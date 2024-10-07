import os
import sys
from accelerate.state import PartialState
from datasets.exceptions import DatasetNotFoundError
from finetune_backends.cohere_peft.peft_arguments import DataArguments, ModelArguments, TrainingArgumentsDefaultChanged
from finetune_backends.cohere_peft.peft_data import get_hf_datasets_from_cohere_data, get_hf_datasets_from_hf_data, preprocess_hf_datasets
from finetune_backends.cohere_peft.peft_model import create_and_prepare_model, load_and_merge_model, save_hf_model
from finetune_backends.cohere_peft.peft_utils import logger
from peft import LoraConfig
from tokenizer_utils import create_and_prepare_tokenizer
from transformers import HfArgumentParser, set_seed
from transformers.integrations.deepspeed import unset_hf_deepspeed_config
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArgumentsDefaultChanged) -> None:
    """Conduct finetuning using given arguments."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project
    if model_args.wandb_dir:
        os.environ["WANDB_DIR"] = model_args.wandb_dir
        os.makedirs(model_args.wandb_dir, exist_ok=True)  # If this folder doesn't exist, W&B can't write to it

    set_seed(training_args.seed)

    # Create and prepare the model
    model = create_and_prepare_model(model_args, data_args, training_args)

    # Create and prepare the tokenizer
    tokenizer = create_and_prepare_tokenizer(model_args.model_name_or_path)

    # Create the Peft configuration
    peft_config = None
    if model_args.use_peft_lora and not model_args.use_unsloth:
        lora_target_modules = (
            model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules
        )
        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            use_rslora=model_args.use_rslora,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )

    # Need to set model.config.use_cache as False, if we use gradient_checkpointing
    model.config.use_cache = not training_args.gradient_checkpointing

    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # Create the HuggingFace datasets
    try:
        raw_datasets = get_hf_datasets_from_hf_data(
            dataset_name=data_args.train_input_path_or_name,
            splits=data_args.splits,
            data_output_path=data_args.data_output_path,
        )
    except (DatasetNotFoundError, FileNotFoundError):
        raw_datasets = get_hf_datasets_from_cohere_data(
            train_input_path=data_args.train_input_path_or_name,
            eval_input_path=data_args.eval_input_path,
            data_output_path=data_args.data_output_path,
            train_ratio=data_args.train_ratio,
        )
    dataset_train, dataset_eval = preprocess_hf_datasets(raw_datasets, tokenizer)
    logger.info(f"Size of the train dataset: {len(dataset_train)}. Size of the evaluation dataset: {len(dataset_eval)}")
    logger.info(f"A sample of the train dataset: {dataset_train[0]}")

    # Create the data collator
    # See https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only
    data_collator = None
    if data_args.response_template:
        data_collator = DataCollatorForCompletionOnlyLM(response_template=data_args.response_template, tokenizer=tokenizer)

    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_kwargs={"append_concat_token": data_args.append_concat_token, "add_special_tokens": data_args.add_special_tokens},
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        data_collator=data_collator,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # Start training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save the final checkpoint
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.state_dict_type = "FULL_STATE_DICT"
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type()
    trainer.save_model()

    # Merge the adapter weights into the base model, and save the merged weights
    if PartialState().process_index == 0 and model_args.merged_weights_dir:
        if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
            # See below:
            # https://github.com/huggingface/peft/issues/297#issuecomment-1639989289
            # https://github.com/huggingface/transformers/issues/28106
            unset_hf_deepspeed_config()
        merged_model = load_and_merge_model(model_args.model_name_or_path, training_args.output_dir)
        save_hf_model(model_args.merged_weights_dir, merged_model, tokenizer, training_args)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsDefaultChanged))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(f"=== Model Arguments: ===\n{model_args}")
    logger.info(f"=== Data Arguments: ===\n{data_args}")
    logger.info(f"=== Training Arguments: ===\n{training_args}")

    main(model_args, data_args, training_args)
