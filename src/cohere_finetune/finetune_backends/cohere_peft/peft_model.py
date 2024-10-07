import os
import torch
from finetune_backends.cohere_peft.peft_arguments import DataArguments, ModelArguments, TrainingArgumentsDefaultChanged
from finetune_backends.cohere_peft.peft_utils import logger
from peft import PeftModel
from transformers import BitsAndBytesConfig, CohereForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast


def create_and_prepare_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArgumentsDefaultChanged,
) -> PreTrainedModel:
    """Create and prepare the model that we will finetune."""
    if model_args.use_unsloth:
        from unsloth import FastLanguageModel
    if (
        torch.distributed.is_available() and
        torch.distributed.is_initialized() and
        torch.distributed.get_world_size() > 1 and
        model_args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    bnb_config = None
    quant_storage_dtype = None
    if model_args.use_4bit_quantization:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, model_args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and model_args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                logger.info("=" * 80)
                logger.info("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                logger.info("=" * 80)
        elif model_args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=model_args.use_8bit_quantization)

    if model_args.use_unsloth:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=model_args.use_4bit_quantization,
        )
        # Do model patching and add fast LoRA weights
        lora_target_modules = (
            model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules
        )
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            target_modules=lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )
    else:
        """
        If the model is not quantized, "torch_dtype" specifies the dtype of all model trainable parameters
        (excluding model buffers which will always be torch.float32).

        If the model is quantized, e.g., loaded in 4-bit, "torch_dtype" specifies the dtype of all the other
        non-quantized trainable submodules of the model (excluding model buffers which will always be torch.float32).

        See https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        """
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.bfloat16
        )
        if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true" or os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
            # Can't use device_map if you use fsdp or deepspeed
            model = CohereForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )
        else:
            model = CohereForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
                device_map="auto",
            )

        if training_args.gradient_checkpointing:
            # See https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/modeling_utils.py#L2380-L2384
            # See https://github.com/huggingface/peft/blob/v0.12.0/src/peft/peft_model.py#L603-L620
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        if model_args.use_postprocessed_model:
            model = postprocess_model(model)

    return model


def load_and_merge_model(base_model_name_or_path: str, adapter_weights_dir: str) -> PreTrainedModel:
    """
    Load the base model and the model finetuned by Peft,
    and merge the adapter weights to the base weights to get a model with merged weights.
    """
    base_model = CohereForCausalLM.from_pretrained(base_model_name_or_path)
    peft_model = PeftModel.from_pretrained(base_model, adapter_weights_dir)
    merged_model = peft_model.merge_and_unload()
    return merged_model


def save_hf_model(
    merged_weights_dir: str,
    model: PreTrainedModel,
    tokenizer: CohereTokenizerFast | None = None,
    args=None,
) -> None:
    """Save a HuggingFace model (and optionally the tokenizer as well as additional args) to a local directory."""
    os.makedirs(merged_weights_dir, exist_ok=False)
    model.save_pretrained(merged_weights_dir, state_dict=None, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(merged_weights_dir)
    if args is not None:
        torch.save(args, os.path.join(merged_weights_dir, "training_args.bin"))


def postprocess_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Postprocess the 4/8-bit model to achieve better training performance.
    - Cast the layer-norm in "float32" for stability.
    - Cast the output of the last layer in "float32" for the same reason.
    """
    for param in model.parameters():
        if param.ndim == 1:
            # Cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    return model
