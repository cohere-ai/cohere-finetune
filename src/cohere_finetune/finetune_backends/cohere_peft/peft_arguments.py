from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType
from typing import Union


@dataclass
class ModelArguments:
    """Arguments about model."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    lora_alpha: int | None = field(
        default=16,
        metadata={"help": "The alpha parameter of LoRA"},
    )
    lora_dropout: float | None = field(
        default=0.0,
        metadata={"help": "The dropout probability for LoRA layers"},
    )
    use_rslora: bool | None = field(
        default=True,
        metadata={"help": "Whether to use Rank-Stabilized LoRA"},
    )
    lora_r: int | None = field(
        default=8,
        metadata={"help": "The rank parameter of LoRA"},
    )
    lora_target_modules: str | None = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: bool | None = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: str | None = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: str | None = field(
        default="bfloat16",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: str | None = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: bool | None = field(
        default=True,
        metadata={"help": "Enables Flash attention for training"},
    )
    use_peft_lora: bool | None = field(
        default=True,
        metadata={"help": "Enables PEFT LoRA for training"},
    )
    use_8bit_quantization: bool | None = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit"},
    )
    use_4bit_quantization: bool | None = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit"},
    )
    use_reentrant: bool | None = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: bool | None = field(
        default=False,
        metadata={"help": "Enables UnSloth for training"},
    )
    use_postprocessed_model: bool | None = field(
        default=True,
        metadata={"help": "Whether we postprocess the model to convert some small layers to float32"},
    )
    wandb_project: str | None = field(
        default="",
        metadata={"help": "Project name in WANDB"},
    )
    wandb_dir: str | None = field(
        default="",
        metadata={"help": "Directory to save the WANDB result"},
    )
    merged_weights_dir: str | None = field(
        default="",
        metadata={"help": "Directory to save the merged weights"},
    )


@dataclass
class DataArguments:
    """Arguments about data."""

    train_input_path_or_name: str = field(
        metadata={"help": "Path to the raw training data or the name of the HF dataset"},
    )
    eval_input_path: str | None = field(
        default="",
        metadata={"help": "Path to the raw evaluation data"},
    )
    data_output_path: str | None = field(
        default="",
        metadata={"help": "Path to the save the HF datasets"},
    )
    train_ratio: float | None = field(
        default=0.8,
        metadata={"help": "Ratio of the training samples if evaluation samples are not provided"},
    )
    packing: bool | None = field(
        default=False,
        metadata={"help": "Use packing dataset creating"},
    )
    dataset_text_field: str | None = field(
        default="text",
        metadata={"help": "Dataset field to use as input text"},
    )
    max_seq_length: int | None = field(
        default=16384,
        metadata={"help": "The max sequence length"},
    )
    append_concat_token: bool | None = field(
        default=True,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed"},
    )
    add_special_tokens: bool | None = field(
        default=True,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed"},
    )
    splits: str | None = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the HF dataset"},
    )
    response_template: str | None = field(
        default="",
        metadata={
            "help": (
                "The response_template used for DataCollatorForCompletionOnlyLM. "
                "If it is an empty string and packing is False, DataCollatorForLanguageModeling will be used"
            )
        },
    )


@dataclass
class TrainingArgumentsDefaultChanged(TrainingArguments):
    """Arguments about training."""

    adam_beta1: float | None = field(
        default=0.8,
        metadata={"help": "Beta1 for AdamW optimizer"},
    )
    adam_beta2: float | None = field(
        default=0.999,
        metadata={"help": "Beta2 for AdamW optimizer"},
    )
    max_grad_norm: float | None = field(
        default=0.0,
        metadata={"help": "Max gradient norm"},
    )
    bf16: bool | None = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change"
            )
        },
    )
    evaluation_strategy: Union[None, IntervalStrategy, str] = field(
        default="epoch",
        metadata={"help": "Deprecated. Use `eval_strategy` instead"},
    )
    hub_private_repo: bool | None = field(
        default=True,
        metadata={"help": "Whether the model repository is private or not"},
    )
    learning_rate: float | None = field(
        default=1e-4,
        metadata={"help": "The initial learning rate for AdamW"},
    )
    log_level: str | None = field(
        default="info",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'"
            ),
            "choices": ["debug", "info", "warning", "error", "critical", "passive"],
        },
    )
    logging_steps: float | None = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps"
            )
        },
    )
    lr_scheduler_type: Union[None, SchedulerType, str] = field(
        default="cosine",
        metadata={"help": "The scheduler type to use"},
    )
    num_train_epochs: float | None = field(
        default=1,
        metadata={"help": "Total number of training epochs to perform"},
    )
    report_to: Union[None, str, list[str]] = field(
        default="none",
        metadata={"help": "The list of integrations to report the results and logs to"},
    )
    save_strategy: Union[None, IntervalStrategy, str] = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use"},
    )
    weight_decay: float | None = field(
        default=0.1,
        metadata={"help": "Weight decay for AdamW if we apply some"},
    )


# See https://github.com/huggingface/trl/blob/v0.11.3/trl/trainer/sft_trainer.py#L151
TrainingArgumentsDefaultChanged.__name__ = "TrainingArguments"
