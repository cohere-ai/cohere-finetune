import os
import subprocess
import torch
from configs import CoherePeftPathConfig, Hyperparameters, PathConfig
from consts import ENVIRONMENT_MODE_KEY, FINETUNE_BACKEND_KEY
from utils import load_file, save_file


def create_parallel_config_from_template(input_path: str, parallel_config_dir: str) -> str:
    """Create parallel config by loading and modifying a template config."""
    parallel_config = load_file(input_path)

    parallel_config["num_processes"] = torch.cuda.device_count()

    output_path = os.path.join(parallel_config_dir, os.path.basename(input_path))
    save_file(parallel_config, output_path)

    return output_path


def train_with_peft(path_config: CoherePeftPathConfig, hyperparameters: Hyperparameters) -> None:
    """Call HuggingFace's Peft to finetune the model, using the given path config and hyperparameters."""
    # Modify the configuration according to different parallel strategies
    if hyperparameters.parallel_strategy != "vanilla":
        n_batch_partitions = torch.cuda.device_count()
        if hyperparameters.parallel_strategy == "fsdp" and not hyperparameters.use_4bit_quantization:
            use_postprocessed_model = False
            config_file_path = create_parallel_config_from_template(
                os.path.join(path_config.peft_template_parallel_configs_dir, "fsdp_config.yaml"),
                path_config.configs_dir,
            )
        elif hyperparameters.parallel_strategy == "fsdp" and hyperparameters.use_4bit_quantization:
            use_postprocessed_model = False
            config_file_path = create_parallel_config_from_template(
                os.path.join(path_config.peft_template_parallel_configs_dir, "fsdp_qlora_config.yaml"),
                path_config.configs_dir,
            )
        elif hyperparameters.parallel_strategy == "deepspeed" and not hyperparameters.use_4bit_quantization:
            use_postprocessed_model = True
            config_file_path = create_parallel_config_from_template(
                os.path.join(path_config.peft_template_parallel_configs_dir, "deepspeed_z3_config.yaml"),
                path_config.configs_dir,
            )
        else:
            use_postprocessed_model = True
            config_file_path = create_parallel_config_from_template(
                os.path.join(path_config.peft_template_parallel_configs_dir, "deepspeed_z3_qlora_config.yaml"),
                path_config.configs_dir,
            )
        peft_cmd = ["accelerate", "launch", "--config_file", config_file_path]
    else:
        n_batch_partitions = 1
        use_postprocessed_model = True
        peft_cmd = ["python"]

    # Get the HuggingFace model name
    model_name_or_path = hyperparameters.base_model_config.get_hf_model_name_or_path()

    # Get the per_device_train_batch_size and per_device_eval_batch_size
    per_device_train_batch_size, r_train = divmod(
        hyperparameters.train_batch_size,
        n_batch_partitions * hyperparameters.gradient_accumulation_steps,
    )
    if r_train != 0:
        raise ValueError(
            f"train_batch_size {hyperparameters.train_batch_size} is not divisible by n_batch_partitions * "
            f"gradient_accumulation_steps = {n_batch_partitions} * {hyperparameters.gradient_accumulation_steps}"
        )
    per_device_eval_batch_size, r_eval = divmod(
        hyperparameters.validation_batch_size,
        n_batch_partitions,
    )
    if r_eval != 0:
        raise ValueError(
            f"validation_batch_size {hyperparameters.validation_batch_size} is not divisible by "
            f"n_batch_partitions = {n_batch_partitions}"
        )

    # Get lora_target_modules with the names used by HuggingFace
    lora_target_modules = []
    for module in hyperparameters.lora_config.target_modules:
        if module == "q":
            lora_target_modules.append("q_proj")
        elif module == "k":
            lora_target_modules.append("k_proj")
        elif module == "v":
            lora_target_modules.append("v_proj")
        elif module == "o":
            lora_target_modules.append("o_proj")
        elif module == "ffn_expansion":
            lora_target_modules.extend(["down_proj", "up_proj", "gate_proj"])
    lora_target_modules = ",".join(lora_target_modules)

    peft_cmd += [
        f"{path_config.finetune_backends_parent_dir}/finetune_backends/cohere_peft/peft_train.py",
        "--model_name_or_path", model_name_or_path,
        "--train_input_path_or_name", path_config.finetune_train_path,
        "--eval_input_path", path_config.finetune_eval_path,
        "--max_seq_len", f"{hyperparameters.max_sequence_length}",
        "--response_template", hyperparameters.base_model_config.get_prompt_last_token(),
        "--output_dir", path_config.checkpoints_dir,
        "--merged_weights_dir", path_config.merged_weights_dir,
        "--gradient_checkpointing", f"{hyperparameters.gradient_checkpointing}",
        "--use_4bit_quantization", f"{hyperparameters.use_4bit_quantization}",
        "--gradient_accumulation_steps", f"{hyperparameters.gradient_accumulation_steps}",
        "--per_device_train_batch_size", f"{per_device_train_batch_size}",
        "--per_device_eval_batch_size", f"{per_device_eval_batch_size}",
        "--use_postprocessed_model", f"{use_postprocessed_model}",
        "--num_train_epochs", f"{hyperparameters.train_epochs}",
        "--learning_rate", f"{hyperparameters.learning_rate}",
        "--lora_r", f"{hyperparameters.lora_config.rank}",
        "--lora_alpha", f"{hyperparameters.lora_config.alpha}",
        "--use_rslora", f"{hyperparameters.lora_config.rslora}",
        "--lora_target_modules", f"{lora_target_modules}",
    ]

    # Add arguments about W&B if hyperparameters.wandb_config is not None
    if hyperparameters.wandb_config:
        peft_cmd += [
            "--report_to", "wandb",
            "--wandb_project", hyperparameters.wandb_config.project,
            "--run_name", hyperparameters.wandb_config.run_id,
            "--wandb_dir", path_config.logs_dir,
        ]

    peft_env = os.environ.copy()
    peft_env["PYTHONPATH"] = path_config.finetune_backends_parent_dir
    try:
        subprocess.run(
            peft_cmd,
            check=True,
            text=True,
            # stderr always passes through, but stdout is suppressed if it is in PROD mode
            stdout=None if os.environ.get(ENVIRONMENT_MODE_KEY, "PROD") == "DEV" else subprocess.DEVNULL,
            env=peft_env,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError from e


def train(path_config: PathConfig, hyperparameters: Hyperparameters) -> None:
    """Conduct the training according to the finetuning backend that is being used."""
    if os.environ[FINETUNE_BACKEND_KEY] == "peft":
        train_with_peft(path_config, hyperparameters)
    else:
        raise NotImplementedError
