import json
import os
import torch
from consts import FinetuneStrategy, ParallelStrategy, FINETUNE_BACKEND_KEY, PATH_PREFIX_KEY
from model_config import ModelConfig
from typing import Any


class BaseConfig:
    """BaseConfig is the base class for any configurations."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize BaseConfig."""
        return

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary that stores all user-defined attributes of the config."""
        return {name: attr.to_dict() if hasattr(attr, "to_dict") else attr for name, attr in self.__dict__.items()}

    def _must_be_in_range(self, key: str, min_value: float, max_value: float) -> None:
        """Validate that the value of the key is in the given range."""
        value = getattr(self, key)
        if not (min_value <= float(value) <= max_value):
            raise ValueError(f"{key} is equal to {value}, which is not between {min_value} and {max_value}")

    def _must_be_in_set(self, key: str, valid_set: set) -> None:
        """Validate that the value of the key is in the given set."""
        value = getattr(self, key)
        if value not in valid_set:
            raise ValueError(f"{key} is equal to {value}, which is not in the set {valid_set}")

    def _must_be_non_empty_subset(self, key: str, valid_set: set) -> None:
        """Validate that the value of the key is a non-empty subset of the given set."""
        value = getattr(self, key)
        if not (len(value) > 0 and set(value).issubset(valid_set)):
            raise ValueError(f"{key} is equal to {value}, which is not a non-empty subset of {valid_set}")


class LoraConfig(BaseConfig):
    """LoraConfig stores and validates the Lora hyperparameters."""

    def __init__(
        self,
        rank: int = 8,
        alpha: int | None = None,
        target_modules: list[str] | None = None,
        rslora: str = "true",
    ) -> None:
        """Initialize LoraConfig."""
        self.rank = int(rank)
        self.alpha = int(alpha) if alpha else 2 * self.rank
        self.target_modules = target_modules if target_modules else ["q", "k", "v", "o"]
        self.rslora = json.loads(rslora)

        self._validate()

    def _validate(self) -> None:
        """Validate LoraConfig."""
        self._must_be_in_range("rank", 1, float("Inf"))
        self._must_be_in_range("alpha", 1, float("Inf"))
        self._must_be_non_empty_subset("target_modules", {"q", "k", "v", "o", "ffn_expansion"})


class WandbConfig(BaseConfig):
    """WandbConfig stores and validates the W&B configuration."""

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize WandbConfig."""
        self.project = project
        self.entity = entity
        self.run_id = run_id

        self._validate()

    def _validate(self) -> None:
        """Validate WandbConfig."""
        if self.project == "":
            raise ValueError("project must be a non-empty string")


class Hyperparameters(BaseConfig):
    """Hyperparameters stores and validates all the hyperparameters for finetuning."""

    def __init__(
        self,
        finetune_name: str,
        base_model_name_or_path: str = "command-r-08-2024",
        parallel_strategy: str = "fsdp",
        finetune_strategy: str = "lora",
        use_4bit_quantization: str = "false",
        gradient_checkpointing: str = "true",
        gradient_accumulation_steps: int = 1,
        train_epochs: int = 1,
        train_batch_size: int = 16,
        validation_batch_size: int = 16,
        learning_rate: float = 1e-4,
        eval_percentage: float = 0.2,
        lora_config: dict[str, Any] | None = None,
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Hyperparameters."""
        self.finetune_name = finetune_name
        self.base_model_config = ModelConfig(base_model_name_or_path)
        self.parallel_strategy = ParallelStrategy(parallel_strategy)
        self.finetune_strategy = FinetuneStrategy(finetune_strategy)
        self.use_4bit_quantization = json.loads(use_4bit_quantization)
        self.gradient_checkpointing = json.loads(gradient_checkpointing)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.train_epochs = int(train_epochs)
        self.train_batch_size = int(train_batch_size)
        self.validation_batch_size = int(validation_batch_size)
        self.learning_rate = float(learning_rate)
        self.eval_percentage = float(eval_percentage)
        self.lora_config = LoraConfig(**lora_config) if lora_config else LoraConfig()
        self.wandb_config = WandbConfig(**wandb_config) if wandb_config else None

        self.max_sequence_length = self.base_model_config.get_max_possible_max_sequence_length()

        self._validate()

    def _validate(self) -> None:
        """
        Validate Hyperparameters.
        Some of the following requirements are based on best practices and can be changed if needed.
        """
        self._must_be_in_range("gradient_accumulation_steps", 1, float("Inf"))
        self._must_be_in_range("train_epochs", 1, float("Inf"))
        self._must_be_in_range("train_batch_size", 1, float("Inf"))
        self._must_be_in_range("validation_batch_size", 1, float("Inf"))
        self._must_be_in_range("learning_rate", 1e-5, 1)
        self._must_be_in_range("eval_percentage", 0.05, 0.5)

        n_batch_partitions = torch.cuda.device_count() if self.parallel_strategy != ParallelStrategy.VANILLA else 1
        if self.train_batch_size % (n_batch_partitions * self.gradient_accumulation_steps) != 0:
            raise ValueError(
                f"train_batch_size {self.train_batch_size} is not divisible by n_batch_partitions * "
                f"gradient_accumulation_steps = {n_batch_partitions} * {self.gradient_accumulation_steps}"
            )
        if self.validation_batch_size % n_batch_partitions != 0:
            raise ValueError(
                f"validation_batch_size {self.validation_batch_size} is not divisible by "
                f"n_batch_partitions = {n_batch_partitions}"
            )


class PathConfig(BaseConfig):
    """PathConfig stores paths for inputs, intermediate results during finetuning, and final outputs."""

    def __init__(self, finetune_name: str) -> None:
        """Initialize PathConfig."""
        self.finetune_name = finetune_name

        # Root directory that contains all paths used in the Docker container
        self.root_dir = os.path.join(os.environ[PATH_PREFIX_KEY], finetune_name)

        # Input: directories of the raw train & eval data
        self.input_train_dir = os.path.join(self.root_dir, "input/data/training")
        self.input_eval_dir = os.path.join(self.root_dir, "input/data/evaluation")

        # Finetune: paths of the preprocessed train & eval data and directories of checkpoints, logs, configs, metrics (optional)
        self.finetune_dir = os.path.join(self.root_dir, "finetune")
        self.finetune_train_path = os.path.join(self.finetune_dir, "data/train.csv")
        self.finetune_eval_path = os.path.join(self.finetune_dir, "data/eval.csv")
        self.checkpoints_dir = os.path.join(self.finetune_dir, "checkpoints")
        self.logs_dir = os.path.join(self.finetune_dir, "logs")
        self.configs_dir = os.path.join(self.finetune_dir, "configs")
        self.metrics_dir = os.path.join(self.finetune_dir, "metrics")

        # Output: directories of trained merged weights, trained adapter weights (optional), inference engine (optional)
        self.output_dir = os.path.join(self.root_dir, "output")
        self.merged_weights_dir = os.path.join(self.output_dir, "merged_weights")
        self.adapter_weights_dir = os.path.join(self.output_dir, "adapter_weights")
        self.inference_engine_dir = os.path.join(self.output_dir, "inference_engine")


class CoherePeftPathConfig(PathConfig):
    """CoherePeftPathConfig stores paths specifically used for Cohere peft backend, in addition to all other paths."""

    def __init__(self, finetune_name: str) -> None:
        """Initialize CoherePeftPathConfig."""
        super().__init__(finetune_name)

        self.finetune_backends_parent_dir = os.path.dirname(__file__)
        self.peft_template_parallel_configs_dir = os.path.join(
            self.finetune_backends_parent_dir, "finetune_backends/cohere_peft/template_parallel_configs"
        )


def get_path_config(finetune_name: str) -> PathConfig:
    """Get the path config according to the finetuning backend that is being used."""
    if os.environ[FINETUNE_BACKEND_KEY] == "peft":
        return CoherePeftPathConfig(finetune_name)
    else:
        raise NotImplementedError
