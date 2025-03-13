import os
from consts import CHAT_PROMPT_TEMPLATE_CMD_A_03_2025, CHAT_PROMPT_TEMPLATE_CMD_R, CHAT_PROMPT_TEMPLATE_CMD_R_08_2024, CHAT_PROMPT_TEMPLATE_CMD_R_7B_12_2024
from utils import load_file, logger


def get_model_name_from_hf_config(hf_config_path: str) -> str:
    """
    According to the config.json file in the HuggingFace checkpoint, get the name of the model.

    It distinguishes only the "supported" Cohere's models based on their config.json files, i.e.,
    it may not correctly identify a model when "un-supported" Cohere's models are used
    """
    hf_config = load_file(hf_config_path)

    if hf_config["architectures"] != ["CohereForCausalLM"] and hf_config["architectures"] != ["Cohere2ForCausalLM"]:
        raise ValueError("The model is not one of Cohere's models for causal LM")

    if hf_config["hidden_size"] == 8192 and hf_config["rope_theta"] == 8000000:
        return "command-r"
    elif hf_config["hidden_size"] == 8192 and hf_config["rope_theta"] == 4000000 and hf_config["max_position_embeddings"] == 131072:
        return "command-r-08-2024"
    elif hf_config["hidden_size"] == 12288 and hf_config["rope_theta"] == 75000000:
        return "command-r-plus"
    elif hf_config["hidden_size"] == 12288 and hf_config["rope_theta"] == 8000000:
        return "command-r-plus-08-2024"
    elif hf_config["hidden_size"] == 4096 and hf_config["rope_theta"] == 50000:
        return "command-r-7b-12-2024"
    elif hf_config["hidden_size"] == 12288 and hf_config["rope_theta"] == 50000:
        return "command-a-03-2025"
    elif hf_config["hidden_size"] == 4096 and hf_config["rope_theta"] == 10000:
        return "aya-expanse-8b"
    elif hf_config["hidden_size"] == 8192 and hf_config["rope_theta"] == 4000000 and hf_config["max_position_embeddings"] == 8192:
        return "aya-expanse-32b"
    else:
        raise ValueError("The model is not one of Cohere's models for causal LM that we support")


def get_model_config_from_model_name_and_model_path(model_name: str, model_path: str | None) -> dict:
    """
    According to model_name and model_path, get the config of the model,
    which contains all information about the model that we will use for cohere-finetune.
    """
    if model_name == "command-r":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_R,
            "prompt_last_token": "<|CHATBOT_TOKEN|>",
            "hf_model_name_or_path": "CohereForAI/c4ai-command-r-v01" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    elif model_name == "command-r-08-2024":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_R_08_2024,
            "prompt_last_token": "<|CHATBOT_TOKEN|>",
            "hf_model_name_or_path": "CohereForAI/c4ai-command-r-08-2024" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    elif model_name == "command-r-plus":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_R,
            "prompt_last_token": "<|CHATBOT_TOKEN|>",
            "hf_model_name_or_path": "CohereForAI/c4ai-command-r-plus" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    elif model_name == "command-r-plus-08-2024":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_R_08_2024,
            "prompt_last_token": "<|CHATBOT_TOKEN|>",
            "hf_model_name_or_path": "CohereForAI/c4ai-command-r-plus-08-2024" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    elif model_name == "command-r-7b-12-2024":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_R_7B_12_2024,
            "prompt_last_token": "<|START_RESPONSE|>",
            "hf_model_name_or_path": "CohereForAI/c4ai-command-r7b-12-2024" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    elif model_name == "command-a-03-2025":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_A_03_2025,
            "prompt_last_token": "<|START_RESPONSE|>",
            "hf_model_name_or_path": "CohereForAI/c4ai-command-a-03-2025" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    elif model_name == "aya-expanse-8b":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_R,
            "prompt_last_token": "<|CHATBOT_TOKEN|>",
            "hf_model_name_or_path": "CohereForAI/aya-expanse-8b" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    elif model_name == "aya-expanse-32b":
        return {
            "model_name": model_name,
            "prompt_template": CHAT_PROMPT_TEMPLATE_CMD_R,
            "prompt_last_token": "<|CHATBOT_TOKEN|>",
            "hf_model_name_or_path": "CohereForAI/aya-expanse-32b" if model_path is None else model_path,
            "max_possible_max_sequence_length": 16384,
        }
    else:
        raise ValueError(f"{model_name} is not a valid and supported model name")


class ModelConfig:
    """Model configuration."""

    def __init__(self, model_name_or_path: str) -> None:
        """Initialize ModelConfig."""
        try:
            model_name = get_model_name_from_hf_config(os.path.join(model_name_or_path, "config.json"))
            model_path = model_name_or_path
        except FileNotFoundError:
            model_name = model_name_or_path
            model_path = None

        self.model_config = get_model_config_from_model_name_and_model_path(model_name, model_path)
        logger.info(f"The model config is as follows:\n{self.model_config}")

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return self.model_config["model_name"]

    def get_prompt_template(self) -> str:
        """Get the prompt template for the model."""
        return self.model_config["prompt_template"]

    def get_prompt_last_token(self) -> str:
        """Get the last token in the prompt for the model, which is used to signal the start of model completion."""
        return self.model_config["prompt_last_token"]

    def get_hf_model_name_or_path(self) -> str:
        """Get the HuggingFace model name or path for the model."""
        return self.model_config["hf_model_name_or_path"]

    def get_max_possible_max_sequence_length(self) -> int:
        """Get the max possible max sequence length for the model."""
        return self.model_config["max_possible_max_sequence_length"]
