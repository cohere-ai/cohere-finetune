import json
import os
import signal
import sys
import time
import torch
from chat_utils import normalize_messages
from flask import Flask, jsonify, request, Response
from tokenizer_utils import create_and_prepare_tokenizer
from transformers import CohereForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
from utils import logger


def load_model_for_inference(model_name_or_path: str) -> PreTrainedModel:
    """Load the model for inference."""
    # The model is set in evaluation mode by default using model.eval()
    # See https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    model = CohereForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


class CohereInference:
    """
    CohereInference is the main class for the Cohere inference service.

    This service should only be used for quick experiments or small-scale evaluations, not for inference in production.
    """

    def __init__(self) -> None:
        """Initialize the Cohere inference service."""
        self.app = Flask(__name__)

        self.model_name_or_path: str | None = None
        self.model: PreTrainedModel | None = None
        self.tokenizer: CohereTokenizerFast | None = None

        self._define_inference_route()
        self._define_info_route()
        self._define_terminate_route()

    def _define_inference_route(self) -> None:

        @self.app.route("/inference", methods=["POST"])
        def inference() -> Response:
            """Conduct model inference on a single text."""
            model_name_or_path = request.json.pop("model_name_or_path")
            message = request.json.pop("message")
            chat_history = request.json.pop("chat_history", [])
            preamble = request.json.pop("preamble", "")
            param = {k: json.loads(v) if v in {"false", "true"} else v for k, v in request.json.items()}

            if model_name_or_path != self.model_name_or_path:
                logger.info("Model is not loaded or needs to be changed, so loading the desired model...")
                self._reset()  # Before loading another model, reset to release the memory used
                self.model_name_or_path = model_name_or_path
                self.model = load_model_for_inference(model_name_or_path)
                self.tokenizer = create_and_prepare_tokenizer(model_name_or_path)

            start_time = time.time()

            normalize_messages(chat_history)
            chat = ([{"role": "system", "content": preamble}] if preamble else []) + \
                chat_history + [{"role": "user", "content": message}]
            templated_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            """
            During training, we didn't add the special tokens like <BOS_TOKEN> to the preprocessed (templated) text,
            so we let the tokenizer add them (setting add_special_tokens=True)

            During inference, we just added these special tokens by apply_chat_template,
            so we don't let the tokenizer add them again (setting add_special_tokens=False)
            """
            tokenized_templated_text = self.tokenizer(
                templated_text,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )

            with torch.no_grad():
                generations = self.model.generate(input_ids=tokenized_templated_text["input_ids"].to("cuda"), **param)

            # We let the tokenizer keep the special tokens such that it is easier for us to extract the final generation
            decoded_generation = self.tokenizer.decode(generations[0], skip_special_tokens=False)
            assert decoded_generation.startswith(templated_text) and decoded_generation.endswith(self.tokenizer.eos_token)
            final_generation = decoded_generation[len(templated_text):-len(self.tokenizer.eos_token)]

            return jsonify({
                "text": final_generation,
                "prompt": templated_text,
                "latency": f"{(time.time() - start_time) * 1000:.0f}ms",
            })

    def _define_info_route(self) -> None:

        @self.app.route("/info", methods=["GET"])
        def get_info() -> Response:
            """Return some information about the Cohere inference service."""
            return jsonify({"model_name_or_path": self.model_name_or_path})

    def _define_terminate_route(self) -> None:

        @self.app.route("/terminate", methods=["GET"])
        def terminate() -> Response:
            """Terminate the Cohere inference service."""
            sys.stdout.flush()
            logger.info("Terminating server")
            os.kill(os.getpid(), signal.SIGINT)
            return jsonify({"message": "Server terminated"})

    def _reset(self) -> None:
        """Reset the model_name_or_path, model, and tokenizer to the original values, and release the memory."""
        self.model_name_or_path = None
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    def run(self, debug: bool = True) -> None:
        """Run the app."""
        self.app.run(debug=debug, host="0.0.0.0", port=5001)


if __name__ == "__main__":
    cohere_inference_service = CohereInference()
    cohere_inference_service.run(debug=False)
