import os
import signal
import sys
import time
import traceback
from configs import get_path_config, Hyperparameters, PathConfig
from consts import Status
from data import CohereDataset
from errors import DatasetExceedsMaxSequenceLengthError
from flask import Flask, jsonify, request, Response
from preprocess import preprocess
from threading import Thread
from tokenizer_utils import create_and_prepare_tokenizer, get_min_possible_max_sequence_length
from train import train
from utils import logger


class CohereFinetune:
    """CohereFinetune is the main class for the Cohere finetune service."""

    def __init__(self) -> None:
        """Initialize the Cohere finetune service."""
        self.app = Flask(__name__)

        self.status: Status = Status.IDLE
        self.error_message: str = ""
        self.path_config: PathConfig | None = None
        self.hyperparameters: Hyperparameters | None = None

        self._define_finetune_route()
        self._define_status_route()
        self._define_terminate_route()

    def _define_finetune_route(self) -> None:

        @self.app.route("/finetune", methods=["POST"])
        def start_finetune() -> tuple[Response, int]:
            """Start the finetune process."""
            if self.status == Status.IN_PROGRESS:
                return jsonify({"message": "There is already a finetune in progress"}), 409

            self._reset()

            self.path_config = get_path_config(finetune_name=request.json.get("finetune_name"))

            try:
                self.hyperparameters = Hyperparameters(**request.json)
                logger.info("Hyperparameters: %s", self.hyperparameters.to_dict())
            except ValueError as e:
                trc = traceback.format_exc()
                logger.exception("Invalid hyperparameter values %s", trc)
                return jsonify({"message": f"Invalid hyperparameter values: {e}"}), 400
            except TypeError as e:
                trc = traceback.format_exc()
                logger.exception("Invalid hyperparameter types %s", trc)
                return jsonify({"message": f"Invalid hyperparameter types: {e}"}), 400

            finetuning_thread = Thread(target=self.finetune)
            finetuning_thread.start()

            return jsonify({"message": "Finetune started."}), 202

    def _define_status_route(self) -> None:

        @self.app.route("/status", methods=["GET"])
        def get_status() -> Response:
            """Return the status of the Cohere finetune service."""
            if self.error_message != "":
                return jsonify({"status": self.status, "message": self.error_message})
            else:
                return jsonify({"status": self.status})

    def _define_terminate_route(self) -> None:

        @self.app.route("/terminate", methods=["GET"])
        def terminate() -> Response:
            """Terminate the Cohere finetune service."""
            sys.stdout.flush()
            logger.info("Terminating server")
            os.kill(os.getpid(), signal.SIGINT)
            return jsonify({"message": "Server terminated"})

    def finetune(self) -> None:
        """Start the finetune process."""

        # There could be race conditions with self.status, but we don't expect concurrent requests
        self.status = Status.IN_PROGRESS
        logger.info("Starting finetuning")
        start_time = time.time()

        # Create the finetuning dataset
        try:
            cohere_dataset = CohereDataset(train_dir=self.path_config.input_train_dir, eval_dir=self.path_config.input_eval_dir)
            cohere_dataset.convert_to_chat_jsonl()
        except FileNotFoundError as e:
            trc = traceback.format_exc()
            self._format_error(
                logging_msg=f"Missing files during dataset creation: {trc}",
                error_msg=f"Missing files during dataset creation: {e}",
            )
            return
        except ValueError as e:
            trc = traceback.format_exc()
            self._format_error(
                logging_msg=f"Exception during dataset creation: {trc}",
                error_msg=f"Exception during dataset creation: {e}",
            )
            return

        # Create and prepare the tokenizer
        tokenizer = create_and_prepare_tokenizer(self.hyperparameters.base_model.get_hf_model_name_or_path())

        # Preprocess the finetuning dataset by doing train eval split (if needed) and putting the texts in template
        try:
            logger.info("Starting preprocessing")
            preprocess(
                input_train_path=cohere_dataset.train_path,
                input_eval_path=cohere_dataset.eval_path,
                output_train_path=self.path_config.finetune_train_path,
                output_eval_path=self.path_config.finetune_eval_path,
                eval_percentage=self.hyperparameters.eval_percentage,
                template=self.hyperparameters.base_model.get_prompt_template(),
                max_sequence_length=self.hyperparameters.base_model.get_max_possible_max_sequence_length(),
                tokenizer=tokenizer,
            )
            logger.info("Preprocessing finished")
        except RuntimeError as e:
            trc = traceback.format_exc()
            self._format_error(
                logging_msg=f"Exception during preprocessing: {trc}",
                error_msg=f"Exception during preprocessing: {e}",
            )
            return

        # Get the length of the longest sequence in the preprocessed dataset and use it as the final max_sequence_length
        try:
            logger.info("Calculating max sequence length from the preprocessed dataset")
            self.hyperparameters.max_sequence_length = get_min_possible_max_sequence_length(
                    train_path=self.path_config.finetune_train_path,
                    eval_path=self.path_config.finetune_eval_path,
                    tokenizer=tokenizer,
                )
            logger.info("Using max sequence length: %d", self.hyperparameters.max_sequence_length)
        except DatasetExceedsMaxSequenceLengthError as e:
            trc = traceback.format_exc()
            self._format_error(
                logging_msg=f"Exception during max sequence length calculation: {trc}",
                error_msg=f"Exception during max sequence length calculation: {e}",
            )
            return
        except Exception:
            # If the above procedure fails for any other reason, use the default max_sequence_length in hyperparameters
            logger.info("Using default max sequence length: %d", self.hyperparameters.max_sequence_length)

        # Finetune the base model using the preprocessed dataset
        try:
            logger.info("Starting training")
            train(self.path_config, self.hyperparameters)
            logger.info("Training finished")
        except FileNotFoundError as e:
            trc = traceback.format_exc()
            self._format_error(
                logging_msg=f"Missing files during training: {trc}",
                error_msg=f"Missing files during training: {e}",
            )
            return
        except (RuntimeError, ValueError) as e:
            trc = traceback.format_exc()
            self._format_error(
                logging_msg=f"Exception during training: {trc}",
                error_msg=f"Exception during training: {e}",
            )
            return

        logger.info(f"Finetuning completed in {int(time.time() - start_time)} seconds.")
        self.status = Status.FINISHED

    def _format_error(self, logging_msg: str | None, error_msg: str) -> None:
        """Format the logging and error messages, and set status as ERROR."""
        if logging_msg:
            logger.exception(logging_msg)
        self.error_message = error_msg
        self.status = Status.ERROR

    def _reset(self) -> None:
        """Reset the status, error message, path_config and hyperparameters to the original values."""
        self.status = Status.IDLE
        self.error_message = ""
        self.path_config = None
        self.hyperparameters = None

    def run(self, debug: bool = True) -> None:
        """Run the app."""
        self.app.run(debug=debug, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    cohere_finetune_service = CohereFinetune()
    cohere_finetune_service.run(debug=False)
