import csv
import json
from pathlib import Path
from utils import get_a_file_of_given_type_from_dir, get_ext


def is_prompt_completion_jsonl(path: str) -> bool:
    """For a given JSONL file, check whether it is in prompt-completion format (return True) or not (return False)."""
    with open(path) as file:
        first_line = file.readline()

    try:
        first_data = json.loads(first_line)
        return "prompt" in first_data and "completion" in first_data
    except json.JSONDecodeError as e:
        raise ValueError(f"File is not a valid JSONL file at line 1: {e.msg}: column {e.colno}") from e


def convert_prompt_completion_csv_to_chat_jsonl(input_csv_path: str) -> str:
    """Convert a prompt-completion CSV file to a chat format JSONL file, where the CSV file must not have a header."""
    output_jsonl_path = Path(input_csv_path).with_suffix(".jsonl")

    with (
        open(input_csv_path, encoding="utf-8") as input_csv_file,
        open(output_jsonl_path, "w", encoding="utf-8") as output_jsonl_file,
    ):
        reader = csv.reader(input_csv_file)

        for row in reader:
            prompt, completion = row[0], row[1]
            converted_data = {
                "messages": [
                    {"role": "User", "content": prompt},
                    {"role": "Chatbot", "content": completion},
                ]
            }
            output_jsonl_file.write(json.dumps(converted_data) + "\n")

    return str(output_jsonl_path)


def convert_prompt_completion_jsonl_to_chat_jsonl(input_jsonl_path: str) -> str:
    """Convert a prompt-completion JSONL file to a chat format JSONL file."""
    output_jsonl_path = Path(input_jsonl_path).with_stem(Path(input_jsonl_path).stem + "_chat")

    with (
        open(input_jsonl_path) as input_jsonl_file,
        open(output_jsonl_path, "w") as output_jsonl_file,
    ):
        lines = input_jsonl_file.readlines()

        for i, line in enumerate(lines):
            line_cleaned = line.strip()
            if not line_cleaned:
                continue

            try:
                data = json.loads(line_cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error processing dataset file at line number {i + 1}: {e.msg}: column {e.colno}") from e

            try:
                # Default to no preamble for prompt-completion format to chat format conversion
                converted_data = {
                    "messages": [
                        {"role": "User", "content": data["prompt"]},
                        {"role": "Chatbot", "content": data["completion"]},
                    ]
                }
            except KeyError as e:
                raise ValueError(f"Missing key: {e.args[0]} in {input_jsonl_path} at line number {i + 1}: {e}") from e

            output_jsonl_file.write(json.dumps(converted_data) + "\n")

    return str(output_jsonl_path)


def convert_data_to_chat_jsonl(input_path: str) -> str:
    """Convert input data to a chat format JSONL file."""
    ext = get_ext(input_path)

    if ext == ".csv":
        # If the input data is a CSV file, it must be in the prompt-completion format and we must convert it
        output_path = convert_prompt_completion_csv_to_chat_jsonl(input_path)
    elif ext == ".jsonl" and is_prompt_completion_jsonl(input_path):
        # If the input data is a prompt-completion format JSONL file, we must convert it
        output_path = convert_prompt_completion_jsonl_to_chat_jsonl(input_path)
    else:
        # If the input data is already a chat format JSONL file, no need to convert it
        output_path = input_path

    return output_path


class CohereDataset:
    """CohereDataset stores important paths about data and provides the method for data format conversion."""

    def __init__(self, train_dir: str, eval_dir: str) -> None:
        """Initialize CohereDataset."""
        self.train_dir = train_dir

        self.eval_dir = eval_dir

        self.train_path = get_a_file_of_given_type_from_dir(train_dir, ".csv")
        if not self.train_path:
            self.train_path = get_a_file_of_given_type_from_dir(train_dir, ".jsonl")
        if not self.train_path:
            raise FileNotFoundError(f"No csv or jsonl files found for training under {train_dir}")

        self.eval_path = get_a_file_of_given_type_from_dir(eval_dir, ".csv")
        if not self.eval_path:
            self.eval_path = get_a_file_of_given_type_from_dir(eval_dir, ".jsonl")

    def convert_to_chat_jsonl(self) -> None:
        """Convert train and eval data to chat format JSONL files, and update the paths to the converted files."""
        self.train_path = convert_data_to_chat_jsonl(self.train_path)

        if self.eval_path:
            self.eval_path = convert_data_to_chat_jsonl(self.eval_path)
