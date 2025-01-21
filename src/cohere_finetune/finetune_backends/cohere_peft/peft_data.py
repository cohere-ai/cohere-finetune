import os
from datasets import Dataset, DatasetDict, Features, get_dataset_split_names, load_dataset, load_from_disk, NamedSplit, Value
from datasets.builder import DatasetGenerationError
from finetune_backends.cohere_peft.peft_utils import logger
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
from utils import load_and_prepare_csv


def get_hf_datasets_from_cohere_data(
    train_input_path: str,
    eval_input_path: str,
    data_output_path: str = "",
    train_ratio: float = 0.8,
) -> DatasetDict:
    """
    Create the HuggingFace datasets from the preprocessed Cohere datasets.

    The preprocessing we did before will split the data into a training set and an evaluation set,
    so eval_input_path will always exist, unless this function is called solely without being embedded in our pipeline.
    """
    column_types = {"prompt": str, "completion": str}
    if eval_input_path:
        df_train, n_dropped_lines_train = load_and_prepare_csv(train_input_path, column_types)
        logger.info(
            f"{df_train.shape[0]} unique training samples loaded, "
            f"while {n_dropped_lines_train} training samples dropped as they cannot be decoded as UTF-8 strings"
        )
        df_eval, n_dropped_lines_eval = load_and_prepare_csv(eval_input_path, column_types)
        logger.info(
            f"{df_eval.shape[0]} unique evaluation samples loaded, "
            f"while {n_dropped_lines_eval} evaluation samples dropped as they cannot be decoded as UTF-8 strings"
        )
    else:
        df_train_eval, n_dropped_lines_train_eval = load_and_prepare_csv(train_input_path, column_types)
        logger.info(
            f"{df_train_eval.shape[0]} unique samples loaded, "
            f"while {n_dropped_lines_train_eval} samples dropped as they cannot be decoded as UTF-8 strings"
        )
        n_train = int(train_ratio * df_train_eval.shape[0])
        df_train = df_train_eval.iloc[:n_train, :].copy()
        df_eval = df_train_eval.iloc[n_train:, :].copy()
    assert df_train.shape[0] >= 1 and df_eval.shape[0] >= 1

    # In the preprocessed data, the prompt will end with, e.g., <|CHATBOT_TOKEN|>, so here we can and should concatenate
    # the prompt and the completion without inserting a white space at the junction
    df_train["text"] = df_train.apply(lambda z: f'{z.loc["prompt"].rstrip(" ")}{z.loc["completion"].lstrip(" ")}', axis=1)
    df_eval["text"] = df_eval.apply(lambda z: f'{z.loc["prompt"].rstrip(" ")}{z.loc["completion"].lstrip(" ")}', axis=1)

    dataset_train = Dataset.from_pandas(
        df_train,
        features=Features(prompt=Value(dtype="string"), completion=Value(dtype="string"), text=Value(dtype="string")),
        split=NamedSplit("train"),
        preserve_index=False,
    )
    dataset_eval = Dataset.from_pandas(
        df_eval,
        features=Features(prompt=Value(dtype="string"), completion=Value(dtype="string"), text=Value(dtype="string")),
        split=NamedSplit("test"),
        preserve_index=False,
    )

    if data_output_path:
        # Save the datasets if data_output_path is not an empty string
        dataset_train.save_to_disk(os.path.join(data_output_path, "train"))
        dataset_eval.save_to_disk(os.path.join(data_output_path, "test"))

    raw_datasets = DatasetDict()
    raw_datasets["train"] = dataset_train
    raw_datasets["test"] = dataset_eval

    return raw_datasets


def get_hf_datasets_from_hf_data(dataset_name: str, splits: str, data_output_path: str = "") -> DatasetDict:
    """Load a HuggingFace dataset."""
    _ = get_dataset_split_names(dataset_name, trust_remote_code=True)  # Check whether the dataset exists in HuggingFace

    raw_datasets = DatasetDict()
    for split in splits.split(","):
        try:
            dataset = load_dataset(dataset_name, split=split)  # Try first if the dataset is on a Hub repo
        except DatasetGenerationError:
            dataset = load_from_disk(os.path.join(dataset_name, split))  # If not, check local dataset

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train")

        if data_output_path:
            # Save the dataset if data_output_path is not an empty string
            dataset.save_to_disk(os.path.join(data_output_path, split))

    return raw_datasets


def preprocess_hf_datasets(
    raw_datasets: DatasetDict,
    tokenizer: CohereTokenizerFast,
    apply_chat_template: bool = False,
) -> tuple[Dataset, Dataset]:
    """
    Preprocess HuggingFace datasets by applying the template of Cohere on them, if the data has not been preprocessed.

    No need to apply the template again, if the data has been preprocessed.
    """
    if apply_chat_template:

        def preprocess_samples(samples):
            batch = []
            for chat in samples["messages"]:
                batch.append(tokenizer.apply_chat_template(chat, tokenize=False))
            return {"text": batch}

        raw_datasets = raw_datasets.map(
            preprocess_samples,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    return raw_datasets["train"], raw_datasets["test"]
