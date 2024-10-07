from consts import CANDIDATE_MAX_SEQUENCE_LENGTHS
from errors import DatasetExceedsMaxSequenceLengthError
from transformers import AutoTokenizer
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
from utils import load_and_prepare_csv, logger


def create_and_prepare_tokenizer(model_name_or_path: str) -> CohereTokenizerFast:
    """Create and prepare the Cohere tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, add_eos_token=True)
    if tokenizer.pad_token_id is None:
        # tokenizer.pad_token will be automatically changed to tokenizer.eos_token, as well
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_n_tokens_in_rendered_prompt_completion(rendered_prompt_completion: str, tokenizer: CohereTokenizerFast) -> int:
    """
    Get the number of tokens in a rendered prompt-completion string, in various scenarios as discussed below:

    If the prompt-completion string starts with <BOS_TOKEN> and ends with
        <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|> (as when we do preprocessing in preprocess_chat):
        we don't add special tokens in tokenization but exclude the last two tokens: <|START_OF_TURN_TOKEN|> and <|CHATBOT_TOKEN|>
    Else if the prompt-completion string starts with <BOS_TOKEN> and ends with <|END_OF_TURN_TOKEN|>:
        we don't add special tokens in tokenization and don't exclude any tokens
    Else (as when we calculate the max sequence length in get_max_sequence_length_in_data):
        we add special tokens in tokenization but don't exclude any tokens
    """
    if (
        rendered_prompt_completion.startswith("<BOS_TOKEN>") and
        rendered_prompt_completion.endswith("<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
    ):
        bos_token_stripped, ignore_last_two_tokens = False, True
    elif (
        rendered_prompt_completion.startswith("<BOS_TOKEN>") and
        rendered_prompt_completion.endswith("<|END_OF_TURN_TOKEN|>")
    ):
        bos_token_stripped, ignore_last_two_tokens = False, False
    else:
        assert (
            not rendered_prompt_completion.startswith("<BOS_TOKEN>") and
            not rendered_prompt_completion.endswith(("<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>", "<|CHATBOT_TOKEN|>"))
        )
        bos_token_stripped, ignore_last_two_tokens = True, False

    return len(tokenizer(rendered_prompt_completion, add_special_tokens=bos_token_stripped)["input_ids"]) - 2 * int(ignore_last_two_tokens)


def get_max_sequence_length_in_data(data_path: str, tokenizer: CohereTokenizerFast) -> int:
    """Get the number of tokens in the longest sequence in the given dataset."""
    column_types = {"prompt": str, "completion": str}
    df, n_dropped_lines = load_and_prepare_csv(data_path, column_types)
    logger.info(
        f"{df.shape[0]} unique samples loaded, "
        f"while {n_dropped_lines} samples dropped as they cannot be decoded as UTF-8 strings"
    )

    max_sequence_length = 0
    for i in df.index:
        rendered_prompt_completion = f'{df.loc[i, "prompt"].rstrip(" ")}{df.loc[i, "completion"].lstrip(" ")}'
        n_tokens = get_n_tokens_in_rendered_prompt_completion(rendered_prompt_completion, tokenizer)
        max_sequence_length = max(max_sequence_length, n_tokens)
    return max_sequence_length


def get_min_possible_max_sequence_length(train_path: str, eval_path: str, tokenizer: CohereTokenizerFast) -> int:
    """
    Get the min possible max sequence length we can use from the CANDIDATE_MAX_SEQUENCE_LENGTHS, where
    the lengths in CANDIDATE_MAX_SEQUENCE_LENGTHS are all powers of two.
    """
    max_sequence_length_in_train = get_max_sequence_length_in_data(train_path, tokenizer)
    max_sequence_length_in_eval = get_max_sequence_length_in_data(eval_path, tokenizer)
    max_sequence_length_in_data = max(max_sequence_length_in_train, max_sequence_length_in_eval)

    logger.info(f"The max sequence length in train and eval data is {max_sequence_length_in_data}")

    for min_possible_max_sequence_length in sorted(CANDIDATE_MAX_SEQUENCE_LENGTHS):
        if min_possible_max_sequence_length > max_sequence_length_in_data:
            return min_possible_max_sequence_length

    # If CANDIDATE_MAX_SEQUENCE_LENGTHS contains the max possible max sequence length the base model can handle,
    # this will never happen, as preprocessing has ensured that all sequences in the data are shorter than that length
    raise DatasetExceedsMaxSequenceLengthError()