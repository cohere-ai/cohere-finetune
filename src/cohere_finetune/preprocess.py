import pandas as pd
import random
from chat_utils import dedupe_chats, is_valid_chat
from liquid import Liquid
from tokenizer_utils import get_n_tokens_in_rendered_prompt_completion
from tqdm import tqdm
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
from utils import load_file, logger, save_file


class Turn:
    """Turn stores all the information about a turn in the chat."""

    def __init__(self) -> None:
        """Initialize Turn."""
        self.user_messages: list[dict] = []
        self.chatbot_message: dict | None = None
        self.n_tokens: int = 0  # Number of tokens in all rendered user messages and the rendered chatbot message


class ChatContext:
    """ChatContext stores all the information about a chat."""

    def __init__(self) -> None:
        """Initialize ChatContext."""
        self.preamble: str = ""
        self.turns: list[Turn] = []
        self.n_tokens: int = 0  # Number of tokens in the rendered preamble and all rendered messages of all turns


def preprocess(
    input_train_path: str,
    input_eval_path: str,
    output_train_path: str,
    output_eval_path: str,
    eval_percentage: float,
    template: str,
    max_sequence_length: int,
    tokenizer: CohereTokenizerFast,
) -> None:
    """Preprocess the finetuning dataset by doing train eval split (if needed) and putting the texts in template."""
    if not input_eval_path:
        raw_chats = load_file(input_train_path)
        chats = get_valid_deduped_chats(raw_chats)
        train_chats, eval_chats = train_eval_split(chats, eval_percentage)
    else:
        raw_train_chats, raw_eval_chats = load_file(input_train_path), load_file(input_eval_path)
        train_chats, eval_chats = get_valid_deduped_chats(raw_train_chats), get_valid_deduped_chats(raw_eval_chats)

    liquid_template = Liquid(template, from_file=False)
    preprocess_chats(train_chats, output_train_path, liquid_template, max_sequence_length, tokenizer)
    preprocess_chats(eval_chats, output_eval_path, liquid_template, max_sequence_length, tokenizer)


def get_valid_deduped_chats(chats: list[dict]) -> list[dict]:
    """Get all the valid and deduped chats from a given list of chats."""
    valid_chats = [chat for chat in chats if is_valid_chat(chat)]
    logger.info(
        f"{len(valid_chats)} valid chats loaded, "
        f"while {len(chats) - len(valid_chats)} chats dropped as they are not in a valid format"
    )

    valid_deduped_chats = dedupe_chats(valid_chats)
    logger.info(
        f"{len(valid_deduped_chats)} unique valid chats loaded, "
        f"while {len(valid_chats) - len(valid_deduped_chats)} valid chats dropped as they are duplicates"
    )

    if len(valid_deduped_chats) < 1:
        raise RuntimeError("There must be at least one valid chat")

    return valid_deduped_chats


def train_eval_split(chats: list[dict], eval_percentage: float) -> tuple[list[dict], list[dict]]:
    """Randomly split the chats into a training set and an evaluation set by eval_percentage."""
    n = len(chats)
    n_eval = int(n * eval_percentage)
    n_train = n - n_eval
    if n_train < 1 or n_eval < 1:
        raise RuntimeError("There must be at least one sample in the training set and the evaluation set, respectively")

    random.seed(42)
    eval_indices = set(random.sample(range(n), k=n_eval))
    train_chats, eval_chats = [], []
    for i in range(n):
        if i in eval_indices:
            eval_chats.append(chats[i])
        else:
            train_chats.append(chats[i])

    logger.info(f"Data split into a training set with {len(train_chats)} chats and an evaluation set with {len(eval_chats)} chats")
    return train_chats, eval_chats


def preprocess_chats(
    chats: list[dict],
    output_path: str,
    liquid_template: Liquid,
    max_sequence_length: int,
    tokenizer: CohereTokenizerFast,
) -> None:
    """Preprocess a list of chats and save the preprocessed data as a prompt-completion CSV file."""
    preprocessed_chats = []
    for i in tqdm(range(len(chats))):
        preprocessed_chats.extend(preprocess_chat(chats[i]["messages"], liquid_template, max_sequence_length, tokenizer))

    df = pd.DataFrame(preprocessed_chats, columns=["prompt", "completion"])
    save_file(df, output_path)


def preprocess_chat(
    chat: list[dict],
    liquid_template: Liquid,
    max_sequence_length: int,
    tokenizer: CohereTokenizerFast,
) -> list[list[str]]:
    """
    Preprocess a single chat.

    A chat is a sequence of messages such as: Preamble, User1, Chatbot1, User2, User3, Chatbot2,
    where there are two turns in this example: [User1, Chatbot1], [User2, User3, Chatbot2]

    If the number of tokens of Preamble + User2 + User3 + Chatbot2 > max_sequence_length,
        we drop the turn [User2, User3, Chatbot2]
    Else if the number of tokens of Preamble + User1 + Chatbot1 + User2 + User3 + Chatbot2 > max_sequence_length,
        we drop the previous turns one by one from left to right until the number of tokens <= max_sequence_length,
        so the final (prompt, completion) is (Preamble + User2 + User3, Chatbot2)
    """
    preprocessed_data = []
    chat_context = ChatContext()
    start_index = 0

    first_message = chat[0]
    if first_message["role"] == "System":
        chat_context.preamble = first_message["content"]
        start_index = 1

    curr_turn = Turn()

    for i in range(start_index, len(chat)):
        message = chat[i]
        if message["role"] != "Chatbot":
            curr_turn.user_messages.append(message)
        else:
            curr_turn.chatbot_message = message

            rendered_turn = render_turn(liquid_template, curr_turn, preamble="")
            curr_turn.n_tokens = get_n_tokens_in_rendered_prompt_completion(rendered_turn, tokenizer)

            if chat_context.preamble:
                rendered_turn_and_preamble = render_turn(liquid_template, curr_turn, preamble=chat_context.preamble)
                n_tokens_in_curr_turn_and_preamble = get_n_tokens_in_rendered_prompt_completion(rendered_turn_and_preamble, tokenizer)
            else:
                n_tokens_in_curr_turn_and_preamble = curr_turn.n_tokens
            if n_tokens_in_curr_turn_and_preamble > max_sequence_length:
                curr_turn = Turn()
                continue

            chat_context.turns.append(curr_turn)

            rendered_chat_context = render_chat_context(liquid_template, chat_context, include_completion=True)
            chat_context.n_tokens = get_n_tokens_in_rendered_prompt_completion(rendered_chat_context, tokenizer)

            if chat_context.n_tokens > max_sequence_length:
                chat_context.turns, chat_context.n_tokens = pop_turns_to_fit_max_sequence_length(chat_context, max_sequence_length)

            rendered_prompt = render_chat_context(liquid_template, chat_context, include_completion=False)
            preprocessed_data.append([strip_bos_token(rendered_prompt), curr_turn.chatbot_message["content"]])

            curr_turn = Turn()

    return preprocessed_data


def strip_bos_token(prompt: str) -> str:
    """Strip the first <BOS_TOKEN> token from the prompt."""
    prompt = prompt.replace("<BOS_TOKEN>", "", 1)
    return prompt


def render_turn(liquid_template: Liquid, turn: Turn, preamble: str) -> str:
    """Render a Turn to a string by the template."""
    messages = []
    for user_message in turn.user_messages:
        messages.append({"role": user_message["role"], "message": user_message["content"]})
    messages.append({"role": turn.chatbot_message["role"], "message": turn.chatbot_message["content"]})

    return liquid_template.render(
        safety_mode="NONE",
        preamble=preamble,
        messages=messages,
    )


def render_chat_context(liquid_template: Liquid, chat_context: ChatContext, include_completion: bool) -> str:
    """Render a ChatContext to a string by the template."""
    messages = []
    for i, turn in enumerate(chat_context.turns):
        for user_message in turn.user_messages:
            messages.append({"role": user_message["role"], "message": user_message["content"]})
        if i < len(chat_context.turns) - 1 or include_completion:
            messages.append({"role": turn.chatbot_message["role"], "message": turn.chatbot_message["content"]})

    return liquid_template.render(
        safety_mode="NONE",
        preamble=chat_context.preamble if chat_context.preamble else "",
        messages=messages,
    )


def pop_turns_to_fit_max_sequence_length(chat_context: ChatContext, max_sequence_length: int) -> tuple[list[Turn], int]:
    """Pop turns one by one from left to right until the total number of tokens <= max_sequence_length."""
    n_tokens_removed = 0
    for i in range(len(chat_context.turns) - 1):
        n_tokens_removed += chat_context.turns[i].n_tokens
        n_tokens_retained = chat_context.n_tokens - n_tokens_removed
        if n_tokens_retained <= max_sequence_length:
            return chat_context.turns[(i + 1):], n_tokens_retained
    raise RuntimeError('This function should be called only when "preamble + current turn" fits in the max sequence length')
