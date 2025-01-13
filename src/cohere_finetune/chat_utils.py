import json


def is_valid_training_sample_chat(chat: dict) -> bool:
    """
    Check whether it is a valid chat we can use as a training sample.

    If the first message is a System message, it will be regarded as preamble. Except for this, a System message
    is regarded as equivalent to a User message (they are exchangeable).
    """
    try:
        assert isinstance(chat, dict) and len(chat) == 1 and isinstance(chat["messages"], list)
        n_system_role_curr_turn, n_user_role_curr_turn = 0, 0
        n_chatbot_role_total = 0
        for i, message in enumerate(chat["messages"]):
            assert isinstance(message, dict) and len(message) == 2 and isinstance(message["content"], str)
            if message["role"] == "System":
                n_system_role_curr_turn += 1
            elif message["role"] == "User":
                n_user_role_curr_turn += 1
            else:
                assert message["role"] == "Chatbot"
                assert n_user_role_curr_turn > 0 or (n_system_role_curr_turn > 0 and i > 1)
                n_system_role_curr_turn, n_user_role_curr_turn = 0, 0
                n_chatbot_role_total += 1
        assert n_chatbot_role_total > 0
        return True
    except (AssertionError, KeyError):
        return False


def is_valid_inference_input_chat(chat: dict) -> bool:
    """
    Check whether it is a valid chat we can use as an inference input.

    It is a valid chat we can use as an inference input, if and only if
    it is a valid chat we can use as a training sample after we append a Chatbot message to it.
    """
    try:
        assert isinstance(chat, dict) and len(chat) == 1 and isinstance(chat["messages"], list)
        assert is_valid_training_sample_chat({"messages": chat["messages"] + [{"role": "Chatbot", "content": ""}]})
        return True
    except (AssertionError, KeyError):
        return False


def chat_to_str(chat: dict) -> str:
    """
    Convert a valid chat (a dictionary) to a string that serves as the fingerprint of the chat.

    This function assumes its input is a valid chat for training, i.e.,
    is_valid_training_sample_chat(chat) must be True.
    """
    # Create a normalized chat, where the order of keys in each message is always: "role" and then "content" (if the
    # only difference between two chats is the order of keys in each message, we want to regard them as the same chat)
    normalized_chat = {
        "messages": [{"role": message["role"], "content": message["content"]} for message in chat["messages"]]
    }
    return json.dumps(normalized_chat)


def dedupe_chats(chats: list[dict]) -> list[dict]:
    """
    Deduplicate a list of chats, while preserving the original order of the chats.

    This function assumes each chat in its input is a valid chat for training, i.e.,
    all(is_valid_training_sample_chat(chat) for chat in chats) must be True.
    """
    seen_chat_strings = set()
    deduped_chats = []
    for chat in chats:
        chat_string = chat_to_str(chat)
        if chat_string not in seen_chat_strings:
            seen_chat_strings.add(chat_string)
            deduped_chats.append(chat)
    return deduped_chats


def normalize_messages(messages: list[dict]) -> None:
    """
    To convert a list of messages into a valid input for Liquid template, normalize the messages in place by doing the following.
    - If a message uses "system", "SYSTEM", etc. as the value of "role", we change it to "System"
    - If a message uses "user", "USER", etc. as the value of "role", we change it to "User"
    - If a message uses "assistant", "ASSISTANT", "chatbot", "CHATBOT", etc. as the value of "role", we change it to "Chatbot"
    - If a message uses the key "content", we rename it to "message"
    """
    for message in messages:
        if message["role"].lower() == "system":
            message["role"] = "System"
        elif message["role"].lower() == "user":
            message["role"] = "User"
        elif message["role"].lower() in {"assistant", "chatbot"}:
            message["role"] = "Chatbot"

        if "content" in message:
            message["message"] = message["content"]
            del message["content"]
