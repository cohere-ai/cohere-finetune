import json


def is_valid_chat(chat: dict) -> bool:
    """Check whether the input is a valid chat in the valid format."""
    try:
        assert isinstance(chat, dict) and len(chat) == 1 and isinstance(chat["messages"], list)
        n_user, n_chatbot = 0, 0
        for i, message in enumerate(chat["messages"]):
            assert isinstance(message, dict) and len(message) == 2 and isinstance(message["content"], str)
            if i == 0:
                if message["role"] == "User":
                    n_user += 1
                else:
                    assert message["role"] == "System"
            else:
                if message["role"] == "User":
                    n_user += 1
                else:
                    assert message["role"] == "Chatbot" and n_user > 0
                    n_chatbot += 1
        assert n_user > 0 and n_chatbot > 0
        return True
    except (AssertionError, KeyError):
        return False


def chat_to_str(chat: dict) -> str:
    """
    Convert a valid chat (a dictionary) to a string that serves as the fingerprint of the chat.

    This function assumes its input is a valid chat, i.e., is_valid_chat(chat) must be True.
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

    This function assumes each chat in its input is a valid chat, i.e., all(is_valid_chat(chat) for chat in chats) must be True.
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
    Normalize a list of messages in place by doing the following.
    - If a message uses "System", "SYSTEM", etc. as the value of "role", we change it to "system"
    - If a message uses "User", "USER", etc. as the value of "role", we change it to "user"
    - If a message uses "Chatbot", "CHATBOT", "Assistant", etc. as the value of "role", we change it to "assistant"
    - If a message uses the key "message", we rename it to "content"
    """
    for message in messages:
        if message["role"].lower() == "system":
            message["role"] = "system"
        elif message["role"].lower() == "user":
            message["role"] = "user"
        elif message["role"].lower() in {"chatbot", "assistant"}:
            message["role"] = "assistant"

        if "message" in message:
            message["content"] = message["message"]
            del message["message"]
