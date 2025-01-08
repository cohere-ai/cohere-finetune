from chat_utils import is_valid_chat


def test_is_valid_chat() -> None:
    """Test the function is_valid_chat."""

    # The following chats are valid.
    assert is_valid_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "User", "content": ""},
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "User", "content": ""},
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "User", "content": ""},
                {"role": "User", "content": ""},
            ]
        }
    )
    assert is_valid_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )

    # The following chats are invalid.
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Assistant", "content": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "User", "message": ""},
                {"role": "Chatbot", "message": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": 0},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "User", "content": ""},
            ]
        }
    )
    assert not is_valid_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
