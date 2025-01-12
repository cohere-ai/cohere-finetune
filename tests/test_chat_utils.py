from chat_utils import is_valid_training_sample_chat


def test_is_valid_training_sample_chat() -> None:
    """Test the function is_valid_training_sample_chat."""

    # The following are valid chats we can use as training samples.
    assert is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert is_valid_training_sample_chat(
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
    assert is_valid_training_sample_chat(
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
    assert is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "User", "content": ""},
                {"role": "User", "content": ""},
            ]
        }
    )
    assert is_valid_training_sample_chat(
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
    assert is_valid_training_sample_chat(
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
    assert is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )

    # The following are invalid chats we cannot use as training samples.
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Assistant", "content": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "User", "message": ""},
                {"role": "Chatbot", "message": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "User", "content": 0},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
                {"role": "Chatbot", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "User", "content": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "User", "content": ""},
            ]
        }
    )
    assert not is_valid_training_sample_chat(
        {
            "messages": [
                {"role": "System", "content": ""},
                {"role": "Chatbot", "content": ""},
            ]
        }
    )
