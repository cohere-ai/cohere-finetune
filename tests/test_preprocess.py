from consts import CHAT_PROMPT_TEMPLATE_CMD_R_08_2024
from liquid import Liquid
from preprocess import preprocess_chat
from tokenizer_utils import create_and_prepare_tokenizer


def test_preprocess_chat() -> None:
    """Test the function preprocess_chat."""
    liquid_template = Liquid(CHAT_PROMPT_TEMPLATE_CMD_R_08_2024, from_file=False)
    tokenizer = create_and_prepare_tokenizer("CohereForAI/c4ai-command-r-08-2024")

    # Test case 1
    chat = [
        {"role": "User", "content": "This is user message one"},
        {"role": "Chatbot", "content": "This is chatbot message one"},
        {"role": "User", "content": "This is user message two"},
        {"role": "User", "content": "This is user message three"},
        {"role": "Chatbot", "content": "This is chatbot message two"},
    ]

    prompt_turn1 = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    completion_turn1 = "This is chatbot message one"

    prompt_turn1_and_turn2 = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>This is chatbot message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message two<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message three<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    prompt_turn2 = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message two<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message three<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    completion_turn2 = "This is chatbot message two"

    preprocessed_data = preprocess_chat(chat, liquid_template=liquid_template, max_sequence_length=16, tokenizer=tokenizer)
    assert preprocessed_data == []

    preprocessed_data = preprocess_chat(chat, liquid_template=liquid_template, max_sequence_length=17, tokenizer=tokenizer)
    assert preprocessed_data == [[prompt_turn1, completion_turn1]]

    preprocessed_data = preprocess_chat(chat, liquid_template=liquid_template, max_sequence_length=24, tokenizer=tokenizer)
    assert preprocessed_data == [[prompt_turn1, completion_turn1]]

    preprocessed_data = preprocess_chat(chat, liquid_template=liquid_template, max_sequence_length=25, tokenizer=tokenizer)
    assert preprocessed_data == [[prompt_turn1, completion_turn1], [prompt_turn2, completion_turn2]]

    preprocessed_data = preprocess_chat(chat, liquid_template=liquid_template, max_sequence_length=40, tokenizer=tokenizer)
    assert preprocessed_data == [[prompt_turn1, completion_turn1], [prompt_turn2, completion_turn2]]

    preprocessed_data = preprocess_chat(chat, liquid_template=liquid_template, max_sequence_length=41, tokenizer=tokenizer)
    assert preprocessed_data == [[prompt_turn1, completion_turn1], [prompt_turn1_and_turn2, completion_turn2]]

    # Test case 2 (tool use data)
    chat = [
        {"role": "System", "content": "This is system message one"},
        {"role": "User", "content": "This is user message one"},
        {"role": "Chatbot", "content": "This is chatbot message one"},
        {"role": "System", "content": "This is system message two"},
        {"role": "Chatbot", "content": "This is chatbot message two"},
        {"role": "System", "content": "This is system message three"},
        {"role": "Chatbot", "content": "This is chatbot message three"},
    ]

    prompt_turn1 = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># User Preamble\nThis is system message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    completion_turn1 = "This is chatbot message one"

    prompt_turn1_and_turn2 = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># User Preamble\nThis is system message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>This is chatbot message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>This is system message two<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    completion_turn2 = "This is chatbot message two"

    prompt_turn1_and_turn2_and_turn3 = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># User Preamble\nThis is system message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is user message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>This is chatbot message one<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>This is system message two<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>This is chatbot message two<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>This is system message three<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    completion_turn3 = "This is chatbot message three"

    preprocessed_data = preprocess_chat(chat, liquid_template=liquid_template, max_sequence_length=100, tokenizer=tokenizer)
    assert preprocessed_data == [[prompt_turn1, completion_turn1], [prompt_turn1_and_turn2, completion_turn2], [prompt_turn1_and_turn2_and_turn3, completion_turn3]]
