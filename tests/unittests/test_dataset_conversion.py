import pytest
from datasets import Dataset

from bespokelabs.curator.constants import _INTERNAL_PROMPT_KEY
from bespokelabs.curator.llm.llm import _convert_to_dataset, _is_message_list


@pytest.fixture
def single_message():
    return [{"role": "user", "content": "test"}]


@pytest.fixture
def conversation():
    return [{"role": "user", "content": "test"}, {"role": "assistant", "content": "test"}]


@pytest.fixture
def conversation_with_system():
    return [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}, {"role": "assistant", "content": "test"}]


def test_is_message_list(single_message, conversation, conversation_with_system):
    assert _is_message_list(single_message)
    assert _is_message_list(conversation)
    assert _is_message_list(conversation_with_system)
    assert not _is_message_list([{"col": "row"}])


def test_convert_to_dataset(single_message, conversation, conversation_with_system):
    assert _convert_to_dataset("test").to_list() == [{_INTERNAL_PROMPT_KEY: "test"}]
    assert _convert_to_dataset(single_message).to_list() == [{_INTERNAL_PROMPT_KEY: single_message}]
    assert _convert_to_dataset(conversation).to_list() == [{_INTERNAL_PROMPT_KEY: conversation}]
    assert _convert_to_dataset(conversation_with_system).to_list() == [{_INTERNAL_PROMPT_KEY: conversation_with_system}]
    assert _convert_to_dataset([conversation, conversation_with_system]).to_list() == [
        {_INTERNAL_PROMPT_KEY: conversation},
        {_INTERNAL_PROMPT_KEY: conversation_with_system},
    ]
    assert _convert_to_dataset(Dataset.from_list([{"prompt": "test"}])).to_list() == [{"prompt": "test"}]
    assert _convert_to_dataset([{"new_prompt": "test"}]).to_list() == [{"new_prompt": "test"}]
