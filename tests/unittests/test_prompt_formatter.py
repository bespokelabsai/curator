import json

import pytest
from pydantic import BaseModel, ValidationError

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter, _validate_messages


def test_validate_messages_valid():
    """Tests that valid message formats pass validation."""
    valid_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    # Should not raise any exceptions
    _validate_messages(valid_messages)


def test_validate_messages_invalid_format():
    """Tests that invalid message formats raise appropriate errors."""
    # Test non-dict message
    with pytest.raises(ValueError, match="must be a dictionary"):
        _validate_messages([["role", "content"]])

    # Test missing required keys
    with pytest.raises(ValueError, match="must contain 'role' and 'content' keys"):
        _validate_messages([{"role": "user"}])

    # Test invalid role
    with pytest.raises(ValueError, match="must be one of: assistant, system, user"):
        _validate_messages([{"role": "invalid", "content": "test"}])


class TestResponse(BaseModel):
    text: str


def test_prompt_formatter_create_generic_request():
    """Tests that PromptFormatter correctly creates GenericRequest objects."""
    # Test with string prompt
    formatter = PromptFormatter(model_name="test-model", prompt_func=lambda x: "Hello", response_format=TestResponse)
    assert formatter.response_to_response_format(json.dumps({"text": "response"})) == TestResponse(text="response")
    assert formatter.parse_response_message(json.dumps({"text": "response"})) == ({"text": "response"}, None)

    request = formatter.create_generic_request({"input": "test"}, 0)
    with pytest.raises(json.JSONDecodeError):
        formatter.response_to_response_format("'text': 'response'}")
    with pytest.raises(ValidationError):
        formatter.response_to_response_format(json.dumps({"other": "response"}))
    formatter.parse_response_message("'text': 'response'}")

    assert request.model == "test-model"
    assert request.messages == [{"role": "user", "content": "Hello"}]
    assert request.original_row == {"input": "test"}
    assert request.original_row_idx == 0
    assert request.response_format is not None

    # Test with message list prompt
    formatter = PromptFormatter(
        model_name="test-model",
        prompt_func=lambda x: [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    )

    request = formatter.create_generic_request({"input": "test"}, 1)

    assert len(request.messages) == 2
    assert request.messages[0]["role"] == "system"
    assert request.messages[1]["role"] == "user"
    assert request.original_row_idx == 1


def test_prompt_formatter_invalid_prompt_func():
    """Tests that PromptFormatter raises errors for invalid prompt functions."""
    # Test prompt function with too many parameters
    with pytest.raises(ValueError, match="must have 0 or 1 arguments"):
        PromptFormatter(model_name="test", prompt_func=lambda x, y: "test").create_generic_request({}, 0)

    # Test invalid prompt function return type
    match = "The return value of the `prompt` method <class 'dict'> did not match the expected format"
    with pytest.raises(ValueError, match=match):
        PromptFormatter(model_name="test", prompt_func=lambda x: {"invalid": "format"}).create_generic_request({}, 0)
