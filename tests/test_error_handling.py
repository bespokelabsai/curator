import os
import pytest
from unittest.mock import patch, MagicMock
from bespokelabs.curator.prompter.prompter import Prompter
from bespokelabs.curator.request_processor.openai_online_request_processor import (
    OpenAIOnlineRequestProcessor,
)
from pydantic import BaseModel, Field
from typing import List
from datasets import Dataset


class TestResponse(BaseModel):
    """Test response format."""

    message: str = Field(description="A test message")


@pytest.mark.asyncio
async def test_model_access_error_handling():
    """Test that model access errors are handled properly with clear messages."""
    # Mock API response for model access error
    mock_response = {
        "error": {
            "message": "You do not have access to model gpt-4o",
            "type": "insufficient_quota",
            "param": None,
            "code": None,
        }
    }

    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__.return_value.json.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value.status = 401

    processor = OpenAIOnlineRequestProcessor(model="gpt-4o")

    with pytest.raises(ValueError) as exc_info:
        await processor.call_single_request(
            request=MagicMock(
                api_specific_request={"model": "gpt-4o", "messages": []},
                generic_request=MagicMock(),
            ),
            session=mock_session,
            status_tracker=MagicMock(),
        )

    assert "API key does not have access to model 'gpt-4o'" in str(exc_info.value)
    assert "Please check your API key permissions" in str(exc_info.value)


def test_function_based_error_handling():
    """Test error handling with function-based Prompter usage."""

    def prompt_func(row=None):
        return {"role": "user", "content": "test prompt"}

    def parse_func(row, response):
        return {"result": response.message}

    # Mock environment with invalid API key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid_key"}):
        prompter = Prompter(
            model_name="gpt-4o",
            prompt_func=prompt_func,
            parse_func=parse_func,
            response_format=TestResponse,
        )

        with patch(
            "bespokelabs.curator.request_processor.openai_online_request_processor.OpenAIOnlineRequestProcessor.call_single_request"
        ) as mock_call:
            mock_call.side_effect = ValueError("API key does not have access to model 'gpt-4o'")

            with pytest.raises(ValueError) as exc_info:
                # Following poem.py pattern: when no dataset is provided, it uses None
                prompter()

            assert "All API requests failed" in str(exc_info.value)
            assert "model access permissions" in str(exc_info.value)
            assert "verify your API key" in str(exc_info.value)


def test_backward_compatibility():
    """Test that error handling changes don't break existing functionality."""

    def prompt_func(row=None):
        return {"role": "user", "content": "test prompt"}

    def parse_func(row, response):
        return {"result": response.message}

    # Use valid API key from environment
    with patch.dict(os.environ, {"OPENAI_API_KEY": os.getenv("openai_key_3")}):
        with patch(
            "bespokelabs.curator.request_processor.openai_online_request_processor.OpenAIOnlineRequestProcessor.call_single_request"
        ) as mock_call:
            mock_call.return_value = MagicMock(
                response_message=TestResponse(message="test response"), response_errors=None
            )

            prompter = Prompter(
                model_name="gpt-3.5-turbo",
                prompt_func=prompt_func,
                parse_func=parse_func,
                response_format=TestResponse,
            )

            # Following poem.py pattern: when no dataset is provided, it uses None
            result = prompter()
            assert result is not None
            assert isinstance(result, Dataset)
