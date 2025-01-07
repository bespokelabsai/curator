import os
from typing import Any, Dict, Optional, Union
from unittest.mock import patch, MagicMock

import pytest
from datasets import Dataset
from pydantic import BaseModel

from bespokelabs.curator import LLM

# Type alias for input/output types
_DictOrBaseModel = Union[Dict[str, Any], BaseModel]


class MockResponseFormat(BaseModel):
    """Mock response format for testing."""

    message: str
    confidence: Optional[float] = None


@pytest.fixture
def prompter() -> LLM:
    """Create a Prompter instance for testing.

    Returns:
        PromptCaller: A configured prompt caller instance.
    """

    def prompt_func(row):
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"Context: {row['context']} Answer this question: {row['question']}",
            },
        ]

    return LLM(
        model_name="gpt-4o-mini",
        prompt_func=prompt_func,
        response_format=MockResponseFormat,
    )


@pytest.mark.test
def test_completions(prompter: LLM, tmp_path):
    """Test that completions processes a dataset correctly.

    Args:
        prompter: Fixture providing a configured Prompter instance.
        tmp_path: Pytest fixture providing temporary directory.
    """
    # Create a simple test dataset
    test_data = {
        "context": ["Test context 1", "Test context 2"],
        "question": ["What is 1+1?", "What is 2+2?"],
    }
    dataset = Dataset.from_dict(test_data)

    # Set up temporary cache directory
    os.environ["BELLA_CACHE_DIR"] = str(tmp_path)

    # Mock OpenAI API response
    mock_response = {
        "choices": [{"message": {"content": "1 + 1 equals 2."}, "finish_reason": "stop"}]
    }

    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        # Process dataset and get responses
        result_dataset = prompter(dataset)

        # Verify the dataset structure
        assert len(result_dataset) == len(dataset)
        assert "response" in result_dataset.column_names
        # Check that each response has the required fields
        for row in result_dataset:
            response = row["response"]
            if isinstance(response, dict):
                assert "message" in response
                assert "confidence" in response
            else:
                assert hasattr(response, "message")
                assert hasattr(response, "confidence")


@pytest.mark.test
def test_single_completion_batch(prompter: LLM):
    """Test that a single completion works with batch=True.

    Args:
        prompter: Fixture providing a configured Prompter instance.
    """

    # Create a prompter with batch=True
    def simple_prompt_func(row: _DictOrBaseModel) -> _DictOrBaseModel:
        """Generate a simple prompt for testing.

        Args:
            row: The input row (unused in this test)

        Returns:
            A list of messages for the LLM
        """
        messages = [
            {
                "role": "user",
                "content": "Write a test message",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ]
        return {"messages": messages}

    # Set dummy OpenAI API key for testing
    os.environ["OPENAI_API_KEY"] = "test-key"
    batch_prompter = LLM(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
        batch=True,
    )

    # Mock response data
    mock_dataset = Dataset.from_list(
        [{"response": {"message": "This is a test message.", "confidence": 0.9}}]
    )

    # Mock the run method of OpenAIBatchRequestProcessor
    with patch(
        "bespokelabs.curator.request_processor.batch.openai_batch_request_processor.OpenAIBatchRequestProcessor.run",
        return_value=mock_dataset,
    ):
        # Get single completion
        result = batch_prompter()

        # Assertions
        assert isinstance(result, Dataset)
        assert len(result) == 1
        assert isinstance(result[0]["response"], dict)
        assert result[0]["response"]["message"] == "This is a test message."
        assert result[0]["response"]["confidence"] == 0.9


@pytest.mark.test
def test_single_completion_no_batch(prompter: LLM):
    """Test that a single completion works without batch parameter.

    Args:
        prompter: Fixture providing a configured Prompter instance.
    """

    # Create a prompter without batch parameter
    def simple_prompt_func(row: _DictOrBaseModel) -> _DictOrBaseModel:
        """Generate a simple prompt for testing.

        Args:
            row: The input row (unused in this test)

        Returns:
            A list of messages for the LLM
        """
        messages = [
            {
                "role": "user",
                "content": "Write a test message",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ]
        return {"messages": messages}

    # Set dummy OpenAI API key for testing
    os.environ["OPENAI_API_KEY"] = "test-key"
    non_batch_prompter = LLM(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
    )

    # Mock response data
    mock_dataset = Dataset.from_list(
        [{"response": {"message": "This is a test message.", "confidence": 0.9}}]
    )

    # Mock the run method of OpenAIOnlineRequestProcessor
    with patch(
        "bespokelabs.curator.request_processor.online.openai_online_request_processor.OpenAIOnlineRequestProcessor.run",
        return_value=mock_dataset,
    ):
        # Get single completion
        result = non_batch_prompter()

        # Assertions
        assert isinstance(result, Dataset)
        assert len(result) == 1
        assert isinstance(result[0]["response"], dict)
        assert result[0]["response"]["message"] == "This is a test message."
        assert result[0]["response"]["confidence"] == 0.9
