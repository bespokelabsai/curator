import os
from typing import Optional

import pytest
from datasets import Dataset
from pydantic import BaseModel

from bespokelabs.curator import Prompter
from bespokelabs.curator.prompter.base_prompter import BasePrompter


class MockResponseFormat(BaseModel):
    """Mock response format for testing."""

    message: str
    confidence: Optional[float] = None


@pytest.fixture
def prompter() -> Prompter:
    """Create a Prompter instance for testing.

    Returns:
        PromptCaller: A configured prompt caller instance.
    """

    def prompt_func(row):
        return {
            "user_prompt": f"Context: {row['context']} Answer this question: {row['question']}",
            "system_prompt": "You are a helpful assistant.",
        }

    return Prompter(
        model_name="gpt-4o-mini",
        prompt_func=prompt_func,
        response_format=MockResponseFormat,
    )


@pytest.mark.test
def test_completions(prompter: Prompter, tmp_path):
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

    result_dataset = prompter(dataset)
    result_dataset = result_dataset.to_huggingface()

    # Assertions
    assert len(result_dataset) == len(dataset)
    assert "message" in result_dataset.column_names
    assert "confidence" in result_dataset.column_names


@pytest.mark.test
def test_single_completion_batch(prompter: Prompter):
    """Test that a single completion works with batch=True.

    Args:
        prompter: Fixture providing a configured Prompter instance.
    """

    # Create a prompter with batch=True
    def simple_prompt_func():
        return [
            {
                "role": "user",
                "content": "Write a test message",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ]

    batch_prompter = Prompter(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
        batch=True,
    )

    # Get single completion
    result = batch_prompter()

    # Assertions
    assert isinstance(result, MockResponseFormat)
    assert hasattr(result, "message")
    assert hasattr(result, "confidence")


@pytest.mark.test
def test_single_completion_no_batch(prompter: Prompter):
    """Test that a single completion works without batch parameter.

    Args:
        prompter: Fixture providing a configured Prompter instance.
    """

    # Create a prompter without batch parameter
    def simple_prompt_func():
        return [
            {
                "role": "user",
                "content": "Write a test message",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ]

    non_batch_prompter = Prompter(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
    )

    # Get single completion
    result = non_batch_prompter()

    # Assertions
    assert isinstance(result, MockResponseFormat)
    assert hasattr(result, "message")
    assert hasattr(result, "confidence")


class TestClassBasedPrompter:
    """Test cases for class-based Prompter implementation."""

    class CustomPrompter(BasePrompter):
        """Test prompter implementation using class-based approach."""

        def prompt_func(self, row=None):
            if row is None:
                return {
                    "user_prompt": "Write a test message",
                    "system_prompt": "You are a helpful assistant.",
                }
            return {
                "user_prompt": f"Context: {row['context']} Answer this question: {row['question']}",
                "system_prompt": "You are a helpful assistant.",
            }

        def parse_func(self, row, response):
            # Custom parsing that adds a prefix to the message
            if isinstance(response, MockResponseFormat):
                return MockResponseFormat(
                    message=f"Parsed: {response.message}", confidence=response.confidence
                )
            return response

    @pytest.fixture
    def class_based_prompter(self):
        """Create a class-based Prompter instance for testing."""
        return self.CustomPrompter(
            model_name="gpt-4o-mini",
            response_format=MockResponseFormat,
        )

    def test_class_based_completion(self, class_based_prompter, tmp_path):
        """Test that class-based prompter processes a dataset correctly."""
        test_data = {
            "context": ["Test context 1", "Test context 2"],
            "question": ["What is 1+1?", "What is 2+2?"],
        }
        dataset = Dataset.from_dict(test_data)

        os.environ["BELLA_CACHE_DIR"] = str(tmp_path)
        result_dataset = class_based_prompter(dataset)
        result_dataset = result_dataset.to_huggingface()

        assert len(result_dataset) == len(dataset)
        assert "message" in result_dataset.column_names
        assert "confidence" in result_dataset.column_names
        # Verify our custom parse_func was applied
        assert all(msg.startswith("Parsed: ") for msg in result_dataset["message"])

    def test_class_based_single_completion(self, class_based_prompter):
        """Test that class-based prompter works for single completions."""
        result = class_based_prompter()

        assert isinstance(result, MockResponseFormat)
        assert hasattr(result, "message")
        assert hasattr(result, "confidence")
        assert result.message.startswith("Parsed: ")

    def test_mixed_approach(self):
        """Test using class-based prompter with function-based parse_func."""

        def custom_parse_func(row, response):
            if isinstance(response, MockResponseFormat):
                return MockResponseFormat(
                    message=f"Function parsed: {response.message}", confidence=response.confidence
                )
            return response

        prompter = self.CustomPrompter(
            model_name="gpt-4o-mini",
            response_format=MockResponseFormat,
            parse_func=custom_parse_func,  # Override class-based parse_func
        )

        result = prompter()
        assert isinstance(result, MockResponseFormat)
        assert result.message.startswith("Function parsed: ")

    def test_invalid_prompt_func(self):
        """Test that invalid prompt_func signature raises ValueError."""

        class InvalidPrompter(BasePrompter):
            def prompt_func(self, row, extra_arg):  # Invalid: too many parameters
                return {"prompt": f"Process {row}"}

        with pytest.raises(ValueError) as exc_info:
            InvalidPrompter(model_name="gpt-4o-mini")
        assert "prompt_func must take one argument or less" in str(exc_info.value)

    def test_invalid_parse_func(self):
        """Test that invalid parse_func signature raises ValueError."""

        class InvalidParsePrompter(BasePrompter):
            def prompt_func(self, row=None):
                return {"prompt": "test"}

            def parse_func(self, response):  # Invalid: too few parameters
                return response

        with pytest.raises(ValueError) as exc_info:
            InvalidParsePrompter(model_name="gpt-4o-mini")
        assert "parse_func must take exactly 2 arguments" in str(exc_info.value)
