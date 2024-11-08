import os
from typing import Optional

import pytest
from datasets import Dataset
from pydantic import BaseModel

from bespokelabs.curator import Prompter


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
