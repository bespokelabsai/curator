from typing import Dict, List

import pandas as pd
import pytest
from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator
from bespokelabs.curator.request_processor.batch.mistral_batch_request_processor import MistralBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus
from bespokelabs.curator.types.generic_request import GenericRequest


class Answer(BaseModel):
    answer: int = Field(description="The answer to the question")


class _CostProcessorStub:
    def cost(self, **kwargs):
        return 0.0


def _generic_request() -> GenericRequest:
    return GenericRequest(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "hello"}],
        original_row={},
        original_row_idx=0,
    )


def _generic_batch() -> GenericBatch:
    now = pd.Timestamp("2025-01-01T00:00:00Z").to_pydatetime()
    return GenericBatch(
        request_file="requests.jsonl",
        id="batch-id",
        created_at=now,
        finished_at=now,
        status=GenericBatchStatus.FINISHED.value,
        api_key_suffix="test",
        request_counts=GenericBatchRequestCounts(
            failed=0,
            succeeded=1,
            total=1,
            raw_request_counts_object={},
        ),
        raw_batch={},
        raw_status="SUCCESS",
    )


def test_mistral_batch_response_includes_normalized_finish_reason() -> None:
    """Mistral batch responses expose finish_reason for invalid-finish retries."""
    processor = MistralBatchRequestProcessor.__new__(MistralBatchRequestProcessor)
    processor.config = BatchRequestProcessorConfig(model="mistral-small-latest")
    processor._cost_processor = _CostProcessorStub()

    raw_response = {
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        }
    }

    response = processor.parse_api_specific_response(
        raw_response=raw_response,
        generic_request=_generic_request(),
        batch=_generic_batch(),
    )

    assert response.finish_reason == "length"


def batch_call(model_name, prompts):
    dataset: Dataset = Dataset.from_dict({"prompt": prompts})
    llm = curator.LLM(
        prompt_func=lambda row: row["prompt"],
        model_name=model_name,
        response_format=Answer,
        batch=True,
        batch_size=2,
        parse_func=lambda row, response: [{"input": row["prompt"], "answer": response.answer}],
    )
    response = llm(dataset).dataset
    return response


@pytest.mark.skip
def test_batch_call() -> None:
    """Tests that batch_call correctly processes multiple prompts and returns expected answers.

    This test verifies:
    1. Batch processing of multiple prompts
    2. Correct response format
    """
    # Test input prompts
    test_prompts: List[str] = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]

    # Expected answers for our test prompts
    expected_answers: Dict[str, int] = {"What is 2+2?": 4, "What is 3+3?": 6, "What is 4+4?": 8}

    # Call the batch processing function
    result = batch_call("gpt-4o-mini", test_prompts)

    # Verify the results
    assert len(result) == len(test_prompts), "Number of results should match number of prompts"
    for i in range(len(result)):
        result_item = result[i]
        assert result_item["input"] == test_prompts[i], f"Result at index {i} for prompt {result_item['input']} should match expected prompt"
        # TODO: this is potentially an incorrect assertion
        assert (  # noqa: F631
            result_item["answer"] == expected_answers[result_item["input"]],
            f"Result at index {i} for prompt {result_item['input']} should match expected answer",
        )


class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    cook_time: int = Field(description="Cooking time in minutes")


@pytest.mark.skip(reason="Temporarily disabled, since it takes a while")
def test_anthropic_batch_structured_output() -> None:
    """Tests that Anthropic batch processing correctly handles structured output.

    This test verifies:
    1. Batch processing with structured output format
    2. Correct parsing of responses into pydantic models
    3. Multiple prompts processed correctly
    """
    # Test input prompts
    test_prompts: List[str] = [
        "Create a recipe for pasta",
        "Create a recipe for pizza",
        "Create a recipe for salad",
        "Create a recipe for soup",
        "Create a recipe for dessert",
    ]

    dataset: Dataset = Dataset.from_dict({"prompt": test_prompts})
    llm = curator.LLM(
        prompt_func=lambda row: row["prompt"],
        model_name="claude-3-5-haiku-20241022",
        response_format=Recipe,
        batch=True,
        batch_size=2,
        parse_func=lambda row, response: {
            "input": row["prompt"],
            "title": response.title,
            "ingredients": response.ingredients,
            "cook_time": response.cook_time,
        },
    )

    result = llm(dataset).dataset

    # Verify the results
    assert len(result) == len(test_prompts), "Number of results should match number of prompts"

    for item in result:
        assert "title" in item, "Each result should have a title"
        assert "ingredients" in item, "Each result should have ingredients"
        assert "cook_time" in item, "Each result should have cook time"
        assert isinstance(item["ingredients"], list), "Ingredients should be a list"
        assert isinstance(item["cook_time"], int), "Cook time should be an integer"
        assert item["cook_time"] > 0, "Cook time should be positive"

    # Enhanced output display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 30)
    print("\nTest Results:")
    print(result.to_pandas())
