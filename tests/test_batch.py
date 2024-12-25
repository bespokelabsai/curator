from typing import List, Dict
import pytest
from datasets import Dataset
from bespokelabs import curator
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: int = Field(description="The answer to the question")


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
    response = llm(dataset)
    return response


def test_batch_call() -> None:
    """Tests that batch_call correctly processes multiple prompts and returns expected answers.

    This test verifies:
    1. Batch processing of multiple prompts
    2. Correct response format"""
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
        assert (
            result_item["input"] == test_prompts[i]
        ), f"Result at index {i} for prompt {result_item['input']} should match expected prompt"
        assert (
            result_item["answer"] == expected_answers[result_item["input"]],
            f"Result at index {i} for prompt {result_item['input']} should match expected answer",
        )
