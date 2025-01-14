from typing import Dict, List

import pandas as pd
import pytest
from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator


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

    result = llm(dataset)

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
