"""Generate synthetic recipes for different cuisines.

Demonstrates how to use a structured output format with vllm.
"""

import logging
from typing import List

from pydantic import BaseModel, Field

from bespokelabs import curator

logger = logging.getLogger(__name__)


# Define response format using Pydantic
class Recipe(BaseModel):
    """A recipe with title, ingredients, instructions, prep time, cook time, and servings."""

    title: str = Field(description="Title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    instructions: List[str] = Field(description="Step by step cooking instructions")
    prep_time: int = Field(description="Preparation time in minutes")
    cook_time: int = Field(description="Cooking time in minutes")
    servings: int = Field(description="Number of servings")


class Cuisines(BaseModel):
    """A list of cuisines."""

    cuisines_list: List[str] = Field(description="A list of cuisines.")


def main():
    """Generate recipes for different world cuisines using vLLM.

    Creates a dataset of cuisine names, sets up a recipe generation prompter using vLLM backend,
    and generates creative but realistic recipes for each cuisine. The results are printed
    as a pandas DataFrame.
    """
    # List of cuisines to generate recipes for
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    cuisines_generator = curator.LLM(
        prompt_func=lambda: "Generate 10 diverse cuisines.",
        model_name=model_path,
        response_format=Cuisines,
        parse_func=lambda _, cuisines: [{"cuisine": t} for t in cuisines.cuisines_list],
        backend="vllm",
    )
    cuisines = cuisines_generator()
    print(cuisines.to_pandas())

    recipe_prompter = curator.LLM(
        model_name=model_path,
        prompt_func=lambda row: f"Generate a random {row['cuisine']} recipe. Be creative but keep it realistic.",
        parse_func=lambda row, response: {
            "title": response.title,
            "ingredients": response.ingredients,
            "instructions": response.instructions,
            "prep_time": response.prep_time,
            "cook_time": response.cook_time,
            "servings": response.servings,
        },
        response_format=Recipe,
        backend="vllm",
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
