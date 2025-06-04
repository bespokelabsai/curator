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


class CuisineGenerator(curator.LLM):
    """A cuisine generator that generates diverse cuisines."""

    response_format = Cuisines

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the cuisine generator."""
        return "Generate 10 diverse cuisines."

    def parse(self, input: dict, response: Cuisines) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        return [{"cuisine": t} for t in response.cuisines_list]


class RecipeGenerator(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    response_format = Recipe

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return f"Generate a random {input['cuisine']} recipe. Be creative but keep it realistic."

    def parse(self, input: dict, response: Recipe) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        return {
            "title": response.title,
            "ingredients": response.ingredients,
            "instructions": response.instructions,
            "prep_time": response.prep_time,
            "cook_time": response.cook_time,
            "servings": response.servings,
        }


def main():
    """Generate recipes for different world cuisines using vLLM.

    Creates a dataset of cuisine names, sets up a recipe generation prompter using vLLM backend,
    and generates creative but realistic recipes for each cuisine. The results are printed
    as a pandas DataFrame.
    """
    # List of cuisines to generate recipes for
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    cuisines_generator = CuisineGenerator(
        model_name=model_path,
        backend="vllm",
    )
    cuisines = cuisines_generator()
    print(cuisines.to_pandas())

    recipe_generator = RecipeGenerator(
        model_name=model_path,
        backend="vllm",
    )

    # Generate recipes for all cuisines
    recipes = recipe_generator(cuisines)

    # Print results
    print(recipes.dataset.to_pandas())


if __name__ == "__main__":
    main()
