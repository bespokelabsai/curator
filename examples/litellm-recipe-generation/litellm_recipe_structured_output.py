"""Generate synthetic recipes for different cuisines.

Demonstrates how to use a structured output format with Litellm.
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
        """Generate a prompt using the cuisine."""
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
            "cuisine": input["cuisine"],
        }


def main():
    """Main function to generate synthetic recipes."""
    #############################################
    # To use Claude models:
    # 1. Go to https://console.anthropic.com/settings/keys
    # 2. Generate an API key or use an existing API key
    # 3. Set environment variable: ANTHROPIC_API_KEY
    #############################################
    cuisines_generator = CuisineGenerator(
        model_name="claude-3-5-haiku-20241022",
        backend="litellm",
    )
    cuisines = cuisines_generator()
    print(cuisines.to_pandas())

    #############################################
    # To use Gemini models:
    # 1. Go to https://aistudio.google.com/app/apikey
    # 2. Generate an API key or use an existing API key
    # 3. Set environment variable: GEMINI_API_KEY
    # 4. If you are a free user, update rate limits:
    #       max_requests_per_minute=15
    #       max_tokens_per_minute=1_000_000
    #       (Up to 1,000 requests per day)
    #############################################
    recipe_generator = RecipeGenerator(
        model_name="gemini/gemini-1.5-flash",
        backend="litellm",
        backend_params={"max_requests_per_minute": 2_000, "max_tokens_per_minute": 4_000_000},
    )

    # Generate recipes for all cuisines
    recipes = recipe_generator(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
