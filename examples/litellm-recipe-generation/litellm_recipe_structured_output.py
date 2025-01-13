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


def main():
    """Main function to generate synthetic recipes."""
    # We define a prompter that generates cuisines
    #############################################
    # To use Claude models:
    # 1. Go to https://console.anthropic.com/settings/keys
    # 2. Generate an API key or use an existing API key
    # 3. Set environment variable: ANTHROPIC_API_KEY
    #############################################
    cuisines_generator = curator.LLM(
        prompt_func=lambda: "Generate 10 diverse cuisines.",
        model_name="claude-3-5-haiku-20241022",
        response_format=Cuisines,
        parse_func=lambda _, cuisines: [{"cuisine": t} for t in cuisines.cuisines_list],
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
    recipe_prompter = curator.LLM(
        model_name="gemini/gemini-1.5-flash",
        prompt_func=lambda row: f"Generate a random {row['cuisine']} recipe. Be creative but keep it realistic.",
        parse_func=lambda row, response: {
            "title": response.title,
            "ingredients": response.ingredients,
            "instructions": response.instructions,
            "prep_time": response.prep_time,
            "cook_time": response.cook_time,
            "servings": response.servings,
            "cuisine": row["cuisine"],
        },
        response_format=Recipe,
        backend="litellm",
        backend_params={"max_requests_per_minute": 2_000, "max_tokens_per_minute": 4_000_000},
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
