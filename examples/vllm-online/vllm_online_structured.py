"""
Generate synthetic recipes for different cuisines using vLLM online API.
To start the vLLM server, run the following command:
vllm serve
Qwen/Qwen2.5-3B-Instruct
--host localhost
--port 8787
--api-key token-abc123
"""

from bespokelabs import curator
from datasets import Dataset
import os
from typing import List
from pydantic import BaseModel, Field


class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    instructions: List[str] = Field(description="Step by step cooking instructions")
    prep_time: int = Field(description="Preparation time in minutes")
    cook_time: int = Field(description="Cooking time in minutes")
    servings: int = Field(description="Number of servings")


def main():
    # List of cuisines to generate recipes for
    cuisines = [
        {"cuisine": cuisine}
        for cuisine in [
            "Chinese",
            "Italian",
            "Mexican",
            "French",
            "Japanese",
            "Indian",
            "Thai",
            "Korean",
            "Vietnamese",
            "Brazilian",
        ]
    ]
    cuisines = Dataset.from_list(cuisines)

    model_path = "Qwen/Qwen2.5-3B-Instruct"
    model_path = f"hosted_vllm/{model_path}"

    API_KEY = "token-abc123"

    os.environ["HOSTED_VLLM_API_KEY"] = API_KEY

    # Define the vLLM server params
    PORT = 8787
    HOST = "localhost"

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
        backend="litellm",
        base_url=f"http://{HOST}:{PORT}/v1",
        response_format=Recipe,
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
