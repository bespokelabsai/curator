"""Example of using the curator library to generate diverse poems.

We generate 10 diverse topics and then generate 2 poems for each topic.
"""

from typing import List

from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator


import base64
from pydantic import BaseModel, Field
from bespokelabs import curator

class MultiModalRecipe(BaseModel):
    recipe: str
    title: str
    instructions: str
    cook_time: str
    ingredients: str


class MultiModalRecipeGenerator(curator.LLM):
    """A recipe generator that can handle multimodal inputs and outputs."""

    response_format = MultiModalRecipe

    def prompt(self, input: dict) -> curator.MultiModalPrompt:
        prompt = f"Generate a {input['cuisine']} recipe given ingredients in the image. Be creative but keep it realistic."
        return prompt, curator.types.Image(url=input["ingredients_url"])


    def parse(self, input: dict, response: MultiModalRecipe) -> dict:
        result = {
            "title": response.title,
            "ingredients": response.ingredients,
            "instructions": response.instructions,
            "cook_time": response.cook_time,
            "cuisine": input["cuisine"],
        }
        return result


def main():
    """Example usage of multimodal recipe generation."""
    recipe_generator = MultiModalRecipeGenerator(
        model_name="gpt-4o",
        backend="openai",
        backend_params={"max_requests_per_minute": 2_000, "max_tokens_per_minute": 4_000_000},
    )

    recipe = recipe_generator({
        "cuisine": "Italian",
        "food_image": "path/to/pizza_reference.jpg"
    })

    print(recipe.to_pandas())
