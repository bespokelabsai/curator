"""Generate synthetic recipes from ingredients image."""

from datasets import Dataset

from bespokelabs import curator


class RecipeGenerator(curator.LLM):
    """A recipe generator that generates recipes for different ingredient images."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt using the ingredients."""
        prompt = f"Create me a {input['spice_level']} recipe from the ingredients image."
        return prompt, curator.types.Image(url=input["image_url"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        return {
            "recipe": response,
        }


def main():
    """Generate synthetic recipes for different spice level and ingredients image."""
    # List of ingredients to generate recipes for
    ingredients = [
        {"spice_level": ingredient[0], "image_url": ingredient[1]}
        for ingredient in [
            ("very spicy", "image1.jpeg"),
            ("not so spicy", "file_example_PNG_500kB.png"),
        ]
    ]
    ingredients = Dataset.from_list(ingredients)

    # Create prompter using LiteLLM backend
    #############################################
    # To use Gemini models:
    # 1. Go to https://aistudio.google.com/app/apikey
    # 2. Generate an API key
    # 3. Set environment variable: GEMINI_API_KEY
    # 4. If you are a free user, update rate limits:
    #       max_requests_per_minute=15
    #       max_tokens_per_minute=1_000_000
    #       (Up to 1,000 requests per day)
    #############################################

    recipe_generator = RecipeGenerator(
        model_name="claude-3-5-sonnet-20241022",
        backend="litellm",
    )

    # Generate recipes for all ingredients
    recipes = recipe_generator(ingredients)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
