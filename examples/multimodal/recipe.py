"""Generate synthetic recipes from ingrients image and cuisine using curator."""

from datasets import Dataset

from bespokelabs import curator


class RecipeGenerator(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt using the template and cuisine."""
        prompt = f"Create me a recipe for {input['cuisine']} cuisine and ingrients from the image."
        return prompt, curator.types.Image(url=input["image_url"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        return {
            "recipe": response,
        }


def main():
    """Generate synthetic recipes for different cuisines."""
    # List of cuisines to generate recipes for
    cuisines = [
        {"cuisine": cuisine[0], "image_url": cuisine[1]}
        for cuisine in [
            ("Indian", "https://cdn.tasteatlas.com//images/ingredients/fcee541cd2354ed8b68b50d1aa1acad8.jpeg"),
            ("Thai", "https://cdn.tasteatlas.com//images/dishes/da5fd425608f48b09555f5257a8d3a86.jpg"),
        ]
    ]
    cuisines = Dataset.from_list(cuisines)

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
        model_name="gpt-4o",
        backend="openai",
    )

    # Generate recipes for all cuisines
    recipes = recipe_generator(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
