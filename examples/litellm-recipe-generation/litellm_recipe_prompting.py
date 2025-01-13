"""Generate synthetic recipes for different cuisines using curator."""

from datasets import Dataset

from bespokelabs import curator


class RecipeGenerator(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    @classmethod
    def prompt(cls, input: dict) -> str:
        """Generate a prompt using the template and cuisine."""
        return f"Generate a random {input['cuisine']} recipe. Be creative but keep it realistic."

    @classmethod
    def parse(cls, input: dict, response: str) -> dict:
        """Parse the model response into the desired output format."""
        return {
            "recipe": response,
            "cuisine": input["cuisine"],
        }


def main():
    """Generate synthetic recipes for different cuisines."""
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
