"""Generate synthetic recipes for different cuisines with offline vLLM.

This script demonstrates using vLLM backend with curator to generate recipes for various
world cuisines in an efficient batched manner. It uses Meta-Llama-3.1-8B-Instruct model.

"""

from datasets import Dataset

from bespokelabs import curator


class RecipeGenerator(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    @classmethod
    def prompt(cls, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return f"Generate a random {input['cuisine']} recipe. Be creative but keep it realistic."

    @classmethod
    def parse(cls, input: dict, response: str) -> dict:
        """Parse the model response into the desired output format."""
        return {
            "recipe": response,
            "cuisine": input["cuisine"],
        }


def main():
    """Generate recipes for different world cuisines using vLLM.

    Creates a dataset of cuisine names, sets up a recipe generation prompter using vLLM backend,
    and generates creative but realistic recipes for each cuisine. The results are printed
    as a pandas DataFrame.
    """
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

    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    recipe_generator = RecipeGenerator(
        model_name=model_path,
        backend="vllm",
        backend_params={"tensor_parallel_size": 4},
    )

    # Generate recipes for all cuisines
    recipes = recipe_generator(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
