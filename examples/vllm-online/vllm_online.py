"""Generate synthetic recipes for different cuisines using vLLM online API.

This script demonstrates using vLLM backend with curator to generate recipes for various
world cuisines in an efficient batched manner. It uses Meta-Llama-3.1-8B-Instruct model.

To start the vLLM server, run the following command:
vllm serve
Qwen/Qwen2.5-3B-Instruct
--host localhost
--port 8787
--api-key token-abc123
"""

import os

from datasets import Dataset

from bespokelabs import curator


class RecipeGenerator(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return f"Generate a random {input['cuisine']} recipe. Be creative but keep it realistic."

    def parse(self, input: dict, response: str) -> dict:
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

    model_path = "Qwen/Qwen2.5-3B-Instruct"
    model_path = f"hosted_vllm/{model_path}"  # Use the hosted_vllm backend

    API_KEY = "token-abc123"
    os.environ["HOSTED_VLLM_API_KEY"] = API_KEY

    # Define the vLLM server params
    PORT = 8787
    HOST = "localhost"

    recipe_generator = RecipeGenerator(
        model_name=model_path,
        backend="litellm",
        backend_params={"base_url": f"http://{HOST}:{PORT}/v1"},
    )

    # Generate recipes for all cuisines
    recipes = recipe_generator(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
