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

    recipe_prompter = curator.LLM(
        model_name=model_path,
        prompt_func=lambda row: f"Generate a random {row['cuisine']} recipe. Be creative but keep it realistic.",
        parse_func=lambda row, response: {
            "recipe": response,
            "cuisine": row["cuisine"],
        },
        backend="litellm",
        backend_params={"base_url": f"http://{HOST}:{PORT}/v1"},
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
