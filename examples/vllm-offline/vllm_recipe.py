"""Generate synthetic recipes for different cuisines with offline vLLM."""

from bespokelabs import curator
from datasets import Dataset


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

    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    recipe_prompter = curator.LLM(
        model_name=model_path,
        prompt_func=lambda row: f"Generate a random {row['cuisine']} recipe. Be creative but keep it realistic.",
        parse_func=lambda row, response: {
            "recipe": response,
            "cuisine": row["cuisine"],
        },
        backend="vllm",
        tensor_parallel_size=4,
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
