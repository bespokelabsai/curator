import argparse
import os

from datasets import load_dataset

from bespokelabs import curator


class Reasoner(curator.LLM):
    """Curator class for reasoning."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [{"role": "user", "content": input["problem"]}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        input["deepseek_reasoning"] = response["choices"][0]["message"]["reasoning_content"]
        input["deepseek_solution"] = response["choices"][0]["message"]["content"]
        return input


def main():
    """Process a dataset with DeepSeek and shard it."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a dataset with DeepSeek and shard it")
    parser.add_argument("--worker", type=int, required=True, help="Worker ID (0-indexed)")
    parser.add_argument("--global", type=int, dest="global_workers", required=True, help="Total number of workers")
    parser.add_argument("--dataset", type=str, required=True, help="Input dataset name (e.g., 'mlfoundations-dev/herorun1_code')")
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output dataset name base (worker ID will be appended). Defaults to input dataset name with 'annotated' suffix.",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.worker < 0 or args.worker >= args.global_workers:
        raise ValueError(f"Worker ID must be between 0 and {args.global_workers-1}")

    if args.global_workers <= 0:
        raise ValueError("Total number of workers must be positive")

    # Initialize the LLM
    llm = Reasoner(
        model_name="deepseek-reasoner",
        backend="openai_client",
        generation_params={"temperature": 0.0},
        backend_params={
            "max_requests_per_minute": 500,
            "max_tokens_per_minute": 1_000_000_000,
            "base_url": "https://api.deepseek.com/",
            "api_key": os.environ.get("DEEPSEEK_API_KEY"),
            "require_all_responses": False,
            "max_retries": 2,
        },
    )

    # Load the dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")

    # Calculate shard size and indices
    total_examples = len(ds)
    shard_size = total_examples // args.global_workers
    start_idx = args.worker * shard_size
    end_idx = start_idx + shard_size if args.worker < args.global_workers - 1 else total_examples

    print(f"Worker {args.worker}/{args.global_workers}: Processing examples {start_idx} to {end_idx-1} (total: {end_idx-start_idx})")

    # Extract the shard
    ds_shard = ds.select(range(start_idx, end_idx))

    # Process the shard with the LLM
    processed_ds = llm(ds_shard)

    # Create output dataset name with worker ID
    output_name = f"{args.output}-{args.worker}" if args.output else f"{args.dataset}-annotated-{args.worker}"

    # Push to hub
    print(f"Pushing processed dataset to {output_name}")
    processed_ds.push_to_hub(output_name)
    print(f"Worker {args.worker} completed successfully")


if __name__ == "__main__":
    main()
