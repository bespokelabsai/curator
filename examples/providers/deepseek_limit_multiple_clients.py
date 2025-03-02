import os

from datasets import load_dataset

from bespokelabs import curator

# This example demonstrates using multiple OpenAI clients in parallel
# to overcome the bottleneck of parallel requests that a single client can send.
# The `num_clients` parameter creates multiple AsyncOpenAI clients that are used
# in a round-robin fashion to distribute requests.


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


# Initialize the LLM with multiple clients for parallel request handling
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
        "num_clients": 5,  # Create 5 OpenAI clients for parallel requests
    },
)

# Load the dataset
print("Loading dataset...")
ds = load_dataset("simplescaling/s1K", split="train")

# Process the dataset with the LLM using multiple clients in parallel
print(f"Processing dataset with {llm.backend_params['num_clients']} parallel clients...")
processed_ds = llm(ds[:20])  # Process a subset for demonstration

# Display results
print(f"Processed {len(processed_ds)} examples")
print("\nSample result:")
print(f"Question: {processed_ds[0]['problem']}")
print(f"Reasoning: {processed_ds[0]['deepseek_reasoning']}")
print(f"Solution: {processed_ds[0]['deepseek_solution']}")
