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
        "max_requests_per_minute": 2_500,
        "max_tokens_per_minute": 1_000_000_000,
        "base_url": "https://api.deepseek.com/",
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "require_all_responses": False,
        "max_retries": 1,
        "num_clients": 4,  # Create 2 OpenAI clients for parallel requests
    },
)


ds = load_dataset("mlfoundations-dev/herorun1_code", split="train")
ds = llm(ds.select(range(50_000, len(ds))))
ds.push_to_hub("mlfoundations-dev/herorun1_code_50000-end")
