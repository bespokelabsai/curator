import os

from datasets import load_dataset

from bespokelabs import curator


class Reasoner(curator.LLM):
    """Curator class for reasoning."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [{"role": "user", "content": input["question"]}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        input["deepseek_reasoning"] = response["choices"][0]["message"]["reasoning_content"]
        input["deepseek_solution"] = response["choices"][0]["message"]["content"]
        return input


llm = Reasoner(
    model_name="deepseek-reasoner",
    backend="openai_client",
    generation_params={"temperature": 0.0},
    backend_params={
        "max_requests_per_minute": 100,
        "max_tokens_per_minute": 100_000_000,
        "base_url": "https://api.deepseek.com/",
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
    },
)

ds = load_dataset("simplescaling/s1K", split="train")
ds = ds.remove_columns(["thinking_trajectories", "cot", "attempt"])
ds = llm(ds)
# print("REASONING: ", ds[0]["deepseek_reasoning"])
# print("\n\nSOLUTION: ", ds[0]["deepseek_solution"])
ds.push_to_hub("mlfoundations-dev/s1K-test-annotation")
