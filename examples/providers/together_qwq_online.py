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
        response = response["choices"][0]["message"]["content"]
        thinking = response.split("</think>")[0].strip("<think>").strip("</think>").strip()
        answer = response.split("</think>")[1].strip()
        input["qwq_thinking_trajectory"] = thinking
        input["qwq_attempt"] = answer
        return input


# https://api.together.ai/models/Qwen/QwQ-32B
# https://docs.together.ai/docs/rate-limits#rate-limit-tiers

llm = Reasoner(
    model_name="Qwen/QwQ-32B",
    backend="openai",
    generation_params={
        "max_tokens": 40_000,
    },
    backend_params={
        "base_url": "https://api.together.ai",
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "max_requests_per_minute": 100,
        "max_tokens_per_minute": 2_000_000,
    },
)

ds = load_dataset("simplescaling/s1K", split="train")
response = llm(ds)
print("THINKING TRAJECTORY: ", response[0]["qwq_thinking_trajectory"])
print("\n\nATTEMPT: ", response[0]["qwq_attempt"])
