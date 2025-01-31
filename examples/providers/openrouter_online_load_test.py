import os

from datasets import load_dataset

from bespokelabs import curator


class Reasoner(curator.LLM):
    """Curator class for processing Numina dataset."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [{"role": "user", "content": input["problem"]}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        return [
            {
                "problem": input["problem"],
                "deepseek_reasoning": response["choices"][0]["message"]["reasoning"],
                "deepseek_solution": response["choices"][0]["message"]["content"],
            }
        ]


llm = Reasoner(
    model_name="deepseek/deepseek-r1",
    backend="openai",
    generation_params={
        "include_reasoning": True,
        "max_tokens": 40000,
    },
    backend_params={
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "max_retries": 0,
        "max_requests_per_minute": 6000,
        "max_tokens_per_minute": 10000000000000,
    },
)

dataset = load_dataset("mlfoundations-dev/math_stratos_scale_judged_and_annotated_with_difficulty", split="train")
dataset = dataset.filter(lambda x: not x["correct"] and x["difficulty"] >= 9)
dataset = dataset.take(10)

response = llm(dataset)
print("REASONING: ", response["deepseek_reasoning"][0])
print("SOLUTION: ", response["deepseek_solution"][0])
