import os

from datasets import load_dataset

from bespokelabs import curator


class Reasoner(curator.LLM):
    """Curator class for processing Numina dataset."""

    return_completions_object = True

    def prompt(self, input):  # noqa: D102
        return input["question"]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        content = response["content"]
        thinking = ""
        text = ""

        for content_block in content:
            if content_block["type"] == "thinking":
                thinking = content_block["thinking"]
            elif content_block["type"] == "text":
                text = content_block["text"]

        input["claude_thinking_trajectory"] = thinking
        input["claude_attempt"] = text
        return input


llm = Reasoner(
    model_name="claude-3-7-sonnet-20250219",
    generation_params={"max_tokens": 20000, "thinking": {"type": "enabled", "budget_tokens": 18000}},
    backend="anthropic",
    backend_params={  # https://docs.anthropic.com/en/api/rate-limits#rate-limits Tier 4
        "max_input_tokens_per_minute": 200_000,
        "max_output_tokens_per_minute": 80_000,
    },
)


def unroll_trajectory(example):  # noqa: D103
    example["gemini_thinking_trajectory"] = example["thinking_trajectories"][0]
    example["gemini_attempt"] = example["attempt"]
    return example


ds = load_dataset("simplescaling/s1K", split="train")
ds = ds.map(unroll_trajectory, num_proc=os.cpu_count())
ds = ds.remove_columns(["thinking_trajectories", "cot", "attempt"])
ds = llm(ds)
ds.push_to_hub("simplescaling/s1K-claude-3-7-sonnet")
