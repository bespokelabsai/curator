import os

from datasets import load_dataset

from bespokelabs import curator


class Reasoner(curator.LLM):
    """Curator class for processing claude reasoning."""

    return_completions_object = True

    def prompt(self, input):
        """Directly pass the question to the model."""
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
            elif content_block["type"] == "redacted_thinking":
                print("Redacted thinking block! (notifying you for fun)")

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
        "max_retries": 20,
    },
)


def unroll_gemini_trajectory(example):
    """Unroll the thinking trajectory and attempt into separate columns."""
    example["gemini_thinking_trajectory"] = example["thinking_trajectories"][0]
    example["gemini_attempt"] = example["attempt"]
    return example


ds = load_dataset("simplescaling/s1K", split="train")
ds = ds.map(unroll_gemini_trajectory, num_proc=os.cpu_count())
ds = ds.remove_columns(["thinking_trajectories", "cot", "attempt"])
ds = llm(ds.take(10))

# Change this to your organization and dataset name
ds.push_to_hub("bespokelabs/test-s1K-claude-3-7-sonnet")
