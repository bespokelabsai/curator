import os

from datasets import load_dataset

from bespokelabs import curator


class Reasoner(curator.LLM):
    """Curator class for processing claude reasoning."""

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
            elif content_block["type"] == "redacted_thinking":
                print("Redacted thinking block! (notifying you for fun)")

        if text == "" or thinking == "":
            raise ValueError("No text or thinking found in the response")

        input["claude_thinking_trajectory"] = thinking
        input["claude_attempt"] = text
        return input


llm = Reasoner(
    model_name="claude-3-7-sonnet-20250219",
    generation_params={"max_tokens": 16000, "thinking": {"type": "enabled", "budget_tokens": 14000}},
    batch=True,
    backend="anthropic",
)


def unroll_trajectory(example):  # noqa: D103
    example["gemini_thinking_trajectory"] = example["thinking_trajectories"][0]
    example["gemini_attempt"] = example["attempt"]
    return example


ds = load_dataset("simplescaling/s1K", split="train")
ds = ds.map(unroll_trajectory, num_proc=os.cpu_count())
ds = ds.remove_columns(["thinking_trajectories", "cot", "attempt"])
ds = llm(ds)
test = ds.filter(lambda x: x["claude_attempt"] == "")
print(test)
ds.push_to_hub("mlfoundations-dev/s1K-claude-3-7-sonnet-16k")
