import os

from datasets import load_dataset

from bespokelabs import curator

# From https://console.anthropic.com/workbench/
"""
REQUEST:  # noqa: W291
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data '{
    "model": "claude-3-7-sonnet-20250219",
    "max_tokens": 16000,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 14000
    },
    "messages": [
        {"role": "user", "content": "Let $N$ denote the number of ordered triples of positive integers $(a,b,c)$ such that $a,b,c \\leq 3^6$ and $a^3 + b^3 + c^3$ is a multiple of $3^7$. Find the remainder when $N$ is divided by 1000."}  # noqa: E501
    ]
}'

RESPONSE:
{
    "id": "msg_01GTn1T84jYjUPFmBpzfFjiK",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-7-sonnet-20250219",
    "content": [
        {
            "type": "thinking",
            "thinking": "...THINKING_TEXT..."
        },
        {
            "type": "text",
            "text": "...FINAL_TEXT..."
        }
    ],
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
        "input_tokens": 114,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 15312
    }
}
"""


class Reasoner(curator.LLM):
    """Curator class for processing Numina dataset."""

    return_completions_object = True

    def prompt(self, input):  # noqa: D102
        return input["question"]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        print(response)
        # input["claude_thinking_trajectory"] = thinking
        # input["claude_attempt"] = text
        return input


"""
The response doesn't include the thinking trajectory.  # noqa: W291

{
    'id': 'chatcmpl-b1a65801-9d63-4a02-b942-1a18b9b189fd',
    'created': 1740430818,
    'model': 'claude-3-7-sonnet-20250219',
    'object': 'chat.completion',
    'system_fingerprint': None,
    'choices': [{
        'finish_reason': 'stop',
        'index': 0,
        'message': {
            'content': "# Proving Constraints on a Non-Negative Trigonometric Function\n\n...",  # Long content truncated for readability
            'role': 'assistant',
            'tool_calls': None,
            'function_call': None,
            'provider_specific_fields': {'citations': []},
            'citations': []
        }
    }],
    'usage': {
        'completion_tokens': 13386,
        'prompt_tokens': 179,
        'total_tokens': 13565,
        'completion_tokens_details': None,
        'prompt_tokens_details': {
            'audio_tokens': None,
            'cached_tokens': 0,
            'text_tokens': None,
            'image_tokens': None
        },
        'cache_creation_input_tokens': 0,
        'cache_read_input_tokens': 0
    }
}
"""

llm = Reasoner(
    model_name="anthropic/claude-3-7-sonnet-20250219",
    generation_params={"max_tokens": 16000, "thinking": {"type": "enabled", "budget_tokens": 14000}},
    backend_params={  # https://docs.anthropic.com/en/api/rate-limits#rate-limits
        "max_input_tokens_per_minute": 200_000,
        "max_output_tokens_per_minute": 80_000,
    },
)

num_cpus = os.cpu_count()
dataset = load_dataset("simplescaling/s1K", split="train")


def unroll_trajectory(example):  # noqa: D103
    example["gemini_thinking_trajectory"] = example["thinking_trajectories"][0]
    example["gemini_attempt"] = example["attempt"]
    return example


dataset = dataset.map(unroll_trajectory, num_proc=num_cpus)
dataset = dataset.remove_columns(["thinking_trajectories", "cot", "attempt"])
print(dataset)

response = llm(dataset.take(1))
print(response)
