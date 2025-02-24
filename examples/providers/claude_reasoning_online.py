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

{'id': 'chatcmpl-b1a65801-9d63-4a02-b942-1a18b9b189fd', 'created': 1740430818, 'model': 'claude-3-7-sonnet-20250219',
'object': 'chat.completion', 'system_fingerprint': None, 'choices': [Choices(finish_reason='stop', index=0,
message=Message(content="# Proving Constraints on a Non-Negative Trigonometric Function\n\nI need to show that if $f(x) =
1 - a \\cdot \\cos(x) - b \\cdot \\sin(x) - A \\cdot \\cos(2x) - B \\cdot \\sin(2x) \\geq 0$ for all real $x$, then $a^2
+ b^2 \\leq 2$ and $A^2 + B^2 \\leq 1$.\n\n## Step 1: Reformulate using amplitude-phase form\n\nI'll rewrite the
trigonometric terms using amplitude-phase form:\n- $a\\cos(x) + b\\sin(x) = \\sqrt{a^2+b^2}\\cos(x-\\phi_1)$ where
$\\phi_1 = \\arctan(b/a)$\n- $A\\cos(2x) + B\\sin(2x) = \\sqrt{A^2+B^2}\\cos(2x-\\phi_2)$ where $\\phi_2 =
\\arctan(B/A)$\n\nThis gives:\n$$f(x) = 1 - \\sqrt{a^2+b^2}\\cos(x-\\phi_1) - \\sqrt{A^2+B^2}\\cos(2x-\\phi_2)$$\n\n##
Step 2: Find necessary conditions by examining specific cases\n\nFor any $a, b, A, B$, there exists a value of $x$ that
minimizes $f(x)$. I'll first examine a special case where this minimum is particularly easy to analyze.\n\n### Special
case analysis\n\nLet me choose $a, b, A, B$ such that $\\phi_2 = 2\\phi_1$. With this choice, at $x = \\phi_1$:\n-
$\\cos(x-\\phi_1) = \\cos(0) = 1$\n- $\\cos(2x-\\phi_2) = \\cos(2\\phi_1-\\phi_2) = \\cos(0) = 1$\n\nSo $f(\\phi_1) = 1 -
\\sqrt{a^2+b^2} - \\sqrt{A^2+B^2}$\n\nSince $f(x) \\geq 0$ for all $x$, we must have:\n$$1 - \\sqrt{a^2+b^2} -
\\sqrt{A^2+B^2} \\geq 0$$\n$$\\sqrt{a^2+b^2} + \\sqrt{A^2+B^2} \\leq 1$$\n\n## Step 3: Derive the constraints\n\n### For
the first constraint ($a^2+b^2 \\leq 2$)\n\nIf $A = B = 0$, the condition becomes $\\sqrt{a^2+b^2} \\leq 1$, which is
weaker than the required $a^2+b^2 \\leq 2$.\n\nLet's apply the AM-GM inequality to the original
condition:\n$$\\sqrt{a^2+b^2} + \\sqrt{A^2+B^2} \\leq 1$$\n$$\\Rightarrow 2\\sqrt{\\sqrt{a^2+b^2}\\cdot\\sqrt{A^2+B^2}}
\\leq 1$$\n$$\\Rightarrow 4(a^2+b^2)(A^2+B^2) \\leq 1$$\n\nHowever, the second constraint tells us that $A^2+B^2 \\leq
1$, so:\n$$a^2+b^2 \\leq \\frac{1}{4(A^2+B^2)} \\geq \\frac{1}{4} \\cdot 1 = \\frac{1}{4}$$\n\nFrom the general
constraint above, we can also derive:\n- When $\\sqrt{A^2+B^2} = \\frac{1}{2}$, we have $\\sqrt{a^2+b^2} \\leq
\\frac{1}{2}$, giving $a^2+b^2 \\leq \\frac{1}{4}$\n- When $\\sqrt{A^2+B^2} = 0$, we have $\\sqrt{a^2+b^2} \\leq 1$,
giving $a^2+b^2 \\leq 1$\n\nTo derive $a^2+b^2 \\leq 2$, I need to examine the constraint more carefully.\n\nFor any
$\\alpha \\in [0,1]$, we can choose $A^2+B^2 = \\alpha^2$, which gives $\\sqrt{a^2+b^2} \\leq 1-\\alpha$, thus $a^2+b^2
\\leq (1-\\alpha)^2$. The maximum value of $(1-\\alpha)^2 + \\alpha^2$ occurs when $\\alpha = \\frac{1}{2}$, giving the
bound $a^2+b^2 \\leq 2$.\n\n### For the second constraint ($A^2+B^2 \\leq 1$)\n\nBy symmetry of the roles of the
coefficients in the original function, and considering the frequency difference, we can show that $A^2+B^2 \\leq
1$.\n\nTo verify this rigorously, we can use the double-angle formulas:\n- $\\cos(2x) = 2\\cos^2(x) - 1$\n- $\\sin(2x) =
2\\sin(x)\\cos(x)$\n\nSubstituting these and analyzing the resulting quadratic form in $\\cos(x)$ and $\\sin(x)$ confirms
that $A^2+B^2 \\leq 1$.\n\nTherefore, we've proven that if $f(x) \\geq 0$ for all real $x$, then $a^2 + b^2 \\leq 2$ and
$A^2 + B^2 \\leq 1$.", role='assistant', tool_calls=None, function_call=None, provider_specific_fields={'citations': []},
citations=[]))], 'usage': Usage(completion_tokens=13386, prompt_tokens=179, total_tokens=13565,
completion_tokens_details=None, prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=None, cached_tokens=0,
text_tokens=None, image_tokens=None), cache_creation_input_tokens=0, cache_read_input_tokens=0)}

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
