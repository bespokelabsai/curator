import os

from datasets import Dataset

from bespokelabs import curator

# https://openrouter.ai/deepseek/deepseek-r1
# https://openrouter.ai/docs/limits
# https://openrouter.ai/credits


# Using openai backed
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


"""
https://openrouter.ai/announcements/reasoning-tokens-for-thinking-models
to get the reasoning traces we need to do extra_body={"include_reasoning": True} above
101.71s

REASONING:  ["\nOkay, so I need to figure out what 2 plus 2 is. Let me think...
SOLUTION:  ['**Answer:**  \n2 + 2 equals **4**.  \n\n**Explanation:**..
"""


llm = Reasoner(
    model_name="deepseek/deepseek-r1",
    backend="openai",
    generation_params={"include_reasoning": True, "max_tokens": 40000},
    backend_params={"base_url": "https://openrouter.ai/api/v1", "api_key": os.environ.get("OPENROUTER_API_KEY"), "max_retries": 0},
)

dataset = Dataset.from_dict({"problem": ["What is 2 + 2?"]})

response = llm(dataset)
print("REASONING: ", response["deepseek_reasoning"])
print("SOLUTION: ", response["deepseek_solution"])

# Another way of doing it - Using litellm backend
# litellm.APIError: APIError: OpenrouterException - AsyncCompletions.create() got an unexpected keyword argument 'include_reasoning'
# llm = Reasoner(
#     model_name="openrouter/deepseek/deepseek-r1",
#     generation_params={"include_reasoning": True, "max_tokens": 40000},
#     backend_params={"max_retries": 0},
# )
# response = llm(dataset)
# print("REASONING: ", response["deepseek_reasoning"])
# print("SOLUTION: ", response["deepseek_solution"])
