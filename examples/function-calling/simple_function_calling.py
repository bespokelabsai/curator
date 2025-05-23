# Test that generation_params from the row are used if specified, otherwise use the generation_params from the config
# So that we can specify a different generation_params per row

import json
from typing import Dict

from datasets import Dataset

from bespokelabs import curator


class FunctionCallGenerator(curator.LLM):
    """A simple function calling generator."""

    return_completions_object = True

    def prompt(self, input: Dict) -> str:
        """The prompt is used to generate the function call."""
        return f"""You are a function calling expert. Given the user request:
        {input['user_request']}.
        Generate a function call that can be used to satisfy the user request.
        """

    def parse(self, input: Dict, response) -> Dict:
        """Parse the response to extract the function call or the message."""
        if "tool_calls" in response["choices"][0]["message"]:
            input["function_call"] = str([tool_call["function"] for tool_call in response["choices"][0]["message"]["tool_calls"]])
        else:
            # Handle the case where the model returns a string instead of a function call
            input["function_call"] = response["choices"][0]["message"]["content"]
        return input


function_docs = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Retrieves current weather for the given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country e.g. Bogotá, Colombia"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Units the temperature will be returned in."},
                },
                "required": ["location", "units"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_time",
            "description": "Get the local time of a given location",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["location", "timezone"],
                "properties": {
                    "location": {"type": "string", "description": "The name or coordinates of the location for which to get the local time"},
                    "timezone": {"type": "string", "description": "The timezone of the location, defaults to the location's timezone if not provided"},
                },
                "additionalProperties": False,
            },
        },
    },
]
llm = FunctionCallGenerator(
    model_name="gpt-4o-mini",
    # Default generation_params has both functions
    generation_params={"tools": function_docs},
    backend_params={"max_retries": 1, "require_all_responses": False},
)

dataset = Dataset.from_dict(
    {
        "user_request": ["What's the current temperature in New York?", "What time is it in Tokyo?"],
        # WARNING: The generation_params in Dataset ust be a string otherwise the Dataset operation automatically expand dictionary keys
        # See https://github.com/bespokelabsai/curator/issues/325 for more detail
        # The generation_params from the row will override the default generation_params during inference
        "generation_params": [json.dumps({"tools": [function_docs[0]]}), json.dumps({"tools": [function_docs[1]]})],
    }
)

function_calls = llm(dataset)
# The model is expected to return a function call for each row
print(function_calls.dataset.to_pandas())

dataset_2 = Dataset.from_dict(
    {
        "user_request": ["What's the current temperature in New York?", "What time is it in Tokyo?"],
        "generation_params": [json.dumps({"tools": [function_docs[1]]}), json.dumps({"tools": [function_docs[0]]})],
    }
)

function_calls = llm(dataset_2)
# Since the tools are specified in the wrong order, the model will return a message instead of a function call
print(function_calls.dataset.to_pandas())

llm_2 = FunctionCallGenerator(
    model_name="gpt-4o-mini",
    # Default generation_params has both functions
    generation_params={"max_completion_tokens": 20, "tools": function_docs},
    backend_params={"max_retries": 1, "require_all_responses": False},
)

# Since the row-level generation_params does not have max_completion_tokens, the request processor will use the default generation_params
# So an error will be reported as the model will not have enough tokens to return the message
# Uncomment the following line to see the error

# function_calls = llm_2(dataset_2)
# print(function_calls.dataset.to_pandas())
