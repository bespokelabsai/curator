"""Example of reannotating the OpenHermes dataset using curator."""

import logging

from datasets import load_dataset

from bespokelabs import curator

# To see more detail about how batches are being processed
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


def convert_row(row: dict) -> dict:
    """Convert a conversation row from OpenHermes format to instruction/response format.

    Args:
        row: Dictionary containing a conversation from the OpenHermes dataset

    Returns:
        Dictionary with 'instruction' and 'original_response' fields extracted from the conversation
    """
    conversation = row["conversations"]
    instruction = next((item["value"] for item in conversation if item["from"] == "human"), None)
    response = next((item["value"] for item in conversation if item["from"] == "gpt"), None)
    return {"instruction": instruction, "original_response": response}


class OpenHermesReannotator(curator.LLM):
    """A reannotator for the OpenHermes dataset."""

    def prompt(self, input: dict) -> str:
        """Extract the instruction to be used as the prompt."""
        return input["instruction"]

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        instruction = input["instruction"]
        return {"instruction": instruction, "new_response": response}


distiller = OpenHermesReannotator(model_name="claude-3-5-sonnet-20241022", batch=True, backend_params={"batch_size": 100})

dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
dataset = dataset.take(500)
dataset = dataset.map(convert_row)
dataset = dataset.select_columns(["instruction", "original_response"])
distilled_dataset = distiller(dataset)
print(distilled_dataset)
print(distilled_dataset[0])
