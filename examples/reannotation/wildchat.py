"""Example of reannotating the WildChat dataset using curator."""

import logging

from datasets import load_dataset

from bespokelabs import curator

dataset = load_dataset("allenai/WildChat", split="train")
dataset = dataset.select(range(3))

# To see more detail about how batches are being processed
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


class WildChatReannotator(curator.LLM):
    """A reannotator for the WildChat dataset."""

    def prompt(self, input: dict) -> str:
        """Extract the first message from a conversation to use as the prompt."""
        return input["conversation"][0]["content"]

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        instruction = input["conversation"][0]["content"]
        return {"instruction": instruction, "new_response": response}


distiller = WildChatReannotator(
    model_name="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
    batch=True,
)  # backend_params={"api_key": "4d002946-403c-4ef9-a632-8b4f2199911f", "completion_window": "1h"})

distilled_dataset = distiller(dataset)
print(distilled_dataset)
print(distilled_dataset[0])
