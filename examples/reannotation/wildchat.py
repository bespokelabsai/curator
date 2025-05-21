"""Example of reannotating the WildChat dataset using curator."""

import logging

from datasets import load_dataset

from bespokelabs import curator

dataset = load_dataset("allenai/WildChat", split="train")
dataset = dataset.select(range(3_000))

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


distiller = WildChatReannotator(model_name="gpt-4o-mini", batch=True, backend_params={"batch_size": 1_000})

distilled_dataset = distiller(dataset)
print(distilled_dataset.dataset)
print(distilled_dataset.dataset[0])
