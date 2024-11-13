from bespokelabs import curator
from datasets import load_dataset
import logging

dataset = load_dataset("allenai/WildChat", split="train")
dataset = dataset.select(range(10_000))

# To see more detail about how batches are being processed
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)

def prompt_func(row):
    return row["conversation"][0]["content"]


def parse_func(row, response):
    instruction = row["conversation"][0]["content"]
    return {"instruction": instruction, "new_response": response}


distill_prompter = curator.Prompter(
    prompt_func=prompt_func,
    parse_func=parse_func,
    model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    api_base="http://192.222.54.64:8000/"
)

distilled_dataset = distill_prompter(dataset)
print(distilled_dataset)
print(distilled_dataset[0])
