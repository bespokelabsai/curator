import logging

from datasets import load_dataset

from bespokelabs import curator

dataset = load_dataset("allenai/WildChat", split="train")
dataset = dataset.select(range(3_000))

# To see more detail about how batches are being processed
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


def prompt_func(row):
    return row["conversation"][0]["content"]


def parse_func(row, response):
    instruction = row["conversation"][0]["content"]
    return {"instruction": instruction, "new_response": response}


distill_prompter = curator.LLM(
    prompt_func=prompt_func,
    parse_func=parse_func,
    model_name="gpt-4o-mini",
    batch=True,
    batch_size=1_000,
)

distilled_dataset = distill_prompter(dataset)
print(distilled_dataset)
print(distilled_dataset[0])
