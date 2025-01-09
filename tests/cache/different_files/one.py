import logging

from datasets import Dataset

from bespokelabs.curator import LLM

logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


dataset = Dataset.from_dict({"prompt": ["just say 'hi'"] * 3})

prompter = LLM(
    prompt_func=lambda row: row["prompt"],
    model_name="gpt-4o-mini",
    response_format=None,
)

dataset = prompter(dataset)
print(dataset.to_pandas())
