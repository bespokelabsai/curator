from bespokelabs.curator.blocks.simplestrat import StratifiedGenerator
from datasets import Dataset

questions = Dataset.from_dict({"question": [f"{i}. Name a periodic element" for i in range(20)]})
generate = StratifiedGenerator(model_name="gpt-4o-mini")
qas = generate(questions)
