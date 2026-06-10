"""Custom FireworksTrainer example: subclass for custom data formatting.

This example demonstrates:
1. Subclassing FireworksTrainer to customize data formatting
2. Using format_example() to convert raw data to chat training examples
3. Working with non-standard data formats (instruction/response, Q&A with context)

This pattern is useful when your data doesn't already use the standard "messages"
format and needs custom conversion before fine-tuning.

Usage:
    poetry run python examples/fireworks/custom_trainer_example.py
"""

from bespokelabs.curator import FireworksTrainer, FireworksTrainerConfig
from bespokelabs.curator.finetune.types import TrainingExample


class InstructionTrainer(FireworksTrainer):
    """Custom trainer for instruction-response datasets.

    Converts data in {"instruction": ..., "response": ...} format to the chat
    format required by Fireworks fine-tuning.
    """

    def format_example(self, row: dict) -> TrainingExample:
        """Convert an instruction-response pair to chat format."""
        return TrainingExample.from_dict_messages(
            [
                {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["response"]},
            ]
        )


class QATrainer(FireworksTrainer):
    """Custom trainer for question-answer datasets with optional context."""

    def format_example(self, row: dict) -> TrainingExample:
        """Convert a Q&A pair (with optional context) to chat format."""
        if row.get("context"):
            user_content = f"Context: {row['context']}\n\nQuestion: {row['question']}"
        else:
            user_content = row["question"]
        return TrainingExample.from_dict_messages(
            [
                {"role": "system", "content": "You are a knowledgeable assistant. Answer accurately and concisely."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row["answer"]},
            ]
        )


def get_instruction_data() -> list[dict]:
    """Sample instruction-response dataset (Fireworks requires >= 3 examples)."""
    return [
        {
            "instruction": "Summarize in one sentence: Machine learning lets systems learn from experience.",
            "response": "Machine learning lets systems learn from experience automatically.",
        },
        {"instruction": "Convert to past tense: The cat jumps over the fence.", "response": "The cat jumped over the fence."},
        {"instruction": "List two benefits of exercise.", "response": "1. Better cardiovascular health\n2. Improved mood"},
        {"instruction": "Explain what an API is in one sentence.", "response": "An API is a contract that lets software components talk to each other."},
    ]


def get_qa_data() -> list[dict]:
    """Sample Q&A dataset with optional context."""
    return [
        {"context": "Python was created by Guido van Rossum in 1991.", "question": "Who created Python?", "answer": "Guido van Rossum created Python."},
        {"context": "The Great Wall of China is ~21,196 km long.", "question": "How long is the Great Wall?", "answer": "About 21,196 kilometers."},
        {"question": "What is the capital of France?", "answer": "Paris is the capital of France."},
        {
            "context": "Photosynthesis converts sunlight into energy.",
            "question": "What is photosynthesis?",
            "answer": "The process by which plants convert sunlight into energy.",
        },
    ]


def train_instruction_model():
    """Train using the InstructionTrainer."""
    print("\n" + "-" * 50)
    print("Training InstructionTrainer")
    print("-" * 50)
    config = FireworksTrainerConfig(base_model="qwen3-4b", epochs=1, lora_rank=8, display_name="curator-instruction-demo")
    data = get_instruction_data()
    # `with` guarantees the on-demand inference deployment is torn down afterward.
    with InstructionTrainer(config) as trainer:
        print(f"Training on {len(data)} instruction-response pairs...")
        result = trainer.train(data)
        print(f"Output model: {result.weights_name}")
        response = trainer.sample("Translate 'Hello, how are you?' to French.")
        print(f"  Response: {response}")
    return trainer


def train_qa_model():
    """Train using the QATrainer."""
    print("\n" + "-" * 50)
    print("Training QATrainer")
    print("-" * 50)
    config = FireworksTrainerConfig(base_model="qwen3-4b", epochs=1, lora_rank=8, display_name="curator-qa-demo")
    data = get_qa_data()
    with QATrainer(config) as trainer:
        print(f"Training on {len(data)} Q&A pairs...")
        result = trainer.train(data)
        print(f"Output model: {result.weights_name}")
        response = trainer.sample("Context: The Eiffel Tower opened in 1889.\n\nQuestion: When did the Eiffel Tower open?")
        print(f"  Response: {response}")
    return trainer


def main():
    """Run both custom trainer examples."""
    print("=" * 60)
    print("Custom FireworksTrainer Examples")
    print("=" * 60)
    instruction_trainer = train_instruction_model()
    qa_trainer = train_qa_model()
    print("\n" + "=" * 60)
    print("Custom Trainer Examples Complete!")
    print("=" * 60)
    return instruction_trainer, qa_trainer


if __name__ == "__main__":
    main()
