"""Custom TinkerTrainer example: Subclass for custom data formatting.

This example demonstrates:
1. Subclassing TinkerTrainer to customize data formatting
2. Using format_example() to convert raw data to training examples
3. Working with non-standard data formats

This pattern is useful when your data doesn't use the standard
"messages" format and needs custom conversion.

Usage:
    poetry run python examples/tinker/custom_trainer_example.py
"""

from bespokelabs.curator import TinkerTrainer, TinkerTrainerConfig
from bespokelabs.curator.finetune.types import TrainingExample


class InstructionTrainer(TinkerTrainer):
    """Custom trainer for instruction-response datasets.

    Converts data in {"instruction": ..., "response": ...} format
    to the chat format required by TinkerTrainer.
    """

    def format_example(self, row: dict) -> TrainingExample:
        """Convert instruction-response pair to chat format.

        Args:
            row: Dict with "instruction" and "response" keys

        Returns:
            TrainingExample with system/user/assistant messages
        """
        return TrainingExample.from_dict_messages(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows instructions precisely.",
                },
                {
                    "role": "user",
                    "content": row["instruction"],
                },
                {
                    "role": "assistant",
                    "content": row["response"],
                },
            ]
        )


class QATrainer(TinkerTrainer):
    """Custom trainer for question-answer datasets.

    Converts data in {"question": ..., "answer": ..., "context": ...} format
    with optional context for RAG-style training.
    """

    def format_example(self, row: dict) -> TrainingExample:
        """Convert Q&A pair to chat format with optional context.

        Args:
            row: Dict with "question", "answer", and optional "context" keys

        Returns:
            TrainingExample with formatted messages
        """
        # Build user message with optional context
        if row.get("context"):
            user_content = f"Context: {row['context']}\n\nQuestion: {row['question']}"
        else:
            user_content = row["question"]

        return TrainingExample.from_dict_messages(
            [
                {
                    "role": "system",
                    "content": "You are a knowledgeable assistant. Answer questions accurately and concisely.",
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": row["answer"],
                },
            ]
        )


def get_instruction_data() -> list[dict]:
    """Sample instruction-response dataset."""
    return [
        {
            "instruction": "Summarize the following text in one sentence: "
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without being explicitly programmed.",
            "response": "Machine learning is an AI subset that allows systems to learn from experience automatically.",
        },
        {
            "instruction": "Convert this sentence to past tense: The cat jumps over the fence.",
            "response": "The cat jumped over the fence.",
        },
        {
            "instruction": "List three benefits of regular exercise.",
            "response": "1. Improved cardiovascular health\n2. Better mental well-being\n3. Increased energy levels",
        },
        {
            "instruction": "Explain what an API is in simple terms.",
            "response": "An API is like a waiter in a restaurant - it takes your request, "
            "communicates it to the kitchen (the system), and brings back what you ordered (the response).",
        },
    ]


def get_qa_data() -> list[dict]:
    """Sample Q&A dataset with context."""
    return [
        {
            "context": "Python was created by Guido van Rossum and first released in 1991.",
            "question": "Who created Python?",
            "answer": "Guido van Rossum created Python.",
        },
        {
            "context": "The Great Wall of China is approximately 21,196 kilometers long.",
            "question": "How long is the Great Wall of China?",
            "answer": "The Great Wall of China is approximately 21,196 kilometers long.",
        },
        {
            "question": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
        },
        {
            "context": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, "
            "carbon dioxide, and water to produce glucose and oxygen.",
        },
    ]


def train_instruction_model():
    """Train using the InstructionTrainer."""
    print("\n" + "-" * 50)
    print("Training InstructionTrainer")
    print("-" * 50)

    config = TinkerTrainerConfig(
        base_model="Qwen/Qwen3-8B",
        epochs=2,
        batch_size=2,
        lora_config={"rank": 8},
        log_every_n_steps=1,
    )

    trainer = InstructionTrainer(config)
    data = get_instruction_data()

    print(f"Training on {len(data)} instruction-response pairs...")
    result = trainer.train(data)

    print(f"\nFinal Loss: {result.final_loss:.4f}")
    print(f"Total Steps: {result.total_steps}")

    # Test sampling
    print("\nSampling test:")
    response = trainer.sample(
        "Translate 'Hello, how are you?' to French.",
        system_prompt="You are a helpful assistant that follows instructions precisely.",
    )
    print(f"  Response: {response}")

    return trainer


def train_qa_model():
    """Train using the QATrainer."""
    print("\n" + "-" * 50)
    print("Training QATrainer")
    print("-" * 50)

    config = TinkerTrainerConfig(
        base_model="Qwen/Qwen3-8B",
        epochs=2,
        batch_size=2,
        lora_config={"rank": 8},
        log_every_n_steps=1,
    )

    trainer = QATrainer(config)
    data = get_qa_data()

    print(f"Training on {len(data)} Q&A pairs...")
    result = trainer.train(data)

    print(f"\nFinal Loss: {result.final_loss:.4f}")
    print(f"Total Steps: {result.total_steps}")

    # Test sampling with context
    print("\nSampling test (with context):")
    response = trainer.sample(
        "Context: The Eiffel Tower was completed in 1889.\n\nQuestion: When was the Eiffel Tower completed?",
        system_prompt="You are a knowledgeable assistant. Answer questions accurately and concisely.",
    )
    print(f"  Response: {response}")

    return trainer


def main():
    """Run both custom trainer examples."""
    print("=" * 60)
    print("Custom TinkerTrainer Examples")
    print("=" * 60)

    # Train instruction-following model
    instruction_trainer = train_instruction_model()

    # Train Q&A model
    qa_trainer = train_qa_model()

    print("\n" + "=" * 60)
    print("Custom Trainer Examples Complete!")
    print("=" * 60)

    return instruction_trainer, qa_trainer


if __name__ == "__main__":
    main()
