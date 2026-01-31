"""Basic TinkerTrainer example: Fine-tune a model on Q&A data.

This example demonstrates:
1. Creating training data in chat format
2. Configuring TinkerTrainer with LoRA settings
3. Running the training loop
4. Sampling from the fine-tuned model

Usage:
    # Run with mock mode (no API keys required)
    poetry run python examples/tinker/basic_example.py

    # With Tinker API key for real training
    export TINKER_API_KEY="your-tinker-key"
    poetry run python examples/tinker/basic_example.py
"""

from bespokelabs.curator import TinkerTrainer, TinkerTrainerConfig


def get_training_data() -> list[dict]:
    """Create sample Q&A training data in chat format."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is a list comprehension in Python?"},
                {
                    "role": "assistant",
                    "content": "A list comprehension is a concise way to create lists in Python. "
                    "It combines a for loop and optional conditions into a single line. "
                    "For example: `[x**2 for x in range(10)]` creates a list of squares.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "How do I handle exceptions in Python?"},
                {
                    "role": "assistant",
                    "content": "Use try-except blocks to handle exceptions. "
                    "The try block contains code that might raise an exception, "
                    "and the except block handles it. Example:\n"
                    "```python\n"
                    "try:\n"
                    "    result = 10 / 0\n"
                    "except ZeroDivisionError:\n"
                    "    print('Cannot divide by zero')\n"
                    "```",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is a decorator in Python?"},
                {
                    "role": "assistant",
                    "content": "A decorator is a function that modifies the behavior of another function. "
                    "It takes a function as input and returns a new function with added functionality. "
                    "Use the @decorator_name syntax above a function definition to apply it.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Explain Python generators."},
                {
                    "role": "assistant",
                    "content": "Generators are functions that use `yield` instead of `return`. "
                    "They produce values lazily, one at a time, saving memory. "
                    "Example: `def count(): yield 1; yield 2; yield 3` creates a generator "
                    "that yields 1, 2, 3 when iterated.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is the difference between a list and a tuple?"},
                {
                    "role": "assistant",
                    "content": "Lists are mutable (can be changed after creation) and use square brackets []. "
                    "Tuples are immutable (cannot be changed) and use parentheses (). "
                    "Tuples are faster and can be used as dictionary keys.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "How do I read a file in Python?"},
                {
                    "role": "assistant",
                    "content": "Use the `open()` function with a context manager:\n"
                    "```python\n"
                    "with open('file.txt', 'r') as f:\n"
                    "    content = f.read()\n"
                    "```\n"
                    "The context manager ensures the file is properly closed.",
                },
            ]
        },
    ]


def main():
    """Run the basic TinkerTrainer example."""
    print("=" * 60)
    print("Basic TinkerTrainer Example")
    print("=" * 60)

    # Step 1: Get training data
    print("\n[Step 1] Preparing training data...")
    training_data = get_training_data()
    print(f"  Training examples: {len(training_data)}")

    # Step 2: Configure the trainer
    print("\n[Step 2] Configuring TinkerTrainer...")
    config = TinkerTrainerConfig(
        base_model="Qwen/Qwen3-8B",
        epochs=2,
        batch_size=2,
        max_seq_length=1024,
        adam_params={
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
        },
        lora_config={
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
        },
        log_every_n_steps=1,
        warmup_steps=2,
        checkpoint_every_epoch=True,
        checkpoint_name_prefix="coding_assistant",
    )

    print(f"  Base Model: {config.base_model}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.adam_params.learning_rate}")
    print(f"  LoRA Rank: {config.lora_config.rank}")

    # Step 3: Create trainer and run training
    print("\n[Step 3] Training...")
    trainer = TinkerTrainer(config)
    result = trainer.train(training_data)

    # Step 4: Display results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Final Loss: {result.final_loss:.4f}")
    print(f"  Total Steps: {result.total_steps}")
    print(f"  Total Time: {result.total_time:.2f}s")
    print(f"  Samples Processed: {result.samples_processed}")
    print(f"  Tokens Processed: {result.tokens_processed:,}")

    # Display checkpoints
    if result.checkpoints:
        print("\n  Checkpoints saved:")
        for cp in result.checkpoints:
            print(f"    - {cp.name}: step {cp.step}, epoch {cp.epoch}, loss {cp.loss:.4f}")
            print(f"      Path: {cp.path}")

    # Step 5: Sample from the model
    print("\n[Step 5] Sampling from fine-tuned model...")
    test_prompts = [
        "What is a lambda function in Python?",
        "How do I create a class in Python?",
    ]

    for prompt in test_prompts:
        print(f"\n  User: {prompt}")
        response = trainer.sample(
            prompt,
            system_prompt="You are a helpful coding assistant.",
        )
        print(f"  Assistant: {response}")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)

    return trainer


if __name__ == "__main__":
    trainer = main()
