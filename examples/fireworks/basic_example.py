"""Basic FireworksTrainer example: managed supervised fine-tuning on Fireworks AI.

This example demonstrates:
1. Creating training data in chat format
2. Configuring FireworksTrainer (base model + LoRA settings)
3. Running a managed supervised fine-tuning (SFT) job
4. Deploying the fine-tuned model and sampling from it

Fireworks runs fine-tuning as a managed, server-side job. Compared to the Tinker
example, there is no client-side training loop: the dataset is uploaded, an SFT job
is submitted, and Fireworks trains the model for you.

Notes:
- Fireworks requires at least 3 training examples.
- Fine-tuned LoRA models cannot be served serverlessly, so `sample()` provisions an
  on-demand deployment (which costs GPU time and takes a few minutes to spin up).
  Call `trainer.close()` (or use the trainer as a context manager) to tear it down
  when finished, so you don't leak a billable deployment.
- Running real training requires a Fireworks account with training quota
  (Tier 2 / credits). Without the SDK or an API key, this runs in mock mode.

Usage:
    # Mock mode (no API key / SDK required)
    poetry run python examples/fireworks/basic_example.py

    # Real fine-tuning (fireworks-ai is an optional add-on installed separately)
    pip install fireworks-ai
    export FIREWORKS_API_KEY="fw_..."
    poetry run python examples/fireworks/basic_example.py
"""

from bespokelabs.curator import FireworksTrainer, FireworksTrainerConfig


def get_training_data() -> list[dict]:
    """Create sample chat training data with a distinctive, learnable style.

    Each assistant answer ends with a fixed catchphrase. After fine-tuning, the
    model should reproduce that catchphrase on unseen questions.
    """
    catchphrase = " Arrr, that's the bespoke truth, matey!"
    facts = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
        ("What color is a clear daytime sky?", "A clear daytime sky is blue."),
        ("Name a popular programming language.", "Python is a popular programming language."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet."),
        ("How many continents are there?", "There are seven continents on Earth."),
        ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide."),
        ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius at sea level."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ]
    return [
        {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer + catchphrase},
            ]
        }
        for question, answer in facts
    ]


def main():
    """Run the basic FireworksTrainer example."""
    print("=" * 60)
    print("Basic FireworksTrainer Example")
    print("=" * 60)

    print("\n[Step 1] Preparing training data...")
    training_data = get_training_data()
    print(f"  Training examples: {len(training_data)}")

    print("\n[Step 2] Configuring FireworksTrainer...")
    config = FireworksTrainerConfig(
        base_model="qwen3-4b",  # smallest fine-tunable Fireworks model
        # Fireworks packs examples into batches by token count, so a SMALL dataset can
        # be a single batch (one optimizer step) per epoch. With few examples, use many
        # epochs so the LoRA gets enough gradient steps to actually learn.
        epochs=30,
        lora_rank=16,
        learning_rate=3e-4,  # or omit to let Fireworks auto-select
        display_name="curator-basic-demo",
        output_model="curator-basic-demo",
    )
    print(f"  Base Model: {config.qualified_base_model}")
    print(f"  Epochs: {config.epochs}")
    print(f"  LoRA Rank: {config.lora_rank}")

    print("\n[Step 3] Running supervised fine-tuning...")
    trainer = FireworksTrainer(config)
    result = trainer.train(training_data)

    print("\n" + "=" * 60)
    print("Fine-tuning Complete!")
    print("=" * 60)
    print(f"  Output Model: {result.weights_name}")
    print(f"  Samples Processed: {result.samples_processed}")
    print(f"  Total Time: {result.total_time:.2f}s")
    if result.loss_history:
        print(f"  Final Loss: {result.final_loss:.4f}")
    print(f"  Job: {result.metadata.get('job_url')}")

    print("\n[Step 4] Sampling from the fine-tuned model...")
    print("  (this deploys the model on-demand; first call may take a few minutes)")
    test_prompts = [
        "What is the capital of Japan?",
        "Who discovered gravity?",
    ]
    for prompt in test_prompts:
        print(f"\n  User: {prompt}")
        response = trainer.sample(prompt)
        print(f"  Assistant: {response}")

    print("\n[Step 5] Cleaning up the deployment...")
    trainer.close()

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    return trainer


if __name__ == "__main__":
    trainer = main()
