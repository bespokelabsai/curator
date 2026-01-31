"""End-to-end example: Curate poem data with Curator, then fine-tune with TinkerTrainer.

This example demonstrates:
1. Using Curator's LLM to generate poem training data
2. Converting the curated data to chat format
3. Fine-tuning a model using TinkerTrainer

Usage:
    # Set your API keys
    export OPENAI_API_KEY="your-openai-key"
    export TINKER_API_KEY="your-tinker-key"  # Optional, runs in mock mode without it

    # Run the example
    poetry run python examples/poem_finetuning_example.py

    # Run with mock data (no API keys required)
    poetry run python examples/poem_finetuning_example.py --mock
"""

import argparse

from pydantic import BaseModel, Field

from bespokelabs.curator import LLM, TinkerTrainer, TinkerTrainerConfig


# =============================================================================
# Step 1: Define the data schema for poem generation
# =============================================================================
class Poem(BaseModel):
    """Schema for generated poems."""

    theme: str = Field(description="The theme or topic of the poem")
    style: str = Field(description="The style of the poem (haiku, sonnet, free verse, etc.)")
    poem: str = Field(description="The complete poem text")
    explanation: str = Field(description="Brief explanation of the poem's meaning")


class PoemRequest(BaseModel):
    """Input schema for poem generation requests."""

    theme: str
    style: str


# =============================================================================
# Step 2: Create the Curator LLM for data generation
# =============================================================================
def create_poem_curator() -> LLM:
    """Create a Curator LLM configured for poem generation."""

    def prompt_func(row: dict) -> str:
        """Generate a prompt for creating a poem."""
        return f"""You are a creative poet. Write a {row['style']} poem about "{row['theme']}".

Be creative and evocative. The poem should capture the essence of the theme
while adhering to the conventions of the specified style.

After the poem, provide a brief explanation of its meaning and imagery."""

    def parse_func(row: dict, response: Poem) -> dict:
        """Parse the response into training format."""
        return {
            "theme": response.theme,
            "style": response.style,
            "poem": response.poem,
            "explanation": response.explanation,
            "request_theme": row["theme"],
            "request_style": row["style"],
        }

    return LLM(
        model_name="gpt-4o-mini",
        prompt_func=prompt_func,
        parse_func=parse_func,
        response_format=Poem,
    )


# =============================================================================
# Step 3: Generate training data using Curator
# =============================================================================
def generate_poem_dataset() -> list:
    """Generate a dataset of poems using Curator."""
    # Define poem requests - combinations of themes and styles
    themes = [
        "autumn leaves falling",
        "a rainy evening",
        "childhood memories",
        "the ocean at dawn",
        "a forgotten garden",
        "city lights at night",
        "first snowfall",
        "a traveler's journey",
        "mountain solitude",
        "spring blossoms",
    ]

    styles = [
        "haiku",
        "free verse",
        "sonnet",
        "limerick",
    ]

    # Create input requests as a list (Curator accepts any iterable)
    requests = []
    for theme in themes:
        for style in styles:
            requests.append({"theme": theme, "style": style})

    print(f"Generated {len(requests)} poem requests")
    print(f"Themes: {len(themes)}, Styles: {len(styles)}")

    # Create curator and generate poems
    poem_curator = create_poem_curator()

    print("\nCurating poems with Curator...")
    poem_response = poem_curator(requests)

    # Convert response to list
    poem_data = poem_response.to_list()

    print(f"Generated {len(poem_data)} poems")
    return poem_data


# =============================================================================
# Step 4: Convert curated data to chat format for fine-tuning
# =============================================================================
def convert_to_chat_format(poem_data: list) -> list:
    """Convert the poem dataset to chat format for fine-tuning."""
    chat_data = []

    for row in dataset:
        # Create a training example where:
        # - User asks for a poem with specific theme and style
        # - Assistant provides the poem with explanation
        messages = [
            {
                "role": "system",
                "content": "You are a creative poet who writes beautiful poems in various styles. "
                "When asked to write a poem, you provide both the poem and a brief explanation of its meaning.",
            },
            {
                "role": "user",
                "content": f"Write a {row['style']} poem about \"{row['theme']}\".",
            },
            {
                "role": "assistant",
                "content": f"{row['poem']}\n\n---\n\n**Explanation:** {row['explanation']}",
            },
        ]

        chat_data.append({"messages": messages})

    return chat_data


# =============================================================================
# Step 5: Fine-tune with TinkerTrainer
# =============================================================================
def finetune_poem_model(training_data: list) -> TinkerTrainer:
    """Fine-tune a model on the poem data using TinkerTrainer."""
    # Configure the trainer
    config = TinkerTrainerConfig(
        base_model="Qwen/Qwen3-8B",
        epochs=3,
        batch_size=4,
        max_seq_length=2048,
        adam_params={
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
        },
        lora_config={
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
        log_every_n_steps=5,
        warmup_steps=10,
    )

    print("\nTrainer Configuration:")
    print(f"  Base Model: {config.base_model}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.adam_params.learning_rate}")
    print(f"  LoRA Rank: {config.lora_config.rank}")

    # Create trainer and run training
    trainer = TinkerTrainer(config)

    print(f"\nStarting fine-tuning on {len(training_data)} examples...")
    result = trainer.train(training_data)

    # Display results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"  Final Loss: {result.final_loss:.4f}")
    print(f"  Total Steps: {result.total_steps}")
    print(f"  Total Time: {result.total_time:.2f}s")
    print(f"  Samples Processed: {result.samples_processed}")
    print(f"  Tokens Processed: {result.tokens_processed:,}")
    print(f"  Weights Saved: {result.weights_name}")

    return trainer


# =============================================================================
# Step 6: Test the fine-tuned model
# =============================================================================
def test_finetuned_model(trainer: TinkerTrainer):
    """Test the fine-tuned model with sample prompts."""
    test_prompts = [
        'Write a haiku about "the stillness of midnight".',
        'Write a free verse poem about "a forgotten melody".',
        'Write a limerick about "a curious cat".',
    ]

    print("\n" + "=" * 50)
    print("Testing Fine-tuned Model")
    print("=" * 50)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)

        response = trainer.sample(
            prompt,
            system_prompt="You are a creative poet who writes beautiful poems in various styles.",
        )

        print(f"Response: {response}")


# =============================================================================
# Main execution
# =============================================================================
def get_mock_poem_data() -> list:
    """Get mock poem data for demonstration when API key is not available."""
    return [
        {
            "theme": "autumn leaves falling",
            "style": "haiku",
            "poem": "Crimson leaves descend\nDancing on the autumn breeze\nEarth's last warm embrace",
            "explanation": "This haiku captures the beauty and melancholy of autumn.",
        },
        {
            "theme": "a rainy evening",
            "style": "free verse",
            "poem": "The rain whispers secrets\nto the lonely streetlamps,\neach drop a memory\nfalling from grey clouds\nthat hold tomorrow's promises.",
            "explanation": "A contemplative piece about finding meaning in rainy evenings.",
        },
        {
            "theme": "childhood memories",
            "style": "free verse",
            "poem": "In the attic of my mind,\ndusty boxes hold\nthe laughter of summers past,\nthe scraped knees and fireflies,\nmoments golden and gone.",
            "explanation": "Nostalgia for the innocence and joy of childhood.",
        },
        {
            "theme": "the ocean at dawn",
            "style": "haiku",
            "poem": "Pink sky meets the sea\nWaves carry the sun's first light\nNew day awakens",
            "explanation": "A peaceful meditation on dawn over the ocean.",
        },
        {
            "theme": "city lights at night",
            "style": "free verse",
            "poem": "Neon dreams flicker\nacross rain-slicked streets,\na thousand windows\neach a story untold,\nthe city never sleeps.",
            "explanation": "An ode to the energy and mystery of urban nights.",
        },
        {
            "theme": "first snowfall",
            "style": "haiku",
            "poem": "White silence descends\nWorld wrapped in frozen wonder\nFootprints yet to come",
            "explanation": "The pristine beauty of the first snow of winter.",
        },
        {
            "theme": "a curious cat",
            "style": "limerick",
            "poem": (
                "A curious cat named Lou\nFound a box that was perfectly new\n"
                "He jumped right inside\nWith a satisfied pride\nAnd declared it the best thing he knew"
            ),
            "explanation": "A playful limerick about cats and their love of boxes.",
        },
        {
            "theme": "mountain solitude",
            "style": "free verse",
            "poem": "Above the treeline,\nwhere the air thins to whispers,\nI find the silence\nthat speaks louder than words,\nthe mountain's ancient wisdom.",
            "explanation": "Finding peace and perspective in mountain solitude.",
        },
    ]


def main(use_mock: bool = False):
    """Run the complete poem curation and fine-tuning pipeline.

    Args:
        use_mock: If True, use mock data instead of calling APIs
    """
    import os

    print("=" * 60)
    print("Poem Generation & Fine-tuning Pipeline")
    print("=" * 60)

    # Step 1: Generate poem dataset using Curator
    print("\n[Step 1] Generating poem dataset with Curator...")

    # Use mock data if requested or if no API key
    if use_mock:
        print("  (Using mock data as requested)")
        poem_data = get_mock_poem_data()
    elif os.environ.get("OPENAI_API_KEY"):
        poem_data = generate_poem_dataset()
    else:
        print("  (No OPENAI_API_KEY found, using mock data)")
        poem_data = get_mock_poem_data()

    print(f"  Dataset size: {len(poem_data)} poems")

    # Step 2: Convert to chat format
    print("\n[Step 2] Converting to chat format for fine-tuning...")

    # Convert poem data to chat format directly (works with list of dicts)
    training_data = []
    for row in poem_data:
        messages = [
            {
                "role": "system",
                "content": "You are a creative poet who writes beautiful poems in various styles. "
                "When asked to write a poem, you provide both the poem and a brief explanation of its meaning.",
            },
            {
                "role": "user",
                "content": f"Write a {row['style']} poem about \"{row['theme']}\".",
            },
            {
                "role": "assistant",
                "content": f"{row['poem']}\n\n---\n\n**Explanation:** {row['explanation']}",
            },
        ]
        training_data.append({"messages": messages})

    print(f"  Training examples: {len(training_data)}")

    # Show a sample
    print("\n  Sample training example:")
    sample = training_data[0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"    [{role}]: {content}")

    # Step 3: Fine-tune with TinkerTrainer
    print("\n[Step 3] Fine-tuning with TinkerTrainer...")
    trainer = finetune_poem_model(training_data)

    # Step 4: Test the model
    print("\n[Step 4] Testing the fine-tuned model...")
    test_finetuned_model(trainer)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    return trainer, poem_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poem curation and fine-tuning example")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of calling APIs (for testing)",
    )
    args = parser.parse_args()

    trainer, dataset = main(use_mock=args.mock)
