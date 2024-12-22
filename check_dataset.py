"""Script to check the generated PII dataset."""
from datasets import load_from_disk
import json

def print_example(example, index: int, total: int, prompt_index: int):
    """Print a single example in a readable format."""
    prompt_types = [
        "Professional Biography",
        "Business Email",
        "Legal Contract",
        "News Article",
        "Bilingual Social Media Post"
    ]
    
    print(f"\n{'='*40}")
    print(f"Example {index + 1}/{total} - Type: {prompt_types[prompt_index]}")
    print(f"{'='*40}")
    
    print("\nGenerated Text:")
    print("-" * 80)
    print(example["input"])
    print("-" * 80)
    
    print("\nDetected PII Entities:")
    for entity in example["output"]:
        print(f"- {entity['entity_type']:<10}: {entity['entity_value']} ({entity['entity_source']})")
    print("\n")

def main():
    dataset = load_from_disk('pii_dataset')
    print(f"\nTotal examples in dataset: {len(dataset)}")
    
    # Print first example of each type
    seen_types = set()
    for i, example in enumerate(dataset):
        prompt_index = i % 5
        if prompt_index not in seen_types and len(seen_types) < 5:
            print_example(example, i, len(dataset), prompt_index)
            seen_types.add(prompt_index)

if __name__ == "__main__":
    main()
