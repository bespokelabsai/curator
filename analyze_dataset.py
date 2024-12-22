"""Script to analyze the PII detection dataset."""
import pandas as pd
from datasets import load_from_disk
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import json
import re

def count_chinese_chars(text):
    """Count the number of Chinese characters in text."""
    return len(re.findall(r'[\u4e00-\u9fff]', text))

def analyze_dataset():
    # Create plots directory if it doesn't exist
    import os
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_from_disk('pii_dataset')
    df = pd.DataFrame(dataset)
    
    # Basic statistics
    total_examples = len(df)
    total_entities = sum(len(row['output']) for row in dataset)
    avg_entities = total_entities / total_examples
    
    print(f"\nDataset Statistics:")
    print(f"Total examples: {total_examples}")
    print(f"Total PII entities: {total_entities}")
    print(f"Average entities per example: {avg_entities:.2f}")
    
    # Analyze entity types and sources
    entity_types = []
    entity_sources = []
    for row in dataset:
        for entity in row['output']:
            entity_types.append(entity['entity_type'])
            entity_sources.append(entity['entity_source'])
    
    # Create entity type distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=entity_types)
    plt.title('Distribution of PII Entity Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/entity_types_distribution.png')
    plt.close()
    
    # Create entity source distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=entity_sources)
    plt.title('Distribution of Entity Sources')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/entity_sources_distribution.png')
    plt.close()
    
    # Analyze text language composition
    chinese_char_counts = [count_chinese_chars(row['input']) for row in dataset]
    total_char_counts = [len(row['input']) for row in dataset]
    chinese_ratios = [c/t if t > 0 else 0 for c, t in zip(chinese_char_counts, total_char_counts)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(chinese_ratios, bins=20)
    plt.title('Distribution of Chinese Character Ratio in Texts')
    plt.xlabel('Ratio of Chinese Characters')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/chinese_ratio_distribution.png')
    plt.close()
    
    # Print detailed statistics
    print("\nEntity Type Distribution:")
    type_counter = Counter(entity_types)
    for entity_type, count in type_counter.most_common():
        print(f"{entity_type}: {count}")
    
    print("\nEntity Source Distribution:")
    source_counter = Counter(entity_sources)
    for source, count in source_counter.most_common():
        print(f"{source}: {count}")
    
    print("\nText Length Statistics:")
    print(f"Average text length: {sum(total_char_counts)/len(total_char_counts):.2f} characters")
    print(f"Average Chinese characters: {sum(chinese_char_counts)/len(chinese_char_counts):.2f}")
    print(f"Average Chinese ratio: {sum(chinese_ratios)/len(chinese_ratios):.2%}")

if __name__ == "__main__":
    analyze_dataset()
