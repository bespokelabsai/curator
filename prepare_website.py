"""Script to prepare the dataset and files for the website."""
import os
import json
import shutil
from datasets import load_from_disk

def prepare_website():
    # Create website directories
    os.makedirs('website/plots', exist_ok=True)
    
    # Copy plot images to website directory
    plot_files = [
        'entity_types_distribution.png',
        'entity_sources_distribution.png',
        'chinese_ratio_distribution.png'
    ]
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            shutil.copy(plot_file, f'website/plots/{plot_file}')
    
    # Convert dataset to JSON
    dataset = load_from_disk('pii_dataset')
    examples = []
    for i, example in enumerate(dataset):
        examples.append({
            'input': example['input'],
            'output': example['output']
        })
    
    # Save dataset as JSON
    with open('website/pii_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    prepare_website()
