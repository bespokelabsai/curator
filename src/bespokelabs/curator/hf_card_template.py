HUGGINGFACE_CARD_TEMPLATE = """
---
language: en
license: mit
tags:
- curator
---

<a href="https://github.com/bespokelabsai/curator/">
 <img src="https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k/resolve/main/made_with_curator.png" alt="Made with Curator" width=200px>
</a>

## Dataset card for {dataset_name}

This dataset was made with [Curator](https://github.com/bespokelabsai/curator/).

## Dataset details

A sample from the dataset:

```python
{sample}
```

## Loading the dataset

You can load this dataset using the following code:

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", split="default")
```

"""
