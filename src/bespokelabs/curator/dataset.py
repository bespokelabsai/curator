from datasets import Dataset as HFDataset
import json
from huggingface_hub import DatasetCard

DATASET_CARD_TEMPLATE = """
---
language: en
license: mit
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

class Dataset(HFDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # def sort(self, *args, **kwargs):
    #     sorted_dataset = super().sort(*args, **kwargs)
    #     return self.__class__(sorted_dataset._data)

    # def remove_columns(self, *args, **kwargs):
    #     removed_dataset = super().remove_columns(*args, **kwargs)
    #     return self.__class__(removed_dataset._data)
     
    def push_to_hub(
        self,                     
        repo_id: str,
        **kwargs
    ):
        super().push_to_hub(repo_id, **kwargs)
        card = DatasetCard(
            DATASET_CARD_TEMPLATE.format(
                dataset_name=repo_id.split("/")[-1],
                repo_id=repo_id,
                sample=json.dumps(self[0], indent=4),
            )
        )
        card.push_to_hub(repo_id, **kwargs)