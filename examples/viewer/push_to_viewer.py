from datasets import load_dataset

from bespokelabs.curator.utils import push_to_viewer

# Upload HF url
link = push_to_viewer("zed-industries/zeta", hf_params={"split": "train[:10]"})
print("Link to the curator viewer: ", link)

# Upload HF dataset

dataset = load_dataset("zed-industries/zeta", split="train[:10]")
link = push_to_viewer(dataset)
print("Link to the curator viewer: ", link)
