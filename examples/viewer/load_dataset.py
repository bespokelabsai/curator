from bespokelabs.curator.utils import load_dataset, push_to_viewer

# Upload HF url
link = push_to_viewer("zed-industries/zeta", hf_params={"split": "train[:10]"})
print("Link to the curator viewer: ", link)
dataset_id = link.split("/")[-1]
dataset = load_dataset(dataset_id)
print(dataset[0])
