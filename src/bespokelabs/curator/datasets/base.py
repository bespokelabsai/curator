# just upload files, a thin wrapper around push_to_viewer

from pathlib import Path
from bespokelabs.curator.utils import push_to_viewer
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def upload(path: str, name: str, split: str = "train"):
    # load file into a huggingface dataset

    # it could be a huggingface dataset or a local file
    # first check if it is a huggingface dataset
    try:
        dataset = load_dataset(path, split=split)

    except Exception as e:
        
        path = Path(path)
        if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

        if not (path.suffix in [".jsonl", ".json", ".csv", ".parquet"]):
            raise ValueError("Only jsonl, json, csv, and parquet files are supported currently")

        try:

            if path.suffix == ".jsonl" or path.suffix == ".json":
                format = "json"
            elif path.suffix == ".csv":
                format = "csv"
            elif path.suffix == ".parquet":
                format = "parquet"

            dataset = load_dataset(format, data_files=str(path), split=split)

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise e

    link = push_to_viewer(dataset, name)
    return link
