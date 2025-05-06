# just upload files, a thin wrapper around push_to_viewer

import logging
from pathlib import Path

from datasets import load_dataset

from bespokelabs.curator.utils import push_to_viewer

logger = logging.getLogger(__name__)


def upload(path: str, split: str = "train"):
    """Uploads a dataset file to the Hugging Face Hub.

    Args:
        path: The local path to the dataset file.
        split: The name of the split to upload (e.g., "train", "test").

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file type is not supported.
    """
    # load file into a huggingface dataset

    # it could be a huggingface dataset or a local file
    # first check if it is a huggingface dataset
    try:
        dataset = load_dataset(path, split=split)

    except Exception:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist") from None

        if path.suffix not in [".jsonl", ".json", ".csv", ".parquet"]:
            raise ValueError("Only jsonl, json, csv, and parquet files are supported currently") from None

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

    link = push_to_viewer(dataset)
    return link
