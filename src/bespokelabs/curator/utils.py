import os
import requests
import asyncio
import json
import logging
import typing as t
import uuid

from datasets import Dataset, DatasetDict, load_dataset
from rich.progress import Progress

from bespokelabs.curator.constants import _CURATOR_DEFAULT_CACHE_DIR
from bespokelabs.curator import _CONSOLE, constants
from bespokelabs.curator.client import Client, _SessionStatus
from bespokelabs.curator.log import USE_RICH_DISPLAY
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop

logger = logging.getLogger(__name__)


def push_to_viewer(
    dataset: Dataset | str,
    session_id: str | None = None,
    hf_params: t.Optional[t.Dict] = None,
    max_concurrent_requests: int = 100,
):
    """Push a dataset to the Curator Viewer.

    Args:
        dataset (Dataset | str): The dataset to push to the Curator Viewer.
        session_id (str | None): Existing session id.
        hf_params: (dict): Huggingface parameters for load dataset.
        max_concurrent_requests (int): Max concurrent requests limit.

    Returns:
        str: The URL to view the data
    """
    if isinstance(dataset, str):
        logger.info(f"Downloading dataset {dataset} from huggingface")
        hf_params = {} or hf_params
        dataset = load_dataset(dataset, **hf_params)
    if isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected a `datasets.Dataset` object, but received a `datasets.DatasetDict`. "
            "Please select a specific split (e.g., `dataset['train']`) before passing it."
        )
    elif not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a `datasets.Dataset` object, but received a `{type(dataset)}`.")

    client = Client(hosted=True)
    uid = str(uuid.uuid4())
    metadata = {
        "run_hash": uid,
        "dataset_hash": uid,
        "prompt_func": "N/A",
        "model_name": "simulated_dataset",
        "response_format": "N/A",
        "batch_mode": False,
        "status": _SessionStatus.STARTED,
    }

    if session_id is None:
        session_id = client.create_session(metadata)
    else:
        client._session = session_id

    if not client.session:
        raise Exception("Failed to create session.")

    view_url = f"{constants.PUBLIC_CURATOR_VIEWER_DATASET_URL}/{session_id}"
    viewer_text = (
        f"[bold white]Curator Viewer:[/bold white] [blue][link={view_url}]:sparkles: Open Curator Viewer[/link] :sparkles:[/blue]\n[dim]{view_url}[/dim]\n"
    )
    if USE_RICH_DISPLAY:
        _CONSOLE.print(viewer_text)
    else:
        logger.info(viewer_text)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def send_responses():
        with Progress() as progress:
            task = progress.add_task("[cyan]Uploading dataset rows...", total=len(dataset))

            async def send_row(idx, row):
                nonlocal task, progress
                response_data = {"parsed_response_message": [row]}
                response_data_json = json.dumps(response_data)
                await client.stream_response(response_data_json, idx)
                progress.update(task, advance=1)

            for idx, row in enumerate(dataset):
                async with semaphore:
                    await send_row(idx, row)

            await client.session_completed()

    run_in_event_loop(send_responses())
    return view_url


def load_dataset(dataset_id: str):
    """Load a dataset from a curator dataset id."""
    url = f"http://localhost:8080/v0/viewer/sessions/{dataset_id}/fetch_data"

    curator_cache_dir = os.environ.get(
        "CURATOR_CACHE_DIR",
        os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
    )
    
    # Create cache directory if it doesn't exist
    os.makedirs(curator_cache_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(curator_cache_dir, f"dataset_{dataset_id}.arrow")
    
    # Check if dataset is already cached
    if os.path.exists(cache_file):
        logger.debug(f"Loading dataset {dataset_id} from cache")
        return Dataset.from_file(cache_file)
    
    # If not cached, download it
    logger.debug(f"Downloading dataset {dataset_id}")
    # url = f"{constants.PUBLIC_CURATOR_VIEWER_DATASET_URL}/{dataset_id}/fetch_data"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(cache_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    # Return the dataset
    return Dataset.from_file(cache_file)
