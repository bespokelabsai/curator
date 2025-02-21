import asyncio
import json
import logging
import typing as t
import uuid

from datasets import Dataset, load_dataset
from rich.progress import Progress

from bespokelabs.curator import _CONSOLE, constants
from bespokelabs.curator.client import Client, _SessionStatus
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop

logger = logging.getLogger(__name__)


def push_to_viewer(dataset: Dataset | str, hf_params: t.Optional[t.Dict] = None, max_concurrent_requests: int = 100):
    """Push a dataset to the Curator Viewer.

    Args:
        dataset (Dataset | str): The dataset to push to the Curator Viewer.
        hf_params: (dict): Huggingface parameters for load dataset.
        max_concurrent_requests (int): Max concurrent requests limit.

    Returns:
        str: The URL to view the data
    """
    if isinstance(dataset, str):
        logger.info(f"Downloading dataset {dataset} from huggingface")
        hf_params = {} or hf_params
        dataset = load_dataset(dataset, **hf_params)

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

    session_id = client.create_session(metadata)
    if not client.session:
        raise Exception("Failed to create session.")

    view_url = f"{constants.PUBLIC_CURATOR_VIEWER_DATASET_URL}/{session_id}"
    viewer_text = (
        f"[bold white]Curator Viewer:[/bold white] "
        f"[blue][link={view_url}]:sparkles: Open Curator Viewer[/link] :sparkles:[/blue]\n"
        f"[dim]{view_url}[/dim]\n"
    )
    _CONSOLE.print(viewer_text)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def send_responses():
        with Progress() as progress:
            task = progress.add_task("[cyan]Uploading dataset rows...", total=len(dataset))

            async def send_row(idx, row):
                nonlocal task, progress
                response_data = {"parsed_response_message": [row]}
                response_data_json = json.dumps(response_data)
                async with semaphore:
                    await client.stream_response(response_data_json, idx)
                progress.update(task, advance=1)

            tasks = [send_row(idx, row) for idx, row in enumerate(dataset)]

            await asyncio.gather(*tasks)
            await client.session_completed()

    run_in_event_loop(send_responses())
    return view_url
