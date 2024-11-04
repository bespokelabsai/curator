"""Bella: Bespoke Labs Synthetic Data Generation Library."""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from itertools import product
from typing import Callable, Iterable, Optional

from datasets import Dataset
from datasets.arrow_writer import ArrowWriter

from api_request_parallel_processor import process_api_requests_from_file
from prompt import Prompter


def _create_requests_file(
    dataset: Iterable, requests_file: str, prompter: Prompter, resume: bool = True
):
    if os.path.exists(requests_file):
        if resume:
            logging.info(f"Loading existing jobs from {requests_file}")
            logging.debug(
                f"Alternatively, delete the jobs file and re-run the annotator: `rm -rf {requests_file}`"
            )
            # count existing jobs in file and print first job
            with open(requests_file, "r") as f:
                num_jobs = sum(1 for _ in f)
                f.seek(0)
                first_job = json.loads(f.readline())
            logging.debug(f"Found {num_jobs} jobs in {requests_file}")
            logging.debug("Example job:")
            logging.debug(json.dumps(first_job, indent=2))
        else:
            error_message = (
                f"Existing job file {requests_file}. "
                f"Delete the jobs file and re-run the annotator: `rm -rf {requests_file}`. "
                f"Or run the annotator with the --resume flag to continue from the previous run."
            )
            raise ValueError(error_message)
    else:
        os.makedirs(os.path.dirname(requests_file), exist_ok=True)
        with open(requests_file, "w") as f:
            if len(dataset) == 0:
                request = prompter.get_request_object(dict(), 0)
                f.write(json.dumps(request) + "\n")
            else:
                for idx, sample in enumerate(dataset):
                    request = prompter.get_request_object(sample, idx)
                    f.write(json.dumps(request) + "\n")
        logging.info(f"Requests file {requests_file} written to disk.")


def _read_responses_file_and_write_to_dataset(
    prompter: Prompter, responses_file, dataset_file, output_column
):
    total_count = 0
    failed_count = 0
    with ArrowWriter(path=dataset_file) as writer:
        with open(responses_file, "r") as f_in:
            for line in f_in:
                total_count += 1
                try:
                    response = json.loads(line)
                    if isinstance(response[1], list):
                        # A failed requests contains a list of all the errors before max_retries
                        logging.info(
                            f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}"
                        )
                        failed_count += 1
                        continue

                    metadata = response[2]
                    response_message = response[1]["choices"][0]["message"]["content"]

                    # Parse the response for structured output
                    if prompter.response_format:
                        response_parsed = prompter.response_format.model_validate_json(
                            response_message
                        )
                        response_parsed = response_parsed.model_dump()
                    else:
                        response_parsed = response_message

                    sample = metadata["sample"]

                    sample[output_column] = response_parsed
                    if sample == None:
                        raise ValueError("Trying to write a None sample")
                    writer.write(sample)

                except Exception as e:
                    logging.warning(f"Error: {e}")
                    logging.warning(f"Full response: {response}")
                    continue
        logging.debug(f"Read {total_count} responses, {failed_count} failed")
        if failed_count == total_count:
            raise ValueError("All requests failed")
        writer.finalize()


def _parse_responses_file(prompter: Prompter, responses_file):
    total_count = 0
    failed_count = 0
    samples = []
    with open(responses_file, "r") as f_in:
        for line in f_in:
            total_count += 1
            try:
                response = json.loads(line)
                if isinstance(response[1], list):
                    # A failed requests contains a list of all the errors before max_retries
                    logging.info(
                        f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}"
                    )
                    failed_count += 1
                    continue

                response_message = response[1]["choices"][0]["message"]["content"]
                metadata = response[2]

                # Parse the response for structured output
                if prompter.response_format:
                    response_parsed = prompter.response_format.model_validate_json(
                        response_message
                    )
                    response_parsed = response_parsed.model_dump()
                else:
                    response_parsed = response_message

                samples.append((metadata["request_idx"], response_parsed))

            except Exception as e:
                logging.warning(f"Error: {e}")
                logging.warning(f"Full response: {response}")
                continue
    logging.debug(f"Read {total_count} responses, {failed_count} failed")
    # Sort by idx then return only the responses
    samples.sort(key=lambda x: x[0])
    samples = [sample[1] for sample in samples]  # Keep only the response_parsed values
    if failed_count == total_count:
        raise ValueError("All requests failed")
    return samples


PROMPT_DUMMY_VALUE = "___bella_dummy_value___"


class EmptyDataset:
    empty_bella_dataset = True
    fingerprint = "empty"


def empty():
    return EmptyDataset()


def completions(
    dataset: Iterable = [],
    prompter: Prompter = None,
    name: Optional[str] = None,
    resume: bool = True,
) -> "Dataset":
    """
    Apply structured completions to the dataset using specified model and prompts.

    Args:
        prompter: A function that goes from row to request object
        output_column: Name of the column to store the response
        name: Name of the task
        resume: Whether to resume from the previous completions run

    Returns:
        A Dataset with the completions added in the output_column
    """
    if prompter is None:
        raise ValueError("prompter is required")

    bella_cache_dir = os.environ.get(
        "BELLA_CACHE_DIR", os.path.expanduser("~/.cache/bella")
    )

    # Convert all elements to strings and join them before hashing
    fingerprint_str = "_".join(
        [
            str(uuid.uuid4()),
            str(prompter.user_prompt),
            str(prompter.system_prompt),
            str(prompter.model_name),
            str(prompter.response_format.schema_json()),
        ]
    )

    fingerprint = hashlib.md5(fingerprint_str.encode("utf-8")).hexdigest()

    name = f"{name.replace(' ', '-')}--{fingerprint}" if name else fingerprint
    requests_path = os.path.join(bella_cache_dir, f"{name}/requests.jsonl")
    responses_path = os.path.join(bella_cache_dir, f"{name}/responses.jsonl")

    _create_requests_file(dataset, requests_path, prompter, resume)
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_path,
            save_filepath=responses_path,
            request_url="https://api.openai.com/v1/chat/completions",
            max_attempts=5,
            resume=True,
            model=prompter.model_name,
        )
    )
    return _parse_responses_file(prompter, responses_path)


def map(dataset: Iterable, func: Callable):
    samples = []
    for sample in dataset:
        result = func(sample)
        if isinstance(result, list):
            samples.extend(result)
        else:
            samples.append(result)
    return samples


def flatten_list(batch):
    """
    Flatten a dataset with rows containing lists.

    If multiple columns contain lists, the cartesian product of all the lists is taken.

    Example:
    dataset: [{"name": "John", "hobbies": ["reading", "traveling"]}]
    flattened dataset: [{"name": "John", "hobbies": "reading"}, {"name": "John", "hobbies": "traveling"}]

    Args:
        batch: A batch with a column of lists.

    Returns:
        A dataset with the column of lists flattened.
    """
    # A batch is a dict of lists of equal length.
    cols = list(batch.values())
    batch_length = len(cols[0])

    # Iterate over the rows of the batch. For each row, we
    # want to create new rows by taking the cartesian product
    # of all the values of the columns in the row.
    # For example, let's say we have the following row:
    # {"a": [1,2], "b": [3,4]}
    # We want to create the following new rows:
    # [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    new_rows = []
    for i in range(batch_length):
        values = []
        for col in cols:
            values.append(col[i] if isinstance(col[i], list) else [col[i]])
        new_rows.extend(list(product(*values)))

    # Convert the new rows into the "column-oriented" format.
    new_batch = {}
    for row in new_rows:
        for k, v in zip(batch.keys(), row):
            if k not in new_batch:
                new_batch[k] = []
            new_batch[k].append(v)

    return new_batch


def flatten_dict(dataset: Dataset) -> Dataset:
    dataset = dataset.flatten()
    return dataset.rename_columns(
        {col_name: col_name.replace(".", "__") for col_name in dataset.column_names}
    )
