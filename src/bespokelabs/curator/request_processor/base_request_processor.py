import asyncio
import glob
import json
import logging
import os
import resource
from abc import ABC, abstractmethod
from math import ceil
from typing import Optional

import aiofiles
import pyarrow
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from pydantic import BaseModel, ValidationError

from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import GenericResponse

logger = logger = logging.getLogger(__name__)

CACHE_MSG = "If you want to regenerate the dataset, disable or delete the cache."


class BaseRequestProcessor(ABC):
    """
    Base class for all request processors.
    """

    def __init__(self, batch_size: Optional[int] = None):
        self.batch_size = batch_size
        # Increase the number of open file descriptors to avoid "Too many open files" errors
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_limit = min(10_000_000, hard)
        logger.debug(
            f"Adjusting file descriptor limit from {soft} to {desired_limit} (hard limit: {hard})"
        )
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_limit, hard))

    @abstractmethod
    def get_rate_limits(self) -> dict:
        """
        Returns the rate limits for the API.

        Returns:
            dict: A dictionary containing the rate limit information.
        """
        pass

    @abstractmethod
    def create_api_specific_request(sself, generic_request: GenericRequest) -> dict:
        """
        Creates a API-specific request body from a GenericRequest.

        Must use request_body.metadata.dataset_row or request_body.metadata.dataset_row_idx

        Note that the generic request body has both the original dataset row or the index of the row in the original dataset.
        What you pass on depends on what is works with the API. For example, OpenAI and Anthropic offline batch API allows you to pass a custom_id (can be index).
        For online, the api_parallel_processor can store the original dataset row in the metadata.

        Returns:
            dict: API specific request body
        """
        pass

    def run(
        self,
        dataset: Dataset,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """
        Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)

        Returns:
            Dataset: Completed dataset
        """
        pass

    def create_request_files(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        prompt_formatter: PromptFormatter,
    ) -> list[str]:
        """
        Creates a request file if they don't already exist or use existing.

        Args:
            dataset (Dataset): The dataset to be processed.
            working_dir (str): The directory where request files will be saved.

        Returns:
            list[str]: Paths to the request files that were created.
        """
        os.makedirs(working_dir, exist_ok=True)
        request_files = glob.glob(f"{working_dir}/requests_*.jsonl")

        # By default use existing requests in working_dir
        if len(request_files) > 0:
            logger.info(f"Using cached requests. {CACHE_MSG}")
            # count existing jobs in file and print first job
            with open(request_files[0], "r") as f:
                # Count lines and store first job
                first_job = None
                num_jobs = 0
                for i, line in enumerate(f):
                    if i == 0:
                        first_job = json.loads(line)
                    num_jobs = i + 1

                if num_jobs > 0:
                    logger.debug(
                        f"There are {num_jobs} existing requests in {request_files[0]}\n"
                        f"Example request in {request_files[0]}:\n{json.dumps(first_job, default=str, indent=2)}"
                    )
                    return request_files

        # Create new requests file
        request_file = f"{working_dir}/requests_0.jsonl"
        request_files = [request_file]

        if dataset is None:
            with open(request_file, "w") as f:
                generic_request = prompt_formatter.create_generic_request(dict(), 0)
                f.write(json.dumps(generic_request.model_dump(), default=str) + "\n")
            return request_files

        if self.batch_size:
            num_batches = ceil(len(dataset) / self.batch_size)
            request_files = [f"{working_dir}/requests_{i}.jsonl" for i in range(num_batches)]

            async def create_all_request_files():
                tasks = [
                    self.acreate_request_file(
                        dataset,
                        prompt_formatter,
                        request_files[i],
                        start_idx=i * self.batch_size,
                    )
                    for i in range(num_batches)
                ]
                await asyncio.gather(*tasks)

            run_in_event_loop(create_all_request_files())
        else:
            run_in_event_loop(self.acreate_request_file(dataset, prompt_formatter, request_file))

        return request_files

    # NOTE(Ryan): Instead of doing this, just iterate over iterable and keep counter and change filename when hit batch_size, this will be slower but this whole thing is dominated by llm calls anyways
    async def acreate_request_file(
        self,
        dataset: Dataset,
        prompt_formatter: PromptFormatter,
        request_file: str,
        start_idx: int = 0,
    ) -> str:
        if self.batch_size is not None:
            end_idx = min(start_idx + self.batch_size, len(dataset))
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            end_idx = len(dataset)

        # NOTE(Ryan): For loops only for IterableDataset which allows for _very_ large datasets, when start_idx and batch_size are not specified
        async with aiofiles.open(request_file, "w") as f:
            for idx, dataset_row in enumerate(dataset):
                dataset_row_idx = idx + start_idx
                # Get the generic request from the map function
                request = prompt_formatter.create_generic_request(dataset_row, dataset_row_idx)
                await f.write(json.dumps(request.model_dump(), default=str) + "\n")
        logger.info(f"Wrote {end_idx - start_idx} requests to {request_file}.")

    def attempt_loading_cached_dataset(
        self, working_dir: str, parse_func_hash: str
    ) -> Optional[Dataset]:
        dataset_file = f"{working_dir}/{parse_func_hash}.arrow"
        if os.path.exists(dataset_file):
            logger.debug(f"Loading dataset from {dataset_file}")
            try:
                output_dataset = Dataset.from_file(dataset_file)
                logger.info(f"Using cached output dataset. {CACHE_MSG}")
                return output_dataset
            except pyarrow.lib.ArrowInvalid as e:
                os.remove(dataset_file)
                logger.warning(
                    f"Failed to load dataset from {dataset_file}, "
                    "which was likely corrupted by a failed previous run. "
                    "Deleted file and attempting to regenerate dataset from cached LLM responses."
                )

    def create_dataset_files(
        self,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """
        Creates the request files if they don't already exist or use existing.
        A single request file (requests_0.jsonl) or multiple request files
        (requests_0.jsonl, requests_1.jsonl, etc.) are created depending on
        batch_size.

        Args:
            dataset (Dataset): The dataset to be processed.
            working_dir (str): The directory where request files will be saved.
            prompt_formatter (PromptFormatter): The prompt formatter to use for parsing the responses.

        Returns:
            Dataset: Completed dataset
        """
        total_responses_count = 0
        failed_responses_count = 0

        responses_files = glob.glob(f"{working_dir}/responses_*.jsonl")
        if len(responses_files) == 0:
            raise ValueError(f"No responses files found in {working_dir}")

        error_help = (
            "Please check your `parse_func` is returning a valid row (dict) "
            "or list of rows (list of dicts) and re-run. "
            "Dataset will be regenerated from cached LLM responses."
        )

        # Process all response files
        dataset_file = f"{working_dir}/{parse_func_hash}.arrow"
        with ArrowWriter(path=dataset_file) as writer:
            for responses_file in responses_files:
                with open(responses_file, "r") as f_in:
                    for generic_response_string in f_in:
                        total_responses_count += 1
                        response = GenericResponse.model_validate_json(generic_response_string)

                        # response.response_errors is not None IFF response.response_message is None
                        if response.response_errors is not None:
                            failed_responses_count += 1
                            continue

                        if prompt_formatter.response_format:
                            # Response message is a string, which is converted to a dict
                            # The dict is then used to construct the response_format Pydantic model
                            try:
                                # First try to parse the response message as JSON
                                if isinstance(response.response_message, str):
                                    try:
                                        response_dict = json.loads(response.response_message)
                                    except json.JSONDecodeError as e:
                                        warning_msg = (
                                            f"Failed to parse response message as JSON: {response.response_message}. "
                                            f"The model likely returned an invalid JSON format. Will skip this response."
                                        )
                                        logger.warning(warning_msg)
                                        failed_responses_count += 1
                                        continue
                                else:
                                    response_dict = response.response_message

                                # Then construct the Pydantic model from the parsed dict
                                response.response_message = prompt_formatter.response_format(
                                    **response_dict
                                )
                            except ValidationError as e:
                                schema_str = json.dumps(
                                    prompt_formatter.response_format.model_json_schema(),
                                    indent=2,
                                )
                                warning_msg = (
                                    f"Pydantic failed to parse response message {response.response_message} with `response_format` {schema_str}. "
                                    f"The model likely returned a JSON that does not match the schema of the `response_format`. Will skip this response."
                                )
                                logger.warning(warning_msg)
                                failed_responses_count += 1
                                continue

                        # parse_func can return a single row or a list of rows
                        if prompt_formatter.parse_func:
                            try:
                                dataset_rows = prompt_formatter.parse_func(
                                    response.generic_request.original_row,
                                    response.response_message,
                                )
                            except Exception as e:
                                logger.error(f"Exception raised in your `parse_func`. {error_help}")
                                os.remove(dataset_file)
                                raise e
                            if not isinstance(dataset_rows, list):
                                dataset_rows = [dataset_rows]
                        else:
                            dataset_rows = [{"response": response.response_message}]

                        for row in dataset_rows:
                            if isinstance(row, BaseModel):
                                row = row.model_dump()

                            if not isinstance(row, dict):
                                os.remove(dataset_file)
                                raise ValueError(
                                    f"Got invalid row {row} of type {type(row)} from `parse_func`. "
                                    f"This should be type <class 'dict'>. {error_help}"
                                )
                            if not row:
                                os.remove(dataset_file)
                                raise ValueError(
                                    f"Got empty row {row} from `parse_func`. {error_help}"
                                )

                            writer.write(row)

            logger.info(f"Read {total_responses_count} responses, {failed_responses_count} failed")
            if failed_responses_count == total_responses_count:
                os.remove(dataset_file)
                raise ValueError("All requests failed")

            logger.info("Finalizing writer")

            writer.finalize()

        return Dataset.from_file(dataset_file)


def parse_response_message(
    response_message: str, response_format: Optional[BaseModel]
) -> tuple[Optional[dict | str], Optional[list[str]]]:
    response_errors = None
    if response_format:
        try:
            response_message = json.loads(response_message)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse response as JSON: {response_message}, skipping this response."
            )
            response_message = None
            response_errors = [f"Failed to parse response as JSON: {response_message}"]
    return response_message, response_errors
