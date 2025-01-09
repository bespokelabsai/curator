import asyncio
import glob
import json
import logging
import os
import resource
from abc import ABC, abstractmethod
from math import ceil
from typing import List, Optional

import aiofiles
import pyarrow
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from pydantic import BaseModel, ValidationError

from bespokelabs.curator.file_utilities import count_lines
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.config import (
    BatchRequestProcessorConfig,
    RequestProcessorConfig,
)
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.types.generic_response import GenericResponse

logger = logger = logging.getLogger(__name__)

CACHE_MSG = "If you want to regenerate the dataset, disable or delete the cache."


class BaseRequestProcessor(ABC):
    """
    Base class for all request processors.
    """

    def __init__(self, config: RequestProcessorConfig):
        # Increase the number of open file descriptors to avoid "Too many open files" errors
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_limit = min(10_000_000, hard)
        logger.debug(
            f"Adjusting file descriptor limit from {soft} to {desired_limit} (hard limit: {hard})"
        )
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_limit, hard))
        self.config = config

    @abstractmethod
    def requests_to_responses(self, generic_request_files: list[str]) -> None:
        pass

    def check_structured_output_support(self) -> bool:
        """Check if the model supports structured output"""
        return True

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
        self.prompt_formatter = prompt_formatter
        self.working_dir = working_dir

        # load from already completed dataset
        output_dataset = self.attempt_loading_cached_dataset(parse_func_hash)
        if output_dataset is not None:
            return output_dataset

        logger.info(
            f"Running {self.__class__.__name__} completions with model: {self.config.model}"
        )

        self.prompt_formatter = prompt_formatter
        if self.prompt_formatter.response_format:
            if not self.check_structured_output_support():
                raise ValueError(
                    f"Model {self.config.model} does not support structured output, "
                    f"response_format: {self.prompt_formatter.response_format}"
                )
        generic_request_files = self.create_request_files(dataset)

        self.requests_to_responses(generic_request_files)

        return self.create_dataset_files(parse_func_hash)

    def _verify_existing_request_files(self, dataset: Optional[Dataset]) -> List[int]:
        """
        Verify integrity of the cache (each request file has associated metadata, and the number of rows is correct),
        and return the indices of request files that need to be regenerated (so that no work is repeated).

        Args:
            working_dir (str): Working directory where cache files are expected to be (requests.jsonl, metadata.json)
            dataset (Optional[Dataset]): The dataset that we want to create requests from

        Returns:
            List[int]: Indices of missing files
        """
        if isinstance(self.config, BatchRequestProcessorConfig) and dataset is not None:
            expected_num_files = ceil(len(dataset) / self.config.batch_size)
        else:
            expected_num_files = 1

        try:
            incomplete_files = []
            for i in range(expected_num_files):
                req_f = os.path.join(self.working_dir, f"requests_{i}.jsonl")
                meta_f = os.path.join(self.working_dir, f"metadata_{i}.json")

                if not os.path.exists(req_f):
                    incomplete_files.append(i)
                    continue

                if not os.path.exists(meta_f):
                    logger.warning(f"Cache missing metadata file {meta_f} for request file {req_f}")
                    incomplete_files.append(i)
                    continue

                with open(req_f, "r") as f:
                    data = f.read()
                num_jobs = len(data.splitlines())

                with open(meta_f, "r") as f:
                    metadata = json.load(f)

                expected_num_jobs = metadata["num_jobs"]
                if num_jobs != expected_num_jobs:
                    logger.warning(
                        f"Request file {req_f} has {num_jobs} jobs, but metadata file {meta_f} has {expected_num_jobs} jobs"
                    )
                    incomplete_files.append(i)

            return incomplete_files

        except Exception as e:
            logger.warning(
                f"Cache verification failed due to {e} - regenerating all request files."
            )
            incomplete_files = list(range(expected_num_files))
            return incomplete_files

    def create_request_files(self, dataset: Optional[Dataset]) -> list[str]:
        """
        Creates a request file if they don't already exist or use existing.

        Args:
            dataset (Dataset): The dataset to be processed.
            working_dir (str): The directory where request files will be saved.

        Returns:
            list[str]: Paths to the request files that were created.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        request_files = glob.glob(os.path.join(self.working_dir, "requests_*.jsonl"))

        # By default use existing requests in working_dir
        incomplete_files = self._verify_existing_request_files(dataset)

        if len(incomplete_files) == 0:
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
        logger.info(f"Preparing request file(s) in {self.working_dir}")
        request_file = os.path.join(self.working_dir, "requests_0.jsonl")
        request_files = [request_file]

        metadata_file = os.path.join(self.working_dir, "metadata_0.json")
        metadata_files = [metadata_file]

        if dataset is None:
            with open(request_file, "w") as f:
                generic_request = self.prompt_formatter.create_generic_request(dict(), 0)
                generic_request.generation_params = self.config.generation_params
                f.write(json.dumps(generic_request.model_dump(), default=str) + "\n")

            metadata_dict = {"num_jobs": 1}
            with open(metadata_file, "w") as f:
                f.write(json.dumps(metadata_dict, indent=4) + "\n")
            return request_files

        if isinstance(self.config, BatchRequestProcessorConfig):
            num_batches = ceil(len(dataset) / self.config.batch_size)
            request_files = [
                os.path.join(self.working_dir, f"requests_{i}.jsonl") for i in range(num_batches)
            ]
            metadata_files = [
                os.path.join(self.working_dir, f"metadata_{i}.json") for i in range(num_batches)
            ]

            async def create_all_request_files():
                tasks = [
                    self.acreate_request_file(
                        dataset,
                        request_files[i],
                        metadata_files[i],
                        start_idx=i * self.config.batch_size,
                    )
                    for i in range(num_batches)
                    if i in incomplete_files
                ]
                await asyncio.gather(*tasks)

            run_in_event_loop(create_all_request_files())
        else:
            run_in_event_loop(self.acreate_request_file(dataset, request_file, metadata_file))

        return request_files

    async def acreate_request_file(
        self,
        dataset: Dataset,
        request_file: str,
        metadata_file: str,
        start_idx: int = 0,
    ) -> None:
        if isinstance(self.config, BatchRequestProcessorConfig):
            end_idx = min(start_idx + self.config.batch_size, len(dataset))
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            end_idx = len(dataset)

        async with aiofiles.open(request_file, "w") as f:
            for idx, dataset_row in enumerate(dataset):
                dataset_row_idx = idx + start_idx
                # Get the generic request from the map function
                request = self.prompt_formatter.create_generic_request(dataset_row, dataset_row_idx)
                request.generation_params = self.config.generation_params
                await f.write(json.dumps(request.model_dump(), default=str) + "\n")

        num_requests = end_idx - start_idx
        metadata_dict = {"num_jobs": num_requests}
        async with aiofiles.open(metadata_file, "w") as f:
            await f.write(json.dumps(metadata_dict, indent=4) + "\n")

        logger.info(f"Wrote {num_requests} requests to {request_file}.")

    def attempt_loading_cached_dataset(self, parse_func_hash: str) -> Optional[Dataset]:
        dataset_file = os.path.join(self.working_dir, f"{parse_func_hash}.arrow")
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
        parse_func_hash: str,
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
        responses_files = glob.glob(os.path.join(self.working_dir, "responses_*.jsonl"))
        if len(responses_files) == 0:
            raise ValueError(f"No responses files found in {self.working_dir}")

        error_help = (
            "Please check your `parse_func` is returning a valid row (dict) "
            "or list of rows (list of dicts) and re-run. "
            "Dataset will be regenerated from cached LLM responses."
        )

        # Process all response files
        total_responses_count = 0
        failed_responses_count = 0
        error_sample = []
        dataset_file = os.path.join(self.working_dir, f"{parse_func_hash}.arrow")
        with ArrowWriter(path=dataset_file) as writer:
            for responses_file in responses_files:
                with open(responses_file, "r") as f_in:
                    for generic_response_string in f_in:
                        total_responses_count += 1
                        response = GenericResponse.model_validate_json(generic_response_string)

                        if response.response_errors is not None:
                            failed_responses_count += 1
                            if len(error_sample) < 10:
                                error_sample.append(str(response.response_errors))
                            continue

                        try:
                            response.response_message = (
                                self.prompt_formatter.response_to_response_format(
                                    response.response_message
                                )
                            )
                        except (json.JSONDecodeError, ValidationError) as e:
                            logger.warning(
                                "Skipping response due to error parsing response message into response format"
                            )
                            failed_responses_count += 1
                            continue

                        # parse_func can return a single row or a list of rows
                        if self.prompt_formatter.parse_func:
                            try:
                                dataset_rows = self.prompt_formatter.parse_func(
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
                            # Convert response to dict before adding to dataset
                            response_value = response.response_message
                            if hasattr(response_value, "model_dump"):
                                response_value = response_value.model_dump()
                            elif hasattr(response_value, "__dict__"):
                                response_value = response_value.__dict__
                            dataset_rows = [{"response": response_value}]

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
                            # Add the original row index to the row so that we can sort by it later.
                            row["__original_row_idx"] = response.generic_request.original_row_idx
                            writer.write(row)

            logger.info(f"Read {total_responses_count} responses.")
            error_sample_str = "\n".join(error_sample)
            error_sample_msg = (
                f"Sample of the first {len(error_sample)} errors encountered: \n {error_sample_str}"
            )
            if failed_responses_count == total_responses_count:
                writer.write({"error": "All requests failed"})
                writer.finalize()
                os.remove(dataset_file)
                raise ValueError(f"All requests failed. {error_sample_msg}")
            else:
                logger.info("Finalizing writer")
                writer.finalize()

            if failed_responses_count > 0:
                logger.warning(f"{failed_responses_count} requests failed.")
                if self.config.require_all_responses:
                    os.remove(dataset_file)
                    raise ValueError(
                        f"Some requests failed and require_all_responses is True. {error_sample_msg}"
                    )

            # number of responses matches number of requests
            request_files = glob.glob(os.path.join(self.working_dir, "requests_*.jsonl"))
            n_requests = 0
            for request_file in request_files:
                n_requests += count_lines(request_file)

            if n_requests != total_responses_count:
                logger.warning(
                    f"{n_requests - total_responses_count} requests do not have responses. n_requests is {n_requests} and n_responses is {total_responses_count}"
                )
                if self.config.require_all_responses:
                    os.remove(dataset_file)
                    raise ValueError(
                        f"Some requests do not have responses and require_all_responses is True."
                    )

        d = Dataset.from_file(dataset_file)
        d = d.sort("__original_row_idx")
        d = d.remove_columns("__original_row_idx")
        return d


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
