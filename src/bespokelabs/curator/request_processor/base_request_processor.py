"""Base module for request processing functionality."""

import asyncio
import functools
import glob
import json
import os
import resource
import typing as t
from abc import ABC, abstractmethod
from math import ceil
from typing import TYPE_CHECKING, List, Optional

import aiofiles
import pyarrow
from pydantic import BaseModel, ValidationError

from bespokelabs.curator.constants import _CACHE_MSG
from bespokelabs.curator.cost import cost_processor_factory
from bespokelabs.curator.file_utilities import count_lines
from bespokelabs.curator.hf_card_template import HUGGINGFACE_CARD_TEMPLATE
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig, RequestProcessorConfig
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.types.generic_response import GenericResponse

if TYPE_CHECKING:
    from datasets import Dataset


class BaseRequestProcessor(ABC):
    """Base class for all request processors.

    Provides core functionality for processing requests through LLM APIs, including:
    - File descriptor limit management
    - Request file creation and caching
    - Response processing and dataset generation
    - Error handling and validation
    """

    def __init__(self, config: RequestProcessorConfig):
        """Initialize the request processor.

        Args:
            config: Configuration object containing processing parameters
        """
        # Increase the number of open file descriptors to avoid "Too many open files" errors
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_limit = min(10_000_000, hard)
        logger.debug(f"Adjusting file descriptor limit from {soft} to {desired_limit} (hard limit: {hard})")
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_limit, hard))
        self._viewer_client = None
        self.config = config
        self._cost_processor = cost_processor_factory(config=self.config, backend=self.backend)

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend property."""
        return "base"

    @abstractmethod
    def validate_config(self):
        """Validate request processor configuration.

        Ensures that configuration parameters are set correctly.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        pass

    @abstractmethod
    def requests_to_responses(self, generic_request_files: list[str]) -> None:
        """Process request files and generate responses.

        Args:
            generic_request_files: List of paths to request files to process
        """
        pass

    @property
    def viewer_client(self):
        """Return the viewer client for the request processor."""
        return self._viewer_client

    @viewer_client.setter
    def viewer_client(self, client):
        """Set the viewer client for the request processor."""
        self._viewer_client = client

    def check_structured_output_support(self) -> bool:
        """Check if the model supports structured output.

        Returns:
            bool: True if structured output is supported, False otherwise
        """
        return True

    def run(
        self,
        dataset: "Dataset",
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> "Dataset":
        """Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset: Dataset that is being mapped over
            working_dir: Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)
            parse_func_hash: Hash of the parse function for caching
            prompt_formatter: Formatter for generating prompts from dataset rows

        Returns:
            Dataset: Completed dataset with LLM responses

        Raises:
            ValueError: If model doesn't support structured output but it's requested
        """
        self.prompt_formatter = prompt_formatter
        self.working_dir = working_dir
        self.total_requests = len(dataset) if dataset is not None else 1

        # load from already completed dataset
        output_dataset = self.attempt_loading_cached_dataset(parse_func_hash)
        if output_dataset is not None:
            return output_dataset
        logger.info(f"Running {self.__class__.__name__} completions with model: {self.config.model}")

        self.validate_config()
        self.prompt_formatter = prompt_formatter
        if self.prompt_formatter.response_format:
            if not self.check_structured_output_support():
                raise ValueError(f"Model {self.config.model} does not support structured output, response_format: {self.prompt_formatter.response_format}")
        generic_request_files = self.create_request_files(dataset)

        self.requests_to_responses(generic_request_files)
        return self.create_dataset_files(parse_func_hash)

    def _verify_existing_request_files(self, dataset: Optional["Dataset"]) -> List[int]:
        """Verify integrity of the cache and identify files needing regeneration.

        Checks that each request file has associated metadata and correct number of rows.

        Args:
            dataset: The dataset to create requests from

        Returns:
            List of indices for request files that need to be regenerated
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
                    logger.warning(f"Request file {req_f} has {num_jobs} jobs, but metadata file {meta_f} has {expected_num_jobs} jobs")
                    incomplete_files.append(i)
            return incomplete_files

        except Exception as e:
            logger.warning(f"Cache verification failed due to {e} - regenerating all request files.")
            incomplete_files = list(range(expected_num_files))
            return incomplete_files

    @property
    def _multimodal_prompt_supported(self) -> bool:
        return False

    def create_request_files(self, dataset: Optional["Dataset"]) -> list[str]:
        """Creates request files if they don't exist or uses existing ones.

        Args:
            dataset: The dataset to be processed

        Returns:
            List of paths to the request files
        """
        os.makedirs(self.working_dir, exist_ok=True)
        request_files = glob.glob(os.path.join(self.working_dir, "requests_*.jsonl"))

        # By default use existing requests in working_dir
        incomplete_files = self._verify_existing_request_files(dataset)

        if len(incomplete_files) == 0:
            logger.info(f"Using cached requests. {_CACHE_MSG}")
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
                generic_request = self.prompt_formatter.create_generic_request(dict(), 0)  # noqa: C408
                if generic_request.is_multimodal_prompt is True:
                    assert self._multimodal_prompt_supported, "Requested processor does not support multimodal prompts."

                generic_request.generation_params = self.config.generation_params
                f.write(json.dumps(generic_request.model_dump(), default=str) + "\n")

            metadata_dict = {"num_jobs": 1}
            with open(metadata_file, "w") as f:
                f.write(json.dumps(metadata_dict, indent=4) + "\n")
            return request_files

        if isinstance(self.config, BatchRequestProcessorConfig):
            num_batches = ceil(len(dataset) / self.config.batch_size)
            request_files = [os.path.join(self.working_dir, f"requests_{i}.jsonl") for i in range(num_batches)]
            metadata_files = [os.path.join(self.working_dir, f"metadata_{i}.json") for i in range(num_batches)]

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
        dataset: "Dataset",
        request_file: str,
        metadata_file: str,
        start_idx: int = 0,
    ) -> None:
        """Asynchronously create a request file and its metadata.

        Args:
            dataset: Dataset to create requests from
            request_file: Path to save request file
            metadata_file: Path to save metadata file
            start_idx: Starting index in dataset for this batch
        """
        if isinstance(self.config, BatchRequestProcessorConfig):
            end_idx = min(start_idx + self.config.batch_size, len(dataset))
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            end_idx = len(dataset)

        # Check if we need to vary generation_params per row
        generation_params_per_row = "generation_params" in dataset.column_names
        if self.prompt_formatter.generation_params and generation_params_per_row:
            logger.warning("Found both default and row-level generation_params. Collided keys will follow values in row-level config.")
        async with aiofiles.open(request_file, "w") as f:
            for idx, dataset_row in enumerate(dataset):
                dataset_row_idx = idx + start_idx
                # Get the generic request from the map function
                request = self.prompt_formatter.create_generic_request(dataset_row, dataset_row_idx, generation_params_per_row)
                await f.write(json.dumps(request.model_dump(), default=str) + "\n")

        num_requests = end_idx - start_idx
        metadata_dict = {"num_jobs": num_requests}
        async with aiofiles.open(metadata_file, "w") as f:
            await f.write(json.dumps(metadata_dict, indent=4) + "\n")

        logger.info(f"Wrote {num_requests} requests to {request_file}.")

    def attempt_loading_cached_dataset(self, parse_func_hash: str) -> Optional["Dataset"]:
        """Attempt to load a cached dataset file.

        Args:
            parse_func_hash: Hash identifying the specific dataset

        Returns:
            Cached dataset if available and valid, None otherwise
        """
        dataset_file = os.path.join(self.working_dir, f"{parse_func_hash}.arrow")
        if os.path.exists(dataset_file):
            logger.debug(f"Loading dataset from {dataset_file}")
            try:
                logger.info(f"Using cached output dataset. {_CACHE_MSG}")
                return self._load_from_dataset_file(dataset_file)
            except pyarrow.lib.ArrowInvalid:
                os.remove(dataset_file)
                logger.warning(
                    f"Failed to load dataset from {dataset_file}, "
                    "which was likely corrupted by a failed previous run. "
                    "Deleted file and attempting to regenerate dataset from cached LLM responses."
                )

    def _process_response(self, data: GenericResponse) -> List | None:
        try:
            data.response_message = self.prompt_formatter.response_to_response_format(data.response_message)
        except (json.JSONDecodeError, ValidationError):
            logger.warning("Skipping response due to error parsing response message into response format")
            return

        # parse_func can return a single row or a list of rows
        responses = None
        if self.prompt_formatter.parse_func:
            try:
                responses = self.prompt_formatter.parse_func(
                    data.generic_request.original_row,
                    data.response_message,
                )
            except Exception as e:
                logger.warning(f"Skipping response due to error in `parse_func` :: {e}")
                return

            if not isinstance(responses, list):
                responses = [responses]
        else:
            # Convert response to dict before adding to dataset
            response_value = data.response_message
            if hasattr(response_value, "model_dump"):
                response_value = response_value.model_dump()
            elif hasattr(response_value, "__dict__"):
                response_value = response_value.__dict__
            responses = [{"response": response_value}]
        return responses

    def create_dataset_files(
        self,
        parse_func_hash: str,
    ) -> "Dataset":
        """Creates dataset from response files.

        Args:
            parse_func_hash: Hash identifying the dataset version

        Returns:
            Dataset containing processed responses

        Raises:
            ValueError: If no responses found or processing fails
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
        from datasets.arrow_writer import ArrowWriter

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

                        # TODO: Find a way to not process responses that have already been processed
                        # We cannot just check if parsed_response_message is not None because it could be from cached previous run
                        # response.
                        response.parsed_response_message = self._process_response(response)
                        if response.parsed_response_message is None:
                            failed_responses_count += 1
                            continue

                        for row in response.parsed_response_message:
                            if isinstance(row, BaseModel):
                                row = row.model_dump()

                            if not isinstance(row, dict):
                                os.remove(dataset_file)
                                raise ValueError(
                                    f"Got invalid row {row} of type {type(row)} from `parse_func`. This should be type <class 'dict'>. {error_help}"
                                )
                            if not row:
                                os.remove(dataset_file)
                                raise ValueError(f"Got empty row {row} from `parse_func`. {error_help}")
                            # Add the original row index to the row so that we can sort by it later.
                            row["__original_row_idx"] = response.generic_request.original_row_idx
                            writer.write(row)

            logger.info(f"Read {total_responses_count} responses.")
            error_sample_str = "\n".join(error_sample)
            error_sample_msg = f"Sample of the first {len(error_sample)} errors encountered: \n {error_sample_str}"
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
                    raise ValueError(f"Some requests failed and require_all_responses is True. {error_sample_msg}")

            # number of responses matches number of requests
            request_files = glob.glob(os.path.join(self.working_dir, "requests_*.jsonl"))
            n_requests = 0
            for request_file in request_files:
                n_requests += count_lines(request_file)

            if n_requests != total_responses_count:
                logger.warning(
                    f"{n_requests - total_responses_count} requests do not have responses. "
                    f"n_requests is {n_requests} and n_responses is {total_responses_count}"
                )
                if self.config.require_all_responses:
                    os.remove(dataset_file)
                    raise ValueError("Some requests do not have responses and require_all_responses is True.")

        return self._load_from_dataset_file(dataset_file)

    def _load_from_dataset_file(self, dataset_file: str) -> "Dataset":
        from datasets import Dataset

        d = Dataset.from_file(dataset_file)
        d = d.sort("__original_row_idx")
        d = d.remove_columns("__original_row_idx")

        push_to_hub = functools.partial(BaseRequestProcessor.push_to_hub, dataset=d, _push_to_hub=d.push_to_hub)
        d.push_to_hub = push_to_hub

        return d

    @staticmethod
    def push_to_hub(repo_id: str, dataset=None, _push_to_hub=None, **kwargs):
        """Push the dataset to the hub and create a dataset card."""
        from huggingface_hub import DatasetCard

        _push_to_hub(repo_id, **kwargs)
        card = DatasetCard(
            HUGGINGFACE_CARD_TEMPLATE.format(
                dataset_name=repo_id.split("/")[-1],
                repo_id=repo_id,
                sample=json.dumps(dataset[0], indent=4),
            )
        )
        card.push_to_hub(repo_id)

    def _get_validated_response(self, line: str) -> tuple[GenericResponse | None, bool]:
        """Check if a response is valid or has errors.

        Args:
            line: The line to process into a GenericResponse.

        Returns:
            response: The response if it is valid, None otherwise.
            is_valid: True if the response is valid, False otherwise.
        """
        response = None
        try:
            response = GenericResponse.model_validate_json(line)
            row_id = response.generic_request.original_row_idx
            if response.response_errors:
                logger.debug(f"Request {row_id} previously failed due to errors: {response.response_errors}, removing from output and will retry")
                return None, False
            if response.response_message is None:
                logger.debug(f"Request {row_id} previously failed due to no response. Removing from output and will retry.")
                return None, False
        except (json.JSONDecodeError, ValidationError):
            logger.warning(f"Skipping response due to error parsing line: {line}")
            return None, False

        return response, True

    def validate_existing_response_file(self, response_file: str) -> t.Union[set[int], int]:
        """Parse an existing response file to identify completed requests and removes failed requests.

        Args:
            response_file: Path to the response file to parse

        Returns:
            set[int]: Set of completed request IDs that were already successfully processed
            int: Number of completed parsed responses
        """
        if not os.path.exists(response_file):
            return set(), 0

        completed_request_ids = set()
        failed_request_ids = set()
        completed_parsed_responses = 0
        parsing_error_responses = 0
        logger.info(f"Resuming progress by reading existing file: {response_file}")
        logger.debug(f"Removing all failed requests from {response_file} so they can be retried")
        temp_filepath = response_file + ".temp"

        with open(response_file, "r") as input_file, open(temp_filepath, "w") as output_file:
            for line in input_file:
                response, is_valid = self._get_validated_response(line)
                if not response:
                    parsing_error_responses += 1
                    continue
                row_id = response.generic_request.original_row_idx
                if response.parsed_response_message:
                    completed_parsed_responses += len(response.parsed_response_message)
                if is_valid:
                    completed_request_ids.add(row_id)
                    output_file.write(line)
                else:
                    failed_request_ids.add(row_id)

        logger.info(
            f"Found {len(completed_request_ids)} successful requests and {len(failed_request_ids)} "
            f"previously failed requests and {parsing_error_responses} parsing errors in {response_file}"
        )
        os.replace(temp_filepath, response_file)

        return completed_request_ids, completed_parsed_responses

    def read_metadata_file(self, request_file: str) -> int:
        """Read the number of jobs from the metadata file.

        Args:
            request_file: Path to the request file to get metadata for

        Returns:
            int: Number of total batch requests

        Raises:
            ValueError: If metadata file is missing or invalid
        """
        metadata_file = request_file.replace("requests_", "metadata_").replace(".jsonl", ".json")

        if not os.path.exists(metadata_file):
            raise ValueError(f"Metadata file not found: {metadata_file}")

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                return metadata
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metadata file: {metadata_file}. Delete cache directory 'rm -rf {self.working_dir}' and try again.") from e
