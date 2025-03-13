"""Base class for the code execution backend."""

import asyncio
import glob
import io
import json
import os
import resource
import tarfile
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional

import aiofiles
import pyarrow
from pydantic import BaseModel, ValidationError

from bespokelabs.curator.code_executor.code_formatter import CodeFormatter
from bespokelabs.curator.code_executor.tracker import CodeExecutionStatusTracker
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequest, CodeExecutionResponse
from bespokelabs.curator.constants import _CACHE_MSG
from bespokelabs.curator.file_utilities import count_lines
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop

if TYPE_CHECKING:
    from datasets import Dataset


class BaseCodeExecutionBackend:
    """Base class for all request processors.

    Provides core functionality for processing requests through LLM APIs, including:
    - File descriptor limit management
    - Request file creation and caching
    - Response processing and dataset generation
    - Error handling and validation
    """

    def __init__(self, config: dict):
        """Initialize the request processor.

        Args:
            config: Configuration object containing processing parameters
        """
        # Increase the number of open file descriptors to avoid "Too many open files" errors
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_limit = min(10_000_000, hard)
        logger.debug(f"Adjusting file descriptor limit from {soft} to {desired_limit} (hard limit: {hard})")
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_limit, hard))
        self.config = config

        self.manual_max_requests_per_minute = config.max_requests_per_minute
        self.default_max_requests_per_minute = 10
        # The rich.Console used for the status tracker, only set for testing
        self._tracker_console = None

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend property."""
        return "base"

    @abstractmethod
    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionResponse:
        """Execute a single request."""
        pass

    @abstractmethod
    def requests_to_responses(self, execution_request_files: list[str]) -> None:
        """Process request files and generate responses.

        Args:
            execution_request_files: List of paths to request files to process
        """
        for request_file in execution_request_files:
            response_file = request_file.replace("requests_", "responses_")
            run_in_event_loop(
                self.process_requests_from_file(
                    execution_request_filepath=request_file,
                    response_file=response_file,
                )
            )

    async def process_requests_from_file(
        self,
        execution_request_filepath: str,
        response_file: str,
    ) -> None:
        """Processes API requests in parallel, throttling to stay under rate limits.

        Args:
            execution_request_filepath: Path to file containing requests
            response_file: Path where the response data will be saved
        """
        # Initialize trackers
        queue_of_requests_to_retry: asyncio.Queue[CodeExecutionRequest] = asyncio.Queue()
        status_tracker = CodeExecutionStatusTracker()

        # Get rate limits
        status_tracker.max_requests_per_minute = self.manual_max_requests_per_minute

        # Resume if a response file exists
        completed_request_ids = self.validate_existing_response_file(response_file)

        # Count total requests
        status_tracker.num_tasks_already_completed = len(completed_request_ids)
        status_tracker.total_requests = self.total_requests
        status_tracker.start_tracker(self._tracker_console)

        # Use higher connector limit for better throughput
        async with aiofiles.open(execution_request_filepath) as file:
            pending_requests = []

            async for line in file:
                execution_request = CodeExecutionRequest.model_validate_json(line)

                if execution_request.original_row_idx in completed_request_ids:
                    continue

                request = CodeAPIRequest(
                    task_id=execution_request.original_row_idx,
                    execution_request=execution_request,
                    attempts_left=self.config.max_retries,
                    code_formatter=self.code_formatter,
                )

                # print(f"Processing request {request.task_id}")

                # while not status_tracker.has_capacity():
                # await asyncio.sleep(0.1)

                status_tracker.consume_capacity()

                # Wait for rate limits cool down if needed
                # todo: implement this
                # await self.cool_down_if_rate_limit_error(status_tracker)

                task = asyncio.create_task(
                    self.handle_single_request_with_retries(
                        request=request,
                        retry_queue=queue_of_requests_to_retry,
                        response_file=response_file,
                        status_tracker=status_tracker,
                    )
                )

                pending_requests.append(task)

                # status_tracker.num_tasks_started += 1
                # status_tracker.num_tasks_in_progress += 1

            # Wait for all tasks to complete
            if pending_requests:
                await asyncio.gather(*pending_requests)

            # Process any remaining retries in the queue
            pending_retries = set()
            while not queue_of_requests_to_retry.empty() or pending_retries:
                # Process new items from the queue if we have capacity
                if not queue_of_requests_to_retry.empty():
                    retry_request = await queue_of_requests_to_retry.get()
                    attempt_number = self.config.max_retries - retry_request.attempts_left
                    logger.debug(
                        f"Retrying request {retry_request.task_id} "
                        f"(attempt #{attempt_number} of {self.config.max_retries})"
                        f"Previous errors: {retry_request.result}"
                    )

                    # Wait for capacity if needed
                    # while not status_tracker.has_capacity():
                    # await asyncio.sleep(0.1)

                    # Consume capacity before making request
                    # status_tracker.consume_capacity()

                    task = asyncio.create_task(
                        self.handle_single_request_with_retries(
                            request=retry_request,
                            retry_queue=queue_of_requests_to_retry,
                            response_file=response_file,
                            status_tracker=status_tracker,
                        )
                    )
                    pending_retries.add(task)

                # Wait for some tasks to complete
                if pending_retries:
                    done, pending_retries = await asyncio.wait(pending_retries, timeout=0.1)

        status_tracker.stop_tracker()

        # Log final status
        logger.info(f"Processing complete. Results saved to {response_file}")
        logger.info(f"Status tracker: {status_tracker}")

        if status_tracker.num_tasks_failed > 0:
            logger.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {response_file}.")

    async def handle_single_request_with_retries(
        self,
        request: CodeAPIRequest,
        retry_queue: asyncio.Queue,
        response_file: str,
        status_tracker: CodeExecutionStatusTracker,
    ) -> None:
        """Common wrapper for handling a single request with error handling and retries.

        This method implements the common try/except logic and retry mechanism,
        while delegating the actual API call to call_single_request.

        Args:
            request: The request to process
            session: Async HTTP session
            retry_queue: Queue for failed requests
            response_file: Path where the response data will be saved
            status_tracker: Tracks request status
        """
        try:
            execution_output = await self.execute_request(request)

            execution_response = CodeExecutionResponse(
                code_api_request=request,
                exec_output=execution_output,
            ).model_dump()

            await self.append_execution_response(execution_response, response_file)

            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            status_tracker.update_stats()

        except Exception as e:
            status_tracker.num_other_errors += 1

            request.result.append(e)

            if request.attempts_left > 0:
                request.attempts_left -= 1
                logger.warning(
                    f"Encountered '{e.__class__.__name__}: {e}' during attempt "
                    f"{self.config.max_retries - request.attempts_left} of {self.config.max_retries} "
                    f"while processing request {request.task_id}"
                )
                retry_queue.put_nowait(request)
            else:
                logger.error(
                    f"Request {request.task_id} failed permanently after exhausting all {self.config.max_retries} retry attempts. "
                    f"Errors: {[str(e) for e in request.result]}"
                )

                execution_response = CodeExecutionResponse(
                    code_api_request=request,
                    exec_output=CodeExecutionOutput(
                        message="Error",
                        error="\n".join([str(e) for e in request.result]),
                    ),
                ).model_dump()

                await self.append_execution_response(execution_response, response_file)

    def run(
        self,
        dataset: "Dataset",
        working_dir: str,
        code_formatter: CodeFormatter,
        all_func_hash_hash: str,
    ) -> "Dataset":
        """Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset: Dataset that is being mapped over
            working_dir: Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)
            code_formatter: Code formatter to use
            all_func_hash_hash: Hash identifying the dataset version

        Returns:
            Dataset: Completed dataset with responses

        Raises:
            ValueError: If model doesn't support structured output but it's requested
        """
        self.code = code_formatter.code
        self.code_input = code_formatter.code_input
        self.code_output = code_formatter.code_output
        self.working_dir = working_dir
        self.total_requests = len(dataset) if dataset is not None else 1
        # load from already completed dataset

        output_dataset = self.attempt_loading_cached_dataset(all_func_hash_hash)
        if output_dataset is not None:
            return output_dataset

        self.code_formatter = code_formatter

        execution_request_files = self.create_request_files(dataset)

        self.requests_to_responses(execution_request_files)

        return self.create_dataset_files(all_func_hash_hash)

    def _verify_existing_request_files(self, dataset: Optional["Dataset"]) -> List[int]:
        """Verify integrity of the cache and identify files needing regeneration.

        Checks that each request file has associated metadata and correct number of rows.

        Args:
            dataset: The dataset to create requests from

        Returns:
            List of indices for request files that need to be regenerated
        """
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

        if dataset is None:
            raise ValueError("Dataset is empty")

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
        end_idx = len(dataset)

        async with aiofiles.open(request_file, "w") as f:
            for idx, dataset_row in enumerate(dataset):
                dataset_row_idx = idx + start_idx
                # Get the generic request from the map function
                execution_directory = os.path.join(self.working_dir, "code_execution", str(dataset_row_idx))
                request = self.code_formatter.create_code_execution_request(dataset_row, dataset_row_idx, execution_directory)
                await f.write(json.dumps(request.model_dump(), default=str) + "\n")

        num_requests = end_idx - start_idx
        metadata_dict = {"num_jobs": num_requests}
        async with aiofiles.open(metadata_file, "w") as f:
            await f.write(json.dumps(metadata_dict, indent=4) + "\n")

        logger.info(f"Wrote {num_requests} requests to {request_file}.")

    def attempt_loading_cached_dataset(self, all_func_hash_hash: str) -> Optional["Dataset"]:
        """Attempt to load a cached dataset file.

        Args:
            all_func_hash_hash: Hash identifying the specific dataset

        Returns:
            Cached dataset if available and valid, None otherwise
        """
        dataset_file = os.path.join(self.working_dir, f"{all_func_hash_hash}.arrow")
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

    def create_dataset_files(
        self,
        all_func_hash_hash: str,
    ) -> "Dataset":
        """Creates dataset from response files.

        Args:
            all_func_hash_hash: Hash identifying the dataset version

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
        dataset_file = os.path.join(self.working_dir, f"{all_func_hash_hash}.arrow")
        from datasets.arrow_writer import ArrowWriter

        with ArrowWriter(path=dataset_file) as writer:
            for responses_file in responses_files:
                with open(responses_file, "r") as f_in:
                    for execution_response_string in f_in:
                        total_responses_count += 1

                        response = CodeExecutionResponse.model_validate_json(execution_response_string)

                        if self.code_formatter.code_output:
                            try:
                                dataset_rows = self.code_formatter.code_output(
                                    response.code_api_request.execution_request.original_row,
                                    response.exec_output,
                                )
                            except Exception as e:
                                logger.error(f"Exception raised in your `parse_func`. {error_help}")
                                os.remove(dataset_file)
                                raise e
                            if not isinstance(dataset_rows, list):
                                dataset_rows = [dataset_rows]
                        else:
                            raise ValueError("code_output is not implemented")

                        for row in dataset_rows:
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
                            row["__original_row_idx"] = response.code_api_request.execution_request.original_row_idx
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

        print(f"Loading dataset from {dataset_file}")
        d = Dataset.from_file(dataset_file)
        d = d.sort("__original_row_idx")
        d = d.remove_columns("__original_row_idx")
        return d

    def validate_existing_response_file(self, response_file: str) -> set[int]:
        """Parse an existing response file to identify completed requests and removes failed requests.

        Args:
            response_file: Path to the response file to parse

        Returns:
            set[int]: Set of completed request IDs that were already successfully processed
        """
        completed_request_ids = set()
        failed_request_ids = set()

        if os.path.exists(response_file):
            logger.info(f"Resuming progress by reading existing file: {response_file}")
            logger.debug(f"Removing all failed requests from {response_file} so they can be retried")
            temp_filepath = response_file + ".temp"

            parsing_error_responses = 0
            with open(response_file, "r") as input_file, open(temp_filepath, "w") as output_file:
                for line in input_file:
                    try:
                        response = CodeExecutionResponse.model_validate_json(line)
                    except (json.JSONDecodeError, ValidationError):
                        logger.warning("Skipping response due to error parsing line")
                        parsing_error_responses += 1
                        continue
                    row_id = response.code_api_request.execution_request.original_row_idx
                    if response.exec_output.error:
                        logger.debug(f"Request {row_id} previously failed due to errors: {response.exec_output.error}, removing from output and will retry")
                        failed_request_ids.add(row_id)
                    elif response.exec_output.message is None:
                        logger.debug(f"Request {row_id} previously failed due to no response. Removing from output and will retry.")
                        failed_request_ids.add(row_id)
                    else:
                        completed_request_ids.add(row_id)
                        output_file.write(line)

            logger.info(
                f"Found {len(completed_request_ids)} successful requests and {len(failed_request_ids)} "
                f"previously failed requests and {parsing_error_responses} parsing errors in {response_file}"
            )
            os.replace(temp_filepath, response_file)

        return completed_request_ids

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

    async def cool_down_if_rate_limit_error(self, status_tracker: CodeExecutionStatusTracker) -> None:
        """Pause processing if a rate limit error is detected.

        Args:
            status_tracker: Tracker containing rate limit status
        """
        seconds_to_pause_on_rate_limit = self.config.seconds_to_pause_on_rate_limit
        seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
        remaining_seconds_to_pause = seconds_to_pause_on_rate_limit - seconds_since_rate_limit_error
        if remaining_seconds_to_pause > 0:
            logger.warn(f"Pausing for {int(remaining_seconds_to_pause)} seconds")
            await asyncio.sleep(remaining_seconds_to_pause)

    async def append_execution_response(self, data: dict, filename: str) -> None:
        """Append a response to a jsonl file with async file operations.

        Args:
            data: Response data to append
            filename: File to append to
        """
        json_string = json.dumps(data, default=str)
        async with aiofiles.open(filename, "a") as f:
            await f.write(json_string + "\n")
        logger.debug(f"Successfully appended response to {filename}")

    @classmethod
    def _create_temp_file(cls, content: str, execution_directory: str) -> str:
        """Create a temporary file with the given content.

        Args:
            content: Content to write to temp file
            execution_directory: Directory to create the temp file in

        Returns:
            Path to the created temp file
        """
        # create execution directory if it doesn't exist
        os.makedirs(execution_directory, exist_ok=True)

        temp_file_path = os.path.join(execution_directory, "program.py")
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
        return temp_file_path

    @classmethod
    def _get_created_files(cls, program_dir: str) -> bytes:
        """Get any files created during code execution.

        Args:
            program_dir: Directory containing the executed program and created files

        Returns:
            Bytes containing a zip archive of any created files
        """
        # Create a BytesIO object to store the zip data
        tar_buffer = io.BytesIO()

        # Create a zip archive containing all files in program_dir
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            # Add all files in program_dir to the archive
            for filename in os.listdir(program_dir):
                file_path = os.path.join(program_dir, filename)
                if os.path.isfile(file_path):
                    tar.add(file_path, arcname=filename)

        # Get the bytes from the buffer
        tar_buffer.seek(0)
        return str(tar_buffer.getvalue())
