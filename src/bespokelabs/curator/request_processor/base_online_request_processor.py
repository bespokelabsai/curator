from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import datetime
import time
from typing import Optional
from tqdm import tqdm
import logging
import asyncio
import aiohttp
import os
import json
import resource

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.prompter.prompter import PromptFormatter
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.generic_response import GenericResponse
import aiofiles

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class StatusTracker:
    """Tracks the status of all requests."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    num_rate_limit_errors: int = 0
    available_request_capacity: float = 0
    available_token_capacity: float = 0
    last_update_time: float = field(default_factory=time.time)
    max_requests_per_minute: int = 0
    max_tokens_per_minute: int = 0
    pbar: tqdm = field(default=None)
    response_cost: float = 0
    time_of_last_rate_limit_error: float = field(default=None)

    def __str__(self):
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}\n"
            f"Errors - API: {self.num_api_errors}, "
            f"Rate Limit: {self.num_rate_limit_errors}, "
            f"Other: {self.num_other_errors}, "
            f"Total: {self.num_other_errors + self.num_api_errors + self.num_rate_limit_errors}"
        )

    def update_capacity(self):
        """Update available capacity based on time elapsed"""
        current_time = time.time()
        seconds_since_update = current_time - self.last_update_time

        self.available_request_capacity = min(
            self.available_request_capacity
            + self.max_requests_per_minute * seconds_since_update / 60.0,
            self.max_requests_per_minute,
        )

        self.available_token_capacity = min(
            self.available_token_capacity
            + self.max_tokens_per_minute * seconds_since_update / 60.0,
            self.max_tokens_per_minute,
        )

        self.last_update_time = current_time

    def has_capacity(self, token_estimate: int) -> bool:
        """Check if there's enough capacity for a request"""
        self.update_capacity()
        has_capacity = (
            self.available_request_capacity >= 1 and self.available_token_capacity >= token_estimate
        )
        if not has_capacity:
            logger.debug(
                f"No capacity for request with {token_estimate} tokens. "
                f"Available capacity: {self.available_token_capacity} tokens, "
                f"{self.available_request_capacity} requests."
            )
        return has_capacity

    def consume_capacity(self, token_estimate: int):
        """Consume capacity for a request"""
        self.available_request_capacity -= 1
        self.available_token_capacity -= token_estimate


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata."""

    task_id: int
    generic_request: GenericRequest
    api_specific_request: dict
    attempts_left: int
    result: list = field(default_factory=list)
    prompt_formatter: PromptFormatter = field(default=None)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class BaseOnlineRequestProcessor(BaseRequestProcessor, ABC):
    """Abstract base class for online request processors that make real-time API calls."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        super().__init__(batch_size=None)
        self.model: str = model
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.presence_penalty: float | None = presence_penalty
        self.frequency_penalty: float | None = frequency_penalty
        self.prompt_formatter: Optional[PromptFormatter] = None

    @abstractmethod
    def estimate_total_tokens(self, messages: list) -> int:
        """Estimate total tokens for a request"""
        pass

    @abstractmethod
    def estimate_output_tokens(self) -> int:
        """Estimate output tokens for a request"""
        pass

    def check_structured_output_support(self) -> bool:
        """Check if the model supports structured output"""
        return True

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Run completions using the online API with async processing."""
        logger.info(f"Running {self.__class__.__name__} completions with model: {self.model}")

        self.prompt_formatter = prompt_formatter
        if self.prompt_formatter.response_format:
            if not self.check_structured_output_support():
                raise ValueError(
                    f"Model {self.model} does not support structured output, "
                    f"response_format: {self.prompt_formatter.response_format}"
                )
        generic_requests_files = self.create_request_files(dataset, working_dir, prompt_formatter)
        generic_responses_files = [
            f"{working_dir}/responses_{i}.jsonl" for i in range(len(generic_requests_files))
        ]

        for request_file, response_file in zip(generic_requests_files, generic_responses_files):
            run_in_event_loop(
                self.process_requests_from_file(
                    generic_requests_filepath=request_file,
                    save_filepath=response_file,
                    max_attempts=5,
                    resume=True,
                )
            )

        return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

    async def process_requests_from_file(
        self,
        generic_requests_filepath: str,
        save_filepath: str,
        max_attempts: int,
        resume: bool,
        resume_no_retry: bool = False,
    ) -> None:
        """Processes API requests in parallel, throttling to stay under rate limits."""

        # Initialize trackers
        queue_of_requests_to_retry: asyncio.Queue[APIRequest] = asyncio.Queue()
        status_tracker = StatusTracker()

        # Get rate limits
        rate_limits = self.get_rate_limits()
        status_tracker.max_requests_per_minute = rate_limits["max_requests_per_minute"]
        status_tracker.max_tokens_per_minute = rate_limits["max_tokens_per_minute"]
        rpm = rate_limits["max_requests_per_minute"]

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (min(hard, int(10 * status_tracker.max_requests_per_minute)), hard),
        )

        # Track completed requests for resume functionality
        completed_request_ids = set()
        if os.path.exists(save_filepath):
            if resume:
                logger.debug(f"Resuming progress from existing file: {save_filepath}")
                logger.debug(
                    f"Removing all failed requests from {save_filepath} so they can be retried"
                )
                temp_filepath = f"{save_filepath}.temp"
                num_previously_failed_requests = 0

                with open(save_filepath, "r") as input_file, open(
                    temp_filepath, "w"
                ) as output_file:
                    for line in input_file:
                        response = GenericResponse.model_validate_json(line)
                        if response.response_errors:
                            logger.debug(
                                f"Request {response.generic_request.original_row_idx} previously failed due to errors: "
                                f"{response.response_errors}, removing from output and will retry"
                            )
                            num_previously_failed_requests += 1
                        else:
                            completed_request_ids.add(response.generic_request.original_row_idx)
                            output_file.write(line)

                logger.info(
                    f"Found {len(completed_request_ids)} completed requests and "
                    f"{num_previously_failed_requests} previously failed requests"
                )
                logger.info("Failed requests and remaining requests will now be processed.")
                os.replace(temp_filepath, save_filepath)

            elif resume_no_retry:
                logger.warning(
                    f"Resuming progress from existing file: {save_filepath}, without retrying failed requests"
                )
                num_previously_failed_requests = 0

                with open(save_filepath, "r") as input_file:
                    for line in input_file:
                        response = GenericResponse.model_validate_json(line)
                        if response.response_errors:
                            logger.debug(
                                f"Request {response.generic_request.original_row_idx} previously failed due to errors: "
                                f"{response.response_errors}, will NOT retry"
                            )
                            num_previously_failed_requests += 1
                        completed_request_ids.add(response.generic_request.original_row_idx)

                logger.info(
                    f"Found {len(completed_request_ids)} total requests and "
                    f"{num_previously_failed_requests} previously failed requests"
                )
                logger.info("Remaining requests will now be processed.")

            else:
                user_input = input(
                    f"File {save_filepath} already exists.\n"
                    f"To resume if there are remaining requests without responses, run with --resume flag.\n"
                    f"Overwrite? (Y/n): "
                )
                if user_input.lower() not in ["y", ""]:
                    logger.info("Aborting operation.")
                    return

        # Count total requests
        total_requests = sum(1 for _ in open(generic_requests_filepath))

        # Create progress bar
        status_tracker.pbar = tqdm(
            initial=len(completed_request_ids),
            total=total_requests,
            desc=f"Processing {self.__class__.__name__} requests",
        )

        # Use higher connector limit for better throughput
        connector = aiohttp.TCPConnector(limit=10 * rpm)
        async with aiohttp.ClientSession(
            connector=connector
        ) as session:  # Initialize ClientSession here
            async with aiofiles.open(generic_requests_filepath) as file:
                pending_requests = []

                async for line in file:
                    generic_request = GenericRequest.model_validate_json(line)

                    if resume and generic_request.original_row_idx in completed_request_ids:
                        status_tracker.num_tasks_already_completed += 1
                        continue

                    request = APIRequest(
                        task_id=status_tracker.num_tasks_started,
                        generic_request=generic_request,
                        api_specific_request=self.create_api_specific_request(generic_request),
                        attempts_left=max_attempts,
                        prompt_formatter=self.prompt_formatter,
                    )

                    token_estimate = self.estimate_total_tokens(request.generic_request.messages)

                    # Wait for capacity if needed
                    while not status_tracker.has_capacity(token_estimate):
                        await asyncio.sleep(0.1)

                    # Consume capacity before making request
                    status_tracker.consume_capacity(token_estimate)

                    task = asyncio.create_task(
                        self.handle_single_request_with_retries(
                            request=request,
                            session=session,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    pending_requests.append(task)

                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1

            # Wait for all tasks to complete
            if pending_requests:
                await asyncio.gather(*pending_requests)

            # Process any remaining retries in the queue
            pending_retries = []
            retry_batch = []

            # Process retries in batches to avoid overwhelming the queue
            while not queue_of_requests_to_retry.empty():
                retry_request = await queue_of_requests_to_retry.get()
                retry_batch.append(retry_request)

                # Process in batches of up to 10 requests
                if len(retry_batch) >= 10 or queue_of_requests_to_retry.empty():
                    for request in retry_batch:
                        token_estimate = self.estimate_total_tokens(
                            request.generic_request.messages
                        )
                        attempt_number = 6 - request.attempts_left
                        logger.info(
                            f"Processing retry for request {request.task_id} "
                            f"(attempt #{attempt_number} of 5). "
                            f"Previous errors: {request.result}"
                        )

                        # Wait for capacity if needed
                        while not status_tracker.has_capacity(token_estimate):
                            await asyncio.sleep(0.1)

                        # Consume capacity before making request
                        status_tracker.consume_capacity(token_estimate)

                        task = asyncio.create_task(
                            self.handle_single_request_with_retries(
                                request=request,
                                session=session,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        pending_retries.append(task)

                    # Wait for current batch to complete before processing more
                    if pending_retries:
                        await asyncio.gather(*pending_retries)
                        pending_retries = []
                    retry_batch = []

        status_tracker.pbar.close()

        # Log final status
        logger.info(f"Processing complete. Results saved to {save_filepath}")
        logger.info(f"Status tracker: {status_tracker}")

        if status_tracker.num_tasks_failed > 0:
            logger.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} "
                f"requests failed. Errors logged to {save_filepath}."
            )

    async def handle_single_request_with_retries(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Common wrapper for handling a single request with error handling and retries.

        This method implements the common try/except logic and retry mechanism,
        while delegating the actual API call to call_single_request.

        Args:
            request (APIRequest): The request to process
            session (aiohttp.ClientSession): Async HTTP session
            retry_queue (asyncio.Queue): Queue for failed requests
            save_filepath (str): Path to save responses
            status_tracker (StatusTracker): Tracks request status
        """
        try:
            generic_response = await self.call_single_request(
                request=request,
                session=session,
                status_tracker=status_tracker,
            )

            # Save response in the base class
            await self.append_generic_response(generic_response, save_filepath)

            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            status_tracker.pbar.update(1)

        except Exception as e:
            logger.warning(
                f"Request {request.task_id} failed with Exception {e}, attempts left {request.attempts_left}"
            )
            status_tracker.num_other_errors += 1
            request.result.append(e)

            if request.attempts_left > 0:
                request.attempts_left -= 1
                # Add retry queue logging
                logger.info(
                    f"Adding request {request.task_id} to retry queue. Will retry in next available slot. "
                    f"Attempts remaining: {request.attempts_left}"
                )
                retry_queue.put_nowait(request)
            else:
                logger.error(
                    f"Request {request.task_id} failed permanently after exhausting all 5 retry attempts. "
                    f"Errors: {[str(e) for e in request.result]}"
                )
                generic_response = GenericResponse(
                    response_message=None,
                    response_errors=[str(e) for e in request.result],
                    raw_request=request.api_specific_request,
                    raw_response=None,
                    generic_request=request.generic_request,
                    created_at=request.created_at,
                    finished_at=datetime.datetime.now(),
                )
                await self.append_generic_response(generic_response, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1

    @abstractmethod
    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: StatusTracker,
    ) -> GenericResponse:
        """Make a single API request without error handling.

        This method should implement the actual API call logic
        without handling retries or errors.

        Args:
            request (APIRequest): Request to process
            session (aiohttp.ClientSession): Async HTTP session
            status_tracker (StatusTracker): Tracks request status

        Returns:
            GenericResponse: The response from the API call
        """
        pass

    async def append_generic_response(self, data: GenericResponse, filename: str) -> None:
        """Append a response to a jsonl file with async file operations."""
        try:
            json_string = json.dumps(data.model_dump(), default=str)
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            async with aiofiles.open(filename, "a") as f:
                await f.write(json_string + "\n")
                # Ensure data is written to disk
                await f.flush()
                os.fsync(f.fileno())
            logger.debug(f"Successfully appended response to {filename}")
        except Exception as e:
            logger.error(f"Failed to write response to {filename}: {e}")
            raise  # Re-raise to ensure the error is handled by the caller
