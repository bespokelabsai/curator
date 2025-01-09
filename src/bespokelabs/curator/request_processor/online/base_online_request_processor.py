"""Base class for online request processors that make real-time API calls.

This module provides the core functionality for making API requests in real-time,
handling rate limiting, retries, and parallel processing.
"""

import asyncio
import datetime
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import aiofiles
import aiohttp
import litellm
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata.

    Attributes:
        task_id: Unique identifier for the request
        generic_request: The generic request object to be processed
        api_specific_request: The request formatted for the specific API
        attempts_left: Number of retry attempts remaining
        result: List to store results/errors from attempts
        prompt_formatter: Formatter for prompts and responses
        created_at: Timestamp when request was created
    """

    task_id: int
    generic_request: GenericRequest
    api_specific_request: dict
    attempts_left: int
    result: list = field(default_factory=list)
    prompt_formatter: PromptFormatter = field(default=None)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class BaseOnlineRequestProcessor(BaseRequestProcessor, ABC):
    """Abstract base class for online request processors that make real-time API calls.

    This class handles rate limiting, retries, parallel processing and other common
    functionality needed for making real-time API requests.

    Args:
        config: Configuration object containing settings for the request processor
    """

    def __init__(self, config: OnlineRequestProcessorConfig):
        """Initialize the BaseOnlineRequestProcessor."""
        super().__init__(config)
        self.manual_max_requests_per_minute = config.max_requests_per_minute
        self.manual_max_tokens_per_minute = config.max_tokens_per_minute
        self.default_max_requests_per_minute = 10
        self.default_max_tokens_per_minute = 100_000
        self.header_based_max_requests_per_minute = None
        self.header_based_max_tokens_per_minute = None

    @property
    def backend(self) -> str:
        """Backend property."""
        return "base"

    @property
    def max_requests_per_minute(self) -> int:
        """Gets the maximum requests per minute rate limit.

        Returns the manually set limit if available, falls back to header-based limit,
        or uses default value as last resort.
        """
        if self.manual_max_requests_per_minute:
            logger.info(f"Manually set max_requests_per_minute to {self.manual_max_requests_per_minute}")
            return self.manual_max_requests_per_minute
        elif self.header_based_max_requests_per_minute:
            logger.info(f"Automatically set max_requests_per_minute to {self.header_based_max_requests_per_minute}")
            return self.header_based_max_requests_per_minute
        else:
            logger.warning(
                f"No manual max_requests_per_minute set, and headers based detection failed, using default value of {self.default_max_requests_per_minute}"
            )
            return self.default_max_requests_per_minute

    @property
    def max_tokens_per_minute(self) -> int:
        """Gets the maximum tokens per minute rate limit.

        Returns the manually set limit if available, falls back to header-based limit,
        or uses default value as last resort.
        """
        if self.manual_max_tokens_per_minute:
            logger.info(f"Manually set max_tokens_per_minute to {self.manual_max_tokens_per_minute}")
            return self.manual_max_tokens_per_minute
        elif self.header_based_max_tokens_per_minute:
            logger.info(f"Automatically set max_tokens_per_minute to {self.header_based_max_tokens_per_minute}")
            return self.header_based_max_tokens_per_minute
        else:
            logger.warning(
                f"No manual max_tokens_per_minute set, and headers based detection failed, using default value of {self.default_max_tokens_per_minute}"
            )
            return self.default_max_tokens_per_minute

    @abstractmethod
    def estimate_total_tokens(self, messages: list) -> int:
        """Estimate total tokens for a request.

        Args:
            messages: List of messages to estimate token count for

        Returns:
            Estimated total number of tokens
        """
        pass

    @abstractmethod
    def estimate_output_tokens(self) -> int:
        """Estimate output tokens for a request.

        Returns:
            Estimated number of output tokens
        """
        pass

    @abstractmethod
    def create_api_specific_request_online(self, generic_request: GenericRequest) -> dict:
        """Create an API-specific request body from a generic request body.

        Args:
            generic_request: The generic request to convert

        Returns:
            API-specific request dictionary
        """
        pass

    def completion_cost(self, response):
        """Calculate the cost of a completion response using litellm.

        Args:
            response: The completion response to calculate cost for

        Returns:
            Calculated cost of the completion
        """
        # Calculate cost using litellm
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            # We should ideally not catch a catch-all exception here. But litellm is not throwing any specific error.
            cost = 0

        return cost

    def requests_to_responses(
        self,
        generic_request_files: list[str],
    ) -> None:
        """Process multiple request files and generate corresponding response files.

        Args:
            generic_request_files: List of request files to process
        """
        for request_file in generic_request_files:
            response_file = request_file.replace("requests_", "responses_")
            run_in_event_loop(
                self.process_requests_from_file(
                    generic_request_filepath=request_file,
                    save_filepath=response_file,
                    resume=True,
                )
            )

    async def cool_down_if_rate_limit_error(self, status_tracker: OnlineStatusTracker) -> None:
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

    async def process_requests_from_file(
        self,
        generic_request_filepath: str,
        save_filepath: str,
        resume: bool,
        resume_no_retry: bool = False,
    ) -> None:
        """Processes API requests in parallel, throttling to stay under rate limits.

        Args:
            generic_request_filepath: Path to file containing requests
            save_filepath: Path to save responses
            resume: Whether to resume from previous progress
            resume_no_retry: Whether to skip retrying failed requests when resuming
        """
        # Initialize trackers
        queue_of_requests_to_retry: asyncio.Queue[APIRequest] = asyncio.Queue()
        status_tracker = OnlineStatusTracker()

        # Get rate limits
        status_tracker.max_requests_per_minute = self.max_requests_per_minute
        status_tracker.max_tokens_per_minute = self.max_tokens_per_minute

        # Track completed requests for resume functionality
        completed_request_ids = set()
        if os.path.exists(save_filepath):
            if resume:
                logger.info(f"Resuming progress by reading existing file: {save_filepath}")
                logger.debug(f"Removing all failed requests from {save_filepath} so they can be retried")
                temp_filepath = save_filepath + ".temp"  # This is a file extension, not a path join
                num_previously_failed_requests = 0

                with open(save_filepath, "r") as input_file, open(temp_filepath, "w") as output_file:
                    for line in input_file:
                        response = GenericResponse.model_validate_json(line)
                        if response.response_errors:
                            logger.debug(
                                f"Request {response.generic_request.original_row_idx} previously failed due to errors: "
                                f"{response.response_errors}, removing from output and will retry"
                            )
                            num_previously_failed_requests += 1
                        if response.response_message is None:
                            logger.debug(
                                f"Request {response.generic_request.original_row_idx} previously failed due to no response, removing from output and will retry"
                            )
                            num_previously_failed_requests += 1
                        else:
                            completed_request_ids.add(response.generic_request.original_row_idx)
                            output_file.write(line)

                logger.info(f"Found {len(completed_request_ids)} completed requests and " f"{num_previously_failed_requests} previously failed requests")
                logger.info("Failed requests and remaining requests will now be processed.")
                os.replace(temp_filepath, save_filepath)

            elif resume_no_retry:
                logger.warning(f"Resuming progress from existing file: {save_filepath}, without retrying failed requests")
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

                logger.info(f"Found {len(completed_request_ids)} total requests and " f"{num_previously_failed_requests} previously failed requests")
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
        total_requests = sum(1 for _ in open(generic_request_filepath))

        # Create progress bar
        status_tracker.pbar = tqdm(
            initial=len(completed_request_ids),
            total=total_requests,
            desc=f"Processing {self.__class__.__name__} requests",
        )

        # Use higher connector limit for better throughput
        connector = aiohttp.TCPConnector(limit=10 * status_tracker.max_requests_per_minute)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with aiofiles.open(generic_request_filepath) as file:
                pending_requests = []

                async for line in file:
                    generic_request = GenericRequest.model_validate_json(line)

                    if resume and generic_request.original_row_idx in completed_request_ids:
                        status_tracker.num_tasks_already_completed += 1
                        continue

                    request = APIRequest(
                        task_id=status_tracker.num_tasks_started,
                        generic_request=generic_request,
                        api_specific_request=self.create_api_specific_request_online(generic_request),
                        attempts_left=self.config.max_retries,
                        prompt_formatter=self.prompt_formatter,
                    )

                    token_estimate = self.estimate_total_tokens(request.generic_request.messages)

                    # Wait for capacity if needed
                    while not status_tracker.has_capacity(token_estimate):
                        await asyncio.sleep(0.1)

                    # Wait for rate limits cool down if needed
                    await self.cool_down_if_rate_limit_error(status_tracker)

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
            pending_retries = set()
            while not queue_of_requests_to_retry.empty() or pending_retries:
                # Process new items from the queue if we have capacity
                if not queue_of_requests_to_retry.empty():
                    retry_request = await queue_of_requests_to_retry.get()
                    token_estimate = self.estimate_total_tokens(retry_request.generic_request.messages)
                    attempt_number = self.config.max_retries - retry_request.attempts_left
                    logger.debug(
                        f"Retrying request {retry_request.task_id} "
                        f"(attempt #{attempt_number} of {self.config.max_retries})"
                        f"Previous errors: {retry_request.result}"
                    )

                    # Wait for capacity if needed
                    while not status_tracker.has_capacity(token_estimate):
                        await asyncio.sleep(0.1)

                    # Consume capacity before making request
                    status_tracker.consume_capacity(token_estimate)

                    task = asyncio.create_task(
                        self.handle_single_request_with_retries(
                            request=retry_request,
                            session=session,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    pending_retries.add(task)

                # Wait for some tasks to complete
                if pending_retries:
                    done, pending_retries = await asyncio.wait(pending_retries, timeout=0.1)

        status_tracker.pbar.close()

        # Log final status
        logger.info(f"Processing complete. Results saved to {save_filepath}")
        logger.info(f"Status tracker: {status_tracker}")

        if status_tracker.num_tasks_failed > 0:
            logger.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} " f"requests failed. Errors logged to {save_filepath}.")

    async def handle_single_request_with_retries(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: OnlineStatusTracker,
    ) -> None:
        """Common wrapper for handling a single request with error handling and retries.

        This method implements the common try/except logic and retry mechanism,
        while delegating the actual API call to call_single_request.

        Args:
            request: The request to process
            session: Async HTTP session
            retry_queue: Queue for failed requests
            save_filepath: Path to save responses
            status_tracker: Tracks request status
        """
        try:
            generic_response = await self.call_single_request(
                request=request,
                session=session,
                status_tracker=status_tracker,
            )

            # Allows us to retry on responses that don't match the response format
            self.prompt_formatter.response_to_response_format(generic_response.response_message)

            # Save response in the base class
            await self.append_generic_response(generic_response, save_filepath)

            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            status_tracker.pbar.update(1)

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
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Make a single API request without error handling.

        This method should implement the actual API call logic
        without handling retries or errors.

        Args:
            request: Request to process
            session: Async HTTP session
            status_tracker: Tracks request status

        Returns:
            The response from the API call
        """
        pass

    async def append_generic_response(self, data: GenericResponse, filename: str) -> None:
        """Append a response to a jsonl file with async file operations.

        Args:
            data: Response data to append
            filename: File to append to
        """
        json_string = json.dumps(data.model_dump(), default=str)
        async with aiofiles.open(filename, "a") as f:
            await f.write(json_string + "\n")
        logger.debug(f"Successfully appended response to {filename}")
