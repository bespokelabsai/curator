"""Base class for online request processors that make real-time API calls.

This module provides the core functionality for making API requests in real-time,
handling rate limiting, retries, and concurrent processing.
"""

import asyncio
import datetime
import json
import os
import time
import typing as t
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field

import aiofiles
import aiohttp

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor import _DEFAULT_COST_MAP
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker, TokenLimitStrategy
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.prompt import _MultiModalPrompt
from bespokelabs.curator.types.token_usage import _TokenUsage

_MAX_OUTPUT_MVA_WINDOW = 50


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

        defaults = _DEFAULT_COST_MAP["online"]["default"]["ratelimit"]
        self.token_limit_strategy = TokenLimitStrategy.default
        self.manual_max_requests_per_minute = config.max_requests_per_minute
        self.manual_max_tokens_per_minute = config.max_tokens_per_minute
        self.default_max_requests_per_minute = defaults["max_requests_per_minute"]
        self.default_max_concurrent_requests = defaults["max_concurrent_requests"]
        self.default_max_tokens_per_minute = defaults["max_tokens_per_minute"][self.token_limit_strategy.value]
        self.header_based_max_requests_per_minute = None
        self.header_based_max_tokens_per_minute = None

        self.manual_max_concurrent_requests = config.max_concurrent_requests
        self.header_based_max_concurrent_requests = None

        # The rich.Console used for the status tracker, only set for testing
        self._tracker_console = None
        self._output_tokens_window = deque(maxlen=_MAX_OUTPUT_MVA_WINDOW)
        self._semaphore = None
        if self.max_concurrent_requests is not None:
            self._semaphore = asyncio.Semaphore(t.cast(int, self.max_concurrent_requests))

    @property
    def backend(self) -> str:
        """Backend property."""
        return "base"

    @property
    def compatible_provider(self) -> str:
        """Compatible provider property."""
        return "base"

    def validate_config(self):
        """Validate online request processor configuration.

        Ensures that configuration parameters are set correctly.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        pass

    def _unpack_multimodal(self, request: GenericRequest) -> GenericRequest:
        messages = request.messages
        if request.is_multimodal_prompt is True and self._multimodal_prompt_supported is False:
            raise TypeError(f"Multimodal prompts are not supported by this model {self.config.model}.")

        if request.is_multimodal_prompt is False:
            return request

        unpacked_messages = []
        for message in messages:
            # Load message content as multimodal prompt
            content = _MultiModalPrompt.model_validate(message["content"])
            content = self._handle_multi_modal_prompt(content)
            message["content"] = content
            unpacked_messages.append(message)

        request.messages = unpacked_messages
        return request

    @abstractmethod
    def file_upload_limit_check(self, base64_image: str) -> None:
        """Check if the image size is within the allowed limit."""
        pass

    def _handle_multi_modal_prompt(self, message):
        content = []
        texts = message.texts

        def _format_multimodal(data, mime_type="image/png"):
            """Format multimodal prompt data for API request."""
            mime_type = mime_type or "image/png"
            if data.url and not data.is_local:
                return {"type": "image_url", "image_url": {"url": data.url}}
            else:
                base64_content = data.serialize()
                self.file_upload_limit_check(base64_content)

                content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_content}",
                    },
                }
                if "image" in mime_type:
                    content["image_url"].update({"detail": data.detail})
                return content

        for text in texts:
            content.append({"type": "text", "text": text})
        for image in message.images:
            content.append(_format_multimodal(image, mime_type=image.mime_type))
        for file in message.files:
            content.append(_format_multimodal(file, mime_type=file.mime_type))

        return content

    @property
    def max_concurrent_requests(self) -> int | None:
        """Gets the maximum concurrent requests rate limit.

        Returns the manually set limit if available, falls back to header-based limit,
        or uses default value as last resort.
        """
        if self.manual_max_concurrent_requests:
            logger.info(f"Manually set max_concurrent_requests to {self.manual_max_concurrent_requests}")
            return self.manual_max_concurrent_requests

        elif self.header_based_max_concurrent_requests:
            logger.info(f"Automatically set max_concurrent_requests to {self.header_based_max_concurrent_requests}")
            return self.header_based_max_concurrent_requests
        else:
            return None

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
    def estimate_total_tokens(self, messages: list) -> _TokenUsage:
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
        cost = self._cost_processor.cost(completion_response=response)
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
                    response_file=response_file,
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
            status_tracker.last_update_time = time.time()

    def free_capacity(self, tracker, tokens):
        """Free blocked capacity."""

    async def process_requests_from_file(
        self,
        generic_request_filepath: str,
        response_file: str,
    ) -> None:
        """Processes API requests in parallel, throttling to stay under rate limits.

        Args:
            generic_request_filepath: Path to file containing requests
            response_file: Path where the response data will be saved
        """
        # Initialize trackers
        queue_of_requests_to_retry: asyncio.Queue[APIRequest] = asyncio.Queue()
        status_tracker = OnlineStatusTracker(
            token_limit_strategy=self.token_limit_strategy,
            max_requests_per_minute=self.max_requests_per_minute,
            max_tokens_per_minute=self.max_tokens_per_minute,
            compatible_provider=self.compatible_provider,
            viewer_client=self._viewer_client,
        )

        # Resume if a response file exists
        completed_request_ids, completed_parsed_responses = self.validate_existing_response_file(response_file)

        # Count total requests
        status_tracker.num_tasks_already_completed = len(completed_request_ids)
        status_tracker.num_parsed_responses = completed_parsed_responses
        status_tracker.total_requests = self.total_requests
        status_tracker.model = self.prompt_formatter.model_name
        status_tracker.start_tracker(self._tracker_console)

        # Use higher connector limit for better throughput
        tcp_limit = self.max_concurrent_requests if status_tracker.max_requests_per_minute is None else status_tracker.max_requests_per_minute
        # Update session status to inprogress
        await self._viewer_client.session_inprogress()
        connector = aiohttp.TCPConnector(limit=10 * tcp_limit)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with aiofiles.open(generic_request_filepath) as file:
                pending_requests = []
                async for line in file:
                    if self._semaphore:
                        await self._semaphore.acquire()

                    generic_request = GenericRequest.model_validate_json(line)

                    if generic_request.original_row_idx in completed_request_ids:
                        continue

                    # Unpack multimodal prompts
                    generic_request = self._unpack_multimodal(generic_request)

                    request = APIRequest(
                        task_id=status_tracker.num_tasks_started,
                        generic_request=generic_request,
                        api_specific_request=self.create_api_specific_request_online(generic_request),
                        attempts_left=self.config.max_retries,
                        prompt_formatter=self.prompt_formatter,
                    )

                    if status_tracker.max_tokens_per_minute is not None:
                        token_estimate = self.estimate_total_tokens(request.generic_request.messages)
                    else:
                        token_estimate = None

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
                            response_file=response_file,
                            status_tracker=status_tracker,
                            blocked_capacity=token_estimate,
                        )
                    )
                    pending_requests.append(task)

                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1

            if pending_requests:
                await asyncio.gather(*pending_requests)

            # Process any remaining retries in the queue
            pending_retries = set()
            while not queue_of_requests_to_retry.empty() or pending_retries:
                # Process new items from the queue if we have capacity
                if self._semaphore:
                    await self._semaphore.acquire()

                if not queue_of_requests_to_retry.empty():
                    retry_request = await queue_of_requests_to_retry.get()
                    if status_tracker.max_tokens_per_minute is not None:
                        token_estimate = self.estimate_total_tokens(retry_request.generic_request.messages)
                    else:
                        token_estimate = None

                    attempt_number = self.config.max_retries - retry_request.attempts_left
                    logger.debug(
                        f"Retrying request {retry_request.task_id} "
                        f"(attempt #{attempt_number} of {self.config.max_retries})"
                        f"Previous errors: {retry_request.result}"
                    )

                    # Wait for capacity if needed
                    while not status_tracker.has_capacity(token_estimate):
                        await asyncio.sleep(0.1)

                    # Wait for rate limits cool down if needed
                    await self.cool_down_if_rate_limit_error(status_tracker)

                    # Consume capacity before making request
                    status_tracker.consume_capacity(token_estimate)

                    task = asyncio.create_task(
                        self.handle_single_request_with_retries(
                            request=retry_request,
                            session=session,
                            retry_queue=queue_of_requests_to_retry,
                            response_file=response_file,
                            status_tracker=status_tracker,
                            blocked_capacity=token_estimate,
                        )
                    )
                    pending_retries.add(task)

                # Wait for some tasks to complete
                if pending_retries:
                    done, pending_retries = await asyncio.wait(pending_retries, timeout=0.1)

        status_tracker.stop_tracker()
        await self.viewer_client.session_completed()

        # Log final status
        logger.info(f"Processing complete. Results saved to {response_file}")
        logger.info(f"Status tracker: {status_tracker}")

        if status_tracker.num_tasks_failed > 0:
            logger.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {response_file}.")

    def _free_capacity(self, status_tracker: OnlineStatusTracker, used_capacity: "_TokenUsage", blocked_capacity: "_TokenUsage"):
        if status_tracker.max_tokens_per_minute is not None:
            status_tracker.free_capacity(used_capacity, blocked_capacity)

    async def handle_single_request_with_retries(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        retry_queue: asyncio.Queue,
        response_file: str,
        status_tracker: OnlineStatusTracker,
        blocked_capacity: "_TokenUsage",
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
            blocked_capacity: Blocked token capacity
        """
        try:
            # Estimate tokens before making request
            generic_response = None
            token_estimate: _TokenUsage = blocked_capacity or self.estimate_total_tokens(request.generic_request.messages)

            # Add new estimate to projection (pre_request=True indicates new estimate)
            status_tracker.update_cost_projection(token_estimate, pre_request=True)

            generic_response = await self.call_single_request(
                request=request,
                session=session,
                status_tracker=status_tracker,
            )

            if generic_response.finish_reason in self.config.invalid_finish_reasons:
                logger.debug(
                    f"Invalid finish_reason {generic_response.finish_reason}."
                    " Raw response {generic_response.raw_response} "
                    "for request {generic_response.raw_request}"
                )
                # Update cost projection with actual usage but mark as failed
                status_tracker.update_stats(generic_response.token_usage, generic_response.response_cost)
                raise ValueError(f"finish_reason was {generic_response.finish_reason}")

            # Allows us to retry on responses that don't match the response format
            self.prompt_formatter.response_to_response_format(generic_response.response_message)

        except Exception as e:
            status_tracker.num_other_errors += 1
            request.result.append(e)

            if generic_response and generic_response.token_usage is not None:
                used_tokens: _TokenUsage = _TokenUsage(input=generic_response.token_usage.input, output=generic_response.token_usage.output)
                status_tracker.update_cost_projection(used_tokens)
            else:
                status_tracker.update_cost_projection(None)

            if request.attempts_left > 0:
                request.attempts_left -= 1
                logger.warning(
                    f"Encountered '{e.__class__.__name__}: {e}' during attempt "
                    f"{self.config.max_retries - request.attempts_left} of {self.config.max_retries} "
                    f"while processing request {request.task_id}"
                )
                retry_queue.put_nowait(request)
            else:
                error_counts = Counter(str(err) for err in request.result)
                formatted_errors = [f"{error}(x{count})" for error, count in error_counts.items()]
                logger.error(
                    f"Request {request.task_id} failed permanently after exhausting all {self.config.max_retries} retry attempts. "
                    f"Errors: {formatted_errors}"
                )

                generic_response = GenericResponse(
                    response_message=None,
                    response_errors=formatted_errors,
                    raw_request=request.api_specific_request,
                    raw_response=None,
                    generic_request=request.generic_request,
                    created_at=request.created_at,
                    finished_at=datetime.datetime.now(),
                )
                await self.append_generic_response(status_tracker, generic_response, response_file)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
            return
        else:
            self._add_output_token_moving_window(generic_response.token_usage.output)
        finally:
            if self._semaphore:
                self._semaphore.release()

        # Save response in the base class
        await self.append_generic_response(status_tracker, generic_response, response_file)

        status_tracker.num_tasks_in_progress -= 1
        status_tracker.num_tasks_succeeded += 1

        # Update cost projection with actual usage
        used_tokens: _TokenUsage = _TokenUsage(input=generic_response.token_usage.input, output=generic_response.token_usage.output)
        # Update stats with actual usage
        status_tracker.update_stats(generic_response.token_usage, generic_response.response_cost)
        status_tracker.update_cost_projection(used_tokens)

        # Free the extra capacity blocked before request with actual consumed capacity.
        self._free_capacity(status_tracker, used_tokens, blocked_capacity)

    def _add_output_token_moving_window(self, tokens):
        self._output_tokens_window.append(tokens)

    def _output_tokens_moving_average(self):
        return sum(self._output_tokens_window) / (len(self._output_tokens_window) or _MAX_OUTPUT_MVA_WINDOW)

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

    async def append_generic_response(self, status_tracker: OnlineStatusTracker, data: GenericResponse, filename: str) -> None:
        """Append a response to a jsonl file with async file operations.

        Args:
            status_tracker: Tracker containing request status
            data: Response data to append
            filename: File to append to
        """
        raw_response = data.response_message
        responses = self._process_response(data)
        if not responses:
            return
        data.parsed_response_message = responses
        data.response_message = raw_response
        data_dump = data.model_dump()
        json_string = json.dumps(data_dump, default=str)
        async with aiofiles.open(filename, "a") as f:
            await f.write(json_string + "\n")
        logger.debug(f"Successfully appended response to {filename}")
        filename = os.path.basename(filename).split(".")[0]
        idx = status_tracker.num_parsed_responses
        status_tracker.num_parsed_responses = idx + len(responses)
        await self.viewer_client.stream_response(json_string, idx)
