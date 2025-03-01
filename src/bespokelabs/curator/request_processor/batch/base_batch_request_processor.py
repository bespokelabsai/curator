import asyncio
import json
import os
from abc import abstractmethod
from collections import Counter
from typing import Optional

import aiofiles
from litellm import model_cost

from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


class BaseBatchRequestProcessor(BaseRequestProcessor):
    """Abstract base class for processing batched API requests.

    This class provides the core functionality for submitting, tracking, and managing
    batch requests across different LLM API providers. It handles file operations,
    request tracking, batch status management, and response processing.

    The batch processing workflow:
    1. Load or initialize batch status tracker
    2. Submit requests in batches to API
    3. Monitor batch processing status
    4. Download and process completed batches
    5. Handle failures and retries

    Attributes:
        working_dir (str): Directory for storing batch-related files.
        prompt_formatter (PromptFormatter): Formatter for structuring prompts.
        tracker (BatchStatusTracker): Tracks status of submitted batches.
        batch_objects_file (str): Path to file storing batch objects.
        semaphore (asyncio.Semaphore): Controls concurrent batch operations.

    Note:
        Subclasses must implement abstract methods to provide API-specific
        functionality while maintaining consistent batch processing behavior.
    """

    def __init__(self, config: BatchRequestProcessorConfig):
        """Initialize the batch request processor.

        Args:
            config: Configuration object containing batch processing parameters.
        """
        super().__init__(config)
        self._tracker_console = None  # Add this line to store console for testing

    @property
    def backend(self) -> str:
        """Backend property."""
        return "base"

    @property
    def compatible_provider(self) -> str:
        """Compatible provider property."""
        return self.backend

    def requests_to_responses(self, generic_request_files: list[str]) -> None:
        """Process multiple request files using batch API operations.

        Orchestrates the complete batch processing workflow:
        1. Validates batch size limits
        2. Initializes concurrent operation controls
        3. Submits requests in batches
        4. Monitors and processes batch results

        Args:
            generic_request_files: List of paths to files containing requests.

        Raises:
            ValueError: If batch size exceeds API limits.
            RuntimeError: If batch processing fails or no successful responses.

        Side Effects:
            - Creates batch tracking files in working directory
            - Updates progress bars for batch submission and processing
            - Generates response files for completed batches
        """
        self.semaphore = asyncio.Semaphore(self.max_concurrent_batch_operations)
        self._batch_objects_file_lock = asyncio.Lock()
        self.batch_objects_file = os.path.join(self.working_dir, "batch_objects.jsonl")

        self._attempt_loading_batch_status_tracker(generic_request_files)

        run_in_event_loop(self.submit_batches_from_request_files(generic_request_files))
        run_in_event_loop(self.poll_and_process_batches())
        self.tracker.stop_tracker()

    @property
    @abstractmethod
    def max_requests_per_batch(self) -> int:
        """Maximum number of requests allowed in a single batch.

        This property must be implemented by subclasses to specify their
        API-specific batch size limits.

        Returns:
            int: Maximum number of requests that can be included in one batch.

        Note:
            This limit is enforced during batch creation to prevent API errors.
        """
        pass

    @property
    @abstractmethod
    def max_bytes_per_batch(self) -> int:
        """Maximum size in bytes allowed for a single batch.

        This property must be implemented by subclasses to specify their
        API-specific batch size limits in bytes.

        Returns:
            int: Maximum allowed size of a batch in bytes.

        Note:
            This limit is enforced during batch file creation to prevent API errors.
        """
        pass

    @property
    @abstractmethod
    def max_concurrent_batch_operations(self) -> int:
        """Maximum number of concurrent batch operations allowed.

        This property must be implemented by subclasses to specify their
        API-specific concurrency limits for batch operations.

        Returns:
            int: Maximum number of batch operations that can run concurrently.

        Note:
            This limit is enforced via semaphore during batch operations.
        """
        pass

    @abstractmethod
    async def submit_batch(self, requests: list[dict], metadata: Optional[dict] = None) -> GenericBatch:
        """Submit a batch of requests to the API provider.

        Args:
            requests: List of API-specific request dictionaries.
            metadata: Optional metadata to associate with the batch.

        Returns:
            GenericBatch: Standardized batch object with submission details.

        Note:
            Implementation must use self.semaphore for concurrency control.
        """
        pass

    @abstractmethod
    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieve current status of a submitted batch.

        Args:
            batch: The batch object to check status for.

        Returns:
            GenericBatch: Updated batch object with current status.

        Note:
            Implementation must use self.semaphore for concurrency control.
        """
        pass

    @abstractmethod
    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancel a running batch job.

        Args:
            batch: The batch object to cancel.

        Returns:
            GenericBatch: Updated batch object after cancellation attempt.

        Note:
            Implementation must use self.semaphore for concurrency control.
        """
        pass

    @abstractmethod
    async def download_batch(self, batch: GenericBatch) -> str | None:
        """Download results of a completed batch.

        Args:
            batch: The completed batch object to download.

        Returns:
            str | None: Raw response content if successful, None if failed.

        Note:
            Implementation should handle API-specific result formats.
        """
        pass

    @abstractmethod
    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parse API-specific response into standardized format.

        Args:
            raw_response: Raw response dictionary from API.
            generic_request: Original generic request object.
            batch: Batch object containing context information.

        Returns:
            GenericResponse: Standardized response object.

        Note:
            Should handle both successful and failed responses.
        """
        pass

    @abstractmethod
    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Convert generic request to API-specific format.

        Args:
            generic_request: Standardized request object.

        Returns:
            dict: API-specific request dictionary.

        Note:
            Must format request according to API provider requirements.
        """
        pass

    @abstractmethod
    def parse_api_specific_batch_object(self, batch: object, request_file: str | None = None) -> GenericBatch:
        """Convert API-specific batch object to generic format.

        Args:
            batch: API-specific batch object.
            request_file: Optional path to associated request file.

        Returns:
            GenericBatch: Standardized batch object.
        """
        pass

    @abstractmethod
    def parse_api_specific_request_counts(self, request_counts: object, request_file: Optional[str] = None) -> GenericBatchRequestCounts:
        """Convert API-specific request counts to generic format.

        Args:
            request_counts: API-specific request count object.
            request_file: Path to associated request file.

        Returns:
            GenericBatchRequestCounts: Standardized request count object.
        """
        pass

    def validate_config(self):
        """Validate batch request processor configuration.

        Ensures that configuration parameters are set correctly for batch processing.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.config.batch_size > self.max_requests_per_batch:
            raise ValueError(
                f"batch_size {self.config.batch_size} is greater than the maximum of "
                f"{self.max_requests_per_batch:,} requests per batch that {self.__class__.__name__} supports. "
                f"Please set your batch_size to be less than or equal to {self.max_requests_per_batch:,}."
            )

    def _attempt_loading_batch_status_tracker(self, request_files: set[str]):
        """Load existing batch status tracker or create new one.

        Args:
            request_files: Set of paths to request files to track.

        Side Effects:
            - Loads tracker from batch_objects_file if it exists
            - Creates new tracker if file doesn't exist
            - Sets self.tracker with loaded/created tracker
        """
        if os.path.exists(self.batch_objects_file):
            with open(self.batch_objects_file, "r") as f:
                self.tracker = BatchStatusTracker.model_validate_json(f.read())
                self.tracker.viewer_client = self._viewer_client
            logger.info(f"Loaded existing tracker from {self.batch_objects_file}")
        else:
            self.tracker = BatchStatusTracker(
                unsubmitted_request_files=set(request_files),
                viewer_client=self._viewer_client,
            )

        self.tracker.model = self.prompt_formatter.model_name
        self.tracker.n_total_requests = self.total_requests

        # Set cost information if available
        if self.prompt_formatter.model_name in model_cost:
            # Batch requests are 50% cheaper
            self.tracker.input_cost_per_million = (model_cost[self.prompt_formatter.model_name]["input_cost_per_token"] * 1_000_000) * 0.5
            self.tracker.output_cost_per_million = (model_cost[self.prompt_formatter.model_name]["output_cost_per_token"] * 1_000_000) * 0.5
        else:
            from bespokelabs.curator.cost import external_model_cost

            self.tracker.input_cost_per_million = (
                external_model_cost(self.prompt_formatter.model_name, provider=self.compatible_provider, completion_window=self.config.completion_window)[
                    "input_cost_per_token"
                ]
                * 1_000_000
            )
            self.tracker.output_cost_per_million = (
                external_model_cost(self.prompt_formatter.model_name, provider=self.compatible_provider, completion_window=self.config.completion_window)[
                    "output_cost_per_token"
                ]
                * 1_000_000
            )

        # Start the tracker with the console from constructor
        self.tracker.start_tracker(self._tracker_console)

    async def update_batch_objects_file(self):
        """Update batch objects file with current tracker state.

        Side Effects:
            - Writes current tracker state to batch_objects_file
            - Uses file lock to prevent concurrent writes
        """
        async with self._batch_objects_file_lock:
            with open(self.batch_objects_file, "w") as f:
                f.write(self.tracker.model_dump_json())

    def create_batch_file(self, api_specific_requests: list[dict]) -> str:
        """Create a batch file from API-specific requests.

        Validates request count and file size against API limits before creating
        the batch file content.

        Args:
            api_specific_requests: List of API-specific request dictionaries.

        Returns:
            str: Encoded file content ready for API upload.

        Raises:
            ValueError: If batch exceeds request count or size limits.

        Side Effects:
            - Logs debug information about batch file size
        """
        n_requests = len(api_specific_requests)
        if n_requests > self.max_requests_per_batch:
            raise ValueError(
                f"Batch file contains {n_requests:,} requests, "
                f"which is more than the maximum of {self.max_requests_per_batch:,} requests per batch that {self.__class__.__name__} supports. "
                f"Preventing batch submission. Please reduce `batch_size`."
            )

        # Join requests with newlines and encode to bytes for upload
        file_content = "\n".join(json.dumps(r) for r in api_specific_requests).encode()
        file_content_size = len(file_content)
        logger.debug(f"Batch file content size: {file_content_size / (1024 * 1024):.2f} MB ({file_content_size:,} bytes)")
        if file_content_size > self.max_bytes_per_batch:
            raise ValueError(
                f"Batch file content size {file_content_size:,} bytes "
                f"is greater than the maximum of {self.max_bytes_per_batch:,} bytes per batch that {self.__class__.__name__} supports. "
                f"Please reduce your batch size or request content size (via prompt_func and response_format)."
            )
        return file_content

    async def submit_batch_from_request_file(
        self,
        request_file: str,
        completed_request_ids: set[int],
    ):
        """Submit a batch of requests from a file.

        Reads requests from file, converts them to API-specific format,
        and submits them as a batch.

        Args:
            request_file: Path to file containing request data.
            completed_request_ids: Set of request IDs that have already been completed
                and should be skipped.

        Side Effects:
            - Updates batch submission progress bar
            - Updates tracker with submitted batch status
            - Creates batch metadata with request file path
            - Updates batch objects file
        """
        metadata = {"request_file": request_file}
        requests = self.requests_from_generic_request_file(request_file, completed_request_ids)
        batch = await self.submit_batch(requests, metadata)
        self.tracker.mark_as_submitted(batch, len(requests))
        await self.update_batch_objects_file()

    def requests_from_generic_request_file(self, request_file: str, completed_request_ids: set[int]) -> list[dict]:
        """Reads and converts generic requests from a file into API-specific request format.

        Args:
            request_file (str): Path to the file containing generic requests in JSONL format.
            completed_request_ids (set[int]): Set of request IDs that already have responses, these will be skipped
        Returns:
            list[dict]: API-specific request bodies ready for submission.

        Side Effects:
            - Reads from request file on disk
        """
        api_specific_requests = []

        with open(request_file, "r") as file:
            for line in file:
                request = GenericRequest.model_validate_json(line.strip())
                if request.original_row_idx in completed_request_ids:
                    continue
                api_specific_request = self.create_api_specific_request_batch(request)
                api_specific_requests.append(api_specific_request)

        return api_specific_requests

    async def generic_response_file_from_responses(self, responses: list[dict], batch: GenericBatch) -> str | None:
        """Process API responses and create generic response file.

        Converts API-specific responses to GenericResponse objects and writes them
        to a response file. Handles successful and failed responses, including
        token usage tracking and cost calculation.

        Args:
            responses: List of raw API response dictionaries.
            batch: Batch object containing request metadata.

        Returns:
            str | None: Path to created response file, or None if creation failed.

        Side Effects:
            - Creates response file from request file name
            - Writes GenericResponse objects in JSONL format
            - Calculates costs with batch discount
            - Handles failed requests with error details
            - Updates tracker with token usage and cost stats
        """
        request_file = batch.request_file
        request_dir = os.path.dirname(request_file)
        request_filename = os.path.basename(request_file)
        response_filename = request_filename.replace("requests_", "responses_")
        response_file = os.path.join(request_dir, response_filename)

        # Load generic requests from request file
        generic_request_map = {}
        with open(request_file, "r") as f:
            for line in f:
                generic_request = GenericRequest.model_validate_json(line)
                generic_request_map[generic_request.original_row_idx] = generic_request

        # Track total token usage and cost for this batch
        total_token_usage = _TokenUsage(input=0, output=0)
        total_cost = 0.0

        # appending allows for the resubmitted resumed batch

        stream_response_tasks = []
        invalid_finish_responses = []
        failed_processed_responses = []
        async with aiofiles.open(response_file, "a") as f:
            for raw_response in responses:
                request_idx = int(raw_response["custom_id"])
                generic_request = generic_request_map[request_idx]
                generic_response = self.parse_api_specific_response(raw_response, generic_request, batch)

                if generic_response.finish_reason in self.config.invalid_finish_reasons:
                    invalid_finish_responses.append({"request_id": request_idx, "finish_reason": generic_response.finish_reason})
                    continue

                processed_responses = self._process_response(generic_response)
                generic_response.parsed_response_message = processed_responses
                if processed_responses is None:
                    failed_processed_responses.append(request_idx)
                    continue

                # Write response to file
                response_dump = generic_response.model_dump(mode="json")
                r = json.dumps(response_dump, default=str)
                await f.write(r + "\n")

                # Update token and cost totals
                if generic_response.token_usage:
                    total_token_usage.input += generic_response.token_usage.input
                    total_token_usage.output += generic_response.token_usage.output
                if generic_response.response_cost:
                    total_cost += generic_response.response_cost

                # Stream responses to viewer client
                idx = self.tracker.num_parsed_responses
                self.tracker.num_parsed_responses = idx + len(processed_responses)
                stream_response_tasks.append(self.viewer_client.stream_response(json.dumps(response_dump), idx))

        await asyncio.gather(*stream_response_tasks)
        if failed_processed_responses:
            logger.warning(f"Batch {batch.id} has {len(failed_processed_responses)} failed responses due to parse function errors.")

        if invalid_finish_responses:
            logger.warning(f"Batch {batch.id} has {len(invalid_finish_responses)} invalid finish responses. Please check the logs above for details.")
            invalid_finish_reasons = dict(Counter([response["finish_reason"] for response in invalid_finish_responses]))
            logger.warning(f"Invalid finish responses: {invalid_finish_reasons}")

        # Update tracker with token usage and cost stats
        self.tracker.update_token_and_cost(total_token_usage, total_cost)
        await self.viewer_client.session_completed()
        return response_file

    async def submit_batches_from_request_files(
        self,
        request_files: set[str],
    ):
        """Submit multiple request files as batches to API.

        Manages the complete batch submission workflow including tracking,
        progress monitoring, and concurrent submission of multiple files.

        Args:
            request_files: Set of paths to request files to process.

        Side Effects:
            - Loads or creates batch status tracker
            - Updates tracker with batch statuses
            - Creates and updates batch submission progress bar
            - Submits batches concurrently using asyncio
            - Updates batch objects file
        """
        tasks = []

        # Update session status to inprogress
        await self._viewer_client.session_inprogress()

        # check existing response files for resuming
        for batch in self.tracker.downloaded_batches.values():
            response_file = batch.request_file.replace("requests_", "responses_")
            completed_request_ids, completed_parsed_responses = self.validate_existing_response_file(response_file)
            self.tracker.num_parsed_responses += completed_parsed_responses
            n_total_batch_requests = self.read_metadata_file(batch.request_file).get("num_jobs")
            if len(completed_request_ids) < n_total_batch_requests:
                tasks.append(self.submit_batch_from_request_file(batch.request_file, completed_request_ids))

        # submit full batches of unsubmitted request files
        for request_file in self.tracker.unsubmitted_request_files:
            tasks.append(self.submit_batch_from_request_file(request_file, set()))

        # exit early if no batches to submit
        if len(tasks) == 0:
            return

        await asyncio.gather(*tasks)
        assert self.tracker.unsubmitted_request_files == set()

    async def check_batch_status(self, batch: GenericBatch) -> GenericBatch | None:
        """Check current status of a batch job.

        Retrieves current batch status from API and updates tracking information.
        Handles batch completion detection and request count updates.

        Args:
            batch: The batch object to check status for.

        Returns:
            GenericBatch | None: Updated batch object if found, None if not found.

        Side Effects:
            - Updates tracker with current batch status
            - Updates request completion counts
            - Logs batch status and request counts
            - Marks completed batches as finished
        """
        async with self.semaphore:
            batch = await self.retrieve_batch(batch)
            if batch is not None:
                self.tracker.update_submitted(batch)

                n_succeeded_requests = batch.request_counts.succeeded
                n_failed_requests = batch.request_counts.failed
                n_total_requests = batch.request_counts.total

                logger.debug(
                    f"Batch {batch.id} status: {batch.raw_status} requests: "
                    f"{n_succeeded_requests}/{n_failed_requests}/{n_total_requests} "
                    "succeeded/failed/total"
                )

                if batch.status == GenericBatchStatus.FINISHED.value:
                    logger.debug(f"Batch {batch.id} finished with status: {batch.raw_status}")
                    self.tracker.mark_as_finished(batch)
                    await self.update_batch_objects_file()

    async def poll_and_process_batches(self) -> None:
        """Monitor and process batches until completion.

        Continuously polls batch status and downloads results when complete.
        Manages batch lifecycle including status checks, downloads, and error handling.

        Returns:
            None

        Raises:
            RuntimeError: If no batches complete successfully.

        Side Effects:
            - Creates and updates request progress bar
            - Updates batch tracker state
            - Downloads and processes completed batches
            - Creates response files for completed batches
            - Logs progress and status information
            - Sleeps between status checks
        """
        # loop until all batches have been returned
        all_response_files = []
        while self.tracker.n_submitted_batches + self.tracker.n_finished_batches > 0:
            # check batch status also updates the tracker
            status_tasks = [self.check_batch_status(batch) for batch in self.tracker.submitted_batches.values()]
            await asyncio.gather(*status_tasks)
            await self.update_batch_objects_file()

            download_tasks = [self.download_batch_to_response_file(batch) for batch in self.tracker.finished_batches.values()]
            # Failed downloads return None and print any errors that occurred
            all_response_files.extend(await asyncio.gather(*download_tasks))

            logger.debug(
                f"Batches returned: {self.tracker.n_finished_or_downloaded_batches:,}/{self.tracker.n_total_batches:,} "
                f"Requests completed: {self.tracker.n_finished_or_downloaded_succeeded_requests:,}/{self.tracker.n_total_requests:,}"
            )

            self.tracker.update_display()
            if self.tracker.n_submitted_batches + self.tracker.n_finished_batches > 0:
                logger.debug(f"Sleeping for {self.config.batch_check_interval} seconds...")
                await asyncio.sleep(self.config.batch_check_interval)

        response_files = filter(None, all_response_files)
        await self.viewer_client.close()
        if self.tracker.n_downloaded_batches == 0 or not response_files:
            raise RuntimeError(f"None of the submitted batches completed successfully. Please check the logs above and {self.web_dashboard} for errors.")

    async def download_batch_to_response_file(self, batch: GenericBatch) -> str | None:
        """Download and process completed batch results."""
        file_content = await self.download_batch(batch)

        if file_content is None:
            return None

        # Write responses to file and update stats
        response_file = await self.generic_response_file_from_responses(file_content, batch)

        logger.debug(f"Batch {batch.id} written to {response_file}")

        if self.config.delete_successful_batch_files:
            await self.delete_file(batch.input_file_id, self.semaphore)
            await self.delete_file(batch.output_file_id, self.semaphore)

        # Update tracker with downloaded batch
        self.tracker.mark_as_downloaded(batch)
        await self.update_batch_objects_file()

        return response_file

    async def cancel_batches(self):
        """Cancel all currently submitted batches.

        Attempts to cancel all batches that are currently in submitted state.
        Handles cases where no batches are submitted.

        Side Effects:
            - Attempts to cancel all submitted batches concurrently
            - Logs warning if no batches to cancel
            - Updates batch status through cancel_batch calls
        """
        if self.tracker.n_submitted_batches == 0:
            logger.warning("No batches to be cancelled, but cancel_batches=True.")
            return
        tasks = [self.cancel_batch(batch) for batch in self.tracker.submitted_batches.values()]
        await asyncio.gather(*tasks)
