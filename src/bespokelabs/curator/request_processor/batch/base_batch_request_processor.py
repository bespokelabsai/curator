import asyncio
import json
import logging
import os
from abc import abstractmethod
from typing import Optional

from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        batch_submit_pbar (tqdm | None): Progress bar for batch submission.
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

    @property
    def backend(self) -> str:
        """Backend property."""
        return "base"

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
        if self.config.batch_size > self.max_requests_per_batch:
            raise ValueError(
                f"batch_size {self.config.batch_size} is greater than the maximum of "
                f"{self.max_requests_per_batch:,} requests per batch that {self.__class__.__name__} supports. "
                f"Please set your batch_size to be less than or equal to {self.max_requests_per_batch:,}."
            )

        self.semaphore = asyncio.Semaphore(self.max_concurrent_batch_operations)
        self._batch_objects_file_lock = asyncio.Lock()
        self.batch_objects_file = os.path.join(self.working_dir, "batch_objects.jsonl")
        self.batch_submit_pbar: tqdm | None = None
        self.request_pbar: tqdm | None = None

        run_in_event_loop(self.submit_batches_from_request_files(generic_request_files))
        logger.info(f"Submitted batches. These can be viewed in the web dashboard: {self.web_dashboard}")
        run_in_event_loop(self.poll_and_process_batches())

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
    def parse_api_specific_request_counts(self, request_counts: object) -> GenericBatchRequestCounts:
        """Convert API-specific request counts to generic format.

        Args:
            request_counts: API-specific request count object.

        Returns:
            GenericBatchRequestCounts: Standardized request count object.
        """
        pass

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
            logger.info(f"Loaded existing tracker from {self.batch_objects_file}:\n{self.tracker}")
        else:
            self.tracker = BatchStatusTracker(unsubmitted_request_files=set(request_files))

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
        logger.debug(f"Batch file content size: {file_content_size / (1024*1024):.2f} MB ({file_content_size:,} bytes)")
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
    ):
        """Submit a batch of requests from a file.

        Reads requests from file, converts them to API-specific format,
        and submits them as a batch.

        Args:
            request_file: Path to file containing request data.

        Side Effects:
            - Updates batch submission progress bar
            - Updates tracker with submitted batch status
            - Creates batch metadata with request file path
            - Updates batch objects file
        """
        metadata = {"request_file": request_file}
        requests = self.requests_from_generic_request_file(request_file)
        batch = await self.submit_batch(requests, metadata)
        self.tracker.mark_as_submitted(batch, len(requests))
        await self.update_batch_objects_file()
        self.batch_submit_pbar.update(1)

    def requests_from_generic_request_file(self, request_file: str) -> list[dict]:
        """Read and convert generic requests to API-specific format.

        Reads JSONL formatted generic requests and converts each to the
        API provider's specific request format.

        Args:
            request_file: Path to file containing generic requests in JSONL.

        Returns:
            list[dict]: API-specific request bodies ready for submission.

        Side Effects:
            - Reads from request file on disk
        """
        api_specific_requests = []

        with open(request_file, "r") as file:
            for line in file:
                request = GenericRequest.model_validate_json(line.strip())
                api_specific_request = self.create_api_specific_request_batch(request)
                api_specific_requests.append(api_specific_request)

        return api_specific_requests

    def generic_response_file_from_responses(self, responses: list[dict], batch: GenericBatch) -> str | None:
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
        """
        request_file = batch.request_file
        request_dir = os.path.dirname(request_file)
        request_filename = os.path.basename(request_file)
        response_filename = request_filename.replace("requests_", "responses_")
        response_file = os.path.join(request_dir, response_filename)
        generic_request_map = {}
        # batch_created_at = batch.created_at
        with open(request_file, "r") as f:
            for line in f:
                generic_request = GenericRequest.model_validate_json(line)
                generic_request_map[generic_request.original_row_idx] = generic_request

        with open(response_file, "w") as f:
            for raw_response in responses:
                request_idx = int(raw_response["custom_id"])
                generic_request = generic_request_map[request_idx]
                generic_response = self.parse_api_specific_response(raw_response, generic_request, batch)
                json.dump(generic_response.model_dump(), f, default=str)
                f.write("\n")

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
        self._attempt_loading_batch_status_tracker(request_files)
        if self.tracker.n_submitted_batches > 0:
            n_remaining = self.tracker.n_total_batches - self.tracker.n_downloaded_batches
            n_submitted = self.tracker.n_submitted_batches + self.tracker.n_finished_batches
            logger.info(f"{n_submitted:,} out of {n_remaining:,} remaining batches previously submitted.")
        # exit early
        if self.tracker.n_unsubmitted_request_files == 0:
            return

        # submit remaining batches
        self.batch_submit_pbar = tqdm(
            total=self.tracker.n_total_batches,
            desc="Submitting batches",
            unit="batch",
            initial=self.tracker.n_submitted_finished_or_downloaded_batches,
        )
        tasks = [self.submit_batch_from_request_file(f) for f in self.tracker.unsubmitted_request_files]
        await asyncio.gather(*tasks)
        self.batch_submit_pbar.close()
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

                if batch.status == GenericBatchStatus.FINISHED:
                    logger.debug(f"Batch {batch.id} finished with status: {batch.raw_status}")
                    self.tracker.mark_as_finished(batch)

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
        # progress bar for finished requests
        self.request_pbar = tqdm(
            total=self.tracker.n_total_requests,
            desc="Finished requests in batches",
            unit="request",
            initial=self.tracker.n_finished_or_downloaded_requests,
        )

        # loop until all batches have been returned
        all_response_files = []
        while self.tracker.n_submitted_batches + self.tracker.n_finished_batches > 0:
            # check batch status also updates the tracker
            status_tasks = [self.check_batch_status(batch) for batch in self.tracker.submitted_batches.values()]
            await asyncio.gather(*status_tasks)
            await self.update_batch_objects_file()

            # update progress bari
            self.request_pbar.n = self.tracker.n_finished_or_downloaded_requests
            self.request_pbar.refresh()

            download_tasks = [self.download_batch_to_response_file(batch) for batch in self.tracker.finished_batches.values()]
            # Failed downloads return None and print any errors that occurred
            all_response_files.extend(await asyncio.gather(*download_tasks))
            if self.tracker.n_finished_or_downloaded_requests < self.tracker.n_total_requests:
                logger.debug(
                    f"Batches returned: {self.tracker.n_finished_or_downloaded_batches:,}/{self.tracker.n_total_batches:,} "
                    f"Requests completed: {self.tracker.n_finished_or_downloaded_requests:,}/{self.tracker.n_total_requests:,}"
                )
                logger.debug(f"Sleeping for {self.config.batch_check_interval} seconds...")
                await asyncio.sleep(self.config.batch_check_interval)

        self.request_pbar.close()
        response_files = filter(None, all_response_files)
        if self.tracker.n_downloaded_batches == 0 or not response_files:
            raise RuntimeError("None of the submitted batches completed successfully. " f"Please check the logs above and {self.web_dashboard} for errors.")

    async def download_batch_to_response_file(self, batch: GenericBatch) -> str | None:
        """Download and process completed batch results.

        Downloads batch results, converts responses to generic format, and handles
        cleanup of completed batches including file deletion if configured.

        Args:
            batch: The completed batch object to process.

        Returns:
            str | None: Path to response file if successful, None if failed.

        Side Effects:
            - Downloads batch results from API
            - Creates response file with processed results
            - Updates batch tracking state
            - Updates batch objects file
            - Optionally deletes API provider's batch files
            - Logs download progress and completion
        """
        file_content = await self.download_batch(batch)

        if file_content is None:
            return None

        request_file = batch.request_file
        request_dir = os.path.dirname(request_file)
        request_filename = os.path.basename(request_file)
        response_filename = request_filename.replace("requests_", "responses_")
        response_file = os.path.join(request_dir, response_filename)
        self.generic_response_file_from_responses(file_content, batch)

        logger.debug(f"Batch {batch.id} written to {response_file}")

        if self.config.delete_successful_batch_files:
            await self.delete_file(batch.input_file_id, self.semaphore)
            await self.delete_file(batch.output_file_id, self.semaphore)

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
