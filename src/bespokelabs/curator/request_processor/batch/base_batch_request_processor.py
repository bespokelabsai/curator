import datetime
import json
import logging
import os
import asyncio

from typing import Optional
from tqdm import tqdm
from abc import abstractmethod

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.types.generic_batch import GenericBatch
from bespokelabs.curator.types.generic_batch import GenericBatchRequestCounts
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig


logger = logging.getLogger(__name__)


class BaseBatchRequestProcessor(BaseRequestProcessor):
    """Abstract base class for batch request processors.

    This class provides the core functionality for processing large batches of requests
    through various LLM provider APIs. It handles request submission, status tracking,
    and result processing while maintaining consistent interfaces across providers.

    Attributes:
        config: Configuration object containing batch processing parameters.
        semaphore: Asyncio semaphore for controlling concurrent operations.
        tracker: Status tracker for monitoring batch progress.
        working_dir: Directory for storing intermediate files and results.
        batch_objects_file: Path to file storing batch status information.
    """

    def __init__(self, config: BatchRequestProcessorConfig):
        super().__init__(config)

    def requests_to_responses(self, generic_request_files: list[str]) -> None:
        """Processes a dataset using the provider's batch API.

        This function orchestrates the complete batch processing workflow:
        1. Attempts to load cached results if available
        2. Creates request files from the dataset
        3. Submits and processes batches
        4. Creates output dataset files

        Args:
            generic_request_files: List of paths to files containing generic requests.

        Raises:
            ValueError: If batch_size exceeds provider's maximum requests per batch.
            RuntimeError: If batch processing fails or no successful responses received.
        """
        if self.config.batch_size > self.max_requests_per_batch:
            raise ValueError(
                f"batch_size {self.config.batch_size} is greater than the maximum of "
                f"{self.max_requests_per_batch:,} requests per batch that {self.__class__.__name__} supports. "
                f"Please set your batch_size to be less than or equal to {self.max_requests_per_batch:,}."
            )

        self.semaphore = asyncio.Semaphore(self.max_concurrent_batch_operations)
        self._batch_objects_file_lock = asyncio.Lock()
        self.batch_objects_file = f"{self.working_dir}/batch_objects.jsonl"
        self.batch_submit_pbar: tqdm | None = None
        self.request_pbar: tqdm | None = None

        run_in_event_loop(self.submit_batches_from_request_files(generic_request_files))
        run_in_event_loop(self.poll_and_process_batches())

    def cancel_batches(self) -> Dataset:
        run_in_event_loop(self.cancel_batches())
        logger.warning("Exiting program after batch cancellation.")
        os._exit(1)

    @property
    @abstractmethod
    def max_requests_per_batch(self) -> int:
        """Maximum number of requests allowed in a single batch.

        Returns:
            int: The maximum number of requests per batch supported by the provider.
        """
        pass

    @property
    @abstractmethod
    def max_bytes_per_batch(self) -> int:
        """Maximum size in bytes allowed for a single batch.

        Returns:
            int: The maximum batch size in bytes supported by the provider.
        """
        pass

    @property
    @abstractmethod
    def max_concurrent_batch_operations(self) -> int:
        """Maximum number of concurrent batch operations allowed.

        Returns:
            int: The maximum number of concurrent operations supported.
        """
        pass

    @abstractmethod
    async def submit_batch(
        self, requests: list[dict], metadata: Optional[dict] = None
    ) -> GenericBatch:
        """Submit a batch of requests to the API provider.

        Args:
            requests: List of API-specific request bodies to submit.
            metadata: Optional metadata to include with the batch.

        Returns:
            GenericBatch: The created batch object from the provider.

        Note:
            Implementations must use self.semaphore for concurrency control.
        """
        pass

    @abstractmethod
    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieve current status of a batch from the provider.

        Args:
            batch: The batch object to retrieve status for.

        Returns:
            GenericBatch: Updated batch object with current status.

        Note:
            Implementations must use self.semaphore for concurrency control.
        """
        pass

    @abstractmethod
    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancel a running batch job.

        Args:
            batch: The batch object to cancel.

        Returns:
            GenericBatch: Updated batch object reflecting cancellation status.

        Note:
            Implementations must use self.semaphore for concurrency control.
        """
        pass

    @abstractmethod
    async def download_batch(self, batch: GenericBatch) -> list[dict] | None:
        """Download results of a completed batch.

        Args:
            batch: The batch object to download results for.

        Returns:
            list[dict] | None: List of response dictionaries or None if download fails.
        """
        pass

    @abstractmethod
    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parse provider-specific response into generic format.

        Args:
            raw_response: Raw response dictionary from the provider's API.
            generic_request: Original generic request object.
            batch: The batch object containing timing information.

        Returns:
            GenericResponse: Standardized response object.
        """
        pass

    @abstractmethod
    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Create provider-specific request body from generic request.

        Args:
            generic_request: The generic request object to convert.

        Returns:
            dict: Provider-specific request body.
        """
        pass

    @abstractmethod
    def parse_api_specific_batch_object(
        self, batch: object, request_file: str | None = None
    ) -> GenericBatch:
        pass

    @abstractmethod
    def parse_api_specific_request_counts(
        self, request_counts: object
    ) -> GenericBatchRequestCounts:
        pass

    def _attempt_loading_batch_status_tracker(self, request_files: set[str]):
        if os.path.exists(self.batch_objects_file):
            with open(self.batch_objects_file, "r") as f:
                self.tracker = BatchStatusTracker.model_validate_json(f.read())
            logger.info(f"Loaded existing tracker from {self.batch_objects_file}:\n{self.tracker}")
        else:
            self.tracker = BatchStatusTracker(unsubmitted_request_files=set(request_files))

    async def update_batch_objects_file(self):
        """Updates the batch objects file with the current tracker state."""
        async with self._batch_objects_file_lock:
            with open(self.batch_objects_file, "w") as f:
                f.write(self.tracker.model_dump_json())

    def create_batch_file(self, api_specific_requests: list[dict]) -> str:
        """Creates a batch file from a list of API-specific requests.

        Args:
            api_specific_requests (list[dict]): List of API-specific request bodies.

        Returns:
            str: The encoded file content ready for upload.

        Raises:
            ValueError: If the batch file contains more requests than OpenAI supports.
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
        logger.debug(
            f"Batch file content size: {file_content_size / (1024*1024):.2f} MB ({file_content_size:,} bytes)"
        )
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
        """Submits a batch from a request file.

        Args:
            request_file (str): Path to the file containing requests.

        Side Effects:
            - Updates batch submission progress bar.
            - Updates tracker with submitted batch status.
        """
        metadata = {"request_file": request_file}
        requests = self.requests_from_generic_request_file(request_file)
        batch = await self.submit_batch(requests, metadata)
        self.tracker.mark_as_submitted(batch, len(requests))
        await self.update_batch_objects_file()
        self.batch_submit_pbar.update(1)

    def requests_from_generic_request_file(self, request_file: str) -> list[dict]:
        """Reads and converts generic requests from a file into API-specific request format.

        Args:
            request_file (str): Path to the file containing generic requests in JSONL format.

        Returns:
            list[dict]: List of API-specific request bodies ready for batch submission.
        """
        api_specific_requests = []

        with open(request_file, "r") as file:
            for line in file:
                request = GenericRequest.model_validate_json(line.strip())
                api_specific_request = self.create_api_specific_request_batch(request)
                api_specific_requests.append(api_specific_request)

        return api_specific_requests

    def generic_response_file_from_responses(
        self, responses: list[dict], batch: GenericBatch
    ) -> str | None:
        """Processes API-specific responses and creates a generic response file.

        Takes raw API responses from a batch request and converts them into GenericResponse objects,
        writing them to a response file. Handles both successful and failed responses, including
        token usage tracking and cost calculation.

        Args:
            responses (str): Raw response text from the API containing JSONL formatted responses.
            batch (Batch): The OpenAI batch object containing metadata about the request batch.
            response_file (str): Path where the generic response file should be written.

        Returns:
            str | None: Path to the created response file, or None if creation failed.

        Note:
            The response file will contain one GenericResponse per line in JSONL format.
            Failed requests will have response_message=None and include error details.
            Costs are calculated using litellm with 50% discount applied for batch requests.
        """
        request_file = batch.request_file
        response_file = request_file.replace("requests_", "responses_")
        generic_request_map = {}
        batch_created_at = batch.created_at
        with open(request_file, "r") as f:
            for line in f:
                generic_request = GenericRequest.model_validate_json(line)
                generic_request_map[generic_request.original_row_idx] = generic_request

        with open(response_file, "w") as f:
            for raw_response in responses:
                request_idx = int(raw_response["custom_id"])
                generic_request = generic_request_map[request_idx]
                generic_response = self.parse_api_specific_response(
                    raw_response, generic_request, batch
                )
                json.dump(generic_response.model_dump(), f, default=str)
                f.write("\n")

    async def submit_batches_from_request_files(
        self,
        request_files: set[str],
    ):
        """Manages the submission of multiple request files as batches.

        Args:
            request_files (set[str]): Set of paths to request files to process.

        Side Effects:
            - Updates tracker with batch statuses.
            - Creates and updates batch submission progress bar.
        """
        self._attempt_loading_batch_status_tracker(request_files)
        if self.tracker.n_submitted_batches > 0:
            n_remaining = self.tracker.n_total_batches - self.tracker.n_downloaded_batches
            n_submitted = self.tracker.n_submitted_batches + self.tracker.n_finished_batches
            logger.info(
                f"{n_submitted:,} out of {n_remaining:,} remaining batches previously submitted."
            )
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
        tasks = [
            self.submit_batch_from_request_file(f) for f in self.tracker.unsubmitted_request_files
        ]
        await asyncio.gather(*tasks)
        self.batch_submit_pbar.close()
        assert self.tracker.unsubmitted_request_files == set()

    async def check_batch_status(self, batch: GenericBatch) -> GenericBatch | None:
        """Checks the current status of a batch job.

        Args:
            batch_id (str): The ID of the batch to check.

        Returns:
            Batch | None: The batch object if found, None if not found.

        Side Effects:
            - Updates tracker with current batch status.
            - Updates request completion counts.
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

                if batch.status == "finished":
                    logger.debug(f"Batch {batch.id} finished with status: {batch.raw_status}")
                    self.tracker.mark_as_finished(batch)

    async def poll_and_process_batches(self) -> None:
        """Monitors and processes batches until all are completed.

        Continuously polls the status of submitted batches and downloads their results
        when complete. Handles successful completions, failures, expirations, and
        cancellations. Progress is tracked via a progress bar showing completed requests.

        Returns:
            None

        Raises:
            RuntimeError: If none of the submitted batches complete successfully.

        Side Effects:
            - Updates the batch tracker state.
            - Creates response files for completed batches.
            - Creates and updates requests progress bar.
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
            status_tasks = [
                self.check_batch_status(batch) for batch in self.tracker.submitted_batches.values()
            ]
            await asyncio.gather(*status_tasks)
            await self.update_batch_objects_file()

            # update progress bari
            self.request_pbar.n = self.tracker.n_finished_or_downloaded_requests
            self.request_pbar.refresh()

            download_tasks = [
                self.download_batch_to_response_file(batch)
                for batch in self.tracker.finished_batches.values()
            ]
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
            raise RuntimeError(
                "None of the submitted batches completed successfully. "
                f"Please check the logs above and {self.web_dashboard} for errors."
            )

    async def download_batch_to_response_file(self, batch: GenericBatch) -> str | None:
        """Downloads and processes the results of a completed batch.

        Handles successful completions, failures, and error cases. Converts API-specific
        responses to generic responses and calculates costs.

        Args:
            batch (Batch): The completed batch object to process.

        Returns:
            str | None: Path to the response file if successful, None if batch failed.

        Side Effects:
            - Creates response file with processed results.
            - Updates batch tracking state.
            - Appends batch object to downloaded batch objects file.
            - Optionally deletes batch files from OpenAI.
        """
        file_content = await self.download_batch(batch)

        if file_content is None:
            return None

        request_file = batch.request_file
        response_file = request_file.replace("requests_", "responses_")
        self.generic_response_file_from_responses(file_content, batch)

        logger.debug(f"Batch {batch.id} written to {response_file}")

        if self.config.delete_successful_batch_files:
            await self.delete_file(batch.input_file_id, self.semaphore)
            await self.delete_file(batch.output_file_id, self.semaphore)

        self.tracker.mark_as_downloaded(batch)
        await self.update_batch_objects_file()
        return response_file

    async def cancel_batches(self):
        if self.tracker.n_submitted_batches == 0:
            logger.warning("No batches to be cancelled, but cancel_batches=True.")
            return
        tasks = [self.cancel_batch(batch) for batch in self.tracker.submitted_batches.values()]
        results = await asyncio.gather(*tasks)
