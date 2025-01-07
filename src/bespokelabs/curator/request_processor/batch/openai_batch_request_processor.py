import asyncio
import datetime
import json
import logging
import litellm

from typing import Optional
from openai import AsyncOpenAI, NotFoundError, APIConnectionError
from openai.types.batch import Batch
from openai.types.file_object import FileObject
from openai.types.batch_request_counts import BatchRequestCounts

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.types.token_usage import TokenUsage
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.generic_batch import (
    GenericBatch,
    GenericBatchRequestCounts,
    GenericBatchStatus,
)
from bespokelabs.curator.request_processor.openai_request_mixin import OpenAIRequestMixin
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig

logger = logging.getLogger(__name__)


class OpenAIBatchRequestProcessor(BaseBatchRequestProcessor, OpenAIRequestMixin):
    def __init__(self, config: BatchRequestProcessorConfig) -> None:
        super().__init__(config)
        if self.config.base_url is None:
            self.client = AsyncOpenAI(max_retries=self.config.max_retries)
        else:
            self.client = AsyncOpenAI(
                max_retries=self.config.max_retries, base_url=self.config.base_url
            )
        self.web_dashboard = "https://platform.openai.com/batches"

    @property
    def max_requests_per_batch(self) -> int:
        return 50_000

    @property
    def max_bytes_per_batch(self) -> int:
        return 200 * 1024 * 1024  # 200 MB

    @property
    def max_concurrent_batch_operations(self) -> int:
        return 100

    def parse_api_specific_request_counts(
        self, request_counts: BatchRequestCounts
    ) -> GenericBatchRequestCounts:
        """Convert OpenAI-specific request counts to generic format.

        Handles the following OpenAI request count statuses:
        - completed: Successfully completed requests
        - failed: Requests that failed
        - total: Total number of requests in batch

        Args:
            request_counts: OpenAI's BatchRequestCounts object.

        Returns:
            GenericBatchRequestCounts: Standardized request count format.
        """
        return GenericBatchRequestCounts(
            failed=request_counts.failed,
            succeeded=request_counts.completed,
            total=request_counts.total,
            raw_request_counts_object=request_counts.model_dump(),
        )

    def parse_api_specific_batch_object(
        self, batch: Batch, request_file: str | None = None
    ) -> GenericBatch:
        """Convert an OpenAI batch object to generic format.

        Maps OpenAI-specific batch statuses and timing information to our
        standardized GenericBatch format.

        Batch statuses:
        - validating/finalizing/cancelling/in_progress: Mapped to SUBMITTED
        - completed/failed/expired/cancelled: Mapped to FINISHED

        Timing fields:
        - created_at: When the batch was created
        - completed_at/failed_at/expired_at/cancelled_at: When processing ended

        Args:
            batch: OpenAI's Batch object.
            request_file: Optional path to the request file.

        Returns:
            GenericBatch: Standardized batch object.

        Raises:
            ValueError: If the batch status is unknown.
        """
        if batch.status in ["validating", "finalizing", "cancelling", "in_progress"]:
            status = GenericBatchStatus.SUBMITTED
        elif batch.status in ["completed", "failed", "expired", "cancelled"]:
            status = GenericBatchStatus.FINISHED
        else:
            raise ValueError(f"Unknown batch status: {batch.status}")

        finished_at = (
            batch.completed_at or batch.failed_at or batch.expired_at or batch.cancelled_at
        )

        return GenericBatch(
            request_file=batch.metadata["request_file"],
            id=batch.id,
            created_at=batch.created_at,
            finished_at=finished_at,
            status=status,
            api_key_suffix=self.client.api_key[-4:],
            request_counts=self.parse_api_specific_request_counts(batch.request_counts),
            raw_batch=batch.model_dump(),
            raw_status=batch.status,
        )

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parse OpenAI API response into generic format.

        Processes raw responses from OpenAI's batch API, handling both successful
        and failed responses. For successful responses, calculates token usage
        and applies batch pricing discount.

        Args:
            raw_response: Raw response dictionary from OpenAI's API.
            generic_request: Original generic request object.
            batch: The batch object containing timing information.

        Returns:
            GenericResponse: Standardized response object with parsed message,
                errors, token usage, and cost information.

        Side Effects:
            - Calculates costs with 50% batch discount
            - Parses response messages using prompt formatter
            - Handles failed requests with error details
        """
        if raw_response["response"]["status_code"] != 200:
            response_message = None
            response_errors = [str(raw_response["response"]["status_code"])]
            token_usage = None
            cost = None
        else:
            response_body = raw_response["response"]["body"]
            response_message_raw = response_body["choices"][0]["message"]["content"]
            usage = response_body.get("usage", {})

            # TODO(Ryan) will want to resubmit requests like in the online case
            # if we get length?
            # can use litellm and my pr https://github.com/BerriAI/litellm/pull/7264
            # resubmission also related to the expiration
            token_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            response_message, response_errors = self.prompt_formatter.parse_response_message(
                response_message_raw
            )

            cost = litellm.completion_cost(
                model=self.config.model,
                prompt=str(generic_request.messages),
                completion=response_message_raw,
            )
            cost *= 0.5  # 50% off for batch

        return GenericResponse(
            response_message=response_message,
            response_errors=response_errors,
            raw_response=raw_response,
            raw_request=None,
            generic_request=generic_request,
            created_at=batch.created_at,
            finished_at=batch.finished_at,
            token_usage=token_usage,
            response_cost=cost,
        )

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """
        Creates an API-specific request body from a generic request body.

        This function transforms a GenericRequest into the format expected by OpenAI's batch API.
        It handles both standard requests and those with JSON schema response formats.

        Args:
            generic_request (GenericRequest): The generic request object containing model, messages,
                and optional response format.

        Returns:
            dict: API specific request body formatted for OpenAI's batch API, including:
                - custom_id: String identifier from the original row index
                - method: Always "POST"
                - url: OpenAI chat completions endpoint
                - body: Request parameters including model, messages, and optional formatting
        """
        request = {
            "custom_id": str(generic_request.original_row_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": self.create_api_specific_request_online(generic_request),
        }

        return request

    async def upload_batch_file(self, file_content: bytes, num_retries: int = 3) -> FileObject:
        """
        Uploads a batch file to OpenAI and waits until ready.

        Args:
            file_content (bytes): The encoded file content to upload

        Returns:
            str: The uploaded file object from OpenAI
        """
        try:
            for retry_id in range(num_retries + 1):
                try:
                    batch_file_upload = await self.client.files.create(
                        file=file_content, purpose="batch"
                    )
                    break
                except APIConnectionError as e:
                    logger.error(
                        f"Failed to connect to OpenAI API. Please check here for more information: https://help.openai.com/en/articles/6897191-apiconnectionerror."
                    )
                    if retry_id == num_retries:
                        raise e
                    else:
                        logger.error(
                            f"Retrying batch file upload (attempt {retry_id+1}/{num_retries})"
                        )
        except Exception as e:
            logger.error(f"Error uploading batch file: {e}")
            raise e

        # When submitting a file, sometimes the file is not ready immediately for status checking
        # Which results in a file not found error, so we briefly pause before checking the status
        await asyncio.sleep(1)

        try:
            batch_file_upload = await self.client.files.wait_for_processing(batch_file_upload.id)
        except Exception as e:
            logger.error(f"Error waiting for batch file to be processed: {e}")
            raise e

        logger.debug(f"File uploaded with id {batch_file_upload.id}")

        return batch_file_upload

    async def create_batch(self, batch_file_id: str, metadata: dict) -> Batch:
        """
        Creates a batch job with OpenAI using an uploaded file.

        Args:
            batch_file_id (str): ID of the uploaded file to use for the batch
            metadata (dict): Metadata to be included with the batch

        Returns:
            Batch: The created batch object from OpenAI

        Raises:
            Exception: If batch creation fails
        """
        try:
            batch = await self.client.batches.create(
                input_file_id=batch_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata,
            )
            logger.debug(f"Batch submitted with id {batch.id}")
        except Exception as e:
            logger.error(f"Error submitting batch: {e}")
            raise e
        return batch

    async def submit_batch(self, requests: list[dict], metadata: dict) -> GenericBatch:
        """
        Handles the complete batch submission process.

        Args:
            requests (list[dict]): List of API-specific requests to submit
            metadata (dict): Metadata to be included with the batch

        Returns:
            Batch: The created batch object from OpenAI

        Side Effects:
            - Updates tracker with submitted batch status
        """
        async with self.semaphore:
            file_content = self.create_batch_file(requests)
            batch_file_upload = await self.upload_batch_file(file_content)
            batch = await self.create_batch(batch_file_upload.id, metadata)
            return self.parse_api_specific_batch_object(batch)

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieve current status of a batch from OpenAI's API.

        Args:
            batch: The batch object to retrieve status for.

        Returns:
            GenericBatch: Updated batch object with current status.
            None: If the batch is not found or inaccessible.

        Side Effects:
            - Logs warnings if batch is not found or inaccessible
            - Uses API key suffix to help identify access issues
        """
        try:
            batch = await self.client.batches.retrieve(batch.id)
        except NotFoundError:
            logger.warning(
                f"batch object {batch.id} not found. "
                f"Your API key (***{self.client.api_key[-4:]}) might not have access to this batch."
            )
            return None
        return self.parse_api_specific_batch_object(batch)

    async def delete_file(self, file_id: str, semaphore: asyncio.Semaphore):
        """
        Deletes a file from OpenAI's storage.

        Args:
            file_id (str): The ID of the file to delete
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent operations
        """
        async with semaphore:
            try:
                delete_response = await self.client.files.delete(file_id)
                if delete_response.deleted:
                    logger.debug(f"Deleted file {file_id}")
                else:
                    logger.warning(f"Failed to delete file {file_id}")
            except NotFoundError:
                # This is fine, the file may have been deleted already. Deletion should be best-effort.
                logger.warning(f"Trying to delete file {file_id} but it was not found.")

    async def download_batch(self, batch: GenericBatch) -> list[dict] | None:
        """Download and process batch results from OpenAI.

        Downloads output and error files for completed batches. Handles different
        batch statuses (completed, failed, cancelled, expired) and manages file
        cleanup based on configuration.

        Args:
            batch: The batch object to download results for.

        Returns:
            list[dict] | None: List of response dictionaries if successful,
                None if download fails or batch has no output.

        Side Effects:
            - Downloads files from OpenAI's API
            - Optionally deletes batch files based on configuration
            - Logs batch status and any errors
            - Handles file cleanup for failed/cancelled/expired batches
        """
        output_file_content = None
        error_file_content = None  # TODO how should we use this?
        openai_batch = Batch.model_validate(batch.raw_batch)
        async with self.semaphore:
            # Completed batches have an output file
            if openai_batch.output_file_id:
                output_file_content = await self.client.files.content(openai_batch.output_file_id)
            if openai_batch.error_file_id:
                error_file_content = await self.client.files.content(openai_batch.error_file_id)

            if openai_batch.status == "completed" and openai_batch.output_file_id:
                logger.debug(f"Batch {batch.id} completed and downloaded")
                if self.config.delete_successful_batch_files:
                    await self.delete_file(openai_batch.input_file_id, self.semaphore)
                    await self.delete_file(openai_batch.output_file_id, self.semaphore)

            # Failed batches with an error file
            elif openai_batch.status == "failed" and openai_batch.error_file_id:
                logger.warning(f"Batch {batch.id} failed\n. Errors will be parsed below.")
                if self.config.delete_failed_batch_files:
                    await self.delete_file(openai_batch.input_file_id, self.semaphore)
                    await self.delete_file(openai_batch.error_file_id, self.semaphore)

            # Failed batches without an error file
            elif openai_batch.status == "failed" and not openai_batch.error_file_id:
                errors = "\n".join([str(error) for error in openai_batch.errors.data])
                logger.error(
                    f"Batch {batch.id} failed and likely failed validation. "
                    f"Batch errors: {errors}. "
                    f"Check https://platform.openai.com/batches/{batch.id} for more details."
                )
                if self.config.delete_failed_batch_files:
                    await self.delete_file(openai_batch.input_file_id, self.semaphore)

            # Cancelled or expired batches
            elif openai_batch.status == "cancelled" or openai_batch.status == "expired":
                logger.warning(f"Batch {batch.id} was cancelled or expired")
                if self.config.delete_successful_batch_files:
                    await self.delete_file(openai_batch.input_file_id, self.semaphore)
                    await self.delete_file(openai_batch.output_file_id, self.semaphore)

        responses = []
        if output_file_content:
            for line in output_file_content.text.splitlines():
                raw_response = json.loads(line)
                responses.append(raw_response)
        return responses

    async def cancel_batch(self, batch: GenericBatch) -> int:
        """Cancel a running batch job.

        Attempts to cancel a batch that hasn't completed yet. Handles cases
        where the batch is already completed or cancellation fails.

        Args:
            batch: The batch object to cancel.

        Returns:
            int: 0 if cancellation succeeds or batch is already complete,
                -1 if cancellation fails.

        Side Effects:
            - Attempts to cancel batch with OpenAI's API
            - Logs success or failure of cancellation
            - Retrieves current batch status before attempting cancellation
        """
        async with self.semaphore:
            batch_object = await self.retrieve_batch(batch)
            if batch_object.status == "completed":
                logger.warning(f"Batch {batch.id} is already completed, cannot cancel")
                return 0
            try:
                await self.client.batches.cancel(batch.id)
                logger.info(f"Successfully cancelled batch: {batch.id}")
                return 0
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to cancel batch {batch.id}: {error_msg}")
                return -1
