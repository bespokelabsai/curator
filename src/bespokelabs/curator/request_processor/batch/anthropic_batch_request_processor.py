import typing as t

import instructor
import litellm
from anthropic import AsyncAnthropic
from anthropic.types.messages import MessageBatch, MessageBatchRequestCounts
from anthropic.types.shared.not_found_error import NotFoundError
from litellm.litellm_core_utils.core_helpers import map_finish_reason

from bespokelabs.curator.log import logger
from bespokelabs.curator.misc import safe_model_dump
from bespokelabs.curator.request_processor.batch.base_batch_request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


class AnthropicBatchRequestProcessor(BaseBatchRequestProcessor):
    """Anthropic-specific implementation of the BatchRequestProcessor.

    This class handles batch processing of requests using Anthropic's API, including
    file uploads, batch submissions, and result retrieval. Supports message batches
    with limitations defined in Anthropic's documentation.

    Information about limits:
    https://docs.anthropic.com/en/api/creating-message-batches
    https://docs.anthropic.com/en/docs/build-with-claude/message-batches#batch-limitations

    Attributes:
        client: AsyncAnthropic client instance for making API calls.
        web_dashboard: URL to Anthropic's web dashboard for batch monitoring.
    """

    def __init__(self, config: BatchRequestProcessorConfig) -> None:
        """Initialize the AnthropicBatchRequestProcessor."""
        super().__init__(config)
        if self.config.base_url is None:
            self.client = AsyncAnthropic(max_retries=self.config.max_retries)
        else:
            self.client = AsyncAnthropic(max_retries=self.config.max_retries, base_url=self.config.base_url)
        self.web_dashboard = "https://console.anthropic.com/settings/workspaces/default/batches"

    @property
    def backend(self):
        """Backend property."""
        return "anthropic"

    @property
    def max_requests_per_batch(self) -> int:
        """Maximum number of requests allowed in a single Anthropic batch.

        Returns:
            int: The maximum number of requests (100,000) per batch.
        """
        return 100_000

    @property
    def max_bytes_per_batch(self) -> int:
        """Maximum size in bytes allowed for a single Anthropic batch.

        Returns:
            int: The maximum batch size (256 MB) in bytes.
        """
        return 256 * 1024 * 1024  # 256 MB

    @property
    def max_concurrent_batch_operations(self) -> int:
        """Maximum number of concurrent batch operations allowed.

        Set to 100 to avoid hitting undocumented rate limits in the batch operation APIs.

        Returns:
            int: The maximum number of concurrent operations (100).
        """
        return 100

    def parse_api_specific_request_counts(self, request_counts: MessageBatchRequestCounts, request_file: t.Optional[str] = None) -> GenericBatchRequestCounts:
        """Converts Anthropic-specific request counts to generic format.

        Reference implementation:
        https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch_request_counts.py#L9

        Handles the following Anthropic request count statuses:
        - processing: Requests still being processed
        - canceled: Requests that were canceled
        - errored: Requests that failed with errors
        - expired: Requests that timed out
        - succeeded: Successfully completed requests

        Args:
            request_counts: Anthropic's MessageBatchRequestCounts object.
            request_file: Optional path to the request file.

        Returns:
            GenericBatchRequestCounts: Standardized request count format.
        """
        failed = request_counts.canceled + request_counts.errored + request_counts.expired
        succeeded = request_counts.succeeded
        processing = request_counts.processing
        return GenericBatchRequestCounts(
            failed=failed,
            succeeded=succeeded,
            total=processing + succeeded + failed,
            raw_request_counts_object=request_counts.model_dump(),
        )

    def parse_api_specific_batch_object(self, batch: MessageBatch, request_file: str | None = None) -> GenericBatch:
        """Converts an Anthropic batch object to generic format.

        Reference implementations:
        - Batch Status: https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch.py#L53
        - Timing: https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch.py#L20-L51

        Maps Anthropic-specific batch statuses and timing information to our
        standardized GenericBatch format.

        Batch statuses:
        - in_progress: Batch is currently processing
        - canceling: Batch is being canceled
        - ended: Batch has completed (success or failure)

        Timing fields:
        - created_at: When the batch was created
        - cancel_initiated_at: When cancellation was requested
        - archived_at: When the batch was archived
        - ended_at: When processing completed
        - expires_at: When the batch will expire

        Args:
            batch: Anthropic's MessageBatch object.
            request_file: Optional path to the request file.

        Returns:
            GenericBatch: Standardized batch object.

        Raises:
            ValueError: If the batch status is unknown.
        """
        if batch.processing_status in ["cancelling", "in_progress"]:
            status = GenericBatchStatus.SUBMITTED.value
        elif batch.processing_status in ["ended"]:
            status = GenericBatchStatus.FINISHED.value
        else:
            raise ValueError(f"Unknown batch status: {batch.processing_status}")

        return GenericBatch(
            request_file=request_file,
            id=batch.id,
            created_at=batch.created_at,
            finished_at=batch.ended_at,
            status=status,
            api_key_suffix=self.client.api_key[-4:],
            request_counts=self.parse_api_specific_request_counts(batch.request_counts),
            raw_batch=batch.model_dump(),
            raw_status=batch.processing_status,
        )

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Creates an API-specific request body from a generic request.

        Transforms a GenericRequest into the format expected by Anthropic's batch API.
        Combines and constructs a system message with schema and instructions using
        the instructor package for JSON response formatting.

        Args:
            generic_request: The generic request object containing model, messages,
                and optional response format.

        Returns:
            dict: API specific request body formatted for Anthropic's batch API,
                including custom_id and request parameters.
        """
        _, kwargs = instructor.handle_response_model(
            self.prompt_formatter.response_format,  # Use the object instead of the dict
            mode=instructor.Mode.ANTHROPIC_JSON,
            messages=generic_request.messages,
        )

        return {
            "custom_id": str(generic_request.original_row_idx),
            "params": {
                "model": generic_request.model,
                "max_tokens": litellm.get_max_tokens(self.config.model),
                **kwargs,  # contains 'system' and 'messages'
                **generic_request.generation_params,  # contains 'temperature', 'top_p', etc.
            },
        }

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parses an Anthropic API response into generic format.

        Processes the raw response from Anthropic's batch API, handling both
        successful and failed responses, including token usage and cost calculation.
        For batch requests, a 50% discount is applied to the cost.

        Args:
            raw_response: Raw response dictionary from Anthropic's API.
            generic_request: Original generic request object.
            batch: The batch object containing timing information.

        Returns:
            GenericResponse: Standardized response object with parsed message,
                errors, token usage, and cost information.
        """
        result_type = raw_response["result"]["type"]

        token_usage = None
        cost = None
        response_message = None
        finish_reason = "unknown"

        if result_type == "succeeded":
            response_body = raw_response["result"]["message"]
            if self.config.return_completions_object:
                response_message_raw = response_body
            else:
                content_blocks = response_body.get("content", [])
                if content_blocks:
                    for block in content_blocks:
                        if block.get("type") == "text":
                            response_message_raw = block.get("text", "")
                            break
                    else:
                        response_message_raw = str(content_blocks)
                else:
                    response_message_raw = ""

            # Get stop reason
            finish_reason = response_body.get("stop_reason", "unknown")

            usage = response_body.get("usage", {})

            token_usage = _TokenUsage(
                input=usage.get("input_tokens", 0),
                output=usage.get("output_tokens", 0),
            )
            response_message, response_errors = self.prompt_formatter.parse_response_message(response_message_raw)

            all_text_response = ""
            for msg in response_body["content"]:
                all_text_response += msg.get("text") or ""  # in case this is None
                all_text_response += msg.get("thinking") or ""  # in case this is None

            # TODO we should directly use the token counts returned by anthropic in the response...
            cost = self._cost_processor.cost(model=self.config.model, prompt=str(generic_request.messages), completion=all_text_response)

        elif result_type == "errored":
            error = raw_response["result"]["error"]
            logger.warning(f"custom_id {raw_response['custom_id']} result was '{result_type}' with error '{error}'")
            response_errors = [str(error)]
        elif result_type == "expired" or result_type == "canceled":
            logger.warning(f"custom_id {raw_response['custom_id']} result was '{result_type}'")
            response_errors = [f"Request {result_type}"]
        else:
            raise ValueError(f"Unknown result type: {result_type}")

        finish_reason = map_finish_reason(finish_reason)
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
            finish_reason=finish_reason,
        )

    async def submit_batch(self, requests: list[dict], metadata: dict) -> GenericBatch:
        """Handles the complete batch submission process.

        Args:
            requests (list[dict]): List of API-specific requests to submit
            metadata (dict): Metadata to be included with the batch

        Returns:
            Batch: The created batch object from OpenAI

        Side Effects:
            - Updates tracker with submitted batch status
        """
        async with self.semaphore:
            batch = await self.client.messages.batches.create(requests=requests)
            return self.parse_api_specific_batch_object(batch, request_file=metadata["request_file"])

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieves the current status of a batch from Anthropic's API.

        Args:
            batch: The batch object to retrieve status for.

        Returns:
            GenericBatch: Updated batch object with current status.
            None: If the batch is not found or inaccessible.

        Side Effects:
            - Logs warnings if batch is not found or inaccessible.
        """
        async with self.semaphore:
            try:
                batch = await self.client.messages.batches.retrieve(batch.id)
            except NotFoundError:
                logger.warning(f"batch object {batch.id} not found. Your API key (***{self.client.api_key[-4:]}) might not have access to this batch.")
                return None
            request_file = self.tracker.submitted_batches[batch.id].request_file
            return self.parse_api_specific_batch_object(batch, request_file=request_file)

    async def download_batch(self, batch: GenericBatch) -> list[dict] | None:
        """Downloads the results of a completed batch.

        Args:
            batch: The batch object to download results for.

        Returns:
            list[dict]: List of response dictionaries.
            None: If download fails.

        Side Effects:
            - Streams results from Anthropic's API.
            - Converts each result to a dictionary format.
        """
        async with self.semaphore:
            MessageBatch.model_validate(batch.raw_batch)
            responses = []
            results_stream = await self.client.messages.batches.results(batch.id)
            async for result in results_stream:
                responses.append(safe_model_dump(result))
            return responses

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancels a running batch job.

        Args:
            batch: The batch object to cancel.

        Returns:
            GenericBatch: Updated batch object after cancellation attempt.

        Side Effects:
            - Attempts to cancel the batch with Anthropic's API.
            - Logs success or failure of cancellation.
            - Updates batch status in generic format.

        Note:
            Cannot cancel already ended batches.
        """
        async with self.semaphore:
            request_file = self.tracker.submitted_batches[batch.id].request_file
            batch_object = await self.retrieve_batch(batch)
            if batch_object.status == "ended":
                logger.warning(f"Batch {batch.id} is already ended, cannot cancel")
                return self.parse_api_specific_batch_object(batch_object, request_file=request_file)
            try:
                await self.client.messages.batches.cancel(batch.id)
                logger.info(f"Successfully cancelled batch: {batch.id}")
                return self.parse_api_specific_batch_object(batch_object, request_file=request_file)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to cancel batch {batch.id}: {error_msg}")
                return self.parse_api_specific_batch_object(batch_object, request_file=request_file)
