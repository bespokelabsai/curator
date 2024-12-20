import logging
import litellm
from litellm import get_max_tokens

from anthropic import AsyncAnthropic
from anthropic.types.messages import MessageBatch
from anthropic.types.messages import MessageBatchRequestCounts
from anthropic.types.shared.not_found_error import NotFoundError

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.types.token_usage import TokenUsage
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig

logger = logging.getLogger(__name__)


class AnthropicBatchRequestProcessor(BaseBatchRequestProcessor):
    """Anthropic-specific implementation of the BatchRequestProcessor.

    This class handles batch processing of requests using Anthropic's API, including
    submitting batches, monitoring their status, and retrieving results. For batch
    limitations and details, see:
    https://docs.anthropic.com/en/api/creating-message-batches
    https://docs.anthropic.com/en/docs/build-with-claude/message-batches#batch-limitations

    Attributes:
        client: AsyncAnthropic client instance for making API calls
        web_dashboard: URL to Anthropic's web dashboard for batch monitoring
    """

    def __init__(self, config: BatchRequestProcessorConfig) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(max_retries=self.config.max_retries)
        self.web_dashboard = "https://console.anthropic.com/settings/workspaces/default/batches"

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

        Returns:
            int: The maximum number of concurrent operations (100).
        """
        return 100

    def parse_api_specific_request_counts(
        self, request_counts: MessageBatchRequestCounts
    ) -> GenericBatchRequestCounts:
        """Converts Anthropic-specific request counts to generic format.

        Anthropic request counts include: "processing", "canceled", "errored",
        "expired", "succeeded". These are mapped to our generic format.

        Args:
            request_counts: Anthropic's MessageBatchRequestCounts object.

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

    def parse_api_specific_batch_object(
        self, batch: MessageBatch, request_file: str | None = None
    ) -> GenericBatch:
        """Converts an Anthropic batch object to generic format.

        Handles status mapping and timing information from Anthropic's format
        to our standardized GenericBatch format.

        Args:
            batch: Anthropic's MessageBatch object.
            request_file: Optional path to the request file.

        Returns:
            GenericBatch: Standardized batch object.

        Raises:
            ValueError: If the batch status is unknown.
        """
        if batch.processing_status in ["cancelling", "in_progress"]:
            status = "submitted"
        elif batch.processing_status in ["ended"]:
            status = "finished"
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
        Handles message formatting and parameter configuration.

        Args:
            generic_request: The generic request object containing model, messages,
                and optional response format.

        Returns:
            dict: API specific request body formatted for Anthropic's batch API.

        Raises:
            NotImplementedError: If response_format is specified (not yet supported).
        """
        if generic_request.response_format:
            raise NotImplementedError("response_format is not yet supported for Anthropic")

        params = {"model": generic_request.model, "max_tokens": get_max_tokens(self.config.model)}
        if generic_request.messages[0]["role"] == "system":
            params["system"] = generic_request.messages[0]["content"]
            params["messages"] = generic_request.messages[1:]
        else:
            params["messages"] = generic_request.messages

        for key, value in generic_request.generation_params.items():
            params[key] = value

        request = {
            "custom_id": str(generic_request.original_row_idx),
            "params": params,
        }

        return request

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parses an Anthropic API response into generic format.

        Processes the raw response from Anthropic's batch API, handling both
        successful and failed responses, including token usage and cost calculation.

        Args:
            raw_response: Raw response dictionary from Anthropic's API.
            generic_request: Original generic request object.
            batch: The batch object containing timing information.

        Returns:
            GenericResponse: Standardized response object with parsed message,
                errors, token usage, and cost information.
        """
        result_type = raw_response["result"]["type"]
        if result_type != "succeeded":
            error = raw_response["result"]["error"]
            logger.warning(
                f"custom_id {raw_response['custom_id']} result was '{result_type}' with error '{error}'"
            )
            response_message = None
            response_errors = [str(error)]
            token_usage = None
            cost = None
        else:
            response_body = raw_response["result"]["message"]
            response_message_raw = response_body["content"][0]["text"]
            stop_reason = response_body["stop_reason"]
            stop_sequence = response_body["stop_sequence"]
            usage = response_body.get("usage", {})

            token_usage = TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            )
            response_message, response_errors = self.prompt_formatter.parse_response_message(
                response_message_raw
            )

            cost = litellm.completion_cost(
                model=self.config.model,
                prompt=str(generic_request.messages),
                completion=response_message,
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

    async def submit_batch(self, requests: list[dict], metadata: dict) -> GenericBatch:
        """Submits a batch of requests to Anthropic's API.

        Args:
            requests: List of API-specific requests to submit.
            metadata: Metadata to be included with the batch.

        Returns:
            GenericBatch: The created batch object.

        Side Effects:
            - Updates tracker with submitted batch status.
        """
        async with self.semaphore:
            batch = await self.client.messages.batches.create(requests=requests)
            return self.parse_api_specific_batch_object(
                batch, request_file=metadata["request_file"]
            )

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieves the current status of a batch from Anthropic's API.

        Args:
            batch: The batch object to retrieve status for.

        Returns:
            GenericBatch: Updated batch object with current status.
            None: If the batch is not found or inaccessible.

        Note:
            Uses API key suffix to help identify access issues.
        """
        async with self.semaphore:
            try:
                batch = await self.client.messages.batches.retrieve(batch.id)
            except NotFoundError:
                logger.warning(
                    f"batch object {batch.id} not found. "
                    f"Your API key (***{self.client.api_key[-4:]}) might not have access to this batch."
                )
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
        """
        async with self.semaphore:
            anthropic_batch = MessageBatch.model_validate(batch.raw_batch)
            responses = []
            results_stream = await self.client.messages.batches.results(batch.id)
            async for result in results_stream:
                responses.append(result.model_dump())
            return responses

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancels a running batch job.

        Args:
            batch: The batch object to cancel.

        Returns:
            GenericBatch: Updated batch object reflecting cancellation status.

        Note:
            Cannot cancel already completed batches.
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
