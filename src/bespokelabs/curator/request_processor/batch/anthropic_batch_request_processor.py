import datetime
import logging
import litellm

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
    """
    Information about limits:
    https://docs.anthropic.com/en/api/creating-message-batches
    https://docs.anthropic.com/en/docs/build-with-claude/message-batches#batch-limitations
    """

    def __init__(self, config: BatchRequestProcessorConfig) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(max_retries=self.config.max_retries_per_operation)

    @property
    def max_requests_per_batch(self) -> int:
        return 100_000

    @property
    def max_bytes_per_batch(self) -> int:
        return 256 * 1024 * 1024  # 256 MB

    @property
    def max_concurrent_batch_operations(self) -> int:
        return 100

    def parse_api_specific_request_counts(
        self, request_counts: MessageBatchRequestCounts
    ) -> GenericBatchRequestCounts:
        """
        https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch_request_counts.py#L9
        Request Counts (Anthropic): "processing", "canceled", "errored", "expired", "succeeded"
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
        """
        https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch.py#L53
        Batch Status (Anthropic): "in_progress", "canceling", "ended"

        https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch.py#L20-L51
        Timing (Anthropic): "created_at", "cancel_initiated_at", "archived_at", "ended_at", "expires_at"
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
        if generic_request.response_format:
            # TODO(Ryan) how can we support this the way litellm does?
            raise NotImplementedError("response_format is not yet supported for Anthropic")

        params = {
            "model": generic_request.model,
        }
        if generic_request.messages[0]["role"] == "system":
            params["system"] = generic_request.messages[0]["content"]
            params["messages"] = generic_request.messages[1:]
        else:
            params["messages"] = generic_request.messages

        for key, value in generic_request.generation_params.items():
            if key in self.supported_params:
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
        batch_created_at: datetime.datetime,
    ) -> GenericResponse:
        if raw_response["result"]["type"] != "succeeded":
            response_message = None
            response_errors = [
                raw_response["result"]["type"]
            ]  # no examples of a failed response, we can probably include more information here
            token_usage = None
            cost = None
        else:
            response_body = raw_response["response"]["body"]
            response_message_raw = response_body["choices"][0]["message"]["content"]
            usage = response_body.get("usage", {})

            token_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            response_message, response_errors = self.prompt_formatter.parse_response_message(
                response_message_raw, self.prompt_formatter.response_format
            )

            cost = litellm.completion_cost(
                model=self.config.model,
                prompt=str(self.generic_request.messages),
                completion=response_message,
            )
            cost *= 0.5  # 50% off for batch

        return GenericResponse(
            response_message=response_message,
            response_errors=response_errors,
            raw_response=raw_response,
            raw_request=None,
            generic_request=generic_request,
            created_at=batch_created_at,
            finished_at=datetime.datetime.now(),
            token_usage=token_usage,
            response_cost=cost,
        )

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
            batch = await self.client.messages.batches.create(requests=requests)
            return self.parse_api_specific_batch_object(
                batch, request_file=metadata["request_file"]
            )

    async def retrieve_batch(self, batch_id: str) -> GenericBatch:
        async with self.semaphore:
            try:
                batch = await self.client.messages.batches.retrieve(batch_id)
            except NotFoundError:
                logger.warning(
                    f"batch object {batch_id} not found. "
                    f"Your API key (***{self.client.api_key[-4:]}) might not have access to this batch."
                )
                return None

            request_file = self.tracker.submitted_batches[batch_id].request_file
            return self.parse_api_specific_batch_object(batch, request_file=request_file)

    async def download_batch(self, batch: GenericBatch) -> list[dict] | None:
        async with self.semaphore:
            anthropic_batch = MessageBatch.model_validate(batch.raw_batch)
            responses = []
            async for result in self.client.messages.batches.results(batch.id):
                responses.append(result.model_dump())
            return responses

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
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
