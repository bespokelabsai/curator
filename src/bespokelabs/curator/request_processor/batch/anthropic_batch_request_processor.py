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

logger = logging.getLogger(__name__)


class AnthropicBatchRequestProcessor(BaseBatchRequestProcessor):
    """Handles batch request processing for Anthropic's Claude models.

    This class implements the batch processing interface for Anthropic's API, handling
    request batching, status tracking, and response processing. It supports Claude models
    and manages batch operations within Anthropic's API limits.

    Attributes:
        working_dir (str): Directory for storing batch-related files
        check_interval (int): Time between batch status checks in seconds
        prompt_formatter (PromptFormatter): Formatter for processing prompts
        delete_successful_batch_files (bool): Whether to delete successful batch files
        delete_failed_batch_files (bool): Whether to delete failed batch files
        max_retries (int): Maximum number of retry attempts

    References:
        https://docs.anthropic.com/en/api/creating-message-batches
        https://docs.anthropic.com/en/docs/build-with-claude/message-batches#batch-limitations
    """

    def __init__(
        self,
        working_dir: str,
        check_interval: int = 60,
        prompt_formatter: PromptFormatter | None = None,
        delete_successful_batch_files: bool = False,
        delete_failed_batch_files: bool = False,
        max_retries: int | None = None,
    ) -> None:
        """Initialize BatchManager to handle Anthropic batch processing operations.

        Args:
            working_dir (str): Directory for storing batch-related files including requests, responses,
                and tracking files.
            check_interval (int): Time interval (in seconds) between batch status checks.
            delete_successful_batch_files (bool): Whether to delete input/output files
                after successful batch completion.
            delete_failed_batch_files (bool): Whether to delete input/error files
                after batch failure.
        """
        super().__init__(
            working_dir=working_dir,
            check_interval=check_interval,
            prompt_formatter=prompt_formatter,
            delete_successful_batch_files=delete_successful_batch_files,
            delete_failed_batch_files=delete_failed_batch_files,
            max_retries=max_retries,
        )
        self.client = AsyncAnthropic(max_retries=self.max_retries_per_operation)

    @property
    def max_requests_per_batch(self) -> int:
        """Maximum number of requests allowed in a single batch.

        Returns:
            int: Maximum of 100,000 requests per batch for Anthropic API
        """
        return 100_000

    @property
    def max_bytes_per_batch(self) -> int:
        """Maximum size in bytes allowed for a single batch.

        Returns:
            int: Maximum of 256MB per batch for Anthropic API
        """
        return 256 * 1024 * 1024  # 256 MB

    @property
    def max_concurrent_batch_operations(self) -> int:
        """Maximum number of concurrent batch operations allowed.

        Returns:
            int: Maximum of 100 concurrent operations for Anthropic API
        """
        return 100

    def parse_api_specific_request_counts(
        self, request_counts: MessageBatchRequestCounts
    ) -> GenericBatchRequestCounts:
        """Convert Anthropic-specific request counts to generic format.

        Args:
            request_counts (MessageBatchRequestCounts): Anthropic's request count object
                containing counts for processing, canceled, errored, expired, and
                succeeded requests

        Returns:
            GenericBatchRequestCounts: Standardized request counts containing:
                - failed: sum of canceled, errored, and expired requests
                - succeeded: count of successful requests
                - total: sum of processing, succeeded, and failed requests
                - raw_request_counts_object: original Anthropic counts as dict

        References:
            https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/beta/messages/beta_message_batch_request_counts.py
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
        """Convert Anthropic-specific batch object to generic format.

        Args:
            batch (MessageBatch): Anthropic batch object containing status, timing,
                and request count information
            request_file (str | None): Path to the file containing batch requests

        Returns:
            GenericBatch: Standardized batch object with mapped status:
                - "submitted" for "in_progress" or "cancelling" status
                - "finished" for "ended" status
                Also includes:
                - request_counts: Standardized request count information
                - timing information: created_at, finished_at
                - raw_batch: Original Anthropic batch object as dict

        Raises:
            ValueError: If batch has unknown processing_status

        References:
            https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/beta/messages/beta_message_batch.py
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
        """Create an Anthropic-specific batch request from a generic request.

        Args:
            generic_request (GenericRequest): Generic request object containing model,
                messages, and generation parameters

        Returns:
            dict: Anthropic-specific request format with custom_id and params

        Raises:
            NotImplementedError: If response_format is specified (not yet supported)
        """
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
        """Parse Anthropic-specific response into generic response format.

        Args:
            raw_response (dict): Raw response from Anthropic API
            generic_request (GenericRequest): Original request that generated this response
            batch_created_at (datetime.datetime): When the batch was created

        Returns:
            GenericResponse: Standardized response object containing message, errors,
                token usage, and cost information

        Note:
            - Failed responses will have None for response_message and token_usage
            - Batch requests receive a 50% discount on cost
        """
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
            # Anthropic responses don't need parsing, use raw content directly
            response_message = response_message_raw
            response_errors = []

            cost = litellm.completion_cost(
                model=self.model,
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
        """Submit a batch of requests to Anthropic's API.

        Args:
            requests (list[dict]): List of Anthropic-specific request objects, each
                containing custom_id and params for message generation
            metadata (dict): Additional metadata for batch tracking, including
                request_file path

        Returns:
            GenericBatch: Standardized batch object containing:
                - batch ID and status
                - request counts and timing information
                - file paths and API key information

        Side Effects:
            - Creates batch request in Anthropic's system
            - Updates local batch tracking with metadata
        """
        async with self.semaphore:
            batch = await self.client.messages.batches.create(requests=requests)
            return self.parse_api_specific_batch_object(
                batch, request_file=metadata["request_file"]
            )

    async def retrieve_batch(self, batch_id: str) -> GenericBatch:
        """Retrieve a batch by ID from Anthropic's API.

        Args:
            batch_id (str): ID of the batch to retrieve

        Returns:
            GenericBatch: Retrieved batch information in generic format, or None if not found

        Side Effects:
            - Logs a warning if batch is not found or API key lacks access
        """
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
        """Download all results from a completed Anthropic batch.

        Args:
            batch (GenericBatch): Batch object containing ID and metadata

        Returns:
            list[dict] | None: List of raw response objects from Anthropic API,
                each containing result type, response body, and usage information.
                Returns None if batch is not found or not completed.

        Side Effects:
            - Streams results from Anthropic's API
            - Logs warnings for missing or incomplete batches
        """
        anthropic_batch = MessageBatch.model_validate(batch.raw_batch)
        responses = []
        async for result in self.client.messages.batches.results(batch.id):
            responses.append(result.model_dump())
        return responses
