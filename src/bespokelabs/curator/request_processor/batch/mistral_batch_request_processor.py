import typing as t

from mistralai import Mistral

from bespokelabs.curator.request_processor.batch.base_batch_request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse

# Reference for Mistral status: https://github.com/mistralai/client-python/blob/main/docs/models/batchjobstatus.md
_PROGRESS_STATE = {"QUEUED", "RUNNING", "CANCELLATION_REQUESTED"}
_FINISHED_STATE = {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}


class MistralBatchRequestProcessor(BaseBatchRequestProcessor):
    """Mistral-specific implementation of the BatchRequestProcessor.

    This class handles batch processing of requests using Mistral's API, including
    file uploads, batch submissions, and result retrieval.
    """

    def __init__(self, config: BatchRequestProcessorConfig) -> None:
        """Initialize the MistralBatchRequestProcessor."""
        super().__init__(config)
        self.client = Mistral(api_key=config.api_key)

    @property
    def backend(self):
        """Backend property."""
        return "mistral"

    @property
    def max_requests_per_batch(self) -> int:
        """The maximum number of requests that can be processed in a batch."""
        return 1000000  # 1 million as per Mistral's documentation - https://docs.mistral.ai/capabilities/batch/

    @property
    def max_bytes_per_batch(self) -> int:
        """The maximum number of bytes that can be processed in a batch."""
        return 100 * 1024 * 1024  # NOTE Example limit (100MB), adjust based on Mistral's documentation (https://mistral.ai/products/la-plateforme#pricing)

    @property
    def max_concurrent_batch_operations(self) -> int:
        """The maximum number of concurrent batch operations."""
        return 100  # NOTE Example limit, adjust based on Mistral's documentation (https://mistral.ai/products/la-plateforme#pricing)

    def parse_api_specific_request_counts(
        self, mistral_batch_object: t.Any
    ) -> GenericBatchRequestCounts:  # TODO: Replace t.Any with the correct type (idea: Should I create a pydantic model for Mistral's Batch object?)
        """Convert Mistral-specific request counts to generic format.

        Handles the following Mistral request count statuses:
        - completed: Successfully completed requests
        - failed: Requests that failed
        - total: Total number of requests in batch

        Args:
            mistral_batch_object: Mistral batch object.

        Returns:
            GenericBatchRequestCounts: Standardized request count format.

        """
        return GenericBatchRequestCounts(
            failed=mistral_batch_object.failed_requests,
            succeeded=mistral_batch_object.succeeded_requests,
            total=mistral_batch_object.total_requests,
            raw_request_counts_object=mistral_batch_object.model_dump(),
        )

    def parse_api_specific_batch_object(self, mistral_batch_object, request_file: str | None = None) -> GenericBatch:
        """Convert a Mistral batch object to generic format.

        Convert Mistral-specific request counts to generic format.

        Maps Mistral-specific batch statuses and timing information to our
        standardized GenericBatch format.

        Batch statuses:
        - QUEUED, RUNNING, CANCELLATION_REQUESTED, CANCELLATION_REQUESTED: Mapped to SUBMITTED
        - SUCCESS, FAILED, TIMEOUT_EXCEEDED, CANCELLED: Mapped to FINISHED

        Timing fields:
        - created_at: When the batch was created
        - completed_at: When processing ended
        - # NOTE: Mistral client-python also as a parameter `started_at` which is not used in this function

        Args:
            mistral_batch_object: Mistral's Batch object.
            request_file: Optional path to the request file.

        Returns:
            GenericBatch: Standardized batch object.

        Reference: Mistral client-python: https://github.com/mistralai/client-python/blob/main/docs/models/batchjobout.md
        """
        if mistral_batch_object.status in _PROGRESS_STATE:
            status = GenericBatchStatus.SUBMITTED
        elif mistral_batch_object.status in _FINISHED_STATE:
            status = GenericBatchStatus.FINISHED
        else:
            raise ValueError(f"Unknown batch status: {mistral_batch_object.status}")

        return GenericBatch(
            request_file=request_file,
            id=mistral_batch_object.id,
            created_at=mistral_batch_object.created_at,
            finished_at=mistral_batch_object.completed_at,
            status=status,
            api_key_suffix=self.client.api_key[-4:],
            raw_batch=mistral_batch_object.model_dump(),
            request_counts=self.parse_api_specific_request_counts(mistral_batch_object.total_requests),
            raw_status=mistral_batch_object.status,
        )

    def parse_api_specific_response(self, raw_response: dict, generic_request: GenericRequest, batch: GenericBatch) -> GenericResponse:
        """Parse Mistral API response into generic format."""
        pass

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Creates an API-specific request body from a generic request body.

        Transforms a GenericRequest into the format expected by MIstral's batch API.
        """
        pass

    async def upload_batch_file(self, file_content: bytes) -> t.Any:
        """Uploads a batch file to Mistral and waits until ready."""
        pass

    async def create_batch(self, batch_file_id: str, metadata: dict) -> t.Any:
        """Creates a batch job with Mistral using an uploaded file."""
        pass

    async def submit_batch(self, requests: list[dict], metadata: dict) -> GenericBatch:
        """Handles the complete batch submission process."""
        pass

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieve current status of a batch from Mistral's API."""
        pass

    async def download_batch(self, batch: GenericBatch) -> list[dict] | None:
        """Download the results of a completed batch."""
        pass

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancel a batch job."""
        pass
