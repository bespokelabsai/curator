import typing as t

from mistralai import Mistral

from bespokelabs.curator.request_processor.batch.base_batch_request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse

# Reference for Mistral status: https://github.com/mistralai/client-python/blob/main/docs/models/batchjobstatus.md
_PROGRESS_STATE = {"QUEUED", "RUNNING", "cancelling", "CANCELLATION_REQUESTED"}
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

    def parse_api_specific_request_counts(self, mistral_batch_object) -> GenericBatchRequestCounts:
        """Convert Mistral-specific request counts to generic format."""
        pass

    def parse_api_specific_batch_object(self, mistral_batch_object, request_file: str | None = None) -> GenericBatch:
        """Convert a Mistral batch object to generic format."""
        # Reference for Mistral BatchJobOut: https://github.com/mistralai/client-python/blob/main/docs/models/batchjobout.md
        pass

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
