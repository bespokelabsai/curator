import json
import os
import typing as t
from functools import lru_cache

import vertexai
from google.cloud import aiplatform, storage
from pydantic import BaseModel
from vertexai.batch_prediction import BatchPredictionJob

from bespokelabs.curator import constants
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.batch.base_batch_request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage

"""
Gemini latest rate limits:

gemini-2.0-pro	50k records
gemini-2.0-flash	150k records
gemini-1.5-pro	50k records
gemini-1.5-flash	150k records
gemini-1.0-pro	150k records
gemini-1.0-pro-vision	50k records
"""
# NOTE: Do not change the order.
_GEMINI_BATCH_RATELIMIT_MAP = {
    "gemini-2.0-pro": 50_000,
    "gemini-2.0-flash": 150_000,
    "gemini-1.5-pro": 50_000,
    "gemini-1.5-flash": 150_000,
    "gemini-1.0-pro-vision": 50_000,
    "gemini-1.0-pro": 150_000,
}
_REQUEST_FILE_SHARD_SIZE = 1024

# Gemini batch state map

_FAILED = {"JOB_STATE_EXPIRED", "JOB_STATE_FAILED", "JOB_STATE_PAUSED"}
_PROGRESS = {"JOB_STATE_CANCELLING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED", "JOB_STATE_PAUSED", "JOB_STATE_RUNNING", "JOB_STATE_UPDATING"}
_FINISHED = {"JOB_STATE_CANCELLED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_SUCCEEDED"}


def _response_format_to_json(cls: BaseModel):
    schema = cls.model_json_schema()
    if "$defs" not in schema:
        return schema

    defs = schema.pop("$defs")

    def _resolve(schema):
        if "$ref" in schema:
            ref = schema.pop("$ref")
            schema.update(defs[ref.split("/")[-1]])
        if "properties" in schema:
            for prop in schema["properties"].values():
                _resolve(prop)
        if "items" in schema:
            _resolve(schema["items"])
        schema.pop("title", None)

    _resolve(schema)
    return schema


class GeminiBatchRequestProcessor(BaseBatchRequestProcessor):
    """Gemini-specific implementation of the BatchRequestProcessor.

    This class handles batch processing of requests using Gemini's API, including
    file uploads, batch submissions, and result retrieval. Supports message batches
    with limitations defined in Gemini's documentation.

    Information about limits:
    https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#batch-requests

    Attributes:
        web_dashboard: URL to Gemini's web dashboard for batch monitoring.
    """

    def __init__(self, config: BatchRequestProcessorConfig) -> None:
        """Initialize the GeminiBatchRequestProcessor."""
        super().__init__(config)

        self._initialize_cloud()

    def _initialize_cloud(self):
        self._location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self._project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._bucket_name = os.environ.get("GEMINI_BUCKET_NAME")

        assert self._bucket_name, "GEMINI_BUCKET_NAME environment variable is not set"
        assert self._project_id is not None, "GOOGLE_CLOUD_PROJECT environment variable is not set"
        assert self._location is not None, "GOOGLE_CLOUD_REGION environment variable is not set"

        vertexai.init(project=self._project_id, location=self._location)

        self._bucket = storage.Client().bucket(self._bucket_name)
        self.web_dashboard = f"https://console.cloud.google.com/ai/platform/locations/{self._location}/batch-predictions"

    @property
    def backend(self):
        """Backend property."""
        return "gemini"

    @property
    def max_requests_per_batch(self) -> int:
        """Maximum number of requests allowed in a single Gemini batch.

        Returns:
            int: The maximum number of requests per batch.
        """
        for prefix in _GEMINI_BATCH_RATELIMIT_MAP:
            if self.config.model.startswith(prefix):
                return _GEMINI_BATCH_RATELIMIT_MAP[prefix]
        else:
            raise ValueError(f"Could not find rate limit for {self.config.model}")

    @property
    def max_bytes_per_batch(self) -> int:
        """Maximum size in bytes allowed for a single Gemini batch.

        Returns:
            int: The maximum batch size in bytes.
        """
        return _REQUEST_FILE_SHARD_SIZE * 1024 * 1024  # 1 GB

    @property
    def max_concurrent_batch_operations(self) -> int:
        """Maximum number of concurrent batch operations allowed.

        Set to 4 to avoid hitting undocumented rate limits in the batch operation APIs.

        Returns:
            int: The maximum number of concurrent operations (4).
        """
        # TODO: This is not for ratelimiting
        # https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#batch-requests
        return 4

    # ruff: noqa: B019
    @lru_cache
    def get_n_requests_in_file(self, request_file: str) -> int:
        """Returns the number of requests in a request file.

        Args:
            request_file: Path to the request file.

        Returns:
            int: Number of requests in the file.
        """
        with open(request_file) as f:
            return sum(1 for _ in f)

    def parse_api_specific_request_counts(self, batch: "BatchPredictionJob", request_file: t.Optional[str] = None) -> GenericBatchRequestCounts:
        """Converts Gemini-specific request counts to generic format.

        Handles the following Gemini request count statuses:
        - failed
        - succeeded
        - inprogress

        Args:
            batch: Gemini's BatchPredictionJob object.
            request_file: Path to the request file.


        Returns:
            GenericBatchRequestCounts: Standardized request count format.
        """
        # TODO: bug in google python sdk, completion_stats are empty
        # TODO: Use batch.completion_stats when it's fixed
        processing = succeeded = failed = 0
        n_requests = self.get_n_requests_in_file(request_file)
        if batch.state.name in _PROGRESS:
            processing = n_requests
        elif batch.state.name in _FINISHED:
            succeeded = n_requests
        elif batch.state.name in _FAILED:
            failed = n_requests
        return GenericBatchRequestCounts(
            failed=failed,
            succeeded=succeeded,
            total=processing + succeeded + failed,
            raw_request_counts_object={},
        )

    def parse_api_specific_batch_object(self, batch, request_file: str | None = None) -> GenericBatch:
        """Converts an Gemini batch object to generic format.

        Maps Gemini-specific batch statuses and timing information to our
        standardized GenericBatch format.

        Args:
            batch: Gemini's BatchPredictionJob object.
            request_file: Optional path to the request file.

        Returns:
            GenericBatch: Standardized batch object.

        Raises:
            ValueError: If the batch status is unknown.
        """
        if batch.state.name in _PROGRESS:
            status = GenericBatchStatus.SUBMITTED.value
        elif batch.state.name in _FINISHED:
            status = GenericBatchStatus.FINISHED.value
        else:
            raise ValueError(f"Unknown batch status: {batch.state.name}")
        return GenericBatch(
            request_file=request_file,
            id=batch.name,
            created_at=batch.create_time,
            finished_at=batch.update_time,
            status=status,
            api_key_suffix="gs",
            request_counts=self.parse_api_specific_request_counts(batch, request_file=request_file),
            raw_batch=batch.to_dict(),
            raw_status=batch.state.name,
        )

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Creates an API-specific request body from a generic request.

        Transforms a GenericRequest into the format expected by Gemini's batch API.
        Combines and constructs a system message with schema and instructions using
        the instructor package for JSON response formatting.

        Args:
            generic_request: The generic request object containing model, messages,
                and optional response format.

        Returns:
            dict: API specific request body formatted for Gemini's batch API,
                including custom_id and request parameters.
        """
        contents = []
        for message in generic_request.messages:
            contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})
        request_object = {"contents": contents}
        if generic_request.response_format:
            request_object.update(
                {
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": _response_format_to_json(self.prompt_formatter.response_format),
                    }
                }
            )

        if self.config.generation_params:
            if "generationConfig" in request_object:
                request_object["generationConfig"].update(self.config.generation_params)
            else:
                request_object.update({"generationConfig": self.config.generation_params})

        return {
            "request": request_object,
            constants.BATCH_REQUEST_ID_TAG: str(generic_request.original_row_idx),
        }

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parses an Gemini API response into generic format.

        Processes the raw response from Gemini's batch API, handling both

        successful and failed responses, including token usage and cost calculation.
        For batch requests, a 50% discount is applied to the cost.

        Args:
            raw_response: Raw response dictionary from Gemini's API.
            generic_request: Original generic request object.
            batch: The batch object containing timing information.

        Returns:
            GenericResponse: Standardized response object with parsed message,
                errors, token usage, and cost information.
        """
        result_type = raw_response.get("status", "succeeded")
        # TODO: This is a bug in gemini batch prediction response
        if result_type == "":
            result_type = "succeeded"

        token_usage = None
        cost = None
        response_message = None

        if result_type == "succeeded":
            response_body = raw_response["response"]
            if self.config.return_completions_object:
                response_message_raw = response_body
            else:
                response_message_raw = response_body["candidates"][0]["content"]["parts"][0]["text"]
            usage = response_body.get("usageMetadata", {})

            token_usage = _TokenUsage(
                input=usage.get("promptTokenCount", 0),
                output=usage.get("candidatesTokenCount", 0),
                total=usage.get("totalTokenCount", 0),
            )

            response_message, response_errors = self.prompt_formatter.parse_response_message(response_message_raw)

            cost = self._cost_processor.cost(model=self.config.model, prompt=str(generic_request.messages), completion=response_message_raw)

        # TODO: check other result types.
        else:
            response_errors = [f"Request {result_type}"]

        # TODO: may be move it to GenericBatch
        return GenericResponse(
            response_message=response_message,
            response_errors=response_errors,
            raw_response=raw_response,
            raw_request=None,
            generic_request=generic_request,
            created_at=batch.created_at,
            finished_at=raw_response["processed_time"],
            token_usage=token_usage,
            response_cost=cost,
        )

    def _upload_batch_file(self, requests: list, metadata: dict):
        path = metadata["request_file"]
        filename = os.path.basename(path)
        gcs_path = f"gs://{self._bucket_name}/batch_requests/{filename}"
        try:
            blob_name = f"batch_requests/{filename}"
            blob = self._bucket.blob(blob_name)
            jsonl_data = "\n".join(json.dumps(item, ensure_ascii=False) for item in requests)
            blob.upload_from_string(jsonl_data, content_type="application/jsonl+json")
        except Exception as e:
            logger.error(f"Could not upload batch file request to gcloud at {gcs_path} :: reason {e}")
            raise
        else:
            logger.info(f"Uploaded request batch file to gcloud at {gcs_path}")
        return gcs_path

    def _create_batch(self, input_dataset: str):
        output_bucket = f"gs://{self._bucket_name}"
        try:
            job = BatchPredictionJob.submit(source_model=self.config.model, input_dataset=input_dataset, output_uri_prefix=output_bucket)
        except Exception as e:
            logger.error(f"Could not create batch prediction job for {input_dataset} :: reason {e}")
            raise
        else:
            logger.info(f"Successfully created batch prediction job for {input_dataset}")

        job = self._get_batch_job_object(job.name)
        return job

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
            input_dataset = self._upload_batch_file(requests, metadata)
            batch = self._create_batch(input_dataset)
            return self.parse_api_specific_batch_object(batch, request_file=metadata["request_file"])

    # ruff: noqa: B019
    @lru_cache
    def _get_batch_job_object(self, id):
        uri = self._get_batch_job_uri(id)
        return aiplatform.BatchPredictionJob(uri)

    def _get_batch_job_uri(self, id):
        return f"projects/{self._project_id}/locations/{self._location}/batchPredictionJobs/{id}"

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieves the current status of a batch from Gemini's API.

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
                job = self._get_batch_job_object(batch.id)
            # TODO: check for specific not found exceptions
            except Exception as e:
                logger.warning(f"batch object {batch.id} not found. :: reason {e}")
                return None

            request_file = self.tracker.submitted_batches[batch.id].request_file
            return self.parse_api_specific_batch_object(job, request_file=request_file)

    async def download_batch(self, batch: GenericBatch) -> list[dict] | None:
        """Downloads the results of a completed batch.

        Args:
            batch: The batch object to download results for.

        Returns:
            list[dict]: List of response dictionaries.
            None: If download fails.

        Side Effects:
            - Streams results from Gemini's API.
            - Converts each result to a dictionary format.
        """
        async with self.semaphore:
            # TODO: validate batch (check if it even makes sense)

            responses = []
            job = self._get_batch_job_object(batch.id)
            for result in job.iter_outputs():
                content = result.download_as_string().decode("utf-8")
                if not content:
                    continue
                for line in content.splitlines():
                    responses.append(json.loads(line))
            return responses

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancels a running batch job.

        Args:
            batch: The batch object to cancel.

        Returns:
            GenericBatch: Updated batch object after cancellation attempt.

        Side Effects:
            - Attempts to cancel the batch with Gemini's API.
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
                job = self._get_batch_job_object(batch.id)
                # TODO: check if delete or cancel
                job.cancel()
                logger.info(f"Successfully cancelled batch: {batch.id}")
                return self.parse_api_specific_batch_object(batch_object, request_file=request_file)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to cancel batch {batch.id}: {error_msg}")
                return self.parse_api_specific_batch_object(batch_object, request_file=request_file)
