"""AWS Bedrock batch request processor implementation."""

import asyncio
import datetime
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from bespokelabs.curator.cost import cost_processor_factory
from bespokelabs.curator.request_processor.batch.base_batch_request_processor import (
    BaseBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import TokenUsage

logger = logging.getLogger(__name__)

# Default rate limits (can be overridden by service quotas)
_DEFAULT_RPM = 100
_DEFAULT_TPM = 100000

# Token estimation constants
_CHARS_PER_TOKEN = 4  # Approximate characters per token for English text
_DEFAULT_OUTPUT_TOKENS = 4096

# Error retry settings
_MAX_RETRIES = 3
_RETRY_DELAY = 5

# Rate limit error strings
_RATE_LIMIT_ERROR_STRINGS = ["rate limit", "throttling", "too many requests", "throttlexception"]

# Supported models and their limits
SUPPORTED_BATCH_MODELS = {
    # Anthropic Models
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "supported_regions": [
            "us-east-1",
            "us-east-2",
            "us-west-2",
            "us-gov-west-1",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-central-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
        ],
    },
    "anthropic.claude-3-opus-20240229-v1:0": {"max_input_tokens": 200000, "max_output_tokens": 4096, "supported_regions": ["us-east-1", "us-west-2"]},
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "supported_regions": [
            "us-east-1",
            "us-west-2",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
        ],
    },
    # Meta Models
    "meta.llama3-1-405b-instruct-v1:0": {"max_input_tokens": 150000, "max_output_tokens": 4096, "supported_regions": ["us-west-2"]},
    "meta.llama3-1-70b-instruct-v1:0": {"max_input_tokens": 150000, "max_output_tokens": 4096, "supported_regions": ["us-east-1", "us-west-2"]},
    # Amazon Nova Models
    "us.amazon.nova-micro-v1:0": {"max_input_tokens": 100000, "max_output_tokens": 4096, "supported_regions": ["us-east-1", "us-east-2", "us-west-2"]},
    "us.amazon.nova-lite-v1:0": {"max_input_tokens": 200000, "max_output_tokens": 4096, "supported_regions": ["us-east-1", "us-east-2", "us-west-2"]},
    "us.amazon.nova-pro-v1:0": {"max_input_tokens": 200000, "max_output_tokens": 4096, "supported_regions": ["us-east-1", "us-east-2", "us-west-2"]},
    # Mistral Models
    "mistral.mistral-large-24-07": {"max_input_tokens": 150000, "max_output_tokens": 4096, "supported_regions": ["us-west-2"]},
}


class BedrockBatchRequestProcessor(BaseBatchRequestProcessor):
    """Request processor for AWS Bedrock Batch API."""

    def __init__(self, config: BatchRequestProcessorConfig):
        """Initialize the Bedrock batch request processor.

        Args:
            config: Configuration for the request processor

        Raises:
            ValueError: If model is not supported or region is invalid
        """
        super().__init__(config)

        # Set up cost tracking
        self._cost_processor = cost_processor_factory(config=config, backend=self.compatible_provider)

        # Validate model and region
        if config.model not in SUPPORTED_BATCH_MODELS:
            supported_models = "\n".join(SUPPORTED_BATCH_MODELS.keys())
            raise ValueError(f"Model {config.model} is not supported for batch inference.\n" f"Supported models are:\n{supported_models}")

        # Get region from config or default to us-east-1
        self.region = config.region or "us-east-1"
        model_config = SUPPORTED_BATCH_MODELS[config.model]
        if self.region not in model_config["supported_regions"]:
            supported_regions = ", ".join(model_config["supported_regions"])
            raise ValueError(f"Model {config.model} is not supported in region {self.region}.\n" f"Supported regions are: {supported_regions}")

        # Initialize clients
        self.client = boto3.client("bedrock", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        self.quotas = boto3.client("service-quotas", region_name=self.region)

        # Store model info
        self.model_id = config.model
        self.model_provider = (
            config.model.split(".")[0] if config.model.split(".")[0] in ["anthropic", "meta", "amazon", "mistral"] else config.model.split(".")[1]
        )
        self.model_name = config.model.split(".")[1] if config.model.split(".")[0] in ["anthropic", "meta", "amazon", "mistral"] else config.model.split(".")[2]
        self.model_limits = SUPPORTED_BATCH_MODELS[config.model]

        # Update model limits from quotas
        self._update_model_limits()

    def backend(self) -> str:
        """Get the backend identifier."""
        return "bedrock"

    def compatible_provider(self) -> str:
        """Get compatible provider name."""
        return "bedrock"

    def _get_all_quotas(self) -> List[Dict]:
        """Get all Bedrock service quotas."""
        try:
            response = self.quotas.list_service_quotas(ServiceCode="bedrock")
            return response.get("Quotas", [])
        except Exception as e:
            logger.warning(f"Could not retrieve quotas: {e}")
            return []

    def _get_quota_by_pattern(self, quotas: List[Dict], pattern: str, default_value: float) -> float:
        """Find quota value by matching pattern in quota name."""
        for quota in quotas:
            if pattern.lower() in quota["QuotaName"].lower():
                return quota["Value"]
        return default_value

    def _update_model_limits(self):
        """Update model limits from service quotas."""
        quotas = self._get_all_quotas()

        # Get batch size limits
        records_pattern = f"records per batch inference job for {self.model_name}"
        min_records_pattern = f"minimum number of records per batch inference job for {self.model_name}"

        records_quota = self._get_quota_by_pattern(quotas, records_pattern, 50000.0)
        min_records_quota = self._get_quota_by_pattern(quotas, min_records_pattern, 100.0)

        self.model_limits.update({"max_batch_size": int(records_quota), "min_batch_size": int(min_records_quota)})

        # Get file size limits
        input_size_pattern = f"batch inference input file size (in gb) for {self.model_name}"
        job_size_pattern = f"batch inference job size (in gb) for {self.model_name}"

        input_size_quota = self._get_quota_by_pattern(quotas, input_size_pattern, 1.0)
        job_size_quota = self._get_quota_by_pattern(quotas, job_size_pattern, 5.0)

        self.model_limits.update({"max_input_file_size_gb": float(input_size_quota), "max_job_size_gb": float(job_size_quota)})

        logger.info(f"Updated model limits for {self.model_id}:")
        logger.info(f"Max batch size: {self.model_limits['max_batch_size']}")
        logger.info(f"Min batch size: {self.model_limits['min_batch_size']}")
        logger.info(f"Max input file size (GB): {self.model_limits['max_input_file_size_gb']}")
        logger.info(f"Max job size (GB): {self.model_limits['max_job_size_gb']}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        # Simple estimation: ~4 characters per token for English text
        return len(text) // 4

    def _calculate_optimal_shard_size(self, requests: List[Dict]) -> int:
        """Calculate optimal shard size based on AWS Bedrock limits and input data."""
        # Sample up to 100 records to estimate average sizes
        sample_size = min(100, len(requests))
        sample_records = requests[:sample_size]

        # Calculate average record size and validate token limits
        total_size_bytes = 0
        max_record_size = 0
        max_tokens_in_record = 0

        for record in sample_records:
            # Calculate exact record size
            record_size = len(json.dumps(record).encode("utf-8"))
            total_size_bytes += record_size
            max_record_size = max(max_record_size, record_size)

            # Estimate tokens for validation
            messages = record.get("messages", [])
            record_tokens = sum(self._estimate_tokens(msg.get("content", "")) for msg in messages)
            max_tokens_in_record = max(max_tokens_in_record, record_tokens)

        # Validate token limits
        if max_tokens_in_record > self.model_limits["max_input_tokens"]:
            raise ValueError(
                f"Input contains records exceeding model's token limit. "
                f"Maximum tokens found: {max_tokens_in_record}, "
                f"Model limit: {self.model_limits['max_input_tokens']}"
            )

        avg_bytes_per_record = total_size_bytes / sample_size
        max_input_file_size_bytes = self.model_limits["max_input_file_size_gb"] * 1024 * 1024 * 1024

        # Calculate size-based limit with 10% safety margin
        size_based_limit = int((max_input_file_size_bytes * 0.9) / avg_bytes_per_record)

        # Calculate optimal shard size
        optimal_size = min(size_based_limit, self.model_limits.get("max_batch_size", 50000))

        # Ensure we don't go below minimum batch size
        optimal_size = max(optimal_size, self.model_limits.get("min_batch_size", 100))

        logger.info("Shard size calculation details:")
        logger.info(f"Average bytes per record: {avg_bytes_per_record:.2f}")
        logger.info(f"Maximum record size: {max_record_size} bytes")
        logger.info(f"Maximum tokens in a record: {max_tokens_in_record}")
        logger.info(f"Optimal shard size: {optimal_size} records")

        return optimal_size

    def _multimodal_prompt_supported(self) -> bool:
        """Check if multimodal prompts are supported."""
        return "anthropic.claude" in self.model_id

    def max_requests_per_batch(self) -> int:
        """Get maximum number of requests per batch."""
        return self.model_limits.get("max_batch_size", 50000)

    def max_bytes_per_batch(self) -> int:
        """Get maximum bytes per batch."""
        return int(self.model_limits.get("max_input_file_size_gb", 1.0) * 1024 * 1024 * 1024)

    def max_concurrent_batch_operations(self) -> int:
        """Get maximum concurrent batch operations."""
        return 20  # Default from AWS Bedrock quotas

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Create a Bedrock-specific batch request from a generic request.

        Args:
            generic_request: The generic request to convert

        Returns:
            Dict containing the Bedrock-specific request

        Raises:
            ValueError: If model provider is not supported
        """
        messages = generic_request.get("messages", [])

        # Common parameters for all models
        base_request = {
            "modelId": self.model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "custom_id": str(generic_request.original_row_idx),  # Required for response mapping
        }

        if self.model_provider == "anthropic":
            # Format for Anthropic Claude models
            system = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            prompt_messages = [msg for msg in messages if msg["role"] != "system"]

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": prompt_messages,
                "max_tokens": generic_request.get("max_tokens", self.model_limits["max_output_tokens"]),
                "temperature": generic_request.get("temperature", 0.7),
                "top_p": generic_request.get("top_p", 0.95),
            }

            if system:
                body["system"] = system

        elif self.model_provider == "meta":
            # Format for Meta Llama models
            formatted_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{msg['content']}<|eot_id|>"
                elif msg["role"] == "user":
                    formatted_prompt += f"<|start_header_id|>user<|end_header_id|>{msg['content']}<|eot_id|>"
                    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>"

            body = {
                "prompt": formatted_prompt,
                "max_gen_len": generic_request.get("max_tokens", self.model_limits["max_output_tokens"]),
                "temperature": generic_request.get("temperature", 0.7),
                "top_p": generic_request.get("top_p", 0.95),
            }

        elif self.model_provider == "amazon":
            # Format for Amazon Nova models
            system = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            prompt_messages = [msg for msg in messages if msg["role"] != "system"]

            body = {
                "messages": prompt_messages,
                "textGenerationConfig": {
                    "maxTokenCount": generic_request.get("max_tokens", self.model_limits["max_output_tokens"]),
                    "temperature": generic_request.get("temperature", 0.7),
                    "topP": generic_request.get("top_p", 0.95),
                    "stopSequences": [],
                },
            }

            if system:
                body["systemPrompt"] = system

        elif self.model_provider == "mistral":
            # Format for Mistral models
            body = {
                "messages": messages,
                "max_tokens": generic_request.get("max_tokens", self.model_limits["max_output_tokens"]),
                "temperature": generic_request.get("temperature", 0.7),
                "top_p": generic_request.get("top_p", 0.95),
            }

        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        base_request["body"] = body
        return base_request

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parse API-specific response into generic format.

        Args:
            raw_response: Raw response dictionary from API
            generic_request: Original generic request object
            batch: Batch object containing context information

        Returns:
            GenericResponse: Standardized response object

        Raises:
            ValueError: If model provider is not supported
        """
        # Estimate input tokens
        input_tokens = self.estimate_total_tokens(generic_request.get("messages", []))

        # Parse response based on model provider
        if self.model_provider == "anthropic":
            completion = raw_response.get("completion", "")
            output_tokens = len(completion) // _CHARS_PER_TOKEN
            response_message = {"choices": [{"message": {"content": completion, "role": "assistant"}}]}

        elif self.model_provider == "meta":
            generation = raw_response.get("generation", "")
            output_tokens = len(generation) // _CHARS_PER_TOKEN
            response_message = {"choices": [{"message": {"content": generation, "role": "assistant"}}]}

        elif self.model_provider == "amazon":
            output_text = raw_response.get("outputText", "")
            output_tokens = len(output_text) // _CHARS_PER_TOKEN
            response_message = {"choices": [{"message": {"content": output_text, "role": "assistant"}}]}

        elif self.model_provider == "mistral":
            text = raw_response.get("text", "")
            output_tokens = len(text) // _CHARS_PER_TOKEN
            response_message = {"choices": [{"message": {"content": text, "role": "assistant"}}]}

        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        # Create token usage object
        token_usage = TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        # Calculate cost
        cost = self._cost_processor.calculate_cost(model=self.model_id, input_tokens=input_tokens, output_tokens=output_tokens)

        # Return formatted response
        return GenericResponse(
            response_message=response_message if self.config.return_completions_object else response_message["choices"][0]["message"]["content"],
            response_errors=None,
            raw_request=generic_request,
            raw_response=raw_response,
            generic_request=generic_request,
            created_at=datetime.datetime.now(),
            finished_at=datetime.datetime.now(),
            token_usage=token_usage,
            response_cost=cost,
            finish_reason="stop",  # Bedrock doesn't provide finish reason
        )

    def parse_api_specific_batch_object(self, batch: object, request_file: Optional[str] = None) -> GenericBatch:
        """Parse API-specific batch object into generic format.

        Args:
            batch: Raw batch object from API
            request_file: Optional path to request file

        Returns:
            Generic batch object
        """
        return {
            "id": batch.get("jobId"),
            "status": batch.get("status"),
            "created": batch.get("creationTime"),
            "model": self.model_id,
            "request_file": request_file,
            "metadata": batch.get("metadata", {}),
            "request_counts": self.parse_api_specific_request_counts(batch.get("requestCounts", {})),
        }

    def parse_api_specific_request_counts(self, request_counts: object, request_file: Optional[str] = None) -> GenericBatchRequestCounts:
        """Parse API-specific request counts into generic format.

        Args:
            request_counts: Raw request count object from API
            request_file: Optional path to request file

        Returns:
            GenericBatchRequestCounts: Standardized request count object
        """
        return GenericBatchRequestCounts(
            total=request_counts.get("total", 0),
            succeeded=request_counts.get("succeeded", 0),
            failed=request_counts.get("failed", 0),
            retried=request_counts.get("retried", 0),
            request_file=request_file,
        )

    async def submit_batch(self, requests: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> GenericBatch:
        """Submit a batch of requests to Bedrock.

        Args:
            requests: List of request dictionaries
            metadata: Optional metadata for the batch

        Returns:
            GenericBatch object with batch information

        Raises:
            ValueError: If requests exceed model limits
            ClientError: If AWS API call fails
            Exception: For other errors
        """
        async with self.semaphore:
            try:
                # Calculate optimal shard size and validate requests
                shard_size = self._calculate_optimal_shard_size(requests)
                if len(requests) > shard_size:
                    raise ValueError(
                        f"Batch size {len(requests)} exceeds optimal shard size {shard_size}. " f"Please split your requests into smaller batches."
                    )

                # Create a batch inference job
                timestamp = int(time.time())
                job_name = f"curator-batch-{uuid.uuid4()}-{timestamp}"

                # Prepare input data for S3
                input_data = "\n".join(json.dumps(req) for req in requests)
                bucket_name = self.config.bucket_name
                input_key = f"input/{job_name}.jsonl"
                input_s3_uri = f"s3://{bucket_name}/{input_key}"

                # Upload input data to S3 with retries
                max_retries = _MAX_RETRIES
                retry_delay = _RETRY_DELAY

                for attempt in range(max_retries):
                    try:
                        self.s3.put_object(Bucket=bucket_name, Key=input_key, Body=input_data.encode("utf-8"))
                        break
                    except ClientError as e:
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to upload input data to S3 after {max_retries} attempts")
                            raise e from None
                        logger.warning(f"S3 upload attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(retry_delay)

                # Create the batch inference job
                try:
                    response = self.client.create_model_inference_job(
                        jobName=job_name,
                        modelId=self.model_id,
                        inputConfig={"s3Uri": input_s3_uri, "dataInputConfig": {"contentType": "application/json", "inputFormat": "jsonlines"}},
                        outputConfig={"s3Uri": f"s3://{bucket_name}/output/{job_name}/", "dataOutputConfig": {"contentType": "application/json"}},
                        roleArn=self.config.role_arn,
                        tags={"CreatedBy": "curator", "BatchId": str(uuid.uuid4()), "Timestamp": str(timestamp)},
                    )

                    logger.info(f"Successfully submitted batch job {job_name}")
                    return self.parse_api_specific_batch_object(
                        {
                            "jobId": response["jobId"],
                            "status": response["status"],
                            "creationTime": response.get("creationTime"),
                            "modelId": self.model_id,
                            "metadata": {**(metadata or {}), "job_name": job_name, "input_key": input_key, "shard_size": shard_size},
                            "requestCounts": {"total": len(requests), "succeeded": 0, "failed": 0, "retried": 0},
                        }
                    )

                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "Unknown")
                    error_msg = e.response.get("Error", {}).get("Message", str(e))
                    logger.error(f"AWS error creating batch job: {error_code} - {error_msg}")
                    raise

            except Exception as e:
                logger.error(f"Failed to submit batch job: {str(e)}")
                raise

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Get current status of a batch job.

        Args:
            batch: The batch to check

        Returns:
            Updated batch object
        """
        try:
            response = self.client.get_model_inference_job(jobId=batch["id"])
            return self.parse_api_specific_batch_object(response)
        except Exception as err:
            logger.error(f"Failed to retrieve batch status: {str(err)}")
            raise Exception(f"Failed to retrieve batch status: {str(err)}") from err

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancel a batch job.

        Args:
            batch: The batch to cancel

        Returns:
            Updated batch object
        """
        try:
            response = self.client.stop_model_inference_job(jobId=batch["id"])
            return self.parse_api_specific_batch_object(response)
        except Exception as err:
            logger.error(f"Failed to cancel batch job: {str(err)}")
            raise Exception(f"Failed to cancel batch job: {str(err)}") from err

    async def download_batch(self, batch: GenericBatch) -> Optional[str]:
        """Download results from a completed batch.

        Args:
            batch: The completed batch

        Returns:
            Path to downloaded results file or None if not ready
        """
        try:
            # Get the job details
            job = await self.retrieve_batch(batch)

            if job["status"] != "COMPLETED":
                return None

            # Download results from S3
            output_prefix = f"output/{batch['id']}/"

            # List objects in output prefix
            response = self.s3.list_objects_v2(Bucket=self.config.bucket_name, Prefix=output_prefix)

            if not response.get("Contents"):
                return None

            # Download and combine all output files
            results = []
            for obj in response["Contents"]:
                data = self.s3.get_object(Bucket=self.config.bucket_name, Key=obj["Key"])
                content = data["Body"].read().decode("utf-8")
                results.extend([json.loads(line) for line in content.splitlines()])

            return results

        except Exception as err:
            logger.error(f"Failed to download batch results: {str(err)}")
            raise Exception(f"Failed to download batch results: {str(err)}") from err
