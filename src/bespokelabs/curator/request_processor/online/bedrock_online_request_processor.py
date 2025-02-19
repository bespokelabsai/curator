"""AWS Bedrock online request processor implementation."""

import datetime
import json
import logging
import time
from typing import Any, Dict, List

import aiohttp
import boto3
from botocore.exceptions import ClientError

from bespokelabs.curator.cost import cost_processor_factory
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import (
    APIRequest,
    BaseOnlineRequestProcessor,
    OnlineStatusTracker,
)
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
SUPPORTED_MODELS = {
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
        "rpm": 100,
        "tpm": 100000,
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "supported_regions": ["us-east-1", "us-west-2"],
        "rpm": 100,
        "tpm": 100000,
    },
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
        "rpm": 100,
        "tpm": 100000,
    },
    # Meta Models
    "meta.llama3-1-405b-instruct-v1:0": {"max_input_tokens": 150000, "max_output_tokens": 4096, "supported_regions": ["us-west-2"], "rpm": 100, "tpm": 100000},
    "meta.llama3-1-70b-instruct-v1:0": {
        "max_input_tokens": 150000,
        "max_output_tokens": 4096,
        "supported_regions": ["us-east-1", "us-west-2"],
        "rpm": 100,
        "tpm": 100000,
    },
    # Amazon Nova Models
    "us.amazon.nova-micro-v1:0": {
        "max_input_tokens": 100000,
        "max_output_tokens": 4096,
        "supported_regions": ["us-east-1", "us-east-2", "us-west-2"],
        "rpm": 100,
        "tpm": 100000,
    },
    "us.amazon.nova-lite-v1:0": {
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "supported_regions": ["us-east-1", "us-east-2", "us-west-2"],
        "rpm": 100,
        "tpm": 100000,
    },
    "us.amazon.nova-pro-v1:0": {
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "supported_regions": ["us-east-1", "us-east-2", "us-west-2"],
        "rpm": 100,
        "tpm": 100000,
    },
    # Mistral Models
    "mistral.mistral-large-24-07": {"max_input_tokens": 150000, "max_output_tokens": 4096, "supported_regions": ["us-west-2"], "rpm": 100, "tpm": 100000},
}


class BedrockOnlineRequestProcessor(BaseOnlineRequestProcessor):
    """Request processor for AWS Bedrock API."""

    def __init__(self, config: OnlineRequestProcessorConfig, compatible_provider: str = None):
        """Initialize the Bedrock request processor.

        Args:
            config: Configuration for the request processor
            compatible_provider: Optional provider name for cost tracking

        Raises:
            ValueError: If model is not supported or region is invalid
        """
        super().__init__(config)

        # Set up cost tracking
        self._compatible_provider = compatible_provider or self.backend()
        self._cost_processor = cost_processor_factory(config=config, backend=self._compatible_provider)

        # Validate model and region
        if config.model not in SUPPORTED_MODELS:
            supported_models = "\n".join(SUPPORTED_MODELS.keys())
            raise ValueError(f"Model {config.model} is not supported.\n" f"Supported models are:\n{supported_models}")

        # Get region from config or default to us-east-1
        self.region = config.region or "us-east-1"
        model_config = SUPPORTED_MODELS[config.model]
        if self.region not in model_config["supported_regions"]:
            supported_regions = ", ".join(model_config["supported_regions"])
            raise ValueError(f"Model {config.model} is not supported in region {self.region}.\n" f"Supported regions are: {supported_regions}")

        # Initialize client
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

        # Store model info
        self.model_id = config.model
        self.model_provider = (
            config.model.split(".")[0] if config.model.split(".")[0] in ["anthropic", "meta", "amazon", "mistral"] else config.model.split(".")[1]
        )
        self.model_name = config.model.split(".")[1] if config.model.split(".")[0] in ["anthropic", "meta", "amazon", "mistral"] else config.model.split(".")[2]
        self.model_limits = SUPPORTED_MODELS[config.model]

    def backend(self) -> str:
        """Get the backend identifier."""
        return "bedrock"

    def compatible_provider(self) -> str:
        """Get compatible provider name."""
        return "bedrock"

    def _multimodal_prompt_supported(self) -> bool:
        """Check if multimodal prompts are supported."""
        # Claude models in Bedrock support multimodal
        return "anthropic.claude" in self.model_id

    def max_requests_per_minute(self) -> int:
        """Get rate limit for requests per minute."""
        return self.model_limits.get("rpm", 100)

    def max_tokens_per_minute(self) -> int:
        """Get rate limit for tokens per minute."""
        return self.model_limits.get("tpm", 100000)

    def estimate_total_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total tokens in messages.

        Args:
            messages: List of message dicts

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token for English text
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):  # Handle multimodal content
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += len(part["text"]) // 4
        return total

    def estimate_output_tokens(self) -> int:
        """Estimate output tokens.

        Returns:
            Estimated token count for output
        """
        return self.model_limits.get("max_output_tokens", 4096)

    def create_api_specific_request_online(self, generic_request: GenericRequest) -> dict:
        """Create a Bedrock-specific request from a generic request.

        Args:
            generic_request: The generic request to convert

        Returns:
            Dict containing the Bedrock-specific request

        Raises:
            ValueError: If model provider is not supported
        """
        messages = generic_request.get("messages", [])

        # Common parameters for all models
        base_request = {"modelId": self.model_id, "contentType": "application/json", "accept": "application/json"}

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

    def completion_cost(self, response: Dict[str, Any]) -> float:
        """Calculate the cost of a completion based on token usage."""
        if not self._cost_processor:
            return 0.0

        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return self._cost_processor.calculate_cost(model=self.model_id, input_tokens=input_tokens, output_tokens=output_tokens)

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Make a single request to the Bedrock API.

        Args:
            request: The request to send
            session: The aiohttp ClientSession (unused for boto3)
            status_tracker: Tracks request statistics

        Returns:
            The API response

        Raises:
            ClientError: If AWS API call fails
            ValueError: If model provider is not supported
            Exception: For other errors
        """
        try:
            api_request = request.to_dict()
            model_id = api_request["modelId"]
            content_type = api_request["contentType"]
            accept = api_request["accept"]
            body = api_request["body"]

            # Estimate input tokens for cost tracking
            input_tokens = self.estimate_total_tokens(request.generic_request.get("messages", []))

            # Make the API call
            response = self.client.invoke_model(modelId=model_id, contentType=content_type, accept=accept, body=json.dumps(body))

            response_body = json.loads(response.get("body").read())

            # Parse response based on model provider
            if self.model_provider == "anthropic":
                completion = response_body.get("completion", "")
                output_tokens = len(completion) // _CHARS_PER_TOKEN
                response_message = {"choices": [{"message": {"content": completion, "role": "assistant"}}]}

            elif self.model_provider == "meta":
                generation = response_body.get("generation", "")
                output_tokens = len(generation) // _CHARS_PER_TOKEN
                response_message = {"choices": [{"message": {"content": generation, "role": "assistant"}}]}

            elif self.model_provider == "amazon":
                output_text = response_body.get("outputText", "")
                output_tokens = len(output_text) // _CHARS_PER_TOKEN
                response_message = {"choices": [{"message": {"content": output_text, "role": "assistant"}}]}

            elif self.model_provider == "mistral":
                text = response_body.get("text", "")
                output_tokens = len(text) // _CHARS_PER_TOKEN
                response_message = {"choices": [{"message": {"content": text, "role": "assistant"}}]}

            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")

            # Add token usage to response for cost calculation
            response_body["usage"] = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}

            # Calculate cost
            cost = self.completion_cost(response_body)

            # Create token usage object
            token_usage = TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

            # Return formatted response
            return GenericResponse(
                response_message=response_message if self.config.return_completions_object else response_message["choices"][0]["message"]["content"],
                response_errors=None,
                raw_request=request.api_specific_request,
                raw_response=response_body,
                generic_request=request.generic_request,
                created_at=request.created_at,
                finished_at=datetime.datetime.now(),
                token_usage=token_usage,
                response_cost=cost,
                finish_reason="stop",  # Bedrock doesn't provide finish reason
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))

            # Check if this is a rate limit error
            is_rate_limit = any(err_str in error_msg.lower() for err_str in _RATE_LIMIT_ERROR_STRINGS)

            if is_rate_limit:
                status_tracker.time_of_last_rate_limit_error = time.time()
                status_tracker.num_rate_limit_errors += 1
                status_tracker.num_api_errors -= 1  # Don't double count
                status_tracker.num_other_errors -= 1

            logger.error(f"AWS error calling Bedrock API: {error_code} - {error_msg}")
            raise

        except Exception as e:
            logger.error(f"Error calling Bedrock API: {str(e)}")
            raise
