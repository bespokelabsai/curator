import datetime
import os
import time
from typing import TypeVar

import aiohttp
import litellm
import tiktoken
from anthropic import Anthropic
from litellm.litellm_core_utils.core_helpers import map_finish_reason

from bespokelabs.curator.cost import cost_processor_factory
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest, BaseOnlineRequestProcessor
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker, TokenLimitStrategy
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse, _TokenUsage

T = TypeVar("T")

_DEFAULT_ANTHROPIC_URL: str = "https://api.anthropic.com/v1/messages"

_ANTHROPIC_MULTIMODAL_SUPPORTED_MODELS = {"claude-3", "claude-3-sonnet", "claude-3-haiku", "claude-3-opus"}
_ANTHROPIC_ALLOWED_IMAGE_SIZE_MB = 20  # MB


class AnthropicOnlineRequestProcessor(BaseOnlineRequestProcessor):
    """Anthropic-specific implementation of the OnlineRequestProcessor.

    Handles API requests to Anthropic's chat completion endpoints with rate limiting,
    token counting, and error handling specific to Anthropic's API.

    Note:
        - Automatically detects and respects API rate limits
        - Handles token counting
        - Supports structured output via JSON schema
        - Supports thinking models
    """

    def __init__(self, config: OnlineRequestProcessorConfig):
        """Initialize the AnthropicOnlineRequestProcessor."""
        super().__init__(config)
        self._compatible_provider = "anthropic"
        self._cost_processor = cost_processor_factory(config=config, backend=self._compatible_provider)

        if self.config.base_url is None:
            if "ANTHROPIC_BASE_URL" in os.environ:
                key_url = os.environ["ANTHROPIC_BASE_URL"].strip().rstrip("/")
                self.url = key_url + "/messages"
            else:
                self.url = _DEFAULT_ANTHROPIC_URL
        else:
            self.url = self.config.base_url + "/messages"

        self.api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")

        # Set token limit strategy to separate for input/output token accounting
        self.token_limit_strategy = TokenLimitStrategy.seperate

        # Handle separate input/output token rate limits
        self._set_manual_tpm(config)

        # Attempt to get rate limits from headers
        self.header_based_max_requests_per_minute, self.header_based_max_tokens_per_minute = self.get_header_based_rate_limits()
        self.token_encoding = self.get_token_encoding()

    def _set_manual_tpm(self, config):
        """Set token per minute limits based on config values.

        For anthropic, we handle separate input and output token tracking.
        """
        if config.max_input_tokens_per_minute and config.max_output_tokens_per_minute:
            self.manual_max_tokens_per_minute = _TokenUsage(input=config.max_input_tokens_per_minute, output=config.max_output_tokens_per_minute)
        else:
            self.manual_max_tokens_per_minute = None

    @property
    def backend(self):
        """Backend property."""
        return "anthropic"

    @property
    def compatible_provider(self) -> str:
        """Compatible provider property."""
        return self._compatible_provider

    def test_call(self):
        """Test call to get rate limits."""
        url = "/".join(self.url.split("/")[:-2])
        client = Anthropic(base_url=url)
        response = client.messages.with_raw_response.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, Claude",
                }
            ],
            model="claude-3-5-sonnet-latest",
        )

        return response.headers

    def get_header_based_rate_limits(self) -> tuple[int, _TokenUsage]:
        """Get rate limits from Anthropic API headers.

        Returns:
            tuple[int, _TokenUsage]: Contains 'max_requests_per_minute' and 'max_tokens_per_minute'
                with separate input and output token limits

        Note:
            - Makes a dummy request to get actual rate limits from headers
        """
        if not self.api_key:
            raise ValueError("Missing Anthropic API Key - Please set ANTHROPIC_API_KEY in your environment vars")

        # Default to these values when header-based limits can't be determined
        # Based on Anthropic's documented default rate limits
        # These defaults are true for tier 4 Claude 3.7
        # More information: https://docs.anthropic.com/en/api/rate-limits#rate-limits
        headers = self.test_call()
        rpm = headers.get("anthropic-ratelimit-requests-limit", 4000)
        input_tpm = headers.get("anthropic-ratelimit-output-tokens-limit", 80000)
        output_tpm = headers.get("anthropic-ratelimit-input-tokens-limit", 400000)
        return int(rpm), _TokenUsage(input=int(input_tpm), output=int(output_tpm))

    def estimate_output_tokens(self) -> int:
        """Estimate number of tokens in the response.

        Returns:
            int: Estimated number of output tokens
        """
        return self._output_tokens_moving_average() or self._get_max_tokens() // 4

    def _get_max_tokens(self):
        if self.config.generation_params.get("max_tokens"):
            return self.config.generation_params["max_tokens"]
        return litellm.get_max_tokens(model=self.config.model)

    def estimate_total_tokens(self, messages: list) -> _TokenUsage:
        """Estimate total tokens for a request using Anthropic's token counting rules.

        Args:
            messages (list): List of message dictionaries with role and content

        Returns:
            _TokenUsage: Estimated input and output tokens including message formatting tokens
        """
        num_tokens = 0

        # Simplistic token counting for Anthropic
        # In reality, Anthropic uses more complex rules, but this is a reasonable approximation
        for message in messages:
            # Count tokens in message content
            content = message.get("content", "")
            if isinstance(content, str):
                num_tokens += len(self.token_encoding.encode(content, disallowed_special=()))
            elif isinstance(content, list):
                # For multimodal content
                for item in content:
                    if item.get("type") == "text":
                        num_tokens += len(self.token_encoding.encode(item.get("text", ""), disallowed_special=()))
                    elif item.get("type") == "image" or item.get("type") == "image_url":
                        # Approximate token count for images
                        num_tokens += 1024  # Approximate token count for images

        # Add tokens for message formatting
        num_tokens += 50  # Approximate overhead for message formatting

        output_tokens = self.estimate_output_tokens()
        return _TokenUsage(input=num_tokens, output=output_tokens)

    def file_upload_limit_check(self, base64_image: str) -> None:
        """Check if the image size is within the allowed limit."""
        from bespokelabs.curator.file_utilities import get_base64_size

        mb = get_base64_size(base64_image)
        if mb > _ANTHROPIC_ALLOWED_IMAGE_SIZE_MB:
            raise ValueError(f"Image size is {mb} MB, which is greater than the allowed size of {_ANTHROPIC_ALLOWED_IMAGE_SIZE_MB} MB.")

    @property
    def _multimodal_prompt_supported(self) -> bool:
        """Check if the model supports multimodal prompts."""
        return any(model_prefix in self.config.model for model_prefix in _ANTHROPIC_MULTIMODAL_SUPPORTED_MODELS)

    def create_api_specific_request_online(self, generic_request: GenericRequest) -> dict:
        """Create an Anthropic-specific request from a generic request.

        Args:
            generic_request (GenericRequest): The generic request to convert

        Returns:
            dict: API-specific request dictionary for Anthropic
        """
        request = {
            "model": generic_request.model,
            "messages": generic_request.messages,
            "max_tokens": generic_request.generation_params.get("max_tokens", 4096),
        }

        # Handle structured output if response_format is provided
        if generic_request.response_format:
            request["system"] = request.get("system", "") + "\nYou must respond in JSON format matching this schema: " + str(generic_request.response_format)
            # Anthropic has native JSON response format in Claude 3.5 and above
            # For those models, we should use the native format
            if "claude-3.5" in generic_request.model or "claude-3-5" in generic_request.model:
                request["response_format"] = {"type": "json_schema", "schema": generic_request.response_format}

        # Handle thinking parameter if provided
        if "thinking" in generic_request.generation_params:
            request["thinking"] = generic_request.generation_params.pop("thinking")

        # Copy over remaining generation parameters
        for key, value in generic_request.generation_params.items():
            if key != "max_tokens":  # Already handled above
                request[key] = value

        return request

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Make a single Anthropic API request.

        Args:
            request (APIRequest): The request to process
            session (aiohttp.ClientSession): Async HTTP session
            status_tracker (OnlineStatusTracker): Tracks request status

        Returns:
            GenericResponse: The response from Anthropic
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with session.post(
            self.url,
            headers=headers,
            json=request.api_specific_request,
            timeout=self.config.request_timeout,
        ) as response_obj:
            response = await response_obj.json()

            if response is None:
                raise Exception("Response is empty")
            elif "error" in response:
                status_tracker.num_api_errors += 1
                error = response["error"]
                error_message = error.get("message", str(error))
                if "rate limit" in error_message.lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
                    # because handle_single_request_with_retries will double count otherwise
                    status_tracker.num_other_errors -= 1
                raise Exception(f"API error: {error}")

            if response_obj.status != 200:
                raise Exception(f"API request failed with status {response_obj.status}: {response}")

            # Handle response based on return_completions_object flag
            if self.config.return_completions_object:
                response_message = dict(response)
            else:
                # Extract content from the first text content block
                content_blocks = response.get("content", [])
                if content_blocks:
                    # Find the first text content block
                    for block in content_blocks:
                        if block.get("type") == "text":
                            response_message = block.get("text", "")
                            break
                    else:
                        # If no text content block found, return the whole message
                        response_message = str(content_blocks)
                else:
                    response_message = ""

            # Get usage information
            usage = response.get("usage", {})
            token_usage = _TokenUsage(
                input=usage.get("input_tokens", 0),
                output=usage.get("output_tokens", 0),
            )

            # Get stop reason
            finish_reason = response.get("stop_reason", "unknown")
            finish_reason = map_finish_reason(finish_reason)

            # Calculate cost
            cost = self.completion_cost(response)

            # Create and return response
            return GenericResponse(
                response_message=response_message,
                response_errors=None,
                raw_request=request.api_specific_request,
                raw_response=response,
                generic_request=request.generic_request,
                created_at=request.created_at,
                finished_at=datetime.datetime.now(),
                token_usage=token_usage,
                response_cost=cost,
                finish_reason=finish_reason,
            )

    def get_token_encoding(self) -> "tiktoken.Encoding":
        """Get the token encoding for Anthropic models."""
        # Anthropic generally uses the cl100k_base encoding, similar to GPT-4
        return tiktoken.get_encoding("cl100k_base")
