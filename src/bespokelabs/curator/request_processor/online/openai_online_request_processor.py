import datetime
import os
import time
from typing import TypeVar

import aiohttp
import litellm
import requests
import tiktoken

from bespokelabs.curator.cost import cost_processor_factory
from bespokelabs.curator.file_utilities import get_base64_size
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor import openai_request_mixin
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest, BaseOnlineRequestProcessor
from bespokelabs.curator.request_processor.openai_request_mixin import OpenAIRequestMixin
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse, _TokenUsage

T = TypeVar("T")

_DEFAULT_OPENAI_URL: str = "https://api.openai.com/v1/chat/completions"

_OPENAI_MULTIMODAL_SUPPORTED_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4o-vision"}
_OPENAI_ALLOWED_IMAGE_SIZE_MB = 20


class OpenAIOnlineRequestProcessor(BaseOnlineRequestProcessor, OpenAIRequestMixin):
    """OpenAI-specific implementation of the OnlineRequestProcessor.

    Handles API requests to OpenAI's chat completion endpoints with rate limiting,
    token counting, and error handling specific to OpenAI's API.

    Note:
        - Supports both OpenAI and Azure OpenAI endpoints
        - Automatically detects and respects API rate limits
        - Handles token counting using tiktoken
        - Supports structured output via JSON schema
    """

    _DEFAULT_COMPLETION_SUFFIX = "/chat/completions"

    def __init__(self, config: OnlineRequestProcessorConfig, compatible_provider: str = None):
        """Initialize the OpenAIOnlineRequestProcessor."""
        super().__init__(config)
        self._compatible_provider = compatible_provider or self.backend
        self._cost_processor = cost_processor_factory(config=config, backend=self._compatible_provider)

        if self.config.base_url is None:
            if "OPENAI_BASE_URL" in os.environ:
                key_url = os.environ["OPENAI_BASE_URL"].strip().rstrip("/")
                self.url = key_url + self._DEFAULT_COMPLETION_SUFFIX
            else:
                self.url = _DEFAULT_OPENAI_URL
        else:
            self.url = self.config.base_url + self._DEFAULT_COMPLETION_SUFFIX

        if self.config.base_url == "https://api.deepseek.com":
            # DeepSeek does not return rate limits in headers
            # https://api-docs.deepseek.com/quick_start/rate_limit.
            # And sending an empty request for rate limits results in a 400 error like this:
            # {'error': {'message': 'Empty input messages', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
            self.api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            self.header_based_max_requests_per_minute, self.header_based_max_tokens_per_minute = self.get_header_based_rate_limits()
        self.token_encoding = self.get_token_encoding()

    @property
    def backend(self):
        """Backend property."""
        return "openai"

    @property
    def compatible_provider(self) -> str:
        """Compatible provider property."""
        return self._compatible_provider

    def get_header_based_rate_limits(self) -> tuple[int, int]:
        """Get rate limits from OpenAI API headers.

        Returns:
            tuple[int, int]: Contains 'max_requests_per_minute' and 'max_tokens_per_minute'

        Note:
            - Makes a dummy request to get actual rate limits
        """
        if not self.api_key:
            raise ValueError("Missing OpenAI API Key - Please set OPENAI_API_KEY in your environment vars")

        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.config.model, "messages": []},
        )
        from bespokelabs.curator.cost import RATE_LIMIT_HEADER

        for provider in RATE_LIMIT_HEADER:
            if provider in self.url:
                rate_limit_header = RATE_LIMIT_HEADER[provider]
                rpm_key = rate_limit_header["request-key"]
                tpm_key = rate_limit_header["token-key"]
                rpm = float(response.headers.get(rpm_key["key"], 0))
                tpm = float(response.headers.get(tpm_key["key"], 0))
                if rpm_key.get("type") == "rps":
                    rpm = rpm * 60
                if tpm_key.get("type") == "tps":
                    tpm = tpm * 60
                return int(rpm), int(tpm)

        rpm = int(response.headers.get("x-ratelimit-limit-requests", 0))
        tpm = int(response.headers.get("x-ratelimit-limit-tokens", 0))

        return rpm, tpm

    def estimate_output_tokens(self) -> int:
        """Estimate number of tokens in the response.

        Returns:
            int: Estimated number of output tokens

        Note:
            Default implementation returns a conservative estimate.
            Override this method for more accurate model-specific estimates.
        """
        if self.config.model in litellm.model_cost:
            return litellm.get_max_tokens(model=self.config.model) // 4
        else:
            return 4096

    def estimate_total_tokens(self, messages: list) -> _TokenUsage:
        """Estimate total tokens for a request using OpenAI's token counting rules.

        Args:
            messages (list): List of message dictionaries with role and content

        Returns:
            _TokenUsage: Estimated input and output tokens including message formatting tokens

        Note:
            Includes:
            - 4 tokens per message for formatting
            - Role/name tokens
            - Content tokens
            - 2 tokens for assistant reply priming
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                try:
                    num_tokens += openai_request_mixin.calculate_input_tokens(value, self.token_encoding)
                except TypeError:
                    logger.warning(f"Failed to encode value {value} with tiktoken. Assuming 1 token per 4 chars.")
                    num_tokens += len(str(value)) // 4
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token

        num_tokens += 2  # every reply is primed with <im_start>assistant
        output_tokens = self.estimate_output_tokens()
        return _TokenUsage(input=num_tokens, output=output_tokens)

    def check_structured_output_support(self) -> bool:
        """Check if the model supports structured output based on model name and date.

        Returns:
            bool: True if model supports structured output, False otherwise

        Note:
            Supports:
            - gpt-4o-mini with date >= 2024-07-18 or latest
            - gpt-4o with date >= 2024-08-06 or latest
            - o1 with date >= 2024-12-17 or latest
            - o3-mini with date >= 2025-01-31 or latest
        """
        model_name = self.config.model.lower()

        # Check gpt-4o-mini support.
        if model_name == "gpt-4o-mini":  # Latest version
            return True
        if "gpt-4o-mini-" in model_name:
            mini_date = datetime.datetime.strptime(model_name.split("gpt-4o-mini-")[1], "%Y-%m-%d")
            if mini_date >= datetime(2024, 7, 18):
                return True

        # Check gpt-4o, o1, o3-mini support.
        if model_name in ["gpt-4o", "o1"]:  # Latest version
            return True
        if "gpt-4o-" in model_name:
            base_date = datetime.datetime.strptime(model_name.split("gpt-4o-")[1], "%Y-%m-%d")
            if base_date >= datetime.datetime(2024, 8, 6):
                return True
        if "o1-" in model_name:
            base_date = datetime.datetime.strptime(model_name.split("o1-")[1], "%Y-%m-%d")
            if base_date >= datetime.datetime(2024, 12, 17):  # Support o1 dated versions from 2024-12-17
                return True
        if "o3-mini-" in model_name:
            base_date = datetime.datetime.strptime(model_name.split("o3-mini-")[1], "%Y-%m-%d")
            if base_date >= datetime.datetime(2025, 1, 31):  # Support o3-mini dated versions from 2025-01-31
                return True

        return False

    def file_upload_limit_check(self, base64_image: str) -> None:
        """Check if the image size is within the allowed limit."""
        mb = get_base64_size(base64_image)
        if mb > _OPENAI_ALLOWED_IMAGE_SIZE_MB:
            raise ValueError(f"Image size is {mb} MB, which is greater than the allowed size of {_OPENAI_ALLOWED_IMAGE_SIZE_MB} MB.")

    @property
    def _multimodal_prompt_supported(self) -> bool:
        return any(self.config.model.startswith(k) for k in _OPENAI_MULTIMODAL_SUPPORTED_MODELS)

    def create_api_specific_request_online(self, generic_request: GenericRequest) -> dict:
        """Create an OpenAI-specific request from a generic request.

        Delegates to the mixin implementation.
        """
        return OpenAIRequestMixin.create_api_specific_request_online(self, generic_request)

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Make a single OpenAI API request.

        Args:
            request (APIRequest): The request to process
            session (aiohttp.ClientSession): Async HTTP session
            status_tracker (OnlineStatusTracker): Tracks request status

        Returns:
            GenericResponse: The response from OpenAI
        """
        request_header = {"Authorization": f"Bearer {self.api_key}"}
        if "/deployments" in self.url:  # Azure deployment
            request_header = {"api-key": f"{self.api_key}"}
        async with session.post(
            self.url,
            headers=request_header,
            json=request.api_specific_request,
            timeout=self.config.request_timeout,
        ) as response_obj:
            response = await response_obj.json()

            if response is None:
                raise Exception("Response is empty")
            elif "error" in response:
                status_tracker.num_api_errors += 1
                error = response["error"]
                error_message = error if isinstance(error, str) else error.get("message", "")
                if "rate limit" in error_message.lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
                    # because handle_single_request_with_retries will double count otherwise
                    status_tracker.num_other_errors -= 1
                raise Exception(f"API error: {error}")

            if response_obj.status != 200:
                raise Exception(f"API request failed with status {response_obj.status}: {response}")

            if self.config.return_completions_object:
                response_message = dict(response)
            else:
                response_message = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason", "unknown")
            usage = response["usage"]
            token_usage = _TokenUsage(
                input=usage["prompt_tokens"],
                output=usage["completion_tokens"],
                total=usage["total_tokens"],
            )

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

    def get_token_encoding(self) -> str:
        """Get the token encoding name for a given model."""
        if self.config.model.startswith("gpt-4"):
            name = "cl100k_base"
        elif self.config.model.startswith("gpt-3.5"):
            name = "cl100k_base"
        else:
            name = "cl100k_base"  # Default to cl100k_base

        return tiktoken.get_encoding(name)
