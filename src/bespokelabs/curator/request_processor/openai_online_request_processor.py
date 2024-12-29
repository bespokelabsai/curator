import asyncio
import datetime
import logging
import os
import re
from typing import Optional, Any, TypeVar

import aiohttp
import requests
import tiktoken
import litellm
import time

from bespokelabs.curator.request_processor.base_online_request_processor import (
    BaseOnlineRequestProcessor,
    APIRequest,
    StatusTracker,
    SECONDS_TO_PAUSE_ON_RATE_LIMIT,
)
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import TokenUsage, GenericResponse

T = TypeVar("T")
logger = logger = logging.getLogger(__name__)


def get_token_encoding_name(model_name: str) -> str:
    """Get the token encoding name for a given model."""
    if model_name.startswith("gpt-4"):
        return "cl100k_base"
    elif model_name.startswith("gpt-3.5"):
        return "cl100k_base"
    else:
        return "cl100k_base"  # Default to cl100k_base


def api_endpoint_from_url(request_url: str) -> str:
    """Extract the API endpoint from the request URL.
    This is used to determine the number of tokens consumed by the request.
    """

    # OpenAI API
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match:
        return match[1]

    # for Azure OpenAI deployment urls
    match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    if match:
        return match[1]

    # Catch all for other API endpoints using OpenAI OpenAPI format
    if "chat/completions" in request_url:
        return "chat/completions"
    elif "completions" in request_url:
        return "completions"
    else:
        raise NotImplementedError(f'API endpoint "{request_url}" not implemented in Curator yet.')


class OpenAIOnlineRequestProcessor(BaseOnlineRequestProcessor):
    """OpenAI-specific implementation of the OnlineRequestProcessor.

    Handles API requests to OpenAI's chat completion endpoints with rate limiting,
    token counting, and error handling specific to OpenAI's API.

    Note:
        - Supports both OpenAI and Azure OpenAI endpoints
        - Automatically detects and respects API rate limits
        - Handles token counting using tiktoken
        - Supports structured output via JSON schema
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = os.getenv("OPENAI_API_KEY"),
        url: str = "https://api.openai.com/v1/chat/completions",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_requests_per_minute: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None,
        require_all_responses: bool = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            require_all_responses=require_all_responses,
            max_retries=max_retries,
        )
        self.url = url
        self.api_key = api_key
        self.token_encoding = tiktoken.get_encoding(get_token_encoding_name(model))
        self.header_based_max_requests_per_minute, self.header_based_max_tokens_per_minute = (
            self.get_header_based_rate_limits()
        )

    def get_header_based_rate_limits(self) -> tuple[int, int]:
        """Get rate limits from OpenAI API headers.

        Returns:
            tuple[int, int]: Contains 'max_requests_per_minute' and 'max_tokens_per_minute'

        Note:
            - Makes a dummy request to get actual rate limits
        """
        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API Key - Please set OPENAI_API_KEY in your environment vars"
            )

        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": []},
        )
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
        try:
            return litellm.get_max_tokens(model=self.model) // 4
        except Exception:
            return 0

    def estimate_total_tokens(self, messages: list) -> int:
        """Estimate total tokens for a request using OpenAI's token counting rules.

        Args:
            messages (list): List of message dictionaries with role and content

        Returns:
            int: Estimated total tokens including message formatting tokens

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
                    num_tokens += len(self.token_encoding.encode(str(value)))
                except TypeError:
                    logger.warning(
                        f"Failed to encode value {value} with tiktoken. Assuming 1 token per 4 chars."
                    )
                    num_tokens += len(str(value)) // 4
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token

        num_tokens += 2  # every reply is primed with <im_start>assistant
        output_tokens = self.estimate_output_tokens()
        return num_tokens + output_tokens

    def check_structured_output_support(self) -> bool:
        """Check if the model supports structured output based on model name and date.

        Returns:
            bool: True if model supports structured output, False otherwise

        Note:
            Supports:
            - gpt-4o-mini with date >= 2024-07-18 or latest
            - gpt-4o with date >= 2024-08-06 or latest
        """
        model_name = self.model.lower()

        # Check gpt-4o-mini support
        if model_name == "gpt-4o-mini":  # Latest version
            return True
        if "gpt-4o-mini-" in model_name:
            mini_date = datetime.datetime.strptime(model_name.split("gpt-4o-mini-")[1], "%Y-%m-%d")
            if mini_date >= datetime(2024, 7, 18):
                return True

        # Check gpt-4o and o1 support
        if model_name in ["gpt-4o", "o1"]:  # Latest version
            return True
        if "gpt-4o-" in model_name:
            base_date = datetime.datetime.strptime(model_name.split("gpt-4o-")[1], "%Y-%m-%d")
            if base_date >= datetime.datetime(2024, 8, 6):
                return True
        if "o1-" in model_name:
            base_date = datetime.datetime.strptime(model_name.split("o1-")[1], "%Y-%m-%d")
            if base_date >= datetime.datetime(
                2024, 12, 17
            ):  # Support o1 dated versions from 2024-12-17
                return True

        return False

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Create an OpenAI-specific request from a generic request.

        Args:
            generic_request (GenericRequest): Generic request object

        Returns:
            dict: OpenAI API-compatible request dictionary

        Note:
            - Handles JSON schema response format if specified
            - Applies optional parameters (temperature, top_p, etc.)
            - Maintains compatibility with both chat and completion endpoints
        """
        request: dict[str, Any] = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }
        if generic_request.response_format:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output_schema",
                    "schema": generic_request.response_format,
                },
            }

        if self.temperature is not None:
            request["temperature"] = self.temperature

        if self.top_p is not None:
            request["top_p"] = self.top_p

        if self.presence_penalty is not None:
            request["presence_penalty"] = self.presence_penalty

        if self.frequency_penalty is not None:
            request["frequency_penalty"] = self.frequency_penalty

        return request

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: StatusTracker,
    ) -> GenericResponse:
        """Make a single OpenAI API request.

        Args:
            request (APIRequest): The request to process
            session (aiohttp.ClientSession): Async HTTP session
            status_tracker (StatusTracker): Tracks request status

        Returns:
            GenericResponse: The response from OpenAI
        """
        api_endpoint = api_endpoint_from_url(self.url)
        request_header = {"Authorization": f"Bearer {self.api_key}"}
        if "/deployments" in self.url:  # Azure deployment
            request_header = {"api-key": f"{self.api_key}"}

        async with session.post(
            self.url,
            headers=request_header,
            json=request.api_specific_request,
            timeout=self.timeout,
        ) as response_obj:
            response = await response_obj.json()

            if "error" in response:
                status_tracker.num_api_errors += 1
                error = response["error"]
                if "rate limit" in error.get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
                    # because handle_single_request_with_retries will double count otherwise
                    status_tracker.num_other_errors -= 1
                    # Get retry-after from headers or use default
                    retry_after = float(response_obj.headers.get("retry-after", SECONDS_TO_PAUSE_ON_RATE_LIMIT))
                    status_tracker.retry_after_seconds = retry_after
                    logger.warning(f"Rate limit reached. Will retry after {retry_after} seconds")
                raise Exception(f"API error: {error}")

            if response_obj.status != 200:
                raise Exception(f"API request failed with status {response_obj.status}: {response}")

            response_message = response["choices"][0]["message"]["content"]
            usage = response["usage"]
            token_usage = TokenUsage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )

            # Calculate cost using litellm
            cost = litellm.completion_cost(completion_response=response)

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
            )
