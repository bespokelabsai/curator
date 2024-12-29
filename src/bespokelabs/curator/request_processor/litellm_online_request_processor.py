import logging
from typing import Optional
import aiohttp
import litellm
from litellm import get_supported_openai_params
import datetime
import instructor
from bespokelabs.curator.request_processor.base_online_request_processor import (
    BaseOnlineRequestProcessor,
    APIRequest,
    StatusTracker,
    SECONDS_TO_PAUSE_ON_RATE_LIMIT,
)
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import TokenUsage, GenericResponse
from pydantic import BaseModel
import time

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


class LiteLLMOnlineRequestProcessor(BaseOnlineRequestProcessor):
    """LiteLLM implementation of the OnlineRequestProcessor for multi-provider LLM support.

    This processor uses LiteLLM to handle requests across different LLM providers (OpenAI, Anthropic, etc.)
    with unified interface and structured output support via instructor.

    Features:
        - Multi-provider support through LiteLLM
        - Structured output via instructor
        - Automatic token counting and rate limiting
        - Cost tracking per request

    Attributes:
        model (str): The model identifier (e.g., "gpt-4", "claude-2")
        temperature (Optional[float]): Temperature for response randomness
        top_p (Optional[float]): Top-p sampling parameter
        presence_penalty (Optional[float]): Presence penalty for response diversity
        frequency_penalty (Optional[float]): Frequency penalty for response diversity
        client: Instructor-wrapped LiteLLM client for structured outputs
    """

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_requests_per_minute: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None,
        require_all_responses: Optional[bool] = None,
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
        self.client = instructor.from_litellm(litellm.acompletion)
        (
            self.header_based_max_requests_per_minute,
            self.header_based_max_tokens_per_minute,
        ) = self.get_header_based_rate_limits()

    def check_structured_output_support(self):
        """Verify if the model supports structured output via instructor.

        Tests the model's capability to handle structured output by making a test request
        with a simple schema.

        Returns:
            bool: True if structured output is supported, False otherwise

        Note:
            - Uses a simple User schema as test case
            - Logs detailed information about support status
            - Required for models that will use JSON schema responses
        """

        class User(BaseModel):
            name: str
            age: int

        try:
            client = instructor.from_litellm(litellm.completion)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Jason is 25 years old."}],
                response_model=User,
            )
            logger.info(f"Check instructor structure output response: {response}")
            assert isinstance(response, User)
            logger.info(
                f"Model {self.model} supports structured output via instructor, response: {response}"
            )
            return True
        except instructor.exceptions.InstructorRetryException as e:
            if "litellm.AuthenticationError" in str(e):
                logger.warning(f"Please provide a valid API key for model {self.model}.")
                raise e
            else:
                logger.warning(
                    f"Model {self.model} does not support structured output via instructor: {e} {type(e)} {e.__cause__}"
                )
                return False

    def estimate_output_tokens(self) -> int:
        """Estimate the number of tokens in the model's response.

        Uses LiteLLM's get_max_tokens and applies a conservative estimate
        by dividing by 4 to avoid hitting context limits.

        Returns:
            int: Estimated number of output tokens

        Note:
            Falls back to 0 if token estimation fails
        """
        try:
            return litellm.get_max_tokens(model=self.model) // 4
        except Exception:
            return 0

    def estimate_total_tokens(self, messages: list) -> int:
        """Calculate the total token usage for a request.

        Uses LiteLLM's token_counter for accurate input token counting
        and adds estimated output tokens.

        Args:
            messages (list): List of message dictionaries

        Returns:
            int: Total estimated tokens (input + output)
        """
        input_tokens = litellm.token_counter(model=self.model, messages=messages)
        output_tokens = self.estimate_output_tokens()
        return input_tokens + output_tokens

    def test_call(self):
        completion = litellm.completion(
            model=self.model,
            messages=[
                {"role": "user", "content": "hi"}
            ],  # Some models (e.g. Claude) require an non-empty message to get rate limits.
        )
        # Try the method of caculating cost
        try:
            litellm.completion_cost(completion_response=completion.model_dump())
        except Exception as e:
            # We should ideally not catch a catch-all exception here. But litellm is not throwing any specific error.
            logger.warning(f"LiteLLM does not support cost estimation for model {self.model}: {e}")

        headers = completion._hidden_params.get("additional_headers", {})
        logger.info(f"Test call headers: {headers}")
        return headers

    def get_header_based_rate_limits(self) -> tuple[int, int]:
        """Retrieve rate limits from the LLM provider via LiteLLM.

        Returns:
            tuple[int, int]: Contains 'max_requests_per_minute' and 'max_tokens_per_minute'

        Note:
            - Makes a test request to get rate limit information from response headers.
            - Some providers (e.g., Claude) require non-empty messages
        """
        logger.info(f"Getting rate limits for model: {self.model}")

        headers = self.test_call()
        rpm = int(headers.get("x-ratelimit-limit-requests", 0))
        tpm = int(headers.get("x-ratelimit-limit-tokens", 0))

        return rpm, tpm

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Convert a generic request into a LiteLLM-compatible format.

        Checks supported parameters for the specific model and only includes
        applicable parameters.

        Args:
            generic_request (GenericRequest): The generic request to convert

        Returns:
            dict: LiteLLM-compatible request parameters

        Note:
            Uses LiteLLM's get_supported_openai_params to check parameter support
        """
        # Get supported parameters for this model
        supported_params = get_supported_openai_params(model=self.model)
        request = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }

        # Only add parameters that are supported by this model
        if "temperature" in supported_params and self.temperature is not None:
            request["temperature"] = self.temperature

        if "top_p" in supported_params and self.top_p is not None:
            request["top_p"] = self.top_p

        if "presence_penalty" in supported_params and self.presence_penalty is not None:
            request["presence_penalty"] = self.presence_penalty

        if "frequency_penalty" in supported_params and self.frequency_penalty is not None:
            request["frequency_penalty"] = self.frequency_penalty

        # Add safety settings for Gemini models
        if "gemini" in generic_request.model.lower():
            request["safety_settings"] = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                    "threshold": "BLOCK_NONE",
                },
            ]

        return request

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: StatusTracker,
    ) -> GenericResponse:
        """Make a single request through LiteLLM.

        Handles both structured and unstructured outputs, tracks token usage
        and costs.

        Args:
            request (APIRequest): Request to process
            session (aiohttp.ClientSession): Async HTTP session
            status_tracker (StatusTracker): Tracks request status

        Returns:
            GenericResponse: The response from LiteLLM
        """
        # Get response directly without extra logging
        try:
            if request.generic_request.response_format:
                (
                    response,
                    completion_obj,
                ) = await self.client.chat.completions.create_with_completion(
                    **request.api_specific_request,
                    response_model=request.prompt_formatter.response_format,
                    timeout=self.timeout,
                )
                response_message = (
                    response.model_dump() if hasattr(response, "model_dump") else response
                )
            else:
                completion_obj = await litellm.acompletion(
                    **request.api_specific_request, timeout=self.timeout
                )
                response_message = completion_obj["choices"][0]["message"]["content"]
        except litellm.RateLimitError as e:
            status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_rate_limit_errors += 1
            # because handle_single_request_with_retries will double count otherwise
            status_tracker.num_api_errors -= 1
            # Get retry-after from error headers or use default
            retry_after = float(
                getattr(e, "retry_after", None)
                or getattr(e, "headers", {}).get("retry-after", SECONDS_TO_PAUSE_ON_RATE_LIMIT)
            )
            status_tracker.retry_after_seconds = retry_after
            logger.warning(f"Rate limit reached. Will retry after {retry_after} seconds")
            raise e

        # Extract token usage
        usage = completion_obj.usage if hasattr(completion_obj, "usage") else {}
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        # Calculate cost using litellm
        try:
            cost = litellm.completion_cost(completion_response=completion_obj.model_dump())
        except Exception as e:
            # We should ideally not catch a catch-all exception here. But litellm is not throwing any specific error.
            cost = 0

        finish_reason = completion_obj.choices[0].finish_reason
        invalid_finish_reasons = ["length", "content_filter"]
        if finish_reason in invalid_finish_reasons:
            logger.debug(
                f"Invalid finish_reason {finish_reason}. Raw response {completion_obj.model_dump()} for request {request.generic_request.messages}"
            )
            raise ValueError(f"finish_reason was {finish_reason}")

        if response_message is None:
            raise ValueError(
                f"response_message was None with raw response {completion_obj.model_dump()}"
            )

        # Create and return response
        return GenericResponse(
            response_message=response_message,
            response_errors=None,
            raw_request=request.api_specific_request,
            raw_response=completion_obj.model_dump(),
            generic_request=request.generic_request,
            created_at=request.created_at,
            finished_at=datetime.datetime.now(),
            token_usage=token_usage,
            response_cost=cost,
        )
