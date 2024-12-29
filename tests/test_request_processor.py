import asyncio
import os
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from litellm.exceptions import RateLimitError
from bespokelabs.curator.request_processor.base_online_request_processor import (
    SECONDS_TO_PAUSE_ON_RATE_LIMIT,
    StatusTracker,
    APIRequest,
)
from bespokelabs.curator.request_processor.openai_online_request_processor import (
    OpenAIOnlineRequestProcessor,
)
from bespokelabs.curator.request_processor.litellm_online_request_processor import (
    LiteLLMOnlineRequestProcessor,
)
from bespokelabs.curator.request_processor.generic_request import GenericRequest


@pytest.fixture
def mock_session():
    """Create a mock aiohttp ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    return session


@pytest.fixture
def status_tracker():
    """Create a StatusTracker instance."""
    return StatusTracker(
        num_tasks_started=1,
        max_requests_per_minute=60,
        max_tokens_per_minute=40000,
    )


@pytest.mark.asyncio
async def test_openai_rate_limit_retry_after(mock_session, status_tracker):
    """Test OpenAI processor respects Retry-After header on rate limit."""
    retry_after = 5.5  # Custom retry time in seconds

    # Mock response with rate limit error and Retry-After header
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.headers = {"retry-after": str(retry_after)}
    mock_response.json.return_value = {"error": {"message": "rate limit exceeded"}}
    mock_session.post.return_value.__aenter__.return_value = mock_response

    processor = OpenAIOnlineRequestProcessor(
        model="gpt-4",
        api_key="test-key",
    )

    generic_request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        original_row={"text": "test"},
        original_row_idx=0,
    )

    request = APIRequest(
        task_id=0,
        generic_request=generic_request,
        api_specific_request=processor.create_api_specific_request(generic_request),
        attempts_left=10,
        prompt_formatter=None,
    )

    # Verify rate limit handling
    start_time = time.time()
    with pytest.raises(Exception) as exc_info:
        await processor.call_single_request(
            request=request,
            session=mock_session,
            status_tracker=status_tracker,
        )

    assert "rate limit" in str(exc_info.value).lower()
    assert status_tracker.retry_after_seconds == retry_after
    assert status_tracker.time_of_last_rate_limit_error >= start_time


@pytest.mark.asyncio
async def test_litellm_rate_limit_retry_after(mock_session, status_tracker):
    """Test LiteLLM processor respects Retry-After from RateLimitError."""
    retry_after = 3.5  # Custom retry time in seconds

    # Mock LiteLLM to raise RateLimitError with retry_after
    error = RateLimitError(message="Rate limit exceeded", llm_provider="openai", model="gpt-4")
    error.retry_after = retry_after
    error.headers = {"retry-after": str(retry_after)}

    # Mock both the test_call during initialization and the actual completion call
    mock_headers = {
        "headers": {
            "x-ratelimit-remaining": "1",
            "x-ratelimit-limit": "100",
            "x-ratelimit-reset": "1000",
        }
    }
    with patch.object(LiteLLMOnlineRequestProcessor, "test_call", return_value=mock_headers), patch(
        "litellm.acompletion", side_effect=error
    ):
        processor = LiteLLMOnlineRequestProcessor(
            model="gpt-4",
        )

        generic_request = GenericRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            original_row={"text": "test"},
            original_row_idx=0,
        )

        request = APIRequest(
            task_id=0,
            generic_request=generic_request,
            api_specific_request=processor.create_api_specific_request(generic_request),
            attempts_left=10,
            prompt_formatter=None,
        )

        # Verify rate limit handling
        start_time = time.time()
        with pytest.raises(RateLimitError):
            await processor.call_single_request(
                request=request,
                session=mock_session,
                status_tracker=status_tracker,
            )

        assert status_tracker.retry_after_seconds == retry_after
        assert status_tracker.time_of_last_rate_limit_error >= start_time


@pytest.mark.asyncio
async def test_rate_limit_fallback_timeout(mock_session, status_tracker):
    """Test fallback to default timeout when no Retry-After header."""
    # Mock response with rate limit error but no Retry-After header
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.headers = {}  # No retry-after header
    mock_response.json.return_value = {"error": {"message": "rate limit exceeded"}}
    mock_session.post.return_value.__aenter__.return_value = mock_response

    processor = OpenAIOnlineRequestProcessor(
        model="gpt-4",
        api_key="test-key",
    )

    generic_request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        original_row={"text": "test"},
        original_row_idx=0,
    )

    request = APIRequest(
        task_id=0,
        generic_request=generic_request,
        api_specific_request=processor.create_api_specific_request(generic_request),
        attempts_left=10,
        prompt_formatter=None,
    )

    # Verify fallback behavior
    start_time = time.time()
    with pytest.raises(Exception) as exc_info:
        await processor.call_single_request(
            request=request,
            session=mock_session,
            status_tracker=status_tracker,
        )

    assert "rate limit" in str(exc_info.value).lower()
    assert status_tracker.retry_after_seconds == SECONDS_TO_PAUSE_ON_RATE_LIMIT
    assert status_tracker.time_of_last_rate_limit_error >= start_time
