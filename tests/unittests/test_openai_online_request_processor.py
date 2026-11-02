from unittest.mock import Mock

import pytest

from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest
from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor, _RateLimitError, fetch_response
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest


class _StubResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status: int, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body if isinstance(self._body, str) else str(self._body)

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _stubbed_session(status: int, body):
    session = Mock()
    session.post.return_value = _StubResponse(status, body)
    return session


def _build_processor() -> OpenAIOnlineRequestProcessor:
    # Constructing the processor fetches rate limits over the network, so bypass __init__.
    processor = object.__new__(OpenAIOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini")
    processor.api_key = "sk-test"
    processor.url = "https://api.openai.com/v1/chat/completions"
    processor._longlived_response = False
    return processor


def _api_request() -> APIRequest:
    return APIRequest(
        task_id=0,
        generic_request=GenericRequest(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], original_row={"q": "hi"}, original_row_idx=0),
        api_specific_request={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        attempts_left=1,
    )


def _tracker_with_history() -> OnlineStatusTracker:
    tracker = OnlineStatusTracker(model="gpt-4o-mini", max_requests_per_minute=None, max_tokens_per_minute=None, total_requests=1)
    tracker.num_api_errors = 3
    tracker.num_other_errors = 2
    tracker.num_rate_limit_errors = 4
    return tracker


def test_special_token_handling():
    """Test that special tokens like <|endoftext|> are handled correctly in token estimation."""
    config = OnlineRequestProcessorConfig(model="gpt-4")
    processor = OpenAIOnlineRequestProcessor(config)

    # Test message containing special token
    messages = [{"role": "user", "content": "Testing <|endoftext|> token"}]

    try:
        total_tokens = processor.estimate_total_tokens(messages)
        assert (total_tokens.input + total_tokens.output) > 0, "Token estimation should return a positive number"
    except ValueError as e:
        if "<|endoftext|>" in str(e):
            pytest.fail("Special token <|endoftext|> should not raise ValueError")
        raise  # Re-raise if it is a different ValueError


@pytest.mark.asyncio
async def test_fetch_response_raises_typed_error_on_429():
    """A 429 raises the typed error with the status and body text."""
    with pytest.raises(_RateLimitError) as exc_info:
        await fetch_response(
            _stubbed_session(429, "rate limit exceeded"),
            url="https://api.openai.com/v1/chat/completions",
            headers={},
            payload={},
            timeout=60,
        )

    error = exc_info.value
    assert error.status == 429
    assert error.response_text == "rate limit exceeded"
    assert str(error) == "API Error: 429 - rate limit exceeded"


@pytest.mark.asyncio
async def test_call_single_request_records_rate_limit_error_on_429():
    """A 429 is recorded exactly like a 200 response carrying a rate-limit error body."""
    processor = _build_processor()
    request = _api_request()
    tracker = _tracker_with_history()

    with pytest.raises(_RateLimitError, match="API Error: 429"):
        await processor.call_single_request(request, _stubbed_session(429, "rate limit exceeded"), tracker)

    assert tracker.time_of_last_rate_limit_error > 0
    assert tracker.num_rate_limit_errors == 5
    assert tracker.num_api_errors == 3
    # A direct call leaves 1; the retry wrapper adds one back in production, netting 2.
    assert tracker.num_other_errors == 1


@pytest.mark.asyncio
async def test_call_single_request_records_same_counters_for_error_body_and_429():
    """A 429 ends at the same counters as a 200 response carrying a rate-limit error body."""
    processor = _build_processor()
    request = _api_request()
    status_tracker = _tracker_with_history()
    body_tracker = _tracker_with_history()

    with pytest.raises(_RateLimitError):
        await processor.call_single_request(request, _stubbed_session(429, "rate limit exceeded"), status_tracker)
    with pytest.raises(Exception, match="API error"):
        await processor.call_single_request(request, _stubbed_session(200, {"error": {"message": "Rate limit exceeded"}}), body_tracker)

    assert (status_tracker.num_rate_limit_errors, status_tracker.num_api_errors, status_tracker.num_other_errors) == (5, 3, 1)
    assert (body_tracker.num_rate_limit_errors, body_tracker.num_api_errors, body_tracker.num_other_errors) == (5, 3, 1)
    assert body_tracker.time_of_last_rate_limit_error > 0


@pytest.mark.asyncio
async def test_call_single_request_raises_original_429_without_status_tracker():
    """The agent processor passes no tracker; the 429 error must still surface unchanged."""
    with pytest.raises(_RateLimitError, match="API Error: 429"):
        await _build_processor().call_single_request(_api_request(), _stubbed_session(429, "rate limit exceeded"), None)


@pytest.mark.asyncio
async def test_call_single_request_does_not_record_rate_limit_on_500():
    """A 500 keeps raising the plain exception and records no rate limit."""
    processor = _build_processor()
    tracker = _tracker_with_history()

    with pytest.raises(Exception, match="API Error: 500 - server error") as exc_info:
        await processor.call_single_request(_api_request(), _stubbed_session(500, "server error"), tracker)

    assert type(exc_info.value) is Exception
    assert tracker.time_of_last_rate_limit_error == 0
    assert (tracker.num_rate_limit_errors, tracker.num_api_errors, tracker.num_other_errors) == (4, 3, 2)
