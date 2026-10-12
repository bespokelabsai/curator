import asyncio
import datetime
from collections import Counter, deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.anthropic_online_request_processor import AnthropicOnlineRequestProcessor
from bespokelabs.curator.status_tracker import online_status_tracker as tracker_module
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker, TokenLimitStrategy
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


@pytest.mark.asyncio
async def test_aiohttp_connector_trusts_env_proxy():
    """Ensure the shared aiohttp session honors HTTP(S)_PROXY/ALL_PROXY env vars.

    aiohttp.ClientSession ignores proxy environment variables unless trust_env=True
    is set explicitly, unlike requests/httpx. Without it, requests (e.g. to the
    Anthropic API) silently bypass any configured system proxy. See issue #699.

    aiohttp_connector doesn't touch instance state, so we bypass __init__ (which
    makes a live network call to fetch header-based rate limits) via object.__new__.
    """
    processor = object.__new__(AnthropicOnlineRequestProcessor)

    async with processor.aiohttp_connector(tcp_limit=1) as session:
        assert session.trust_env is True


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [TokenLimitStrategy.combined, TokenLimitStrategy.seperate])
@pytest.mark.parametrize("retry_oversized", [False, True])
async def test_oversized_request_and_following_rows_complete(tmp_path, monkeypatch, strategy, retry_oversized):
    """Exercise the real dispatch/retry loops without API calls or refill delays."""
    clock = SimpleNamespace(now=0.0, sleeps=0)
    monkeypatch.setattr(tracker_module, "time", SimpleNamespace(time=lambda: clock.now))
    real_sleep = asyncio.sleep

    async def refill_sleep(seconds):
        clock.sleeps += 1
        assert clock.sleeps < 20, "Request never acquired token capacity"
        clock.now += 60
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", refill_sleep)
    processor = object.__new__(AnthropicOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini", max_retries=1)
    processor.manual_max_concurrent_requests = 1
    processor._semaphore = None
    processor._tracker_console = None
    processor._output_tokens_window = deque()
    processor._viewer_client = AsyncMock()
    processor.prompt_formatter = Mock()
    processor.validate_existing_response_file = Mock(return_value=(set(), 0))
    processor.create_api_specific_request_online = Mock(return_value={})
    processor.estimate_total_tokens = Mock(
        side_effect=lambda messages: _TokenUsage(input=150, output=75) if messages[0]["content"] == "oversized" else _TokenUsage(input=10, output=5)
    )
    processor.append_generic_response = AsyncMock()
    calls = Counter()

    async def respond(request, **kwargs):
        row = request.generic_request.original_row_idx
        calls[row] += 1
        if row == 1 and retry_oversized and calls[row] == 1:
            raise RuntimeError("Temporary provider error")
        return GenericResponse(
            response_message="answer",
            raw_response={},
            generic_request=request.generic_request,
            created_at=request.created_at,
            finished_at=datetime.datetime.now(),
            token_usage=_TokenUsage(input=10, output=5),
            response_cost=0,
        )

    processor.call_single_request = AsyncMock(side_effect=respond)
    tracker = OnlineStatusTracker(
        model="gpt-4o-mini",
        token_limit_strategy=strategy,
        max_requests_per_minute=60,
        max_tokens_per_minute=100 if strategy == TokenLimitStrategy.combined else _TokenUsage(input=100, output=50),
        last_update_time=0,
        total_requests=3,
    )
    tracker.start_tracker = Mock()
    tracker.stop_tracker = Mock()
    tracker.update_stats = Mock()
    tracker.update_cost_projection = Mock()
    requests_file = tmp_path / "requests_0.jsonl"
    requests_file.write_text(
        "\n".join(
            GenericRequest(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], original_row={"q": prompt}, original_row_idx=index
            ).model_dump_json()
            for index, prompt in enumerate(["first", "oversized", "last"])
        )
        + "\n"
    )
    await asyncio.wait_for(processor.process_requests_from_file(str(requests_file), str(tmp_path / "responses_0.jsonl"), tracker), timeout=5)
    assert calls == {0: 1, 1: 2 if retry_oversized else 1, 2: 1}
    assert tracker.num_tasks_started == tracker.num_tasks_succeeded == 3
    assert tracker.num_tasks_in_progress == tracker.num_tasks_failed == 0
    assert processor.append_generic_response.await_count == 3
