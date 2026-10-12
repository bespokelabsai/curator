import asyncio
import datetime
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
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


@pytest.fixture
def online_cache_processor(tmp_path):
    """Use real dispatch, response persistence, and cache validation with a fake API."""
    processor = object.__new__(AnthropicOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini", max_retries=0, require_all_responses=True)
    processor.working_dir = str(tmp_path)
    processor.manual_max_concurrent_requests = 1
    processor._semaphore = None
    processor._tracker_console = None
    processor._output_tokens_window = deque()
    processor._viewer_client = AsyncMock()
    processor.prompt_formatter = PromptFormatter(
        model_name="gpt-4o-mini",
        prompt_func=lambda row: row["q"],
        parse_func=lambda row, response: {"q": row["q"], "a": response},
    )
    processor.create_api_specific_request_online = Mock(return_value={})

    async def respond(request, **kwargs):
        return GenericResponse(
            response_message=f"answer to {request.generic_request.original_row['q']}",
            raw_response={},
            generic_request=request.generic_request,
            created_at=request.created_at,
            finished_at=datetime.datetime.now(),
            token_usage=_TokenUsage(input=5, output=3),
            response_cost=0,
        )

    processor.call_single_request = AsyncMock(side_effect=respond)
    requests_file = tmp_path / "requests_0.jsonl"
    requests_file.write_text(
        "\n".join(
            GenericRequest(
                model="gpt-4o-mini", messages=[{"role": "user", "content": f"q{i}"}], original_row={"q": f"q{i}"}, original_row_idx=i
            ).model_dump_json()
            for i in range(10)
        )
        + "\n"
    )
    return processor


async def process_cached_requests(processor):
    tracker = OnlineStatusTracker(model="gpt-4o-mini", max_requests_per_minute=None, max_tokens_per_minute=None, total_requests=10)
    tracker.start_tracker = Mock()
    tracker.stop_tracker = Mock()
    tracker.update_stats = Mock()
    tracker.update_cost_projection = Mock()
    await asyncio.wait_for(
        processor.process_requests_from_file(
            str(Path(processor.working_dir) / "requests_0.jsonl"), str(Path(processor.working_dir) / "responses_0.jsonl"), tracker
        ),
        timeout=5,
    )
    return tracker


@pytest.mark.asyncio
@pytest.mark.parametrize("parse_behavior", ["error", "filter_some", "filter_all"])
async def test_reparse_cached_responses_without_api_calls(online_cache_processor, tmp_path, parse_behavior):
    processor = online_cache_processor
    fixed_parse = processor.prompt_formatter.parse_func

    def parse(row, response):
        if parse_behavior == "filter_all" or row["q"] in {"q3", "q7"}:
            if parse_behavior == "error":
                raise KeyError("parse typo")
            return []
        return fixed_parse(row, response)

    processor.prompt_formatter.parse_func = parse
    tracker = await process_cached_requests(processor)
    assert processor.call_single_request.await_count == 10
    responses_file = tmp_path / "responses_0.jsonl"
    responses = [GenericResponse.model_validate_json(line) for line in responses_file.read_text().splitlines()]
    assert len(responses) == 10
    assert {response.generic_request.original_row_idx for response in responses} == set(range(10))
    assert all(response.response_message == f"answer to {response.generic_request.original_row['q']}" for response in responses)
    assert tracker.num_parsed_responses == (0 if parse_behavior == "filter_all" else 8)
    assert processor.viewer_client.stream_response.await_count == tracker.num_parsed_responses

    if parse_behavior == "error":
        assert responses[3].parsed_response_message is None
        with pytest.raises(ValueError, match="Some requests failed"):
            processor.create_dataset_files("original-parse")
    else:
        assert responses[3].parsed_response_message == []
        dataset = processor.create_dataset_files("original-parse")
        assert len(dataset) == (0 if parse_behavior == "filter_all" else 8)
        assert (tmp_path / "failed_requests.jsonl").read_text() == ""
        assert len(processor.attempt_loading_cached_dataset("original-parse")) == len(dataset)

    # Resume with the same parser: all API results, including filtered/rejected
    # responses, must already be complete and require no additional calls.
    tracker = await process_cached_requests(processor)
    assert tracker.num_tasks_already_completed == 10
    assert processor.call_single_request.await_count == 10

    # Changing only the parser should recover every row from the original results.
    processor.prompt_formatter.parse_func = fixed_parse
    await process_cached_requests(processor)
    dataset = processor.create_dataset_files("fixed-parse")
    assert dataset["q"] == [f"q{i}" for i in range(10)]
    assert dataset["a"] == [f"answer to q{i}" for i in range(10)]
    assert processor.call_single_request.await_count == 10
    assert len(responses_file.read_text().splitlines()) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_parsing_does_not_mutate_cached_raw_response(online_cache_processor, tmp_path, raises):
    processor = online_cache_processor

    def mutate_parse(row, response):
        response.pop("answer")
        row.pop("q")
        if raises:
            raise ValueError("Parser failed after mutation")
        return []

    processor.prompt_formatter.parse_func = mutate_parse
    now = datetime.datetime.now()
    response = GenericResponse(
        response_message={"answer": "original"},
        raw_response={},
        generic_request=GenericRequest(model="gpt-4o-mini", messages=[], original_row={"q": "question"}, original_row_idx=0),
        created_at=now,
        finished_at=now,
    )
    filename = str(tmp_path / "responses_0.jsonl")
    await processor.append_generic_response(SimpleNamespace(num_parsed_responses=0), response, filename)
    cached = GenericResponse.model_validate_json(Path(filename).read_text())
    assert cached.response_message == {"answer": "original"}
    assert cached.generic_request.original_row == {"q": "question"}
    assert processor.validate_existing_response_file(filename) == ({0}, 0)


@pytest.mark.asyncio
async def test_provider_failures_are_still_retried_on_resume(online_cache_processor, tmp_path):
    processor = online_cache_processor
    respond = processor.call_single_request.side_effect

    async def fail_one(request, **kwargs):
        if request.generic_request.original_row_idx == 3:
            raise RuntimeError("Provider error")
        return await respond(request=request, **kwargs)

    processor.call_single_request.side_effect = fail_one
    await process_cached_requests(processor)
    filename = tmp_path / "responses_0.jsonl"
    responses = [GenericResponse.model_validate_json(line) for line in filename.read_text().splitlines()]
    assert len(responses) == 10
    assert responses[3].response_errors
    assert responses[3].response_message is None
    processor.call_single_request.side_effect = respond
    tracker = await process_cached_requests(processor)
    assert tracker.num_tasks_already_completed == 9
    assert processor.call_single_request.await_count == 11
    assert len(processor.create_dataset_files("parse-hash")) == 10
