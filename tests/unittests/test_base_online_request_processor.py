import asyncio
import datetime
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import litellm
import pytest

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online import base_online_request_processor as base_module
from bespokelabs.curator.request_processor.online.anthropic_online_request_processor import AnthropicOnlineRequestProcessor
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest
from bespokelabs.curator.request_processor.online.litellm_online_request_processor import LiteLLMOnlineRequestProcessor
from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor
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


_REAL_ASYNCIO_SLEEP = asyncio.sleep


class _FakeClock:
    """Wall clock whose sleeps advance time instantly instead of waiting."""

    def __init__(self):
        self.now = 1_000_000.0
        self.sleeps = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await _REAL_ASYNCIO_SLEEP(0)


@pytest.fixture
def fake_clock(monkeypatch):
    """Drive the processors' clock and sleeps from a fake clock."""
    clock = _FakeClock()
    monkeypatch.setattr(base_module, "time", SimpleNamespace(time=clock.time))
    monkeypatch.setattr(tracker_module, "time", SimpleNamespace(time=clock.time))
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    return clock


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


def _stubbed_session(status: int, body) -> Mock:
    session = Mock()
    session.post.return_value = _StubResponse(status, body)
    return session


def _api_request(attempts_left: int = 2) -> APIRequest:
    return APIRequest(
        task_id=0,
        generic_request=GenericRequest(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], original_row={"q": "hi"}, original_row_idx=0),
        api_specific_request={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        attempts_left=attempts_left,
    )


def _retry_tracker(total_requests: int) -> OnlineStatusTracker:
    tracker = OnlineStatusTracker(model="gpt-4o-mini", max_requests_per_minute=None, max_tokens_per_minute=None, total_requests=total_requests)
    tracker.start_tracker = Mock()
    tracker.stop_tracker = Mock()
    tracker.update_stats = Mock()
    tracker.update_cost_projection = Mock()
    return tracker


def _success_response(request: APIRequest) -> GenericResponse:
    return GenericResponse(
        response_message="answer",
        raw_response={},
        generic_request=request.generic_request,
        created_at=request.created_at,
        finished_at=datetime.datetime.now(),
        token_usage=_TokenUsage(input=10, output=5),
        response_cost=0,
    )


def _write_requests(path: Path, count: int) -> str:
    path.write_text(
        "\n".join(
            GenericRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"q{index}"}],
                original_row={"q": f"q{index}"},
                original_row_idx=index,
            ).model_dump_json()
            for index in range(count)
        )
        + "\n"
    )
    return str(path)


@pytest.fixture
def retry_processor(tmp_path):
    """Processor whose dispatch and retry drain run for real against a mocked API."""
    processor = object.__new__(AnthropicOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini", max_retries=2)
    processor.working_dir = str(tmp_path)
    processor.manual_max_concurrent_requests = 1
    processor._semaphore = None
    processor._tracker_console = None
    processor._output_tokens_window = deque()
    processor._viewer_client = AsyncMock()
    processor.prompt_formatter = Mock()
    processor.validate_existing_response_file = Mock(return_value=(set(), 0))
    processor.create_api_specific_request_online = Mock(return_value={})
    processor.estimate_total_tokens = Mock(return_value=_TokenUsage(input=10, output=5))
    processor.append_generic_response = AsyncMock()
    return processor


async def _run_to_completion(processor, requests_file: str, tracker: OnlineStatusTracker) -> None:
    await asyncio.wait_for(
        processor.process_requests_from_file(requests_file, str(Path(processor.working_dir) / "responses_0.jsonl"), tracker),
        timeout=5,
    )


@pytest.mark.parametrize(
    ("base", "jitter", "expected_delays"),
    [
        (10, 1.0, [10, 20, 40, 80, 160, 300, 300]),
        (10, 1.25, [12.5, 25, 50, 100, 200, 375, 375]),
        (400, 1.0, [400, 400, 400]),
    ],
)
def test_consecutive_rate_limits_double_the_delay_until_the_cap(monkeypatch, base, jitter, expected_delays):
    """Backoff starts at the configured pause, doubles per 429, and never exceeds the cap."""
    clock = _FakeClock()
    monkeypatch.setattr(base_module, "time", SimpleNamespace(time=clock.time))
    jitter_bounds = []

    def uniform(low, high):
        jitter_bounds.append((low, high))
        return jitter

    monkeypatch.setattr(base_module.random, "uniform", uniform)
    processor = object.__new__(AnthropicOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini", seconds_to_pause_on_rate_limit=base)
    tracker = _retry_tracker(total_requests=1)
    request = _api_request()

    delays = []
    for _ in expected_delays:
        processor._record_rate_limit_error(tracker, request)
        delays.append(request.not_before - clock.now)

    assert delays == pytest.approx(expected_delays)
    assert jitter_bounds == [(1.0, 1.25)] * len(expected_delays)
    assert request.rate_limit_retries == tracker.num_rate_limit_errors == len(expected_delays)


def _build_openai_processor() -> OpenAIOnlineRequestProcessor:
    processor = object.__new__(OpenAIOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini")
    processor.api_key = "sk-test"
    processor.url = "https://api.openai.com/v1/chat/completions"
    processor._longlived_response = False
    return processor


async def _openai_http_429(monkeypatch, request, tracker):
    await _build_openai_processor().call_single_request(request, _stubbed_session(429, "rate limit exceeded"), tracker)


async def _openai_rate_limit_error_body(monkeypatch, request, tracker):
    await _build_openai_processor().call_single_request(request, _stubbed_session(200, {"error": {"message": "Rate limit exceeded"}}), tracker)


async def _litellm_rate_limit(monkeypatch, request, tracker):
    async def raise_rate_limit(**kwargs):
        raise litellm.RateLimitError(message="rate limited", llm_provider="openai", model="gpt-4o-mini")

    processor = object.__new__(LiteLLMOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini")
    monkeypatch.setattr(litellm, "acompletion", raise_rate_limit)
    await processor.call_single_request(request, Mock(), tracker)


async def _anthropic_rate_limit_error_body(monkeypatch, request, tracker):
    processor = object.__new__(AnthropicOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini")
    processor.api_key = "sk-ant-test"
    processor.url = "https://api.anthropic.com/v1/messages"
    await processor.call_single_request(request, _stubbed_session(200, {"error": {"message": "rate limit exceeded"}}), tracker)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "error_match", "expected_counters"),
    [
        (_openai_http_429, "API Error: 429", (5, 3, 1)),
        (_openai_rate_limit_error_body, "API error", (5, 3, 1)),
        (_litellm_rate_limit, "rate limited", (5, 2, 2)),
        (_anthropic_rate_limit_error_body, "API error", (5, 3, 1)),
    ],
    ids=["openai-429", "openai-error-body", "litellm", "anthropic"],
)
async def test_rate_limit_sites_back_off_the_request_and_keep_net_error_counts(monkeypatch, trigger, error_match, expected_counters):
    """Every recording site schedules backoff with the same net counter compensation as before."""
    clock = _FakeClock()
    monkeypatch.setattr(base_module, "time", SimpleNamespace(time=clock.time))
    monkeypatch.setattr(base_module.random, "uniform", lambda low, high: 1.0)
    tracker = _retry_tracker(total_requests=1)
    tracker.num_api_errors = 3
    tracker.num_other_errors = 2
    tracker.num_rate_limit_errors = 4

    request = _api_request()
    with pytest.raises(Exception, match=error_match):
        await trigger(monkeypatch, request, tracker)

    assert (tracker.num_rate_limit_errors, tracker.num_api_errors, tracker.num_other_errors) == expected_counters
    assert request.rate_limit_retries == 1
    assert request.not_before == clock.now + 10


@pytest.mark.asyncio
async def test_backing_off_request_does_not_block_a_ready_retry(retry_processor, fake_clock, monkeypatch, tmp_path):
    """A ready retry runs while an earlier request is still waiting out its backoff."""
    processor = retry_processor
    monkeypatch.setattr(base_module.random, "uniform", lambda low, high: 1.25)
    attempt_times = defaultdict(list)
    seen_requests = {}

    async def respond(request, **kwargs):
        row = request.generic_request.original_row_idx
        attempt_times[row].append(fake_clock.now)
        seen_requests[row] = request
        if len(attempt_times[row]) == 1:
            if row == 0:
                processor._record_rate_limit_error(kwargs["status_tracker"], request)
            raise RuntimeError("provider error")
        return _success_response(request)

    processor.call_single_request = AsyncMock(side_effect=respond)
    tracker = _retry_tracker(total_requests=2)

    await _run_to_completion(processor, _write_requests(tmp_path / "requests_0.jsonl", 2), tracker)

    assert tracker.num_tasks_succeeded == 2
    assert tracker.num_tasks_failed == 0
    assert attempt_times[1][1] < seen_requests[0].not_before
    assert attempt_times[1][1] < attempt_times[0][1]


@pytest.mark.asyncio
async def test_retry_loop_sleeps_until_the_backoff_request_is_ready(retry_processor, fake_clock, monkeypatch, tmp_path):
    """While every queued retry is in backoff the loop sleeps in short slices instead of spinning."""
    processor = retry_processor
    monkeypatch.setattr(base_module.random, "uniform", lambda low, high: 1.0)
    attempt_times = []
    not_before = None

    async def respond(request, **kwargs):
        nonlocal not_before
        attempt_times.append(fake_clock.now)
        if len(attempt_times) == 1:
            processor._record_rate_limit_error(kwargs["status_tracker"], request)
            not_before = request.not_before
            raise RuntimeError("provider error")
        return _success_response(request)

    processor.call_single_request = AsyncMock(side_effect=respond)
    tracker = _retry_tracker(total_requests=1)

    await _run_to_completion(processor, _write_requests(tmp_path / "requests_0.jsonl", 1), tracker)

    assert tracker.num_tasks_succeeded == 1
    assert attempt_times[1] >= not_before
    assert attempt_times[1] == pytest.approx(not_before, abs=0.1)
    assert fake_clock.sleeps
    assert all(0 < seconds <= 0.1 + 1e-9 for seconds in fake_clock.sleeps)
    assert sum(fake_clock.sleeps) == pytest.approx(10, abs=0.05)
    assert len(fake_clock.sleeps) == pytest.approx(100, abs=2)


@pytest.mark.asyncio
async def test_consecutive_backoffs_complete_with_one_concurrency_slot(retry_processor, fake_clock, monkeypatch, tmp_path):
    """A single semaphore slot is not held while a request waits out consecutive 429 backoffs."""
    processor = retry_processor
    monkeypatch.setattr(base_module.random, "uniform", lambda low, high: 1.0)
    attempt_times = []

    async def respond(request, **kwargs):
        attempt_times.append(fake_clock.now)
        if len(attempt_times) < 3:
            processor._record_rate_limit_error(kwargs["status_tracker"], request)
            raise RuntimeError("provider error")
        return _success_response(request)

    processor.call_single_request = AsyncMock(side_effect=respond)
    tracker = _retry_tracker(total_requests=1)

    await _run_to_completion(processor, _write_requests(tmp_path / "requests_0.jsonl", 1), tracker)

    assert len(attempt_times) == 3
    assert attempt_times[1] - attempt_times[0] == pytest.approx(10, abs=0.1)
    assert attempt_times[2] - attempt_times[1] == pytest.approx(20, abs=0.1)
    assert tracker.num_tasks_succeeded == 1
    assert tracker.num_tasks_in_progress == 0


@pytest.mark.asyncio
async def test_rate_limit_on_terminal_attempt_fails_the_request(retry_processor, fake_clock, monkeypatch, tmp_path):
    """Once attempts are exhausted the request fails instead of staying queued behind its backoff."""
    processor = retry_processor
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini", max_retries=1)
    monkeypatch.setattr(base_module.random, "uniform", lambda low, high: 1.0)
    attempt_times = []

    async def respond(request, **kwargs):
        attempt_times.append(fake_clock.now)
        processor._record_rate_limit_error(kwargs["status_tracker"], request)
        raise RuntimeError("provider error")

    processor.call_single_request = AsyncMock(side_effect=respond)
    tracker = _retry_tracker(total_requests=1)

    await _run_to_completion(processor, _write_requests(tmp_path / "requests_0.jsonl", 1), tracker)

    assert len(attempt_times) == 2
    assert tracker.num_tasks_failed == 1
    assert tracker.num_tasks_succeeded == 0
    assert tracker.num_tasks_in_progress == 0
    failed_response = processor.append_generic_response.await_args.args[1]
    assert failed_response.response_message is None
    assert failed_response.response_errors == ["provider error(x2)"]


@pytest.mark.asyncio
async def test_non_rate_limit_failure_keeps_the_request_immediately_retryable(retry_processor, fake_clock, tmp_path):
    """A non-429 failure schedules no backoff, so the retry runs in the same loop iteration."""
    processor = retry_processor
    attempt_times = []
    seen_requests = []

    async def respond(request, **kwargs):
        attempt_times.append(fake_clock.now)
        seen_requests.append(request)
        if len(attempt_times) == 1:
            raise RuntimeError("provider error")
        return _success_response(request)

    processor.call_single_request = AsyncMock(side_effect=respond)
    tracker = _retry_tracker(total_requests=1)

    await _run_to_completion(processor, _write_requests(tmp_path / "requests_0.jsonl", 1), tracker)

    assert seen_requests[0].rate_limit_retries == 0
    assert seen_requests[0].not_before == 0.0
    assert attempt_times[0] == attempt_times[1]
    assert fake_clock.sleeps == []
    assert tracker.num_tasks_succeeded == 1
