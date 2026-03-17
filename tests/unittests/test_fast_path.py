import asyncio
import datetime
from contextlib import asynccontextmanager

import pytest

from bespokelabs import curator
from bespokelabs.curator.constants import _INTERNAL_PROMPT_KEY
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import BaseOnlineRequestProcessor
from bespokelabs.curator.request_processor.online.in_memory_request_processor import _process_requests_in_memory
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


class DummyFastPathProcessor(BaseOnlineRequestProcessor):
    def __init__(
        self,
        config: OnlineRequestProcessorConfig,
        scripted_results: dict[int, list[dict | Exception]],
        token_estimate: _TokenUsage | None = None,
    ):
        super().__init__(config)
        self.scripted_results = scripted_results
        self.token_estimate = token_estimate or _TokenUsage(input=60, output=0)
        self.call_counts: dict[int, int] = {}
        self.connector_calls = 0
        self.session_object = None
        self.sessions_seen = []

    @property
    def backend(self) -> str:
        return "dummy"

    @property
    def compatible_provider(self) -> str:
        return "dummy"

    def validate_config(self):
        return None

    def file_upload_limit_check(self, base64_image: str) -> None:
        return None

    def estimate_total_tokens(self, messages: list) -> _TokenUsage:
        return self.token_estimate.model_copy()

    def estimate_output_tokens(self) -> int:
        return 0

    def create_api_specific_request_online(self, generic_request) -> dict:
        return {"messages": generic_request.messages}

    @asynccontextmanager
    async def aiohttp_connector(self, tcp_limit: int):
        self.connector_calls += 1
        self.session_object = object()
        yield self.session_object

    async def call_single_request(self, request, session, status_tracker) -> GenericResponse:
        self.sessions_seen.append(session)
        attempt = self.call_counts.get(request.task_id, 0)
        self.call_counts[request.task_id] = attempt + 1
        event = self.scripted_results[request.task_id][attempt]
        if isinstance(event, Exception):
            raise event

        return GenericResponse(
            response_message=event.get("response_message"),
            response_errors=None,
            raw_request=request.api_specific_request,
            raw_response=event.get("raw_response", {"ok": True}),
            generic_request=request.generic_request,
            created_at=request.created_at,
            finished_at=datetime.datetime.now(),
            token_usage=event.get("token_usage"),
            response_cost=event.get("response_cost"),
            finish_reason=event.get("finish_reason"),
        )


class RecordingFastPathProcessor(DummyFastPathProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_generation_params = []

    async def call_single_request(self, request, session, status_tracker) -> GenericResponse:
        self.seen_generation_params.append(dict(request.generic_request.generation_params))
        return await super().call_single_request(request, session, status_tracker)


def test_fast_path_response_preserves_failed_requests_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("CURATOR_DISABLE_CACHE", "true")
    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CURATOR_VIEWER", "false")
    monkeypatch.delenv("CURATOR_DISABLE_FAST_PATH", raising=False)

    llm = curator.LLM(model_name="gpt-4o-mini")
    processor = DummyFastPathProcessor(
        config=OnlineRequestProcessorConfig(
            model="gpt-4o-mini",
            max_retries=0,
            require_all_responses=False,
            max_requests_per_minute=10,
            max_tokens_per_minute=1000,
            max_concurrent_requests=1,
        ),
        scripted_results={
            0: [
                {
                    "response_message": "ok",
                    "token_usage": _TokenUsage(input=5, output=7),
                    "response_cost": 0.25,
                    "finish_reason": "stop",
                }
            ],
            1: [RuntimeError("boom")],
        },
        token_estimate=_TokenUsage(input=12, output=0),
    )
    llm._request_processor = processor

    response = llm(["first", "second"])

    assert response.cache_dir is not None
    assert response.metadata["session_id"]
    assert response.metadata["run_hash"]
    assert response.metadata["dataset_hash"]
    assert response.request_stats.total == 2
    assert response.request_stats.succeeded == 1
    assert response.request_stats.failed == 1
    assert len(response.dataset) == 1
    assert response.failed_requests_path is not None
    assert response.failed_requests_path.exists()
    assert list(response.get_failed_requests()) == [
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "second"}],
            "response_format": None,
            "original_row": {_INTERNAL_PROMPT_KEY: "second"},
            "original_row_idx": 1,
            "generation_params": {},
            "is_multimodal_prompt": False,
        }
    ]
    assert (tmp_path / response.metadata["run_hash"] / "response.json").exists()
    assert processor.connector_calls == 1
    assert len(processor.sessions_seen) == 2
    assert all(session is processor.session_object for session in processor.sessions_seen)


def test_fast_path_keeps_consumed_token_capacity_on_invalid_finish(monkeypatch):
    monkeypatch.setenv("CURATOR_DISABLE_CACHE", "true")

    llm = curator.LLM(model_name="gpt-4o-mini")
    processor = DummyFastPathProcessor(
        config=OnlineRequestProcessorConfig(
            model="gpt-4o-mini",
            max_retries=0,
            require_all_responses=False,
            max_requests_per_minute=100,
            max_tokens_per_minute=100,
        ),
        scripted_results={
            0: [
                {
                    "response_message": "too long",
                    "token_usage": _TokenUsage(input=1, output=0),
                    "response_cost": 0.1,
                    "finish_reason": "length",
                }
            ]
        },
        token_estimate=_TokenUsage(input=1, output=0),
    )

    request = llm.prompt_formatter.create_generic_request({_INTERNAL_PROMPT_KEY: "hello"}, 0, False)
    result = asyncio.run(
        _process_requests_in_memory(
            processor=processor,
            requests=[request],
            prompt_formatter=llm.prompt_formatter,
        )
    )

    assert result.responses[0].response_errors == ["finish_reason was length"]
    assert result.tracker.available_token_capacity < 0.5
    assert result.tracker.total_prompt_tokens == 1
    assert result.tracker.total_tokens == 1


def test_fast_path_does_not_retry_runtime_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("CURATOR_DISABLE_CACHE", "true")

    llm = curator.LLM(model_name="gpt-4o-mini")
    llm._request_processor = DummyFastPathProcessor(
        config=OnlineRequestProcessorConfig(
            model="gpt-4o-mini",
            max_retries=0,
            require_all_responses=False,
        ),
        scripted_results={},
    )

    call_count = 0

    async def fail_in_memory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("processing exploded")

    monkeypatch.setattr(
        "bespokelabs.curator.request_processor.online.in_memory_request_processor._process_requests_in_memory",
        fail_in_memory,
    )

    with pytest.raises(RuntimeError, match="processing exploded"):
        llm._run_fast_path(
            raw_input=["hello"],
            run_cache_dir=str(tmp_path),
            metadata={},
        )

    assert call_count == 1


def test_fast_path_matches_standard_generation_params_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv("CURATOR_DISABLE_CACHE", "true")
    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CURATOR_VIEWER", "false")
    monkeypatch.delenv("CURATOR_DISABLE_FAST_PATH", raising=False)

    llm = curator.LLM(
        model_name="gpt-4o-mini",
        generation_params={"temperature": None, "max_tokens": 7},
    )
    processor = RecordingFastPathProcessor(
        config=OnlineRequestProcessorConfig(
            model="gpt-4o-mini",
            max_retries=0,
            require_all_responses=False,
            generation_params={"temperature": None, "max_tokens": 7},
        ),
        scripted_results={
            0: [
                {
                    "response_message": "ok",
                    "token_usage": _TokenUsage(input=1, output=1),
                    "response_cost": 0.01,
                    "finish_reason": "stop",
                }
            ]
        },
    )
    llm._request_processor = processor

    response = llm(["hello"])

    assert len(response.dataset) == 1
    assert processor.seen_generation_params == [{"max_tokens": 7}]
