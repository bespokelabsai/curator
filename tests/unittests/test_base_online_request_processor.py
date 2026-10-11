import datetime
import json
import os
import types

import pytest

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.anthropic_online_request_processor import AnthropicOnlineRequestProcessor
from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse


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


_QUESTIONS = ["q0", "q1", "q2"]
_REJECTED = "q1"


class _ViewerClient:
    async def stream_response(self, response_data, idx):
        pass


def _parse(row, response):
    return {"question": row["question"], "answer": response}


def _parse_raises_on_rejected_row(row, response):
    if row["question"] == _REJECTED:
        raise KeyError("bug in parse_func")
    return _parse(row, response)


def _parse_filters_out_rejected_row(row, response):
    if row["question"] == _REJECTED:
        return []
    return _parse(row, response)


def _online_processor(working_dir, parse_func):
    # append_generic_response and the cache helpers only need these attributes, so skip __init__,
    # which makes a network call to read rate limits.
    processor = object.__new__(OpenAIOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini")
    processor.working_dir = str(working_dir)
    processor.prompt_formatter = PromptFormatter(model_name="gpt-4o-mini", prompt_func=lambda row: row["question"], parse_func=parse_func)
    processor.viewer_client = _ViewerClient()
    return processor


async def _record_api_responses(processor, working_dir):
    """Write the request file and record an API response for every request, as an online run does."""
    tracker = types.SimpleNamespace(num_parsed_responses=0)
    now = datetime.datetime.now()
    with open(os.path.join(working_dir, "requests_0.jsonl"), "w") as requests_file:
        for idx, question in enumerate(_QUESTIONS):
            request = GenericRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
                original_row={"question": question},
                original_row_idx=idx,
            )
            requests_file.write(json.dumps(request.model_dump(), default=str) + "\n")
            response = GenericResponse(
                response_message=f"answer to {question}",
                raw_response={},
                generic_request=request,
                created_at=now,
                finished_at=now,
            )
            await processor.append_generic_response(tracker, response, os.path.join(working_dir, "responses_0.jsonl"))
    return tracker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parse_func",
    [_parse_raises_on_rejected_row, _parse_filters_out_rejected_row],
    ids=["parse_func_raises", "parse_func_filters_out_row"],
)
async def test_response_rejected_by_parse_func_is_not_requested_again_on_resume(tmp_path, parse_func):
    processor = _online_processor(tmp_path, parse_func)
    tracker = await _record_api_responses(processor, tmp_path)

    completed_request_ids, _ = processor.validate_existing_response_file(str(tmp_path / "responses_0.jsonl"))

    assert completed_request_ids == {0, 1, 2}
    assert tracker.num_parsed_responses == 2


@pytest.mark.asyncio
async def test_dataset_is_rebuilt_from_cached_responses_after_parse_func_is_fixed(tmp_path):
    await _record_api_responses(_online_processor(tmp_path, _parse_raises_on_rejected_row), tmp_path)

    dataset = _online_processor(tmp_path, _parse).create_dataset_files("fixed_parse_func_hash")

    assert dataset["answer"] == [f"answer to {question}" for question in _QUESTIONS]


@pytest.mark.asyncio
async def test_rows_filtered_out_by_parse_func_do_not_fail_the_run(tmp_path):
    await _record_api_responses(_online_processor(tmp_path, _parse_filters_out_rejected_row), tmp_path)

    dataset = _online_processor(tmp_path, _parse_filters_out_rejected_row).create_dataset_files("parse_func_hash")

    assert dataset["question"] == ["q0", "q2"]
