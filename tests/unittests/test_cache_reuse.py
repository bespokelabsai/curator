import datetime
import json
from pathlib import Path

import pytest

from bespokelabs.curator.llm.llm import LLM
from bespokelabs.curator.request_processor.cache import reuse_cached_responses
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse


def _request(
    *,
    idx: int,
    question: str,
    max_tokens: int,
    prompt: str | None = None,
) -> GenericRequest:
    return GenericRequest(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": prompt or question,
            }
        ],
        response_format=None,
        original_row={"question": question},
        original_row_idx=idx,
        generation_params={"max_tokens": max_tokens},
    )


def _response(
    request: GenericRequest,
    answer: str | None,
    errors: list[str] | None = None,
    finish_reason: str | None = None,
) -> GenericResponse:
    now = datetime.datetime.now(datetime.timezone.utc)
    return GenericResponse(
        response_message=answer,
        response_errors=errors,
        raw_response={},
        raw_request={},
        generic_request=request,
        created_at=now,
        finished_at=now,
        finish_reason=finish_reason or ("stop" if not errors else "length"),
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        for row in rows:
            if isinstance(row, (GenericRequest, GenericResponse)):
                output.write(row.model_dump_json() + "\n")
            else:
                output.write(json.dumps(row) + "\n")


def test_reuse_cache_copies_only_successful_exact_request_matches(tmp_path):
    source = tmp_path / "old-cache"
    destination = tmp_path / "new-cache"

    source_a = _request(
        idx=0,
        question="Question A",
        max_tokens=8192,
    )
    source_b = _request(
        idx=1,
        question="Question B",
        max_tokens=8192,
    )
    source_c = _request(
        idx=2,
        question="Question C",
        max_tokens=8192,
    )
    _write_jsonl(
        source / "responses_0.jsonl",
        [
            _response(source_a, "Answer A"),
            _response(
                source_b,
                "Truncated answer B",
                finish_reason="length",
            ),
            _response(source_c, "Answer C"),
        ],
    )

    current_b = _request(
        idx=0,
        question="Question B",
        max_tokens=16384,
    )
    current_c_changed_prompt = _request(
        idx=1,
        question="Question C",
        max_tokens=16384,
        prompt="A changed prompt for Question C",
    )
    current_a_reordered = _request(
        idx=2,
        question="Question A",
        max_tokens=16384,
    )
    request_file = destination / "requests_0.jsonl"
    _write_jsonl(
        request_file,
        [current_b, current_c_changed_prompt, current_a_reordered],
    )

    assert (
        reuse_cached_responses(
            source,
            [str(request_file)],
            {"length", "content_filter"},
        )
        == 1
    )

    reused_lines = (destination / "responses_0.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(reused_lines) == 1
    reused = GenericResponse.model_validate_json(reused_lines[0])
    assert reused.response_message == "Answer A"
    assert reused.generic_request.original_row_idx == 2
    assert reused.generic_request.generation_params == {"max_tokens": 8192}
    assert reused.parsed_response_message is None

    # Reapplying the same source is idempotent.
    assert (
        reuse_cached_responses(
            source,
            [str(request_file)],
            {"length", "content_filter"},
        )
        == 0
    )
    assert (
        len(
            (destination / "responses_0.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 1
    )


def test_reuse_cache_ignores_reserved_row_generation_params(tmp_path):
    source = tmp_path / "old-cache"
    destination = tmp_path / "new-cache"
    source_request = _request(
        idx=0,
        question="Question A",
        max_tokens=8192,
    )
    source_request.original_row["generation_params"] = json.dumps(
        {"max_tokens": 8192}
    )
    current_request = _request(
        idx=0,
        question="Question A",
        max_tokens=16384,
    )
    current_request.original_row["generation_params"] = json.dumps(
        {"max_tokens": 16384}
    )
    _write_jsonl(
        source / "responses_0.jsonl",
        [_response(source_request, "Answer A")],
    )
    request_file = destination / "requests_0.jsonl"
    _write_jsonl(request_file, [current_request])

    assert reuse_cached_responses(source, [str(request_file)]) == 1


@pytest.mark.parametrize(
    "reuse_cache",
    [True, 123, object()],
)
def test_reuse_cache_rejects_invalid_source_types(tmp_path, reuse_cache):
    with pytest.raises(TypeError, match="reuse_cache"):
        LLM._resolve_reuse_cache_dir(reuse_cache, str(tmp_path))


def test_reuse_cache_resolves_run_fingerprint_from_working_dir(tmp_path):
    source = tmp_path / "old-run-hash"
    source.mkdir()

    assert (
        LLM._resolve_reuse_cache_dir("old-run-hash", str(tmp_path))
        == source.resolve()
    )


def test_reuse_cache_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        LLM._resolve_reuse_cache_dir("missing-run", str(tmp_path))
