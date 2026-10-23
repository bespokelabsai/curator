"""Utilities for safely reusing successful responses from another Curator run."""

import glob
import json
import os
from collections import defaultdict, deque
from pathlib import Path

from pydantic import ValidationError

from bespokelabs.curator.log import logger
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse


def _request_identity(request: GenericRequest) -> str:
    """Return the generation-parameter-independent identity of a request."""
    request_data = request.model_dump(
        mode="json",
        exclude={"generation_params", "original_row_idx"},
    )
    original_row = request_data.get("original_row")
    if isinstance(original_row, dict):
        # Curator reserves this column for per-row generation controls.
        original_row.pop("generation_params", None)
    return json.dumps(
        request_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _successful_cached_responses(
    cache_dir: Path,
    invalid_finish_reasons: set[str],
) -> dict[str, deque[GenericResponse]]:
    responses_by_request = defaultdict(deque)
    for response_file in sorted(glob.glob(str(cache_dir / "responses_*.jsonl"))):
        with open(response_file, encoding="utf-8") as source:
            for line in source:
                try:
                    response = GenericResponse.model_validate_json(line)
                except (json.JSONDecodeError, ValidationError):
                    continue
                if (
                    response.response_errors
                    or response.response_message is None
                    or response.finish_reason in invalid_finish_reasons
                ):
                    continue
                identity = _request_identity(response.generic_request)
                responses_by_request[identity].append(response)
    return responses_by_request


def _completed_request_ids(response_file: Path) -> set[int]:
    completed = set()
    if not response_file.exists():
        return completed
    with open(response_file, encoding="utf-8") as existing:
        for line in existing:
            try:
                response = GenericResponse.model_validate_json(line)
            except (json.JSONDecodeError, ValidationError):
                continue
            if not response.response_errors and response.response_message is not None:
                completed.add(response.generic_request.original_row_idx)
    return completed


def reuse_cached_responses(
    source_cache_dir: str | Path,
    request_files: list[str],
    invalid_finish_reasons: set[str] | None = None,
) -> int:
    """Seed a new run with exact request matches from a previous cache.

    Generation parameters and row positions may differ. The model, rendered
    messages, response schema, multimodal flag, and original input row must
    otherwise match exactly. Failed responses are never reused.

    Args:
        source_cache_dir: Previous Curator run directory containing response
            JSONL files.
        request_files: Newly generated request JSONL files for the current run.
        invalid_finish_reasons: Finish reasons that must be retried rather than
            reused.

    Returns:
        Number of successful responses copied into the current run.
    """
    source_cache_dir = Path(source_cache_dir).expanduser().resolve()
    if not source_cache_dir.is_dir():
        raise ValueError(
            f"Reuse cache directory does not exist: {source_cache_dir}"
        )
    if not request_files:
        return 0

    destination_dir = Path(request_files[0]).resolve().parent
    if source_cache_dir == destination_dir:
        return 0

    cached_responses = _successful_cached_responses(
        source_cache_dir,
        invalid_finish_reasons or set(),
    )
    reused_count = 0

    for request_file_name in request_files:
        request_file = Path(request_file_name)
        response_file = Path(
            str(request_file).replace("requests_", "responses_")
        )
        completed_ids = _completed_request_ids(response_file)
        responses_to_append = []

        with open(request_file, encoding="utf-8") as requests:
            for line in requests:
                try:
                    current_request = GenericRequest.model_validate_json(line)
                except (json.JSONDecodeError, ValidationError):
                    continue
                if current_request.original_row_idx in completed_ids:
                    continue

                matching = cached_responses.get(
                    _request_identity(current_request)
                )
                if not matching:
                    continue

                reused_response = matching.popleft().model_copy(deep=True)
                source_generation_params = dict(
                    reused_response.generic_request.generation_params
                )
                reused_response.generic_request = current_request.model_copy(deep=True)
                reused_response.generic_request.generation_params = (
                    source_generation_params
                )
                reused_response.parsed_response_message = None
                responses_to_append.append(reused_response.model_dump_json())
                completed_ids.add(current_request.original_row_idx)
                reused_count += 1

        if responses_to_append:
            os.makedirs(response_file.parent, exist_ok=True)
            with open(response_file, "a", encoding="utf-8") as destination:
                destination.write("\n".join(responses_to_append) + "\n")

    logger.info(
        f"Reused {reused_count} successful responses from {source_cache_dir}"
    )
    return reused_count
