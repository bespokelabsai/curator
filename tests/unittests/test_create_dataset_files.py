import datetime
import json
import os

import pytest
from datasets import config as datasets_config

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse

# The dataset writer flushes rows to disk in batches of this size. Failing after the first batch
# means part of the dataset is already in the file when the build stops.
_BATCH = datasets_config.DEFAULT_MAX_BATCH_SIZE
_N_ROWS = _BATCH + 50
_FAIL_AT_ROW = _BATCH + 10
_PARSE_FUNC_HASH = "parse_func_hash"


def _write_cached_run(working_dir, corrupt_row=None):
    """Write the request and response files a finished online run leaves in its cache directory."""
    now = datetime.datetime.now()
    with open(os.path.join(working_dir, "requests_0.jsonl"), "w") as requests_file, open(os.path.join(working_dir, "responses_0.jsonl"), "w") as responses_file:
        for idx in range(_N_ROWS):
            request = GenericRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"question {idx}"}],
                original_row={"question": f"question {idx}"},
                original_row_idx=idx,
            )
            requests_file.write(json.dumps(request.model_dump(), default=str) + "\n")
            if idx == corrupt_row:
                responses_file.write('{"response_message": "unterminated\n')
                continue
            response = GenericResponse(
                response_message=f"answer {idx}",
                raw_response={},
                generic_request=request,
                created_at=now,
                finished_at=now,
            )
            responses_file.write(json.dumps(response.model_dump(), default=str) + "\n")
    with open(os.path.join(working_dir, "metadata_0.json"), "w") as f:
        json.dump({"num_jobs": _N_ROWS}, f)


def _processor(working_dir, parse_func):
    # create_dataset_files only needs the config, working directory and prompt formatter, so skip
    # __init__, which makes a network call to read rate limits.
    processor = object.__new__(OpenAIOnlineRequestProcessor)
    processor.config = OnlineRequestProcessorConfig(model="gpt-4o-mini")
    processor.working_dir = str(working_dir)
    processor.prompt_formatter = PromptFormatter(model_name="gpt-4o-mini", prompt_func=lambda row: row["question"], parse_func=parse_func)
    return processor


def _parse(row, response):
    return {"question": row["question"], "answer": response}


def _interrupting_parse():
    calls = {"n": 0}

    def parse(row, response):
        calls["n"] += 1
        if calls["n"] == _FAIL_AT_ROW:
            raise KeyboardInterrupt
        return _parse(row, response)

    return parse


@pytest.mark.parametrize(
    "corrupt_row, parse_func, error",
    [
        pytest.param(None, _interrupting_parse, KeyboardInterrupt, id="keyboard_interrupt"),
        pytest.param(_FAIL_AT_ROW, lambda: _parse, ValueError, id="unreadable_response_line"),
    ],
)
def test_interrupted_dataset_build_is_not_loaded_as_cached_dataset(tmp_path, corrupt_row, parse_func, error):
    """A dataset build that stops partway must not leave a dataset file that the next run loads from the cache."""
    _write_cached_run(tmp_path, corrupt_row=corrupt_row)

    with pytest.raises(error):
        _processor(tmp_path, parse_func()).create_dataset_files(_PARSE_FUNC_HASH)

    assert _processor(tmp_path, _parse).attempt_loading_cached_dataset(_PARSE_FUNC_HASH) is None
    # No partial dataset or temporary file is left in the cache directory.
    assert sorted(os.listdir(tmp_path)) == ["metadata_0.json", "requests_0.jsonl", "responses_0.jsonl"]


def test_dataset_is_rebuilt_from_cached_responses_after_interrupted_build(tmp_path):
    _write_cached_run(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _processor(tmp_path, _interrupting_parse()).create_dataset_files(_PARSE_FUNC_HASH)

    dataset = _processor(tmp_path, _parse).create_dataset_files(_PARSE_FUNC_HASH)

    assert len(dataset) == _N_ROWS
    cached = _processor(tmp_path, _parse).attempt_loading_cached_dataset(_PARSE_FUNC_HASH)
    assert cached["answer"] == [f"answer {idx}" for idx in range(_N_ROWS)]
    assert sorted(os.listdir(tmp_path)) == sorted(
        [f"{_PARSE_FUNC_HASH}.arrow", "failed_requests.jsonl", "metadata_0.json", "requests_0.jsonl", "responses_0.jsonl"]
    )


def test_failed_move_into_place_leaves_no_temporary_file(tmp_path, monkeypatch):
    """If the final rename fails (for example a permission error), the error surfaces and no dataset or temporary file is left behind."""
    _write_cached_run(tmp_path)

    def failing_replace(src, dst):
        raise PermissionError(f"cannot move {src} to {dst}")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failing_replace)
        with pytest.raises(PermissionError):
            _processor(tmp_path, _parse).create_dataset_files(_PARSE_FUNC_HASH)

    assert sorted(os.listdir(tmp_path)) == sorted(["failed_requests.jsonl", "metadata_0.json", "requests_0.jsonl", "responses_0.jsonl"])
    assert _processor(tmp_path, _parse).attempt_loading_cached_dataset(_PARSE_FUNC_HASH) is None
