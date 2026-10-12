import datetime
from pathlib import Path

import pytest
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from pydantic import ValidationError

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor import base_request_processor as processor_module
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.anthropic_online_request_processor import AnthropicOnlineRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse


@pytest.fixture
def cached_processor(tmp_path):
    """Real cached responses large enough to flush multiple Arrow write batches."""
    processor = object.__new__(AnthropicOnlineRequestProcessor)
    processor.working_dir = str(tmp_path)
    processor.config = OnlineRequestProcessorConfig(model="test-model", require_all_responses=True)
    processor.prompt_formatter = PromptFormatter(
        model_name="test-model",
        prompt_func=lambda row: str(row["index"]),
        parse_func=lambda row, response: {"index": row["index"], "answer": response},
    )
    now = datetime.datetime.now()
    with (tmp_path / "requests_0.jsonl").open("w") as requests, (tmp_path / "responses_0.jsonl").open("w") as responses:
        # Reverse order also exercises sorting when the completed file is loaded.
        for index in reversed(range(2500)):
            request = GenericRequest(model="test-model", messages=[], original_row={"index": index}, original_row_idx=index)
            response = GenericResponse(generic_request=request, response_message=f"answer {index}", raw_response={}, created_at=now, finished_at=now)
            requests.write(request.model_dump_json() + "\n")
            responses.write(response.model_dump_json() + "\n")
    return processor


def assert_complete_cache(processor):
    dataset = processor.create_dataset_files("parse-hash")
    assert dataset["index"] == list(range(2500))
    assert dataset["answer"] == [f"answer {index}" for index in range(2500)]
    cached = processor.attempt_loading_cached_dataset("parse-hash")
    assert cached.to_list() == dataset.to_list()
    cache_path = Path(processor.working_dir) / "parse-hash.arrow"
    assert len(Dataset.from_file(str(cache_path))) == 2500
    assert str(cache_path) in {entry["filename"] for entry in dataset.cache_files}
    assert not list(Path(processor.working_dir).glob(".curator-dataset-*"))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("existing_cache", [False, True])
def test_interrupted_dataset_build_is_not_published(cached_processor, tmp_path, interruption, existing_cache):
    processor = cached_processor
    cache_path = tmp_path / "parse-hash.arrow"
    if existing_cache:
        processor.create_dataset_files("parse-hash")
        previous_cache = cache_path.read_bytes()
    response_path = tmp_path / "responses_0.jsonl"
    original_responses = response_path.read_bytes()
    parse = processor.prompt_formatter.parse_func
    calls = 0

    def interrupt_parse(row, response):
        nonlocal calls
        calls += 1
        if calls == 2200:
            raise interruption("Interrupted dataset build")
        return parse(row, response)

    processor.prompt_formatter.parse_func = interrupt_parse
    with pytest.raises(interruption, match="Interrupted dataset build"):
        processor.create_dataset_files("parse-hash")
    assert calls == 2200
    assert not list(tmp_path.glob(".curator-dataset-*"))
    assert response_path.read_bytes() == original_responses
    if existing_cache:
        assert cache_path.read_bytes() == previous_cache
        assert len(processor.attempt_loading_cached_dataset("parse-hash")) == 2500
    else:
        assert not cache_path.exists()
        assert processor.attempt_loading_cached_dataset("parse-hash") is None

    processor.prompt_formatter.parse_func = parse
    assert_complete_cache(processor)


def test_malformed_cached_response_does_not_publish_partial_dataset(cached_processor, tmp_path):
    response_path = tmp_path / "responses_0.jsonl"
    original = response_path.read_text()
    lines = original.splitlines(keepends=True)
    lines[2200] = '{"broken":\n'
    response_path.write_text("".join(lines))
    with pytest.raises(ValidationError):
        cached_processor.create_dataset_files("parse-hash")
    assert cached_processor.attempt_loading_cached_dataset("parse-hash") is None
    assert not list(tmp_path.glob(".curator-dataset-*"))
    response_path.write_text(original)
    assert_complete_cache(cached_processor)


@pytest.mark.parametrize("failure_stage", ["finalize", "publish"])
def test_dataset_write_failure_preserves_existing_cache(cached_processor, tmp_path, monkeypatch, failure_stage):
    processor = cached_processor
    processor.create_dataset_files("parse-hash")
    cache_path = tmp_path / "parse-hash.arrow"
    original = cache_path.read_bytes()
    if failure_stage == "finalize":
        finalize = ArrowWriter.finalize

        def fail_finalize(self, *args, **kwargs):
            finalize(self, *args, **kwargs)
            raise OSError("Failed to finalize dataset")

        monkeypatch.setattr(ArrowWriter, "finalize", fail_finalize)
    else:

        def fail_publish(source, destination):
            assert Path(source).parent.parent == tmp_path  # Same filesystem as the destination.
            assert Path(destination) == cache_path
            assert len(Dataset.from_file(source)) == 2500
            raise OSError("Failed to publish dataset")

        monkeypatch.setattr(processor_module.os, "replace", fail_publish)

    with pytest.raises(OSError, match="Failed to"):
        processor.create_dataset_files("parse-hash")
    assert cache_path.read_bytes() == original
    assert not list(tmp_path.glob(".curator-dataset-*"))
    assert len(processor.attempt_loading_cached_dataset("parse-hash")) == 2500


@pytest.mark.parametrize("failure", ["missing_response", "empty_row", "invalid_row", "failed_response", "all_failed"])
def test_dataset_validation_failure_is_not_cached(cached_processor, tmp_path, failure):
    processor = cached_processor
    response_path = tmp_path / "responses_0.jsonl"
    if failure == "missing_response":
        response_path.write_text("\n".join(response_path.read_text().splitlines()[:-1]) + "\n")
    elif failure == "empty_row":
        processor.prompt_formatter.parse_func = lambda row, response: {}
    elif failure == "invalid_row":
        processor.prompt_formatter.parse_func = lambda row, response: "invalid"
    else:
        lines = response_path.read_text().splitlines()
        for index in range(len(lines) if failure == "all_failed" else 1):
            response = GenericResponse.model_validate_json(lines[index])
            response.response_errors = ["Provider error"]
            lines[index] = response.model_dump_json()
        response_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        processor.create_dataset_files("parse-hash")
    assert processor.attempt_loading_cached_dataset("parse-hash") is None
    assert not list(tmp_path.glob(".curator-dataset-*"))
