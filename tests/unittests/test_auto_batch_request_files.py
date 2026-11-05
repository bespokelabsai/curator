import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import Dataset

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig


# ---- Local fixture helpers ----
@pytest.fixture
def batch_file_processor(tmp_path, monkeypatch):
    """Avoid constructing API clients while exercising real request-file orchestration."""
    processor = object.__new__(OpenAIBatchRequestProcessor)
    processor.config = BatchRequestProcessorConfig(model="gpt-4o-mini", batch_size="auto")
    processor.prompt_formatter = PromptFormatter(model_name="gpt-4o-mini", prompt_func=lambda row: row["prompt"])
    processor.working_dir = str(tmp_path)
    monkeypatch.setattr(OpenAIBatchRequestProcessor, "max_requests_per_batch", property(lambda self: 100))
    monkeypatch.setattr(OpenAIBatchRequestProcessor, "max_bytes_per_batch", property(lambda self: 1_000_000))
    return processor


def read_requests(paths):
    return [[json.loads(line) for line in Path(path).read_text().splitlines()] for path in paths]


# ---- End local fixture helpers ----


@pytest.mark.parametrize("split", ["count", "bytes"])
def test_auto_batches_all_reach_the_response_processor(batch_file_processor, tmp_path, monkeypatch, split):
    processor = batch_file_processor
    dataset = Dataset.from_list([{"prompt": "hello"}] * 3)
    if split == "count":
        monkeypatch.setattr(OpenAIBatchRequestProcessor, "max_requests_per_batch", property(lambda self: 2))
        expected_lengths = [2, 1]
    else:
        request = processor.prompt_formatter.create_generic_request(dataset[0], 0)
        request_bytes = len(json.dumps(request.model_dump(), default=str).encode())
        monkeypatch.setattr(OpenAIBatchRequestProcessor, "max_bytes_per_batch", property(lambda self: request_bytes * 2 - 1))
        expected_lengths = [1, 1, 1]

    responses = Mock()
    monkeypatch.setattr(processor, "requests_to_responses", responses)
    monkeypatch.setattr(processor, "attempt_loading_cached_dataset", lambda parse_hash: None)
    monkeypatch.setattr(processor, "create_dataset_files", lambda parse_hash: dataset)

    processor.run(dataset, str(tmp_path), "parse-hash", processor.prompt_formatter)

    responses.assert_called_once()
    paths = responses.call_args.args[0]
    assert paths == [str(tmp_path / f"requests_{index}.jsonl") for index in range(len(expected_lengths))]
    requests = read_requests(paths)
    assert [len(batch) for batch in requests] == expected_lengths
    assert [row["original_row_idx"] for batch in requests for row in batch] == list(range(len(dataset)))


@pytest.mark.parametrize("batch_size, expected_lengths", [("auto", [3]), (2, [2, 1])])
def test_batch_file_creation_preserves_single_and_numeric_batches(batch_file_processor, tmp_path, batch_size, expected_lengths):
    processor = batch_file_processor
    processor.config.batch_size = batch_size
    dataset = Dataset.from_list([{"prompt": "hello"}] * 3)

    paths = processor.create_request_files(dataset)

    assert paths == [str(tmp_path / f"requests_{index}.jsonl") for index in range(len(expected_lengths))]
    requests = read_requests(paths)
    assert [len(batch) for batch in requests] == expected_lengths
    assert [row["original_row_idx"] for batch in requests for row in batch] == [0, 1, 2]


def test_cached_auto_batch_files_are_reused(batch_file_processor, tmp_path, monkeypatch):
    processor = batch_file_processor
    dataset = Dataset.from_list([{"prompt": "hello"}] * 3)
    monkeypatch.setattr(OpenAIBatchRequestProcessor, "max_requests_per_batch", property(lambda self: 2))
    processor.create_request_files(dataset)
    original_contents = {str(path): path.read_bytes() for path in tmp_path.glob("requests_*.jsonl")}
    create = Mock(side_effect=AssertionError("Cached request files must not be rewritten"))
    monkeypatch.setattr(processor, "acreate_request_file", create)

    cached_paths = processor.create_request_files(dataset)

    assert set(cached_paths) == {str(tmp_path / "requests_0.jsonl"), str(tmp_path / "requests_1.jsonl")}
    assert {path: Path(path).read_bytes() for path in cached_paths} == original_contents
    create.assert_not_called()
