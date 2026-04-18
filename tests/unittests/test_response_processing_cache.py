import datetime

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.config import RequestProcessorConfig
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse


class DummyRequestProcessor(BaseRequestProcessor):
    @property
    def backend(self) -> str:
        return "dummy"

    def validate_config(self):
        pass

    def requests_to_responses(self, generic_request_files: list[str]) -> None:
        pass


def _make_generic_response(parse_hash: str | None = None, parsed=None) -> GenericResponse:
    return GenericResponse(
        response_message="hello",
        parsed_response_message=parsed,
        parsed_response_message_parse_func_hash=parse_hash,
        response_errors=None,
        raw_response={},
        raw_request={},
        generic_request=GenericRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            original_row={"value": "hello"},
            original_row_idx=0,
        ),
        created_at=datetime.datetime.now(),
        finished_at=datetime.datetime.now(),
    )


def test_create_dataset_files_reuses_matching_parsed_response(tmp_path):
    def parse_func(_row, _response):
        raise AssertionError("create_dataset_files should reuse the cached parsed response")

    processor = DummyRequestProcessor(RequestProcessorConfig(model="test-model", require_all_responses=False))
    processor.working_dir = str(tmp_path)
    processor.prompt_formatter = PromptFormatter(
        model_name="test-model",
        prompt_func=lambda row: row["value"],
        parse_func=parse_func,
    )

    request = _make_generic_response()
    request_path = tmp_path / "requests_0.jsonl"
    request_path.write_text(request.generic_request.model_dump_json() + "\n")

    response = _make_generic_response(parse_hash="parse-hash", parsed=[{"cached": "row"}])
    response_path = tmp_path / "responses_0.jsonl"
    response_path.write_text(response.model_dump_json() + "\n")

    dataset = processor.create_dataset_files("parse-hash")
    assert dataset.to_list() == [{"cached": "row"}]


def test_create_dataset_files_reprocesses_stale_parsed_response(tmp_path):
    parse_calls = 0

    def parse_func(row, response):
        nonlocal parse_calls
        parse_calls += 1
        return {"value": f"{row['value']}::{response}"}

    processor = DummyRequestProcessor(RequestProcessorConfig(model="test-model", require_all_responses=False))
    processor.working_dir = str(tmp_path)
    processor.prompt_formatter = PromptFormatter(
        model_name="test-model",
        prompt_func=lambda row: row["value"],
        parse_func=parse_func,
    )

    request = _make_generic_response()
    request_path = tmp_path / "requests_0.jsonl"
    request_path.write_text(request.generic_request.model_dump_json() + "\n")

    response = _make_generic_response(parse_hash="old-hash", parsed=[{"stale": "row"}])
    response_path = tmp_path / "responses_0.jsonl"
    response_path.write_text(response.model_dump_json() + "\n")

    dataset = processor.create_dataset_files("new-hash")
    assert parse_calls == 1
    assert dataset.to_list() == [{"value": "hello::hello"}]
