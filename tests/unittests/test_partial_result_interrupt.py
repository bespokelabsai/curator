import datetime
import json
import os
from pathlib import Path

import pytest
from datasets import Dataset

from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.config import RequestProcessorConfig
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse


class _PromptFormatter:
    response_format = None

    def response_to_response_format(self, response):
        return response

    def parse_func(self, row, response):
        return {"prompt": row["prompt"], "answer": response}


class _InterruptingProcessor(BaseRequestProcessor):
    @property
    def backend(self) -> str:
        return "test"

    def validate_config(self):
        pass

    def create_request_files(self, dataset):
        request_file = os.path.join(self.working_dir, "requests_0.jsonl")
        with open(request_file, "w") as f:
            for idx, row in enumerate(dataset):
                request = GenericRequest(
                    model=self.config.model,
                    messages=[{"role": "user", "content": row["prompt"]}],
                    original_row=row,
                    original_row_idx=idx,
                )
                f.write(request.model_dump_json() + "\n")
        return [request_file]

    def requests_to_responses(self, generic_request_files):
        request = GenericRequest(
            model=self.config.model,
            messages=[{"role": "user", "content": "first"}],
            original_row={"prompt": "first"},
            original_row_idx=0,
        )
        response = GenericResponse(
            response_message="done",
            raw_response={},
            raw_request={},
            generic_request=request,
            created_at=datetime.datetime.now(),
            finished_at=datetime.datetime.now(),
        )
        response_file = os.path.join(self.working_dir, "responses_0.jsonl")
        with open(response_file, "w") as f:
            f.write(json.dumps(response.model_dump(), default=str) + "\n")
        raise KeyboardInterrupt


def test_interrupt_preserves_successful_responses_when_enabled(tmp_path: Path) -> None:
    processor = _InterruptingProcessor(
        RequestProcessorConfig(
            model="test-model",
            allow_partial_result_on_interrupt=True,
        )
    )

    dataset = Dataset.from_list([{"prompt": "first"}, {"prompt": "second"}])
    partial = processor.run(
        dataset=dataset,
        working_dir=str(tmp_path),
        parse_func_hash="parse-hash",
        prompt_formatter=_PromptFormatter(),
    )

    assert len(partial) == 1
    assert partial[0] == {"prompt": "first", "answer": "done"}

    failed_requests_path = tmp_path / "failed_requests.jsonl"
    failed_requests = [json.loads(line) for line in failed_requests_path.read_text().splitlines()]
    assert [request["original_row_idx"] for request in failed_requests] == [1]


def test_interrupt_keeps_default_keyboard_interrupt_behavior(tmp_path: Path) -> None:
    processor = _InterruptingProcessor(RequestProcessorConfig(model="test-model"))
    dataset = Dataset.from_list([{"prompt": "first"}, {"prompt": "second"}])

    with pytest.raises(KeyboardInterrupt):
        processor.run(
            dataset=dataset,
            working_dir=str(tmp_path),
            parse_func_hash="parse-hash",
            prompt_formatter=_PromptFormatter(),
        )
