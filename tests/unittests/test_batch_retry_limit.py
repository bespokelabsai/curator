import asyncio
from datetime import datetime, timezone

import pytest

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts
from bespokelabs.curator.types.generic_request import GenericRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [0, 1, 2])
async def test_failed_batches_exhaust_the_retry_budget(tmp_path, monkeypatch, max_retries):
    processor = OpenAIBatchRequestProcessor(
        BatchRequestProcessorConfig(
            model="gpt-4o-mini",
            api_key="test-key",
            max_retries=max_retries,
        )
    )
    processor.prompt_formatter = PromptFormatter(model_name="gpt-4o-mini", prompt_func=lambda row: "hello")
    request_file = tmp_path / "requests_0.jsonl"
    request_file.write_text(
        GenericRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            original_row={},
            original_row_idx=0,
        ).model_dump_json()
        + "\n"
    )
    processor.batch_objects_file = str(tmp_path / "batch_objects.jsonl")
    processor._batch_objects_file_lock = asyncio.Lock()
    processor.tracker = BatchStatusTracker(unsubmitted_request_files={str(request_file)}, n_total_requests=1)
    monkeypatch.setattr(BatchStatusTracker, "update_display", lambda self: None)
    submitted = []

    async def submit(requests, metadata):
        now = datetime.now(timezone.utc)
        batch = GenericBatch(
            request_file=metadata["request_file"],
            id=f"batch-{len(submitted)}",
            created_at=now,
            finished_at=None,
            status="submitted",
            api_key_suffix="test",
            raw_status="validating",
            raw_batch={},
            request_counts=GenericBatchRequestCounts(failed=0, succeeded=0, total=1, raw_request_counts_object={}),
        )
        submitted.append(batch)
        return batch

    monkeypatch.setattr(processor, "submit_batch", submit)
    try:
        batch = await processor.submit_batch_from_request_file(str(request_file), set())
        assert batch.attempts_left == max_retries
        for remaining in range(max_retries - 1, -1, -1):
            await processor._tag_batch_as_retry(batch)
            assert batch.attempts_left == remaining
            await processor.resubmit_batch(batch)
            batch = submitted[-1]
            assert batch.attempts_left == remaining
        await processor._tag_batch_as_retry(batch)
        assert processor.tracker.to_resubmit_batches == {}
        assert len(submitted) == max_retries + 1
    finally:
        await processor.client.close()
