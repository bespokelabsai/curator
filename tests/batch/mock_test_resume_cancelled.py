import pytest
import time
import os
from unittest.mock import AsyncMock, patch
from openai.types import Batch, FileObject
from openai.types.batch import Errors
from openai.types.batch_request_counts import BatchRequestCounts
from tests.helpers import run_script
from tests.helpers import prepare_test_cache
from bespokelabs.curator.request_processor.openai_batch_request_processor import BatchManager

"""
USAGE:
pytest -s tests/batch/mock_test_resume_cancelled.py
"""

# https://platform.openai.com/docs/api-reference/batch/create
def create_mock_batch(
    batch_id: str,
    request_file_name: str,
    status: str = "in_progress",
    total_requests: int = 1,
) -> Batch:
    """Helper function to create mock Batch objects"""
    return Batch(
        id=batch_id,
        created_at=1234567890,
        error_file_id=None,
        errors=None,
        expires_at=None,
        failed_at=None,
        input_file_id="file-123",
        output_file_id="file-456" if status == "completed" else None,
        status=status,
        request_counts=BatchRequestCounts(
            completed=total_requests if status == "completed" else 0,
            failed=0,
            total=total_requests,
        ),
        metadata={"request_file_name": request_file_name},
        completion_window="24h",
        endpoint="/v1/chat/completions",
        object="batch",
        in_progress_at=None,
        finalizing_at=None,
        completed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
    )


@pytest.mark.cache_dir(
    os.path.expanduser("~/.cache/curator-tests/mock-test-batch-resume-cancelled")
)
@pytest.mark.usefixtures("prepare_test_cache")
@patch("openai.AsyncOpenAI")
async def test_batch_resume(mock_openai):
    # Setup mock responses
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client

    # Mock batch creation
    mock_client.batches.create.return_value = create_mock_batch(
        "batch_" + "a" * 32, request_file_name="requests.jsonl"
    )

    # Mock file creation
    mock_client.files.create.return_value = FileObject(
        id="file-123",
        bytes=1000,
        created_at=1234567890,
        filename="test.jsonl",
        object="file",
        purpose="batch",
        status="processed",
    )

    # Mock file processing wait
    mock_client.files.wait_for_processing.return_value = FileObject(
        id="file-123",
        bytes=1000,
        created_at=1234567890,
        filename="test.jsonl",
        object="file",
        purpose="batch",
        status="processed",
    )

    # Setup batch retrieval sequence
    mock_batch_sequence = [
        create_mock_batch(
            "batch_" + "a" * 32, "requests.jsonl", status="in_progress"
        ),  # First check
        create_mock_batch(
            "batch_" + "a" * 32, "requests.jsonl", status="cancelled"
        ),  # After cancellation
    ]
    mock_client.batches.retrieve.side_effect = mock_batch_sequence

    script = [
        "python",
        "tests/batch/simple_batch.py",
        "--log-level",
        "DEBUG",
        "--n-requests",
        "2",
        "--batch-size",
        "1",
        "--batch-check-interval",
        "10",
    ]

    env = os.environ.copy()

    print("FIRST RUN")
    stop_line_pattern = r"Marked batch ID batch_[a-f0-9]{32} as downloaded"
    output1, _ = run_script(script, stop_line_pattern, env=env)
    print(output1)

    # Small delay to ensure files are written
    time.sleep(1)

    # cache_dir = os.getenv("CURATOR_CACHE_DIR")
    # child_folder = os.listdir(cache_dir)[0]
    # working_dir = os.path.join(cache_dir, child_folder)
    # print(f"CANCELLING BATCHES in {working_dir}")
    # batch_manager = BatchManager(
    #     working_dir,
    #     delete_successful_batch_files=True,
    #     delete_failed_batch_files=True,
    # )
    # submitted_batch_ids = batch_manager.get_submitted_batch_ids()
    # downloaded_batch_ids = batch_manager.get_downloaded_batch_ids()
    # not_downloaded_batch_id = list(submitted_batch_ids - downloaded_batch_ids)[0]
    # print(f"Submitted batch IDs: {submitted_batch_ids}")
    # print(f"Downloaded batch IDs: {downloaded_batch_ids}")
    # print(f"Not downloaded batch ID: {not_downloaded_batch_id}")

    # Mock batch cancellation
    # mock_client.batches.cancel.return_value = None

    # # Reset batch retrieval sequence for second run
    # mock_batch_sequence = [
    #     create_mock_batch(
    #         "batch_" + "a" * 32, "requests.jsonl", status="cancelled"
    #     ),  # Initial check
    #     create_mock_batch(
    #         "batch_" + "b" * 32, "requests.jsonl", status="completed", total_requests=2
    #     ),  # New batch
    # ]
    # mock_client.batches.retrieve.side_effect = mock_batch_sequence

    # batch_manager.cancel_batch(not_downloaded_batch_id)
    # batch_object = batch_manager.retrieve_batch(not_downloaded_batch_id)
    # # takes a while for the batch to be cancelled
    # while batch_object.status != "cancelled":
    #     time.sleep(10)
    #     batch_object = batch_manager.retrieve_batch(not_downloaded_batch_id)

    # # Second run should process the remaining batch, and resubmit the cancelled batch
    # print("SECOND RUN")
    # output2, _ = run_script(script, env=env)
    # print(output2)

    # # checks
    # assert "1 out of 2 batches already downloaded." in output2
    # assert "0 out of 1 remaining batches are already submitted." in output2
