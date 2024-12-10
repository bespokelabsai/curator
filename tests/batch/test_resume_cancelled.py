import pytest
import time
import os
from tests.helpers import run_script
from tests.helpers import prepare_test_cache
from bespokelabs.curator.request_processor.openai_batch_request_processor import BatchManager

"""
USAGE:
pytest -s tests/batch/test_resume_cancelled.py
"""


@pytest.mark.cache_dir(os.path.expanduser("~/.cache/curator-tests/test-batch-resume-cancelled"))
@pytest.mark.usefixtures("prepare_test_cache")
def test_batch_resume():
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

    cache_dir = os.getenv("CURATOR_CACHE_DIR")
    child_folder = os.listdir(cache_dir)[0]
    working_dir = os.path.join(cache_dir, child_folder)
    print(f"CANCELLING BATCHES in {working_dir}")
    batch_manager = BatchManager(
        working_dir,
        delete_successful_batch_files=True,
        delete_failed_batch_files=True,
    )
    submitted_batch_ids = batch_manager.get_submitted_batch_ids()
    downloaded_batch_ids = batch_manager.get_downloaded_batch_ids()
    not_downloaded_batch_id = list(submitted_batch_ids - downloaded_batch_ids)[0]
    print(f"Submitted batch IDs: {submitted_batch_ids}")
    print(f"Downloaded batch IDs: {downloaded_batch_ids}")
    print(f"Not downloaded batch ID: {not_downloaded_batch_id}")
    batch_manager.cancel_batch(not_downloaded_batch_id)
    batch_object = batch_manager.retrieve_batch(not_downloaded_batch_id)
    # takes a while for the batch to be cancelled
    while batch_object.status != "cancelled":
        time.sleep(10)
        batch_object = batch_manager.retrieve_batch(not_downloaded_batch_id)

    # Second run should process the remaining batch, and resubmit the cancelled batch
    print("SECOND RUN")
    output2, _ = run_script(script, env=env)
    print(output2)

    # checks
    assert "1 out of 2 batches already downloaded." in output2
    assert "0 out of 1 remaining batches are already submitted." in output2
