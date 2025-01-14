import os
import time

import pytest

from tests.helpers import run_script

"""
USAGE:
pytest -s tests/batch/test_resume.py
"""


@pytest.mark.skip(reason="Temporarily disabled, since it takes a while")
@pytest.mark.cache_dir(os.path.expanduser("~/.cache/curator-tests/test-batch-resume"))
@pytest.mark.usefixtures("clear_test_cache")
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

    # Second run should process the remaining batch
    print("SECOND RUN")
    output2, _ = run_script(script, env=env)
    print(output2)

    # checks
    assert "1 out of 2 batches already downloaded." in output2
    assert "1 out of 1 remaining batches are already submitted." in output2
