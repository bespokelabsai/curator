import os

import pytest

from tests.helpers import run_script

"""
USAGE:
pytest -s tests/cache/test_different_files.py
"""


@pytest.mark.skip(reason="Temporarily disabled, since it takes a while")
@pytest.mark.cache_dir(os.path.expanduser("~/.cache/curator-tests/test-different-files"))
@pytest.mark.usefixtures("clear_test_cache")
def test_cache_behavior():
    cache_hit_log = "Using cached output dataset."

    # Run one.py twice and check for cache behavior
    print("RUNNING ONE.PY")
    output1, _ = run_script(["python", "tests/cache/different_files/one.py"])
    assert cache_hit_log not in output1, "First run of one.py should not hit cache"

    print("RUNNING ONE.PY AGAIN")
    output2, _ = run_script(["python", "tests/cache/different_files/one.py"])
    assert cache_hit_log in output2, "Second run of one.py should hit cache"

    # Run two.py and check for cache behavior
    print("RUNNING TWO.PY")
    output3, _ = run_script(["python", "tests/cache/different_files/two.py"])
    assert cache_hit_log in output3, "First run of two.py should hit cache"
