import os
import re
import shutil
import subprocess
import time

import pytest


@pytest.fixture
def clear_test_cache(request):
    """Fixture to ensure clean caches before tests"""
    # Get cache_dir from marker if provided, otherwise use default
    marker = request.node.get_closest_marker("cache_dir")
    if marker and callable(marker.args[0]):
        # If the marker arg is a function (lambda), call it with the test params
        cache_dir = marker.args[0](request.node.callspec.params["script_path"])
    else:
        cache_dir = marker.args[0] if marker else None

    os.environ["CURATOR_CACHE_DIR"] = cache_dir

    # Delete caches before test
    shutil.rmtree(cache_dir, ignore_errors=True)

    # Run test
    yield


def run_script(script, stop_line_pattern=None, env=None):
    process = subprocess.Popen(
        script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )

    lines = ""
    for line in process.stderr:
        print(line, end="")  # Print each line as it is received
        lines += line
        if stop_line_pattern and re.search(stop_line_pattern, line):
            process.terminate()
            break

    for line in process.stdout:
        print(line, end="")  # Print each line as it is received
        lines += line
        if stop_line_pattern and re.search(stop_line_pattern, line):
            process.terminate()
            break

    process.wait()
    return lines, process.returncode
