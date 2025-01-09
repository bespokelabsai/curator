import os
import subprocess
import sys
from pathlib import Path

import pytest

examples_dir = Path(__file__).parent.parent / "examples"
sys.path.append(str(examples_dir))


def get_example_scripts():
    """Get all Python scripts from the examples directory and its subdirectories."""
    example_scripts = []
    for root, _, files in os.walk(examples_dir):
        for file in files:
            if file.endswith(".py"):
                example_scripts.append(os.path.join(root, file))
    # Filter out prompt_templates.py and reannotate.py
    example_scripts = [script for script in example_scripts if not script.endswith(("prompt_templates.py", "wildchat.py", "openhermes.py"))]
    return example_scripts


@pytest.mark.integration
@pytest.mark.parametrize("script_path", get_example_scripts())
@pytest.mark.cache_dir(lambda script_path: os.path.expanduser(f"~/.cache/curator-tests/{Path(script_path).stem}"))
@pytest.mark.usefixtures("clear_test_cache")
@pytest.mark.dependency(name="first_run_{script_path}")
def test_example_script_first_run(script_path, monkeypatch, tmp_path):
    """Test that all example scripts can run without error (first run, no cache)."""
    print(f"\n\n====== RUNNING FIRST RUN of {script_path} ======\n\n")
    _run_example_script(script_path, monkeypatch, tmp_path)


def _run_example_script(script_path, monkeypatch, tmp_path):
    """Helper function to run an example script and handle errors."""
    # Change to a temporary directory to avoid writing files to the actual directory
    monkeypatch.chdir(tmp_path)

    script_name = os.path.basename(script_path)
    cmd = [sys.executable, script_path]
    if script_name == "synthesize.py":
        cmd.extend(["--template", "math", "--output_path", "math.jsonl"])
    try:
        subprocess.run(cmd, check=True, timeout=120)  # 2 minute timeout
    except subprocess.TimeoutExpired:
        pytest.fail(f"Script {script_path} timed out after 2 minutes")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script {script_path} failed with exit code {e.returncode}")
