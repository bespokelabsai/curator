"""Check that a source install can share its namespace with another project."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("source_first", [True, False])
def test_separate_namespace_portions(tmp_path: Path, source_first: bool):
    source = Path(__file__).resolve().parents[2] / "src"
    sibling = tmp_path / "other-project"
    package = sibling / "bespokelabs" / "namespace_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 42\n")
    paths = [str(source), str(sibling)]
    if not source_first:
        paths.reverse()
    code = f"""
import sys
from importlib.util import find_spec
sys.path[:0] = {paths!r}
from bespokelabs import namespace_probe
import bespokelabs
assert namespace_probe.VALUE == 42
assert bespokelabs.__file__ is None
assert find_spec('bespokelabs.curator').origin == {str(source / 'bespokelabs/curator/__init__.py')!r}
"""
    # Disable site-packages so installed siblings cannot mask the source layout.
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True, capture_output=True, text=True)
