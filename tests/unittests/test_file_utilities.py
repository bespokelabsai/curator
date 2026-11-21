import os
import subprocess
import sys
import textwrap
import types

import pytest

from bespokelabs.curator.file_utilities import raise_open_file_limit


@pytest.mark.parametrize(
    "hard, expected_soft",
    [
        (4096, 4096),
        (2**63 - 1, 10_000_000),  # RLIM_INFINITY on Linux/macOS
    ],
)
def test_raise_open_file_limit(monkeypatch, hard, expected_soft):
    calls = []
    fake_resource = types.SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda which: (256, hard),
        setrlimit=lambda which, limits: calls.append((which, limits)),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    raise_open_file_limit()

    assert calls == [(7, (expected_soft, hard))]


def test_raise_open_file_limit_without_resource_module(monkeypatch):
    # Setting a module to None in sys.modules makes importing it fail,
    # which is what happens on Windows.
    monkeypatch.setitem(sys.modules, "resource", None)

    raise_open_file_limit()


def test_curator_works_without_resource_module():
    """Regression test for #290: creating an LLM or CodeExecutor failed on Windows."""
    script = textwrap.dedent(
        """
        import sys

        sys.modules["resource"] = None

        from bespokelabs import curator

        curator.LLM(model_name="gpt-4o-mini", batch=True)


        class Executor(curator.CodeExecutor):
            def code(self, row):
                return "print(1)"


        Executor(backend="local")
        """
    )
    env = {**os.environ, "OPENAI_API_KEY": "sk-test", "TELEMETRY_ENABLED": "false"}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr
