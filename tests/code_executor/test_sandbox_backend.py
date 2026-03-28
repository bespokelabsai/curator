import sys
from types import ModuleType, SimpleNamespace

from bespokelabs.curator.code_executor.code_execution_backend import sandbox_backend


def _install_fake_sandbox(monkeypatch, sandbox_cls):
    module = ModuleType("bespokelabs.sandbox")
    module.Sandbox = sandbox_cls
    monkeypatch.setitem(sys.modules, "bespokelabs.sandbox", module)


def test_execute_in_sandbox_returns_error_for_nonzero_exit(monkeypatch):
    class FakeSandbox:
        def __init__(self, backend_name, **kwargs):
            self.backend_name = backend_name
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_file(self, path, content):
            pass

        def execute_command(self, command, args=None):
            return SimpleNamespace(exit_code=1, stdout="partial output", stderr="Traceback")

    _install_fake_sandbox(monkeypatch, FakeSandbox)
    monkeypatch.setattr(sandbox_backend, "_collect_sandbox_files", lambda sandbox: "files-archive")

    output = sandbox_backend._execute_in_sandbox(
        code="print('hi')",
        code_input="",
        timeout=10,
        backend_name="local",
        sandbox_kwargs={},
    )

    assert output.message == "error"
    assert output.error == "Program exited with status code 1\n\nError details:\nTraceback"
    assert output.stdout == "partial output"
    assert output.stderr == "Traceback"
    assert output.files == "files-archive"


def test_execute_in_sandbox_returns_timeout_for_timeout_exit(monkeypatch):
    class FakeSandbox:
        def __init__(self, backend_name, **kwargs):
            self.backend_name = backend_name
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_file(self, path, content):
            pass

        def execute_command(self, command, args=None):
            return SimpleNamespace(exit_code=124, stdout="", stderr="")

    _install_fake_sandbox(monkeypatch, FakeSandbox)
    monkeypatch.setattr(sandbox_backend, "_collect_sandbox_files", lambda sandbox: "files-archive")

    output = sandbox_backend._execute_in_sandbox(
        code="print('hi')",
        code_input="",
        timeout=7,
        backend_name="local",
        sandbox_kwargs={},
    )

    assert output.message == "timeout"
    assert output.error == "Execution timed out after 7s"
    assert output.files == "files-archive"


def test_execute_in_sandbox_preserves_files_when_command_raises(monkeypatch):
    class FakeSandbox:
        def __init__(self, backend_name, **kwargs):
            self.backend_name = backend_name
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_file(self, path, content):
            pass

        def execute_command(self, command, args=None):
            raise RuntimeError("sandbox command failed")

    _install_fake_sandbox(monkeypatch, FakeSandbox)
    monkeypatch.setattr(sandbox_backend, "_collect_sandbox_files", lambda sandbox: "files-archive")

    output = sandbox_backend._execute_in_sandbox(
        code="print('hi')",
        code_input="",
        timeout=10,
        backend_name="local",
        sandbox_kwargs={},
    )

    assert output.message == "error"
    assert output.error == "sandbox command failed"
    assert output.stdout is None
    assert output.stderr is None
    assert output.files == "files-archive"
