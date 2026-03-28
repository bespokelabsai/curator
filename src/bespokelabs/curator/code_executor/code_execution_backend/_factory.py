"""Factory for creating code execution backends."""

from typing import Optional

from bespokelabs.curator.code_executor.types import CodeExecutionBackendConfig

# Map old backend names to sandbox backend names
_BACKEND_ALIASES = {
    "multiprocessing": "local",
}


class _CodeExecutionBackendFactory:
    """Factory for creating code execution backends."""

    @classmethod
    def create(cls, backend: str, backend_params: Optional[dict] = None):
        """Create a code execution backend.

        Args:
            backend: Backend name (e.g. "local", "docker", "e2b", "modal", "daytona")
            backend_params: Configuration parameters for the backend

        Returns:
            SandboxCodeExecutionBackend instance
        """
        if backend_params is None:
            backend_params = {}

        backend_name = _BACKEND_ALIASES.get(backend, backend)
        config = CodeExecutionBackendConfig(**backend_params)

        from bespokelabs.curator.code_executor.code_execution_backend.sandbox_backend import SandboxCodeExecutionBackend

        return SandboxCodeExecutionBackend(config, backend_name=backend_name)
