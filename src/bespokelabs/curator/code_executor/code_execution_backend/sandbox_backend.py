"""Sandbox Code Execution Backend using bespokelabs-sandbox."""

import asyncio
import io
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput
from bespokelabs.curator.log import logger

WORKSPACE_DIR = "/workspace"


class SandboxCodeExecutionBackend(BaseCodeExecutionBackend):
    """Code execution backend using bespokelabs-sandbox."""

    def __init__(self, config, backend_name="local"):
        """Initialize the sandbox backend.

        Args:
            config: Backend configuration
            backend_name: Sandbox backend name (e.g. "local", "docker", "e2b", "modal", "daytona")
        """
        super().__init__(config)
        self.config = config
        self.backend_name = backend_name
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Build sandbox constructor kwargs from config
        self.sandbox_kwargs = {}
        # Prefer explicit `image`, fall back to legacy `docker_image`
        image = config.image or config.docker_image
        if image:
            self.sandbox_kwargs["image"] = image
        if config.base_url:
            self.sandbox_kwargs["base_url"] = config.base_url

        logger.debug(f"Initialized sandbox backend with backend={backend_name}")

    @property
    def backend(self) -> str:
        """Backend property."""
        return self.backend_name

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionOutput:
        """Execute a single request using a sandbox."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            partial(
                _execute_in_sandbox,
                code=request.execution_request.code,
                code_input=request.execution_request.code_input or "",
                timeout=request.execution_request.execution_params.timeout if request.execution_request.execution_params else 10,
                backend_name=self.backend_name,
                sandbox_kwargs=self.sandbox_kwargs,
            ),
        )

    def __del__(self):
        """Clean up thread pool when object is destroyed."""
        self.thread_pool.shutdown(wait=True)
        logger.debug("Shutting down sandbox backend")


def _execute_in_sandbox(
    code: str,
    code_input: str,
    timeout: int,
    backend_name: str,
    sandbox_kwargs: dict,
) -> CodeExecutionOutput:
    """Execute code in a bespokelabs-sandbox.

    This is a module-level function so it can be used with ProcessPoolExecutor if needed.
    """
    from bespokelabs.sandbox import Sandbox

    try:
        kwargs = {**sandbox_kwargs}
        kwargs.setdefault("timeout_secs", timeout + 30)

        with Sandbox(backend_name, **kwargs) as sb:
            sb.write_file(f"{WORKSPACE_DIR}/program.py", code)
            sb.write_file(f"{WORKSPACE_DIR}/input.txt", code_input)

            result = sb.execute_command("bash", args=["-c", f"cd {WORKSPACE_DIR} && python program.py < input.txt"])

            files = _collect_sandbox_files(sb)

            return CodeExecutionOutput(
                message="success",
                stdout=result.stdout,
                stderr=result.stderr,
                files=files,
            )

    except Exception as e:
        return CodeExecutionOutput(
            message="error",
            error=str(e),
        )


def _collect_sandbox_files(sandbox) -> str:
    """Collect files created during execution as a tar archive string.

    Preserves directory structure relative to WORKSPACE_DIR so that
    nested output paths (e.g. media/videos/…) survive round-tripping.
    """
    try:
        file_list = sandbox.list_files(WORKSPACE_DIR)
        tar_buffer = io.BytesIO()

        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for file_info in file_list:
                if file_info.is_dir:
                    continue
                rel_path = os.path.relpath(file_info.path, WORKSPACE_DIR)
                content = sandbox.read_file(file_info.path)
                if isinstance(content, str):
                    content = content.encode()
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.size = len(content)
                tar.addfile(tarinfo, io.BytesIO(content))

        tar_buffer.seek(0)
        return str(tar_buffer.getvalue())
    except Exception:
        return ""
