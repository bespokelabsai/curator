"""Docker Code Execution Backend."""

import io
import logging
import os
import tarfile
import tempfile

import aiodocker
from aiodocker.exceptions import DockerError

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequestParams, CodeExecutionResponse

logger = logging.getLogger(__name__)


class DockerCodeExecutionBackend(BaseCodeExecutionBackend):
    """Docker-based code execution backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.client = None  # Will be initialized in execute_request
        logger.debug("Initialized Docker backend")

    async def _ensure_client(self):
        """Ensure aiodocker client is initialized."""
        if self.client is None:
            self.client = aiodocker.Docker()
            # Pull image asynchronously
            try:
                await self.client.images.pull("python:3.9-slim")
            except DockerError as e:
                raise RuntimeError(f"Failed to pull Docker image: {e}") from e

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionResponse:
        """Execute a single request in a Docker container."""
        return await self.execute_standard_input_request(
            request.execution_request.code, request.execution_request.code_input, request.execution_request.execution_params
        )

    @classmethod
    def _create_temp_file(cls, content: str) -> str:
        """Create a temporary file with the given content.

        Args:
            content: Content to write to temp file

        Returns:
            Path to the created temp file
        """
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            return temp_file.name

    @staticmethod
    async def get_file_content(container, file_path="/tmp/stdout.txt"):
        """Get the content of a file from a Docker container."""
        # Get the tar archive stream and metadata
        stream, stat = await container.get_archive(file_path)

        # Read all bytes from the stream.
        # Note: Depending on your Docker SDK version, stream might be an async iterator.
        file_bytes = b""
        async for chunk in stream:
            file_bytes += chunk

        # Create a file-like object from the bytes
        file_like_object = io.BytesIO(file_bytes)

        # Open the tar archive
        with tarfile.open(fileobj=file_like_object) as tar:
            # The tar archive contains files with relative paths.
            # List the members to see what the file is named.
            # It may be just "stdout.txt" or include a folder name.
            for member in tar.getmembers():
                if member.name.endswith("stdout.txt"):
                    extracted_file = tar.extractfile(member)
                    if extracted_file is None:
                        raise ValueError("Failed to extract the file from the archive")
                    # Read and decode the content
                    content = extracted_file.read().decode("utf-8")
                    return content
        # If the file wasn't found in the archive, raise an error
        raise FileNotFoundError(f"{file_path} not found in the container archive")

    async def execute_standard_input_request(self, code: str, code_input: str, execution_params: CodeExecutionRequestParams) -> CodeExecutionResponse:
        """Execute code in Docker container.

        Args:
            code: Source code
            code_input: Input to the code
            execution_params: Execution parameters

        Returns:
            CodeExecutionResponse: Execution results
        """
        await self._ensure_client()
        temp_program_path = None
        output = None
        try:
            temp_program_path = self._create_temp_file(code)

            try:
                config = {
                    "Cmd": ["/bin/sh", "-c", "python /code/program.py"],
                    "Image": "python:3.9-slim",
                    "Volumes": {"/code": {}},
                    "OpenStdin": True,
                    "Tty": True,
                    "StopTimeout": 1,
                }

                container = await self.client.containers.create(config=config)
                await container.put_archive("/code", self._make_tarfile(temp_program_path))

                await container.put_archive("/code/input.txt", code_input.encode())

                try:
                    await container.start()
                    exec_config = {
                        "Cmd": [
                            "sh",
                            "-c",
                            f"timeout {execution_params.timeout} python /code/program.py < /code/input.txt 2>/tmp/stderr.txt 1>/tmp/stdout.txt",
                        ],
                        "AttachStdout": True,
                        "AttachStderr": True,
                    }

                    exec_obj = await container.exec.create(exec_config)
                    output = await exec_obj.start()

                    stdout_file = await self.get_file_content(container, "/tmp/stdout.txt")
                    stderr_file = await self.get_file_content(container, "/tmp/stderr.txt")

                    output = CodeExecutionOutput(message="success", stdout=stdout_file, stderr=stderr_file)

                except DockerError as e:
                    if "timeout" in str(e).lower():
                        output = CodeExecutionOutput(message="error", error=f"Execution timed out after {execution_params.timeout_seconds} seconds")
                    else:
                        output = CodeExecutionOutput(message="error", error=str(e))

                finally:
                    try:
                        await container.stop()
                        await container.delete()
                    except Exception:
                        pass

            except DockerError as e:
                output = CodeExecutionOutput(message="error", error=f"Docker error: {str(e)}")

        finally:
            if temp_program_path and os.path.exists(temp_program_path):
                os.unlink(temp_program_path)

        return output

    async def shutdown(self):
        """Cleanup Docker resources."""
        if self.client:
            await self.client.close()
            logger.debug("Shutting down Docker backend")

    def _make_tarfile(self, source_path):
        """Create a tar archive containing the source file."""
        import io
        import tarfile

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(source_path, arcname="program.py")
        tar_stream.seek(0)
        return tar_stream
