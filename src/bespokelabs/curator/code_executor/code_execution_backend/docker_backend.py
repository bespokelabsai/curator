"""Docker Code Execution Backend."""

import asyncio
import io
import multiprocessing
import os
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import docker
from docker.errors import DockerException

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequestParams, CodeExecutionResponse
from bespokelabs.curator.log import logger


class DockerCodeExecutionBackend(BaseCodeExecutionBackend):
    """Docker-based code execution backend."""

    PYTHON_IMAGE = "python:3.11-slim"
    WORKSPACE_DIR = "/workspace"

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.PYTHON_IMAGE = config.docker_image if config.docker_image else self.PYTHON_IMAGE
        self.client = docker.from_env()

        # Initialize process pool for parallel execution
        self.max_workers = getattr(config, "max_workers", multiprocessing.cpu_count())
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

        logger.debug(f"Initialized Docker backend with {self.max_workers} workers")

    def _ensure_image(self):
        """Ensure required Docker image is pulled."""
        try:
            self.client.images.get(self.PYTHON_IMAGE)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling {self.PYTHON_IMAGE} image...")
            self.client.images.pull(self.PYTHON_IMAGE)
        except DockerException as e:
            raise RuntimeError(f"Error ensuring Docker image: {str(e)}") from e

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionResponse:
        """Execute a single request in a Docker container using a process pool."""
        return await self.execute_standard_input_request(
            request.execution_request.code, request.execution_request.code_input, request.execution_request.execution_params
        )

    @classmethod
    def _create_temp_file(cls, content: str) -> str:
        """Create a temporary file with the given content."""
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            return temp_file.name

    def _create_tar_archive(self, data: bytes, filename: str) -> io.BytesIO:
        """Create a tar archive containing a single file."""
        tarstream = io.BytesIO()
        with tarfile.TarFile(fileobj=tarstream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        tarstream.seek(0)
        return tarstream

    def _get_container_file(self, container, file_path: str) -> str:
        """Get content of a file from a Docker container."""
        try:
            bits, _ = container.get_archive(file_path)
            bio = io.BytesIO()
            for chunk in bits:
                bio.write(chunk)
            bio.seek(0)

            with tarfile.open(fileobj=bio) as tar:
                member = tar.next()
                f = tar.extractfile(member)
                if f:
                    return f.read().decode("utf-8")
                return ""

        except docker.errors.NotFound:
            return ""

    def _setup_volume_with_files(self, volume, code_file: str, input_file: str):
        """Set up a volume with code and input files."""
        setup_container = self.client.containers.create(
            self.PYTHON_IMAGE, command=["sleep", "5"], volumes={volume.name: {"bind": self.WORKSPACE_DIR, "mode": "rw"}}
        )

        try:
            setup_container.start()
            for src, dest in [(code_file, "program.py"), (input_file, "input.txt")]:
                with open(src, "rb") as f:
                    data = f.read()
                    tarstream = self._create_tar_archive(data, dest)
                    setup_container.put_archive(self.WORKSPACE_DIR, tarstream)
        finally:
            setup_container.stop()
            setup_container.remove()

    def _get_created_files(self, container: docker.models.containers.Container, directory: str) -> str:
        """Get any files created during code execution."""
        # Get all files in the directory as a tar archive
        bits, _ = container.get_archive(directory, encode_stream=True)
        bio = io.BytesIO()
        for chunk in bits:
            bio.write(chunk)
        bio.seek(0)
        return str(bio.getvalue())

    def _create_execution_container(self, volume_name: str) -> docker.models.containers.Container:
        """Create the main execution container."""
        # TODO: Add a name to the container and volume so that they are unique for a each row
        return self.client.containers.create(
            self.PYTHON_IMAGE,
            command=[  # noqa: E501
                f"bash -c python {self.WORKSPACE_DIR}/program.py < {self.WORKSPACE_DIR}/input.txt > {self.WORKSPACE_DIR}/output.txt 2> {self.WORKSPACE_DIR}/error.txt; echo $? > {self.WORKSPACE_DIR}/exit_code.txt"  # noqa: E501
            ],
            volumes={volume_name: {"bind": self.WORKSPACE_DIR, "mode": "rw"}},
            working_dir=self.WORKSPACE_DIR,
            entrypoint=["/bin/sh", "-c"],
            user="root",
        )

    def _handle_execution_result(self, container, status_code: int) -> CodeExecutionOutput:
        """Handle the execution result and create appropriate output."""
        stdout = self._get_container_file(container, f"{self.WORKSPACE_DIR}/output.txt")
        stderr = self._get_container_file(container, f"{self.WORKSPACE_DIR}/error.txt")
        files = self._get_created_files(container, self.WORKSPACE_DIR)

        if status_code == 0:
            return CodeExecutionOutput(
                message="success",
                stdout=stdout,
                stderr=stderr,
                files=files,
            )

        # Create a more descriptive error message
        error_message = f"Program exited with status code {status_code}"
        if stderr:
            error_message = f"{error_message}\n\nError details:\n{stderr}"

        return CodeExecutionOutput(
            message="error",
            error=error_message,
            stdout=stdout,
            stderr=stderr,
            files=files,
        )

    async def execute_standard_input_request(self, code: str, code_input: str, execution_params: CodeExecutionRequestParams) -> CodeExecutionOutput:
        """Execute code in Docker container using a worker from the process pool."""
        # Create a function that can be executed in a separate process
        execution_func = partial(
            self._execute_in_process, code=code, code_input=code_input, timeout=120, python_image=self.PYTHON_IMAGE, workspace_dir=self.WORKSPACE_DIR
        )

        # Submit the task to the process pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.process_pool, execution_func)

        return result

    @staticmethod
    def _execute_in_process(code: str, code_input: str, timeout: int, python_image: str, workspace_dir: str) -> CodeExecutionOutput:
        """Static method to execute code in a separate process."""
        try:
            # Create a new Docker client in this process
            client = docker.from_env()

            # Create temporary files
            with (
                tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as code_file,
                tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as input_file,
            ):
                code_file.write(code)
                code_file_path = code_file.name

                input_file.write(code_input)
                input_file_path = input_file.name

            # Ensure image exists
            try:
                client.images.get(python_image)
            except docker.errors.ImageNotFound:
                client.images.pull(python_image)
            except DockerException as e:
                raise RuntimeError(f"Error ensuring Docker image: {str(e)}") from e

            container = None
            volume = None

            try:
                volume = client.volumes.create()

                # Set up volume with files
                setup_container = client.containers.create(python_image, command=["sleep", "5"], volumes={volume.name: {"bind": workspace_dir, "mode": "rw"}})

                try:
                    setup_container.start()
                    for src, dest in [(code_file_path, "program.py"), (input_file_path, "input.txt")]:
                        with open(src, "rb") as f:
                            data = f.read()
                            tarstream = DockerCodeExecutionBackend._create_tar_archive_static(data, dest)
                            setup_container.put_archive(workspace_dir, tarstream)
                finally:
                    setup_container.stop()
                    setup_container.remove()

                # Create and run execution container
                cmd = (
                    f"echo 'DEBUG: Starting container execution' && "
                    f"echo 'DEBUG: Workspace contents:' && ls -la {workspace_dir} && "
                    f"python {workspace_dir}/program.py < {workspace_dir}/input.txt > {workspace_dir}/output.txt 2> {workspace_dir}/error.txt; "
                    f"exit_code=$?; "
                    f"echo $exit_code > {workspace_dir}/exit_code.txt; "
                    f"echo 'DEBUG: Execution finished with code:' $exit_code; "
                    f"exit $exit_code"
                )

                container = client.containers.create(
                    python_image,
                    command=[cmd],
                    volumes={volume.name: {"bind": workspace_dir, "mode": "rw"}},
                    working_dir=workspace_dir,
                    entrypoint=["/bin/bash", "-c"],
                    user="root",
                )

                container.start()
                result = container.wait(timeout=timeout)

                # Handle results
                return DockerCodeExecutionBackend._handle_execution_result_static(container, result["StatusCode"], workspace_dir)

            except docker.errors.ContainerError as e:
                return CodeExecutionOutput(message="error", error=f"Docker container error: {str(e)}\n\nExit code: {e.exit_status}\n\nStderr: {e.stderr}")
            except docker.errors.ImageNotFound as e:
                return CodeExecutionOutput(message="error", error=f"Docker image not found: {str(e)}")
            except docker.errors.APIError as e:
                return CodeExecutionOutput(message="error", error=f"Docker API error: {str(e)}")
            except Exception as e:
                return CodeExecutionOutput(message="error", error=f"Error during code execution: {str(e)}\n\nType: {type(e).__name__}")
            finally:
                # Clean up
                try:
                    if container:
                        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
                        print(logs)
                        # container.remove(force=True)
                    # if volume:
                    # volume.remove(force=True)
                    for f in [code_file_path, input_file_path]:
                        if os.path.exists(f):
                            os.unlink(f)
                    client.close()
                except Exception:
                    pass

        except Exception as e:
            return CodeExecutionOutput(message="error", error=f"Process execution error: {str(e)}")

    @staticmethod
    def _create_tar_archive_static(data: bytes, filename: str) -> io.BytesIO:
        """Static version of _create_tar_archive for use in processes."""
        tarstream = io.BytesIO()
        with tarfile.TarFile(fileobj=tarstream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        tarstream.seek(0)
        return tarstream

    @staticmethod
    def _handle_execution_result_static(container, status_code: int, workspace_dir: str) -> CodeExecutionOutput:
        """Static version of _handle_execution_result for use in processes."""
        stdout = DockerCodeExecutionBackend._get_container_file_static(container, f"{workspace_dir}/output.txt")
        stderr = DockerCodeExecutionBackend._get_container_file_static(container, f"{workspace_dir}/error.txt")

        # Get created files
        try:
            bits, _ = container.get_archive(workspace_dir, encode_stream=True)
            bio = io.BytesIO()
            for chunk in bits:
                bio.write(chunk)
            bio.seek(0)
            files = str(bio.getvalue())
        except Exception:
            files = ""

        if status_code == 0:
            return CodeExecutionOutput(
                message="success",
                stdout=stdout,
                stderr=stderr,
                files=files,
            )

        # Create a more descriptive error message
        error_message = f"Program exited with status code {status_code}"
        if stderr:
            error_message = f"{error_message}\n\nError details:\n{stderr}"

        return CodeExecutionOutput(
            message="error",
            error=error_message,
            stdout=stdout,
            stderr=stderr,
            files=files,
        )

    @staticmethod
    def _get_container_file_static(container, file_path: str) -> str:
        """Static version of _get_container_file for use in processes."""
        try:
            bits, _ = container.get_archive(file_path)
            bio = io.BytesIO()
            for chunk in bits:
                bio.write(chunk)
            bio.seek(0)

            with tarfile.open(fileobj=bio) as tar:
                member = tar.next()
                f = tar.extractfile(member)
                if f:
                    return f.read().decode("utf-8")
                return ""

        except docker.errors.NotFound:
            return ""

    async def shutdown(self):
        """Cleanup resources including process pool."""
        self.process_pool.shutdown(wait=True)
        self.client.close()
        logger.debug("Shutting down Docker backend and process pool")
