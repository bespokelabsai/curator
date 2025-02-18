"""Ray-based Code Execution Backend."""

import asyncio
import logging
import os
import subprocess

import ray

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequest


@ray.remote
class CodeExecutorActor:
    """Ray actor for executing code in isolation."""

    def __init__(self):
        """Initialize the actor."""
        self.temp_files = []

    def __del__(self):
        """Cleanup temporary files on actor shutdown."""
        if hasattr(self, "temp_files") and self.temp_files:
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception:
                    pass

    @classmethod
    def execute_standard_input_request(cls, request: CodeExecutionRequest) -> CodeExecutionOutput:
        """Execute code with function calls and test cases.

        Args:
            request: CodeExecutionRequest

        Returns:
            CodeExecutionOutput: Execution results
        """
        temp_program_path = None
        output = None
        try:
            temp_program_path = BaseCodeExecutionBackend._create_temp_file(request.code, request.execution_directory)
            try:
                # Get directory containing the program file
                program_dir = os.path.dirname(temp_program_path)

                # Run program from its directory
                result = subprocess.run(
                    ["python", "program.py"],
                    input=request.code_input,
                    text=True,
                    capture_output=True,
                    timeout=request.execution_params.timeout,
                    cwd=request.execution_directory,
                )

                output = CodeExecutionOutput(
                    message="success",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    files=BaseCodeExecutionBackend._get_created_files(program_dir),
                )

            except subprocess.TimeoutExpired:
                output = CodeExecutionOutput(
                    message="timeout",
                    error=f"Execution timed out after {request.execution_params.timeout}s",
                )

            except Exception as e:
                output = CodeExecutionOutput(
                    message="error",
                    error=str(e),
                )

        finally:
            if temp_program_path:
                os.unlink(temp_program_path)

        return output


class RayCodeExecutionBackend(BaseCodeExecutionBackend):
    """Ray-based code execution backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize Ray if needed
        if not ray.is_initialized():
            if self.config.base_url:
                ray.init(address=self.config.base_url)
            else:
                ray.init(num_cpus=os.cpu_count())

        self.executor = CodeExecutorActor.remote()

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionOutput:
        """Execute a single request using Ray actors."""
        # Execute request in the remote executor
        result = await asyncio.get_event_loop().run_in_executor(None, ray.get, self.executor.execute_standard_input_request.remote(request.execution_request))

        return result

    async def shutdown(self):
        """Cleanup resources when shutting down the backend."""
        if ray.is_initialized():
            ray.shutdown()
