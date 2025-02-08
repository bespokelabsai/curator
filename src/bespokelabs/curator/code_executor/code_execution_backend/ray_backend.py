"""Ray-based Code Execution Backend."""

import asyncio
import logging
import os
import subprocess
import tempfile

import ray

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequestParams


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

    async def execute_standard_input_request(self, code: str, code_input: str, execution_params: CodeExecutionRequestParams) -> CodeExecutionOutput:
        """Execute code with standard input.

        Args:
            code: Source code
            code_input: Input to the code
            execution_params: Execution parameters

        Returns:
            CodeExecutionOutput: Execution results
        """
        temp_program_path = None
        output = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(code)
                temp_program_path = temp_file.name
                self.temp_files.append(temp_program_path)

            try:
                result = subprocess.run(["python", temp_program_path], input=code_input, text=True, capture_output=True, timeout=execution_params.timeout)
                output = CodeExecutionOutput(
                    message="success",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            except subprocess.TimeoutExpired:
                output = CodeExecutionOutput(message="timeout", error=f"Execution timed out after {execution_params.timeout}s")

            except Exception as e:
                output = CodeExecutionOutput(message="error", error=str(e))

        finally:
            if temp_program_path and temp_program_path in self.temp_files:
                try:
                    os.unlink(temp_program_path)
                    self.temp_files.remove(temp_program_path)
                except Exception:
                    pass

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
        code = request.execution_request.code
        code_input = request.execution_request.code_input
        execution_params = request.execution_request.execution_params

        # Execute request in the remote executor
        result = await asyncio.get_event_loop().run_in_executor(
            None, ray.get, self.executor.execute_standard_input_request.remote(code, code_input, execution_params)
        )

        return result

    def shutdown(self):
        """Cleanup resources when shutting down the backend."""
        if ray.is_initialized():
            ray.shutdown()
