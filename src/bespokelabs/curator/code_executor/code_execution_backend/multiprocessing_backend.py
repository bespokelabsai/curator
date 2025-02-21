"""Multiprocessing Code Execution Backend."""

import asyncio
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequest
from bespokelabs.curator.log import logger


class MultiprocessingCodeExecutionBackend(BaseCodeExecutionBackend):
    """Multiprocessing Code Execution Backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())
        logger.debug(f"Initialized multiprocessing backend with {os.cpu_count()} workers")

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionOutput:
        """Execute a single request."""
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self.process_pool,
            self.execute_standard_input_request,
            request.execution_request,
        )

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
            temp_program_path = cls._create_temp_file(request.code, request.execution_directory)
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
                    files=cls._get_created_files(program_dir),
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

    def __del__(self):
        """Clean up pool when object is destroyed."""
        self.process_pool.shutdown(wait=True)
        logger.debug("Shutting down multiprocessing backend")
