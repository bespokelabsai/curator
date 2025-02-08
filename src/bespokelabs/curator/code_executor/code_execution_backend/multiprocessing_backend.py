"""Multiprocessing Code Execution Backend."""

import asyncio
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequestParams

logger = logging.getLogger(__name__)


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
            request.execution_request.code,
            request.execution_request.code_input,
            request.execution_request.execution_params,
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

    @classmethod
    def execute_standard_input_request(cls, code: str, code_input: str, execution_params: CodeExecutionRequestParams) -> CodeExecutionOutput:
        """Execute code with function calls and test cases.

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
            temp_program_path = cls._create_temp_file(code)
            try:
                result = subprocess.run(["python", temp_program_path], input=code_input, text=True, capture_output=True, timeout=execution_params.timeout)
                output = CodeExecutionOutput(
                    message="success",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            except subprocess.TimeoutExpired:
                output = CodeExecutionOutput(
                    message="timeout",
                    error=f"Execution timed out after {execution_params.timeout}s",
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
