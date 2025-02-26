"""E2B Code Execution Backend.

See https://e2b.dev/ for more information.
"""

import logging

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequest

logger = logging.getLogger(__name__)


class E2BCodeExecutionBackend(BaseCodeExecutionBackend):
    """E2B Code Execution Backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        try:
            from e2b_code_interpreter import AsyncSandbox
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("""Please install 'e2b' extra to use E2BExecutor""") from e
        self.Sandbox = AsyncSandbox  # Store the class instead of instance

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionOutput:
        """Execute a single request."""
        self.sandbox = await self.Sandbox.create()  # Initialize sandbox for this request
        try:
            return await self.execute_standard_input_request(request.execution_request)
        finally:
            await self.sandbox.kill()

    async def execute_standard_input_request(self, request: CodeExecutionRequest) -> CodeExecutionOutput:
        """Execute code with E2B sandbox.

        Args:
            request: CodeExecutionRequest

        Returns:
            CodeExecutionOutput: Execution results
        """
        try:
            # Run program in E2B sandbox
            # save code and input to files in the sandbox root directory
            await self.sandbox.files.write("/code.py", request.code)
            await self.sandbox.files.write("/input.txt", request.code_input)

            result = await self.sandbox.run_code(
                "python /code.py < /input.txt > /output.txt 2> /error.txt",
                language="bash",
                timeout=request.execution_params.timeout,
            )

            if result.error:
                return CodeExecutionOutput(
                    message="error",
                    error=f"{result.error.name}: {result.error.value}\n{result.error.traceback}",
                )

            return CodeExecutionOutput(
                message="success",
                stdout=await self.sandbox.files.read("/output.txt"),
                stderr=await self.sandbox.files.read("/error.txt"),
                # files=await self.sandbox.files.read("/"),
                # TODO: e2b doesn't support seamless reading of an entire directory
            )

        except Exception as e:
            return CodeExecutionOutput(
                message="error",
                error=str(e),
            )

    def __del__(self):
        """Clean up when object is destroyed."""
        logger.debug("Shutting down E2B backend")
