from dataclasses import dataclass
from typing import Callable

from bespokelabs.curator.code_executor.types import CodeExecutionRequest, CodeExecutionRequestParams


@dataclass
class CodeFormatter:
    """Formatter for the code execution backend."""

    code: Callable
    code_input: Callable
    code_output: Callable
    execution_params: CodeExecutionRequestParams

    def create_code_execution_request(self, row: dict, idx: int, execution_directory: str) -> CodeExecutionRequest:
        """Format the request object based off of `LLM` attributes.

        Args:
            row: Input data to format into a prompt
            idx: Index of the row in the dataset
            execution_directory: Directory to create the temp file in

        Returns:
            CodeExecutionRequest object containing the formatted request
        """
        return CodeExecutionRequest(
            code=self.code(row),
            code_input=self.code_input(row),
            execution_params=self.execution_params,
            original_row=row,
            original_row_idx=idx,
            execution_directory=execution_directory,
        )
