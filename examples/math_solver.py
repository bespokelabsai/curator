"""Math problem solver using curator.Prompter with safe code execution."""

import re
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class MathProblem(BaseModel):
    """A math problem with its solution."""

    question: str = Field(description="The math problem to solve")
    python_code: str = Field(description="Python code that solves the problem")
    expected_answer: Optional[Union[int, float, str]] = Field(
        None, description="Expected answer if known"
    )
    explanation: Optional[str] = Field(None, description="Explanation of the solution approach")

    @field_validator("python_code")
    @classmethod
    def validate_python_code(cls, code: str) -> str:
        """Validate Python code for safety."""
        # Disallow imports
        if re.search(r"^\s*import\s+", code, re.MULTILINE):
            raise ValueError("Import statements are not allowed")

        # Disallow exec/eval
        if re.search(r"\b(exec|eval)\s*\(", code):
            raise ValueError("exec() and eval() are not allowed")

        # Disallow file operations
        if re.search(r"\b(open|file|read|write)\s*\(", code):
            raise ValueError("File operations are not allowed")

        # Disallow system calls
        if re.search(r"\b(os|subprocess|system)\.", code):
            raise ValueError("System calls are not allowed")

        return code


class MathSolution(BaseModel):
    """The solution to a math problem."""

    python_code: str = Field(description="Generated Python code that solves the problem")
    answer: Union[int, float, str] = Field(description="The computed answer")
    explanation: str = Field(description="Step-by-step explanation of the solution")

    @field_validator("python_code")
    @classmethod
    def validate_python_code(cls, code: str) -> str:
        """Validate Python code for safety."""
        return MathProblem.validate_python_code(code)


class MathSolutions(BaseModel):
    """A list of math solutions."""

    solutions: List[MathSolution] = Field(description="List of solutions to math problems")


class MathResult(BaseModel):
    """The result of solving a math problem."""

    question: str = Field(description="Original math problem")
    solution: MathSolution = Field(description="Solution details")
    is_correct: Optional[bool] = Field(
        None, description="Whether the solution is correct (if expected_answer is provided)"
    )
    error: Optional[str] = Field(None, description="Error message if code execution failed")

    def __str__(self) -> str:
        """Format the result for display."""
        status = "✓" if self.is_correct else "✗" if self.is_correct is False else "?"
        error_msg = f"\nError: {self.error}" if self.error else ""
        return (
            f"Question: {self.question}\n"
            f"Answer: {self.solution.answer} [{status}]\n"
            f"Explanation: {self.solution.explanation}{error_msg}"
        )
