"""Tests for the math problem solver implementation."""

import pytest
from math_solver import MathProblem, MathSolution, MathResult, MathSolutions
from math_prompter import MathPrompter
from math_executor import execute_math_code, is_safe_ast


def test_execute_math_code():
    """Test safe code execution."""
    # Test valid code
    code = "x = 5 + 3\nprint(x)"
    result, error = execute_math_code(code)
    assert result == "8"
    assert error is None

    # Test unsafe code - imports
    unsafe_code = "import os\nos.system('ls')"
    result, error = execute_math_code(unsafe_code)
    assert result is None
    assert "unsafe operations" in error

    # Test unsafe code - eval
    unsafe_code = "eval('2 + 2')"
    result, error = execute_math_code(unsafe_code)
    assert result is None
    assert "unsafe operations" in error

    # Test syntax error
    invalid_code = "print(x"
    result, error = execute_math_code(invalid_code)
    assert result is None
    assert "Syntax error" in error

    # Test timeout
    infinite_loop = "while True: pass"
    result, error = execute_math_code(infinite_loop, timeout=1)
    assert result is None
    assert "Runtime error" in error


def test_math_solution_validation():
    """Test MathSolution validation."""
    # Test valid solution
    valid_solution = MathSolution(
        python_code="print(2 + 2)", answer=4, explanation="Adding 2 and 2"
    )
    assert valid_solution is not None

    # Test invalid solution with unsafe code
    with pytest.raises(ValueError) as exc_info:
        MathSolution(
            python_code="import sys; print('unsafe')", answer="unsafe", explanation="Unsafe code"
        )
    assert "Import statements are not allowed" in str(exc_info.value)


def test_math_prompter():
    """Test MathPrompter functionality."""
    prompter = MathPrompter(model_name="gpt-4o-mini")

    # Test string input
    result = prompter.solve("What is 5 + 3?")
    assert isinstance(result, MathResult)
    assert result.question == "What is 5 + 3?"
    assert not result.error

    # Test dict input with expected answer
    problem = {"question": "What is 10 * 5?", "expected_answer": 50}
    result = prompter.solve(problem)
    assert isinstance(result, MathResult)
    assert result.is_correct is not None

    # Test batch processing
    problems = ["What is 2 + 2?", "What is 3 * 4?", "What is 15 / 3?"]
    results = prompter.solve(problems)
    assert len(results) == 3

    # Test error handling
    problem = MathProblem(
        question="What is x?", python_code="print(undefined_variable)", expected_answer=None
    )
    result = prompter.solve(problem)
    assert result.error is not None


if __name__ == "__main__":
    pytest.main([__file__])
