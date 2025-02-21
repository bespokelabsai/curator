import pytest

from bespokelabs.curator.code_executor.code_formatter import CodeFormatter
from bespokelabs.curator.code_executor.types import CodeExecutionRequest, CodeExecutionRequestParams


@pytest.fixture
def sample_row():
    return {"code": "print('hello')", "input": "test input", "output": "test output"}


@pytest.fixture
def code_formatter():
    def code_fn(row):
        return row["code"]

    def input_fn(row):
        return row["input"]

    def output_fn(row):
        return row["output"]

    params = CodeExecutionRequestParams(
        timeout=10,
        memory_limit=1024 * 1024 * 1024,  # 1GB in bytes
    )

    return CodeFormatter(code=code_fn, code_input=input_fn, code_output=output_fn, execution_params=params)


def test_code_formatter_initialization(code_formatter):
    """Test that CodeFormatter initializes correctly."""
    assert callable(code_formatter.code)
    assert callable(code_formatter.code_input)
    assert callable(code_formatter.code_output)
    assert isinstance(code_formatter.execution_params, CodeExecutionRequestParams)
    assert code_formatter.execution_params.timeout == 10
    assert code_formatter.execution_params.memory_limit == 1024 * 1024 * 1024


def test_create_code_execution_request(code_formatter, sample_row):
    """Test creating a code execution request."""
    execution_directory = "/tmp/test"
    request = code_formatter.create_code_execution_request(row=sample_row, idx=1, execution_directory=execution_directory)

    assert isinstance(request, CodeExecutionRequest)
    assert request.code == "print('hello')"
    assert request.code_input == "test input"
    assert request.original_row == sample_row
    assert request.original_row_idx == 1
    assert request.execution_directory == execution_directory
    assert request.execution_params == code_formatter.execution_params


def test_code_formatter_with_custom_functions():
    """Test CodeFormatter with custom formatting functions."""

    def custom_code_fn(row):
        return f"def main():\n    {row['code']}"

    def custom_input_fn(row):
        return f"Input: {row['input']}"

    def custom_output_fn(row):
        return f"Expected: {row['output']}"

    params = CodeExecutionRequestParams(
        timeout=5,
        memory_limit=512 * 1024 * 1024,  # 512MB
    )

    formatter = CodeFormatter(code=custom_code_fn, code_input=custom_input_fn, code_output=custom_output_fn, execution_params=params)

    sample_row = {"code": "return x + y", "input": "x=1, y=2", "output": "3"}

    request = formatter.create_code_execution_request(row=sample_row, idx=0, execution_directory="/tmp/test")

    assert request.code == "def main():\n    return x + y"
    assert request.code_input == "Input: x=1, y=2"
    assert formatter.code_output(sample_row) == "Expected: 3"
    assert request.execution_params.timeout == 5
    assert request.execution_params.memory_limit == 512 * 1024 * 1024


def test_code_formatter_with_missing_data():
    """Test CodeFormatter handling of missing data."""

    def safe_code_fn(row):
        return row.get("code", "")

    def safe_input_fn(row):
        return row.get("input", "")

    def safe_output_fn(row):
        return row.get("output", "")

    params = CodeExecutionRequestParams()  # Use default values

    formatter = CodeFormatter(code=safe_code_fn, code_input=safe_input_fn, code_output=safe_output_fn, execution_params=params)

    empty_row = {}
    request = formatter.create_code_execution_request(row=empty_row, idx=0, execution_directory="/tmp/test")

    assert request.code == ""
    assert request.code_input == ""
    assert formatter.code_output(empty_row) == ""
    assert request.execution_params.timeout == 10  # Default value
    assert request.execution_params.memory_limit == 1024 * 1024 * 1024  # Default value


def test_code_formatter_with_default_params():
    """Test CodeFormatter with default execution parameters."""
    formatter = CodeFormatter(
        code=lambda x: x.get("code", ""),
        code_input=lambda x: x.get("input", ""),
        code_output=lambda x: x.get("output", ""),
        execution_params=CodeExecutionRequestParams(),
    )

    request = formatter.create_code_execution_request(row={"code": "test", "input": "test", "output": "test"}, idx=0, execution_directory="/tmp/test")

    assert request.execution_params.timeout == 10  # Default from types.py
    assert request.execution_params.memory_limit == 1024 * 1024 * 1024  # Default from types.py
