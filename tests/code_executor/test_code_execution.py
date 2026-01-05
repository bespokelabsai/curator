# test with multiprocessing backend
import pytest

from bespokelabs import curator
from tests.conftest import importorskip


@pytest.mark.asyncio
async def test_simple_code_execution_multiprocessing():
    """Test simple code execution with basic input/output."""

    # Initialize backend
    class TestCodeExecutor(curator.CodeExecutor):
        def code(self, row):
            return """
input_value = input()
print(f"You entered: {input_value}")
"""

        def code_input(self, row):
            return row["input"]

        def code_output(self, row, exec_output):
            row["output"] = exec_output.stdout
            return row

    executor = TestCodeExecutor(backend="multiprocessing")
    sample_data = [{"input": "Hello World multiprocessing"}]
    result = executor(sample_data)
    assert result[0]["output"] == "You entered: Hello World multiprocessing\n"


@pytest.mark.asyncio
async def test_simple_code_execution_ray():
    """Test simple code execution with basic input/output."""
    importorskip("ray")

    # Initialize backend
    class TestCodeExecutor(curator.CodeExecutor):
        def code(self, row):
            return """
input_value = input()
print(f"You entered: {input_value}")
"""

        def code_input(self, row):
            return row["input"]

        def code_output(self, row, exec_output):
            row["output"] = exec_output.stdout
            return row

    executor = TestCodeExecutor(backend="ray")
    sample_data = [{"input": "Hello World ray"}]
    result = executor(sample_data)
    assert result[0]["output"] == "You entered: Hello World ray\n"
