# test with sandbox backend
import pytest

from bespokelabs import curator


@pytest.mark.asyncio
async def test_simple_code_execution_local():
    """Test simple code execution with local sandbox backend."""

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

    executor = TestCodeExecutor(backend="local")
    sample_data = [{"input": "Hello World local"}]
    result = executor(sample_data)
    assert result[0]["output"] == "You entered: Hello World local\n"
