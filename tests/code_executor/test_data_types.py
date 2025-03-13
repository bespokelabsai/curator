import datetime

import pytest

from bespokelabs.curator.code_executor.types import (
    CodeAPIRequest,
    CodeExecutionBackendConfig,
    CodeExecutionOutput,
    CodeExecutionRequest,
    CodeExecutionRequestParams,
    CodeExecutionResponse,
    CodeExecutionResult,
)


def test_code_execution_result():
    """Test CodeExecutionResult model."""
    result = CodeExecutionResult(stdout="test output", stderr="test error", exit_code=0)

    assert result.stdout == "test output"
    assert result.stderr == "test error"
    assert result.exit_code == 0


def test_code_execution_request_params():
    """Test CodeExecutionRequestParams model."""
    params = CodeExecutionRequestParams()

    # Test default values
    assert params.timeout == 10
    assert params.memory_limit == 1024 * 1024 * 1024

    # Test custom values
    custom_params = CodeExecutionRequestParams(timeout=20, memory_limit=512 * 1024 * 1024)
    assert custom_params.timeout == 20
    assert custom_params.memory_limit == 512 * 1024 * 1024


def test_code_execution_request():
    """Test CodeExecutionRequest model."""
    request = CodeExecutionRequest(
        code="print('hello')",
        code_input="test input",
        execution_params=CodeExecutionRequestParams(),
        original_row={"test": "data"},
        original_row_idx=1,
        execution_directory="/tmp/test",
    )

    assert request.code == "print('hello')"
    assert request.code_input == "test input"
    assert isinstance(request.execution_params, CodeExecutionRequestParams)
    assert request.original_row == {"test": "data"}
    assert request.original_row_idx == 1
    assert request.execution_directory == "/tmp/test"


def test_code_api_request():
    """Test CodeAPIRequest model."""
    execution_request = CodeExecutionRequest(code="test code", code_input="test input")

    api_request = CodeAPIRequest(task_id=1, execution_request=execution_request, attempts_left=3, code_formatter=lambda x: x)

    assert api_request.task_id == 1
    assert api_request.execution_request == execution_request
    assert api_request.attempts_left == 3
    assert callable(api_request.code_formatter)
    assert isinstance(api_request.created_at, datetime.datetime)
    assert api_request.result == []


def test_code_execution_output():
    """Test CodeExecutionOutput model."""
    output = CodeExecutionOutput(message="test message", error="test error", stdout="test stdout", stderr="test stderr", files=b"test files")

    assert output.message == "test message"
    assert output.error == "test error"
    assert output.stdout == "test stdout"
    assert output.stderr == "test stderr"
    assert output.files == "test files"


def test_code_execution_response():
    """Test CodeExecutionResponse model."""
    exec_output = CodeExecutionOutput(message="test")
    api_request = CodeAPIRequest(execution_request=CodeExecutionRequest(code="test", code_input="test"), attempts_left=3, code_formatter=lambda x: x)

    response = CodeExecutionResponse(exec_output=exec_output, code_api_request=api_request)

    assert response.exec_output == exec_output
    assert response.code_api_request == api_request
    assert isinstance(response.created_at, datetime.datetime)
    assert isinstance(response.finished_at, datetime.datetime)


def test_code_execution_backend_config():
    """Test CodeExecutionBackendConfig model."""
    config = CodeExecutionBackendConfig()

    # Test default values
    assert config.max_requests_per_minute == 10000
    assert config.max_retries == 3
    assert config.seconds_to_pause_on_rate_limit == 10
    assert config.base_url is None

    # Test custom values
    custom_config = CodeExecutionBackendConfig(max_requests_per_minute=5000, max_retries=5, seconds_to_pause_on_rate_limit=20, base_url="http://test.com")

    assert custom_config.max_requests_per_minute == 5000
    assert custom_config.max_retries == 5
    assert custom_config.seconds_to_pause_on_rate_limit == 20
    assert custom_config.base_url == "http://test.com"


def test_invalid_code_execution_result():
    """Test invalid CodeExecutionResult validation."""
    with pytest.raises(ValueError):
        CodeExecutionResult(
            stdout=123,  # Should be string
            stderr="test",
            exit_code="0",  # Should be int
        )


def test_invalid_code_execution_request_params():
    """Test invalid CodeExecutionRequestParams validation."""
    with pytest.raises(ValueError):
        CodeExecutionRequestParams(
            timeout="invalid",  # Should be int
            memory_limit="invalid",  # Should be int
        )
