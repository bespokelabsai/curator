import ast
import asyncio
import io
import json
import os
import tarfile
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequest, CodeExecutionResponse


class TestBackend(BaseCodeExecutionBackend):
    """Test implementation of abstract base class"""

    def __init__(self, config):
        super().__init__(config)
        self.working_dir = None  # Add working_dir attribute
        self.total_requests = 0  # Add missing total_requests attribute

    @property
    def backend(self) -> str:
        return "test"

    async def execute_request(self, request):
        return CodeExecutionOutput(message="test")

    def requests_to_responses(self, execution_request_files):
        pass


class Config:
    """Config class for testing"""

    def __init__(self):
        self.max_retries = 3
        self.max_requests_per_minute = 10
        self.seconds_to_pause_on_rate_limit = 10
        self.require_all_responses = True


@pytest.fixture
def backend():
    return TestBackend(Config())


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def status_tracker():
    tracker = Mock()
    tracker.num_tasks_in_progress = 1
    tracker.num_tasks_succeeded = 0
    tracker.num_other_errors = 0
    tracker.time_of_last_rate_limit_error = time.time() - 5
    tracker.update_stats = Mock()
    return tracker


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.__len__ = Mock(return_value=2)
    dataset.__iter__ = Mock(return_value=iter([{"data": "row1"}, {"data": "row2"}]))
    return dataset


def test_backend_property(backend):
    assert backend.backend == "test"


@pytest.mark.asyncio
async def test_create_temp_file(temp_dir):
    content = "print('hello')"
    file_path = BaseCodeExecutionBackend._create_temp_file(content, temp_dir)

    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert f.read() == content


@pytest.mark.asyncio
async def test_get_created_files(temp_dir):
    # Create some test files
    test_files = {"test1.txt": "content1"}

    for path, content in test_files.items():
        full_path = os.path.join(temp_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    # Add program.py which should be included
    with open(os.path.join(temp_dir, "program.py"), "w") as f:
        f.write("print('test')")

    # Get tar bytes
    tar_bytes = BaseCodeExecutionBackend._get_created_files(temp_dir)

    # Read back the tar contents
    tar_buffer = io.BytesIO(ast.literal_eval(tar_bytes))
    with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
        files = tar.getnames()
        # Convert paths to use forward slashes for consistency
        files = [f.replace("\\", "/") for f in files]

        # Verify program.py is included (since base implementation includes all files)
        assert "program.py" in files

        # Verify other files are included with correct content
        for path, content in test_files.items():
            assert path in files
            # Extract and read the file content
            member = tar.extractfile(path)
            assert member is not None
            assert member.read().decode() == content


@pytest.mark.asyncio
async def test_append_execution_response(temp_dir):
    response_file = os.path.join(temp_dir, "responses.jsonl")

    data = {"test": "data"}
    # Use classmethod call syntax with cls parameter
    await BaseCodeExecutionBackend.append_execution_response(BaseCodeExecutionBackend, data, response_file)

    with open(response_file) as f:
        content = f.read().strip()
        assert content == '{"test": "data"}'


@pytest.mark.asyncio
async def test_validate_existing_response_file(backend, temp_dir):
    response_file = os.path.join(temp_dir, "responses.jsonl")

    # Create test responses
    responses = [
        CodeExecutionResponse(
            code_api_request=CodeAPIRequest(
                task_id=1,
                execution_request=CodeExecutionRequest(
                    original_row_idx=1,
                    code="test",
                    code_input="test input",
                    original_row={"input": "test"},
                    execution_directory=temp_dir,
                    language="python",
                    timeout=30,
                ),
                attempts_left=3,
                code_formatter=None,
            ),
            exec_output=CodeExecutionOutput(message="success"),
        ),
    ]

    # Write test responses
    with open(response_file, "w") as f:
        for response in responses:
            f.write(response.model_dump_json() + "\n")

    completed = backend.validate_existing_response_file(response_file)
    assert completed == {1}


@pytest.mark.asyncio
async def test_process_requests_from_file(backend, temp_dir):
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    response_file = os.path.join(temp_dir, "responses_0.jsonl")

    requests = [
        CodeExecutionRequest(
            original_row_idx=1,
            code="test1",
            code_input="test input",
            original_row={},
            execution_directory=temp_dir,
        )
    ]

    backend.code_formatter = Mock()
    backend.code_formatter.create_code_execution_request = Mock(return_value=requests[0])

    with open(request_file, "w") as f:
        for req in requests:
            f.write(req.model_dump_json() + "\n")

    await backend.process_requests_from_file(request_file, response_file)

    with open(response_file) as f:
        responses = [CodeExecutionResponse.model_validate_json(line) for line in f]
        assert len(responses) == 1


@pytest.mark.asyncio
async def test_handle_single_request_with_retries(backend, temp_dir, status_tracker):
    response_file = os.path.join(temp_dir, "responses.jsonl")
    retry_queue = asyncio.Queue()

    request = CodeAPIRequest(
        task_id=1,
        execution_request=CodeExecutionRequest(
            original_row_idx=1,
            code="test",
            code_input="test input",
            original_row={},
            execution_directory=temp_dir,
        ),
        attempts_left=3,
        code_formatter=None,
    )

    await backend.handle_single_request_with_retries(request=request, retry_queue=retry_queue, response_file=response_file, status_tracker=status_tracker)

    with open(response_file) as f:
        response = CodeExecutionResponse.model_validate_json(f.read())
        assert response.exec_output.message == "test"


def test_read_metadata_file(backend, temp_dir):
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    metadata_file = os.path.join(temp_dir, "metadata_0.json")

    # Create test metadata
    metadata = {"num_jobs": 10}
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    result = backend.read_metadata_file(request_file)
    assert result == metadata


def test_read_metadata_file_missing(backend, temp_dir):
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    with pytest.raises(ValueError, match="Metadata file not found"):
        backend.read_metadata_file(request_file)


def test_read_metadata_file_invalid(backend, temp_dir):
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    metadata_file = os.path.join(temp_dir, "metadata_0.json")

    # Create invalid metadata
    with open(metadata_file, "w") as f:
        f.write("invalid json")

    with pytest.raises(ValueError, match="Invalid JSON in metadata file"):
        backend.read_metadata_file(request_file)


@pytest.mark.asyncio
async def test_cool_down_if_rate_limit_error(backend, status_tracker):
    with patch("asyncio.sleep") as mock_sleep:
        await backend.cool_down_if_rate_limit_error(status_tracker)
        mock_sleep.assert_called_once()
        assert abs(mock_sleep.call_args[0][0] - 5) < 0.1


def test_verify_existing_request_files(backend, temp_dir):
    backend.working_dir = temp_dir

    # Create valid request and metadata files
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    metadata_file = os.path.join(temp_dir, "metadata_0.json")

    with open(request_file, "w") as f:
        f.write("test\n")

    with open(metadata_file, "w") as f:
        json.dump({"num_jobs": 1}, f)

    incomplete = backend._verify_existing_request_files(None)
    assert len(incomplete) == 0


def test_verify_existing_request_files_missing_metadata(backend, temp_dir):
    backend.working_dir = temp_dir

    # Create only request file
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    with open(request_file, "w") as f:
        f.write("test\n")

    incomplete = backend._verify_existing_request_files(None)
    assert incomplete == [0]


def test_verify_existing_request_files_mismatched_count(backend, temp_dir):
    backend.working_dir = temp_dir

    # Create files with mismatched counts
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    metadata_file = os.path.join(temp_dir, "metadata_0.json")

    with open(request_file, "w") as f:
        f.write("test\n")

    with open(metadata_file, "w") as f:
        json.dump({"num_jobs": 2}, f)

    incomplete = backend._verify_existing_request_files(None)
    assert incomplete == [0]


@pytest.mark.asyncio
async def test_create_request_files_cached(backend, temp_dir):
    backend.working_dir = temp_dir

    # Create cached request files
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    metadata_file = os.path.join(temp_dir, "metadata_0.json")

    with open(request_file, "w") as f:
        f.write('{"original_row_idx": 1, "code": "test"}\n')

    with open(metadata_file, "w") as f:
        json.dump({"num_jobs": 1}, f)

    request_files = backend.create_request_files(None)

    assert len(request_files) == 1
    assert request_files[0] == request_file
