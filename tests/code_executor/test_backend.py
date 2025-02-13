import asyncio
import io
import json
import os
import tempfile
import time
import zipfile
from unittest.mock import Mock, patch

import pytest

from bespokelabs.curator.code_executor.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.code_executor.types import CodeAPIRequest, CodeExecutionOutput, CodeExecutionRequest, CodeExecutionResponse


class TestBackend(BaseCodeExecutionBackend):
    """Test implementation of abstract base class"""

    @property
    def backend(self) -> str:
        return "test"

    async def execute_request(self, request):
        return CodeExecutionOutput(message="test")

    def requests_to_responses(self, execution_request_files):
        pass


@pytest.fixture
def backend():
    return TestBackend({"max_retries": 3, "max_requests_per_minute": 10})


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


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
    test_files = {"test1.txt": "content1", "subdir/test2.txt": "content2"}

    for path, content in test_files.items():
        full_path = os.path.join(temp_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    # Add program.py which should be excluded
    with open(os.path.join(temp_dir, "program.py"), "w") as f:
        f.write("print('test')")

    # Get zip bytes
    zip_bytes = BaseCodeExecutionBackend._get_created_files(temp_dir)

    # Read back the zip contents
    zip_buffer = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(zip_buffer) as zipf:
        files = zipf.namelist()

        # Verify program.py is excluded
        assert "program.py" not in files

        # Verify other files are included with correct content
        for path, content in test_files.items():
            assert path in files
            assert zipf.read(path).decode() == content


@pytest.mark.asyncio
async def test_append_execution_response(temp_dir):
    response_file = os.path.join(temp_dir, "responses.jsonl")

    data = {"test": "data"}
    await BaseCodeExecutionBackend.append_execution_response(data, response_file)

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
                task_id=1, execution_request=CodeExecutionRequest(original_row_idx=1, code="test"), attempts_left=3, code_formatter=None
            ),
            exec_output=CodeExecutionOutput(message="success"),
        ),
        CodeExecutionResponse(
            code_api_request=CodeAPIRequest(
                task_id=2, execution_request=CodeExecutionRequest(original_row_idx=2, code="test"), attempts_left=3, code_formatter=None
            ),
            exec_output=CodeExecutionOutput(error="failed"),
        ),
    ]

    # Write test responses
    with open(response_file, "w") as f:
        for response in responses:
            f.write(response.model_dump_json() + "\n")

    completed = backend.validate_existing_response_file(response_file)

    # Should only include successful response
    assert completed == {1}


@pytest.mark.asyncio
async def test_process_requests_from_file(backend, temp_dir):
    # Create test request file
    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    response_file = os.path.join(temp_dir, "responses_0.jsonl")

    requests = [CodeExecutionRequest(original_row_idx=1, code="test1"), CodeExecutionRequest(original_row_idx=2, code="test2")]

    with open(request_file, "w") as f:
        for req in requests:
            f.write(req.model_dump_json() + "\n")

    # Process requests
    await backend.process_requests_from_file(request_file, response_file)

    # Verify responses
    with open(response_file) as f:
        responses = [CodeExecutionResponse.model_validate_json(line) for line in f]

    assert len(responses) == 2
    assert all(r.exec_output.message == "test" for r in responses)


@pytest.mark.asyncio
async def test_handle_single_request_with_retries(backend, temp_dir):
    response_file = os.path.join(temp_dir, "responses.jsonl")
    retry_queue = asyncio.Queue()
    status_tracker = Mock()

    request = CodeAPIRequest(task_id=1, execution_request=CodeExecutionRequest(original_row_idx=1, code="test"), attempts_left=3, code_formatter=None)

    await backend.handle_single_request_with_retries(request=request, retry_queue=retry_queue, response_file=response_file, status_tracker=status_tracker)

    # Verify response was written
    with open(response_file) as f:
        response = CodeExecutionResponse.model_validate_json(f.read())
        assert response.exec_output.message == "test"

    # Verify status updates
    assert status_tracker.num_tasks_in_progress == -1
    assert status_tracker.num_tasks_succeeded == 1


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
async def test_cool_down_if_rate_limit_error(backend):
    status_tracker = Mock()
    status_tracker.time_of_last_rate_limit_error = time.time() - 5

    backend.config.seconds_to_pause_on_rate_limit = 10

    with patch("asyncio.sleep") as mock_sleep:
        await backend.cool_down_if_rate_limit_error(status_tracker)
        mock_sleep.assert_called_once_with(5)


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
async def test_create_request_files_new(backend, temp_dir, mocker):
    backend.working_dir = temp_dir
    mock_dataset = mocker.Mock()
    mock_dataset.__len__.return_value = 2

    request_files = backend.create_request_files(mock_dataset)

    assert len(request_files) == 1
    assert os.path.exists(request_files[0])
    assert os.path.exists(request_files[0].replace("requests_", "metadata_").replace(".jsonl", ".json"))


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


def test_attempt_loading_cached_dataset(backend, temp_dir, mocker):
    backend.working_dir = temp_dir
    dataset_file = os.path.join(temp_dir, "test_hash.arrow")

    # Mock Dataset.from_file
    mock_dataset = mocker.Mock()
    mock_from_file = mocker.patch("datasets.Dataset.from_file", return_value=mock_dataset)

    # Create dummy arrow file
    with open(dataset_file, "wb") as f:
        f.write(b"dummy arrow data")

    result = backend.attempt_loading_cached_dataset("test_hash")

    assert result == mock_dataset
    mock_from_file.assert_called_once_with(dataset_file)


def test_attempt_loading_cached_dataset_missing(backend, temp_dir):
    backend.working_dir = temp_dir
    result = backend.attempt_loading_cached_dataset("test_hash")
    assert result is None


def test_attempt_loading_cached_dataset_invalid(backend, temp_dir):
    backend.working_dir = temp_dir
    dataset_file = os.path.join(temp_dir, "test_hash.arrow")

    # Create invalid arrow file
    with open(dataset_file, "wb") as f:
        f.write(b"invalid arrow data")

    result = backend.attempt_loading_cached_dataset("test_hash")
    assert result is None
    assert not os.path.exists(dataset_file)


@pytest.mark.asyncio
async def test_acreate_request_file(backend, temp_dir, mocker):
    mock_dataset = mocker.Mock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.__iter__.return_value = [{"data": "row1"}, {"data": "row2"}]

    request_file = os.path.join(temp_dir, "requests_0.jsonl")
    metadata_file = os.path.join(temp_dir, "metadata_0.json")

    await backend.acreate_request_file(dataset=mock_dataset, request_file=request_file, metadata_file=metadata_file)

    assert os.path.exists(request_file)
    assert os.path.exists(metadata_file)

    with open(metadata_file) as f:
        metadata = json.load(f)
        assert metadata["num_jobs"] == 2


def test_create_dataset_files(backend, temp_dir, mocker):
    backend.working_dir = temp_dir
    response_file = os.path.join(temp_dir, "responses_0.jsonl")

    # Create test response
    response = CodeExecutionResponse(
        code_api_request=CodeAPIRequest(
            task_id=1, execution_request=CodeExecutionRequest(original_row_idx=1, code="test"), attempts_left=3, code_formatter=None
        ),
        exec_output=CodeExecutionOutput(message="success"),
    )

    with open(response_file, "w") as f:
        f.write(response.model_dump_json() + "\n")

    # Mock code formatter and dataset loading
    mock_formatter = mocker.Mock()
    mock_formatter.code_output.return_value = {"result": "success"}
    backend.code_formatter = mock_formatter

    mock_dataset = mocker.Mock()
    mocker.patch("datasets.Dataset.from_file", return_value=mock_dataset)

    result = backend.create_dataset_files("test_hash")

    assert result == mock_dataset
