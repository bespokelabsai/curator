import datetime
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from bespokelabs.curator.request_processor.batch.gemini_batch_request_processor import GeminiBatchRequestProcessor  # noqa
from tests.integrations.helper import BasicLLM


def _hash_string(input_string):
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()


def _mock_job_side_effect(*args, **kwargs):
    mock_instance = MagicMock()
    mock_instance.name = "mocked_job_id"
    mock_instance.uri = args[0]
    mock_instance.submit = MagicMock()
    mocked_job = MagicMock()
    mocked_job.name = "mocked_job_id"
    mock_instance.submit.return_value = mocked_job
    mock_instance.state.name = "JOB_STATE_SUCCEEDED"

    mock_instance.created_time = "now"
    mock_instance.updated_time = "now"

    mock_instance.completion_stats = MagicMock()
    mock_instance.completion_stats.sucessful_count = 3
    to_dict = MagicMock()
    to_dict.return_value = {}
    mock_instance.to_dict = to_dict

    mock_instance.iter_outputs = MagicMock()

    class _MockedResponse:
        def download_as_string():
            file_path = "tests/integrations/common_fixtures/gemini_batch_response.jsonl"

            with open(file_path, "rb") as file:
                return file.read()

    mock_instance.iter_outputs.return_value = [_MockedResponse]
    return mock_instance


def _create_mock_batch_job():
    mock_batch_job_class = MagicMock()
    mock_batch_job_class.side_effect = _mock_job_side_effect
    return mock_batch_job_class


class _BlobObject:
    def __init__(self):
        self._traces = {}

    def upload_from_string(self, str_json, content_type=None):
        self._traces["upload_from_string"] = str_json


class _MockedGoogleBucket:
    def __init__(self) -> None:
        self._mocked_blob = _BlobObject()
        self._traces = {}

    def blob(self, path):
        self._traces["blob.input"] = path
        return self._mocked_blob


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "gemini"}]), indirect=True)
def test_basic_batch_gemini(temp_working_dir, mock_dataset):
    temp_working_dir, backend, _ = temp_working_dir

    with patch("vertexai.batch_prediction.BatchPredictionJob.submit") as mocked_batch_job:
        with patch("google.cloud.aiplatform.BatchPredictionJob", new=_create_mock_batch_job()):
            with patch("bespokelabs.curator.request_processor.batch.gemini_batch_request_processor.GeminiBatchRequestProcessor._initialize_cloud") as init:
                init.return_value = None
                job = MagicMock()
                job.name = "mocked_job"
                mocked_batch_job.return_value = job
                prompter = BasicLLM(
                    model_name="gemini-1.5-flash-002",
                    backend="gemini",
                    batch=True,
                )
                prompter._request_processor._location = "mocked_location"
                prompter._request_processor._project_id = "mocked_id"
                prompter._request_processor._bucket_name = "mocked_bucket"
                prompter._request_processor._bucket = _MockedGoogleBucket()

                dataset = prompter(mock_dataset, working_dir=temp_working_dir)
                recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
                assert _hash_string(recipes) == "2131be1c57623eb8e27bc4437476990bfd4b0c954a274ed3af6d46dc26138d1e"
                assert len(dataset) == 3


count = 0


class MockedParse:
    def _parse(self, *args, **kwargs):
        global count
        from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus

        request_count = GenericBatchRequestCounts(
            failed=0,
            succeeded=3,
            total=3,
            raw_request_counts_object={},
        )
        count += 1
        if count > 4:
            raw_status = "JOB_STATE_SUCCEEDED"
            status = GenericBatchStatus.FINISHED
        else:
            status = GenericBatchStatus.SUBMITTED
            raw_status = "JOB_STATE_PENDING"

        return GenericBatch(
            request_file=kwargs["request_file"],
            id="mock",
            created_at=datetime.datetime.now(),
            finished_at=datetime.datetime.now(),
            status=status,
            api_key_suffix="gs",
            request_counts=request_count,
            raw_batch={},
            raw_status=raw_status,
        )


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "gemini"}]), indirect=True)
def test_polled_batch_gemini(temp_working_dir, mock_dataset):
    temp_working_dir, backend, _ = temp_working_dir

    with patch("vertexai.batch_prediction.BatchPredictionJob.submit") as mocked_batch_job:
        with patch("google.cloud.aiplatform.BatchPredictionJob", new=_create_mock_batch_job()):
            with patch(
                "bespokelabs.curator.request_processor.batch.gemini_batch_request_processor.GeminiBatchRequestProcessor.parse_api_specific_batch_object"
            ) as parse:
                with patch("bespokelabs.curator.request_processor.batch.gemini_batch_request_processor.GeminiBatchRequestProcessor._initialize_cloud") as init:
                    init.return_value = None
                    parse.side_effect = MockedParse._parse
                    job = MagicMock()
                    job.name = "mocked_job"
                    mocked_batch_job.return_value = job
                    prompter = BasicLLM(model_name="gemini-1.5-flash-002", backend="gemini", batch=True, backend_params={"batch_check_interval": 1})
                    prompter._request_processor._location = "mocked_location"
                    prompter._request_processor._project_id = "mocked_id"
                    prompter._request_processor._bucket_name = "mocked_bucket"
                    prompter._request_processor._bucket = _MockedGoogleBucket()

                    dataset = prompter(
                        mock_dataset,
                        working_dir=temp_working_dir,
                    )
                    recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
                    assert _hash_string(recipes) == "2131be1c57623eb8e27bc4437476990bfd4b0c954a274ed3af6d46dc26138d1e"
                    assert len(dataset) == 3
