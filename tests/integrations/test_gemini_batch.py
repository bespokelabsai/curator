from unittest.mock import MagicMock, patch

import pytest

from bespokelabs.curator.request_processor.batch.gemini_batch_request_processor import GeminiBatchRequestProcessor  # noqa
from tests.integrations.helper import BasicLLM


def _create_mock_batch_job():
    mock_batch_job_class = MagicMock()

    def side_effect(*args, **kwargs):
        mock_instance = MagicMock()
        mock_instance.uri = args[0]
        mock_instance.submit = MagicMock()
        mocked_job = MagicMock()
        mocked_job.name = "mocked_job_id"
        mock_instance.submit.return_value = mocked_job
        mock_instance.state.name = "JOB_STATE_SUCCEEDED"
        mock_instance.name = "mocked_job_id"
        mock_instance.created_time = "now"
        mock_instance.updated_time = "now"

        mock_instance.completion_stats = MagicMock()
        mock_instance.completion_stats.sucessful_count = 3
        mock_instance.to_dict.return_value = {}

        mock_instance.iter_outputs = MagicMock()

        class _MockedResponse:
            def download_as_string():
                file_path = "tests/integrations/common_fixtures/gemini_batch_response.jsonl"

                with open(file_path, "rb") as file:
                    return file.read()

        mock_instance.iter_outputs.return_value = [_MockedResponse]
        return mock_instance

    mock_batch_job_class.side_effect = side_effect
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
    temp_working_dir, backend, vcr_config = temp_working_dir

    with patch("google.cloud.aiplatform.BatchPredictionJob", new=_create_mock_batch_job()):
        with patch("bespokelabs.curator.request_processor.batch.gemini_batch_request_processor.GeminiBatchRequestProcessor.__init__") as init:
            init.return_value = None
            prompter = BasicLLM(
                model_name="gemini-1.5-flash-002",
                backend="gemini",
                batch=True,
            )
            prompter._request_processor._location = "mocked_location"
            prompter._request_processor._project_id = "mocked_id"
            prompter._request_processor._bucket_name = "mocked_bucket"
            prompter._request_processor._bucket = _MockedGoogleBucket()

            dataset = prompter(mock_dataset)
            assert len(dataset) == 3
