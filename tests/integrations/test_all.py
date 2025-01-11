import hashlib
import logging
import signal
import time
from io import StringIO

import numpy as np
import pytest
from rich.console import Console

from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from tests.integrations import helper

##############################
# Online                     #
##############################


def _hash_string(input_string):
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()


_ONLINE_BACKENDS = [{"integration": backend} for backend in {"openai", "litellm"}]
_BATCH_BACKENDS = [{"integration": backend} for backend in {"openai"}]
_FAILED_BATCH_BACKENDS = [{"integration": backend, "cached_working_dir": True} for backend in {"openai"}]


class TimeoutError(Exception):
    pass


class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)

    @staticmethod
    def _handle_timeout(signum, frame):
        raise TimeoutError("Function execution exceeded time limit!")


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic(temp_working_dir, mock_dataset):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "54456b7dcce5826036a52f242e589b02c945ef0891af6e8b786020ba2737fc09",
        "litellm": "7bf42717b5e516eca1f92ca69680c1278c8a9a0e351365d1aa808f92e1b59086",
    }

    with vcr_config.use_cassette("basic_completion.yaml"):
        # Capture the output to verify status tracker
        output = StringIO()
        console = Console(file=output, width=300)

        dataset = helper.create_basic(
            temp_working_dir,
            mock_dataset,
            backend=backend,
            llm_params={
                "generation_params": {"seed": 42},
            },
            tracker_console=console,
        )

        # Verify status tracker output
        captured = output.getvalue()
        assert "Generating data using gpt-3.5-turbo" in captured, captured
        assert "3" in captured, captured  # Verify total requests processed
        assert "Final Curator Statistics" in captured, captured
        # Verify response content
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]


@pytest.mark.skip
@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
def test_camel(temp_working_dir, camel_gt_dataset):
    temp_working_dir, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("camel_completion.yaml"):
        qa_dataset = helper.create_camel(temp_working_dir)
        assert np.array_equal(qa_dataset.to_pandas().values, camel_gt_dataset.to_pandas().values)


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
def test_basic_cache(caplog, temp_working_dir, mock_dataset):
    temp_working_dir, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("basic_completion.yaml"):
        st = time.time()
        distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset)
        tt = time.time() - st

        # This should use cache
        from bespokelabs.curator.request_processor.base_request_processor import CACHE_MSG

        logger = "bespokelabs.curator.request_processor.base_request_processor"
        with caplog.at_level(logging.INFO, logger=logger):
            st = time.time()
            helper.create_basic(temp_working_dir, mock_dataset)
            cached_tt = time.time() - st
            distilled_dataset.cleanup_cache_files()
            assert f"Using cached output dataset. {CACHE_MSG}" in caplog.text
            assert cached_tt < tt - 0.2


@pytest.mark.skip
@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
def test_low_rpm_setting(temp_working_dir, mock_dataset):
    temp_working_dir, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("basic_completion.yaml"):
        helper.create_basic(temp_working_dir, mock_dataset, llm_params={"max_requests_per_minute": 5})


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_auto_rpm(temp_working_dir):
    _, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("basic_completion.yaml"):
        llm = helper.create_llm()
        assert llm._request_processor.header_based_max_requests_per_minute == 10_000
        assert llm._request_processor.header_based_max_tokens_per_minute == 200_000


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_resume(caplog, temp_working_dir, mock_dataset):
    temp_working_dir, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("basic_resume.yaml"):
        with pytest.raises(TimeoutError):
            with Timeout(3):
                helper.create_basic(temp_working_dir, mock_dataset, llm_params={"max_requests_per_minute": 1})

        logger = "bespokelabs.curator.request_processor.online.base_online_request_processor"
        with caplog.at_level(logging.INFO, logger=logger):
            helper.create_basic(temp_working_dir, mock_dataset)
            resume_msg = "Already Completed: 1"
            assert resume_msg in caplog.text


##############################
# Batch                      #
##############################


@pytest.mark.parametrize("temp_working_dir", (_BATCH_BACKENDS), indirect=True)
def test_batch_resume(temp_working_dir, mock_dataset):
    temp_working_dir, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("basic_batch_resume.yaml"):
        from unittest.mock import patch

        with patch("bespokelabs.curator.request_processor.event_loop.run_in_event_loop") as mocked_run_loop:

            def _run_loop(func):
                if "poll_and_process_batches" in str(func):
                    return
                return run_in_event_loop(func)

            mocked_run_loop.side_effect = _run_loop
            with pytest.raises(ValueError):
                helper.create_basic(temp_working_dir, mock_dataset, batch=True)
        mocked_run_loop.side_effect = run_in_event_loop
        from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker

        tracker_batch_file_path = temp_working_dir + "/testing_hash_123/batch_objects.jsonl"
        with open(tracker_batch_file_path, "r") as f:
            tracker = BatchStatusTracker.model_validate_json(f.read())
        assert tracker.n_total_requests == 3
        assert len(tracker.submitted_batches) == 1
        assert len(tracker.downloaded_batches) == 0

        helper.create_basic(temp_working_dir, mock_dataset, batch=True)
        with open(tracker_batch_file_path, "r") as f:
            tracker = BatchStatusTracker.model_validate_json(f.read())
        assert len(tracker.submitted_batches) == 0
        assert len(tracker.downloaded_batches) == 1


@pytest.mark.parametrize("temp_working_dir", (_FAILED_BATCH_BACKENDS), indirect=True)
def test_failed_request_in_batch_resume(caplog, temp_working_dir, mock_dataset):
    temp_working_dir, backend, vcr_config = temp_working_dir
    with vcr_config.use_cassette("failed_request_batch_resume.yaml"):
        tracker_batch_file_path = temp_working_dir + "/testing_hash_123/batch_objects.jsonl"
        from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker

        with open(tracker_batch_file_path, "r") as f:
            failed_tracker = BatchStatusTracker.model_validate_json(f.read())
        assert failed_tracker.n_total_requests == 3
        assert failed_tracker.n_downloaded_failed_requests == 1
        assert len(failed_tracker.submitted_batches) == 0
        assert len(failed_tracker.downloaded_batches) == 1
        RESUBMIT_MSG = f"Request file tests/integrations/{backend}/fixtures/.test_cache/testing_hash_123/requests_0.jsonl is being re-submitted."

        logger = "bespokelabs.curator.status_tracker.batch_status_tracker"

        with caplog.at_level(logging.INFO, logger=logger):
            helper.create_basic(temp_working_dir, mock_dataset, batch=True)
            assert RESUBMIT_MSG in caplog.text

        with open(tracker_batch_file_path, "r") as f:
            tracker = BatchStatusTracker.model_validate_json(f.read())
        assert len(tracker.submitted_batches) == 0
        resubmitted_sucess_batch = [key for key in tracker.downloaded_batches.keys() if key not in failed_tracker.downloaded_batches.keys()][0]
        assert tracker.downloaded_batches[resubmitted_sucess_batch].request_counts.total == 1
        assert tracker.downloaded_batches[resubmitted_sucess_batch].request_counts.succeeded == 1


@pytest.mark.parametrize("temp_working_dir", (_BATCH_BACKENDS), indirect=True)
def test_basic_batch(temp_working_dir, mock_dataset):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {"openai": "47127d9dcb428c18e5103dffcb0406ba2f9acab2f1ea974606962caf747b0ad5"}
    with vcr_config.use_cassette("basic_batch_completion.yaml"):
        dataset = helper.create_basic(temp_working_dir, mock_dataset, batch=True)
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]
