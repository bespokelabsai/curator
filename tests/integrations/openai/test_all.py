import logging
import os
import signal
import time

import numpy as np
import pytest
import vcr

from tests.integrations import helper

mode = os.environ.get("VCR_MODE", None)

vcr_config = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/integrations/openai/fixtures",
    record_mode=mode,
)

##############################
# Online                     #
##############################


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


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_basic(temp_working_dir, mock_dataset, basic_gt_dataset):
    distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset)
    assert np.array_equal(distilled_dataset.to_pandas().values, basic_gt_dataset.to_pandas().values)


@pytest.mark.skip
@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("camel_completion.yaml")
def test_camel(temp_working_dir, camel_gt_dataset):
    qa_dataset = helper.create_camel(temp_working_dir)
    assert np.array_equal(qa_dataset.to_pandas().values, camel_gt_dataset.to_pandas().values)


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_basic_cache(caplog, temp_working_dir, mock_dataset):
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
        assert cached_tt < tt // 2


@pytest.mark.skip
@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_low_rpm_setting(temp_working_dir, mock_dataset):
    helper.create_basic(temp_working_dir, mock_dataset, llm_params={"max_requests_per_minute": 5})


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_auto_rpm(temp_working_dir):
    llm = helper.create_llm()
    assert llm._request_processor.header_based_max_requests_per_minute == 10_000
    assert llm._request_processor.header_based_max_tokens_per_minute == 200_000


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_resume.yaml")
def test_resume(caplog, temp_working_dir, mock_dataset):
    with pytest.raises(TimeoutError):
        with Timeout(3):
            helper.create_basic(temp_working_dir, mock_dataset, llm_params={"max_requests_per_minute": 1})

    logger = "bespokelabs.curator.request_processor.online.base_online_request_processor"
    with caplog.at_level(logging.INFO, logger=logger):
        helper.create_basic(temp_working_dir, mock_dataset)
        resume_msg = "Resuming progress by reading existing file: tests/integrations/"
        assert resume_msg in caplog.text


##############################
# Batch                      #
##############################


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_batch_completion.yaml")
def test_basic_batch(temp_working_dir, mock_dataset, batch_gt_dataset):
    distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset, batch=True)
    assert np.array_equal(distilled_dataset.to_pandas().values, batch_gt_dataset.to_pandas().values)
