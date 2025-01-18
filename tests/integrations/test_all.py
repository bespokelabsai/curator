import hashlib
import importlib
import logging
import signal
import time
from io import StringIO
from unittest.mock import patch

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
_FAILED_BATCH_BACKENDS = [{"integration": backend, "cached_working_dir": True} for backend in {"anthropic", "openai"}]
_BATCH_BACKENDS = [{"integration": backend} for backend in {"anthropic", "openai"}]


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
@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_camel(temp_working_dir):
    temp_working_dir, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("camel_completion.yaml"):
        qa_dataset = helper.create_camel(temp_working_dir)
        assert ["subject", "subsubject", "question", "answer"] == qa_dataset.column_names


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


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "litellm/anthropic"}]), indirect=True)
def test_seperate_rpm_tpm(caplog, temp_working_dir, mock_dataset):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "litellm": "8c4d5d0d647a04bc724c9000db14619a444c918d9b6423fbe31f7308e6f8f94c",
    }

    with vcr_config.use_cassette("basic_completion_seperate_rpm_tpm.yaml"):
        # Capture the output to verify status tracker
        output = StringIO()
        console = Console(file=output, width=300)

        logger = "bespokelabs.curator.request_processor.online.base_online_request_processor"
        RPM_MSG = "Automatically set max_requests_per_minute to 4000"
        TPM_MSG = "Automatically set max_tokens_per_minute to input=400000 output=80000"

        with caplog.at_level(logging.INFO, logger=logger):
            dataset = helper.create_basic(temp_working_dir, mock_dataset, backend=backend, tracker_console=console, model="anthropic/claude-3-haiku-20240307")

        assert RPM_MSG in caplog.text
        assert TPM_MSG in caplog.text
        captured = output.getvalue()
        assert "with seperate input and output" in captured
        assert "input=400000 output=80000" in captured
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]


##############################
# Batch                      #
##############################


def _reload_batch_patch_deps():
    from bespokelabs.curator.request_processor.batch import base_batch_request_processor

    importlib.reload(base_batch_request_processor)


@pytest.mark.parametrize("temp_working_dir", (_BATCH_BACKENDS), indirect=True)
def test_batch_resume(temp_working_dir, mock_dataset):
    temp_working_dir, backend, vcr_config = temp_working_dir
    with vcr_config.use_cassette("basic_batch_resume.yaml"):
        with patch("bespokelabs.curator.request_processor.event_loop.run_in_event_loop") as mocked_run_loop:

            def _run_loop(func):
                if "poll_and_process_batches" in str(func):
                    return
                return run_in_event_loop(func)

            mocked_run_loop.side_effect = _run_loop
            with pytest.raises(ValueError):
                _reload_batch_patch_deps()
                helper.create_basic(temp_working_dir, mock_dataset, batch=True, backend=backend)
        from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker

        tracker_batch_file_path = temp_working_dir + "/testing_hash_123/batch_objects.jsonl"
        with open(tracker_batch_file_path, "r") as f:
            tracker = BatchStatusTracker.model_validate_json(f.read())
        assert tracker.n_total_requests == 3
        assert len(tracker.submitted_batches) == 1
        assert len(tracker.downloaded_batches) == 0

        patch.stopall()
        _reload_batch_patch_deps()
        helper.create_basic(temp_working_dir, mock_dataset, batch=True, backend=backend)
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
            helper.create_basic(temp_working_dir, mock_dataset, batch=True, backend=backend)
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
    hash_book = {
        "openai": "47127d9dcb428c18e5103dffcb0406ba2f9acab2f1ea974606962caf747b0ad5",
        "anthropic": "f38e7406448e95160ebe4d9b6148920ef37b019f23a4e2c57094fdd4bafb09be",
    }
    with vcr_config.use_cassette("basic_batch_completion.yaml"):
        output = StringIO()
        console = Console(file=output, width=300)

        dataset = helper.create_basic(temp_working_dir, mock_dataset, batch=True, backend=backend, tracker_console=console)
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]

        # Verify status tracker output
        captured = output.getvalue()
        assert "Processing batches using" in captured, captured
        assert "Batches: Total: 1 • Submitted: 0⋯ • Downloaded: 1✓" in captured, captured
        assert "Requests: Total: 3 • Submitted: 0⋯ • Succeeded: 3✓ • Failed: 0✗" in captured, captured
        assert "Final Curator Statistics" in captured, captured
        assert "Total Requests             │ 3" in captured, captured
        assert "Successful                 │ 3" in captured, captured
        assert "Failed                     │ 0" in captured, captured


##############################
# Offline                    #
##############################


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "vllm"}]), indirect=True)
def test_basic_offline(temp_working_dir, mock_dataset):
    """Test basic completion with VLLM backend"""
    temp_working_dir, _, _ = temp_working_dir

    import json
    import os

    # Load mock responses from fixture file
    fixture_path = os.path.join(os.path.dirname(__file__), "vllm", "fixtures", "basic_responses.json")
    with open(fixture_path) as f:
        mock_responses = json.load(f)

    # Mock the vllm.LLM.generate method based on replay output
    class MockVLLMOutput:
        def __init__(self, text, request_id):
            self.text = text
            self.request_id = request_id
            self.finished = True
            self.prompt = None  # From replay output
            self.encoder_prompt = None
            self.metrics = None

        @property
        def outputs(self):
            return [type("MockOutput", (), {"text": self.text})]

    def mock_generate(prompts, sampling_params):
        """Mock the generate method based on replay output"""
        assert len(prompts) == 3  # Verify batch size
        # Verify prompts match the expected format
        template = (
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            "<|im_end|>\n"
            "<|im_start|>user\n{}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for i, prompt in enumerate(prompts):
            assert prompt == template.format(mock_dataset[i]["dish"])

        return [MockVLLMOutput(mock_responses[str(i)], i) for i in range(len(prompts))]

    def mock_apply_chat_template(conversation=None, tokenize=None, add_generation_prompt=None, **kwargs):
        """Mock the tokenizer's apply_chat_template method"""
        assert len(conversation) == 1  # We expect single message per prompt
        assert conversation[0]["role"] == "user"
        template = (
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            "<|im_end|>\n"
            "<|im_start|>user\n{}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return template.format(conversation[0]["content"])

    # Mock CUDA-related methods
    mock_cuda = type(
        "MockCuda",
        (),
        {
            "synchronize": lambda: None,
            "empty_cache": lambda: None,
            "is_available": lambda: True,
            "get_device_name": lambda device: "Mock GPU",
            "device_count": lambda: 1,
        },
    )

    with (
        patch("vllm.LLM") as mock_llm,
        patch("torch.cuda", mock_cuda),
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.empty_cache"),
    ):
        mock_llm.return_value.generate = mock_generate
        mock_llm.return_value.get_tokenizer.return_value.apply_chat_template = mock_apply_chat_template

        dataset = helper.create_basic(
            temp_working_dir,
            mock_dataset,
            backend="vllm",
        )

        # Verify response content
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == "f0e229cb0b9c6d60930abda07998fe5870c7e94331ca877af8f400f9697213ee"
