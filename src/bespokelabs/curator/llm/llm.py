"""Curator: Bespoke Labs Synthetic Data Generation Library."""

import inspect
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar

from datasets import Dataset
from datasets.utils._dill import Pickler
from pydantic import BaseModel
from xxhash import xxh64

from bespokelabs.curator.db import MetadataDB
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor._factory import _RequestProcessorFactory

_CURATOR_DEFAULT_CACHE_DIR = "~/.cache/curator"
T = TypeVar("T")
_DictOrBaseModel = Dict[str, Any] | BaseModel

logger = logging.getLogger(__name__)


class LLM:
    """Interface for prompting LLMs."""

    def __init__(
        self,
        model_name: str,
        prompt_func: Callable[[_DictOrBaseModel], _DictOrBaseModel],
        parse_func: Callable[[_DictOrBaseModel, _DictOrBaseModel], _DictOrBaseModel] | None = None,
        base_url: str | None = None,
        response_format: Type[BaseModel] | None = None,
        batch: bool = False,
        backend: str | None = None,
        max_requests_per_minute: int | None = None,
        max_tokens_per_minute: int | None = None,
        batch_size: int | None = None,
        batch_check_interval: int | None = None,
        delete_successful_batch_files: bool | None = None,
        delete_failed_batch_files: bool | None = None,
        max_retries: int | None = None,
        require_all_responses: bool | None = None,
        generation_params: dict | None = None,
        seconds_to_pause_on_rate_limit: int | None = None,
        tensor_parallel_size: int | None = None,
        enforce_eager: bool | None = None,
        max_model_length: int | None = None,
        max_tokens: int | None = None,
        gpu_memory_utilization: float | None = None,
    ):
        """Initialize a LLM.

        Args:
            model_name: The name of the LLM to use
            prompt_func: A function that takes a single row
                and returns either a string (assumed to be a user prompt) or messages list
            parse_func: A function that takes the input row and
                response object and returns the parsed output
            base_url: Optional base URL for the API endpoint
            response_format: A Pydantic model specifying the
                response format from the LLM
            batch: Whether to use batch processing
            backend: The backend to use ("openai", "litellm", or "vllm"). If None, will be auto-determined
            max_requests_per_minute: Maximum number of requests per minute for rate limiting
            max_tokens_per_minute: Maximum number of tokens per minute for rate limiting
            batch_size: The size of the batch to use, only used if batch is True
            batch_check_interval: The interval to check for batch completions, only used if batch is True
            delete_successful_batch_files: Whether to delete successful batch files, only used if batch is True
            delete_failed_batch_files: Whether to delete failed batch files, only used if batch is True
            max_retries: The maximum number of retries to use for the LLM
            require_all_responses: Whether to require all responses
            generation_params: Additional parameters to pass to the generation API
            seconds_to_pause_on_rate_limit: Number of seconds to pause when rate limited
            tensor_parallel_size: The tensor parallel size to use for the VLLM backend
            enforce_eager: Whether to enforce eager execution for the VLLM backend
            max_model_length: The maximum model length to use for the VLLM backend
            max_tokens: The maximum tokens to use for the VLLM backend
            gpu_memory_utilization: The GPU memory utilization to use for the VLLM backend
        """
        if generation_params is None:
            generation_params = {}
        else:
            generation_params = _remove_none_values(generation_params)

        self.prompt_formatter = PromptFormatter(model_name, prompt_func, parse_func, response_format, generation_params)
        self.batch_mode = batch

        backend_params = {
            "model": model_name,
            "base_url": base_url,
            "batch_size": batch_size,
            "batch_check_interval": batch_check_interval,
            "delete_successful_batch_files": delete_successful_batch_files,
            "delete_failed_batch_files": delete_failed_batch_files,
            "require_all_responses": require_all_responses,
            "generation_params": generation_params,
            "max_requests_per_minute": max_requests_per_minute,
            "max_tokens_per_minute": max_tokens_per_minute,
            "max_retries": max_retries,
            "seconds_to_pause_on_rate_limit": seconds_to_pause_on_rate_limit,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": enforce_eager,
            "max_model_length": max_model_length,
            "max_tokens": max_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        self._request_processor = _RequestProcessorFactory.create(backend_params, batch=batch, response_format=response_format, backend=backend)

    def __call__(
        self,
        dataset: Optional[Iterable] = None,
        working_dir: str = None,
        batch_cancel: bool = False,
    ) -> Dataset:
        """Apply structured completions in parallel to a dataset using specified model and prompts.

        Args:
            dataset (Iterable): A dataset consisting of a list of items to apply completions
            working_dir (str): The working directory to save the requests.jsonl, responses.jsonl, and dataset.arrow files.
            batch_cancel (bool): Whether to cancel the batch if it is running
        Returns:
            Iterable: A list of structured outputs from the completions
        """
        # We convert from iterable to Dataset because Dataset has random access via row_idx
        if not isinstance(dataset, Dataset) and dataset is not None:
            dataset = Dataset.from_generator(dataset)

        if working_dir is None:
            curator_cache_dir = os.environ.get(
                "CURATOR_CACHE_DIR",
                os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
            )
        else:
            curator_cache_dir = working_dir

        dataset_hash = dataset._fingerprint if dataset is not None else xxh64("").hexdigest()

        prompt_func_hash = _get_function_hash(self.prompt_formatter.prompt_func)

        # Used to name the dataset .arrow file, but not the cache directory name
        # Modifying `parse_func` creates a new dataset file from cached responses
        parse_func_hash = _get_function_hash(self.prompt_formatter.parse_func)

        fingerprint_str = "_".join(
            [
                str(dataset_hash),
                str(prompt_func_hash),
                str(self.prompt_formatter.model_name),
                str(self.prompt_formatter.response_format.model_json_schema() if self.prompt_formatter.response_format else "text"),
                str(self.batch_mode),
                str(self._request_processor.backend),
            ]
        )

        if self.prompt_formatter.generation_params:
            generation_params_str = str(sorted(self.prompt_formatter.generation_params.items()))
            fingerprint_str += f"_{generation_params_str}"

        fingerprint = xxh64(fingerprint_str.encode("utf-8")).hexdigest()
        logger.debug(f"Curator Cache Fingerprint String: {fingerprint_str}")
        logger.debug(f"Curator Cache Fingerprint: {fingerprint}")

        metadata_db_path = os.path.join(curator_cache_dir, "metadata.db")
        metadata_db = MetadataDB(metadata_db_path)

        # Get the source code of the prompt function
        prompt_func_source = _get_function_source(self.prompt_formatter.prompt_func)
        if self.prompt_formatter.parse_func is not None:
            parse_func_source = _get_function_source(self.prompt_formatter.parse_func)
        else:
            parse_func_source = ""

        metadata_dict = {
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "prompt_func": prompt_func_source,
            "parse_func": parse_func_source,
            "model_name": self.prompt_formatter.model_name,
            "response_format": (str(self.prompt_formatter.response_format.model_json_schema()) if self.prompt_formatter.response_format else "text"),
            "run_hash": fingerprint,
            "batch_mode": self.batch_mode,
        }
        metadata_db.store_metadata(metadata_dict)

        run_cache_dir = os.path.join(curator_cache_dir, fingerprint)

        if batch_cancel:
            from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor

            if not isinstance(self._request_processor, OpenAIBatchRequestProcessor):
                raise ValueError("batch_cancel can only be used with batch mode")

            dataset = self._request_processor.cancel_batches(
                working_dir=run_cache_dir,
            )
        else:
            dataset = self._request_processor.run(
                dataset=dataset,
                working_dir=run_cache_dir,
                parse_func_hash=parse_func_hash,
                prompt_formatter=self.prompt_formatter,
            )

        return dataset


def _get_function_hash(func) -> str:
    """Get a hash of a function's source code."""
    if func is None:
        return xxh64("").hexdigest()

    file = BytesIO()
    Pickler(file, recurse=True).dump(func)
    return xxh64(file.getvalue()).hexdigest()


def _get_function_source(func) -> str:
    """Get the source code of a function.

    Purpose of this function is that during Python interpreter (REPL),
    `inspect.getsource` will fail with an OSError because functions defined in the
    interpreter don't have an associated source file. We have to use this wrapper
    to gracefully handle this case.
    """
    try:
        return inspect.getsource(func)
    except OSError:
        return ""


def _remove_none_values(d: dict) -> dict:
    """Remove all None values from a dictionary."""
    return {k: v for k, v in d.items() if v is not None}
