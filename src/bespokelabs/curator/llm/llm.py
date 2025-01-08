"""Curator: Bespoke Labs Synthetic Data Generation Library."""

import inspect
import logging
import os
import shutil
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar, Union

from datasets import Dataset
from datasets.utils._dill import Pickler
from pydantic import BaseModel
from xxhash import xxh64

from bespokelabs.curator.db import MetadataDB
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor import (
    AnthropicBatchRequestProcessor,
    LiteLLMOnlineRequestProcessor,
    OpenAIBatchRequestProcessor,
    OpenAIOnlineRequestProcessor,
)
from bespokelabs.curator.request_processor.config import (
    BatchRequestProcessorConfig,
    OnlineRequestProcessorConfig,
)

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
    ):
        """Initialize a LLM.

        Args:
            model_name: The name of the LLM to use
            prompt_func: A function that takes a single row
                and returns either a string (assumed to be a user prompt) or messages list
            parse_func: A function that takes the input row and
                response object and returns the parsed output
            response_format: A Pydantic model specifying the
                response format from the LLM.
            backend: The backend to use ("openai" or "litellm"). If None, will be auto-determined
            batch: Whether to use batch processing
            batch_size: The size of the batch to use, only used if batch is True
            batch_check_interval: The interval to check for batch completions, only used if batch is True
            delete_successful_batch_files: Whether to delete successful batch files, only used if batch is True
            delete_failed_batch_files: Whether to delete failed batch files, only used if batch is True
            max_retries: The maximum number of retries to use for the LLM
            require_all_responses: Whether to require all responses
            generation_params: The generation kwargs to use for the LLM
            seconds_to_pause_on_rate_limit: The number of seconds to pause for if a rate limit error occurs
        """
        if generation_params is None:
            generation_params = {}
        else:
            generation_params = _remove_none_values(generation_params)

        self.prompt_formatter = PromptFormatter(
            model_name, prompt_func, parse_func, response_format, generation_params
        )
        self.batch_mode = batch

        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._determine_backend(model_name, response_format, batch)

        if batch:
            config_params = {
                "model": model_name,
                "base_url": base_url,
                "batch_size": batch_size,
                "batch_check_interval": batch_check_interval,
                "delete_successful_batch_files": delete_successful_batch_files,
                "delete_failed_batch_files": delete_failed_batch_files,
                "max_retries": max_retries,
                "require_all_responses": require_all_responses,
                "generation_params": generation_params,
            }
            config = BatchRequestProcessorConfig(**_remove_none_values(config_params))
        else:
            config_params = {
                "model": model_name,
                "base_url": base_url,
                "max_requests_per_minute": max_requests_per_minute,
                "max_tokens_per_minute": max_tokens_per_minute,
                "max_retries": max_retries,
                "require_all_responses": require_all_responses,
                "generation_params": generation_params,
                "seconds_to_pause_on_rate_limit": seconds_to_pause_on_rate_limit,
            }
            config = OnlineRequestProcessorConfig(**_remove_none_values(config_params))

        if self.backend == "openai" and not batch:
            self._request_processor = OpenAIOnlineRequestProcessor(config)
        elif self.backend == "openai" and batch:
            self._request_processor = OpenAIBatchRequestProcessor(config)
        elif self.backend == "anthropic" and batch:
            self._request_processor = AnthropicBatchRequestProcessor(config)
        elif self.backend == "anthropic" and not batch:
            raise ValueError("Online mode is not currently supported with Anthropic backend.")
        elif self.backend == "litellm" and batch:
            raise ValueError("Batch mode is not supported with LiteLLM backend")
        elif self.backend == "litellm":
            self._request_processor = LiteLLMOnlineRequestProcessor(config)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    @staticmethod
    def _check_openai_structured_output_support(model_name: str) -> bool:
        config = OnlineRequestProcessorConfig(model=model_name)
        return OpenAIOnlineRequestProcessor(config).check_structured_output_support()

    @staticmethod
    def _determine_backend(
        model_name: str, response_format: Optional[Type[BaseModel]] = None, batch: bool = False
    ) -> str:
        """Determine which backend to use based on model name and response format.

        Args:
            model_name (str): Name of the model
            response_format (Optional[Type[BaseModel]]): Response format if specified
            batch (bool): Whether to use batch mode
        Returns:
            str: Backend to use ("openai" or "litellm")
        """
        model_name = model_name.lower()

        # GPT-4o models with response format should use OpenAI
        if response_format and LLM._check_openai_structured_output_support(model_name):
            logger.info(f"Requesting structured output from {model_name}, using OpenAI backend")
            return "openai"

        # GPT models and O1 models without response format should use OpenAI
        if not response_format and any(x in model_name for x in ["gpt-", "o1-preview", "o1-mini"]):
            logger.info(f"Requesting text output from {model_name}, using OpenAI backend")
            return "openai"

        if batch and "claude" in model_name:
            logger.info(f"Requesting output from {model_name}, using Anthropic backend")
            return "anthropic"

        # Default to LiteLLM for all other cases
        logger.info(
            f"Requesting {'structured' if response_format else 'text'} output from {model_name}, using LiteLLM backend"
        )
        return "litellm"

    def __call__(
        self,
        dataset: Optional[Iterable] = None,
        working_dir: Optional[str] = None,
        batch_cancel: bool = False,
    ) -> Dataset:
        """
        Apply structured completions in parallel to a dataset using specified model and
        prompts.

        Args:
            dataset (Iterable): A dataset consisting of a list of items to apply completions
            prompter (LLM): A LLM that contains the logic for formatting each
                item in the dataset
            working_dir (str): The working directory to save the requests.jsonl, responses.jsonl, and dataset.arrow files.

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
                str(
                    self.prompt_formatter.response_format.model_json_schema()
                    if self.prompt_formatter.response_format
                    else "text"
                ),
                str(self.batch_mode),
                str(self.backend),
            ]
        )

        if self.prompt_formatter.generation_params:
            generation_params_str = str(sorted(self.prompt_formatter.generation_params.items()))
            fingerprint_str += f"_{generation_params_str}"

        fingerprint = xxh64(fingerprint_str.encode("utf-8")).hexdigest()
        logger.debug(f"Curator Cache Fingerprint String: {fingerprint_str}")
        logger.debug(f"Curator Cache Fingerprint: {fingerprint}")

        # Check if cache is disabled
        cache_disabled = os.getenv("CURATOR_DISABLE_CACHE", "").lower() in ["true", "1"]

        # Only create metadata if caching is enabled
        if not cache_disabled:
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
                "response_format": (
                    str(self.prompt_formatter.response_format.model_json_schema())
                    if self.prompt_formatter.response_format
                    else "text"
                ),
                "run_hash": fingerprint,
                "batch_mode": self.batch_mode,
            }
            metadata_db.store_metadata(metadata_dict)
        else:
            logger.info("Cache disabled by CURATOR_DISABLE_CACHE env variable.")
            metadata_db = None

        # Use the same directory structure but clean it up later if cache is disabled
        run_cache_dir = os.path.join(curator_cache_dir, fingerprint)

        # Check for invalid combination of batch_cancel and cache_disabled
        if batch_cancel and cache_disabled:
            raise ValueError("Cannot use batch_cancel when CURATOR_DISABLE_CACHE is set.")

        try:
            if batch_cancel:
                if not isinstance(self._request_processor, OpenAIBatchRequestProcessor):
                    raise ValueError("batch_cancel can only be used with batch mode")

                from bespokelabs.curator.request_processor.event_loop import run_in_event_loop

                dataset = run_in_event_loop(self._request_processor.cancel_batches())
            else:
                dataset = self._request_processor.run(
                    dataset=dataset,
                    working_dir=run_cache_dir,
                    parse_func_hash=parse_func_hash,
                    prompt_formatter=self.prompt_formatter,
                )
        finally:
            # Clean up cache directory if caching is disabled
            if cache_disabled:
                shutil.rmtree(run_cache_dir, ignore_errors=True)
                logger.debug(f"Cleaned up directory for non-cached run: {run_cache_dir}")

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
