"""Curator: Bespoke Labs Synthetic Data Generation Library."""

import inspect
import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar, Union

from datasets import Dataset
from pydantic import BaseModel
from xxhash import xxh64
import logging

from bespokelabs.curator.db import MetadataDB
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
)
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.openai_batch_request_processor import (
    OpenAIBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.openai_online_request_processor import (
    OpenAIOnlineRequestProcessor,
)

_CURATOR_DEFAULT_CACHE_DIR = "~/.cache/curator"
T = TypeVar("T")

logger = logging.getLogger(__name__)


class Prompter:
    """Interface for prompting LLMs."""

    def __init__(
        self,
        model_name: str,
        prompt_func: Callable[
            [Union[Dict[str, Any], BaseModel]], Dict[str, str]
        ],
        parse_func: Optional[
            Callable[
                [
                    Union[Dict[str, Any], BaseModel],
                    Union[Dict[str, Any], BaseModel],
                ],
                T,
            ]
        ] = None,
        response_format: Optional[Type[BaseModel]] = None,
        batch: bool = False,
        batch_size: Optional[int] = None,
        api_base: Optional[str] = "https://api.openai.com",
        max_requests_per_minute: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None,
    ):
        """Initialize a Prompter.

        Args:
            model_name (str): The name of the LLM to use
            prompt_func (Callable[[Dict[str, Any]], Union[str, List[Dict[str, Any]]]]): A function that takes a single row
                and returns either a string (assumed to be a user prompt) or messages list
            parse_func (Callable[[Dict[str, Any], Any], T]): A function that takes the input row and
                response object and returns the parsed output
            response_format (Optional[Type[BaseModel]]): A Pydantic model specifying the
                response format from the LLM.
            batch (bool): Whether to use batch processing
            batch_size (Optional[int]): The size of the batch to use, only used if batch is True
        """
        prompt_sig = inspect.signature(prompt_func)
        if len(prompt_sig.parameters) > 1:
            raise ValueError(
                f"prompt_func must take one argument or less, got {len(prompt_sig.parameters)}"
            )

        if parse_func is not None:
            parse_sig = inspect.signature(parse_func)
            if len(parse_sig.parameters) != 2:
                raise ValueError(
                    f"parse_func must take exactly 2 arguments, got {len(parse_sig.parameters)}"
                )

        self.prompt_formatter = PromptFormatter(
            model_name, prompt_func, parse_func, response_format
        )

        if api_base is not None:
            api_base = api_base.rstrip("/")
            url = api_base + "/v1/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"

        self.batch_mode = batch
        if batch:
            self._request_processor = OpenAIBatchRequestProcessor(
                model=model_name, batch_size=batch_size, url=url, max_requests_per_minute=max_requests_per_minute, max_tokens_per_minute=max_tokens_per_minute
            )
        else:
            if batch_size is not None:
                logger.warning(
                    f"Prompter argument `batch_size` {batch_size} is ignored because `batch` is False"
                )
            self._request_processor = OpenAIOnlineRequestProcessor(
                model=model_name, url=url, max_requests_per_minute=max_requests_per_minute, max_tokens_per_minute=max_tokens_per_minute
            )

    def __call__(
        self, dataset: Optional[Iterable] = None, working_dir: str = None
    ) -> Dataset:
        """
        Run completions on a dataset.

        Args:
            dataset (Iterable): A dataset consisting of a list of items to apply completions
            working_dir (str): The working directory to save the requests.jsonl, responses.jsonl, and dataset.arrow files.
        """
        return self._completions(self._request_processor, dataset, working_dir)

    def _completions(
        self,
        request_processor: BaseRequestProcessor,
        dataset: Optional[Iterable] = None,
        working_dir: str = None,
    ) -> Dataset:
        """
        Apply structured completions in parallel to a dataset using specified model and
        prompts.

        Args:
            dataset (Iterable): A dataset consisting of a list of items to apply completions
            prompter (Prompter): A Prompter that contains the logic for formatting each
                item in the dataset
            working_dir (str): The working directory to save the requests.jsonl, responses.jsonl, and dataset.arrow files.

        Returns:
            Iterable: A list of structured outputs from the completions
        """
        # NOTE(Ryan): We convert from iterable to Dataset because Dataset has random access via row_idx
        if not isinstance(dataset, Dataset) and dataset is not None:
            dataset = Dataset.from_generator(dataset)

        if self is None:
            raise ValueError("Prompter must be provided")

        if working_dir is None:
            curator_cache_dir = os.environ.get(
                "CURATOR_CACHE_DIR",
                os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
            )
        else:
            curator_cache_dir = working_dir

        dataset_hash = (
            dataset._fingerprint
            if dataset is not None
            else xxh64("").hexdigest()
        )

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
                    self.prompt_formatter.response_format.schema_json()
                    if self.prompt_formatter.response_format
                    else "text"
                ),
                str(self.batch_mode),
            ]
        )

        fingerprint = xxh64(fingerprint_str.encode("utf-8")).hexdigest()
        metadata_db_path = os.path.join(curator_cache_dir, "metadata.db")
        metadata_db = MetadataDB(metadata_db_path)

        # Get the source code of the prompt function
        prompt_func_source = inspect.getsource(
            self.prompt_formatter.prompt_func
        )
        if self.prompt_formatter.parse_func is not None:
            parse_func_source = inspect.getsource(
                self.prompt_formatter.parse_func
            )
        else:
            parse_func_source = ""

        metadata_dict = {
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "prompt_func": prompt_func_source,
            "parse_func": parse_func_source,
            "model_name": self.prompt_formatter.model_name,
            "response_format": (
                self.prompt_formatter.response_format.schema_json()
                if self.prompt_formatter.response_format
                else "text"
            ),
            "run_hash": fingerprint,
            "batch_mode": self.batch_mode,
        }
        metadata_db.store_metadata(metadata_dict)

        dataset = request_processor.run(
            dataset=dataset,
            working_dir=os.path.join(curator_cache_dir, fingerprint),
            parse_func_hash=parse_func_hash,
            prompt_formatter=self.prompt_formatter,
        )

        return dataset


def _get_function_hash(func) -> str:
    """Get a hash of a function's source code."""
    if func is None:
        return xxh64("").hexdigest()

    return xxh64(inspect.getsource(func)).hexdigest()
