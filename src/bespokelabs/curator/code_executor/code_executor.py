"""CodeExecutor is a class that handles code execution and caching of results.

This class provides methods to execute code against test cases and cache the results
for future runs. It supports different backend execution engines and customizable
execution parameters.
"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, TypeVar

from pydantic import BaseModel
from xxhash import xxh64

from bespokelabs.curator.code_executor.code_execution_backend._factory import _CodeExecutionBackendFactory
from bespokelabs.curator.code_executor.code_formatter import CodeFormatter
from bespokelabs.curator.code_executor.db import CodeMetadataDB
from bespokelabs.curator.code_executor.types import CodeExecutionRequestParams
from bespokelabs.curator.llm.llm import _convert_to_dataset, _get_function_hash, _get_function_source
from bespokelabs.curator.log import logger

if TYPE_CHECKING:
    from datasets import Dataset

# Constants
_CURATOR_DEFAULT_CACHE_DIR = "~/.cache/curator"
T = TypeVar("T")
_DictOrBaseModel = Dict[str, Any] | BaseModel


class CodeExecutor:
    """A class that handles code execution and caching of results.

    This class provides methods to execute code against test cases and cache the results
    for future runs. It supports different backend execution engines and customizable
    execution parameters.
    """

    def code(self, row: dict):
        """Extract code from a dataset row."""
        pass

    def code_input(self, row: dict):
        """Extract input from a dataset row."""
        return ""

    def code_output(self, row: dict, execution_output: Any):
        """Extract output from a dataset row."""
        pass

    def __init__(
        self,
        backend: str = "multiprocessing",
        backend_params: dict = None,
    ):
        """Initialize the CodeExecutor with specified backend and parameters.

        Args:
            backend: The execution backend to use ("multiprocessing" by default)
            backend_params: Configuration parameters for the backend
        """
        self._code_executor = _CodeExecutionBackendFactory.create(backend=backend, backend_params=backend_params)

    def _hash_fingerprint(self, dataset_hash, disable_cache):
        """Generate a unique fingerprint for caching based on dataset and methods.

        Args:
            dataset_hash: Hash of the input dataset
            disable_cache: Flag to disable caching

        Returns:
            str: A unique hash fingerprint
        """
        if disable_cache:
            # Generate random fingerprint if caching is disabled
            fingerprint = xxh64(os.urandom(8)).hexdigest()
        else:
            # Generate deterministic fingerprint based on dataset and method hashes
            code_hash = _get_function_hash(self.code)
            input_hash = _get_function_hash(self.code_input)
            output_hash = _get_function_hash(self.code_output)

            fingerprint_str = "_".join(
                [
                    str(dataset_hash),
                    str(code_hash),
                    str(input_hash),
                    str(output_hash),
                ]
            )

            fingerprint = xxh64(fingerprint_str.encode("utf-8")).hexdigest()
            logger.debug(f"Curator Cache Fingerprint String: {fingerprint_str}")
            logger.debug(f"Curator Cache Fingerprint: {fingerprint}")

        return fingerprint

    def __call__(
        self,
        dataset: Optional[Iterable] = None,
        working_dir: str = None,
        execution_params: Optional[CodeExecutionRequestParams | dict] = None,
    ) -> "Dataset":
        """Execute code on the provided dataset.

        Args:
            dataset: Input dataset to process
            working_dir: Directory for cache and temporary files
            execution_params: Parameters controlling code execution

        Returns:
            Dataset: Processed dataset with execution results
        """
        # Convert input to Dataset format
        if dataset:
            dataset = _convert_to_dataset(dataset)

        # Set up cache directory
        if working_dir is None:
            curator_cache_dir = os.environ.get(
                "CURATOR_CACHE_DIR",
                os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
            )
        else:
            curator_cache_dir = working_dir

        if execution_params is None:
            execution_params = CodeExecutionRequestParams()
        elif isinstance(execution_params, dict):
            execution_params = CodeExecutionRequestParams(**execution_params)

        # Generate dataset hash and cache fingerprint
        dataset_hash = dataset._fingerprint if dataset is not None else xxh64("").hexdigest()
        disable_cache = os.getenv("CURATOR_DISABLE_CACHE", "").lower() in ["true", "1"]
        fingerprint = self._hash_fingerprint(dataset_hash, disable_cache)

        # Initialize and store metadata
        metadata_db_path = os.path.join(curator_cache_dir, "metadata.db")
        metadata_db = CodeMetadataDB(metadata_db_path)

        metadata_dict = {
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "code": _get_function_source(self.code),
            "code_input": _get_function_source(self.code_input),
            "code_output": _get_function_source(self.code_output),
            "run_hash": fingerprint,
        }

        metadata_db.store_metadata(metadata_dict)

        # Set up run-specific cache directory
        run_cache_dir = os.path.join(curator_cache_dir, fingerprint)

        # Generate hash of all function implementations
        all_func_hash = [
            _get_function_hash(self.code),
            _get_function_hash(self.code_input),
            _get_function_hash(self.code_output),
        ]

        all_func_hash_str = "_".join(all_func_hash)
        all_func_hash_hash = xxh64(all_func_hash_str.encode("utf-8")).hexdigest()

        # Initialize code formatter
        self.code_formatter = CodeFormatter(
            code=self.code,
            code_input=self.code_input,
            code_output=self.code_output,
            execution_params=execution_params,
        )

        # Execute code using configured backend
        dataset = self._code_executor.run(
            dataset=dataset,
            working_dir=run_cache_dir,
            code_formatter=self.code_formatter,
            all_func_hash_hash=all_func_hash_hash,
        )

        return dataset
