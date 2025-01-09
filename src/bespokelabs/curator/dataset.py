import glob
import logging
import os
from typing import Any, Dict, Iterable, Iterator, List, TypeVar

from datasets import Dataset as HFDataset
from datasets.arrow_writer import ArrowWriter, SchemaInferenceError
from pydantic import BaseModel

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.types.generic_response import GenericResponse

T = TypeVar("T")


class Dataset:
    """A dataset class that can be constructed from an iterable or working directory.

    This class provides functionality to work with datasets, including iterating over items,
    converting to different formats, and saving/loading from disk.

    Args:
        iterable: An optional iterable of dictionaries or BaseModel objects to construct the dataset from
        working_dir: Optional directory path where dataset files are stored
        prompt_formatter: Optional PromptFormatter instance for formatting prompts and responses
    """

    def __init__(
        self,
        iterable: Iterable[Dict[str, Any] | BaseModel] | None = None,
        working_dir: str | None = None,
        prompt_formatter: PromptFormatter | None = None,
    ):
        """Initialize the Dataset instance."""
        self.working_dir = working_dir
        self.prompt_formatter = prompt_formatter
        self.iterable = iterable

    @staticmethod
    def from_iterable(iterable: Iterable[Dict[str, Any] | BaseModel]):
        """Creates a Dataset instance from an iterable.

        Args:
            iterable: An iterable of dictionaries or BaseModel objects

        Returns:
            Dataset: A new Dataset instance
        """
        return Dataset(iterable=iterable)

    @staticmethod
    def from_working_dir(working_dir: str, prompt_formatter: PromptFormatter):
        """Creates a Dataset instance from a working directory.

        Args:
            working_dir: Directory path containing dataset files
            prompt_formatter: PromptFormatter instance for formatting prompts and responses

        Returns:
            Dataset: A new Dataset instance
        """
        return Dataset(working_dir=working_dir, prompt_formatter=prompt_formatter)

    def __iter__(self) -> Iterator[Dict[str, Any] | BaseModel]:
        """Iterates over items in the dataset.

        Yields items either from the provided iterable or from response files in the working directory.

        Yields:
            Dict[str, Any] | BaseModel: Dataset items
        """
        if self.iterable is not None:
            yield from self.iterable
            return

        if self.working_dir:
            response_file = os.path.join(self.working_dir, "responses.jsonl")

            for line in open(response_file, "r"):
                response = GenericResponse.model_validate_json(line)
                if self.prompt_formatter.response_format:
                    response.response = self.prompt_formatter.response_format(**response.response)
                if self.prompt_formatter.parse_func:
                    response = self.prompt_formatter.parse_func(response.row, response.response)
                else:
                    response = [response.response]

                if isinstance(response, list):
                    yield from response
                else:
                    yield response

    def to_list(self) -> List[Dict[str, Any] | BaseModel]:
        """Converts the dataset to a list.

        Returns:
            List[Dict[str, Any] | BaseModel]: List containing all dataset items
        """
        return list(self)

    def to_huggingface(self, in_memory: bool = False) -> None:
        """Converts the dataset to a HuggingFace Dataset format.

        Processes response files in the working directory and writes them to an Arrow file,
        which is then loaded as a HuggingFace Dataset.

        Args:
            in_memory: Whether to load the dataset into memory

        Returns:
            HFDataset: A HuggingFace Dataset instance

        Raises:
            ValueError: If no response files are found or if all requests failed
            ValueError: If the Arrow writer encounters schema inference errors
        """
        total_responses_count = 0
        failed_responses_count = 0

        os.makedirs(self.working_dir, exist_ok=True)

        dataset_file = os.path.join(self.working_dir, "dataset.arrow")
        responses_files = glob.glob(os.path.join(self.working_dir, "responses_*.jsonl"))
        if len(responses_files) == 0:
            raise ValueError(f"No responses files found in {self.working_dir}, can't construct dataset")

        # Process all response files
        with ArrowWriter(path=dataset_file) as writer:
            for responses_file in responses_files:
                with open(responses_file, "r") as f_in:
                    for line in f_in:
                        total_responses_count += 1
                        response = GenericResponse.model_validate_json(line)
                        if self.prompt_formatter.response_format:
                            response.response = self.prompt_formatter.response_format(**response.response)

                        if response is None:
                            failed_responses_count += 1
                            continue

                        if self.prompt_formatter.parse_func:
                            dataset_rows = self.prompt_formatter.parse_func(response.row, response.response)
                        else:
                            dataset_rows = [response.response]

                        for row in dataset_rows:
                            if isinstance(row, BaseModel):
                                row = row.model_dump()
                            writer.write(row)

            logger = logging.getLogger(__name__)
            logger.info(f"Read {total_responses_count} responses, {failed_responses_count} failed")
            logger.info("Finalizing writer")

            if failed_responses_count == total_responses_count:
                raise ValueError("All requests failed")

            # TODO(Ryan): Look at what this file looks like before finalize. What happens during finalize?
            try:
                writer.finalize()
            except SchemaInferenceError as e:
                raise ValueError(  # type: ignore
                    "Arrow writer is complaining about the schema: likely all of your parsed rows were None and writer.write only wrote None objects."
                ) from e

        return HFDataset.from_file(dataset_file, in_memory=in_memory)
