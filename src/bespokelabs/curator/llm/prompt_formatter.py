"""Module for formatting prompts and handling responses for LLM interactions."""

import copy
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from bespokelabs.curator.constants import _INTERNAL_PROMPT_KEY
from bespokelabs.curator.log import logger
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.prompt import _MultiModalPrompt

T = TypeVar("T")
_DictOrBaseModel = Union[Dict[str, Any], BaseModel]


def _validate_messages(messages: list[dict]) -> None:
    """Validates that messages conform to the expected chat format.

    Args:
        messages: A list of message dictionaries to validate.

    Raises:
        ValueError: If messages don't meet the required format:
            - Must be a list of dictionaries
            - Each message must have 'role' and 'content' keys
            - Role must be one of: 'system', 'user', 'assistant'
    """
    valid_roles = {"system", "user", "assistant"}

    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("In the return value (a list) of the prompt_func, each message must be a dictionary")

        if "role" not in msg or "content" not in msg:
            raise ValueError("In the return value (a list) of the prompt_func, each message must contain 'role' and 'content' keys")

        if msg["role"] not in valid_roles:
            raise ValueError(f"In the return value (a list) of the prompt_func, each message role must be one of: {', '.join(sorted(valid_roles))}")


@dataclass
class PromptFormatter:
    """Class for formatting prompts and handling responses for LLM interactions.

    Attributes:
        model_name: Name of the LLM model to use
        prompt_func: Function that takes input and returns formatted prompts
        parse_func: Optional function to parse model responses
        response_format: Optional Pydantic model defining expected response format
        generation_params: Dictionary of parameters for generation
    """

    model_name: str
    prompt_func: Callable[[_DictOrBaseModel], Dict[str, str]]
    parse_func: Optional[Callable[[_DictOrBaseModel, _DictOrBaseModel], T]] = None
    response_format: Optional[Type[BaseModel]] = None
    generation_params: dict = field(default_factory=dict)

    def get_prompts(self, row: _DictOrBaseModel) -> str | list[str]:
        """Get the prompts for the given row."""
        sig = inspect.signature(self.prompt_func)
        if len(sig.parameters) == 0:
            return self.prompt_func()
        if len(sig.parameters) == 1:
            if isinstance(row, dict):
                if _INTERNAL_PROMPT_KEY in row:
                    return self.prompt_func(row[_INTERNAL_PROMPT_KEY])
            return self.prompt_func(row)
        raise ValueError(f"Prompting function {self.prompt_func} must have 0 or 1 arguments.")

    def get_messages(self, prompts: str | list[str]) -> list[dict]:
        """Get the messages for the given prompts."""
        if isinstance(prompts, str):
            return [{"role": "user", "content": prompts}]
        elif isinstance(prompts, list):
            _validate_messages(prompts)
            return prompts
        elif isinstance(prompts, tuple):
            return [{"role": "user", "content": _MultiModalPrompt.load(prompts)}]
        raise ValueError(f"The return value of the `prompt` method {type(prompts)} did not match the expected format.")

    def create_generic_request(self, row: _DictOrBaseModel, idx: int, generation_params_per_row: bool = False) -> GenericRequest:
        """Format the request object based off of `LLM` attributes.

        Args:
            row: Input data to format into a prompt
            idx: Index of the row in the dataset
            generation_params_per_row: Whether the dataset has a generation_params column
        Returns:
            GenericRequest object containing the formatted request

        Raises:
            ValueError: If prompt_func has invalid number of arguments or returns invalid format
        """
        # Convert BaseModel to dict for serialization
        if isinstance(row, BaseModel):
            row = row.model_dump()
        prompts = self.get_prompts(row)
        messages = self.get_messages(prompts)
        multimodal_prompt = isinstance(prompts, tuple)

        row_generation_params = copy.deepcopy(self.generation_params)
        # Specify generation_params given in the row if applicable
        if generation_params_per_row:
            try:
                # Convert generation_params back to dict
                # Previously it was passed into the Dataset as a string to avoid automatic dictionary expansion
                # See https://github.com/bespokelabsai/curator/issues/325 for more detail
                loaded_params = json.loads(row["generation_params"])
                # Update only the keys that exist in row-level generation_params
                row_generation_params.update(loaded_params)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse generation params as JSON: {row['generation_params']}. Using default generation params.")

        return GenericRequest(
            model=self.model_name,
            messages=messages,
            original_row=row,
            original_row_idx=idx,
            response_format=(self.response_format.model_json_schema() if self.response_format else None),
            is_multimodal_prompt=multimodal_prompt,
            generation_params=row_generation_params,
        )

    def response_to_response_format(self, response_message: str | dict) -> Optional[dict | str]:
        """Converts a response message to a specified Pydantic model format.

        This method takes a response message (either as a string or dict) and validates/converts it
        according to the provided Pydantic model format. If the response message is a string,
        it first attempts to parse it as JSON. The resulting dict is then used to construct
        an instance of the specified Pydantic model.

        Args:
            response_message: The response message to convert, either as a JSON string
                or a dictionary.

        Returns:
            The validated response message as a Pydantic model instance.

        Raises:
            json.JSONDecodeError: If the response_message is a string but cannot be parsed as valid JSON.
            ValidationError: If the parsed response does not match the schema defined by response_format.
        """
        # Response message is a string, which is converted to a dict
        # The dict is then used to construct the response_format Pydantic model
        if self.response_format is None:
            return response_message

        try:
            # First try to parse the response message as JSON
            if isinstance(response_message, str):
                try:
                    response_dict = json.loads(response_message)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse response message as JSON: {response_message}. The model likely returned an invalid JSON format.")
                    raise e
            else:
                response_dict = response_message

            # Then construct the Pydantic model from the parsed dict
            response_message = self.response_format(**response_dict)
            return response_message

        except ValidationError as e:
            schema_str = json.dumps(self.response_format.model_json_schema(), indent=2)
            logger.warning(
                f"Pydantic failed to parse response message {response_message} with `response_format` {schema_str}. "
                f"The model likely returned a JSON that does not match the schema of the `response_format`."
            )
            raise e

    def parse_response_message(self, response_message: str) -> tuple[Optional[dict | str], Optional[list[str]]]:
        """Parse a response message from the model.

        Args:
            response_message: Raw response string from the model

        Returns:
            Tuple containing:
                - Parsed response message (as dict or str) or None if parsing failed
                - List of error messages if any occurred during parsing, or None
        """
        response_errors = None
        if self.response_format:
            try:
                response_message = json.loads(response_message)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse response as JSON: {response_message}, skipping this response.")
                response_message = None
                response_errors = [f"Failed to parse response as JSON: {response_message}"]
        return response_message, response_errors
