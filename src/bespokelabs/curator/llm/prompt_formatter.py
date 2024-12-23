import dataclasses
import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from bespokelabs.curator.request_processor.generic_request import GenericRequest

T = TypeVar("T")
_DictOrBaseModel = Union[Dict[str, Any], BaseModel]
logger = logging.getLogger(__name__)


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
            raise ValueError(
                "In the return value (a list) of the prompt_func, each "
                "message must be a dictionary"
            )

        if "role" not in msg or "content" not in msg:
            raise ValueError(
                "In the return value (a list) of the prompt_func, each "
                "message must contain 'role' and 'content' keys"
            )

        if msg["role"] not in valid_roles:
            raise ValueError(
                f"In the return value (a list) of the prompt_func, "
                f"each message role must be one of: {', '.join(sorted(valid_roles))}"
            )


@dataclasses.dataclass
class PromptFormatter:
    model_name: str
    prompt_func: Callable[[_DictOrBaseModel], Dict[str, str]]
    parse_func: Optional[Callable[[_DictOrBaseModel, _DictOrBaseModel], T]] = None
    response_format: Optional[Type[BaseModel]] = None

    def create_generic_request(self, row: _DictOrBaseModel, idx: int) -> GenericRequest:
        """Format the request object based off of `LLM` attributes."""
        sig = inspect.signature(self.prompt_func)
        if len(sig.parameters) == 0:
            prompts = self.prompt_func()
        elif len(sig.parameters) == 1:
            prompts = self.prompt_func(row)
        else:
            raise ValueError(f"Prompting function {self.prompt_func} must have 0 or 1 arguments.")

        if isinstance(prompts, str):
            messages = [{"role": "user", "content": prompts}]
        elif isinstance(prompts, list):
            _validate_messages(prompts)
            messages = prompts
        else:
            raise ValueError("The return value of the prompt_func must be a list of dictionaries.")

        # Convert BaseModel to dict for serialization
        if isinstance(row, BaseModel):
            row = row.model_dump()

        return GenericRequest(
            model=self.model_name,
            messages=messages,
            original_row=row,
            original_row_idx=idx,
            response_format=(
                self.response_format.model_json_schema() if self.response_format else None
            ),
        )

    def response_to_response_format(self, response_message: str | dict) -> Optional[dict | str]:
        """
        Converts a response message to a specified Pydantic model format.

        This method takes a response message (either as a string or dict) and validates/converts it
        according to the provided Pydantic model format. If the response message is a string,
        it first attempts to parse it as JSON. The resulting dict is then used to construct
        an instance of the specified Pydantic model.

        Args:
            response_message (str | dict): The response message to convert, either as a JSON string
                or a dictionary.
            response_format (Optional[BaseModel]): The Pydantic model class that defines the
                expected format of the response.

        Returns:
            Optional[dict | str]: The validated response message as a Pydantic model instance.

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
                    logger.warning(
                        f"Failed to parse response message as JSON: {response_message}. "
                        f"The model likely returned an invalid JSON format."
                    )
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
