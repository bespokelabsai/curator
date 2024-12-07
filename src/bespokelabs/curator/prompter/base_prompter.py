"""Base class for Prompter implementations.

This module provides the abstract base class for implementing custom prompters.
The BasePrompter class defines the interface that all prompter implementations
must follow.

Example:
    Creating a custom prompter:
    ```python
    class CustomPrompter(BasePrompter):
        def prompt_func(self, row: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
            # Generate prompts for your specific use case
            if row is None:
                return {
                    "user_prompt": "Default prompt",
                    "system_prompt": "System instructions",
                }
            return {
                "user_prompt": f"Process input: {row['data']}",
                "system_prompt": "System instructions",
            }

        def parse_func(self, row: Dict[str, Any], response: Dict[str, Any]) -> Any:
            # Optional: Override to customize response parsing
            return response

    # Usage
    prompter = CustomPrompter(
        model_name="gpt-4",
        response_format=MyResponseFormat,
    )
    result = prompter(dataset)  # Process dataset
    single_result = prompter()  # Single completion
    ```

For simpler use cases where you don't need a full class implementation,
you can use the function-based approach with the Prompter class directly.
See the Prompter class documentation for details.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T")


class BasePrompter(ABC):
    """Abstract base class for prompter implementations.

    This class defines the interface for prompter implementations. Subclasses must
    implement prompt_func and may optionally override parse_func.
    """

    def __init__(
        self,
        model_name: str,
        response_format: Optional[Type[BaseModel]] = None,
        batch: bool = False,
        batch_size: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        """Initialize a BasePrompter.

        Args:
            model_name (str): The name of the LLM to use
            response_format (Optional[Type[BaseModel]]): A Pydantic model specifying the
                response format from the LLM.
            batch (bool): Whether to use batch processing
            batch_size (Optional[int]): The size of the batch to use, only used if batch is True
            temperature (Optional[float]): The temperature to use for the LLM, only used if batch is False
            top_p (Optional[float]): The top_p to use for the LLM, only used if batch is False
            presence_penalty (Optional[float]): The presence_penalty to use for the LLM, only used if batch is False
            frequency_penalty (Optional[float]): The frequency_penalty to use for the LLM, only used if batch is False
        """
        self.model_name = model_name
        self.response_format = response_format
        self.batch_mode = batch
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    @abstractmethod
    def prompt_func(
        self,
        row: Optional[Union[Dict[str, Any], BaseModel]] = None,
    ) -> Dict[str, str]:
        """Override this method to define how prompts are generated.

        Args:
            row (Optional[Union[Dict[str, Any], BaseModel]]): The input row to generate a prompt for.
                If None, generate a prompt without input data.

        Returns:
            Dict[str, str]: A dictionary containing the prompt components (e.g., user_prompt, system_prompt).
        """
        pass

    def parse_func(
        self,
        row: Union[Dict[str, Any], BaseModel],
        response: Union[Dict[str, Any], BaseModel],
    ) -> T:
        """Override this method to define how responses are parsed.

        Args:
            row (Union[Dict[str, Any], BaseModel]): The input row that generated the response.
            response (Union[Dict[str, Any], BaseModel]): The response from the LLM.

        Returns:
            T: The parsed response in the desired format.
        """
        return response
