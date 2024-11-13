from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

"""A generic request model for LLM API requests.

Attributes:
    model: The name/identifier of the LLM model to use.
    messages: List of message dictionaries forming the conversation history.
    response_format: Optional json schema of a Pydantic model class for structured output validation.
        None indicates non-structured (str) output is expected.
    original_row: The source data being processed. The original row as a dictionary.
    original_row_idx: The index of the original row in the dataset.
"""


class GenericRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    response_format: Dict[str, Any] | None = None
    original_row: Dict[str, Any]
    original_row_idx: int
