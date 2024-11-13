from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .generic_request import GenericRequest

"""A generic response model for LLM API requests.

Attributes:
    response_message: The main response content. Can be:
        - None when there are errors
        - str for non-structured output
        - Dict[str, Any] for structured output
    response_errors: List of error messages. None when there are no errors.
    raw_response: The raw response data from the API.
    raw_request: The raw request data. Will be None for BatchAPI requests.
    generic_request: The associated GenericRequest object.
"""


class GenericResponse(BaseModel):
    response_message: Optional[Dict[str, Any]] | str = None
    response_errors: Optional[List[str]] = None
    raw_response: Optional[Dict[str, Any]]
    raw_request: Optional[Dict[str, Any]] = None
    generic_request: GenericRequest
