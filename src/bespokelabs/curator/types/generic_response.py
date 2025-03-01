import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.token_usage import _TokenUsage

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
    created_at: The datetime when the request was created.
    finished_at: The datetime when the request was finished.
    token_usage: Token usage information for the request.
    response_cost: The cost of the request in USD.
    finish_reason: The reason for completion finish.
"""


class GenericResponse(BaseModel):
    """A generic response model for LLM API requests."""

    response_message: Optional[Dict[str, Any]] | str | BaseModel = None
    parsed_response_message: Optional[list] = None
    response_errors: Optional[List[str]] = None
    raw_response: Optional[Dict[str, Any]]
    raw_request: Optional[Dict[str, Any]] = None
    generic_request: GenericRequest
    created_at: datetime.datetime
    finished_at: datetime.datetime
    token_usage: Optional[_TokenUsage] = None
    response_cost: Optional[float] = None
    finish_reason: Optional[str] = None

    model_config = {
        "json_encoders": {datetime.datetime: lambda dt: dt.isoformat()},
    }
