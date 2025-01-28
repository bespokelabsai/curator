import logging
from typing import Any

from bespokelabs.curator.types.generic_request import GenericRequest

logger = logger = logging.getLogger(__name__)


class OpenAIRequestMixin:
    """Mixin class for creating OpenAI-specific API request schema.

    Provides shared functionality for both batch and online OpenAI request processors.
    """

    def create_api_specific_request_online(self, generic_request: GenericRequest) -> dict:
        """Create an OpenAI-specific request from a generic request.

        Args:
            generic_request (GenericRequest): Generic request object

        Returns:
            dict: OpenAI API-compatible request dictionary

        Note:
            - Handles JSON schema response format if specified
            - Applies optional parameters (temperature, top_p, etc.)
        """
        request: dict[str, Any] = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }

        if generic_request.response_format:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output_schema",  # NOTE: not using 'strict': True
                    "schema": generic_request.response_format,
                },
            }

        for key, value in generic_request.generation_params.items():
            request[key] = value

        return request
