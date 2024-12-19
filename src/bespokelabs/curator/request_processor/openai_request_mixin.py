from typing import Any
from bespokelabs.curator.types.generic_request import GenericRequest


class OpenAIRequestMixin:
    """Mixin class for creating OpenAI-specific API requests.

    Provides shared functionality for both batch and online OpenAI request processors.
    """

    def create_api_specific_request_online(
        self, generic_request: GenericRequest, generation_kwargs: dict | None = None
    ) -> dict:
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

        # TODO probably want to call to litellm here to get supported params
        if generation_kwargs is None:
            return request

        if generation_kwargs.get("temperature") is not None:
            request["temperature"] = generation_kwargs["temperature"]

        if generation_kwargs.get("top_p") is not None:
            request["top_p"] = generation_kwargs["top_p"]

        if generation_kwargs.get("presence_penalty") is not None:
            request["presence_penalty"] = generation_kwargs["presence_penalty"]

        if generation_kwargs.get("frequency_penalty") is not None:
            request["frequency_penalty"] = generation_kwargs["frequency_penalty"]

        return request
