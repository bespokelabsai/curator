from typing import Any

from bespokelabs.curator.types.generic_request import GenericRequest

# TODO: Add logic for high res detailed images
_OPENAI_TOKENS_PER_IMAGE = {"low": 85}


def calculate_input_tokens(message, token_encoding) -> int:
    """Calculate the number of tokens in the input message."""
    if isinstance(message, str):
        return len(token_encoding.encode(str(message), disallowed_special=()))
    else:
        tokens = 0
        for msg in message:
            if msg["type"] == "text":
                msg = msg["text"]
                tokens += len(token_encoding.encode(str(msg), disallowed_special=()))
            else:
                msg = msg["image_url"]
                # Note: Currently estimating low res image tokens. Need to add logic for high res images
                tokens += _OPENAI_TOKENS_PER_IMAGE["low"]

        return tokens


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
