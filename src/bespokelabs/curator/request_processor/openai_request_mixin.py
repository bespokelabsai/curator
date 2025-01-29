import logging
from typing import Any

from bespokelabs.curator.types.generic_request import GenericRequest

logger = logger = logging.getLogger(__name__)


class OpenAIRequestMixin:
    """Mixin class for creating OpenAI-specific API requests.

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

    @classmethod
    def patch_external_openai_compatibles(cls, config):
        """Patch the configuration for external OpenAI-compatible models."""
        # Route DeepSeek to OpenAI online backend since LiteLLM does not return
        # reasoning_content
        if "deepseek" in config.model:
            config.base_url = "https://api.deepseek.com"
            logger.warn("Overriding base_url to https://api.deepseek.com config")

        elif "klusterai" in config.model:
            config.base_url = "https://api.kluster.ai/v1"
            logger.warn("Overriding base_url to https://api.kluster.ai/v1 config")
        return config
