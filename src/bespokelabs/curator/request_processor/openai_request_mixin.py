import logging
from typing import Any

import pydantic

from bespokelabs.curator.file_utilities import get_base64_size
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.prompt import _MultiModalPrompt

logger = logger = logging.getLogger(__name__)

_OPENAI_ALLOWED_IMAGE_SIZE_MB = 20  # MB


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
            "messages": self._unpack(generic_request.messages),
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

    def _unpack(self, messages):
        unpacked_messages = []
        for message in messages:
            try:
                content = _MultiModalPrompt.model_validate(message["content"])
                content = self._handle_multi_modal_prompt(content)
                message["content"] = content
                unpacked_messages.append(message)

            except pydantic.ValidationError:
                unpacked_messages.append(message)
        return unpacked_messages

    def _handle_multi_modal_prompt(self, message):
        content = []
        texts = message.texts
        for text in texts:
            content.append({"type": "text", "text": text})
        for image in message.images:
            if image.url:
                content.append({"type": "image_url", "image_url": {"url": image.url}})
            elif image.content:
                base64_image = image.serialize()
                mb = get_base64_size(base64_image)
                if mb > _OPENAI_ALLOWED_IMAGE_SIZE_MB:
                    raise ValueError(f"Image size is {mb} MB, which is greater than the " f"allowed size of {_OPENAI_ALLOWED_IMAGE_SIZE_MB} MB in OpenAI.")

                # TODO: add detail option in Image types.
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "low",
                        },
                    }
                )

        return content
