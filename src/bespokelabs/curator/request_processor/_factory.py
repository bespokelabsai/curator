import os
import typing as t

from pydantic import BaseModel

from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.config import (
    BackendParamsType,
    BatchRequestProcessorConfig,
    OfflineRequestProcessorConfig,
    OnlineRequestProcessorConfig,
    _validate_backend_params,
)

if t.TYPE_CHECKING:
    from pydantic import BaseModel

    from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor


# TODO: Redundant move to misc module.
def _remove_none_values(d: dict) -> dict:
    """Remove all None values from a dictionary."""
    return {k: v for k, v in d.items() if v is not None}


class _RequestProcessorFactory:
    @classmethod
    def _create_config(cls, params, batch, backend):
        if backend == "vllm":
            return OfflineRequestProcessorConfig(**_remove_none_values(params))
        elif batch:
            return BatchRequestProcessorConfig(**_remove_none_values(params))
        return OnlineRequestProcessorConfig(**_remove_none_values(params))

    @staticmethod
    def _check_openai_structured_output_support(config_params) -> bool:
        from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor

        config = OnlineRequestProcessorConfig(**_remove_none_values(config_params))
        return OpenAIOnlineRequestProcessor(config).check_structured_output_support()

    @staticmethod
    def _determine_backend(
        model_name: str,
        config_params: BackendParamsType,
        response_format: t.Type["BaseModel"] | None = None,
        batch: bool = False,
    ) -> str:
        model_name = model_name.lower()

        # TODO: Move the following logic to corresponding client implementation
        # GPT-4o models with response format should use OpenAI
        if response_format and _RequestProcessorFactory._check_openai_structured_output_support(config_params):
            logger.info(f"Requesting structured output from {model_name}, using OpenAI backend")
            return "openai"

        # GPT models and O1 models and DeepSeek without response format should use OpenAI
        if not response_format and any(x in model_name for x in ["gpt-", "o1-preview", "o1-mini"]):
            logger.info(f"Requesting text output from {model_name}, using OpenAI backend")
            return "openai"

        if "claude" in model_name:
            logger.info(f"Requesting output from {model_name}, using Anthropic backend")
            return "anthropic"

        # Default to LiteLLM for all other cases
        logger.info(f"Requesting {'structured' if response_format else 'text'} output from {model_name}, using LiteLLM backend")
        return "litellm"

    @classmethod
    def create(
        cls,
        model_name: str,
        params: BackendParamsType | None,
        generation_params: t.Dict,
        batch: bool,
        backend: str | None,
        response_format: t.Type[BaseModel] | None,
        return_completions_object: bool = False,
    ) -> "BaseRequestProcessor":
        """Create appropriate processor instance based on config params."""
        if params:
            params["model"] = model_name
            _validate_backend_params(params)
        else:
            params = t.cast(BackendParamsType, {"model": model_name})

        params.update({"generation_params": generation_params})  # noqa

        if backend is None:
            backend = cls._determine_backend(model_name, params, response_format, batch)

        params["return_completions_object"] = return_completions_object
        config = cls._create_config(params, batch, backend)

        if backend == "klusterai" and not batch:
            config.base_url = "https://api.kluster.ai/v1"
            config.api_key = config.api_key or os.getenv("KLUSTERAI_API_KEY")
            if not config.api_key:
                raise ValueError("KLUSTERAI_API_KEY is not set")
            from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor

            _request_processor = OpenAIOnlineRequestProcessor(config, compatible_provider="klusterai")
        elif backend == "klusterai" and batch:
            config.base_url = "https://api.kluster.ai/v1"
            config.api_key = config.api_key or os.getenv("KLUSTERAI_API_KEY")
            if not config.api_key:
                raise ValueError("KLUSTERAI_API_KEY is not set")
            from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor

            _request_processor = OpenAIBatchRequestProcessor(config, compatible_provider="klusterai")
        elif backend == "inference.net" and not batch:
            config.base_url = "https://api.inference.net/v1"
            config.api_key = config.api_key or os.getenv("INFERENCE_API_KEY")
            if not config.api_key:
                raise ValueError("INFERENCE_API_KEY is not set")
            from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor

            _request_processor = OpenAIOnlineRequestProcessor(config, compatible_provider="inference.net")
        elif backend == "inference.net" and batch:
            config.base_url = "https://batch.inference.net/v1"
            config.api_key = config.api_key or os.getenv("INFERENCE_API_KEY")
            if not config.api_key:
                raise ValueError("INFERENCE_API_KEY is not set")
            from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor

            _request_processor = OpenAIBatchRequestProcessor(config, compatible_provider="inference.net")
        elif backend == "openai" and not batch:
            from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor

            _request_processor = OpenAIOnlineRequestProcessor(config)
        elif backend == "openai" and batch:
            from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor

            _request_processor = OpenAIBatchRequestProcessor(config)
        elif backend == "anthropic" and batch:
            from bespokelabs.curator.request_processor.batch.anthropic_batch_request_processor import AnthropicBatchRequestProcessor

            _request_processor = AnthropicBatchRequestProcessor(config)
        elif backend == "gemini" and batch:
            from bespokelabs.curator.request_processor.batch.gemini_batch_request_processor import GeminiBatchRequestProcessor

            _request_processor = GeminiBatchRequestProcessor(config)
        elif backend == "anthropic" and not batch:
            from bespokelabs.curator.request_processor.online.anthropic_online_request_processor import AnthropicOnlineRequestProcessor

            _request_processor = AnthropicOnlineRequestProcessor(config)
        elif backend == "litellm" and batch:
            raise ValueError("Batch mode is not supported with LiteLLM backend")
        elif backend == "litellm":
            from bespokelabs.curator.request_processor.online.litellm_online_request_processor import LiteLLMOnlineRequestProcessor

            _request_processor = LiteLLMOnlineRequestProcessor(config)
        elif backend == "vllm":
            from bespokelabs.curator.request_processor.offline.vllm_offline_request_processor import VLLMOfflineRequestProcessor

            _request_processor = VLLMOfflineRequestProcessor(config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return _request_processor
