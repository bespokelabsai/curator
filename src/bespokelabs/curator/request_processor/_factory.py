import logging
import typing as t

from bespokelabs.curator.request_processor.config import (
    BatchRequestProcessorConfig,
    OfflineRequestProcessorConfig,
    OnlineRequestProcessorConfig,
)

logger = logging.getLogger(__name__)

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
        if backend == 'vllm':
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
        config_params: t.Dict,
        response_format: t.Optional["BaseModel"] = None,
        batch: bool = False,
    ) -> str:
        model_name = config_params['model'].lower()

        # TODO: Move the following logic to corresponding client implementation
        # GPT-4o models with response format should use OpenAI
        if response_format and _RequestProcessorFactory._check_openai_structured_output_support(
            config_params
        ):
            logger.info(f"Requesting structured output from {model_name}, using OpenAI backend")
            return "openai"

        # GPT models and O1 models without response format should use OpenAI
        if not response_format and any(x in model_name for x in ["gpt-", "o1-preview", "o1-mini"]):
            logger.info(f"Requesting text output from {model_name}, using OpenAI backend")
            return "openai"

        if batch and "claude" in model_name:
            logger.info(f"Requesting output from {model_name}, using Anthropic backend")
            return "anthropic"

        # Default to LiteLLM for all other cases
        logger.info(
            f"Requesting {'structured' if response_format else 'text'} output from {model_name}, using LiteLLM backend"
        )
        return "litellm"

    @classmethod
    def create(cls, params, batch: bool, backend, response_format) -> "BaseRequestProcessor":
        """Create appropriate processor instance based on config params."""
        if backend == "vllm":
            config_params = {
                "model": params['model'],
                "generation_params": params['generation_params'],
                "tensor_parallel_size": params['tensor_parallel_size'],
                "enforce_eager": params['enforce_eager'],
                "max_model_length": params['max_model_length'],
                "max_tokens": params['max_tokens'],
                "gpu_memory_utilization": params['gpu_memory_utilization'],
                "batch_size": params['batch_size'] if params['batch_size'] is not None else 256,
            }
        elif batch:
            config_params = {
                "model": params['model'],
                "base_url": params['base_url'],
                "batch_size": params['batch_size'],
                "batch_check_interval": params['batch_check_interval'],
                "delete_successful_batch_files": params['delete_successful_batch_files'],
                "delete_failed_batch_files": params['delete_failed_batch_files'],
                "max_retries": params['max_retries'],
                "require_all_responses": params['require_all_responses'],
                "generation_params": params['generation_params'],
            }
        else:
            config_params = {
                "model": params['model'],
                "base_url": params['base_url'],
                "max_requests_per_minute": params['max_requests_per_minute'],
                "max_tokens_per_minute": params['max_tokens_per_minute'],
                "max_retries": params['max_retries'],
                "require_all_responses": params['require_all_responses'],
                "generation_params": params['generation_params'],
                "seconds_to_pause_on_rate_limit": params['seconds_to_pause_on_rate_limit'],
            }

        if backend is not None:
            backend = backend
        else:
            backend = cls._determine_backend(config_params, response_format, batch)

        config = cls._create_config(params, batch, backend)

        if backend == "openai" and not batch:
            from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor
            _request_processor = OpenAIOnlineRequestProcessor(config)
        elif backend == "openai" and batch:
            from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor
            _request_processor = OpenAIBatchRequestProcessor(config)
        elif backend == "anthropic" and batch:
            from bespokelabs.curator.request_processor.batch.anthropic_batch_request_processor import AnthropicBatchRequestProcessor
            _request_processor = AnthropicBatchRequestProcessor(config)
        elif backend == "anthropic" and not batch:
            raise ValueError("Online mode is not currently supported with Anthropic backend.")
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
