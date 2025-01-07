import logging
import typing as t

from bespokelabs.curator.request_processor import (
    LiteLLMOnlineRequestProcessor,
    OpenAIOnlineRequestProcessor,
    AnthropicBatchRequestProcessor,
    OpenAIBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.config import (
    BatchRequestProcessorConfig,
    OnlineRequestProcessorConfig,
)

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from pydantic import BaseModel
    from bespokelabs.curator.request_processor.config import (
        RequestProcessorConfig,
    )
    from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor 

#TODO: Redundant move to misc module.
def _remove_none_values(d: dict) -> dict:
    """Remove all None values from a dictionary."""
    return {k: v for k, v in d.items() if v is not None}

class _RequestProcessorFactory:
    @classmethod
    def _create_config(cls, params, batch):
        if batch:
            return BatchRequestProcessorConfig(**_remove_none_values(params))
        return OnlineRequestProcessorConfig(**_remove_none_values(params))
    
    @staticmethod
    def _check_openai_structured_output_support(config) -> bool:
        return OpenAIOnlineRequestProcessor(config).check_structured_output_support()
    
    @staticmethod
    def _determine_backend(
        config: 'RequestProcessorConfig', response_format: t.Optional['BaseModel'] = None, batch: bool = False
    ) -> str:
        model_name = config.model.lower()
        
        # TODO: Move the following logic to corresponding client implementation
        # GPT-4o models with response format should use OpenAI
        if response_format and _RequestProcessorFactory._check_openai_structured_output_support(config):
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
    def create(cls, params, batch: bool, backend, response_format) -> 'BaseRequestProcessor':
        """Create appropriate processor instance based on config params"""
        config = cls._create_config(params, batch)

        if backend is not None:
            backend = backend
        else:
            backend = cls._determine_backend(config, response_format, batch)

        if backend == "openai" and not batch:
            _request_processor = OpenAIOnlineRequestProcessor(config)
        elif backend == "openai" and batch:
            _request_processor = OpenAIBatchRequestProcessor(config)
        elif backend == "anthropic" and batch:
            _request_processor = AnthropicBatchRequestProcessor(config)
        elif backend == "anthropic" and not batch:
            raise ValueError("Online mode is not currently supported with Anthropic backend.")
        elif backend == "litellm" and batch:
            raise ValueError("Batch mode is not supported with LiteLLM backend")
        elif backend == "litellm":
            _request_processor = LiteLLMOnlineRequestProcessor(config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return _request_processor
