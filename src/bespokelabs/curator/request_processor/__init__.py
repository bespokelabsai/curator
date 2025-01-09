def get_batch_processors():
    from .batch.anthropic_batch_request_processor import AnthropicBatchRequestProcessor
    from .batch.base_batch_request_processor import BaseBatchRequestProcessor
    from .batch.openai_batch_request_processor import OpenAIBatchRequestProcessor

    return BaseBatchRequestProcessor, AnthropicBatchRequestProcessor, OpenAIBatchRequestProcessor


from .online.base_online_request_processor import APIRequest, BaseOnlineRequestProcessor
from .online.litellm_online_request_processor import LiteLLMOnlineRequestProcessor
from .online.openai_online_request_processor import OpenAIOnlineRequestProcessor
from .offline.base_offline_request_processor import BaseOfflineRequestProcessor
from .offline.vllm_offline_request_processor import VLLMOfflineRequestProcessor

__all__ = [
    "get_batch_processors",
    "BaseOnlineRequestProcessor",
    "LiteLLMOnlineRequestProcessor",
    "OpenAIOnlineRequestProcessor",
    "BaseOfflineRequestProcessor",
    "VLLMOfflineRequestProcessor",
    "APIRequest",
]
