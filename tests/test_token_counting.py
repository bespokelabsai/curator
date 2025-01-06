"""Tests for token counting consistency across different request processors."""

import pytest
import os
from bespokelabs.curator.request_processor.online.litellm_online_request_processor import LiteLLMOnlineRequestProcessor
from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig

@pytest.fixture(autouse=True)
def check_environment():
    """Ensure required API keys are available for testing."""
    env = os.environ.copy()
    required_keys = [
        "OPENAI_API_KEY",
    ]
    for key in required_keys:
        assert key in env, f"{key} must be set"


def test_identical_prompts_have_consistent_token_counts():
    """Test that identical prompts get the same token count across processors."""
    # Configure both processors with the same model
    config = OnlineRequestProcessorConfig(
        model="gpt-4",
        request_timeout=30,
        max_attempts=1
    )
    
    litellm_processor = LiteLLMOnlineRequestProcessor(config)
    openai_processor = OpenAIOnlineRequestProcessor(config)
    
    # Test message that mimics the real-world example from the bug report
    test_messages = [
        {
            "role": "user",
            "content": "Generate a random {'cuisine': 'Chinese'} recipe. Be creative but keep it realistic."
        }
    ]
    
    # Get token counts from both processors
    litellm_count = litellm_processor.estimate_total_tokens(test_messages)
    openai_count = openai_processor.estimate_total_tokens(test_messages)
    
    # Assert that both processors return the same token count
    assert litellm_count == openai_count, (
        f"Token count mismatch: LiteLLM counted {litellm_count} tokens "
        f"while OpenAI counted {openai_count} tokens for identical messages"
    )
    
    # Test with a more complex message structure
    complex_messages = [
        {
            "role": "system",
            "content": "You are a helpful cooking assistant."
        },
        {
            "role": "user",
            "content": "Generate a recipe."
        },
        {
            "role": "assistant",
            "content": "What cuisine would you like?"
        },
        {
            "role": "user",
            "content": "Chinese cuisine please."
        }
    ]
    
    # Get token counts for complex messages
    litellm_complex_count = litellm_processor.estimate_total_tokens(complex_messages)
    openai_complex_count = openai_processor.estimate_total_tokens(complex_messages)
    
    # Assert consistency with complex messages
    assert litellm_complex_count == openai_complex_count, (
        f"Token count mismatch with complex messages: LiteLLM counted {litellm_complex_count} tokens "
        f"while OpenAI counted {openai_complex_count} tokens"
    )
