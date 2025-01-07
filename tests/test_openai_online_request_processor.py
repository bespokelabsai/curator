import pytest

from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor


def test_special_token_handling():
    """Test that special tokens like <|endoftext|> are handled correctly in token estimation."""
    config = OnlineRequestProcessorConfig(model="gpt-4")
    processor = OpenAIOnlineRequestProcessor(config)

    # Test message containing special token
    messages = [{"role": "user", "content": "Testing <|endoftext|> token"}]

    try:
        total_tokens = processor.estimate_total_tokens(messages)
        assert total_tokens > 0, "Token estimation should return a positive number"
    except ValueError as e:
        if "<|endoftext|>" in str(e):
            pytest.fail("Special token <|endoftext|> should not raise ValueError")
        raise  # Re-raise if it is a different ValueError
