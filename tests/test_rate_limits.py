import pytest
from unittest.mock import patch, MagicMock
from bespokelabs.curator.request_processor.online.litellm_online_request_processor import LiteLLMOnlineRequestProcessor
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig

class TestRateLimits:
    @patch('litellm.completion')
    @patch.object(LiteLLMOnlineRequestProcessor, 'test_call')
    def test_anthropic_rate_limits(self, mock_test_call, mock_completion):
        """Test that Anthropic-specific rate limits are correctly parsed."""
        # Setup mock completion response for initialization
        mock_completion_response = MagicMock()
        mock_completion_response._hidden_params = MagicMock()
        mock_completion.return_value = mock_completion_response

        # Setup initial test_call response for initialization
        mock_test_call.return_value = {
            "x-ratelimit-limit-requests": "5000",
            "x-ratelimit-limit-tokens": "480000"
        }

        config = OnlineRequestProcessorConfig(model="claude-3-opus-20240229")
        processor = LiteLLMOnlineRequestProcessor(config)

        # Test case 1: When Anthropic-specific header is present
        mock_test_call.return_value = {
            "x-ratelimit-limit-requests": "5000",
            "x-ratelimit-limit-tokens": "480000",  # Combined limit
            "llm_provider-anthropic-ratelimit-output-tokens-limit": "80000"  # Output limit
        }
        rpm, tpm = processor.get_header_based_rate_limits()
        assert rpm == 5000, "Request limit should be 5000"
        assert tpm == 80000, "Token limit should use Anthropic output limit (80k)"

        # Test case 2: When Anthropic-specific header is not present
        mock_test_call.return_value = {
            "x-ratelimit-limit-requests": "5000",
            "x-ratelimit-limit-tokens": "480000"
        }
        rpm, tpm = processor.get_header_based_rate_limits()
        assert rpm == 5000, "Request limit should be 5000"
        assert tpm == 480000, "Token limit should fall back to combined limit (480k)"

        # Test case 3: When no headers are present
        mock_test_call.return_value = {}
        rpm, tpm = processor.get_header_based_rate_limits()
        assert rpm == 0, "Request limit should default to 0"
        assert tpm == 0, "Token limit should default to 0"
