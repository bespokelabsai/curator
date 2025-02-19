import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.bedrock_online_request_processor import BedrockOnlineRequestProcessor


def test_model_validation():
    """Test model validation during initialization."""
    # Valid model and region
    config = OnlineRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1")
    processor = BedrockOnlineRequestProcessor(config)
    assert processor.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
    assert processor.region == "us-east-1"

    # Invalid model
    with pytest.raises(ValueError, match="is not supported"):
        config = OnlineRequestProcessorConfig(model="invalid-model")
        BedrockOnlineRequestProcessor(config)

    # Invalid region
    with pytest.raises(ValueError, match="is not supported in region"):
        config = OnlineRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", region="invalid-region")
        BedrockOnlineRequestProcessor(config)


def test_token_estimation():
    """Test token estimation for different content types."""
    config = OnlineRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0")
    processor = BedrockOnlineRequestProcessor(config)

    # Test text content
    messages = [{"role": "user", "content": "Hello world"}]
    tokens = processor.estimate_total_tokens(messages)
    assert tokens > 0

    # Test multimodal content
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello world"}, {"type": "image", "image_url": "test.jpg"}]}]
    tokens = processor.estimate_total_tokens(messages)
    assert tokens > 0


def test_request_formatting():
    """Test request formatting for different model providers."""
    config = OnlineRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0")
    processor = BedrockOnlineRequestProcessor(config)

    # Test Anthropic format
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]
    request = processor.create_api_specific_request_online({"messages": messages})
    assert "anthropic_version" in request["body"]
    assert "system" in request["body"]
    assert "messages" in request["body"]

    # Test with different model provider
    config = OnlineRequestProcessorConfig(model="meta.llama3-1-70b-instruct-v1:0")
    processor = BedrockOnlineRequestProcessor(config)
    request = processor.create_api_specific_request_online({"messages": messages})
    assert "prompt" in request["body"]
    assert "<|begin_of_text|>" in request["body"]["prompt"]


@pytest.mark.asyncio
@patch("boto3.client")
async def test_api_call_error_handling(mock_boto3):
    """Test error handling during API calls."""
    config = OnlineRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0")
    processor = BedrockOnlineRequestProcessor(config)

    # Mock boto3 client
    mock_client = Mock()
    mock_boto3.return_value = mock_client

    # Test rate limit error
    error_response = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
    mock_client.invoke_model.side_effect = ClientError(error_response, "invoke_model")

    with pytest.raises(ClientError) as exc_info:
        await processor.call_single_request(
            request=Mock(to_dict=lambda: {"modelId": "test", "contentType": "application/json", "accept": "application/json", "body": {}}),
            session=AsyncMock(),
            status_tracker=Mock(),
        )
    assert "Rate exceeded" in str(exc_info.value)

    # Test successful response
    mock_client.invoke_model.side_effect = None
    mock_client.invoke_model.return_value = {"body": Mock(read=lambda: json.dumps({"completion": "Test response", "stop_reason": "stop"}).encode())}

    response = await processor.call_single_request(
        request=Mock(
            to_dict=lambda: {
                "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                "contentType": "application/json",
                "accept": "application/json",
                "body": {"prompt": "test"},
            },
            generic_request={"messages": [{"role": "user", "content": "test"}]},
        ),
        session=AsyncMock(),
        status_tracker=Mock(),
    )

    assert response.response_message["choices"][0]["message"]["content"] == "Test response"
    assert response.token_usage is not None
    assert response.response_cost is not None


@pytest.mark.asyncio
async def test_response_parsing():
    """Test response parsing for different model providers."""
    config = OnlineRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0")
    processor = BedrockOnlineRequestProcessor(config)

    # Test Anthropic response
    response = {"completion": "Hello, I am Claude", "stop_reason": "stop"}
    parsed = processor.parse_api_specific_response(response)
    assert parsed["choices"][0]["message"]["content"] == "Hello, I am Claude"

    # Test Meta response
    config = OnlineRequestProcessorConfig(model="meta.llama3-1-70b-instruct-v1:0")
    processor = BedrockOnlineRequestProcessor(config)
    response = {"generation": "Hello, I am Llama", "stop_reason": "stop"}
    parsed = processor.parse_api_specific_response(response)
    assert parsed["choices"][0]["message"]["content"] == "Hello, I am Llama"

    # Test Amazon Nova response
    config = OnlineRequestProcessorConfig(model="us.amazon.nova-lite-v1:0")
    processor = BedrockOnlineRequestProcessor(config)
    response = {"outputText": "Hello, I am Nova", "stop_reason": "stop"}
    parsed = processor.parse_api_specific_response(response)
    assert parsed["choices"][0]["message"]["content"] == "Hello, I am Nova"

    # Test Mistral response
    config = OnlineRequestProcessorConfig(model="mistral.mistral-large-24-07")
    processor = BedrockOnlineRequestProcessor(config)
    response = {"text": "Hello, I am Mistral", "stop_reason": "stop"}
    parsed = processor.parse_api_specific_response(response)
    assert parsed["choices"][0]["message"]["content"] == "Hello, I am Mistral"
