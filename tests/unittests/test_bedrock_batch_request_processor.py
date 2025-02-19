import json
from unittest.mock import Mock, patch

import pytest

from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch
from bespokelabs.curator.types.generic_request import GenericRequest


def test_model_validation():
    """Test model validation during initialization."""
    # Valid model and region
    config = BatchRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)
    assert processor.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
    assert processor.region == "us-east-1"

    # Invalid model
    with pytest.raises(ValueError, match="is not supported"):
        config = BatchRequestProcessorConfig(model="invalid-model", bucket_name="test-bucket", role_arn="test-role")
        BedrockBatchRequestProcessor(config)

    # Invalid region
    with pytest.raises(ValueError, match="is not supported in region"):
        config = BatchRequestProcessorConfig(
            model="anthropic.claude-3-haiku-20240307-v1:0", region="invalid-region", bucket_name="test-bucket", role_arn="test-role"
        )
        BedrockBatchRequestProcessor(config)


def test_shard_size_calculation():
    """Test optimal shard size calculation."""
    config = BatchRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)

    # Test with small requests
    requests = [{"messages": [{"role": "user", "content": "Hello"}]}, {"messages": [{"role": "user", "content": "World"}]}]
    shard_size = processor._calculate_optimal_shard_size(requests)
    assert shard_size >= processor.model_limits["min_batch_size"]
    assert shard_size <= processor.model_limits["max_batch_size"]

    # Test with large requests that exceed token limits
    large_content = "x" * (processor.model_limits["max_input_tokens"] * 4)  # 4 chars per token
    requests = [{"messages": [{"role": "user", "content": large_content}]}]
    with pytest.raises(ValueError, match="exceeding model's token limit"):
        processor._calculate_optimal_shard_size(requests)


def test_request_formatting():
    """Test batch request formatting for different model providers."""
    config = BatchRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)

    # Test Anthropic format
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]
    generic_request = GenericRequest(messages=messages, original_row_idx=0, original_row={})
    request = processor.create_api_specific_request_batch(generic_request)
    assert "custom_id" in request  # Required for response mapping
    assert "anthropic_version" in request["body"]
    assert "system" in request["body"]
    assert "messages" in request["body"]

    # Test with different model provider
    config = BatchRequestProcessorConfig(model="meta.llama3-1-70b-instruct-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)
    request = processor.create_api_specific_request_batch(generic_request)
    assert "custom_id" in request
    assert "prompt" in request["body"]
    assert "<|begin_of_text|>" in request["body"]["prompt"]


@pytest.mark.asyncio
@patch("boto3.client")
async def test_batch_submission(mock_boto3):
    """Test batch job submission and S3 interactions."""
    config = BatchRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)

    # Mock S3 and Bedrock clients
    mock_s3 = Mock()
    mock_bedrock = Mock()
    mock_quotas = Mock()
    mock_boto3.side_effect = lambda service, **kwargs: {"s3": mock_s3, "bedrock": mock_bedrock, "service-quotas": mock_quotas}[service]

    # Mock successful job creation
    mock_bedrock.create_model_inference_job.return_value = {"jobId": "test-job", "status": "IN_PROGRESS", "creationTime": "2024-02-18T12:00:00Z"}

    # Test batch submission
    requests = [
        GenericRequest(messages=[{"role": "user", "content": "Hello"}], original_row_idx=0, original_row={}),
        GenericRequest(messages=[{"role": "user", "content": "World"}], original_row_idx=1, original_row={}),
    ]
    await processor.submit_batch(requests)

    # Verify S3 upload
    mock_s3.put_object.assert_called_once()
    assert mock_s3.put_object.call_args[1]["Bucket"] == "test-bucket"
    assert "input/" in mock_s3.put_object.call_args[1]["Key"]

    # Verify job creation
    mock_bedrock.create_model_inference_job.assert_called_once()
    job_args = mock_bedrock.create_model_inference_job.call_args[1]
    assert job_args["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert job_args["roleArn"] == "test-role"
    assert "CreatedBy" in job_args["tags"]


@pytest.mark.asyncio
@patch("boto3.client")
async def test_batch_operations(mock_boto3):
    """Test batch operations like retrieve, cancel, and download."""
    config = BatchRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)

    # Mock clients
    mock_s3 = Mock()
    mock_bedrock = Mock()
    mock_quotas = Mock()
    mock_boto3.side_effect = lambda service, **kwargs: {"s3": mock_s3, "bedrock": mock_bedrock, "service-quotas": mock_quotas}[service]

    # Test retrieve batch
    mock_bedrock.get_model_inference_job.return_value = {"jobId": "test-job", "status": "COMPLETED", "creationTime": "2024-02-18T12:00:00Z"}

    batch = GenericBatch(id="test-job", status="IN_PROGRESS", model="test-model", created="2024-02-18T12:00:00Z")

    updated_batch = await processor.retrieve_batch(batch)
    assert updated_batch["status"] == "COMPLETED"

    # Test cancel batch
    mock_bedrock.stop_model_inference_job.return_value = {"jobId": "test-job", "status": "STOPPED", "creationTime": "2024-02-18T12:00:00Z"}

    cancelled_batch = await processor.cancel_batch(batch)
    assert cancelled_batch["status"] == "STOPPED"

    # Test download batch
    mock_s3.list_objects_v2.return_value = {"Contents": [{"Key": "output/test-job/results.jsonl"}]}
    mock_s3.get_object.return_value = {"Body": Mock(read=lambda: json.dumps({"completion": "Test response", "stop_reason": "stop"}).encode())}

    results = await processor.download_batch(batch)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "completion" in results[0]


def test_response_parsing():
    """Test batch response parsing for different model providers."""
    config = BatchRequestProcessorConfig(model="anthropic.claude-3-haiku-20240307-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)

    # Test Anthropic response
    raw_response = {"completion": "Hello, I am Claude", "stop_reason": "stop"}
    generic_request = GenericRequest(messages=[{"role": "user", "content": "Hello"}], original_row_idx=0, original_row={})
    batch = GenericBatch(id="test-job", status="COMPLETED", model="test-model", created="2024-02-18T12:00:00Z")

    response = processor.parse_api_specific_response(raw_response, generic_request, batch)
    assert response.response_message["choices"][0]["message"]["content"] == "Hello, I am Claude"
    assert response.token_usage is not None
    assert response.response_cost is not None

    # Test Meta response
    config = BatchRequestProcessorConfig(model="meta.llama3-1-70b-instruct-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)
    raw_response = {"generation": "Hello, I am Llama", "stop_reason": "stop"}
    response = processor.parse_api_specific_response(raw_response, generic_request, batch)
    assert response.response_message["choices"][0]["message"]["content"] == "Hello, I am Llama"

    # Test Amazon Nova response
    config = BatchRequestProcessorConfig(model="us.amazon.nova-lite-v1:0", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)
    raw_response = {"outputText": "Hello, I am Nova", "stop_reason": "stop"}
    response = processor.parse_api_specific_response(raw_response, generic_request, batch)
    assert response.response_message["choices"][0]["message"]["content"] == "Hello, I am Nova"

    # Test Mistral response
    config = BatchRequestProcessorConfig(model="mistral.mistral-large-24-07", bucket_name="test-bucket", role_arn="test-role")
    processor = BedrockBatchRequestProcessor(config)
    raw_response = {"text": "Hello, I am Mistral", "stop_reason": "stop"}
    response = processor.parse_api_specific_response(raw_response, generic_request, batch)
    assert response.response_message["choices"][0]["message"]["content"] == "Hello, I am Mistral"
