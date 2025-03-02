import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest
from bespokelabs.curator.request_processor.online.openai_client_online_request_processor import OpenAIClientOnlineRequestProcessor
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest


class TestOpenAIClientMultiple(unittest.TestCase):
    """Test the multiple clients feature of OpenAIClientOnlineRequestProcessor."""

    @patch("openai.AsyncOpenAI")
    def test_multiple_clients_initialization(self, mock_async_openai):
        """Test that multiple clients are initialized correctly."""
        config = OnlineRequestProcessorConfig(model="test-model", num_clients=3)

        processor = OpenAIClientOnlineRequestProcessor(config)

        # Check that the correct number of clients were created
        self.assertEqual(len(processor.clients), 3)
        self.assertEqual(mock_async_openai.call_count, 3)

    @patch("aiohttp.ClientSession")
    @patch("openai.AsyncOpenAI")
    @patch("datetime.datetime")
    async def test_round_robin_client_selection(self, mock_datetime, mock_async_openai, mock_session):
        """Test that clients are selected in round-robin fashion."""
        # Set up the mock OpenAI client instances
        mock_clients = []
        for i in range(3):
            client = MagicMock()
            client.chat = MagicMock()
            client.chat.completions = MagicMock()
            client.chat.completions.create = AsyncMock()

            # Setup response
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = f"Response from client {i}"
            response.choices[0].finish_reason = "stop"
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 10
            response.usage.total_tokens = 20
            response.model_dump.return_value = {"model": "test-model"}

            client.chat.completions.create.return_value = response
            mock_clients.append(client)

        # Make the AsyncOpenAI constructor return our mock clients
        mock_async_openai.side_effect = mock_clients

        # Create a processor with 3 clients
        config = OnlineRequestProcessorConfig(model="test-model", num_clients=3)

        processor = OpenAIClientOnlineRequestProcessor(config)

        # Create test data
        generic_request = GenericRequest(messages=[], model="test-model")
        api_request = APIRequest(generic_request=generic_request, api_specific_request={"model": "test-model", "messages": []})
        status_tracker = OnlineStatusTracker(config)

        # Test that each client is used in turn
        for i in range(3):
            await processor.call_single_request(api_request, mock_session, status_tracker)
            # Check that the i-th client was used
            mock_clients[i].chat.completions.create.assert_called_once()

        # Reset the call counts
        for client in mock_clients:
            client.chat.completions.create.reset_mock()

        # The 4th call should use the first client again (round-robin)
        await processor.call_single_request(api_request, mock_session, status_tracker)
        mock_clients[0].chat.completions.create.assert_called_once()
