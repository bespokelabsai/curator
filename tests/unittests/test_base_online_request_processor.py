import pytest

from bespokelabs.curator.request_processor.online.anthropic_online_request_processor import AnthropicOnlineRequestProcessor


@pytest.mark.asyncio
async def test_aiohttp_connector_trusts_env_proxy():
    """Ensure the shared aiohttp session honors HTTP(S)_PROXY/ALL_PROXY env vars.

    aiohttp.ClientSession ignores proxy environment variables unless trust_env=True
    is set explicitly, unlike requests/httpx. Without it, requests (e.g. to the
    Anthropic API) silently bypass any configured system proxy. See issue #699.

    aiohttp_connector doesn't touch instance state, so we bypass __init__ (which
    makes a live network call to fetch header-based rate limits) via object.__new__.
    """
    processor = object.__new__(AnthropicOnlineRequestProcessor)

    async with processor.aiohttp_connector(tcp_limit=1) as session:
        assert session.trust_env is True
