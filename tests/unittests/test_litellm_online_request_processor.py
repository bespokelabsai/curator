import litellm

from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import TokenLimitStrategy
from bespokelabs.curator.request_processor.online.litellm_online_request_processor import LiteLLMOnlineRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.prompt import File, _MultiModalPrompt
from bespokelabs.curator.types.token_usage import _TokenUsage


def test_multimodal_support_falls_back_for_claude_aliases(monkeypatch):
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)

    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))

    assert processor._multimodal_prompt_supported is True


def test_multimodal_support_does_not_fallback_for_non_claude_models(monkeypatch):
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)

    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="openai/gpt-4o-mini"))

    assert processor._multimodal_prompt_supported is False


def test_requests_to_responses_uses_resolved_token_limit_strategy(monkeypatch):
    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))
    processor.prompt_formatter = type("PromptFormatter", (), {"model_name": "anthropic/claude-sonnet-4-6"})()
    processor.total_requests = 0

    def fake_rate_limits():
        processor.token_limit_strategy = TokenLimitStrategy.seperate
        return 20000, _TokenUsage(input=10000000, output=2000000)

    monkeypatch.setattr(processor, "get_header_based_rate_limits", fake_rate_limits)

    processor.requests_to_responses([])

    assert processor.tracker.token_limit_strategy == TokenLimitStrategy.seperate
    assert processor.tracker.max_tokens_per_minute == _TokenUsage(input=10000000, output=2000000)


def test_anthropic_file_prompts_use_document_blocks_and_pdf_beta_header():
    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))
    request = GenericRequest(
        model="anthropic/claude-sonnet-4-6",
        messages=[
            {
                "role": "user",
                "content": _MultiModalPrompt(texts=["Describe the pdf"], files=[File(url="https://example.com/sample.pdf")]).model_dump(),
            }
        ],
        original_row={"pdf": "https://example.com/sample.pdf", "text": "Describe the pdf"},
        original_row_idx=0,
        is_multimodal_prompt=True,
    )

    request = processor._unpack_multimodal(request)
    api_request = processor.create_api_specific_request_online(request)

    assert api_request["messages"][0]["content"][1]["type"] == "document"
    assert api_request["messages"][0]["content"][1]["source"] == {
        "type": "url",
        "url": "https://example.com/sample.pdf",
    }
    assert api_request["extra_headers"]["anthropic-beta"] == "pdfs-2024-09-25"


def test_estimate_total_tokens_falls_back_for_anthropic_document_prompts():
    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))

    tokens = processor.estimate_total_tokens(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the pdf"},
                    {"type": "document", "source": {"type": "url", "url": "https://example.com/sample.pdf"}},
                ],
            }
        ]
    )

    assert tokens.input >= 2048
