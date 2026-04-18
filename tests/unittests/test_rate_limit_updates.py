from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.anthropic_online_request_processor import AnthropicOnlineRequestProcessor
from bespokelabs.curator.request_processor.online.openai_online_request_processor import OpenAIOnlineRequestProcessor
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker, TokenLimitStrategy
from bespokelabs.curator.types.generic_response import _TokenUsage


def _anthropic_processor() -> AnthropicOnlineRequestProcessor:
    return AnthropicOnlineRequestProcessor(OnlineRequestProcessorConfig(model="claude-3-5-sonnet-latest"))


def _separate_tracker(input_tpm: int = 100_000, output_tpm: int = 40_000, rpm: int = 200) -> OnlineStatusTracker:
    return OnlineStatusTracker(
        token_limit_strategy=TokenLimitStrategy.seperate,
        max_requests_per_minute=rpm,
        max_tokens_per_minute=_TokenUsage(input=input_tpm, output=output_tpm),
    )


def test_anthropic_parse_returns_zeros_when_headers_absent():
    """A proxy/base_url that strips `anthropic-ratelimit-*` headers should surface as zeros."""
    processor = _anthropic_processor()
    rpm, tpm = processor.parse_header_based_rate_limits({})
    assert rpm == 0
    assert tpm == _TokenUsage(input=0, output=0)


def test_anthropic_parse_returns_header_values_when_present():
    processor = _anthropic_processor()
    headers = {
        "anthropic-ratelimit-requests-limit": "4000",
        "anthropic-ratelimit-input-tokens-limit": "400000",
        "anthropic-ratelimit-output-tokens-limit": "80000",
    }
    rpm, tpm = processor.parse_header_based_rate_limits(headers)
    assert rpm == 4000
    assert tpm == _TokenUsage(input=400_000, output=80_000)


def test_maybe_update_rate_limits_noop_when_all_zero():
    """No headers present → leave tracker on its conservative defaults."""
    processor = _anthropic_processor()
    tracker = _separate_tracker()

    processor._maybe_update_rate_limits(rpm=0, tpm=_TokenUsage(input=0, output=0), status_tracker=tracker)

    assert processor.header_based_max_requests_per_minute is None
    assert processor.header_based_max_tokens_per_minute is None
    assert tracker.max_requests_per_minute == 200
    assert tracker.max_tokens_per_minute == _TokenUsage(input=100_000, output=40_000)


def test_maybe_update_rate_limits_preserves_output_when_only_input_present():
    """Partial header set must not zero the axis we have no fresh signal on."""
    processor = _anthropic_processor()
    tracker = _separate_tracker(input_tpm=100_000, output_tpm=40_000)

    processor._maybe_update_rate_limits(
        rpm=4000,
        tpm=_TokenUsage(input=400_000, output=0),
        status_tracker=tracker,
    )

    assert tracker.max_requests_per_minute == 4000
    assert tracker.max_tokens_per_minute.input == 400_000
    assert tracker.max_tokens_per_minute.output == 40_000  # preserved
    assert tracker.available_token_capacity.output == 40_000  # not zeroed


def test_maybe_update_rate_limits_preserves_input_when_only_output_present():
    processor = _anthropic_processor()
    tracker = _separate_tracker(input_tpm=100_000, output_tpm=40_000)

    processor._maybe_update_rate_limits(
        rpm=0,
        tpm=_TokenUsage(input=0, output=80_000),
        status_tracker=tracker,
    )

    assert tracker.max_tokens_per_minute.input == 100_000  # preserved
    assert tracker.max_tokens_per_minute.output == 80_000
    assert tracker.available_token_capacity.input == 100_000


def test_maybe_update_rate_limits_full_update_separate():
    processor = _anthropic_processor()
    tracker = _separate_tracker(input_tpm=100_000, output_tpm=40_000)

    processor._maybe_update_rate_limits(
        rpm=4000,
        tpm=_TokenUsage(input=400_000, output=80_000),
        status_tracker=tracker,
    )

    assert processor.header_based_max_requests_per_minute == 4000
    assert processor.header_based_max_tokens_per_minute == _TokenUsage(input=400_000, output=80_000)
    assert tracker.max_tokens_per_minute == _TokenUsage(input=400_000, output=80_000)
    # Capacity expands by the delta on each axis.
    assert tracker.available_token_capacity.input == 400_000
    assert tracker.available_token_capacity.output == 80_000


def test_maybe_update_rate_limits_respects_manual_override():
    """Manual overrides are not clobbered by header-based updates."""
    config = OnlineRequestProcessorConfig(
        model="claude-3-5-sonnet-latest",
        max_requests_per_minute=10,
        max_input_tokens_per_minute=1000,
        max_output_tokens_per_minute=500,
    )
    processor = AnthropicOnlineRequestProcessor(config)
    tracker = _separate_tracker(input_tpm=1000, output_tpm=500, rpm=10)

    processor._maybe_update_rate_limits(
        rpm=4000,
        tpm=_TokenUsage(input=400_000, output=80_000),
        status_tracker=tracker,
    )

    # Cache still records the header values for observability.
    assert processor.header_based_max_requests_per_minute == 4000
    # But tracker keeps the manual limits.
    assert tracker.max_requests_per_minute == 10
    assert tracker.max_tokens_per_minute == _TokenUsage(input=1000, output=500)


def test_openai_combined_noop_when_tpm_zero():
    """OpenAI (combined strategy) should skip the update when header values are missing."""
    processor = OpenAIOnlineRequestProcessor(OnlineRequestProcessorConfig(model="gpt-4"))
    tracker = OnlineStatusTracker(
        token_limit_strategy=TokenLimitStrategy.combined,
        max_requests_per_minute=200,
        max_tokens_per_minute=100_000,
    )

    processor._maybe_update_rate_limits(rpm=0, tpm=0, status_tracker=tracker)

    assert processor.header_based_max_requests_per_minute is None
    assert processor.header_based_max_tokens_per_minute is None
    assert tracker.max_requests_per_minute == 200
    assert tracker.max_tokens_per_minute == 100_000
