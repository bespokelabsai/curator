from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from bespokelabs.curator.status_tracker import online_status_tracker as tracker_module
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker, TokenLimitStrategy
from bespokelabs.curator.types.token_usage import _TokenUsage


def test_online_status_tracker_display():
    """Test that the status tracker display works correctly."""
    # Create a string buffer to capture output
    output = StringIO()
    # Set width=200 and force_terminal=True to prevent truncation
    console = Console(file=output, width=200, force_terminal=True)

    # Create and configure the status tracker
    tracker = OnlineStatusTracker()
    tracker.model = "test-model"
    tracker.total_requests = 100

    # Start display
    tracker.start_tracker(console)

    # Simulate some progress
    tracker.total_requests = 100
    tracker.num_tasks_succeeded = 50
    tracker.num_tasks_failed = 5
    tracker.num_tasks_in_progress = 10
    tracker.total_prompt_tokens = 1000
    tracker.total_completion_tokens = 2000
    tracker.total_cost = 0.123
    tracker.update_stats(None, None)

    # Stop display
    tracker.stop_tracker()

    # Get the captured output
    captured = output.getvalue()

    # Verify key elements are present
    assert "test-model" in captured
    assert "100" in captured  # Total requests
    assert "50✓" in captured.replace(" ", "")  # Success count
    assert "5✗" in captured.replace(" ", "")  # Failed count
    assert "10⋯" in captured.replace(" ", "")  # In progress count
    assert "$0.123" in captured  # Cost
    assert "1000" in captured  # Input tokens
    assert "2000" in captured  # Output tokens


def test_online_status_tracker_final_stats():
    """Test that the final statistics table is formatted correctly."""
    output = StringIO()
    # Same here - set width and force_terminal
    console = Console(file=output, width=200, force_terminal=True)

    tracker = OnlineStatusTracker()
    tracker.model = "test-model"
    tracker.start_tracker(console)

    # Set some statistics
    tracker.num_tasks_succeeded = 50
    tracker.total_prompt_tokens = 1000
    tracker.total_completion_tokens = 2000
    tracker.total_cost = 0.123
    tracker.input_cost_per_million = 0.15
    tracker.output_cost_per_million = 0.60

    # Generate final stats
    tracker.stop_tracker()

    # Get the captured output
    captured = output.getvalue()

    # Verify the final statistics table
    assert "Final Curator Statistics" in captured
    assert "test-model" in captured
    assert "Total Tokens Used" in captured
    assert "1,000" in captured  # Input tokens
    assert "2,000" in captured  # Output tokens
    assert "$0.123" in captured  # Cost


def test_online_status_tracker_missing_cost_keys():
    """Test that missing cost keys in model pricing don't cause KeyError."""
    with patch("bespokelabs.curator.status_tracker.online_status_tracker.model_cost", {"test-model": {}}):
        tracker = OnlineStatusTracker()
        tracker.model = "test-model"
        # Should not raise KeyError and should set costs to None
        assert tracker.input_cost_per_million is None
        assert tracker.output_cost_per_million is None
        assert "N/A" in tracker.input_cost_str
        assert "N/A" in tracker.output_cost_str


def test_online_status_tracker_external_model_cost_error():
    """Test that external_model_cost errors are handled gracefully."""
    with patch("bespokelabs.curator.status_tracker.online_status_tracker.model_cost", {}):
        with patch("bespokelabs.curator.status_tracker.online_status_tracker.external_model_cost", side_effect=KeyError("Missing key")):
            tracker = OnlineStatusTracker()
            tracker.model = "unknown-model"
            # Should not raise KeyError and should set costs to None
            assert tracker.input_cost_per_million is None
            assert tracker.output_cost_per_million is None
            assert "N/A" in tracker.input_cost_str
            assert "N/A" in tracker.output_cost_str


def test_online_status_tracker_external_model_cost_missing_keys():
    """Test that missing keys in external_model_cost result don't cause KeyError."""
    with patch("bespokelabs.curator.status_tracker.online_status_tracker.model_cost", {}):
        with patch(
            "bespokelabs.curator.status_tracker.online_status_tracker.external_model_cost",
            return_value={"input_cost_per_token": None, "output_cost_per_token": None},
        ):
            tracker = OnlineStatusTracker()
            tracker.model = "unknown-model"
            # Should not raise KeyError and should set costs to None
            assert tracker.input_cost_per_million is None
            assert tracker.output_cost_per_million is None
            assert "N/A" in tracker.input_cost_str
            assert "N/A" in tracker.output_cost_str


@pytest.fixture
def capacity_clock(monkeypatch):
    """Control refill time without sleeping or changing the process-wide clock."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(tracker_module, "time", SimpleNamespace(time=lambda: clock.now))
    return clock


@pytest.mark.parametrize("estimate,required", [(90, 90), (100, 100), (150, 100)])
def test_combined_capacity_preserves_request_and_token_limits(capacity_clock, estimate, required):
    tracker = OnlineStatusTracker(model="gpt-4o-mini", max_requests_per_minute=60, max_tokens_per_minute=100, last_update_time=0)
    tokens = _TokenUsage(input=estimate)
    tracker.available_request_capacity = 1
    tracker.available_token_capacity = required - 1
    assert not tracker.has_capacity(tokens)
    tracker.available_token_capacity = required
    tracker.available_request_capacity = 0
    assert not tracker.has_capacity(tokens)
    tracker.available_request_capacity = 1
    assert tracker.has_capacity(tokens)


@pytest.mark.parametrize("input_tokens,output_tokens", [(50, 25), (100, 50), (150, 25), (50, 75), (150, 75)])
def test_separate_capacity_requires_each_bucket(capacity_clock, input_tokens, output_tokens):
    tracker = OnlineStatusTracker(
        model="gpt-4o-mini",
        token_limit_strategy=TokenLimitStrategy.seperate,
        max_requests_per_minute=60,
        max_tokens_per_minute=_TokenUsage(input=100, output=50),
        last_update_time=0,
    )
    tokens = _TokenUsage(input=input_tokens, output=output_tokens)
    required_input, required_output = min(input_tokens, 100), min(output_tokens, 50)
    tracker.available_token_capacity = _TokenUsage(input=required_input - 1, output=required_output)
    assert not tracker.has_capacity(tokens)
    tracker.available_token_capacity = _TokenUsage(input=required_input, output=required_output - 1)
    assert not tracker.has_capacity(tokens)
    tracker.available_token_capacity = _TokenUsage(input=required_input, output=required_output)
    tracker.available_request_capacity = 0
    assert not tracker.has_capacity(tokens)
    tracker.available_request_capacity = 1
    assert tracker.has_capacity(tokens)


@pytest.mark.parametrize("strategy", [TokenLimitStrategy.combined, TokenLimitStrategy.seperate])
@pytest.mark.parametrize("refund", [False, True])
def test_oversized_capacity_debt_recovers(capacity_clock, caplog, strategy, refund):
    separate = strategy == TokenLimitStrategy.seperate
    tracker = OnlineStatusTracker(
        model="gpt-4o-mini",
        token_limit_strategy=strategy,
        max_requests_per_minute=60,
        max_tokens_per_minute=_TokenUsage(input=100, output=50) if separate else 100,
        last_update_time=0,
    )
    tokens = _TokenUsage(input=150, output=75 if separate else 0)
    assert not tracker.has_capacity(tokens)
    assert not tracker.has_capacity(tokens)
    assert not caplog.records  # No warning on each capacity poll.
    capacity_clock.now = 60
    assert tracker.has_capacity(tokens)
    tracker.consume_capacity(tokens)
    assert "charging the full estimate" in caplog.text
    if separate:
        assert tracker.available_token_capacity.input == -50
        assert tracker.available_token_capacity.output == -25
    else:
        assert tracker.available_token_capacity == -50
    next_tokens = _TokenUsage(input=1, output=1)
    assert not tracker.has_capacity(next_tokens)
    assert not tracker.has_capacity(tokens)  # A retry must also wait for refill.

    if refund:
        tracker.free_capacity(used=_TokenUsage(input=10, output=5 if separate else 0), blocked=tokens)
        if separate:
            assert tracker.available_token_capacity.input == 90
            assert tracker.available_token_capacity.output == 45
        else:
            assert tracker.available_token_capacity == 90
    else:
        capacity_clock.now += 30
        assert not tracker.has_capacity(next_tokens)  # Paying the debt alone is insufficient.
        capacity_clock.now += 2
    assert tracker.has_capacity(next_tokens)


def test_unlimited_capacity_still_accepts_large_requests(capacity_clock):
    tracker = OnlineStatusTracker(model="gpt-4o-mini", max_requests_per_minute=None, max_tokens_per_minute=None, last_update_time=0)
    assert tracker.has_capacity(_TokenUsage(input=1_000_000, output=1_000_000))
