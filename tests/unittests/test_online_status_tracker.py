from io import StringIO
from unittest.mock import patch

from rich.console import Console

from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker


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
