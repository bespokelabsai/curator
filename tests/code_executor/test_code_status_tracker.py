import time
from io import StringIO

import pytest
from rich.console import Console

from bespokelabs.curator.code_executor.tracker import CodeExecutionStatusTracker


def test_code_execution_tracker_initialization():
    """Test that the tracker initializes with correct default values."""
    tracker = CodeExecutionStatusTracker()

    assert tracker.num_tasks_started == 0
    assert tracker.num_tasks_in_progress == 0
    assert tracker.num_tasks_succeeded == 0
    assert tracker.num_tasks_failed == 0
    assert tracker.num_tasks_already_completed == 0
    assert tracker.num_execution_errors == 0
    assert tracker.num_other_errors == 0
    assert tracker.num_rate_limit_errors == 0
    assert tracker.time_of_last_rate_limit_error == 0.0
    assert tracker.total_requests == 0
    assert hasattr(tracker, "start_time")


def test_code_execution_tracker_display():
    """Test that the status tracker display works correctly."""
    # Create a string buffer to capture output
    output = StringIO()
    # Set width=200 and force_terminal=True to prevent truncation
    console = Console(file=output, width=200, force_terminal=True)

    # Create and configure the status tracker
    tracker = CodeExecutionStatusTracker()
    tracker.total_requests = 100

    # Start display
    tracker.start_tracker(console)

    # Simulate some progress
    tracker.num_tasks_started = 60
    tracker.num_tasks_succeeded = 50
    tracker.num_tasks_failed = 5
    tracker.num_tasks_in_progress = 5
    tracker.num_tasks_already_completed = 10
    tracker.num_execution_errors = 3
    tracker.num_other_errors = 2

    # Update display
    tracker.update_stats()

    # Stop display
    tracker.stop_tracker()

    # Get the captured output
    captured = output.getvalue()

    # Verify key elements are present
    assert "Processing Tasks" in captured
    assert "50✓" in captured.replace(" ", "")  # Success count
    assert "5✗" in captured.replace(" ", "")  # Failed count
    assert "5⋯" in captured.replace(" ", "")  # In progress count
    assert "60" in captured  # Started count
    assert "100" in captured  # Total requests


def test_code_execution_tracker_final_stats():
    """Test that the final statistics table is formatted correctly."""
    output = StringIO()
    console = Console(file=output, width=200, force_terminal=True)

    tracker = CodeExecutionStatusTracker()
    tracker.start_tracker(console)

    # Set some statistics
    tracker.total_requests = 100
    tracker.num_tasks_started = 60
    tracker.num_tasks_succeeded = 50
    tracker.num_tasks_failed = 5
    tracker.num_tasks_already_completed = 10
    tracker.num_execution_errors = 3
    tracker.num_other_errors = 2

    # Generate final stats
    tracker.stop_tracker()

    # Get the captured output
    captured = output.getvalue()

    # Verify the final statistics table
    assert "Final Statistics" in captured
    assert "Total Requests" in captured
    assert "100" in captured  # Total requests
    assert "50" in captured  # Successful tasks
    assert "5" in captured  # Failed tasks
    assert "10" in captured  # Already completed
    assert "3" in captured  # Execution errors
    assert "2" in captured  # Other errors


def test_code_execution_tracker_str_representation():
    """Test the string representation of the tracker."""
    tracker = CodeExecutionStatusTracker()
    tracker.num_tasks_started = 60
    tracker.num_tasks_succeeded = 50
    tracker.num_tasks_failed = 5
    tracker.num_tasks_in_progress = 5
    tracker.num_tasks_already_completed = 10
    tracker.num_execution_errors = 3
    tracker.num_other_errors = 2

    str_rep = str(tracker)

    assert "Tasks - Started: 60" in str_rep
    assert "In Progress: 5" in str_rep
    assert "Succeeded: 50" in str_rep
    assert "Failed: 5" in str_rep
    assert "Already Completed: 10" in str_rep
    assert "Errors - Execution: 3" in str_rep
    assert "Other: 2" in str_rep


@pytest.mark.asyncio
async def test_code_execution_tracker_capacity_management():
    """Test the capacity management functions."""
    tracker = CodeExecutionStatusTracker()
    tracker.max_requests_per_minute = 10

    # Test initial capacity
    assert tracker.has_capacity() is True
    assert tracker.num_tasks_in_progress == 0

    # Test consuming capacity
    tracker.consume_capacity()
    assert tracker.num_tasks_in_progress == 1

    # Test freeing capacity
    tracker.free_capacity()
    assert tracker.num_tasks_in_progress == 0

    # Test capacity limit
    for _ in range(10):
        tracker.consume_capacity()
    assert tracker.has_capacity() is False


def test_code_execution_tracker_disabled():
    """Test that the tracker can be disabled via environment variable."""
    import os

    os.environ["BESPOKE_CURATOR_TRACKER_DISABLED"] = "1"

    output = StringIO()
    console = Console(file=output, width=200, force_terminal=True)

    tracker = CodeExecutionStatusTracker()
    tracker.start_tracker(console)
    tracker.update_stats()
    tracker.stop_tracker()

    captured = output.getvalue()
    assert captured == ""

    # Clean up
    os.environ["BESPOKE_CURATOR_TRACKER_DISABLED"] = "0"


def test_code_execution_tracker_rpm_calculation():
    """Test the requests per minute calculation."""
    output = StringIO()
    console = Console(file=output, width=200, force_terminal=True)

    tracker = CodeExecutionStatusTracker()
    tracker.start_tracker(console)

    # Simulate some completed requests
    tracker.num_tasks_succeeded = 30

    # Force the elapsed time to be exactly 1 minute
    tracker.start_time = time.time() - 60

    tracker.update_stats()
    tracker.stop_tracker()

    captured = output.getvalue()
    assert "30.0" in captured  # RPM should be 30.0


def test_code_execution_tracker_progress_updates():
    """Test that progress updates correctly reflect task state changes."""
    output = StringIO()
    console = Console(file=output, width=200, force_terminal=True, color_system=None)  # Disable colors

    tracker = CodeExecutionStatusTracker()
    tracker.total_requests = 100
    tracker.start_tracker(console)

    # Test progressive updates
    test_states = [
        {"started": 20, "succeeded": 15, "failed": 3, "in_progress": 2},
        {"started": 40, "succeeded": 30, "failed": 5, "in_progress": 5},
        {"started": 60, "succeeded": 45, "failed": 8, "in_progress": 7},
    ]

    for state in test_states:
        tracker.num_tasks_started = state["started"]
        tracker.num_tasks_succeeded = state["succeeded"]
        tracker.num_tasks_failed = state["failed"]
        tracker.num_tasks_in_progress = state["in_progress"]
        tracker.update_stats()

        # Force the progress display to refresh
        tracker._progress.refresh()

        # Get the captured output and verify the numbers are present
        captured = output.getvalue()
        print(f"Captured output: {repr(captured)}")  # Debug print
        assert str(state["succeeded"]) in captured
        assert str(state["failed"]) in captured
        assert str(state["in_progress"]) in captured
        assert "Success" in captured
        assert "Failed" in captured
        assert "In Progress" in captured

        # Clear the output buffer for the next iteration
        output.seek(0)
        output.truncate()

    tracker.stop_tracker()


def test_code_execution_tracker_error_handling():
    """Test error counting and tracking."""
    tracker = CodeExecutionStatusTracker()

    # Test execution errors
    tracker.num_execution_errors += 1
    assert tracker.num_execution_errors == 1

    # Test other errors
    tracker.num_other_errors += 1
    assert tracker.num_other_errors == 1

    # Test rate limit errors
    tracker.num_rate_limit_errors += 1
    assert tracker.num_rate_limit_errors == 1

    # Test time of last rate limit error
    current_time = time.time()
    tracker.time_of_last_rate_limit_error = current_time
    assert tracker.time_of_last_rate_limit_error == current_time
