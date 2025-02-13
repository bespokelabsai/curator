import datetime
from io import StringIO

from rich.console import Console

from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchRequestCounts, GenericBatchStatus
from bespokelabs.curator.types.generic_response import TokenUsage


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
    tracker.num_tasks_succeeded = 50
    tracker.num_tasks_failed = 5
    tracker.num_tasks_in_progress = 10
    tracker.total_prompt_tokens = 1000
    tracker.total_completion_tokens = 2000
    tracker.total_cost = 0.123

    # Update display
    tracker.update_stats(None, None)

    # Stop display
    tracker.stop_tracker()

    # Get the captured output
    captured = output.getvalue()

    # Verify key elements are present
    print(captured)
    assert "test-model" in captured
    assert "50✓" in captured.replace(" ", "")  # Success count
    assert "5✗" in captured.replace(" ", "")  # Failed count
    assert "10⋯" in captured.replace(" ", "")  # In progress count
    assert "$0.123" in captured  # Cost
    assert "1000" in captured  # Input tokens
    assert "2000" in captured  # Output tokens


def test_status_tracker_final_stats():
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


def test_batch_status_tracker_display():
    """Test that the batch status tracker display works correctly."""
    # Create a string buffer to capture output
    output = StringIO()
    # Set width=200 and force_terminal=True to prevent truncation
    console = Console(file=output, width=200, force_terminal=True)

    # Create and configure the status tracker
    tracker = BatchStatusTracker()
    tracker.model = "test-model"
    tracker.input_cost_per_million = 0.15
    tracker.output_cost_per_million = 0.60

    # Start display
    tracker.start_tracker(console)

    # Create a test batch
    batch = GenericBatch(
        id="test-batch-1",
        request_file="test.jsonl",
        status=GenericBatchStatus.SUBMITTED,
        created_at=datetime.datetime.now(),
        finished_at=None,
        api_key_suffix="1234",
        raw_status="submitted",
        raw_batch={},
        request_counts=GenericBatchRequestCounts(total=20, succeeded=15, failed=5, raw_request_counts_object={"total": 20, "succeeded": 15, "failed": 5}),
    )

    # Simulate batch progress
    tracker.mark_as_submitted(batch, n_requests=20)

    # Update token usage and cost
    token_usage = TokenUsage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)
    tracker.update_token_and_cost(token_usage, cost=0.123)

    # Mark batch as downloaded
    tracker.mark_as_finished(batch)
    tracker.mark_as_downloaded(batch)  # Add this to move it to downloaded state

    # Stop display
    tracker.stop_tracker()

    # Get the captured output
    captured = output.getvalue()

    # Verify key elements are present
    print(captured)
    assert "test-model" in captured
    assert "1✓" in captured.replace(" ", "")  # Downloaded batch count
    assert "15✓" in captured.replace(" ", "")  # Success count
    assert "5✗" in captured.replace(" ", "")  # Failed count
    assert "$0.123" in captured  # Cost


def test_batch_status_tracker_final_stats():
    """Test that the final statistics table for batch tracker is formatted correctly."""
    output = StringIO()
    console = Console(file=output, width=200, force_terminal=True)

    tracker = BatchStatusTracker()
    tracker.model = "test-model"
    tracker.input_cost_per_million = 0.15
    tracker.output_cost_per_million = 0.60

    tracker.start_tracker(console)

    # Create and submit a test batch
    batch = GenericBatch(
        id="test-batch-1",
        request_file="test.jsonl",
        status=GenericBatchStatus.SUBMITTED,
        created_at=datetime.datetime.now(),
        finished_at=None,
        api_key_suffix="1234",
        raw_status="submitted",
        raw_batch={},
        request_counts=GenericBatchRequestCounts(total=50, succeeded=45, failed=5, raw_request_counts_object={"total": 50, "succeeded": 45, "failed": 5}),
    )

    tracker.mark_as_submitted(batch, n_requests=50)

    # Update token usage and cost
    token_usage = TokenUsage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)
    tracker.update_token_and_cost(token_usage, cost=0.123)

    # Mark as finished and downloaded
    tracker.mark_as_finished(batch)
    tracker.mark_as_downloaded(batch)

    # Generate final stats
    tracker.stop_tracker()

    # Get the captured output
    captured = output.getvalue()
    # Test serialization
    assert isinstance(tracker.json(), str)

    # Verify the final statistics table
    assert "Final Curator Statistics" in captured, captured
    assert "test-model" in captured, captured
    assert "Total Tokens Used" in captured, captured
    assert "1,000" in captured, captured  # Input tokens
    assert "2,000" in captured, captured  # Output tokens
    assert "$0.123" in captured, captured  # Cost
    assert "45" in captured, captured  # Successful requests
    assert "5" in captured, captured  # Failed requests
    assert "$0.15" in captured, captured  # Input cost per million
    assert "$0.60" in captured, captured  # Output cost per million
