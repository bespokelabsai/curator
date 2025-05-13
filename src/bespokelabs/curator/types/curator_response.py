import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from datasets import Dataset
from pydantic import BaseModel

from bespokelabs.curator.log import logger
from bespokelabs.curator.status_tracker.offline_status_tracker import OfflineStatusTracker
from bespokelabs.curator.status_tracker.tqdm_constants.colors import COST, END, ERROR, HEADER, METRIC, MODEL, SUCCESS


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input: int = 0
    output: int = 0
    total: int = 0

    def add(self, other: "TokenUsage"):
        """Add token usage statistics from another TokenUsage instance.

        This method adds the input, output, and total token counts from another
        TokenUsage instance to this one.

        Args:
            other (TokenUsage): Another TokenUsage instance whose values will be added.
        """
        self.input += other.input
        self.output += other.output
        self.total += other.total


@dataclass
class CostInfo:
    """Cost information."""

    total_cost: float = 0.0
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None
    projected_remaining_cost: float = 0.0

    def add(self, other: "CostInfo"):
        """Add cost information from another CostInfo instance.

        This method adds the total cost and projected remaining cost from another
        CostInfo instance to this one. If both instances have input/output costs
        per million tokens, those are also added.

        Args:
            other (CostInfo): Another CostInfo instance whose values will be added.
        """
        self.total_cost += other.total_cost
        if self.input_cost_per_million and other.input_cost_per_million:
            self.input_cost_per_million += other.input_cost_per_million
        if self.output_cost_per_million and other.output_cost_per_million:
            self.output_cost_per_million += other.output_cost_per_million

        self.projected_remaining_cost += other.projected_remaining_cost


@dataclass
class RequestStats:
    """Request statistics."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    in_progress: int = 0
    cached: int = 0

    def add(self, other: "RequestStats"):
        """Add request statistics from another RequestStats instance.

        This method adds all request counts (total, succeeded, failed, in_progress,
        and cached) from another RequestStats instance to this one.

        Args:
            other (RequestStats): Another RequestStats instance whose values will be added.
        """
        self.total += other.total
        self.succeeded += other.succeeded
        self.failed += other.failed
        self.in_progress += other.in_progress
        self.cached += other.cached


@dataclass
class PerformanceStats:
    """Performance statistics."""

    total_time: float = 0.0
    requests_per_minute: float = 0.0
    input_tokens_per_minute: float = 0.0
    output_tokens_per_minute: float = 0.0
    max_concurrent_requests: int = 0

    def add(self, other: "PerformanceStats"):
        """Add performance statistics from another PerformanceStats instance.

        This method adds all performance metrics (total time, requests per minute,
        input/output tokens per minute, and max concurrent requests) from another
        PerformanceStats instance to this one.

        Args:
            other (PerformanceStats): Another PerformanceStats instance whose values will be added.
        """
        self.total_time += other.total_time
        self.requests_per_minute += other.requests_per_minute
        self.input_tokens_per_minute += other.input_tokens_per_minute
        self.output_tokens_per_minute += other.output_tokens_per_minute
        self.max_concurrent_requests += other.max_concurrent_requests


@dataclass
class CuratorResponse:
    """Response from Curator LLM processing.

    This class encapsulates all the information about a Curator processing run,
    including the dataset, failed requests, and various statistics.
    """

    # Core data
    dataset: Dataset
    cache_dir: Optional[str] = None
    failed_requests_path: Optional[Path] = None
    viewer_url: Optional[str] = None
    batch_mode: bool = False

    # Model information
    model_name: str = "gpt-4"
    max_requests_per_minute: int | None = None
    max_tokens_per_minute: int | None = None

    # Statistics
    token_usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    cost_info: CostInfo = field(default_factory=lambda: CostInfo(0.0, 0.0, 0.0, 0.0))
    request_stats: RequestStats = field(default_factory=lambda: RequestStats(0, 0, 0, 0, 0))
    performance_stats: PerformanceStats = field(default_factory=lambda: PerformanceStats(0.0, 0.0, 0.0, 0.0, 0))

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization hook."""
        if self.metadata is None:
            self.metadata = {}

    def __getattr__(self, name):
        """Get an attribute from the response.

        This method is called when an attribute is not found in the normal way.
        It raises a deprecation warning and AttributeError to indicate that the
        Huggingface response API is deprecated.

        Args:
            name (str): The name of the attribute being accessed.

        Raises:
            AttributeError: Always raises this error with a deprecation warning.
        """
        warnings.warn("Warning: Huggingface response from the curator LLM api is deprecated, please check out the `CuratorResponse`", stacklevel=2)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def update_tracker_stats(self, tracker) -> None:
        """Update the response with statistics from a tracker.

        This method updates various statistics in the response object based on
        the provided tracker's data, including token usage, costs, request stats,
        and performance metrics.

        Args:
            tracker: The tracker object containing statistics about the processing run.
                    Can be either an OfflineStatusTracker or None.
        """
        if isinstance(tracker, OfflineStatusTracker) or tracker is None:
            logger.warning("Tracker is offline or None, skipping stats update")
            return

        if not self.batch_mode:
            if isinstance(tracker.max_tokens_per_minute, BaseModel):
                max_tokens_per_minute = tracker.max_tokens_per_minute.model_dump()
            else:
                max_tokens_per_minute = tracker.max_tokens_per_minute
            self.max_requests_per_minute = tracker.max_requests_per_minute
            self.max_tokens_per_minute = max_tokens_per_minute

        # Update token usage
        self.token_usage = TokenUsage(input=tracker.total_prompt_tokens, output=tracker.total_completion_tokens, total=tracker.total_tokens)

        # Update cost information
        self.cost_info = CostInfo(
            total_cost=tracker.total_cost,
            input_cost_per_million=tracker.input_cost_per_million,
            output_cost_per_million=tracker.output_cost_per_million,
            projected_remaining_cost=tracker.projected_remaining_cost,
        )

        # Update request statistics
        if self.batch_mode:
            total = tracker.n_total_requests
            succeeded = tracker.n_final_success_requests
            failed = tracker.n_final_failed_requests
            in_progress = 0
            cached = 0
        else:
            total = tracker.total_requests
            succeeded = tracker.num_tasks_succeeded
            failed = tracker.num_tasks_failed
            in_progress = tracker.num_tasks_in_progress
            cached = tracker.num_tasks_already_completed

        self.request_stats = RequestStats(total=total, succeeded=succeeded, failed=failed, in_progress=in_progress, cached=cached)

        # Update performance statistics
        self.performance_stats = PerformanceStats(
            total_time=time.time() - tracker.start_time,
            requests_per_minute=self.request_stats.succeeded / max(0.001, (time.time() - tracker.start_time) / 60),
            input_tokens_per_minute=self.token_usage.input / max(0.001, (time.time() - tracker.start_time) / 60),
            output_tokens_per_minute=self.token_usage.output / max(0.001, (time.time() - tracker.start_time) / 60),
            max_concurrent_requests=tracker.max_concurrent_requests_seen if not self.batch_mode else 0,
        )

    def get_failed_requests(self) -> Iterable[Dict[str, Any]]:
        """Get an iterator over failed requests.

        This method reads the failed requests from a JSONL file and yields
        each request as a dictionary. If no failed requests file exists,
        returns an empty iterator.

        Returns:
            Iterable[Dict[str, Any]]: An iterator that yields dictionaries
                                     containing failed request data.
        """
        if self.failed_requests_path is None or not self.failed_requests_path.exists():
            return iter([])

        def _iter_failed_requests():
            with open(self.failed_requests_path, "r") as f:
                for line in f:
                    yield json.loads(line)

        return _iter_failed_requests()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary.

        This method creates a dictionary representation of the CuratorResponse
        object, including dataset information, model details, statistics,
        and metadata.

        Returns:
            Dict[str, Any]: A dictionary containing all the response data
                           in a serializable format.
        """
        return {
            "dataset": {
                "fingerprint": self.dataset._fingerprint,
                "size": len(self.dataset),
                "columns": list(self.dataset.column_names),
            },
            "failed_requests_path": str(self.failed_requests_path) if self.failed_requests_path else None,
            "model_name": self.model_name,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "token_usage": {
                "input": self.token_usage.input,
                "output": self.token_usage.output,
                "total": self.token_usage.total,
            },
            "cost_info": {
                "total_cost": self.cost_info.total_cost,
                "input_cost_per_million": self.cost_info.input_cost_per_million,
                "output_cost_per_million": self.cost_info.output_cost_per_million,
                "projected_remaining_cost": self.cost_info.projected_remaining_cost,
            },
            "request_stats": {
                "total": self.request_stats.total,
                "succeeded": self.request_stats.succeeded,
                "failed": self.request_stats.failed,
                "in_progress": self.request_stats.in_progress,
                "cached": self.request_stats.cached,
            },
            "performance_stats": {
                "total_time": self.performance_stats.total_time,
                "requests_per_minute": self.performance_stats.requests_per_minute,
                "input_tokens_per_minute": self.performance_stats.input_tokens_per_minute,
                "output_tokens_per_minute": self.performance_stats.output_tokens_per_minute,
                "max_concurrent_requests": self.performance_stats.max_concurrent_requests,
            },
            "metadata": self.metadata,
        }

    def save(self, cache_dir: Union[str, Path]) -> None:
        """Save the response to a cache directory.

        This method saves the response data to a JSON file in the specified
        cache directory. The directory will be created if it doesn't exist.

        Args:
            cache_dir (Union[str, Path]): Directory where the response will be saved.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save response.json
        with open(cache_dir / "response.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, cache_dir: Union[str, Path], dataset: Dataset) -> "CuratorResponse":
        """Load a response from a cache directory.

        This class method creates a new CuratorResponse instance from saved
        data in the cache directory. It requires a dataset to be provided
        as the dataset itself is not cached.

        Args:
            cache_dir (Union[str, Path]): Directory containing the cached response data.
            dataset (Dataset): The dataset to use for the response.

        Returns:
            CuratorResponse: A new CuratorResponse instance loaded from cache.
        """
        cache_dir = Path(cache_dir)

        # Load response.json
        with open(cache_dir / "response.json", "r") as f:
            data = json.load(f)

        # Create response object
        response = cls(
            dataset=dataset,
            failed_requests_path=cache_dir / "failed_requests.jsonl" if (cache_dir / "failed_requests.jsonl").exists() else None,
            model_name=data["model_name"],
            max_requests_per_minute=data["max_requests_per_minute"],
            max_tokens_per_minute=data["max_tokens_per_minute"],
            token_usage=TokenUsage(**data["token_usage"]),
            cost_info=CostInfo(**data["cost_info"]),
            request_stats=RequestStats(**data["request_stats"]),
            performance_stats=PerformanceStats(**data["performance_stats"]),
            metadata=data["metadata"],
        )

        return response

    def append(self, other: "CuratorResponse"):
        """Appends another CuratorResponse to this one.

        This method combines the statistics from another CuratorResponse
        with this one. It adds up token usage, costs, request stats,
        and performance metrics.

        Args:
            other (CuratorResponse): Another CuratorResponse to append.

        Raises:
            AssertionError: If the model names of the two responses don't match.
        """
        assert self.model_name == other.model_name

        self.token_usage.add(other.token_usage)
        self.cost_info.add(other.cost_info)
        self.request_stats.add(other.request_stats)
        self.performance_stats.add(other.performance_stats)

    def display_stats(self):
        """Display final statistics in plain text format.

        This method prints a formatted summary of all statistics including
        model information, request statistics, token usage, costs, and
        performance metrics. The output is color-coded for better readability.
        """
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        rpm = self.request_stats.num_tasks_succeeded / max(0.001, elapsed_minutes)
        input_tpm = self.token_usage.prompt_tokens / max(0.001, elapsed_minutes)
        output_tpm = self.token_usage.completion_tokens / max(0.001, elapsed_minutes)

        stats = [
            f"\n{HEADER}Final Statistics:{END}",
            f"{HEADER}Model Information:{END}",
            f"  Model: {MODEL}{self.model_name}{END}",
            f"  Rate Limit (RPM): {METRIC}{self.max_requests_per_minute}{END}",
            f"  Rate Limit (TPM): {METRIC}{self.max_tokens_per_minute}{END}",
            "",
            f"{HEADER}Request Statistics:{END}",
            f"  Total Requests: {METRIC}{self.request_stats.total_requests}{END}",
            f"  Cached: {SUCCESS}{self.request_stats.num_tasks_already_completed}{END}",
            f"  Successful: {SUCCESS}{self.request_stats.num_tasks_succeeded}{END}",
            f"  Failed: {ERROR}{self.request_stats.num_tasks_failed}{END}",
            "",
            f"{HEADER}Token Statistics:{END}",
            f"  Total Tokens Used: {METRIC}{self.token_usage.total_tokens:,}{END}",
            f"  Total Input Tokens: {METRIC}{self.token_usage.prompt_tokens:,}{END}",
            f"  Total Output Tokens: {METRIC}{self.token_usage.completion_tokens:,}{END}",
            f"  Average Tokens per Request: {METRIC}{int(self.token_usage.total_tokens / max(1, self.request_stats.num_tasks_succeeded))}{END}",
            f"  Average Input Tokens: {METRIC}{int(self.token_usage.prompt_tokens / max(1, self.request_stats.num_tasks_succeeded))}{END}",
            f"  Average Output Tokens: {METRIC}{int(self.token_usage.completion_tokens / max(1, self.request_stats.num_tasks_succeeded))}{END}",
            "",
            f"{HEADER}Cost Statistics:{END}",
            f"  Total Cost: {COST}${self.cost_info.total_cost:.3f}{END}",
            f"  Average Cost per Request: {COST}${self.cost_info.total_cost / max(1, self.request_stats.num_tasks_succeeded):.3f}{END}",
            f"  Input Cost per 1M Tokens: {COST}${self.cost_info.input_cost_per_million:.3f}{END}",
            f"  Output Cost per 1M Tokens: {COST}${self.cost_info.output_cost_per_million:.3f}{END}",
            "",
            f"{HEADER}Performance Statistics:{END}",
            f"  Total Time: {METRIC}{elapsed_time:.2f}s{END}",
            f"  Average Time per Request: {METRIC}{elapsed_time / max(1, self.request_stats.num_tasks_succeeded):.2f}s{END}",
            f"  Requests per Minute: {METRIC}{rpm:.1f}{END}",
            f"  Max Concurrent Requests: {METRIC}{self.performance_stats.max_concurrent_requests}{END}",
            f"  Input Tokens per Minute: {METRIC}{input_tpm:.1f}{END}",
            f"  Output Tokens per Minute: {METRIC}{output_tpm:.1f}{END}",
        ]
        logger.info("\n".join(stats))
