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


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input: int = 0
    output: int = 0
    total: int = 0


@dataclass
class CostInfo:
    """Cost information."""

    total_cost: float = 0.0
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None
    projected_remaining_cost: float = 0.0


@dataclass
class RequestStats:
    """Request statistics."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    in_progress: int = 0
    cached: int = 0


@dataclass
class PerformanceStats:
    """Performance statistics."""

    total_time: float = 0.0
    requests_per_minute: float = 0.0
    input_tokens_per_minute: float = 0.0
    output_tokens_per_minute: float = 0.0
    max_concurrent_requests: int = 0


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
        """Get an attribute from the response."""
        warnings.warn("Warning: Huggingface response from the curator LLM api is deprecated, please check out the `CuratorResponse`", stacklevel=2)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def update_tracker_stats(self, tracker) -> None:
        """Update the response with statistics from a tracker.

        Args:
            tracker: The tracker object containing statistics about the processing run.
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

        Returns:
            Iterable[Dict[str, Any]]: Iterator over failed requests
        """
        if self.failed_requests_path is None or not self.failed_requests_path.exists():
            return iter([])

        def _iter_failed_requests():
            with open(self.failed_requests_path, "r") as f:
                for line in f:
                    yield json.loads(line)

        return _iter_failed_requests()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary."""
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

        Args:
            cache_dir: Directory to save the response to
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save response.json
        with open(cache_dir / "response.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, cache_dir: Union[str, Path], dataset: Dataset) -> "CuratorResponse":
        """Load a response from a cache directory.

        Args:
            cache_dir: Directory to load the response from
            dataset: Dataset to use for the response

        Returns:
            CuratorResponse: Loaded response
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
