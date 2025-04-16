from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path
import json
import os

from datasets import Dataset


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
    failed_requests_path: Optional[Path] = None
    
    # Model information
    model_name: str = ""
    max_requests_per_minute: int = 0
    max_tokens_per_minute: int = 0
    
    # Statistics
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_info: CostInfo = field(default_factory=CostInfo)
    request_stats: RequestStats = field(default_factory=RequestStats)
    performance_stats: PerformanceStats = field(default_factory=PerformanceStats)
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary."""
        return {
            "model_name": self.model_name,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "token_usage": {
                "input": self.token_usage.input,
                "output": self.token_usage.output,
                "total": self.token_usage.total
            },
            "cost_info": {
                "total_cost": self.cost_info.total_cost,
                "input_cost_per_million": self.cost_info.input_cost_per_million,
                "output_cost_per_million": self.cost_info.output_cost_per_million,
                "projected_remaining_cost": self.cost_info.projected_remaining_cost
            },
            "request_stats": {
                "total": self.request_stats.total,
                "succeeded": self.request_stats.succeeded,
                "failed": self.request_stats.failed,
                "in_progress": self.request_stats.in_progress,
                "cached": self.request_stats.cached
            },
            "performance_stats": {
                "total_time": self.performance_stats.total_time,
                "requests_per_minute": self.performance_stats.requests_per_minute,
                "input_tokens_per_minute": self.performance_stats.input_tokens_per_minute,
                "output_tokens_per_minute": self.performance_stats.output_tokens_per_minute,
                "max_concurrent_requests": self.performance_stats.max_concurrent_requests
            },
            "metadata": self.metadata
        }
    
    def save(self, cache_dir: Union[str, Path]) -> None:
        """Save the response to a cache directory.
        
        Args:
            cache_dir: Directory to save the response to
        """
        cache_dir = Path(cache_dir)
        
        # Save the response metadata
        with open(os.path.join(cache_dir, "curator_response.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, cache_dir: Union[str, Path], dataset: Dataset) -> "CuratorResponse":
        """Load a response from a cache directory.
        
        Args:
            cache_dir: Directory containing the cached response
            dataset: The dataset to use for the response
            
        Returns:
            CuratorResponse: The loaded response
        """
        cache_dir = Path(cache_dir)
        
        # Load the response metadata
        with open(os.path.join(cache_dir, "curator_response.json"), "r") as f:
            data = json.load(f)
        
        # Create the response object
        response = cls(
            dataset=dataset,
            failed_requests_path=os.path.join(cache_dir, "failed_requests.jsonl") if os.path.exists(os.path.join(cache_dir, "failed_requests.jsonl")) else None,
            model_name=data["model_name"],
            max_requests_per_minute=data["max_requests_per_minute"],
            max_tokens_per_minute=data["max_tokens_per_minute"],
            token_usage=TokenUsage(**data["token_usage"]),
            cost_info=CostInfo(**data["cost_info"]),
            request_stats=RequestStats(**data["request_stats"]),
            performance_stats=PerformanceStats(**data["performance_stats"]),
            metadata=data["metadata"]
        )
        
        return response 