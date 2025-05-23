import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from datasets import Dataset

from bespokelabs.curator.types.curator_response import CostInfo, PerformanceStats, RequestStats, TokenUsage
from bespokelabs.curator.types.generic_response import GenericResponse


@dataclass
class AgentResponse(GenericResponse):
    """A response from an agent in a multi-turn conversation.

    This class extends the GenericResponse class to include agent-specific information,
    particularly the name of the agent that generated the response.

    Attributes:
        name (str): The name of the agent that generated this response.
    """

    name: str


@dataclass
class MultiTurnResponse:
    """Response from a multi-turn conversation between agents.

    This class encapsulates all the information about a multi-turn conversation,
    including the dataset, conversation history, and various statistics.

    Attributes:
        dataset (Dataset): The conversation dataset.
        cache_dir (Optional[str]): Directory where conversation results are cached.
        conversation_history (list): List of message dictionaries containing role and content.
        seeder_model (str): Name of the model used by the seeder agent.
        partner_model (str): Name of the model used by the partner agent.
        token_usage (TokenUsage): Token usage statistics for both agents.
        cost_info (CostInfo): Cost information for both agents.
        request_stats (RequestStats): Request statistics for both agents.
        performance_stats (PerformanceStats): Performance statistics for the conversation.
        metadata (Dict[str, Any]): Additional metadata about the conversation.
    """

    # Core data
    dataset: Dataset
    cache_dir: Optional[str] = None
    conversation_history: list = field(default_factory=list)
    seeder_model: str = ""
    partner_model: str = ""

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

    def update_tracker_stats(self, tracker) -> None:
        """Update the response with statistics from a tracker.

        Args:
            tracker: The tracker object containing statistics about the conversation.
        """
        if tracker is None:
            return

        # Update token usage
        self.token_usage = TokenUsage(input=tracker.total_tokens.input, output=tracker.total_tokens.output, total=tracker.total_tokens.total)

        # Update cost information
        self.cost_info = CostInfo(
            total_cost=tracker.total_cost,
            input_cost_per_million=tracker.input_cost_per_million,
            output_cost_per_million=tracker.output_cost_per_million,
            projected_remaining_cost=0.0,  # AgentStatusTracker doesn't track this
        )

        # Update request statistics
        self.request_stats = RequestStats(
            total=tracker.max_turns,
            succeeded=tracker.num_responses,
            failed=tracker.num_errors,
            in_progress=0,  # AgentStatusTracker doesn't track this
            cached=0,  # AgentStatusTracker doesn't track this
        )

        # Update performance statistics
        elapsed_time = time.time() - tracker.start_time
        elapsed_minutes = elapsed_time / 60
        self.performance_stats = PerformanceStats(
            total_time=elapsed_time,
            requests_per_minute=tracker.num_responses / max(0.001, elapsed_minutes),
            input_tokens_per_minute=tracker.total_tokens.input / max(0.001, elapsed_minutes),
            output_tokens_per_minute=tracker.total_tokens.output / max(0.001, elapsed_minutes),
            max_concurrent_requests=1,  # AgentStatusTracker doesn't track this
        )
