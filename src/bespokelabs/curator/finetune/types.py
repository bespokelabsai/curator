"""Type definitions for fine-tuning module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str
    content: str


class TrainingExample(BaseModel):
    """A single training example with chat messages."""

    messages: List[ChatMessage]

    @classmethod
    def from_dict_messages(cls, messages: List[Dict[str, str]]) -> "TrainingExample":
        """Create a TrainingExample from a list of message dictionaries."""
        return cls(messages=[ChatMessage(**msg) for msg in messages])


class SamplingConfig(BaseModel):
    """Configuration for inference/sampling."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = Field(default_factory=list)


@dataclass
class TrainingStats:
    """Real-time training statistics."""

    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    tokens_processed: int = 0
    samples_processed: int = 0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    name: str
    path: str
    step: int
    epoch: int
    loss: float


@dataclass
class TrainingResult:
    """Result from a training run."""

    final_loss: float
    total_steps: int
    total_epochs: int
    total_time: float
    tokens_processed: int
    samples_processed: int
    loss_history: List[float] = field(default_factory=list)
    weights_name: Optional[str] = None
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
