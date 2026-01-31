"""Abstract base class for trainers."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from bespokelabs.curator.finetune.types import TrainingExample, TrainingResult


class BaseTrainer(ABC):
    """Abstract base class for fine-tuning trainers."""

    @abstractmethod
    def train(self, dataset: Any) -> TrainingResult:
        """Run training on the provided dataset.

        Args:
            dataset: The training dataset (HuggingFace Dataset or similar)

        Returns:
            TrainingResult with training metrics and metadata
        """
        pass

    @abstractmethod
    def save_weights(self, name: str) -> str:
        """Save the trained weights.

        Args:
            name: Name for the saved weights

        Returns:
            Identifier or path to the saved weights
        """
        pass

    @abstractmethod
    def format_example(self, row: Dict[str, Any]) -> TrainingExample:
        """Format a dataset row into a TrainingExample.

        Override this method to customize how your data is formatted.

        Args:
            row: A dictionary containing the training data

        Returns:
            TrainingExample with formatted messages
        """
        pass

    @abstractmethod
    def sample(self, prompt: str, **kwargs) -> str:
        """Generate a sample from the trained model.

        Args:
            prompt: The input prompt
            **kwargs: Additional sampling parameters

        Returns:
            Generated text
        """
        pass
