"""Fine-tuning module for curator.

This module provides LoRA-based fine-tuning via the Tinker API.

Example usage:
    ```python
    from bespokelabs.curator import TinkerTrainer, TinkerTrainerConfig
    from datasets import Dataset

    # Prepare data (chat format)
    data = [
        {"messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."}
        ]},
    ]
    dataset = Dataset.from_list(data)

    # Configure and train
    config = TinkerTrainerConfig(
        base_model="Qwen3-8B",
        epochs=3,
        batch_size=4,
        adam_params={"learning_rate": 1e-4},
    )

    trainer = TinkerTrainer(config)
    result = trainer.train(dataset)

    # Sample from fine-tuned model
    response = trainer.sample("Explain recursion")
    ```

Custom data formatting:
    ```python
    from bespokelabs.curator.finetune import TrainingExample

    class MyTrainer(TinkerTrainer):
        def format_example(self, row: dict) -> TrainingExample:
            return TrainingExample(
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
            )
    ```
"""

from bespokelabs.curator.finetune.config import AdamParams, LoRAConfig, TinkerTrainerConfig
from bespokelabs.curator.finetune.data_formatter import DataFormatter
from bespokelabs.curator.finetune.status_tracker import FinetuneStatusTracker
from bespokelabs.curator.finetune.trainer import BaseTrainer, TinkerTrainer
from bespokelabs.curator.finetune.types import (
    ChatMessage,
    CheckpointInfo,
    SamplingConfig,
    TrainingExample,
    TrainingResult,
    TrainingStats,
)

__all__ = [
    # Config
    "TinkerTrainerConfig",
    "AdamParams",
    "LoRAConfig",
    # Trainers
    "TinkerTrainer",
    "BaseTrainer",
    # Types
    "TrainingExample",
    "TrainingResult",
    "TrainingStats",
    "ChatMessage",
    "SamplingConfig",
    "CheckpointInfo",
    # Utilities
    "DataFormatter",
    "FinetuneStatusTracker",
]
