"""Configuration models for fine-tuning."""

import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AdamParams(BaseModel):
    """Adam optimizer parameters."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=1e-4, gt=0)
    beta1: float = Field(default=0.9, ge=0, le=1)
    beta2: float = Field(default=0.999, ge=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0)
    epsilon: float = Field(default=1e-8, gt=0)


class LoRAConfig(BaseModel):
    """LoRA-specific configuration."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0, le=1)
    target_modules: list = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


class TinkerTrainerConfig(BaseModel):
    """Main configuration for TinkerTrainer.

    Attributes:
        base_model: Name of the base model to fine-tune (e.g., "Qwen/Qwen3-8B")
        epochs: Number of training epochs
        batch_size: Training batch size
        max_seq_length: Maximum sequence length for training
        adam_params: Adam optimizer parameters
        lora_config: LoRA-specific configuration
        api_key: Tinker API key (defaults to TINKER_API_KEY env var)
        gradient_accumulation_steps: Number of gradient accumulation steps
        warmup_steps: Number of warmup steps for learning rate scheduler
        log_every_n_steps: Log training stats every N steps
        save_weights_on_complete: Whether to save weights when training completes
        checkpoint_every_n_steps: Save checkpoint every N steps (0 to disable)
        checkpoint_every_epoch: Save checkpoint at the end of each epoch
        checkpoint_name_prefix: Prefix for checkpoint names
    """

    base_model: str
    epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=4, gt=0)
    max_seq_length: int = Field(default=2048, gt=0)
    adam_params: AdamParams = Field(default_factory=AdamParams)
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)
    api_key: Optional[str] = None
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    log_every_n_steps: int = Field(default=10, gt=0)
    save_weights_on_complete: bool = True
    checkpoint_every_n_steps: int = Field(default=0, ge=0)
    checkpoint_every_epoch: bool = False
    checkpoint_name_prefix: str = "checkpoint"

    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data):
        """Initialize config, loading API key from environment if not provided."""
        if "api_key" not in data or data["api_key"] is None:
            data["api_key"] = os.environ.get("TINKER_API_KEY")
        super().__init__(**data)
