"""Configuration models for fine-tuning."""

import os
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


_VALID_LORA_RANKS = (1, 2, 4, 8, 16, 32)
_RESOURCE_NAME_RE = re.compile(r"^[a-z0-9-]+$")


class FireworksTrainerConfig(BaseModel):
    """Configuration for FireworksTrainer (Fireworks AI managed fine-tuning).

    Fireworks runs supervised fine-tuning (SFT) as a managed, server-side job: a
    chat-format JSONL dataset is uploaded, an SFT job is submitted against a base
    model, and the result is a new (LoRA) model that is served via an on-demand
    deployment. This mirrors :class:`TinkerTrainerConfig` but uses the parameters
    that the Fireworks API actually exposes.

    Attributes:
        base_model: Base model to fine-tune. Either a short id (e.g. ``"qwen3-4b"``)
            or a fully-qualified id (``"accounts/fireworks/models/qwen3-4b"``).
        epochs: Number of training epochs (``examples * epochs`` must be <= 3,000,000).
            Note: Fireworks packs examples into batches by token count, so a small
            dataset may be a single batch (one optimizer step) per epoch. For tiny
            datasets, use many epochs so the LoRA gets enough gradient steps to learn.
        learning_rate: Learning rate. ``None`` lets Fireworks auto-select per base
            model (recommended).
        lora_rank: LoRA rank; must be a power of two up to 32. ``None`` requests a
            full-parameter fine-tune instead of LoRA.
        batch_size: Training batch size (tokens packed per step). ``None`` uses the
            Fireworks default.
        max_context_length: Maximum context length during training. ``None`` uses
            the base-model default.
        early_stop: Whether to enable early stopping.
        output_model: Desired id for the resulting fine-tuned model. ``None`` lets
            Fireworks derive it from the job id.
        display_name: Human-readable job name; must match ``^[a-z0-9-]+$``.
        api_key: Fireworks API key (defaults to the ``FIREWORKS_API_KEY`` env var).
        account_id: Fireworks account id. ``None`` auto-discovers it from the key.
        deployment_type: Deployment strategy used to serve the fine-tuned model for
            inference. Fine-tuned LoRA models cannot be served serverlessly, so this
            defaults to ``"on-demand"`` (Live Merge).
        accelerator_type: Optional accelerator override for the inference deployment.
        accelerator_count: Optional accelerator count for the inference deployment.
        auto_deploy: If True, :meth:`FireworksTrainer.sample` provisions an on-demand
            deployment automatically on first use. Always release it afterwards with
            :meth:`FireworksTrainer.close` or by using the trainer as a context manager.
        poll_interval_seconds: How often to poll the fine-tuning job for progress.
        max_wait_seconds: Maximum time to wait for a fine-tuning job to complete.
        inference_base_url: OpenAI-compatible base URL for inference.
    """

    base_model: str
    epochs: int = Field(default=1, gt=0)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    lora_rank: Optional[int] = Field(default=8)
    batch_size: Optional[int] = Field(default=None, gt=0)
    max_context_length: Optional[int] = Field(default=None, gt=0)
    early_stop: Optional[bool] = None
    output_model: Optional[str] = None
    display_name: str = "curator-sft"
    api_key: Optional[str] = None
    account_id: Optional[str] = None

    # Inference / deployment
    deployment_type: str = "on-demand"
    accelerator_type: Optional[str] = None
    accelerator_count: Optional[int] = Field(default=None, gt=0)
    auto_deploy: bool = True

    # Polling / inference
    poll_interval_seconds: int = Field(default=10, gt=0)
    max_wait_seconds: int = Field(default=3600, gt=0)
    inference_base_url: str = "https://api.fireworks.ai/inference/v1"

    model_config = ConfigDict(extra="forbid")

    @field_validator("lora_rank")
    @classmethod
    def _validate_lora_rank(cls, value: Optional[int]) -> Optional[int]:
        """Ensure lora_rank is a power of two up to 32 (Fireworks constraint)."""
        if value is not None and value not in _VALID_LORA_RANKS:
            raise ValueError(f"lora_rank must be one of {_VALID_LORA_RANKS} or None for full fine-tuning, got {value}")
        return value

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        """Ensure display_name only contains lowercase letters, digits, and hyphens."""
        if not _RESOURCE_NAME_RE.fullmatch(value):
            raise ValueError(f"display_name must match ^[a-z0-9-]+$ (lowercase a-z, 0-9, hyphen), got {value!r}")
        return value

    def __init__(self, **data):
        """Initialize config, loading API key from environment if not provided."""
        if data.get("api_key") is None:
            data["api_key"] = os.environ.get("FIREWORKS_API_KEY")
        super().__init__(**data)

    @property
    def qualified_base_model(self) -> str:
        """Return the fully-qualified base model id (``accounts/fireworks/models/...``)."""
        if self.base_model.startswith("accounts/"):
            return self.base_model
        return f"accounts/fireworks/models/{self.base_model}"
