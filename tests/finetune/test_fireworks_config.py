"""Tests for FireworksTrainerConfig."""

import pytest

from bespokelabs.curator.finetune.config import FireworksTrainerConfig


class TestFireworksTrainerConfig:
    """Tests for FireworksTrainerConfig."""

    def test_minimal_config(self):
        """Test config with only required fields."""
        config = FireworksTrainerConfig(base_model="qwen3-4b")
        assert config.base_model == "qwen3-4b"
        assert config.epochs == 1
        assert config.lora_rank == 8
        assert config.learning_rate is None
        assert config.deployment_type == "on-demand"
        assert config.inference_base_url == "https://api.fireworks.ai/inference/v1"

    def test_qualified_base_model_short_id(self):
        """Test short ids are expanded to fully-qualified resource names."""
        config = FireworksTrainerConfig(base_model="qwen3-4b")
        assert config.qualified_base_model == "accounts/fireworks/models/qwen3-4b"

    def test_qualified_base_model_full_id(self):
        """Test fully-qualified ids are left unchanged."""
        config = FireworksTrainerConfig(base_model="accounts/fireworks/models/qwen3-8b")
        assert config.qualified_base_model == "accounts/fireworks/models/qwen3-8b"

    def test_full_config(self):
        """Test config with all common fields."""
        config = FireworksTrainerConfig(
            base_model="qwen3-8b",
            epochs=3,
            learning_rate=1e-4,
            lora_rank=16,
            batch_size=32768,
            max_context_length=4096,
            early_stop=True,
            output_model="my-model",
            display_name="my-job",
        )
        assert config.epochs == 3
        assert config.learning_rate == 1e-4
        assert config.lora_rank == 16
        assert config.batch_size == 32768
        assert config.max_context_length == 4096
        assert config.early_stop is True
        assert config.output_model == "my-model"
        assert config.display_name == "my-job"

    @pytest.mark.parametrize("rank", [1, 2, 4, 8, 16, 32, None])
    def test_valid_lora_ranks(self, rank):
        """Test all valid LoRA ranks (powers of two up to 32, or None)."""
        config = FireworksTrainerConfig(base_model="qwen3-4b", lora_rank=rank)
        assert config.lora_rank == rank

    @pytest.mark.parametrize("rank", [3, 7, 12, 64])
    def test_invalid_lora_ranks(self, rank):
        """Test invalid LoRA ranks are rejected."""
        with pytest.raises(ValueError):
            FireworksTrainerConfig(base_model="qwen3-4b", lora_rank=rank)

    def test_invalid_display_name(self):
        """Test display names with invalid characters are rejected."""
        with pytest.raises(ValueError):
            FireworksTrainerConfig(base_model="qwen3-4b", display_name="Bad Name")
        with pytest.raises(ValueError):
            FireworksTrainerConfig(base_model="qwen3-4b", display_name="UPPER")
        with pytest.raises(ValueError):
            # `$` alone would match before a trailing newline; fullmatch must reject it.
            FireworksTrainerConfig(base_model="qwen3-4b", display_name="good-name\n")

    def test_valid_display_name(self):
        """Test valid display names are accepted."""
        config = FireworksTrainerConfig(base_model="qwen3-4b", display_name="good-name-123")
        assert config.display_name == "good-name-123"

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loading from environment variable."""
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test-key")
        config = FireworksTrainerConfig(base_model="qwen3-4b")
        assert config.api_key == "fw-test-key"

    def test_api_key_explicit_overrides_env(self, monkeypatch):
        """Test explicit API key overrides environment."""
        monkeypatch.setenv("FIREWORKS_API_KEY", "env-key")
        config = FireworksTrainerConfig(base_model="qwen3-4b", api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_validation(self):
        """Test numeric field validation."""
        with pytest.raises(ValueError):
            FireworksTrainerConfig(base_model="qwen3-4b", epochs=0)
        with pytest.raises(ValueError):
            FireworksTrainerConfig(base_model="qwen3-4b", batch_size=-1)
        with pytest.raises(ValueError):
            FireworksTrainerConfig(base_model="qwen3-4b", learning_rate=0)
