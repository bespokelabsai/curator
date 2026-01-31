"""Tests for finetune configuration models."""

import pytest

from bespokelabs.curator.finetune.config import AdamParams, LoRAConfig, TinkerTrainerConfig


class TestAdamParams:
    """Tests for AdamParams configuration."""

    def test_default_values(self):
        """Test default parameter values."""
        params = AdamParams()
        assert params.learning_rate == 1e-4
        assert params.beta1 == 0.9
        assert params.beta2 == 0.999
        assert params.weight_decay == 0.01
        assert params.epsilon == 1e-8

    def test_custom_values(self):
        """Test custom parameter values."""
        params = AdamParams(
            learning_rate=1e-5,
            beta1=0.85,
            beta2=0.995,
            weight_decay=0.001,
            epsilon=1e-7,
        )
        assert params.learning_rate == 1e-5
        assert params.beta1 == 0.85
        assert params.beta2 == 0.995
        assert params.weight_decay == 0.001
        assert params.epsilon == 1e-7

    def test_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            AdamParams(learning_rate=-0.001)

        with pytest.raises(ValueError):
            AdamParams(beta1=1.5)


class TestLoRAConfig:
    """Tests for LoRAConfig configuration."""

    def test_default_values(self):
        """Test default parameter values."""
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]

    def test_custom_values(self):
        """Test custom parameter values."""
        config = LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.1,
            target_modules=["q_proj", "k_proj"],
        )
        assert config.rank == 8
        assert config.alpha == 16
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "k_proj"]


class TestTinkerTrainerConfig:
    """Tests for TinkerTrainerConfig."""

    def test_minimal_config(self):
        """Test config with only required fields."""
        config = TinkerTrainerConfig(base_model="Qwen3-8B")
        assert config.base_model == "Qwen3-8B"
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.max_seq_length == 2048
        assert isinstance(config.adam_params, AdamParams)
        assert isinstance(config.lora_config, LoRAConfig)

    def test_full_config(self):
        """Test config with all fields."""
        config = TinkerTrainerConfig(
            base_model="Qwen3-8B",
            epochs=5,
            batch_size=8,
            max_seq_length=4096,
            adam_params={"learning_rate": 1e-5},
            lora_config={"rank": 32},
            gradient_accumulation_steps=4,
            warmup_steps=100,
            log_every_n_steps=5,
            save_weights_on_complete=False,
        )
        assert config.epochs == 5
        assert config.batch_size == 8
        assert config.max_seq_length == 4096
        assert config.adam_params.learning_rate == 1e-5
        assert config.lora_config.rank == 32
        assert config.gradient_accumulation_steps == 4
        assert config.warmup_steps == 100
        assert config.log_every_n_steps == 5
        assert config.save_weights_on_complete is False

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loading from environment variable."""
        monkeypatch.setenv("TINKER_API_KEY", "test-api-key")
        config = TinkerTrainerConfig(base_model="Qwen3-8B")
        assert config.api_key == "test-api-key"

    def test_api_key_explicit(self, monkeypatch):
        """Test explicit API key overrides environment."""
        monkeypatch.setenv("TINKER_API_KEY", "env-key")
        config = TinkerTrainerConfig(base_model="Qwen3-8B", api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            TinkerTrainerConfig(base_model="Qwen3-8B", epochs=0)

        with pytest.raises(ValueError):
            TinkerTrainerConfig(base_model="Qwen3-8B", batch_size=-1)
