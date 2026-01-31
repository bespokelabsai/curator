"""Tests for finetune type definitions."""

from bespokelabs.curator.finetune.types import (
    ChatMessage,
    SamplingConfig,
    TrainingExample,
    TrainingResult,
    TrainingStats,
)


class TestChatMessage:
    """Tests for ChatMessage."""

    def test_create_message(self):
        """Test creating a chat message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_roles(self):
        """Test different message roles."""
        for role in ["system", "user", "assistant"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role


class TestTrainingExample:
    """Tests for TrainingExample."""

    def test_create_example(self):
        """Test creating a training example."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        example = TrainingExample(messages=messages)
        assert len(example.messages) == 2
        assert example.messages[0].role == "user"
        assert example.messages[1].role == "assistant"

    def test_from_dict_messages(self):
        """Test creating from dictionary messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        example = TrainingExample.from_dict_messages(messages)
        assert len(example.messages) == 2
        assert example.messages[0].role == "user"
        assert example.messages[0].content == "Hello"

    def test_from_dict_messages_with_system(self):
        """Test creating with system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        example = TrainingExample.from_dict_messages(messages)
        assert len(example.messages) == 3
        assert example.messages[0].role == "system"


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_values(self):
        """Test default sampling config values."""
        config = SamplingConfig()
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.stop_sequences == []

    def test_custom_values(self):
        """Test custom sampling config values."""
        config = SamplingConfig(
            max_tokens=256,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            stop_sequences=["END", "STOP"],
        )
        assert config.max_tokens == 256
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.stop_sequences == ["END", "STOP"]


class TestTrainingStats:
    """Tests for TrainingStats."""

    def test_default_values(self):
        """Test default training stats values."""
        stats = TrainingStats()
        assert stats.current_epoch == 0
        assert stats.total_epochs == 0
        assert stats.current_step == 0
        assert stats.total_steps == 0
        assert stats.current_loss == 0.0
        assert stats.tokens_processed == 0
        assert stats.samples_processed == 0
        assert stats.learning_rate == 0.0
        assert stats.elapsed_time == 0.0

    def test_custom_values(self):
        """Test custom training stats values."""
        stats = TrainingStats(
            current_epoch=2,
            total_epochs=5,
            current_step=100,
            total_steps=500,
            current_loss=0.5,
            tokens_processed=10000,
            samples_processed=50,
            learning_rate=1e-4,
            elapsed_time=120.5,
        )
        assert stats.current_epoch == 2
        assert stats.total_epochs == 5
        assert stats.current_step == 100
        assert stats.total_steps == 500
        assert stats.current_loss == 0.5
        assert stats.tokens_processed == 10000
        assert stats.samples_processed == 50
        assert stats.learning_rate == 1e-4
        assert stats.elapsed_time == 120.5


class TestTrainingResult:
    """Tests for TrainingResult."""

    def test_create_result(self):
        """Test creating a training result."""
        result = TrainingResult(
            final_loss=0.25,
            total_steps=1000,
            total_epochs=3,
            total_time=3600.0,
            tokens_processed=100000,
            samples_processed=500,
        )
        assert result.final_loss == 0.25
        assert result.total_steps == 1000
        assert result.total_epochs == 3
        assert result.total_time == 3600.0
        assert result.tokens_processed == 100000
        assert result.samples_processed == 500
        assert result.loss_history == []
        assert result.weights_name is None
        assert result.metadata == {}

    def test_with_optional_fields(self):
        """Test creating a result with optional fields."""
        result = TrainingResult(
            final_loss=0.25,
            total_steps=1000,
            total_epochs=3,
            total_time=3600.0,
            tokens_processed=100000,
            samples_processed=500,
            loss_history=[2.5, 1.5, 0.5, 0.25],
            weights_name="my_weights_v1",
            metadata={"learning_rate": 1e-4, "batch_size": 8},
        )
        assert result.loss_history == [2.5, 1.5, 0.5, 0.25]
        assert result.weights_name == "my_weights_v1"
        assert result.metadata["learning_rate"] == 1e-4
        assert result.metadata["batch_size"] == 8
