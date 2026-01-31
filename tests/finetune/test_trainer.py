"""Tests for TinkerTrainer."""

import pytest

from bespokelabs.curator.finetune.config import TinkerTrainerConfig
from bespokelabs.curator.finetune.trainer import TinkerTrainer
from bespokelabs.curator.finetune.types import TrainingExample, TrainingResult


class TestTinkerTrainer:
    """Tests for TinkerTrainer."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TinkerTrainerConfig(
            base_model="Qwen3-8B",
            epochs=1,
            batch_size=2,
        )

    @pytest.fixture
    def trainer(self, config):
        """Create a test trainer."""
        return TinkerTrainer(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is Java?"},
                    {"role": "assistant", "content": "Java is also a programming language."},
                ]
            },
        ]

    def test_trainer_initialization(self, config):
        """Test trainer initialization."""
        trainer = TinkerTrainer(config)
        assert trainer.config == config
        assert trainer._is_trained is False
        assert trainer._weights_name is None

    def test_trainer_repr(self, trainer):
        """Test trainer string representation."""
        repr_str = repr(trainer)
        assert "Qwen3-8B" in repr_str
        assert "epochs=1" in repr_str
        assert "batch_size=2" in repr_str
        assert "trained=False" in repr_str

    def test_format_example(self, trainer, sample_data):
        """Test formatting a single example."""
        example = trainer.format_example(sample_data[0])
        assert isinstance(example, TrainingExample)
        assert len(example.messages) == 2
        assert example.messages[0].role == "user"
        assert example.messages[1].role == "assistant"

    def test_train_returns_result(self, trainer, sample_data):
        """Test training returns a TrainingResult."""
        result = trainer.train(sample_data)
        assert isinstance(result, TrainingResult)
        assert result.total_epochs == 1
        assert result.samples_processed == 2
        assert result.final_loss > 0
        assert result.total_time > 0

    def test_train_updates_state(self, trainer, sample_data):
        """Test training updates trainer state."""
        assert trainer._is_trained is False
        trainer.train(sample_data)
        assert trainer._is_trained is True
        assert trainer._weights_name is not None

    def test_train_with_more_data(self, config):
        """Test training with more examples."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ]
            }
            for i in range(10)
        ]

        config = TinkerTrainerConfig(
            base_model="Qwen3-8B",
            epochs=2,
            batch_size=3,
        )
        trainer = TinkerTrainer(config)
        result = trainer.train(data)

        assert result.total_epochs == 2
        assert result.samples_processed == 20  # 10 samples * 2 epochs
        assert len(result.loss_history) >= 2  # At least one loss per epoch

    def test_sample(self, trainer):
        """Test sampling from the model."""
        response = trainer.sample("What is Python?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_sample_with_config(self, trainer):
        """Test sampling with custom config."""
        from bespokelabs.curator.finetune.types import SamplingConfig

        sampling_config = SamplingConfig(
            max_tokens=256,
            temperature=0.5,
        )
        response = trainer.sample("What is Python?", sampling_config=sampling_config)
        assert isinstance(response, str)

    def test_sample_with_system_prompt(self, trainer):
        """Test sampling with system prompt."""
        response = trainer.sample(
            "What is Python?",
            system_prompt="You are a helpful assistant.",
        )
        assert isinstance(response, str)

    def test_save_weights(self, trainer, sample_data):
        """Test saving weights."""
        trainer.train(sample_data)
        weight_id = trainer.save_weights("test_weights")
        assert weight_id is not None
        assert "test_weights" in weight_id

    def test_save_weights_before_training(self, trainer):
        """Test saving weights before training gives warning but works."""
        weight_id = trainer.save_weights("pretrain_weights")
        assert weight_id is not None

    def test_get_sampling_client(self, trainer):
        """Test getting sampling client (mock mode returns None)."""
        client = trainer.get_sampling_client()
        # In mock mode, client is None
        assert client is None


class TestCustomTrainer:
    """Tests for custom trainer implementations."""

    def test_custom_format_example(self):
        """Test creating a custom trainer with custom format_example."""

        class CustomTrainer(TinkerTrainer):
            def format_example(self, row):
                return TrainingExample.from_dict_messages(
                    [
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["answer"]},
                    ]
                )

        config = TinkerTrainerConfig(base_model="Qwen3-8B", epochs=1)
        trainer = CustomTrainer(config)

        example = trainer.format_example({"question": "What is Python?", "answer": "A language."})
        assert len(example.messages) == 3
        assert example.messages[0].role == "system"
        assert example.messages[0].content == "You are a helpful coding assistant."

    def test_custom_trainer_training(self):
        """Test training with custom trainer."""

        class CustomTrainer(TinkerTrainer):
            def format_example(self, row):
                return TrainingExample.from_dict_messages(
                    [
                        {"role": "user", "content": row["q"]},
                        {"role": "assistant", "content": row["a"]},
                    ]
                )

        data = [
            {"q": "What is 2+2?", "a": "4"},
            {"q": "What is 3+3?", "a": "6"},
        ]

        config = TinkerTrainerConfig(base_model="Qwen3-8B", epochs=1, batch_size=2)
        trainer = CustomTrainer(config)
        result = trainer.train(data)

        assert result.samples_processed == 2
