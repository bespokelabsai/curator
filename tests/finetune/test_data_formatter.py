"""Tests for data formatter."""

import pytest

from bespokelabs.curator.finetune.data_formatter import DataFormatter
from bespokelabs.curator.finetune.types import ChatMessage, TrainingExample


class TestDataFormatter:
    """Tests for DataFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create a test data formatter."""
        return DataFormatter(max_seq_length=1024)

    def test_format_chat_messages(self, formatter):
        """Test formatting chat messages from dicts."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = formatter.format_chat_messages(messages)
        assert len(result) == 2
        assert all(isinstance(m, ChatMessage) for m in result)
        assert result[0].role == "user"
        assert result[1].content == "Hi there!"

    def test_format_example_with_messages_key(self, formatter):
        """Test formatting example with 'messages' key."""
        row = {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "A programming language."},
            ]
        }
        example = formatter.format_example(row)
        assert isinstance(example, TrainingExample)
        assert len(example.messages) == 2

    def test_format_example_with_chat_message_objects(self, formatter):
        """Test formatting example with ChatMessage objects."""
        row = {
            "messages": [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
            ]
        }
        example = formatter.format_example(row)
        assert isinstance(example, TrainingExample)
        assert len(example.messages) == 2

    def test_format_example_missing_messages_key(self, formatter):
        """Test formatting example without 'messages' key raises error."""
        row = {"question": "What is Python?", "answer": "A language."}
        with pytest.raises(ValueError, match="Cannot format row"):
            formatter.format_example(row)

    def test_to_tinker_datum_structure(self, formatter):
        """Test Tinker Datum structure."""
        example = TrainingExample.from_dict_messages(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        datum = formatter.to_tinker_datum(example)

        assert "model_input" in datum
        assert "loss_fn_inputs" in datum
        assert "metadata" in datum
        assert "target_tokens" in datum["loss_fn_inputs"]
        assert "weights" in datum["loss_fn_inputs"]
        assert "original_text" in datum["metadata"]
        assert "num_messages" in datum["metadata"]
        assert datum["metadata"]["num_messages"] == 2

    def test_to_tinker_datum_with_system_message(self, formatter):
        """Test Tinker Datum with system message."""
        example = TrainingExample.from_dict_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )
        datum = formatter.to_tinker_datum(example)

        assert datum["metadata"]["num_messages"] == 3
        assert "<|system|>" in datum["metadata"]["original_text"]

    def test_format_batch(self, formatter):
        """Test formatting a batch of examples."""
        examples = [
            TrainingExample.from_dict_messages(
                [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ]
            ),
            TrainingExample.from_dict_messages(
                [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            ),
        ]
        batch = formatter.format_batch(examples)

        assert len(batch) == 2
        assert all("model_input" in d for d in batch)
        assert all("loss_fn_inputs" in d for d in batch)

    def test_max_seq_length(self):
        """Test max sequence length is respected."""
        short_formatter = DataFormatter(max_seq_length=50)
        long_content = "x" * 1000

        example = TrainingExample.from_dict_messages(
            [
                {"role": "user", "content": long_content},
                {"role": "assistant", "content": "Response"},
            ]
        )
        datum = short_formatter.to_tinker_datum(example)

        # Mock tokenization uses char count / 4, capped at max_seq_length
        assert len(datum["model_input"]) <= 50
