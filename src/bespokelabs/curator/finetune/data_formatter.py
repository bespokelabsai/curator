"""Data formatter for converting datasets to Tinker Datum format."""

from typing import Any, Dict, List, Optional

from bespokelabs.curator.finetune.types import ChatMessage, TrainingExample

try:
    import tinker

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False


class DataFormatter:
    """Converts curator datasets to Tinker Datum format."""

    def __init__(self, max_seq_length: int = 2048, train_on_assistant_only: bool = True):
        """Initialize the data formatter.

        Args:
            max_seq_length: Maximum sequence length for training
            train_on_assistant_only: If True, only compute loss on assistant responses
        """
        self.max_seq_length = max_seq_length
        self.train_on_assistant_only = train_on_assistant_only

    def format_chat_messages(self, messages: List[Dict[str, str]]) -> List[ChatMessage]:
        """Convert raw message dicts to ChatMessage objects.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            List of ChatMessage objects
        """
        return [ChatMessage(**msg) for msg in messages]

    def format_example(self, row: Dict[str, Any]) -> TrainingExample:
        """Format a single dataset row into a TrainingExample.

        This is the default implementation that expects a 'messages' key.
        Override this method for custom data formats.

        Args:
            row: A dictionary containing the training data

        Returns:
            TrainingExample with formatted messages
        """
        if "messages" in row:
            messages = row["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict):
                    return TrainingExample.from_dict_messages(messages)
                elif isinstance(messages[0], ChatMessage):
                    return TrainingExample(messages=messages)
        raise ValueError(f"Cannot format row: expected 'messages' key with list of chat messages, got {row.keys()}")

    def _compute_weights(self, messages: List[ChatMessage], tokens: List[int], tokenizer: Any) -> List[float]:
        """Compute loss weights for tokens based on which ones are assistant responses.

        Args:
            messages: List of chat messages
            tokens: Tokenized input
            tokenizer: Tokenizer to use for finding boundaries

        Returns:
            List of weights (1.0 for assistant tokens, 0.0 for others)
        """
        if not self.train_on_assistant_only:
            return [1.0] * len(tokens)

        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        weights = [0.0] * len(tokens)

        try:
            cumulative_messages = []

            for msg in message_dicts:
                cumulative_messages.append(msg)

                if msg["role"] == "assistant":
                    # Tokenize up to before this assistant message
                    pre_assistant = cumulative_messages[:-1]
                    if pre_assistant:
                        pre_text = tokenizer.apply_chat_template(pre_assistant, tokenize=False, add_generation_prompt=True)
                        pre_tokens = tokenizer.encode(pre_text, add_special_tokens=False)
                        start_idx = len(pre_tokens)
                    else:
                        start_idx = 0

                    # Tokenize including this assistant message
                    curr_text = tokenizer.apply_chat_template(cumulative_messages, tokenize=False, add_generation_prompt=False)
                    curr_tokens = tokenizer.encode(curr_text, add_special_tokens=False)
                    end_idx = len(curr_tokens)

                    # Set weights for assistant tokens
                    for i in range(start_idx, min(end_idx, len(weights))):
                        weights[i] = 1.0

        except Exception:
            # If boundary detection fails, fall back to training on all tokens
            weights = [1.0] * len(tokens)

        return weights

    def to_tinker_datum(self, example: TrainingExample, tokenizer: Optional[Any] = None) -> Any:
        """Convert a TrainingExample to Tinker Datum format.

        Args:
            example: The training example to convert
            tokenizer: Tokenizer for creating model_input (required for real training)

        Returns:
            Tinker Datum object or dictionary in Tinker Datum format
        """
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in example.messages]

        if tokenizer is not None:
            # Use the tokenizer's chat template
            chat_text = tokenizer.apply_chat_template(message_dicts, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(chat_text, max_length=self.max_seq_length, truncation=True)

            weights = self._compute_weights(example.messages, tokens, tokenizer)
        else:
            chat_text = ""
            for msg in example.messages:
                chat_text += f"<|{msg.role}|>\n{msg.content}\n"
            mock_token_count = min(len(chat_text) // 4, self.max_seq_length)
            tokens = list(range(mock_token_count))
            weights = [1.0] * len(tokens)

        if TINKER_AVAILABLE and tokenizer is not None:
            model_input = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=tokens)])
            datum = tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": list(tokens),
                    "weights": weights,
                },
            )
            return datum

        return {
            "model_input": tokens,
            "loss_fn_inputs": {
                "target_tokens": tokens,
                "weights": weights,
            },
            "metadata": {
                "original_text": chat_text if tokenizer is None else "",
                "num_messages": len(example.messages),
            },
        }

    def format_batch(self, examples: List[TrainingExample], tokenizer: Optional[Any] = None) -> List[Any]:
        """Format a batch of examples into Tinker Datum format.

        Args:
            examples: List of TrainingExample objects
            tokenizer: Tokenizer for creating model_input

        Returns:
            List of Tinker Datum objects or dictionaries
        """
        return [self.to_tinker_datum(example, tokenizer) for example in examples]
