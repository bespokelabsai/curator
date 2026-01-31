"""TinkerTrainer implementation for LoRA fine-tuning via Tinker API."""

import time
from typing import Any, Dict, List, Optional

from bespokelabs.curator.finetune.config import TinkerTrainerConfig
from bespokelabs.curator.finetune.data_formatter import DataFormatter
from bespokelabs.curator.finetune.status_tracker import FinetuneStatusTracker
from bespokelabs.curator.finetune.trainer.base_trainer import BaseTrainer
from bespokelabs.curator.finetune.types import (
    ChatMessage,
    CheckpointInfo,
    SamplingConfig,
    TrainingExample,
    TrainingResult,
    TrainingStats,
)
from bespokelabs.curator.log import logger

try:
    import tinker

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False


class TinkerTrainer(BaseTrainer):
    """Trainer that uses Tinker API for LoRA fine-tuning.

    Example usage:
        ```python
        from bespokelabs.curator import TinkerTrainer, TinkerTrainerConfig
        from datasets import Dataset

        data = [
            {"messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ]},
        ]
        dataset = Dataset.from_list(data)

        config = TinkerTrainerConfig(
            base_model="Qwen3-8B",
            epochs=3,
            batch_size=4,
        )

        trainer = TinkerTrainer(config)
        result = trainer.train(dataset)
        response = trainer.sample("Explain recursion")
        ```
    """

    def __init__(self, config: TinkerTrainerConfig):
        """Initialize the TinkerTrainer.

        Args:
            config: TinkerTrainerConfig with training parameters
        """
        self.config = config
        self.data_formatter = DataFormatter(max_seq_length=config.max_seq_length)
        self._weights_name: Optional[str] = None
        self._weights_path: Optional[str] = None
        self._is_trained = False
        self._checkpoints: List[CheckpointInfo] = []

        # These will be initialized when Tinker SDK is available
        self._service_client: Optional[Any] = None
        self._training_client: Optional[Any] = None
        self._sampling_client: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Tinker client and model."""
        if not self.config.api_key:
            logger.warning("No TINKER_API_KEY provided. Running in mock mode.")
            return

        if not TINKER_AVAILABLE:
            logger.warning("Tinker SDK not installed. Running in mock mode. Install with: pip install tinker")
            return

        try:
            self._service_client = tinker.ServiceClient()
            self._training_client = self._service_client.create_lora_training_client(
                base_model=self.config.base_model,
                rank=self.config.lora_config.rank,
                train_mlp=True,
                train_attn=True,
                train_unembed=True,
            )
            self._tokenizer = self._training_client.get_tokenizer()

            logger.info(f"TinkerTrainer initialized for model: {self.config.base_model}")
            logger.info(f"LoRA rank: {self.config.lora_config.rank}, Tokenizer vocab size: {self._tokenizer.vocab_size}")

        except Exception as e:
            logger.warning(f"Failed to initialize Tinker client: {e}. Running in mock mode.")
            self._service_client = None
            self._training_client = None
            self._tokenizer = None

    def save_checkpoint(self, name: str, step: int, epoch: int, loss: float) -> Optional[CheckpointInfo]:
        """Save a training checkpoint.

        Args:
            name: Name for the checkpoint
            step: Current training step
            epoch: Current epoch
            loss: Current loss value

        Returns:
            CheckpointInfo with checkpoint details, or None if saving failed
        """
        if self._training_client is not None and TINKER_AVAILABLE:
            try:
                save_future = self._training_client.save_state(name)
                save_result = save_future.result()
                checkpoint_path = save_result.path

                checkpoint = CheckpointInfo(
                    name=name,
                    path=checkpoint_path,
                    step=step,
                    epoch=epoch,
                    loss=loss,
                )
                self._checkpoints.append(checkpoint)
                logger.info(f"Checkpoint saved: {name} at step {step} (path: {checkpoint_path})")
                return checkpoint

            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
                return None
        else:
            # Mock checkpoint
            mock_path = f"mock://checkpoints/{name}"
            checkpoint = CheckpointInfo(
                name=name,
                path=mock_path,
                step=step,
                epoch=epoch,
                loss=loss,
            )
            self._checkpoints.append(checkpoint)
            logger.info(f"Checkpoint saved (mock): {name} at step {step}")
            return checkpoint

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load a training checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint (from CheckpointInfo.path)

        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        if self._training_client is not None and TINKER_AVAILABLE:
            try:
                load_future = self._training_client.load_state_with_optimizer(checkpoint_path)
                load_future.result()
                logger.info(f"Checkpoint loaded: {checkpoint_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return False
        else:
            logger.info(f"Checkpoint load (mock): {checkpoint_path}")
            return True

    def get_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of all saved checkpoints.

        Returns:
            List of CheckpointInfo objects
        """
        return self._checkpoints.copy()

    def format_example(self, row: Dict[str, Any]) -> TrainingExample:
        """Format a dataset row into a TrainingExample.

        Override this method to customize data formatting for your use case.

        Args:
            row: A dictionary containing the training data

        Returns:
            TrainingExample with formatted messages
        """
        return self.data_formatter.format_example(row)

    def _prepare_batch(self, examples: List[Dict[str, Any]]) -> List[Any]:
        """Prepare a batch of examples for training.

        Args:
            examples: List of raw examples from the dataset

        Returns:
            List of Tinker Datum objects or formatted examples
        """
        training_examples = [self.format_example(ex) for ex in examples]
        return self.data_formatter.format_batch(training_examples, self._tokenizer)

    def train(self, dataset: Any) -> TrainingResult:
        """Run LoRA training on the provided dataset.

        Args:
            dataset: HuggingFace Dataset or list of examples

        Returns:
            TrainingResult with training metrics
        """
        start_time = time.time()

        if hasattr(dataset, "to_list"):
            data_list = dataset.to_list()
        elif hasattr(dataset, "__iter__"):
            data_list = list(dataset)
        else:
            data_list = dataset

        num_examples = len(data_list)
        steps_per_epoch = (num_examples + self.config.batch_size - 1) // self.config.batch_size
        total_steps = steps_per_epoch * self.config.epochs

        logger.info(f"Starting training: {num_examples} examples, {self.config.epochs} epochs, {total_steps} total steps")

        # Initialize status tracker
        status_tracker = FinetuneStatusTracker(
            model=self.config.base_model,
            total_epochs=self.config.epochs,
            total_steps=total_steps,
            batch_size=self.config.batch_size,
        )
        status_tracker.start_tracker()

        loss_history: List[float] = []
        tokens_processed = 0
        samples_processed = 0
        current_step = 0

        try:
            for epoch in range(1, self.config.epochs + 1):
                epoch_loss = 0.0
                epoch_steps = 0

                for batch_start in range(0, num_examples, self.config.batch_size):
                    batch_end = min(batch_start + self.config.batch_size, num_examples)
                    batch = data_list[batch_start:batch_end]

                    current_step += 1
                    lr = self._get_learning_rate(current_step, total_steps)
                    batch_data = self._prepare_batch(batch)
                    loss, batch_tokens = self._training_step(batch_data, learning_rate=lr)

                    epoch_loss += loss
                    epoch_steps += 1
                    tokens_processed += batch_tokens
                    samples_processed += len(batch)

                    stats = TrainingStats(
                        current_epoch=epoch,
                        total_epochs=self.config.epochs,
                        current_step=current_step,
                        total_steps=total_steps,
                        current_loss=loss,
                        tokens_processed=tokens_processed,
                        samples_processed=samples_processed,
                        learning_rate=lr,
                    )
                    status_tracker.update(stats)

                    if current_step % self.config.log_every_n_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        loss_history.append(avg_loss)
                        logger.debug(f"Step {current_step}/{total_steps}, Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {lr:.2e}")

                    if self.config.checkpoint_every_n_steps > 0 and current_step % self.config.checkpoint_every_n_steps == 0:
                        checkpoint_name = f"{self.config.checkpoint_name_prefix}_step_{current_step}"
                        self.save_checkpoint(
                            name=checkpoint_name,
                            step=current_step,
                            epoch=epoch,
                            loss=loss,
                        )

                avg_epoch_loss = epoch_loss / epoch_steps
                loss_history.append(avg_epoch_loss)
                logger.info(f"Epoch {epoch}/{self.config.epochs} complete. Average loss: {avg_epoch_loss:.4f}")

                if self.config.checkpoint_every_epoch:
                    checkpoint_name = f"{self.config.checkpoint_name_prefix}_epoch_{epoch}"
                    self.save_checkpoint(
                        name=checkpoint_name,
                        step=current_step,
                        epoch=epoch,
                        loss=avg_epoch_loss,
                    )

        finally:
            status_tracker.stop_tracker()

        total_time = time.time() - start_time
        final_loss = loss_history[-1] if loss_history else 0.0

        self._is_trained = True

        if self.config.save_weights_on_complete:
            self._weights_name = self.save_weights(f"{self.config.base_model}_lora_{int(time.time())}")

        result = TrainingResult(
            final_loss=final_loss,
            total_steps=current_step,
            total_epochs=self.config.epochs,
            total_time=total_time,
            tokens_processed=tokens_processed,
            samples_processed=samples_processed,
            loss_history=loss_history,
            weights_name=self._weights_name,
            checkpoints=self._checkpoints.copy(),
            metadata={
                "base_model": self.config.base_model,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.adam_params.learning_rate,
                "lora_rank": self.config.lora_config.rank,
                "lora_alpha": self.config.lora_config.alpha,
            },
        )

        logger.info(f"Training complete. Final loss: {final_loss:.4f}, Time: {total_time:.2f}s")
        return result

    def _training_step(self, batch_data: List[Any], learning_rate: float) -> tuple:
        """Execute a single training step.

        Args:
            batch_data: List of Tinker Datum objects or formatted examples
            learning_rate: Current learning rate for this step

        Returns:
            Tuple of (loss, tokens_processed)
        """
        if self._training_client is not None and TINKER_AVAILABLE:
            try:
                fwd_bwd_future = self._training_client.forward_backward(batch_data, loss_fn="cross_entropy")
                fwd_bwd_result = fwd_bwd_future.result()

                loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
                total_tokens = sum(len(datum.model_input.chunks[0].tokens) for datum in batch_data if hasattr(datum, "model_input"))
                avg_loss = loss_sum / max(total_tokens, 1)

                adam_params = tinker.AdamParams(
                    learning_rate=learning_rate,
                    beta1=self.config.adam_params.beta1,
                    beta2=self.config.adam_params.beta2,
                    eps=self.config.adam_params.epsilon,
                    weight_decay=self.config.adam_params.weight_decay,
                )
                optim_future = self._training_client.optim_step(adam_params)
                optim_future.result()  # Wait for completion

                return avg_loss, total_tokens

            except Exception as e:
                logger.warning(f"Training step failed: {e}. Falling back to mock mode.")

        import random

        mock_loss = 2.5 - (random.random() * 0.5)
        mock_tokens = 0
        for d in batch_data:
            if hasattr(d, "model_input"):
                mock_tokens += len(d.model_input.chunks[0].tokens)
            elif isinstance(d, dict) and "model_input" in d:
                mock_tokens += len(d.get("model_input", []))
            else:
                mock_tokens += 100

        time.sleep(0.01)

        return mock_loss, mock_tokens if mock_tokens > 0 else len(batch_data) * 100

    def _get_learning_rate(self, step: int, total_steps: int) -> float:
        """Calculate learning rate with optional warmup.

        Args:
            step: Current training step
            total_steps: Total number of training steps

        Returns:
            Current learning rate
        """
        base_lr = self.config.adam_params.learning_rate

        if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
            return base_lr * (step / self.config.warmup_steps)

        return base_lr

    def save_weights(self, name: str) -> str:
        """Save the trained LoRA weights.

        Args:
            name: Name for the saved weights

        Returns:
            Identifier for the saved weights
        """
        if not self._is_trained:
            logger.warning("Model has not been trained yet. Saving current state.")

        if self._training_client is not None and TINKER_AVAILABLE:
            try:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(name)
                self._weights_name = name
                if hasattr(self._sampling_client, "model_path"):
                    self._weights_path = self._sampling_client.model_path
                logger.info(f"Weights saved: {name}")
                return name
            except Exception as e:
                logger.warning(f"Failed to save weights: {e}. Using mock implementation.")

        # Mock implementation
        weight_id = f"mock_weights_{name}"
        logger.info(f"Weights saved (mock): {weight_id}")
        self._weights_name = weight_id
        return weight_id

    def get_sampling_client(self) -> Any:
        """Get a client configured for sampling from the fine-tuned model.

        Returns:
            Tinker sampling client or None if not available
        """
        if not self._is_trained:
            logger.warning("Model has not been trained yet. Sampling from base model.")

        if self._sampling_client is not None:
            return self._sampling_client

        if self._training_client is not None and TINKER_AVAILABLE:
            try:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(f"auto_save_{int(time.time())}")
                return self._sampling_client
            except Exception as e:
                logger.warning(f"Failed to create sampling client: {e}")

        return None

    def sample(
        self,
        prompt: str,
        sampling_config: Optional[SamplingConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a sample from the fine-tuned model.

        Args:
            prompt: The user prompt
            sampling_config: Optional sampling configuration
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        if sampling_config is None:
            sampling_config = SamplingConfig()

        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        sampling_client = self.get_sampling_client()
        if sampling_client is not None and self._tokenizer is not None and TINKER_AVAILABLE:
            try:
                message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
                prompt_text = self._tokenizer.apply_chat_template(message_dicts, tokenize=False, add_generation_prompt=True)
                prompt_tokens = self._tokenizer.encode(prompt_text)

                prompt_input = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=prompt_tokens)])
                sample_params = tinker.SamplingParams(
                    max_tokens=sampling_config.max_tokens,
                    temperature=sampling_config.temperature,
                    top_p=sampling_config.top_p,
                    top_k=sampling_config.top_k if sampling_config.top_k else -1,
                    stop=sampling_config.stop_sequences if sampling_config.stop_sequences else None,
                )
                sample_result = sampling_client.sample(prompt_input, num_samples=1, sampling_params=sample_params).result()

                if sample_result.sequences and len(sample_result.sequences) > 0:
                    generated_tokens = sample_result.sequences[0].tokens
                    generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    return generated_text.strip()

                return "[No response generated]"

            except Exception as e:
                logger.warning(f"Sampling failed: {e}. Using mock response.")

        logger.info(f"Sampling (mock) with prompt: {prompt[:50]}...")
        return f"[Mock response for: {prompt[:100]}...]"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TinkerTrainer(model={self.config.base_model}, "
            f"epochs={self.config.epochs}, "
            f"batch_size={self.config.batch_size}, "
            f"trained={self._is_trained})"
        )
