"""FireworksTrainer implementation for managed fine-tuning via the Fireworks AI API.

Unlike :class:`TinkerTrainer`, which drives a low-level forward/backward training
loop, Fireworks AI exposes *managed* fine-tuning: you upload a chat-format JSONL
dataset, submit a supervised fine-tuning (SFT) job against a base model, poll the
job until it completes, and then serve the resulting (LoRA) model from an on-demand
deployment for inference. This trainer wraps that flow behind the same
:class:`BaseTrainer` interface used by the rest of the ``finetune`` module.

The official Fireworks "build SDK" (``pip install fireworks-ai``) is used for the
control plane (dataset upload, job submission, deployment), and the OpenAI-compatible
inference endpoint is used for sampling. When the SDK is not installed or no API key
is available, the trainer runs in "mock mode" so examples and tests work offline.
"""

import inspect
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

from bespokelabs.curator.finetune.config import FireworksTrainerConfig
from bespokelabs.curator.finetune.fireworks_data_formatter import FireworksDataFormatter
from bespokelabs.curator.finetune.trainer.base_trainer import BaseTrainer
from bespokelabs.curator.finetune.types import (
    SamplingConfig,
    TrainingExample,
    TrainingResult,
)
from bespokelabs.curator.log import logger

fireworks = None
try:
    import fireworks as _fireworks

    required_fireworks_attrs = ("LLM", "Dataset")
    if all(hasattr(_fireworks, attr) for attr in required_fireworks_attrs):
        fireworks = _fireworks
        FIREWORKS_AVAILABLE = True
    else:
        FIREWORKS_AVAILABLE = False
except Exception:  # noqa: BLE001 - optional dep; never let a broken install break curator import
    FIREWORKS_AVAILABLE = False


# Terminal job states reported by the Fireworks API (normalized JobState names; a
# leading "JOB_STATE_" prefix from the REST API is stripped before comparison).
# Intermediate states (CREATING, RUNNING, VALIDATING, WRITING_RESULTS, PENDING,
# EVALUATION, ROLLOUT, etc.) are not listed here and simply continue polling.
# Default accelerator for on-demand inference deployments when none is configured.
_DEFAULT_DEPLOY_ACCELERATOR = "NVIDIA_H100_80GB"

# Fireworks JobState enum values (the SDK stores `_proto.state` as a raw int).
_JOB_STATE_NAMES_BY_VALUE = {
    0: "UNSPECIFIED",
    1: "CREATING",
    2: "RUNNING",
    3: "COMPLETED",
    4: "FAILED",
    5: "CANCELLED",
    6: "DELETING",
    7: "WRITING_RESULTS",
    8: "VALIDATING",
    9: "ROLLOUT",
    10: "EVALUATION",
    11: "FAILED_CLEANING_UP",
    12: "DELETING_CLEANING_UP",
    13: "POLICY_UPDATE",
    14: "PENDING",
    15: "EXPIRED_CLEANING_UP",
    16: "EXPIRED",
    17: "CREATING_DEPENDENCIES",
    18: "RE_QUEUEING",
    19: "CREATING_INPUT_DATASET",
}

_COMPLETED_STATES = {"COMPLETED"}
_FAILED_STATES = {
    "FAILED",
    "FAILED_CLEANING_UP",
    "CANCELLED",
    "DELETING",
    "DELETING_CLEANING_UP",
    "EXPIRED",
    "EXPIRED_CLEANING_UP",
}


class FireworksTrainer(BaseTrainer):
    """Trainer that uses the Fireworks AI managed fine-tuning API.

    Example usage:
        ```python
        from bespokelabs.curator import FireworksTrainer, FireworksTrainerConfig
        from datasets import Dataset

        data = [
            {"messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ]},
        ]
        dataset = Dataset.from_list(data * 3)  # Fireworks requires >= 3 examples

        config = FireworksTrainerConfig(
            base_model="qwen3-4b",
            epochs=1,
            lora_rank=8,
        )

        trainer = FireworksTrainer(config)
        result = trainer.train(dataset)        # uploads data, runs the SFT job
        response = trainer.sample("Explain recursion")  # deploys + runs inference
        trainer.close()                        # tears down the deployment (optional)
        ```
    """

    def __init__(self, config: FireworksTrainerConfig):
        """Initialize the FireworksTrainer.

        Args:
            config: FireworksTrainerConfig with training parameters.
        """
        self.config = config
        self.data_formatter = FireworksDataFormatter()
        self._output_model: Optional[str] = None
        self._job_name: Optional[str] = None
        self._job_url: Optional[str] = None
        self._is_trained = False

        # Populated when the SDK and an API key are available.
        self._base_llm: Optional[Any] = None
        self._account_id: Optional[str] = config.account_id
        self._deployment_llm: Optional[Any] = None
        self._deployment_ready = False

        self._initialize_client()

    @property
    def is_mock(self) -> bool:
        """Whether the trainer is running in mock mode (no SDK/key)."""
        return self._base_llm is None

    def _initialize_client(self) -> None:
        """Initialize the Fireworks base-model handle used to launch SFT jobs."""
        if not self.config.api_key:
            logger.warning("No FIREWORKS_API_KEY provided. Running in mock mode.")
            return

        if not FIREWORKS_AVAILABLE:
            logger.warning("Fireworks SDK not installed. Running in mock mode. Install with: pip install fireworks-ai")
            return

        try:
            self._base_llm = self._build_base_llm()
            if self._account_id is None:
                self._account_id = self._base_llm._gateway.account_id()
            logger.info(f"FireworksTrainer initialized for base model: {self.config.qualified_base_model}")
            logger.info(f"Account: {self._account_id}, LoRA rank: {self.config.lora_rank}")
        except Exception as e:
            logger.warning(f"Failed to initialize Fireworks client: {e}. Running in mock mode.")
            self._base_llm = None

    def _build_base_llm(self) -> Any:
        """Construct a (non-provisioned) LLM handle for the base model.

        The base model is only used to submit the SFT job, so the deployment is
        never applied/provisioned. An ``id`` is still required by the SDK for the
        on-demand deployment strategy.
        """
        deployment_id = self._sanitize_resource_name(f"curator-sft-{self.config.base_model}")
        return fireworks.LLM(
            model=self.config.qualified_base_model,
            deployment_type="on-demand",
            id=deployment_id,
            api_key=self.config.api_key,
        )

    @staticmethod
    def _sanitize_resource_name(name: str) -> str:
        """Sanitize a string into a valid Fireworks resource name (``[a-z0-9-]``)."""
        sanitized = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
        return sanitized or "curator"

    def _qualify_model_id(self, model_id: str) -> str:
        """Expand a bare model id into a fully-qualified ``accounts/<acct>/models/<id>``.

        The Fireworks API requires ``output_model`` to be a fully-qualified resource
        name, so a bare id from the config is expanded using the discovered account id.
        """
        if model_id.startswith("accounts/"):
            return model_id
        account = self._account_id or "fireworks"
        return f"accounts/{account}/models/{model_id}"

    @staticmethod
    def _get_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """Filter keyword arguments to those accepted by the callable (forward-compat)."""
        try:
            parameters = inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return kwargs, []

        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return kwargs, []

        supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        ignored_kwargs = [key for key in kwargs if key not in supported_kwargs]
        return supported_kwargs, ignored_kwargs

    def format_example(self, row: Dict[str, Any]) -> TrainingExample:
        """Format a dataset row into a TrainingExample.

        Override this method to customize data formatting for your use case.

        Args:
            row: A dictionary containing the training data.

        Returns:
            TrainingExample with formatted messages.
        """
        return self.data_formatter.format_example(row)

    @staticmethod
    def _normalize_dataset(dataset: Any) -> List[Dict[str, Any]]:
        """Normalize a HuggingFace Dataset / iterable into a list of rows."""
        if hasattr(dataset, "to_list"):
            return dataset.to_list()
        if hasattr(dataset, "__iter__"):
            return list(dataset)
        return dataset

    def train(self, dataset: Any) -> TrainingResult:
        """Run a supervised fine-tuning job on the provided dataset.

        Args:
            dataset: HuggingFace Dataset or list of examples.

        Returns:
            TrainingResult with job metadata and the resulting model id.
        """
        start_time = time.time()
        data_list = self._normalize_dataset(dataset)
        num_examples = len(data_list)
        examples = [self.format_example(row) for row in data_list]

        if num_examples < 3:
            logger.warning(f"Fireworks fine-tuning requires at least 3 examples; got {num_examples}. The job may be rejected.")

        if self.is_mock:
            return self._mock_train(num_examples, start_time)

        # Reset all trained state up front so a failed (re)train leaves the trainer in a
        # clean "not trained" state instead of silently retaining the previous run's
        # model. Also tear down any deployment of the now-stale previous model.
        self._is_trained = False
        self._output_model = None
        self._job_name = None
        self._job_url = None
        self.delete_deployment()

        logger.info(f"Starting Fireworks SFT: {num_examples} examples, {self.config.epochs} epochs, base model {self.config.qualified_base_model}")

        tmp_path = self._write_dataset_file(examples)
        try:
            dataset_obj = fireworks.Dataset.from_file(tmp_path)
            dataset_obj.sync()
            logger.info(f"Dataset uploaded: {dataset_obj.name}")

            job = self._create_job(dataset_obj)
            self._job_name = getattr(job, "name", None)
            self._job_url = getattr(job, "url", None)
            logger.info(f"Fine-tuning job submitted: {self._job_name} ({self._job_url})")

            job = self._poll_job(job)
            self._output_model = job.output_model
            logger.info(f"Fine-tuning complete. Output model: {self._output_model}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        self._is_trained = True
        total_time = time.time() - start_time
        loss_history = self._maybe_fetch_loss_history(job)

        return TrainingResult(
            final_loss=loss_history[-1] if loss_history else 0.0,
            total_steps=0,
            total_epochs=self.config.epochs,
            total_time=total_time,
            tokens_processed=0,
            samples_processed=num_examples,
            loss_history=loss_history,
            weights_name=self._output_model,
            checkpoints=[],
            metadata={
                "provider": "fireworks",
                "base_model": self.config.qualified_base_model,
                "output_model": self._output_model,
                "job_name": self._job_name,
                "job_url": self._job_url,
                "epochs": self.config.epochs,
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
                "account_id": self._account_id,
            },
        )

    def _write_dataset_file(self, examples: List[TrainingExample]) -> str:
        """Write training examples to a ``.jsonl`` file with a stable, meaningful name.

        Fireworks derives the dataset id from the file's contents and name, so using a
        stable name (derived from the job's display name) lets repeated runs with the
        same data deduplicate to a single uploaded dataset instead of creating orphans.
        """
        filename = f"curator-{self._sanitize_resource_name(self.config.display_name)}.jsonl"
        path = os.path.join(tempfile.gettempdir(), filename)
        return self.data_formatter.write_jsonl(examples, path)

    def _create_job(self, dataset_obj: Any) -> Any:
        """Submit the supervised fine-tuning job with the configured hyperparameters."""
        candidate_kwargs = {
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "lora_rank": self.config.lora_rank,
            "batch_size": self.config.batch_size,
            "max_context_length": self.config.max_context_length,
            "early_stop": self.config.early_stop,
            # output_model must be a fully-qualified resource name for the API.
            "output_model": self._qualify_model_id(self.config.output_model) if self.config.output_model else None,
        }
        # Only forward parameters the user actually set.
        candidate_kwargs = {key: value for key, value in candidate_kwargs.items() if value is not None}
        supported_kwargs, ignored_kwargs = self._get_supported_kwargs(self._base_llm.create_supervised_fine_tuning_job, candidate_kwargs)
        if ignored_kwargs:
            logger.warning("Installed Fireworks SDK does not support SFT args %s; continuing without them.", ", ".join(ignored_kwargs))
        return self._base_llm.create_supervised_fine_tuning_job(self.config.display_name, dataset_obj, **supported_kwargs)

    @staticmethod
    def _job_state_name(job: Any) -> str:
        """Best-effort extraction of a readable job-state name from an SFT job.

        Handles three representations of ``_proto.state``: an enum with ``.name``, a
        raw integer (the SDK actually stores the enum field as an int — e.g. ``2`` for
        RUNNING, ``3`` for COMPLETED), and a string. Returns ``"UNKNOWN"`` if the
        state cannot be read. Note: a job's ``output_model`` is populated at *creation*
        time (not completion), so completion is keyed on state only, never on it.
        """
        proto = getattr(job, "_proto", None)
        state = getattr(proto, "state", None) if proto is not None else None
        if state is None:
            return "UNKNOWN"

        name = getattr(state, "name", None)
        if not name:
            # Integer (or numeric-string) enum value -> map to a name.
            try:
                name = _JOB_STATE_NAMES_BY_VALUE[int(state)]
            except (KeyError, ValueError, TypeError):
                name = str(state)

        # Normalize the REST-style "JOB_STATE_RUNNING" to the gRPC-style "RUNNING".
        return name[len("JOB_STATE_") :] if name.startswith("JOB_STATE_") else name

    def _poll_job(self, job: Any) -> Any:
        """Poll a fine-tuning job until it completes, fails, or times out.

        Completion is determined strictly by the job ``state`` (``COMPLETED``). If the
        state cannot be read from this SDK version, defer to the SDK's own waiter.
        """
        start = time.time()
        last_state: Optional[str] = None
        while time.time() - start < self.config.max_wait_seconds:
            state = self._job_state_name(job)
            if state != last_state:
                elapsed = int(time.time() - start)
                logger.info(f"Fine-tuning job '{self.config.display_name}' state: {state} (elapsed {elapsed}s)")
                last_state = state
            if state in _COMPLETED_STATES:
                return job
            if state in _FAILED_STATES:
                raise RuntimeError(f"Fireworks fine-tuning job '{self.config.display_name}' ended in state {state}")
            if state == "UNKNOWN":
                # This SDK build doesn't expose job state; fall back to its own waiter.
                if hasattr(job, "wait_for_completion"):
                    return job.wait_for_completion()
                raise RuntimeError(f"Cannot determine state for job '{self.config.display_name}' and no wait_for_completion is available")
            time.sleep(self.config.poll_interval_seconds)
            refreshed = job.get()
            if refreshed is None:
                raise RuntimeError(f"Fireworks fine-tuning job '{self.config.display_name}' could not be found")
            job = refreshed
        logger.warning(
            "Stopped waiting for fine-tuning job '%s' after %ss, but the job is STILL RUNNING server-side and "
            "continues to incur charges. Cancel or monitor it at %s",
            self.config.display_name,
            self.config.max_wait_seconds,
            self._job_url or "the Fireworks dashboard",
        )
        raise TimeoutError(
            f"Fireworks fine-tuning job '{self.config.display_name}' did not finish within {self.config.max_wait_seconds}s (still running server-side)"
        )

    def _maybe_fetch_loss_history(self, job: Any) -> List[float]:
        """Best-effort retrieval of training loss values from the job metrics file."""
        proto = getattr(job, "_proto", None)
        url = getattr(proto, "metrics_file_signed_url", None) if proto is not None else None
        if not url:
            return []
        try:
            import json

            import requests

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            losses: List[float] = []
            for line in response.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for key in ("loss", "train_loss", "training_loss"):
                    if key in record and record[key] is not None:
                        losses.append(float(record[key]))
                        break
            return losses
        except Exception as e:  # noqa: BLE001 - metrics are best-effort only
            logger.debug(f"Could not fetch loss history: {e}")
            return []

    def _mock_train(self, num_examples: int, start_time: float) -> TrainingResult:
        """Produce a mock TrainingResult when running without the SDK or a key."""
        import random

        time.sleep(0.01)
        self._is_trained = True
        self._output_model = f"mock://accounts/{self._account_id or 'mock'}/models/{self.config.display_name}"
        mock_loss = 2.5 - (random.random() * 0.5)
        logger.info(f"Training (mock) complete. Output model: {self._output_model}")
        return TrainingResult(
            final_loss=mock_loss,
            total_steps=0,
            total_epochs=self.config.epochs,
            total_time=time.time() - start_time,
            tokens_processed=0,
            samples_processed=num_examples,
            loss_history=[mock_loss],
            weights_name=self._output_model,
            checkpoints=[],
            metadata={"provider": "fireworks", "mock": True, "base_model": self.config.qualified_base_model},
        )

    def save_weights(self, name: str) -> str:
        """Return the identifier of the fine-tuned model.

        Fireworks persists the fine-tuned model server-side as part of the job, so
        there is no separate save step; this returns the resulting model id.

        Args:
            name: Unused; present for interface compatibility with BaseTrainer.

        Returns:
            The fine-tuned model id (or a mock id when untrained).
        """
        if not self._is_trained:
            logger.warning("Model has not been trained yet.")
        return self._output_model or f"mock_weights_{name}"

    def deploy(self, wait: bool = True) -> Any:
        """Provision an on-demand deployment for the fine-tuned model.

        Fine-tuned LoRA models cannot be served serverlessly, so an on-demand
        (Live Merge) deployment is required before inference.

        Args:
            wait: Whether to block until the deployment is ready.

        Returns:
            The Fireworks LLM deployment handle, or a mock id in mock mode.
        """
        if not self._is_trained or not self._output_model:
            logger.warning("Model has not been trained yet; cannot deploy.")
            return None

        if self.is_mock or self._output_model.startswith("mock://"):
            self._deployment_ready = True
            return self._output_model

        if self._deployment_llm is not None and self._deployment_ready:
            return self._deployment_llm

        deployment_id = self._sanitize_resource_name(f"{self.config.display_name}-deploy")
        # On-demand deployments require an accelerator type (unlike SFT jobs, where it
        # must be auto-selected). Fall back to a sensible default if unset.
        accelerator_type = self.config.accelerator_type or _DEFAULT_DEPLOY_ACCELERATOR
        deploy_kwargs = {
            "model": self._output_model,
            "deployment_type": self.config.deployment_type,
            "id": deployment_id,
            "api_key": self.config.api_key,
            "accelerator_type": accelerator_type,
            "accelerator_count": self.config.accelerator_count,
        }
        deploy_kwargs = {key: value for key, value in deploy_kwargs.items() if value is not None}
        supported_kwargs, _ = self._get_supported_kwargs(fireworks.LLM.__init__, deploy_kwargs)
        logger.info(f"Deploying fine-tuned model {self._output_model} ({self.config.deployment_type})...")
        self._deployment_llm = fireworks.LLM(**supported_kwargs)
        self._deployment_llm.apply(wait=wait)
        self._deployment_ready = True
        logger.info(f"Deployment ready: {getattr(self._deployment_llm, 'deployment_url', '<unknown>')}")
        return self._deployment_llm

    def get_sampling_client(self) -> Any:
        """Return the deployment handle used for sampling (deploying if needed)."""
        if not self._is_trained:
            logger.warning("Model has not been trained yet.")
            return None
        if self.is_mock:
            return None
        if not self._deployment_ready and self.config.auto_deploy:
            self.deploy()
        return self._deployment_llm

    def sample(
        self,
        prompt: str,
        sampling_config: Optional[SamplingConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a sample from the fine-tuned model via the inference endpoint.

        Args:
            prompt: The user prompt.
            sampling_config: Optional sampling configuration.
            system_prompt: Optional system prompt.

        Returns:
            Generated text.
        """
        if sampling_config is None:
            sampling_config = SamplingConfig()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Genuine mock mode (no SDK / no API key): return a clearly-fake response so
        # examples and tests run offline.
        if self.is_mock:
            logger.info(f"Sampling (mock) with prompt: {prompt[:50]}...")
            return f"[Mock response for: {prompt[:100]}...]"

        # Real mode: a fine-tuned model is required. Never fabricate a mock response
        # here — that would mask failed or skipped training. Run actual inference and
        # let failures propagate.
        if not self._is_trained or not self._output_model or self._output_model.startswith("mock://"):
            raise RuntimeError("No fine-tuned model available for inference. Call train() and ensure it completed successfully before sampling.")
        return self._sample_real(messages, sampling_config)

    def _inference_model_id(self) -> str:
        """Return the model id to use for inference, with LoRA deployment routing.

        A fine-tuned LoRA adapter served on a dedicated deployment must be addressed
        as ``<model>#<deployment>`` (the plain model id returns 404). When we created
        the deployment, append its resource name; otherwise use the bare model id.
        """
        if self._deployment_llm is None:
            return self._output_model
        deployment_name = getattr(self._deployment_llm, "deployment_name", None)
        if not deployment_name:
            deployment_id = self._sanitize_resource_name(f"{self.config.display_name}-deploy")
            deployment_name = f"accounts/{self._account_id}/deployments/{deployment_id}"
        return f"{self._output_model}#{deployment_name}"

    def _sample_real(self, messages: List[Dict[str, str]], sampling_config: SamplingConfig) -> str:
        """Run inference against the deployed fine-tuned model (with scale-up retry)."""
        from openai import OpenAI

        if self.config.auto_deploy and not self._deployment_ready:
            self.deploy()

        client = OpenAI(base_url=self.config.inference_base_url, api_key=self.config.api_key)
        request_kwargs: Dict[str, Any] = {
            "model": self._inference_model_id(),
            "messages": messages,
            "max_tokens": sampling_config.max_tokens,
            "temperature": sampling_config.temperature,
            "top_p": sampling_config.top_p,
        }
        if sampling_config.stop_sequences:
            request_kwargs["stop"] = sampling_config.stop_sequences
        if sampling_config.top_k:
            request_kwargs["extra_body"] = {"top_k": sampling_config.top_k}

        max_attempts = 6
        backoff = 5.0
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content
                return content.strip() if content else "[No response generated]"
            except Exception as e:  # noqa: BLE001 - inspect message for scale-up signal
                last_error = e
                message = str(e).lower()
                if "503" in message or "scal" in message or "not ready" in message:
                    logger.info(f"Deployment scaling up; retrying in {backoff:.0f}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                raise
        raise RuntimeError(f"Inference failed after {max_attempts} attempts: {last_error}")

    def delete_deployment(self) -> None:
        """Tear down the on-demand inference deployment, if one was created.

        Deletes the deployment *resource* directly via the gateway with
        ``ignore_checks=True`` (so a deployment that recently served requests is still
        removed — Fireworks otherwise blocks deletion for ~1 hour). This is more
        reliable than ``LLM.delete_deployment``, which takes a PEFT-addon code path and
        raises "is not a deployed model" for a dedicated on-demand deployment. Avoiding
        a leaked, billable GPU deployment matters, so we fall back across methods.
        """
        if self._deployment_llm is not None and not self.is_mock:
            if not self._gateway_delete_deployment():
                self._llm_delete_deployment()
        self._deployment_llm = None
        self._deployment_ready = False

    def _gateway_delete_deployment(self) -> bool:
        """Delete the deployment resource directly via the gateway. Returns success."""
        gateway = getattr(self._deployment_llm, "_gateway", None)
        name = getattr(self._deployment_llm, "deployment_name", None) or getattr(self._deployment_llm, "deployment_id", None)
        if gateway is None or name is None or not hasattr(gateway, "delete_deployment_sync"):
            return False
        try:
            gateway.delete_deployment_sync(name, ignore_checks=True)
            logger.info("Inference deployment deleted.")
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"gateway.delete_deployment_sync failed: {e}")
            return False

    def _llm_delete_deployment(self) -> None:
        """Fallback: delete via the LLM handle (handles SDKs without ignore_checks)."""
        try:
            self._deployment_llm.delete_deployment(ignore_checks=True)
            logger.info("Inference deployment deleted.")
        except TypeError:
            try:
                self._deployment_llm.delete_deployment()
                logger.info("Inference deployment deleted.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to delete deployment: {e}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to delete deployment: {e}")

    def close(self) -> None:
        """Release resources, tearing down the on-demand inference deployment.

        This always deletes the deployment (a billable GPU resource), so calling
        ``close()`` — or using the trainer as a context manager — is the way to avoid
        leaking it. If you want the deployment to persist, simply don't call ``close()``.
        """
        self.delete_deployment()

    def __enter__(self) -> "FireworksTrainer":
        """Enter a context that guarantees the inference deployment is torn down."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Always tear down the on-demand deployment on context exit (avoids cost leaks)."""
        self.delete_deployment()
        return False

    @property
    def output_model(self) -> Optional[str]:
        """The fully-qualified id of the fine-tuned model (after training)."""
        return self._output_model

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FireworksTrainer(base_model={self.config.qualified_base_model}, "
            f"epochs={self.config.epochs}, "
            f"lora_rank={self.config.lora_rank}, "
            f"trained={self._is_trained})"
        )
