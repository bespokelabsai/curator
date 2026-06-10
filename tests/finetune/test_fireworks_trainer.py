"""Tests for FireworksTrainer."""

from types import SimpleNamespace

import pytest

from bespokelabs.curator.finetune.config import FireworksTrainerConfig
from bespokelabs.curator.finetune.trainer import FireworksTrainer
from bespokelabs.curator.finetune.trainer import fireworks_trainer as fireworks_trainer_module
from bespokelabs.curator.finetune.types import TrainingExample, TrainingResult


# ---------------------------------------------------------------------------
# Fakes that emulate the Fireworks build SDK (fireworks.LLM / fireworks.Dataset)
# ---------------------------------------------------------------------------
class FakeProto:
    """Stand-in for the protobuf job message."""

    def __init__(self, state_name, output_model=None, metrics_url=None):
        self.state = SimpleNamespace(name=state_name)
        self.output_model = output_model
        self.metrics_file_signed_url = metrics_url
        self.create_time = None


class FakeJob:
    """Self-advancing fake fine-tuning job."""

    def __init__(self, states, output_model, display_name="job"):
        self.name = f"accounts/mahesh/supervisedFineTuningJobs/{display_name}"
        self.url = "https://app.fireworks.ai/job"
        self.display_name = display_name
        self._states = list(states)
        self._i = 0
        self._final_output = output_model
        initial = self._states[0]
        self._proto = FakeProto(initial, output_model if initial == "COMPLETED" else None)

    @property
    def output_model(self):
        return self._proto.output_model

    def get(self):
        self._i = min(self._i + 1, len(self._states) - 1)
        name = self._states[self._i]
        self._proto = FakeProto(name, self._final_output if name == "COMPLETED" else None)
        return self


class FakeDataset:
    """Stand-in for fireworks.Dataset."""

    last_path = None

    def __init__(self, path=None):
        self.name = "accounts/mahesh/datasets/fake-dataset"
        self._path = path
        self.synced = False

    @classmethod
    def from_file(cls, path):
        cls.last_path = path
        return cls(path)

    def sync(self):
        self.synced = True


class FakeLLM:
    """Stand-in for fireworks.LLM."""

    last_init_kwargs = None
    last_sft_kwargs = None

    def __init__(self, **kwargs):
        FakeLLM.last_init_kwargs = kwargs
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self._gateway = SimpleNamespace(account_id=lambda: "mahesh", _api_key=kwargs.get("api_key"))
        self.deployment_url = "https://app.fireworks.ai/deployment"
        self.applied = False
        self.deleted = False

    def create_supervised_fine_tuning_job(self, display_name, dataset, **kwargs):
        FakeLLM.last_sft_kwargs = kwargs
        return FakeJob(["CREATING", "RUNNING", "COMPLETED"], "accounts/mahesh/models/curator-out", display_name)

    def apply(self, wait=True):
        self.applied = True
        return self

    def delete_deployment(self, ignore_checks=False, wait=True):
        self.deleted = True


def install_fake_sdk(monkeypatch, llm_cls=FakeLLM):
    """Patch the trainer module to use the fake SDK."""
    fake_fireworks = SimpleNamespace(LLM=llm_cls, Dataset=FakeDataset)
    monkeypatch.setattr(fireworks_trainer_module, "FIREWORKS_AVAILABLE", True)
    monkeypatch.setattr(fireworks_trainer_module, "fireworks", fake_fireworks)
    # keep tests fast: never actually sleep while polling
    monkeypatch.setattr(fireworks_trainer_module.time, "sleep", lambda *a, **k: None)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    return [{"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]} for i in range(4)]


# ---------------------------------------------------------------------------
# Mock mode (no SDK / no key)
# ---------------------------------------------------------------------------
class TestFireworksTrainerMockMode:
    """Tests for FireworksTrainer running without the SDK or an API key."""

    @pytest.fixture
    def config(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        return FireworksTrainerConfig(base_model="qwen3-4b", epochs=1, lora_rank=8)

    @pytest.fixture
    def trainer(self, config):
        return FireworksTrainer(config)

    def test_initialization_is_mock(self, trainer):
        assert trainer.is_mock is True
        assert trainer._is_trained is False
        assert trainer._output_model is None

    def test_repr(self, trainer):
        repr_str = repr(trainer)
        assert "qwen3-4b" in repr_str
        assert "lora_rank=8" in repr_str
        assert "trained=False" in repr_str

    def test_format_example(self, trainer, sample_data):
        example = trainer.format_example(sample_data[0])
        assert isinstance(example, TrainingExample)
        assert len(example.messages) == 2
        assert example.messages[0].role == "user"

    def test_train_returns_result(self, trainer, sample_data):
        result = trainer.train(sample_data)
        assert isinstance(result, TrainingResult)
        assert result.samples_processed == 4
        assert result.total_epochs == 1
        assert result.metadata["mock"] is True

    def test_train_updates_state(self, trainer, sample_data):
        assert trainer._is_trained is False
        trainer.train(sample_data)
        assert trainer._is_trained is True
        assert trainer._output_model is not None

    def test_sample_returns_str(self, trainer, sample_data):
        trainer.train(sample_data)
        response = trainer.sample("What is Python?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_save_weights(self, trainer, sample_data):
        trainer.train(sample_data)
        weights = trainer.save_weights("unused")
        assert weights == trainer._output_model


# ---------------------------------------------------------------------------
# Real mode against a faked SDK
# ---------------------------------------------------------------------------
class TestFireworksTrainerWithFakeSDK:
    """Tests that exercise the real code paths against a faked Fireworks SDK."""

    @pytest.fixture
    def config(self):
        return FireworksTrainerConfig(
            base_model="qwen3-4b",
            epochs=2,
            lora_rank=8,
            learning_rate=1e-4,
            api_key="fw-test-key",
            poll_interval_seconds=1,
        )

    def test_initialize_client_builds_base_llm(self, monkeypatch, config):
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)
        assert trainer.is_mock is False
        assert trainer._account_id == "mahesh"
        assert FakeLLM.last_init_kwargs["model"] == "accounts/fireworks/models/qwen3-4b"
        assert FakeLLM.last_init_kwargs["deployment_type"] == "on-demand"

    def test_train_full_flow(self, monkeypatch, config, sample_data):
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)
        result = trainer.train(sample_data)
        assert trainer._is_trained is True
        assert result.weights_name == "accounts/mahesh/models/curator-out"
        assert result.samples_processed == 4
        assert result.metadata["provider"] == "fireworks"
        # dataset was written and uploaded
        assert FakeDataset.last_path.endswith(".jsonl")
        # hyperparameters were forwarded
        assert FakeLLM.last_sft_kwargs["epochs"] == 2
        assert FakeLLM.last_sft_kwargs["lora_rank"] == 8
        assert FakeLLM.last_sft_kwargs["learning_rate"] == 1e-4

    def test_train_omits_none_hyperparameters(self, monkeypatch, sample_data):
        install_fake_sdk(monkeypatch)
        config = FireworksTrainerConfig(base_model="qwen3-4b", api_key="k", learning_rate=None, batch_size=None, poll_interval_seconds=1)
        trainer = FireworksTrainer(config)
        trainer.train(sample_data)
        assert "learning_rate" not in FakeLLM.last_sft_kwargs
        assert "batch_size" not in FakeLLM.last_sft_kwargs

    def test_create_job_filters_unsupported_kwargs(self, monkeypatch, sample_data):
        """SFT args unsupported by an older SDK signature are dropped, not passed."""

        class NarrowLLM(FakeLLM):
            def create_supervised_fine_tuning_job(self, display_name, dataset, epochs=None, lora_rank=None):
                NarrowLLM.last_sft_kwargs = {"epochs": epochs, "lora_rank": lora_rank}
                return FakeJob(["COMPLETED"], "accounts/mahesh/models/curator-out", display_name)

        install_fake_sdk(monkeypatch, llm_cls=NarrowLLM)
        config = FireworksTrainerConfig(base_model="qwen3-4b", api_key="k", learning_rate=1e-4, batch_size=128, poll_interval_seconds=1)
        trainer = FireworksTrainer(config)
        result = trainer.train(sample_data)  # should not raise despite unsupported kwargs
        assert result.weights_name == "accounts/mahesh/models/curator-out"

    def test_job_state_name(self):
        job = FakeJob(["RUNNING"], None)
        assert FireworksTrainer._job_state_name(job) == "RUNNING"
        completed = FakeJob(["COMPLETED"], "m")
        completed._proto = FakeProto("COMPLETED", "m")
        assert FireworksTrainer._job_state_name(completed) == "COMPLETED"

    def test_job_state_name_handles_integer_and_prefixed_states(self):
        """The SDK stores _proto.state as a raw int; REST uses JOB_STATE_ prefixes."""
        # Raw integer enum values (the real SDK representation).
        assert FireworksTrainer._job_state_name(SimpleNamespace(_proto=SimpleNamespace(state=2))) == "RUNNING"
        assert FireworksTrainer._job_state_name(SimpleNamespace(_proto=SimpleNamespace(state=3))) == "COMPLETED"
        assert FireworksTrainer._job_state_name(SimpleNamespace(_proto=SimpleNamespace(state=4))) == "FAILED"
        # REST-style prefixed string is normalized.
        assert FireworksTrainer._job_state_name(SimpleNamespace(_proto=SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_COMPLETED")))) == "COMPLETED"
        # Missing proto/state -> UNKNOWN.
        assert FireworksTrainer._job_state_name(SimpleNamespace(_proto=None)) == "UNKNOWN"

    def test_poll_job_completes_on_integer_state(self, monkeypatch, config):
        """_poll_job must detect completion when state is the raw int 3 (COMPLETED)."""
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)

        class IntStateJob:
            def __init__(self, states):
                self._states = states
                self._i = 0
                self._proto = SimpleNamespace(state=states[0])
                self.display_name = "j"

            @property
            def output_model(self):
                return "accounts/mahesh/models/m" if self._proto.state == 3 else None

            def get(self):
                self._i = min(self._i + 1, len(self._states) - 1)
                self._proto = SimpleNamespace(state=self._states[self._i])
                return self

        done = trainer._poll_job(IntStateJob([1, 2, 3]))  # CREATING -> RUNNING -> COMPLETED
        assert done.output_model == "accounts/mahesh/models/m"

    def test_poll_job_raises_on_failure(self, monkeypatch, config):
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)
        failing = FakeJob(["RUNNING", "FAILED"], None)
        with pytest.raises(RuntimeError):
            trainer._poll_job(failing)

    def test_sample_real_path(self, monkeypatch, config, sample_data):
        install_fake_sdk(monkeypatch)

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                msg = SimpleNamespace(content="Paris is the capital of France.")
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        class FakeOpenAI:
            def __init__(self, base_url=None, api_key=None):
                captured["base_url"] = base_url
                self.chat = SimpleNamespace(completions=FakeCompletions())

        import openai

        monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

        trainer = FireworksTrainer(config)
        trainer.train(sample_data)
        response = trainer.sample("What is the capital of France?")
        assert response == "Paris is the capital of France."
        # LoRA inference must route through the dedicated deployment: model#deployment
        assert captured["model"].startswith("accounts/mahesh/models/curator-out#")
        assert "/deployments/" in captured["model"]
        assert captured["base_url"] == config.inference_base_url
        # deployment was provisioned for inference
        assert trainer._deployment_ready is True

    def test_deploy_and_delete(self, monkeypatch, config, sample_data):
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)
        trainer.train(sample_data)
        deployment = trainer.deploy()
        assert deployment.applied is True
        trainer.delete_deployment()
        assert trainer._deployment_ready is False

    def test_close_always_tears_down_deployment(self, monkeypatch, config, sample_data):
        """close() must release the billable deployment unconditionally (P1)."""
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)
        trainer.train(sample_data)
        trainer.deploy()
        assert trainer._deployment_ready is True
        trainer.close()
        assert trainer._deployment_ready is False
        assert trainer._deployment_llm is None

    def test_context_manager_tears_down_deployment(self, monkeypatch, config, sample_data):
        """Using the trainer as a context manager tears down the deployment on exit."""
        install_fake_sdk(monkeypatch)
        with FireworksTrainer(config) as trainer:
            trainer.train(sample_data)
            trainer.deploy()
            assert trainer._deployment_ready is True
        assert trainer._deployment_ready is False
        assert trainer._deployment_llm is None

    def test_failed_retrain_resets_trained_state(self, monkeypatch, config, sample_data):
        """A failed re-train must not leave stale trained-model state behind (P2)."""
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)
        trainer.train(sample_data)
        assert trainer._is_trained is True
        assert trainer._output_model == "accounts/mahesh/models/curator-out"

        def boom(_dataset_obj):
            raise RuntimeError("fireworks job failed")

        monkeypatch.setattr(trainer, "_create_job", boom)
        with pytest.raises(RuntimeError):
            trainer.train(sample_data)
        # The previous run's model must NOT be retained after a failed retrain.
        assert trainer._is_trained is False
        assert trainer._output_model is None

    def test_sample_in_real_mode_without_training_raises(self, monkeypatch, config):
        """In real mode, sampling without a trained model must raise, not fabricate (P3)."""
        install_fake_sdk(monkeypatch)
        trainer = FireworksTrainer(config)
        assert trainer.is_mock is False
        assert trainer._is_trained is False
        with pytest.raises(RuntimeError):
            trainer.sample("What is the capital of France?")


class TestCustomFireworksTrainer:
    """Tests for custom trainer subclasses."""

    def test_custom_format_example(self):
        class CustomTrainer(FireworksTrainer):
            def format_example(self, row):
                return TrainingExample.from_dict_messages(
                    [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["answer"]},
                    ]
                )

        config = FireworksTrainerConfig(base_model="qwen3-4b")
        trainer = CustomTrainer(config)
        example = trainer.format_example({"question": "What is Python?", "answer": "A language."})
        assert len(example.messages) == 2
        assert example.messages[0].content == "What is Python?"
