import logging
from collections import defaultdict

import litellm

litellm.suppress_debug_info = True
logger = logging.getLogger(__name__)


class _LitellmCostProcessor:
    def __init__(self, batch=False) -> None:
        self.batch = batch

    def cost(self, *args, completion_window="*", **kwargs):
        import litellm

        model = kwargs.get("model", None)
        cost_to_complete = 0.0
        if model in litellm.model_cost:
            cost_to_complete = litellm.completion_cost(*args, **kwargs)
        if self.batch:
            cost_to_complete *= 0.5
        return cost_to_complete


# source: https://www.kluster.ai/#pricing
_EXTERNAL_MODEL_COST_MAP = {
    "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo": {
        "max_tokens": 8192,
        "cost_per_million": {"*": 0.18, "1h": 0.09, "3h": 0.08, "6h": 0.07, "12h": 0.06, "24h": 0.05},  # Input/Output
    },
    "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo": {
        "max_tokens": 8192,
        "cost_per_million": {"*": 0.70, "1h": 0.35, "3h": 0.33, "6h": 0.30, "12h": 0.25, "24h": 0.20},  # Input/Output
    },
    "klusterai/Meta-Llama-3.1-405B-Instruct-Turbo": {
        "max_tokens": 8192,
        "cost_per_million": {"*": 3.50, "1h": 1.75, "3h": 1.60, "6h": 1.45, "12h": 1.20, "24h": 0.99},  # Input/Output
    },
    "deepseek-ai/DeepSeek-R1": {
        "max_tokens": 8192,
        "cost_per_million": {"*": 2.00, "1h": 1.00, "3h": 0.90, "6h": 0.80, "12h": 0.70, "24h": 0.60},  # Input/Output
    },
}


def _get_litellm_cost_map(model, completion_window="*"):
    cost = external_model_cost(model, completion_window=completion_window)
    litellm_cost_map = {
        model: {
            "max_tokens": 8192,
            "input_cost_per_token": cost["input_cost_per_token"],  # source: https://www.kluster.ai/#pricing
            "output_cost_per_token": cost["output_cost_per_token"],
            "litellm_provider": "openai",
        }
    }

    return litellm_cost_map


def external_model_cost(model, completion_window="*"):
    """Get the cost of the model from the external providers registered."""
    if model not in _EXTERNAL_MODEL_COST_MAP:
        return {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
    cost = _EXTERNAL_MODEL_COST_MAP[model]["cost_per_million"][completion_window]
    return {"input_cost_per_token": cost / 1e6, "output_cost_per_token": cost / 1e6}


class _KlusterAICostProcessor(_LitellmCostProcessor):
    _registered_models = set()

    def __init__(self, batch=False) -> None:
        self.batch = batch
        super().__init__(batch=batch)

    @staticmethod
    def _wrap(model, completion_window):
        return model + "." + completion_window

    def cost(self, *args, completion_window="*", **kwargs):
        if kwargs.get("completion_response") is not None:
            model = kwargs["completion_response"]["model"]
        else:
            model = kwargs.get("model", None) or args[0]
        times = 2 if self.batch else 1
        if _KlusterAICostProcessor._wrap(model, completion_window) in _KlusterAICostProcessor._registered_models:  #
            return super().cost(model, *args, **kwargs) * times

        import litellm

        litellm.register_model(_get_litellm_cost_map(model))
        _KlusterAICostProcessor._registered_models.add(_KlusterAICostProcessor._wrap(model, completion_window))
        return super().cost(*args, **kwargs) * times


COST_PROCESSOR = defaultdict(lambda: _LitellmCostProcessor)
COST_PROCESSOR["klusterai"] = _KlusterAICostProcessor


def cost_processor_factory(backend, batch=False):
    """Factory function to return the cost processor for the given backend."""
    return COST_PROCESSOR[backend](batch=batch)
