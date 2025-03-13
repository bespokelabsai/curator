from collections import defaultdict

import litellm

from bespokelabs.curator.request_processor import _DEFAULT_COST_MAP

litellm.suppress_debug_info = True

RATE_LIMIT_HEADER = {
    "api.together.xyz": {"request-key": {"key": "x-ratelimit-limit", "type": "rps"}, "token-key": {"key": "x-ratelimit-limit-tokens", "type": "tps"}}
}


class _LitellmCostProcessor:
    def __init__(self, config, batch=False) -> None:
        self.batch = batch
        self.config = config
        if config.in_mtok_cost is not None:
            cost_per_input_token = config.in_mtok_cost / 1e6
            if config.out_mtok_cost is not None:
                cost_per_output_token = config.out_mtok_cost / 1e6
            else:
                cost_per_output_token = cost_per_input_token

            litellm_provider = "openai"
            if "/" in config.model:
                litellm_provider = config.model.split("/")[0]

            litellm.register_model(
                {
                    config.model: {
                        "max_tokens": 8192,
                        "input_cost_per_token": cost_per_input_token,
                        "output_cost_per_token": cost_per_output_token,
                        "litellm_provider": litellm_provider,
                    }
                }
            )

    def cost(self, *, completion_window="*", **kwargs):
        cost_to_complete = 0.0
        if self.config.model in litellm.model_cost:
            if "model" not in kwargs:
                kwargs["model"] = self.config.model
            cost_to_complete = litellm.completion_cost(**kwargs)
        if self.batch:
            cost_to_complete *= 0.5
        return cost_to_complete


def _get_litellm_cost_map(model, completion_window="*", provider="default"):
    cost = external_model_cost(model, completion_window=completion_window, provider=provider)
    litellm_cost_map = {
        model: {
            "max_tokens": 8192,
            "input_cost_per_token": cost["input_cost_per_token"],
            "output_cost_per_token": cost["output_cost_per_token"],
            "litellm_provider": "openai",
        }
    }

    return litellm_cost_map


def external_model_cost(model, completion_window="*", provider="default"):
    """Get the cost of the model from the external providers registered."""
    if provider not in _DEFAULT_COST_MAP["external"]["providers"]:
        return {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
    provider_cost = _DEFAULT_COST_MAP["external"]["providers"][provider]["cost"]
    if model not in provider_cost:
        raise ValueError(f"Model {model} is not supported by {provider}.")

    if completion_window not in provider_cost[model]["input_cost_per_million"]:
        raise ValueError(f"Completion window {completion_window} is not supported for {model} by {provider}.")

    cost = provider_cost[model]["input_cost_per_million"][completion_window]
    return {"input_cost_per_token": cost / 1e6, "output_cost_per_token": cost / 1e6}


# source: https://www.kluster.ai/#pricing
class _KlusterAICostProcessor(_LitellmCostProcessor):
    _registered_models = set()

    def __init__(self, config, batch=False) -> None:
        self.batch = batch
        super().__init__(config=config, batch=batch)

    @staticmethod
    def _wrap(model, completion_window):
        return model + "." + completion_window

    def cost(self, *, completion_window="*", **kwargs):
        times = 2 if self.batch else 1
        if _KlusterAICostProcessor._wrap(self.config.model, completion_window) in _KlusterAICostProcessor._registered_models:
            return super().cost(completion_window=completion_window, **kwargs) * times

        litellm.register_model(_get_litellm_cost_map(self.config.model, provider="klusterai", completion_window=completion_window))
        _KlusterAICostProcessor._registered_models.add(_KlusterAICostProcessor._wrap(self.config.model, completion_window))
        return super().cost(completion_window=completion_window, **kwargs) * times


# source: https://docs.inference.net/resources/pricing
class _InferenceNetCostProcessor(_LitellmCostProcessor):
    _registered_models = set()

    def __init__(self, config, batch=False) -> None:
        self.batch = batch
        super().__init__(config=config, batch=batch)

    @staticmethod
    def _wrap(model, completion_window):
        return model + "." + completion_window

    def cost(self, *, completion_window="*", **kwargs):
        if "completion_response" in kwargs:
            model = kwargs["completion_response"]["model"]
        else:
            model = kwargs.get("model", None)
        times = 2 if self.batch else 1
        if _InferenceNetCostProcessor._wrap(model, completion_window) in _InferenceNetCostProcessor._registered_models:
            return super().cost(completion_window=completion_window, **kwargs) * times

        litellm.register_model(_get_litellm_cost_map(model, provider="inference.net", completion_window=completion_window))
        _InferenceNetCostProcessor._registered_models.add(_InferenceNetCostProcessor._wrap(model, completion_window))
        return super().cost(completion_window=completion_window, **kwargs) * times


COST_PROCESSOR = defaultdict(lambda: _LitellmCostProcessor)
COST_PROCESSOR["klusterai"] = _KlusterAICostProcessor
COST_PROCESSOR["inference.net"] = _InferenceNetCostProcessor


def cost_processor_factory(backend, config=None, batch=False):
    """Factory function to return the cost processor for the given backend."""
    return COST_PROCESSOR[backend](config=config, batch=batch)
