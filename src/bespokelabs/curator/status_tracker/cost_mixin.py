from typing import Optional

from litellm import model_cost

from bespokelabs.curator.cost import external_model_cost
from bespokelabs.curator.log import USE_RICH_DISPLAY, logger


class CostMixin:
    """Mixin class for handling cost calculations in status trackers."""

    def __init__(self):
        """Initialize cost-related attributes."""
        self.input_cost_per_million: Optional[float] = None
        self.output_cost_per_million: Optional[float] = None
        self.compatible_provider: Optional[str] = None

    def initialize_model_costs(self, model: str) -> None:
        """Initialize model costs based on the model name.

        Args:
            model: The name of the model to get costs for.
        """
        try:
            if model in model_cost:
                model_pricing = model_cost[model]
                if model_pricing.get("input_cost_per_token") is not None:
                    self.input_cost_per_million = model_pricing.get("input_cost_per_token", 0) * 1_000_000
                else:
                    self.input_cost_per_million = None
                if model_pricing.get("output_cost_per_token") is not None:
                    self.output_cost_per_million = model_pricing.get("output_cost_per_token", 0) * 1_000_000
                else:
                    self.output_cost_per_million = None
            else:
                try:
                    external_pricing = external_model_cost(model, provider=self.compatible_provider)
                    self.input_cost_per_million = (
                        external_pricing.get("input_cost_per_token", 0) * 1_000_000 if external_pricing.get("input_cost_per_token") is not None else None
                    )
                    self.output_cost_per_million = (
                        external_pricing.get("output_cost_per_token", 0) * 1_000_000 if external_pricing.get("output_cost_per_token") is not None else None
                    )
                except (KeyError, TypeError):
                    self.input_cost_per_million = None
                    self.output_cost_per_million = None

            self._format_cost_strings()
        except Exception as e:
            logger.warning(f"Could not determine model costs: {e}")
            self.input_cost_per_million = None
            self.output_cost_per_million = None
            self._format_cost_strings()

    def _format_cost_strings(self) -> None:
        """Format the cost strings based on the values."""
        if self.input_cost_per_million is not None:
            if USE_RICH_DISPLAY:
                self.input_cost_str = f"[red]${self.input_cost_per_million:.3f}[/red]"
            else:
                self.input_cost_str = f"${self.input_cost_per_million:.3f}"
        else:
            self.input_cost_str = "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A"

        if self.output_cost_per_million is not None:
            if USE_RICH_DISPLAY:
                self.output_cost_str = f"[red]${self.output_cost_per_million:.3f}[/red]"
            else:
                self.output_cost_str = f"${self.output_cost_per_million:.3f}"
        else:
            self.output_cost_str = "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A"

    def estimate_request_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request based on token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            float: Estimated cost for the request
        """
        input_cost = (input_tokens * (self.input_cost_per_million or 0)) / 1_000_000
        output_cost = (output_tokens * (self.output_cost_per_million or 0)) / 1_000_000
        return input_cost + output_cost
