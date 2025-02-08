import logging
import time
import typing as t
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import tqdm
from litellm import model_cost
from pydantic import BaseModel
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from bespokelabs.curator import _CONSOLE
from bespokelabs.curator.telemetry.client import TelemetryEvent, telemetry_client
from bespokelabs.curator.types.generic_response import TokenUsage

logger = logging.getLogger(__name__)


class TokenLimitStrategy(str, Enum):
    """Token limit Strategy enum."""

    combined = "combined"
    seperate = "seperate"
    default = "combined"

    def __str__(self):
        """String representation of the token limit strategy."""
        return self.value


class _TokenCount(BaseModel):
    input: int | float | None = 0
    output: int | float | None = 0

    @property
    def total(self):
        if self.input is None:
            return None
        return self.input + self.output


@dataclass
class OnlineStatusTracker:
    """Tracks the status of all requests."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    num_rate_limit_errors: int = 0
    available_request_capacity: float = 1.0
    available_token_capacity: float | _TokenCount = 0
    last_update_time: float = field(default_factory=time.time)
    max_requests_per_minute: int = 0
    max_tokens_per_minute: int | _TokenCount = 0
    max_concurrent_requests: int | None = None
    max_tokens_per_minute: int = 0
    pbar: tqdm = field(default=None)
    response_cost: float = 0
    time_of_last_rate_limit_error: float = field(default=0.0)

    # Stats tracking
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0

    # Cost per million tokens
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None
    compatible_provider: Optional[str] = None

    start_time: float = field(default_factory=time.time, init=False)

    # Add model name field
    model: str = ""
    token_limit_strategy: TokenLimitStrategy = TokenLimitStrategy.default

    def __post_init__(self):
        """Post init."""
        if self.token_limit_strategy == TokenLimitStrategy.combined:
            self.available_token_capacity = t.cast(float, self.available_token_capacity)
        else:
            self.available_token_capacity = t.cast(_TokenCount, self.available_token_capacity)
            self.available_token_capacity = _TokenCount()
            if not self.max_tokens_per_minute:
                self.max_tokens_per_minute = _TokenCount()

    def start_tracker(self, console: Optional[Console] = None):
        """Start the tracker."""
        self._console = _CONSOLE if console is None else console

        # Create progress display
        self._progress = Progress(
            TextColumn(
                "[cyan]{task.description}[/cyan]\n"
                "{task.fields[requests_text]}\n"
                "{task.fields[tokens_text]}\n"
                "{task.fields[cost_text]}\n"
                "{task.fields[rate_limit_text]}\n"
                "{task.fields[price_text]}",
                justify="left",
            ),
            TextColumn("\n\n\n\n\n\n"),  # Spacer
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold white]•[/bold white]"),
            TimeElapsedColumn(),
            TextColumn("[bold white]•[/bold white]"),
            TimeRemainingColumn(),
            console=self._console,
        )

        # Add task
        self._task_id = self._progress.add_task(
            description=f"[cyan]Generating data using {self.model} with {self.token_limit_strategy.value} input and output token Strategy.",
            total=self.total_requests,
            completed=self.num_tasks_already_completed,
            requests_text="[bold white]Requests:[/bold white] [dim]--[/dim]",
            tokens_text="[bold white]Tokens:[/bold white] [dim]--[/dim]",
            cost_text="[bold white]Cost:[/bold white] [dim]--[/dim]",
            model_name_text="[bold white]Model:[/bold white] [dim]--[/dim]",
            rate_limit_text="[bold white]Rate Limits:[/bold white] [dim]--[/dim]",
            price_text="[bold white]Model Pricing:[/bold white] [dim]--[/dim]",
        )

        if self.model in model_cost:
            self.input_cost_per_million = model_cost[self.model]["input_cost_per_token"] * 1_000_000
            self.output_cost_per_million = model_cost[self.model]["output_cost_per_token"] * 1_000_000
        else:
            from bespokelabs.curator.cost import external_model_cost

            self.input_cost_per_million = external_model_cost(self.model, provider=self.compatible_provider)["input_cost_per_token"] * 1_000_000
            self.output_cost_per_million = external_model_cost(self.model, provider=self.compatible_provider)["output_cost_per_token"] * 1_000_000

        # Create Live display that will show both logs and progress
        self._live = Live(
            Group(
                Panel(self._progress),
            ),
            console=self._console,
            refresh_per_second=4,
            transient=True,  # This ensures logs stay above the progress bar
        )
        self._live.start()

    def update_stats(self, token_usage: TokenUsage, cost: float):
        """Update statistics in the tracker with token usage and cost."""
        if token_usage:
            self.total_prompt_tokens += token_usage.prompt_tokens
            self.total_completion_tokens += token_usage.completion_tokens
            self.total_tokens += token_usage.total_tokens
        if cost:
            self.total_cost += cost

        avg_prompt = self.total_prompt_tokens / max(1, self.num_tasks_succeeded)
        avg_completion = self.total_completion_tokens / max(1, self.num_tasks_succeeded)
        avg_cost = self.total_cost / max(1, self.num_tasks_succeeded)
        projected_cost = avg_cost * self.total_requests

        # Calculate current rpm
        elapsed_minutes = (time.time() - self.start_time) / 60
        current_rpm = self.num_tasks_succeeded / max(0.001, elapsed_minutes)

        # Format the text for each line with properly closed tags

        requests_text = (
            "[bold white]Requests:[/bold white] "
            f"[white]•[/white] "
            f"[white]Total:[/white] [blue]{self.total_requests}[/blue] "
            f"[white]•[/white] "
            f"[white]Cached:[/white] [green]{self.num_tasks_already_completed}✓[/green] "
            f"[white]•[/white] "
            f"[white]Success:[/white] [green]{self.num_tasks_succeeded}✓[/green] "
            f"[white]•[/white] "
            f"[white]Failed:[/white] [red]{self.num_tasks_failed}✗[/red] "
            f"[white]•[/white] "
            f"[white]In Progress:[/white] [yellow]{self.num_tasks_in_progress}⋯[/yellow] "
            f"[white]•[/white] "
            f"[white]RPM:[/white] [blue]{current_rpm:.1f}[/blue]"
        )

        input_tpm = self.total_prompt_tokens / max(0.001, elapsed_minutes)
        output_tpm = self.total_completion_tokens / max(0.001, elapsed_minutes)

        tokens_text = (
            "[bold white]Tokens:[/bold white] "
            f"[white]Avg Input:[/white] [blue]{avg_prompt:.0f}[/blue] "
            f"[white]•[/white] "
            f"[white]Input TPM:[/white] [blue]{input_tpm:.0f}[/blue] "
            f"[white]•[/white] "
            f"[white]Avg Output:[/white] [blue]{avg_completion:.0f}[/blue] "
            f"[white]•[/white] "
            f"[white]Output TPM:[/white] [blue]{output_tpm:.0f}[/blue]"
        )

        cost_text = (
            "[bold white]Cost:[/bold white] "
            f"[white]Current:[/white] [magenta]${self.total_cost:.3f}[/magenta] "
            f"[white]•[/white] "
            f"[white]Projected:[/white] [magenta]${projected_cost:.3f}[/magenta] "
            f"[white]•[/white] "
            f"[white]Rate:[/white] [magenta]${self.total_cost / max(0.01, elapsed_minutes):.3f}/min[/magenta]"
        )

        model_name_text = f"[bold white]Model:[/bold white] [blue]{self.model}[/blue]"

        rate_limit_text = (
            "[bold white]Rate Limits:[/bold white] "
            f"[white]RPM:[/white] [blue]{self.max_requests_per_minute}[/blue] "
            f"[white]•[/white] "
            f"[white]TPM:[/white] [blue]{self.max_tokens_per_minute}[/blue]"
        )

        input_cost_str = f"${self.input_cost_per_million:.3f}" if isinstance(self.input_cost_per_million, float) else "N/A"
        output_cost_str = f"${self.output_cost_per_million:.3f}" if isinstance(self.output_cost_per_million, float) else "N/A"

        price_text = (
            "[bold white]Model Pricing:[/bold white] "
            f"[white]Per 1M tokens:[/white] "
            f"[white]Input:[/white] [red]{input_cost_str}[/red] "
            f"[white]•[/white] "
            f"[white]Output:[/white] [red]{output_cost_str}[/red]"
        )

        # Update progress through the live display
        self._progress.update(
            self._task_id,
            completed=self.num_tasks_succeeded + self.num_tasks_already_completed,
            requests_text=requests_text,
            tokens_text=tokens_text,
            cost_text=cost_text,
            model_name_text=model_name_text,
            rate_limit_text=rate_limit_text,
            price_text=price_text,
        )

    def stop_tracker(self):
        """Stop the tracker."""
        if hasattr(self, "_live"):
            # Refresh one last time to show final state
            self._progress.refresh()
            # Stop the live display
            self._live.stop()
            # Print the final progress state
            self._console.print(self._progress)

        table = Table(title="Final Curator Statistics", box=box.ROUNDED)
        table.add_column("Section/Metric", style="cyan")
        table.add_column("Value", style="yellow")

        # Model Information
        table.add_row("Model", "", style="bold magenta")
        table.add_row("Name", f"[blue]{self.model}[/blue]")
        table.add_row("Rate Limit (RPM)", f"[blue]{self.max_requests_per_minute}[/blue]")
        table.add_row("Rate Limit (TPM)", f"[blue]{self.max_tokens_per_minute}[/blue]")

        # Request Statistics
        table.add_row("Requests", "", style="bold magenta")
        table.add_row("Total Processed", str(self.num_tasks_succeeded + self.num_tasks_failed))
        table.add_row("Successful", f"[green]{self.num_tasks_succeeded}[/green]")
        table.add_row("Failed", f"[red]{self.num_tasks_failed}[/red]")

        # Token Statistics
        table.add_row("Tokens", "", style="bold magenta")
        table.add_row("Total Tokens Used", f"{self.total_tokens:,}")
        table.add_row("Total Input Tokens", f"{self.total_prompt_tokens:,}")
        table.add_row("Total Output Tokens", f"{self.total_completion_tokens:,}")
        if self.num_tasks_succeeded > 0:
            table.add_row("Average Tokens per Request", f"{int(self.total_tokens / self.num_tasks_succeeded)}")
            table.add_row("Average Input Tokens", f"{int(self.total_prompt_tokens / self.num_tasks_succeeded)}")
            table.add_row("Average Output Tokens", f"{int(self.total_completion_tokens / self.num_tasks_succeeded)}")
        # Cost Statistics
        table.add_row("Costs", "", style="bold magenta")
        table.add_row("Total Cost", f"[red]${self.total_cost:.4f}[/red]")
        table.add_row("Average Cost per Request", f"[red]${self.total_cost / max(1, self.num_tasks_succeeded):.4f}[/red]")

        # Handle None values for cost per million tokens
        input_cost_str = f"[red]${self.input_cost_per_million:.4f}[/red]" if self.input_cost_per_million is not None else "[dim]N/A[/dim]"
        output_cost_str = f"[red]${self.output_cost_per_million:.4f}[/red]" if self.output_cost_per_million is not None else "[dim]N/A[/dim]"

        table.add_row("Input Cost per 1M Tokens", input_cost_str)
        table.add_row("Output Cost per 1M Tokens", output_cost_str)

        # Performance Statistics
        table.add_row("Performance", "", style="bold magenta")
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        rpm = self.num_tasks_succeeded / max(0.001, elapsed_minutes)
        input_tpm = self.total_prompt_tokens / max(0.001, elapsed_minutes)
        output_tpm = self.total_completion_tokens / max(0.001, elapsed_minutes)

        table.add_row("Total Time", f"{elapsed_time:.2f}s")
        table.add_row("Average Time per Request", f"{elapsed_time / max(1, self.num_tasks_succeeded):.2f}s")
        table.add_row("Requests per Minute", f"{rpm:.1f}")
        table.add_row("Input Tokens per Minute", f"{input_tpm:.1f}")
        table.add_row("Output Tokens per Minute", f"{output_tpm:.1f}")

        self._console.print(table)

        telemetry_client.capture(
            TelemetryEvent(
                event_type="OnlineRequest",
                metadata=asdict(self),
            )
        )

    def __str__(self):
        """String representation of the status tracker."""
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}\n"
            f"Errors - API: {self.num_api_errors}, "
            f"Rate Limit: {self.num_rate_limit_errors}, "
            f"Other: {self.num_other_errors}, "
            f"Total: {self.num_other_errors + self.num_api_errors + self.num_rate_limit_errors}"
        )

    def update_capacity(self):
        """Update available capacity based on time elapsed."""
        current_time = time.time()
        seconds_since_update = current_time - self.last_update_time
        if self.max_requests_per_minute is not None:
            self.available_request_capacity = min(
                self.available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0,
                self.max_requests_per_minute,
            )

        if self.token_limit_strategy == TokenLimitStrategy.combined:
            if self.max_tokens_per_minute is not None:
                self.available_token_capacity = t.cast(int, self.available_token_capacity)
                self.max_tokens_per_minute = t.cast(int, self.max_tokens_per_minute)
                self.available_token_capacity = min(
                    self.available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0,
                    self.max_tokens_per_minute,
                )
        else:
            self.available_token_capacity = t.cast(_TokenCount, self.available_token_capacity)
            self.max_tokens_per_minute = t.cast(_TokenCount, self.max_tokens_per_minute)
            if self.max_tokens_per_minute.input is not None:
                self.available_token_capacity.input = min(
                    self.available_token_capacity.input + self.max_tokens_per_minute.input * seconds_since_update / 60.0,
                    self.max_tokens_per_minute.input,
                )
            if self.max_tokens_per_minute.output is not None:
                self.available_token_capacity.output = min(
                    self.available_token_capacity.output + self.max_tokens_per_minute.output * seconds_since_update / 60.0, self.max_tokens_per_minute.output
                )

        self.last_update_time = current_time

    def has_capacity(self, token_estimate: _TokenCount) -> bool:
        """Check if there's enough capacity for a request."""
        self.update_capacity()
        if self.token_limit_strategy == TokenLimitStrategy.combined:
            has_capacity = self._check_combined_capacity(token_estimate)
        else:
            has_capacity = self._check_seperate_capacity(token_estimate)

        if not has_capacity:
            logger.debug(
                f"No capacity for request with {token_estimate} tokens."
                f"Available capacity: {self.available_token_capacity} tokens, "
                f"{int(self.available_request_capacity)} requests."
            )
        return has_capacity

    def _check_combined_capacity(self, token_estimate):
        self.available_token_capacity = t.cast(int, self.available_token_capacity)
        if self.max_requests_per_minute is None and self.max_tokens_per_minute is None:
            return True

        token_estimate = token_estimate.total
        has_capacity = self.available_request_capacity >= 1 and self.available_token_capacity >= token_estimate
        return has_capacity

    def _check_seperate_capacity(self, token_estimate: _TokenCount):
        self.available_token_capacity = t.cast(_TokenCount, self.available_token_capacity)
        if self.max_tokens_per_minute.total is None and self.max_requests_per_minute is None:
            return True

        has_capacity = (
            self.available_request_capacity >= 1
            and self.available_token_capacity.input >= token_estimate.input
            and self.available_token_capacity.output >= token_estimate.output
        )
        return has_capacity

    def consume_capacity(self, token_estimate: _TokenCount):
        """Consume capacity for a request."""
        if self.max_requests_per_minute is not None:
            self.available_request_capacity -= 1
        if self.token_limit_strategy == TokenLimitStrategy.combined:
            if self.max_tokens_per_minute is not None:
                self.available_token_capacity = t.cast(float, self.available_token_capacity)
                self.available_token_capacity -= token_estimate.total
        else:
            self.available_token_capacity = t.cast(_TokenCount, self.available_token_capacity)

            if self.max_tokens_per_minute is not None:
                self.available_token_capacity.input -= token_estimate.input
                self.available_token_capacity.output -= token_estimate.output

    def free_capacity(self, used: _TokenCount, blocked: _TokenCount):
        """Free extra consumed capacity.

        Note: This can be a negative number
        incase of under estimation of consumed capacity.
        """
        if self.token_limit_strategy == TokenLimitStrategy.seperate:
            input_free = blocked.input - used.input
            output_free = blocked.output - used.output
            self.available_token_capacity.input += input_free
            self.available_token_capacity.output += output_free
        else:
            free = blocked.total - used.total
            self.available_token_capacity += free

    def __del__(self):
        """Ensure live display is stopped on deletion."""
        if hasattr(self, "_live"):
            self._live.stop()
