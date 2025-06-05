import time
import typing as t
from dataclasses import asdict, dataclass, field
from enum import Enum

import tqdm
from litellm import model_cost
from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from bespokelabs.curator import _CONSOLE
from bespokelabs.curator.cost import external_model_cost
from bespokelabs.curator.log import USE_RICH_DISPLAY, logger
from bespokelabs.curator.status_tracker.tqdm_constants.colors import COST, END, ERROR, HEADER, METRIC, MODEL, SUCCESS
from bespokelabs.curator.telemetry.client import TelemetryEvent, telemetry_client
from bespokelabs.curator.types.generic_response import _TokenUsage

# Time between status updates in seconds
_STATUS_UPDATE_INTERVAL = 5


class AgentTurn(str, Enum):
    """Enum for tracking which agent's turn it is in the conversation."""

    SEEDER = "seeder"
    PARTNER = "partner"


@dataclass
class AgentStatusTracker:
    """Tracks the status of multi-turn conversations between agents.

    This tracker monitors the progress of conversations between two agents,
    tracking turns, responses, and overall conversation statistics.

    Attributes:
        seeder_name (str): Name of the seeder agent.
        partner_name (str): Name of the partner agent.
        max_turns (int): Maximum number of turns in the conversation.
        current_turn (int): Current turn number in the conversation.
        current_agent (AgentTurn): Which agent's turn it currently is.
        num_responses (int): Total number of responses generated.
        num_errors (int): Number of errors encountered.
        start_time (float): Time when the conversation started.
        last_update_time (float): Time of the last status update.
        pbar (Optional[tqdm.tqdm]): Progress bar for tracking progress.
        total_tokens (_TokenUsage): Total tokens used in the conversation.
        total_cost (float): Total cost of the conversation.
    """

    seeder_name: str
    partner_name: str
    max_turns: int
    current_turn: int = 0
    current_agent: AgentTurn = AgentTurn.SEEDER
    num_responses: int = 0
    num_errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    pbar: t.Optional[tqdm.tqdm] = field(default=None, repr=False, compare=False)

    # Token and cost tracking
    total_tokens: _TokenUsage = field(default_factory=_TokenUsage)
    total_cost: float = 0.0

    # Model information for both agents
    seeder_model: str = ""
    partner_model: str = ""
    seeder_compatible_provider: t.Optional[str] = None
    partner_compatible_provider: t.Optional[str] = None

    # Cost tracking for both models using dictionaries
    input_cost_per_million: dict[str, t.Optional[float]] = field(default_factory=lambda: {"seeder": None, "partner": None})
    output_cost_per_million: dict[str, t.Optional[float]] = field(default_factory=lambda: {"seeder": None, "partner": None})
    input_cost_str: dict[str, str] = field(
        default_factory=lambda: {"seeder": "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A", "partner": "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A"}
    )
    output_cost_str: dict[str, str] = field(
        default_factory=lambda: {"seeder": "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A", "partner": "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A"}
    )

    def __post_init__(self):
        """Initialize the tracker."""
        self.input_cost_per_million = self.input_cost_per_million or {"seeder": None, "partner": None}
        self.output_cost_per_million = self.output_cost_per_million or {"seeder": None, "partner": None}

        # Initialize model costs for both agents
        if self.seeder_model:
            self._initialize_model_costs(self.seeder_model, is_seeder=True)
        if self.partner_model:
            self._initialize_model_costs(self.partner_model, is_seeder=False)

        if USE_RICH_DISPLAY:
            self._start_rich_tracker()
        else:
            self._start_tqdm_tracker()

    def _initialize_model_costs(self, model: str, is_seeder: bool) -> None:
        """Initialize model costs for a specific agent.

        Args:
            model: The name of the model to get costs for.
            is_seeder: Whether this is for the seeder agent (True) or partner agent (False).
        """
        agent_type = "seeder" if is_seeder else "partner"
        try:
            if model in model_cost:
                model_pricing = model_cost[model]
                self.input_cost_per_million[agent_type] = (
                    model_pricing.get("input_cost_per_token", 0) * 1_000_000 if model_pricing.get("input_cost_per_token") is not None else None
                )
                self.output_cost_per_million[agent_type] = (
                    model_pricing.get("output_cost_per_token", 0) * 1_000_000 if model_pricing.get("output_cost_per_token") is not None else None
                )
            else:
                try:
                    provider = self.seeder_compatible_provider if is_seeder else self.partner_compatible_provider
                    external_pricing = external_model_cost(model, provider=provider)
                    self.input_cost_per_million[agent_type] = (
                        external_pricing.get("input_cost_per_token", 0) * 1_000_000 if external_pricing.get("input_cost_per_token") is not None else None
                    )
                    self.output_cost_per_million[agent_type] = (
                        external_pricing.get("output_cost_per_token", 0) * 1_000_000 if external_pricing.get("output_cost_per_token") is not None else None
                    )
                except (KeyError, TypeError):
                    self.input_cost_per_million[agent_type] = None
                    self.output_cost_per_million[agent_type] = None

            self._format_cost_strings(agent_type)
        except Exception as e:
            logger.warning(f"Could not determine model costs for {agent_type}: {e}")
            self.input_cost_per_million[agent_type] = None
            self.output_cost_per_million[agent_type] = None
            self._format_cost_strings(agent_type)

    def _format_cost_strings(self, agent_type: str) -> None:
        """Format the cost strings based on the values.

        Args:
            agent_type: Either "seeder" or "partner"
        """
        if self.input_cost_per_million[agent_type] is not None:
            if USE_RICH_DISPLAY:
                self.input_cost_str[agent_type] = f"[red]${self.input_cost_per_million[agent_type]:.3f}[/red]"
            else:
                self.input_cost_str[agent_type] = f"${self.input_cost_per_million[agent_type]:.3f}"
        else:
            self.input_cost_str[agent_type] = "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A"

        if self.output_cost_per_million[agent_type] is not None:
            if USE_RICH_DISPLAY:
                self.output_cost_str[agent_type] = f"[red]${self.output_cost_per_million[agent_type]:.3f}[/red]"
            else:
                self.output_cost_str[agent_type] = f"${self.output_cost_per_million[agent_type]:.3f}"
        else:
            self.output_cost_str[agent_type] = "[dim]N/A[/dim]" if USE_RICH_DISPLAY else "N/A"

    def estimate_request_cost(self, input_tokens: int, output_tokens: int, is_seeder: bool) -> float:
        """Estimate cost for a request based on token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            is_seeder: Whether this is for the seeder agent (True) or partner agent (False)

        Returns:
            float: Estimated cost for the request
        """
        agent_type = "seeder" if is_seeder else "partner"
        input_cost = (input_tokens * (self.input_cost_per_million[agent_type] or 0)) / 1_000_000
        output_cost = (output_tokens * (self.output_cost_per_million[agent_type] or 0)) / 1_000_000
        return input_cost + output_cost

    def _start_rich_tracker(self):
        """Start the rich progress tracker."""
        self._console = _CONSOLE

        # Create progress bar display
        self._progress = Progress(
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold white]•[/bold white] Time Elapsed"),
            TimeElapsedColumn(),
            TextColumn("[bold white]•[/bold white] Time Remaining"),
            TimeRemainingColumn(),
            console=self._console,
        )

        # Create stats display with just text columns
        self._stats = Progress(
            TextColumn("{task.description}"),
            console=self._console,
        )

        # Add tasks
        self._task_id = self._progress.add_task(
            description="",
            total=self.max_turns,
            completed=0,
        )

        self._stats_task_id = self._stats.add_task(
            total=None,
            description=(
                f"Preparing conversation between [blue]{self.seeder_name}[/blue] "
                f"and [blue]{self.partner_name}[/blue] "
                f"for [blue]{self.max_turns}[/blue] turns"
            ),
        )

        # Create Live display with both progress and stats in one panel
        self._live = Live(
            Panel(
                Group(
                    self._progress,
                    self._stats,
                ),
                title="Multi-Turn Agent Conversation",
                box=box.ROUNDED,
            ),
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def _start_tqdm_tracker(self):
        """Start the tqdm progress tracker."""
        self.pbar = tqdm.tqdm(
            total=self.max_turns,
            initial=0,
            desc=f"Conversation between {self.seeder_name} and {self.partner_name}",
            unit="turn",
        )

    def update_turn(self, agent: AgentTurn, response_success: bool = True, token_usage: t.Optional[_TokenUsage] = None, cost: t.Optional[float] = None):
        """Update the tracker with a new turn.

        Args:
            agent (AgentTurn): The agent that just took a turn.
            response_success (bool): Whether the response was successful.
            token_usage (Optional[_TokenUsage]): Token usage for this turn.
            cost (Optional[float]): Cost for this turn.
        """
        self.current_turn += 1
        self.current_agent = agent
        self.num_responses += 1
        if not response_success:
            self.num_errors += 1

        if token_usage:
            self.total_tokens.input += token_usage.input
            self.total_tokens.output += token_usage.output
            self.total_tokens.total += token_usage.total

        if cost:
            self.total_cost += cost

        self.update_display()

    def update_display(self):
        """Update the display with current status."""
        current_time = time.time()

        if USE_RICH_DISPLAY:
            self._refresh_console()
        else:
            if self.pbar:
                self.pbar.n = self.current_turn
                self.pbar.set_description(
                    f"Conversation: {MODEL}{self.current_agent.value}{END} "
                    f"[{SUCCESS}{self.num_responses} responses{END} • "
                    f"{ERROR}{self.num_errors} errors{END} • "
                    f"{METRIC}{self.current_turn}/{self.max_turns} turns{END} • "
                    f"{COST}${self.total_cost:.3f} spent{END} • "
                    f"{METRIC}{self.total_tokens.total:,} tokens{END}]"
                )
                self.pbar.refresh()

        self.last_update_time = current_time

    def _refresh_console(self):
        """Refresh the console display with latest stats."""
        # Calculate stats
        elapsed_minutes = (time.time() - self.start_time) / 60
        input_tpm = self.total_tokens.input / max(0.001, elapsed_minutes)
        output_tpm = self.total_tokens.output / max(0.001, elapsed_minutes)
        cost_per_minute = self.total_cost / max(0.01, elapsed_minutes)

        # Format stats text
        stats_text = (
            f"[bold white]Conversation:[/bold white] "
            f"[white]Turn:[/white] [blue]{self.current_turn}/{self.max_turns}[/blue] "
            f"[white]•[/white] "
            f"[white]Agent:[/white] [blue]{self.current_agent.value}[/blue] "
            f"[white]•[/white] "
            f"[white]Responses:[/white] [green]{self.num_responses}[/green] "
            f"[white]•[/white] "
            f"[white]Errors:[/white] [red]{self.num_errors}[/red]\n"
            f"[bold white]Time:[/bold white] "
            f"[white]Elapsed:[/white] [blue]{time.time() - self.start_time:.1f}s[/blue]\n"
            f"[bold white]Tokens:[/bold white] "
            f"[white]Input:[/white] [blue]{self.total_tokens.input:,}[/blue] "
            f"([blue]{input_tpm:.0f}/min[/blue]) "
            f"[white]•[/white] "
            f"[white]Output:[/white] [blue]{self.total_tokens.output:,}[/blue] "
            f"([blue]{output_tpm:.0f}/min[/blue]) "
            f"[white]•[/white] "
            f"[white]Total:[/white] [blue]{self.total_tokens.total:,}[/blue]\n"
            f"[bold white]Cost:[/bold white] "
            f"[white]Total:[/white] [red]${self.total_cost:.3f}[/red] "
            f"[white]•[/white] "
            f"[white]Rate:[/white] [red]${cost_per_minute:.3f}/min[/red]\n"
            f"[bold white]Model Pricing:[/bold white]\n"
            f"  [white]Seeder ({self.seeder_model}):[/white] "
            f"[white]Input:[/white] {self.input_cost_str['seeder']} "
            f"[white]•[/white] "
            f"[white]Output:[/white] {self.output_cost_str['seeder']}\n"
            f"  [white]Partner ({self.partner_model}):[/white] "
            f"[white]Input:[/white] {self.input_cost_str['partner']} "
            f"[white]•[/white] "
            f"[white]Output:[/white] {self.output_cost_str['partner']}"
        )

        # Update main progress bar
        self._progress.update(
            self._task_id,
            completed=self.current_turn,
        )

        # Update stats display
        self._stats.update(
            self._stats_task_id,
            description=stats_text,
        )

    def stop_tracker(self):
        """Stop the tracker and display final statistics."""
        if USE_RICH_DISPLAY:
            if hasattr(self, "_live"):
                self._live.stop()
                self._console.print(self._progress)
                self._console.print(self._stats)
        else:
            if self.pbar:
                self.pbar.close()

        # Display final statistics
        self.display_final_stats()

        # Clean up non-serializable fields before telemetry
        temp_pbar = self.pbar
        self.pbar = None
        metadata = asdict(self)
        metadata.pop("pbar", None)
        # Restore pbar if needed
        self.pbar = temp_pbar

        telemetry_client.capture(
            TelemetryEvent(
                event_type="AgentConversation",
                metadata=metadata,
            )
        )

    def display_final_stats(self):
        """Display final statistics."""
        if USE_RICH_DISPLAY:
            self._display_rich_final_stats()
        else:
            self._display_simple_final_stats()

    def _display_rich_final_stats(self):
        """Display final statistics using rich table."""
        table = Table(title="Final Agent Conversation Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        # Conversation Information
        table.add_row("Agents", f"{self.seeder_name} ↔ {self.partner_name}")
        table.add_row("Total Turns", str(self.max_turns))
        table.add_row("Completed Turns", str(self.current_turn))
        table.add_row("Successful Responses", f"[green]{self.num_responses}[/green]")
        table.add_row("Errors", f"[red]{self.num_errors}[/red]")
        table.add_row("Total Time", f"{time.time() - self.start_time:.2f}s")
        table.add_row("Average Time per Turn", f"{(time.time() - self.start_time) / max(1, self.current_turn):.2f}s")

        # Token Statistics
        table.add_row("", "")  # Empty row for spacing
        table.add_row("Token Statistics", "", style="bold magenta")
        table.add_row("Total Input Tokens", f"{self.total_tokens.input:,}")
        table.add_row("Total Output Tokens", f"{self.total_tokens.output:,}")
        table.add_row("Total Tokens", f"{self.total_tokens.total:,}")
        if self.current_turn > 0:
            table.add_row("Average Input Tokens per Turn", f"{self.total_tokens.input / self.current_turn:.0f}")
            table.add_row("Average Output Tokens per Turn", f"{self.total_tokens.output / self.current_turn:.0f}")

        # Cost Statistics
        table.add_row("", "")  # Empty row for spacing
        table.add_row("Cost Statistics", "", style="bold magenta")
        table.add_row("Total Cost", f"[red]${self.total_cost:.3f}[/red]")
        if self.current_turn > 0:
            table.add_row("Average Cost per Turn", f"[red]${self.total_cost / self.current_turn:.3f}[/red]")
        elapsed_minutes = (time.time() - self.start_time) / 60
        table.add_row("Cost per Minute", f"[red]${self.total_cost / max(0.01, elapsed_minutes):.3f}[/red]")

        # Model Pricing
        table.add_row("", "")  # Empty row for spacing
        table.add_row("Model Pricing", "", style="bold magenta")
        table.add_row(f"Seeder ({self.seeder_model})", "")
        table.add_row("  Input Cost per 1M Tokens", self.input_cost_str["seeder"])
        table.add_row("  Output Cost per 1M Tokens", self.output_cost_str["seeder"])
        table.add_row(f"Partner ({self.partner_model})", "")
        table.add_row("  Input Cost per 1M Tokens", self.input_cost_str["partner"])
        table.add_row("  Output Cost per 1M Tokens", self.output_cost_str["partner"])

        self._console.print(table)

    def _display_simple_final_stats(self):
        """Display final statistics in plain text format."""
        elapsed_minutes = (time.time() - self.start_time) / 60
        input_tpm = self.total_tokens.input / max(0.001, elapsed_minutes)
        output_tpm = self.total_tokens.output / max(0.001, elapsed_minutes)
        cost_per_minute = self.total_cost / max(0.01, elapsed_minutes)

        stats = [
            f"\n{HEADER}Final Agent Conversation Statistics:{END}",
            f"{HEADER}Agents:{END} {MODEL}{self.seeder_name}{END} ↔ {MODEL}{self.partner_name}{END}",
            f"{HEADER}Total Turns:{END} {METRIC}{self.max_turns}{END}",
            f"{HEADER}Completed Turns:{END} {METRIC}{self.current_turn}{END}",
            f"{HEADER}Successful Responses:{END} {SUCCESS}{self.num_responses}{END}",
            f"{HEADER}Errors:{END} {ERROR}{self.num_errors}{END}",
            f"{HEADER}Total Time:{END} {METRIC}{time.time() - self.start_time:.2f}s{END}",
            f"{HEADER}Average Time per Turn:{END} {METRIC}{(time.time() - self.start_time) / max(1, self.current_turn):.2f}s{END}",
            "",
            f"{HEADER}Token Statistics:{END}",
            f"  Total Input Tokens: {METRIC}{self.total_tokens.input:,}{END} ({input_tpm:.0f}/min)",
            f"  Total Output Tokens: {METRIC}{self.total_tokens.output:,}{END} ({output_tpm:.0f}/min)",
            f"  Total Tokens: {METRIC}{self.total_tokens.total:,}{END}",
            f"  Average Input Tokens per Turn: {METRIC}{self.total_tokens.input / max(1, self.current_turn):.0f}{END}",
            f"  Average Output Tokens per Turn: {METRIC}{self.total_tokens.output / max(1, self.current_turn):.0f}{END}",
            "",
            f"{HEADER}Cost Statistics:{END}",
            f"  Total Cost: {COST}${self.total_cost:.3f}{END}",
            f"  Average Cost per Turn: {COST}${self.total_cost / max(1, self.current_turn):.3f}{END}",
            f"  Cost per Minute: {COST}${cost_per_minute:.3f}{END}",
            "",
            f"{HEADER}Model Pricing:{END}",
            f"  Seeder ({self.seeder_model}):",
            f"    Input Cost per 1M Tokens: {COST}{self.input_cost_str['seeder']}{END}",
            f"    Output Cost per 1M Tokens: {COST}{self.output_cost_str['seeder']}{END}",
            f"  Partner ({self.partner_model}):",
            f"    Input Cost per 1M Tokens: {COST}{self.input_cost_str['partner']}{END}",
            f"    Output Cost per 1M Tokens: {COST}{self.output_cost_str['partner']}{END}",
        ]
        logger.info("\n".join(stats))

    def __del__(self):
        """Ensure live display is stopped on deletion."""
        if hasattr(self, "_live"):
            self._live.stop()
