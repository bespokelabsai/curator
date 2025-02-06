"""Module for tracking the status of batches during curation."""

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Group

from bespokelabs.curator.telemetry.client import TelemetryEvent, telemetry_client
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus
from bespokelabs.curator.types.generic_response import TokenUsage

logger = logging.getLogger(__name__)


class BatchStatusTracker(BaseModel):
    """Class for tracking the status of batches during curation.

    Tracks unsubmitted request files, submitted batches, finished batches,
    and downloaded batches. Provides properties and methods to monitor and
    update the status of batches and requests throughout the curation process.
    """

    model_config = {
        "arbitrary_types_allowed": True,  # Allow non-serializable types
        "exclude": {"console", "progress", "task_id"},  # Exclude from serialization
    }

    n_total_requests: int = Field(default=0)
    unsubmitted_request_files: set[str] = Field(default_factory=set)
    submitted_batches: dict[str, GenericBatch] = Field(default_factory=dict)
    finished_batches: dict[str, GenericBatch] = Field(default_factory=dict)
    downloaded_batches: dict[str, GenericBatch] = Field(default_factory=dict)

    # Add fields for tracking costs and tokens
    total_prompt_tokens: int = Field(default=0)
    total_completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)

    # Model information
    model: str = Field(default="")
    input_cost_per_million: Optional[float] = Field(default=None)
    output_cost_per_million: Optional[float] = Field(default=None)

    # Track start time
    start_time: float = Field(default_factory=time.time)

    def start_tracker(self, console: Optional[Console] = None):
        """Start the progress tracker with rich console output."""
        self._console = Console(stderr=True) if console is None else console
        
        # Set up rich logging handler
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s",
            datefmt="[%Y-%m-%d %X]",
            handlers=[RichHandler(console=self._console)]
        )
        
        # Create progress display
        self._progress = Progress(
            TextColumn(
                "[cyan]{task.description}[/cyan]\n"
                "{task.fields[batches_text]}\n"
                "{task.fields[requests_text]}\n"
                "{task.fields[tokens_text]}\n"
                "{task.fields[cost_text]}\n"
                "{task.fields[price_text]}",
                justify="left",
            ),
            TextColumn("\n\n\n\n\n\n"),  # Spacer
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold white]•[/bold white]"),
            TimeElapsedColumn(),
            console=self._console,
        )

        self._task_id = self._progress.add_task(
            description=f"[cyan]Processing batches using {self.model}",
            total=self.n_total_requests,
            # Since we don't automatically retry failed requests within a run, we can count
            # failed downloaded requests as "completed". Users who require
            # 100% success rate can set require_all_responses to True and
            # manually retry.
            completed=self.n_downloaded_succeeded_requests + self.n_downloaded_failed_requests,
            batches_text="[bold white]Batches:[/bold white] [dim]--[/dim]",
            requests_text="[bold white]Requests:[/bold white] [dim]--[/dim]",
            tokens_text="[bold white]Tokens:[/bold white] [dim]--[/dim]",
            cost_text="[bold white]Cost:[/bold white] [dim]--[/dim]",
            price_text="[bold white]Model Pricing:[/bold white] [dim]--[/dim]",
        )
        
        # Create Live display that will show both logs and progress
        self._live = Live(
            Group(
                Panel(self._progress),
            ),
            console=self._console,
            refresh_per_second=4,
            transient=True  # This ensures logs stay above the progress bar
        )
        self._live.start()

    def stop_tracker(self):
        """Stop the tracker and display final statistics."""
        if hasattr(self, '_live'):
            # Refresh one last time to show final state
            self._progress.refresh()
            # Stop the live display
            self._live.stop()
            # Print the final progress state
            self._console.print(self._progress)
            self.display_final_stats()

        # update anonymized telemetry
        telemetry_client.capture(
            TelemetryEvent(
                event_type="BatchRequest",
                metadata=self.model_dump(),
            )
        )

    def __del__(self):
        """Ensure live display is stopped on deletion."""
        if hasattr(self, '_live'):
            self._live.stop()

    def update_display(self):
        """Update statistics with token usage and cost information."""
        # Format display texts
        batches_text = (
            "[bold white]Batches:[/bold white] "
            f"[white]Total:[/white] [blue]{self.n_total_batches}[/blue] "
            f"[white]•[/white] "
            f"[white]Submitted:[/white] [yellow]{self.n_submitted_batches}⋯[/yellow] "
            f"[white]•[/white] "
            f"[white]Downloaded:[/white] [green]{self.n_downloaded_batches}✓[/green]"
        )

        n_submitted_requests = self.n_total_requests - (self.n_downloaded_succeeded_requests + self.n_downloaded_failed_requests)
        requests_text = (
            "[bold white]Requests:[/bold white] "
            f"[white]Total:[/white] [blue]{self.n_total_requests}[/blue] "
            f"[white]•[/white] "
            f"[white]Submitted:[/white] [yellow]{n_submitted_requests}⋯[/yellow] "
            f"[white]•[/white] "
            f"[white]Succeeded:[/white] [green]{self.n_downloaded_succeeded_requests}✓[/green] "
            f"[white]•[/white] "
            f"[white]Failed:[/white] [red]{self.n_downloaded_failed_requests}✗[/red] "
        )

        avg_prompt = self.total_prompt_tokens / max(1, self.n_finished_or_downloaded_succeeded_requests)
        avg_completion = self.total_completion_tokens / max(1, self.n_finished_or_downloaded_succeeded_requests)

        tokens_text = (
            "[bold white]Tokens:[/bold white] "
            f"[white]Avg Input:[/white] [blue]{avg_prompt:.0f}[/blue] "
            f"[white]•[/white] "
            f"[white]Avg Output:[/white] [blue]{avg_completion:.0f}[/blue] "
        )

        avg_cost = self.total_cost / max(1, self.n_downloaded_succeeded_requests)
        projected_cost = avg_cost * self.n_total_requests

        cost_text = (
            "[bold white]Cost:[/bold white] "
            f"[white]Current:[/white] [magenta]${self.total_cost:.3f}[/magenta] "
            f"[white]•[/white] "
            f"[white]Projected:[/white] [magenta]${projected_cost:.3f}[/magenta] "
            f"[white]•[/white] "
            f"[white]Rate:[/white] [magenta]${avg_cost:.3f}/request[/magenta]"
        )

        input_cost_str = f"${self.input_cost_per_million:.3f}" if self.input_cost_per_million else "N/A"
        output_cost_str = f"${self.output_cost_per_million:.3f}" if self.output_cost_per_million else "N/A"

        price_text = (
            "[bold white]Model Pricing:[/bold white] "
            f"[white]Per 1M tokens:[/white] "
            f"[white]Input:[/white] [red]{input_cost_str}[/red] "
            f"[white]•[/white] "
            f"[white]Output:[/white] [red]{output_cost_str}[/red]"
        )

        # Update progress display
        self._progress.update(
            self._task_id,
            completed=self.n_downloaded_succeeded_requests + self.n_downloaded_failed_requests,
            batches_text=batches_text,
            requests_text=requests_text,
            tokens_text=tokens_text,
            cost_text=cost_text,
            price_text=price_text,
        )

    def display_final_stats(self):
        """Display final statistics."""
        table = Table(title="Final Curator Statistics", box=box.ROUNDED)
        table.add_column("Section/Metric", style="cyan")
        table.add_column("Value", style="yellow")

        # Model Information
        table.add_row("Model", "", style="bold magenta")
        table.add_row("Model", f"[blue]{self.model}[/blue]")

        # Batch Statistics
        table.add_row("Batches", "", style="bold magenta")
        table.add_row("Total Batches", str(self.n_total_batches))
        table.add_row("Submitted", f"[yellow]{self.n_submitted_batches}[/yellow]")
        table.add_row("Downloaded", f"[green]{self.n_downloaded_batches}[/green]")

        # Request Statistics
        table.add_row("Requests", "", style="bold magenta")
        table.add_row("Total Requests", str(self.n_total_requests))
        table.add_row("Successful", f"[green]{self.n_finished_or_downloaded_succeeded_requests}[/green]")
        table.add_row("Failed", f"[red]{self.n_finished_failed_requests + self.n_downloaded_failed_requests}[/red]")

        # Token Statistics
        table.add_row("Tokens", "", style="bold magenta")
        table.add_row("Total Tokens Used", f"{self.total_tokens:,}")
        table.add_row("Total Input Tokens", f"{self.total_prompt_tokens:,}")
        table.add_row("Total Output Tokens", f"{self.total_completion_tokens:,}")
        if self.n_finished_or_downloaded_succeeded_requests > 0:
            table.add_row("Average Tokens per Request", f"{int(self.total_tokens / self.n_finished_or_downloaded_succeeded_requests)}")
            table.add_row("Average Input Tokens", f"{int(self.total_prompt_tokens / self.n_finished_or_downloaded_succeeded_requests)}")
            table.add_row("Average Output Tokens", f"{int(self.total_completion_tokens / self.n_finished_or_downloaded_succeeded_requests)}")

        # Cost Statistics
        table.add_row("Costs", "", style="bold magenta")
        table.add_row("Total Cost", f"[red]${self.total_cost:.4f}[/red]")
        if self.n_finished_or_downloaded_succeeded_requests > 0:
            table.add_row("Average Cost per Request", f"[red]${self.total_cost / self.n_finished_or_downloaded_succeeded_requests:.4f}[/red]")

        # Handle None values for cost per million tokens
        input_cost_str = f"[red]${self.input_cost_per_million:.4f}[/red]" if self.input_cost_per_million else "[dim]N/A[/dim]"
        output_cost_str = f"[red]${self.output_cost_per_million:.4f}[/red]" if self.output_cost_per_million else "[dim]N/A[/dim]"

        table.add_row("Input Cost per 1M Tokens", input_cost_str)
        table.add_row("Output Cost per 1M Tokens", output_cost_str)

        # Performance Statistics
        table.add_row("Performance", "", style="bold magenta")
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        rpm = (self.n_downloaded_succeeded_requests + self.n_downloaded_failed_requests) / max(0.001, elapsed_minutes)
        input_tpm = self.total_prompt_tokens / max(0.001, elapsed_minutes)
        output_tpm = self.total_completion_tokens / max(0.001, elapsed_minutes)

        table.add_row("Total Time", f"{elapsed_time:.2f}s")
        table.add_row("Average Time per Request", f"{elapsed_time / max(1, self.n_finished_or_downloaded_succeeded_requests):.2f}s")
        table.add_row("Requests per Minute", f"{rpm:.1f}")
        table.add_row("Input Tokens per Minute", f"{input_tpm:.1f}")
        table.add_row("Output Tokens per Minute", f"{output_tpm:.1f}")

        self._console.print(table)

    @property
    def n_total_batches(self) -> int:
        """Get the total number of batches across all states."""
        return self.n_unsubmitted_request_files + self.n_submitted_batches + self.n_finished_batches + self.n_downloaded_batches

    @property
    def n_unsubmitted_request_files(self) -> int:
        """Get the number of unsubmitted request files."""
        return len(self.unsubmitted_request_files)

    @property
    def n_submitted_batches(self) -> int:
        """Get the number of submitted batches."""
        return len(self.submitted_batches)

    @property
    def n_finished_batches(self) -> int:
        """Get the number of finished batches."""
        return len(self.finished_batches)

    @property
    def n_downloaded_batches(self) -> int:
        """Get the number of downloaded batches."""
        return len(self.downloaded_batches)

    @property
    def n_finished_succeeded_requests(self) -> int:
        """Get the number of succeeded requests in submitted and finished batches.

        Returns:
            int: Total count of succeeded requests across submitted and finished batches.
        """
        batches = list(self.submitted_batches.values()) + list(self.finished_batches.values())
        return sum(b.request_counts.succeeded for b in batches)

    @property
    def n_finished_failed_requests(self) -> int:
        """Get the number of failed requests in finished batches.

        Returns:
            int: Total count of failed requests in finished batches.
        """
        batches = list(self.finished_batches.values())
        return sum(b.request_counts.failed for b in batches)

    @property
    def n_downloaded_succeeded_requests(self) -> int:
        """Get the number of succeeded requests in downloaded batches.

        Returns:
            int: Total count of succeeded requests in downloaded batches.
        """
        batches = list(self.downloaded_batches.values())
        return sum(b.request_counts.succeeded for b in batches)

    @property
    def n_downloaded_failed_requests(self) -> int:
        """Get the number of failed requests in downloaded batches.

        Returns:
            int: Total count of failed requests in downloaded batches.
        """
        batches = list(self.downloaded_batches.values())
        return sum(b.request_counts.failed for b in batches)

    @property
    def n_finished_or_downloaded_succeeded_requests(self) -> int:
        """Get the total number of succeeded requests across finished and downloaded batches.

        Returns:
            int: Combined count of succeeded requests from both finished and downloaded batches.
        """
        return self.n_finished_succeeded_requests + self.n_downloaded_succeeded_requests

    @property
    def n_submitted_finished_or_downloaded_batches(self) -> int:
        """Get the total number of batches that are submitted, finished, or downloaded."""
        return self.n_submitted_batches + self.n_finished_batches + self.n_downloaded_batches

    @property
    def n_finished_or_downloaded_batches(self) -> int:
        """Get the total number of batches that are either finished or downloaded."""
        return self.n_finished_batches + self.n_downloaded_batches

    def mark_as_submitted(self, batch: GenericBatch, n_requests: int):
        """Mark a batch as submitted and update tracking counters.

        Args:
            batch: The batch to mark as submitted
            n_requests: The number of requests in the batch
        """
        assert n_requests > 0
        batch.status = GenericBatchStatus.SUBMITTED
        if batch.request_file in self.unsubmitted_request_files:
            self.unsubmitted_request_files.remove(batch.request_file)
        else:
            logger.warning(f"Request file {batch.request_file} is being re-submitted.")
        self.submitted_batches[batch.id] = batch
        logger.debug(f"Marked {batch.request_file} as submitted with batch {batch.id}")
        self.update_display()

    def mark_as_finished(self, batch: GenericBatch):
        """Mark a batch as finished and move it to finished batches.

        Args:
            batch: The batch to mark as finished
        """
        assert batch.id in self.submitted_batches
        batch.status = GenericBatchStatus.FINISHED
        self.submitted_batches.pop(batch.id)
        self.finished_batches[batch.id] = batch
        logger.debug(f"Marked batch {batch.id} as finished")
        self.update_display()

    def mark_as_downloaded(self, batch: GenericBatch):
        """Mark a batch as downloaded and move it to downloaded batches.

        Args:
            batch: The batch to mark as downloaded
        """
        assert batch.id in self.finished_batches
        batch.status = GenericBatchStatus.DOWNLOADED
        self.finished_batches.pop(batch.id)
        self.downloaded_batches[batch.id] = batch
        logger.debug(f"Marked batch {batch.id} as downloaded")
        self.update_display()

    def update_submitted(self, batch: GenericBatch):
        """Update the request counts for a submitted batch.

        Args:
            batch: The batch to update
        """
        assert batch.id in self.submitted_batches
        self.submitted_batches[batch.id] = batch
        logger.debug(f"Updated submitted batch {batch.id} with new request counts")
        self.update_display()

    def __str__(self) -> str:
        """Return a human-readable string representation of the batch status."""
        status_lines = [
            f"Total batches: {self.n_total_batches}",
            f"Unsubmitted files: {self.n_unsubmitted_request_files}",
            f"Submitted batches: {self.n_submitted_batches}",
            f"Finished batches: {self.n_finished_batches}",
            f"Downloaded batches: {self.n_downloaded_batches}",
            "",
            f"Total requests: {self.n_total_requests}",
            f"Finished failed requests: {self.n_finished_failed_requests}",
            f"Finished succeeded requests: {self.n_finished_succeeded_requests}",
            f"Downloaded failed requests: {self.n_downloaded_failed_requests}",
            f"Downloaded succeeded requests: {self.n_downloaded_succeeded_requests}",
        ]
        return "\n".join(status_lines)

    def update_token_and_cost(self, token_usage: TokenUsage, cost: float):
        """Update statistics with token usage and cost information."""
        if token_usage:
            self.total_prompt_tokens += token_usage.prompt_tokens
            self.total_completion_tokens += token_usage.completion_tokens
            self.total_tokens += token_usage.total_tokens
        if cost:
            self.total_cost += cost
        self.update_display()
