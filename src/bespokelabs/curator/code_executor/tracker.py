import os
import time
from dataclasses import dataclass, field
from typing import Optional

from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table


@dataclass
class CodeExecutionStatusTracker:
    """Status tracker for code execution."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_execution_errors: int = 0
    num_other_errors: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0
    # Stats tracking
    total_requests: int = 0

    start_time: float = field(default_factory=time.time, init=False)

    def start_tracker(self, console: Optional[Console] = None):
        """Start the tracker."""
        if os.environ.get("BESPOKE_CURATOR_TRACKER_DISABLED", "0") == "1":
            return

        """Start the tracker."""
        self._console = Console() if console is None else console
        self._progress = Progress(
            TextColumn(
                "[cyan]{task.description}[/cyan]\n{task.fields[requests_text]}\n{task.fields[time_text]}",
                justify="left",
            ),
            TextColumn("\n\n\n"),  # Spacer
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self._console,
        )
        self._task_id = self._progress.add_task(
            description="[cyan]Processing Tasks",
            total=self.total_requests,
            completed=self.num_tasks_already_completed,
            requests_text="[bold white]Requests:[/bold white] [dim]--[/dim]",
            time_text="[bold white]Time:[/bold white] [dim]--[/dim]",
        )
        self._progress.start()

    def update_stats(self):
        """Update statistics in the tracker."""
        if os.environ.get("BESPOKE_CURATOR_TRACKER_DISABLED", "0") == "1":
            return

        """Update statistics in the tracker."""
        # Calculate current rpm
        elapsed_minutes = (time.time() - self.start_time) / 60
        current_rpm = self.num_tasks_succeeded / max(0.001, elapsed_minutes)

        requests_text = (
            "[bold white]Requests:[/bold white] "
            f"[white]Total:[/white] [blue]{self.total_requests}[/blue] "
            f"[white]•[/white] "
            f"[white]Started:[/white] [blue]{self.num_tasks_started}[/blue] "
            f"[white]•[/white] "
            f"[white]Success:[/white] [green]{self.num_tasks_succeeded}✓[/green] "
            f"[white]•[/white] "
            f"[white]Failed:[/white] [red]{self.num_tasks_failed}✗[/red] "
            f"[white]•[/white] "
            f"[white]In Progress:[/white] [yellow]{self.num_tasks_in_progress}⋯[/yellow]"
        )

        time_text = (
            "[bold white]Time:[/bold white] "
            f"[white]Elapsed:[/white] [blue]{elapsed_minutes:.1f}m[/blue] "
            f"[white]•[/white] "
            f"[white]RPM:[/white] [blue]{current_rpm:.1f}[/blue]"
        )

        # Update the progress display
        self._progress.update(
            self._task_id,
            completed=self.num_tasks_succeeded + self.num_tasks_already_completed,
            requests_text=requests_text,
            time_text=time_text,
        )

    def stop_tracker(self):
        """Stop the tracker."""
        if os.environ.get("BESPOKE_CURATOR_TRACKER_DISABLED", "0") == "1":
            return

        self._progress.stop()
        table = Table(title="Final Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        # Request Statistics
        table.add_row("Total Requests", str(self.total_requests))
        table.add_row("Tasks Started", str(self.num_tasks_started))
        table.add_row("Successful Tasks", f"[green]{self.num_tasks_succeeded}[/green]")
        table.add_row("Failed Tasks", f"[red]{self.num_tasks_failed}[/red]")
        table.add_row("Already Completed", str(self.num_tasks_already_completed))

        # Error Statistics
        table.add_row("Execution Errors", f"[red]{self.num_execution_errors}[/red]")
        table.add_row("Other Errors", f"[red]{self.num_other_errors}[/red]")

        # Performance Statistics
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        rpm = self.num_tasks_succeeded / max(0.001, elapsed_minutes)

        table.add_row("Total Time", f"{elapsed_time:.2f}s")
        table.add_row("Requests per Minute", f"{rpm:.1f}")

        self._console.print(table)

    def __str__(self):
        """String representation of the status tracker."""
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}\n"
            f"Errors - Execution: {self.num_execution_errors}, "
            f"Other: {self.num_other_errors}"
        )

    def has_capacity(self):
        """Check if there is capacity."""
        return self.num_tasks_in_progress < self.max_requests_per_minute

    def consume_capacity(self):
        """Consume capacity."""
        self.num_tasks_in_progress += 1

    def free_capacity(self):
        """Free up capacity."""
        self.num_tasks_in_progress -= 1
