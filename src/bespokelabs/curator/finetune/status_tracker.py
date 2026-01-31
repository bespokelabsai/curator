"""Status tracker for training progress display."""

import time
from dataclasses import dataclass, field
from typing import Optional

import tqdm
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from bespokelabs.curator.finetune.types import TrainingStats
from bespokelabs.curator.log import _CONSOLE, USE_RICH_DISPLAY
from bespokelabs.curator.status_tracker.tqdm_constants.colors import END, HEADER, METRIC, MODEL


@dataclass
class FinetuneStatusTracker:
    """Tracks and displays fine-tuning progress."""

    model: str = ""
    total_epochs: int = 0
    total_steps: int = 0
    batch_size: int = 1

    # Current stats
    current_epoch: int = 0
    current_step: int = 0
    current_loss: float = 0.0
    tokens_processed: int = 0
    samples_processed: int = 0
    learning_rate: float = 0.0
    loss_history: list = field(default_factory=list)

    # Timing
    start_time: float = field(default_factory=time.time, init=False)

    # Progress bar
    pbar: Optional[tqdm.tqdm] = field(default=None, repr=False, compare=False)

    def start_tracker(self, console: Optional[Console] = None):
        """Start the progress tracker."""
        if USE_RICH_DISPLAY:
            self._start_rich_tracker(console)
        else:
            self._start_tqdm_tracker()

    def _start_rich_tracker(self, console: Optional[Console] = None):
        """Start the rich progress tracker."""
        self._console = _CONSOLE if console is None else console

        # Create progress bar display
        self._progress = Progress(
            TextColumn("[bold blue]Training[/bold blue]"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold white]•[/bold white]"),
            TextColumn("Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
            TextColumn("[bold white]•[/bold white]"),
            TimeElapsedColumn(),
            TextColumn("[bold white]•[/bold white]"),
            TimeRemainingColumn(),
            console=self._console,
        )

        # Create stats display
        self._stats = Progress(
            TextColumn("{task.description}"),
            console=self._console,
        )

        # Add tasks
        self._task_id = self._progress.add_task(
            description="",
            total=self.total_steps,
            completed=0,
            epoch=1,
            total_epochs=self.total_epochs,
        )

        self._stats_task_id = self._stats.add_task(
            total=None,
            description=self._format_stats_text(),
        )

        # Create Live display
        self._live = Live(
            Panel(
                self._progress,
                title=f"[bold]Fine-tuning {self.model}[/bold]",
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
            total=self.total_steps,
            desc=f"Training {self.model}",
            unit="step",
        )
        self._last_stats_update = time.time()

    def _format_stats_text(self) -> str:
        """Format the stats text for display."""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.current_step / max(0.001, elapsed)
        tokens_per_sec = self.tokens_processed / max(0.001, elapsed)

        if USE_RICH_DISPLAY:
            return (
                f"[bold white]Loss:[/bold white] [yellow]{self.current_loss:.4f}[/yellow] "
                f"[bold white]•[/bold white] "
                f"[bold white]LR:[/bold white] [blue]{self.learning_rate:.2e}[/blue] "
                f"[bold white]•[/bold white] "
                f"[bold white]Steps/sec:[/bold white] [green]{steps_per_sec:.2f}[/green] "
                f"[bold white]•[/bold white] "
                f"[bold white]Tokens/sec:[/bold white] [green]{tokens_per_sec:.0f}[/green] "
                f"[bold white]•[/bold white] "
                f"[bold white]Samples:[/bold white] [blue]{self.samples_processed}[/blue]"
            )
        else:
            return (
                f"Loss: {self.current_loss:.4f} | "
                f"LR: {self.learning_rate:.2e} | "
                f"Steps/sec: {steps_per_sec:.2f} | "
                f"Tokens/sec: {tokens_per_sec:.0f} | "
                f"Samples: {self.samples_processed}"
            )

    def update(self, stats: TrainingStats):
        """Update the tracker with new training stats."""
        self.current_epoch = stats.current_epoch
        self.current_step = stats.current_step
        self.current_loss = stats.current_loss
        self.tokens_processed = stats.tokens_processed
        self.samples_processed = stats.samples_processed
        self.learning_rate = stats.learning_rate

        if stats.current_loss > 0:
            self.loss_history.append(stats.current_loss)

        self._update_display()

    def _update_display(self):
        """Update the progress display."""
        if USE_RICH_DISPLAY:
            self._progress.update(
                self._task_id,
                completed=self.current_step,
                epoch=self.current_epoch,
                total_epochs=self.total_epochs,
            )
            self._stats.update(
                self._stats_task_id,
                description=self._format_stats_text(),
            )

            # Update the panel to include stats
            self._live.update(
                Panel(
                    f"{self._progress}\n{self._stats}",
                    title=f"[bold]Fine-tuning {self.model}[/bold]",
                    box=box.ROUNDED,
                )
            )
        else:
            if self.pbar:
                self.pbar.n = self.current_step
                self.pbar.set_description(
                    f"Training {MODEL}{self.model}{END} " f"[Epoch {self.current_epoch}/{self.total_epochs} | " f"Loss: {METRIC}{self.current_loss:.4f}{END}]"
                )
                self.pbar.refresh()

    def stop_tracker(self):
        """Stop the tracker and display final statistics."""
        if USE_RICH_DISPLAY:
            if hasattr(self, "_live"):
                self._live.stop()
        else:
            if self.pbar:
                self.pbar.close()

        self.display_final_stats()

    def display_final_stats(self):
        """Display final training statistics."""
        elapsed = time.time() - self.start_time
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0
        final_loss = self.loss_history[-1] if self.loss_history else 0.0

        if USE_RICH_DISPLAY:
            self._display_rich_final_stats(elapsed, avg_loss, final_loss)
        else:
            self._display_simple_final_stats(elapsed, avg_loss, final_loss)

    def _display_rich_final_stats(self, elapsed: float, avg_loss: float, final_loss: float):
        """Display final stats using rich table."""
        table = Table(title="Fine-tuning Complete", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Model", f"[blue]{self.model}[/blue]")
        table.add_row("Total Epochs", str(self.total_epochs))
        table.add_row("Total Steps", str(self.current_step))
        table.add_row("Final Loss", f"[green]{final_loss:.4f}[/green]")
        table.add_row("Average Loss", f"{avg_loss:.4f}")
        table.add_row("Total Time", f"{elapsed:.2f}s")
        table.add_row("Samples Processed", str(self.samples_processed))
        table.add_row("Tokens Processed", f"{self.tokens_processed:,}")
        table.add_row("Steps/second", f"{self.current_step / max(0.001, elapsed):.2f}")
        table.add_row("Tokens/second", f"{self.tokens_processed / max(0.001, elapsed):.0f}")

        self._console.print(table)

    def _display_simple_final_stats(self, elapsed: float, avg_loss: float, final_loss: float):
        """Display final stats in plain text."""
        stats = [
            f"\n{HEADER}Fine-tuning Complete{END}",
            f"  Model: {MODEL}{self.model}{END}",
            f"  Total Epochs: {METRIC}{self.total_epochs}{END}",
            f"  Total Steps: {METRIC}{self.current_step}{END}",
            f"  Final Loss: {METRIC}{final_loss:.4f}{END}",
            f"  Average Loss: {METRIC}{avg_loss:.4f}{END}",
            f"  Total Time: {METRIC}{elapsed:.2f}s{END}",
            f"  Samples Processed: {METRIC}{self.samples_processed}{END}",
            f"  Tokens Processed: {METRIC}{self.tokens_processed:,}{END}",
            f"  Steps/second: {METRIC}{self.current_step / max(0.001, elapsed):.2f}{END}",
            f"  Tokens/second: {METRIC}{self.tokens_processed / max(0.001, elapsed):.0f}{END}",
        ]
        print("\n".join(stats))

    def get_stats(self) -> TrainingStats:
        """Get current training stats."""
        return TrainingStats(
            current_epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            current_step=self.current_step,
            total_steps=self.total_steps,
            current_loss=self.current_loss,
            tokens_processed=self.tokens_processed,
            samples_processed=self.samples_processed,
            learning_rate=self.learning_rate,
            elapsed_time=time.time() - self.start_time,
        )
