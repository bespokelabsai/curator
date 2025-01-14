"""Enhanced installation UI for bespokelabs-curator.

This module provides a rich, interactive UI for the installation process of the Curator package.
It includes progress tracking, status updates, and a polished success message.
"""

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text


class InstallationStage(Enum):
    """Enum representing different stages of the installation process."""

    PREPARING = ("Preparing your environment...", 0.0)
    COLLECTING = ("Downloading packages...", 0.2)
    DOWNLOADING = ("Downloading packages...", 0.4)
    INSTALLING = ("Installing...", 0.7)
    FINALIZING = ("Finalizing...", 0.9)
    COMPLETE = ("Installation complete!", 1.0)

    def __init__(self, message: str, progress: float):
        """Initialize the InstallationStage with a message and progress."""
        self.message = message
        self.progress = progress


@dataclass
class InstallationUI:
    """Class to manage the installation UI components and styling."""

    package_name: str
    console: Console = Console()

    def create_progress_bar(self, completed: float = 0) -> Text:
        """Create a stylish progress bar with the given completion percentage."""
        width = 40
        filled = int(width * completed)
        bar = Text()
        bar.append("\n╭", style="dim white")
        bar.append("─" * (width + 2), style="dim white")
        bar.append("╮\n", style="dim white")
        bar.append("│ ", style="dim white")
        bar.append("█" * filled, style="green")
        bar.append("▒" * (width - filled), style="dim white")
        bar.append(" │", style="dim white")
        bar.append("\n╰", style="dim white")
        bar.append("─" * (width + 2), style="dim white")
        bar.append("╯", style="dim white")
        return bar

    def create_loading_text(self, stage: InstallationStage, progress: float) -> Text:
        """Create the loading text with current stage and progress."""
        return Text.assemble(
            ("✨ Installing ", "bold"),
            (self.package_name, "bold cyan"),
            "\n",
            ("↳ ", "dim white"),
            ("Your synthetic data journey begins in moments", "dim white"),
            self.create_progress_bar(progress),
            ("\n ", ""),
            (stage.message, "italic dim white"),
        )

    def create_success_text(self) -> Text:
        """Create the success message with links."""
        text = Text()
        text.append("✨ Curator installed successfully!\n\n", style="bold green")
        text.append("Start building production-ready synthetic data pipelines:\n\n", style="dim white")
        text.append("   📚 ", style="")
        text.append("docs.bespokelabs.ai", style="dim cyan link https://docs.bespokelabs.ai")
        text.append("\n   📦 ", style="")
        text.append(
            "github.com/bespokelabsai/curator",
            style="dim cyan link https://github.com/bespokelabsai/curator",
        )
        text.append("\n   💬 ", style="")
        text.append("discord.gg/KqpXvpzVBS", style="dim cyan link https://discord.com/invite/KqpXvpzVBS")
        return text


class PackageInstaller:
    """Class to handle the package installation process."""

    def __init__(self, package_name: str, version: Optional[str] = None):
        """Initialize the PackageInstaller with the package name and optional version.

        Args:
            package_name: The name of the package to install
            version: Optional specific version to install
        """
        self.package_spec = f"{package_name}=={version}" if version else package_name
        self.ui = InstallationUI(package_name)

    def run_pip_install(self) -> subprocess.Popen:
        """Run pip install and capture output."""
        return subprocess.Popen(
            [sys.executable, "-m", "pip", "install", self.package_spec],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

    def parse_pip_output(self, line: str) -> Tuple[InstallationStage, float]:
        """Parse pip output to determine installation stage and progress."""
        line = line.strip().lower()

        if "collecting" in line:
            return InstallationStage.COLLECTING, InstallationStage.COLLECTING.progress
        elif "downloading" in line:
            if "%" in line:
                try:
                    percent = float(line.split("%")[0].split()[-1])
                    # Scale download progress between 20% and 60%
                    return InstallationStage.DOWNLOADING, 0.2 + (percent / 100.0 * 0.4)
                except Exception:
                    pass
            return InstallationStage.DOWNLOADING, InstallationStage.DOWNLOADING.progress
        elif "installing" in line:
            return InstallationStage.INSTALLING, InstallationStage.INSTALLING.progress
        elif "successfully installed" in line:
            return InstallationStage.FINALIZING, InstallationStage.FINALIZING.progress

        return InstallationStage.PREPARING, InstallationStage.PREPARING.progress

    def install(self) -> None:
        """Execute the installation with progress tracking and UI updates."""
        spinner = Spinner("dots2", text=self.ui.create_loading_text(InstallationStage.PREPARING, 0), style="green")

        with Live(spinner, console=self.ui.console, refresh_per_second=30) as live:
            try:
                process = self.run_pip_install()

                while True:
                    output_line = process.stdout.readline()
                    if output_line == "" and process.poll() is not None:
                        break

                    stage, progress = self.parse_pip_output(output_line)
                    spinner.text = self.ui.create_loading_text(stage, progress)

                # Show completion
                spinner.text = self.ui.create_loading_text(InstallationStage.COMPLETE, 1.0)

                if process.poll() == 0:
                    live.update(self.ui.create_success_text())
                else:
                    error = process.stderr.read()
                    error_text = Text(error, style="red")
                    live.update(error_text)
                    sys.exit(1)

            except Exception as e:
                error_text = Text(f"Error: {str(e)}", style="red")
                live.update(error_text)
                sys.exit(1)

        self.ui.console.print()


def enhanced_install(package_name: str, version: Optional[str] = None) -> None:
    """Enhance pip installation with a professional progress UI.

    Args:
        package_name: Name of the package to install
        version: Optional specific version to install
    """
    installer = PackageInstaller(package_name, version)
    installer.install()


if __name__ == "__main__":
    enhanced_install("bespokelabs-curator")
