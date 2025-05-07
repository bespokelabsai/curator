from __future__ import annotations

import datetime
import platform
from dataclasses import dataclass, field

try:
    import torch
except ImportError:
    torch = None  # type: ignore


@dataclass
class System:
    """Information about the current host and (optionally) PyTorch device."""

    system: str = field(init=False)
    version: str = field(init=False)
    release: str = field(init=False)

    device: str = field(init=False)
    device_count: int = field(init=False)
    cuda_version: str = field(init=False)
    pytorch_version: str = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        # ――― OS info ―――
        self.system = platform.system()
        self.version = platform.version()
        self.release = platform.release()

        # ――― PyTorch‑specific details (if available) ―――
        if torch is not None:
            self.pytorch_version = torch.__version__

            # CUDA GPUs
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                idx = torch.cuda.current_device()
                self.device = torch.cuda.get_device_name(idx)
                self.device_count = torch.cuda.device_count()
                self.cuda_version = torch.version.cuda or "Unknown"

            # Apple‑Silicon / Metal backend (PyTorch ≥ 1.13)
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "MPS"
                self.device_count = 1
                self.cuda_version = "N/A"

            # CPU‑only PyTorch build
            else:
                self.device = "CPU"
                self.device_count = 0
                self.cuda_version = "N/A"

        else:  # PyTorch not installed
            self.pytorch_version = "Not installed"
            self.device = "CPU"
            self.device_count = 0
            self.cuda_version = "N/A"

    def __str__(self) -> str:
        """String representation of the System class."""
        return (
            f"System:          {self.system}\n"
            f"Version:         {self.version}\n"
            f"Release:         {self.release}\n"
            f"Device:          {self.device}\n"
            f"Device Count:    {self.device_count}\n"
            f"PyTorch Version: {self.pytorch_version}\n"
            f"CUDA Version:    {self.cuda_version}"
        )


@dataclass
class OfflineStatusTracker:
    """Tracks the status of all requests."""

    time_started: datetime.datetime = field(default_factory=datetime.datetime.now)
    time_finished: datetime.datetime = None
    finished_successfully: bool = False
    num_total_requests: int = 0
    system: System = field(default_factory=System)
    num_parsed_responses: int = 0

    def __str__(self):
        """String representation of the OfflineStatusTracker class."""
        return (
            f"Started: {self.time_started}\n"
            f"Finished: {self.time_finished}\n"
            f"Success: {self.finished_successfully}\n"
            f"Total Requests: {self.num_total_requests}\n"
            f"System: {self.system}"
        )
