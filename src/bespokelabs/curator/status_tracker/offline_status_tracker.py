import datetime
import platform
from dataclasses import dataclass, field

try:
    import torch
except ImportError:
    pass


@dataclass
class System:
    """Tracks the system setup."""

    system: str = platform.system()
    version: str = platform.version()
    release: str = platform.release()
    device: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_count: int = torch.cuda.device_count()
    cuda_version: str = torch.version.cuda
    pytorch_version: str = torch.__version__

    def __str__(self):
        """String representation of the System class."""
        return (
            f"System: {self.system}\n"
            f"Version: {self.version}\n"
            f"Release: {self.release}\n"
            f"Device: {self.device}\n"
            f"Device Count: {self.device_count}\n"
            f"PyTorch Version: {self.pytorch_version}\n"
            f"CUDA Version: {self.cuda_version}"
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
