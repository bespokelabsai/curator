import logging
import logging.handlers
import os
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d - %(message)s"
ROOT_LOG_LEVEL = logging.DEBUG

# Check if rich CLI is disabled
RICH_CLI_DISABLED = os.environ.get("CURATOR_DISABLE_RICH_CLI", "0") == "1"

# Use standard console if rich is disabled
_CONSOLE = None if RICH_CLI_DISABLED else Console(stderr=True)


class Logger:
    """Curator Logger class with handlers."""

    _instance = None

    def __new__(cls):
        """Python new method."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        self.logger = logging.getLogger("curator")
        self.logger.setLevel(ROOT_LOG_LEVEL)
        if not self.logger.handlers:
            if RICH_CLI_DISABLED:
                # Use standard logging handler when rich CLI is disabled
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(LOG_FORMAT))
            else:
                handler = RichHandler(console=_CONSOLE)
            handler.setLevel(logging.INFO)
            self.logger.addHandler(handler)

    def get_logger(self, name):
        """Get logger instance."""
        return self.logger.getChild(name)


logger = Logger().get_logger(__name__)


def get_progress_bar(total: int, desc: Optional[str] = None):
    """Get appropriate progress bar based on CLI settings."""
    if RICH_CLI_DISABLED:
        return tqdm(total=total, desc=desc)
    return None


def add_file_handler(log_dir):
    """Create a file handler and attach it to logger."""
    global logger
    log_file = os.path.join(log_dir, "curator.log")
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
