import logging
import logging.handlers
import os
import sys
import threading

from rich.console import Console
from rich.logging import RichHandler

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d - %(message)s"
ROOT_LOG_LEVEL = logging.DEBUG

# Check environment variable for display mode
USE_RICH_DISPLAY = os.environ.get("CURATOR_DISABLE_RICH_DISPLAY", "0").lower() not in (
    "1",
    "true",
    "yes",
)

# Create console based on display mode
if USE_RICH_DISPLAY:
    _CONSOLE = Console(stderr=True)
else:
    _CONSOLE = None


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
        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            self.logger.propagate = False
        if not self.logger.handlers:
            if USE_RICH_DISPLAY:
                rich_handler = RichHandler(console=_CONSOLE)
                rich_handler.setLevel(logging.INFO)
                self.logger.addHandler(rich_handler)
            else:
                # Use standard logging handler for non-rich mode
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
                stream_handler.setLevel(logging.INFO)
                self.logger.addHandler(stream_handler)

    def get_logger(self, name):
        """Get logger instance."""
        return self.logger.getChild(name)


logger = Logger().get_logger(__name__)
_FILE_HANDLER_LOCK = threading.RLock()


class _CuratorRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Detach cleanly when the backing run directory has been removed."""

    def __init__(self, *args, owner_logger: logging.Logger, **kwargs):
        self._owner_logger = owner_logger
        super().__init__(*args, **kwargs)
        self._curator_managed_file_handler = True

    def emit(self, record):
        if not os.path.isdir(os.path.dirname(self.baseFilename)):
            self._detach()
            return
        super().emit(record)

    def _detach(self):
        with _FILE_HANDLER_LOCK:
            if self._owner_logger is not None:
                self._owner_logger.removeHandler(self)
                self._owner_logger = None
            self.close()


def _iter_managed_file_handlers():
    return [handler for handler in logger.handlers if getattr(handler, "_curator_managed_file_handler", False)]


def remove_file_handlers():
    """Remove and close Curator-managed file handlers."""
    global logger
    with _FILE_HANDLER_LOCK:
        for handler in _iter_managed_file_handlers():
            logger.removeHandler(handler)
            handler.close()


def add_file_handler(log_dir):
    """Attach a single Curator-managed file handler for the active run."""
    global logger
    log_file = os.path.abspath(os.path.join(log_dir, "curator.log"))
    with _FILE_HANDLER_LOCK:
        for handler in _iter_managed_file_handlers():
            if os.path.abspath(handler.baseFilename) == log_file:
                return

        remove_file_handlers()
        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(LOG_FORMAT)
        file_handler = _CuratorRotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            owner_logger=logger,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
