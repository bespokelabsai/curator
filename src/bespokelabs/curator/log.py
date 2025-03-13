import logging
import logging.handlers
import os

from rich.console import Console
from rich.logging import RichHandler

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d - %(message)s"
ROOT_LOG_LEVEL = logging.DEBUG

_CONSOLE = Console(stderr=True)


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
            rich_handler = RichHandler(console=_CONSOLE)
            rich_handler.setLevel(logging.INFO)

            self.logger.addHandler(rich_handler)

    def get_logger(self, name):
        """Get logger instance."""
        return self.logger.getChild(name)


logger = Logger().get_logger(__name__)


def add_file_handler(log_dir):
    """Create a file handler and attach it to logger."""
    global logger
    log_file = os.path.join(log_dir, "curator.log")
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
