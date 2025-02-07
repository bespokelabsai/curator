"""BespokeLabs Curator."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from .code_executor.code_executor import CodeExecutor
from .llm.llm import LLM
from .types import prompt as types

__all__ = ["LLM", "CodeExecutor", "types"]

_CONSOLE = Console(stderr=True)
logger = logging.getLogger("bespokelabs.curator")

logger.setLevel(logging.WARNING)

logger.addHandler(RichHandler(console=_CONSOLE))
