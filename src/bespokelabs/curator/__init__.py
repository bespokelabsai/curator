"""BespokeLabs Curator."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("bespokelabs-curator")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .code_executor.code_executor import CodeExecutor
from .llm.llm import LLM
from .types import prompt as types
from .utils import load_dataset, push_to_viewer

__all__ = ["LLM", "CodeExecutor", "types", "push_to_viewer", "load_dataset", "__version__"]

from .log import _CONSOLE  # noqa: F401
