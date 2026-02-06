"""module for curator blocks."""

__all__: list[str] = []

try:
    from bespokelabs.curator.blocks.gepa import CuratorAdapter, EvaluationResult  # noqa: F401
except ModuleNotFoundError as exc:
    # `gepa` is an optional dependency (optimizer extra).
    if exc.name not in {"gepa", "gepa.core", "gepa.core.adapter"}:
        raise
else:
    __all__.extend(["CuratorAdapter", "EvaluationResult"])
