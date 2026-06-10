"""Data formatter for converting datasets to the Fireworks chat JSONL format.

Fireworks AI supervised fine-tuning expects an OpenAI-compatible chat-completion
JSONL file: one JSON object per line of the form
``{"messages": [{"role": ..., "content": ...}, ...]}``. This formatter reuses the
shared :class:`DataFormatter` row -> :class:`TrainingExample` logic and adds the
JSONL serialization that the Fireworks Dataset upload requires.
"""

import json
from typing import Any, Dict, List

from bespokelabs.curator.finetune.data_formatter import DataFormatter
from bespokelabs.curator.finetune.types import TrainingExample


class FireworksDataFormatter(DataFormatter):
    """Converts curator datasets to the Fireworks chat JSONL format."""

    def example_to_dict(self, example: TrainingExample) -> Dict[str, Any]:
        """Convert a TrainingExample into a Fireworks chat-format dict.

        Args:
            example: The training example to convert.

        Returns:
            A dict ``{"messages": [{"role": ..., "content": ...}, ...]}``.
        """
        return {"messages": [{"role": msg.role, "content": msg.content} for msg in example.messages]}

    def to_jsonl_lines(self, examples: List[TrainingExample]) -> List[str]:
        """Serialize a list of examples to Fireworks chat JSONL lines.

        Args:
            examples: The training examples to serialize.

        Returns:
            A list of JSON strings, one per training example.
        """
        return [json.dumps(self.example_to_dict(example), ensure_ascii=False) for example in examples]

    def write_jsonl(self, examples: List[TrainingExample], path: str) -> str:
        """Write examples to a JSONL file in Fireworks chat format.

        Args:
            examples: The training examples to write.
            path: Destination ``.jsonl`` file path.

        Returns:
            The path that was written.
        """
        lines = self.to_jsonl_lines(examples)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        return path
