"""Custom Pickler that combines HuggingFace's path normalization with type annotation support.

This module provides a CustomPickler class that extends HuggingFace's Pickler to support
both path normalization (for consistent function hashing across different environments)
and type annotations in function signatures.
"""

import os
from io import BytesIO
from typing import Any, Optional, Type, TypeVar, Union

import cloudpickle
from datasets.utils._dill import Pickler as HFPickler


class CustomPickler(HFPickler):
    """A custom pickler that combines HuggingFace's path normalization with type annotation support.

    This pickler extends HuggingFace's Pickler to:
    1. Preserve path normalization for consistent function hashing
    2. Support type annotations in function signatures
    3. Handle closure variables properly
    """

    def __init__(self, file: BytesIO, recurse: bool = True):
        """Initialize the CustomPickler.

        Args:
            file: The file-like object to pickle to
            recurse: Whether to recursively handle object attributes
        """
        super().__init__(file, recurse=recurse)

    def save(self, obj: Any, save_persistent_id: bool = True) -> None:
        """Save an object, handling type annotations properly.

        This method attempts to use cloudpickle's type annotation handling while
        preserving HuggingFace's path normalization logic.

        Args:
            obj: The object to pickle
            save_persistent_id: Whether to save persistent IDs
        """
        try:
            # First try HuggingFace's pickler for path normalization
            super().save(obj, save_persistent_id=save_persistent_id)
        except Exception as e:
            if "No default __reduce__ due to non-trivial __cinit__" in str(e):
                # If HF's pickler fails with type annotation error, use cloudpickle
                cloudpickle.dump(obj, self._file)
            else:
                # Re-raise other exceptions
                raise


def dumps(obj: Any) -> bytes:
    """Pickle an object to bytes using CustomPickler.

    Args:
        obj: The object to pickle

    Returns:
        The pickled object as bytes
    """
    file = BytesIO()
    CustomPickler(file, recurse=True).dump(obj)
    return file.getvalue()


def loads(data: bytes) -> Any:
    """Unpickle an object from bytes.

    Args:
        data: The pickled data

    Returns:
        The unpickled object
    """
    return cloudpickle.loads(data)
