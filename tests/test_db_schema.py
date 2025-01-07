import pytest
import sqlite3
from bespokelabs.curator.db import MetadataDB


def test_valid_schema(tmp_path):
    """Test that a valid schema does not raise any exceptions."""
    db_path = tmp_path / "metadata.db"
    db = MetadataDB(str(db_path))
    db.store_metadata({
        "run_hash": "test",
        "dataset_hash": "hash",
        "prompt_func": "def prompt_func(): pass",
        "model_name": "test-model",
        "response_format": "{}",
        "batch_mode": False,
        "timestamp": "2023-01-01T00:00:00Z",
    })
    # If no exception is raised, the test passes


def test_invalid_schema(tmp_path):
    """Test that an invalid schema raises RuntimeError with mismatch message."""
    db_path = tmp_path / "metadata.db"
    # Manually create a runs table with incorrect columns
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE runs (wrong_col TEXT)")
    
    db = MetadataDB(str(db_path))
    with pytest.raises(RuntimeError, match="mismatch"):
        db.store_metadata({
            "run_hash": "test2",
            "dataset_hash": "hash2",
            "prompt_func": "def prompt_func(): pass",
            "model_name": "test-model-2",
            "response_format": "{}",
            "batch_mode": True,
            "timestamp": "2023-01-01T01:00:00Z",
        })
