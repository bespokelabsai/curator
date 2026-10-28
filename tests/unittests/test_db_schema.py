import sqlite3

import pytest

from bespokelabs.curator.db import MetadataDB


def test_valid_schema(tmp_path):
    """Test that a valid schema does not raise any exceptions."""
    db_path = tmp_path / "metadata.db"
    db = MetadataDB(str(db_path))
    db.store_metadata(
        {
            "run_hash": "test",
            "session_id": "mysession",
            "dataset_hash": "hash",
            "prompt_func": "def prompt_func(): pass",
            "model_name": "test-model",
            "response_format": "{}",
            "batch_mode": False,
            "timestamp": "2023-01-01T00:00:00Z",
            "is_hosted_viewer_synced": False,
        }
    )
    # If no exception is raised, the test passes


def test_invalid_schema(tmp_path):
    """Test that an invalid schema raises RuntimeError with mismatch message."""
    db_path = tmp_path / "metadata.db"
    # Manually create a runs table with incorrect columns
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE runs (wrong_col TEXT)")

    db = MetadataDB(str(db_path))
    with pytest.raises(RuntimeError, match="mismatch"):
        db.store_metadata(
            {
                "run_hash": "test2",
                "session_id": "mysession2",
                "dataset_hash": "hash2",
                "prompt_func": "def prompt_func(): pass",
                "model_name": "test-model-2",
                "response_format": "{}",
                "batch_mode": True,
                "timestamp": "2023-01-01T01:00:00Z",
            }
        )


def test_get_metadata_allowlist(tmp_path):
    """Test that _get_metadata only accepts allowed column names and rejects malicious input."""
    db_path = tmp_path / "metadata.db"
    db = MetadataDB(str(db_path))

    # Store valid metadata
    db.store_metadata(
        {
            "run_hash": "test_hash",
            "session_id": "test_session",
            "dataset_hash": "hash",
            "prompt_func": "def prompt_func(): pass",
            "model_name": "test-model",
            "response_format": "{}",
            "batch_mode": False,
            "timestamp": "2023-01-01T00:00:00Z",
            "is_hosted_viewer_synced": True,
        }
    )

    # Test valid column names work
    assert db._get_metadata("test_hash", "session_id") == "test_session"
    assert db._get_metadata("test_hash", "is_hosted_viewer_synced") is True

    # Test that malicious column names raise ValueError without executing SQL
    with pytest.raises(ValueError, match="Invalid input"):
        db._get_metadata("test_hash", "run_hash; DROP TABLE runs--")

    with pytest.raises(ValueError, match="Invalid input"):
        db._get_metadata("test_hash", "dataset_hash")

    with pytest.raises(ValueError, match="Invalid input"):
        db._get_metadata("test_hash", "model_name")

    # Verify the table still exists and data is intact (SQL injection was prevented)
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM runs WHERE run_hash = ?", ("test_hash",))
        result = cursor.fetchone()
        assert result[0] == "test_session"
