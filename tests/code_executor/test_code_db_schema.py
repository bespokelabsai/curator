import sqlite3

import pytest

from bespokelabs.curator.code_executor.db import CodeMetadataDB


def test_valid_schema(tmp_path):
    """Test that a valid schema does not raise any exceptions."""
    db_path = tmp_path / "metadata.db"
    db = CodeMetadataDB(str(db_path))
    db.store_metadata(
        {
            "run_hash": "test",
            "dataset_hash": "hash",
            "code": "def test(): pass",
            "code_input": "test input",
            "code_output": "test output",
            "timestamp": "2023-01-01T00:00:00Z",
        }
    )
    # If no exception is raised, the test passes


def test_invalid_schema(tmp_path):
    """Test that an invalid schema raises RuntimeError with mismatch message."""
    db_path = tmp_path / "metadata.db"
    # Manually create a runs_code table with incorrect columns
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE runs_code (wrong_col TEXT)")

    db = CodeMetadataDB(str(db_path))
    with pytest.raises(RuntimeError, match="mismatch"):
        db.store_metadata(
            {
                "run_hash": "test2",
                "dataset_hash": "hash2",
                "code": "def test2(): pass",
                "code_input": "test input 2",
                "code_output": "test output 2",
                "timestamp": "2023-01-01T01:00:00Z",
            }
        )


def test_get_current_schema(tmp_path):
    """Test that _get_current_schema returns correct schema info."""
    db_path = tmp_path / "metadata.db"
    db = CodeMetadataDB(str(db_path))

    # Create table with known schema
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE runs_code (
                run_hash TEXT PRIMARY KEY,
                dataset_hash TEXT,
                code TEXT,
                code_input TEXT,
                code_output TEXT,
                created_time TEXT,
                last_edited_time TEXT
            )
        """)

    schema_info = db._get_current_schema()

    # Verify schema info contains expected columns
    column_names = [col[1] for col in schema_info]
    expected_columns = ["run_hash", "dataset_hash", "code", "code_input", "code_output", "created_time", "last_edited_time"]
    assert set(column_names) == set(expected_columns)


def test_store_metadata_new_entry(tmp_path):
    """Test storing new metadata entry."""
    db_path = tmp_path / "metadata.db"
    db = CodeMetadataDB(str(db_path))

    metadata = {
        "run_hash": "test3",
        "dataset_hash": "hash3",
        "code": "def test3(): pass",
        "code_input": "test input 3",
        "code_output": "test output 3",
        "timestamp": "2023-01-01T02:00:00Z",
    }

    db.store_metadata(metadata)

    # Verify entry was stored correctly
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM runs_code WHERE run_hash = ?", (metadata["run_hash"],))
        row = cursor.fetchone()

    assert row is not None
    assert row[0] == metadata["run_hash"]
    assert row[1] == metadata["dataset_hash"]
    assert row[2] == metadata["code"]
    assert row[3] == metadata["code_input"]
    assert row[4] == metadata["code_output"]
    assert row[5] == metadata["timestamp"]
    assert row[6] == "-"  # Default last_edited_time


def test_store_metadata_update_existing(tmp_path):
    """Test updating existing metadata entry."""
    db_path = tmp_path / "metadata.db"
    db = CodeMetadataDB(str(db_path))

    # First insertion
    metadata = {
        "run_hash": "test4",
        "dataset_hash": "hash4",
        "code": "def test4(): pass",
        "code_input": "test input 4",
        "code_output": "test output 4",
        "timestamp": "2023-01-01T03:00:00Z",
    }
    db.store_metadata(metadata)

    # Update with same run_hash
    metadata_update = metadata.copy()
    metadata_update["timestamp"] = "2023-01-01T04:00:00Z"
    db.store_metadata(metadata_update)

    # Verify entry was updated
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM runs_code WHERE run_hash = ?", (metadata["run_hash"],))
        row = cursor.fetchone()

    assert row is not None
    assert row[6] == metadata_update["timestamp"]  # last_edited_time should be updated
