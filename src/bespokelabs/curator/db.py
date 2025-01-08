"""Database class for storing metadata for Bella runs."""

import os
import sqlite3


class MetadataDB:
    """Database class for storing Bella run metadata."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_current_schema(self) -> list:
        """Get the current schema of the runs table from the database.

        Returns:
            list: List of tuples containing column information.
                  Each tuple contains (cid, name, type, notnull, dflt_value, pk)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(runs)")
            schema_info = cursor.fetchall()
        return schema_info

    def validate_schema(self):
        """Validate that the current database schema matches the expected schema.

        Raises:
            RuntimeError: If there is a mismatch between the current schema and expected schema,
                        with instructions to clear the cache.
        """
        # IMPORTANT: If you modify the CREATE TABLE statement in store_metadata(),
        # you must update this expected_columns list to match the new schema.
        # Otherwise, schema validation will fail and users will need to clear their cache.
        expected_columns = [
            "run_hash",
            "dataset_hash",
            "prompt_func",
            "model_name",
            "response_format",
            "batch_mode",
            "created_time",
            "last_edited_time",
        ]
        current_info = self._get_current_schema()
        current_columns = [col[1] for col in current_info]  # col[1] = column name

        if set(current_columns) != set(expected_columns):
            msg = (
                "Detected a mismatch between the local DB schema and the expected schema. "
                "Please clear your cache with `rm -rf ~/.cache/curator` or "
                "`rm -rf $CURATOR_CACHE_DIR` if set."
            )
            raise RuntimeError(msg)

    def store_metadata(self, metadata: dict):
        """Store metadata about a Bella run in the database.

        Args:
            metadata: Dictionary containing run metadata with keys:
                - timestamp: ISO format timestamp
                - dataset_hash: Unique hash of input dataset
                - prompt_func: Source code of prompt function
                - model_name: Name of model used
                - response_format: JSON schema of response format
                - run_hash: Unique hash identifying the run
                - batch_mode: Boolean indicating batch mode or online mode (True = batch, False = online)
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_hash TEXT PRIMARY KEY,
                    dataset_hash TEXT,
                    prompt_func TEXT,
                    model_name TEXT,
                    response_format TEXT,
                    batch_mode BOOLEAN,
                    created_time TEXT,
                    last_edited_time TEXT
                )
                """
            )
            self.validate_schema()

            # Check if run_hash exists
            cursor.execute(
                "SELECT run_hash FROM runs WHERE run_hash = ?",
                (metadata["run_hash"],),
            )
            existing_run = cursor.fetchone()

            if existing_run:
                # Update last_edited_time for existing entry
                cursor.execute(
                    """
                    UPDATE runs 
                    SET last_edited_time = ?
                    WHERE run_hash = ?
                    """,
                    (metadata["timestamp"], metadata["run_hash"]),
                )
            else:
                # Insert new entry
                cursor.execute(
                    """
                    INSERT INTO runs (
                        run_hash, dataset_hash, prompt_func, model_name, 
                        response_format, batch_mode, created_time, last_edited_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata["run_hash"],
                        metadata["dataset_hash"],
                        metadata["prompt_func"],
                        metadata["model_name"],
                        metadata["response_format"],
                        metadata["batch_mode"],
                        metadata["timestamp"],
                        "-",
                    ),
                )
            conn.commit()
