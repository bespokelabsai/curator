"""Database class for storing metadata for Bella runs."""

import os
import sqlite3


class MetadataDB:
    """Database class for storing Bella run metadata."""

    def __init__(self, db_path: str):
        self.db_path = db_path

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
