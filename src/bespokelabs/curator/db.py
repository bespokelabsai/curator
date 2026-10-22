"""Database class for storing metadata for Bella runs."""

import os
import sqlite3

from bespokelabs.curator.log import logger


class MetadataDB:
    """Database class for storing Bella run metadata."""

    def __init__(self, db_path: str):
        """Initialize the MetadataDB with a given database path.

        Args:
            db_path: Path to the SQLite database file.
        """
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
        expected_columns = [
            "run_hash",
            "session_id",
            "dataset_hash",
            "prompt_func",
            "model_name",
            "response_format",
            "batch_mode",
            "created_time",
            "last_edited_time",
            "is_hosted_viewer_synced",
            # Cost management columns
            "total_cost_milli_dollars",
            "total_requests",
            "total_prompt_tokens",
            "total_completion_tokens",
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
            # IMPORTANT: If you modify the CREATE TABLE schema below,
            # you must update the expected_columns list in validate_schema()
            # to match the new schema. Otherwise, schema validation will fail.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_hash TEXT PRIMARY KEY,
                    session_id TEXT,
                    dataset_hash TEXT,
                    prompt_func TEXT,
                    model_name TEXT,
                    response_format TEXT,
                    batch_mode BOOLEAN,
                    created_time TEXT,
                    last_edited_time TEXT,
                    is_hosted_viewer_synced BOOLEAN,

                    total_cost_milli_dollars TEXT,
                    total_requests TEXT,
                    total_prompt_tokens TEXT,
                    total_completion_tokens TEXT
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
                        run_hash, session_id, dataset_hash, prompt_func, model_name,
                        response_format, batch_mode, created_time, last_edited_time, is_hosted_viewer_synced
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata["run_hash"],
                        metadata["session_id"],
                        metadata["dataset_hash"],
                        metadata["prompt_func"],
                        metadata["model_name"],
                        metadata["response_format"],
                        metadata["batch_mode"],
                        metadata["timestamp"],
                        metadata["timestamp"],  # last_edited_time same as created_time initially
                        metadata["is_hosted_viewer_synced"],
                    ),
                )
            conn.commit()

    def get_existing_session_id(self, run_hash: str):
        """Get existing session id from previous run."""
        return self._get_metadata(run_hash, "session_id")

    def _get_metadata(self, run_hash: str, column: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {column} FROM runs WHERE run_hash = ?",
                    (run_hash,),
                )
                fetch = cursor.fetchone()
                if fetch:
                    return fetch[0]

        except Exception:
            return None

    def update_session_id(self, run_hash: str, session_id: str):
        """Point a run at the viewer session that it now uses.

        The new session holds none of the run's data yet, so the run is not
        synced to the viewer until its upload to that session completes.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET session_id = ?, is_hosted_viewer_synced = ? WHERE run_hash = ?",
                (session_id, False, run_hash),
            )
            conn.commit()

    def store_viewer_write_token(self, session_id: str | None, write_token: str | None):
        """Keep the write token of a viewer session made without an API key.

        The viewer refuses any change to such a session without the token, so a
        later run that adds to the same session needs it. It has a table of its
        own: validate_schema checks the runs table's columns.

        Only the file's owner may read a token. If the file cannot be limited
        to its owner, for example in a cache that another user owns, no token
        is kept, and the next run starts a new session.
        """
        if not session_id or not write_token:
            return
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Before the first token is written, so SQLite's journal files get
            # the same mode.
            try:
                os.chmod(self.db_path, 0o600)
            except OSError as e:
                logger.warning(f"Not keeping the Curator Viewer write token: {self.db_path} could not be made readable only by its owner ({e}).")
                return
            conn.execute("CREATE TABLE IF NOT EXISTS viewer_write_tokens (session_id TEXT PRIMARY KEY, write_token TEXT NOT NULL)")
            conn.execute(
                "INSERT OR REPLACE INTO viewer_write_tokens (session_id, write_token) VALUES (?, ?)",
                (session_id, write_token),
            )
            conn.commit()

    def get_viewer_write_token(self, session_id: str | None) -> str | None:
        """Get the write token kept for a viewer session, or None."""
        if not session_id:
            return None
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT write_token FROM viewer_write_tokens WHERE session_id = ?", (session_id,)).fetchone()
        except sqlite3.Error:
            # A cache from before write tokens has no such table.
            return None
        return row[0] if row else None

    def check_existing_hosted_sync(self, run_hash: str) -> bool:
        """Check if the run is already hosted on the viewer."""
        return bool(self._get_metadata(run_hash, "is_hosted_viewer_synced"))

    def update_sync_viewer_flag(self, run_hash: str, hosted: bool):
        """Update the hosted_viewer boolean for a run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE runs SET is_hosted_viewer_synced = ? WHERE run_hash = ?",
                (hosted, run_hash),
            )
            conn.commit()
