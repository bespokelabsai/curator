import logging
import shutil
from pathlib import Path

from bespokelabs.curator.log import add_file_handler, logger, remove_file_handlers


def _managed_file_handlers():
    return [handler for handler in logger.handlers if getattr(handler, "_curator_managed_file_handler", False)]


def test_add_file_handler_is_idempotent_and_replaces_stale_directories(tmp_path, monkeypatch):
    remove_file_handlers()

    run_dir = tmp_path / "run-1"
    next_run_dir = tmp_path / "run-2"

    def fail_on_logging_error(self, record):
        raise AssertionError("logging handler emitted an internal error")

    monkeypatch.setattr(logging.Handler, "handleError", fail_on_logging_error)

    try:
        add_file_handler(str(run_dir))
        add_file_handler(str(run_dir))

        handlers = _managed_file_handlers()
        assert len(handlers) == 1
        assert Path(handlers[0].baseFilename) == run_dir / "curator.log"

        logger.info("first run log")
        assert (run_dir / "curator.log").exists()

        shutil.rmtree(run_dir)
        logger.info("log after the working directory was deleted")
        assert _managed_file_handlers() == []

        add_file_handler(str(next_run_dir))
        handlers = _managed_file_handlers()
        assert len(handlers) == 1
        assert Path(handlers[0].baseFilename) == next_run_dir / "curator.log"

        logger.info("second run log")
        assert "second run log" in (next_run_dir / "curator.log").read_text()
    finally:
        remove_file_handlers()
