import logging

import pytest

from conftest import REPO_ROOT
from utils.logging_utils import LogManager, add_run_separator, setup_logging

LOG_CONFIG = str(REPO_ROOT / "configs/logging_config.ini")


def test_add_run_separator(tmp_path):
    log = tmp_path / "run.log"
    add_run_separator(str(log))
    add_run_separator(str(log))
    assert log.read_text().count("New Run Started at") == 2


def test_setup_logging_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        setup_logging(str(tmp_path / "app.log"), str(tmp_path / "missing.ini"))


def test_log_manager_writes_to_file(tmp_path, restore_logging):
    log = tmp_path / "logs" / "application_test.log"  # parent dir does not exist yet
    LogManager.initialize(log_file_path=str(log), log_config_path=LOG_CONFIG)

    logger = LogManager.get_logger("test_logger")
    logger.info("hello from the test")
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = log.read_text()
    assert "New Run Started at" in text
    assert "hello from the test" in text
    assert LogManager._log_file_path == str(log)


def test_get_logger_is_cached():
    assert LogManager.get_logger("same") is LogManager.get_logger("same")


def test_get_logger_defaults_to_caller_module():
    assert LogManager.get_logger().name == __name__


def test_progress_logging_keeps_console_level(tmp_path, restore_logging, capsys):
    from utils.logging_utils import progress_logging

    LogManager.initialize(log_file_path=str(tmp_path / "app.log"), log_config_path=LOG_CONFIG)
    logger = LogManager.get_logger("progress_test")
    capsys.readouterr()
    with progress_logging():
        logger.info("info message")
        logger.warning("warning message")
    out = capsys.readouterr()
    console = out.out + out.err
    assert "warning message" in console
    assert "info message" not in console, "INFO must stay out of the console (WARNING level)"
    assert "info message" in (tmp_path / "app.log").read_text()
