"""
This module provides utility functions and classes for setting up and managing logging in the project.
"""

import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime
from typing import Dict
import inspect
from contextlib import contextmanager

from tqdm.contrib.logging import logging_redirect_tqdm


class ReopeningFileHandler(logging.FileHandler):
    """
    FileHandler that reopens its log file when writing to it fails.

    File sync clients (e.g. OneDrive) can replace a file while it is open, after which every
    write through the old handle fails ("TimeoutError: [Errno 60] Operation timed out").
    The record is then written through a newly opened handle instead of being lost.
    """

    def emit(self, record):
        try:
            msg = self.format(record) + self.terminator
            if self.stream is None:
                self.stream = self._open()
            try:
                self.stream.write(msg)
                self.stream.flush()
            except OSError:
                self._reopen()
                self.stream.write(msg)
                self.stream.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def _reopen(self):
        try:
            self.stream.close()
        except OSError:
            pass  # buffered data of the failed write cannot be flushed through the old handle
        self.stream = self._open()


def add_run_separator(log_file_path: str):
    """
    Add a separator to the log file to mark the beginning of a new run.

    Args:
        log_file_path (str): Path to the log file.
    """
    separator = f"\n{'='*50}\n"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_header = f"New Run Started at {timestamp}\n"
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{separator}{run_header}{separator}\n")

def setup_logging(log_file_path: str, log_config_path: str) -> None:
    """
    Set up logging configuration for the project using logging_config.ini.

    Args:
        log_file_path (str): Path to the log file
        log_config_path (str): Path to the log configuration file
    """
    # Path to the logging config file
    print(f"Reading logging config from: {log_config_path}")
    if not Path(log_config_path).exists():
        raise FileNotFoundError(f"Logging config file not found: {log_config_path}")
    

    # Ensure the logs directory exists
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add run separator to the log file
    add_run_separator(log_file_path)

    # Load the logging configuration
    logging.config.fileConfig(
        log_config_path,
        defaults={'log_file_path': log_file_path},
        disable_existing_loggers=False
    )
    
    logging.info(f"Logging initialized. Check {log_file_path} for logs.")

@contextmanager
def progress_logging():
    """
    Route console log messages through tqdm so they do not break progress bars.

    tqdm's logging_redirect_tqdm replaces the console handler without copying its level,
    which would print every INFO message; the original console level is restored here.
    """
    root = logging.getLogger()
    console_levels = [
        h.level for h in root.handlers
        if type(h) is logging.StreamHandler  # console handlers, not FileHandler
    ]
    with logging_redirect_tqdm():
        if console_levels:
            for handler in root.handlers:
                if type(handler).__name__ == "_TqdmLoggingHandler":
                    handler.setLevel(min(console_levels))
        yield

class LogManager:
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _log_file_path: str = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, log_file_path: str, log_config_path: str):
        """
        Initialize the LogManager with the log file path.
        
        Args:
            log_file_path (str): Path to the log file.
            log_config_path (str): Path to the log configuration file.
        """
        cls._log_file_path = log_file_path
        setup_logging(log_file_path, log_config_path)

    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """
        Get or create a logger with the given name. If no name is provided,
        use the name of the module that called this method.
        
        Args:
            name (str, optional): Name of the logger to get or create.
        
        Returns:
            logging.Logger: The requested logger.
        """
        if name is None:
            # Get the caller's module name
            frame = inspect.currentframe().f_back
            module = inspect.getmodule(frame)
            name = module.__name__

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        return cls._loggers[name]

# Example usage
if __name__ == "__main__":
    # This is just for demonstration. In real use, the log_file_path
    # should be read from config.yaml in the main script.
    log_config = 'configs/logging_config.ini'
    example_log_path = "logs/test_logutils.log"
    LogManager.initialize(example_log_path, log_config)
    
    logger1 = LogManager.get_logger()  # Will use '__main__' as the logger name
    logger2 = LogManager.get_logger("custom_name")

    logger1.info("This is a log from the main script")
    logger2.warning("This is a warning from a custom named logger")
    
    print(f"Logs are being written to: {LogManager._log_file_path}")