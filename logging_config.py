"""Application logging configuration utilities."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(log_dir: Path, *, max_bytes: int = 1_048_576, backup_count: int = 5) -> Tuple[logging.Logger, Path]:
    """Configure application-wide logging with rotation.

    Parameters
    ----------
    log_dir:
        Directory where log files should be written.
    max_bytes:
        Maximum size of each log file before rotation.
    backup_count:
        Number of historical log files to retain.

    Returns
    -------
    tuple
        A tuple containing the configured logger and the path to the active log file.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    logger = logging.getLogger("flow_training_ui")
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        rotating_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        rotating_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(rotating_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(stream_handler)

        logger.propagate = False

    return logger, log_path

