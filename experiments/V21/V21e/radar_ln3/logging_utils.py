from __future__ import annotations

import logging
import math
from pathlib import Path


TRAINING_LOGGER_NAME = "radar_ln3.training"


class _CompactFormatter(logging.Formatter):
    """Short human-readable records suitable for both console and file."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        return f"{timestamp}.{record.msecs:03.0f} | {record.levelname:<5} | {record.getMessage()}"


def configure_training_logger(
    log_path: str | Path,
    *,
    logger_name: str = TRAINING_LOGGER_NAME,
    console: bool = True,
) -> logging.Logger:
    """Create a non-propagating training logger without duplicate handlers.

    Console records go to stderr so the training CLI can keep stdout as a
    machine-readable JSON result. The same compact records are appended to
    the artifact-local ``training.log`` file.
    """

    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = _CompactFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def close_training_logger(logger: logging.Logger) -> None:
    """Flush and close all handlers owned by a training run."""

    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        handler.close()


def format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:05.1f}s"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}h{minutes:02d}m{remainder:02d}s"
    return f"{minutes:02d}m{remainder:02d}s"


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.4f}"








__all__ = ["TRAINING_LOGGER_NAME", "close_training_logger", "configure_training_logger", "format_duration", "format_metric"]