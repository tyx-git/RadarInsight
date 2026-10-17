"""Live, artifact-local logging for independently reproducible V27d runs."""

from __future__ import annotations

import logging
from pathlib import Path




class _CompactFormatter(logging.Formatter):
    """Render concise tail-friendly records for file and stderr output."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        message = (
            f"{timestamp}.{record.msecs:03.0f} | "
            f"{record.levelname:<5} | {record.getMessage()}"
        )
        if record.exc_info:
            return f"{message}\n{self.formatException(record.exc_info)}"
        return message


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        handler.close()


def configure_logger(
    path: str | Path,
    *,
    name: str,
    console: bool = True,
) -> logging.Logger:
    """Configure one non-propagating file logger and optional stderr mirror."""

    logger_name = str(name).strip()
    if not logger_name:
        raise ValueError("V27d logger name must be non-empty")
    target = Path(path)
    if not target.name:
        raise ValueError("V27d log path must name a file")
    target.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _close_handlers(logger)
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


def close_logger(logger: logging.Logger) -> None:
    """Flush and detach all handlers created for one V27d run."""

    _close_handlers(logger)














__all__ = ["close_logger", "configure_logger"]