"""Live, artifact-local logging for independently reproducible V27d runs."""

from __future__ import annotations

import logging
import math
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




def format_metric(value: object, *, precision: int = 4) -> str:
    if int(precision) < 0:
        raise ValueError("V27d metric precision must be non-negative")
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{int(precision)}f}"




def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    name = str(event).strip()
    if not name:
        raise ValueError("V27d log event must be non-empty")
    if not fields:
        logger.info("%s", name)
        return
    parts: list[str] = []
    for raw_key in sorted(fields):
        key = str(raw_key).strip()
        if not key or "|" in key or "=" in key:
            raise ValueError("V27d log field names must be plain text")
        parts.append(f"{key}={_field_value(fields[raw_key])}")
    logger.info("%s | %s", name, " | ".join(parts))


def _field_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format_metric(value, precision=6)
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "/")




__all__ = ["close_logger", "configure_logger", "format_metric", "log_event"]