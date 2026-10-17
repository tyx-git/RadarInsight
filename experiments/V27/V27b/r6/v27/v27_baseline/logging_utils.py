"""Live, artifact-local logging for V27 training runs.

The implementation is deliberately compatible with the established LN3
training logger: human-readable records are appended to a task-local file and
mirrored to stderr, leaving CLI stdout reserved for the final JSON result.
It has no model, data, or calibration dependency, so future V27 stages can
reuse the same lifecycle and progress helpers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path


TRAINING_LOGGER_PREFIX = "radar_ln3.v27.training"


class _CompactFormatter(logging.Formatter):
    """Render one concise, tail-friendly line for console and file output."""

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
    """Flush, detach, and close handlers owned by a configured run logger."""

    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        handler.close()


def configure_training_logger(
    log_path: str | Path,
    *,
    logger_name: str = TRAINING_LOGGER_PREFIX,
    console: bool = True,
) -> logging.Logger:
    """Configure one non-propagating file-and-console training logger.

    Reconfiguration is intentional: it first removes old handlers for the
    exact logger name, preventing duplicate records in notebook, smoke, or
    sequential CLI invocations.  Console messages use stderr so stdout stays
    machine-readable for the V27 CLI result.
    """

    name = str(logger_name).strip()
    if not name:
        raise ValueError("V27 training logger_name must be non-empty")
    target = Path(log_path)
    if not target.name:
        raise ValueError("V27 training log_path must name a file")
    target.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
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


def close_training_logger(logger: logging.Logger) -> None:
    """Close all handlers created for a V27 training logger."""

    _close_handlers(logger)


def format_duration(seconds: float) -> str:
    """Format elapsed wall time without embedding presentation policy in loops."""

    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:05.1f}s"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}h{minutes:02d}m{remainder:02d}s"
    return f"{minutes:02d}m{remainder:02d}s"


def format_metric(value: object, *, precision: int = 4) -> str:
    """Render optional finite scalar metrics consistently in training logs."""

    if int(precision) < 0:
        raise ValueError("V27 metric precision must be non-negative")
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{int(precision)}f}"


def progress_interval(steps_per_epoch: int) -> int:
    """Return a bounded interval that yields about ten progress records/epoch."""

    steps = int(steps_per_epoch)
    if steps <= 0:
        raise ValueError("V27 logging requires a positive steps_per_epoch")
    return max(1, min(50, steps // 10))


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    """Write a stable key-value event shared by all V27 run stages.

    Keeping the envelope here makes log lines easy to tail and parse while
    allowing future knownness, calibration, and audit stages to add fields
    without duplicating formatting code.
    """

    name = str(event).strip()
    if not name:
        raise ValueError("V27 log event must be non-empty")
    if not fields:
        logger.info("%s", name)
        return
    parts: list[str] = []
    for key in sorted(fields):
        field_name = str(key).strip()
        if not field_name or "|" in field_name or "=" in field_name:
            raise ValueError("V27 log field names must be non-empty plain text")
        parts.append(f"{field_name}={_format_field_value(fields[key])}")
    logger.info("%s | %s", name, " | ".join(parts))


def _format_field_value(value: object) -> str:
    """Render a primitive event field without introducing ambiguous pipes."""

    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format_metric(value, precision=6)
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text.replace("|", "/")




__all__ = ["TRAINING_LOGGER_PREFIX", "close_training_logger", "configure_training_logger", "format_duration", "format_metric", "log_event", "progress_interval"]