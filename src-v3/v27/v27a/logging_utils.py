"""Small, reusable logging primitives for V27a experiment runs."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


LOGGER_PREFIX = "radar_ln3.v27a"


class _CompactFormatter(logging.Formatter):
    """Render one tail-friendly line while preserving exception details."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        line = (
            f"{timestamp}.{record.msecs:03.0f} | "
            f"{record.levelname:<5} | {record.getMessage()}"
        )
        if record.exc_info:
            return f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        handler.close()


def configure_logger(
    log_path: str | Path,
    *,
    logger_name: str,
    console: bool = True,
) -> logging.Logger:
    """Configure one non-propagating file logger and optional stderr mirror."""

    name = str(logger_name).strip()
    if not name:
        raise ValueError("V27a logger_name must be non-empty")
    target = Path(log_path)
    if not target.name:
        raise ValueError("V27a log_path must name a file")
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


def close_logger(logger: logging.Logger) -> None:
    """Flush and detach all handlers owned by the supplied logger."""

    _close_handlers(logger)


def format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:05.1f}s"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}h{minutes:02d}m{remainder:02d}s"
    return f"{minutes:02d}m{remainder:02d}s"


def format_metric(value: object, *, precision: int = 4) -> str:
    if int(precision) < 0:
        raise ValueError("V27a metric precision must be non-negative")
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
    steps = int(steps_per_epoch)
    if steps <= 0:
        raise ValueError("V27a logging requires a positive step count")
    return max(1, min(50, steps // 10))


def _field_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format_metric(value, precision=6)
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "/")


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    name = str(event).strip()
    if not name:
        raise ValueError("V27a log event must be non-empty")
    if not fields:
        logger.info("%s", name)
        return
    parts: list[str] = []
    for raw_key in sorted(fields):
        key = str(raw_key).strip()
        if not key or "|" in key or "=" in key:
            raise ValueError("V27a log field names must be plain text")
        parts.append(f"{key}={_field_value(fields[raw_key])}")
    logger.info("%s | %s", name, " | ".join(parts))


def log_metric_set(
    logger: logging.Logger,
    scope: str,
    metrics: Mapping[str, Any] | None,
) -> None:
    name = str(scope).strip()
    if not name:
        raise ValueError("V27a metric scope must be non-empty")
    values = metrics or {}
    logger.info(
        "metrics | scope=%s | support=%s | accuracy=%s | macro_precision=%s "
        "| macro_recall=%s | macro_f1=%s | balanced_accuracy=%s",
        name,
        values.get("support", "n/a"),
        format_metric(values.get("accuracy")),
        format_metric(values.get("macro_precision")),
        format_metric(values.get("macro_recall")),
        format_metric(values.get("macro_f1")),
        format_metric(values.get("balanced_accuracy")),
    )
    for item in values.get("per_class", ()):
        if not isinstance(item, Mapping):
            raise ValueError("V27a per-class metric entry must be a mapping")
        logger.info(
            "class_metrics | scope=%s | class=%s | precision=%s | recall=%s "
            "| f1=%s | support=%s | confusion_to=%s | confusion_count=%s",
            name,
            item.get("class", "n/a"),
            format_metric(item.get("precision")),
            format_metric(item.get("recall")),
            format_metric(item.get("f1")),
            item.get("support", "n/a"),
            item.get("major_confusion_to", "n/a"),
            item.get("major_confusion_count", "n/a"),
        )


__all__ = [
    "LOGGER_PREFIX",
    "close_logger",
    "configure_logger",
    "format_duration",
    "format_metric",
    "log_event",
    "log_metric_set",
    "progress_interval",
]
