"""Tail-friendly logging shared by V27b training stages."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


LOGGER_PREFIX = "radar_ln3.v27b"


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        line = f"{timestamp}.{record.msecs:03.0f} | {record.levelname:<5} | {record.getMessage()}"
        if record.exc_info:
            line += f"\n{self.formatException(record.exc_info)}"
        return line


def _close(logger: logging.Logger) -> None:
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
    logger_name = str(name).strip()
    if not logger_name:
        raise ValueError("V27b logger name must be non-empty")
    target = Path(path)
    if not target.name:
        raise ValueError("V27b log path must name a file")
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _close(logger)
    formatter = _Formatter(datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    _close(logger)


def format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:05.1f}s"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}h{minutes:02d}m{remainder:02d}s" if hours else f"{minutes:02d}m{remainder:02d}s"


def format_metric(value: object, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(numeric) else f"{numeric:.{int(precision)}f}"


def progress_interval(steps_per_epoch: int) -> int:
    steps = int(steps_per_epoch)
    if steps <= 0:
        raise ValueError("V27b logging requires positive steps_per_epoch")
    return max(1, min(50, steps // 10))


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    label = str(event).strip()
    if not label:
        raise ValueError("V27b log event must be non-empty")
    parts: list[str] = []
    for key in sorted(fields):
        field = str(key).strip()
        if not field or "|" in field or "=" in field:
            raise ValueError("V27b log field name is invalid")
        value = fields[key]
        if value is None:
            rendered = "n/a"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, float):
            rendered = format_metric(value, precision=6)
        else:
            rendered = str(value).replace("\r", " ").replace("\n", " ").replace("|", "/")
        parts.append(f"{field}={rendered}")
    logger.info("%s%s", label, f" | {' | '.join(parts)}" if parts else "")


def log_metric_set(logger: logging.Logger, scope: str, metrics: Mapping[str, Any]) -> None:
    logger.info(
        "metrics | scope=%s | support=%s | accuracy=%s | macro_precision=%s | macro_recall=%s | macro_f1=%s | balanced_accuracy=%s",
        scope,
        metrics.get("support", "n/a"),
        format_metric(metrics.get("accuracy")),
        format_metric(metrics.get("macro_precision")),
        format_metric(metrics.get("macro_recall")),
        format_metric(metrics.get("macro_f1")),
        format_metric(metrics.get("balanced_accuracy")),
    )
    for item in metrics.get("per_class", ()):
        if not isinstance(item, Mapping):
            raise ValueError("V27b per-class metric entry must be a mapping")
        logger.info(
            "class_metrics | scope=%s | class=%s | precision=%s | recall=%s | f1=%s | support=%s | confusion_to=%s | confusion_count=%s",
            scope,
            item.get("class", "n/a"),
            format_metric(item.get("precision")),
            format_metric(item.get("recall")),
            format_metric(item.get("f1")),
            item.get("support", "n/a"),
            item.get("major_confusion_to", "n/a"),
            item.get("major_confusion_count", "n/a"),
        )


__all__ = ["LOGGER_PREFIX", "close_logger", "configure_logger", "format_duration", "format_metric", "log_event", "log_metric_set", "progress_interval"]