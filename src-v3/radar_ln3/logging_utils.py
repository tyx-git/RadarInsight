from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any


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


def progress_interval(steps_per_epoch: int) -> int:
    """Log about ten progress records per epoch, capped at fifty steps."""

    return max(1, min(50, max(1, int(steps_per_epoch) // 10)))


def log_metric_set(logger: logging.Logger, scope: str, metrics: dict[str, Any] | None) -> None:
    metrics = metrics or {}
    logger.info(
        "metrics | scope=%s | support=%s | accuracy=%s | macro_precision=%s | macro_recall=%s | macro_f1=%s | balanced_accuracy=%s | target_90=%s",
        scope,
        metrics.get("support", "n/a"),
        format_metric(metrics.get("accuracy")),
        format_metric(metrics.get("macro_precision")),
        format_metric(metrics.get("macro_recall")),
        format_metric(metrics.get("macro_f1")),
        format_metric(metrics.get("balanced_accuracy")),
        metrics.get("target_90_percent_status", "n/a"),
    )
    for item in metrics.get("per_class", []):
        logger.info(
            "class_metrics | scope=%s | class=%s | precision=%s | recall=%s | f1=%s | support=%s | confusion_to=%s | confusion_count=%s | target_90=%s",
            scope,
            item.get("class", "n/a"),
            format_metric(item.get("precision")),
            format_metric(item.get("recall")),
            format_metric(item.get("f1")),
            item.get("support", "n/a"),
            item.get("major_confusion_to", "n/a"),
            item.get("major_confusion_count", "n/a"),
            item.get("reaches_90_percent_all_metrics", "n/a"),
        )
    confusion = metrics.get("drone_bird_confusion", {})
    logger.info(
        "confusion | scope=%s | DroneTarget_to_BirdTarget=%s | BirdTarget_to_DroneTarget=%s",
        scope,
        confusion.get("DroneTarget_to_BirdTarget", "n/a"),
        confusion.get("BirdTarget_to_DroneTarget", "n/a"),
    )


def log_low_confidence_records(
    logger: logging.Logger,
    validation: dict[str, Any],
    *,
    threshold: float = 0.60,
) -> None:
    records = validation.get("records", {})
    logger.info("low_confidence | threshold=%.2f | scopes=window,parent,stable", threshold)
    if not isinstance(records, dict):
        logger.info("low_confidence | scope=all | count=0 | reason=records_unavailable")
        return
    for scope in ("window", "parent", "stable"):
        rows = [
            row for row in records.get(scope, [])
            if isinstance(row, dict) and float(row.get("confidence", 1.0)) < threshold
        ]
        logger.info("low_confidence | scope=%s | count=%d", scope, len(rows))
        for row in rows:
            logger.info(
                "low_confidence_sample | scope=%s | parent=%s | label=%s | true_id=%s | pred_id=%s | confidence=%.4f | event_index=%s | window_length=%s | stable_confirmed=%s",
                scope,
                row.get("parent_sample_id", "n/a"),
                row.get("label_name", "n/a"),
                row.get("true_id", "n/a"),
                row.get("pred_id", "n/a"),
                float(row.get("confidence", 0.0)),
                row.get("event_index", "n/a"),
                row.get("window_length", row.get("event_count", "n/a")),
                row.get("stable_confirmed", "n/a"),
            )


__all__ = [
    "TRAINING_LOGGER_NAME",
    "close_training_logger",
    "configure_training_logger",
    "format_duration",
    "format_metric",
    "log_low_confidence_records",
    "log_metric_set",
    "progress_interval",
]
