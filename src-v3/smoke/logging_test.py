from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_low_confidence_records,
    log_metric_set,
    progress_interval,
)
from radar_ln3.task_models import summarize_task_records


def run() -> dict[str, object]:
    with TemporaryDirectory(prefix="radar-ln3-logging-") as temporary:
        log_path = Path(temporary) / "training.log"
        logger = configure_training_logger(log_path, console=False)
        if len(logger.handlers) != 1:
            raise AssertionError("file-only logger must have exactly one handler")
        logger.info("progress | epoch=01/20 | step=0001/0541 | loss=1.234567")
        metrics = summarize_task_records(
            [{"true_id": 0, "pred_id": 0}, {"true_id": 1, "pred_id": 0}],
            class_names=("DBB", "CU"),
            scope="probe",
        )
        log_metric_set(logger, "val:parent_final_prefix", metrics)
        log_low_confidence_records(
            logger,
            {
                "records": {
                    "parent": [
                        {"parent_sample_id": "low-parent", "confidence": 0.59},
                        {"parent_sample_id": "high-parent", "confidence": 0.95},
                    ]
                }
            },
            threshold=0.60,
        )
        close_training_logger(logger)
        content = log_path.read_text(encoding="utf-8")
        for token in ("INFO", "progress", "loss=1.234567", " | "):
            if token not in content:
                raise AssertionError(f"training log is missing token: {token}")
        for token in ("class_metrics", "class=DBB", "class=CU", "parent=low-parent"):
            if token not in content:
                raise AssertionError(f"training metric log is missing token: {token}")
        if "parent=high-parent" in content:
            raise AssertionError("training log emitted a sample at or above the confidence threshold")

        second_logger = configure_training_logger(log_path, console=False)
        if len(second_logger.handlers) != 1:
            raise AssertionError("reconfigured logger must not duplicate handlers")
        close_training_logger(second_logger)
        result = {
            "status": "passed",
            "format_duration_65": format_duration(65.0),
            "progress_interval_541": progress_interval(541),
            "handler_duplication": "not_detected",
            "per_class_metrics_logged": True,
            "low_confidence_only": True,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
