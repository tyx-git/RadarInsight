from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.evaluator import strip_detailed_records, write_sample_detail_information
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    log_low_confidence_records,
    log_metric_set,
)


def migrate(
    summary_path: Path,
    *,
    backup_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training summary must be a JSON object")
    if backup_path is not None:
        backup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    history = payload.get("history", [])
    best_epoch = int(payload.get("best_epoch", -1))
    best_record = next((item for item in history if int(item.get("epoch", -1)) == best_epoch), None)
    if not isinstance(best_record, dict):
        raise ValueError(f"best epoch {best_epoch} is not present in training history")
    validation = best_record.get("validation")
    if not isinstance(validation, dict) or "records" not in validation:
        raise ValueError("best epoch validation does not contain detailed records")
    target_summary_path = summary_path if output_path is None else output_path
    artifact_dir = target_summary_path.parent
    best_checkpoint = payload.get("best_checkpoint") or payload.get("best_checkpoint_path")
    if best_checkpoint is None:
        raise ValueError("training summary has no best checkpoint field")
    detail_path = write_sample_detail_information(
        artifact_dir,
        validation,
        checkpoint_path=best_checkpoint,
        epoch=best_epoch,
        selection_metric=best_record.get("selection_metric"),
    )
    logger = configure_training_logger(artifact_dir / "training.log")
    try:
        logger.info("migration | source=legacy_training_summary | best_epoch=%d", best_epoch)
        log_metric_set(logger, "val:parent_final_prefix", validation.get("parent_metrics", {}))
        log_metric_set(logger, "val:stable_final", validation.get("stable_metrics") or {})
        log_metric_set(logger, "val:window_causal_prefix", validation.get("window_metrics", {}))
        log_low_confidence_records(logger, validation)
        logger.info("final_metrics | status=complete | best_checkpoint=%s | sample_details=%s", best_checkpoint, detail_path)
    finally:
        close_training_logger(logger)
    migrated = dict(payload)
    migrated.pop("best_checkpoint_path", None)
    migrated["best_checkpoint"] = str(best_checkpoint)
    migrated["sample_detail_information"] = str(detail_path)
    migrated["best_validation_metrics"] = strip_detailed_records(validation)
    migrated["history"] = []
    for item in history:
        row = dict(item)
        if isinstance(row.get("validation"), dict):
            row["validation"] = strip_detailed_records(row["validation"])
        migrated["history"].append(row)
    target_summary_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "passed",
        "summary_path": str(target_summary_path),
        "legacy_backup": None if backup_path is None else str(backup_path),
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": best_epoch,
        "sample_detail_information": str(detail_path),
        "summary_contains_records": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a RADAR-LN3 training summary to the compact output contract.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--backup", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = migrate(
        args.summary.resolve(),
        backup_path=None if args.backup is None else args.backup.resolve(),
        output_path=None if args.output is None else args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
