"""Diagnose completed V12 DBB-subclass auxiliary treatments without using test data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from radar_ln3.task_models import (
    DBB_AUXILIARY_CLASS_NAMES,
    DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION,
    DBB_AUXILIARY_PAIRED_PROTOCOL,
    DBB_AUXILIARY_TREATMENT_WEIGHT,
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    summarize_task_records,
    task_device,
)


SEEDS = (42, 43, 44)
ARMS = ("control", "treatment")
DBB_LABELS = frozenset(DBB_AUXILIARY_CLASS_NAMES)
PAIRED_REPORT = Path("src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-paired-validation.json")
OUTPUT_REPORT = Path("src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-diagnostic.json")


def _config_path(seed: int, arm: str) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V12-cq08-liquid-dbb-cu-gate-dbb-auxiliary-{arm}-seed{seed}-config.json"
    )


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _model_inputs(tensors: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tr_features": tensors["tr_features"],
        "tr_missing_mask": tensors["tr_missing_mask"],
        "tr_dt_seconds": tensors["tr_dt_seconds"],
        "tr_dt_valid": tensors["tr_dt_valid"],
        "tr_motion_features": tensors.get("tr_motion_features"),
        "tr_motion_valid_mask": tensors.get("tr_motion_valid_mask"),
        "rd_physical_features": tensors.get("rd_physical_features"),
        "rd_physical_valid": tensors.get("rd_physical_valid"),
        "rd_frames": tensors["rd_frames"],
        "rd_frame_mask": tensors["rd_frame_mask"],
        "rd_quality": tensors["rd_quality"],
        "tr_padding_mask": tensors["tr_padding_mask"],
    }


def _evaluate_final_states(
    root: Path,
    *,
    seed: int,
    split_name: str,
) -> dict[str, Any]:
    config_path = root / _config_path(seed, "treatment")
    config = load_task_config(config_path)
    if (
        config.schema_version != DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION
        or config.train.paired_protocol != DBB_AUXILIARY_PAIRED_PROTOCOL
        or config.train.paired_arm != "treatment"
        or config.train.dbb_auxiliary_loss_weight != DBB_AUXILIARY_TREATMENT_WEIGHT
    ):
        raise ValueError(f"not a frozen V12 Treatment config: {config_path}")
    dataset = build_task_dataset(root, config, split_name=split_name)
    loader = build_task_dataloader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
    )
    device = task_device(config)
    model = LiquidTaskClassifier(config).to(device)
    checkpoint_path = root / config.train.checkpoint_path
    payload = load_task_checkpoint(
        checkpoint_path,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(
                **_model_inputs(tensors),
                return_dbb_auxiliary_logits=True,
            )
            gate_probabilities = torch.softmax(output["window_logits"], dim=-1)
            auxiliary_sequence = output["dbb_auxiliary_sequence_logits"]
            valid = ~tensors["tr_padding_mask"]
            last_indices = valid.long().sum(dim=1).clamp_min(1) - 1
            batch_indices = torch.arange(valid.shape[0], device=device)
            auxiliary_probabilities = torch.softmax(
                auxiliary_sequence[batch_indices, last_indices],
                dim=-1,
            )
            for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                gate_probability = gate_probabilities[batch_index]
                auxiliary_probability = auxiliary_probabilities[batch_index]
                gate_pred_id = int(gate_probability.argmax().item())
                auxiliary_pred_id = int(auxiliary_probability.argmax().item())
                auxiliary_true_id = int(batch["dbb_auxiliary_labels"][batch_index].item())
                sorted_auxiliary = torch.sort(auxiliary_probability, descending=True).values
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "label_name": str(batch["label_names"][batch_index]),
                        "gate_true_id": int(batch["labels"][batch_index].item()),
                        "gate_pred_id": gate_pred_id,
                        "gate_confidence": float(gate_probability[gate_pred_id].item()),
                        "auxiliary_true_id": auxiliary_true_id,
                        "auxiliary_pred_id": auxiliary_pred_id,
                        "auxiliary_pred_name": DBB_AUXILIARY_CLASS_NAMES[auxiliary_pred_id],
                        "auxiliary_confidence": float(auxiliary_probability[auxiliary_pred_id].item()),
                        "auxiliary_margin": float((sorted_auxiliary[0] - sorted_auxiliary[1]).item()),
                    }
                )
    if len(records) != len(dataset):
        raise AssertionError("V12 auxiliary diagnostic did not emit exactly one final-state row per parent")
    dbb_records = [row for row in records if int(row["auxiliary_true_id"]) >= 0]
    metrics = summarize_task_records(
        (
            {
                "true_id": int(row["auxiliary_true_id"]),
                "pred_id": int(row["auxiliary_pred_id"]),
            }
            for row in dbb_records
        ),
        class_names=DBB_AUXILIARY_CLASS_NAMES,
        scope=f"{split_name}_dbb_parent_final_state_auxiliary_head",
    )

    by_label: dict[str, Any] = {}
    for label_name in (
        "DroneTarget",
        "BirdTarget",
        "BalloonTarget",
        "ClutterTarget",
        "UnknownTarget",
    ):
        selected = [row for row in records if row["label_name"] == label_name]
        by_label[label_name] = {
            "support": len(selected),
            "gate_prediction_counts": dict(sorted(Counter("DBB" if row["gate_pred_id"] == 0 else "CU" for row in selected).items())),
            "forced_auxiliary_prediction_counts": dict(
                sorted(Counter(str(row["auxiliary_pred_name"]) for row in selected).items())
            ),
            "auxiliary_max_probability": _summary(float(row["auxiliary_confidence"]) for row in selected),
            "auxiliary_margin": _summary(float(row["auxiliary_margin"]) for row in selected),
        }

    def mechanism_rows(label_name: str, gate_pred_id: int) -> list[dict[str, Any]]:
        return [
            row
            for row in records
            if row["label_name"] == label_name and int(row["gate_pred_id"]) == gate_pred_id
        ]

    bird_to_cu = mechanism_rows("BirdTarget", 1)
    unknown_to_dbb = mechanism_rows("UnknownTarget", 0)
    return {
        "seed": seed,
        "split_name": split_name,
        "device": str(device),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "parent_count": len(records),
        "dbb_parent_count": len(dbb_records),
        "cu_parent_count": len(records) - len(dbb_records),
        "dbb_auxiliary_metrics": metrics,
        "by_semantic_label": by_label,
        "gate_error_mechanism": {
            "BirdTarget_to_CU": {
                "count": len(bird_to_cu),
                "auxiliary_still_predicts_BirdTarget_count": sum(
                    row["auxiliary_pred_name"] == "BirdTarget" for row in bird_to_cu
                ),
                "auxiliary_still_predicts_BirdTarget_rate": (
                    sum(row["auxiliary_pred_name"] == "BirdTarget" for row in bird_to_cu) / len(bird_to_cu)
                    if bird_to_cu
                    else None
                ),
                "forced_auxiliary_prediction_counts": dict(
                    sorted(Counter(str(row["auxiliary_pred_name"]) for row in bird_to_cu).items())
                ),
                "parent_sample_ids": sorted(str(row["parent_sample_id"]) for row in bird_to_cu),
            },
            "UnknownTarget_to_DBB": {
                "count": len(unknown_to_dbb),
                "forced_auxiliary_prediction_counts": dict(
                    sorted(Counter(str(row["auxiliary_pred_name"]) for row in unknown_to_dbb).items())
                ),
                "auxiliary_max_probability": _summary(
                    float(row["auxiliary_confidence"]) for row in unknown_to_dbb
                ),
                "parent_sample_ids": sorted(str(row["parent_sample_id"]) for row in unknown_to_dbb),
            },
        },
        "interpretation_guard": (
            "Clutter/Unknown were excluded from auxiliary training; their forced three-way softmax label and "
            "confidence are descriptive only and are not a knownness or rejection score."
        ),
    }


def _class_metrics(parent_metrics: Mapping[str, Any], class_name: str) -> Mapping[str, Any]:
    matches = [row for row in parent_metrics["per_class"] if row["class"] == class_name]
    if len(matches) != 1:
        raise ValueError(f"missing {class_name} parent metric")
    return matches[0]


def _all_epoch_feasibility(root: Path) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    total_epoch_count = 0
    total_feasible_count = 0
    for seed in SEEDS:
        seed_runs: dict[str, Any] = {}
        for arm in ARMS:
            config = load_task_config(root / _config_path(seed, arm))
            summary_path = root / config.train.formal_artifact_dir / "training-summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            history = list(summary.get("history") or [])
            if summary.get("status") != "passed" or len(history) != 20:
                raise ValueError(f"V12 all-epoch diagnostic requires a complete 20-epoch run: {summary_path}")
            rows: list[dict[str, Any]] = []
            for epoch in history:
                parent_metrics = epoch["validation"]["parent_metrics"]
                dbb = _class_metrics(parent_metrics, "DBB")
                cu = _class_metrics(parent_metrics, "CU")
                dbb_floor = min(float(dbb[name]) for name in ("precision", "recall", "f1"))
                cu_floor = min(float(cu[name]) for name in ("precision", "recall", "f1"))
                matrix = parent_metrics["confusion_matrix_rows_true_columns_pred"]
                rows.append(
                    {
                        "epoch": int(epoch["epoch"]),
                        "dbb_floor": dbb_floor,
                        "cu_floor": cu_floor,
                        "dbb_to_cu": int(matrix[0][1]),
                        "cu_to_dbb": int(matrix[1][0]),
                        "stage_one_feasible": dbb_floor >= 0.95 and cu_floor >= 0.90,
                    }
                )
            eligible = sorted(
                (row for row in rows if row["dbb_floor"] >= 0.95),
                key=lambda row: (row["cu_floor"], row["dbb_floor"]),
                reverse=True,
            )
            feasible = [row for row in rows if row["stage_one_feasible"]]
            total_epoch_count += len(rows)
            total_feasible_count += len(feasible)
            seed_runs[arm] = {
                "epoch_count": len(rows),
                "stage_one_feasible_epoch_count": len(feasible),
                "stage_one_feasible_epochs": [row["epoch"] for row in feasible],
                "dbb_floor_eligible_epoch_count": len(eligible),
                "best_cu_floor_subject_to_dbb_floor": eligible[0] if eligible else None,
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
            }
        runs[str(seed)] = seed_runs
    return {
        "total_arm_epoch_count": total_epoch_count,
        "stage_one_feasible_arm_epoch_count": total_feasible_count,
        "failure_is_not_explained_by_best_checkpoint_selection": total_feasible_count == 0,
        "runs": runs,
    }


def _training_auxiliary_loss_history(root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for seed in SEEDS:
        config = load_task_config(root / _config_path(seed, "treatment"))
        summary_path = root / config.train.formal_artifact_dir / "training-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        history = list(summary.get("history") or [])
        best_epoch = int(summary.get("best_epoch", -1))
        if summary.get("status") != "passed" or len(history) != 20 or not 1 <= best_epoch <= 20:
            raise ValueError(f"V12 auxiliary-loss diagnostic requires a complete run: {summary_path}")
        losses = [float(row["train_dbb_auxiliary_loss"]) for row in history]
        results[str(seed)] = {
            "epoch_count": len(history),
            "epoch_1": losses[0],
            "best_gate_checkpoint_epoch": best_epoch,
            "at_best_gate_checkpoint": losses[best_epoch - 1],
            "epoch_20": losses[-1],
            "minimum": min(losses),
            "minimum_epoch": losses.index(min(losses)) + 1,
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "interpretation_guard": "training auxiliary loss is optimization evidence, not validation accuracy",
        }
    return results


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    paired_path = root / PAIRED_REPORT
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired.get("status") != "completed_but_gate_failed" or paired.get("v12_retention_passed") is not False:
        raise ValueError("V12 auxiliary diagnostic requires the completed failed frozen paired report")
    validation_results: dict[str, Any] = {}
    for seed in SEEDS:
        validation_results[str(seed)] = _evaluate_final_states(root, seed=seed, split_name="val")
    report = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V12-dbb-auxiliary-diagnostic-v1",
        "scope": "CQ08 V3 validation parent-final states plus recorded train losses; Treatment best checkpoints; test sealed",
        "paired_report": str(paired_path),
        "paired_report_sha256": sha256_file(paired_path),
        "v12_retention_passed": False,
        "validation_results": validation_results,
        "training_auxiliary_loss_history": _training_auxiliary_loss_history(root),
        "all_epoch_feasibility": _all_epoch_feasibility(root),
        "conclusion_guard": (
            "This is a post-training mechanism diagnostic. It cannot change the frozen retention decision, "
            "select another checkpoint, fit a rejection threshold, or authorize test/deployment."
        ),
        "test_evaluation": "未执行",
        "OOF": "未执行",
        "threshold_fitting": "未执行",
        "stable_evaluation": "未执行",
        "P95": "未执行",
    }
    path = output_path or (root / OUTPUT_REPORT)
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose completed V12 DBB auxiliary Treatment runs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
