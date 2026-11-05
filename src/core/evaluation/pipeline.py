"""Evaluation helpers for the product cascade."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from core.inference.pipeline import ONNXProductInferencePipeline, ProductInferencePipeline


_ACTION_LABEL_ALIASES = {
    "ClutterTarget": "Clutter",
    "UnknownTarget": "UNKNOWN_REJECTED",
}


def evaluation_action_label(label: Any) -> str:
    """Map dataset labels to the semantic action vocabulary used by L3."""

    value = str(label)
    return _ACTION_LABEL_ALIASES.get(value, value)


def action_matches_label(action: Any, label: Any) -> bool:
    """Compare a model action with a dataset label in L3 semantic space."""

    return str(action) == evaluation_action_label(label)


def evaluate_asd_packages(
    project_root: str | Path,
    paths: Iterable[str | Path],
    *,
    timestamp_unit: str | None = None,
    binding_path: str | Path = "src/config/radar_ln3_l3.json",
    onnx_path: str | Path | None = "artifacts/radar_ln3/onnx/radarln3.onnx",
    prefer_directml: bool = False,
    amplitude_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate labeled `.asd` packages and return auditable metrics."""

    root = Path(project_root).resolve()
    target_onnx = None if onnx_path is None else Path(onnx_path)
    if target_onnx is not None and not target_onnx.is_absolute():
        target_onnx = root / target_onnx
    pipeline = (
        ONNXProductInferencePipeline(
            root,
            target_onnx,
            prefer_directml=prefer_directml,
            binding_path=binding_path,
        )
        if target_onnx is not None and target_onnx.is_file()
        else ProductInferencePipeline(root, binding_path=binding_path)
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        result = pipeline.predict_asd(
            path,
            timestamp_unit=timestamp_unit,
            amplitude_mode=amplitude_mode,
        )
        records.append({"path": str(path), **result})
    action_counts = Counter(str(row["action"]) for row in records)
    labeled = [row for row in records if row.get("label_name")]
    pending = [row for row in labeled if row.get("action") == "PENDING_ABSTAIN"]
    non_pending = [row for row in labeled if row not in pending]
    raw_correct = [
        row
        for row in labeled
        if str(row["action"]) == str(row["label_name"])
    ]
    correct = [
        row for row in labeled
        if action_matches_label(row["action"], row["label_name"])
    ]
    correct_non_pending = [
        row
        for row in non_pending
        if action_matches_label(row["action"], row["label_name"])
    ]
    l2_labels = {"DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2}
    l2_rows = [row for row in labeled if row.get("label_name") in l2_labels]
    l2_correct = [
        row for row in l2_rows
        if int(max(range(3), key=lambda index: row["l2_probabilities"][index]))
        == l2_labels[row["label_name"]]
    ]
    cascade_correct = [
        row for row in l2_rows
        if int(max(range(3), key=lambda index: row["cascade_probabilities"][index]))
        == l2_labels[row["label_name"]]
    ]
    return {
        "count": len(records),
        "timestamp_unit": timestamp_unit,
        "amplitude_mode": amplitude_mode,
        "evaluation_note": (
            "research-only replay; ASD packages may contain raw Stable8/Micro-Doppler evidence, "
            "and the current radar_ln3 calibration is not deployment eligible"
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "labeled_count": len(labeled),
        "labeled_action_accuracy": (len(correct) / len(labeled)) if labeled else None,
        "raw_action_string_accuracy": (
            len(raw_correct) / len(labeled) if labeled else None
        ),
        "pending_count": len(pending),
        "action_coverage": len(non_pending) / len(labeled) if labeled else None,
        "labeled_action_accuracy_excluding_pending": (
            len(correct_non_pending) / len(non_pending) if non_pending else None
        ),
        "l2_closed_set_top1_accuracy": (len(l2_correct) / len(l2_rows)) if l2_rows else None,
        "cascade_known_class_top1_accuracy": (len(cascade_correct) / len(l2_rows)) if l2_rows else None,
        "stable8_available_count": sum(bool(row.get("stable8_available")) for row in records),
        "micro_doppler_available_count": sum(bool(row.get("micro_doppler_available")) for row in records),
        "formal_available": bool(pipeline.binding.get("formal_available", False)),
        "binding": pipeline.binding,
        "records": records,
    }


def write_evaluation_report(path: str | Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "action_matches_label",
    "evaluation_action_label",
    "evaluate_asd_packages",
    "write_evaluation_report",
]
