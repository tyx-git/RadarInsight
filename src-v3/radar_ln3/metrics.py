from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from . import LABELS


def summarize_records(records: Iterable[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    rows = list(records)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for row in rows:
        true_id = int(row["true_id"])
        pred_id = int(row["pred_id"])
        if not 0 <= true_id < len(LABELS) or not 0 <= pred_id < len(LABELS):
            raise ValueError("metric record contains an invalid class id")
        matrix[true_id, pred_id] += 1
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    per_class: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_id, label in enumerate(LABELS):
        tp = int(matrix[class_id, class_id])
        precision = (tp / int(predicted[class_id])) if predicted[class_id] else 0.0
        recall = (tp / int(support[class_id])) if support[class_id] else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0.0 else 0.0
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        confusion = matrix[class_id].copy()
        confusion[class_id] = 0
        destination = int(confusion.argmax()) if int(confusion.sum()) else None
        per_class.append({
            "class": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[class_id]),
            "major_confusion_to": LABELS[destination] if destination is not None and confusion[destination] > 0 else None,
            "major_confusion_count": int(confusion[destination]) if destination is not None else 0,
            "reaches_90_percent_all_metrics": bool(support[class_id] > 0 and min(precision, recall, f1) >= 0.9),
        })
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else None
    macro_precision = float(np.mean(precision_values)) if precision_values else None
    macro_recall = float(np.mean(recall_values)) if recall_values else None
    macro_f1 = float(np.mean(f1_values)) if f1_values else None
    drone = LABELS.index("DroneTarget")
    bird = LABELS.index("BirdTarget")
    return {
        "scope": scope,
        "support": total,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_accuracy": macro_recall,
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
        "drone_bird_confusion": {
            "DroneTarget_to_BirdTarget": int(matrix[drone, bird]),
            "BirdTarget_to_DroneTarget": int(matrix[bird, drone]),
        },
        "target_90_percent_status": "met" if total > 0 and all(item["reaches_90_percent_all_metrics"] for item in per_class) else "not_met_or_unavailable",
    }


def grouped_summaries(records: Iterable[dict[str, Any]], *, field: str, scope_prefix: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        value = row.get(field)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(row)
    return {key: summarize_records(value, scope=f"{scope_prefix}:{key}") for key, value in sorted(groups.items())}


def confidence_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["confidence"]) for row in records if row.get("confidence") is not None]
    if not values:
        return {"support": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"support": len(values), "min": float(array.min()), "median": float(np.median(array)), "max": float(array.max())}


__all__ = ["confidence_summary", "grouped_summaries", "summarize_records"]
