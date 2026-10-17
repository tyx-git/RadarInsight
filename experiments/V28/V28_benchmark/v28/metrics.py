"""Parent-level metrics for V28 classification and diagnostics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def confusion_metrics(records: Iterable[Mapping[str, Any]], *, class_names: Sequence[str], scope: str) -> dict[str, Any]:
    names = tuple(str(value) for value in class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("V28 metric class names are invalid")
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for record in records:
        true_id, pred_id = int(record["true_id"]), int(record["pred_id"])
        if not 0 <= true_id < len(names) or not 0 <= pred_id < len(names):
            raise ValueError("V28 metric class ID is out of range")
        matrix[true_id, pred_id] += 1
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    for index, name in enumerate(names):
        tp = int(matrix[index, index])
        precision = tp / int(predicted[index]) if predicted[index] else 0.0
        recall = tp / int(support[index]) if support[index] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        off = matrix[index].copy()
        off[index] = 0
        confusion_to = names[int(off.argmax())] if off.sum() else None
        confusion_count = int(off.max()) if off.sum() else 0
        rows.append({"class": name, "precision": float(precision), "recall": float(recall), "f1": float(f1), "support": int(support[index]), "major_confusion_to": confusion_to, "major_confusion_count": confusion_count})
        precision_values.append(float(precision)); recall_values.append(float(recall)); f1_values.append(float(f1))
    total = int(matrix.sum())
    return {"scope": str(scope), "support": total, "accuracy": float(np.trace(matrix) / total) if total else None, "macro_precision": float(np.mean(precision_values)) if precision_values else None, "macro_recall": float(np.mean(recall_values)) if recall_values else None, "macro_f1": float(np.mean(f1_values)) if f1_values else None, "balanced_accuracy": float(np.mean(recall_values)) if recall_values else None, "per_class": rows, "confusion_matrix_rows_true_columns_pred": matrix.tolist()}




__all__ = ["confusion_metrics"]