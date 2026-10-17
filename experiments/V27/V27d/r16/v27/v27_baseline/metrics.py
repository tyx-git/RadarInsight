"""Parent-level metrics used by the V27 DBB baseline.

The functions in this module deliberately distinguish one final prediction per
parent from event/prefix observations.  A long track therefore cannot acquire
extra weight in the primary metric merely because it contains more events.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _as_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("V27 metrics require at least two unique class names")
    return names


def _matrix(
    records: Iterable[Mapping[str, Any]], class_names: Sequence[str]
) -> np.ndarray:
    names = _as_class_names(class_names)
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for record in records:
        true_id = int(record["true_id"])
        pred_id = int(record["pred_id"])
        if not 0 <= true_id < len(names) or not 0 <= pred_id < len(names):
            raise ValueError(
                f"V27 metric record has an out-of-range class id: true={true_id}, pred={pred_id}"
            )
        matrix[true_id, pred_id] += 1
    return matrix


def metrics_from_matrix(
    matrix: np.ndarray,
    *,
    class_names: Sequence[str],
    scope: str,
) -> dict[str, Any]:
    names = _as_class_names(class_names)
    value = np.asarray(matrix)
    if value.shape != (len(names), len(names)) or value.dtype.kind not in "iu":
        raise ValueError("V27 confusion matrix has an invalid shape or dtype")
    if bool((value < 0).any()):
        raise ValueError("V27 confusion matrix cannot contain negative counts")
    support = value.sum(axis=1)
    predicted = value.sum(axis=0)
    per_class: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(names):
        true_positive = int(value[class_id, class_id])
        precision = (
            true_positive / int(predicted[class_id])
            if int(predicted[class_id])
            else 0.0
        )
        recall = (
            true_positive / int(support[class_id])
            if int(support[class_id])
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        off_diagonal = value[class_id].copy()
        off_diagonal[class_id] = 0
        if int(off_diagonal.sum()) > 0:
            destination = int(np.argmax(off_diagonal))
            destination_name: str | None = names[destination]
            destination_count = int(off_diagonal[destination])
        else:
            destination_name = None
            destination_count = 0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_values.append(float(f1))
        per_class.append(
            {
                "class": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support[class_id]),
                "major_confusion_to": destination_name,
                "major_confusion_count": destination_count,
            }
        )
    total = int(value.sum())
    return {
        "scope": str(scope),
        "support": total,
        "accuracy": float(np.trace(value) / total) if total else None,
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": value.astype(np.int64).tolist(),
    }


def summarize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
    scope: str,
) -> dict[str, Any]:
    return metrics_from_matrix(_matrix(records, class_names), class_names=class_names, scope=scope)


def parent_balanced_prefix_records(
    event_records: Iterable[Mapping[str, Any]],
    *,
    minimum_events: int,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Make one equal-weight record per parent from eligible prefix evidence.

    The prefix protocol is frozen for V27 as follows: events at positions
    ``minimum_events .. parent_length-1`` (one-based), excluding the final
    event, are eligible; their class probabilities are averaged per parent and
    the averaged vector is converted to one prediction.  This is a diagnostic
    metric only and is never used for training or checkpoint selection.
    """

    if int(minimum_events) < 1:
        raise ValueError("V27 prefix minimum_events must be positive")
    names = _as_class_names(class_names)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in event_records:
        parent_id = str(record.get("parent_sample_id", ""))
        if not parent_id:
            raise ValueError("V27 prefix record has no parent_sample_id")
        grouped[parent_id].append(record)
    output: list[dict[str, Any]] = []
    for parent_id in sorted(grouped):
        rows = sorted(grouped[parent_id], key=lambda row: int(row["event_index"]))
        if not rows:
            continue
        parent_length = int(rows[0].get("parent_length", len(rows)))
        eligible = [
            row
            for row in rows
            if int(row["window_length"]) >= int(minimum_events)
            and int(row["window_length"]) < parent_length
        ]
        if not eligible:
            continue
        probabilities: list[np.ndarray] = []
        for row in eligible:
            raw = row.get("probabilities")
            if raw is None:
                logits = np.asarray(row.get("logits"), dtype=np.float64)
                if logits.shape != (len(names),) or not np.isfinite(logits).all():
                    raise ValueError("V27 prefix record has no finite class probability/logit vector")
                shifted = logits - float(np.max(logits))
                exp = np.exp(shifted)
                probability = exp / float(exp.sum())
            else:
                probability = np.asarray(raw, dtype=np.float64)
                if probability.shape != (len(names),) or not np.isfinite(probability).all():
                    raise ValueError("V27 prefix probability vector has an invalid shape/value")
                if np.any(probability < 0.0) or not math.isclose(
                    float(probability.sum()), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6
                ):
                    raise ValueError("V27 prefix probabilities must be non-negative and sum to one")
            probabilities.append(probability)
        mean_probability = np.mean(np.stack(probabilities, axis=0), axis=0)
        true_id = int(eligible[0]["true_id"])
        output.append(
            {
                "parent_sample_id": parent_id,
                "track_id": eligible[0].get("track_id"),
                "source_name": eligible[0].get("source_name"),
                "true_id": true_id,
                "pred_id": int(np.argmax(mean_probability)),
                "probabilities": [float(value) for value in mean_probability],
                "eligible_prefix_count": len(eligible),
                "parent_length": parent_length,
                "label_name": eligible[0].get("label_name"),
            }
        )
    return output










__all__ = ["metrics_from_matrix", "parent_balanced_prefix_records", "summarize_records"]