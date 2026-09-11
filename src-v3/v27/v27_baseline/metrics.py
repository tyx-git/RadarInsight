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


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"):
        raw = metrics.get(name)
        values[name] = float(raw) if raw is not None and np.isfinite(float(raw)) else float("nan")
    return values


def parent_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
    seed: int,
    replicates: int = 2000,
) -> dict[str, Any]:
    """Return stratified parent bootstrap intervals for validation metrics."""

    names = _as_class_names(class_names)
    rows = list(records)
    if not rows:
        raise ValueError("V27 bootstrap requires at least one parent record")
    strata: dict[int, list[int]] = {class_id: [] for class_id in range(len(names))}
    for index, row in enumerate(rows):
        true_id = int(row["true_id"])
        if true_id not in strata:
            raise ValueError("V27 bootstrap record contains an invalid true class")
        strata[true_id].append(index)
    if any(not values for values in strata.values()):
        raise ValueError("V27 stratified bootstrap requires positive support for every class")
    count = int(replicates)
    if count < 100:
        raise ValueError("V27 bootstrap replicates must be at least 100")
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {
        name: []
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    }
    for _ in range(count):
        selected: list[Mapping[str, Any]] = []
        for indices in strata.values():
            sampled = rng.choice(np.asarray(indices, dtype=np.int64), size=len(indices), replace=True)
            selected.extend(rows[int(index)] for index in sampled.tolist())
        vector = _metric_vector(summarize_records(selected, class_names=names, scope="bootstrap"))
        for name in samples:
            samples[name].append(vector[name])
    intervals: dict[str, Any] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[name] = {
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
    return {
        "method": "stratified_parent_bootstrap",
        "seed": int(seed),
        "replicates": count,
        "stratum_support": {
            names[class_id]: len(indices) for class_id, indices in strata.items()
        },
        "metrics": intervals,
    }


def distribution_summary(values: Iterable[float], *, name: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"name": str(name), "count": 0, "status": "no_valid_values"}
    if not np.isfinite(array).all():
        raise ValueError(f"V27 telemetry distribution {name} contains non-finite values")
    return {
        "name": str(name),
        "count": int(array.size),
        "status": "ok",
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def summarize_parent_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    class_names: Sequence[str],
    scope_prefix: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if group_field not in record:
            raise ValueError(f"V27 record lacks stratification field {group_field}")
        grouped[str(record[group_field])].append(record)
    return {
        group: {
            "support": len(rows),
            "metrics": summarize_records(
                rows,
                class_names=class_names,
                scope=f"{scope_prefix}:{group}",
            ),
        }
        for group, rows in sorted(grouped.items())
    }


__all__ = [
    "distribution_summary",
    "metrics_from_matrix",
    "parent_balanced_prefix_records",
    "parent_bootstrap",
    "summarize_parent_groups",
    "summarize_records",
]
