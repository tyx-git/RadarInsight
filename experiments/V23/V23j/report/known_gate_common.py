from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate import summarize_known_gate_records


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 21543
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")






def _record_key(scope: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    parent_id = str(row["parent_sample_id"])
    return (parent_id,) if scope == "parent" else (parent_id, int(row["event_index"]))


def _ordered_pair(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before = {_record_key(scope, row): dict(row) for row in baseline}
    after = {_record_key(scope, row): dict(row) for row in treatment}
    if len(before) != len(baseline) or len(after) != len(treatment) or set(before) != set(after):
        raise ValueError(f"V21e paired {scope} sets differ or repeat keys")
    ordered = sorted(before)
    for key in ordered:
        if (
            before[key]["label_name"] != after[key]["label_name"]
            or int(before[key]["true_id"]) != int(after[key]["true_id"])
        ):
            raise ValueError(f"V21e paired {scope} labels differ")
    return [before[key] for key in ordered], [after[key] for key in ordered]


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = {str(row["class"]): row for row in metrics["per_class"]}
    target = rows["KNOWN_TARGET_EVIDENCE"]
    clutter = rows["KNOWN_CLUTTER_EVIDENCE"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "known_target_precision": float(target["precision"]),
        "known_target_recall": float(target["recall"]),
        "known_target_f1": float(target["f1"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_f1": float(clutter["f1"]),
        "weak_class_precision_recall_f1_floor": min(
            float(row[field])
            for row in metrics["per_class"]
            for field in ("precision", "recall", "f1")
        ),
    }


def _base_label_recall_vector(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for label in LABELS:
        selected = [row for row in records if str(row["label_name"]) == label]
        if not selected:
            raise ValueError(f"V21e bootstrap has no {label} parent")
        result[f"{label}_acceptance_recall"] = float(
            np.mean([int(row["pred_id"]) == int(row["true_id"]) for row in selected])
        )
    result["four_label_macro_acceptance_recall"] = float(
        np.mean([result[f"{label}_acceptance_recall"] for label in LABELS])
    )
    return result


def _ci(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _bootstrap(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment, scope="parent")
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(before) if row["label_name"] == label],
            dtype=np.int64,
        )
        for label in LABELS
    }
    expected = {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
    }
    if {label: len(indices) for label, indices in strata.items()} != expected:
        raise ValueError("V21e bootstrap support changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    before_values: dict[str, list[float]] = {}
    after_values: dict[str, list[float]] = {}
    delta_values: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        before_sample = [before[int(index)] for index in sampled]
        after_sample = [after[int(index)] for index in sampled]
        before_vector = _metric_vector(
            summarize_known_gate_records(before_sample, scope="v21e_parent_bootstrap")
        )
        after_vector = _metric_vector(
            summarize_known_gate_records(after_sample, scope="v21e_parent_bootstrap")
        )
        before_vector.update(_base_label_recall_vector(before_sample))
        after_vector.update(_base_label_recall_vector(after_sample))
        for key in before_vector:
            before_values.setdefault(key, []).append(before_vector[key])
            after_values.setdefault(key, []).append(after_vector[key])
            delta_values.setdefault(key, []).append(after_vector[key] - before_vector[key])
    return {
        "method": "paired parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": expected,
        "baseline": {key: _ci(values) for key, values in before_values.items()},
        "treatment": {key: _ci(values) for key, values in after_values.items()},
        "treatment_minus_baseline": {
            key: _ci(values) for key, values in delta_values.items()
        },
    }


















