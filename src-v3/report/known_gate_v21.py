from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_LABELS,
    summarize_known_gate_records,
)
from radar_ln3.known_gate_features import (
    BASELINE_VARIANT,
    FEATURE_NAMES_BY_VARIANT,
    V21A_VARIANT,
    V21B_VARIANT,
    V21C_VARIANT,
    V21D_VARIANT,
)


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 21042
PRIMARY_DELTA = 0.02
VARIANTS = (V21A_VARIANT, V21B_VARIANT, V21C_VARIANT, V21D_VARIANT)
DETAILS: Mapping[str, dict[str, Any]] = {
    BASELINE_VARIANT: {"version": "V21", "slug": "baseline", "expected_labels": []},
    V21A_VARIANT: {
        "version": "V21a",
        "slug": "spectral-temporal-consistency",
        "expected_labels": ["BirdTarget", "BalloonTarget", "ClutterTarget"],
    },
    V21B_VARIANT: {
        "version": "V21b",
        "slug": "modal-consistency",
        "expected_labels": ["DroneTarget", "BalloonTarget", "ClutterTarget"],
    },
    V21C_VARIANT: {
        "version": "V21c",
        "slug": "target-row-iso",
        "expected_labels": ["DroneTarget", "ClutterTarget"],
    },
    V21D_VARIANT: {
        "version": "V21d",
        "slug": "d-range-stability",
        "expected_labels": ["BirdTarget", "ClutterTarget"],
    },
}


def _artifact_relative(variant: str) -> Path:
    detail = DETAILS[variant]
    return Path(
        f"artifacts/RADAR-LN3-{detail['version']}-cq08-liquid-known-gate-"
        f"{detail['slug']}-seed42-formal"
    )


def _load_run(root: Path, variant: str) -> dict[str, Any]:
    artifact = root / _artifact_relative(variant)
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    if not summary_path.is_file() or not detail_path.is_file():
        raise FileNotFoundError(f"V21 {variant} formal summary/detail is missing: {artifact}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    records = list(dict(detail.get("records") or {}).get("parent") or [])
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "known_gate"
        or summary.get("variant") != variant
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("unknown_is_model_class") is not False
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or len(records) != 232
    ):
        raise ValueError(f"V21 {variant} summary violates the frozen local full-training contract")
    if detail.get("variant") != variant or int(detail.get("epoch", -1)) != int(summary["best_epoch"]):
        raise ValueError(f"V21 {variant} sample detail is not bound to its best epoch")
    parent_ids = [str(row["parent_sample_id"]) for row in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError(f"V21 {variant} sample detail repeats a parent")
    metrics = summarize_known_gate_records(records, scope="parent_final_recomputed")
    recorded = dict(dict(summary["best_validation_metrics"])["parent_metrics"])
    for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(metrics[field]) - float(recorded[field])) > 1.0e-12:
            raise ValueError(f"V21 {variant} recomputed metric differs at {field}")
    return {
        "variant": variant,
        "artifact": str(artifact),
        "summary_path": str(summary_path),
        "detail_path": str(detail_path),
        "summary": summary,
        "records": records,
        "metrics": metrics,
    }


def _metrics_from_indices(records: Sequence[Mapping[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    return summarize_known_gate_records(
        [records[int(index)] for index in indices.tolist()],
        scope="parent_bootstrap",
    )


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = {str(row["class"]): row for row in metrics["per_class"]}
    result = {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    for class_name in KNOWN_GATE_CLASS_NAMES:
        row = rows[class_name]
        short = "known_target" if class_name == KNOWN_GATE_CLASS_NAMES[0] else "known_clutter"
        for field in ("precision", "recall", "f1"):
            result[f"{short}_{field}"] = float(row[field])
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
    treatment: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    baseline_by_parent = {str(row["parent_sample_id"]): dict(row) for row in baseline}
    if treatment is not None:
        treatment_by_parent = {str(row["parent_sample_id"]): dict(row) for row in treatment}
        if set(treatment_by_parent) != set(baseline_by_parent):
            raise ValueError("V21 paired bootstrap parent sets differ")
        for parent_id in baseline_by_parent:
            before, after = baseline_by_parent[parent_id], treatment_by_parent[parent_id]
            if before["label_name"] != after["label_name"] or int(before["true_id"]) != int(after["true_id"]):
                raise ValueError("V21 paired bootstrap label/true-id differs")
    ordered_ids = sorted(baseline_by_parent)
    baseline_rows = [baseline_by_parent[parent_id] for parent_id in ordered_ids]
    treatment_rows = None if treatment is None else [treatment_by_parent[parent_id] for parent_id in ordered_ids]
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(baseline_rows) if row["label_name"] == label],
            dtype=np.int64,
        )
        for label in KNOWN_LABELS
    }
    if {label: len(indices) for label, indices in strata.items()} != {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
    }:
        raise ValueError("V21 bootstrap base-label support changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    baseline_values: dict[str, list[float]] = {}
    treatment_values: dict[str, list[float]] = {}
    delta_values: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        before_vector = _metric_vector(_metrics_from_indices(baseline_rows, sampled))
        for key, value in before_vector.items():
            baseline_values.setdefault(key, []).append(value)
        if treatment_rows is not None:
            after_vector = _metric_vector(_metrics_from_indices(treatment_rows, sampled))
            for key, value in after_vector.items():
                treatment_values.setdefault(key, []).append(value)
                delta_values.setdefault(key, []).append(value - before_vector[key])
    result = {
        "method": "parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": {label: len(indices) for label, indices in strata.items()},
        "baseline": {key: _ci(values) for key, values in baseline_values.items()},
    }
    if treatment_rows is not None:
        result["treatment"] = {key: _ci(values) for key, values in treatment_values.items()}
        result["treatment_minus_baseline"] = {key: _ci(values) for key, values in delta_values.items()}
    return result


def _weak_floor(metrics: Mapping[str, Any]) -> float:
    return min(
        float(row[field])
        for row in metrics["per_class"]
        for field in ("precision", "recall", "f1")
    )


def _base_label_outcomes(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before = {str(row["parent_sample_id"]): row for row in baseline}
    after = {str(row["parent_sample_id"]): row for row in treatment}
    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        selected = [parent_id for parent_id, row in before.items() if row["label_name"] == label]
        rescued = harmed = changed = 0
        for parent_id in selected:
            before_correct = int(before[parent_id]["pred_id"]) == int(before[parent_id]["true_id"])
            after_correct = int(after[parent_id]["pred_id"]) == int(after[parent_id]["true_id"])
            changed += int(int(before[parent_id]["pred_id"]) != int(after[parent_id]["pred_id"]))
            rescued += int((not before_correct) and after_correct)
            harmed += int(before_correct and (not after_correct))
        baseline_correct = sum(
            int(before[parent_id]["pred_id"]) == KNOWN_GATE_LABEL_TO_ID[label]
            for parent_id in selected
        )
        treatment_correct = sum(
            int(after[parent_id]["pred_id"]) == KNOWN_GATE_LABEL_TO_ID[label]
            for parent_id in selected
        )
        result[label] = {
            "support": len(selected),
            "baseline_correct": int(baseline_correct),
            "treatment_correct": int(treatment_correct),
            "correct_delta": int(treatment_correct - baseline_correct),
            "changed": int(changed),
            "rescued": int(rescued),
            "harmed": int(harmed),
            "net_rescue": int(rescued - harmed),
        }
    return result


def _comparison(baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    before_metrics = baseline["metrics"]
    after_metrics = treatment["metrics"]
    bootstrap = _bootstrap(baseline["records"], treatment["records"])
    outcomes = _base_label_outcomes(baseline["records"], treatment["records"])
    variant = str(treatment["variant"])
    expected_labels = list(DETAILS[variant]["expected_labels"])
    expected_net = sum(int(outcomes[label]["correct_delta"]) for label in expected_labels)
    macro_delta = float(after_metrics["macro_f1"] - before_metrics["macro_f1"])
    accuracy_delta = float(after_metrics["accuracy"] - before_metrics["accuracy"])
    weak_delta = float(_weak_floor(after_metrics) - _weak_floor(before_metrics))
    delta_ci = bootstrap["treatment_minus_baseline"]["macro_f1"]
    primary = macro_delta > PRIMARY_DELTA
    robust = float(delta_ci["lower_95"]) > 0.0
    expected_direction = expected_net > 0
    return {
        "variant": variant,
        "feature_names": list(FEATURE_NAMES_BY_VARIANT[variant]),
        "expected_direction_labels": expected_labels,
        "baseline_metrics": before_metrics,
        "treatment_metrics": after_metrics,
        "treatment_minus_baseline": {
            "accuracy": accuracy_delta,
            "macro_precision": float(after_metrics["macro_precision"] - before_metrics["macro_precision"]),
            "macro_recall": float(after_metrics["macro_recall"] - before_metrics["macro_recall"]),
            "macro_f1": macro_delta,
            "weak_class_precision_recall_f1_floor": weak_delta,
        },
        "base_label_paired_outcomes": outcomes,
        "expected_direction_correct_delta_sum": expected_net,
        "bootstrap": bootstrap,
        "frozen_decision": {
            "primary_macro_f1_delta_strictly_greater_than_0_02": primary,
            "paired_bootstrap_macro_f1_lower_95_greater_than_0": robust,
            "expected_direction_labels_net_correct_positive": expected_direction,
            "positive_signal": primary and expected_direction,
            "robust_positive_signal": primary and expected_direction and robust,
            "advance_only_to_multiseed_or_oof_confirmation": primary and expected_direction,
            "deployment_claim_allowed": False,
        },
        "clutter_support_warning": {
            "validation_parent_support": 9,
            "bootstrap_required_and_executed": True,
            "single_point_metric_is_not_sufficient_for_release": True,
        },
    }


def build_report(root: Path, variants: Sequence[str]) -> dict[str, Any]:
    baseline = _load_run(root, BASELINE_VARIANT)
    comparisons = {
        variant: _comparison(baseline, _load_run(root, variant))
        for variant in variants
    }
    accepted = [
        variant
        for variant, comparison in comparisons.items()
        if comparison["frozen_decision"]["positive_signal"]
    ]
    return {
        "status": "completed",
        "schema_version": "RADAR-LN3-V21-known-gate-feature-validation-v1",
        "scope": "development validation parent-final KnownGate local task only",
        "baseline": {
            "variant": BASELINE_VARIANT,
            "metrics": baseline["metrics"],
            "bootstrap": _bootstrap(baseline["records"], None),
            "summary_path": baseline["summary_path"],
            "validation_clutter_support": 9,
        },
        "comparisons": comparisons,
        "positive_signal_variants": accepted,
        "complementarity_not_tested": True,
        "combining_features_is_forbidden_in_this_single_factor_stage": True,
        "interpretation": (
            "A positive local signal may advance only to multi-seed and train parent-grouped OOF confirmation. "
            "It is not an Unknown-rejection, full-cascade, test or deployment result."
        ),
        "unknown_is_model_class": False,
        "local_argmax_is_product_action": False,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }


def _output_path(variants: Sequence[str]) -> Path:
    if tuple(variants) == VARIANTS:
        return Path("src-v3/processed/report/RADAR-LN3-V21-known-gate-feature-validation.json")
    if len(variants) == 1:
        detail = DETAILS[variants[0]]
        return Path(
            f"src-v3/processed/report/RADAR-LN3-{detail['version']}-known-gate-paired-validation.json"
        )
    return Path("src-v3/processed/report/RADAR-LN3-V21-known-gate-selected-feature-validation.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V21 KnownGate single-factor full-training experiments with parent bootstrap."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(VARIANTS) + ["all"],
        default=None,
    )
    args = parser.parse_args()
    requested = list(args.variant or ["all"])
    variants = list(VARIANTS) if "all" in requested else requested
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("V21 report variants must be non-empty and unique")
    root = args.project_root.resolve()
    report = build_report(root, variants)
    output = root / _output_path(variants)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_path": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
