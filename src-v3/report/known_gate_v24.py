"""Development-only audit report for the completed V24 KnownGate baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from build.known_gate_v24 import (
    ARTIFACT,
    CONFIG,
    EXPERIMENT_CONTRACT,
    READINESS,
    VALIDATION_CLUTTER_PARENT_SUPPORT,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V24_READINESS_SCHEMA,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_features import V24_COURSE_REASON_AWARE_BASELINE_VARIANT
from train.known_gate_v21 import _verify_contract, _verify_readiness
from radar_ln3.known_gate import load_known_gate_config, sha256_file


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-development.json"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 24043
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
EXPECTED_SUPPORT = {
    "DroneTarget": 113,
    "BirdTarget": 56,
    "BalloonTarget": 54,
    "ClutterTarget": 9,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _records(detail: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in ("parent", "eligible_prefix", "window")
    }
    if (
        len(rows["parent"]) != 232
        or len(rows["eligible_prefix"]) != 3756
        or len(rows["window"]) != 4452
    ):
        raise ValueError("V24 detail record counts differ from the frozen V21 scope")
    parent_ids = [str(row.get("parent_sample_id", "")) for row in rows["parent"]]
    if len(parent_ids) != len(set(parent_ids)) or not all(parent_ids):
        raise ValueError("V24 parent detail records are incomplete or duplicated")
    support = {
        label: sum(str(row.get("label_name")) == label for row in rows["parent"])
        for label in LABELS
    }
    if support != EXPECTED_SUPPORT:
        raise ValueError(f"V24 parent label support changed: {support}")
    return rows


def _metrics(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "parent_final": summarize_known_gate_records(
            rows["parent"], scope="parent_final"
        ),
        "eligible_prefix": summarize_known_gate_records(
            rows["eligible_prefix"], scope="eligible_causal_prefix"
        ),
        "parent_balanced_eligible_prefix": (
            summarize_parent_balanced_eligible_prefix_records(
                rows["eligible_prefix"],
                scope="parent_balanced_eligible_causal_prefix",
            )
        ),
        "window": summarize_known_gate_records(rows["window"], scope="causal_windows"),
    }


def _wilson(
    successes: int,
    support: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float | int | None | str]:
    """Return a Wilson interval without hiding a zero predicted denominator.

    Recall always has the nine true Clutter parents as its denominator.  In a
    failed model, however, precision can have zero predicted Clutter parents.
    That is a meaningful result (precision is undefined), not a reason to
    abort the entire development report.
    """

    if support < 0 or successes < 0 or successes > support:
        raise ValueError("Wilson interval count is invalid")
    if support == 0:
        return {
            "support": 0,
            "successes": 0,
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "status": "undefined_zero_denominator",
        }
    p = float(successes / support)
    denominator = 1.0 + z * z / support
    center = (p + z * z / (2.0 * support)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / support + z * z / (4.0 * support * support)) / denominator
    return {
        "support": int(support),
        "successes": int(successes),
        "estimate": p,
        "lower_95": float(max(0.0, center - half)),
        "upper_95": float(min(1.0, center + half)),
        "status": "defined",
    }


def _clutter_wilson(parent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in parent_rows:
        matrix[int(row["true_id"]), int(row["pred_id"])] += 1
    tp = int(matrix[1, 1])
    predicted_clutter = int(matrix[:, 1].sum())
    true_clutter = int(matrix[1].sum())
    return {
        "method": "Wilson score interval, two-sided 95 percent",
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
        "clutter_precision": _wilson(tp, predicted_clutter),
        "clutter_recall": _wilson(tp, true_clutter),
        "warning": "Only nine validation Clutter parents; point estimates and intervals are development-only diagnostics.",
    }


def _metric_vector(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics = summarize_known_gate_records(records, scope="v24_parent_bootstrap")
    per_class = {str(row["class"]): row for row in metrics["per_class"]}
    target = per_class["KNOWN_TARGET_EVIDENCE"]
    clutter = per_class["KNOWN_CLUTTER_EVIDENCE"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "known_target_recall": float(target["recall"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_f1": float(clutter["f1"]),
    }


def _ci(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _bootstrap(parent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [dict(row) for row in sorted(parent_rows, key=lambda row: str(row["parent_sample_id"]))]
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(ordered) if str(row["label_name"]) == label],
            dtype=np.int64,
        )
        for label in LABELS
    }
    if {label: len(indices) for label, indices in strata.items()} != EXPECTED_SUPPORT:
        raise ValueError("V24 bootstrap strata differ from frozen support")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_indices = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        vector = _metric_vector([ordered[int(index)] for index in sampled_indices])
        for name, value in vector.items():
            values.setdefault(name, []).append(value)
    return {
        "method": "parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": EXPECTED_SUPPORT,
        "metrics": {name: _ci(values_) for name, values_ in values.items()},
    }


def _confusions(parent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known_to_clutter = []
    clutter_to_known = []
    correct_clutter = []
    for row in parent_rows:
        item = {
            "parent_sample_id": str(row["parent_sample_id"]),
            "track_id": str(row["track_id"]),
            "source_name": str(row["source_name"]),
            "label_name": str(row["label_name"]),
            "true_id": int(row["true_id"]),
            "pred_id": int(row["pred_id"]),
            "known_target_probability": float(row["known_target_probability"]),
            "known_clutter_probability": float(row["known_clutter_probability"]),
            "event_index": int(row["event_index"]),
            "parent_length": int(row["parent_length"]),
        }
        if item["true_id"] == 0 and item["pred_id"] == 1:
            known_to_clutter.append(item)
        elif item["true_id"] == 1 and item["pred_id"] == 0:
            clutter_to_known.append(item)
        elif item["true_id"] == 1 and item["pred_id"] == 1:
            correct_clutter.append(item)
    return {
        "KnownTarget_to_Clutter": known_to_clutter,
        "Clutter_to_KnownTarget": clutter_to_known,
        "Clutter_correct": correct_clutter,
        "counts": {
            "KnownTarget_to_Clutter": len(known_to_clutter),
            "Clutter_to_KnownTarget": len(clutter_to_known),
            "Clutter_correct": len(correct_clutter),
        },
    }


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    artifact = root / ARTIFACT
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    if not all(
        path.is_file()
        for path in (
            config_path,
            readiness_path,
            contract_path,
            summary_path,
            detail_path,
            checkpoint_path,
        )
    ):
        raise FileNotFoundError("V24 training output is incomplete")
    config = load_known_gate_config(config_path)
    contract_evidence = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract_evidence)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (
        config.variant != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or summary.get("status") != "completed"
        or summary.get("variant") != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or int(summary.get("seed", -1)) != 43
        or len(list(summary.get("history") or ())) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or Path(str(summary.get("config_path", ""))).resolve() != config_path.resolve()
        or Path(str(summary.get("best_checkpoint", ""))).resolve() != checkpoint_path.resolve()
        or Path(str(summary.get("sample_detail_information", ""))).resolve()
        != detail_path.resolve()
        or dict(summary.get("training_readiness") or {}).get("sha256")
        != readiness["sha256"]
        or dict(summary.get("paired_initialization") or {}).get(
            "sealed_v24_initial_model_state_reproduced"
        )
        is not True
    ):
        raise ValueError("V24 training summary violates the sealed baseline contract")
    rows = _records(detail)
    metrics = _metrics(rows)
    recorded = dict(dict(summary.get("best_validation_metrics") or {}).get("parent_metrics") or {})
    recomputed = metrics["parent_final"]
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(recorded.get(key, np.nan)) - float(recomputed[key])) > 1.0e-12:
            raise ValueError(f"V24 parent-final summary metric differs at {key}")
    report = {
        "status": "completed_development_baseline_only",
        "schema_version": "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-development-v1",
        "scope": "V24 development KnownGate L1 baseline only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "training_summary_path": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": readiness["sha256"],
        "experiment_contract_path": str(contract_path),
        "experiment_contract_sha256": sha256_file(contract_path),
        "best_checkpoint_path": str(checkpoint_path),
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": int(summary["best_epoch"]),
        "initialization": {
            "fresh_training_not_finetune": True,
            "initial_model_state_sha256": str(summary["initial_model_state_sha256"]),
            "paired_initialization": dict(summary["paired_initialization"]),
        },
        "metrics": metrics,
        "clutter_parent_wilson_intervals": _clutter_wilson(rows["parent"]),
        "parent_stratified_bootstrap": _bootstrap(rows["parent"]),
        "parent_confusion_audit": _confusions(rows["parent"]),
        "clutter_support_warning": {
            "validation_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
            "no_window_duplication_is_used_as_independent_support": True,
            "development_only": True,
        },
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    output = root / OUTPUT
    _write_json(output, report)
    return {**report, "report_path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the completed V24 KnownGate development baseline."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
