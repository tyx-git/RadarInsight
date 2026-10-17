from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from build.known_gate_v21_multiseed import (
    BASE_READINESS,
    MULTISEED_READINESS,
    artifact_relative,
    config_relative,
    selection_protocol,
)
from build.micro_doppler_v17 import _write_json_new_or_identical
from radar_ln3.known_gate import (
    KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA,
    KNOWN_GATE_BASELINE_MULTISEED_SEEDS,
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_LABELS,
    sha256_file,
    summarize_known_gate_records,
)


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 214244
REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21-known-gate-baseline-multiseed-validation.json"
)
SELECTION_RELATIVE = Path(
    "data/processed/RADAR-LN3-V21-known-gate-baseline-development-selection.json"
)


def _class_rows(metrics: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {str(row["class"]): row for row in metrics["per_class"]}
    if set(rows) != set(KNOWN_GATE_CLASS_NAMES):
        raise ValueError("V21 multiseed metrics do not contain both KnownGate classes")
    return rows


def _selection_key(metrics: Mapping[str, Any], seed: int) -> tuple[float, ...]:
    rows = _class_rows(metrics)
    weak_precision_recall = min(
        float(row[field])
        for row in rows.values()
        for field in ("precision", "recall")
    )
    clutter = rows[KNOWN_GATE_CLASS_NAMES[1]]
    target = rows[KNOWN_GATE_CLASS_NAMES[0]]
    return (
        weak_precision_recall,
        float(metrics["macro_f1"]),
        float(clutter["f1"]),
        float(target["recall"]),
        -float(seed),
    )


def _selection_record(metrics: Mapping[str, Any], seed: int) -> dict[str, Any]:
    key = _selection_key(metrics, seed)
    return {
        "seed": int(seed),
        "parent_min_class_precision_recall": key[0],
        "parent_macro_f1": key[1],
        "known_clutter_f1": key[2],
        "known_target_recall": key[3],
        "ranking_key": list(key),
    }


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = _class_rows(metrics)
    target = rows[KNOWN_GATE_CLASS_NAMES[0]]
    clutter = rows[KNOWN_GATE_CLASS_NAMES[1]]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "parent_min_class_precision_recall": min(
            float(row[field])
            for row in rows.values()
            for field in ("precision", "recall")
        ),
        "known_target_precision": float(target["precision"]),
        "known_target_recall": float(target["recall"]),
        "known_target_f1": float(target["f1"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_f1": float(clutter["f1"]),
    }


def _ci(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _load_run(root: Path, seed: int, readiness_sha256: str) -> dict[str, Any]:
    artifact = root / artifact_relative(seed)
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    config_path = root / config_relative(seed)
    if not all(path.is_file() for path in (summary_path, detail_path, checkpoint_path, config_path)):
        raise FileNotFoundError(f"V21 Baseline seed{seed} formal artifact is incomplete: {artifact}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    inferred_seed = int(summary.get("seed", dict(summary.get("dataloader_contract") or {}).get("seed", -1)))
    records = list(dict(detail.get("records") or {}).get("parent") or [])
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "known_gate"
        or summary.get("variant") != "baseline"
        or inferred_seed != int(seed)
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or len(list(summary.get("history") or [])) != 20
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("model_state_changed_from_initialization") is not True
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or dict(summary.get("training_readiness") or {}).get("sha256") != readiness_sha256
        or detail.get("variant") != "baseline"
        or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -1))
        or len(records) != 232
    ):
        raise ValueError(f"V21 Baseline seed{seed} violates the formal multiseed contract")
    parent_ids = [str(row["parent_sample_id"]) for row in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError(f"V21 Baseline seed{seed} repeats validation parents")
    metrics = summarize_known_gate_records(records, scope=f"seed{seed}_parent_final_recomputed")
    recorded = dict(dict(summary["best_validation_metrics"])["parent_metrics"])
    for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(metrics[field]) - float(recorded[field])) > 1.0e-12:
            raise ValueError(f"V21 Baseline seed{seed} recomputed metric differs at {field}")
    return {
        "seed": int(seed),
        "artifact": str(artifact),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "detail_path": str(detail_path),
        "detail_sha256": sha256_file(detail_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": int(summary["best_epoch"]),
        "performance_status": summary.get("performance_status"),
        "metrics": metrics,
        "records": records,
        "selection": _selection_record(metrics, seed),
    }


def _ordered_records(runs: Mapping[int, Mapping[str, Any]]) -> tuple[list[str], dict[int, list[dict[str, Any]]]]:
    reference = {
        str(row["parent_sample_id"]): dict(row)
        for row in runs[KNOWN_GATE_BASELINE_MULTISEED_SEEDS[0]]["records"]
    }
    ordered_ids = sorted(reference)
    ordered: dict[int, list[dict[str, Any]]] = {}
    for seed, run in runs.items():
        by_parent = {str(row["parent_sample_id"]): dict(row) for row in run["records"]}
        if set(by_parent) != set(reference):
            raise ValueError(f"V21 Baseline seed{seed} validation parent set differs")
        for parent_id in ordered_ids:
            before, after = reference[parent_id], by_parent[parent_id]
            if before["label_name"] != after["label_name"] or int(before["true_id"]) != int(after["true_id"]):
                raise ValueError(f"V21 Baseline seed{seed} parent label alignment differs")
        ordered[seed] = [by_parent[parent_id] for parent_id in ordered_ids]
    return ordered_ids, ordered


def _bootstrap(runs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    _, ordered = _ordered_records(runs)
    reference = ordered[KNOWN_GATE_BASELINE_MULTISEED_SEEDS[0]]
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(reference) if row["label_name"] == label],
            dtype=np.int64,
        )
        for label in KNOWN_LABELS
    }
    expected_support = {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
    }
    if {label: len(indices) for label, indices in strata.items()} != expected_support:
        raise ValueError("V21 multiseed bootstrap base-label support changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_seed: dict[int, dict[str, list[float]]] = {
        seed: {} for seed in KNOWN_GATE_BASELINE_MULTISEED_SEEDS
    }
    pairwise: dict[str, dict[str, list[float]]] = {
        f"seed{right}_minus_seed{left}": {}
        for left, right in combinations(KNOWN_GATE_BASELINE_MULTISEED_SEEDS, 2)
    }
    selection_counts = {seed: 0 for seed in KNOWN_GATE_BASELINE_MULTISEED_SEEDS}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        metrics_by_seed: dict[int, dict[str, Any]] = {}
        vectors: dict[int, dict[str, float]] = {}
        for seed in KNOWN_GATE_BASELINE_MULTISEED_SEEDS:
            selected_rows = [ordered[seed][int(index)] for index in sampled.tolist()]
            metrics = summarize_known_gate_records(selected_rows, scope="multiseed_parent_bootstrap")
            metrics_by_seed[seed] = metrics
            vector = _metric_vector(metrics)
            vectors[seed] = vector
            for name, value in vector.items():
                per_seed[seed].setdefault(name, []).append(value)
        winner = max(
            KNOWN_GATE_BASELINE_MULTISEED_SEEDS,
            key=lambda seed: _selection_key(metrics_by_seed[seed], seed),
        )
        selection_counts[winner] += 1
        for left, right in combinations(KNOWN_GATE_BASELINE_MULTISEED_SEEDS, 2):
            key = f"seed{right}_minus_seed{left}"
            for name in vectors[left]:
                pairwise[key].setdefault(name, []).append(
                    vectors[right][name] - vectors[left][name]
                )
    return {
        "method": "paired parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": expected_support,
        "per_seed": {
            str(seed): {name: _ci(values) for name, values in metrics.items()}
            for seed, metrics in per_seed.items()
        },
        "pairwise_deltas": {
            pair: {name: _ci(values) for name, values in metrics.items()}
            for pair, metrics in pairwise.items()
        },
        "selection_frequency": {
            str(seed): {
                "count": int(selection_counts[seed]),
                "fraction": float(selection_counts[seed] / BOOTSTRAP_REPLICATES),
            }
            for seed in KNOWN_GATE_BASELINE_MULTISEED_SEEDS
        },
    }


def _aggregate(runs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    vectors = {seed: _metric_vector(run["metrics"]) for seed, run in runs.items()}
    names = next(iter(vectors.values())).keys()
    return {
        name: {
            "mean": float(np.mean([vectors[seed][name] for seed in vectors])),
            "sample_std": float(np.std([vectors[seed][name] for seed in vectors], ddof=1)),
            "minimum": float(min(vectors[seed][name] for seed in vectors)),
            "maximum": float(max(vectors[seed][name] for seed in vectors)),
        }
        for name in names
    }


def build_report(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    multiseed_readiness_path = root / MULTISEED_READINESS
    base_readiness_path = root / BASE_READINESS
    if not multiseed_readiness_path.is_file() or not base_readiness_path.is_file():
        raise FileNotFoundError("V21 multiseed/base readiness is missing")
    readiness = json.loads(multiseed_readiness_path.read_text(encoding="utf-8"))
    checks = dict(readiness.get("checks") or {})
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA
        or readiness.get("seeds") != list(KNOWN_GATE_BASELINE_MULTISEED_SEEDS)
        or not checks
        or not all(value is True for value in checks.values())
        or readiness.get("test") != "未执行"
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V21 multiseed readiness is stale or failed")
    readiness_hash = sha256_file(multiseed_readiness_path)
    base_readiness_hash = sha256_file(base_readiness_path)
    if readiness.get("base_readiness_sha256") != base_readiness_hash:
        raise ValueError("V21 multiseed readiness no longer binds the seed42 readiness")
    runs = {
        seed: _load_run(
            root,
            seed,
            base_readiness_hash if seed == 42 else readiness_hash,
        )
        for seed in KNOWN_GATE_BASELINE_MULTISEED_SEEDS
    }
    ranked = sorted(
        KNOWN_GATE_BASELINE_MULTISEED_SEEDS,
        key=lambda seed: _selection_key(runs[seed]["metrics"], seed),
        reverse=True,
    )
    selected_seed = int(ranked[0])
    bootstrap = _bootstrap(runs)
    selected_frequency = bootstrap["selection_frequency"][str(selected_seed)]
    report_path = root / REPORT_RELATIVE
    selection_path = root / SELECTION_RELATIVE
    report = {
        "status": "completed",
        "schema_version": "RADAR-LN3-V21-known-gate-baseline-multiseed-validation-v1",
        "scope": "development validation parent-final KnownGate Baseline seeds 42/43/44",
        "selection_protocol": selection_protocol(),
        "ranked_seeds": [int(seed) for seed in ranked],
        "selected_seed": selected_seed,
        "selected_role": "development_champion_only",
        "candidates": {
            str(seed): {
                key: value
                for key, value in run.items()
                if key != "records"
            }
            for seed, run in runs.items()
        },
        "aggregate_across_seeds": _aggregate(runs),
        "bootstrap": bootstrap,
        "selection_stability": {
            "selected_seed_bootstrap_win_fraction": selected_frequency["fraction"],
            "selected_seed_bootstrap_win_count": selected_frequency["count"],
            "clutter_validation_support": 9,
            "low_support_selection_warning": True,
        },
        "selection_artifact_path": str(selection_path),
        "unknown_is_model_class": False,
        "selection_is_unbiased_test_estimate": False,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(report_path, report)
    selected = runs[selected_seed]
    selection = {
        "status": "selected_development_champion",
        "schema_version": "RADAR-LN3-V21-known-gate-baseline-development-selection-v1",
        "selected_seed": selected_seed,
        "selection_rank": selected["selection"],
        "validation_metrics": selected["metrics"],
        "config_path": selected["config_path"],
        "config_sha256": selected["config_sha256"],
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "summary_path": selected["summary_path"],
        "summary_sha256": selected["summary_sha256"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "bootstrap_selection_frequency": selected_frequency,
        "clutter_validation_support": 9,
        "selection_bias_warning": (
            "The champion is selected on the same development validation set and is not an unbiased "
            "generalization estimate. Any V21a-d comparison must use a Baseline/Treatment pair with "
            "the same selected seed or, preferably, paired deltas across all three seeds."
        ),
        "unknown_is_model_class": False,
        "test": "未执行",
        "OOF": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(selection_path, selection)
    return report, selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the V21 KnownGate Baseline development champion across seeds 42/43/44."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    report, selection = build_report(root)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": str(root / REPORT_RELATIVE),
                "selection_path": str(root / SELECTION_RELATIVE),
                "ranked_seeds": report["ranked_seeds"],
                "selected_seed": selection["selected_seed"],
                "selected_metrics": selection["validation_metrics"],
                "bootstrap_selection_frequency": selection["bootstrap_selection_frequency"],
                "test": "未执行",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
