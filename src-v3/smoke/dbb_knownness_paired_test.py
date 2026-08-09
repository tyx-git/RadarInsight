"""Focused synthetic contract tests for the V12 paired knownness diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from report.dbb_knownness_paired import (
    ARMS,
    COMPARISONS,
    DBB_LABELS,
    DISTANCE_SCORES,
    SCORE_NAMES,
    SEEDS,
    auc,
    build_paired_comparison,
    evaluate_v13_design_evidence,
    fit_subclass_reference,
    knownness_scores,
)


ACTUAL_REPORT = Path("src-v3/processed/report/RADAR-LN3-V12-dbb-knownness-paired-diagnostic.json")


def _synthetic_run(*, arm: str, score_auc: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        "arm": arm,
        "scores": {
            score: {
                "validation_auc": {
                    comparison: {"positive_class_auc": float(value)}
                    for comparison, value in comparisons.items()
                }
            }
            for score, comparisons in score_auc.items()
        },
    }


def _base_auc() -> dict[str, dict[str, float]]:
    return {
        score: {comparison: 0.90 for comparison in COMPARISONS}
        for score in SCORE_NAMES
    }


def _evidence_runs(*, passing: bool) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        control = _base_auc()
        treatment = _base_auc()
        treatment["p_cu"]["Unknown_vs_Bird"] = 0.75
        treatment["p_cu"]["CU_vs_DBB"] = 0.95
        control["nearest_subclass_cosine_distance"]["Unknown_vs_Bird"] = 0.76
        treatment["nearest_subclass_cosine_distance"]["Unknown_vs_Bird"] = (
            0.80 if passing or seed == 42 else 0.77
        )
        treatment["nearest_subclass_cosine_distance"]["Unknown_vs_DBB"] = 0.92
        treatment["nearest_subclass_cosine_distance"]["CU_vs_DBB"] = 0.945
        control["nearest_subclass_diagonal_mahalanobis"]["Unknown_vs_Bird"] = 0.80
        treatment["nearest_subclass_diagonal_mahalanobis"]["Unknown_vs_Bird"] = 0.79
        treatment["nearest_subclass_diagonal_mahalanobis"]["Unknown_vs_DBB"] = 0.91
        treatment["nearest_subclass_diagonal_mahalanobis"]["CU_vs_DBB"] = 0.94
        runs[seed] = {
            "control": _synthetic_run(arm="control", score_auc=control),
            "treatment": _synthetic_run(arm="treatment", score_auc=treatment),
        }
    return runs


def run(project_root: Path) -> dict[str, Any]:
    if auc(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.0])) != 1.0:
        raise AssertionError("knownness AUC orientation is not higher-score-positive")
    if auc(np.asarray([1.0]), np.asarray([1.0])) != 0.5:
        raise AssertionError("knownness AUC tie handling drifted")

    dbb_states = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [0.0, 1.0],
            [0.0, 1.2],
        ],
        dtype=np.float64,
    )
    dbb_labels = np.asarray(
        [
            "DroneTarget",
            "DroneTarget",
            "BirdTarget",
            "BirdTarget",
            "BalloonTarget",
            "BalloonTarget",
        ],
        dtype=object,
    )
    cu_states_a = np.asarray([[100.0, 100.0], [-100.0, -100.0]], dtype=np.float64)
    cu_states_b = np.asarray([[1.0e6, -1.0e6], [-1.0e6, 1.0e6]], dtype=np.float64)
    cu_labels = np.asarray(["ClutterTarget", "UnknownTarget"], dtype=object)
    reference_a = fit_subclass_reference(
        np.concatenate([dbb_states, cu_states_a], axis=0),
        np.concatenate([dbb_labels, cu_labels], axis=0),
    )
    reference_b = fit_subclass_reference(
        np.concatenate([dbb_states, cu_states_b], axis=0),
        np.concatenate([dbb_labels, cu_labels], axis=0),
    )
    if not np.array_equal(reference_a["centers"], reference_b["centers"]):
        raise AssertionError("CU states contaminated train-only DBB subclass centers")
    if not np.array_equal(reference_a["pooled_variance"], reference_b["pooled_variance"]):
        raise AssertionError("CU states contaminated train-only DBB pooled variance")
    if reference_a["cu_rows_excluded"] != 2 or tuple(reference_a["class_counts"]) != DBB_LABELS:
        raise AssertionError("knownness reference did not preserve DBB-only class accounting")
    probes = np.asarray([[0.1, 0.0], [10.0, 10.0]], dtype=np.float64)
    logits = np.asarray([[4.0, -4.0], [-4.0, 4.0]], dtype=np.float64)
    scores = knownness_scores(probes, logits, reference_a)
    if tuple(scores) != SCORE_NAMES:
        raise AssertionError("knownness score names drifted")
    for score in DISTANCE_SCORES:
        if not float(scores[score][1]) > float(scores[score][0]):
            raise AssertionError(f"knownness distance orientation drifted: {score}")
    if not float(scores["p_cu"][1]) > float(scores["p_cu"][0]):
        raise AssertionError("P(CU) orientation drifted")

    passing_runs = _evidence_runs(passing=True)
    passing_comparison = build_paired_comparison(passing_runs)
    passing_gate = evaluate_v13_design_evidence(passing_runs, passing_comparison)
    if passing_gate["passed"] is not True:
        raise AssertionError("V13 design evidence gate rejected a frozen passing example")
    if passing_gate["eligible_candidates"] != ["nearest_subclass_cosine_distance"]:
        raise AssertionError("V13 design evidence gate selected an unexpected candidate")
    failing_runs = _evidence_runs(passing=False)
    failing_gate = evaluate_v13_design_evidence(
        failing_runs,
        build_paired_comparison(failing_runs),
    )
    if failing_gate["passed"] is not False:
        raise AssertionError("V13 design evidence gate accepted an unstable paired effect")
    serialized = json.dumps(failing_gate, ensure_ascii=False)
    if "threshold" in serialized and '"threshold_values_emitted": false' not in serialized:
        raise AssertionError("knownness diagnostic unexpectedly emitted a threshold")
    if tuple(ARMS) != ("control", "treatment"):
        raise AssertionError("paired arm contract drifted")

    actual_report_validated = False
    actual_path = project_root / ACTUAL_REPORT
    if actual_path.is_file():
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        if (
            actual.get("status") != "passed"
            or actual.get("test_evaluation") != "未执行"
            or actual.get("threshold_fitting") != "未执行"
            or actual.get("threshold_values_emitted") is not False
            or actual.get("new_training") != "未执行"
            or actual.get("v13_design_evidence_gate", {}).get("passed") is not False
            or actual.get("v13_design_evidence_gate", {}).get("eligible_candidates") != []
        ):
            raise AssertionError("actual V12 knownness report violates its terminal safety contract")
        if set(actual.get("runs", {})) != {str(seed) for seed in SEEDS}:
            raise AssertionError("actual V12 knownness report does not contain exactly three seeds")
        for seed in SEEDS:
            seed_runs = actual["runs"][str(seed)]
            if set(seed_runs) != set(ARMS):
                raise AssertionError("actual V12 knownness report does not contain both paired arms")
            if (
                seed_runs["control"]["train_parent_order_sha256"]
                != seed_runs["treatment"]["train_parent_order_sha256"]
                or seed_runs["control"]["validation_parent_order_sha256"]
                != seed_runs["treatment"]["validation_parent_order_sha256"]
            ):
                raise AssertionError("actual V12 knownness paired parent order differs")
            for arm in ARMS:
                reference = seed_runs[arm]["reference_contract"]
                if (
                    reference.get("fit_split") != "train_only"
                    or reference.get("validation_in_fit") is not False
                    or reference.get("unknown_prototype") is not False
                    or reference.get("clutter_prototype") is not False
                    or int(reference.get("excluded_CU_train_parent_count", -1)) != 105
                ):
                    raise AssertionError("actual V12 knownness reference contract is contaminated")
        score_table = actual.get("validation_score_table") or {}
        score_path = Path(str(score_table.get("path", "")))
        if not score_path.is_file() or int(score_table.get("row_count", -1)) != 1392:
            raise AssertionError("actual V12 knownness score table is missing or has an invalid row count")
        digest = hashlib.sha256(score_path.read_bytes()).hexdigest()
        if digest != score_table.get("sha256"):
            raise AssertionError("actual V12 knownness score table hash drifted")
        with score_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            score_rows = list(reader)
        if len(score_rows) != 1392 or list(reader.fieldnames or []) != list(score_table.get("columns") or []):
            raise AssertionError("actual V12 knownness score table schema drifted")
        if any("threshold" in name.lower() for name in reader.fieldnames or []):
            raise AssertionError("actual V12 knownness score table contains a threshold column")
        if (
            score_table.get("contains_hidden_state") is not False
            or score_table.get("contains_TR_or_RD_payload") is not False
            or score_table.get("contains_threshold") is not False
        ):
            raise AssertionError("actual V12 knownness score table payload boundary drifted")
        recomputed_auc_count = 0
        for seed in SEEDS:
            for arm in ARMS:
                selected = [
                    row
                    for row in score_rows
                    if int(row["seed"]) == seed and row["arm"] == arm
                ]
                if len(selected) != 232:
                    raise AssertionError("actual V12 knownness score table paired row count drifted")
                labels = np.asarray([row["label_name"] for row in selected], dtype=object)
                masks = {
                    "CU_vs_DBB": (np.isin(labels, ("ClutterTarget", "UnknownTarget")), np.isin(labels, DBB_LABELS)),
                    "Unknown_vs_DBB": (labels == "UnknownTarget", np.isin(labels, DBB_LABELS)),
                    "Unknown_vs_Bird": (labels == "UnknownTarget", labels == "BirdTarget"),
                    "Clutter_vs_DBB": (labels == "ClutterTarget", np.isin(labels, DBB_LABELS)),
                    "Clutter_vs_Bird": (labels == "ClutterTarget", labels == "BirdTarget"),
                }
                for score_name in SCORE_NAMES:
                    values = np.asarray([float(row[score_name]) for row in selected], dtype=np.float64)
                    for comparison_name, (positive, negative) in masks.items():
                        recomputed = auc(values[positive], values[negative])
                        recorded = float(
                            actual["runs"][str(seed)][arm]["scores"][score_name][
                                "validation_auc"
                            ][comparison_name]["positive_class_auc"]
                        )
                        if recomputed != recorded:
                            raise AssertionError(
                                "actual V12 knownness AUC does not reproduce from the unified score table"
                            )
                        recomputed_auc_count += 1
        if recomputed_auc_count != len(SEEDS) * len(ARMS) * len(SCORE_NAMES) * len(COMPARISONS):
            raise AssertionError("actual V12 knownness AUC recomputation coverage drifted")
        actual_report_validated = True

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V12-dbb-knownness-paired-test-v1",
        "auc_orientation_and_ties": "passed",
        "CU_excluded_from_subclass_reference": True,
        "score_orientation": "passed",
        "passing_evidence_example_accepted": True,
        "unstable_evidence_example_rejected": True,
        "actual_report_contract_validated": actual_report_validated,
        "actual_report_auc_cells_recomputed": (
            len(SEEDS) * len(ARMS) * len(SCORE_NAMES) * len(COMPARISONS)
            if actual_report_validated
            else 0
        ),
        "threshold_values_emitted": False,
        "formal_training": "未执行",
        "test_evaluation": "未执行",
        "OOF": "未执行",
    }
    output = project_root / "src-v3/smoke/RADAR-LN3-V12-dbb-knownness-paired-test-result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the frozen V12 paired knownness diagnostic contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
