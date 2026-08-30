"""Focused contract and numerical smoke tests for the V26k L2 audit.

The smoke test intentionally does not construct a model or run a CQ08
forward.  It verifies the immutable preparation contract and the numerical
orientation/boundaries of the train-only Mahalanobis helper on a tiny fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radar_ln3.known_gate_v26k import (
    CONTRACT_SCHEMA,
    L2_CLASSES,
    OOD_SCORE_POLICY,
    READINESS_SCHEMA,
    binary_ranking_metrics,
    fit_class_conditional_reference,
    mahalanobis_scores,
    train_only_retention_threshold,
    validate_prepared_contract,
)


def _math_smoke() -> dict[str, object]:
    """Check class-conditional fit, score orientation, and threshold semantics."""

    train = np.asarray(
        [
            [-1.10, 0.00],
            [-0.90, 0.10],
            [-1.00, -0.10],
            [0.90, 0.00],
            [1.10, -0.10],
            [1.00, 0.10],
            [3.90, 0.00],
            [4.10, -0.10],
            [4.00, 0.10],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    fitted = fit_class_conditional_reference(train, labels)
    if fitted["class_counts"] != {
        L2_CLASSES[0]: 3,
        L2_CLASSES[1]: 3,
        L2_CLASSES[2]: 3,
    }:
        raise AssertionError("V26k class-conditional support drifted")

    probes = np.asarray(
        [
            [-1.00, 0.00],
            [1.00, 0.00],
            [4.00, 0.00],
            [20.00, 20.00],
        ],
        dtype=np.float64,
    )
    scores, closest = mahalanobis_scores(probes, fitted)
    if scores.shape != (len(probes),) or closest.shape != (len(probes),):
        raise AssertionError("V26k score shape contract drifted")
    if not (scores[-1] > scores[:3]).all():
        raise AssertionError("V26k high-distance outlier did not score as novel")
    if tuple(closest[:3].tolist()) != (0, 1, 2):
        raise AssertionError("V26k closest-class orientation drifted")

    ranking = binary_ranking_metrics(scores[:3], scores[3:])
    if ranking is None or ranking["auroc_unknown_vs_known"] != 1.0:
        raise AssertionError("V26k ranking orientation is invalid")
    threshold = train_only_retention_threshold(scores[:3], target_retention=2.0 / 3.0)
    if threshold["threshold"] != float(np.sort(scores[:3])[1]):
        raise AssertionError("V26k empirical retention threshold is not order-statistic based")
    if threshold["unknown_threshold_fit_exposure"] != 0:
        raise AssertionError("V26k threshold helper exposed Unknown")

    return {
        "class_conditional_train_only_fit": True,
        "three_known_classes_supported": True,
        "high_distance_means_less_like_known": True,
        "closest_class_orientation_is_stable": True,
        "threshold_is_train_only_empirical_order_statistic": True,
        "unknown_reference_fit_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
    }


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    contract, readiness = validate_prepared_contract(root)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or int(contract.get("fold_count", -1)) != 5
        or int(contract.get("fixed_l2_epoch", -1)) != 15
        or dict(contract.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_reference_fit_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or int(contract.get("unknown_checkpoint_selection_exposure", -1)) != 0
        or contract.get("new_training") is not False
        or contract.get("new_threshold_fit") is not False
        or contract.get("product_threshold_mutation") is not False
        or contract.get("semantic_action_mutation") is not False
        or contract.get("deployment_eligible") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise AssertionError("V26k immutable diagnostic scope was violated")
    checks = dict(readiness.get("checks") or {})
    if not checks or not all(value is True for value in checks.values()):
        raise AssertionError("V26k readiness contains a failed or missing check")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-smoke-v1",
        "prepared_contract": True,
        "math_smoke": _math_smoke(),
        "fold_count": int(contract["fold_count"]),
        "fixed_l2_epoch": int(contract["fixed_l2_epoch"]),
        "unknown_reference_fit_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "training": "not_executed",
        "product_threshold_calibration": "not_executed",
        "semantic_action_mutation": False,
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V26k contract and numerical smoke tests.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
