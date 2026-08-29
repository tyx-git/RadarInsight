"""Focused no-training contract tests for V26h."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radar_ln3.known_gate_v26h import (
    L1_CLASSES,
    OOD_SCORE_POLICY,
    binary_ranking_metrics,
    fit_class_conditional_reference,
    mahalanobis_scores,
    validate_prepared_contract,
)


def _math_smoke() -> dict[str, object]:
    # Two well-separated train classes with an obvious outlier.  This is a
    # mathematical contract test, not a claim about CQ08 performance.
    train = np.asarray(
        [[-1.1, 0.0], [-0.9, 0.1], [-1.0, -0.1], [0.9, 0.0], [1.1, -0.1], [1.0, 0.1]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    fitted = fit_class_conditional_reference(train, labels)
    probes = np.asarray([[-1.0, 0.0], [1.0, 0.0], [6.0, 6.0]], dtype=np.float64)
    scores, closest = mahalanobis_scores(probes, fitted)
    if not (scores[2] > scores[0] and scores[2] > scores[1]):
        raise AssertionError("V26h novelty orientation is invalid")
    ranking = binary_ranking_metrics(scores[:2], scores[2:])
    if ranking["auroc_unknown_vs_known"] != 1.0 or ranking["average_precision_unknown"] != 1.0:
        raise AssertionError("V26h threshold-free ranking contract is invalid")
    if fitted["class_counts"] != {L1_CLASSES[0]: 3, L1_CLASSES[1]: 3}:
        raise AssertionError("V26h class-conditional train reference drifted")
    return {
        "two_known_classes_have_train_only_references": True,
        "outlier_has_higher_novelty": True,
        "threshold_free_ranking_orientation_is_high_equals_unknown": True,
        "unknown_reference_fit_exposure": 0,
    }


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    contract, readiness = validate_prepared_contract(root)
    if (
        dict(contract.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
        or int(contract.get("unknown_reference_fit_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or contract.get("semantic_action_mutation") is not False
        or contract.get("new_training") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise AssertionError("V26h immutable diagnostic scope was violated")
    return {
        "status": "passed",
        "prepared_contract": True,
        "math_smoke": _math_smoke(),
        "unknown_reference_fit_exposure": 0,
        "threshold_fitting": "not_executed",
        "semantic_action_mutation": False,
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V26h focused contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
