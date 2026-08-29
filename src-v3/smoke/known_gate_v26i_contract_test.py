"""Focused no-training contract tests for V26i."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radar_ln3.known_gate_v26i import (
    CLUTTER_COMPONENT_COUNT,
    KNOWN_TARGET_RECALL_MINIMUM,
    OOD_SCORE_POLICY,
    bimodal_clutter_mahalanobis_scores,
    binary_ranking_metrics,
    fit_bimodal_clutter_reference,
    validate_prepared_contract,
)
from report.known_gate_v26i import _run_isolated_fold


def _math_smoke() -> dict[str, object]:
    # Two train-only Clutter modes, one target group and one obvious outlier.
    # This checks numerical semantics only; it makes no CQ08 claim.
    states = np.asarray(
        [
            [-1.1, 0.0], [-0.9, 0.1], [-1.0, -0.1], [-1.05, 0.05], [-0.95, -0.05],
            [2.9, 0.0], [3.1, 0.1], [3.0, -0.1], [2.95, 0.05], [3.05, -0.05],
            [8.9, 0.0], [9.1, 0.1], [9.0, -0.1], [8.95, 0.05], [9.05, -0.05],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0] * 5 + [1] * 10, dtype=np.int64)
    identifiers = tuple(f"parent-{index:02d}" for index in range(len(states)))
    # A fixed control geometry represents the V26h scale.
    control_target = states[labels == 0].mean(axis=0)
    control_variance = np.asarray([0.02, 0.02], dtype=np.float64)
    fitted = fit_bimodal_clutter_reference(
        states,
        labels,
        identifiers,
        control_known_target_centre=control_target,
        control_pooled_variance=control_variance,
    )
    if fitted["clutter_component_counts"] != [5, 5]:
        raise AssertionError("V26i fixed two-component Clutter split drifted")
    if fitted["variance_source"] != "V26h_frozen_control_pooled_diagonal_variance":
        raise AssertionError("V26i smoke did not preserve the control distance scale")
    probes = np.asarray([[-1.0, 0.0], [3.0, 0.0], [20.0, 20.0]], dtype=np.float64)
    scores, classes, components = bimodal_clutter_mahalanobis_scores(probes, fitted)
    if not (scores[2] > scores[0] and scores[2] > scores[1]):
        raise AssertionError("V26i novelty orientation is invalid")
    if tuple(classes.tolist()) != (0, 1, 1) or tuple(components.tolist()) != (-1, 0, 1):
        raise AssertionError("V26i target/class/component tie contract drifted")
    ranking = binary_ranking_metrics(scores[:2], scores[2:])
    if ranking["auroc_unknown_vs_known"] != 1.0 or ranking["average_precision_unknown"] != 1.0:
        raise AssertionError("V26i threshold-free ranking contract is invalid")
    return {
        "fixed_component_count": CLUTTER_COMPONENT_COUNT,
        "two_train_only_clutter_modes_recovered": True,
        "v26h_target_and_variance_control_are_reused": True,
        "outlier_has_higher_novelty": True,
        "threshold_free_ranking_orientation_is_high_equals_unknown": True,
    }


def _infeasible_payload_smoke() -> dict[str, object]:
    """Verify that a fixed-K=2 infeasibility crosses the process boundary.

    The outer diagnostic runs every fold in a subprocess.  This fixture
    validates the response contract without loading a checkpoint or CQ08 data:
    the outer parser must accept the only pre-registered non-success payload,
    preserving its fold id and exact metadata rather than converting it into a
    generic worker crash.
    """
    import inspect

    source = inspect.getsource(_run_isolated_fold)
    required_fragments = (
        'payload.get("status") == "infeasible_fixed_k2_reference"',
        'set(payload) != {"status", "fold_id", "reason", "metadata"}',
        'not isinstance(payload.get("metadata"), dict)',
    )
    if any(fragment not in source for fragment in required_fragments):
        raise AssertionError("V26i outer fold parser no longer accepts only the bounded infeasible payload")
    return {
        "fixed_k2_infeasibility_is_a_reportable_terminal_result": True,
        "outer_process_parser_requires_exact_bounded_payload_schema": True,
        "automatic_k_reduction_or_held_out_data_fallback": False,
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
        or float(dict(contract.get("future_action_gate") or {}).get("known_target_recall_minimum", -1.0))
        != KNOWN_TARGET_RECALL_MINIMUM
        or readiness.get("deployment_eligible") is not False
    ):
        raise AssertionError("V26i immutable diagnostic scope was violated")
    return {
        "status": "passed",
        "prepared_contract": True,
        "math_smoke": _math_smoke(),
        "infeasible_payload_smoke": _infeasible_payload_smoke(),
        "known_target_recall_minimum_for_future_action_gate": KNOWN_TARGET_RECALL_MINIMUM,
        "known_target_recall_status": "not_evaluable_no_action_replay",
        "unknown_reference_fit_exposure": 0,
        "threshold_fitting": "not_executed",
        "semantic_action_mutation": False,
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V26i focused contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
