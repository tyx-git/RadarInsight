"""Immutable contract for V26i's fixed two-component Clutter geometry diagnostic.

V26i is a deliberately narrow, *read-only* follow-up to V26h.  It reuses the
same five frozen V26e L1 epoch-32 checkpoints, the same LayerNorm classifier
readout state, and the same V26f mixed CQ08 parent populations.  The only
analytical change is the train-only ``KNOWN_CLUTTER_EVIDENCE`` reference:

* V26h: one Clutter centre;
* V26i: exactly two deterministic Clutter centres.

The KnownTarget centre remains singular.  All three centres share one pooled
diagonal within-reference variance, so V26i tests support geometry rather than
adding a small-sample per-component covariance estimator.  Unknown samples do
not participate in clustering, normalisation, reference fitting, threshold
selection, checkpoint selection, or semantic actions.

This module contains the immutable source/score contract and pure numerical
helpers.  The report runner owns checkpoint inference in short-lived CPU fold
workers to keep Windows allocator use bounded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .known_gate_v26h import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONTRACT as V26H_CONTRACT,
    DETAIL_CSV as V26H_DETAIL_CSV,
    DOCUMENT as V26H_DOCUMENT,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    KNOWN_BASE_LABELS,
    L1_CLASSES,
    L1_EPOCH,
    MAX_CHECKPOINT_BYTES,
    MAX_FEATURE_CACHE_BYTES,
    MAX_JSON_SOURCE_BYTES,
    OOD_SCORE_POLICY as V26H_OOD_SCORE_POLICY,
    READINESS as V26H_READINESS,
    REPORT as V26H_REPORT,
    REPORT_SCHEMA as V26H_REPORT_SCHEMA,
    UNKNOWN_BASE_LABEL,
    binary_ranking_metrics,
    implementation_sha256 as v26h_implementation_sha256,
    parent_id_sequence_sha256,
    read_json,
    relative_path,
    require_bound_file,
    resolve_path,
    sha256_file,
    source_binding,
    state_matrix_sha256,
    validate_prepared_contract as validate_v26h_prepared_contract,
    write_immutable_json,
    write_immutable_text,
)


CONTRACT = Path(
    "data/processed/RADAR-LN3-V26i-cq08-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic.json"
)
DETAIL_CSV = Path(
    "src-v3/processed/report/RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-parent-details.csv"
)
DOCUMENT = Path("src-v3/docs/radar_ln3_v26i_bimodal_clutter_mahalanobis_diagnostic.md")

CONTRACT_SCHEMA = "RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-readiness-v1"
REPORT_SCHEMA = "RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-v1"

MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_DETAIL_BYTES = 3 * 1024 * 1024
MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_READINESS_BYTES = 512 * 1024
MAX_ISOLATED_FOLD_OUTPUT_BYTES = 2 * 1024 * 1024

# Fixed before any V26i score is computed.  It is intentionally not selected
# from Unknown, validation, test, or a sweep of candidate K values.
CLUTTER_COMPONENT_COUNT = 2
MIN_CLUTTER_COMPONENT_PARENT_SUPPORT = 5
KMEANS_MAX_ITERATIONS = 100
KMEANS_CONVERGENCE_ATOL = 1.0e-12
KMEANS_TIE_ATOL = 1.0e-12
VARIANCE_FLOOR_MULTIPLIER = 1.0e-6
VARIANCE_FLOOR_ABSOLUTE = 1.0e-10

REPRESENTATION_NAME = "classifier_layer_norm_liquid_final_state"
SCORE_NAME = "minimum_known_target_or_bimodal_clutter_pooled_diagonal_mahalanobis_distance"
SCORE_ORIENTATION = "higher_means_less_like_train_known"
KNOWN_TARGET_RECALL_MINIMUM = 0.95

OOD_SCORE_POLICY: dict[str, Any] = {
    "score_name": SCORE_NAME,
    "score_orientation": SCORE_ORIENTATION,
    "representation": REPRESENTATION_NAME,
    "representation_dimension": 48,
    "reference_fit": "per_fold_train_known_only_parent_final_states",
    "known_target_reference": "one class-conditional centre, unchanged from V26h",
    "known_clutter_reference": {
        "component_count_fixed": CLUTTER_COMPONENT_COUNT,
        "fit": "deterministic train-only two-means on LayerNorm readout states",
        "initialization": "global farthest train-Clutter parent pair with parent-id tie break",
        "distance_for_assignment": "unweighted squared Euclidean readout-state distance",
        "minimum_component_parent_support": MIN_CLUTTER_COMPONENT_PARENT_SUPPORT,
        "component_count_selection": "not_executed_fixed_before_scoring",
    },
    "covariance": "one pooled diagonal within-reference variance across KnownTarget and both Clutter components",
    "variance_floor_multiplier_of_median_positive_variance": VARIANCE_FLOOR_MULTIPLIER,
    "variance_floor_absolute": VARIANCE_FLOOR_ABSOLUTE,
    "unknown_in_reference_fit": False,
    "unknown_in_normalization": False,
    "unknown_in_threshold_fit": False,
    "unknown_in_model_or_checkpoint_selection": False,
    "threshold_fitting": "not_executed",
    "semantic_action_mutation": False,
    "score_fusion_with_msp_or_energy": False,
    "cross_fold_state_pooling": False,
}

IMPLEMENTATION_FILES = (
    Path("src-v3/radar_ln3/known_gate_v26i.py"),
    Path("src-v3/build/known_gate_v26i.py"),
    Path("src-v3/report/known_gate_v26i.py"),
    Path("src-v3/smoke/known_gate_v26i_contract_test.py"),
    Path("src-v3/scripts/v26i_known_gate_bimodal_clutter_mahalanobis_diagnostic_pipeline.ps1"),
)


class BimodalClutterReferenceInfeasibleError(ValueError):
    """Raised when fixed K=2 cannot be estimated without changing the protocol."""

    def __init__(self, message: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


def _finite_matrix(states: np.ndarray, *, description: str) -> np.ndarray:
    value = np.asarray(states, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1 or not np.isfinite(value).all():
        raise ValueError(f"V26i {description} must be a finite non-empty two-dimensional matrix")
    return value


def _farthest_pair_indices(states: np.ndarray, parent_ids: Sequence[str]) -> tuple[int, int]:
    """Return a deterministic farthest pair without random initialisation."""

    array = _finite_matrix(states, description="Clutter seed states")
    identifiers = tuple(str(value) for value in parent_ids)
    if len(identifiers) != array.shape[0] or len(set(identifiers)) != len(identifiers):
        raise ValueError("V26i Clutter seed parent identifiers must be unique and aligned")
    if array.shape[0] < CLUTTER_COMPONENT_COUNT:
        raise BimodalClutterReferenceInfeasibleError(
            "V26i fixed K=2 has insufficient train Clutter parents",
            metadata={"train_clutter_parent_count": int(array.shape[0])},
        )
    differences = array[:, None, :] - array[None, :, :]
    squared = np.square(differences).sum(axis=2)
    upper = np.triu_indices(array.shape[0], k=1)
    candidates = squared[upper]
    maximum = float(candidates.max())
    tied: list[tuple[str, str, int, int]] = []
    for index, distance in enumerate(candidates):
        if abs(float(distance) - maximum) <= KMEANS_TIE_ATOL:
            left = int(upper[0][index])
            right = int(upper[1][index])
            first, second = sorted((identifiers[left], identifiers[right]))
            tied.append((first, second, left, right))
    if not tied:
        raise AssertionError("V26i farthest-pair initialization found no candidate")
    _first, _second, left, right = min(tied)
    return int(left), int(right)


def _deterministic_two_means(
    states: np.ndarray,
    parent_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit exactly two deterministic centres to train Clutter states only."""

    array = _finite_matrix(states, description="Clutter clustering states")
    identifiers = tuple(str(value) for value in parent_ids)
    left, right = _farthest_pair_indices(array, identifiers)
    centres = np.stack((array[left], array[right]), axis=0)
    assignments: np.ndarray | None = None
    iterations = 0
    for iterations in range(1, KMEANS_MAX_ITERATIONS + 1):
        squared = np.square(array[:, None, :] - centres[None, :, :]).sum(axis=2)
        # ``<`` rather than ``<=`` gives component zero a deterministic tie rule.
        next_assignments = (squared[:, 1] < squared[:, 0]).astype(np.int64, copy=False)
        counts = np.bincount(next_assignments, minlength=CLUTTER_COMPONENT_COUNT)
        if counts.shape != (CLUTTER_COMPONENT_COUNT,) or bool((counts == 0).any()):
            raise BimodalClutterReferenceInfeasibleError(
                "V26i fixed K=2 deterministic Clutter clustering produced an empty component",
                metadata={
                    "component_counts": [int(value) for value in counts.tolist()],
                    "seed_parent_ids": [identifiers[left], identifiers[right]],
                    "iteration": int(iterations),
                },
            )
        updated = np.stack(
            [array[next_assignments == component].mean(axis=0) for component in range(CLUTTER_COMPONENT_COUNT)],
            axis=0,
        )
        if assignments is not None and np.array_equal(next_assignments, assignments):
            centres = updated
            assignments = next_assignments
            break
        if np.allclose(updated, centres, rtol=0.0, atol=KMEANS_CONVERGENCE_ATOL):
            centres = updated
            assignments = next_assignments
            break
        centres = updated
        assignments = next_assignments
    else:
        raise BimodalClutterReferenceInfeasibleError(
            "V26i fixed K=2 deterministic Clutter clustering did not converge",
            metadata={"maximum_iterations": int(KMEANS_MAX_ITERATIONS)},
        )
    if assignments is None:
        raise AssertionError("V26i deterministic Clutter clustering produced no assignments")
    # Recompute once after the final centres so the recorded assignment exactly
    # corresponds to the returned centres.  Empty components remain a hard,
    # reportable infeasibility rather than a silent repair.
    final_squared = np.square(array[:, None, :] - centres[None, :, :]).sum(axis=2)
    final_assignments = (final_squared[:, 1] < final_squared[:, 0]).astype(np.int64, copy=False)
    counts = np.bincount(final_assignments, minlength=CLUTTER_COMPONENT_COUNT)
    if counts.shape != (CLUTTER_COMPONENT_COUNT,) or bool((counts == 0).any()):
        raise BimodalClutterReferenceInfeasibleError(
            "V26i fixed K=2 final Clutter assignment produced an empty component",
            metadata={"component_counts": [int(value) for value in counts.tolist()]},
        )
    final_centres = np.stack(
        [array[final_assignments == component].mean(axis=0) for component in range(CLUTTER_COMPONENT_COUNT)],
        axis=0,
    )
    component_parent_ids = [
        tuple(identifiers[index] for index in np.flatnonzero(final_assignments == component))
        for component in range(CLUTTER_COMPONENT_COUNT)
    ]
    return final_centres, final_assignments, {
        "algorithm": "deterministic_two_means",
        "initialization": "global_farthest_pair_parent_id_tie_break",
        "seed_parent_ids": [identifiers[left], identifiers[right]],
        "iterations": int(iterations),
        "component_counts": [int(value) for value in counts.tolist()],
        "component_parent_ids_sha256": [
            parent_id_sequence_sha256(values) for values in component_parent_ids
        ],
    }


def fit_bimodal_clutter_reference(
    states: np.ndarray,
    class_ids: np.ndarray,
    parent_ids: Sequence[str],
    *,
    control_known_target_centre: np.ndarray | None = None,
    control_pooled_variance: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit the fixed V26i two-component train-only Clutter reference.

    For the strictest one-variable comparison, callers may provide V26h's
    frozen KnownTarget centre and pooled variance.  In that mode V26i changes
    only the Clutter centre geometry, preserving every V26h distance scale.
    The standalone recomputation mode exists solely for numerical unit tests;
    the production runner is required to use the frozen-control mode.
    """

    array = _finite_matrix(states, description="reference states")
    labels = np.asarray(class_ids, dtype=np.int64)
    identifiers = tuple(str(value) for value in parent_ids)
    if (
        labels.shape != (array.shape[0],)
        or len(identifiers) != array.shape[0]
        or len(set(identifiers)) != len(identifiers)
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("V26i reference states, binary labels, and parent identifiers must align")
    target = array[labels == 0]
    clutter = array[labels == 1]
    clutter_parent_ids = tuple(
        identifiers[index] for index in np.flatnonzero(labels == 1)
    )
    if target.shape[0] < 2:
        raise BimodalClutterReferenceInfeasibleError(
            "V26i reference has inadequate train KnownTarget support",
            metadata={"train_known_target_parent_count": int(target.shape[0])},
        )
    if clutter.shape[0] < CLUTTER_COMPONENT_COUNT * MIN_CLUTTER_COMPONENT_PARENT_SUPPORT:
        raise BimodalClutterReferenceInfeasibleError(
            "V26i fixed K=2 train Clutter support is below the pre-registered minimum",
            metadata={
                "train_clutter_parent_count": int(clutter.shape[0]),
                "minimum_required": int(CLUTTER_COMPONENT_COUNT * MIN_CLUTTER_COMPONENT_PARENT_SUPPORT),
            },
        )
    clutter_centres, assignments, clustering = _deterministic_two_means(clutter, clutter_parent_ids)
    component_counts = np.bincount(assignments, minlength=CLUTTER_COMPONENT_COUNT)
    if bool((component_counts < MIN_CLUTTER_COMPONENT_PARENT_SUPPORT).any()):
        raise BimodalClutterReferenceInfeasibleError(
            "V26i fixed K=2 Clutter component is below the pre-registered support minimum",
            metadata={
                **clustering,
                "minimum_component_parent_support": int(MIN_CLUTTER_COMPONENT_PARENT_SUPPORT),
            },
        )
    control_mode = control_known_target_centre is not None or control_pooled_variance is not None
    if control_mode:
        if control_known_target_centre is None or control_pooled_variance is None:
            raise ValueError("V26i frozen-control reference needs both centre and pooled variance")
        target_centre = np.asarray(control_known_target_centre, dtype=np.float64)
        pooled_variance = np.asarray(control_pooled_variance, dtype=np.float64)
        if (
            target_centre.shape != (array.shape[1],)
            or pooled_variance.shape != (array.shape[1],)
            or not np.isfinite(target_centre).all()
            or not np.isfinite(pooled_variance).all()
            or bool((pooled_variance <= 0.0).any())
        ):
            raise ValueError("V26i frozen-control centre or pooled variance is invalid")
        degrees_of_freedom = None
        variance_floor = None
        variance_source = "V26h_frozen_control_pooled_diagonal_variance"
    else:
        target_centre = target.mean(axis=0)
        residual_sum = np.square(target - target_centre).sum(axis=0)
        for component in range(CLUTTER_COMPONENT_COUNT):
            selected = clutter[assignments == component]
            residual_sum += np.square(selected - clutter_centres[component]).sum(axis=0)
        degrees_of_freedom = int(array.shape[0] - (1 + CLUTTER_COMPONENT_COUNT))
        if degrees_of_freedom <= 0:
            raise BimodalClutterReferenceInfeasibleError(
                "V26i pooled diagonal variance has no degrees of freedom",
                metadata={"train_parent_count": int(array.shape[0])},
            )
        pooled_variance = residual_sum / float(degrees_of_freedom)
        positive = pooled_variance[pooled_variance > 0.0]
        variance_floor = max(
            (float(np.median(positive)) if positive.size else 1.0) * VARIANCE_FLOOR_MULTIPLIER,
            VARIANCE_FLOOR_ABSOLUTE,
        )
        pooled_variance = np.maximum(pooled_variance, variance_floor)
        variance_source = "recomputed_pooled_diagonal_variance_unit_test_only"
    if (
        target_centre.shape != (array.shape[1],)
        or clutter_centres.shape != (CLUTTER_COMPONENT_COUNT, array.shape[1])
        or pooled_variance.shape != (array.shape[1],)
        or not np.isfinite(target_centre).all()
        or not np.isfinite(clutter_centres).all()
        or not np.isfinite(pooled_variance).all()
        or bool((pooled_variance <= 0.0).any())
    ):
        raise AssertionError("V26i fitted bimodal reference violates the finite diagonal contract")
    return {
        "known_target_centre": target_centre,
        "clutter_centres": clutter_centres,
        "pooled_variance": pooled_variance,
        "variance_floor": float(variance_floor) if variance_floor is not None else None,
        "variance_source": variance_source,
        "class_counts": {
            L1_CLASSES[0]: int(target.shape[0]),
            L1_CLASSES[1]: int(clutter.shape[0]),
        },
        "clutter_component_counts": [int(value) for value in component_counts.tolist()],
        "train_parent_count": int(array.shape[0]),
        "pooled_degrees_of_freedom": int(degrees_of_freedom) if degrees_of_freedom is not None else None,
        "state_dimension": int(array.shape[1]),
        "clustering": clustering,
    }


def bimodal_clutter_mahalanobis_scores(
    states: np.ndarray,
    fitted: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return novelty, closest L1 class, and closest Clutter component.

    ``closest_clutter_component`` is ``-1`` whenever the KnownTarget centre is
    closer.  The score is the minimum over one target and two Clutter centres.
    """

    array = _finite_matrix(states, description="bimodal scoring states")
    target = np.asarray(fitted.get("known_target_centre"), dtype=np.float64)
    clutter = np.asarray(fitted.get("clutter_centres"), dtype=np.float64)
    variance = np.asarray(fitted.get("pooled_variance"), dtype=np.float64)
    if (
        target.shape != (array.shape[1],)
        or clutter.shape != (CLUTTER_COMPONENT_COUNT, array.shape[1])
        or variance.shape != (array.shape[1],)
        or not np.isfinite(target).all()
        or not np.isfinite(clutter).all()
        or not np.isfinite(variance).all()
        or bool((variance <= 0.0).any())
    ):
        raise ValueError("V26i bimodal Mahalanobis scoring received an invalid reference")
    # The one KnownTarget and two Clutter centres all use the same pooled
    # diagonal variance.  This keeps the sole change confined to Clutter's
    # support geometry, not a new covariance family.
    target_distance = np.sqrt(
        (np.square(array - target[None, :]) / variance[None, :]).sum(axis=1)
    )
    clutter_squared = (
        np.square(array[:, None, :] - clutter[None, :, :]) / variance[None, None, :]
    ).sum(axis=2)
    clutter_distances = np.sqrt(clutter_squared)
    closest_component = clutter_distances.argmin(axis=1).astype(np.int64, copy=False)
    closest_clutter_distance = clutter_distances[
        np.arange(array.shape[0]), closest_component
    ]
    closest_class = (closest_clutter_distance < target_distance).astype(np.int64, copy=False)
    # Tie rule matches V26h's lower-index class behaviour: KnownTarget wins a
    # tie, and a component is only meaningful if Clutter actually wins.
    component_result = np.where(closest_class == 1, closest_component, -1).astype(np.int64)
    score = np.minimum(target_distance, closest_clutter_distance)
    if (
        score.shape != (array.shape[0],)
        or closest_class.shape != score.shape
        or component_result.shape != score.shape
        or not np.isfinite(score).all()
        or not np.isin(closest_class, (0, 1)).all()
        or not np.isin(component_result, (-1, 0, 1)).all()
    ):
        raise AssertionError("V26i bimodal Mahalanobis score vector is invalid")
    return score, closest_class, component_result


def _source_v26h_report(root: Path) -> tuple[dict[str, Any], Path, Path]:
    contract, readiness = validate_v26h_prepared_contract(root)
    report_path = require_bound_file(root, V26H_REPORT, "V26h source report", max_bytes=MAX_REPORT_BYTES)
    detail_path = require_bound_file(root, V26H_DETAIL_CSV, "V26h source parent detail", max_bytes=MAX_DETAIL_BYTES)
    report = read_json(report_path, description="V26h source report", max_bytes=MAX_REPORT_BYTES)
    if (
        report.get("schema_version") != V26H_REPORT_SCHEMA
        or report.get("status") != "completed_v26h_train_only_mahalanobis_ood_ranking_diagnostic"
        or report.get("contract_sha256") != sha256_file(root / V26H_CONTRACT)
        or report.get("readiness_sha256") != sha256_file(root / V26H_READINESS)
        or report.get("parent_detail_csv_sha256") != sha256_file(detail_path)
        or dict(report.get("ood_score_policy") or {}) != V26H_OOD_SCORE_POLICY
        or dict(report.get("implementation_sha256") or {}) != dict(contract.get("implementation_sha256") or {})
        or int(dict(report.get("source_protocol") or {}).get("unknown_reference_fit_exposure", -1)) != 0
        or int(dict(report.get("source_protocol") or {}).get("unknown_threshold_fit_exposure", -1)) != 0
        or report.get("deployment_eligible") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V26i V26h source report is stale or outside the frozen diagnostic scope")
    return report, report_path, detail_path


def validate_v26h_source(root: Path) -> dict[str, Any]:
    """Bind V26i to the completed V26h source chain without recomputing it."""

    v26h_contract, v26h_readiness = validate_v26h_prepared_contract(root)
    report, report_path, detail_path = _source_v26h_report(root)
    if (
        v26h_contract.get("schema_version")
        != "RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-contract-v1"
        or v26h_readiness.get("schema_version")
        != "RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-readiness-v1"
        or v26h_readiness.get("status") != "passed"
        or v26h_readiness.get("contract_sha256") != sha256_file(root / V26H_CONTRACT)
        or int(v26h_contract.get("unknown_reference_fit_exposure", -1)) != 0
        or int(v26h_contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or v26h_contract.get("new_training") is not False
        or v26h_contract.get("new_threshold_fit") is not False
        or v26h_contract.get("semantic_action_mutation") is not False
        or v26h_contract.get("deployment_eligible") is not False
    ):
        raise ValueError("V26i V26h source contract is stale or out of scope")
    return {
        "v26h_contract_sha256": sha256_file(root / V26H_CONTRACT),
        "v26h_readiness_sha256": sha256_file(root / V26H_READINESS),
        "v26h_report_sha256": sha256_file(report_path),
        "v26h_parent_detail_sha256": sha256_file(detail_path),
        "v26h_contract_schema": v26h_contract.get("schema_version"),
        "v26h_readiness_schema": v26h_readiness.get("schema_version"),
        "v26h_report_schema": report.get("schema_version"),
        "v26h_score_policy": V26H_OOD_SCORE_POLICY,
        "fixed_epoch": int(L1_EPOCH),
        "fold_count": int(FOLD_COUNT),
        "mixed_parent_counts": EXPECTED_PARENT_COUNTS,
    }


def implementation_sha256(root: Path) -> dict[str, str]:
    return {
        relative_path(root, path): sha256_file(require_bound_file(root, path, f"implementation {path}"))
        for path in IMPLEMENTATION_FILES
    }


def validate_prepared_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless V26i remains the frozen V26h-derived diagnostic."""

    source_v26h = validate_v26h_source(root)
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    contract = read_json(contract_path, description="V26i contract", max_bytes=MAX_CONTRACT_BYTES)
    readiness = read_json(readiness_path, description="V26i readiness", max_bytes=MAX_READINESS_BYTES)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or dict(contract.get("source_v26h") or {}) != source_v26h
        or dict(contract.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
        or dict(contract.get("implementation_sha256") or {}) != implementation_sha256(root)
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or float(dict(contract.get("future_action_gate") or {}).get("known_target_recall_minimum", -1.0))
        != KNOWN_TARGET_RECALL_MINIMUM
        or int(contract.get("unknown_reference_fit_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or contract.get("new_training") is not False
        or contract.get("new_threshold_fit") is not False
        or contract.get("semantic_action_mutation") is not False
        or contract.get("deployment_eligible") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V26i prepared contract/readiness is invalid or stale")
    for raw in list(contract.get("source_bindings") or ()):
        item = dict(raw)
        path = require_bound_file(
            root,
            str(item.get("path", "")),
            f"source binding {item.get('name', '')}",
            expected_sha256=str(item.get("sha256", "")),
            max_bytes=MAX_REPORT_BYTES,
        )
        if int(item.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError("V26i source binding size differs")
    return contract, readiness


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "BimodalClutterReferenceInfeasibleError",
    "CLUTTER_COMPONENT_COUNT",
    "CONTRACT",
    "CONTRACT_SCHEMA",
    "DETAIL_CSV",
    "DOCUMENT",
    "EXPECTED_PARENT_COUNTS",
    "FOLD_COUNT",
    "KMEANS_MAX_ITERATIONS",
    "KNOWN_BASE_LABELS",
    "KNOWN_TARGET_RECALL_MINIMUM",
    "L1_CLASSES",
    "L1_EPOCH",
    "MAX_CONTRACT_BYTES",
    "MAX_DETAIL_BYTES",
    "MAX_ISOLATED_FOLD_OUTPUT_BYTES",
    "MAX_READINESS_BYTES",
    "MAX_REPORT_BYTES",
    "MIN_CLUTTER_COMPONENT_PARENT_SUPPORT",
    "OOD_SCORE_POLICY",
    "READINESS",
    "READINESS_SCHEMA",
    "REPORT",
    "REPORT_SCHEMA",
    "REPRESENTATION_NAME",
    "SCORE_NAME",
    "UNKNOWN_BASE_LABEL",
    "V26H_CONTRACT",
    "V26H_DETAIL_CSV",
    "V26H_DOCUMENT",
    "V26H_READINESS",
    "V26H_REPORT",
    "bimodal_clutter_mahalanobis_scores",
    "binary_ranking_metrics",
    "fit_bimodal_clutter_reference",
    "implementation_sha256",
    "parent_id_sequence_sha256",
    "read_json",
    "relative_path",
    "require_bound_file",
    "resolve_path",
    "sha256_file",
    "source_binding",
    "state_matrix_sha256",
    "validate_prepared_contract",
    "validate_v26h_source",
    "write_immutable_json",
    "write_immutable_text",
]
