"""Paired known-only evaluation for the V27c motion residual."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records

from .features import load_feature_cache, load_feature_manifest
from .protocol import (
    V27C_FROZEN_ERROR_PARENT_IDS,
    V27C_FROZEN_MAIN_CONFUSION,
    V27C_L2_CLASSES,
    V27C_REPORT_SCHEMA_VERSION,
    V27CContext,
    v27c_artifact_path,
    write_sealed_json_if_identical,
)
from .training import _MotionDataset, _collate, load_v27c_checkpoint


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _records(
    payload: Mapping[str, Any], logits: np.ndarray, *, arm: str, residual: np.ndarray | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lengths = payload["tensors"]["lengths"].numpy().astype(np.int64)
    padding = payload["tensors"]["padding_mask"].numpy()
    values = payload["tensors"]["motion_values"].numpy()
    labels = [str(value) for value in payload["label_names"]]
    parents = [str(value) for value in payload["parent_ids"]]
    probabilities = _softmax(logits)
    parent_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for row, (parent_id, label, length) in enumerate(zip(parents, labels, lengths, strict=True)):
        if length <= 0 or not np.array_equal(padding[row], np.arange(logits.shape[1]) >= length):
            raise ValueError("V27c evaluation parent padding differs")
        true_id = V27C_L2_CLASSES.index(label)
        final = int(length) - 1
        final_nonzero = bool(np.abs(values[row, final]).max() > 0.0)
        total_nonzero = int((np.abs(values[row, :length]).max(axis=-1) > 0.0).sum())
        coverage = total_nonzero / int(length)
        parent_rows.append({
            "parent_sample_id": parent_id,
            "true_id": true_id,
            "pred_id": int(np.argmax(probabilities[row, final])),
            "probabilities": [float(value) for value in probabilities[row, final].tolist()],
            "logits": [float(value) for value in logits[row, final].tolist()],
            "event_count": int(length),
            "motion_nonzero_event_count": total_nonzero,
            "motion_nonzero_event_fraction": float(coverage),
            "final_motion_nonzero": final_nonzero,
            "arm": arm,
            "residual": (
                [float(value) for value in residual[row, final].tolist()]
                if residual is not None else [0.0] * len(V27C_L2_CLASSES)
            ),
        })
        for event in range(int(length)):
            event_rows.append({
                "parent_sample_id": parent_id,
                "true_id": true_id,
                "pred_id": int(np.argmax(probabilities[row, event])),
                "probabilities": [float(value) for value in probabilities[row, event].tolist()],
                "logits": [float(value) for value in logits[row, event].tolist()],
                "event_index": event,
                "window_length": event + 1,
                "parent_length": int(length),
                "motion_nonzero": bool(np.abs(values[row, event]).max() > 0.0),
                "arm": arm,
            })
    return parent_rows, event_rows


def _strata(record: Mapping[str, Any]) -> str:
    coverage = float(record["motion_nonzero_event_fraction"])
    if coverage == 0.0:
        return "motion_zero"
    if coverage < 0.5:
        return "motion_sparse_lt_0_5"
    return "motion_coverage_ge_0_5"


def _group_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_strata(record)].append(record)
    return {
        name: {
            "support": len(rows),
            "metrics": summarize_records(rows, class_names=V27C_L2_CLASSES, scope=f"parent_final:{name}"),
        }
        for name, rows in sorted(groups.items())
    }


def _motion_coverage_by_class(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, Any]] = {}
    for class_id, class_name in enumerate(V27C_L2_CLASSES):
        rows = [record for record in records if int(record["true_id"]) == class_id]
        if not rows:
            raise ValueError("V27c coverage audit lacks a DBB class")
        coverage = np.asarray(
            [float(record["motion_nonzero_event_fraction"]) for record in rows],
            dtype=np.float64,
        )
        final = np.asarray([bool(record["final_motion_nonzero"]) for record in rows])
        result[class_name] = {
            "parent_support": len(rows),
            "mean_motion_nonzero_event_fraction": float(coverage.mean()),
            "median_motion_nonzero_event_fraction": float(np.median(coverage)),
            "final_motion_nonzero_fraction": float(final.mean()),
        }
    return result


def _length_preserving_parent_permutation(lengths: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute motion rows only among identical sequence lengths.

    This is label-free and keeps every non-zero token inside an active event;
    applying a generic parent permutation would violate the padding contract.
    """

    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("V27c permutation needs non-empty parent lengths")
    result = np.arange(lengths.size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for length in np.unique(lengths):
        indices = np.flatnonzero(lengths == length)
        if indices.size > 1:
            result[indices] = _cyclic_derangement(indices, rng=rng)
    return result


def _cyclic_derangement(
    indices: np.ndarray, *, rng: np.random.Generator
) -> np.ndarray:
    """Return a deterministic seeded derangement of distinct row indices."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size < 2 or len(set(values.tolist())) != values.size:
        raise ValueError("V27c derangement needs at least two distinct indices")
    return np.roll(values, int(rng.integers(1, values.size)))


def _support_pattern_preserving_parent_permutation(
    motion_values: np.ndarray, lengths: np.ndarray, *, seed: int
) -> tuple[np.ndarray, int]:
    """Permute values only within identical active-event support patterns.

    Unlike the required length-preserving permutation, this stricter diagnostic
    holds each parent's numeric-zero pattern fixed.  It therefore isolates the
    use of motion values from a possible missingness/coverage shortcut.  Rows
    in singleton patterns remain in place, so the resulting support count is
    reported rather than overstating its coverage.
    """

    if (
        motion_values.ndim != 3
        or lengths.ndim != 1
        or motion_values.shape[0] != lengths.size
        or lengths.size == 0
    ):
        raise ValueError("V27c support-pattern permutation has invalid shapes")
    result = np.arange(lengths.size, dtype=np.int64)
    groups: dict[tuple[int, tuple[bool, ...]], list[int]] = defaultdict(list)
    for row, length in enumerate(lengths.tolist()):
        if not 0 < int(length) <= motion_values.shape[1]:
            raise ValueError("V27c support-pattern permutation length differs")
        active = tuple(np.abs(motion_values[row, : int(length)]).max(axis=-1) > 0.0)
        groups[(int(length), active)].append(row)
    rng = np.random.default_rng(int(seed))
    moved = 0
    for indices in groups.values():
        if len(indices) > 1:
            group = np.asarray(indices, dtype=np.int64)
            result[group] = _cyclic_derangement(group, rng=rng)
            moved += len(indices)
    return result, moved


def _component_support_preserving_event_value_permutation(
    motion_values: np.ndarray, lengths: np.ndarray, *, seed: int
) -> tuple[np.ndarray, int, int]:
    """Permute values while preserving every active event's nine-value support.

    Each donor has the same parent length, causal event index, and per-feature
    zero/non-zero pattern as its recipient.  Thus the counterfactual preserves
    padding, prefix position, event-presence, and component-level availability
    while breaking the motion magnitudes/signs and their parent trajectories.
    It is label-free and evaluation-only.
    """

    if (
        motion_values.ndim != 3
        or lengths.ndim != 1
        or motion_values.shape[0] != lengths.size
        or lengths.size == 0
    ):
        raise ValueError("V27c component-support permutation has invalid shapes")
    result = motion_values.copy()
    groups: dict[tuple[int, int, tuple[bool, ...]], list[int]] = defaultdict(list)
    for row, length in enumerate(lengths.tolist()):
        if not 0 < int(length) <= motion_values.shape[1]:
            raise ValueError("V27c component-support permutation length differs")
        for event in range(int(length)):
            support = tuple((motion_values[row, event] != 0.0).tolist())
            groups[(int(length), event, support)].append(row)
    rng = np.random.default_rng(int(seed))
    moved_events = 0
    moved_parents: set[int] = set()
    for (_, event, _), rows in groups.items():
        if len(rows) < 2:
            continue
        group = np.asarray(rows, dtype=np.int64)
        donors = _cyclic_derangement(group, rng=rng)
        result[group, event] = motion_values[donors, event]
        moved_events += len(group)
        moved_parents.update(group.tolist())
    if not np.array_equal(motion_values == 0.0, result == 0.0):
        raise AssertionError("V27c component-support permutation altered a zero pattern")
    return result, moved_events, len(moved_parents)


def _paired_transitions(control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right):
        raise ValueError("V27c paired arms have different parent sets")
    transitions: dict[str, int] = defaultdict(int)
    confusion_delta = np.zeros((3, 3), dtype=np.int64)
    frozen_error_rows: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        base, motion = left[parent_id], right[parent_id]
        if int(base["true_id"]) != int(motion["true_id"]):
            raise ValueError("V27c paired labels differ")
        base_correct = int(base["pred_id"]) == int(base["true_id"])
        motion_correct = int(motion["pred_id"]) == int(motion["true_id"])
        transitions[f"{'correct' if base_correct else 'wrong'}_to_{'correct' if motion_correct else 'wrong'}"] += 1
        confusion_delta[int(motion["true_id"]), int(motion["pred_id"])] += 1
        confusion_delta[int(base["true_id"]), int(base["pred_id"])] -= 1
        if parent_id in V27C_FROZEN_ERROR_PARENT_IDS:
            frozen_error_rows.append({
                "parent_sample_id": parent_id,
                "control_pred": V27C_L2_CLASSES[int(base["pred_id"])],
                "motion_pred": V27C_L2_CLASSES[int(motion["pred_id"])],
                "motion_final_nonzero": bool(motion["final_motion_nonzero"]),
                "motion_nonzero_event_fraction": float(motion["motion_nonzero_event_fraction"]),
                "motion_residual": motion["residual"],
            })
    expected_errors = set(V27C_FROZEN_ERROR_PARENT_IDS)
    observed_errors = {str(row["parent_sample_id"]) for row in frozen_error_rows}
    if observed_errors != expected_errors:
        raise AssertionError("V27c frozen Drone-to-Bird error identities differ")
    if any(
        int(left[str(row["parent_sample_id"])]["true_id"]) != 0
        or row["control_pred"] != "BirdTarget"
        for row in frozen_error_rows
    ):
        raise AssertionError("V27c archived frozen errors no longer are Drone-to-Bird")
    return {
        "parent_count": len(left),
        "correctness_transitions": dict(sorted(transitions.items())),
        "treatment_minus_control_confusion": confusion_delta.tolist(),
        "frozen_drone_to_bird_error_audit": frozen_error_rows,
    }


def _paired_bootstrap(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    strata: dict[int, list[str]] = {index: [] for index in range(len(V27C_L2_CLASSES))}
    for parent_id, row in left.items():
        if int(row["true_id"]) != int(right[parent_id]["true_id"]):
            raise ValueError("V27c bootstrap labels differ")
        strata[int(row["true_id"])].append(parent_id)
    if any(not values for values in strata.values()) or replicates < 100:
        raise ValueError("V27c bootstrap requires class support and at least 100 replicates")
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = {name: [] for name in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "balanced_accuracy")}
    for _ in range(replicates):
        ids = [parent_id for rows in strata.values() for parent_id in rng.choice(rows, size=len(rows), replace=True).tolist()]
        base = summarize_records([left[item] for item in ids], class_names=V27C_L2_CLASSES, scope="bootstrap")
        motion = summarize_records([right[item] for item in ids], class_names=V27C_L2_CLASSES, scope="bootstrap")
        for name in values:
            values[name].append(float(motion[name]) - float(base[name]))
    return {
        "method": "stratified_parent_paired_bootstrap",
        "seed": int(seed),
        "replicates": int(replicates),
        "metric_deltas_treatment_minus_control": {
            name: {
                "lower_95": float(np.quantile(value, 0.025)),
                "upper_95": float(np.quantile(value, 0.975)),
                "mean": float(np.mean(value)),
            }
            for name, value in values.items()
        },
    }


def _metric_delta(
    reference: Mapping[str, Any], comparison: Mapping[str, Any]
) -> dict[str, float]:
    return {
        name: float(comparison[name]) - float(reference[name])
        for name in (
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "balanced_accuracy",
        )
    }


def _inference(context: V27CContext, *, seed: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model, checkpoint = load_v27c_checkpoint(context, seed=seed, device=torch.device("cpu"))
    payload = load_feature_cache(context, split="val")
    dataset = _MotionDataset(payload)
    loader = DataLoader(dataset, batch_size=context.config.training.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    logits: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    parents: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(
                frozen_endpoint_logits=batch["frozen_endpoint_logits"],
                motion_values=batch["motion_values"],
                padding_mask=batch["padding_mask"],
            )
            logits.append(output["sequence_logits"])
            residuals.append(output["residual_logits"])
            parents.extend(batch["parent_ids"])
    if parents != [str(value) for value in payload["parent_ids"]]:
        raise ValueError("V27c evaluation changed parent order")
    return checkpoint, torch.cat(logits).numpy(), torch.cat(residuals).numpy()


def evaluate_v27c_motion(context: V27CContext, *, seed: int) -> dict[str, Any]:
    """Build a complete paired report for one fixed V27c seed."""

    payload = load_feature_cache(context, split="val")
    baseline = payload["tensors"]["frozen_endpoint_logits"].numpy()
    checkpoint, motion_logits, residual = _inference(context, seed=seed)
    control_parents, control_events = _records(payload, baseline, arm="frozen_control", residual=None)
    motion_parents, motion_events = _records(payload, motion_logits, arm="motion_treatment", residual=residual)
    prefix_control = parent_balanced_prefix_records(control_events, minimum_events=context.config.training.eligible_prefix_min_events, class_names=V27C_L2_CLASSES)
    prefix_motion = parent_balanced_prefix_records(motion_events, minimum_events=context.config.training.eligible_prefix_min_events, class_names=V27C_L2_CLASSES)
    paired = _paired_transitions(control_parents, motion_parents)
    control_metrics = summarize_records(
        control_parents, class_names=V27C_L2_CLASSES, scope="parent_final"
    )
    control_matrix = control_metrics["confusion_matrix_rows_true_columns_pred"]
    if control_matrix != [list(row) for row in V27C_FROZEN_MAIN_CONFUSION]:
        raise AssertionError("V27c frozen control no longer replays V27a Main")
    zero_motion = np.zeros_like(payload["tensors"]["motion_values"].numpy())
    model, _ = load_v27c_checkpoint(context, seed=seed, device=torch.device("cpu"))
    with torch.inference_mode():
        zeroed = model(
            frozen_endpoint_logits=torch.from_numpy(baseline),
            motion_values=torch.from_numpy(zero_motion),
            padding_mask=payload["tensors"]["padding_mask"],
        )["sequence_logits"].numpy()
    zeroing_exact = bool(np.array_equal(zeroed, baseline))
    if not zeroing_exact:
        raise AssertionError("V27c zero-motion counterfactual must exactly replay frozen control")
    original_motion = payload["tensors"]["motion_values"].numpy()
    permutation_seed = 27_399 + int(seed)
    permutation = _length_preserving_parent_permutation(
        payload["tensors"]["lengths"].numpy(), seed=permutation_seed
    )
    with torch.inference_mode():
        permuted = model(
            frozen_endpoint_logits=torch.from_numpy(baseline),
            motion_values=torch.from_numpy(original_motion[permutation].copy()),
            padding_mask=payload["tensors"]["padding_mask"],
        )["sequence_logits"].numpy()
    permutation_changed = bool(not np.array_equal(permuted, motion_logits))
    permuted_parents, _ = _records(
        payload, permuted, arm="length_preserving_motion_permutation", residual=None
    )
    support_permutation_seed = 27_799 + int(seed)
    support_permutation, support_permutation_movable_parent_count = (
        _support_pattern_preserving_parent_permutation(
            original_motion,
            payload["tensors"]["lengths"].numpy(),
            seed=support_permutation_seed,
        )
    )
    with torch.inference_mode():
        support_permuted = model(
            frozen_endpoint_logits=torch.from_numpy(baseline),
            motion_values=torch.from_numpy(original_motion[support_permutation].copy()),
            padding_mask=payload["tensors"]["padding_mask"],
        )["sequence_logits"].numpy()
    support_permutation_changed = bool(
        not np.array_equal(support_permuted, motion_logits)
    )
    component_permutation_seed = 28_199 + int(seed)
    component_permuted_motion, component_moved_events, component_moved_parents = (
        _component_support_preserving_event_value_permutation(
            original_motion,
            payload["tensors"]["lengths"].numpy(),
            seed=component_permutation_seed,
        )
    )
    with torch.inference_mode():
        component_permuted = model(
            frozen_endpoint_logits=torch.from_numpy(baseline),
            motion_values=torch.from_numpy(component_permuted_motion),
            padding_mask=payload["tensors"]["padding_mask"],
        )["sequence_logits"].numpy()
    component_permutation_changed = bool(
        not np.array_equal(component_permuted, motion_logits)
    )
    active_event_mask = (
        np.arange(original_motion.shape[1])[None, :]
        < payload["tensors"]["lengths"].numpy()[:, None]
    )
    component_value_changed = (
        np.abs(component_permuted_motion - original_motion).max(axis=-1) > 0.0
    )
    component_value_changed_event_count = int(component_value_changed[active_event_mask].sum())
    component_active_event_count = int(active_event_mask.sum())
    if component_active_event_count <= 0:
        raise ValueError("V27c component-support counterfactual lacks active events")
    motion_metrics = summarize_records(
        motion_parents, class_names=V27C_L2_CLASSES, scope="parent_final"
    )
    support_permuted_parents, _ = _records(
        payload,
        support_permuted,
        arm="support_pattern_preserving_motion_permutation",
        residual=None,
    )
    support_permuted_metrics = summarize_records(
        support_permuted_parents,
        class_names=V27C_L2_CLASSES,
        scope="parent_final:support_pattern_preserving_motion_permutation",
    )
    component_permuted_parents, _ = _records(
        payload,
        component_permuted,
        arm="component_support_preserving_motion_value_permutation",
        residual=None,
    )
    component_permuted_metrics = summarize_records(
        component_permuted_parents,
        class_names=V27C_L2_CLASSES,
        scope="parent_final:component_support_preserving_motion_value_permutation",
    )
    report = {
        "schema_version": V27C_REPORT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "paired_known_only_motion_validation",
        "seed": int(seed),
        "manifest_sha256": context.manifest_sha256,
        "feature_manifest_sha256": load_feature_manifest(context)["payload_sha256"],
        "validation_motion_cache_sha256": payload["cache_sha256"],
        "frozen_v27a_main_checkpoint_sha256": context.config.base.expected_v27a_main_checkpoint_sha256,
        "checkpoint_state_sha256": checkpoint["model_state_sha256"],
        "class_names": list(V27C_L2_CLASSES),
        "arms": {
            "frozen_control": {
                "parent_metrics": control_metrics,
                "parent_balanced_eligible_prefix_metrics": summarize_records(prefix_control, class_names=V27C_L2_CLASSES, scope="parent_balanced_eligible_prefix"),
            },
            "motion_treatment": {
                "parent_metrics": motion_metrics,
                "parent_balanced_eligible_prefix_metrics": summarize_records(prefix_motion, class_names=V27C_L2_CLASSES, scope="parent_balanced_eligible_prefix"),
                "motion_coverage_groups": _group_metrics(motion_parents),
                "motion_coverage_by_true_class": _motion_coverage_by_class(motion_parents),
            },
        },
        "paired": paired,
        "parent_final_metric_delta_treatment_minus_control": _metric_delta(
            control_metrics, motion_metrics
        ),
        "paired_bootstrap": _paired_bootstrap(
            control_parents, motion_parents, seed=27_300 + int(seed),
            replicates=context.config.training.bootstrap_replicates,
        ),
        "counterfactuals": {
            "all_motion_values_zero_preserves_original_mask": True,
            "zero_motion_replays_frozen_control_bitwise": zeroing_exact,
            "label_free_length_preserving_parent_motion_permutation_seed": permutation_seed,
            "parent_motion_permutation_changes_treatment_output": permutation_changed,
            "parent_motion_permutation_max_abs_logit_delta": float(
                np.abs(permuted - motion_logits).max()
            ),
            "support_pattern_preserving_parent_motion_permutation_seed": support_permutation_seed,
            "support_pattern_permutation_movable_parent_count": support_permutation_movable_parent_count,
            "support_pattern_permutation_changes_treatment_output": support_permutation_changed,
            "support_pattern_permutation_max_abs_logit_delta": float(
                np.abs(support_permuted - motion_logits).max()
            ),
            "support_pattern_permuted_motion_parent_metrics": support_permuted_metrics,
            "motion_treatment_minus_support_pattern_permuted_metric_delta": _metric_delta(
                support_permuted_metrics, motion_metrics
            ),
            "component_support_preserving_motion_value_permutation_seed": component_permutation_seed,
            "component_support_permutation_moved_event_count": component_moved_events,
            "component_support_permutation_moved_parent_count": component_moved_parents,
            "component_support_permutation_numeric_changed_event_count": component_value_changed_event_count,
            "component_support_permutation_numeric_changed_active_event_fraction": float(
                component_value_changed_event_count / component_active_event_count
            ),
            "component_support_permutation_zero_pattern_preserved": bool(
                np.array_equal(component_permuted_motion == 0.0, original_motion == 0.0)
            ),
            "component_support_permutation_changes_treatment_output": component_permutation_changed,
            "component_support_permutation_max_abs_logit_delta": float(
                np.abs(component_permuted - motion_logits).max()
            ),
            "component_support_permuted_motion_parent_metrics": component_permuted_metrics,
            "motion_treatment_minus_component_support_permuted_metric_delta": _metric_delta(
                component_permuted_metrics, motion_metrics
            ),
            "permuted_motion_parent_metrics": summarize_records(
                permuted_parents, class_names=V27C_L2_CLASSES,
                scope="parent_final:length_preserving_motion_permutation",
            ),
        },
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
        "records": {"control_parent": control_parents, "motion_parent": motion_parents},
    }
    path = v27c_artifact_path(context.project_root, context.config.artifacts.evaluation_report_template, seed=seed)
    write_sealed_json_if_identical(path, report)
    return {**report, "report_path": str(path)}


def summarize_v27c_seeds(context: V27CContext) -> dict[str, Any]:
    """Summarize immutable per-seed reports without selecting a best seed."""

    reports: list[dict[str, Any]] = []
    for seed in context.config.training.seeds:
        path = v27c_artifact_path(context.project_root, context.config.artifacts.evaluation_report_template, seed=seed)
        from .protocol import read_sealed_json
        report = read_sealed_json(path, schema=V27C_REPORT_SCHEMA_VERSION)
        if (
            int(report.get("seed", -1)) != seed
            or report.get("manifest_sha256") != context.manifest_sha256
            or report.get("feature_manifest_sha256")
            != load_feature_manifest(context)["payload_sha256"]
        ):
            raise ValueError("V27c seed report does not belong to the active experiment")
        reports.append(report)
    point_deltas = [
        report["parent_final_metric_delta_treatment_minus_control"]["macro_f1"]
        for report in reports
    ]
    bootstrap_mean_deltas = [
        report["paired_bootstrap"]["metric_deltas_treatment_minus_control"]
        ["macro_f1"]["mean"]
        for report in reports
    ]
    drone_to_bird = [
        report["arms"]["motion_treatment"]["parent_metrics"]["confusion_matrix_rows_true_columns_pred"][0][1]
        for report in reports
    ]
    safe = []
    bootstrap_safe = []
    for report in reports:
        matrix = report["arms"]["motion_treatment"]["parent_metrics"]["confusion_matrix_rows_true_columns_pred"]
        macro_f1_lower_95 = report["paired_bootstrap"]["metric_deltas_treatment_minus_control"]["macro_f1"]["lower_95"]
        safe.append(
            sum(matrix[index][index] for index in range(3)) >= 215
            and all(matrix[index][index] >= (109, 54, 52)[index] for index in range(3))
            and report["counterfactuals"]["zero_motion_replays_frozen_control_bitwise"]
            and report["counterfactuals"]["parent_motion_permutation_changes_treatment_output"]
            and report["counterfactuals"]["support_pattern_permutation_movable_parent_count"] > 0
            and report["counterfactuals"]["support_pattern_permutation_changes_treatment_output"]
            and report["counterfactuals"]["component_support_permutation_moved_event_count"] > 0
            and report["counterfactuals"]["component_support_permutation_numeric_changed_event_count"] > 0
            and report["counterfactuals"]["component_support_permutation_zero_pattern_preserved"]
            and report["counterfactuals"]["component_support_permutation_changes_treatment_output"]
        )
        bootstrap_safe.append(float(macro_f1_lower_95) >= -(1.0 / 223.0))
    positive = [
        sum(
            item["motion_pred"] == "DroneTarget"
            for item in report["paired"]["frozen_drone_to_bird_error_audit"]
        ) >= 1
        and value <= 3
        and report["parent_final_metric_delta_treatment_minus_control"]["macro_f1"] > 0.0
        and report["arms"]["motion_treatment"]["parent_metrics"]["per_class"][1]["precision"] > 0.9152542372881356
        for value, report in zip(drone_to_bird, reports, strict=True)
    ]
    numeric_attribution = [
        report["counterfactuals"]["support_pattern_permutation_movable_parent_count"] > 0
        and report["counterfactuals"]["support_pattern_permutation_changes_treatment_output"]
        and report["counterfactuals"]
        ["motion_treatment_minus_support_pattern_permuted_metric_delta"]["macro_f1"]
        > 0.0
        and report["counterfactuals"]
        ["motion_treatment_minus_component_support_permuted_metric_delta"]["macro_f1"]
        > 0.0
        for report in reports
    ]
    credible_positive = [
        positive_value and numeric_value
        for positive_value, numeric_value in zip(
            positive, numeric_attribution, strict=True
        )
    ]
    summary = {
        "schema_version": "RADAR-LN3-V27c-motion-three-seed-summary-v1",
        "experiment_id": context.config.experiment_id,
        "status": "three_seed_known_only_motion_summary",
        "manifest_sha256": context.manifest_sha256,
        "seeds": list(context.config.training.seeds),
        "macro_f1_point_delta_treatment_minus_control": {
            "per_seed": point_deltas,
            "mean": float(np.mean(point_deltas)),
            "min": float(np.min(point_deltas)),
            "max": float(np.max(point_deltas)),
        },
        "macro_f1_paired_bootstrap_mean_delta_treatment_minus_control": {
            "per_seed": bootstrap_mean_deltas,
            "mean": float(np.mean(bootstrap_mean_deltas)),
        },
        "drone_to_bird_errors_per_seed": drone_to_bird,
        "safety_gate_passed_per_seed": safe,
        "bootstrap_macro_f1_lower_95_within_one_parent_per_seed": bootstrap_safe,
        "positive_signal_per_seed": positive,
        "numeric_motion_attribution_per_seed": numeric_attribution,
        "credible_positive_numeric_motion_signal_per_seed": credible_positive,
        "preliminary_verdict": {
            "safety_all_seeds": bool(all(safe)),
            "positive_at_least_two_seeds": int(sum(positive)) >= 2,
            "numeric_motion_attribution_at_least_two_seeds": int(sum(numeric_attribution)) >= 2,
            "credible_positive_numeric_motion_at_least_two_seeds": int(
                sum(credible_positive)
            ) >= 2,
            "all_three_seed_candidate": bool(
                all(safe)
                and all(bootstrap_safe)
                and int(sum(credible_positive)) >= 2
                and float(np.mean(point_deltas)) > 0.0
            ),
            "scope": "known_only_classification; not an open-set, knownness, calibration, or L3 result",
        },
        "test_payload_access_count": 0,
    }
    path = v27c_artifact_path(context.project_root, context.config.artifacts.summary_report_path)
    write_sealed_json_if_identical(path, summary)
    return {**summary, "summary_report_path": str(path)}


__all__ = ["evaluate_v27c_motion", "summarize_v27c_seeds"]
