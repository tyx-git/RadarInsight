"""Knownness ranking evaluation for the frozen-endpoint V27b-r6 ablation.

R6 evaluates an L2 knownness score only.  It does not fit an L3 threshold,
change the frozen V27a classifier, or count Pending/Clutter as Unknown success.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .features import load_feature_cache, load_feature_manifest
from .protocol import (
    R6_ARMS,
    R6_DIAGNOSTIC_LABEL,
    R6_KNOWN_LABELS,
    R6_NEGATIVE_LABEL,
    R6_REPORT_SCHEMA_VERSION,
    R6_ROLES,
    R6Context,
    r6_artifact_path,
    write_sealed_json_if_identical,
)
from .training import _FeatureDataset, _collate, load_r6_checkpoint


def _score_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Summarize a higher-is-known score without a scikit-learn dependency."""

    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("V27b-r6 score summary shapes differ")
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("V27b-r6 score summary values differ")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("V27b-r6 score summary needs Known and Unknown support")
    # Mann-Whitney AUROC with half credit for equal scores.  A stable sort is
    # not sufficient here: its arbitrary within-tie ordering would otherwise
    # make an all-equal score appear better or worse than chance.
    known_scores = scores[labels == 1]
    unknown_scores = scores[labels == 0]
    comparisons = known_scores[:, None] - unknown_scores[None, :]
    auroc = float(
        (np.count_nonzero(comparisons > 0.0) + 0.5 * np.count_nonzero(comparisons == 0.0))
        / float(positives * negatives)
    )
    # Interpolated average precision is tie-safe: every sample sharing a score
    # uses the precision after the whole score group is admitted.
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    true_positive = np.cumsum(ordered_labels == 1)
    group_end = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0.0), ordered_scores.size - 1]
    precision_at_group_end = true_positive[group_end] / (group_end + 1.0)
    positives_at_group_end = true_positive[group_end]
    positives_before_group = np.r_[0, positives_at_group_end[:-1]]
    average_precision = float(
        ((positives_at_group_end - positives_before_group) * precision_at_group_end).sum()
        / positives
    )
    return {
        "known_positive_support": positives,
        "unknown_negative_support": negatives,
        "auroc_higher_is_known": auroc,
        "average_precision_known_positive": average_precision,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
    }


def _threshold_at_known_recall(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    *,
    recall_floor: float,
) -> dict[str, Any]:
    """Choose the least permissive threshold that preserves the recall floor."""

    if known_scores.ndim != 1 or unknown_scores.ndim != 1 or not known_scores.size or not unknown_scores.size:
        raise ValueError("V27b-r6 operating point needs Known and Unknown scores")
    if not (0.0 < float(recall_floor) <= 1.0):
        raise ValueError("V27b-r6 known recall floor is invalid")
    # ``higher_is_known``: threshold at the required lower empirical quantile.
    required = int(math.ceil(float(recall_floor) * known_scores.size))
    threshold = float(np.sort(known_scores)[known_scores.size - required])
    known_accept = known_scores >= threshold
    unknown_accept = unknown_scores >= threshold
    return {
        "known_recall_floor": float(recall_floor),
        "threshold_higher_is_known": threshold,
        "known_acceptance": float(known_accept.mean()),
        "unknown_rejection": float((~unknown_accept).mean()),
        "unknown_acceptance": float(unknown_accept.mean()),
        "known_support": int(known_scores.size),
        "unknown_support": int(unknown_scores.size),
    }


def _per_class_coverage(
    labels: Sequence[str], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels_array = np.asarray([str(value) for value in labels], dtype=str)
    for label in R6_KNOWN_LABELS:
        values = scores[labels_array == label]
        if not values.size:
            raise ValueError("V27b-r6 known class is absent from an evaluation role")
        result[label] = {
            "support": int(values.size),
            "known_acceptance": float((values >= threshold).mean()),
            "score_mean": float(values.mean()),
        }
    return result


def _score_groups(
    labels: Sequence[str],
    fit_eligible: np.ndarray,
    scores: np.ndarray,
    endpoint_correct: np.ndarray | None,
) -> dict[str, Any]:
    labels_array = np.asarray([str(value) for value in labels], dtype=str)
    known_or_unknown = np.isin(labels_array, (*R6_KNOWN_LABELS, R6_NEGATIVE_LABEL))
    groups: dict[str, np.ndarray] = {
        "eligible_known": fit_eligible & np.isin(labels_array, R6_KNOWN_LABELS),
        "eligible_unknown": fit_eligible & (labels_array == R6_NEGATIVE_LABEL),
        # Clutter is diagnostic-only and must never inflate a Pending count.
        "pending_low_quality_or_insufficient_reliable": ~fit_eligible & known_or_unknown,
        "clutter_diagnostic_only": labels_array == R6_DIAGNOSTIC_LABEL,
    }
    if endpoint_correct is not None:
        groups["eligible_known_correct_by_frozen_endpoint"] = groups["eligible_known"] & endpoint_correct
        groups["eligible_known_wrong_by_frozen_endpoint"] = groups["eligible_known"] & ~endpoint_correct
    result: dict[str, Any] = {}
    for name, mask in groups.items():
        values = scores[mask]
        result[name] = {
            "support": int(mask.sum()),
            "score_mean": float(values.mean()) if values.size else None,
            "score_median": float(np.median(values)) if values.size else None,
            "score_min": float(values.min()) if values.size else None,
            "score_max": float(values.max()) if values.size else None,
        }
    return result


def _reference_endpoint_correct(
    context: R6Context,
    *,
    role: str,
    parent_ids: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray | None:
    """Return frozen V27a correctness only as a non-training diagnostic."""

    if role not in {"representation_fit", "calibration"}:
        return None
    from v27.v27a.cache import load_cache_payload
    from v27.v27a.model import V27AEvidenceResidual
    from v27.v27a.protocol import load_v27a_config

    checkpoint_path = context.project_root / context.config.base.v27a_main_checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_v27a_config(context.project_root, context.config.base.v27a_config_path)
    model = V27AEvidenceResidual(config, arm="main")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    cache_path = context.project_root / (
        context.config.base.v27a_train_reference_cache_path
        if role == "representation_fit"
        else context.config.base.v27a_validation_reference_cache_path
    )
    reference = load_cache_payload(cache_path)
    positions = {str(value): index for index, value in enumerate(reference["parent_sample_ids"])}
    results: list[bool] = []
    with torch.inference_mode():
        for parent_id, label in zip(parent_ids, labels, strict=True):
            if label not in R6_KNOWN_LABELS:
                results.append(False)
                continue
            index = positions.get(str(parent_id))
            if index is None:
                raise ValueError("V27b-r6 calibration Known parent is absent from frozen V27a cache")
            length = int(reference["lengths"][index])
            output = model(
                baseline_logits=torch.from_numpy(reference["event_baseline_logits"][index:index + 1]),
                summary_tokens=torch.from_numpy(reference["event_summary_tokens"][index:index + 1]),
                summary_valid=torch.from_numpy(reference["event_summary_valid"][index:index + 1]),
                frame_values=torch.from_numpy(reference["event_frame_values"][index:index + 1]),
                frame_mask=torch.from_numpy(reference["event_frame_mask"][index:index + 1]),
                padding_mask=torch.from_numpy(reference["padding_mask"][index:index + 1]),
            )
            prediction = int(output["sequence_logits"][0, length - 1].argmax().item())
            truth = int(reference["labels"][index])
            results.append(prediction == truth)
    return np.asarray(results, dtype=np.bool_)


def _infer_scores(
    context: R6Context,
    *,
    arm: str,
    seed: int,
    role: str,
) -> dict[str, Any]:
    if arm not in R6_ARMS or role not in R6_ROLES:
        raise ValueError("V27b-r6 evaluation arm or role is invalid")
    device = torch.device("cpu")
    model, checkpoint = load_r6_checkpoint(context, arm=arm, seed=seed, device=device)
    cache = load_feature_cache(context, role=role)
    dataset = _FeatureDataset(cache)
    loader = DataLoader(
        dataset,
        batch_size=context.config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    parent_ids: list[str] = []
    labels: list[str] = []
    scores: list[float] = []
    fit_eligible: list[bool] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            logits = model(
                event_trajectory=batch["event_trajectory"],
                summary_features=batch["summary_features"],
                task_facts_sequence=batch["task_facts_sequence"],
                stable_features=batch["stable_features"],
                padding_mask=batch["padding_mask"],
            )
            parent_ids.extend(batch["parent_ids"])
            labels.extend(batch["label_names"])
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            fit_eligible.extend(batch["fit_eligible"].to(dtype=torch.bool).tolist())
    if parent_ids != [str(value) for value in cache["parent_ids"]]:
        raise ValueError("V27b-r6 evaluation changed cache parent order")
    return {
        "checkpoint": checkpoint,
        "cache": cache,
        "parent_ids": parent_ids,
        "labels": labels,
        "scores": np.asarray(scores, dtype=np.float64),
        "fit_eligible": np.asarray(fit_eligible, dtype=np.bool_),
    }


def _pending_population(labels: np.ndarray, fit_eligible: np.ndarray) -> int:
    """Count only Known/Unknown parents that become Pending by quality facts."""

    relevant = np.isin(labels, (*R6_KNOWN_LABELS, R6_NEGATIVE_LABEL))
    return int((~fit_eligible & relevant).sum())


def _arm_report_from_inferred(
    context: R6Context,
    *,
    arm: str,
    seed: int,
    role: str,
    inferred: Mapping[str, Any],
) -> dict[str, Any]:
    labels = list(inferred["labels"])
    scores = np.asarray(inferred["scores"], dtype=np.float64)
    fit_eligible = np.asarray(inferred["fit_eligible"], dtype=np.bool_)
    label_array = np.asarray(labels, dtype=str)
    known_mask = fit_eligible & np.isin(label_array, R6_KNOWN_LABELS)
    unknown_mask = fit_eligible & (label_array == R6_NEGATIVE_LABEL)
    summary = _score_summary(
        np.concatenate((np.ones(int(known_mask.sum()), dtype=np.int64), np.zeros(int(unknown_mask.sum()), dtype=np.int64))),
        np.concatenate((scores[known_mask], scores[unknown_mask])),
    )
    operating_points = [
        _threshold_at_known_recall(
            scores[known_mask], scores[unknown_mask], recall_floor=recall_floor
        )
        for recall_floor in context.config.evaluation.known_recall_floors
    ]
    for operating in operating_points:
        operating["per_known_class_coverage"] = _per_class_coverage(
            labels, scores, float(operating["threshold_higher_is_known"])
        )
    endpoint_correct = _reference_endpoint_correct(
        context,
        role=role,
        parent_ids=inferred["parent_ids"],
        labels=labels,
    )
    return {
        "schema_version": R6_REPORT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "knownness_ranking_diagnostic",
        "arm": arm,
        "seed": int(seed),
        "role": role,
        "r6_manifest_sha256": context.manifest_sha256,
        "feature_manifest_sha256": load_feature_manifest(context)["feature_manifest_sha256"],
        "feature_cache_sha256": inferred["cache"]["cache_sha256"],
        "checkpoint_state_sha256": inferred["checkpoint"]["model_state_sha256"],
        "frozen_v27a_main_checkpoint_sha256": context.config.base.expected_v27a_main_checkpoint_sha256,
        "classifier_logits_input_to_knownness": False,
        "classifier_logits_must_remain_bitwise_unchanged": True,
        "unknown_is_classifier_class": False,
        "knownness_score_direction": "higher_is_known",
        "fit_population": {
            "eligible_known": int(known_mask.sum()),
            "eligible_unknown": int(unknown_mask.sum()),
            "pending_low_quality_or_insufficient_reliable": _pending_population(label_array, fit_eligible),
            "clutter_diagnostic_only": int((label_array == R6_DIAGNOSTIC_LABEL).sum()),
        },
        "ranking": summary,
        "operating_points": operating_points,
        "score_groups": _score_groups(labels, fit_eligible, scores, endpoint_correct),
        "classification_endpoint_diagnostic": {
            "available": endpoint_correct is not None,
            "source": (
                "frozen_v27a_main_epoch_20_archived_train_cache"
                if role == "representation_fit"
                else "frozen_v27a_main_epoch_20_archived_validation_cache"
                if role == "calibration"
                else "not_available_for_disjoint_representation_development"
            ),
            "used_as_knownness_input": False,
            "used_for_training": False,
        },
        "scope": {
            "calibration": "not_fitted",
            "l3": "not_fitted",
            "fixed_softmax_threshold": "not_used",
            "single_distance_gate": "not_used",
            "pending_counted_as_unknown_rejection": False,
            "clutter_counted_as_unknown_rejection": False,
            "test_payload_access_count": 0,
        },
    }


def evaluate_r6_arm(
    context: R6Context,
    *,
    arm: str,
    seed: int,
    role: str,
) -> dict[str, Any]:
    """Evaluate one frozen-knownness head without fitting a decision boundary."""

    inferred = _infer_scores(context, arm=arm, seed=seed, role=role)
    return _arm_report_from_inferred(
        context, arm=arm, seed=seed, role=role, inferred=inferred
    )


def _paired_bootstrap(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if int(replicates) < 100:
        raise ValueError("V27b-r6 paired bootstrap requires at least 100 replicates")
    control_scores = np.asarray(control["_scores"], dtype=np.float64)
    treatment_scores = np.asarray(treatment["_scores"], dtype=np.float64)
    labels = np.asarray(control["_binary_labels"], dtype=np.int64)
    if control_scores.shape != treatment_scores.shape or labels.shape != control_scores.shape:
        raise ValueError("V27b-r6 paired score arrays differ")
    known_indices = np.flatnonzero(labels == 1)
    unknown_indices = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(int(seed))
    deltas: dict[str, list[float]] = {"auroc": []}
    for recall_floor in control["_recall_floors"]:
        deltas[f"unknown_rejection_at_known_recall_{recall_floor:.2f}"] = []
    for _ in range(int(replicates)):
        sampled = np.concatenate(
            (
                rng.choice(known_indices, size=known_indices.size, replace=True),
                rng.choice(unknown_indices, size=unknown_indices.size, replace=True),
            )
        )
        sampled_labels = labels[sampled]
        control_summary = _score_summary(sampled_labels, control_scores[sampled])
        treatment_summary = _score_summary(sampled_labels, treatment_scores[sampled])
        deltas["auroc"].append(
            treatment_summary["auroc_higher_is_known"] - control_summary["auroc_higher_is_known"]
        )
        known = sampled_labels == 1
        unknown = ~known
        for recall_floor in control["_recall_floors"]:
            key = f"unknown_rejection_at_known_recall_{recall_floor:.2f}"
            control_point = _threshold_at_known_recall(
                control_scores[sampled][known], control_scores[sampled][unknown], recall_floor=recall_floor
            )
            treatment_point = _threshold_at_known_recall(
                treatment_scores[sampled][known], treatment_scores[sampled][unknown], recall_floor=recall_floor
            )
            deltas[key].append(
                treatment_point["unknown_rejection"] - control_point["unknown_rejection"]
            )
    return {
        "method": "stratified_paired_known_unknown_bootstrap",
        "seed": int(seed),
        "replicates": int(replicates),
        "treatment_minus_control": {
            name: {
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for name, values in deltas.items()
        },
    }


def compare_r6_arms(
    context: R6Context,
    *,
    seed: int,
    role: str,
) -> dict[str, Any]:
    """Create one paired control/treatment report for a fixed head seed."""

    control_raw = _infer_scores(context, arm="control", seed=seed, role=role)
    treatment_raw = _infer_scores(context, arm="stable8_treatment", seed=seed, role=role)
    control = _arm_report_from_inferred(
        context, arm="control", seed=seed, role=role, inferred=control_raw
    )
    treatment = _arm_report_from_inferred(
        context,
        arm="stable8_treatment",
        seed=seed,
        role=role,
        inferred=treatment_raw,
    )
    if control_raw["parent_ids"] != treatment_raw["parent_ids"] or control_raw["labels"] != treatment_raw["labels"]:
        raise ValueError("V27b-r6 paired arms have different evaluation populations")
    labels = np.asarray(control_raw["labels"], dtype=str)
    eligible = control_raw["fit_eligible"]
    if not np.array_equal(eligible, treatment_raw["fit_eligible"]):
        raise ValueError("V27b-r6 paired arms changed fit eligibility")
    selected = eligible & np.isin(labels, (*R6_KNOWN_LABELS, R6_NEGATIVE_LABEL))
    binary_labels = np.isin(labels[selected], R6_KNOWN_LABELS).astype(np.int64)
    report = {
        "schema_version": R6_REPORT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "paired_knownness_ranking_diagnostic",
        "seed": int(seed),
        "role": role,
        "r6_manifest_sha256": context.manifest_sha256,
        "feature_manifest_sha256": load_feature_manifest(context)["feature_manifest_sha256"],
        "arms": {"control": control, "stable8_treatment": treatment},
        "paired": _paired_bootstrap(
            {
                "_scores": control_raw["scores"][selected],
                "_binary_labels": binary_labels,
                "_recall_floors": context.config.evaluation.known_recall_floors,
            },
            {
                "_scores": treatment_raw["scores"][selected],
                "_binary_labels": binary_labels,
            },
            seed=27_060 + int(seed),
            replicates=context.config.evaluation.paired_bootstrap_replicates,
        ),
        "scope": {
            "classification_endpoint": "frozen_v27a_main_epoch_20",
            "calibration": "not_fitted",
            "l3": "not_fitted",
            "test_payload_access_count": 0,
        },
    }
    return report


def write_r6_arm_report(
    context: R6Context, report: Mapping[str, Any], *, arm: str, seed: int, role: str
) -> Any:
    path = r6_artifact_path(
        context.project_root,
        context.config.artifacts.evaluation_report_template,
        arm=arm,
        seed=seed,
        role=role,
    )
    write_sealed_json_if_identical(path, dict(report))
    return path


def write_r6_comparison_report(
    context: R6Context, report: Mapping[str, Any], *, seed: int, role: str
) -> Any:
    path = r6_artifact_path(
        context.project_root,
        context.config.artifacts.comparison_report_template,
        seed=seed,
        role=role,
    )
    write_sealed_json_if_identical(path, dict(report))
    return path


__all__ = [
    "compare_r6_arms",
    "evaluate_r6_arm",
    "write_r6_arm_report",
    "write_r6_comparison_report",
]
