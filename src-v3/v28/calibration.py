"""Independent V28 calibration on the frozen calibration role."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file

from .head_training import _load_head_payload
from .heads import V28TrajectoryHeads, deterministic_quality_target
from .representations import load_representation, representation_path
from .protocol import V28Context, V28_L1_TASK, V28_L2_TASK, V28_L2_CLASSES, V28_ROLES, artifact_path, checkpoint_manifest_compatibility, resolve_project_path


def _calibrated_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits / float(temperature)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _temperature(logits: np.ndarray, labels: np.ndarray, context: V28Context) -> dict[str, Any]:
    if not len(labels):
        return {"status": "unavailable", "reason": "no known calibration labels"}
    lo = float(context.config.calibration["temperature_min"])
    hi = float(context.config.calibration["temperature_max"])
    count = int(context.config.calibration["temperature_grid_size"])
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), count))
    losses: list[tuple[float, float]] = []
    for temperature in grid:
        probabilities = _calibrated_probabilities(logits, float(temperature))
        losses.append((float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0)).mean()), float(temperature)))
    loss, selected = min(losses)
    return {"status": "fitted", "temperature": selected, "nll": loss, "grid_size": count, "min": lo, "max": hi, "support": int(len(labels))}


def _conformal(probabilities: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...], alpha: float) -> dict[str, Any]:
    quantiles: dict[str, float] = {}
    floors: dict[str, float] = {}
    supports: dict[str, int] = {}
    coverages: dict[str, float] = {}
    for class_id, class_name in enumerate(class_names):
        selected = probabilities[labels == class_id, class_id]
        supports[class_name] = int(len(selected))
        if len(selected) == 0:
            quantiles[class_name] = 1.0; floors[class_name] = 0.0; coverages[class_name] = 0.0
            continue
        nonconformity = 1.0 - selected
        rank = min(len(nonconformity), max(1, int(math.ceil((len(nonconformity) + 1) * (1.0 - alpha)))))
        quantile = float(np.sort(nonconformity)[rank - 1])
        quantiles[class_name] = quantile
        floors[class_name] = 1.0 - quantile
        coverages[class_name] = float((nonconformity <= quantile + 1.0e-12).mean())
    return {"method": "class_conditional_split_conformal_1_minus_calibrated_probability_higher", "error_rate": float(alpha), "quantile_by_class": quantiles, "candidate_probability_floor_by_class": floors, "support_by_class": supports, "coverage_by_class": coverages, "support": int(len(labels))}


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    values = sorted(set(float(value) for value in np.asarray(scores, dtype=np.float64).tolist()))
    candidates = {0.0, 1.0, *values}
    candidates.update(float(np.nextafter(value, 1.0)) for value in values)
    return sorted(value for value in candidates if 0.0 <= value <= 1.0)


def _knownness_scores(payload: Mapping[str, Any], head_payload: Mapping[str, Any], context: V28Context, *, task: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    heads = V28TrajectoryHeads(context, task=task)
    heads.load_state_dict(head_payload["state_dict"], strict=True)
    tensors = payload["tensors"]
    with torch.inference_mode():
        output = heads.forward_features(tensors["knownness_features"].float(), tensors["quality_features"].float(), tensors["padding_mask"].bool())
    lengths = tensors["lengths"].long()
    rows = torch.arange(lengths.shape[0])
    known = torch.sigmoid(output["knownness_logits"])[rows, lengths - 1].numpy()
    quality = torch.sigmoid(output["quality_logits"])[rows, lengths - 1].numpy()
    return known, [str(value) for value in payload["label_names"]], quality


def _knownness_threshold(scores: np.ndarray, labels: list[str], context: V28Context, *, task: str) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    names = np.asarray(labels, dtype=object)
    # Target recall and Clutter recall are separate safety constraints.  L1
    # Clutter remains a known positive for head training, but its calibration
    # floor is checked independently below, matching the head selector.
    positive = np.isin(names, list(V28_L2_CLASSES))
    unknown = names == "UnknownTarget"
    clutter = names == "ClutterTarget"
    floor = float(context.config.calibration["known_target_recall_floor"])
    clutter_floor = float(context.config.calibration["clutter_recall_floor"])
    max_reject = int(context.config.selection["max_correct_known_new_rejections"])
    minimum_negative_support = int(context.config.selection["minimum_negative_support"])
    support_sufficient = int(unknown.sum()) >= minimum_negative_support
    candidates = []
    for threshold in _threshold_candidates(values):
        accepted = values >= float(threshold)
        target_recall = float(accepted[positive].mean()) if positive.any() else 0.0
        clutter_recall = float(accepted[clutter].mean()) if clutter.any() else None
        unknown_rejection = float((~accepted[unknown]).mean()) if unknown.any() else None
        new_rejections = int((positive & ~accepted).sum())
        clutter_safe = not clutter.any() or bool(clutter_recall >= clutter_floor)
        candidates.append({"threshold": float(threshold), "known_target_recall": target_recall, "clutter_recall": clutter_recall, "unknown_rejection": unknown_rejection, "correct_known_new_rejections": new_rejections, "unknown_support": int(unknown.sum()), "minimum_negative_support": minimum_negative_support, "support_sufficient": support_sufficient, "safe": bool(positive.any() and target_recall >= floor and clutter_safe and new_rejections <= max_reject and support_sufficient)})
    safe = [row for row in candidates if row["safe"]]
    if not safe:
        return {"status": "unavailable", "reason": "knownness safety constraints infeasible", "candidate_count": len(candidates), "support_sufficient": support_sufficient}
    selected = max(safe, key=lambda row: (row["unknown_rejection"] if row["unknown_rejection"] is not None else -1.0, row["threshold"]))
    return {"status": "fitted", "direction": "higher_is_known", "threshold_space": "sigmoid_probability", "selected": selected, "candidate_count": len(candidates), "positive_support": int(positive.sum()), "negative_support": int(unknown.sum()), "clutter_support": int(clutter.sum()), "known_target_recall_floor": floor, "clutter_recall_floor": clutter_floor, "max_correct_known_new_rejections": max_reject}


def _quality_threshold(scores: np.ndarray, targets: np.ndarray, context: V28Context) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=bool)
    floor = float(context.config.selection["quality_recall_floor"])
    candidates = []
    for threshold in _threshold_candidates(values):
        accepted = values >= float(threshold)
        recall = float(accepted[labels].mean()) if labels.any() else 0.0
        rejection = float((~accepted[~labels]).mean()) if (~labels).any() else None
        candidates.append({"threshold": float(threshold), "positive_recall": recall, "negative_rejection": rejection, "safe": bool(labels.any() and recall >= floor)})
    safe = [row for row in candidates if row["safe"]]
    if not safe:
        return {"status": "unavailable", "reason": "quality recall floor infeasible", "candidate_count": len(candidates)}
    selected = max(safe, key=lambda row: (row["negative_rejection"] if row["negative_rejection"] is not None else -1.0, row["threshold"]))
    return {"status": "fitted", "direction": "higher_is_eligible", "selected": selected, "candidate_count": len(candidates), "positive_support": int(labels.sum()), "negative_support": int((~labels).sum()), "quality_recall_floor": floor}


def fit_calibration(context: V28Context, *, endpoint_binding: Mapping[str, Any], head_bindings: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    calibration_unknown_parents: set[str] = set()
    for task in (V28_L1_TASK, V28_L2_TASK):
        representation = load_representation(context, task=task, role="calibration")
        representation_path_value = representation_path(context, task=task, role="calibration")
        if dict(representation.get("endpoint_binding") or {}) != dict(endpoint_binding[task]):
            raise ValueError(f"V28 calibration endpoint binding differs: {task}")
        head_path = resolve_project_path(context.project_root, head_bindings[task]["path"], must_exist=True)
        head_payload = _load_head_payload(head_path)
        if head_payload.get("task") != task or head_payload.get("seed") != int(head_bindings[task]["seed"]):
            raise ValueError(f"V28 calibration head binding differs: {task}")
        if head_payload.get("system_manifest_sha256") != context.manifest_sha256:
            checkpoint_manifest_compatibility(
                context.project_root,
                {**context.manifest, "_manifest_path": context.manifest_path.relative_to(context.project_root).as_posix()},
                str(head_payload.get("system_manifest_sha256")),
                task=task,
            )
        expected_contract = V28TrajectoryHeads(context, task=task).model_contract()
        if head_payload.get("model_contract") != expected_contract or head_payload.get("model_contract_sha256") != canonical_sha256(expected_contract):
            raise ValueError(f"V28 calibration head contract differs: {task}")
        known_scores, labels, quality_scores = _knownness_scores(representation, head_payload, context, task=task)
        tensors = representation["tensors"]
        lengths = tensors["lengths"].long()
        rows = torch.arange(lengths.shape[0])
        logits = tensors["sequence_logits"][rows, lengths - 1].numpy()
        quality_features = tensors["quality_features"].float()
        quality_targets_sequence = deterministic_quality_target(context, task=task, task_facts=quality_features[..., 38:], event_facts=quality_features[..., 19:38], padding_mask=tensors["padding_mask"].bool())
        quality_targets = quality_targets_sequence[rows, lengths - 1].numpy().astype(bool)
        known_label_ids = []
        known_indices = []
        class_names = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE") if task == V28_L1_TASK else V28_L2_CLASSES
        for index, label in enumerate(labels):
            if task == V28_L1_TASK and label in {"DroneTarget", "BirdTarget", "BalloonTarget"}:
                known_indices.append(index); known_label_ids.append(0)
            elif task == V28_L1_TASK and label == "ClutterTarget":
                known_indices.append(index); known_label_ids.append(1)
            elif task == V28_L2_TASK and label in V28_L2_CLASSES:
                known_indices.append(index); known_label_ids.append(class_names.index(label))
        known_indices_array = np.asarray(known_indices, dtype=np.int64)
        temperature = _temperature(logits[known_indices_array], np.asarray(known_label_ids, dtype=np.int64), context)
        probabilities = _calibrated_probabilities(logits[known_indices_array], float(temperature.get("temperature", 1.0))) if len(known_indices_array) else np.zeros((0, len(class_names)))
        conformal = _conformal(probabilities, np.asarray(known_label_ids, dtype=np.int64), class_names, float(context.config.calibration["conformal_error_rate"])) if len(known_indices_array) else {"support": 0}
        knownness = _knownness_threshold(known_scores, labels, context, task=task)
        quality_threshold = _quality_threshold(quality_scores, quality_targets, context)
        calibration_unknown_parents.update(parent_id for parent_id, label in zip(representation["parent_ids"], labels, strict=True) if label == "UnknownTarget")
        scorer = {"status": "implemented_diagnostic_only" if task == V28_L2_TASK and conformal.get("support", 0) else "unavailable", "method": "class_conditional_probability_nonconformity", "score": "1 - calibrated_probability", "reference_quantile_by_class": conformal.get("quantile_by_class", {}), "candidate_rule": "score <= reference_quantile"} if task == V28_L2_TASK else None
        results[task] = {"class_names": list(class_names), "representation_calibration": {"path": relative_project_path(context.project_root, representation_path_value), "file_sha256": sha256_file(representation_path_value), "cache_sha256": representation["cache_sha256"], "parent_ids_sha256": representation["parent_ids_sha256"]}, "temperature": temperature, "conformal": conformal, "knownness": knownness, "quality": {"status": "diagnostic_only", "score_direction": "higher_is_eligible", "quality_score_support": int(len(quality_scores)), "deterministic_gate": {"minimum_events": int(context.config.heads["quality_min_events"]), "minimum_rd_fraction": float(context.config.heads["quality_min_rd_fraction"]), "minimum_dt_fraction": float(context.config.heads["quality_min_dt_fraction"]), "minimum_aux_fraction": float(context.config.heads["quality_min_aux_fraction"])}, "learned_threshold": quality_threshold}, "scorer": scorer, "support": {"parent_count": len(labels), "known_parent_count": len(known_indices), "unknown_parent_count": sum(label == "UnknownTarget" for label in labels), "clutter_parent_count": sum(label == "ClutterTarget" for label in labels)}}
    unknown_support = len(calibration_unknown_parents)
    minimum_unknown = int(context.config.calibration["minimum_unknown_support_for_formal"])
    audit_required = bool(context.config.calibration["post_selection_audit_required"])
    reasons: list[str] = []
    if unknown_support < minimum_unknown:
        reasons.append("unknown_support_below_formal_minimum")
    if audit_required:
        reasons.append("post_selection_audit_pool_unavailable")
    if any(results[task]["temperature"].get("status") != "fitted" or results[task]["knownness"].get("status") != "fitted" for task in (V28_L1_TASK, V28_L2_TASK)):
        reasons.append("one_or_more_calibration_components_unavailable")
    status = "fitted" if not reasons else "internal_exploratory"
    return {"schema_version": "RADAR-LN3-V28-calibration-v2", "experiment_id": "V28", "status": status, "deployment_eligible": status == "fitted", "status_reasons": reasons, "system_manifest_sha256": context.manifest_sha256, "endpoint_binding": dict(endpoint_binding), "head_bindings": dict(head_bindings), "unknown_support": unknown_support, "unknown_parent_ids_sha256": canonical_sha256(sorted(calibration_unknown_parents)), "minimum_unknown_support": minimum_unknown, "post_selection_audit": {"required": audit_required, "available": False}, "calibration_role": "validation_split_only", "tasks": results, "test_payload_access_count": 0}


__all__ = ["fit_calibration"]
