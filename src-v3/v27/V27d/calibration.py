"""Independent calibration-pool fitting for V27d dynamic L3."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical

from .features import load_feature_cache
from .heads import V27dHeads
from .l3 import L3_REGISTRY_SCHEMA
from .protocol import V27DContext, V27D_KNOWN_LABELS, V27D_VARIANTS, artifact_path, project_path, read_sealed_json, write_sealed_json
from .scorers import (
    KNOWN_CLASS_NAMES,
    ScorerConfig,
    classifier_conformal_p_values,
    classifier_nonconformity,
    conformal_nonconformity,
    conformal_p_values,
    fit_scorer,
    scorer_digest,
)
from .training import load_head_checkpoint


CALIBRATION_ARTIFACT_SCHEMA = "RADAR-LN3-V27d-calibration-artifact-v1"


def _device(context: V27DContext) -> torch.device:
    if context.config.training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d calibration requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _head_output(model: V27dHeads, payload: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    indices = torch.arange(len(payload["parent_ids"]))
    inputs = {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "event_facts_sequence": tensors["event_facts_sequence"][indices].to(device),
        "raw_scale_features": tensors["raw_scale_features"][indices].to(device),
        "stable8_causal_features": tensors["stable8_causal_features"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        output = model(**inputs)
    return {key: value.detach().cpu() for key, value in output.items()}


def _load_variant_cache(context: V27DContext, *, variant: str, role: str, task: str) -> dict[str, Any]:
    path = artifact_path(context.project_root, context.config.artifacts.feature_cache_template, variant=variant, role=role, task=task)
    return load_feature_cache(context, path)


def _candidate_thresholds(values: torch.Tensor, *, lower: float | None = None, upper: float | None = None) -> list[float]:
    if values.ndim != 1 or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("V27d calibration threshold values must be finite [N] floats")
    if lower is not None and not math.isfinite(float(lower)):
        raise ValueError("V27d calibration threshold lower bound must be finite")
    if upper is not None and not math.isfinite(float(upper)):
        raise ValueError("V27d calibration threshold upper bound must be finite")
    if lower is not None and upper is not None and float(lower) > float(upper):
        raise ValueError("V27d calibration threshold bounds are reversed")
    if values.numel() == 0:
        unique: list[float] = []
    else:
        unique = sorted(set(float(value) for value in values.tolist()))
    candidates = [float("-inf")]
    if lower is not None:
        candidates.append(float(lower))
    candidates.extend(unique)
    if unique:
        candidates.append(math.nextafter(unique[-1], math.inf))
    if lower is not None:
        candidates = [value for value in candidates if value >= lower]
    if upper is not None:
        candidates = [value for value in candidates if value <= upper or value == float("-inf")]
    return sorted(set(candidates))


def _safe_l1_thresholds(
    scores: torch.Tensor,
    p_values: torch.Tensor,
    labels: list[str],
    quality: torch.Tensor,
    endpoint_prediction: torch.Tensor,
    endpoint_correct: torch.Tensor,
    context: V27DContext,
) -> tuple[float, float, dict[str, Any]]:
    if (
        scores.ndim != 1
        or scores.shape[0] != len(labels)
        or quality.ndim != 1
        or quality.shape[0] != len(labels)
        or endpoint_prediction.ndim != 1
        or endpoint_prediction.shape[0] != len(labels)
        or endpoint_correct.ndim != 1
        or endpoint_correct.shape[0] != len(labels)
        or quality.dtype != torch.bool
        or endpoint_prediction.dtype != torch.long
        or endpoint_correct.dtype != torch.bool
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("V27d L1 calibration inputs differ")
    if bool((endpoint_prediction < 0).any()) or bool((endpoint_prediction >= 2).any()):
        raise ValueError("V27d L1 endpoint predictions are out of bounds")
    known = torch.tensor([label in V27D_KNOWN_LABELS for label in labels], dtype=torch.bool)
    clutter = torch.tensor([label == "ClutterTarget" for label in labels], dtype=torch.bool)
    unknown = torch.tensor([label == "UnknownTarget" for label in labels], dtype=torch.bool)
    known_eligible = known & quality
    clutter_eligible = clutter & quality
    unknown_eligible = unknown & quality
    threshold_population = known_eligible | clutter_eligible
    if p_values.ndim != 2 or p_values.shape[0] != len(labels) or p_values.shape[1] != 2:
        raise ValueError("V27d L1 classifier conformal values differ")
    if not torch.isfinite(p_values).all() or bool((p_values < 0).any()) or bool((p_values > 1).any()):
        raise ValueError("V27d L1 classifier conformal values are invalid")
    class_masks = {
        class_name: known_eligible & torch.tensor([label == class_name for label in labels], dtype=torch.bool)
        for class_name in V27D_KNOWN_LABELS
    }
    if any(not bool(mask.any()) for mask in class_masks.values()):
        raise ValueError("V27d L1 calibration lacks a known-class quality support group")
    if not bool(clutter_eligible.any()):
        raise ValueError("V27d L1 calibration lacks a quality-eligible Clutter support group")
    if not bool(unknown_eligible.any()):
        raise ValueError("V27d L1 calibration lacks a quality-eligible Unknown support group")
    score_candidates = _candidate_thresholds(scores[threshold_population])
    p_candidates = _candidate_thresholds(
        p_values[threshold_population].reshape(-1),
        lower=context.config.calibration.conformal_error_rate,
        upper=1.0,
    )
    best: tuple[tuple[float, float, float, float], float, float, dict[str, Any]] | None = None
    for threshold in score_candidates:
        for p_threshold in p_candidates:
            candidate_set = p_values.ge(p_threshold)
            singleton = candidate_set.sum(dim=1).eq(1)
            accept_target = candidate_set[:, 0] & singleton & endpoint_prediction.eq(0) & scores.ge(threshold) & quality
            accept_clutter = candidate_set[:, 1] & singleton & endpoint_prediction.eq(1) & scores.ge(threshold) & quality
            known_recall = float(accept_target[known_eligible].float().mean().item()) if bool(known_eligible.any()) else 0.0
            clutter_recall = float(accept_clutter[clutter_eligible].float().mean().item()) if bool(clutter_eligible.any()) else 0.0
            known_recall_by_class = {
                class_name: float(
                    accept_target[known_eligible & torch.tensor([label == class_name for label in labels], dtype=torch.bool)]
                    .float()
                    .mean()
                    .item()
                )
                if bool((known_eligible & torch.tensor([label == class_name for label in labels], dtype=torch.bool)).any())
                else 0.0
                for class_name in V27D_KNOWN_LABELS
            }
            accepted = accept_target | accept_clutter
            unknown_rejection = float((~accepted[unknown_eligible]).float().mean().item()) if bool(unknown_eligible.any()) else 0.0
            correct_known_loss = int((endpoint_correct & known_eligible & ~accepted).sum().item())
            correct_known_loss_by_class = {
                class_name: int((endpoint_correct & class_masks[class_name] & ~accepted).sum().item())
                for class_name in V27D_KNOWN_LABELS
            }
            correct_clutter_loss = int((endpoint_correct & clutter_eligible & ~accepted).sum().item())
            feasible = (
                min(known_recall_by_class.values()) >= context.config.calibration.known_target_recall_floor
                and clutter_recall >= context.config.calibration.clutter_recall_floor
                and correct_known_loss <= context.config.calibration.max_correct_known_new_rejections
                and max(correct_known_loss_by_class.values()) <= context.config.calibration.max_per_class_correct_known_loss
            )
            if feasible:
                ranking = (unknown_rejection, min(known_recall_by_class.values()), float(threshold), float(p_threshold))
                if best is None or ranking > best[0]:
                    best = (
                        ranking,
                        threshold,
                        p_threshold,
                        {
                            "known_recall": known_recall,
                            "known_recall_by_class": known_recall_by_class,
                            "clutter_recall": clutter_recall,
                            "unknown_rejection": unknown_rejection,
                            "correct_known_new_rejections": correct_known_loss,
                            "correct_known_new_rejections_by_class": correct_known_loss_by_class,
                            "correct_clutter_new_rejections": correct_clutter_loss,
                            "unknown_support": int(unknown_eligible.sum().item()),
                            "conformal_p_threshold": p_threshold,
                            "feasible": True,
                        },
                    )
    if best is None:
        return float("inf"), float("inf"), {
            "feasible": False,
            "known_recall": 0.0,
            "clutter_recall": 0.0,
            "unknown_rejection": 0.0,
        }
    return best[1], best[2], best[3]


def _safe_l2_thresholds(
    *,
    knownness: torch.Tensor,
    p_values: torch.Tensor,
    labels: list[str],
    quality: torch.Tensor,
    endpoint_prediction: torch.Tensor,
    endpoint_correct: torch.Tensor,
    context: V27DContext,
) -> tuple[float, float, dict[str, Any]]:
    if (
        knownness.ndim != 1
        or knownness.shape[0] != len(labels)
        or quality.ndim != 1
        or quality.shape[0] != len(labels)
        or endpoint_prediction.ndim != 1
        or endpoint_prediction.shape[0] != len(labels)
        or endpoint_correct.ndim != 1
        or endpoint_correct.shape[0] != len(labels)
        or quality.dtype != torch.bool
        or endpoint_prediction.dtype != torch.long
        or endpoint_correct.dtype != torch.bool
        or not torch.isfinite(knownness).all()
    ):
        raise ValueError("V27d L2 calibration inputs differ")
    known = torch.tensor([label in V27D_KNOWN_LABELS for label in labels], dtype=torch.bool)
    unknown = torch.tensor([label == "UnknownTarget" for label in labels], dtype=torch.bool)
    eligible = known & quality
    unknown_eligible = unknown & quality
    if p_values.ndim != 2 or p_values.shape[0] != len(labels) or p_values.shape[1] != len(KNOWN_CLASS_NAMES):
        raise ValueError("V27d L2 conformal values differ")
    if not torch.isfinite(p_values).all() or bool((p_values < 0).any()) or bool((p_values > 1).any()):
        raise ValueError("V27d L2 conformal values are invalid")
    class_masks = {
        class_name: eligible & torch.tensor([label == class_name for label in labels], dtype=torch.bool)
        for class_name in V27D_KNOWN_LABELS
    }
    if any(not bool(mask.any()) for mask in class_masks.values()):
        raise ValueError("V27d L2 calibration lacks a known-class quality support group")
    if not bool(unknown_eligible.any()):
        raise ValueError("V27d L2 calibration lacks a quality-eligible Unknown support group")
    score_candidates = _candidate_thresholds(knownness[eligible])
    p_candidates = _candidate_thresholds(
        p_values[eligible].reshape(-1),
        lower=context.config.calibration.conformal_error_rate,
        upper=1.0,
    )
    best: tuple[tuple[float, float, float, float], float, float, dict[str, Any]] | None = None
    for score_threshold in score_candidates:
        for p_threshold in p_candidates:
            candidate_set = p_values.ge(p_threshold)
            predicted_in_set = candidate_set.gather(1, endpoint_prediction.unsqueeze(-1)).squeeze(-1)
            singleton_set = candidate_set.sum(dim=1).eq(1)
            accepted = knownness.ge(score_threshold) & predicted_in_set & singleton_set & quality
            recalls: dict[str, float] = {}
            safe = True
            correct_known_loss_by_class: dict[str, int] = {}
            for class_name in V27D_KNOWN_LABELS:
                class_mask = class_masks[class_name]
                recall = float(accepted[class_mask].float().mean().item()) if bool(class_mask.any()) else 0.0
                recalls[class_name] = recall
                if recall < context.config.calibration.known_target_recall_floor:
                    safe = False
                correct_mask = class_mask & endpoint_correct
                loss = int((correct_mask & ~accepted).sum().item())
                correct_known_loss_by_class[class_name] = loss
                if loss > context.config.calibration.max_per_class_correct_known_loss:
                    safe = False
            correct_known_loss = int((eligible & endpoint_correct & ~accepted).sum().item())
            if correct_known_loss > context.config.calibration.max_correct_known_new_rejections:
                safe = False
            unknown_rejection = float((~accepted[unknown_eligible]).float().mean().item()) if bool(unknown_eligible.any()) else 0.0
            if not safe:
                continue
            ranking = (unknown_rejection, min(recalls.values()), float(score_threshold), float(p_threshold))
            if best is None or ranking > best[0]:
                best = (ranking, score_threshold, p_threshold, {"feasible": True, "known_recall_by_class": recalls, "correct_known_new_rejections": correct_known_loss, "correct_known_new_rejections_by_class": correct_known_loss_by_class, "unknown_rejection": unknown_rejection, "unknown_support": int(unknown_eligible.sum().item())})
    if best is None:
        return float("inf"), float("inf"), {
            "feasible": False,
            "unknown_support": int((unknown & quality).sum().item()),
        }
    return best[1], best[2], best[3]


def fit_dynamic_calibration(context: V27DContext, *, variant: str, seed: int) -> dict[str, Any]:
    if variant not in V27D_VARIANTS:
        raise ValueError("V27d calibration variant is invalid")
    if int(seed) not in context.config.training.seeds:
        raise ValueError("V27d calibration seed is not registered")
    device = _device(context)
    fit_l1 = _load_variant_cache(context, variant=variant, role="representation_fit", task="known_gate")
    fit_l2 = _load_variant_cache(context, variant=variant, role="representation_fit", task="dbb_specialist")
    cal_l2 = _load_variant_cache(context, variant=variant, role="calibration", task="dbb_specialist")
    cal_l1 = _load_variant_cache(context, variant=variant, role="calibration", task="known_gate")
    l2_model, _ = load_head_checkpoint(context, variant=variant, task="dbb_specialist", seed=seed)
    l1_model, _ = load_head_checkpoint(context, variant=variant, task="known_gate", seed=seed)
    fit_l2_out = _head_output(l2_model, fit_l2, device)
    cal_l2_out = _head_output(l2_model, cal_l2, device)
    cal_l1_out = _head_output(l1_model, cal_l1, device)
    if (
        fit_l1["parent_ids"] != fit_l2["parent_ids"]
        or cal_l1["parent_ids"] != cal_l2["parent_ids"]
        or fit_l1["label_names"] != fit_l2["label_names"]
        or cal_l1["label_names"] != cal_l2["label_names"]
        or not torch.equal(fit_l1["tensors"]["lengths"], fit_l2["tensors"]["lengths"])
        or not torch.equal(cal_l1["tensors"]["lengths"], cal_l2["tensors"]["lengths"])
    ):
        raise ValueError("V27d calibration L1/L2 parent, label or length order differs")
    fit_joint_quality = fit_l1["tensors"]["quality_target_final"] & fit_l2["tensors"]["quality_target_final"]
    fit_mask = fit_l2["tensors"]["fit_eligible"] & fit_joint_quality & torch.tensor([label in V27D_KNOWN_LABELS for label in fit_l2["label_names"]], dtype=torch.bool)
    fit_projection = fit_l2_out["knownness_projection_final"][fit_mask]
    fit_labels = [label for label, selected in zip(fit_l2["label_names"], fit_mask.tolist(), strict=True) if selected]
    if any(fit_labels.count(class_name) == 0 for class_name in KNOWN_CLASS_NAMES):
        raise ValueError("V27d scorer fit pool lacks a known class")
    scorer = fit_scorer(fit_projection, fit_labels, config=ScorerConfig(prototype_count=3, ridge=1.0e-3, knn_k=5))
    cal_projection = cal_l2_out["knownness_projection_final"]
    cal_endpoint_logits = cal_l2["tensors"]["endpoint_final_logits"]
    cal_prediction = cal_endpoint_logits.argmax(dim=-1)
    calibration_nonconformity: dict[str, torch.Tensor] = {}
    cal_quality = cal_l1["tensors"]["quality_target_final"] & cal_l2["tensors"]["quality_target_final"]
    cal_known = cal_l2["tensors"]["fit_eligible"] & cal_quality & torch.tensor([label in V27D_KNOWN_LABELS for label in cal_l2["label_names"]], dtype=torch.bool)
    cal_nonconformity = conformal_nonconformity(cal_projection[cal_known], [label for label, selected in zip(cal_l2["label_names"], cal_known.tolist(), strict=True) if selected], scorer)
    for class_name in KNOWN_CLASS_NAMES:
        calibration_nonconformity[class_name] = cal_nonconformity[torch.tensor([label == class_name for label, selected in zip(cal_l2["label_names"], cal_known.tolist(), strict=True) if selected], dtype=torch.bool)]
        if calibration_nonconformity[class_name].numel() == 0:
            raise ValueError(f"V27d calibration pool lacks known class support: {class_name}")
    p_values = conformal_p_values(cal_projection, scorer, calibration_nonconformity)
    l2_scores = torch.sigmoid(cal_l2_out["knownness_logit_final"])
    l2_quality = cal_quality
    l2_correct = cal_l2["tensors"]["classification_correct"]
    l2_score_threshold, l2_p_threshold, l2_selection = _safe_l2_thresholds(
        knownness=l2_scores,
        p_values=p_values,
        labels=cal_l2["label_names"],
        quality=l2_quality,
        endpoint_prediction=cal_prediction,
        endpoint_correct=l2_correct,
        context=context,
    )
    l1_scores = torch.sigmoid(cal_l1_out["knownness_logit_final"])
    l1_quality = cal_l1["tensors"]["quality_target_final"]
    l1_conformal_fit = torch.tensor([label in {*V27D_KNOWN_LABELS, "ClutterTarget"} for label in cal_l1["label_names"]], dtype=torch.bool) & l1_quality
    l1_calibration_labels = ["Target" if label in V27D_KNOWN_LABELS else "Clutter" for label, selected in zip(cal_l1["label_names"], l1_conformal_fit.tolist(), strict=True) if selected]
    l1_calibration_nonconformity_values = classifier_nonconformity(
        cal_l1["tensors"]["endpoint_final_logits"][l1_conformal_fit],
        l1_calibration_labels,
        class_names=("Target", "Clutter"),
    )
    l1_classifier_nonconformity = {
        class_name: l1_calibration_nonconformity_values[
            torch.tensor([label == class_name for label in l1_calibration_labels], dtype=torch.bool)
        ]
        for class_name in ("Target", "Clutter")
    }
    l1_classifier_p_values = classifier_conformal_p_values(
        cal_l1["tensors"]["endpoint_final_logits"],
        l1_classifier_nonconformity,
        class_names=("Target", "Clutter"),
    )
    l1_prediction = cal_l1["tensors"]["endpoint_final_logits"].argmax(dim=-1)
    l1_endpoint_correct = torch.tensor(
        [
            (label in V27D_KNOWN_LABELS and int(l1_prediction[row]) == 0)
            or (label == "ClutterTarget" and int(l1_prediction[row]) == 1)
            for row, label in enumerate(cal_l1["label_names"])
        ],
        dtype=torch.bool,
    )
    l1_threshold, l1_p_threshold, l1_selection = _safe_l1_thresholds(
        l1_scores,
        l1_classifier_p_values,
        cal_l1["label_names"],
        l1_quality,
        l1_prediction,
        l1_endpoint_correct,
        context,
    )
    unknown_support = int(sum(label == "UnknownTarget" and bool(quality) for label, quality in zip(cal_l2["label_names"], l2_quality.tolist(), strict=True)))
    # Thresholds and conformal cutoffs are selected on this same pool.  Until a
    # post-selection audit pool exists, this is fitted but exploratory only.
    formal = False
    scorer_path = artifact_path(context.project_root, f"src-v3/v27/V27d/calibration/V27d-{variant}-l2-scorer-seed{seed}.pt")
    scorer_payload = {key: value for key, value in scorer.items()}
    write_torch_if_absent_or_identical(scorer_path, scorer_payload)
    serialized_nonconformity = {
        class_name: [float(value) for value in calibration_nonconformity[class_name].tolist()]
        for class_name in KNOWN_CLASS_NAMES
    }
    calibration_available = bool(l1_selection.get("feasible")) and bool(l2_selection.get("feasible"))
    registry: dict[str, Any] = {
        "schema_version": L3_REGISTRY_SCHEMA,
        "experiment_id": "V27d",
        "variant": variant,
        "seed": int(seed),
        "status": "fitted" if calibration_available else "unavailable",
        "calibration_scope": "internal_exploratory" if not formal else "formal_risk_constrained",
        "formal_status_reason": "post_selection_audit_pool_not_available",
        "calibration_unknown_support": unknown_support,
        "minimum_formal_unknown_support": context.config.calibration.minimum_calibration_unknown_support_for_formal,
        "feature_cache_role": "calibration",
        "v27d_manifest_sha256": context.manifest_sha256,
        "head_checkpoint_sha256": {"l1": sha256_file(artifact_path(context.project_root, context.config.artifacts.head_checkpoint_template, variant=variant, task="known_gate", seed=seed)), "l2": sha256_file(artifact_path(context.project_root, context.config.artifacts.head_checkpoint_template, variant=variant, task="dbb_specialist", seed=seed))},
        "l1": {
            "knownness_threshold": l1_threshold if calibration_available else None,
            "classifier_conformal_p_threshold": l1_p_threshold if calibration_available else None,
            "conformal_error_rate": context.config.calibration.conformal_error_rate,
            "class_names": ["Target", "Clutter"],
            "selection": l1_selection,
        },
        "l1_classifier_calibration_nonconformity": {
            class_name: [float(value) for value in values.tolist()]
            for class_name, values in l1_classifier_nonconformity.items()
        },
        "l2": {"knownness_threshold": l2_score_threshold if calibration_available else None, "conformal_p_threshold": l2_p_threshold if calibration_available else None, "conformal_error_rate": context.config.calibration.conformal_error_rate, "selection": l2_selection, "class_names": list(KNOWN_CLASS_NAMES)},
        "scorer_calibration_nonconformity": serialized_nonconformity,
        "scorer_path": relative_project_path(context.project_root, scorer_path),
        "scorer_sha256": sha256_file(scorer_path),
        "scorer_payload_sha256": scorer["scorer_sha256"],
        "risk_constraints": {
            "known_target_recall_floor": context.config.calibration.known_target_recall_floor,
            "clutter_recall_floor": context.config.calibration.clutter_recall_floor,
            "max_correct_known_new_rejections": context.config.calibration.max_correct_known_new_rejections,
            "max_per_class_correct_known_loss": context.config.calibration.max_per_class_correct_known_loss,
        },
        "fixed_softmax_threshold_used": False,
        "single_global_distance_gate_used": False,
        "unknown_is_classifier_class": False,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    registry_path = artifact_path(context.project_root, context.config.artifacts.calibration_registry_path, variant=variant)
    write_sealed_json(registry_path, registry)
    return {**registry, "registry_path": relative_project_path(context.project_root, registry_path), "registry_file_sha256": sha256_file(registry_path)}


def _validated_probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"V27d {field} must be a finite probability")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27d {field} must be a finite probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"V27d {field} must be within [0, 1]")
    return result


def _validated_score_threshold(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"V27d {field} must be a probability or -inf")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27d {field} must be a probability or -inf") from exc
    if result == float("-inf"):
        return result
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"V27d {field} must be within [0, 1] or -inf")
    return result


def _validated_reference(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"V27d {field} must contain non-empty support")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"V27d {field} contains a boolean")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"V27d {field} contains a non-numeric value") from exc
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"V27d {field} contains an invalid nonconformity")
        result.append(number)
    return result


def _validate_scorer_payload(scorer: Mapping[str, Any], context: V27DContext) -> None:
    if scorer.get("schema_version") != "RADAR-LN3-V27d-scorer-v1":
        raise ValueError("V27d scorer schema differs")
    if list(scorer.get("class_names") or ()) != list(KNOWN_CLASS_NAMES):
        raise ValueError("V27d scorer class order differs")
    projection_dim = scorer.get("projection_dim")
    if isinstance(projection_dim, bool) or not isinstance(projection_dim, int) or projection_dim != context.config.model.knownness_projection_dim:
        raise ValueError("V27d scorer projection dimension differs")
    for field in ("prototype_count", "knn_k"):
        value = scorer.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"V27d scorer {field} differs")
    ridge = scorer.get("ridge")
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("V27d scorer ridge differs")
    dimension = context.config.model.knownness_projection_dim
    prototypes = scorer.get("prototypes")
    means = scorer.get("means")
    inverse = scorer.get("inverse_covariances")
    references = scorer.get("reference_points")
    supports = scorer.get("supports")
    if (
        not isinstance(prototypes, torch.Tensor)
        or not isinstance(means, torch.Tensor)
        or not isinstance(inverse, torch.Tensor)
        or not isinstance(references, list)
        or not isinstance(supports, list)
        or len(references) != len(KNOWN_CLASS_NAMES)
        or len(supports) != len(KNOWN_CLASS_NAMES)
        or prototypes.ndim != 3
        or prototypes.shape[0] != len(KNOWN_CLASS_NAMES)
        or prototypes.shape[1] <= 0
        or prototypes.shape[1] > int(scorer["prototype_count"])
        or prototypes.shape[2] != dimension
        or means.shape != (len(KNOWN_CLASS_NAMES), dimension)
        or inverse.shape != (len(KNOWN_CLASS_NAMES), dimension, dimension)
        or not all(value.is_floating_point() and torch.isfinite(value).all() for value in (prototypes, means, inverse))
    ):
        raise ValueError("V27d scorer tensor contract differs")
    for class_index, reference in enumerate(references):
        if not isinstance(reference, torch.Tensor) or reference.ndim != 2 or reference.shape[0] <= 0 or reference.shape[1] != dimension or not reference.is_floating_point() or not torch.isfinite(reference).all():
            raise ValueError("V27d scorer reference-point contract differs")
        support = supports[class_index]
        if isinstance(support, bool) or not isinstance(support, int) or support != reference.shape[0]:
            raise ValueError("V27d scorer support contract differs")


def load_calibration_registry(context: V27DContext, *, variant: str, seed: int | None = None) -> dict[str, Any]:
    path = artifact_path(context.project_root, context.config.artifacts.calibration_registry_path, variant=variant)
    payload = read_sealed_json(path)
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    declared = body.pop("registry_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError("V27d calibration registry hash differs")
    if payload.get("schema_version") != L3_REGISTRY_SCHEMA or payload.get("experiment_id") != "V27d":
        raise ValueError("V27d calibration registry identity differs")
    if payload.get("variant") != variant:
        raise ValueError("V27d calibration registry variant differs")
    if payload.get("v27d_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27d calibration registry context differs")
    status = payload.get("status")
    if status not in {"fitted", "unavailable"}:
        raise ValueError("V27d calibration registry status differs")
    registry_seed = payload.get("seed")
    if isinstance(registry_seed, bool) or not isinstance(registry_seed, int) or registry_seed not in context.config.training.seeds:
        raise ValueError("V27d calibration registry seed differs")
    if seed is not None and int(seed) != registry_seed:
        raise ValueError("V27d calibration registry and evaluation seed differ")
    if payload.get("calibration_scope") != "internal_exploratory":
        raise ValueError("V27d formal calibration is not enabled for this experiment")
    if not isinstance(payload.get("formal_status_reason"), str) or not payload["formal_status_reason"].strip():
        raise ValueError("V27d calibration formal-status reason is missing")
    if payload.get("feature_cache_role") != "calibration" or payload.get("test_payload_access_count") != 0 or payload.get("deployment_eligible") is not False:
        raise ValueError("V27d calibration registry protocol differs")
    for field in ("calibration_unknown_support", "minimum_formal_unknown_support"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"V27d calibration registry {field} differs")
    if payload["minimum_formal_unknown_support"] != context.config.calibration.minimum_calibration_unknown_support_for_formal:
        raise ValueError("V27d calibration formal support requirement differs")
    for level, field in ((payload.get("l1"), "l1"), (payload.get("l2"), "l2")):
        if not isinstance(level, Mapping):
            raise ValueError(f"V27d calibration registry lacks {field}")
    l1 = payload["l1"]
    l2 = payload["l2"]
    if list(l1.get("class_names") or ()) != ["Target", "Clutter"] or list(l2.get("class_names") or ()) != list(KNOWN_CLASS_NAMES):
        raise ValueError("V27d calibration registry class order differs")
    configured_alpha = context.config.calibration.conformal_error_rate
    if not math.isclose(_validated_probability(l1.get("conformal_error_rate"), field="l1.conformal_error_rate"), configured_alpha, rel_tol=0.0, abs_tol=1.0e-12) or not math.isclose(_validated_probability(l2.get("conformal_error_rate"), field="l2.conformal_error_rate"), configured_alpha, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("V27d calibration conformal error rate differs from configuration")
    if status == "fitted":
        _validated_score_threshold(l1.get("knownness_threshold"), field="l1.knownness_threshold")
        _validated_probability(l1.get("classifier_conformal_p_threshold"), field="l1.classifier_conformal_p_threshold")
        _validated_score_threshold(l2.get("knownness_threshold"), field="l2.knownness_threshold")
        _validated_probability(l2.get("conformal_p_threshold"), field="l2.conformal_p_threshold")
        if float(l1["classifier_conformal_p_threshold"]) < configured_alpha or float(l2["conformal_p_threshold"]) < configured_alpha:
            raise ValueError("V27d calibration p-value threshold is below configured error rate")
    elif l1.get("knownness_threshold") is not None or l1.get("classifier_conformal_p_threshold") is not None or l2.get("knownness_threshold") is not None or l2.get("conformal_p_threshold") is not None:
        raise ValueError("V27d unavailable calibration must not contain active thresholds")
    for level, field in ((l1, "l1"), (l2, "l2")):
        selection = level.get("selection")
        if not isinstance(selection, Mapping) or selection.get("feasible") != (status == "fitted"):
            raise ValueError(f"V27d calibration {field} selection is infeasible")
    risk_constraints = payload.get("risk_constraints")
    expected_risk_constraints = {
        "known_target_recall_floor": context.config.calibration.known_target_recall_floor,
        "clutter_recall_floor": context.config.calibration.clutter_recall_floor,
        "max_correct_known_new_rejections": context.config.calibration.max_correct_known_new_rejections,
        "max_per_class_correct_known_loss": context.config.calibration.max_per_class_correct_known_loss,
    }
    if not isinstance(risk_constraints, Mapping) or dict(risk_constraints) != expected_risk_constraints:
        raise ValueError("V27d calibration risk constraints differ")
    raw_l1_nonconformity = payload.get("l1_classifier_calibration_nonconformity")
    if not isinstance(raw_l1_nonconformity, Mapping) or set(raw_l1_nonconformity) != {"Target", "Clutter"}:
        raise ValueError("V27d calibration registry lacks L1 conformal references")
    for class_name in ("Target", "Clutter"):
        _validated_reference(raw_l1_nonconformity[class_name], field=f"l1.{class_name}.nonconformity")
    head_hashes = payload.get("head_checkpoint_sha256")
    if not isinstance(head_hashes, Mapping) or set(head_hashes) != {"l1", "l2"}:
        raise ValueError("V27d calibration registry head bindings differ")
    for key, task in (("l1", "known_gate"), ("l2", "dbb_specialist")):
        checkpoint_path = artifact_path(context.project_root, context.config.artifacts.head_checkpoint_template, variant=variant, task=task, seed=registry_seed)
        if head_hashes.get(key) != sha256_file(checkpoint_path):
            raise ValueError(f"V27d calibration registry {key} head hash differs")
    scorer_name = f"src-v3/v27/V27d/calibration/V27d-{variant}-l2-scorer-seed{registry_seed}.pt"
    if not isinstance(payload.get("scorer_path"), str) or not payload["scorer_path"].strip():
        raise ValueError("V27d scorer path is missing")
    scorer_path = project_path(context.project_root, payload["scorer_path"], must_exist=True)
    expected_scorer_path = artifact_path(context.project_root, scorer_name)
    if scorer_path != expected_scorer_path:
        raise ValueError("V27d scorer path differs from registry binding")
    if sha256_file(scorer_path) != payload.get("scorer_sha256"):
        raise ValueError("V27d scorer artifact hash differs")
    scorer = torch.load(scorer_path, map_location="cpu", weights_only=False)
    if not isinstance(scorer, Mapping) or scorer.get("scorer_sha256") != scorer_digest(scorer):
        raise ValueError("V27d scorer payload hash differs")
    _validate_scorer_payload(scorer, context)
    if payload.get("scorer_payload_sha256") != scorer.get("scorer_sha256"):
        raise ValueError("V27d scorer payload binding differs")
    raw_nonconformity = payload.get("scorer_calibration_nonconformity")
    if not isinstance(raw_nonconformity, Mapping) or set(raw_nonconformity) != set(KNOWN_CLASS_NAMES):
        raise ValueError("V27d calibration registry lacks scorer nonconformity")
    nonconformity: dict[str, torch.Tensor] = {}
    for class_name in KNOWN_CLASS_NAMES:
        values = _validated_reference(raw_nonconformity.get(class_name), field=f"l2.{class_name}.nonconformity")
        tensor = torch.tensor(values, dtype=torch.float32)
        nonconformity[class_name] = tensor
    result = dict(payload)
    result["registry_path"] = relative_project_path(context.project_root, path)
    result["registry_file_sha256"] = sha256_file(path)
    result["scorer"] = dict(scorer)
    result["scorer_calibration_nonconformity"] = nonconformity
    return result


__all__ = ["CALIBRATION_ARTIFACT_SCHEMA", "fit_dynamic_calibration", "load_calibration_registry"]
