"""Evaluation reports for the frozen-endpoint V27d control."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import relative_project_path

from .features import load_feature_cache
from .l3 import resolve_l3_action
from .metrics import binary_auroc
from .protocol import V27DContext, V27D_KNOWN_LABELS, artifact_path, write_sealed_json
from .training import load_head_checkpoint
from .calibration import load_calibration_registry


def _head_inputs(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "summary_features": tensors["summary_features"],
        "event_trajectory": tensors["event_trajectory"],
        "task_facts_sequence": tensors["task_facts_sequence"],
        "event_facts_sequence": tensors["event_facts_sequence"],
        "raw_scale_features": tensors["raw_scale_features"],
        "stable8_causal_features": tensors["stable8_causal_features"],
        "padding_mask": tensors["padding_mask"],
    }


def _output(model: torch.nn.Module, payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        output = model(**_head_inputs(payload["tensors"]))
    return {key: value.detach().cpu() for key, value in output.items()}


def _binary_metrics(scores: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    if (
        scores.ndim != 1
        or target.ndim != 1
        or scores.shape != target.shape
        or not scores.is_floating_point()
        or target.dtype != torch.bool
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("V27d binary evaluation inputs must be aligned finite tensors")
    scores = scores.detach().cpu().to(dtype=torch.float64)
    target = target.detach().cpu().bool()
    predicted = scores.ge(0.5)
    tp = int((predicted & target).sum().item())
    fp = int((predicted & ~target).sum().item())
    fn = int((~predicted & target).sum().item())
    tn = int((~predicted & ~target).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    positive = int(target.sum().item())
    negative = int((~target).sum().item())
    if positive and negative:
        order = torch.argsort(scores, descending=True, stable=True)
        ordered_target = target[order]
        auc = binary_auroc(scores, target)
        cumulative_positive = ordered_target.to(torch.float64).cumsum(0)
        cumulative_count = torch.arange(1, len(order) + 1, dtype=torch.float64)
        average_precision = float(
            (cumulative_positive / cumulative_count * ordered_target).sum().item() / positive
        )
    else:
        auc = None
        average_precision = None
    return {
        "support": int(target.numel()),
        "positive": positive,
        "negative": negative,
        "accuracy": (tp + tn) / max(1, int(target.numel())),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auc,
        "auprc": average_precision,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _class_metrics(labels: list[str], predictions: torch.Tensor, names: tuple[str, ...]) -> dict[str, Any]:
    predicted = predictions.detach().cpu().tolist()
    result: dict[str, Any] = {"support": len(labels), "accuracy": None, "by_class": {}}
    expected = [index for index, label in enumerate(labels) if label in names]
    correct = [row for row in expected if predicted[row] == names.index(labels[row])]
    result["evaluated_support"] = len(expected)
    result["accuracy"] = len(correct) / max(1, len(expected))
    for class_index, class_name in enumerate(names):
        actual = {row for row, label in enumerate(labels) if label == class_name}
        selected = {row for row, value in enumerate(predicted) if value == class_index}
        tp = len(actual & selected)
        fp = len(selected - actual)
        fn = len(actual - selected)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        result["by_class"][class_name] = {
            "support": len(actual),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def _cache_path(context: V27DContext, *, variant: str, role: str, task: str) -> Path:
    return artifact_path(
        context.project_root,
        context.config.artifacts.feature_cache_template,
        variant=variant,
        role=role,
        task=task,
    )


def evaluate_task_role(
    context: V27DContext,
    *,
    variant: str,
    task: str,
    role: str,
    seed: int,
) -> dict[str, Any]:
    payload = load_feature_cache(context, _cache_path(context, variant=variant, role=role, task=task))
    model, checkpoint = load_head_checkpoint(context, variant=variant, task=task, seed=seed)
    output = _output(model, payload)
    tensors = payload["tensors"]
    labels = [str(value) for value in payload["label_names"]]
    endpoint_logits = tensors["endpoint_final_logits"]
    endpoint_prediction = endpoint_logits.argmax(dim=-1)
    if task == "known_gate":
        classification_labels = [
            "Target" if label in V27D_KNOWN_LABELS else "Clutter" if label == "ClutterTarget" else label
            for label in labels
        ]
        class_names = ("Target", "Clutter")
    else:
        classification_labels = labels
        class_names = V27D_KNOWN_LABELS
    quality_target = tensors["quality_target_final"].bool()
    quality_score = torch.sigmoid(output["quality_logit_final"])
    known_population = tensors["knownness_population"].bool() & quality_target
    known_score = torch.sigmoid(output["knownness_logit_final"])
    known_target = tensors["knownness_target"].bool()
    population_description = (
        "DBB, Clutter and Unknown; quality-eligible parents"
        if task == "known_gate"
        else "DBB and Unknown; quality-eligible parents"
    )
    return {
        "schema_version": "RADAR-LN3-V27d-task-evaluation-v1",
        "experiment_id": "V27d",
        "variant": variant,
        "task": task,
        "role": role,
        "seed": int(seed),
        "parent_count": len(labels),
        "label_counts": dict(Counter(labels)),
        "endpoint": {
            "classification_endpoint": payload["endpoint_policy"],
            "metrics": _class_metrics(classification_labels, endpoint_prediction, class_names),
        },
        "knownness": {
            "population": population_description,
            "eligible_parent_count": int(known_population.sum().item()),
            "metrics": _binary_metrics(known_score[known_population], known_target[known_population]),
        },
        "quality": {
            "metrics": _binary_metrics(quality_score, quality_target),
            "positive_count": int(quality_target.sum().item()),
        },
        "classification_logits_changed": False,
        "unknown_is_classifier_class": False,
        "checkpoint": {
            "path": relative_project_path(context.project_root, artifact_path(context.project_root, context.config.artifacts.head_checkpoint_template, variant=variant, task=task, seed=seed)),
            "head_state_sha256": checkpoint["head_state_sha256"],
        },
    }


def evaluate_l3_role(
    context: V27DContext,
    *,
    variant: str,
    role: str,
    seed: int,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    l1 = load_feature_cache(context, _cache_path(context, variant=variant, role=role, task="known_gate"))
    l2 = load_feature_cache(context, _cache_path(context, variant=variant, role=role, task="dbb_specialist"))
    if l1["parent_ids"] != l2["parent_ids"]:
        raise ValueError(f"V27d L1/L2 parent order differs for {role}")
    l1_model, _ = load_head_checkpoint(context, variant=variant, task="known_gate", seed=seed)
    l2_model, _ = load_head_checkpoint(context, variant=variant, task="dbb_specialist", seed=seed)
    l1_output = _output(l1_model, l1)
    l2_output = _output(l2_model, l2)
    labels = [str(value) for value in l2["label_names"]]
    actions: list[str] = []
    reasons: list[str] = []
    for row in range(len(labels)):
        decision = resolve_l3_action(
            registry=registry,
            l1_endpoint_logits=l1["tensors"]["endpoint_final_logits"][row],
            l1_knownness_logit=l1_output["knownness_logit_final"][row],
            l1_quality_logit=l1_output["quality_logit_final"][row],
            l2_endpoint_logits=l2["tensors"]["endpoint_final_logits"][row],
            l2_knownness_projection=l2_output["knownness_projection_final"][row],
            l2_knownness_logit=l2_output["knownness_logit_final"][row],
            l2_quality_logit=l2_output["quality_logit_final"][row],
            l1_deterministic_quality_eligible=bool(l1["tensors"]["quality_target_final"][row]),
            l2_deterministic_quality_eligible=bool(l2["tensors"]["quality_target_final"][row]),
        )
        actions.append(str(decision["action"]))
        reasons.append(str(decision["reason_code"]))
    action_counts = Counter(actions)
    reason_counts = Counter(reasons)
    by_label: dict[str, Any] = {}
    for label in sorted(set(labels)):
        rows = [row for row, item in enumerate(labels) if item == label]
        by_label[label] = {
            "support": len(rows),
            "action_counts": dict(Counter(actions[row] for row in rows)),
            "reason_counts": dict(Counter(reasons[row] for row in rows)),
        }
    l1_quality = l1["tensors"]["quality_target_final"].bool()
    l2_quality = l2["tensors"]["quality_target_final"].bool()
    target_quality = l1_quality & l2_quality
    unknown = torch.tensor([label == "UnknownTarget" for label in labels]) & target_quality
    known = torch.tensor([label in V27D_KNOWN_LABELS for label in labels]) & target_quality
    clutter = torch.tensor([label == "ClutterTarget" for label in labels]) & l1_quality
    dbb_actions = {"DroneTarget", "BirdTarget", "BalloonTarget"}
    known_rows = torch.where(known)[0].tolist()
    unknown_rows = torch.where(unknown)[0].tolist()
    clutter_rows = torch.where(clutter)[0].tolist()
    unknown_rejected = sum(actions[row] == "UNKNOWN_REJECTED" for row in unknown_rows)
    known_accepted = sum(actions[row] in dbb_actions for row in known_rows)
    clutter_accepted = sum(actions[row] == "Clutter" for row in clutter_rows)
    known_exact_accepted = sum(actions[row] == labels[row] for row in known_rows)
    known_wrong_accepted = sum(
        actions[row] in dbb_actions | {"Clutter"} and actions[row] != labels[row]
        for row in known_rows
    )
    known_pending = sum(actions[row] == "PENDING_ABSTAIN" for row in known_rows)
    known_rejected = sum(actions[row] == "UNKNOWN_REJECTED" for row in known_rows)
    unknown_error_accepted = sum(actions[row] in dbb_actions | {"Clutter"} for row in unknown_rows)
    unknown_pending = sum(actions[row] == "PENDING_ABSTAIN" for row in unknown_rows)
    return {
        "schema_version": "RADAR-LN3-V27d-l3-evaluation-v1",
        "experiment_id": "V27d",
        "variant": variant,
        "role": role,
        "seed": int(seed),
        "calibration_scope": registry.get("calibration_scope"),
        "calibration_status": registry.get("status"),
        "calibration_formal_status_reason": registry.get("formal_status_reason"),
        "action_counts": dict(action_counts),
        "reason_counts": dict(reason_counts),
        "by_label": by_label,
        "quality_eligible_counts": {
            "unknown": int(unknown.sum().item()),
            "known": int(known.sum().item()),
            "clutter": int(clutter.sum().item()),
            "l1": int(l1_quality.sum().item()),
            "l2": int(l2_quality.sum().item()),
        },
        "rates": {
            "unknown_rejection": unknown_rejected / max(1, int(unknown.sum().item())),
            "known_dbb_acceptance": known_accepted / max(1, int(known.sum().item())),
            "clutter_acceptance": clutter_accepted / max(1, int(clutter.sum().item())),
        },
        "open_set": {
            "known": {
                "eligible_support": len(known_rows),
                "dbb_route_accepted": known_accepted,
                "dbb_route_recall": known_accepted / max(1, len(known_rows)),
                "exact_class_accepted": known_exact_accepted,
                "exact_class_acceptance_rate": known_exact_accepted / max(1, len(known_rows)),
                "wrong_nonreject_accepted": known_wrong_accepted,
                "wrong_nonreject_acceptance_rate": known_wrong_accepted / max(1, len(known_rows)),
                "unknown_rejected": known_rejected,
                "pending": known_pending,
                "not_accepted_rate": (len(known_rows) - known_accepted) / max(1, len(known_rows)),
            },
            "unknown": {
                "eligible_support": len(unknown_rows),
                "strictly_rejected": unknown_rejected,
                "strict_rejection_rate": unknown_rejected / max(1, len(unknown_rows)),
                "error_accepted": unknown_error_accepted,
                "error_acceptance_rate": unknown_error_accepted / max(1, len(unknown_rows)),
                "pending": unknown_pending,
                "pending_rate": unknown_pending / max(1, len(unknown_rows)),
            },
            "clutter": {
                "eligible_support": len(clutter_rows),
                "accepted": clutter_accepted,
                "recall": clutter_accepted / max(1, len(clutter_rows)),
                "nonclutter_action": len(clutter_rows) - clutter_accepted,
            },
        },
        "low_quality_pending_count": sum(
            actions[row] == "PENDING_ABSTAIN" and reasons[row] == "LOW_QUALITY_OR_IMMATURE"
            for row in range(len(labels))
        ),
        "l1_low_quality_pending_count": sum(
            actions[row] == "PENDING_ABSTAIN"
            and reasons[row] == "LOW_QUALITY_OR_IMMATURE"
            and not bool(l1_quality[row])
            for row in range(len(labels))
        ),
        "l2_low_quality_pending_count": sum(
            actions[row] == "PENDING_ABSTAIN"
            and reasons[row] == "LOW_QUALITY_OR_IMMATURE"
            and bool(l1_quality[row])
            and not bool(l2_quality[row])
            for row in range(len(labels))
        ),
    }


def evaluate_v27d(
    context: V27DContext,
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    registry = load_calibration_registry(context, variant=variant, seed=seed)
    task_reports = {
        f"{task}:{role}": evaluate_task_role(
            context, variant=variant, task=task, role=role, seed=seed
        )
        for task in ("known_gate", "dbb_specialist")
        for role in ("representation_development", "calibration")
    }
    l3_reports = {
        role: evaluate_l3_role(
            context, variant=variant, role=role, seed=seed, registry=registry
        )
        for role in ("representation_development", "calibration")
    }
    report = {
        "schema_version": "RADAR-LN3-V27d-evaluation-v1",
        "experiment_id": "V27d",
        "variant": variant,
        "seed": int(seed),
        "calibration_registry_path": registry.get("registry_path"),
        "calibration_scope": registry.get("calibration_scope"),
        "calibration_status": registry.get("status"),
        "tasks": task_reports,
        "l3": l3_reports,
        "deployment_eligible": False,
        "formal_conformal_support": (
            registry.get("calibration_scope") == "formal_risk_constrained"
            and int(registry.get("calibration_unknown_support", 0))
            >= int(registry.get("minimum_formal_unknown_support", 10**9))
        ),
    }
    report_path = artifact_path(
        context.project_root,
        context.config.artifacts.evaluation_report_template,
        variant=variant,
        task="cascade",
        seed=seed,
        role="all",
    )
    write_sealed_json(report_path, report)
    return {"report_path": relative_project_path(context.project_root, report_path), "report": report}


__all__ = ["evaluate_l3_role", "evaluate_task_role", "evaluate_v27d"]
