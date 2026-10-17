"""Frozen-backbone V28 knownness/quality training and safe selection."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


from .heads import deterministic_quality_target
from .protocol import V28Context, V28_L1_TASK








def _quality_targets(task: str, task_facts: torch.Tensor, event_facts: torch.Tensor, padding: torch.Tensor, context: V28Context) -> torch.Tensor:
    """Use the shared deterministic quality contract for training and replay."""

    return deterministic_quality_target(
        context,
        task=task,
        task_facts=task_facts,
        event_facts=event_facts,
        padding_mask=padding,
    )






def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    comparisons = (positives[:, None] > negatives[None, :]).mean() + 0.5 * (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    if positive_count <= 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    positions = np.arange(1, len(labels) + 1)
    return float((cumulative[sorted_labels == 1] / positions[sorted_labels == 1]).sum() / positive_count)


def _diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (scores >= lower) & (scores <= upper if upper == 1.0 else scores < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return {"auroc": _roc_auc(labels, scores), "auprc": _average_precision(labels, scores), "brier": float(np.mean((scores - labels) ** 2)), "ece_10bin": float(ece), "positive_support": int(labels.sum()), "negative_support": int((labels == 0).sum())}


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    unique = sorted(set(float(value) for value in scores.tolist()))
    candidates = {0.0, 1.0}
    for value in unique:
        candidates.add(value)
        candidates.add(float(np.nextafter(value, 1.0)))
    return sorted(value for value in candidates if 0.0 <= value <= 1.0)


def _select_knownness(task: str, development: Mapping[str, Any], config: V28Context, *, route: str = "target") -> dict[str, Any]:
    labels = np.asarray(development["labels"], dtype=object)
    scores = np.asarray(development["known_scores"], dtype=np.float64)
    unknown = labels == "UnknownTarget"
    known_target = (
        np.isin(labels, ["DroneTarget", "BirdTarget", "BalloonTarget"])
        if route == "target"
        else labels == "ClutterTarget"
    )
    clutter = labels == "ClutterTarget"
    if task == V28_L1_TASK and route == "clutter":
        floor = float(config.config.heads["l1_clutter_knownness_recall_floor"])
        max_reject = int(config.config.selection["max_correct_known_new_rejections"])
    elif task == V28_L1_TASK:
        floor = float(config.config.heads["l1_target_knownness_recall_floor"])
        max_reject = int(config.config.selection["max_correct_known_new_rejections"])
    else:
        floor = float(config.config.heads["l2_drone_knownness_recall_floor"])
        max_reject = int(config.config.selection["max_correct_known_new_rejections"])
    clutter_floor = float(config.config.selection["clutter_recall_floor"])
    minimum_negative_support = int(config.config.selection["minimum_negative_support"])
    unknown_support = int(unknown.sum())
    support_sufficient = unknown_support >= minimum_negative_support
    candidates: list[dict[str, Any]] = []
    for threshold in _threshold_candidates(scores):
        accepted = scores >= threshold
        target_recall = float(accepted[known_target].mean()) if known_target.any() else 0.0
        clutter_recall = float(accepted[clutter].mean()) if clutter.any() else None
        unknown_rejection = float((~accepted[unknown]).mean()) if unknown.any() else None
        new_rejections = int((known_target & ~accepted).sum())
        clutter_safe = route != "clutter" or not clutter.any() or bool(clutter_recall >= clutter_floor)
        safe = target_recall >= floor and clutter_safe and new_rejections <= max_reject and support_sufficient
        candidates.append({"threshold": threshold, "known_target_recall": target_recall, "clutter_recall": clutter_recall, "unknown_rejection": unknown_rejection, "correct_known_new_rejections": new_rejections, "unknown_support": unknown_support, "minimum_negative_support": minimum_negative_support, "support_sufficient": support_sufficient, "safe": safe})
    safe = [row for row in candidates if row["safe"]]
    if safe:
        selected = max(safe, key=lambda row: (row["unknown_rejection"] if row["unknown_rejection"] is not None else -1.0, row["threshold"]))
        status = "feasible"
        eligible = True
    elif bool(config.config.selection["fallback_allowed"]):
        selected = min(candidates, key=lambda row: (max(0.0, floor - row["known_target_recall"]) + (max(0.0, clutter_floor - float(row["clutter_recall"])) if row["clutter_recall"] is not None else 0.0) + max(0, row["correct_known_new_rejections"] - max_reject) + (0.0 if row["support_sufficient"] else 1.0), -float(row["unknown_rejection"] or 0.0)))
        status = "fallback_infeasible"
        eligible = False
    else:
        raise ValueError(f"V28 {task} knownness safety gate is infeasible")
    return {"route": route, "selection_status": status, "deployment_eligible": eligible, "selected": selected, "candidate_count": len(candidates), "support_sufficient": support_sufficient, "diagnostics": _diagnostics(known_target.astype(np.int64), scores)}


def _select_quality(development: Mapping[str, Any], context: V28Context) -> dict[str, Any]:
    labels = np.asarray(development["quality_targets"], dtype=np.int64)
    scores = np.asarray(development["quality_scores"], dtype=np.float64)
    floor = float(context.config.selection["quality_recall_floor"])
    candidates = []
    for threshold in _threshold_candidates(scores):
        accepted = scores >= threshold
        recall = float(accepted[labels == 1].mean()) if bool((labels == 1).any()) else 0.0
        negative_rejection = float((~accepted[labels == 0]).mean()) if bool((labels == 0).any()) else None
        candidates.append({"threshold": threshold, "positive_recall": recall, "negative_rejection": negative_rejection, "safe": recall >= floor})
    safe = [row for row in candidates if row["safe"]]
    if safe:
        selected, status, eligible = max(safe, key=lambda row: (row["negative_rejection"] or -1.0, row["threshold"])), "feasible", True
    elif bool(context.config.selection["fallback_allowed"]):
        selected, status, eligible = min(candidates, key=lambda row: (max(0.0, floor - row["positive_recall"]), -(row["negative_rejection"] or 0.0))), "fallback_infeasible", False
    else:
        raise ValueError("V28 quality safety gate is infeasible")
    return {"selection_status": status, "deployment_eligible": eligible, "selected": selected, "candidate_count": len(candidates), "diagnostics": _diagnostics(labels, scores)}






