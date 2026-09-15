"""Frozen-backbone V28 knownness/quality training and safe selection."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from radar_ln3.runtime import atomic_torch_save, atomic_write_text, canonical_sha256, json_text, relative_project_path, sha256_file

from .heads import V28TrajectoryHeads
from .representations import load_representation
from .protocol import V28Context, V28_HEAD_SCHEMA, V28_L1_TASK, V28_L2_TASK, artifact_path


def _seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _quality_parts(payload: Mapping[str, Any], task: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    quality = tensors["quality_features"].float()
    padding = tensors["padding_mask"].bool()
    if quality.shape[-1] != 19 + 19 + (8 if task == V28_L1_TASK else 7):
        raise ValueError("V28 cached quality feature width differs")
    return quality[..., :19], quality[..., 19:38], quality[..., 38:], padding


def _known_targets(task: str, labels: list[str], *, device: torch.device) -> torch.Tensor:
    if task == V28_L1_TASK:
        return torch.tensor([0.0 if label == "UnknownTarget" else 1.0 for label in labels], device=device, dtype=torch.float32)
    return torch.tensor([1.0 if label in {"DroneTarget", "BirdTarget", "BalloonTarget"} else 0.0 for label in labels], device=device, dtype=torch.float32)


def _quality_targets(task: str, task_facts: torch.Tensor, event_facts: torch.Tensor, padding: torch.Tensor, context: V28Context) -> torch.Tensor:
    active = ~padding
    event_number = active.long().cumsum(dim=1)
    rd_fraction = task_facts[..., 1]
    dt_fraction = task_facts[..., 3]
    aux_fraction = task_facts[..., 7] if task == V28_L1_TASK else torch.minimum(task_facts[..., 5], task_facts[..., 6])
    heads = context.config.heads
    return active & (event_number >= int(heads["quality_min_events"])) & (rd_fraction >= float(heads["quality_min_rd_fraction"])) & (dt_fraction >= float(heads["quality_min_dt_fraction"])) & (aux_fraction >= float(heads["quality_min_aux_fraction"]))


def _parent_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, *, positive_weight: float) -> torch.Tensor:
    if logits.ndim != 2 or mask.shape != logits.shape:
        raise ValueError("V28 auxiliary BCE expects logits and mask shaped [B,T]")
    if targets.ndim == 1:
        if targets.shape[0] != logits.shape[0]:
            raise ValueError("V28 parent targets do not match logits batch")
        target_values = targets.unsqueeze(1).expand_as(logits)
    elif targets.ndim == 2:
        if targets.shape != logits.shape:
            raise ValueError("V28 event targets do not match logits")
        target_values = targets
    else:
        raise ValueError("V28 auxiliary BCE targets must be shaped [B] or [B,T]")
    values = nn.functional.binary_cross_entropy_with_logits(logits, target_values, pos_weight=logits.new_tensor(float(positive_weight)), reduction="none")
    counts = mask.sum(dim=1)
    eligible = counts > 0
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    return ((values * mask).sum(dim=1) / counts.clamp_min(1))[eligible].mean()


def _final_scores(payload: Mapping[str, Any], heads: V28TrajectoryHeads, *, task: str, context: V28Context, device: torch.device) -> dict[str, Any]:
    tensors = payload["tensors"]
    known = tensors["knownness_features"].to(device=device, dtype=torch.float32)
    quality = tensors["quality_features"].to(device=device, dtype=torch.float32)
    padding = tensors["padding_mask"].to(device=device, dtype=torch.bool)
    with torch.inference_mode():
        output = heads.forward_features(known, quality, padding)
    lengths = tensors["lengths"].long().to(device)
    rows = torch.arange(lengths.shape[0], device=device)
    known_scores = torch.sigmoid(output["knownness_logits"][rows, lengths - 1]).cpu().numpy()
    quality_scores = torch.sigmoid(output["quality_logits"][rows, lengths - 1]).cpu().numpy()
    task_facts, event_facts, task_fact_values, _ = _quality_parts(payload, task)
    quality_targets = _quality_targets(task, task_fact_values, event_facts, tensors["padding_mask"].bool(), context)[rows.cpu(), lengths.cpu() - 1].numpy().astype(np.int64)
    labels = [str(value) for value in payload["label_names"]]
    known_targets = _known_targets(task, labels, device=torch.device("cpu")).numpy().astype(np.int64)
    return {"known_scores": known_scores, "quality_scores": quality_scores, "known_targets": known_targets, "quality_targets": quality_targets, "labels": labels, "parent_ids": [str(value) for value in payload["parent_ids"]]}


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


def _select_knownness(task: str, development: Mapping[str, Any], config: V28Context) -> dict[str, Any]:
    labels = np.asarray(development["labels"], dtype=object)
    scores = np.asarray(development["known_scores"], dtype=np.float64)
    unknown = labels == "UnknownTarget"
    known_target = np.isin(labels, ["DroneTarget", "BirdTarget", "BalloonTarget"])
    clutter = labels == "ClutterTarget"
    floor = float(config.config.selection["knownness_recall_floor"])
    clutter_floor = float(config.config.selection["clutter_recall_floor"])
    max_reject = int(config.config.selection["max_correct_known_new_rejections"])
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
        clutter_safe = not clutter.any() or bool(clutter_recall >= clutter_floor)
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
    return {"selection_status": status, "deployment_eligible": eligible, "selected": selected, "candidate_count": len(candidates), "support_sufficient": support_sufficient, "diagnostics": _diagnostics((labels != "UnknownTarget").astype(np.int64) if task == V28_L1_TASK else np.isin(labels, ["DroneTarget", "BirdTarget", "BalloonTarget"]).astype(np.int64), scores)}


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


def _load_head_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != V28_HEAD_SCHEMA:
        raise ValueError("V28 head checkpoint schema differs")
    value = dict(payload)
    state = value.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("V28 head checkpoint state is missing")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("V28 head checkpoint state contains a non-tensor entry")
    state_hash = canonical_sha256({key: tensor.detach().cpu().numpy().tobytes().hex() for key, tensor in sorted(state.items())})
    if value.get("state_dict_sha256") != state_hash:
        raise ValueError("V28 head checkpoint state hash differs")
    if value.get("model_contract_sha256") != canonical_sha256(value.get("model_contract")):
        raise ValueError("V28 head checkpoint model contract hash differs")
    return value


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def train_heads(context: V28Context, *, task: str, seed: int, device: torch.device) -> dict[str, Any]:
    """Fit knownness then quality on representation_fit and select on development."""

    _seed(seed)
    fit = load_representation(context, task=task, role="representation_fit")
    development = load_representation(context, task=task, role="representation_development")
    heads = V28TrajectoryHeads(context, task=task).to(device)
    fit_tensors = fit["tensors"]
    padding = fit_tensors["padding_mask"].to(device).bool()
    known_features = fit_tensors["knownness_features"].to(device).float()
    quality_features = fit_tensors["quality_features"].to(device).float()
    labels = [str(value) for value in fit["label_names"]]
    known_targets = _known_targets(task, labels, device=device)
    _, event_facts, task_facts, _ = _quality_parts(fit, task)
    event_facts, task_facts = event_facts.to(device).float(), task_facts.to(device).float()
    quality_targets = _quality_targets(task, task_facts, event_facts, padding, context).float()
    event_number = (~padding).long().cumsum(dim=1)
    known_mask = (~padding) & (event_number >= int(context.config.heads["knownness_min_events"]))
    quality_mask = ~padding
    pos = float(known_targets.unsqueeze(1).expand_as(known_mask).masked_select(known_mask).sum().item())
    total = float(known_mask.sum().item())
    if pos <= 0 or total <= pos:
        raise ValueError("V28 knownness fit lacks both classes")
    known_weight = (total - pos) / pos
    quality_pos = float(quality_targets.masked_select(quality_mask).sum().item())
    quality_total = float(quality_mask.sum().item())
    if quality_pos <= 0 or quality_total <= quality_pos:
        raise ValueError("V28 quality fit lacks both classes")
    quality_weight = (quality_total - quality_pos) / quality_pos
    optimizer = torch.optim.AdamW(list(heads.knownness_projection.parameters()) + list(heads.knownness_head.parameters()), lr=float(context.config.heads["learning_rate"]), weight_decay=float(context.config.heads["weight_decay"]))
    history: list[dict[str, Any]] = []
    known_states: dict[int, dict[str, torch.Tensor]] = {}
    for epoch in range(1, int(context.config.heads["epochs"]) + 1):
        heads.train(); optimizer.zero_grad(set_to_none=True)
        output = heads.forward_features(known_features, quality_features, padding)
        loss = _parent_bce(output["knownness_logits"], known_targets, known_mask, positive_weight=known_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(heads.knownness_projection.parameters()) + list(heads.knownness_head.parameters()), float(context.config.heads["gradient_clip_norm"]), error_if_nonfinite=True); optimizer.step()
        heads.eval()
        fit_scores = _final_scores(fit, heads, task=task, context=context, device=device)
        dev_scores = _final_scores(development, heads, task=task, context=context, device=device)
        selection = _select_knownness(task, dev_scores, context)
        history.append({"epoch": epoch, "knownness_loss": float(loss.detach().cpu()), "selection": selection})
        known_states[epoch] = {
            **{f"knownness_projection.{key}": value for key, value in _clone_state(heads.knownness_projection).items()},
            **{f"knownness_head.{key}": value for key, value in _clone_state(heads.knownness_head).items()},
        }
    best_known = min(history, key=lambda row: (not bool(row["selection"]["deployment_eligible"]), -float(row["selection"]["selected"]["unknown_rejection"] or 0.0), -int(row["epoch"])))
    selected_known_state = known_states[int(best_known["epoch"])]
    heads.knownness_projection.load_state_dict({key.split(".", 1)[1]: value for key, value in selected_known_state.items() if key.startswith("knownness_projection.")}, strict=True)
    heads.knownness_head.load_state_dict({key.split(".", 1)[1]: value for key, value in selected_known_state.items() if key.startswith("knownness_head.")}, strict=True)
    # Quality is trained separately; knownness parameters are frozen during it.
    for parameter in heads.knownness_projection.parameters(): parameter.requires_grad_(False)
    for parameter in heads.knownness_head.parameters(): parameter.requires_grad_(False)
    quality_optimizer = torch.optim.AdamW(heads.quality_head.parameters(), lr=float(context.config.heads["learning_rate"]), weight_decay=float(context.config.heads["weight_decay"]))
    quality_history: list[dict[str, Any]] = []
    quality_states: dict[int, dict[str, torch.Tensor]] = {}
    for epoch in range(1, int(context.config.heads["epochs"]) + 1):
        heads.train(); quality_optimizer.zero_grad(set_to_none=True)
        output = heads.forward_features(known_features, quality_features, padding)
        quality_loss = _parent_bce(output["quality_logits"], quality_targets, quality_mask, positive_weight=quality_weight)
        quality_loss.backward(); torch.nn.utils.clip_grad_norm_(heads.quality_head.parameters(), float(context.config.heads["gradient_clip_norm"]), error_if_nonfinite=True); quality_optimizer.step()
        heads.eval(); dev_scores = _final_scores(development, heads, task=task, context=context, device=device)
        quality_selection = _select_quality(dev_scores, context)
        quality_history.append({"epoch": epoch, "quality_loss": float(quality_loss.detach().cpu()), "selection": quality_selection})
        quality_states[epoch] = _clone_state(heads.quality_head)
    best_quality = min(quality_history, key=lambda row: (not bool(row["selection"]["deployment_eligible"]), -float(row["selection"]["selected"]["negative_rejection"] or 0.0), -int(row["epoch"])))
    heads.quality_head.load_state_dict(quality_states[int(best_quality["epoch"])], strict=True)
    target = artifact_path(context.project_root, str(context.config.artifacts["head_checkpoint_template"]), task=task, seed=seed)
    payload = {"schema_version": V28_HEAD_SCHEMA, "experiment_id": "V28", "task": task, "seed": seed, "status": "fitted", "system_manifest_sha256": context.manifest_sha256, "representation_fit_cache_sha256": fit["cache_sha256"], "representation_development_cache_sha256": development["cache_sha256"], "model_contract": heads.model_contract(), "model_contract_sha256": canonical_sha256(heads.model_contract()), "state_dict": heads.state_dict(), "state_dict_sha256": canonical_sha256({key: value.detach().cpu().numpy().tobytes().hex() for key, value in sorted(heads.state_dict().items())}), "knownness_selection": best_known, "quality_selection": best_quality, "knownness_history": history, "quality_history": quality_history, "quality_gate_active": False, "test_payload_access_count": 0}
    atomic_torch_save(target, payload)
    summary = artifact_path(context.project_root, str(context.config.artifacts["head_summary_template"]), task=task, seed=seed)
    atomic_write_text(summary, json_text({key: value for key, value in payload.items() if key != "state_dict"}))
    return {"checkpoint_path": relative_project_path(context.project_root, target), "checkpoint_sha256": sha256_file(target), "summary_path": relative_project_path(context.project_root, summary), "knownness_selection": best_known, "quality_selection": best_quality}


__all__ = ["train_heads"]
