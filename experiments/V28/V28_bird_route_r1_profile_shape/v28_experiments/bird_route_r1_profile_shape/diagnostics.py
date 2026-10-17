"""Control/treatment diagnostics and calibration for Bird-route r1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from radar_ln3.runtime import sha256_file

from v28.heads import V28TrajectoryHeads, route_knownness_score
from v28.representations import load_representation

from .features import load_role
from .profile_head import _final_scores, _load_role_tensors, _select_threshold, load_profile_head
from .protocol import R1Context, resolve_artifact, write_sealed_json


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = (positive[:, None] > negative[None, :]).mean() + 0.5 * (positive[:, None] == negative[None, :]).mean()
    return float(comparisons)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    count = int(labels.sum())
    if count <= 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    positions = np.arange(1, len(labels) + 1)
    return float((cumulative[sorted_labels == 1] / positions[sorted_labels == 1]).sum() / count)


def _distribution(labels: np.ndarray, routes: np.ndarray, scores: np.ndarray, *, route: str = "BirdTarget") -> dict[str, Any]:
    mask = routes == route
    bird = mask & (labels == "BirdTarget")
    unknown = mask & (labels == "UnknownTarget")
    binary_labels = np.concatenate((np.ones(int(bird.sum()), dtype=np.int64), np.zeros(int(unknown.sum()), dtype=np.int64)))
    binary_scores = np.concatenate((scores[bird], scores[unknown]))
    return {
        "route": route,
        "route_support": int(mask.sum()),
        "bird_support": int(bird.sum()),
        "unknown_support": int(unknown.sum()),
        "bird_mean": float(scores[bird].mean()) if bird.any() else None,
        "unknown_mean": float(scores[unknown].mean()) if unknown.any() else None,
        "bird_median": float(np.median(scores[bird])) if bird.any() else None,
        "unknown_median": float(np.median(scores[unknown])) if unknown.any() else None,
        "auroc": _auc(binary_labels, binary_scores),
        "auprc": _average_precision(binary_labels, binary_scores),
    }


def _control_scores(context: R1Context, role: str) -> dict[str, Any]:
    payload = load_representation(context.v28_context, task="l2_dbb_specialist", role=role)
    head_payload = torch.load(context.project_root / context.config["base"]["l2_head_path"], map_location="cpu", weights_only=False)
    head = V28TrajectoryHeads(context.v28_context, task="l2_dbb_specialist")
    head.load_state_dict(head_payload["state_dict"], strict=True)
    tensors = payload["tensors"]
    with torch.inference_mode():
        output = head.forward_features(tensors["knownness_features"].float(), tensors["quality_features"].float(), tensors["padding_mask"].bool())
    lengths = tensors["lengths"].long()
    rows = torch.arange(lengths.shape[0])
    target = torch.sigmoid(output["target_knownness_logits"])[rows, lengths - 1].numpy()
    consistency = tensors["trajectory_consistency"][rows, lengths - 1].numpy()
    logits = tensors["sequence_logits"][rows, lengths - 1].numpy()
    temperature = float(context.v28_registry["calibration"]["tasks"]["l2_dbb_specialist"]["temperature"]["temperature"])
    routes = np.asarray([("DroneTarget", "BirdTarget", "BalloonTarget")[int(index)] for index in (logits / temperature).argmax(axis=1)], dtype=object)
    score = target.copy()
    bird = routes == "BirdTarget"
    score[bird] = np.asarray(route_knownness_score(target[bird], consistency[bird], route="BirdTarget", consistency_weight=float(context.config["model"]["consistency_weight_control"])), dtype=np.float64)
    return {"labels": np.asarray(payload["label_names"], dtype=object), "routes": routes, "scores": score, "target_scores": target, "consistency": consistency, "parent_ids": [str(value) for value in payload["parent_ids"]]}


def _treatment_scores(context: R1Context, role: str) -> dict[str, Any]:
    payload = _load_role_tensors(context, role)
    head, checkpoint = load_profile_head(context)
    device = torch.device("cpu")
    mean = checkpoint["normalization_mean"].float().to(device)
    scale = checkpoint["normalization_scale"].float().to(device)
    scores = _final_scores(head, payload, mean, scale, device)
    return {"labels": payload["labels"], "routes": payload["routes"], "scores": scores, "parent_ids": [str(value) for value in load_role(context, role)["parent_ids"]]}


def _comparison(context: R1Context, role: str) -> dict[str, Any]:
    control = _control_scores(context, role)
    treatment = _treatment_scores(context, role)
    if control["parent_ids"] != treatment["parent_ids"] or not np.array_equal(control["labels"], treatment["labels"]):
        raise ValueError(f"Bird-route r1 control/treatment parent alignment differs: {role}")
    return {
        "role": role,
        "parent_count": len(control["parent_ids"]),
        "control": _distribution(control["labels"], control["routes"], control["scores"]),
        "treatment": _distribution(treatment["labels"], treatment["routes"], treatment["scores"]),
        "control_route_counts": {name: int((control["routes"] == name).sum()) for name in ("DroneTarget", "BirdTarget", "BalloonTarget")},
        "treatment_route_counts": {name: int((treatment["routes"] == name).sum()) for name in ("DroneTarget", "BirdTarget", "BalloonTarget")},
    }


def run_diagnostic(context: R1Context) -> dict[str, Any]:
    result = {
        "schema_version": "RADAR-LN3-V28-bird-route-r1-profile-diagnostic-v1",
        "experiment_id": "V28-bird-route-r1-profile-shape",
        "status": "completed",
        "test_access_count": 0,
        "representation_development": _comparison(context, "representation_development"),
        "calibration": _comparison(context, "calibration"),
    }
    write_sealed_json(resolve_artifact(context, "diagnostic_report_path"), result)
    return result


def fit_calibration(context: R1Context) -> dict[str, Any]:
    head, checkpoint = load_profile_head(context)
    payload = _load_role_tensors(context, "calibration")
    mean = checkpoint["normalization_mean"].float()
    scale = checkpoint["normalization_scale"].float()
    scores = _final_scores(head, payload, mean, scale, torch.device("cpu"))
    selection = _select_threshold(scores, payload["labels"], payload["routes"], context.config)
    result = {
        "schema_version": "RADAR-LN3-V28-bird-route-r1-calibration-v1",
        "experiment_id": "V28-bird-route-r1-profile-shape",
        "status": "fitted" if selection["status"] == "fitted" else "internal_exploratory",
        "deployment_eligible": False,
        "test_access_count": 0,
        "head_checkpoint_sha256": sha256_file(resolve_artifact(context, "training_checkpoint_path", must_exist=True)),
        "calibration_role": "V28_calibration_only",
        "selection": selection,
        "support": {"parent_count": len(payload["labels"]), "bird_route_count": int((payload["routes"] == "BirdTarget").sum()), "unknown_route_count": int(((payload["routes"] == "BirdTarget") & (payload["labels"] == "UnknownTarget")).sum())},
    }
    write_sealed_json(resolve_artifact(context, "calibration_report_path"), result)
    return result


__all__ = ["fit_calibration", "run_diagnostic"]
