"""Bird projection prototype scores and threshold-free diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from radar_ln3.runtime import canonical_sha256

from v27.V27d.metrics import binary_auroc


def _check_projection(value: torch.Tensor) -> None:
    if value.ndim != 2 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("V27d-r5 projection must be finite [N,D]")


def _digest(payload: dict[str, Any]) -> str:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            normalized[key] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": tensor.numpy().tobytes().hex(),
            }
        else:
            normalized[key] = value
    return canonical_sha256(normalized)


def _quantile_prototypes(values: torch.Tensor, count: int) -> torch.Tensor:
    count = min(int(count), int(values.shape[0]))
    order = torch.argsort(values[:, 0], stable=True)
    chunks = torch.tensor_split(values[order], count, dim=0)
    return torch.stack([chunk.mean(dim=0) for chunk in chunks if chunk.shape[0]], dim=0)


def fit_bird_prototypes(projection: torch.Tensor, *, prototype_count: int = 3, ridge: float = 1.0e-3) -> dict[str, Any]:
    _check_projection(projection)
    if projection.shape[0] < 1 or int(prototype_count) <= 0 or float(ridge) <= 0.0:
        raise ValueError("V27d-r5 prototype configuration is invalid")
    mean = projection.mean(dim=0)
    centered = projection - mean
    covariance = centered.T @ centered / float(max(1, projection.shape[0] - 1))
    covariance = covariance + torch.eye(projection.shape[1], dtype=projection.dtype) * float(ridge)
    payload: dict[str, Any] = {
        "schema_version": "RADAR-LN3-V27d-r5-bird-prototype-v1",
        "class_name": "BirdTarget",
        "projection_dim": int(projection.shape[1]),
        "prototype_count": int(prototype_count),
        "ridge": float(ridge),
        "prototypes": _quantile_prototypes(projection, prototype_count),
        "mean": mean,
        "inverse_covariance": torch.linalg.pinv(covariance),
        "support": int(projection.shape[0]),
    }
    payload["scorer_sha256"] = _digest(payload)
    return payload


def validate_scorer(payload: dict[str, Any]) -> None:
    declared = payload.get("scorer_sha256")
    body = {key: value for key, value in payload.items() if key != "scorer_sha256"}
    if not isinstance(declared, str) or declared != _digest(body):
        raise ValueError("V27d-r5 scorer hash differs")
    if payload.get("schema_version") != "RADAR-LN3-V27d-r5-bird-prototype-v1":
        raise ValueError("V27d-r5 scorer schema differs")
    if int(payload.get("projection_dim", -1)) <= 0:
        raise ValueError("V27d-r5 scorer projection dimension is invalid")


def score_bird_projection(projection: torch.Tensor, scorer: dict[str, Any]) -> dict[str, torch.Tensor]:
    _check_projection(projection)
    validate_scorer(scorer)
    prototypes = scorer["prototypes"].to(device=projection.device, dtype=projection.dtype)
    mean = scorer["mean"].to(device=projection.device, dtype=projection.dtype)
    inverse = scorer["inverse_covariance"].to(device=projection.device, dtype=projection.dtype)
    if prototypes.shape[-1] != projection.shape[-1]:
        raise ValueError("V27d-r5 scorer projection width differs")
    normalized_projection = torch.nn.functional.normalize(projection, dim=-1)
    normalized_prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    cosine_by_prototype = normalized_projection @ normalized_prototypes.T
    distance_by_prototype = torch.cdist(projection, prototypes)
    centered = projection - mean
    mahalanobis = torch.sqrt(torch.einsum("nd,de,ne->n", centered, inverse, centered).clamp_min(0.0))
    return {
        "prototype_cosine": cosine_by_prototype.max(dim=-1).values,
        "prototype_distance": distance_by_prototype.min(dim=-1).values,
        "mahalanobis_distance": mahalanobis,
    }


def _threshold_candidates(values: torch.Tensor) -> list[float]:
    finite = sorted(set(float(value) for value in values.detach().cpu().tolist()))
    if not finite:
        return []
    return [*finite, float(torch.nextafter(torch.tensor(finite[-1]), torch.tensor(float("inf"))).item())]


def select_high_known_threshold(scores: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, *, higher_is_known: bool, recall_floor: float) -> dict[str, Any]:
    if scores.ndim != 1 or positive.shape != scores.shape or negative.shape != scores.shape:
        raise ValueError("V27d-r5 threshold inputs are not aligned")
    candidates = _threshold_candidates(scores[positive])
    best: tuple[float, float, float, dict[str, Any]] | None = None
    for threshold in candidates:
        accepted = scores.ge(threshold) if higher_is_known else scores.le(threshold)
        recall = float(accepted[positive].float().mean().item()) if bool(positive.any()) else 0.0
        rejection = float((~accepted[negative]).float().mean().item()) if bool(negative.any()) else 0.0
        if recall < float(recall_floor):
            continue
        ranking = (rejection, recall, -abs(float(threshold)))
        record = {
            "threshold": float(threshold),
            "higher_is_known": bool(higher_is_known),
            "bird_recall": recall,
            "unknown_rejection": rejection,
            "positive_support": int(positive.sum()),
            "negative_support": int(negative.sum()),
            "feasible": True,
        }
        if best is None or ranking > best[0]:
            best = (ranking, threshold, recall, record)
    if best is None:
        return {
            "threshold": None,
            "higher_is_known": bool(higher_is_known),
            "bird_recall": 0.0,
            "unknown_rejection": 0.0,
            "positive_support": int(positive.sum()),
            "negative_support": int(negative.sum()),
            "feasible": False,
        }
    return best[3]


def ranking_metrics(scores: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> dict[str, Any]:
    mask = positive | negative
    target = positive[mask]
    selected = scores[mask]
    return {
        "support": int(mask.sum()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "auroc_higher_is_known": binary_auroc(selected, target),
        "score_min": float(selected.min().item()) if selected.numel() else None,
        "score_max": float(selected.max().item()) if selected.numel() else None,
        "score_mean_positive": float(scores[positive].mean().item()) if bool(positive.any()) else None,
        "score_mean_negative": float(scores[negative].mean().item()) if bool(negative.any()) else None,
    }


__all__ = ["fit_bird_prototypes", "ranking_metrics", "score_bird_projection", "select_high_known_threshold", "validate_scorer"]