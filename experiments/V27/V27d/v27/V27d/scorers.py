"""Low-dimensional knownness scorers for V27d ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from radar_ln3.runtime import canonical_sha256


KNOWN_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")


@dataclass(frozen=True)
class ScorerConfig:
    prototype_count: int = 3
    ridge: float = 1.0e-3
    knn_k: int = 5


def _check_projection(value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] <= 0 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("V27d projection must be finite [N,D]")


def _class_id(label: str) -> int:
    if label not in KNOWN_CLASS_NAMES:
        raise ValueError(f"V27d scorer received non-known label: {label}")
    return KNOWN_CLASS_NAMES.index(label)


def _quantile_prototypes(values: torch.Tensor, count: int) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("V27d prototype values are empty")
    count = min(int(count), int(values.shape[0]))
    order = torch.argsort(values[:, 0], stable=True)
    chunks = torch.tensor_split(values[order], count, dim=0)
    return torch.stack([chunk.mean(dim=0) for chunk in chunks if chunk.shape[0]], dim=0)


def fit_scorer(
    projections: torch.Tensor,
    labels: Sequence[str],
    *,
    config: ScorerConfig = ScorerConfig(),
) -> dict[str, Any]:
    _check_projection(projections)
    if len(labels) != projections.shape[0] or not labels:
        raise ValueError("V27d scorer labels do not align")
    if (
        isinstance(config.prototype_count, bool)
        or isinstance(config.knn_k, bool)
        or isinstance(config.ridge, bool)
        or int(config.prototype_count) <= 0
        or int(config.knn_k) <= 0
        or float(config.ridge) <= 0.0
    ):
        raise ValueError("V27d scorer configuration is invalid")
    prototypes: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    inv_covariances: list[torch.Tensor] = []
    reference_points: list[torch.Tensor] = []
    supports: list[int] = []
    for class_name in KNOWN_CLASS_NAMES:
        selected = projections[
            torch.tensor(
                [label == class_name for label in labels],
                dtype=torch.bool,
                device=projections.device,
            )
        ]
        if selected.shape[0] == 0:
            raise ValueError(f"V27d scorer lacks class support: {class_name}")
        center = selected.mean(dim=0)
        centered = selected - center
        covariance = centered.T @ centered / float(max(1, selected.shape[0] - 1))
        covariance = covariance + torch.eye(
            selected.shape[1], dtype=selected.dtype, device=selected.device
        ) * float(config.ridge)
        inv = torch.linalg.pinv(covariance)
        prototypes.append(_quantile_prototypes(selected, config.prototype_count))
        means.append(center)
        inv_covariances.append(inv)
        reference_points.append(selected)
        supports.append(int(selected.shape[0]))
    payload = {
        "schema_version": "RADAR-LN3-V27d-scorer-v1",
        "class_names": list(KNOWN_CLASS_NAMES),
        "projection_dim": int(projections.shape[1]),
        "prototype_count": int(config.prototype_count),
        "knn_k": int(config.knn_k),
        "ridge": float(config.ridge),
        "prototypes": torch.stack(prototypes),
        "means": torch.stack(means),
        "inverse_covariances": torch.stack(inv_covariances),
        "reference_points": reference_points,
        "supports": supports,
    }
    payload["scorer_sha256"] = scorer_digest(payload)
    return payload


def scorer_digest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "scorer_sha256"}
    normalized: dict[str, Any] = {}
    for key, value in body.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            normalized[key] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": tensor.numpy().tobytes().hex(),
            }
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            normalized[key] = [
                {"dtype": str(item.dtype), "shape": list(item.shape), "bytes": item.detach().cpu().contiguous().numpy().tobytes().hex()}
                for item in value
            ]
        else:
            normalized[key] = value
    return canonical_sha256(normalized)


def _distance_to_prototypes(projection: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return torch.cdist(projection, prototypes.reshape(-1, prototypes.shape[-1])).reshape(
        projection.shape[0], prototypes.shape[0], prototypes.shape[1]
    )


def score_projection(
    projection: torch.Tensor,
    endpoint_logits: torch.Tensor,
    scorer: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    _check_projection(projection)
    if (
        endpoint_logits.ndim != 2
        or endpoint_logits.shape[0] != projection.shape[0]
        or endpoint_logits.shape[1] != 3
        or not endpoint_logits.is_floating_point()
        or endpoint_logits.device != projection.device
        or not torch.isfinite(endpoint_logits).all()
    ):
        raise ValueError("V27d scorer endpoint logits differ")
    prototypes = scorer["prototypes"].to(device=projection.device, dtype=projection.dtype)
    means = scorer["means"].to(device=projection.device, dtype=projection.dtype)
    inv_covariances = scorer["inverse_covariances"].to(device=projection.device, dtype=projection.dtype)
    if prototypes.ndim != 3 or prototypes.shape[0] != 3 or prototypes.shape[2] != projection.shape[1]:
        raise ValueError("V27d scorer prototype shape differs")
    normalized_projection = torch.nn.functional.normalize(projection, dim=-1)
    normalized_prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    cosine = torch.einsum("nd,kpd->nkp", normalized_projection, normalized_prototypes).amax(dim=-1)
    distances = _distance_to_prototypes(projection, prototypes)
    min_distance = distances.amin(dim=-1)
    centered = projection.unsqueeze(1) - means.unsqueeze(0)
    mahalanobis_distance = torch.einsum("nkd,kde,nke->nk", centered, inv_covariances, centered).clamp_min(0.0).sqrt()
    energy = torch.logsumexp(-distances, dim=-1)
    sorted_logits, sorted_indices = torch.sort(endpoint_logits, dim=-1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    predicted_class = sorted_indices[:, 0]
    predicted_cosine = cosine.gather(1, predicted_class.unsqueeze(-1)).squeeze(-1)
    predicted_distance = min_distance.gather(1, predicted_class.unsqueeze(-1)).squeeze(-1)
    predicted_mahalanobis = mahalanobis_distance.gather(1, predicted_class.unsqueeze(-1)).squeeze(-1)
    predicted_energy = energy.gather(1, predicted_class.unsqueeze(-1)).squeeze(-1)
    return {
        "cosine_by_class": cosine,
        "distance_by_class": min_distance,
        "mahalanobis_by_class": mahalanobis_distance,
        "energy_by_class": energy,
        "classification_margin": margin,
        "predicted_class": predicted_class,
        "predicted_cosine": predicted_cosine,
        "predicted_distance": predicted_distance,
        "predicted_mahalanobis": predicted_mahalanobis,
        "predicted_energy": predicted_energy,
    }


def conformal_nonconformity(
    projection: torch.Tensor,
    labels: Sequence[str],
    scorer: Mapping[str, Any],
) -> torch.Tensor:
    """Return class-conditional nonconformity for each true class."""

    _check_projection(projection)
    if len(labels) != projection.shape[0]:
        raise ValueError("V27d conformal labels do not align")
    distances = score_projection(
        projection,
        torch.zeros((projection.shape[0], 3), dtype=projection.dtype, device=projection.device),
        scorer,
    )["mahalanobis_by_class"]
    result = torch.empty((projection.shape[0],), dtype=projection.dtype, device=projection.device)
    for row, label in enumerate(labels):
        result[row] = distances[row, _class_id(str(label))]
    return result


def conformal_p_values(
    projection: torch.Tensor,
    scorer: Mapping[str, Any],
    calibration_nonconformity: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Compute class-conditional p-values without a fixed softmax threshold."""

    _check_projection(projection)
    distances = score_projection(
        projection,
        torch.zeros((projection.shape[0], 3), dtype=projection.dtype, device=projection.device),
        scorer,
    )["mahalanobis_by_class"]
    p_values = torch.zeros_like(distances)
    for class_index, class_name in enumerate(KNOWN_CLASS_NAMES):
        reference = calibration_nonconformity.get(class_name)
        if reference is None or reference.numel() == 0:
            raise ValueError(f"V27d conformal calibration lacks class: {class_name}")
        reference = reference.to(device=projection.device, dtype=projection.dtype)
        p_values[:, class_index] = (1.0 + (reference.unsqueeze(0) >= distances[:, class_index].unsqueeze(1)).sum(dim=1).to(dtype=projection.dtype)) / float(reference.numel() + 1)
    return p_values


def classifier_nonconformity(
    logits: torch.Tensor,
    labels: Sequence[str],
    *,
    class_names: Sequence[str],
) -> torch.Tensor:
    """Return true-class negative log probability for conformal routing."""

    if logits.ndim != 2 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("V27d classifier logits must be finite [N,C]")
    names = tuple(str(name) for name in class_names)
    if len(names) != logits.shape[1] or len(set(names)) != len(names):
        raise ValueError("V27d classifier class names differ")
    if len(labels) != logits.shape[0]:
        raise ValueError("V27d classifier labels do not align")
    positions = {name: index for index, name in enumerate(names)}
    try:
        target = torch.tensor([positions[str(label)] for label in labels], device=logits.device)
    except KeyError as exc:
        raise ValueError(f"V27d classifier conformal label is invalid: {exc.args[0]}") from exc
    return -torch.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)


def classifier_conformal_p_values(
    logits: torch.Tensor,
    calibration_nonconformity: Mapping[str, torch.Tensor],
    *,
    class_names: Sequence[str],
) -> torch.Tensor:
    """Compute class-conditional route p-values from frozen classifier logits."""

    if logits.ndim != 2 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("V27d classifier logits must be finite [N,C]")
    names = tuple(str(name) for name in class_names)
    if len(names) != logits.shape[1] or len(set(names)) != len(names):
        raise ValueError("V27d classifier class names differ")
    scores = -torch.log_softmax(logits, dim=-1)
    p_values = torch.empty_like(scores)
    for class_index, class_name in enumerate(names):
        reference = calibration_nonconformity.get(class_name)
        if not isinstance(reference, torch.Tensor) or reference.ndim != 1 or reference.numel() == 0:
            raise ValueError(f"V27d classifier conformal reference is missing: {class_name}")
        reference = reference.to(device=logits.device, dtype=logits.dtype)
        if not torch.isfinite(reference).all() or bool((reference < 0).any()):
            raise ValueError(f"V27d classifier conformal reference is invalid: {class_name}")
        p_values[:, class_index] = (
            1.0
            + (reference.unsqueeze(0) >= scores[:, class_index].unsqueeze(1)).sum(dim=1).to(dtype=logits.dtype)
        ) / float(reference.numel() + 1)
    return p_values


__all__ = ["KNOWN_CLASS_NAMES", "ScorerConfig", "classifier_conformal_p_values", "classifier_nonconformity", "conformal_nonconformity", "conformal_p_values", "fit_scorer", "score_projection", "scorer_digest"]