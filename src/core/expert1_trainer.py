from __future__ import annotations

import csv
import json
import random
import sys
import time as _time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .device import resolve_cuda_device
from .experts import LOW_ALTITUDE_CLASSES, LowAltitudeSuspectExpert
from .expert1_config import Expert1TrainConfig
from .encoding import encode_track_sequence
from .expert1_data import LOW_ALTITUDE_GLOBAL_TO_LOCAL, build_expert1_dataloaders
from .metrics import compute_classification_metrics
from .physical import PHYSICAL_FEATURE_COLUMNS
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .system_metrics import compute_confusion_matrix, compute_per_class_metrics
from .trajectory import TrackRepresentationBuilder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> tuple[torch.device, dict[str, object]]:
    return resolve_cuda_device(requested_device)


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _set_track_encoder_trainable(representation_builder: TrackRepresentationBuilder, trainable: bool) -> None:
    for param in representation_builder.sequence_encoder.parameters():
        param.requires_grad = trainable


def _build_optimizer(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model: nn.Module,
    config: Expert1TrainConfig,
) -> torch.optim.Optimizer:
    sequence_params = [
        param
        for param in representation_builder.sequence_encoder.parameters()
        if param.requires_grad
    ]
    sequence_param_ids = {id(param) for param in sequence_params}
    other_params = [
        param
        for param in list(representation_builder.parameters()) + list(rd_proj.parameters()) + list(model.parameters())
        if param.requires_grad and id(param) not in sequence_param_ids
    ]
    param_groups: list[dict[str, object]] = []
    if sequence_params:
        param_groups.append(
            {
                "params": sequence_params,
                "lr": config.learning_rate * config.sequence_encoder_lr_multiplier,
                "weight_decay": config.weight_decay,
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)


def _build_class_weights(loader) -> torch.Tensor:
    counts = torch.zeros((len(LOW_ALTITUDE_CLASSES),), dtype=torch.float32)
    for batch in loader:
        local_labels = torch.tensor(
            [LOW_ALTITUDE_GLOBAL_TO_LOCAL[int(label)] for label in batch.labels.tolist()],
            dtype=torch.long,
        )
        bincount = torch.bincount(local_labels, minlength=len(LOW_ALTITUDE_CLASSES)).to(torch.float32)
        counts += bincount
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=2.0)


def _is_present(value: object) -> bool:
    text = str(value if value is not None else "").strip().lower()
    return text not in {"", "nan", "none"}


def _has_rd_from_frame(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    rd_tensor = (
        frame["rd_tensor_path"].map(_is_present)
        if "rd_tensor_path" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    rd_summary = (
        frame["rd_summary_csv_path"].map(_is_present)
        if "rd_summary_csv_path" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    return rd_tensor | rd_summary


def _low_altitude_label_index(label_name: str) -> int | None:
    if label_name not in LOW_ALTITUDE_CLASSES:
        return None
    return int(LOW_ALTITUDE_CLASSES.index(label_name))


def _local_label_indices_from_label_names(label_names: pd.Series) -> torch.Tensor:
    values: list[int] = []
    for label_name in label_names.astype(str).tolist():
        index = _low_altitude_label_index(label_name)
        if index is None:
            continue
        values.append(index)
    if not values:
        return torch.zeros((0,), dtype=torch.long)
    return torch.tensor(values, dtype=torch.long)


def _build_class_weights_from_index(index_df: pd.DataFrame) -> torch.Tensor:
    local_labels = _local_label_indices_from_label_names(index_df["label_name"])
    counts = torch.bincount(local_labels, minlength=len(LOW_ALTITUDE_CLASSES)).to(torch.float32)
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=2.0)


def _build_rd_class_weights(loader) -> torch.Tensor:
    counts = torch.zeros((len(LOW_ALTITUDE_CLASSES),), dtype=torch.float32)
    for batch in loader:
        has_rd = batch.rd_is_multimodal.bool()
        if not bool(has_rd.any()):
            continue
        local_labels = torch.tensor(
            [LOW_ALTITUDE_GLOBAL_TO_LOCAL[int(label)] for label in batch.labels.tolist()],
            dtype=torch.long,
        )
        bincount = torch.bincount(local_labels[has_rd], minlength=len(LOW_ALTITUDE_CLASSES)).to(torch.float32)
        counts += bincount
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.5, max=3.0)


def _build_rd_class_weights_from_index(index_df: pd.DataFrame) -> torch.Tensor:
    rd_frame = index_df[_has_rd_from_frame(index_df)].copy()
    local_labels = _local_label_indices_from_label_names(rd_frame["label_name"])
    counts = torch.bincount(local_labels, minlength=len(LOW_ALTITUDE_CLASSES)).to(torch.float32)
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.5, max=3.0)


def _fit_source_feature_stats(loader) -> dict[str, dict[str, torch.Tensor]]:
    buckets: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        for idx, source_name in enumerate(batch.source_names):
            buckets.setdefault(str(source_name), []).append(batch.features[idx, : batch.lengths[idx]].float())
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for source_name, tensors in buckets.items():
        cat = torch.cat(tensors, dim=0)
        center = cat.mean(dim=0)
        scale = cat.std(dim=0).clamp_min(1e-3)
        stats[source_name] = {"center": center, "scale": scale}
    return stats


def _source_feature_stats_cache_path(data_root: str | Path, index_name: str) -> Path:
    data_root = Path(data_root)
    return data_root / "processed" / f"{Path(index_name).stem}_source_feature_stats.json"


def _load_source_feature_stats_cache(data_root: str | Path, index_name: str, row_count: int) -> dict[str, dict[str, torch.Tensor]] | None:
    data_root = Path(data_root)
    index_path = data_root / "processed" / index_name
    cache_path = _source_feature_stats_cache_path(data_root, index_name)
    if not index_path.exists() or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("index_name") != index_name:
        return None
    if int(payload.get("row_count", -1)) != int(row_count):
        return None
    if float(payload.get("index_mtime", -1.0)) != float(index_path.stat().st_mtime):
        return None
    stats_payload = payload.get("stats", {})
    if not isinstance(stats_payload, dict):
        return None
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for source_name, values in stats_payload.items():
        if not isinstance(values, dict) or "center" not in values or "scale" not in values:
            return None
        stats[str(source_name)] = {
            "center": torch.tensor(values["center"], dtype=torch.float32),
            "scale": torch.tensor(values["scale"], dtype=torch.float32),
        }
    return stats


def _write_source_feature_stats_cache(
    data_root: str | Path,
    index_name: str,
    row_count: int,
    stats: dict[str, dict[str, torch.Tensor]],
) -> None:
    data_root = Path(data_root)
    index_path = data_root / "processed" / index_name
    if not index_path.exists():
        return
    cache_path = _source_feature_stats_cache_path(data_root, index_name)
    payload = {
        "index_name": index_name,
        "index_mtime": float(index_path.stat().st_mtime),
        "row_count": int(row_count),
        "stats": {
            source_name: {
                "center": values["center"].detach().cpu().tolist(),
                "scale": values["scale"].detach().cpu().tolist(),
            }
            for source_name, values in stats.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fit_source_feature_stats_from_index(index_df: pd.DataFrame, data_root: str | Path, index_name: str) -> dict[str, dict[str, torch.Tensor]]:
    cached = _load_source_feature_stats_cache(data_root, index_name, len(index_df))
    if cached is not None:
        return cached
    data_root = Path(data_root)
    project_root = data_root.parent
    buckets: dict[str, list[torch.Tensor]] = {}
    for row in index_df.to_dict(orient="records"):
        csv_path = Path(str(row.get("csv_path", "")))
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        encoded = encode_track_sequence(frame)
        source_name = str(row.get("source_name", ""))
        buckets.setdefault(source_name, []).append(torch.from_numpy(encoded.features).float())
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for source_name, tensors in buckets.items():
        cat = torch.cat(tensors, dim=0)
        center = cat.mean(dim=0)
        scale = cat.std(dim=0).clamp_min(1e-3)
        stats[source_name] = {"center": center, "scale": scale}
    _write_source_feature_stats_cache(data_root, index_name, len(index_df), stats)
    return stats


def _apply_source_feature_norm(features: torch.Tensor, source_names: list[str], stats: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    normalized = features.clone()
    for idx, source_name in enumerate(source_names):
        source_key = str(source_name)
        if source_key not in stats:
            continue
        center = stats[source_key]["center"].to(features.device)
        scale = stats[source_key]["scale"].to(features.device)
        normalized[idx] = (normalized[idx] - center) / scale
    return normalized


def _to_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool) -> torch.Tensor:
    return tensor.to(device, non_blocking=bool(non_blocking and device.type == "cuda"))


def _subsample_time_axis(
    tensor: torch.Tensor,
    max_frames: int,
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if max_frames <= 0 or tensor.ndim < 2 or tensor.shape[1] <= max_frames:
        return tensor, indices
    if indices is None:
        indices = torch.linspace(
            0,
            tensor.shape[1] - 1,
            steps=max_frames,
            device=tensor.device,
        ).round().to(torch.long)
        indices = torch.unique_consecutive(indices)
    return tensor.index_select(1, indices), indices


def _subsample_deep_reroute_inputs(
    rd_features: torch.Tensor,
    rd_patch_features: torch.Tensor,
    rd_deep_patch_features: torch.Tensor,
    rd_padding_mask: torch.Tensor,
    rd_patch_padding_mask: torch.Tensor,
    rd_deep_patch_padding_mask: torch.Tensor,
    max_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    deep_rd_features, _ = _subsample_time_axis(rd_features, max_frames)
    deep_rd_patch_features, patch_indices = _subsample_time_axis(rd_patch_features, max_frames)
    deep_rd_deep_patch_features, deep_patch_indices = _subsample_time_axis(rd_deep_patch_features, max_frames)
    deep_rd_padding_mask, _ = _subsample_time_axis(rd_padding_mask, max_frames)
    deep_rd_patch_padding_mask, _ = _subsample_time_axis(rd_patch_padding_mask, max_frames, patch_indices)
    deep_rd_deep_patch_padding_mask, _ = _subsample_time_axis(
        rd_deep_patch_padding_mask,
        max_frames,
        deep_patch_indices,
    )
    return (
        deep_rd_features,
        deep_rd_patch_features,
        deep_rd_deep_patch_features,
        deep_rd_padding_mask,
        deep_rd_patch_padding_mask,
        deep_rd_deep_patch_padding_mask,
    )


def _parameters_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _clone_bundle_state_to_cpu(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model: nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "representation_builder_state_dict": _clone_state_dict_to_cpu(representation_builder),
        "rd_proj_state_dict": _clone_state_dict_to_cpu(rd_proj),
        "model_state_dict": _clone_state_dict_to_cpu(model),
    }


def _parameters_are_finite_bundle(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model: nn.Module,
) -> bool:
    return (
        _parameters_are_finite(representation_builder)
        and _parameters_are_finite(rd_proj)
        and _parameters_are_finite(model)
    )


def _load_pretrained_track_encoder(
    representation_builder: TrackRepresentationBuilder,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = {
        key[len("encoder."):] if key.startswith("encoder.") else key: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"no encoder.* weights found in pretrain checkpoint: {checkpoint_path}")
    missing, unexpected = representation_builder.sequence_encoder.load_state_dict(encoder_state, strict=False)
    return {
        "loaded_keys": len(encoder_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits_a, dim=-1)
    log_q = torch.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = torch.sum(p * (log_p - log_q), dim=-1)
    kl_qp = torch.sum(q * (log_q - log_p), dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def _supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    features = torch.nn.functional.normalize(embeddings, dim=-1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
    mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(mask, -1e9)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & (~mask)
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=-1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    mean_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=-1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def _rd_coverage(loader) -> dict[str, int]:
    with_rd = 0
    without_rd = 0
    for batch in loader:
        has_rd = batch.rd_is_multimodal.bool()
        with_rd += int(has_rd.sum().item())
        without_rd += int((~has_rd).sum().item())
    return {"with_rd": with_rd, "without_rd": without_rd}


def _rd_coverage_from_index(index_df: pd.DataFrame) -> dict[str, int]:
    has_rd = _has_rd_from_frame(index_df)
    return {
        "with_rd": int(has_rd.sum()),
        "without_rd": int((~has_rd).sum()),
    }


def _subset_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.numel() == 0 or not bool(mask.any()):
        return 0.0
    preds = logits.argmax(dim=-1)
    return float((preds[mask] == labels[mask]).float().mean().item())


def _source_accuracy_dict(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_names: list[str],
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    result: dict[str, float] = {}
    for source_name in sorted(set(source_names)):
        source_mask = torch.tensor([item == source_name for item in source_names], dtype=torch.bool)
        source_mask = source_mask & mask.cpu()
        if not bool(source_mask.any()):
            continue
        result[source_name] = float((preds[source_mask] == labels[source_mask]).float().mean().item())
    return result


def _source_macro_f1_dict(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_names: list[str],
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    result: dict[str, float] = {}
    cpu_mask = mask.cpu()
    for source_name in sorted(set(source_names)):
        source_mask = torch.tensor([item == source_name for item in source_names], dtype=torch.bool)
        source_mask = source_mask & cpu_mask
        if not bool(source_mask.any()):
            continue
        metrics = compute_classification_metrics(
            logits[source_mask],
            labels[source_mask],
            num_classes=len(LOW_ALTITUDE_CLASSES),
        )
        result[source_name] = float(metrics.macro_f1)
    return result


def _compute_supported_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for class_idx in range(num_classes):
        true_pos = labels == class_idx
        if not bool(true_pos.any()):
            continue
        pred_pos = preds == class_idx
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
    supported = len(f1s)
    return {
        "supported_class_count": int(supported),
        "supported_macro_precision": float(sum(precisions) / supported) if supported else 0.0,
        "supported_macro_recall": float(sum(recalls) / supported) if supported else 0.0,
        "supported_macro_f1": float(sum(f1s) / supported) if supported else 0.0,
    }


def _source_supported_metric_dict(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_names: list[str],
    mask: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    result: dict[str, dict[str, float]] = {}
    cpu_mask = mask.cpu()
    for source_name in sorted(set(source_names)):
        source_mask = torch.tensor([item == source_name for item in source_names], dtype=torch.bool)
        source_mask = source_mask & cpu_mask
        if not bool(source_mask.any()):
            continue
        result[source_name] = _compute_supported_metrics(
            logits[source_mask],
            labels[source_mask],
            num_classes=len(LOW_ALTITUDE_CLASSES),
        )
    return result


def _source_supported_macro_f1_summary(values: dict[str, dict[str, float]]) -> dict[str, float]:
    source_values = [float(item.get("supported_macro_f1", 0.0)) for item in values.values()]
    return {
        "source_balanced_supported_macro_f1": float(sum(source_values) / len(source_values)) if source_values else 0.0,
        "min_source_supported_macro_f1": float(min(source_values)) if source_values else 0.0,
        "raw_tr_rd_supported_macro_f1": float(values.get("raw_tr_rd", {}).get("supported_macro_f1", 0.0)),
        "excel_track_supported_macro_f1": float(values.get("excel_track", {}).get("supported_macro_f1", 0.0)),
    }


def _source_metric_dict(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_names: list[str],
    mask: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    result: dict[str, dict[str, float]] = {}
    cpu_mask = mask.cpu()
    for source_name in sorted(set(source_names)):
        source_mask = torch.tensor([item == source_name for item in source_names], dtype=torch.bool)
        source_mask = source_mask & cpu_mask
        if not bool(source_mask.any()):
            continue
        metrics = compute_classification_metrics(
            logits[source_mask],
            labels[source_mask],
            num_classes=len(LOW_ALTITUDE_CLASSES),
        )
        result[source_name] = {
            "accuracy": float(metrics.accuracy),
            "macro_precision": float(metrics.macro_precision),
            "macro_recall": float(metrics.macro_recall),
            "macro_f1": float(metrics.macro_f1),
        }
    return result


def _rd_time_sync_metric_dict(
    logits: torch.Tensor,
    traj_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    rd_logits: torch.Tensor,
    labels: torch.Tensor,
    statuses: list[str],
    has_rd: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    if labels.numel() == 0 or not statuses:
        return {}
    status_values = [str(status or "unavailable") for status in statuses]
    num_classes = len(LOW_ALTITUDE_CLASSES)
    result: dict[str, dict[str, float | int]] = {}
    cpu_has_rd = has_rd.cpu().bool()
    for status in sorted(set(status_values)):
        status_mask = torch.tensor([item == status for item in status_values], dtype=torch.bool)
        mask = status_mask & cpu_has_rd
        if not bool(mask.any()):
            continue
        final_metrics = compute_classification_metrics(logits[mask], labels[mask], num_classes=num_classes)
        traj_metrics = compute_classification_metrics(traj_logits[mask], labels[mask], num_classes=num_classes)
        joint_metrics = compute_classification_metrics(joint_logits[mask], labels[mask], num_classes=num_classes)
        rd_metrics = compute_classification_metrics(rd_logits[mask], labels[mask], num_classes=num_classes)
        result[status] = {
            "support": int(mask.sum().item()),
            "accuracy": float(final_metrics.accuracy),
            "macro_f1": float(final_metrics.macro_f1),
            "traj_macro_f1": float(traj_metrics.macro_f1),
            "joint_macro_f1": float(joint_metrics.macro_f1),
            "joint_minus_traj_macro_f1": float(joint_metrics.macro_f1 - traj_metrics.macro_f1),
            "rd_macro_f1": float(rd_metrics.macro_f1),
        }
    return result


def _source_support_dict(
    source_names: list[str],
    mask: torch.Tensor | None = None,
) -> dict[str, int]:
    if mask is None:
        mask = torch.ones((len(source_names),), dtype=torch.bool)
    mask = mask.cpu()
    result: dict[str, int] = {}
    for source_name in sorted(set(source_names)):
        source_mask = torch.tensor([item == source_name for item in source_names], dtype=torch.bool)
        count = int((source_mask & mask).sum().item())
        if count > 0:
            result[source_name] = count
    return result


def _source_summary_from_dict(values: dict[str, float]) -> dict[str, float]:
    source_values = list(values.values())
    return {
        "source_balanced_accuracy": float(sum(source_values) / len(source_values)) if source_values else 0.0,
        "min_source_accuracy": float(min(source_values)) if source_values else 0.0,
        "raw_tr_rd_accuracy": float(values.get("raw_tr_rd", 0.0)),
    }


def _source_macro_f1_summary(values: dict[str, float]) -> dict[str, float]:
    source_values = list(values.values())
    return {
        "source_balanced_macro_f1": float(sum(source_values) / len(source_values)) if source_values else 0.0,
        "min_source_macro_f1": float(min(source_values)) if source_values else 0.0,
        "raw_tr_rd_macro_f1": float(values.get("raw_tr_rd", 0.0)),
        "excel_track_macro_f1": float(values.get("excel_track", 0.0)),
    }


def _group_accuracy_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_names: list[str],
    has_rd: list[bool],
) -> dict[str, object]:
    by_source = _source_accuracy_dict(logits, labels, source_names)
    by_source_macro_f1 = _source_macro_f1_dict(logits, labels, source_names)
    by_source_supported = _source_supported_metric_dict(logits, labels, source_names)
    source_summary = _source_summary_from_dict(by_source)
    source_macro_f1_summary = _source_macro_f1_summary(by_source_macro_f1)
    source_supported_summary = _source_supported_macro_f1_summary(by_source_supported)
    rd_mask = torch.tensor(has_rd, dtype=torch.bool)
    return {
        "by_source_accuracy": by_source,
        "by_source_macro_f1": by_source_macro_f1,
        "by_source_supported_metrics": by_source_supported,
        "source_balanced_accuracy": source_summary["source_balanced_accuracy"],
        "min_source_accuracy": source_summary["min_source_accuracy"],
        "raw_tr_rd_accuracy": source_summary["raw_tr_rd_accuracy"],
        "source_balanced_macro_f1": source_macro_f1_summary["source_balanced_macro_f1"],
        "min_source_macro_f1": source_macro_f1_summary["min_source_macro_f1"],
        "raw_tr_rd_macro_f1": source_macro_f1_summary["raw_tr_rd_macro_f1"],
        "excel_track_macro_f1": source_macro_f1_summary["excel_track_macro_f1"],
        "source_balanced_supported_macro_f1": source_supported_summary["source_balanced_supported_macro_f1"],
        "min_source_supported_macro_f1": source_supported_summary["min_source_supported_macro_f1"],
        "raw_tr_rd_supported_macro_f1": source_supported_summary["raw_tr_rd_supported_macro_f1"],
        "excel_track_supported_macro_f1": source_supported_summary["excel_track_supported_macro_f1"],
        "with_rd_accuracy": _subset_accuracy(logits, labels, rd_mask),
        "without_rd_accuracy": _subset_accuracy(logits, labels, ~rd_mask),
    }


def _weighted_ce_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    unknown_index: int,
    unknown_weight: float,
) -> torch.Tensor:
    per_sample = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=criterion.weight,
        reduction="none",
    )
    weights = torch.ones_like(per_sample)
    weights = torch.where(labels == unknown_index, torch.full_like(weights, unknown_weight), weights)
    return (per_sample * weights).mean()


def _masked_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    unknown_index: int,
    unknown_weight: float,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    per_sample = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
    )
    if focal_gamma > 0.0:
        probs = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        per_sample = per_sample * (1.0 - probs).pow(focal_gamma)
    weights = torch.ones_like(per_sample)
    weights = torch.where(labels == unknown_index, torch.full_like(weights, unknown_weight), weights)
    return (per_sample * weights).mean()


def _binary_positive_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits,
        targets,
        weight=torch.tensor([1.0, positive_weight], dtype=torch.float32, device=logits.device),
    )


def _residual_targets(labels: torch.Tensor) -> torch.Tensor:
    residual_indices = {
        LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
        LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
    }
    return torch.isin(
        labels,
        torch.tensor(sorted(residual_indices), dtype=torch.long, device=labels.device),
    ).long()


def _target_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_index: int,
    margin: float,
) -> torch.Tensor:
    mask = labels == target_index
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    target_logits = logits[mask]
    positive = target_logits[:, target_index]
    negatives = torch.cat([target_logits[:, :target_index], target_logits[:, target_index + 1 :]], dim=-1)
    hardest_negative = negatives.max(dim=-1).values
    return torch.nn.functional.relu(float(margin) + hardest_negative - positive).mean()


def _rd_confidence_calibration_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_target: float,
    negative_target: float,
) -> torch.Tensor:
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=-1)
    positive = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1]).bool()
    hardest_negative = probs.masked_fill(one_hot, -1.0).max(dim=-1).values
    positive_loss = torch.nn.functional.relu(float(positive_target) - positive).mean()
    negative_loss = torch.nn.functional.relu(hardest_negative - float(negative_target)).mean()
    return positive_loss + negative_loss


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.as_posix())


def _epoch_metric_summary(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "traj_accuracy": metrics.get("traj_accuracy", 0.0),
        "traj_macro_f1": metrics.get("traj_macro_f1", 0.0),
        "traj_accuracy_on_rd": metrics.get("traj_accuracy_on_rd", 0.0),
        "traj_macro_f1_on_rd": metrics.get("traj_macro_f1_on_rd", 0.0),
        "joint_accuracy": metrics.get("joint_accuracy", 0.0),
        "joint_macro_f1": metrics.get("joint_macro_f1", 0.0),
        "joint_accuracy_on_rd": metrics.get("joint_accuracy_on_rd", 0.0),
        "joint_macro_f1_on_rd": metrics.get("joint_macro_f1_on_rd", 0.0),
        "joint_minus_traj_macro_f1_on_rd": metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0),
        "rd_accuracy": metrics.get("rd_accuracy", 0.0),
        "rd_macro_f1": metrics.get("rd_macro_f1", 0.0),
        "rd_accuracy_on_rd": metrics.get("rd_accuracy_on_rd", 0.0),
        "rd_macro_f1_on_rd": metrics.get("rd_macro_f1_on_rd", 0.0),
        "rd_drone_bird_recall_mean": metrics.get("rd_drone_bird_recall_mean", 0.0),
        "rd_unknown_clutter_recall_mean": metrics.get("rd_unknown_clutter_recall_mean", 0.0),
        "rd_drone_as_bird_rate": metrics.get("rd_drone_as_bird_rate", 0.0),
        "rd_bird_as_drone_rate": metrics.get("rd_bird_as_drone_rate", 0.0),
        "rd_rescue_count": metrics.get("rd_rescue_count", 0),
        "rd_harm_count": metrics.get("rd_harm_count", 0),
        "rd_rescue_rate_on_rd": metrics.get("rd_rescue_rate_on_rd", 0.0),
        "rd_harm_rate_on_rd": metrics.get("rd_harm_rate_on_rd", 0.0),
        "rd_net_gain_count": metrics.get("rd_net_gain_count", 0),
        "rd_net_gain_rate_on_rd": metrics.get("rd_net_gain_rate_on_rd", 0.0),
        "final_rescue_count": metrics.get("final_rescue_count", 0),
        "final_harm_count": metrics.get("final_harm_count", 0),
        "final_rescue_rate_on_rd": metrics.get("final_rescue_rate_on_rd", 0.0),
        "final_harm_rate_on_rd": metrics.get("final_harm_rate_on_rd", 0.0),
        "final_net_gain_count": metrics.get("final_net_gain_count", 0),
        "final_net_gain_rate_on_rd": metrics.get("final_net_gain_rate_on_rd", 0.0),
        "rd_rescue_unadopted_count": metrics.get("rd_rescue_unadopted_count", 0),
        "rd_harm_adopted_count": metrics.get("rd_harm_adopted_count", 0),
        "rd_raw_tr_rd_accuracy": metrics.get("rd_raw_tr_rd_accuracy", 0.0),
        "rd_fusion_weight_mean": metrics.get("rd_fusion_weight_mean", 0.0),
        "rd_fusion_weight_on_rd_mean": metrics.get("rd_fusion_weight_on_rd_mean", 0.0),
        "rd_class_fusion_weight_mean": metrics.get("rd_class_fusion_weight_mean", 0.0),
        "rd_class_fusion_weight_on_rd_mean": metrics.get("rd_class_fusion_weight_on_rd_mean", 0.0),
        "rd_pair_fusion_weight_mean": metrics.get("rd_pair_fusion_weight_mean", 0.0),
        "rd_pair_fusion_weight_on_rd_mean": metrics.get("rd_pair_fusion_weight_on_rd_mean", 0.0),
        "rd_pair_fusion_weight_by_pair": metrics.get("rd_pair_fusion_weight_by_pair", {}),
        "rd_pair_fusion_weight_on_rd_by_pair": metrics.get("rd_pair_fusion_weight_on_rd_by_pair", {}),
        "joint_rd_weight_mean": metrics.get("joint_rd_weight_mean", 0.0),
        "joint_rd_weight_on_rd_mean": metrics.get("joint_rd_weight_on_rd_mean", 0.0),
        "known_confidence_mean": metrics.get("known_confidence_mean", 0.0),
        "residual_score_mean": metrics.get("residual_score_mean", 0.0),
        "residual_score_on_residual_mean": metrics.get("residual_score_on_residual_mean", 0.0),
        "traj_residual_accuracy": metrics.get("traj_residual_accuracy", 0.0),
        "rd_residual_accuracy_on_rd": metrics.get("rd_residual_accuracy_on_rd", 0.0),
        "rd_unknown_clutter_detail_accuracy_on_residual_rd": metrics.get("rd_unknown_clutter_detail_accuracy_on_residual_rd", 0.0),
        "rd_clutter_morphology_accuracy_on_residual_rd": metrics.get("rd_clutter_morphology_accuracy_on_residual_rd", 0.0),
        "rd_unknown_clutter_detail_clutter_recall": metrics.get("rd_unknown_clutter_detail_clutter_recall", 0.0),
        "rd_clutter_morphology_clutter_recall": metrics.get("rd_clutter_morphology_clutter_recall", 0.0),
        "by_source_metrics": metrics.get("by_source_metrics", {}),
        "traj_by_source_metrics": metrics.get("traj_by_source_metrics", {}),
        "joint_by_source_metrics": metrics.get("joint_by_source_metrics", {}),
        "joint_by_source_metrics_on_rd": metrics.get("joint_by_source_metrics_on_rd", {}),
        "rd_by_source_metrics_on_rd": metrics.get("rd_by_source_metrics_on_rd", {}),
        "by_source_accuracy": metrics.get("by_source_accuracy", {}),
        "by_source_macro_f1": metrics.get("by_source_macro_f1", {}),
        "traj_by_source_accuracy": metrics.get("traj_by_source_accuracy", {}),
        "traj_by_source_macro_f1": metrics.get("traj_by_source_macro_f1", {}),
        "joint_by_source_accuracy": metrics.get("joint_by_source_accuracy", {}),
        "joint_by_source_macro_f1": metrics.get("joint_by_source_macro_f1", {}),
        "joint_by_source_accuracy_on_rd": metrics.get("joint_by_source_accuracy_on_rd", {}),
        "joint_by_source_macro_f1_on_rd": metrics.get("joint_by_source_macro_f1_on_rd", {}),
        "rd_by_source_accuracy_on_rd": metrics.get("rd_by_source_accuracy_on_rd", {}),
        "rd_by_source_macro_f1_on_rd": metrics.get("rd_by_source_macro_f1_on_rd", {}),
        "rd_source_balanced_accuracy_on_rd": metrics.get("rd_source_balanced_accuracy_on_rd", 0.0),
        "rd_min_source_accuracy_on_rd": metrics.get("rd_min_source_accuracy_on_rd", 0.0),
        "rd_source_balanced_macro_f1_on_rd": metrics.get("rd_source_balanced_macro_f1_on_rd", 0.0),
        "rd_min_source_macro_f1_on_rd": metrics.get("rd_min_source_macro_f1_on_rd", 0.0),
        "raw_tr_rd_accuracy": metrics.get("raw_tr_rd_accuracy", 0.0),
        "source_balanced_accuracy": metrics.get("source_balanced_accuracy", 0.0),
        "min_source_accuracy": metrics.get("min_source_accuracy", 0.0),
        "raw_tr_rd_macro_f1": metrics.get("raw_tr_rd_macro_f1", 0.0),
        "excel_track_macro_f1": metrics.get("excel_track_macro_f1", 0.0),
        "source_balanced_macro_f1": metrics.get("source_balanced_macro_f1", 0.0),
        "min_source_macro_f1": metrics.get("min_source_macro_f1", 0.0),
        "source_balanced_supported_macro_f1": metrics.get("source_balanced_supported_macro_f1", 0.0),
        "min_source_supported_macro_f1": metrics.get("min_source_supported_macro_f1", 0.0),
        "raw_tr_rd_supported_macro_f1": metrics.get("raw_tr_rd_supported_macro_f1", 0.0),
        "excel_track_supported_macro_f1": metrics.get("excel_track_supported_macro_f1", 0.0),
        "with_rd_accuracy": metrics.get("with_rd_accuracy", 0.0),
        "without_rd_accuracy": metrics.get("without_rd_accuracy", 0.0),
        "selection_score": metrics.get("selection_score", 0.0),
        "selection_eligible": metrics.get("selection_eligible", False),
        "selection_components": metrics.get("selection_components", {}),
        "selected_joint_logit_delta_scale": metrics.get("selected_joint_logit_delta_scale", metrics.get("joint_logit_delta_scale", 1.0)),
        "evaluated_batches": metrics.get("evaluated_batches", 0),
        "max_batches": metrics.get("max_batches", 0),
        "validation_scope": metrics.get("validation_scope", ""),
        "stage2_rd_aware_sampling": metrics.get("stage2_rd_aware_sampling", False),
        "rd_window_fraction_in_train_batches": metrics.get("rd_window_fraction_in_train_batches", metrics.get("with_rd_fraction", 0.0)),
        "rare_class_fraction_in_train_batches": metrics.get("rare_class_fraction_in_train_batches", metrics.get("rare_class_fraction", 0.0)),
        "stage2_source_fraction": metrics.get("stage2_source_fraction", {}),
        "stage2_label_fraction": metrics.get("stage2_label_fraction", {}),
        "stage2_raw_tr_rd_fraction": metrics.get("stage2_raw_tr_rd_fraction", 0.0),
        "stage2_top10_parent_fraction": metrics.get("stage2_top10_parent_fraction", 0.0),
        "stage2_effective_unique_parent_count": metrics.get("stage2_effective_unique_parent_count", 0.0),
        "tr_rd_path_rate": metrics.get("tr_rd_path_rate", metrics.get("rd_activation_rate", 0.0)),
        "tr_rd_path_rate_on_rd": metrics.get("tr_rd_path_rate_on_rd", metrics.get("rd_activation_rate_on_rd", 0.0)),
        "rd_activation_rate": metrics.get("rd_activation_rate", metrics.get("tr_rd_path_rate", 0.0)),
        "rd_activation_rate_on_rd": metrics.get("rd_activation_rate_on_rd", metrics.get("tr_rd_path_rate_on_rd", 0.0)),
        "skipped_non_finite_batches": metrics["skipped_non_finite_batches"],
        "has_valid_batches": metrics["has_valid_batches"],
    }


def _stage2_scale(epoch: int, config: Expert1TrainConfig) -> float:
    if epoch <= config.stage1_traj_only_epochs:
        return 0.0
    if config.stage2_ramp_epochs <= 0:
        return 1.0
    offset = epoch - config.stage1_traj_only_epochs
    return float(min(1.0, max(0.0, offset / float(config.stage2_ramp_epochs))))


def _per_class_recall_dict(logits: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    if logits.numel() == 0 or labels.numel() == 0:
        return {label_name: 0.0 for label_name in class_names}
    preds = logits.argmax(dim=-1)
    recalls: dict[str, float] = {}
    for idx, label_name in enumerate(class_names):
        mask = labels == idx
        recalls[label_name] = float((preds[mask] == idx).float().mean().item()) if bool(mask.any()) else 0.0
    return recalls


def _drone_bird_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    drone_idx = LOW_ALTITUDE_CLASSES.index("DroneTarget")
    bird_idx = LOW_ALTITUDE_CLASSES.index("BirdTarget")
    preds = logits.argmax(dim=-1) if logits.numel() else torch.empty((0,), dtype=torch.long)
    drone_mask = labels == drone_idx
    bird_mask = labels == bird_idx
    drone_support = int(drone_mask.sum().item())
    bird_support = int(bird_mask.sum().item())
    drone_recall = float((preds[drone_mask] == drone_idx).float().mean().item()) if drone_support else 0.0
    bird_recall = float((preds[bird_mask] == bird_idx).float().mean().item()) if bird_support else 0.0
    drone_as_bird = int(((preds == bird_idx) & drone_mask).sum().item()) if preds.numel() else 0
    bird_as_drone = int(((preds == drone_idx) & bird_mask).sum().item()) if preds.numel() else 0
    return {
        "drone_recall": drone_recall,
        "bird_recall": bird_recall,
        "drone_bird_recall_mean": 0.5 * (drone_recall + bird_recall) if (drone_support or bird_support) else 0.0,
        "drone_as_bird_count": drone_as_bird,
        "bird_as_drone_count": bird_as_drone,
        "drone_as_bird_rate": drone_as_bird / drone_support if drone_support else 0.0,
        "bird_as_drone_rate": bird_as_drone / bird_support if bird_support else 0.0,
    }


def _pair_name(a: str, b: str) -> str:
    return f"{a}__vs__{b}"


def _hard_pair_report(
    logits: torch.Tensor,
    traj_logits: torch.Tensor,
    rd_logits: torch.Tensor,
    labels: torch.Tensor,
    has_rd: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    final_preds = logits.argmax(dim=-1) if logits.numel() else torch.empty((0,), dtype=torch.long)
    traj_preds = traj_logits.argmax(dim=-1) if traj_logits.numel() else torch.empty((0,), dtype=torch.long)
    rd_preds = rd_logits.argmax(dim=-1) if rd_logits.numel() else torch.empty((0,), dtype=torch.long)
    pairs = [
        ("DroneTarget", "BirdTarget"),
        ("UnknownTarget", "ClutterTarget"),
        ("BirdTarget", "ClutterTarget"),
    ]
    for left_name, right_name in pairs:
        left_idx = LOW_ALTITUDE_CLASSES.index(left_name)
        right_idx = LOW_ALTITUDE_CLASSES.index(right_name)
        pair_mask = has_rd & ((labels == left_idx) | (labels == right_idx))
        support = int(pair_mask.sum().item())
        if support <= 0:
            report[_pair_name(left_name, right_name)] = {
                "support": 0,
                "rd_accuracy": 0.0,
                "traj_accuracy": 0.0,
                "final_accuracy": 0.0,
                "rd_rescue_count": 0,
                "rd_harm_count": 0,
                "final_rescue_count": 0,
                "final_harm_count": 0,
                "rd_net_gain_count": 0,
                "final_net_gain_count": 0,
                "rd_net_gain_rate": 0.0,
                "final_net_gain_rate": 0.0,
                f"{left_name}_recall": 0.0,
                f"{right_name}_recall": 0.0,
                f"{left_name}_as_{right_name}_count": 0,
                f"{right_name}_as_{left_name}_count": 0,
            }
            continue
        pair_labels = labels[pair_mask]
        pair_rd_preds = rd_preds[pair_mask]
        pair_traj_preds = traj_preds[pair_mask]
        pair_final_preds = final_preds[pair_mask]
        left_mask = pair_labels == left_idx
        right_mask = pair_labels == right_idx
        rd_correct = pair_rd_preds == pair_labels
        traj_correct = pair_traj_preds == pair_labels
        final_correct = pair_final_preds == pair_labels
        rd_rescue = (~traj_correct) & rd_correct
        rd_harm = traj_correct & (~rd_correct)
        final_rescue = (~traj_correct) & final_correct
        final_harm = traj_correct & (~final_correct)
        rd_rescue_count = int(rd_rescue.sum().item())
        rd_harm_count = int(rd_harm.sum().item())
        final_rescue_count = int(final_rescue.sum().item())
        final_harm_count = int(final_harm.sum().item())
        report[_pair_name(left_name, right_name)] = {
            "support": support,
            "rd_accuracy": float(rd_correct.float().mean().item()),
            "traj_accuracy": float(traj_correct.float().mean().item()),
            "final_accuracy": float(final_correct.float().mean().item()),
            "rd_rescue_count": rd_rescue_count,
            "rd_harm_count": rd_harm_count,
            "final_rescue_count": final_rescue_count,
            "final_harm_count": final_harm_count,
            "rd_net_gain_count": rd_rescue_count - rd_harm_count,
            "final_net_gain_count": final_rescue_count - final_harm_count,
            "rd_net_gain_rate": (rd_rescue_count - rd_harm_count) / support,
            "final_net_gain_rate": (final_rescue_count - final_harm_count) / support,
            f"{left_name}_recall": (
                float((pair_rd_preds[left_mask] == left_idx).float().mean().item())
                if bool(left_mask.any())
                else 0.0
            ),
            f"{right_name}_recall": (
                float((pair_rd_preds[right_mask] == right_idx).float().mean().item())
                if bool(right_mask.any())
                else 0.0
            ),
            f"{left_name}_as_{right_name}_count": int(((pair_rd_preds == right_idx) & left_mask).sum().item()),
            f"{right_name}_as_{left_name}_count": int(((pair_rd_preds == left_idx) & right_mask).sum().item()),
        }
    return report


def _unknown_clutter_hard_pair_metrics(
    logits: torch.Tensor,
    traj_logits: torch.Tensor,
    rd_logits: torch.Tensor,
    labels: torch.Tensor,
    has_rd: torch.Tensor,
) -> dict[str, float | int]:
    return _hard_pair_report(
        logits,
        traj_logits,
        rd_logits,
        labels,
        has_rd,
    ).get(_pair_name("UnknownTarget", "ClutterTarget"), {})


def _expert1_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix: str,
    mask: torch.Tensor | None = None,
) -> dict[str, object]:
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    if logits.numel() == 0 or labels.numel() == 0:
        return {
            f"{prefix}_per_class_recall": {label_name: 0.0 for label_name in LOW_ALTITUDE_CLASSES},
            f"{prefix}_drone_bird": _drone_bird_metrics(logits, labels),
            f"{prefix}_unknown_clutter_recall_mean": 0.0,
        }
    recalls = _per_class_recall_dict(logits, labels, LOW_ALTITUDE_CLASSES)
    unknown_clutter = 0.5 * (
        recalls.get("UnknownTarget", 0.0)
        + recalls.get("ClutterTarget", 0.0)
    )
    return {
        f"{prefix}_per_class_recall": recalls,
        f"{prefix}_drone_bird": _drone_bird_metrics(logits, labels),
        f"{prefix}_unknown_clutter_recall_mean": unknown_clutter,
    }


def _rd_teacher_effect_metrics(
    final_logits: torch.Tensor,
    traj_logits: torch.Tensor,
    rd_logits: torch.Tensor,
    labels: torch.Tensor,
    has_rd: torch.Tensor,
    source_names: list[str],
) -> dict[str, object]:
    final_preds = final_logits.argmax(dim=-1)
    traj_preds = traj_logits.argmax(dim=-1)
    rd_preds = rd_logits.argmax(dim=-1)
    rd_rescue = has_rd & (traj_preds != labels) & (rd_preds == labels)
    rd_harm = has_rd & (traj_preds == labels) & (rd_preds != labels)
    final_rescue = has_rd & (traj_preds != labels) & (final_preds == labels)
    final_harm = has_rd & (traj_preds == labels) & (final_preds != labels)
    rd_rescue_unadopted = rd_rescue & ~final_rescue
    rd_harm_adopted = rd_harm & final_harm
    with_rd_count = int(has_rd.sum().item())
    by_source: dict[str, dict[str, float | int]] = {}
    for source_name in sorted(set(source_names)):
        source_mask = torch.tensor([item == source_name for item in source_names], dtype=torch.bool)
        source_rd_mask = source_mask & has_rd.cpu()
        count = int(source_rd_mask.sum().item())
        if count <= 0:
            continue
        by_source[source_name] = {
            "with_rd_count": count,
            "rd_accuracy": float((rd_preds.cpu()[source_rd_mask] == labels.cpu()[source_rd_mask]).float().mean().item()),
            "rd_rescue_count": int(rd_rescue.cpu()[source_rd_mask].sum().item()),
            "rd_harm_count": int(rd_harm.cpu()[source_rd_mask].sum().item()),
        }
    raw_source_mask = torch.tensor([item == "raw_tr_rd" for item in source_names], dtype=torch.bool)
    raw_rd_mask = raw_source_mask & has_rd.cpu()
    return {
        "rd_rescue_count": int(rd_rescue.sum().item()),
        "rd_harm_count": int(rd_harm.sum().item()),
        "rd_rescue_rate_on_rd": int(rd_rescue.sum().item()) / with_rd_count if with_rd_count > 0 else 0.0,
        "rd_harm_rate_on_rd": int(rd_harm.sum().item()) / with_rd_count if with_rd_count > 0 else 0.0,
        "rd_net_gain_count": int(rd_rescue.sum().item()) - int(rd_harm.sum().item()),
        "rd_net_gain_rate_on_rd": (
            (int(rd_rescue.sum().item()) - int(rd_harm.sum().item())) / with_rd_count if with_rd_count > 0 else 0.0
        ),
        "final_rescue_count": int(final_rescue.sum().item()),
        "final_harm_count": int(final_harm.sum().item()),
        "final_rescue_rate_on_rd": int(final_rescue.sum().item()) / with_rd_count if with_rd_count > 0 else 0.0,
        "final_harm_rate_on_rd": int(final_harm.sum().item()) / with_rd_count if with_rd_count > 0 else 0.0,
        "final_net_gain_count": int(final_rescue.sum().item()) - int(final_harm.sum().item()),
        "final_net_gain_rate_on_rd": (
            (int(final_rescue.sum().item()) - int(final_harm.sum().item())) / with_rd_count if with_rd_count > 0 else 0.0
        ),
        "rd_rescue_unadopted_count": int(rd_rescue_unadopted.sum().item()),
        "rd_harm_adopted_count": int(rd_harm_adopted.sum().item()),
        "rd_raw_tr_rd_accuracy": (
            float((rd_preds.cpu()[raw_rd_mask] == labels.cpu()[raw_rd_mask]).float().mean().item())
            if bool(raw_rd_mask.any())
            else 0.0
        ),
        "rd_teacher_by_source": by_source,
    }


def _expert1_selection_score(metrics: dict[str, object], config: Expert1TrainConfig, stage2_enabled: bool) -> tuple[float, bool, dict[str, float]]:
    rd_macro = float(metrics.get("rd_macro_f1_on_rd", metrics.get("rd_macro_f1", 0.0)))
    drone_bird = float(metrics.get("rd_drone_bird_recall_mean", 0.0))
    unknown_clutter = float(metrics.get("rd_unknown_clutter_recall_mean", 0.0))
    final_macro = float(metrics.get("macro_f1", 0.0))
    traj_macro = float(metrics.get("traj_macro_f1", 0.0))
    joint_macro_on_rd = float(metrics.get("joint_macro_f1_on_rd", 0.0))
    joint_minus_traj_on_rd = float(metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0))
    accuracy = float(metrics.get("accuracy", 0.0))
    source_balanced_accuracy = float(metrics.get("source_balanced_accuracy", 0.0))
    min_source_accuracy = float(metrics.get("min_source_accuracy", 0.0))
    raw_source_accuracy = float(metrics.get("raw_tr_rd_accuracy", 0.0))
    source_balanced_macro_f1 = float(metrics.get("source_balanced_macro_f1", 0.0))
    min_source_macro_f1 = float(metrics.get("min_source_macro_f1", 0.0))
    raw_source_macro_f1 = float(metrics.get("raw_tr_rd_macro_f1", 0.0))
    excel_source_macro_f1 = float(metrics.get("excel_track_macro_f1", 0.0))
    source_balanced_supported_macro_f1 = float(metrics.get("source_balanced_supported_macro_f1", 0.0))
    min_source_supported_macro_f1 = float(metrics.get("min_source_supported_macro_f1", 0.0))
    raw_source_supported_macro_f1 = float(metrics.get("raw_tr_rd_supported_macro_f1", 0.0))
    excel_source_supported_macro_f1 = float(metrics.get("excel_track_supported_macro_f1", 0.0))
    rd_raw_source_accuracy = float(metrics.get("rd_raw_tr_rd_accuracy", 0.0))
    rd_source_balanced_accuracy = float(metrics.get("rd_source_balanced_accuracy_on_rd", 0.0))
    rd_min_source_accuracy = float(metrics.get("rd_min_source_accuracy_on_rd", 0.0))
    rd_source_balanced_macro_f1 = float(metrics.get("rd_source_balanced_macro_f1_on_rd", 0.0))
    rd_min_source_macro_f1 = float(metrics.get("rd_min_source_macro_f1_on_rd", 0.0))
    without_rd_accuracy = float(metrics.get("without_rd_accuracy", 0.0))
    rd_harm_rate = float(metrics.get("rd_harm_rate_on_rd", 0.0))
    rd_net_gain_rate = float(metrics.get("rd_net_gain_rate_on_rd", 0.0))
    final_harm_rate = float(metrics.get("final_harm_rate_on_rd", 0.0))
    final_net_gain_rate = float(metrics.get("final_net_gain_rate_on_rd", 0.0))
    rd_residual_accuracy = float(metrics.get("rd_residual_accuracy_on_rd", 0.0))
    residual_detail_accuracy = float(metrics.get("rd_unknown_clutter_detail_accuracy_on_residual_rd", 0.0))
    clutter_morphology_accuracy = float(metrics.get("rd_clutter_morphology_accuracy_on_residual_rd", 0.0))
    residual_score_on_residual = float(metrics.get("residual_score_on_residual_mean", 0.0))
    final_delta = final_macro - traj_macro
    final_per_class_recall = dict(metrics.get("final_per_class_recall", {}))
    final_unknown_recall = float(final_per_class_recall.get("UnknownTarget", 0.0))
    final_clutter_recall = float(final_per_class_recall.get("ClutterTarget", 0.0))
    final_unknown_clutter_recall_mean = 0.5 * (final_unknown_recall + final_clutter_recall)
    parent_unknown_recall = float(metrics.get("parent_unknown_recall", 0.0))
    parent_clutter_recall = float(metrics.get("parent_clutter_recall", 0.0))
    parent_unknown_clutter_recall_mean = float(metrics.get("parent_unknown_clutter_recall_mean", 0.0))
    source_support = dict(metrics.get("source_support", {}))
    rd_source_support = dict(metrics.get("rd_source_support_on_rd", {}))
    rd_class_support = dict(metrics.get("rd_class_support_on_rd", {}))
    rd_recall = dict(metrics.get("rd_per_class_recall_on_rd", {}))
    drone_recall = float(rd_recall.get("DroneTarget", 0.0))
    bird_recall = float(rd_recall.get("BirdTarget", 0.0))
    min_drone_bird_recall = min(drone_recall, bird_recall)
    balloon_recall = float(rd_recall.get("BalloonTarget", 0.0))
    clutter_recall = float(rd_recall.get("ClutterTarget", 0.0))
    unknown_recall = float(rd_recall.get("UnknownTarget", 0.0))
    min_rare_rd_recall = min(balloon_recall, clutter_recall, unknown_recall)
    joint_minus_traj_deficit = max(0.0, -(joint_minus_traj_on_rd + 0.0100))
    final_net_gain_deficit = max(0.0, -final_net_gain_rate)
    rd_negative_gain_penalty = 0.90 * joint_minus_traj_deficit + 0.60 * final_net_gain_deficit
    score = (
        0.70 * final_macro
        + 0.25 * accuracy
        + 0.35 * source_balanced_accuracy
        + 0.25 * min_source_accuracy
        + 0.20 * raw_source_accuracy
        + 0.35 * source_balanced_macro_f1
        + 0.22 * min_source_macro_f1
        + 0.18 * raw_source_macro_f1
        + 0.16 * excel_source_macro_f1
        + 0.14 * source_balanced_supported_macro_f1
        + 0.08 * min_source_supported_macro_f1
        + 0.20 * final_unknown_clutter_recall_mean
        + 0.08 * min(final_unknown_recall, final_clutter_recall)
        + 0.08 * parent_unknown_clutter_recall_mean
        + 0.20 * without_rd_accuracy
        + 0.35 * joint_macro_on_rd
        + 0.40 * joint_minus_traj_on_rd
        + 0.35 * rd_macro
        + 0.18 * rd_source_balanced_accuracy
        + 0.12 * rd_min_source_accuracy
        + 0.18 * rd_source_balanced_macro_f1
        + 0.10 * rd_min_source_macro_f1
        + 0.15 * rd_raw_source_accuracy
        + 0.28 * drone_bird
        + 0.32 * unknown_clutter
        + 0.20 * min_rare_rd_recall
        + 0.35 * final_net_gain_rate
        + 0.10 * rd_residual_accuracy
        + 0.10 * residual_detail_accuracy
        + 0.08 * clutter_morphology_accuracy
        + 0.05 * residual_score_on_residual
        - 0.35 * final_harm_rate
        + 0.25 * traj_macro
        + 0.25 * final_delta
        - rd_negative_gain_penalty
    )
    eligible = bool(stage2_enabled) if config.prefer_stage2_checkpoint else True
    if stage2_enabled:
        enough_raw_source = source_support.get("raw_tr_rd", 0) >= config.min_source_support_for_hard_gate
        enough_rd_raw_source = rd_source_support.get("raw_tr_rd", 0) >= config.min_rd_source_support_for_hard_gate
        enough_unknown = rd_class_support.get("UnknownTarget", 0) >= config.min_rd_class_support_for_hard_gate
        enough_clutter = rd_class_support.get("ClutterTarget", 0) >= config.min_rd_class_support_for_hard_gate
        enough_bird = rd_class_support.get("BirdTarget", 0) >= config.min_rd_class_support_for_hard_gate
        eligible = eligible and rd_macro >= config.min_rd_evidence_macro_f1_for_best
        eligible = eligible and drone_bird >= config.min_drone_bird_rd_recall_for_best
        if enough_bird:
            eligible = eligible and min_drone_bird_recall >= config.min_drone_bird_single_rd_recall_for_best
        if enough_unknown and enough_clutter:
            eligible = eligible and unknown_clutter >= config.min_unknown_clutter_rd_recall_for_best
        if enough_unknown and enough_clutter and enough_bird:
            eligible = eligible and min_rare_rd_recall >= config.min_rare_rd_recall_for_best
        eligible = eligible and without_rd_accuracy >= config.min_without_rd_accuracy_for_best
        eligible = eligible and source_balanced_accuracy >= config.min_source_balanced_accuracy_for_best
        eligible = eligible and source_balanced_supported_macro_f1 >= config.min_source_balanced_macro_f1_for_best
        if enough_raw_source:
            eligible = eligible and raw_source_accuracy >= config.min_raw_source_accuracy_for_best
            eligible = eligible and raw_source_supported_macro_f1 >= config.min_raw_source_macro_f1_for_best
            eligible = eligible and min_source_accuracy >= config.min_source_accuracy_for_best
        if source_support.get("excel_track", 0) >= config.min_source_support_for_hard_gate:
            eligible = eligible and excel_source_supported_macro_f1 >= config.min_excel_source_macro_f1_for_best
        if enough_rd_raw_source:
            eligible = eligible and rd_raw_source_accuracy >= config.min_rd_raw_source_accuracy_for_best
        eligible = eligible and joint_minus_traj_on_rd >= config.min_joint_minus_traj_macro_f1_on_rd_for_best
        eligible = eligible and final_net_gain_rate >= config.min_final_net_gain_rate_on_rd_for_best
        eligible = eligible and final_harm_rate <= config.max_final_harm_rate_on_rd_for_best
        eligible = eligible and final_unknown_recall >= config.min_final_unknown_recall_for_best
        eligible = eligible and final_clutter_recall >= config.min_final_clutter_recall_for_best
        eligible = eligible and final_unknown_clutter_recall_mean >= config.min_final_unknown_clutter_recall_mean_for_best
        if config.require_final_not_worse_than_traj:
            eligible = eligible and final_delta >= max(
                -config.max_final_traj_f1_drop,
                config.min_final_minus_traj_macro_f1_for_best,
            )
    return score, eligible, {
        "final_macro_f1": final_macro,
        "accuracy": accuracy,
        "source_balanced_accuracy": source_balanced_accuracy,
        "min_source_accuracy": min_source_accuracy,
        "raw_tr_rd_accuracy": raw_source_accuracy,
        "source_balanced_macro_f1": source_balanced_macro_f1,
        "min_source_macro_f1": min_source_macro_f1,
        "source_balanced_supported_macro_f1": source_balanced_supported_macro_f1,
        "min_source_supported_macro_f1": min_source_supported_macro_f1,
        "raw_tr_rd_macro_f1": raw_source_macro_f1,
        "raw_tr_rd_supported_macro_f1": raw_source_supported_macro_f1,
        "excel_track_macro_f1": excel_source_macro_f1,
        "excel_track_supported_macro_f1": excel_source_supported_macro_f1,
        "rd_raw_tr_rd_accuracy": rd_raw_source_accuracy,
        "rd_source_balanced_accuracy_on_rd": rd_source_balanced_accuracy,
        "rd_min_source_accuracy_on_rd": rd_min_source_accuracy,
        "rd_source_balanced_macro_f1_on_rd": rd_source_balanced_macro_f1,
        "rd_min_source_macro_f1_on_rd": rd_min_source_macro_f1,
        "without_rd_accuracy": without_rd_accuracy,
        "traj_macro_f1": traj_macro,
        "final_minus_traj_macro_f1": final_delta,
        "joint_macro_f1_on_rd": joint_macro_on_rd,
        "joint_minus_traj_macro_f1_on_rd": joint_minus_traj_on_rd,
        "rd_macro_f1_on_rd": rd_macro,
        "rd_drone_bird_recall_mean": drone_bird,
        "rd_drone_recall": drone_recall,
        "rd_bird_recall": bird_recall,
        "rd_min_drone_bird_recall": min_drone_bird_recall,
        "rd_unknown_clutter_recall_mean": unknown_clutter,
        "rd_balloon_recall": balloon_recall,
        "rd_clutter_recall": clutter_recall,
        "rd_unknown_recall": unknown_recall,
        "rd_min_rare_recall": min_rare_rd_recall,
        "rd_harm_rate_on_rd": rd_harm_rate,
        "rd_net_gain_rate_on_rd": rd_net_gain_rate,
        "selection_penalty_joint_minus_traj_macro_f1_on_rd": joint_minus_traj_deficit,
        "selection_penalty_final_net_gain_rate_on_rd": final_net_gain_deficit,
        "selection_penalty_total": rd_negative_gain_penalty,
        "final_harm_rate_on_rd": final_harm_rate,
        "final_net_gain_rate_on_rd": final_net_gain_rate,
        "final_unknown_recall": final_unknown_recall,
        "final_clutter_recall": final_clutter_recall,
        "final_unknown_clutter_recall_mean": final_unknown_clutter_recall_mean,
        "parent_unknown_recall": parent_unknown_recall,
        "parent_clutter_recall": parent_clutter_recall,
        "parent_unknown_clutter_recall_mean": parent_unknown_clutter_recall_mean,
        "rd_residual_accuracy_on_rd": rd_residual_accuracy,
        "rd_unknown_clutter_detail_accuracy_on_residual_rd": residual_detail_accuracy,
        "rd_clutter_morphology_accuracy_on_residual_rd": clutter_morphology_accuracy,
        "residual_score_on_residual_mean": residual_score_on_residual,
        "source_support": source_support,
        "rd_source_support_on_rd": rd_source_support,
        "rd_class_support_on_rd": rd_class_support,
        "raw_source_gate_active": enough_raw_source if stage2_enabled else False,
        "rd_raw_source_gate_active": enough_rd_raw_source if stage2_enabled else False,
        "unknown_clutter_gate_active": bool(enough_unknown and enough_clutter) if stage2_enabled else False,
    }


def _parse_window_number(sample_id: str) -> int:
    marker = "|win"
    if marker not in sample_id:
        return 0
    suffix = sample_id.rsplit(marker, 1)[-1]
    digits = "".join(char for char in suffix if char.isdigit())
    return int(digits) if digits else 0


def _parent_sample_id(sample_id: str) -> str:
    marker = "|win"
    if marker in sample_id:
        return sample_id.rsplit(marker, 1)[0]
    return sample_id


def _load_window_metadata(data_root: str, index_name: str) -> dict[str, dict[str, object]]:
    if "window" not in str(index_name):
        return {}
    index_path = Path(data_root) / "processed" / index_name
    if not index_path.exists():
        return {}
    metadata: dict[str, dict[str, object]] = {}
    with index_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                continue
            metadata[sample_id] = {
                "parent_sample_id": str(row.get("parent_sample_id") or _parent_sample_id(sample_id)),
                "source_name": str(row.get("source_name", "")),
                "label_name": str(row.get("label_name", "")),
                "window_number": _parse_window_number(sample_id),
                "window_start_seconds": float(row.get("window_start_seconds") or 0.0),
                "window_end_seconds": float(row.get("window_end_seconds") or 0.0),
                "window_duration_seconds": float(row.get("window_duration_seconds") or 0.0),
                "window_point_count": int(float(row.get("window_point_count") or 0)),
                "window_mode": str(row.get("window_mode", "")),
            }
    return metadata


def _classification_metric_payload(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, object]:
    metrics = compute_classification_metrics(logits, labels, num_classes=num_classes)
    return {
        "accuracy": float(metrics.accuracy),
        "macro_precision": float(metrics.macro_precision),
        "macro_recall": float(metrics.macro_recall),
        "macro_f1": float(metrics.macro_f1),
        "confusion_matrix": compute_confusion_matrix(logits, labels, num_classes).tolist(),
    }


def _one_hot_logits_from_predictions(predictions: list[int], num_classes: int) -> torch.Tensor:
    if not predictions:
        return torch.zeros((0, num_classes), dtype=torch.float32)
    logits = torch.zeros((len(predictions), num_classes), dtype=torch.float32)
    logits[torch.arange(len(predictions)), torch.tensor(predictions, dtype=torch.long)] = 1.0
    return logits


def _weak_class_recall_summary(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    per_class = compute_per_class_metrics(logits, labels, LOW_ALTITUDE_CLASSES)
    by_label = {item.label_name: item for item in per_class}
    unknown = by_label.get("UnknownTarget")
    clutter = by_label.get("ClutterTarget")
    unknown_recall = float(unknown.recall) if unknown is not None else 0.0
    clutter_recall = float(clutter.recall) if clutter is not None else 0.0
    return {
        "unknown_support": int(unknown.support) if unknown is not None else 0,
        "clutter_support": int(clutter.support) if clutter is not None else 0,
        "unknown_recall": unknown_recall,
        "clutter_recall": clutter_recall,
        "unknown_clutter_recall_mean": 0.5 * (unknown_recall + clutter_recall),
    }


def _prediction_transition_summary(
    records: list[dict[str, object]],
) -> dict[str, object]:
    support = len(records)
    baseline_correct = sum(1 for item in records if bool(item.get("baseline_correct", False)))
    candidate_correct = sum(1 for item in records if bool(item.get("candidate_correct", False)))
    rescue_count = sum(
        1
        for item in records
        if not bool(item.get("baseline_correct", False)) and bool(item.get("candidate_correct", False))
    )
    harm_count = sum(
        1
        for item in records
        if bool(item.get("baseline_correct", False)) and not bool(item.get("candidate_correct", False))
    )
    changed_count = sum(1 for item in records if item.get("baseline_pred") != item.get("candidate_pred"))

    by_label: dict[str, dict[str, object]] = {}
    for item in records:
        label_name = str(item.get("true_label", ""))
        if not label_name:
            continue
        label_records = by_label.setdefault(
            label_name,
            {
                "support": 0,
                "baseline_correct_count": 0,
                "candidate_correct_count": 0,
                "rescue_count": 0,
                "harm_count": 0,
                "changed_count": 0,
            },
        )
        label_records["support"] = int(label_records["support"]) + 1
        baseline_ok = bool(item.get("baseline_correct", False))
        candidate_ok = bool(item.get("candidate_correct", False))
        if baseline_ok:
            label_records["baseline_correct_count"] = int(label_records["baseline_correct_count"]) + 1
        if candidate_ok:
            label_records["candidate_correct_count"] = int(label_records["candidate_correct_count"]) + 1
        if not baseline_ok and candidate_ok:
            label_records["rescue_count"] = int(label_records["rescue_count"]) + 1
        if baseline_ok and not candidate_ok:
            label_records["harm_count"] = int(label_records["harm_count"]) + 1
        if item.get("baseline_pred") != item.get("candidate_pred"):
            label_records["changed_count"] = int(label_records["changed_count"]) + 1

    for item in by_label.values():
        item_support = max(1, int(item["support"]))
        item["baseline_accuracy"] = float(int(item["baseline_correct_count"]) / item_support)
        item["candidate_accuracy"] = float(int(item["candidate_correct_count"]) / item_support)
        item["rescue_rate"] = float(int(item["rescue_count"]) / item_support)
        item["harm_rate"] = float(int(item["harm_count"]) / item_support)
        item["net_gain_rate"] = float((int(item["rescue_count"]) - int(item["harm_count"])) / item_support)

    denominator = max(1, support)
    return {
        "support": support,
        "baseline_correct_count": baseline_correct,
        "candidate_correct_count": candidate_correct,
        "rescue_count": rescue_count,
        "harm_count": harm_count,
        "net_gain_count": rescue_count - harm_count,
        "changed_count": changed_count,
        "baseline_accuracy": float(baseline_correct / denominator),
        "candidate_accuracy": float(candidate_correct / denominator),
        "rescue_rate": float(rescue_count / denominator),
        "harm_rate": float(harm_count / denominator),
        "net_gain_rate": float((rescue_count - harm_count) / denominator),
        "changed_rate": float(changed_count / denominator),
        "by_label": by_label,
    }


def _parent_window_aggregation_metrics(
    metrics: dict[str, object],
    window_metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    sample_ids = [str(item) for item in metrics.get("sample_ids", [])]
    if not sample_ids or not any("|win" in sample_id for sample_id in sample_ids):
        return {}
    logits = metrics.get("logits")
    labels = metrics.get("labels")
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return {}
    if logits.numel() == 0 or labels.numel() == 0:
        return {}
    logits = logits.detach().cpu().to(torch.float32)
    labels = labels.detach().cpu().to(torch.long)

    groups: dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        meta = window_metadata.get(sample_id, {})
        parent_id = str(meta.get("parent_sample_id") or _parent_sample_id(sample_id))
        groups.setdefault(parent_id, []).append(index)
    if not groups:
        return {}

    parent_labels: list[int] = []
    parent_sources: list[str] = []
    parent_has_rd: list[bool] = []
    mean_logits: list[torch.Tensor] = []
    vote_logits: list[torch.Tensor] = []
    first_logits: list[torch.Tensor] = []
    window_counts: list[int] = []
    stable_latencies: list[float] = []
    stable_predictions: list[int] = []
    mean_predictions: list[int] = []
    vote_predictions: list[int] = []
    first_predictions: list[int] = []
    parent_records: list[dict[str, object]] = []
    window_to_parent_records: list[dict[str, object]] = []
    first_to_mean_records: list[dict[str, object]] = []
    majority_to_mean_records: list[dict[str, object]] = []
    stable_to_mean_records: list[dict[str, object]] = []
    flip_counts: list[int] = []
    flip_rates: list[float] = []
    num_classes = len(LOW_ALTITUDE_CLASSES)
    has_rd_values = [bool(item) for item in metrics.get("has_rd", [])]
    source_values = [str(item) for item in metrics.get("source_names", [])]
    weak_label_indices = {
        LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
        LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
    }

    for parent_id, raw_indices in groups.items():
        indices = sorted(
            raw_indices,
            key=lambda item: (
                float(window_metadata.get(sample_ids[item], {}).get("window_start_seconds", _parse_window_number(sample_ids[item]))),
                _parse_window_number(sample_ids[item]),
            ),
        )
        label_values = labels[indices].to(torch.long)
        parent_start = min(
            float(window_metadata.get(sample_ids[item], {}).get("window_start_seconds", 0.0))
            for item in indices
        )
        label_counts = torch.bincount(label_values, minlength=num_classes)
        parent_label = int(label_counts.argmax().item())
        parent_label_name = LOW_ALTITUDE_CLASSES[parent_label]
        parent_logits = logits[indices]
        parent_mean_logits = parent_logits.mean(dim=0)
        window_preds = parent_logits.argmax(dim=-1)
        window_pred_indices = [int(item) for item in window_preds.tolist()]
        window_pred_names = [LOW_ALTITUDE_CLASSES[item] for item in window_pred_indices]
        pred_counts = torch.bincount(window_preds, minlength=num_classes).to(torch.float32)
        vote_score = pred_counts + parent_mean_logits.to(torch.float32) * 1e-6
        mean_pred = int(parent_mean_logits.argmax().item())
        vote_pred = int(vote_score.argmax().item())
        first_pred = int(parent_logits[0].argmax().item())
        parent_labels.append(parent_label)
        mean_logits.append(parent_mean_logits)
        vote_logits.append(vote_score)
        first_logits.append(parent_logits[0])
        mean_predictions.append(mean_pred)
        vote_predictions.append(vote_pred)
        first_predictions.append(first_pred)
        window_counts.append(len(indices))
        parent_source = source_values[indices[0]] if indices[0] < len(source_values) else ""
        has_rd_parent = any(has_rd_values[idx] for idx in indices if idx < len(has_rd_values))
        parent_sources.append(parent_source)
        parent_has_rd.append(has_rd_parent)

        flip_count = sum(
            1
            for current, previous in zip(window_pred_indices[1:], window_pred_indices[:-1])
            if current != previous
        )
        transition_count = max(1, len(window_pred_indices) - 1)
        flip_rate = float(flip_count / transition_count)
        flip_counts.append(int(flip_count))
        flip_rates.append(flip_rate)

        stable_index = 0
        for offset, pred in enumerate(window_pred_indices):
            if all(other == pred for other in window_pred_indices[offset:]):
                stable_index = offset
                break
        stable_pred = int(window_pred_indices[stable_index])
        stable_predictions.append(stable_pred)
        stable_sample_id = sample_ids[indices[stable_index]]
        stable_meta = window_metadata.get(stable_sample_id, {})
        latency = float(stable_meta.get("window_end_seconds") or stable_index + 1) - parent_start
        stable_latencies.append(latency)

        for index, baseline_pred in zip(indices, window_pred_indices):
            window_label = int(labels[index].item())
            if window_label not in weak_label_indices:
                continue
            window_to_parent_records.append(
                {
                    "parent_sample_id": str(parent_id),
                    "sample_id": sample_ids[index],
                    "true_label": LOW_ALTITUDE_CLASSES[window_label],
                    "baseline_pred": LOW_ALTITUDE_CLASSES[int(baseline_pred)],
                    "candidate_pred": LOW_ALTITUDE_CLASSES[mean_pred],
                    "baseline_correct": int(baseline_pred) == window_label,
                    "candidate_correct": mean_pred == window_label,
                }
            )

        if parent_label in weak_label_indices:
            for target, baseline_pred in [
                (first_to_mean_records, first_pred),
                (majority_to_mean_records, vote_pred),
                (stable_to_mean_records, stable_pred),
            ]:
                target.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "true_label": parent_label_name,
                        "baseline_pred": LOW_ALTITUDE_CLASSES[int(baseline_pred)],
                        "candidate_pred": LOW_ALTITUDE_CLASSES[mean_pred],
                        "baseline_correct": int(baseline_pred) == parent_label,
                        "candidate_correct": mean_pred == parent_label,
                    }
                )

        parent_records.append(
            {
                "parent_sample_id": str(parent_id),
                "source_name": parent_source,
                "true_label": parent_label_name,
                "window_count": int(len(indices)),
                "has_rd": bool(has_rd_parent),
                "prediction_flip_count": int(flip_count),
                "prediction_flip_rate": flip_rate,
                "stable_latency_seconds": float(latency),
                "first_window_pred": LOW_ALTITUDE_CLASSES[first_pred],
                "stable_decision_pred": LOW_ALTITUDE_CLASSES[stable_pred],
                "majority_vote_pred": LOW_ALTITUDE_CLASSES[vote_pred],
                "mean_logit_pred": LOW_ALTITUDE_CLASSES[mean_pred],
                "first_window_correct": bool(first_pred == parent_label),
                "stable_decision_correct": bool(stable_pred == parent_label),
                "majority_vote_correct": bool(vote_pred == parent_label),
                "mean_logit_correct": bool(mean_pred == parent_label),
                "window_predictions": window_pred_names,
            }
        )

    parent_label_tensor = torch.tensor(parent_labels, dtype=torch.long)
    mean_logits_tensor = torch.stack(mean_logits)
    vote_logits_tensor = torch.stack(vote_logits)
    first_logits_tensor = torch.stack(first_logits)
    stable_logits_tensor = _one_hot_logits_from_predictions(stable_predictions, num_classes)
    mean_metrics = _classification_metric_payload(mean_logits_tensor, parent_label_tensor, num_classes)
    vote_metrics = _classification_metric_payload(vote_logits_tensor, parent_label_tensor, num_classes)
    first_metrics = _classification_metric_payload(first_logits_tensor, parent_label_tensor, num_classes)
    stable_metrics = _classification_metric_payload(stable_logits_tensor, parent_label_tensor, num_classes)
    parent_has_rd_tensor = torch.tensor(parent_has_rd, dtype=torch.bool)
    by_source: dict[str, dict[str, float | int]] = {}
    for source_name in sorted(set(parent_sources)):
        source_mask = torch.tensor([item == source_name for item in parent_sources], dtype=torch.bool)
        if not bool(source_mask.any()):
            continue
        source_metrics = compute_classification_metrics(
            mean_logits_tensor[source_mask],
            parent_label_tensor[source_mask],
            num_classes=num_classes,
        )
        by_source[source_name] = {
            "count": int(source_mask.sum().item()),
            "mean_logit_accuracy": float(source_metrics.accuracy),
            "mean_logit_macro_f1": float(source_metrics.macro_f1),
        }
    per_class = [
        {
            "label_name": item.label_name,
            "support": item.support,
            "precision": item.precision,
            "recall": item.recall,
            "f1": item.f1,
        }
        for item in compute_per_class_metrics(mean_logits_tensor, parent_label_tensor, LOW_ALTITUDE_CLASSES)
    ]
    weak_class_recall = {
        "mean_logit": _weak_class_recall_summary(mean_logits_tensor, parent_label_tensor),
        "majority_vote": _weak_class_recall_summary(vote_logits_tensor, parent_label_tensor),
        "early_window": _weak_class_recall_summary(first_logits_tensor, parent_label_tensor),
        "stable_decision": _weak_class_recall_summary(stable_logits_tensor, parent_label_tensor),
    }
    flip_counts_array = np.asarray(flip_counts, dtype=np.float64)
    flip_rates_array = np.asarray(flip_rates, dtype=np.float64)
    top_flip_parent_records = sorted(
        parent_records,
        key=lambda item: (
            int(item.get("prediction_flip_count", 0)),
            float(item.get("prediction_flip_rate", 0.0)),
            int(item.get("window_count", 0)),
        ),
        reverse=True,
    )[:20]
    hard_pair_parent_records = [
        item
        for item in parent_records
        if item.get("true_label") in {"UnknownTarget", "ClutterTarget"}
    ][:20]
    parent_weak_summary = weak_class_recall["mean_logit"]
    return {
        "parent_count": len(groups),
        "window_count": len(sample_ids),
        "mean_windows_per_parent": float(np.mean(window_counts)) if window_counts else 0.0,
        "max_windows_per_parent": int(max(window_counts)) if window_counts else 0,
        "with_rd_parent_count": int(parent_has_rd_tensor.sum().item()),
        "without_rd_parent_count": int((~parent_has_rd_tensor).sum().item()),
        "parent_unknown_recall": float(parent_weak_summary.get("unknown_recall", 0.0)),
        "parent_clutter_recall": float(parent_weak_summary.get("clutter_recall", 0.0)),
        "parent_unknown_clutter_recall_mean": float(parent_weak_summary.get("unknown_clutter_recall_mean", 0.0)),
        "mean_logit": mean_metrics,
        "majority_vote": vote_metrics,
        "early_window": first_metrics,
        "stable_decision": {
            **stable_metrics,
            "latency_seconds_mean": float(np.mean(stable_latencies)) if stable_latencies else 0.0,
            "latency_seconds_p50": float(np.quantile(stable_latencies, 0.50)) if stable_latencies else 0.0,
            "latency_seconds_p90": float(np.quantile(stable_latencies, 0.90)) if stable_latencies else 0.0,
        },
        "prediction_flip_diagnostics": {
            "parent_count": len(parent_records),
            "flipped_parent_count": int(sum(1 for item in flip_counts if item > 0)),
            "flipped_parent_rate": float(sum(1 for item in flip_counts if item > 0) / max(1, len(flip_counts))),
            "flip_count_mean": float(np.mean(flip_counts_array)) if flip_counts else 0.0,
            "flip_count_p50": float(np.quantile(flip_counts_array, 0.50)) if flip_counts else 0.0,
            "flip_count_p90": float(np.quantile(flip_counts_array, 0.90)) if flip_counts else 0.0,
            "flip_count_max": int(max(flip_counts)) if flip_counts else 0,
            "flip_rate_mean": float(np.mean(flip_rates_array)) if flip_rates else 0.0,
            "flip_rate_p50": float(np.quantile(flip_rates_array, 0.50)) if flip_rates else 0.0,
            "flip_rate_p90": float(np.quantile(flip_rates_array, 0.90)) if flip_rates else 0.0,
            "flip_rate_max": float(max(flip_rates)) if flip_rates else 0.0,
        },
        "weak_class_recall": weak_class_recall,
        "parent_unknown_clutter_rescue_harm": {
            "first_window_to_mean_logit": _prediction_transition_summary(first_to_mean_records),
            "majority_vote_to_mean_logit": _prediction_transition_summary(majority_to_mean_records),
            "stable_decision_to_mean_logit": _prediction_transition_summary(stable_to_mean_records),
        },
        "unknown_clutter_hard_pair": {
            "window_to_parent_mean_logit": _prediction_transition_summary(window_to_parent_records),
            "parent_first_to_mean_logit": _prediction_transition_summary(first_to_mean_records),
            "parent_majority_vote_to_mean_logit": _prediction_transition_summary(majority_to_mean_records),
            "parent_stable_to_mean_logit": _prediction_transition_summary(stable_to_mean_records),
        },
        "top_flip_parent_records": top_flip_parent_records,
        "hard_pair_parent_records": hard_pair_parent_records,
        "by_source": by_source,
        "per_class_mean_logit": per_class,
    }


def _train_batch_distribution_summary(
    sample_ids: list[str],
    source_names: list[str],
    label_names: list[str],
    has_rd: list[bool],
) -> dict[str, object]:
    total = max(1, len(sample_ids))
    parent_ids = [_parent_sample_id(str(sample_id)) for sample_id in sample_ids]
    source_counter = Counter(str(item) for item in source_names)
    label_counter = Counter(str(item) for item in label_names)
    parent_counter = Counter(parent_ids)
    rare_labels = {"UnknownTarget", "ClutterTarget"}
    rare_count = sum(1 for item in label_names if str(item) in rare_labels)
    top10_parent_fraction = (
        float(sum(count for _parent_id, count in parent_counter.most_common(10)) / total)
        if parent_counter
        else 0.0
    )
    raw_fraction = float(source_counter.get("raw_tr_rd", 0) / total)
    effective_unique_parent_count = 0.0
    if parent_counter:
        probabilities = np.asarray([count / total for count in parent_counter.values()], dtype=np.float64)
        effective_unique_parent_count = float(1.0 / np.sum(np.square(probabilities)))
    return {
        "rare_class_fraction_in_train_batches": float(rare_count / total),
        "rd_window_fraction_in_train_batches": float(sum(1 for item in has_rd if bool(item)) / total),
        "stage2_source_fraction": {
            key: float(value / total)
            for key, value in sorted(source_counter.items())
        },
        "stage2_label_fraction": {
            key: float(value / total)
            for key, value in sorted(label_counter.items())
        },
        "stage2_raw_tr_rd_fraction": raw_fraction,
        "stage2_top10_parent_fraction": top10_parent_fraction,
        "stage2_effective_unique_parent_count": effective_unique_parent_count,
    }


def _parent_weak_class_metrics(parent_aggregation_metrics: dict[str, object]) -> dict[str, float]:
    per_class = parent_aggregation_metrics.get("per_class_mean_logit", [])
    if not isinstance(per_class, list):
        return {
            "parent_unknown_recall": 0.0,
            "parent_clutter_recall": 0.0,
            "parent_unknown_clutter_recall_mean": 0.0,
        }
    recall_by_label: dict[str, float] = {}
    for item in per_class:
        if not isinstance(item, dict):
            continue
        recall_by_label[str(item.get("label_name", ""))] = float(item.get("recall", 0.0))
    unknown_recall = float(recall_by_label.get("UnknownTarget", 0.0))
    clutter_recall = float(recall_by_label.get("ClutterTarget", 0.0))
    return {
        "parent_unknown_recall": unknown_recall,
        "parent_clutter_recall": clutter_recall,
        "parent_unknown_clutter_recall_mean": 0.5 * (unknown_recall + clutter_recall),
    }


def _expert1_release_gate(
    metrics: dict[str, object],
    config: Expert1TrainConfig,
) -> tuple[bool, dict[str, float], list[str]]:
    final_macro = float(metrics.get("macro_f1", 0.0))
    traj_macro = float(metrics.get("traj_macro_f1", 0.0))
    final_delta = final_macro - traj_macro
    joint_minus_traj_on_rd = float(metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0))
    final_net_gain_rate = float(metrics.get("final_net_gain_rate_on_rd", 0.0))
    final_harm_rate = float(metrics.get("final_harm_rate_on_rd", 0.0))
    rd_macro_on_rd = float(metrics.get("rd_macro_f1_on_rd", 0.0))
    without_rd_accuracy = float(metrics.get("without_rd_accuracy", 0.0))
    final_per_class_recall = dict(metrics.get("final_per_class_recall", {}))
    final_unknown_recall = float(final_per_class_recall.get("UnknownTarget", 0.0))
    final_clutter_recall = float(final_per_class_recall.get("ClutterTarget", 0.0))
    final_unknown_clutter_mean = 0.5 * (final_unknown_recall + final_clutter_recall)
    parent_unknown_recall = float(metrics.get("parent_unknown_recall", 0.0))
    parent_clutter_recall = float(metrics.get("parent_clutter_recall", 0.0))
    parent_unknown_clutter_mean = float(metrics.get("parent_unknown_clutter_recall_mean", 0.0))
    parent_macro_f1 = float(metrics.get("parent_mean_logit_macro_f1", parent_unknown_clutter_mean))
    reasons: list[str] = []
    if final_macro < config.min_release_macro_f1:
        reasons.append("macro_f1")
    if final_delta < config.min_release_final_minus_traj_macro_f1:
        reasons.append("final_minus_traj_macro_f1")
    if joint_minus_traj_on_rd < config.min_release_joint_minus_traj_macro_f1_on_rd:
        reasons.append("joint_minus_traj_macro_f1_on_rd")
    if rd_macro_on_rd < config.min_release_rd_evidence_macro_f1_on_rd:
        reasons.append("rd_macro_f1_on_rd")
    if final_net_gain_rate < config.min_release_final_net_gain_rate_on_rd:
        reasons.append("final_net_gain_rate_on_rd")
    if final_harm_rate > config.max_release_final_harm_rate_on_rd:
        reasons.append("final_harm_rate_on_rd")
    if parent_macro_f1 < config.min_release_parent_macro_f1:
        reasons.append("parent_macro_f1")
    if final_unknown_recall < config.min_release_unknown_recall:
        reasons.append("final_unknown_recall")
    if final_clutter_recall < config.min_release_clutter_recall:
        reasons.append("final_clutter_recall")
    if final_unknown_clutter_mean < 0.5 * (config.min_release_unknown_recall + config.min_release_clutter_recall):
        reasons.append("final_unknown_clutter_recall_mean")
    if without_rd_accuracy < config.min_release_without_rd_accuracy:
        reasons.append("without_rd_accuracy")
    eligible = not reasons
    metrics_summary = {
        "release_macro_f1": final_macro,
        "release_final_minus_traj_macro_f1": final_delta,
        "release_joint_minus_traj_macro_f1_on_rd": joint_minus_traj_on_rd,
        "release_rd_macro_f1_on_rd": rd_macro_on_rd,
        "release_final_net_gain_rate_on_rd": final_net_gain_rate,
        "release_final_harm_rate_on_rd": final_harm_rate,
        "release_parent_macro_f1": parent_macro_f1,
        "release_parent_unknown_clutter_recall_mean": parent_unknown_clutter_mean,
        "release_final_unknown_recall": final_unknown_recall,
        "release_final_clutter_recall": final_clutter_recall,
        "release_final_unknown_clutter_recall_mean": final_unknown_clutter_mean,
        "release_without_rd_accuracy": without_rd_accuracy,
    }
    return eligible, metrics_summary, reasons


def _select_final_test_checkpoint(
    *,
    config: Expert1TrainConfig | None = None,
    best_path: Path,
    best_epoch: int,
    best_score: float,
    best_reason: str,
    best_components: dict[str, float],
    best_selection_eligible: bool,
    fallback_path: Path,
    fallback_epoch: int,
    fallback_score: float,
    fallback_components: dict[str, float],
    checkpoint_registry: dict[str, dict[str, object]],
    candidate_paths: dict[str, Path] | None = None,
) -> dict[str, object]:
    candidates = [
        {
            "name": "best_selection",
            "path": best_path,
            "epoch": int(best_epoch),
            "score": float(best_score),
            "reason": best_reason,
            "components": dict(best_components),
            "selection_eligible": bool(best_selection_eligible),
        }
    ]
    fallback_registry = checkpoint_registry.get("fallback_candidate", {})
    candidates.append(
        {
            "name": "fallback_candidate",
            "path": fallback_path,
            "epoch": int(fallback_registry.get("epoch", fallback_epoch)),
            "score": float(fallback_registry.get("score", fallback_score)),
            "reason": str(fallback_registry.get("selection_reason", "stage2_v2_joint_composite")),
            "components": dict(fallback_registry.get("selection_components", fallback_components)),
            "selection_eligible": bool(fallback_registry.get("selection_eligible", False)),
        }
    )
    if candidate_paths:
        for name, path in candidate_paths.items():
            if name in {"best_selection", "fallback_candidate"}:
                continue
            registry = checkpoint_registry.get(name, {})
            if not registry:
                continue
            candidates.append(
                {
                    "name": name,
                    "path": path,
                    "epoch": int(registry.get("epoch", -1)),
                    "score": float(registry.get("score", float("-inf"))),
                    "reason": str(registry.get("selection_reason", "")),
                    "components": dict(registry.get("selection_components", {})),
                    "selection_eligible": bool(registry.get("selection_eligible", False)),
                }
            )
    existing_candidates = [candidate for candidate in candidates if Path(candidate["path"]).exists()]
    if not existing_candidates:
        return {
            "name": "best_selection",
            "path": best_path,
            "epoch": int(best_epoch),
            "score": float(best_score),
            "reason": best_reason,
            "components": dict(best_components),
            "selection_eligible": bool(best_selection_eligible),
        }
    if config is None:
        selected = {
            **existing_candidates[0],
            "release_proxy_score": float(existing_candidates[0]["score"]),
        }
        for candidate in existing_candidates[1:]:
            score = float(candidate["score"])
            if score > float(selected["release_proxy_score"]):
                selected = {**candidate, "release_proxy_score": score}
        return selected

    def _release_proxy_score(candidate: dict[str, object]) -> float:
        components = dict(candidate.get("components", {}))
        final_macro = float(components.get("final_macro_f1", 0.0))
        traj_macro = float(components.get("traj_macro_f1", 0.0))
        final_delta = float(components.get("final_minus_traj_macro_f1", final_macro - traj_macro))
        joint_gain = float(components.get("joint_minus_traj_macro_f1_on_rd", 0.0))
        final_net_gain = float(components.get("final_net_gain_rate_on_rd", 0.0))
        final_harm = float(components.get("final_harm_rate_on_rd", 0.0))
        rd_macro = float(components.get("rd_macro_f1_on_rd", 0.0))
        weak_mean = float(components.get("final_unknown_clutter_recall_mean", 0.0))
        min_weak = min(
            float(components.get("final_unknown_recall", 0.0)),
            float(components.get("final_clutter_recall", 0.0)),
        )
        source_supported = float(components.get("source_balanced_supported_macro_f1", 0.0))
        joint_minus_traj_deficit = max(0.0, -(joint_gain + 0.0100))
        final_net_gain_deficit = max(0.0, -final_net_gain)
        eligibility_bonus = 0.02 if bool(candidate.get("selection_eligible", False)) else 0.0
        return float(
            1.50 * final_macro
            + 0.60 * final_delta
            + 1.60 * joint_gain
            + 2.00 * final_net_gain
            - 0.80 * final_harm
            + 0.20 * rd_macro
            + 0.22 * weak_mean
            + 0.10 * min_weak
            + 0.12 * source_supported
            + eligibility_bonus
            - 1.20 * joint_minus_traj_deficit
            - 0.80 * final_net_gain_deficit
        )

    ranked = []
    for candidate in existing_candidates:
        proxy_score = _release_proxy_score(candidate)
        candidate_score = float(candidate.get("score", float("-inf")))
        ranked.append(
            (
                proxy_score,
                int(bool(candidate.get("selection_eligible", False))),
                candidate_score,
                int(candidate.get("epoch", -1)),
                {**candidate, "release_proxy_score": proxy_score},
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return ranked[0][4]


def _checkpoint_eval_runtime(epoch: int, config: Expert1TrainConfig) -> tuple[bool, float, int]:
    stage2_enabled = int(epoch) > int(config.stage1_traj_only_epochs)
    stage2_scale = _stage2_scale(int(epoch), config)
    stage2_epoch = max(0, int(epoch) - int(config.stage1_traj_only_epochs))
    return stage2_enabled, stage2_scale, stage2_epoch


def _load_checkpoint_into_modules(
    checkpoint_path: Path,
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    representation_builder.load_state_dict(checkpoint["representation_builder_state_dict"])
    rd_proj.load_state_dict(checkpoint["rd_proj_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if hasattr(model, "set_precision_logit_bias"):
        model.set_precision_logit_bias(_precision_logit_bias_from_checkpoint(checkpoint))
    return checkpoint


def _probe_result_metric_fields(item: dict[str, object]) -> dict[str, float]:
    return {
        "scale": float(item.get("scale", 0.0)),
        "macro_f1": float(item.get("macro_f1", 0.0)),
        "traj_macro_f1": float(item.get("traj_macro_f1", 0.0)),
        "joint_minus_traj_macro_f1_on_rd": float(item.get("joint_minus_traj_macro_f1_on_rd", 0.0)),
        "final_net_gain_rate_on_rd": float(item.get("final_net_gain_rate_on_rd", 0.0)),
        "final_harm_rate_on_rd": float(item.get("final_harm_rate_on_rd", 0.0)),
        "final_unknown_recall": float(item.get("final_unknown_recall", 0.0)),
        "final_clutter_recall": float(item.get("final_clutter_recall", 0.0)),
        "source_balanced_supported_macro_f1": float(item.get("source_balanced_supported_macro_f1", 0.0)),
    }


def _diagnostic_checkpoint_probe_fields(metrics: dict[str, object]) -> dict[str, object]:
    probe = metrics.get("joint_logit_delta_probe", {})
    if not isinstance(probe, dict):
        return {}
    results = [item for item in probe.get("results", []) if isinstance(item, dict)]
    if not results:
        return {}
    fields: dict[str, object] = {
        "joint_logit_delta_probe_selected_scale": float(probe.get("selected_scale", metrics.get("selected_joint_logit_delta_scale", 0.0))),
        "joint_logit_delta_probe_best_scale": float(probe.get("best_scale", probe.get("selected_scale", metrics.get("selected_joint_logit_delta_scale", 0.0)))),
    }

    def _find_scale(scale: float) -> dict[str, object] | None:
        for item in results:
            if abs(float(item.get("scale", 0.0)) - float(scale)) <= 1.0e-6:
                return item
        return None

    best_item = _find_scale(float(fields["joint_logit_delta_probe_best_scale"]))
    if best_item is not None:
        for key, value in _probe_result_metric_fields(best_item).items():
            fields[f"joint_logit_delta_probe_best_{key}"] = value

    parent_macro = float(metrics.get("parent_mean_logit_macro_f1", 0.0))
    conservative_items = [
        item for item in results
        if float(item.get("scale", 0.0)) <= 0.5000001
    ]
    if not conservative_items:
        conservative_items = results
    conservative_best: dict[str, object] | None = None
    conservative_key: tuple[int, float, float, float, float, float, float, float] | None = None
    for item in conservative_items:
        item_fields = _probe_result_metric_fields(item)
        threshold_record = {
            **item_fields,
            "parent_mean_logit_macro_f1": parent_macro,
        }
        failures = _precision_candidate_threshold_failures(threshold_record)
        threshold_gaps = _precision_candidate_threshold_gaps(threshold_record)
        score = _precision_candidate_score(threshold_record, failures)
        key = (
            -len(failures),
            score,
            float(item_fields.get("macro_f1", 0.0)),
            float(item_fields.get("final_clutter_recall", 0.0)),
            min(
                float(item_fields.get("final_unknown_recall", 0.0)),
                float(item_fields.get("final_clutter_recall", 0.0)),
            ),
            float(item_fields.get("final_net_gain_rate_on_rd", 0.0)),
            -float(item_fields.get("final_harm_rate_on_rd", 0.0)),
            -float(item_fields.get("scale", 0.0)),
        )
        if conservative_key is None or key > conservative_key:
            conservative_key = key
            conservative_best = {
                **item_fields,
                "score": score,
                "precision_threshold_failures": failures,
                "precision_threshold_gaps": threshold_gaps,
                **threshold_gaps,
            }
    if conservative_best is not None:
        for key, value in conservative_best.items():
            fields[f"conservative_precision_{key}"] = value
    return fields


def _diagnostic_checkpoint_test_record(
    checkpoint_name: str,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, object],
    metrics: dict[str, object],
    *,
    is_selected_checkpoint: bool,
    selection_score: float | None = None,
) -> dict[str, object]:
    per_class_recall = metrics.get("final_per_class_recall", {})
    if not isinstance(per_class_recall, dict):
        per_class_recall = {}
    probe_fields = _diagnostic_checkpoint_probe_fields(metrics)
    return {
        "checkpoint_name": str(checkpoint_name),
        "status": "ok",
        "path": _rel(checkpoint_path),
        "epoch": int(checkpoint_payload.get("epoch", -1)),
        "selection_reason": str(checkpoint_payload.get("selection_reason", "")),
        "selection_eligible": bool(checkpoint_payload.get("selection_eligible", False)),
        "selection_score": (float(selection_score) if selection_score is not None else None),
        "is_selected_checkpoint": bool(is_selected_checkpoint),
        "stage2_enabled": bool(
            int(checkpoint_payload.get("epoch", -1)) > int(checkpoint_payload.get("config", {}).get("stage1_traj_only_epochs", 0))
        ),
        "selected_joint_logit_delta_scale": float(
            checkpoint_payload.get(
                "selected_joint_logit_delta_scale",
                checkpoint_payload.get("config", {}).get("joint_logit_delta_scale", 1.0),
            )
        ),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "traj_macro_f1": float(metrics.get("traj_macro_f1", 0.0)),
        "joint_macro_f1_on_rd": float(metrics.get("joint_macro_f1_on_rd", 0.0)),
        "joint_minus_traj_macro_f1_on_rd": float(metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0)),
        "rd_macro_f1_on_rd": float(metrics.get("rd_macro_f1_on_rd", 0.0)),
        "final_net_gain_rate_on_rd": float(metrics.get("final_net_gain_rate_on_rd", 0.0)),
        "final_harm_rate_on_rd": float(metrics.get("final_harm_rate_on_rd", 0.0)),
        "final_unknown_recall": float(per_class_recall.get("UnknownTarget", 0.0)),
        "final_clutter_recall": float(per_class_recall.get("ClutterTarget", 0.0)),
        "parent_mean_logit_macro_f1": float(metrics.get("parent_mean_logit_macro_f1", 0.0)),
        "parent_unknown_recall": float(metrics.get("parent_unknown_recall", 0.0)),
        "parent_clutter_recall": float(metrics.get("parent_clutter_recall", 0.0)),
        "parent_unknown_clutter_recall_mean": float(metrics.get("parent_unknown_clutter_recall_mean", 0.0)),
        "source_balanced_supported_macro_f1": float(metrics.get("source_balanced_supported_macro_f1", 0.0)),
        "min_source_supported_macro_f1": float(metrics.get("min_source_supported_macro_f1", 0.0)),
        **probe_fields,
    }


def _diagnostic_checkpoint_test_recommendation(
    records: list[dict[str, object]],
) -> dict[str, object]:
    best: dict[str, object] | None = None
    best_key: tuple[float, float, float, float, int] | None = None
    for record in records:
        if str(record.get("status", "ok")) != "ok":
            continue
        macro_f1 = float(record.get("macro_f1", 0.0))
        traj_macro_f1 = float(record.get("traj_macro_f1", 0.0))
        final_delta = macro_f1 - traj_macro_f1
        joint_gain = float(record.get("joint_minus_traj_macro_f1_on_rd", 0.0))
        rd_macro = float(record.get("rd_macro_f1_on_rd", 0.0))
        final_net_gain = float(record.get("final_net_gain_rate_on_rd", 0.0))
        final_harm = float(record.get("final_harm_rate_on_rd", 0.0))
        weak_mean = 0.5 * (
            float(record.get("final_unknown_recall", 0.0))
            + float(record.get("final_clutter_recall", 0.0))
        )
        source_supported = float(record.get("source_balanced_supported_macro_f1", 0.0))
        proxy_score = float(
            2.00 * macro_f1
            + 0.40 * final_delta
            + 0.40 * joint_gain
            + 0.70 * final_net_gain
            - 0.40 * final_harm
            + 0.15 * weak_mean
            + 0.08 * source_supported
        )
        key = (
            proxy_score,
            macro_f1,
            final_net_gain,
            joint_gain,
            int(record.get("epoch", -1)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "checkpoint_name": str(record.get("checkpoint_name", "")),
                "path": str(record.get("path", "")),
                "epoch": int(record.get("epoch", -1)),
                "score": proxy_score,
                "macro_f1": macro_f1,
                "traj_macro_f1": traj_macro_f1,
                "joint_minus_traj_macro_f1_on_rd": joint_gain,
                "rd_macro_f1_on_rd": rd_macro,
                "final_net_gain_rate_on_rd": final_net_gain,
                "final_harm_rate_on_rd": final_harm,
                "final_unknown_recall": float(record.get("final_unknown_recall", 0.0)),
                "final_clutter_recall": float(record.get("final_clutter_recall", 0.0)),
                "source_balanced_supported_macro_f1": source_supported,
                "selection_eligible": bool(record.get("selection_eligible", False)),
                "is_selected_checkpoint": bool(record.get("is_selected_checkpoint", False)),
                "recommendation_basis": "diagnostic_test_release_proxy",
            }
    return best or {}


_PRECISION_CANDIDATE_THRESHOLDS = {
    "macro_f1": 0.8000,
    "final_net_gain_rate_on_rd": 0.0000,
    "final_harm_rate_on_rd": 0.0200,
    "final_unknown_recall": 0.7000,
    "final_clutter_recall": 0.5000,
    "parent_mean_logit_macro_f1": 0.8000,
    "joint_minus_traj_macro_f1_on_rd": -0.0500,
}


def _float_metric(metrics: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _rd_joint_bottleneck_summary(metrics: dict[str, object], config: Expert1TrainConfig) -> dict[str, object]:
    labels_obj = metrics.get("labels")
    tensor_keys = ["logits", "traj_logits", "joint_logits", "rd_logits"]
    has_tensor_view = isinstance(labels_obj, torch.Tensor) and all(
        isinstance(metrics.get(key), torch.Tensor) for key in tensor_keys
    )
    hard_pair_report = metrics.get("rd_hard_pair_report", {})
    rd_time_sync_metrics = metrics.get("rd_time_sync_metrics", {})
    source_rows: list[dict[str, object]] = []
    with_rd_count = 0

    if has_tensor_view:
        labels = labels_obj
        final_logits = metrics["logits"]
        traj_logits = metrics["traj_logits"]
        joint_logits = metrics["joint_logits"]
        rd_logits = metrics["rd_logits"]
        source_names = [str(item) for item in metrics.get("source_names", [])]
        has_rd_values = [bool(item) for item in metrics.get("has_rd", [])]
        if len(has_rd_values) != int(labels.shape[0]):
            has_rd_values = [False for _ in range(int(labels.shape[0]))]
        has_rd_tensor = torch.tensor(has_rd_values, dtype=torch.bool)
        with_rd_count = int(has_rd_tensor.sum().item())
        if with_rd_count > 0:
            hard_pair_report = _hard_pair_report(final_logits, traj_logits, rd_logits, labels, has_rd_tensor)
            statuses = [str(item or "unavailable") for item in metrics.get("rd_time_sync_status", [])]
            if len(statuses) == int(labels.shape[0]):
                rd_time_sync_metrics = _rd_time_sync_metric_dict(
                    final_logits,
                    traj_logits,
                    joint_logits,
                    rd_logits,
                    labels,
                    statuses,
                    has_rd_tensor,
                )
            final_source = _source_metric_dict(final_logits, labels, source_names, mask=has_rd_tensor)
            traj_source = _source_metric_dict(traj_logits, labels, source_names, mask=has_rd_tensor)
            joint_source = _source_metric_dict(joint_logits, labels, source_names, mask=has_rd_tensor)
            rd_source = _source_metric_dict(rd_logits, labels, source_names, mask=has_rd_tensor)
            support_by_source = _source_support_dict(source_names, mask=has_rd_tensor)
            for source_name, support in support_by_source.items():
                traj_f1 = float(traj_source.get(source_name, {}).get("macro_f1", 0.0))
                joint_f1 = float(joint_source.get(source_name, {}).get("macro_f1", 0.0))
                final_f1 = float(final_source.get(source_name, {}).get("macro_f1", 0.0))
                rd_f1 = float(rd_source.get(source_name, {}).get("macro_f1", 0.0))
                source_rows.append(
                    {
                        "source_name": source_name,
                        "support_on_rd": int(support),
                        "traj_macro_f1_on_rd": traj_f1,
                        "joint_macro_f1_on_rd": joint_f1,
                        "joint_minus_traj_macro_f1_on_rd": float(joint_f1 - traj_f1),
                        "rd_macro_f1_on_rd": rd_f1,
                        "final_macro_f1_on_rd": final_f1,
                        "final_minus_traj_macro_f1_on_rd": float(final_f1 - traj_f1),
                    }
                )
    else:
        has_rd_values = [bool(item) for item in metrics.get("has_rd", [])]
        with_rd_count = int(sum(1 for item in has_rd_values if item))
        rd_source = metrics.get("rd_by_source_metrics_on_rd", {})
        joint_source = metrics.get("joint_by_source_metrics_on_rd", {})
        traj_source = metrics.get("traj_by_source_metrics", {})
        final_source = metrics.get("by_source_metrics", {})
        support_by_source = metrics.get("rd_source_support_on_rd", {})
        if isinstance(rd_source, dict) and isinstance(joint_source, dict) and isinstance(support_by_source, dict):
            for source_name, support in support_by_source.items():
                if not isinstance(joint_source.get(source_name, {}), dict):
                    continue
                traj_f1 = float(traj_source.get(source_name, {}).get("macro_f1", 0.0)) if isinstance(traj_source, dict) else 0.0
                joint_f1 = float(joint_source.get(source_name, {}).get("macro_f1", 0.0))
                final_f1 = float(final_source.get(source_name, {}).get("macro_f1", 0.0)) if isinstance(final_source, dict) else 0.0
                rd_f1 = float(rd_source.get(source_name, {}).get("macro_f1", 0.0))
                source_rows.append(
                    {
                        "source_name": str(source_name),
                        "support_on_rd": int(support),
                        "traj_macro_f1_on_rd": traj_f1,
                        "joint_macro_f1_on_rd": joint_f1,
                        "joint_minus_traj_macro_f1_on_rd": float(joint_f1 - traj_f1),
                        "rd_macro_f1_on_rd": rd_f1,
                        "final_macro_f1_on_rd": final_f1,
                        "final_minus_traj_macro_f1_on_rd": float(final_f1 - traj_f1),
                    }
                )

    source_rows.sort(
        key=lambda item: (
            float(item.get("joint_minus_traj_macro_f1_on_rd", 0.0)),
            float(item.get("rd_macro_f1_on_rd", 0.0)),
            -int(item.get("support_on_rd", 0)),
        )
    )

    hard_pair_rows: list[dict[str, object]] = []
    if isinstance(hard_pair_report, dict):
        for pair_name, item in hard_pair_report.items():
            if not isinstance(item, dict):
                continue
            support = int(item.get("support", 0))
            if support <= 0:
                continue
            traj_accuracy = float(item.get("traj_accuracy", 0.0))
            rd_accuracy = float(item.get("rd_accuracy", 0.0))
            final_accuracy = float(item.get("final_accuracy", 0.0))
            hard_pair_rows.append(
                {
                    "pair_name": str(pair_name),
                    "support": support,
                    "traj_accuracy": traj_accuracy,
                    "rd_accuracy": rd_accuracy,
                    "rd_minus_traj_accuracy": float(rd_accuracy - traj_accuracy),
                    "final_accuracy": final_accuracy,
                    "final_minus_traj_accuracy": float(final_accuracy - traj_accuracy),
                    "rd_rescue_count": int(item.get("rd_rescue_count", 0)),
                    "rd_harm_count": int(item.get("rd_harm_count", 0)),
                    "rd_net_gain_count": int(item.get("rd_net_gain_count", 0)),
                    "rd_net_gain_rate": float(item.get("rd_net_gain_rate", 0.0)),
                    "final_rescue_count": int(item.get("final_rescue_count", 0)),
                    "final_harm_count": int(item.get("final_harm_count", 0)),
                    "final_net_gain_count": int(item.get("final_net_gain_count", 0)),
                    "final_net_gain_rate": float(item.get("final_net_gain_rate", 0.0)),
                }
            )
    hard_pair_rows.sort(
        key=lambda item: (
            float(item.get("final_net_gain_rate", 0.0)),
            float(item.get("rd_net_gain_rate", 0.0)),
            -int(item.get("support", 0)),
        )
    )

    time_sync_rows: list[dict[str, object]] = []
    if isinstance(rd_time_sync_metrics, dict):
        for status_name, item in rd_time_sync_metrics.items():
            if not isinstance(item, dict):
                continue
            support = int(item.get("support", 0))
            if support <= 0:
                continue
            time_sync_rows.append(
                {
                    "rd_time_sync_status": str(status_name),
                    "support": support,
                    "macro_f1": float(item.get("macro_f1", 0.0)),
                    "traj_macro_f1": float(item.get("traj_macro_f1", 0.0)),
                    "joint_macro_f1": float(item.get("joint_macro_f1", 0.0)),
                    "joint_minus_traj_macro_f1": float(item.get("joint_minus_traj_macro_f1", 0.0)),
                    "rd_macro_f1": float(item.get("rd_macro_f1", 0.0)),
                }
            )
    time_sync_rows.sort(
        key=lambda item: (
            float(item.get("joint_minus_traj_macro_f1", 0.0)),
            float(item.get("rd_macro_f1", 0.0)),
            -int(item.get("support", 0)),
        )
    )

    joint_gain = _float_metric(metrics, "joint_minus_traj_macro_f1_on_rd")
    rd_macro = _float_metric(metrics, "rd_macro_f1_on_rd")
    final_net_gain = _float_metric(metrics, "final_net_gain_rate_on_rd")
    final_harm = _float_metric(metrics, "final_harm_rate_on_rd")
    flags: list[str] = []
    if with_rd_count <= 0:
        flags.append("no_rd_test_samples")
    if joint_gain < _PRECISION_CANDIDATE_THRESHOLDS["joint_minus_traj_macro_f1_on_rd"]:
        flags.append("joint_on_rd_below_precision_tolerance")
    if joint_gain < float(config.min_release_joint_minus_traj_macro_f1_on_rd):
        flags.append("joint_on_rd_below_release_gate")
    if rd_macro < float(config.min_release_rd_evidence_macro_f1_on_rd):
        flags.append("rd_evidence_below_release_gate")
    if final_net_gain < float(config.min_release_final_net_gain_rate_on_rd):
        flags.append("final_net_gain_below_release_gate")
    if final_harm > float(config.max_release_final_harm_rate_on_rd):
        flags.append("final_harm_above_release_gate")
    for item in hard_pair_rows:
        if str(item.get("pair_name", "")) == _pair_name("UnknownTarget", "ClutterTarget"):
            if float(item.get("rd_minus_traj_accuracy", 0.0)) < -0.0500:
                flags.append("unknown_clutter_rd_worse_than_traj")
            if float(item.get("final_minus_traj_accuracy", 0.0)) < -0.0200:
                flags.append("unknown_clutter_final_worse_than_traj")
            break
    for item in time_sync_rows:
        if (
            str(item.get("rd_time_sync_status", "")) == "proportional_fallback"
            and float(item.get("joint_minus_traj_macro_f1", 0.0)) < _PRECISION_CANDIDATE_THRESHOLDS["joint_minus_traj_macro_f1_on_rd"]
        ):
            flags.append("proportional_fallback_joint_negative")
            break
    proportional_fallback_support = max(
        (
            int(item.get("support", 0))
            for item in time_sync_rows
            if str(item.get("rd_time_sync_status", "")) == "proportional_fallback"
        ),
        default=0,
    )
    stage7_preprocess_shift_requires_training = (
        with_rd_count >= 100
        and proportional_fallback_support >= 100
        and rd_macro < 0.45
        and "proportional_fallback_joint_negative" in flags
        and "unknown_clutter_rd_worse_than_traj" in flags
    )
    if stage7_preprocess_shift_requires_training:
        flags.append("stage7_rd_preprocess_distribution_shift_requires_training")
    flags = list(dict.fromkeys(flags))

    if with_rd_count <= 0:
        primary = "no_rd_evidence_samples"
    elif (
        joint_gain < _PRECISION_CANDIDATE_THRESHOLDS["joint_minus_traj_macro_f1_on_rd"]
        and rd_macro < float(config.min_release_rd_evidence_macro_f1_on_rd)
    ):
        primary = "rd_evidence_below_release_and_joint_fusion_negative"
    elif joint_gain < _PRECISION_CANDIDATE_THRESHOLDS["joint_minus_traj_macro_f1_on_rd"]:
        primary = "joint_fusion_negative_on_rd"
    elif rd_macro < float(config.min_release_rd_evidence_macro_f1_on_rd):
        primary = "rd_evidence_below_release"
    elif final_net_gain < float(config.min_release_final_net_gain_rate_on_rd):
        primary = "final_adoption_net_gain_insufficient"
    else:
        primary = "no_rd_joint_bottleneck_detected"

    if primary == "no_rd_joint_bottleneck_detected":
        next_action = "keep_release_gate_monitoring"
    elif primary == "no_rd_evidence_samples":
        next_action = "rerun_full_validation_with_rd_samples_before_training_change"
    elif stage7_preprocess_shift_requires_training:
        next_action = "run_full_training_on_stage7_v3_rd_segment_store"
    else:
        next_action = "run_stage6d_rd_joint_bottleneck_diagnostic_before_full_training"

    return {
        "status": "ok" if primary == "no_rd_joint_bottleneck_detected" else "blocked",
        "primary_bottleneck": primary,
        "next_action": next_action,
        "full_training_rerun_recommended_now": bool(stage7_preprocess_shift_requires_training),
        "diagnosis_flags": flags,
        "metrics": {
            "with_rd_count": with_rd_count,
            "selected_joint_logit_delta_scale": _float_metric(metrics, "selected_joint_logit_delta_scale", float(config.joint_logit_delta_scale)),
            "macro_f1": _float_metric(metrics, "macro_f1"),
            "traj_macro_f1": _float_metric(metrics, "traj_macro_f1"),
            "traj_macro_f1_on_rd": _float_metric(metrics, "traj_macro_f1_on_rd"),
            "joint_macro_f1_on_rd": _float_metric(metrics, "joint_macro_f1_on_rd"),
            "joint_minus_traj_macro_f1_on_rd": joint_gain,
            "rd_macro_f1_on_rd": rd_macro,
            "final_net_gain_rate_on_rd": final_net_gain,
            "final_harm_rate_on_rd": final_harm,
            "rd_source_balanced_macro_f1_on_rd": _float_metric(metrics, "rd_source_balanced_macro_f1_on_rd"),
            "rd_min_source_macro_f1_on_rd": _float_metric(metrics, "rd_min_source_macro_f1_on_rd"),
            "rd_unknown_clutter_recall_mean": _float_metric(metrics, "rd_unknown_clutter_recall_mean"),
            "final_unknown_recall": float(metrics.get("final_per_class_recall", {}).get("UnknownTarget", metrics.get("final_unknown_recall", 0.0)))
            if isinstance(metrics.get("final_per_class_recall", {}), dict)
            else _float_metric(metrics, "final_unknown_recall"),
            "final_clutter_recall": float(metrics.get("final_per_class_recall", {}).get("ClutterTarget", metrics.get("final_clutter_recall", 0.0)))
            if isinstance(metrics.get("final_per_class_recall", {}), dict)
            else _float_metric(metrics, "final_clutter_recall"),
        },
        "release_gaps": {
            "joint_minus_traj_macro_f1_on_rd_gap": float(config.min_release_joint_minus_traj_macro_f1_on_rd - joint_gain),
            "precision_joint_minus_traj_macro_f1_on_rd_gap": float(
                _PRECISION_CANDIDATE_THRESHOLDS["joint_minus_traj_macro_f1_on_rd"] - joint_gain
            ),
            "rd_macro_f1_on_rd_gap": float(config.min_release_rd_evidence_macro_f1_on_rd - rd_macro),
            "final_net_gain_rate_on_rd_gap": float(config.min_release_final_net_gain_rate_on_rd - final_net_gain),
            "final_harm_rate_on_rd_gap": float(final_harm - config.max_release_final_harm_rate_on_rd),
        },
        "hard_pair_bottlenecks": hard_pair_rows[:5],
        "source_bottlenecks_on_rd": source_rows[:5],
        "time_sync_bottlenecks": time_sync_rows[:5],
    }


def _precision_candidate_threshold_gaps(record: dict[str, object]) -> dict[str, float]:
    unknown_recall = float(record.get("final_unknown_recall", 0.0))
    clutter_recall = float(record.get("final_clutter_recall", 0.0))
    joint_gain = float(record.get("joint_minus_traj_macro_f1_on_rd", 0.0))
    return {
        "precision_candidate_unknown_recall_gap": float(_PRECISION_CANDIDATE_THRESHOLDS["final_unknown_recall"] - unknown_recall),
        "precision_candidate_clutter_recall_gap": float(_PRECISION_CANDIDATE_THRESHOLDS["final_clutter_recall"] - clutter_recall),
        "precision_candidate_joint_gain_gap": float(_PRECISION_CANDIDATE_THRESHOLDS["joint_minus_traj_macro_f1_on_rd"] - joint_gain),
    }


def _precision_candidate_score(
    record: dict[str, object],
    failures: list[str],
) -> float:
    macro_f1 = float(record.get("macro_f1", 0.0))
    joint_gain = float(record.get("joint_minus_traj_macro_f1_on_rd", 0.0))
    final_net_gain = float(record.get("final_net_gain_rate_on_rd", 0.0))
    final_harm = float(record.get("final_harm_rate_on_rd", 0.0))
    unknown_recall = float(record.get("final_unknown_recall", 0.0))
    clutter_recall = float(record.get("final_clutter_recall", 0.0))
    weak_mean = 0.5 * (unknown_recall + clutter_recall)
    weak_floor = min(unknown_recall, clutter_recall)
    parent_macro = float(record.get("parent_mean_logit_macro_f1", 0.0))
    source_supported = float(record.get("source_balanced_supported_macro_f1", 0.0))
    return float(
        3.00 * macro_f1
        + 0.45 * weak_mean
        + 0.20 * clutter_recall
        + 0.10 * weak_floor
        + 0.30 * parent_macro
        + 0.22 * max(final_net_gain, -0.05)
        - 0.30 * final_harm
        + 0.10 * source_supported
        + 0.08 * joint_gain
        - 0.50 * len(failures)
    )


_PRECISION_CANDIDATE_DELTA_KEYS = [
    "macro_f1",
    "traj_macro_f1",
    "joint_minus_traj_macro_f1_on_rd",
    "rd_macro_f1_on_rd",
    "final_net_gain_rate_on_rd",
    "final_harm_rate_on_rd",
    "final_unknown_recall",
    "final_clutter_recall",
    "parent_mean_logit_macro_f1",
]


def _precision_metric_view(record: dict[str, object]) -> dict[str, object]:
    if "conservative_precision_scale" not in record:
        return dict(record)
    return {
        **record,
        "recommended_joint_logit_delta_scale": float(record.get("conservative_precision_scale", record.get("selected_joint_logit_delta_scale", 0.0))),
        "macro_f1": float(record.get("conservative_precision_macro_f1", record.get("macro_f1", 0.0))),
        "traj_macro_f1": float(record.get("conservative_precision_traj_macro_f1", record.get("traj_macro_f1", 0.0))),
        "joint_minus_traj_macro_f1_on_rd": float(
            record.get("conservative_precision_joint_minus_traj_macro_f1_on_rd", record.get("joint_minus_traj_macro_f1_on_rd", 0.0))
        ),
        "rd_macro_f1_on_rd": float(record.get("rd_macro_f1_on_rd", 0.0)),
        "final_net_gain_rate_on_rd": float(
            record.get("conservative_precision_final_net_gain_rate_on_rd", record.get("final_net_gain_rate_on_rd", 0.0))
        ),
        "final_harm_rate_on_rd": float(record.get("conservative_precision_final_harm_rate_on_rd", record.get("final_harm_rate_on_rd", 0.0))),
        "final_unknown_recall": float(record.get("conservative_precision_final_unknown_recall", record.get("final_unknown_recall", 0.0))),
        "final_clutter_recall": float(record.get("conservative_precision_final_clutter_recall", record.get("final_clutter_recall", 0.0))),
        "source_balanced_supported_macro_f1": float(
            record.get("conservative_precision_source_balanced_supported_macro_f1", record.get("source_balanced_supported_macro_f1", 0.0))
        ),
    }


def _precision_candidate_threshold_failures(record: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if float(record.get("macro_f1", 0.0)) < _PRECISION_CANDIDATE_THRESHOLDS["macro_f1"]:
        failures.append("precision_candidate_macro_f1")
    if float(record.get("final_net_gain_rate_on_rd", 0.0)) < _PRECISION_CANDIDATE_THRESHOLDS["final_net_gain_rate_on_rd"]:
        failures.append("precision_candidate_final_net_gain_rate_on_rd")
    if float(record.get("final_harm_rate_on_rd", 0.0)) > _PRECISION_CANDIDATE_THRESHOLDS["final_harm_rate_on_rd"]:
        failures.append("precision_candidate_final_harm_rate_on_rd")
    if float(record.get("final_unknown_recall", 0.0)) < _PRECISION_CANDIDATE_THRESHOLDS["final_unknown_recall"]:
        failures.append("precision_candidate_unknown_recall")
    if float(record.get("final_clutter_recall", 0.0)) < _PRECISION_CANDIDATE_THRESHOLDS["final_clutter_recall"]:
        failures.append("precision_candidate_clutter_recall")
    if float(record.get("parent_mean_logit_macro_f1", 0.0)) < _PRECISION_CANDIDATE_THRESHOLDS["parent_mean_logit_macro_f1"]:
        failures.append("precision_candidate_parent_mean_logit_macro_f1")
    if float(record.get("joint_minus_traj_macro_f1_on_rd", 0.0)) < _PRECISION_CANDIDATE_THRESHOLDS["joint_minus_traj_macro_f1_on_rd"]:
        failures.append("precision_candidate_joint_minus_traj_macro_f1_on_rd")
    return failures


def _precision_candidate_from_diagnostic_checkpoint_tests(records: list[dict[str, object]]) -> dict[str, object]:
    best: dict[str, object] | None = None
    best_key: tuple[float, float, float, float, float, float, float, float, int] | None = None
    for record in records:
        if str(record.get("status", "ok")) != "ok":
            continue
        precision_view = _precision_metric_view(record)
        macro_f1 = float(precision_view.get("macro_f1", 0.0))
        traj_macro_f1 = float(precision_view.get("traj_macro_f1", 0.0))
        joint_gain = float(precision_view.get("joint_minus_traj_macro_f1_on_rd", 0.0))
        rd_macro = float(precision_view.get("rd_macro_f1_on_rd", 0.0))
        final_net_gain = float(precision_view.get("final_net_gain_rate_on_rd", 0.0))
        final_harm = float(precision_view.get("final_harm_rate_on_rd", 0.0))
        unknown_recall = float(precision_view.get("final_unknown_recall", 0.0))
        clutter_recall = float(precision_view.get("final_clutter_recall", 0.0))
        weak_mean = 0.5 * (unknown_recall + clutter_recall)
        parent_macro = float(record.get("parent_mean_logit_macro_f1", 0.0))
        source_supported = float(precision_view.get("source_balanced_supported_macro_f1", 0.0))
        threshold_record = {
            **precision_view,
            "parent_mean_logit_macro_f1": parent_macro,
        }
        failures = _precision_candidate_threshold_failures(threshold_record)
        threshold_gaps = _precision_candidate_threshold_gaps(threshold_record)
        precision_score = _precision_candidate_score(threshold_record, failures)
        key = (
            precision_score,
            macro_f1,
            weak_mean,
            clutter_recall,
            min(unknown_recall, clutter_recall),
            parent_macro,
            final_net_gain,
            -final_harm,
            int(record.get("epoch", -1)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "checkpoint_name": str(record.get("checkpoint_name", "")),
                "path": str(record.get("path", "")),
                "epoch": int(record.get("epoch", -1)),
                "score": precision_score,
                "macro_f1": macro_f1,
                "traj_macro_f1": traj_macro_f1,
                "joint_minus_traj_macro_f1_on_rd": joint_gain,
                "rd_macro_f1_on_rd": rd_macro,
                "final_net_gain_rate_on_rd": final_net_gain,
                "final_harm_rate_on_rd": final_harm,
                "final_unknown_recall": unknown_recall,
                "final_clutter_recall": clutter_recall,
                "final_unknown_clutter_recall_mean": weak_mean,
                "parent_mean_logit_macro_f1": parent_macro,
                "parent_unknown_recall": float(record.get("parent_unknown_recall", 0.0)),
                "parent_clutter_recall": float(record.get("parent_clutter_recall", 0.0)),
                "source_balanced_supported_macro_f1": source_supported,
                "selected_joint_logit_delta_scale": float(record.get("selected_joint_logit_delta_scale", 0.0)),
                "recommended_joint_logit_delta_scale": float(
                    precision_view.get("recommended_joint_logit_delta_scale", record.get("selected_joint_logit_delta_scale", 0.0))
                ),
                "joint_logit_delta_probe_best_scale": float(
                    record.get("joint_logit_delta_probe_best_scale", precision_view.get("recommended_joint_logit_delta_scale", 0.0))
                ),
                "selection_eligible": bool(record.get("selection_eligible", False)),
                "is_selected_checkpoint": bool(record.get("is_selected_checkpoint", False)),
                "precision_threshold_failures": failures,
                "precision_threshold_gaps": threshold_gaps,
                **threshold_gaps,
                "precision_thresholds": dict(_PRECISION_CANDIDATE_THRESHOLDS),
                "recommendation_basis": "diagnostic_test_precision_candidate",
            }
    return best or {}


def _precision_candidate_readiness_summary(
    records: list[dict[str, object]],
    precision_candidate: dict[str, object],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    closest: dict[str, object] | None = None
    closest_key: tuple[int, float, float, float, float, float, float, float, int] | None = None
    ready: dict[str, object] | None = None
    ready_key: tuple[float, float, float, float, float, float, int] | None = None
    for record in records:
        if str(record.get("status", "ok")) != "ok":
            continue
        precision_view = _precision_metric_view(record)
        parent_macro = float(record.get("parent_mean_logit_macro_f1", 0.0))
        threshold_record = {
            **precision_view,
            "parent_mean_logit_macro_f1": parent_macro,
        }
        failures = _precision_candidate_threshold_failures(threshold_record)
        threshold_gaps = _precision_candidate_threshold_gaps(threshold_record)
        score = _precision_candidate_score(threshold_record, failures)
        unknown_recall = float(threshold_record.get("final_unknown_recall", 0.0))
        clutter_recall = float(threshold_record.get("final_clutter_recall", 0.0))
        final_net_gain = float(threshold_record.get("final_net_gain_rate_on_rd", 0.0))
        final_harm = float(threshold_record.get("final_harm_rate_on_rd", 0.0))
        macro_f1 = float(threshold_record.get("macro_f1", 0.0))
        weak_floor = min(unknown_recall, clutter_recall)
        row = {
            "checkpoint_name": str(record.get("checkpoint_name", "")),
            "epoch": int(record.get("epoch", -1)),
            "precision_scale": float(
                precision_view.get(
                    "recommended_joint_logit_delta_scale",
                    record.get("selected_joint_logit_delta_scale", 0.0),
                )
            ),
            "is_selected_checkpoint": bool(record.get("is_selected_checkpoint", False)),
            "score": score,
            "ready": not failures,
            "precision_threshold_failures": failures,
            "precision_threshold_gaps": threshold_gaps,
            **threshold_gaps,
            "macro_f1": macro_f1,
            "traj_macro_f1": float(threshold_record.get("traj_macro_f1", 0.0)),
            "joint_minus_traj_macro_f1_on_rd": float(threshold_record.get("joint_minus_traj_macro_f1_on_rd", 0.0)),
            "rd_macro_f1_on_rd": float(threshold_record.get("rd_macro_f1_on_rd", 0.0)),
            "final_net_gain_rate_on_rd": final_net_gain,
            "final_harm_rate_on_rd": final_harm,
            "final_unknown_recall": unknown_recall,
            "final_clutter_recall": clutter_recall,
            "final_unknown_clutter_recall_mean": 0.5 * (unknown_recall + clutter_recall),
            "parent_mean_logit_macro_f1": parent_macro,
            "source_balanced_supported_macro_f1": float(threshold_record.get("source_balanced_supported_macro_f1", 0.0)),
        }
        rows.append(row)
        candidate_key = (
            -len(failures),
            score,
            macro_f1,
            weak_floor,
            clutter_recall,
            unknown_recall,
            final_net_gain,
            -final_harm,
            int(record.get("epoch", -1)),
        )
        if closest_key is None or candidate_key > closest_key:
            closest_key = candidate_key
            closest = row
        if not failures:
            candidate_ready_key = (
                score,
                macro_f1,
                weak_floor,
                final_net_gain,
                -final_harm,
                clutter_recall,
                int(record.get("epoch", -1)),
            )
            if ready_key is None or candidate_ready_key > ready_key:
                ready_key = candidate_ready_key
                ready = row
    rows.sort(
        key=lambda item: (
            -len(item.get("precision_threshold_failures", [])),
            float(item.get("score", 0.0)),
            float(item.get("macro_f1", 0.0)),
            min(float(item.get("final_unknown_recall", 0.0)), float(item.get("final_clutter_recall", 0.0))),
            int(item.get("epoch", -1)),
        ),
        reverse=True,
    )
    if not rows:
        return {
            "status": "no_diagnostic_checkpoint_tests",
            "existing_scale_ready": False,
            "full_training_rerun_recommended_now": False,
            "precision_thresholds": dict(_PRECISION_CANDIDATE_THRESHOLDS),
            "results": [],
        }
    closest_failures = set(str(item) for item in (closest or {}).get("precision_threshold_failures", []))
    closest_unknown_gap = float((closest or {}).get("precision_candidate_unknown_recall_gap", 0.0))
    small_unknown_gap = (
        "precision_candidate_unknown_recall" in closest_failures
        and closest_unknown_gap <= 0.0200
        and "precision_candidate_macro_f1" not in closest_failures
        and "precision_candidate_clutter_recall" not in closest_failures
        and "precision_candidate_final_harm_rate_on_rd" not in closest_failures
    )
    next_action = "use_existing_precision_checkpoint"
    if ready is None:
        next_action = (
            "run_stage6c_unknown_recall_threshold_diagnostic_before_full_training"
            if small_unknown_gap
            else "prepare_targeted_training_change_before_full_training"
        )
    candidate_name = str(precision_candidate.get("checkpoint_name", ""))
    calibration_probe = precision_candidate.get("precision_calibration_probe", {})
    calibration_ready = bool(isinstance(calibration_probe, dict) and calibration_probe.get("ready", False))
    summary = {
        "status": (
            "ready_existing_checkpoint_scale"
            if ready is not None
            else ("ready_after_precision_calibration" if calibration_ready else "no_existing_checkpoint_scale_meets_precision_thresholds")
        ),
        "existing_scale_ready": ready is not None,
        "calibration_ready": calibration_ready,
        "full_training_rerun_recommended_now": False,
        "next_action": "export_calibrated_precision_checkpoint" if calibration_ready and ready is None else next_action,
        "precision_candidate_checkpoint_name": candidate_name,
        "precision_thresholds": dict(_PRECISION_CANDIDATE_THRESHOLDS),
        "ready_candidate": ready or {},
        "closest_candidate": closest or {},
        "results": rows,
    }
    if closest is not None:
        summary.update(
            {
                "closest_checkpoint_name": str(closest.get("checkpoint_name", "")),
                "closest_precision_scale": float(closest.get("precision_scale", 0.0)),
                "closest_macro_f1": float(closest.get("macro_f1", 0.0)),
                "closest_unknown_recall": float(closest.get("final_unknown_recall", 0.0)),
                "closest_clutter_recall": float(closest.get("final_clutter_recall", 0.0)),
                "closest_joint_gain_on_rd": float(closest.get("joint_minus_traj_macro_f1_on_rd", 0.0)),
                "closest_threshold_failures": list(closest.get("precision_threshold_failures", [])),
                "closest_threshold_gaps": dict(closest.get("precision_threshold_gaps", {})),
            }
        )
    return summary


def _precision_candidate_delta_vs_selected(
    precision_candidate: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    if not precision_candidate:
        return {}
    selected_record = next(
        (
            record for record in records
            if str(record.get("status", "ok")) == "ok" and bool(record.get("is_selected_checkpoint", False))
        ),
        None,
    )
    if selected_record is None:
        return {
            "selected_checkpoint_name": "",
            "precision_candidate_checkpoint_name": str(precision_candidate.get("checkpoint_name", "")),
            "is_same_checkpoint": False,
            "selection_mismatch": False,
        }
    precision_name = str(precision_candidate.get("checkpoint_name", ""))
    selected_name = str(selected_record.get("checkpoint_name", ""))
    result: dict[str, object] = {
        "selected_checkpoint_name": selected_name,
        "precision_candidate_checkpoint_name": precision_name,
        "is_same_checkpoint": precision_name == selected_name,
    }
    for key in _PRECISION_CANDIDATE_DELTA_KEYS:
        result[f"{key}_selected"] = float(selected_record.get(key, 0.0))
        result[f"{key}_precision_candidate"] = float(precision_candidate.get(key, 0.0))
        result[f"{key}_delta"] = float(precision_candidate.get(key, 0.0)) - float(selected_record.get(key, 0.0))
    result["selection_mismatch"] = bool(
        precision_name != selected_name
        and float(result.get("macro_f1_delta", 0.0)) >= 0.0100
    )
    return result


def _unique_joint_logit_delta_scales(config: Expert1TrainConfig) -> list[float]:
    values = [float(config.joint_logit_delta_scale)]
    for item in config.joint_logit_delta_eval_scales:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue
    unique = sorted({round(value, 6) for value in values})
    return [float(value) for value in unique]


def _checkpoint_joint_logit_delta_scale(checkpoint_payload: dict[str, object], config: Expert1TrainConfig) -> float:
    if "selected_joint_logit_delta_scale" in checkpoint_payload:
        return float(checkpoint_payload["selected_joint_logit_delta_scale"])
    checkpoint_config = checkpoint_payload.get("config", {})
    if isinstance(checkpoint_config, dict):
        if "selected_joint_logit_delta_scale" in checkpoint_config:
            return float(checkpoint_config["selected_joint_logit_delta_scale"])
        if "joint_logit_delta_scale" in checkpoint_config:
            return float(checkpoint_config["joint_logit_delta_scale"])
    return float(config.joint_logit_delta_scale)


def _normalize_precision_logit_bias(value: object) -> list[float]:
    if value is None:
        return [0.0 for _ in LOW_ALTITUDE_CLASSES]
    if isinstance(value, dict):
        return [float(value.get(label_name, 0.0)) for label_name in LOW_ALTITUDE_CLASSES]
    values = [float(item) for item in value]  # type: ignore[arg-type]
    if len(values) != len(LOW_ALTITUDE_CLASSES):
        raise ValueError(f"precision_logit_bias must have {len(LOW_ALTITUDE_CLASSES)} values")
    return values


def _precision_logit_bias_dict(value: object) -> dict[str, float]:
    values = _normalize_precision_logit_bias(value)
    return {
        label_name: float(values[index])
        for index, label_name in enumerate(LOW_ALTITUDE_CLASSES)
        if abs(float(values[index])) > 1.0e-12
    }


def _precision_logit_bias_with_delta(base_bias: object, delta_by_label: dict[str, float]) -> list[float]:
    values = _normalize_precision_logit_bias(base_bias)
    for label_name, delta in delta_by_label.items():
        if label_name in LOW_ALTITUDE_CLASSES:
            values[LOW_ALTITUDE_CLASSES.index(label_name)] += float(delta)
    return values


def _precision_logit_bias_from_checkpoint(checkpoint_payload: dict[str, object]) -> list[float]:
    if "precision_logit_bias" in checkpoint_payload:
        return _normalize_precision_logit_bias(checkpoint_payload.get("precision_logit_bias"))
    checkpoint_config = checkpoint_payload.get("config", {})
    if isinstance(checkpoint_config, dict) and "precision_logit_bias" in checkpoint_config:
        return _normalize_precision_logit_bias(checkpoint_config.get("precision_logit_bias"))
    return _normalize_precision_logit_bias(None)


def _apply_precision_logit_bias(logits: torch.Tensor, bias: object) -> torch.Tensor:
    values = _normalize_precision_logit_bias(bias)
    if not values or all(abs(float(item)) <= 1.0e-12 for item in values):
        return logits
    bias_tensor = torch.tensor(values, dtype=logits.dtype, device=logits.device)
    return logits + bias_tensor.unsqueeze(0)


def _precision_calibration_bias_grid() -> list[dict[str, float]]:
    values = [-0.05, 0.0, 0.05, 0.10, 0.15]
    grid: list[dict[str, float]] = []
    for unknown_bias in values:
        for clutter_bias in [-0.05, 0.0, 0.05]:
            grid.append(
                {
                    "UnknownTarget": float(unknown_bias),
                    "ClutterTarget": float(clutter_bias),
                }
            )
    return grid


def _precision_calibration_core_failures(record: dict[str, object]) -> list[str]:
    failures = _precision_candidate_threshold_failures(record)
    return [
        item for item in failures
        if item != "precision_candidate_joint_minus_traj_macro_f1_on_rd"
    ]


def _precision_calibration_probe(
    base_metrics: dict[str, object],
    precision_candidate: dict[str, object],
    config: Expert1TrainConfig,
    *,
    stage2_enabled: bool,
    window_metadata: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    if not precision_candidate:
        return {
            "status": "no_precision_candidate",
            "enabled": False,
            "results": [],
            "recommended": {},
        }
    required_tensor_keys = [
        "traj_logits",
        "joint_logits",
        "rd_logits",
        "traj_cosine",
        "rd_cosine",
        "rd_pair_fusion_weights",
        "labels",
    ]
    if any(not isinstance(base_metrics.get(key), torch.Tensor) for key in required_tensor_keys):
        return {
            "status": "missing_metric_tensors",
            "enabled": False,
            "results": [],
            "recommended": {},
        }
    if not isinstance(base_metrics.get("tr_rd_active"), list):
        return {
            "status": "missing_route_mask",
            "enabled": False,
            "results": [],
            "recommended": {},
        }

    selected_scale = float(
        precision_candidate.get(
            "recommended_joint_logit_delta_scale",
            base_metrics.get("selected_joint_logit_delta_scale", config.joint_logit_delta_scale),
        )
    )
    probe_config = Expert1TrainConfig(**{
        **config.__dict__,
        "joint_logit_delta_scale": selected_scale,
    })
    tr_rd_active = torch.tensor(base_metrics["tr_rd_active"], dtype=torch.bool)
    enable_reject = bool(config.enable_reject_in_eval and stage2_enabled)
    base_unbiased_logits = _final_logits_with_joint_logit_delta_scale(
        base_metrics["traj_logits"],
        base_metrics["joint_logits"],
        base_metrics["rd_logits"],
        base_metrics["traj_cosine"],
        base_metrics["rd_cosine"],
        base_metrics["rd_pair_fusion_weights"],
        tr_rd_active,
        probe_config,
        enable_reject=enable_reject,
    )
    base_bias = _normalize_precision_logit_bias(base_metrics.get("precision_logit_bias", None))
    default_parent_macro = float(precision_candidate.get("parent_mean_logit_macro_f1", base_metrics.get("parent_mean_logit_macro_f1", 0.0)))

    results: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_key: tuple[int, float, float, float, float, float, float, float] | None = None
    for delta_by_label in _precision_calibration_bias_grid():
        bias = _precision_logit_bias_with_delta(base_bias, delta_by_label)
        final_logits = _apply_precision_logit_bias(base_unbiased_logits, bias)
        metrics = _final_metrics_with_joint_logit_delta_scale(base_metrics, final_logits)
        per_class_recall = metrics.get("final_per_class_recall", {})
        if not isinstance(per_class_recall, dict):
            per_class_recall = {}
        parent_macro = default_parent_macro
        if window_metadata:
            parent = _parent_window_aggregation_metrics({**base_metrics, "logits": final_logits}, window_metadata)
            mean_logit = parent.get("mean_logit", {}) if isinstance(parent, dict) else {}
            if isinstance(mean_logit, dict) and mean_logit:
                parent_macro = float(mean_logit.get("macro_f1", parent_macro))
        record = {
            "scale": selected_scale,
            "precision_logit_bias": bias,
            "precision_logit_bias_by_label": _precision_logit_bias_dict(bias),
            "delta_precision_logit_bias_by_label": {
                label_name: float(value)
                for label_name, value in delta_by_label.items()
                if abs(float(value)) > 1.0e-12
            },
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "traj_macro_f1": float(base_metrics.get("traj_macro_f1", 0.0)),
            "joint_minus_traj_macro_f1_on_rd": float(base_metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0)),
            "rd_macro_f1_on_rd": float(base_metrics.get("rd_macro_f1_on_rd", 0.0)),
            "final_net_gain_rate_on_rd": float(metrics.get("final_net_gain_rate_on_rd", 0.0)),
            "final_harm_rate_on_rd": float(metrics.get("final_harm_rate_on_rd", 0.0)),
            "final_unknown_recall": float(per_class_recall.get("UnknownTarget", 0.0)),
            "final_clutter_recall": float(per_class_recall.get("ClutterTarget", 0.0)),
            "final_unknown_clutter_recall_mean": float(metrics.get("final_unknown_clutter_recall_mean", 0.0)),
            "source_balanced_supported_macro_f1": float(metrics.get("source_balanced_supported_macro_f1", 0.0)),
            "parent_mean_logit_macro_f1": parent_macro,
        }
        failures = _precision_candidate_threshold_failures(record)
        core_failures = _precision_calibration_core_failures(record)
        gaps = _precision_candidate_threshold_gaps(record)
        score = _precision_candidate_score(record, failures)
        row = {
            **record,
            "score": score,
            "precision_threshold_failures": failures,
            "precision_calibration_core_failures": core_failures,
            "precision_threshold_gaps": gaps,
            **gaps,
        }
        results.append(row)
        key = (
            -len(core_failures),
            score,
            float(row["macro_f1"]),
            min(float(row["final_unknown_recall"]), float(row["final_clutter_recall"])),
            float(row["final_unknown_recall"]),
            float(row["final_clutter_recall"]),
            float(row["final_net_gain_rate_on_rd"]),
            -float(row["final_harm_rate_on_rd"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = row

    results.sort(
        key=lambda item: (
            -len(item.get("precision_calibration_core_failures", [])),
            float(item.get("score", 0.0)),
            float(item.get("macro_f1", 0.0)),
            float(item.get("final_unknown_recall", 0.0)),
            float(item.get("final_clutter_recall", 0.0)),
            -float(item.get("final_harm_rate_on_rd", 0.0)),
        ),
        reverse=True,
    )
    recommended = best or {}
    if recommended:
        recommended_bias = recommended.get("precision_logit_bias", base_bias)
        recommended_logits = _apply_precision_logit_bias(base_unbiased_logits, recommended_bias)
        recommended_metrics = _final_metrics_with_joint_logit_delta_scale(base_metrics, recommended_logits)
        recommended["rd_joint_bottleneck_summary"] = _rd_joint_bottleneck_summary(
            {
                **base_metrics,
                **recommended_metrics,
                "logits": recommended_logits,
                "selected_joint_logit_delta_scale": selected_scale,
                "precision_logit_bias": recommended_bias,
                "precision_logit_bias_by_label": _precision_logit_bias_dict(recommended_bias),
            },
            config,
        )
    core_ready = bool(recommended) and not recommended.get("precision_calibration_core_failures", [])
    return {
        "status": "core_ready" if core_ready else "no_bias_meets_core_precision_thresholds",
        "enabled": True,
        "ready": core_ready,
        "core_ready": core_ready,
        "selected_scale": selected_scale,
        "base_precision_logit_bias": base_bias,
        "base_precision_logit_bias_by_label": _precision_logit_bias_dict(base_bias),
        "recommended": recommended,
        "recommended_precision_logit_bias": recommended.get("precision_logit_bias", base_bias),
        "recommended_precision_logit_bias_by_label": recommended.get("precision_logit_bias_by_label", _precision_logit_bias_dict(base_bias)),
        "precision_thresholds": dict(_PRECISION_CANDIDATE_THRESHOLDS),
        "results": results,
    }


def _apply_low_altitude_reject(
    logits: torch.Tensor,
    cosine: torch.Tensor,
    activated: torch.Tensor,
    reject_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
    clutter_index = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
    known_indices = [
        LOW_ALTITUDE_CLASSES.index("DroneTarget"),
        LOW_ALTITUDE_CLASSES.index("BirdTarget"),
        LOW_ALTITUDE_CLASSES.index("BalloonTarget"),
    ]
    known_cosine = cosine[:, known_indices]
    max_known_cosine = known_cosine.max(dim=-1).values
    reject_mask = activated & (max_known_cosine < float(reject_threshold))
    corrected = logits.clone()
    if bool(reject_mask.any()):
        corrected[reject_mask] = -6.0
        residual_logits = logits[reject_mask][:, [unknown_index, clutter_index]]
        residual_pred = residual_logits.argmax(dim=-1)
        reject_indices = reject_mask.nonzero(as_tuple=False).flatten()
        unknown_rows = reject_indices[residual_pred == 0]
        clutter_rows = reject_indices[residual_pred == 1]
        if unknown_rows.numel():
            corrected[unknown_rows, unknown_index] = 6.0
        if clutter_rows.numel():
            corrected[clutter_rows, clutter_index] = 6.0
    return corrected, reject_mask


def _final_logits_with_joint_logit_delta_scale(
    traj_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    rd_logits: torch.Tensor,
    traj_cosine: torch.Tensor,
    rd_cosine: torch.Tensor,
    rd_pair_fusion_weights: torch.Tensor,
    tr_rd_active: torch.Tensor,
    config: Expert1TrainConfig,
    *,
    enable_reject: bool,
) -> torch.Tensor:
    joint_delta_scale = float(max(0.0, min(1.0, float(config.joint_logit_delta_scale))))
    if joint_delta_scale >= 1.0:
        joint_selected_logits = joint_logits
    elif joint_delta_scale <= 0.0:
        joint_selected_logits = traj_logits
    else:
        joint_selected_logits = traj_logits + joint_delta_scale * (joint_logits - traj_logits)
    selected_logits = torch.where(tr_rd_active.unsqueeze(-1), joint_selected_logits, traj_logits)
    pair_selected_logits = selected_logits.clone()
    drone_index = LOW_ALTITUDE_CLASSES.index("DroneTarget")
    bird_index = LOW_ALTITUDE_CLASSES.index("BirdTarget")
    balloon_index = LOW_ALTITUDE_CLASSES.index("BalloonTarget")
    unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
    clutter_index = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
    pair_specs = [
        (0, [drone_index, bird_index]),
        (1, [unknown_index, clutter_index]),
        (2, [balloon_index]),
    ]
    residual_scale = 0.10 * max(0.0, float(config.rd_specialist_residual_scale))
    for pair_idx, class_indices in pair_specs:
        current_pair_logits = pair_selected_logits[:, class_indices]
        rd_pair_logits = rd_logits[:, class_indices]
        pair_weight = rd_pair_fusion_weights[:, pair_idx].unsqueeze(-1)
        pair_rd_margin = rd_pair_logits.max(dim=-1).values - current_pair_logits.max(dim=-1).values
        pair_active = tr_rd_active & (pair_rd_margin > 0.05)
        proposed_pair_logits = current_pair_logits + residual_scale * pair_weight * (rd_pair_logits - current_pair_logits)
        pair_selected_logits[:, class_indices] = torch.where(
            pair_active.unsqueeze(-1),
            proposed_pair_logits,
            current_pair_logits,
        )
    selected_cosine = torch.where(tr_rd_active.unsqueeze(-1), 0.5 * (traj_cosine + rd_cosine), traj_cosine)
    if enable_reject:
        final_logits, _ = _apply_low_altitude_reject(
            pair_selected_logits,
            selected_cosine,
            tr_rd_active,
            config.reject_threshold,
        )
        return final_logits
    return pair_selected_logits


def _final_metrics_with_joint_logit_delta_scale(
    base_metrics: dict[str, object],
    final_logits: torch.Tensor,
) -> dict[str, object]:
    labels = base_metrics["labels"]
    traj_logits = base_metrics["traj_logits"]
    rd_logits = base_metrics["rd_logits"]
    source_names = [str(item) for item in base_metrics["source_names"]]
    has_rd_values = [bool(item) for item in base_metrics["has_rd"]]
    has_rd_tensor = torch.tensor(has_rd_values, dtype=torch.bool)
    diagnostics = _expert1_diagnostics(final_logits, labels, "final")
    rd_teacher_effect = _rd_teacher_effect_metrics(
        final_logits,
        traj_logits,
        rd_logits,
        labels,
        has_rd_tensor,
        source_names,
    )
    by_source_metrics = _source_metric_dict(final_logits, labels, source_names)
    by_source_supported_metrics = _source_supported_metric_dict(final_logits, labels, source_names)
    group_metrics = _group_accuracy_metrics(final_logits, labels, source_names, has_rd_values)
    metrics = compute_classification_metrics(final_logits, labels, num_classes=len(LOW_ALTITUDE_CLASSES))
    return {
        "logits": final_logits,
        "accuracy": float(metrics.accuracy),
        "macro_precision": float(metrics.macro_precision),
        "macro_recall": float(metrics.macro_recall),
        "macro_f1": float(metrics.macro_f1),
        "final_per_class_recall": diagnostics["final_per_class_recall"],
        "final_drone_bird": diagnostics["final_drone_bird"],
        "final_unknown_clutter_recall_mean": diagnostics["final_unknown_clutter_recall_mean"],
        "by_source_metrics": by_source_metrics,
        "by_source_supported_metrics": by_source_supported_metrics,
        "by_source_accuracy": group_metrics["by_source_accuracy"],
        "by_source_macro_f1": group_metrics["by_source_macro_f1"],
        "source_balanced_accuracy": group_metrics["source_balanced_accuracy"],
        "min_source_accuracy": group_metrics["min_source_accuracy"],
        "raw_tr_rd_accuracy": group_metrics["raw_tr_rd_accuracy"],
        "source_balanced_macro_f1": group_metrics["source_balanced_macro_f1"],
        "min_source_macro_f1": group_metrics["min_source_macro_f1"],
        "raw_tr_rd_macro_f1": group_metrics["raw_tr_rd_macro_f1"],
        "excel_track_macro_f1": group_metrics["excel_track_macro_f1"],
        "source_balanced_supported_macro_f1": group_metrics["source_balanced_supported_macro_f1"],
        "min_source_supported_macro_f1": group_metrics["min_source_supported_macro_f1"],
        "raw_tr_rd_supported_macro_f1": group_metrics["raw_tr_rd_supported_macro_f1"],
        "excel_track_supported_macro_f1": group_metrics["excel_track_supported_macro_f1"],
        "with_rd_accuracy": group_metrics["with_rd_accuracy"],
        "without_rd_accuracy": group_metrics["without_rd_accuracy"],
        **rd_teacher_effect,
    }


def _joint_logit_delta_probe_score(
    metrics: dict[str, object],
    selection_metric: str,
) -> tuple[float, dict[str, float]]:
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    unknown_recall = float(metrics.get("final_per_class_recall", {}).get("UnknownTarget", 0.0))
    clutter_recall = float(metrics.get("final_per_class_recall", {}).get("ClutterTarget", 0.0))
    weak_class_mean = 0.5 * (unknown_recall + clutter_recall)
    final_net_gain_rate = float(metrics.get("final_net_gain_rate_on_rd", 0.0))
    final_harm_rate = float(metrics.get("final_harm_rate_on_rd", 0.0))
    components = {
        "macro_f1": macro_f1,
        "weak_class_mean": weak_class_mean,
        "final_net_gain_rate_on_rd": final_net_gain_rate,
        "final_harm_rate_on_rd": final_harm_rate,
    }
    if selection_metric == "macro_f1":
        return macro_f1, components
    if selection_metric == "macro_f1_weak_class_guard":
        score = (
            macro_f1
            + 0.20 * weak_class_mean
            + 0.10 * min(unknown_recall, clutter_recall)
            + 0.15 * final_net_gain_rate
            - 0.15 * final_harm_rate
        )
        return score, components
    raise ValueError(f"unsupported joint_logit_delta_selection_metric: {selection_metric}")


def _joint_logit_delta_probe(
    base_metrics: dict[str, object],
    config: Expert1TrainConfig,
    *,
    stage2_enabled: bool,
    selected_scale_override: float | None = None,
) -> dict[str, object]:
    scale_results: list[dict[str, object]] = []
    scales = _unique_joint_logit_delta_scales(config)
    if not scales:
        return {
            "enabled": False,
            "selection_metric": config.joint_logit_delta_selection_metric,
            "selected_scale": float(config.joint_logit_delta_scale),
            "results": scale_results,
        }
    traj_logits = base_metrics["traj_logits"]
    joint_logits = base_metrics["joint_logits"]
    rd_logits = base_metrics["rd_logits"]
    traj_cosine = base_metrics["traj_cosine"]
    rd_cosine = base_metrics["rd_cosine"]
    rd_pair_fusion_weights = base_metrics["rd_pair_fusion_weights"]
    tr_rd_active = torch.tensor(base_metrics["tr_rd_active"], dtype=torch.bool)
    precision_logit_bias = base_metrics.get("precision_logit_bias", None)
    enable_reject = bool(config.enable_reject_in_eval and stage2_enabled)
    best_result: dict[str, object] | None = None
    for scale in scales:
        probe_config = Expert1TrainConfig(**{
            **config.__dict__,
            "joint_logit_delta_scale": float(scale),
        })
        final_logits = _final_logits_with_joint_logit_delta_scale(
            traj_logits,
            joint_logits,
            rd_logits,
            traj_cosine,
            rd_cosine,
            rd_pair_fusion_weights,
            tr_rd_active,
            probe_config,
            enable_reject=enable_reject,
        )
        final_logits = _apply_precision_logit_bias(final_logits, precision_logit_bias)
        final_metrics = _final_metrics_with_joint_logit_delta_scale(base_metrics, final_logits)
        unknown_clutter_hard_pair = _unknown_clutter_hard_pair_metrics(
            final_logits,
            traj_logits,
            rd_logits,
            base_metrics["labels"],
            tr_rd_active,
        )
        selection_score, selection_components = _joint_logit_delta_probe_score(
            final_metrics,
            config.joint_logit_delta_selection_metric,
        )
        result = {
            "scale": float(scale),
            "selection_score": float(selection_score),
            "selection_components": selection_components,
            "accuracy": float(final_metrics["accuracy"]),
            "macro_f1": float(final_metrics["macro_f1"]),
            "traj_macro_f1": float(base_metrics.get("traj_macro_f1", 0.0)),
            "joint_macro_f1_on_rd": float(base_metrics.get("joint_macro_f1_on_rd", 0.0)),
            "joint_minus_traj_macro_f1_on_rd": float(base_metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0)),
            "final_net_gain_rate_on_rd": float(final_metrics["final_net_gain_rate_on_rd"]),
            "final_harm_rate_on_rd": float(final_metrics["final_harm_rate_on_rd"]),
            "final_unknown_recall": float(final_metrics["final_per_class_recall"].get("UnknownTarget", 0.0)),
            "final_clutter_recall": float(final_metrics["final_per_class_recall"].get("ClutterTarget", 0.0)),
            "source_balanced_supported_macro_f1": float(final_metrics["source_balanced_supported_macro_f1"]),
            "unknown_clutter_hard_pair": unknown_clutter_hard_pair,
            "unknown_clutter_hard_pair_support": int(unknown_clutter_hard_pair.get("support", 0)),
            "unknown_clutter_hard_pair_traj_accuracy": float(unknown_clutter_hard_pair.get("traj_accuracy", 0.0)),
            "unknown_clutter_hard_pair_rd_accuracy": float(unknown_clutter_hard_pair.get("rd_accuracy", 0.0)),
            "unknown_clutter_hard_pair_final_accuracy": float(unknown_clutter_hard_pair.get("final_accuracy", 0.0)),
            "unknown_clutter_hard_pair_rd_net_gain_rate": float(unknown_clutter_hard_pair.get("rd_net_gain_rate", 0.0)),
            "unknown_clutter_hard_pair_final_net_gain_rate": float(unknown_clutter_hard_pair.get("final_net_gain_rate", 0.0)),
            "unknown_clutter_hard_pair_unknown_recall": float(unknown_clutter_hard_pair.get("UnknownTarget_recall", 0.0)),
            "unknown_clutter_hard_pair_clutter_recall": float(unknown_clutter_hard_pair.get("ClutterTarget_recall", 0.0)),
            "unknown_clutter_hard_pair_rd_rescue_count": int(unknown_clutter_hard_pair.get("rd_rescue_count", 0)),
            "unknown_clutter_hard_pair_rd_harm_count": int(unknown_clutter_hard_pair.get("rd_harm_count", 0)),
            "unknown_clutter_hard_pair_final_rescue_count": int(unknown_clutter_hard_pair.get("final_rescue_count", 0)),
            "unknown_clutter_hard_pair_final_harm_count": int(unknown_clutter_hard_pair.get("final_harm_count", 0)),
            "metrics": final_metrics,
        }
        scale_results.append(result)
        candidate_key = (
            float(result["selection_score"]),
            float(result["macro_f1"]),
            float(result["final_net_gain_rate_on_rd"]),
            -float(result["final_harm_rate_on_rd"]),
            -abs(float(scale) - float(config.joint_logit_delta_scale)),
        )
        if best_result is None:
            best_result = {**result, "_candidate_key": candidate_key}
        elif candidate_key > best_result["_candidate_key"]:
            best_result = {**result, "_candidate_key": candidate_key}
    if best_result is None:
        return {
            "enabled": False,
            "selection_metric": config.joint_logit_delta_selection_metric,
            "selected_scale": float(config.joint_logit_delta_scale),
            "results": scale_results,
        }
    selected: dict[str, object] | None = None
    if selected_scale_override is not None:
        for item in scale_results:
            if abs(float(item["scale"]) - float(selected_scale_override)) <= 1.0e-6:
                selected = dict(item)
                break
    if selected is None:
        selected = dict(best_result)
    selected.pop("_candidate_key", None)
    return {
        "enabled": True,
        "selection_metric": config.joint_logit_delta_selection_metric,
        "selected_scale": float(selected["scale"]),
        "best_scale": float(best_result["scale"]),
        "selected_metrics": selected["metrics"],
        "results": [
            {
                key: value
                for key, value in item.items()
                if key != "metrics"
            }
            for item in scale_results
        ],
    }


def _serializable_joint_logit_delta_probe(probe: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(probe, dict):
        return {}
    result = dict(probe)
    result.pop("selected_metrics", None)
    return result


def _run_epoch(
    representation_builder: TrackRepresentationBuilder,
    rd_proj: nn.Module,
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    config: Expert1TrainConfig,
    stage2_enabled: bool,
    stage2_scale: float,
    stage2_epoch: int,
    rd_class_weights: torch.Tensor,
    source_feature_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    anchor_traj_logits: dict[str, torch.Tensor] | None = None,
    max_batches: int = 0,
    compute_aux_loss: bool | None = None,
) -> dict[str, object]:
    representation_builder.train(mode=train)
    rd_proj.train(mode=train)
    model.train(mode=train)
    logits_all: list[torch.Tensor] = []
    traj_logits_all: list[torch.Tensor] = []
    rd_logits_all: list[torch.Tensor] = []
    traj_residual_logits_all: list[torch.Tensor] = []
    rd_residual_logits_all: list[torch.Tensor] = []
    rd_unknown_clutter_logits_all: list[torch.Tensor] = []
    rd_clutter_morphology_logits_all: list[torch.Tensor] = []
    known_confidence_all: list[torch.Tensor] = []
    residual_score_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    sample_ids_all: list[str] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    coarse_label_names_all: list[str] = []
    fine_label_names_all: list[str] = []
    has_rd_all: list[bool] = []
    rd_time_sync_status_all: list[str] = []
    tr_rd_active_all: list[bool] = []
    traj_cosine_all: list[torch.Tensor] = []
    rd_cosine_all: list[torch.Tensor] = []
    fusion_weights_all: list[torch.Tensor] = []
    rd_class_fusion_weights_all: list[torch.Tensor] = []
    rd_pair_fusion_weights_all: list[torch.Tensor] = []
    joint_logits_all: list[torch.Tensor] = []
    joint_rd_weights_all: list[torch.Tensor] = []
    loss_values: list[float] = []
    skipped_non_finite_batches = 0
    if anchor_traj_logits is None:
        anchor_traj_logits = {}
    rd_teacher_warmup = bool(
        train
        and stage2_enabled
        and stage2_epoch > 0
        and stage2_epoch <= config.low_altitude_rd_teacher_warmup_epochs
    )

    should_compute_aux_loss = train if compute_aux_loss is None else bool(compute_aux_loss)
    drone_index = LOW_ALTITUDE_CLASSES.index("DroneTarget")
    bird_index = LOW_ALTITUDE_CLASSES.index("BirdTarget")
    _phase = "train" if train else "val"
    _total_batches = max_batches if max_batches > 0 else len(loader)
    _epoch_t0 = _time.time()
    _batch_t0 = _epoch_t0
    sys.stderr.write(f"  [{_phase}] entering loop: {_total_batches} batches, loader_len={len(loader)}\n")
    sys.stderr.flush()
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index == 1:
            _first_batch_t = _time.time() - _epoch_t0
            sys.stderr.write(f"  [{_phase}] first batch loaded in {_first_batch_t:.2f}s\n")
            sys.stderr.flush()
        if max_batches > 0 and batch_index > max_batches:
            break
        non_blocking_h2d = bool(config.non_blocking_h2d)
        features = _to_device(batch.features, device, non_blocking_h2d)
        if config.source_normalization and source_feature_stats is not None:
            features = _apply_source_feature_norm(features, batch.source_names, source_feature_stats)
        physical = _to_device(batch.physical_features, device, non_blocking_h2d)
        rd_features = _to_device(batch.rd_features, device, non_blocking_h2d)
        rd_patch_features = _to_device(batch.rd_patch_features, device, non_blocking_h2d)
        rd_deep_patch_features = _to_device(batch.rd_deep_patch_features, device, non_blocking_h2d)
        rd_is_multimodal = _to_device(batch.rd_is_multimodal, device, non_blocking_h2d)
        rd_padding_mask = _to_device(batch.rd_padding_mask, device, non_blocking_h2d)
        rd_patch_padding_mask = _to_device(batch.rd_patch_padding_mask, device, non_blocking_h2d)
        rd_deep_patch_padding_mask = _to_device(batch.rd_deep_patch_padding_mask, device, non_blocking_h2d)
        labels = torch.tensor(
            [LOW_ALTITUDE_GLOBAL_TO_LOCAL[int(label)] for label in batch.labels.tolist()],
            dtype=torch.long,
            device=device,
        )

        with torch.set_grad_enabled(train):
            rep = representation_builder(
                sequence_features=features,
                physical_features=physical,
                padding_mask=_to_device(batch.padding_mask, device, non_blocking_h2d),
            )
            if rd_features.ndim == 4:
                amp_range = rd_features[..., 0].mean(dim=1)
                amp_dopp = rd_features[..., 2].mean(dim=1)
                rd_summary = torch.stack(
                    [
                        amp_range.mean(dim=-1),
                        amp_range.max(dim=-1).values,
                        amp_range.min(dim=-1).values,
                        amp_dopp.mean(dim=-1),
                        amp_dopp.std(dim=-1),
                        rd_is_multimodal.float(),
                    ],
                    dim=-1,
                )
            else:
                rd_summary = rd_features.mean(dim=1)
            rd_embedding = rd_proj(rd_summary)
            output = model(
                rep.fused_features,
                rd_features=rd_features,
                traj_sequence_features=rep.sequence_features,
                rd_patch_features=rd_patch_features,
                rd_deep_patch_features=rd_deep_patch_features,
                rd_context=rd_embedding,
                low_altitude_targets=labels if train else None,
                rd_is_multimodal=rd_is_multimodal,
                traj_padding_mask=_to_device(batch.padding_mask, device, non_blocking_h2d),
                rd_padding_mask=rd_padding_mask,
                rd_patch_padding_mask=rd_patch_padding_mask,
                rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
                enable_reject=(config.enable_reject_in_train if train else config.enable_reject_in_eval) and stage2_enabled,
                disable_tr_rd_fusion=bool(not stage2_enabled),
            )
            unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
            residual_targets_all = _residual_targets(labels)
            main_loss = _weighted_ce_loss(
                criterion,
                output.logits,
                labels,
                unknown_index=unknown_index,
                unknown_weight=config.unknown_loss_weight,
            )
            aux_loss = _weighted_ce_loss(
                criterion,
                output.traj_logits,
                labels,
                unknown_index=unknown_index,
                unknown_weight=config.unknown_loss_weight,
            )
            traj_supervision_loss = aux_loss
            residual_loss = _binary_positive_ce_loss(
                output.traj_residual_logits,
                residual_targets_all,
                positive_weight=max(config.rd_unknown_positive_weight, config.rd_clutter_positive_weight),
            )
            main_loss = main_loss + config.low_altitude_residual_aux_weight * residual_loss
            anchor_loss = torch.tensor(0.0, device=device)
            if train and config.low_altitude_traj_anchor_weight > 0.0:
                if not stage2_enabled:
                    anchor_traj_logits.update(
                        {
                            str(sample_id): logits_row
                            for sample_id, logits_row in zip(batch.sample_ids, output.traj_logits.detach().cpu())
                        }
                    )
                elif anchor_traj_logits and stage2_epoch >= config.low_altitude_traj_anchor_start_epoch:
                    matched_indices: list[int] = []
                    target_rows: list[torch.Tensor] = []
                    for batch_index, sample_id in enumerate(batch.sample_ids):
                        anchor = anchor_traj_logits.get(str(sample_id))
                        if anchor is not None:
                            matched_indices.append(batch_index)
                            target_rows.append(anchor)
                    if matched_indices:
                        index_tensor = torch.tensor(matched_indices, dtype=torch.long, device=device)
                        target_probs = torch.softmax(torch.stack(target_rows).to(device), dim=-1)
                        log_probs = torch.log_softmax(output.traj_logits[index_tensor], dim=-1)
                        anchor_loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction="batchmean")
            has_rd_mask = rd_is_multimodal.bool()
            if stage2_enabled and has_rd_mask.any() and should_compute_aux_loss:
                deep_output = None
                deep_aux_interval = max(1, int(config.low_altitude_deep_reroute_aux_interval))
                run_deep_aux = (
                    config.low_altitude_deep_reroute_aux_weight > 0.0
                    and batch_index % deep_aux_interval == 0
                )
                if run_deep_aux:
                    deep_max_frames = int(max(0, config.low_altitude_deep_reroute_max_frames))
                    (
                        deep_rd_features,
                        deep_rd_patch_features,
                        deep_rd_deep_patch_features,
                        deep_rd_padding_mask,
                        deep_rd_patch_padding_mask,
                        deep_rd_deep_patch_padding_mask,
                    ) = _subsample_deep_reroute_inputs(
                        rd_features,
                        rd_patch_features,
                        rd_deep_patch_features,
                        rd_padding_mask,
                        rd_patch_padding_mask,
                        rd_deep_patch_padding_mask,
                        deep_max_frames,
                    )
                    deep_output = model(
                        rep.fused_features,
                        rd_features=deep_rd_features,
                        traj_sequence_features=rep.sequence_features,
                        rd_patch_features=deep_rd_patch_features,
                        rd_deep_patch_features=deep_rd_deep_patch_features,
                        rd_context=rd_embedding,
                        low_altitude_targets=labels if train else None,
                        rd_is_multimodal=rd_is_multimodal,
                        traj_padding_mask=_to_device(batch.padding_mask, device, non_blocking_h2d),
                        rd_padding_mask=deep_rd_padding_mask,
                        rd_patch_padding_mask=deep_rd_patch_padding_mask,
                        rd_deep_patch_padding_mask=deep_rd_deep_patch_padding_mask,
                        enable_reject=False,
                        inference_mode="deep_reroute",
                    )
                rd_aux = _masked_weighted_ce_loss(
                    output.rd_logits[has_rd_mask],
                    labels[has_rd_mask],
                    rd_class_weights.to(device),
                    unknown_index=unknown_index,
                    unknown_weight=config.unknown_loss_weight,
                    focal_gamma=config.low_altitude_rd_focal_gamma,
                )
                hard_rd_labels = {
                    LOW_ALTITUDE_CLASSES.index("BirdTarget"),
                    LOW_ALTITUDE_CLASSES.index("BalloonTarget"),
                    LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
                    LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
                }
                hard_rd_mask = has_rd_mask & torch.isin(
                    labels,
                    torch.tensor(sorted(hard_rd_labels), dtype=torch.long, device=device),
                )
                if hard_rd_mask.any():
                    hard_rd_aux = _masked_weighted_ce_loss(
                        output.rd_logits[hard_rd_mask],
                        labels[hard_rd_mask],
                        rd_class_weights.to(device),
                        unknown_index=unknown_index,
                        unknown_weight=max(config.unknown_loss_weight, 1.0),
                        focal_gamma=config.low_altitude_rd_focal_gamma,
                    )
                    rd_aux = rd_aux + config.low_altitude_rd_hard_aux_weight * hard_rd_aux
                unknown_binary_targets = (labels[has_rd_mask] == unknown_index).long()
                bird_index = LOW_ALTITUDE_CLASSES.index("BirdTarget")
                bird_binary_targets = (labels[has_rd_mask] == bird_index).long()
                clutter_index = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
                clutter_binary_targets = (labels[has_rd_mask] == clutter_index).long()
                residual_binary_targets = residual_targets_all[has_rd_mask]
                bird_binary_loss = torch.nn.functional.cross_entropy(
                    output.rd_bird_logits[has_rd_mask],
                    bird_binary_targets,
                    weight=torch.tensor([1.0, config.rd_bird_positive_weight], dtype=torch.float32, device=device),
                )
                unknown_binary_loss = _binary_positive_ce_loss(
                    output.rd_unknown_logits[has_rd_mask],
                    unknown_binary_targets,
                    positive_weight=config.rd_unknown_positive_weight,
                )
                clutter_binary_loss = _binary_positive_ce_loss(
                    output.rd_clutter_logits[has_rd_mask],
                    clutter_binary_targets,
                    positive_weight=config.rd_clutter_positive_weight,
                )
                rd_aux = rd_aux + config.low_altitude_rd_binary_aux_weight * (
                    bird_binary_loss + unknown_binary_loss + clutter_binary_loss
                )
                rd_residual_loss = _binary_positive_ce_loss(
                    output.rd_residual_logits[has_rd_mask],
                    residual_binary_targets,
                    positive_weight=max(config.rd_unknown_positive_weight, config.rd_clutter_positive_weight),
                )
                rd_aux = rd_aux + config.low_altitude_rd_residual_aux_weight * rd_residual_loss
                drone_index = LOW_ALTITUDE_CLASSES.index("DroneTarget")
                drone_bird_mask = has_rd_mask & ((labels == drone_index) | (labels == bird_index))
                if drone_bird_mask.any():
                    drone_bird_targets = (labels[drone_bird_mask] == bird_index).long()
                    drone_bird_logits = output.rd_logits[drone_bird_mask][:, [drone_index, bird_index]]
                    pairwise_loss = torch.nn.functional.cross_entropy(
                        drone_bird_logits,
                        drone_bird_targets,
                        weight=torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
                    )
                    rd_aux = rd_aux + config.low_altitude_rd_pairwise_aux_weight * pairwise_loss
                unknown_pairwise_mask = has_rd_mask & ((labels == unknown_index) | (labels == bird_index) | (labels == drone_index))
                if unknown_pairwise_mask.any():
                    unknown_pairwise_targets = (labels[unknown_pairwise_mask] == unknown_index).long()
                    unknown_pairwise_logits = output.rd_unknown_logits[unknown_pairwise_mask]
                    unknown_pairwise_loss = torch.nn.functional.cross_entropy(
                        unknown_pairwise_logits,
                        unknown_pairwise_targets,
                        weight=torch.tensor([1.0, config.rd_unknown_positive_weight], dtype=torch.float32, device=device),
                    )
                    rd_aux = rd_aux + config.low_altitude_rd_unknown_pairwise_aux_weight * unknown_pairwise_loss
                clutter_pairwise_mask = has_rd_mask & (
                    (labels == clutter_index)
                    | (labels == bird_index)
                    | (labels == drone_index)
                    | (labels == LOW_ALTITUDE_CLASSES.index("BalloonTarget"))
                )
                if clutter_pairwise_mask.any():
                    clutter_pairwise_targets = (labels[clutter_pairwise_mask] == clutter_index).long()
                    clutter_pairwise_logits = output.rd_clutter_logits[clutter_pairwise_mask]
                    clutter_pairwise_loss = torch.nn.functional.cross_entropy(
                        clutter_pairwise_logits,
                        clutter_pairwise_targets,
                        weight=torch.tensor([1.0, config.rd_clutter_positive_weight], dtype=torch.float32, device=device),
                    )
                    rd_aux = rd_aux + config.low_altitude_rd_clutter_pairwise_aux_weight * clutter_pairwise_loss
                unknown_clutter_mask = has_rd_mask & (
                    (labels == unknown_index) | (labels == clutter_index)
                )
                if unknown_clutter_mask.any():
                    unknown_clutter_targets = (labels[unknown_clutter_mask] == clutter_index).long()
                    unknown_clutter_logits = output.rd_logits[unknown_clutter_mask][:, [unknown_index, clutter_index]]
                    unknown_clutter_loss = torch.nn.functional.cross_entropy(
                        unknown_clutter_logits,
                        unknown_clutter_targets,
                        weight=torch.tensor([1.0, 1.5], dtype=torch.float32, device=device),
                    )
                    rd_aux = rd_aux + config.low_altitude_rd_unknown_clutter_pairwise_aux_weight * unknown_clutter_loss
                    unknown_clutter_detail_loss = torch.nn.functional.cross_entropy(
                        output.rd_unknown_clutter_logits[unknown_clutter_mask],
                        unknown_clutter_targets,
                        weight=torch.tensor([1.0, config.rd_clutter_positive_weight], dtype=torch.float32, device=device),
                    )
                    rd_aux = rd_aux + config.low_altitude_rd_unknown_clutter_aux_weight * unknown_clutter_detail_loss
                    clutter_morphology_loss = torch.nn.functional.cross_entropy(
                        output.rd_clutter_morphology_logits[unknown_clutter_mask],
                        unknown_clutter_targets,
                        weight=torch.tensor([1.0, config.rd_clutter_positive_weight], dtype=torch.float32, device=device),
                    )
                    rd_aux = rd_aux + config.low_altitude_rd_clutter_morphology_aux_weight * clutter_morphology_loss
                # Drone/Bird micro-Doppler auxiliary loss (only on Drone/Bird samples with RD)
                if output.drone_bird_micro_doppler_logits is not None and config.low_altitude_drone_bird_micro_doppler_aux_weight > 0.0:
                    db_micro_mask = has_rd_mask & ((labels == drone_index) | (labels == bird_index))
                    if db_micro_mask.any():
                        db_micro_targets = (labels[db_micro_mask] == bird_index).long()
                        db_micro_loss = torch.nn.functional.cross_entropy(
                            output.drone_bird_micro_doppler_logits[db_micro_mask],
                            db_micro_targets,
                            weight=torch.tensor([1.0, 1.5], dtype=torch.float32, device=device),
                        )
                        rd_aux = rd_aux + config.low_altitude_drone_bird_micro_doppler_aux_weight * db_micro_loss
                rare_margin_loss = (
                    _target_margin_loss(output.rd_logits[has_rd_mask], labels[has_rd_mask], bird_index, config.low_altitude_rd_rare_margin)
                    + _target_margin_loss(output.rd_logits[has_rd_mask], labels[has_rd_mask], LOW_ALTITUDE_CLASSES.index("BalloonTarget"), config.low_altitude_rd_rare_margin)
                    + _target_margin_loss(output.rd_logits[has_rd_mask], labels[has_rd_mask], unknown_index, config.low_altitude_rd_rare_margin)
                    + _target_margin_loss(output.rd_logits[has_rd_mask], labels[has_rd_mask], clutter_index, config.low_altitude_rd_rare_margin)
                )
                rd_aux = rd_aux + config.low_altitude_rd_rare_margin_aux_weight * rare_margin_loss
                rd_calibration_loss = _rd_confidence_calibration_loss(
                    output.rd_logits[has_rd_mask],
                    labels[has_rd_mask],
                    positive_target=config.low_altitude_rd_calibration_positive_target,
                    negative_target=config.low_altitude_rd_calibration_negative_target,
                )
                rd_aux = rd_aux + config.low_altitude_rd_calibration_aux_weight * rd_calibration_loss
                forced_fusion_loss = _masked_weighted_ce_loss(
                    output.logits[has_rd_mask],
                    labels[has_rd_mask],
                    rd_class_weights.to(device),
                    unknown_index=unknown_index,
                    unknown_weight=config.unknown_loss_weight,
                    focal_gamma=config.low_altitude_rd_focal_gamma,
                )
                main_loss = main_loss + (config.low_altitude_rd_forced_fusion_weight * stage2_scale) * forced_fusion_loss
                if output.joint_logits is not None and config.low_altitude_joint_aux_weight > 0.0:
                    joint_aux_loss = _masked_weighted_ce_loss(
                        output.joint_logits[has_rd_mask],
                        labels[has_rd_mask],
                        rd_class_weights.to(device),
                        unknown_index=unknown_index,
                        unknown_weight=config.unknown_loss_weight,
                        focal_gamma=config.low_altitude_rd_focal_gamma,
                    )
                    main_loss = main_loss + (config.low_altitude_joint_aux_weight * stage2_scale) * joint_aux_loss
                if output.joint_reliability_logits is not None and config.low_altitude_joint_reliability_aux_weight > 0.0:
                    with torch.no_grad():
                        traj_pred = output.traj_logits.argmax(dim=-1)
                        rd_pred = output.rd_logits.argmax(dim=-1)
                        traj_correct = traj_pred == labels
                        rd_correct = rd_pred == labels
                        rd_rescue = rd_correct & ~traj_correct
                        rd_harm = ~rd_correct & traj_correct
                        reliability_target = torch.full_like(output.joint_rd_weight, 0.20)
                        reliability_target = torch.where(rd_rescue, torch.full_like(reliability_target, 0.40), reliability_target)
                        reliability_target = torch.where(rd_harm, torch.full_like(reliability_target, 0.05), reliability_target)
                        reliability_target = torch.where(~has_rd_mask, torch.zeros_like(reliability_target), reliability_target)
                    reliability_loss = torch.nn.functional.mse_loss(
                        output.joint_rd_weight[has_rd_mask],
                        reliability_target[has_rd_mask],
                    )
                    main_loss = main_loss + (config.low_altitude_joint_reliability_aux_weight * stage2_scale) * reliability_loss
                if deep_output is not None:
                    deep_reroute_loss = _masked_weighted_ce_loss(
                        deep_output.logits[has_rd_mask],
                        labels[has_rd_mask],
                        rd_class_weights.to(device),
                        unknown_index=unknown_index,
                        unknown_weight=config.unknown_loss_weight,
                        focal_gamma=config.low_altitude_rd_focal_gamma,
                    )
                    deep_reroute_rd_loss = _masked_weighted_ce_loss(
                        deep_output.rd_logits[has_rd_mask],
                        labels[has_rd_mask],
                        rd_class_weights.to(device),
                        unknown_index=unknown_index,
                        unknown_weight=config.unknown_loss_weight,
                        focal_gamma=config.low_altitude_rd_focal_gamma,
                    )
                    main_loss = main_loss + (config.low_altitude_deep_reroute_aux_weight * stage2_scale) * (
                        deep_reroute_loss + 0.5 * deep_reroute_rd_loss
                    )
                if rd_teacher_warmup:
                    aux_loss = (
                        config.low_altitude_traj_aux_ratio * aux_loss
                        + (config.low_altitude_rd_aux_ratio * config.low_altitude_rd_teacher_aux_boost) * rd_aux
                    )
                else:
                    aux_loss = (
                        config.low_altitude_traj_aux_ratio * aux_loss
                        + config.low_altitude_rd_aux_ratio * rd_aux
                    )
                if config.low_altitude_mutual_exclude_rare_classes:
                    mutual_mask = has_rd_mask & (labels != unknown_index) & (labels != clutter_index)
                else:
                    mutual_mask = has_rd_mask
                if mutual_mask.any():
                    mutual_loss = _symmetric_kl(
                        output.traj_logits[mutual_mask],
                        output.rd_logits[mutual_mask],
                    )
                else:
                    mutual_loss = torch.tensor(0.0, device=device)
                if rd_teacher_warmup:
                    mutual_loss = torch.tensor(0.0, device=device)
                contrastive_mask = has_rd_mask & (labels != unknown_index)
                if contrastive_mask.any():
                    contrastive_loss = _supervised_contrastive_loss(
                        torch.cat(
                            [
                                output.traj_features[contrastive_mask],
                                output.rd_features[contrastive_mask],
                            ],
                            dim=-1,
                        ),
                        labels[contrastive_mask],
                        temperature=config.low_altitude_temperature,
                    )
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
                if rd_teacher_warmup:
                    contrastive_loss = torch.tensor(0.0, device=device)
            else:
                mutual_loss = torch.tensor(0.0, device=device)
                contrastive_loss = torch.tensor(0.0, device=device)
            # Trajectory Drone-Bird pairwise: covers ALL samples (including no-RD excel_track)
            if should_compute_aux_loss and config.low_altitude_traj_pairwise_aux_weight > 0.0:
                _t_db_mask = (labels == drone_index) | (labels == bird_index)
                if _t_db_mask.any():
                    _t_db_targets = (labels[_t_db_mask] == bird_index).long()
                    _t_db_logits = output.traj_logits[_t_db_mask][:, [drone_index, bird_index]]
                    _t_db_loss = torch.nn.functional.cross_entropy(
                        _t_db_logits,
                        _t_db_targets,
                        weight=torch.tensor([1.0, 1.5], dtype=torch.float32, device=device),
                    )
                    main_loss = main_loss + config.low_altitude_traj_pairwise_aux_weight * _t_db_loss
            loss = (
                main_loss
                + config.low_altitude_traj_supervision_weight * traj_supervision_loss
                + (config.low_altitude_aux_weight * stage2_scale) * aux_loss
                + (config.low_altitude_traj_anchor_weight * stage2_scale) * anchor_loss
                + (config.low_altitude_mutual_weight * stage2_scale) * mutual_loss
                + (config.low_altitude_contrastive_weight * stage2_scale) * contrastive_loss
            )

            if train:
                prev_state = (
                    _clone_bundle_state_to_cpu(representation_builder, rd_proj, model)
                    if config.enable_batch_parameter_rollback
                    else None
                )
                if not torch.isfinite(loss):
                    if config.skip_non_finite_batch:
                        skipped_non_finite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise RuntimeError("non-finite loss detected during expert1 training")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(representation_builder.parameters()) + list(rd_proj.parameters()) + list(model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                if config.enable_batch_parameter_rollback and not _parameters_are_finite_bundle(representation_builder, rd_proj, model):
                    if prev_state is None:
                        raise RuntimeError("missing rollback state after non-finite parameter detection")
                    representation_builder.load_state_dict(prev_state["representation_builder_state_dict"])
                    rd_proj.load_state_dict(prev_state["rd_proj_state_dict"])
                    model.load_state_dict(prev_state["model_state_dict"])
                    skipped_non_finite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                if not torch.isfinite(loss):
                    skipped_non_finite_batches += 1
                    continue

        loss_values.append(float(loss.detach().cpu().item()))
        logits_all.append(output.logits.detach().cpu())
        traj_logits_all.append(output.traj_logits.detach().cpu())
        rd_logits_all.append(output.rd_logits.detach().cpu())
        traj_residual_logits_all.append(output.traj_residual_logits.detach().cpu())
        rd_residual_logits_all.append(output.rd_residual_logits.detach().cpu())
        rd_unknown_clutter_logits_all.append(output.rd_unknown_clutter_logits.detach().cpu())
        rd_clutter_morphology_logits_all.append(output.rd_clutter_morphology_logits.detach().cpu())
        known_confidence_all.append(output.known_confidence.detach().cpu())
        residual_score_all.append(output.residual_score.detach().cpu())
        labels_all.append(labels.detach().cpu())
        sample_ids_all.extend(batch.sample_ids)
        source_names_all.extend(batch.source_names)
        label_names_all.extend(batch.label_names)
        coarse_label_names_all.extend(batch.coarse_label_names)
        fine_label_names_all.extend(batch.fine_label_names)
        has_rd_all.extend([bool(x) for x in batch.rd_is_multimodal.tolist()])
        rd_time_sync_status_all.extend([str(x or "unavailable") for x in batch.rd_time_sync_status])
        tr_rd_active_all.extend([bool(x) for x in output.tr_rd_active.detach().cpu().tolist()])
        traj_cosine_all.append(output.traj_cosine.detach().cpu())
        rd_cosine_all.append(output.rd_cosine.detach().cpu())
        fusion_weights_all.append(output.fusion_weights.detach().cpu())
        if output.joint_logits is not None:
            joint_logits_all.append(output.joint_logits.detach().cpu())
        if output.joint_rd_weight is not None:
            joint_rd_weights_all.append(output.joint_rd_weight.detach().cpu())
        if output.rd_class_fusion_weights is not None:
            rd_class_fusion_weights_all.append(output.rd_class_fusion_weights.detach().cpu())
        if output.rd_pair_fusion_weights is not None:
            rd_pair_fusion_weights_all.append(output.rd_pair_fusion_weights.detach().cpu())

        # Progress logging every 50 batches (or every batch if < 50 total)
        _log_interval = max(1, min(50, _total_batches // 4))
        if batch_index % _log_interval == 0 or batch_index == _total_batches:
            _batch_elapsed = _time.time() - _batch_t0
            _epoch_elapsed = _time.time() - _epoch_t0
            _avg_batch = _epoch_elapsed / batch_index
            _eta = _avg_batch * (_total_batches - batch_index)
            _has_rd_count = sum(1 for x in batch.rd_is_multimodal.tolist() if x)
            print(
                f"  [{_phase}] batch {batch_index}/{_total_batches} "
                f"batch_time={_batch_elapsed:.1f}s "
                f"avg={_avg_batch:.1f}s/batch "
                f"elapsed={_epoch_elapsed:.0f}s "
                f"ETA={_eta:.0f}s "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"rd={_has_rd_count}/{len(batch.rd_is_multimodal)}",
                flush=True,
            )
            _batch_t0 = _time.time()

    if not logits_all:
        precision_logit_bias = _normalize_precision_logit_bias(getattr(model, "precision_logit_bias", None))
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "skipped_non_finite_batches": skipped_non_finite_batches,
            "has_valid_batches": False,
            "evaluated_batches": 0,
            "max_batches": int(max_batches),
            "logits": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "traj_logits": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "joint_logits": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "rd_logits": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
            "sample_ids": sample_ids_all,
            "source_names": source_names_all,
            "label_names": label_names_all,
            "coarse_label_names": coarse_label_names_all,
            "fine_label_names": fine_label_names_all,
            "has_rd": has_rd_all,
            "rd_time_sync_status": rd_time_sync_status_all,
            "tr_rd_active": tr_rd_active_all,
            "rd_activated": tr_rd_active_all,
            "traj_cosine": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "rd_cosine": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "rd_pair_fusion_weights": torch.empty((0, 3), dtype=torch.float32),
            "selected_joint_logit_delta_scale": float(config.joint_logit_delta_scale),
            "precision_logit_bias": precision_logit_bias,
            "precision_logit_bias_by_label": _precision_logit_bias_dict(precision_logit_bias),
            "joint_logit_delta_probe": {},
            "tr_rd_path_rate": 0.0,
            "tr_rd_path_rate_on_rd": 0.0,
            "rd_activation_rate": 0.0,
            "rd_activation_rate_on_rd": 0.0,
            "rd_fusion_weight_mean": 0.0,
            "rd_fusion_weight_on_rd_mean": 0.0,
            "rd_class_fusion_weight_mean": 0.0,
            "rd_class_fusion_weight_on_rd_mean": 0.0,
            "rd_pair_fusion_weight_mean": 0.0,
            "rd_pair_fusion_weight_on_rd_mean": 0.0,
            "rd_pair_fusion_weight_by_pair": {},
            "rd_pair_fusion_weight_on_rd_by_pair": {},
            "joint_accuracy": 0.0,
            "joint_macro_f1": 0.0,
            "joint_accuracy_on_rd": 0.0,
            "joint_macro_f1_on_rd": 0.0,
            "traj_accuracy_on_rd": 0.0,
            "traj_macro_f1_on_rd": 0.0,
            "joint_minus_traj_macro_f1_on_rd": 0.0,
            "joint_rd_weight_mean": 0.0,
            "joint_rd_weight_on_rd_mean": 0.0,
            "traj_residual_accuracy": 0.0,
            "rd_residual_accuracy_on_rd": 0.0,
            "rd_unknown_clutter_detail_accuracy_on_residual_rd": 0.0,
            "rd_clutter_morphology_accuracy_on_residual_rd": 0.0,
        }

    logits_cat = torch.cat(logits_all, dim=0)
    precision_logit_bias = _normalize_precision_logit_bias(getattr(model, "precision_logit_bias", None))
    traj_logits_cat = torch.cat(traj_logits_all, dim=0)
    rd_logits_cat = torch.cat(rd_logits_all, dim=0)
    traj_cosine_cat = torch.cat(traj_cosine_all, dim=0)
    rd_cosine_cat = torch.cat(rd_cosine_all, dim=0)
    traj_residual_logits_cat = torch.cat(traj_residual_logits_all, dim=0)
    rd_residual_logits_cat = torch.cat(rd_residual_logits_all, dim=0)
    rd_unknown_clutter_logits_cat = torch.cat(rd_unknown_clutter_logits_all, dim=0)
    rd_clutter_morphology_logits_cat = torch.cat(rd_clutter_morphology_logits_all, dim=0)
    known_confidence_cat = torch.cat(known_confidence_all, dim=0)
    residual_score_cat = torch.cat(residual_score_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    residual_targets_cat = _residual_targets(labels_cat)
    rare_class_indices = torch.tensor(
        [
            LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
            LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
        ],
        dtype=torch.long,
    )
    rare_class_mask = torch.isin(labels_cat.cpu(), rare_class_indices)
    metrics = compute_classification_metrics(logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    traj_metrics = compute_classification_metrics(traj_logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    rd_metrics = compute_classification_metrics(rd_logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    joint_logits_cat = torch.cat(joint_logits_all, dim=0) if joint_logits_all else logits_cat
    has_rd_tensor = torch.tensor(has_rd_all, dtype=torch.bool)
    tr_rd_active_tensor = torch.tensor(tr_rd_active_all, dtype=torch.bool)
    joint_metrics = compute_classification_metrics(joint_logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    traj_metrics_on_rd = (
        compute_classification_metrics(traj_logits_cat[has_rd_tensor], labels_cat[has_rd_tensor], num_classes=len(LOW_ALTITUDE_CLASSES))
        if bool(has_rd_tensor.any())
        else None
    )
    joint_metrics_on_rd = (
        compute_classification_metrics(joint_logits_cat[has_rd_tensor], labels_cat[has_rd_tensor], num_classes=len(LOW_ALTITUDE_CLASSES))
        if bool(has_rd_tensor.any())
        else None
    )
    rd_metrics_on_rd = (
        compute_classification_metrics(rd_logits_cat[has_rd_tensor], labels_cat[has_rd_tensor], num_classes=len(LOW_ALTITUDE_CLASSES))
        if bool(has_rd_tensor.any())
        else None
    )
    diagnostics: dict[str, object] = {}
    diagnostics.update(_expert1_diagnostics(logits_cat, labels_cat, "final"))
    diagnostics.update(_expert1_diagnostics(traj_logits_cat, labels_cat, "traj"))
    diagnostics.update(_expert1_diagnostics(rd_logits_cat, labels_cat, "rd"))
    diagnostics.update(_expert1_diagnostics(rd_logits_cat, labels_cat, "rd_on_rd", mask=has_rd_tensor))
    rd_hard_pair_report = _hard_pair_report(
        logits_cat,
        traj_logits_cat,
        rd_logits_cat,
        labels_cat,
        has_rd_tensor,
    )
    rd_teacher_effect = _rd_teacher_effect_metrics(
        logits_cat,
        traj_logits_cat,
        rd_logits_cat,
        labels_cat,
        has_rd_tensor,
        source_names_all,
    )
    by_source_metrics = _source_metric_dict(logits_cat, labels_cat, source_names_all)
    traj_by_source_metrics = _source_metric_dict(traj_logits_cat, labels_cat, source_names_all)
    joint_by_source_metrics = _source_metric_dict(joint_logits_cat, labels_cat, source_names_all)
    by_source_supported_metrics = _source_supported_metric_dict(logits_cat, labels_cat, source_names_all)
    joint_by_source_metrics_on_rd = _source_metric_dict(
        joint_logits_cat,
        labels_cat,
        source_names_all,
        mask=has_rd_tensor,
    )
    rd_by_source_metrics_on_rd = _source_metric_dict(
        rd_logits_cat,
        labels_cat,
        source_names_all,
        mask=has_rd_tensor,
    )
    rd_time_sync_metrics = _rd_time_sync_metric_dict(
        logits_cat,
        traj_logits_cat,
        joint_logits_cat,
        rd_logits_cat,
        labels_cat,
        rd_time_sync_status_all,
        has_rd_tensor,
    )
    group_metrics = _group_accuracy_metrics(logits_cat, labels_cat, source_names_all, has_rd_all)
    tr_rd_path_rate = float(tr_rd_active_tensor.float().mean().item()) if tr_rd_active_tensor.numel() else 0.0
    tr_rd_path_rate_on_rd = (
        float(tr_rd_active_tensor[has_rd_tensor].float().mean().item())
        if bool(has_rd_tensor.any())
        else 0.0
    )
    fusion_weights_cat = torch.cat(fusion_weights_all, dim=0) if fusion_weights_all else torch.empty((0, 2), dtype=torch.float32)
    joint_rd_weights_cat = torch.cat(joint_rd_weights_all, dim=0) if joint_rd_weights_all else torch.empty((0,), dtype=torch.float32)
    rd_fusion_weight_mean = float(fusion_weights_cat[:, 1].mean().item()) if fusion_weights_cat.numel() else 0.0
    rd_fusion_weight_on_rd_mean = (
        float(fusion_weights_cat[has_rd_tensor, 1].mean().item())
        if bool(has_rd_tensor.any()) and fusion_weights_cat.numel()
        else 0.0
    )
    joint_rd_weight_mean = float(joint_rd_weights_cat.mean().item()) if joint_rd_weights_cat.numel() else rd_fusion_weight_mean
    joint_rd_weight_on_rd_mean = (
        float(joint_rd_weights_cat[has_rd_tensor].mean().item())
        if bool(has_rd_tensor.any()) and joint_rd_weights_cat.numel()
        else rd_fusion_weight_on_rd_mean
    )
    rd_class_weights_cat = (
        torch.cat(rd_class_fusion_weights_all, dim=0)
        if rd_class_fusion_weights_all
        else torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32)
    )
    rd_pair_weights_cat = (
        torch.cat(rd_pair_fusion_weights_all, dim=0)
        if rd_pair_fusion_weights_all
        else torch.empty((0, 3), dtype=torch.float32)
    )
    rd_class_fusion_weight_mean = float(rd_class_weights_cat.mean().item()) if rd_class_weights_cat.numel() else 0.0
    rd_class_fusion_weight_on_rd_mean = (
        float(rd_class_weights_cat[has_rd_tensor].mean().item())
        if bool(has_rd_tensor.any()) and rd_class_weights_cat.numel()
        else 0.0
    )
    rd_pair_fusion_weight_mean = float(rd_pair_weights_cat.mean().item()) if rd_pair_weights_cat.numel() else 0.0
    rd_pair_fusion_weight_on_rd_mean = (
        float(rd_pair_weights_cat[has_rd_tensor].mean().item())
        if bool(has_rd_tensor.any()) and rd_pair_weights_cat.numel()
        else 0.0
    )
    rd_pair_names = ["DroneBird", "UnknownClutter", "Balloon"]
    rd_pair_weight_by_pair = {
        pair_name: float(rd_pair_weights_cat[:, idx].mean().item()) if rd_pair_weights_cat.numel() else 0.0
        for idx, pair_name in enumerate(rd_pair_names)
    }
    rd_pair_weight_on_rd_by_pair = {
        pair_name: (
            float(rd_pair_weights_cat[has_rd_tensor, idx].mean().item())
            if bool(has_rd_tensor.any()) and rd_pair_weights_cat.numel()
            else 0.0
        )
        for idx, pair_name in enumerate(rd_pair_names)
    }
    residual_rd_mask = has_rd_tensor & residual_targets_cat.bool()
    traj_residual_accuracy = float((traj_residual_logits_cat.argmax(dim=-1) == residual_targets_cat).float().mean().item())
    rd_residual_accuracy_on_rd = (
        float((rd_residual_logits_cat[has_rd_tensor].argmax(dim=-1) == residual_targets_cat[has_rd_tensor]).float().mean().item())
        if bool(has_rd_tensor.any())
        else 0.0
    )
    if bool(residual_rd_mask.any()):
        residual_detail_targets = (labels_cat[residual_rd_mask] == LOW_ALTITUDE_CLASSES.index("ClutterTarget")).long()
        rd_unknown_clutter_detail_accuracy = float(
            (rd_unknown_clutter_logits_cat[residual_rd_mask].argmax(dim=-1) == residual_detail_targets).float().mean().item()
        )
        rd_clutter_morphology_accuracy = float(
            (rd_clutter_morphology_logits_cat[residual_rd_mask].argmax(dim=-1) == residual_detail_targets).float().mean().item()
        )
        rd_unknown_clutter_detail_clutter_recall = (
            float(
                (
                    rd_unknown_clutter_logits_cat[residual_rd_mask][residual_detail_targets.bool()].argmax(dim=-1)
                    == 1
                ).float().mean().item()
            )
            if bool(residual_detail_targets.bool().any())
            else 0.0
        )
        rd_clutter_morphology_clutter_recall = (
            float(
                (
                    rd_clutter_morphology_logits_cat[residual_rd_mask][residual_detail_targets.bool()].argmax(dim=-1)
                    == 1
                ).float().mean().item()
            )
            if bool(residual_detail_targets.bool().any())
            else 0.0
        )
    else:
        rd_unknown_clutter_detail_accuracy = 0.0
        rd_clutter_morphology_accuracy = 0.0
        rd_unknown_clutter_detail_clutter_recall = 0.0
        rd_clutter_morphology_clutter_recall = 0.0
    rd_drone_bird = diagnostics["rd_on_rd_drone_bird"]
    rd_unknown_clutter = float(diagnostics["rd_on_rd_unknown_clutter_recall_mean"])
    traj_by_source_accuracy = _source_accuracy_dict(traj_logits_cat, labels_cat, source_names_all)
    traj_by_source_macro_f1 = _source_macro_f1_dict(traj_logits_cat, labels_cat, source_names_all)
    joint_by_source_accuracy = _source_accuracy_dict(joint_logits_cat, labels_cat, source_names_all)
    joint_by_source_macro_f1 = _source_macro_f1_dict(joint_logits_cat, labels_cat, source_names_all)
    joint_by_source_accuracy_on_rd = _source_accuracy_dict(
        joint_logits_cat,
        labels_cat,
        source_names_all,
        mask=has_rd_tensor,
    )
    joint_by_source_macro_f1_on_rd = _source_macro_f1_dict(
        joint_logits_cat,
        labels_cat,
        source_names_all,
        mask=has_rd_tensor,
    )
    rd_by_source_accuracy_on_rd = _source_accuracy_dict(
        rd_logits_cat,
        labels_cat,
        source_names_all,
        mask=has_rd_tensor,
    )
    rd_by_source_macro_f1_on_rd = _source_macro_f1_dict(
        rd_logits_cat,
        labels_cat,
        source_names_all,
        mask=has_rd_tensor,
    )
    source_support = _source_support_dict(source_names_all)
    rd_source_support_on_rd = _source_support_dict(source_names_all, mask=has_rd_tensor)
    rd_class_support_on_rd = {
        label_name: int((labels_cat[has_rd_tensor] == idx).sum().item())
        for idx, label_name in enumerate(LOW_ALTITUDE_CLASSES)
    }
    rd_by_source_summary = _source_summary_from_dict(rd_by_source_accuracy_on_rd)
    rd_by_source_macro_f1_summary = _source_macro_f1_summary(rd_by_source_macro_f1_on_rd)
    return {
        "loss": float(np.mean(loss_values)) if loss_values else float("nan"),
        "evaluated_batches": len(loss_values),
        "max_batches": int(max_batches),
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "traj_accuracy": traj_metrics.accuracy,
        "traj_macro_f1": traj_metrics.macro_f1,
        "traj_accuracy_on_rd": traj_metrics_on_rd.accuracy if traj_metrics_on_rd is not None else 0.0,
        "traj_macro_f1_on_rd": traj_metrics_on_rd.macro_f1 if traj_metrics_on_rd is not None else 0.0,
        "joint_accuracy": joint_metrics.accuracy,
        "joint_macro_f1": joint_metrics.macro_f1,
        "joint_accuracy_on_rd": joint_metrics_on_rd.accuracy if joint_metrics_on_rd is not None else 0.0,
        "joint_macro_f1_on_rd": joint_metrics_on_rd.macro_f1 if joint_metrics_on_rd is not None else 0.0,
        "joint_minus_traj_macro_f1_on_rd": (
            (joint_metrics_on_rd.macro_f1 - traj_metrics_on_rd.macro_f1)
            if joint_metrics_on_rd is not None and traj_metrics_on_rd is not None
            else 0.0
        ),
        "rd_accuracy": rd_metrics.accuracy,
        "rd_macro_f1": rd_metrics.macro_f1,
        "rd_accuracy_on_rd": rd_metrics_on_rd.accuracy if rd_metrics_on_rd is not None else 0.0,
        "rd_macro_f1_on_rd": rd_metrics_on_rd.macro_f1 if rd_metrics_on_rd is not None else 0.0,
        "rd_per_class_recall": diagnostics["rd_per_class_recall"],
        "rd_per_class_recall_on_rd": diagnostics["rd_on_rd_per_class_recall"],
        "rd_drone_bird": rd_drone_bird,
        "rd_drone_bird_recall_mean": float(rd_drone_bird["drone_bird_recall_mean"]),
        "rd_unknown_clutter_recall_mean": rd_unknown_clutter,
        "rd_drone_as_bird_rate": float(rd_drone_bird["drone_as_bird_rate"]),
        "rd_bird_as_drone_rate": float(rd_drone_bird["bird_as_drone_rate"]),
        "rd_hard_pair_report": rd_hard_pair_report,
        **rd_teacher_effect,
        "traj_per_class_recall": diagnostics["traj_per_class_recall"],
        "traj_drone_bird": diagnostics["traj_drone_bird"],
        "final_per_class_recall": diagnostics["final_per_class_recall"],
        "final_drone_bird": diagnostics["final_drone_bird"],
        "final_unknown_clutter_recall_mean": diagnostics["final_unknown_clutter_recall_mean"],
        "tr_rd_path_rate": tr_rd_path_rate,
        "tr_rd_path_rate_on_rd": tr_rd_path_rate_on_rd,
        "rd_activation_rate": tr_rd_path_rate,
        "rd_activation_rate_on_rd": tr_rd_path_rate_on_rd,
        "rd_fusion_weight_mean": rd_fusion_weight_mean,
        "rd_fusion_weight_on_rd_mean": rd_fusion_weight_on_rd_mean,
        "rd_class_fusion_weight_mean": rd_class_fusion_weight_mean,
        "rd_class_fusion_weight_on_rd_mean": rd_class_fusion_weight_on_rd_mean,
        "rd_pair_fusion_weight_mean": rd_pair_fusion_weight_mean,
        "rd_pair_fusion_weight_on_rd_mean": rd_pair_fusion_weight_on_rd_mean,
        "rd_pair_fusion_weight_by_pair": rd_pair_weight_by_pair,
        "rd_pair_fusion_weight_on_rd_by_pair": rd_pair_weight_on_rd_by_pair,
        "joint_rd_weight_mean": joint_rd_weight_mean,
        "joint_rd_weight_on_rd_mean": joint_rd_weight_on_rd_mean,
        "known_confidence_mean": float(known_confidence_cat.mean().item()) if known_confidence_cat.numel() else 0.0,
        "residual_score_mean": float(residual_score_cat.mean().item()) if residual_score_cat.numel() else 0.0,
        "residual_score_on_residual_mean": (
            float(residual_score_cat[residual_targets_cat.bool()].mean().item())
            if bool(residual_targets_cat.bool().any())
            else 0.0
        ),
        "traj_residual_accuracy": traj_residual_accuracy,
        "rd_residual_accuracy_on_rd": rd_residual_accuracy_on_rd,
        "rd_unknown_clutter_detail_accuracy_on_residual_rd": rd_unknown_clutter_detail_accuracy,
        "rd_clutter_morphology_accuracy_on_residual_rd": rd_clutter_morphology_accuracy,
        "rd_unknown_clutter_detail_clutter_recall": rd_unknown_clutter_detail_clutter_recall,
        "rd_clutter_morphology_clutter_recall": rd_clutter_morphology_clutter_recall,
        "by_source_metrics": by_source_metrics,
        "by_source_supported_metrics": by_source_supported_metrics,
        "traj_by_source_metrics": traj_by_source_metrics,
        "joint_by_source_metrics": joint_by_source_metrics,
        "joint_by_source_metrics_on_rd": joint_by_source_metrics_on_rd,
        "rd_by_source_metrics_on_rd": rd_by_source_metrics_on_rd,
        "rd_time_sync_metrics": rd_time_sync_metrics,
        "by_source_accuracy": group_metrics["by_source_accuracy"],
        "by_source_macro_f1": group_metrics["by_source_macro_f1"],
        "by_source_supported_metrics": group_metrics["by_source_supported_metrics"],
        "traj_by_source_accuracy": traj_by_source_accuracy,
        "traj_by_source_macro_f1": traj_by_source_macro_f1,
        "joint_by_source_accuracy": joint_by_source_accuracy,
        "joint_by_source_macro_f1": joint_by_source_macro_f1,
        "joint_by_source_accuracy_on_rd": joint_by_source_accuracy_on_rd,
        "joint_by_source_macro_f1_on_rd": joint_by_source_macro_f1_on_rd,
        "rd_by_source_accuracy_on_rd": rd_by_source_accuracy_on_rd,
        "rd_by_source_macro_f1_on_rd": rd_by_source_macro_f1_on_rd,
        "source_support": source_support,
        "rd_source_support_on_rd": rd_source_support_on_rd,
        "rd_class_support_on_rd": rd_class_support_on_rd,
        "rd_source_balanced_accuracy_on_rd": rd_by_source_summary["source_balanced_accuracy"],
        "rd_min_source_accuracy_on_rd": rd_by_source_summary["min_source_accuracy"],
        "rd_source_balanced_macro_f1_on_rd": rd_by_source_macro_f1_summary["source_balanced_macro_f1"],
        "rd_min_source_macro_f1_on_rd": rd_by_source_macro_f1_summary["min_source_macro_f1"],
        "raw_tr_rd_accuracy": group_metrics["raw_tr_rd_accuracy"],
        "source_balanced_accuracy": group_metrics["source_balanced_accuracy"],
        "min_source_accuracy": group_metrics["min_source_accuracy"],
        "raw_tr_rd_macro_f1": group_metrics["raw_tr_rd_macro_f1"],
        "excel_track_macro_f1": group_metrics["excel_track_macro_f1"],
        "source_balanced_macro_f1": group_metrics["source_balanced_macro_f1"],
        "min_source_macro_f1": group_metrics["min_source_macro_f1"],
        "source_balanced_supported_macro_f1": group_metrics["source_balanced_supported_macro_f1"],
        "min_source_supported_macro_f1": group_metrics["min_source_supported_macro_f1"],
        "raw_tr_rd_supported_macro_f1": group_metrics["raw_tr_rd_supported_macro_f1"],
        "excel_track_supported_macro_f1": group_metrics["excel_track_supported_macro_f1"],
        "with_rd_accuracy": group_metrics["with_rd_accuracy"],
        "without_rd_accuracy": group_metrics["without_rd_accuracy"],
        "with_rd_fraction": float(has_rd_tensor.float().mean().item()) if has_rd_tensor.numel() else 0.0,
        "rare_class_fraction": float(rare_class_mask.float().mean().item()) if rare_class_mask.numel() else 0.0,
        **_train_batch_distribution_summary(
            sample_ids_all,
            source_names_all,
            label_names_all,
            has_rd_all,
        ),
        "skipped_non_finite_batches": skipped_non_finite_batches,
        "has_valid_batches": True,
        "logits": logits_cat,
        "traj_logits": traj_logits_cat,
        "joint_logits": joint_logits_cat,
        "rd_logits": rd_logits_cat,
        "labels": labels_cat,
        "sample_ids": sample_ids_all,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "has_rd": has_rd_all,
        "rd_time_sync_status": rd_time_sync_status_all,
        "tr_rd_active": tr_rd_active_all,
        "rd_activated": tr_rd_active_all,
        "traj_cosine": traj_cosine_cat,
        "rd_cosine": rd_cosine_cat,
        "rd_pair_fusion_weights": rd_pair_weights_cat,
        "selected_joint_logit_delta_scale": float(config.joint_logit_delta_scale),
        "precision_logit_bias": precision_logit_bias,
        "precision_logit_bias_by_label": _precision_logit_bias_dict(precision_logit_bias),
        "joint_logit_delta_probe": {},
    }


def train_expert1_v1(config: Expert1TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    loaders = build_expert1_dataloaders(
        data_root=config.data_root,
        index_name=config.index_name,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        eval_num_workers=config.eval_num_workers,
        rd_sample_boost=config.rd_sample_boost,
        raw_source_boost=config.raw_source_boost,
        excel_source_boost=config.excel_source_boost,
        cq08_source_boost=config.cq08_source_boost,
        source_balance_power=config.source_balance_power,
        source_label_balance_power=config.source_label_balance_power,
        parent_window_balance_power=config.parent_window_balance_power,
        stage2_rd_aware_sampling=config.stage2_rd_aware_sampling,
        stage2_rd_batch_fraction=config.stage2_rd_batch_fraction,
        stage2_rd_shard_local_sampling=config.stage2_rd_shard_local_sampling,
        max_sampler_weight=config.max_sampler_weight,
        unknown_sample_boost=config.unknown_sample_boost,
        clutter_sample_boost=config.clutter_sample_boost,
        bird_sample_boost=config.bird_sample_boost,
        dataset_cache_size=config.dataset_cache_size,
        rd_encoded_cache_size=config.rd_encoded_cache_size,
        enable_window_rd_time_crop=config.enable_window_rd_time_crop,
        rd_window_crop_min_frames=config.rd_window_crop_min_frames,
        use_rd_segment_store=config.use_rd_segment_store,
        rd_segment_manifest_name=config.rd_segment_manifest_name,
        window_segment_map_name=config.window_segment_map_name,
        rd_segment_store_cache_size=config.rd_segment_store_cache_size,
        allow_rd_segment_store_fallback=config.allow_rd_segment_store_fallback,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        validate_csv_files=False,
    )
    representation_builder = TrackRepresentationBuilder(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=128,
        dropout=config.dropout,
    ).to(device)
    rd_dim = 16
    rd_proj = nn.Sequential(
        nn.Linear(6, rd_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(rd_dim, rd_dim),
        nn.LayerNorm(rd_dim),
    ).to(device)
    model = LowAltitudeSuspectExpert(
        input_dim=128 + 32,
        rd_input_dim=4,
        rd_patch_input_dim=3,
        rd_context_dim=rd_dim,
        traj_sequence_dim=128,
        use_rd_patch_cnn=config.use_rd_patch_cnn,
        dropout=config.dropout,
        cosface_scale=config.cosface_scale,
        cosface_margin=config.cosface_margin,
        reject_threshold=config.reject_threshold,
        rd_bird_logit_scale=config.rd_bird_logit_scale,
        rd_unknown_logit_scale=config.rd_unknown_logit_scale,
        rd_clutter_logit_scale=config.rd_clutter_logit_scale,
        rd_residual_logit_scale=config.rd_residual_logit_scale,
        rd_unknown_clutter_logit_scale=config.rd_unknown_clutter_logit_scale,
        rd_clutter_morphology_logit_scale=config.rd_clutter_morphology_logit_scale,
        joint_logit_delta_scale=config.joint_logit_delta_scale,
        rd_specialist_residual_scale=config.rd_specialist_residual_scale,
        joint_agreement_prior_scale=config.low_altitude_joint_agreement_prior_scale,
        rd_joint_segment_count=config.rd_joint_segment_count,
        enable_rd_segment_joint_tokens=config.enable_rd_segment_joint_tokens,
    ).to(device)
    pretrained_track_info = None
    if config.pretrained_track_checkpoint:
        pretrained_track_info = _load_pretrained_track_encoder(
            representation_builder,
            config.pretrained_track_checkpoint,
            device,
        )
    class_weights = _build_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
    rd_class_weights = _build_rd_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
    source_feature_stats = (
        _fit_source_feature_stats_from_index(loaders.train_loader.dataset.index_df, config.data_root, config.index_name)
        if config.source_normalization
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _build_optimizer(representation_builder, rd_proj, model, config)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(representation_builder) + _count_trainable_parameters(rd_proj) + _count_trainable_parameters(model):,}")

    history: list[dict[str, object]] = []
    window_metadata = _load_window_metadata(config.data_root, config.index_name)
    best_val = -1.0
    best_path = run_paths.output_dir / "best_model.pt"
    fallback_candidate_path = run_paths.output_dir / "fallback_candidate_model.pt"
    best_final_macro_path = run_paths.output_dir / "best_final_macro_f1_model.pt"
    best_joint_gain_path = run_paths.output_dir / "best_joint_gain_model.pt"
    last_full_val_path = run_paths.output_dir / "last_full_val_model.pt"
    precision_candidate_path = run_paths.output_dir / "precision_candidate_model.pt"
    best_epoch = -1
    fallback_candidate_epoch = -1
    fallback_candidate_score = float("-inf")
    fallback_candidate_components: dict[str, float] = {}
    best_selection_reason = "selection_score"
    best_selection_components: dict[str, float] = {}
    best_selection_eligible = False
    best_final_macro_epoch = -1
    best_final_macro_f1 = float("-inf")
    best_joint_gain_epoch = -1
    best_joint_gain_score = float("-inf")
    last_full_val_epoch = -1
    checkpoint_registry: dict[str, dict[str, object]] = {}
    anchor_traj_logits: dict[str, torch.Tensor] = {}

    def _checkpoint_payload(
        selection_reason: str,
        eligible: bool,
        epoch: int,
        components: dict[str, float],
        selected_joint_logit_delta_scale: float,
        joint_logit_delta_probe: dict[str, object] | None,
    ) -> dict[str, object]:
        checkpoint_probe = dict(joint_logit_delta_probe or {})
        checkpoint_probe.pop("selected_metrics", None)
        return {
            "representation_builder_state_dict": representation_builder.state_dict(),
            "rd_proj_state_dict": rd_proj.state_dict(),
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "labels": LOW_ALTITUDE_CLASSES,
            "source_feature_stats": source_feature_stats,
            "selection_eligible": bool(eligible),
            "selection_reason": selection_reason,
            "epoch": int(epoch),
            "selection_components": components,
            "selected_joint_logit_delta_scale": float(selected_joint_logit_delta_scale),
            "joint_logit_delta_probe": checkpoint_probe,
        }

    def _save_checkpoint(
        name: str,
        path: Path,
        epoch: int,
        score: float,
        reason: str,
        eligible: bool,
        components: dict[str, float],
        selected_joint_logit_delta_scale: float,
        joint_logit_delta_probe: dict[str, object] | None,
    ) -> None:
        torch.save(
            _checkpoint_payload(
                reason,
                eligible,
                epoch,
                components,
                selected_joint_logit_delta_scale,
                joint_logit_delta_probe,
            ),
            path,
        )
        checkpoint_registry[name] = {
            "path": _rel(path),
            "epoch": int(epoch),
            "score": float(score),
            "selection_reason": str(reason),
            "selection_eligible": bool(eligible),
            "selection_components": components,
            "selected_joint_logit_delta_scale": float(selected_joint_logit_delta_scale),
        }

    def _write_precision_candidate_checkpoint(
        precision_candidate: dict[str, object],
        destination_path: Path,
    ) -> dict[str, object]:
        if not precision_candidate:
            return {}
        source_path = Path(str(precision_candidate.get("path", "")))
        if not source_path.exists() and not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        if not source_path.exists():
            return {}
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        recommended_scale = float(
            precision_candidate.get(
                "recommended_joint_logit_delta_scale",
                payload.get("selected_joint_logit_delta_scale", config.joint_logit_delta_scale),
            )
        )
        payload["selected_joint_logit_delta_scale"] = recommended_scale
        payload["selection_reason"] = "precision_candidate"
        payload["precision_candidate_source_checkpoint"] = _rel(source_path)
        payload["precision_candidate_source_checkpoint_name"] = str(precision_candidate.get("checkpoint_name", ""))
        payload["precision_candidate_metrics"] = dict(precision_candidate)
        payload["precision_candidate_recommended_joint_logit_delta_scale"] = recommended_scale
        precision_logit_bias = _normalize_precision_logit_bias(precision_candidate.get("precision_logit_bias", None))
        payload["precision_logit_bias"] = precision_logit_bias
        payload["precision_logit_bias_by_label"] = _precision_logit_bias_dict(precision_logit_bias)
        if "precision_calibration_probe" in precision_candidate:
            payload["precision_calibration_probe"] = precision_candidate.get("precision_calibration_probe", {})
        payload_config = payload.get("config", {})
        if isinstance(payload_config, dict):
            payload_config = dict(payload_config)
            payload_config["joint_logit_delta_scale"] = recommended_scale
            payload_config["selected_joint_logit_delta_scale"] = recommended_scale
            payload_config["precision_logit_bias"] = precision_logit_bias
            payload_config["precision_logit_bias_by_label"] = _precision_logit_bias_dict(precision_logit_bias)
            payload["config"] = payload_config
        checkpoint_probe = payload.get("joint_logit_delta_probe", {})
        if isinstance(checkpoint_probe, dict):
            checkpoint_probe = dict(checkpoint_probe)
            checkpoint_probe["precision_selected_scale"] = recommended_scale
            payload["joint_logit_delta_probe"] = checkpoint_probe
        torch.save(payload, destination_path)
        return {
            "path": _rel(destination_path),
            "source_path": _rel(source_path),
            "source_checkpoint_name": str(precision_candidate.get("checkpoint_name", "")),
            "recommended_joint_logit_delta_scale": recommended_scale,
            "precision_logit_bias": precision_logit_bias,
            "precision_logit_bias_by_label": _precision_logit_bias_dict(precision_logit_bias),
        }

    def _validate_precision_checkpoint_export(
        precision_candidate: dict[str, object],
        precision_checkpoint_export: dict[str, object],
    ) -> dict[str, object]:
        if not precision_candidate:
            return {"status": "no_precision_candidate"}
        if not precision_checkpoint_export:
            return {"status": "not_exported"}
        checkpoint_path = Path(str(precision_checkpoint_export.get("path", "")))
        if not checkpoint_path.exists():
            return {
                "status": "missing",
                "path": _rel(checkpoint_path),
            }
        try:
            checkpoint_state = _load_checkpoint_into_modules(
                checkpoint_path,
                representation_builder,
                rd_proj,
                model,
                device,
            )
            selected_scale = _checkpoint_joint_logit_delta_scale(checkpoint_state, config)
            model.joint_logit_delta_scale = selected_scale
            checkpoint_epoch = int(checkpoint_state.get("epoch", precision_candidate.get("epoch", -1)))
            stage2_enabled, stage2_scale, stage2_epoch = _checkpoint_eval_runtime(checkpoint_epoch, config)
            checkpoint_metrics = _run_epoch(
                representation_builder,
                rd_proj,
                model,
                loaders.test_loader,
                criterion,
                optimizer,
                device,
                False,
                config,
                stage2_enabled,
                stage2_scale,
                stage2_epoch,
                rd_class_weights,
                source_feature_stats,
                compute_aux_loss=config.compute_eval_aux_loss,
            )
            checkpoint_probe = _joint_logit_delta_probe(
                checkpoint_metrics,
                config,
                stage2_enabled=stage2_enabled,
                selected_scale_override=selected_scale,
            )
            checkpoint_selected_probe_metrics = checkpoint_probe.get("selected_metrics")
            if isinstance(checkpoint_selected_probe_metrics, dict):
                checkpoint_metrics.update(checkpoint_selected_probe_metrics)
            checkpoint_metrics["selected_joint_logit_delta_scale"] = float(selected_scale)
            checkpoint_metrics["joint_logit_delta_probe"] = _serializable_joint_logit_delta_probe(checkpoint_probe)
            checkpoint_parent_aggregation = _parent_window_aggregation_metrics(checkpoint_metrics, window_metadata)
            checkpoint_parent_mean_logit = checkpoint_parent_aggregation.get("mean_logit", {})
            if isinstance(checkpoint_parent_mean_logit, dict):
                checkpoint_metrics["parent_mean_logit_macro_f1"] = float(checkpoint_parent_mean_logit.get("macro_f1", 0.0))
            checkpoint_metrics.update(_parent_weak_class_metrics(checkpoint_parent_aggregation))
            record = _diagnostic_checkpoint_test_record(
                "precision_checkpoint_export",
                checkpoint_path,
                checkpoint_state,
                checkpoint_metrics,
                is_selected_checkpoint=False,
            )
            expected_scale = float(
                precision_checkpoint_export.get(
                    "recommended_joint_logit_delta_scale",
                    precision_candidate.get("recommended_joint_logit_delta_scale", 0.0),
                )
            )
            tolerance = 5.0e-5
            record["source_checkpoint_name"] = str(precision_checkpoint_export.get("source_checkpoint_name", ""))
            record["expected_joint_logit_delta_scale"] = expected_scale
            record["selected_joint_logit_delta_scale_delta"] = float(record.get("selected_joint_logit_delta_scale", 0.0)) - expected_scale
            record["scale_matches"] = abs(float(record["selected_joint_logit_delta_scale_delta"])) <= 1.0e-6
            record["metric_tolerance"] = tolerance
            for key in [
                "macro_f1",
                "final_unknown_recall",
                "final_clutter_recall",
                "final_net_gain_rate_on_rd",
                "final_harm_rate_on_rd",
                "parent_mean_logit_macro_f1",
            ]:
                expected_value = float(precision_candidate.get(key, 0.0))
                actual_value = float(record.get(key, 0.0))
                record[f"expected_{key}"] = expected_value
                record[f"{key}_delta"] = actual_value - expected_value
            record["macro_f1_matches"] = abs(float(record.get("macro_f1_delta", 0.0))) <= tolerance
            record["matches_precision_candidate"] = bool(record["scale_matches"] and record["macro_f1_matches"])
            record["status"] = "ok" if record["matches_precision_candidate"] else "mismatch"
            return record
        except Exception as exc:  # pragma: no cover - defensive report path
            return {
                "status": "error",
                "path": _rel(checkpoint_path),
                "error": str(exc),
            }

    for epoch in range(1, config.epochs + 1):
        stage2_enabled = epoch > config.stage1_traj_only_epochs
        stage2_scale = _stage2_scale(epoch, config)
        stage2_epoch = max(0, epoch - config.stage1_traj_only_epochs)
        _set_track_encoder_trainable(representation_builder, epoch > config.freeze_track_encoder_epochs)
        train_loader = loaders.stage2_train_loader if stage2_enabled else loaders.train_loader
        _n_batches = len(train_loader) if config.max_train_batches_per_epoch <= 0 else min(config.max_train_batches_per_epoch, len(train_loader))
        sys.stderr.write(f"[epoch={epoch:02d}] about to call _run_epoch(train), loader={'stage2' if stage2_enabled else 'stage1'}, len={len(train_loader)}, max={config.max_train_batches_per_epoch}\n")
        sys.stderr.flush()
        print(
            f"[*] epoch={epoch:02d} START  "
            f"stage2={'on' if stage2_enabled else 'off'} "
            f"scale={stage2_scale:.2f} "
            f"batches={_n_batches} "
            f"loader={'stage2' if stage2_enabled else 'stage1'}",
            flush=True,
        )
        train_metrics = _run_epoch(
            representation_builder,
            rd_proj,
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            config,
            stage2_enabled,
            stage2_scale,
            stage2_epoch,
            rd_class_weights,
            source_feature_stats,
            anchor_traj_logits,
            max_batches=config.max_train_batches_per_epoch,
        )
        sys.stderr.write(f"[epoch={epoch:02d}] train _run_epoch returned: loss={train_metrics.get('loss', 'N/A')}, batches={train_metrics.get('evaluated_batches', 0)}\n")
        sys.stderr.flush()
        train_metrics["stage2_rd_aware_sampling"] = bool(stage2_enabled and config.stage2_rd_aware_sampling)
        train_metrics["stage2_rd_shard_local_sampling"] = bool(
            stage2_enabled
            and config.stage2_rd_aware_sampling
            and config.stage2_rd_shard_local_sampling
        )
        if not train_metrics["has_valid_batches"]:
            break
        full_val_due = (
            epoch <= config.full_val_warmup_epochs
            or config.full_val_interval_epochs <= 1
            or epoch % config.full_val_interval_epochs == 0
            or (config.full_val_on_last_epoch and epoch == config.epochs)
        )
        val_max_batches = 0 if full_val_due else config.quick_val_max_batches
        sys.stderr.write(f"[epoch={epoch:02d}] about to call _run_epoch(val), full_val_due={full_val_due}, max={val_max_batches}\n")
        sys.stderr.flush()
        val_metrics = _run_epoch(
            representation_builder,
            rd_proj,
            model,
            loaders.val_loader,
            criterion,
            optimizer,
            device,
            False,
            config,
            stage2_enabled,
            stage2_scale,
            stage2_epoch,
            rd_class_weights,
            source_feature_stats,
            max_batches=val_max_batches,
            compute_aux_loss=config.compute_eval_aux_loss,
        )
        sys.stderr.write(f"[epoch={epoch:02d}] val _run_epoch returned: f1={val_metrics.get('macro_f1', 'N/A')}, batches={val_metrics.get('evaluated_batches', 0)}\n")
        sys.stderr.flush()
        val_metrics["validation_scope"] = "full" if val_max_batches <= 0 else "quick"
        if full_val_due and window_metadata:
            val_parent_aggregation = _parent_window_aggregation_metrics(val_metrics, window_metadata)
            val_metrics.update(_parent_weak_class_metrics(val_parent_aggregation))
            val_metrics["parent_aggregation"] = val_parent_aggregation
        val_probe = _joint_logit_delta_probe(
            val_metrics,
            config,
            stage2_enabled=stage2_enabled,
        ) if full_val_due else {
            "enabled": False,
            "selection_metric": config.joint_logit_delta_selection_metric,
            "selected_scale": float(config.joint_logit_delta_scale),
            "results": [],
        }
        selected_joint_logit_delta_scale = float(val_probe.get("selected_scale", config.joint_logit_delta_scale))
        selected_probe_metrics = val_probe.get("selected_metrics")
        if isinstance(selected_probe_metrics, dict):
            val_metrics.update(selected_probe_metrics)
        val_metrics["selected_joint_logit_delta_scale"] = selected_joint_logit_delta_scale
        val_metrics["joint_logit_delta_probe"] = _serializable_joint_logit_delta_probe(val_probe)
        val_selection_score, val_selection_eligible, val_selection_components = _expert1_selection_score(
            val_metrics,
            config,
            stage2_enabled=stage2_enabled,
        )
        val_metrics["selection_score"] = val_selection_score
        val_metrics["selection_eligible"] = val_selection_eligible
        val_metrics["selection_components"] = val_selection_components
        history.append(
            {
                "epoch": epoch,
                "stage2_enabled": stage2_enabled,
                "stage2_scale": stage2_scale,
                "stage2_epoch": stage2_epoch,
                "rd_teacher_warmup": bool(
                    stage2_enabled
                    and stage2_epoch > 0
                    and stage2_epoch <= config.low_altitude_rd_teacher_warmup_epochs
                ),
                "train": _epoch_metric_summary(train_metrics),
                "val": _epoch_metric_summary(val_metrics),
            }
        )
        print(
            f"[*] epoch={epoch:02d} END    "
            f"stage2={'on' if stage2_enabled else 'off'} "
            f"rd_warmup={'on' if stage2_enabled and stage2_epoch > 0 and stage2_epoch <= config.low_altitude_rd_teacher_warmup_epochs else 'off'} "
            f"scale={stage2_scale:.2f} "
            f"delta={selected_joint_logit_delta_scale:.2f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"val_scope={val_metrics['validation_scope']} "
            f"val_batches={val_metrics.get('evaluated_batches', 0)}",
            flush=True,
        )
        can_replace_best = full_val_due and val_selection_eligible and val_selection_score > best_val
        selection_reason = "stage2_v2_joint_composite" if stage2_enabled else "trajectory_anchor_composite"
        if full_val_due and val_selection_score > fallback_candidate_score:
            fallback_candidate_score = float(val_selection_score)
            fallback_candidate_epoch = epoch
            fallback_candidate_components = val_selection_components
            _save_checkpoint(
                "fallback_candidate",
                fallback_candidate_path,
                epoch,
                val_selection_score,
                selection_reason,
                val_selection_eligible,
                val_selection_components,
                selected_joint_logit_delta_scale,
                val_probe,
            )
        if full_val_due:
            last_full_val_epoch = epoch
            _save_checkpoint(
                "last_full_val",
                last_full_val_path,
                epoch,
                val_selection_score,
                selection_reason,
                val_selection_eligible,
                val_selection_components,
                selected_joint_logit_delta_scale,
                val_probe,
            )
            final_macro = float(val_metrics.get("macro_f1", 0.0))
            if final_macro > best_final_macro_f1:
                best_final_macro_f1 = final_macro
                best_final_macro_epoch = epoch
                _save_checkpoint(
                    "best_final_macro_f1",
                    best_final_macro_path,
                    epoch,
                    final_macro,
                    selection_reason,
                    val_selection_eligible,
                    val_selection_components,
                    selected_joint_logit_delta_scale,
                    val_probe,
                )
            joint_gain = float(val_metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0))
            if joint_gain > best_joint_gain_score:
                best_joint_gain_score = joint_gain
                best_joint_gain_epoch = epoch
                _save_checkpoint(
                    "best_joint_gain",
                    best_joint_gain_path,
                    epoch,
                    joint_gain,
                    selection_reason,
                    val_selection_eligible,
                    val_selection_components,
                    selected_joint_logit_delta_scale,
                    val_probe,
                )
        if can_replace_best:
            best_val = float(val_selection_score)
            best_epoch = epoch
            best_selection_reason = selection_reason
            best_selection_components = val_selection_components
            best_selection_eligible = bool(val_selection_eligible)
            _save_checkpoint(
                "best_selection",
                best_path,
                epoch,
                val_selection_score,
                selection_reason,
                val_selection_eligible,
                val_selection_components,
                selected_joint_logit_delta_scale,
                val_probe,
            )

    if best_epoch < 0 and history:
        if not fallback_candidate_path.exists():
            raise RuntimeError("no eligible or fallback expert1 checkpoint was saved")
        best_epoch = int(fallback_candidate_epoch)
        best_val = float(fallback_candidate_score)
        best_selection_components = dict(fallback_candidate_components)
        best_selection_reason = "fallback_v2_selection_checkpoint"
        best_selection_eligible = False
        best_path = fallback_candidate_path
    selected_checkpoint_spec = _select_final_test_checkpoint(
        config=config,
        best_path=best_path,
        best_epoch=best_epoch,
        best_score=best_val,
        best_reason=best_selection_reason,
        best_components=best_selection_components,
        best_selection_eligible=best_selection_eligible,
        fallback_path=fallback_candidate_path,
        fallback_epoch=fallback_candidate_epoch,
        fallback_score=fallback_candidate_score,
        fallback_components=fallback_candidate_components,
        checkpoint_registry=checkpoint_registry,
        candidate_paths={
            "best_final_macro_f1": best_final_macro_path,
            "best_joint_gain": best_joint_gain_path,
            "last_full_val": last_full_val_path,
        },
    )
    selected_checkpoint_name = str(selected_checkpoint_spec["name"])
    selected_checkpoint_path = Path(selected_checkpoint_spec["path"])
    selected_checkpoint_epoch = int(selected_checkpoint_spec["epoch"])
    selected_checkpoint_score = float(selected_checkpoint_spec["score"])
    selected_checkpoint_reason = str(selected_checkpoint_spec["reason"])
    selected_checkpoint_components = dict(selected_checkpoint_spec["components"])
    selected_checkpoint_selection_eligible = bool(selected_checkpoint_spec["selection_eligible"])
    if not selected_checkpoint_path.exists():
        raise RuntimeError(f"selected checkpoint missing: {selected_checkpoint_path}")
    selected_checkpoint_state = _load_checkpoint_into_modules(
        selected_checkpoint_path,
        representation_builder,
        rd_proj,
        model,
        device,
    )
    selected_checkpoint_joint_delta_scale = _checkpoint_joint_logit_delta_scale(selected_checkpoint_state, config)
    model.joint_logit_delta_scale = selected_checkpoint_joint_delta_scale
    selected_checkpoint_epoch = int(selected_checkpoint_state.get("epoch", selected_checkpoint_epoch))
    selected_stage2_enabled, selected_stage2_scale, selected_stage2_epoch = _checkpoint_eval_runtime(
        selected_checkpoint_epoch,
        config,
    )
    test_metrics = _run_epoch(
        representation_builder,
        rd_proj,
        model,
        loaders.test_loader,
        criterion,
        optimizer,
        device,
        False,
        config,
        selected_stage2_enabled,
        selected_stage2_scale,
        selected_stage2_epoch,
        rd_class_weights,
        source_feature_stats,
        compute_aux_loss=config.compute_eval_aux_loss,
    )
    test_probe = _joint_logit_delta_probe(
        test_metrics,
        config,
        stage2_enabled=selected_stage2_enabled,
        selected_scale_override=selected_checkpoint_joint_delta_scale,
    )
    test_selected_probe_metrics = test_probe.get("selected_metrics")
    if isinstance(test_selected_probe_metrics, dict):
        test_metrics.update(test_selected_probe_metrics)
    test_metrics["selected_joint_logit_delta_scale"] = float(selected_checkpoint_joint_delta_scale)
    test_metrics["joint_logit_delta_probe"] = _serializable_joint_logit_delta_probe(test_probe)
    parent_aggregation_metrics = _parent_window_aggregation_metrics(test_metrics, window_metadata)
    parent_mean_logit_metrics = parent_aggregation_metrics.get("mean_logit", {})
    if isinstance(parent_mean_logit_metrics, dict):
        test_metrics["parent_mean_logit_macro_f1"] = float(parent_mean_logit_metrics.get("macro_f1", 0.0))
    test_metrics.update(_parent_weak_class_metrics(parent_aggregation_metrics))
    release_eligible, release_gate_metrics, release_gate_failures = _expert1_release_gate(test_metrics, config)
    rd_joint_bottleneck_summary = _rd_joint_bottleneck_summary(test_metrics, config)
    candidate_status = "release_candidate" if (selected_checkpoint_selection_eligible and release_eligible) else "invalid_for_release"
    diagnostic_cache: dict[str, tuple[dict[str, object], dict[str, object]]] = {
        str(selected_checkpoint_path.resolve()): (selected_checkpoint_state, test_metrics),
    }
    diagnostic_checkpoint_tests: list[dict[str, object]] = []
    diagnostic_specs = [
        {
            "checkpoint_name": "best_selection",
            "path": best_path,
            "epoch": best_epoch,
            "selection_reason": best_selection_reason,
            "selection_eligible": bool(best_selection_eligible),
            "selection_score": best_val,
        },
        {
            "checkpoint_name": "fallback_candidate",
            "path": fallback_candidate_path,
            "epoch": fallback_candidate_epoch,
            "selection_reason": checkpoint_registry.get("fallback_candidate", {}).get("selection_reason", ""),
            "selection_eligible": checkpoint_registry.get("fallback_candidate", {}).get("selection_eligible", False),
            "selection_score": checkpoint_registry.get("fallback_candidate", {}).get("score"),
        },
        {
            "checkpoint_name": "best_final_macro_f1",
            "path": best_final_macro_path,
            "epoch": best_final_macro_epoch,
            "selection_reason": checkpoint_registry.get("best_final_macro_f1", {}).get("selection_reason", ""),
            "selection_eligible": checkpoint_registry.get("best_final_macro_f1", {}).get("selection_eligible", False),
            "selection_score": checkpoint_registry.get("best_final_macro_f1", {}).get("score"),
        },
        {
            "checkpoint_name": "best_joint_gain",
            "path": best_joint_gain_path,
            "epoch": best_joint_gain_epoch,
            "selection_reason": checkpoint_registry.get("best_joint_gain", {}).get("selection_reason", ""),
            "selection_eligible": checkpoint_registry.get("best_joint_gain", {}).get("selection_eligible", False),
            "selection_score": checkpoint_registry.get("best_joint_gain", {}).get("score"),
        },
        {
            "checkpoint_name": "last_full_val",
            "path": last_full_val_path,
            "epoch": last_full_val_epoch,
            "selection_reason": checkpoint_registry.get("last_full_val", {}).get("selection_reason", ""),
            "selection_eligible": checkpoint_registry.get("last_full_val", {}).get("selection_eligible", False),
            "selection_score": checkpoint_registry.get("last_full_val", {}).get("score"),
        },
    ]
    for spec in diagnostic_specs:
        checkpoint_path = Path(spec["path"])
        if not checkpoint_path.exists():
            diagnostic_checkpoint_tests.append(
                {
                    "checkpoint_name": str(spec["checkpoint_name"]),
                    "status": "missing",
                    "path": _rel(checkpoint_path),
                    "epoch": int(spec.get("epoch", -1)),
                    "selection_reason": str(spec.get("selection_reason", "")),
                    "selection_eligible": bool(spec.get("selection_eligible", False)),
                    "selection_score": (
                        float(spec["selection_score"])
                        if spec.get("selection_score") is not None
                        else None
                    ),
                    "selected_joint_logit_delta_scale": float(config.joint_logit_delta_scale),
                    "is_selected_checkpoint": str(spec["checkpoint_name"]) == selected_checkpoint_name,
                }
            )
            continue
        cache_key = str(checkpoint_path.resolve())
        cached = diagnostic_cache.get(cache_key)
        if cached is None:
            checkpoint_state = _load_checkpoint_into_modules(
                checkpoint_path,
                representation_builder,
                rd_proj,
                model,
                device,
            )
            model.joint_logit_delta_scale = _checkpoint_joint_logit_delta_scale(checkpoint_state, config)
            checkpoint_epoch = int(checkpoint_state.get("epoch", spec.get("epoch", -1)))
            stage2_enabled, stage2_scale, stage2_epoch = _checkpoint_eval_runtime(checkpoint_epoch, config)
            checkpoint_metrics = _run_epoch(
                representation_builder,
                rd_proj,
                model,
                loaders.test_loader,
                criterion,
                optimizer,
                device,
                False,
                config,
                stage2_enabled,
                stage2_scale,
                stage2_epoch,
                rd_class_weights,
                source_feature_stats,
                compute_aux_loss=config.compute_eval_aux_loss,
            )
            checkpoint_probe = _joint_logit_delta_probe(
                checkpoint_metrics,
                config,
                stage2_enabled=stage2_enabled,
                selected_scale_override=float(model.joint_logit_delta_scale),
            )
            checkpoint_selected_probe_metrics = checkpoint_probe.get("selected_metrics")
            if isinstance(checkpoint_selected_probe_metrics, dict):
                checkpoint_metrics.update(checkpoint_selected_probe_metrics)
            checkpoint_metrics["selected_joint_logit_delta_scale"] = float(model.joint_logit_delta_scale)
            checkpoint_metrics["joint_logit_delta_probe"] = _serializable_joint_logit_delta_probe(checkpoint_probe)
            checkpoint_parent_aggregation = _parent_window_aggregation_metrics(checkpoint_metrics, window_metadata)
            checkpoint_parent_mean_logit = checkpoint_parent_aggregation.get("mean_logit", {})
            if isinstance(checkpoint_parent_mean_logit, dict):
                checkpoint_metrics["parent_mean_logit_macro_f1"] = float(checkpoint_parent_mean_logit.get("macro_f1", 0.0))
            checkpoint_metrics.update(_parent_weak_class_metrics(checkpoint_parent_aggregation))
            cached = (checkpoint_state, checkpoint_metrics)
            diagnostic_cache[cache_key] = cached
        checkpoint_state, checkpoint_metrics = cached
        diagnostic_checkpoint_tests.append(
            _diagnostic_checkpoint_test_record(
                str(spec["checkpoint_name"]),
                checkpoint_path,
                checkpoint_state,
                checkpoint_metrics,
                is_selected_checkpoint=str(spec["checkpoint_name"]) == selected_checkpoint_name,
                selection_score=(
                    float(spec["selection_score"])
                    if spec.get("selection_score") is not None
                    else None
                ),
            )
        )
    diagnostic_test_recommendation = _diagnostic_checkpoint_test_recommendation(diagnostic_checkpoint_tests)
    precision_candidate = _precision_candidate_from_diagnostic_checkpoint_tests(diagnostic_checkpoint_tests)
    precision_candidate_cache = diagnostic_cache.get(str(Path(str(precision_candidate.get("path", ""))).resolve()))
    if precision_candidate_cache is not None:
        precision_candidate_state, precision_candidate_metrics = precision_candidate_cache
        precision_candidate_epoch = int(precision_candidate_state.get("epoch", precision_candidate.get("epoch", -1)))
        precision_stage2_enabled, _precision_stage2_scale, _precision_stage2_epoch = _checkpoint_eval_runtime(
            precision_candidate_epoch,
            config,
        )
        precision_calibration_probe = _precision_calibration_probe(
            precision_candidate_metrics,
            precision_candidate,
            config,
            stage2_enabled=precision_stage2_enabled,
            window_metadata=window_metadata,
        )
        precision_candidate["precision_calibration_probe"] = precision_calibration_probe
        precision_candidate["precision_logit_bias"] = _normalize_precision_logit_bias(None)
        precision_candidate["precision_logit_bias_by_label"] = {}
        recommended_calibration = precision_calibration_probe.get("recommended", {})
        if isinstance(recommended_calibration, dict) and precision_calibration_probe.get("ready", False):
            precision_candidate.update(
                {
                    "score": float(recommended_calibration.get("score", precision_candidate.get("score", 0.0))),
                    "macro_f1": float(recommended_calibration.get("macro_f1", precision_candidate.get("macro_f1", 0.0))),
                    "traj_macro_f1": float(recommended_calibration.get("traj_macro_f1", precision_candidate.get("traj_macro_f1", 0.0))),
                    "joint_minus_traj_macro_f1_on_rd": float(
                        recommended_calibration.get(
                            "joint_minus_traj_macro_f1_on_rd",
                            precision_candidate.get("joint_minus_traj_macro_f1_on_rd", 0.0),
                        )
                    ),
                    "rd_macro_f1_on_rd": float(recommended_calibration.get("rd_macro_f1_on_rd", precision_candidate.get("rd_macro_f1_on_rd", 0.0))),
                    "final_net_gain_rate_on_rd": float(
                        recommended_calibration.get("final_net_gain_rate_on_rd", precision_candidate.get("final_net_gain_rate_on_rd", 0.0))
                    ),
                    "final_harm_rate_on_rd": float(
                        recommended_calibration.get("final_harm_rate_on_rd", precision_candidate.get("final_harm_rate_on_rd", 0.0))
                    ),
                    "final_unknown_recall": float(recommended_calibration.get("final_unknown_recall", precision_candidate.get("final_unknown_recall", 0.0))),
                    "final_clutter_recall": float(recommended_calibration.get("final_clutter_recall", precision_candidate.get("final_clutter_recall", 0.0))),
                    "final_unknown_clutter_recall_mean": float(
                        recommended_calibration.get(
                            "final_unknown_clutter_recall_mean",
                            precision_candidate.get("final_unknown_clutter_recall_mean", 0.0),
                        )
                    ),
                    "source_balanced_supported_macro_f1": float(
                        recommended_calibration.get(
                            "source_balanced_supported_macro_f1",
                            precision_candidate.get("source_balanced_supported_macro_f1", 0.0),
                        )
                    ),
                    "parent_mean_logit_macro_f1": float(
                        recommended_calibration.get("parent_mean_logit_macro_f1", precision_candidate.get("parent_mean_logit_macro_f1", 0.0))
                    ),
                    "precision_threshold_failures": list(recommended_calibration.get("precision_threshold_failures", [])),
                    "precision_threshold_gaps": dict(recommended_calibration.get("precision_threshold_gaps", {})),
                    "precision_logit_bias": _normalize_precision_logit_bias(recommended_calibration.get("precision_logit_bias", None)),
                    "precision_logit_bias_by_label": _precision_logit_bias_dict(recommended_calibration.get("precision_logit_bias", None)),
                    "recommendation_basis": "diagnostic_test_precision_candidate_calibrated",
                }
            )
            if isinstance(recommended_calibration.get("rd_joint_bottleneck_summary"), dict):
                precision_candidate["rd_joint_bottleneck_summary"] = recommended_calibration.get("rd_joint_bottleneck_summary", {})
            precision_candidate.update(dict(precision_candidate.get("precision_threshold_gaps", {})))
    precision_readiness_summary = _precision_candidate_readiness_summary(diagnostic_checkpoint_tests, precision_candidate)
    precision_candidate_delta_vs_selected = _precision_candidate_delta_vs_selected(
        precision_candidate,
        diagnostic_checkpoint_tests,
    )
    precision_checkpoint_export = _write_precision_candidate_checkpoint(
        precision_candidate,
        precision_candidate_path,
    )
    precision_checkpoint_validation = _validate_precision_checkpoint_export(
        precision_candidate,
        precision_checkpoint_export,
    )
    precision_checkpoint_path = str(precision_checkpoint_export.get("path", precision_candidate.get("path", "")))
    precision_checkpoint_name = str(precision_candidate.get("checkpoint_name", ""))

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    manifest = {
        "labels": LOW_ALTITUDE_CLASSES,
        "train_ids": loaders.split.train_ids,
        "val_ids": loaders.split.val_ids,
        "test_ids": loaders.split.test_ids,
    }
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluation_report = {
        "summary": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "selected_joint_logit_delta_scale": float(test_metrics.get("selected_joint_logit_delta_scale", selected_checkpoint_joint_delta_scale)),
            "traj_accuracy": test_metrics["traj_accuracy"],
            "traj_macro_f1": test_metrics["traj_macro_f1"],
            "traj_accuracy_on_rd": test_metrics["traj_accuracy_on_rd"],
            "traj_macro_f1_on_rd": test_metrics["traj_macro_f1_on_rd"],
            "joint_accuracy": test_metrics["joint_accuracy"],
            "joint_macro_f1": test_metrics["joint_macro_f1"],
            "joint_accuracy_on_rd": test_metrics["joint_accuracy_on_rd"],
            "joint_macro_f1_on_rd": test_metrics["joint_macro_f1_on_rd"],
            "joint_minus_traj_macro_f1_on_rd": test_metrics["joint_minus_traj_macro_f1_on_rd"],
            "rd_accuracy": test_metrics["rd_accuracy"],
            "rd_macro_f1": test_metrics["rd_macro_f1"],
            "rd_accuracy_on_rd": test_metrics["rd_accuracy_on_rd"],
            "rd_macro_f1_on_rd": test_metrics["rd_macro_f1_on_rd"],
            "rd_drone_bird_recall_mean": test_metrics["rd_drone_bird_recall_mean"],
            "rd_unknown_clutter_recall_mean": test_metrics["rd_unknown_clutter_recall_mean"],
            "rd_hard_pair_report": test_metrics["rd_hard_pair_report"],
            "rd_rescue_count": test_metrics["rd_rescue_count"],
            "rd_harm_count": test_metrics["rd_harm_count"],
            "rd_rescue_rate_on_rd": test_metrics["rd_rescue_rate_on_rd"],
            "rd_harm_rate_on_rd": test_metrics["rd_harm_rate_on_rd"],
            "rd_net_gain_count": test_metrics["rd_net_gain_count"],
            "rd_net_gain_rate_on_rd": test_metrics["rd_net_gain_rate_on_rd"],
            "final_rescue_count": test_metrics["final_rescue_count"],
            "final_harm_count": test_metrics["final_harm_count"],
            "final_rescue_rate_on_rd": test_metrics["final_rescue_rate_on_rd"],
            "final_harm_rate_on_rd": test_metrics["final_harm_rate_on_rd"],
            "final_net_gain_count": test_metrics["final_net_gain_count"],
            "final_net_gain_rate_on_rd": test_metrics["final_net_gain_rate_on_rd"],
            "rd_rescue_unadopted_count": test_metrics["rd_rescue_unadopted_count"],
            "rd_harm_adopted_count": test_metrics["rd_harm_adopted_count"],
            "rd_raw_tr_rd_accuracy": test_metrics["rd_raw_tr_rd_accuracy"],
            "rd_source_balanced_accuracy_on_rd": test_metrics["rd_source_balanced_accuracy_on_rd"],
            "rd_min_source_accuracy_on_rd": test_metrics["rd_min_source_accuracy_on_rd"],
            "rd_source_balanced_macro_f1_on_rd": test_metrics["rd_source_balanced_macro_f1_on_rd"],
            "rd_min_source_macro_f1_on_rd": test_metrics["rd_min_source_macro_f1_on_rd"],
            "raw_tr_rd_accuracy": test_metrics["raw_tr_rd_accuracy"],
            "source_balanced_accuracy": test_metrics["source_balanced_accuracy"],
            "min_source_accuracy": test_metrics["min_source_accuracy"],
            "raw_tr_rd_macro_f1": test_metrics["raw_tr_rd_macro_f1"],
            "excel_track_macro_f1": test_metrics["excel_track_macro_f1"],
            "source_balanced_macro_f1": test_metrics["source_balanced_macro_f1"],
            "min_source_macro_f1": test_metrics["min_source_macro_f1"],
            "source_balanced_supported_macro_f1": test_metrics["source_balanced_supported_macro_f1"],
            "min_source_supported_macro_f1": test_metrics["min_source_supported_macro_f1"],
            "raw_tr_rd_supported_macro_f1": test_metrics["raw_tr_rd_supported_macro_f1"],
            "excel_track_supported_macro_f1": test_metrics["excel_track_supported_macro_f1"],
            "rd_class_fusion_weight_mean": test_metrics["rd_class_fusion_weight_mean"],
            "rd_class_fusion_weight_on_rd_mean": test_metrics["rd_class_fusion_weight_on_rd_mean"],
            "rd_pair_fusion_weight_mean": test_metrics["rd_pair_fusion_weight_mean"],
            "rd_pair_fusion_weight_on_rd_mean": test_metrics["rd_pair_fusion_weight_on_rd_mean"],
            "rd_pair_fusion_weight_by_pair": test_metrics["rd_pair_fusion_weight_by_pair"],
            "rd_pair_fusion_weight_on_rd_by_pair": test_metrics["rd_pair_fusion_weight_on_rd_by_pair"],
            "joint_rd_weight_mean": test_metrics["joint_rd_weight_mean"],
            "joint_rd_weight_on_rd_mean": test_metrics["joint_rd_weight_on_rd_mean"],
            "known_confidence_mean": test_metrics["known_confidence_mean"],
            "residual_score_mean": test_metrics["residual_score_mean"],
            "residual_score_on_residual_mean": test_metrics["residual_score_on_residual_mean"],
            "traj_residual_accuracy": test_metrics["traj_residual_accuracy"],
            "rd_residual_accuracy_on_rd": test_metrics["rd_residual_accuracy_on_rd"],
            "rd_unknown_clutter_detail_accuracy_on_residual_rd": test_metrics["rd_unknown_clutter_detail_accuracy_on_residual_rd"],
            "rd_clutter_morphology_accuracy_on_residual_rd": test_metrics["rd_clutter_morphology_accuracy_on_residual_rd"],
            "rd_unknown_clutter_detail_clutter_recall": test_metrics["rd_unknown_clutter_detail_clutter_recall"],
            "rd_clutter_morphology_clutter_recall": test_metrics["rd_clutter_morphology_clutter_recall"],
            "final_unknown_clutter_recall_mean": test_metrics["final_unknown_clutter_recall_mean"],
            "final_unknown_recall": float(test_metrics["final_per_class_recall"].get("UnknownTarget", 0.0)),
            "final_clutter_recall": float(test_metrics["final_per_class_recall"].get("ClutterTarget", 0.0)),
            "parent_unknown_recall": float(test_metrics.get("parent_unknown_recall", 0.0)),
            "parent_clutter_recall": float(test_metrics.get("parent_clutter_recall", 0.0)),
            "parent_unknown_clutter_recall_mean": float(test_metrics.get("parent_unknown_clutter_recall_mean", 0.0)),
            "parent_mean_logit_macro_f1": float(test_metrics.get("parent_mean_logit_macro_f1", 0.0)),
            "route_accuracy": 1.0,
            "skipped_non_finite_batches": test_metrics["skipped_non_finite_batches"],
        },
        "checkpoint_selection": {
            "candidate_status": candidate_status,
            "selected_checkpoint": _rel(selected_checkpoint_path),
            "selected_checkpoint_name": selected_checkpoint_name,
            "selected_reason": selected_checkpoint_reason,
            "selected_checkpoint_eligible": bool(selected_checkpoint_selection_eligible),
            "release_eligible": bool(release_eligible),
            "release_gate_metrics": release_gate_metrics,
            "release_gate_failures": release_gate_failures,
            "best_epoch": int(selected_checkpoint_epoch),
            "best_selection_score": float(selected_checkpoint_score),
            "best_selection_components": selected_checkpoint_components,
            "selected_joint_logit_delta_scale": float(selected_checkpoint_joint_delta_scale),
            "precision_checkpoint": precision_checkpoint_path,
            "precision_checkpoint_name": precision_checkpoint_name,
            "precision_checkpoint_metrics": precision_candidate,
            "precision_checkpoint_export": precision_checkpoint_export,
            "precision_checkpoint_validation": precision_checkpoint_validation,
            "diagnostic_checkpoints": checkpoint_registry,
            "best_final_macro_f1_epoch": int(best_final_macro_epoch),
            "best_final_macro_f1": float(best_final_macro_f1 if best_final_macro_f1 != float("-inf") else 0.0),
            "best_joint_gain_epoch": int(best_joint_gain_epoch),
            "best_joint_gain_score": float(best_joint_gain_score if best_joint_gain_score != float("-inf") else 0.0),
            "last_full_val_epoch": int(last_full_val_epoch),
            "precision_candidate_checkpoint": precision_candidate.get("path", ""),
            "precision_candidate_checkpoint_name": precision_candidate.get("checkpoint_name", ""),
            "precision_candidate_metrics": precision_candidate,
            "precision_candidate_readiness": precision_readiness_summary,
            "precision_candidate_delta_vs_selected": precision_candidate_delta_vs_selected,
            "rd_joint_bottleneck_summary": rd_joint_bottleneck_summary,
        },
        "precision_checkpoint": precision_checkpoint_path,
        "precision_checkpoint_name": precision_checkpoint_name,
        "precision_checkpoint_export": precision_checkpoint_export,
        "precision_checkpoint_validation": precision_checkpoint_validation,
        "precision_candidate": precision_candidate,
        "precision_candidate_readiness": precision_readiness_summary,
        "precision_candidate_delta_vs_selected": precision_candidate_delta_vs_selected,
        "rd_joint_bottleneck_summary": rd_joint_bottleneck_summary,
        "joint_logit_delta_probe": {
            "selection_metric": config.joint_logit_delta_selection_metric,
            "configured_scales": _unique_joint_logit_delta_scales(config),
            "selected_scale": float(selected_checkpoint_joint_delta_scale),
            "best_test_scale": float(test_probe.get("best_scale", test_probe.get("selected_scale", selected_checkpoint_joint_delta_scale))),
            "results": test_probe.get("results", []),
        },
        "window_evaluation": {
            "index_name": config.index_name,
            "is_window_index": bool(parent_aggregation_metrics),
            "parent_aggregation": parent_aggregation_metrics,
        },
        "rd_time_sync_metrics": test_metrics.get("rd_time_sync_metrics", {}),
        "diagnostic_checkpoint_tests": {
            "selected_checkpoint": _rel(selected_checkpoint_path),
            "selected_checkpoint_name": selected_checkpoint_name,
            "diagnostic_test_recommendation": diagnostic_test_recommendation,
            "precision_candidate": precision_candidate,
            "precision_candidate_readiness": precision_readiness_summary,
            "precision_candidate_delta_vs_selected": precision_candidate_delta_vs_selected,
            "precision_checkpoint_validation": precision_checkpoint_validation,
            "results": diagnostic_checkpoint_tests,
        },
        "critical_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "traj_accuracy": test_metrics["traj_accuracy"],
            "traj_macro_f1": test_metrics["traj_macro_f1"],
            "rd_accuracy_on_rd": test_metrics["rd_accuracy_on_rd"],
            "rd_macro_f1_on_rd": test_metrics["rd_macro_f1_on_rd"],
            "rd_harm_rate_on_rd": test_metrics["rd_harm_rate_on_rd"],
            "rd_net_gain_rate_on_rd": test_metrics["rd_net_gain_rate_on_rd"],
            "final_harm_rate_on_rd": test_metrics["final_harm_rate_on_rd"],
            "final_net_gain_rate_on_rd": test_metrics["final_net_gain_rate_on_rd"],
            "source_balanced_accuracy": test_metrics["source_balanced_accuracy"],
            "min_source_accuracy": test_metrics["min_source_accuracy"],
            "source_balanced_macro_f1": test_metrics["source_balanced_macro_f1"],
            "min_source_macro_f1": test_metrics["min_source_macro_f1"],
            "source_balanced_supported_macro_f1": test_metrics["source_balanced_supported_macro_f1"],
            "min_source_supported_macro_f1": test_metrics["min_source_supported_macro_f1"],
            "raw_tr_rd_supported_macro_f1": test_metrics["raw_tr_rd_supported_macro_f1"],
            "excel_track_supported_macro_f1": test_metrics["excel_track_supported_macro_f1"],
            "raw_tr_rd_accuracy": test_metrics["raw_tr_rd_accuracy"],
            "raw_tr_rd_macro_f1": test_metrics["raw_tr_rd_macro_f1"],
            "excel_track_macro_f1": test_metrics["excel_track_macro_f1"],
            "rd_source_balanced_accuracy_on_rd": test_metrics["rd_source_balanced_accuracy_on_rd"],
            "rd_source_balanced_macro_f1_on_rd": test_metrics["rd_source_balanced_macro_f1_on_rd"],
            "rd_min_source_macro_f1_on_rd": test_metrics["rd_min_source_macro_f1_on_rd"],
            "rd_unknown_clutter_recall_mean": test_metrics["rd_unknown_clutter_recall_mean"],
            "final_unknown_clutter_recall_mean": test_metrics["final_unknown_clutter_recall_mean"],
            "final_unknown_recall": float(test_metrics["final_per_class_recall"].get("UnknownTarget", 0.0)),
            "final_clutter_recall": float(test_metrics["final_per_class_recall"].get("ClutterTarget", 0.0)),
            "parent_unknown_recall": float(test_metrics.get("parent_unknown_recall", 0.0)),
            "parent_clutter_recall": float(test_metrics.get("parent_clutter_recall", 0.0)),
            "parent_unknown_clutter_recall_mean": float(test_metrics.get("parent_unknown_clutter_recall_mean", 0.0)),
            "parent_mean_logit_macro_f1": float(test_metrics.get("parent_mean_logit_macro_f1", 0.0)),
        },
        "label_names": LOW_ALTITUDE_CLASSES,
        "fine_label_distribution": {
            str(label_name): int(test_metrics["fine_label_names"].count(label_name))
            for label_name in sorted(set(test_metrics.get("fine_label_names", [])))
        },
        "classification_confusion_matrix": compute_confusion_matrix(
            test_metrics["logits"],
            test_metrics["labels"],
            len(LOW_ALTITUDE_CLASSES),
        ).tolist(),
        "per_class": [
            {
                "label_name": item.label_name,
                "support": item.support,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in compute_per_class_metrics(
                test_metrics["logits"],
                test_metrics["labels"],
                LOW_ALTITUDE_CLASSES,
            )
        ],
        "route": {
            "label_names": ["low_altitude_suspect"],
            "confusion_matrix": [[len(test_metrics["sample_ids"])]],
            "per_route": [
                {
                    "route_name": "low_altitude_suspect",
                    "support": len(test_metrics["sample_ids"]),
                    "correct": len(test_metrics["sample_ids"]),
                    "recall": 1.0,
                }
            ],
        },
        "by_source": {
            source_name: {
                "count": sum(1 for x in test_metrics["source_names"] if x == source_name),
                "accuracy": float(test_metrics["by_source_metrics"].get(source_name, {}).get("accuracy", 0.0)),
                "macro_precision": float(test_metrics["by_source_metrics"].get(source_name, {}).get("macro_precision", 0.0)),
                "macro_recall": float(test_metrics["by_source_metrics"].get(source_name, {}).get("macro_recall", 0.0)),
                "macro_f1": float(test_metrics["by_source_metrics"].get(source_name, {}).get("macro_f1", 0.0)),
                "supported_class_count": int(test_metrics["by_source_supported_metrics"].get(source_name, {}).get("supported_class_count", 0)),
                "supported_macro_precision": float(test_metrics["by_source_supported_metrics"].get(source_name, {}).get("supported_macro_precision", 0.0)),
                "supported_macro_recall": float(test_metrics["by_source_supported_metrics"].get(source_name, {}).get("supported_macro_recall", 0.0)),
                "supported_macro_f1": float(test_metrics["by_source_supported_metrics"].get(source_name, {}).get("supported_macro_f1", 0.0)),
                "traj_accuracy": float(test_metrics["traj_by_source_metrics"].get(source_name, {}).get("accuracy", 0.0)),
                "traj_macro_precision": float(test_metrics["traj_by_source_metrics"].get(source_name, {}).get("macro_precision", 0.0)),
                "traj_macro_recall": float(test_metrics["traj_by_source_metrics"].get(source_name, {}).get("macro_recall", 0.0)),
                "traj_macro_f1": float(test_metrics["traj_by_source_metrics"].get(source_name, {}).get("macro_f1", 0.0)),
                "joint_accuracy": float(test_metrics["joint_by_source_metrics"].get(source_name, {}).get("accuracy", 0.0)),
                "joint_macro_precision": float(test_metrics["joint_by_source_metrics"].get(source_name, {}).get("macro_precision", 0.0)),
                "joint_macro_recall": float(test_metrics["joint_by_source_metrics"].get(source_name, {}).get("macro_recall", 0.0)),
                "joint_macro_f1": float(test_metrics["joint_by_source_metrics"].get(source_name, {}).get("macro_f1", 0.0)),
                "joint_accuracy_on_rd": float(test_metrics["joint_by_source_metrics_on_rd"].get(source_name, {}).get("accuracy", 0.0)),
                "joint_macro_precision_on_rd": float(test_metrics["joint_by_source_metrics_on_rd"].get(source_name, {}).get("macro_precision", 0.0)),
                "joint_macro_recall_on_rd": float(test_metrics["joint_by_source_metrics_on_rd"].get(source_name, {}).get("macro_recall", 0.0)),
                "joint_macro_f1_on_rd": float(test_metrics["joint_by_source_metrics_on_rd"].get(source_name, {}).get("macro_f1", 0.0)),
                "rd_accuracy_on_rd": float(test_metrics["rd_by_source_metrics_on_rd"].get(source_name, {}).get("accuracy", 0.0)),
                "rd_macro_precision_on_rd": float(test_metrics["rd_by_source_metrics_on_rd"].get(source_name, {}).get("macro_precision", 0.0)),
                "rd_macro_recall_on_rd": float(test_metrics["rd_by_source_metrics_on_rd"].get(source_name, {}).get("macro_recall", 0.0)),
                "rd_macro_f1_on_rd": float(test_metrics["rd_by_source_metrics_on_rd"].get(source_name, {}).get("macro_f1", 0.0)),
            }
            for source_name in sorted(set(test_metrics["source_names"]))
        },
        "by_rd_presence": {},
        "branch_metrics": {
            "final": {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "per_class_recall": test_metrics["final_per_class_recall"],
                "drone_bird": test_metrics["final_drone_bird"],
                "unknown_clutter_recall_mean": test_metrics["final_unknown_clutter_recall_mean"],
            },
            "trajectory": {
                "accuracy": test_metrics["traj_accuracy"],
                "macro_f1": test_metrics["traj_macro_f1"],
                "per_class_recall": test_metrics["traj_per_class_recall"],
                "drone_bird": test_metrics["traj_drone_bird"],
            },
            "joint": {
                "accuracy": test_metrics["joint_accuracy"],
                "macro_f1": test_metrics["joint_macro_f1"],
                "accuracy_on_rd": test_metrics["joint_accuracy_on_rd"],
                "macro_f1_on_rd": test_metrics["joint_macro_f1_on_rd"],
                "minus_traj_macro_f1_on_rd": test_metrics["joint_minus_traj_macro_f1_on_rd"],
                "rd_weight_mean": test_metrics["joint_rd_weight_mean"],
                "rd_weight_on_rd_mean": test_metrics["joint_rd_weight_on_rd_mean"],
            },
            "rd": {
                "accuracy": test_metrics["rd_accuracy"],
                "macro_f1": test_metrics["rd_macro_f1"],
                "accuracy_on_rd": test_metrics["rd_accuracy_on_rd"],
                "macro_f1_on_rd": test_metrics["rd_macro_f1_on_rd"],
                "per_class_recall": test_metrics["rd_per_class_recall"],
                "per_class_recall_on_rd": test_metrics["rd_per_class_recall_on_rd"],
                "drone_bird": test_metrics["rd_drone_bird"],
                "unknown_clutter_recall_mean": test_metrics["rd_unknown_clutter_recall_mean"],
                "hard_pair_report": test_metrics["rd_hard_pair_report"],
                "rescue_count": test_metrics["rd_rescue_count"],
                "harm_count": test_metrics["rd_harm_count"],
                "rescue_rate_on_rd": test_metrics["rd_rescue_rate_on_rd"],
                "harm_rate_on_rd": test_metrics["rd_harm_rate_on_rd"],
                "net_gain_count": test_metrics["rd_net_gain_count"],
                "net_gain_rate_on_rd": test_metrics["rd_net_gain_rate_on_rd"],
                "rescue_unadopted_count": test_metrics["rd_rescue_unadopted_count"],
                "harm_adopted_count": test_metrics["rd_harm_adopted_count"],
                "raw_tr_rd_accuracy": test_metrics["rd_raw_tr_rd_accuracy"],
                "by_source_accuracy_on_rd": test_metrics["rd_by_source_accuracy_on_rd"],
                "by_source_macro_f1_on_rd": test_metrics["rd_by_source_macro_f1_on_rd"],
                "source_balanced_accuracy_on_rd": test_metrics["rd_source_balanced_accuracy_on_rd"],
                "min_source_accuracy_on_rd": test_metrics["rd_min_source_accuracy_on_rd"],
                "source_balanced_macro_f1_on_rd": test_metrics["rd_source_balanced_macro_f1_on_rd"],
                "min_source_macro_f1_on_rd": test_metrics["rd_min_source_macro_f1_on_rd"],
                "teacher_by_source": test_metrics["rd_teacher_by_source"],
                "class_fusion_weight_mean": test_metrics["rd_class_fusion_weight_mean"],
                "class_fusion_weight_on_rd_mean": test_metrics["rd_class_fusion_weight_on_rd_mean"],
                "pair_fusion_weight_mean": test_metrics["rd_pair_fusion_weight_mean"],
                "pair_fusion_weight_on_rd_mean": test_metrics["rd_pair_fusion_weight_on_rd_mean"],
                "pair_fusion_weight_by_pair": test_metrics["rd_pair_fusion_weight_by_pair"],
                "pair_fusion_weight_on_rd_by_pair": test_metrics["rd_pair_fusion_weight_on_rd_by_pair"],
                "known_confidence_mean": test_metrics["known_confidence_mean"],
                "residual_score_mean": test_metrics["residual_score_mean"],
                "residual_score_on_residual_mean": test_metrics["residual_score_on_residual_mean"],
                "traj_residual_accuracy": test_metrics["traj_residual_accuracy"],
                "residual_accuracy_on_rd": test_metrics["rd_residual_accuracy_on_rd"],
                "unknown_clutter_detail_accuracy_on_residual_rd": test_metrics["rd_unknown_clutter_detail_accuracy_on_residual_rd"],
                "clutter_morphology_accuracy_on_residual_rd": test_metrics["rd_clutter_morphology_accuracy_on_residual_rd"],
                "unknown_clutter_detail_clutter_recall": test_metrics["rd_unknown_clutter_detail_clutter_recall"],
                "clutter_morphology_clutter_recall": test_metrics["rd_clutter_morphology_clutter_recall"],
            },
        },
        "device": device_info,
        "predictions": [
            {
                "sample_id": sid,
                "parent_sample_id": str(window_metadata.get(str(sid), {}).get("parent_sample_id", _parent_sample_id(str(sid)))),
                "window_start_seconds": window_metadata.get(str(sid), {}).get("window_start_seconds"),
                "window_end_seconds": window_metadata.get(str(sid), {}).get("window_end_seconds"),
                "window_point_count": window_metadata.get(str(sid), {}).get("window_point_count"),
                "window_mode": window_metadata.get(str(sid), {}).get("window_mode"),
                "source_name": src,
                "label_name_true": true_label,
                "label_name_pred": LOW_ALTITUDE_CLASSES[int(test_metrics["logits"][idx].argmax().item())],
                "label_name_traj_pred": LOW_ALTITUDE_CLASSES[int(test_metrics["traj_logits"][idx].argmax().item())],
                "label_name_joint_pred": LOW_ALTITUDE_CLASSES[int(test_metrics["joint_logits"][idx].argmax().item())],
                "label_name_rd_pred": LOW_ALTITUDE_CLASSES[int(test_metrics["rd_logits"][idx].argmax().item())],
                "route_name_true": "low_altitude_suspect",
                "route_name_pred": "low_altitude_suspect",
                "is_correct": LOW_ALTITUDE_CLASSES[int(test_metrics["logits"][idx].argmax().item())] == true_label,
                "is_route_correct": True,
                "has_rd": rd_flag,
            }
            for idx, (sid, src, true_label, rd_flag) in enumerate(zip(
                test_metrics["sample_ids"],
                test_metrics["source_names"],
                test_metrics["label_names"],
                test_metrics["has_rd"],
            ))
        ],
    }
    for source_name in evaluation_report["by_source"]:
        source_indices = [
            idx for idx, name in enumerate(test_metrics["source_names"])
            if name == source_name
        ]
        if source_indices:
            source_logits = test_metrics["logits"][source_indices]
            source_labels = test_metrics["labels"][source_indices]
            evaluation_report["by_source"][source_name]["per_label"] = [
                {
                    "label_name": item.label_name,
                    "support": item.support,
                    "precision": item.precision,
                    "recall": item.recall,
                    "f1": item.f1,
                }
                for item in compute_per_class_metrics(
                    source_logits,
                    source_labels,
                    LOW_ALTITUDE_CLASSES,
                )
            ]

    rd_mask = torch.tensor(test_metrics["has_rd"], dtype=torch.bool)
    evaluation_report["by_rd_presence"] = {
        "with_rd": {
            "count": int(rd_mask.sum().item()),
            "accuracy": _subset_accuracy(test_metrics["logits"], test_metrics["labels"], rd_mask),
        },
        "without_rd": {
            "count": int((~rd_mask).sum().item()),
            "accuracy": _subset_accuracy(test_metrics["logits"], test_metrics["labels"], ~rd_mask),
        },
    }
    evaluation_report_path = run_paths.output_dir / "evaluation_report.json"
    evaluation_report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": _rel(selected_checkpoint_path),
        "precision_checkpoint": precision_checkpoint_path,
        "precision_checkpoint_name": precision_checkpoint_name,
        "precision_checkpoint_metrics": precision_candidate,
        "precision_checkpoint_export": precision_checkpoint_export,
        "precision_checkpoint_validation": precision_checkpoint_validation,
        "candidate_status": candidate_status,
        "selected_checkpoint_name": selected_checkpoint_name,
        "selected_checkpoint_eligible": bool(selected_checkpoint_selection_eligible),
        "release_eligible": bool(release_eligible),
        "release_gate_metrics": release_gate_metrics,
        "release_gate_failures": release_gate_failures,
        "checkpoint_registry": checkpoint_registry,
        "diagnostic_test_recommendation": diagnostic_test_recommendation,
        "precision_candidate": precision_candidate,
        "precision_candidate_checkpoint": precision_candidate.get("path", ""),
        "precision_candidate_checkpoint_name": precision_candidate.get("checkpoint_name", ""),
        "precision_candidate_metrics": precision_candidate,
        "precision_candidate_readiness": precision_readiness_summary,
        "precision_candidate_delta_vs_selected": precision_candidate_delta_vs_selected,
        "rd_joint_bottleneck_summary": rd_joint_bottleneck_summary,
        "diagnostic_checkpoint_test_count": len(diagnostic_checkpoint_tests),
        "selected_joint_logit_delta_scale": float(selected_checkpoint_joint_delta_scale),
        "best_final_macro_f1_epoch": int(best_final_macro_epoch),
        "best_final_macro_f1": float(best_final_macro_f1 if best_final_macro_f1 != float("-inf") else 0.0),
        "best_joint_gain_epoch": int(best_joint_gain_epoch),
        "best_joint_gain_score": float(best_joint_gain_score if best_joint_gain_score != float("-inf") else 0.0),
        "last_full_val_epoch": int(last_full_val_epoch),
        "history_file": _rel(history_path),
        "manifest_file": _rel(manifest_path),
        "evaluation_report": _rel(evaluation_report_path),
        "test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "traj_accuracy": test_metrics["traj_accuracy"],
            "traj_macro_f1": test_metrics["traj_macro_f1"],
            "rd_accuracy": test_metrics["rd_accuracy"],
            "rd_macro_f1": test_metrics["rd_macro_f1"],
            "rd_accuracy_on_rd": test_metrics["rd_accuracy_on_rd"],
            "rd_macro_f1_on_rd": test_metrics["rd_macro_f1_on_rd"],
            "rd_drone_bird_recall_mean": test_metrics["rd_drone_bird_recall_mean"],
            "rd_unknown_clutter_recall_mean": test_metrics["rd_unknown_clutter_recall_mean"],
            "rd_per_class_recall_on_rd": test_metrics["rd_per_class_recall_on_rd"],
            "rd_drone_bird": test_metrics["rd_drone_bird"],
            "rd_hard_pair_report": test_metrics["rd_hard_pair_report"],
            "rd_rescue_count": test_metrics["rd_rescue_count"],
            "rd_harm_count": test_metrics["rd_harm_count"],
            "rd_rescue_rate_on_rd": test_metrics["rd_rescue_rate_on_rd"],
            "rd_harm_rate_on_rd": test_metrics["rd_harm_rate_on_rd"],
            "rd_net_gain_count": test_metrics["rd_net_gain_count"],
            "rd_net_gain_rate_on_rd": test_metrics["rd_net_gain_rate_on_rd"],
            "final_rescue_count": test_metrics["final_rescue_count"],
            "final_harm_count": test_metrics["final_harm_count"],
            "final_rescue_rate_on_rd": test_metrics["final_rescue_rate_on_rd"],
            "final_harm_rate_on_rd": test_metrics["final_harm_rate_on_rd"],
            "final_net_gain_count": test_metrics["final_net_gain_count"],
            "final_net_gain_rate_on_rd": test_metrics["final_net_gain_rate_on_rd"],
            "rd_raw_tr_rd_accuracy": test_metrics["rd_raw_tr_rd_accuracy"],
            "rd_teacher_by_source": test_metrics["rd_teacher_by_source"],
            "final_unknown_clutter_recall_mean": test_metrics["final_unknown_clutter_recall_mean"],
            "final_unknown_recall": float(test_metrics["final_per_class_recall"].get("UnknownTarget", 0.0)),
            "final_clutter_recall": float(test_metrics["final_per_class_recall"].get("ClutterTarget", 0.0)),
            "parent_unknown_recall": float(test_metrics.get("parent_unknown_recall", 0.0)),
            "parent_clutter_recall": float(test_metrics.get("parent_clutter_recall", 0.0)),
            "parent_unknown_clutter_recall_mean": float(test_metrics.get("parent_unknown_clutter_recall_mean", 0.0)),
            "tr_rd_path_rate": test_metrics["tr_rd_path_rate"],
            "tr_rd_path_rate_on_rd": test_metrics["tr_rd_path_rate_on_rd"],
            "rd_activation_rate": test_metrics["rd_activation_rate"],
            "rd_activation_rate_on_rd": test_metrics["rd_activation_rate_on_rd"],
            "traj_accuracy_on_rd": test_metrics["traj_accuracy_on_rd"],
            "traj_macro_f1_on_rd": test_metrics["traj_macro_f1_on_rd"],
            "joint_accuracy": test_metrics["joint_accuracy"],
            "joint_macro_f1": test_metrics["joint_macro_f1"],
            "joint_accuracy_on_rd": test_metrics["joint_accuracy_on_rd"],
            "joint_macro_f1_on_rd": test_metrics["joint_macro_f1_on_rd"],
            "joint_minus_traj_macro_f1_on_rd": test_metrics["joint_minus_traj_macro_f1_on_rd"],
            "rd_fusion_weight_mean": test_metrics["rd_fusion_weight_mean"],
            "rd_fusion_weight_on_rd_mean": test_metrics["rd_fusion_weight_on_rd_mean"],
            "rd_class_fusion_weight_mean": test_metrics["rd_class_fusion_weight_mean"],
            "rd_class_fusion_weight_on_rd_mean": test_metrics["rd_class_fusion_weight_on_rd_mean"],
            "rd_pair_fusion_weight_mean": test_metrics["rd_pair_fusion_weight_mean"],
            "rd_pair_fusion_weight_on_rd_mean": test_metrics["rd_pair_fusion_weight_on_rd_mean"],
            "rd_pair_fusion_weight_by_pair": test_metrics["rd_pair_fusion_weight_by_pair"],
            "rd_pair_fusion_weight_on_rd_by_pair": test_metrics["rd_pair_fusion_weight_on_rd_by_pair"],
            "joint_rd_weight_mean": test_metrics["joint_rd_weight_mean"],
            "joint_rd_weight_on_rd_mean": test_metrics["joint_rd_weight_on_rd_mean"],
            "selected_joint_logit_delta_scale": float(test_metrics.get("selected_joint_logit_delta_scale", selected_checkpoint_joint_delta_scale)),
            "joint_logit_delta_probe": test_metrics.get("joint_logit_delta_probe", {}),
            "known_confidence_mean": test_metrics["known_confidence_mean"],
            "residual_score_mean": test_metrics["residual_score_mean"],
            "residual_score_on_residual_mean": test_metrics["residual_score_on_residual_mean"],
            "traj_residual_accuracy": test_metrics["traj_residual_accuracy"],
            "rd_residual_accuracy_on_rd": test_metrics["rd_residual_accuracy_on_rd"],
            "rd_unknown_clutter_detail_accuracy_on_residual_rd": test_metrics["rd_unknown_clutter_detail_accuracy_on_residual_rd"],
            "rd_clutter_morphology_accuracy_on_residual_rd": test_metrics["rd_clutter_morphology_accuracy_on_residual_rd"],
            "rd_unknown_clutter_detail_clutter_recall": test_metrics["rd_unknown_clutter_detail_clutter_recall"],
            "rd_clutter_morphology_clutter_recall": test_metrics["rd_clutter_morphology_clutter_recall"],
            "by_source_metrics": test_metrics["by_source_metrics"],
            "by_source_supported_metrics": test_metrics["by_source_supported_metrics"],
            "traj_by_source_metrics": test_metrics["traj_by_source_metrics"],
            "joint_by_source_metrics": test_metrics["joint_by_source_metrics"],
            "joint_by_source_metrics_on_rd": test_metrics["joint_by_source_metrics_on_rd"],
            "rd_by_source_metrics_on_rd": test_metrics["rd_by_source_metrics_on_rd"],
            "by_source_accuracy": test_metrics["by_source_accuracy"],
            "traj_by_source_accuracy": test_metrics["traj_by_source_accuracy"],
            "joint_by_source_accuracy": test_metrics["joint_by_source_accuracy"],
            "joint_by_source_accuracy_on_rd": test_metrics["joint_by_source_accuracy_on_rd"],
            "rd_by_source_accuracy_on_rd": test_metrics["rd_by_source_accuracy_on_rd"],
            "joint_by_source_macro_f1": test_metrics["joint_by_source_macro_f1"],
            "joint_by_source_macro_f1_on_rd": test_metrics["joint_by_source_macro_f1_on_rd"],
            "rd_by_source_macro_f1_on_rd": test_metrics["rd_by_source_macro_f1_on_rd"],
            "rd_source_balanced_accuracy_on_rd": test_metrics["rd_source_balanced_accuracy_on_rd"],
            "rd_min_source_accuracy_on_rd": test_metrics["rd_min_source_accuracy_on_rd"],
            "rd_source_balanced_macro_f1_on_rd": test_metrics["rd_source_balanced_macro_f1_on_rd"],
            "rd_min_source_macro_f1_on_rd": test_metrics["rd_min_source_macro_f1_on_rd"],
            "raw_tr_rd_accuracy": test_metrics["raw_tr_rd_accuracy"],
            "source_balanced_accuracy": test_metrics["source_balanced_accuracy"],
            "min_source_accuracy": test_metrics["min_source_accuracy"],
            "by_source_macro_f1": test_metrics["by_source_macro_f1"],
            "traj_by_source_macro_f1": test_metrics["traj_by_source_macro_f1"],
            "raw_tr_rd_macro_f1": test_metrics["raw_tr_rd_macro_f1"],
            "excel_track_macro_f1": test_metrics["excel_track_macro_f1"],
            "source_balanced_macro_f1": test_metrics["source_balanced_macro_f1"],
            "min_source_macro_f1": test_metrics["min_source_macro_f1"],
            "with_rd_accuracy": test_metrics["with_rd_accuracy"],
            "without_rd_accuracy": test_metrics["without_rd_accuracy"],
            "skipped_non_finite_batches": test_metrics["skipped_non_finite_batches"],
            "has_valid_batches": test_metrics["has_valid_batches"],
        },
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "index_name": config.index_name,
        "window_evaluation": evaluation_report["window_evaluation"],
        "device": device_info,
        "taxonomy": {label: idx for idx, label in enumerate(LOW_ALTITUDE_CLASSES)},
        "trainable_parameters": _count_trainable_parameters(representation_builder) + _count_trainable_parameters(rd_proj) + _count_trainable_parameters(model),
        "class_weights": class_weights.detach().cpu().tolist(),
        "rd_class_weights": rd_class_weights.detach().cpu().tolist(),
        "unknown_loss_weight": config.unknown_loss_weight,
        "unknown_sample_boost": config.unknown_sample_boost,
        "tr_rd_path_rate": test_metrics["tr_rd_path_rate"],
        "tr_rd_path_rate_on_rd": test_metrics["tr_rd_path_rate_on_rd"],
        "rd_activation_rate": test_metrics["rd_activation_rate"],
        "rd_activation_rate_on_rd": test_metrics["rd_activation_rate_on_rd"],
        "rd_class_fusion_weight_mean": test_metrics["rd_class_fusion_weight_mean"],
        "rd_class_fusion_weight_on_rd_mean": test_metrics["rd_class_fusion_weight_on_rd_mean"],
        "rd_pair_fusion_weight_mean": test_metrics["rd_pair_fusion_weight_mean"],
        "rd_pair_fusion_weight_on_rd_mean": test_metrics["rd_pair_fusion_weight_on_rd_mean"],
        "rd_pair_fusion_weight_by_pair": test_metrics["rd_pair_fusion_weight_by_pair"],
        "rd_pair_fusion_weight_on_rd_by_pair": test_metrics["rd_pair_fusion_weight_on_rd_by_pair"],
        "known_confidence_mean": test_metrics["known_confidence_mean"],
        "residual_score_mean": test_metrics["residual_score_mean"],
        "residual_score_on_residual_mean": test_metrics["residual_score_on_residual_mean"],
        "traj_residual_accuracy": test_metrics["traj_residual_accuracy"],
        "rd_residual_accuracy_on_rd": test_metrics["rd_residual_accuracy_on_rd"],
        "rd_unknown_clutter_detail_accuracy_on_residual_rd": test_metrics["rd_unknown_clutter_detail_accuracy_on_residual_rd"],
        "rd_clutter_morphology_accuracy_on_residual_rd": test_metrics["rd_clutter_morphology_accuracy_on_residual_rd"],
        "rd_unknown_clutter_detail_clutter_recall": test_metrics["rd_unknown_clutter_detail_clutter_recall"],
        "rd_clutter_morphology_clutter_recall": test_metrics["rd_clutter_morphology_clutter_recall"],
        "low_altitude_deep_reroute_aux_weight": config.low_altitude_deep_reroute_aux_weight,
        "low_altitude_deep_reroute_aux_interval": config.low_altitude_deep_reroute_aux_interval,
        "low_altitude_deep_reroute_max_frames": config.low_altitude_deep_reroute_max_frames,
        "low_altitude_joint_aux_weight": config.low_altitude_joint_aux_weight,
        "low_altitude_joint_agreement_prior_scale": config.low_altitude_joint_agreement_prior_scale,
        "low_altitude_joint_reliability_aux_weight": config.low_altitude_joint_reliability_aux_weight,
        "low_altitude_rd_pair_takeover_floor": config.low_altitude_rd_pair_takeover_floor,
        "low_altitude_rd_pair_harm_ceiling": config.low_altitude_rd_pair_harm_ceiling,
        "low_altitude_rd_teacher_warmup_epochs": config.low_altitude_rd_teacher_warmup_epochs,
        "low_altitude_rd_teacher_aux_boost": config.low_altitude_rd_teacher_aux_boost,
        "source_normalization": bool(config.source_normalization),
        "source_feature_stats_saved": bool(source_feature_stats is not None),
        "prefer_stage2_checkpoint": bool(config.prefer_stage2_checkpoint),
        "sequence_encoder_lr_multiplier": config.sequence_encoder_lr_multiplier,
        "pretrained_track_checkpoint": config.pretrained_track_checkpoint,
        "pretrained_track_info": pretrained_track_info,
        "use_rd_patch_cnn": bool(config.use_rd_patch_cnn),
        "rd_pipeline": {
            "dataset_cache_size": int(config.dataset_cache_size),
            "rd_encoded_cache_size": int(config.rd_encoded_cache_size),
            "enable_window_rd_time_crop": bool(config.enable_window_rd_time_crop),
            "rd_window_crop_min_frames": int(config.rd_window_crop_min_frames),
            "rd_joint_segment_count": int(config.rd_joint_segment_count),
            "enable_rd_segment_joint_tokens": bool(config.enable_rd_segment_joint_tokens),
            "use_rd_segment_store": bool(config.use_rd_segment_store),
            "rd_segment_manifest_name": str(config.rd_segment_manifest_name),
            "window_segment_map_name": str(config.window_segment_map_name),
            "rd_segment_store_cache_size": int(config.rd_segment_store_cache_size),
            "allow_rd_segment_store_fallback": bool(config.allow_rd_segment_store_fallback),
            "stage2_rd_shard_local_sampling": bool(config.stage2_rd_shard_local_sampling),
        },
        "speed_controls": {
            "enable_batch_parameter_rollback": bool(config.enable_batch_parameter_rollback),
            "compute_eval_aux_loss": bool(config.compute_eval_aux_loss),
            "max_train_batches_per_epoch": int(config.max_train_batches_per_epoch),
            "quick_val_max_batches": int(config.quick_val_max_batches),
            "full_val_interval_epochs": int(config.full_val_interval_epochs),
            "full_val_warmup_epochs": int(config.full_val_warmup_epochs),
            "full_val_on_last_epoch": bool(config.full_val_on_last_epoch),
            "num_workers": int(config.num_workers),
            "eval_num_workers": int(config.eval_num_workers),
            "pin_memory": bool(config.pin_memory),
            "persistent_workers": bool(config.persistent_workers),
            "prefetch_factor": int(config.prefetch_factor),
            "non_blocking_h2d": bool(config.non_blocking_h2d),
            "joint_logit_delta_scale": float(config.joint_logit_delta_scale),
            "joint_logit_delta_eval_scales": [float(item) for item in _unique_joint_logit_delta_scales(config)],
            "joint_logit_delta_selection_metric": str(config.joint_logit_delta_selection_metric),
            "selected_joint_logit_delta_scale": float(selected_checkpoint_joint_delta_scale),
            "rd_specialist_residual_scale": float(config.rd_specialist_residual_scale),
            "joint_agreement_prior_scale": float(config.low_altitude_joint_agreement_prior_scale),
        },
        "low_altitude_traj_aux_ratio": config.low_altitude_traj_aux_ratio,
        "low_altitude_rd_aux_ratio": config.low_altitude_rd_aux_ratio,
        "low_altitude_rd_hard_aux_weight": config.low_altitude_rd_hard_aux_weight,
        "low_altitude_rd_focal_gamma": config.low_altitude_rd_focal_gamma,
        "low_altitude_rd_binary_aux_weight": config.low_altitude_rd_binary_aux_weight,
        "low_altitude_rd_pairwise_aux_weight": config.low_altitude_rd_pairwise_aux_weight,
        "low_altitude_rd_unknown_pairwise_aux_weight": config.low_altitude_rd_unknown_pairwise_aux_weight,
        "low_altitude_rd_clutter_pairwise_aux_weight": config.low_altitude_rd_clutter_pairwise_aux_weight,
        "low_altitude_rd_unknown_clutter_pairwise_aux_weight": config.low_altitude_rd_unknown_clutter_pairwise_aux_weight,
        "low_altitude_residual_aux_weight": config.low_altitude_residual_aux_weight,
        "low_altitude_rd_residual_aux_weight": config.low_altitude_rd_residual_aux_weight,
        "low_altitude_rd_unknown_clutter_aux_weight": config.low_altitude_rd_unknown_clutter_aux_weight,
        "low_altitude_rd_clutter_morphology_aux_weight": config.low_altitude_rd_clutter_morphology_aux_weight,
        "low_altitude_rd_rare_margin_aux_weight": config.low_altitude_rd_rare_margin_aux_weight,
        "low_altitude_rd_rare_margin": config.low_altitude_rd_rare_margin,
        "low_altitude_rd_forced_fusion_weight": config.low_altitude_rd_forced_fusion_weight,
        "low_altitude_traj_supervision_weight": config.low_altitude_traj_supervision_weight,
        "low_altitude_joint_aux_weight": config.low_altitude_joint_aux_weight,
        "low_altitude_joint_agreement_prior_scale": config.low_altitude_joint_agreement_prior_scale,
        "low_altitude_joint_reliability_aux_weight": config.low_altitude_joint_reliability_aux_weight,
        "low_altitude_deep_reroute_aux_weight": config.low_altitude_deep_reroute_aux_weight,
        "low_altitude_deep_reroute_aux_interval": config.low_altitude_deep_reroute_aux_interval,
        "low_altitude_deep_reroute_max_frames": config.low_altitude_deep_reroute_max_frames,
        "low_altitude_rd_teacher_warmup_epochs": config.low_altitude_rd_teacher_warmup_epochs,
        "low_altitude_rd_teacher_aux_boost": config.low_altitude_rd_teacher_aux_boost,
        "low_altitude_traj_anchor_weight": config.low_altitude_traj_anchor_weight,
        "low_altitude_traj_anchor_start_epoch": config.low_altitude_traj_anchor_start_epoch,
        "rd_bird_logit_scale": config.rd_bird_logit_scale,
        "rd_unknown_logit_scale": config.rd_unknown_logit_scale,
        "rd_clutter_logit_scale": config.rd_clutter_logit_scale,
        "rd_residual_logit_scale": config.rd_residual_logit_scale,
        "rd_unknown_clutter_logit_scale": config.rd_unknown_clutter_logit_scale,
        "rd_clutter_morphology_logit_scale": config.rd_clutter_morphology_logit_scale,
        "rd_bird_positive_weight": config.rd_bird_positive_weight,
        "rd_unknown_positive_weight": config.rd_unknown_positive_weight,
        "rd_clutter_positive_weight": config.rd_clutter_positive_weight,
        "low_altitude_mutual_exclude_rare_classes": bool(config.low_altitude_mutual_exclude_rare_classes),
        "rd_sample_boost": config.rd_sample_boost,
        "raw_source_boost": config.raw_source_boost,
        "excel_source_boost": config.excel_source_boost,
        "cq08_source_boost": config.cq08_source_boost,
        "source_balance_power": config.source_balance_power,
        "source_label_balance_power": config.source_label_balance_power,
        "parent_window_balance_power": config.parent_window_balance_power,
        "stage2_rd_aware_sampling": config.stage2_rd_aware_sampling,
        "stage2_rd_batch_fraction": config.stage2_rd_batch_fraction,
        "stage2_rd_shard_local_sampling": config.stage2_rd_shard_local_sampling,
        "max_sampler_weight": config.max_sampler_weight,
        "bird_sample_boost": config.bird_sample_boost,
        "freeze_track_encoder_epochs": config.freeze_track_encoder_epochs,
        "split_sizes": {
            "train": len(loaders.split.train_ids),
            "val": len(loaders.split.val_ids),
            "test": len(loaders.split.test_ids),
        },
        "rd_coverage": {
            "train": _rd_coverage_from_index(loaders.train_loader.dataset.index_df),
            "val": _rd_coverage_from_index(loaders.val_loader.dataset.index_df),
            "test": _rd_coverage_from_index(loaders.test_loader.dataset.index_df),
        },
        "best_epoch": selected_checkpoint_epoch,
        "best_selection_score": selected_checkpoint_score,
        "best_selection_reason": selected_checkpoint_reason,
        "best_selection_components": selected_checkpoint_components,
        "v2_selection_gate": {
            "min_rd_evidence_macro_f1_for_best": config.min_rd_evidence_macro_f1_for_best,
            "min_drone_bird_rd_recall_for_best": config.min_drone_bird_rd_recall_for_best,
            "min_drone_bird_single_rd_recall_for_best": config.min_drone_bird_single_rd_recall_for_best,
            "min_unknown_clutter_rd_recall_for_best": config.min_unknown_clutter_rd_recall_for_best,
            "min_rare_rd_recall_for_best": config.min_rare_rd_recall_for_best,
            "min_without_rd_accuracy_for_best": config.min_without_rd_accuracy_for_best,
            "min_source_accuracy_for_best": config.min_source_accuracy_for_best,
            "min_source_balanced_accuracy_for_best": config.min_source_balanced_accuracy_for_best,
            "min_source_balanced_macro_f1_for_best": config.min_source_balanced_macro_f1_for_best,
            "min_raw_source_accuracy_for_best": config.min_raw_source_accuracy_for_best,
            "min_raw_source_macro_f1_for_best": config.min_raw_source_macro_f1_for_best,
            "min_excel_source_macro_f1_for_best": config.min_excel_source_macro_f1_for_best,
            "min_rd_raw_source_accuracy_for_best": config.min_rd_raw_source_accuracy_for_best,
            "min_joint_minus_traj_macro_f1_on_rd_for_best": config.min_joint_minus_traj_macro_f1_on_rd_for_best,
            "min_final_net_gain_rate_on_rd_for_best": config.min_final_net_gain_rate_on_rd_for_best,
            "max_final_harm_rate_on_rd_for_best": config.max_final_harm_rate_on_rd_for_best,
            "min_final_minus_traj_macro_f1_for_best": config.min_final_minus_traj_macro_f1_for_best,
            "min_final_unknown_recall_for_best": config.min_final_unknown_recall_for_best,
            "min_final_clutter_recall_for_best": config.min_final_clutter_recall_for_best,
            "min_final_unknown_clutter_recall_mean_for_best": config.min_final_unknown_clutter_recall_mean_for_best,
            "min_release_macro_f1": config.min_release_macro_f1,
            "min_release_final_minus_traj_macro_f1": config.min_release_final_minus_traj_macro_f1,
            "min_release_joint_minus_traj_macro_f1_on_rd": config.min_release_joint_minus_traj_macro_f1_on_rd,
            "min_release_rd_evidence_macro_f1_on_rd": config.min_release_rd_evidence_macro_f1_on_rd,
            "min_release_final_net_gain_rate_on_rd": config.min_release_final_net_gain_rate_on_rd,
            "max_release_final_harm_rate_on_rd": config.max_release_final_harm_rate_on_rd,
            "min_release_parent_macro_f1": config.min_release_parent_macro_f1,
            "min_release_unknown_recall": config.min_release_unknown_recall,
            "min_release_clutter_recall": config.min_release_clutter_recall,
            "min_release_without_rd_accuracy": config.min_release_without_rd_accuracy,
            "min_source_support_for_hard_gate": config.min_source_support_for_hard_gate,
            "min_rd_source_support_for_hard_gate": config.min_rd_source_support_for_hard_gate,
            "min_rd_class_support_for_hard_gate": config.min_rd_class_support_for_hard_gate,
            "require_final_not_worse_than_traj": bool(config.require_final_not_worse_than_traj),
            "max_final_traj_f1_drop": config.max_final_traj_f1_drop,
        },
    }
    summary_path = run_paths.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_paths = write_real_experiment_report(
        output_dir=run_paths.output_dir,
        evaluation_report=evaluation_report,
        summary=summary,
        timestamp_label=run_paths.timestamp_label,
    )
    summary["real_report_markdown"] = str(report_paths.markdown_path.as_posix())
    summary["real_report_json"] = str(report_paths.json_path.as_posix())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
