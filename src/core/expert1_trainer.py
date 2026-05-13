from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .device import resolve_cuda_device
from .experts import LOW_ALTITUDE_CLASSES, LowAltitudeSuspectExpert
from .expert1_config import Expert1TrainConfig
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
    source_summary = _source_summary_from_dict(by_source)
    source_macro_f1_summary = _source_macro_f1_summary(by_source_macro_f1)
    rd_mask = torch.tensor(has_rd, dtype=torch.bool)
    return {
        "by_source_accuracy": by_source,
        "by_source_macro_f1": by_source_macro_f1,
        "source_balanced_accuracy": source_summary["source_balanced_accuracy"],
        "min_source_accuracy": source_summary["min_source_accuracy"],
        "raw_tr_rd_accuracy": source_summary["raw_tr_rd_accuracy"],
        "source_balanced_macro_f1": source_macro_f1_summary["source_balanced_macro_f1"],
        "min_source_macro_f1": source_macro_f1_summary["min_source_macro_f1"],
        "raw_tr_rd_macro_f1": source_macro_f1_summary["raw_tr_rd_macro_f1"],
        "excel_track_macro_f1": source_macro_f1_summary["excel_track_macro_f1"],
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
        "with_rd_accuracy": metrics.get("with_rd_accuracy", 0.0),
        "without_rd_accuracy": metrics.get("without_rd_accuracy", 0.0),
        "selection_score": metrics.get("selection_score", 0.0),
        "selection_eligible": metrics.get("selection_eligible", False),
        "selection_components": metrics.get("selection_components", {}),
        "evaluated_batches": metrics.get("evaluated_batches", 0),
        "max_batches": metrics.get("max_batches", 0),
        "validation_scope": metrics.get("validation_scope", ""),
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
        eligible = eligible and source_balanced_macro_f1 >= config.min_source_balanced_macro_f1_for_best
        if enough_raw_source:
            eligible = eligible and raw_source_accuracy >= config.min_raw_source_accuracy_for_best
            eligible = eligible and raw_source_macro_f1 >= config.min_raw_source_macro_f1_for_best
            eligible = eligible and min_source_accuracy >= config.min_source_accuracy_for_best
        if source_support.get("excel_track", 0) >= config.min_source_support_for_hard_gate:
            eligible = eligible and excel_source_macro_f1 >= config.min_excel_source_macro_f1_for_best
        if enough_rd_raw_source:
            eligible = eligible and rd_raw_source_accuracy >= config.min_rd_raw_source_accuracy_for_best
        eligible = eligible and joint_minus_traj_on_rd >= config.min_joint_minus_traj_macro_f1_on_rd_for_best
        eligible = eligible and final_net_gain_rate >= config.min_final_net_gain_rate_on_rd_for_best
        eligible = eligible and final_harm_rate <= config.max_final_harm_rate_on_rd_for_best
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
        "raw_tr_rd_macro_f1": raw_source_macro_f1,
        "excel_track_macro_f1": excel_source_macro_f1,
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
        "final_harm_rate_on_rd": final_harm_rate,
        "final_net_gain_rate_on_rd": final_net_gain_rate,
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
    stable_correct: list[bool] = []
    num_classes = len(LOW_ALTITUDE_CLASSES)
    has_rd_values = [bool(item) for item in metrics.get("has_rd", [])]
    source_values = [str(item) for item in metrics.get("source_names", [])]

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
        parent_logits = logits[indices]
        parent_mean_logits = parent_logits.mean(dim=0)
        window_preds = parent_logits.argmax(dim=-1)
        pred_counts = torch.bincount(window_preds, minlength=num_classes).to(torch.float32)
        vote_score = pred_counts + parent_mean_logits.detach().cpu().to(torch.float32) * 1e-6
        parent_labels.append(parent_label)
        mean_logits.append(parent_mean_logits)
        vote_logits.append(vote_score)
        first_logits.append(parent_logits[0])
        window_counts.append(len(indices))
        parent_sources.append(source_values[indices[0]] if indices[0] < len(source_values) else "")
        parent_has_rd.append(any(has_rd_values[idx] for idx in indices if idx < len(has_rd_values)))

        stable_index = 0
        for offset, pred in enumerate(window_preds.tolist()):
            if all(other == pred for other in window_preds.tolist()[offset:]):
                stable_index = offset
                break
        stable_sample_id = sample_ids[indices[stable_index]]
        stable_meta = window_metadata.get(stable_sample_id, {})
        latency = float(stable_meta.get("window_end_seconds") or stable_index + 1) - parent_start
        stable_latencies.append(latency)
        stable_correct.append(int(window_preds[stable_index].item()) == parent_label)

    parent_label_tensor = torch.tensor(parent_labels, dtype=torch.long)
    mean_logits_tensor = torch.stack(mean_logits)
    vote_logits_tensor = torch.stack(vote_logits)
    first_logits_tensor = torch.stack(first_logits)
    mean_metrics = compute_classification_metrics(mean_logits_tensor, parent_label_tensor, num_classes=num_classes)
    vote_metrics = compute_classification_metrics(vote_logits_tensor, parent_label_tensor, num_classes=num_classes)
    first_metrics = compute_classification_metrics(first_logits_tensor, parent_label_tensor, num_classes=num_classes)
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
    return {
        "parent_count": len(groups),
        "window_count": len(sample_ids),
        "mean_windows_per_parent": float(np.mean(window_counts)) if window_counts else 0.0,
        "max_windows_per_parent": int(max(window_counts)) if window_counts else 0,
        "with_rd_parent_count": int(parent_has_rd_tensor.sum().item()),
        "without_rd_parent_count": int((~parent_has_rd_tensor).sum().item()),
        "mean_logit": {
            "accuracy": float(mean_metrics.accuracy),
            "macro_precision": float(mean_metrics.macro_precision),
            "macro_recall": float(mean_metrics.macro_recall),
            "macro_f1": float(mean_metrics.macro_f1),
        },
        "majority_vote": {
            "accuracy": float(vote_metrics.accuracy),
            "macro_precision": float(vote_metrics.macro_precision),
            "macro_recall": float(vote_metrics.macro_recall),
            "macro_f1": float(vote_metrics.macro_f1),
        },
        "early_window": {
            "accuracy": float(first_metrics.accuracy),
            "macro_precision": float(first_metrics.macro_precision),
            "macro_recall": float(first_metrics.macro_recall),
            "macro_f1": float(first_metrics.macro_f1),
        },
        "stable_decision": {
            "accuracy": float(sum(stable_correct) / len(stable_correct)) if stable_correct else 0.0,
            "latency_seconds_mean": float(np.mean(stable_latencies)) if stable_latencies else 0.0,
            "latency_seconds_p50": float(np.quantile(stable_latencies, 0.50)) if stable_latencies else 0.0,
            "latency_seconds_p90": float(np.quantile(stable_latencies, 0.90)) if stable_latencies else 0.0,
        },
        "by_source": by_source,
        "per_class_mean_logit": per_class,
    }


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
    tr_rd_active_all: list[bool] = []
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
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        features = batch.features.to(device)
        if config.source_normalization and source_feature_stats is not None:
            features = _apply_source_feature_norm(features, batch.source_names, source_feature_stats)
        physical = batch.physical_features.to(device)
        rd_features = batch.rd_features.to(device)
        rd_patch_features = batch.rd_patch_features.to(device)
        rd_deep_patch_features = batch.rd_deep_patch_features.to(device)
        rd_is_multimodal = batch.rd_is_multimodal.to(device)
        rd_padding_mask = batch.rd_padding_mask.to(device)
        rd_patch_padding_mask = batch.rd_patch_padding_mask.to(device)
        rd_deep_patch_padding_mask = batch.rd_deep_patch_padding_mask.to(device)
        labels = torch.tensor(
            [LOW_ALTITUDE_GLOBAL_TO_LOCAL[int(label)] for label in batch.labels.tolist()],
            dtype=torch.long,
            device=device,
        )

        with torch.set_grad_enabled(train):
            rep = representation_builder(
                sequence_features=features,
                physical_features=physical,
                padding_mask=batch.padding_mask.to(device),
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
                traj_padding_mask=batch.padding_mask.to(device),
                rd_padding_mask=rd_padding_mask,
                rd_patch_padding_mask=rd_patch_padding_mask,
                rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
                enable_reject=(config.enable_reject_in_train if train else config.enable_reject_in_eval) and stage2_enabled,
                disable_tr_rd_fusion=bool(train and not stage2_enabled),
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
                if config.low_altitude_deep_reroute_aux_weight > 0.0:
                    deep_output = model(
                        rep.fused_features,
                        rd_features=rd_features,
                        traj_sequence_features=rep.sequence_features,
                        rd_patch_features=rd_patch_features,
                        rd_deep_patch_features=rd_deep_patch_features,
                        rd_context=rd_embedding,
                        low_altitude_targets=labels if train else None,
                        rd_is_multimodal=rd_is_multimodal,
                        traj_padding_mask=batch.padding_mask.to(device),
                        rd_padding_mask=rd_padding_mask,
                        rd_patch_padding_mask=rd_patch_padding_mask,
                        rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
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
        tr_rd_active_all.extend([bool(x) for x in output.tr_rd_active.detach().cpu().tolist()])
        fusion_weights_all.append(output.fusion_weights.detach().cpu())
        if output.joint_logits is not None:
            joint_logits_all.append(output.joint_logits.detach().cpu())
        if output.joint_rd_weight is not None:
            joint_rd_weights_all.append(output.joint_rd_weight.detach().cpu())
        if output.rd_class_fusion_weights is not None:
            rd_class_fusion_weights_all.append(output.rd_class_fusion_weights.detach().cpu())
        if output.rd_pair_fusion_weights is not None:
            rd_pair_fusion_weights_all.append(output.rd_pair_fusion_weights.detach().cpu())

    if not logits_all:
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
            "tr_rd_active": tr_rd_active_all,
            "rd_activated": tr_rd_active_all,
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
    traj_logits_cat = torch.cat(traj_logits_all, dim=0)
    rd_logits_cat = torch.cat(rd_logits_all, dim=0)
    traj_residual_logits_cat = torch.cat(traj_residual_logits_all, dim=0)
    rd_residual_logits_cat = torch.cat(rd_residual_logits_all, dim=0)
    rd_unknown_clutter_logits_cat = torch.cat(rd_unknown_clutter_logits_all, dim=0)
    rd_clutter_morphology_logits_cat = torch.cat(rd_clutter_morphology_logits_all, dim=0)
    known_confidence_cat = torch.cat(known_confidence_all, dim=0)
    residual_score_cat = torch.cat(residual_score_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    residual_targets_cat = _residual_targets(labels_cat)
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
        "traj_by_source_metrics": traj_by_source_metrics,
        "joint_by_source_metrics": joint_by_source_metrics,
        "joint_by_source_metrics_on_rd": joint_by_source_metrics_on_rd,
        "rd_by_source_metrics_on_rd": rd_by_source_metrics_on_rd,
        "by_source_accuracy": group_metrics["by_source_accuracy"],
        "by_source_macro_f1": group_metrics["by_source_macro_f1"],
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
        "with_rd_accuracy": group_metrics["with_rd_accuracy"],
        "without_rd_accuracy": group_metrics["without_rd_accuracy"],
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
        "tr_rd_active": tr_rd_active_all,
        "rd_activated": tr_rd_active_all,
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
        rd_sample_boost=config.rd_sample_boost,
        raw_source_boost=config.raw_source_boost,
        excel_source_boost=config.excel_source_boost,
        cq08_source_boost=config.cq08_source_boost,
        source_balance_power=config.source_balance_power,
        source_label_balance_power=config.source_label_balance_power,
        parent_window_balance_power=config.parent_window_balance_power,
        max_sampler_weight=config.max_sampler_weight,
        unknown_sample_boost=config.unknown_sample_boost,
        clutter_sample_boost=config.clutter_sample_boost,
        bird_sample_boost=config.bird_sample_boost,
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
    ).to(device)
    pretrained_track_info = None
    if config.pretrained_track_checkpoint:
        pretrained_track_info = _load_pretrained_track_encoder(
            representation_builder,
            config.pretrained_track_checkpoint,
            device,
        )
    class_weights = _build_class_weights(loaders.train_loader).to(device)
    rd_class_weights = _build_rd_class_weights(loaders.train_loader).to(device)
    source_feature_stats = _fit_source_feature_stats(loaders.train_loader) if config.source_normalization else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _build_optimizer(representation_builder, rd_proj, model, config)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(representation_builder) + _count_trainable_parameters(rd_proj) + _count_trainable_parameters(model):,}")

    history: list[dict[str, object]] = []
    best_val = -1.0
    best_path = run_paths.output_dir / "best_model.pt"
    fallback_candidate_path = run_paths.output_dir / "fallback_candidate_model.pt"
    best_epoch = -1
    fallback_candidate_epoch = -1
    fallback_candidate_score = float("-inf")
    fallback_candidate_components: dict[str, float] = {}
    best_selection_reason = "selection_score"
    best_selection_components: dict[str, float] = {}
    anchor_traj_logits: dict[str, torch.Tensor] = {}

    for epoch in range(1, config.epochs + 1):
        stage2_enabled = epoch > config.stage1_traj_only_epochs
        stage2_scale = _stage2_scale(epoch, config)
        stage2_epoch = max(0, epoch - config.stage1_traj_only_epochs)
        _set_track_encoder_trainable(representation_builder, epoch > config.freeze_track_encoder_epochs)
        train_metrics = _run_epoch(
            representation_builder,
            rd_proj,
            model,
            loaders.train_loader,
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
        if not train_metrics["has_valid_batches"]:
            break
        full_val_due = (
            config.full_val_interval_epochs <= 1
            or epoch % config.full_val_interval_epochs == 0
            or (config.full_val_on_last_epoch and epoch == config.epochs)
        )
        val_max_batches = 0 if full_val_due else config.quick_val_max_batches
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
        val_metrics["validation_scope"] = "full" if val_max_batches <= 0 else "quick"
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
            f"[*] epoch={epoch:02d} "
            f"stage2={'on' if stage2_enabled else 'off'} "
            f"rd_warmup={'on' if stage2_enabled and stage2_epoch > 0 and stage2_epoch <= config.low_altitude_rd_teacher_warmup_epochs else 'off'} "
            f"scale={stage2_scale:.2f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"val_scope={val_metrics['validation_scope']} "
            f"val_batches={val_metrics.get('evaluated_batches', 0)}"
        )
        can_replace_best = full_val_due and val_selection_eligible and val_selection_score > best_val
        selection_reason = "stage2_v2_joint_composite" if stage2_enabled else "trajectory_anchor_composite"
        if full_val_due and val_selection_score > fallback_candidate_score:
            fallback_candidate_score = float(val_selection_score)
            fallback_candidate_epoch = epoch
            fallback_candidate_components = val_selection_components
            torch.save(
                {
                    "representation_builder_state_dict": representation_builder.state_dict(),
                    "rd_proj_state_dict": rd_proj.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "labels": LOW_ALTITUDE_CLASSES,
                    "source_feature_stats": source_feature_stats,
                    "selection_eligible": bool(val_selection_eligible),
                    "selection_reason": selection_reason,
                },
                fallback_candidate_path,
            )
        if can_replace_best:
            best_val = float(val_selection_score)
            best_epoch = epoch
            best_selection_reason = selection_reason
            best_selection_components = val_selection_components
            torch.save(
                {
                    "representation_builder_state_dict": representation_builder.state_dict(),
                    "rd_proj_state_dict": rd_proj.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "labels": LOW_ALTITUDE_CLASSES,
                    "source_feature_stats": source_feature_stats,
                },
                best_path,
            )

    if best_epoch < 0 and history:
        if not fallback_candidate_path.exists():
            raise RuntimeError("no eligible or fallback expert1 checkpoint was saved")
        best_epoch = int(fallback_candidate_epoch)
        best_val = float(fallback_candidate_score)
        best_selection_components = dict(fallback_candidate_components)
        best_selection_reason = "fallback_v2_selection_checkpoint"
        best_path = fallback_candidate_path

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=True)
        representation_builder.load_state_dict(state["representation_builder_state_dict"])
        rd_proj.load_state_dict(state["rd_proj_state_dict"])
        model.load_state_dict(state["model_state_dict"])
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
        True,
        1.0,
        config.low_altitude_rd_teacher_warmup_epochs + 1,
        rd_class_weights,
        source_feature_stats,
        compute_aux_loss=config.compute_eval_aux_loss,
    )
    window_metadata = _load_window_metadata(config.data_root, config.index_name)
    parent_aggregation_metrics = _parent_window_aggregation_metrics(test_metrics, window_metadata)

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
            "route_accuracy": 1.0,
            "skipped_non_finite_batches": test_metrics["skipped_non_finite_batches"],
        },
        "window_evaluation": {
            "index_name": config.index_name,
            "is_window_index": bool(parent_aggregation_metrics),
            "parent_aggregation": parent_aggregation_metrics,
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
            "raw_tr_rd_accuracy": test_metrics["raw_tr_rd_accuracy"],
            "raw_tr_rd_macro_f1": test_metrics["raw_tr_rd_macro_f1"],
            "excel_track_macro_f1": test_metrics["excel_track_macro_f1"],
            "rd_source_balanced_accuracy_on_rd": test_metrics["rd_source_balanced_accuracy_on_rd"],
            "rd_source_balanced_macro_f1_on_rd": test_metrics["rd_source_balanced_macro_f1_on_rd"],
            "rd_min_source_macro_f1_on_rd": test_metrics["rd_min_source_macro_f1_on_rd"],
            "rd_unknown_clutter_recall_mean": test_metrics["rd_unknown_clutter_recall_mean"],
            "final_unknown_clutter_recall_mean": test_metrics["final_unknown_clutter_recall_mean"],
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
        "best_checkpoint": _rel(best_path),
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
        "speed_controls": {
            "enable_batch_parameter_rollback": bool(config.enable_batch_parameter_rollback),
            "compute_eval_aux_loss": bool(config.compute_eval_aux_loss),
            "max_train_batches_per_epoch": int(config.max_train_batches_per_epoch),
            "quick_val_max_batches": int(config.quick_val_max_batches),
            "full_val_interval_epochs": int(config.full_val_interval_epochs),
            "full_val_on_last_epoch": bool(config.full_val_on_last_epoch),
            "joint_logit_delta_scale": float(config.joint_logit_delta_scale),
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
        "max_sampler_weight": config.max_sampler_weight,
        "bird_sample_boost": config.bird_sample_boost,
        "freeze_track_encoder_epochs": config.freeze_track_encoder_epochs,
        "split_sizes": {
            "train": len(loaders.split.train_ids),
            "val": len(loaders.split.val_ids),
            "test": len(loaders.split.test_ids),
        },
        "rd_coverage": {
            "train": _rd_coverage(loaders.train_loader),
            "val": _rd_coverage(loaders.val_loader),
            "test": _rd_coverage(loaders.test_loader),
        },
        "best_epoch": best_epoch,
        "best_selection_score": best_val,
        "best_selection_reason": best_selection_reason,
        "best_selection_components": best_selection_components,
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
