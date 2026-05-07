from __future__ import annotations

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


def _group_accuracy_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_names: list[str],
    has_rd: list[bool],
) -> dict[str, object]:
    by_source = _source_accuracy_dict(logits, labels, source_names)
    source_summary = _source_summary_from_dict(by_source)
    rd_mask = torch.tensor(has_rd, dtype=torch.bool)
    return {
        "by_source_accuracy": by_source,
        "source_balanced_accuracy": source_summary["source_balanced_accuracy"],
        "min_source_accuracy": source_summary["min_source_accuracy"],
        "raw_tr_rd_accuracy": source_summary["raw_tr_rd_accuracy"],
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


def _fusion_gate_aux_loss(
    output,
    labels: torch.Tensor,
    mask: torch.Tensor,
    rare_indices: set[int],
    temperature: float,
    rare_gate_floor: float,
) -> torch.Tensor:
    if not bool(mask.any()):
        return labels.new_tensor(0.0, dtype=torch.float32)
    with torch.no_grad():
        traj_ce = torch.nn.functional.cross_entropy(output.traj_logits[mask], labels[mask], reduction="none")
        rd_ce = torch.nn.functional.cross_entropy(output.rd_logits[mask], labels[mask], reduction="none")
        rd_target = torch.sigmoid((traj_ce - rd_ce) / max(float(temperature), 1e-3))
        rare_mask = torch.zeros_like(labels[mask], dtype=torch.bool)
        for index in rare_indices:
            rare_mask = rare_mask | (labels[mask] == index)
        if bool(rare_mask.any()):
            rd_target = torch.where(
                rare_mask,
                torch.maximum(rd_target, torch.full_like(rd_target, float(rare_gate_floor))),
                rd_target,
            )
        target = torch.stack([1.0 - rd_target, rd_target], dim=-1)
    log_weights = torch.log(output.fusion_weights[mask].clamp_min(1e-6))
    return torch.nn.functional.kl_div(log_weights, target, reduction="batchmean")


def _rd_gate_target_loss(
    output,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    rare_floor: float,
) -> torch.Tensor:
    if not bool(mask.any()):
        return labels.new_tensor(0.0, dtype=torch.float32)
    with torch.no_grad():
        masked_labels = labels[mask]
        traj_pred = output.traj_logits[mask].argmax(dim=-1)
        rd_pred = output.rd_logits[mask].argmax(dim=-1)
        traj_confidence = torch.softmax(output.traj_logits[mask], dim=-1).max(dim=-1).values
        rd_confidence = torch.softmax(output.rd_logits[mask], dim=-1).max(dim=-1).values
        target = torch.sigmoid((rd_confidence - traj_confidence) / max(float(temperature), 1e-3))
        rd_correct = rd_pred == masked_labels
        traj_correct = traj_pred == masked_labels
        rd_rescue = rd_correct & ~traj_correct
        rd_harm = ~rd_correct & traj_correct
        target = torch.where(~rd_correct, torch.minimum(target, torch.full_like(target, 0.25)), target)
        target = torch.where(rd_rescue, torch.full_like(target, 0.80), target)
        target = torch.where(rd_harm, torch.minimum(target, torch.full_like(target, 0.15)), target)
        rare_mask = torch.zeros_like(labels[mask], dtype=torch.bool)
        rare_indices = {
            LOW_ALTITUDE_CLASSES.index("BirdTarget"),
            LOW_ALTITUDE_CLASSES.index("BalloonTarget"),
            LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
            LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
        }
        for index in rare_indices:
            rare_mask = rare_mask | (masked_labels == index)
        hard_pair_mask = torch.zeros_like(masked_labels, dtype=torch.bool)
        hard_pairs = {
            LOW_ALTITUDE_CLASSES.index("DroneTarget"),
            LOW_ALTITUDE_CLASSES.index("BirdTarget"),
            LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
            LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
        }
        for index in hard_pairs:
            hard_pair_mask = hard_pair_mask | (masked_labels == index)
        rd_confident_hard_pair = hard_pair_mask & rd_correct & (rd_confidence >= 0.45)
        if bool(rd_confident_hard_pair.any()):
            target = torch.where(
                rd_confident_hard_pair,
                torch.maximum(target, torch.full_like(target, 0.65)),
                target,
            )
        if bool(rare_mask.any()):
            target = torch.where(
                rare_mask & rd_correct,
                torch.maximum(target, torch.full_like(target, float(rare_floor))),
                target,
            )
        target = torch.stack([1.0 - target, target], dim=-1)
    log_weights = torch.log(output.fusion_weights[mask].clamp_min(1e-6))
    return torch.nn.functional.kl_div(log_weights, target, reduction="batchmean")


def _rd_class_gate_target_loss(
    output,
    labels: torch.Tensor,
    mask: torch.Tensor,
    traj_logits: torch.Tensor,
    rd_logits: torch.Tensor,
    rare_floor: float,
) -> torch.Tensor:
    if not bool(mask.any()):
        return labels.new_tensor(0.0, dtype=torch.float32)
    gate_logits = output.rd_class_gate_logits
    if gate_logits is None:
        return labels.new_tensor(0.0, dtype=torch.float32)
    with torch.no_grad():
        traj_pred = traj_logits[mask].argmax(dim=-1)
        rd_pred = rd_logits[mask].argmax(dim=-1)
        traj_conf = torch.softmax(traj_logits[mask], dim=-1).max(dim=-1).values
        rd_conf = torch.softmax(rd_logits[mask], dim=-1).max(dim=-1).values
        masked_labels = labels[mask]
        target = torch.full_like(gate_logits[mask], 0.05)
        row_indices = torch.arange(target.shape[0], dtype=torch.long, device=labels.device)
        true_cols = masked_labels.clamp(min=0, max=target.shape[-1] - 1)
        rd_rescue = (traj_pred != masked_labels) & (rd_pred == masked_labels)
        rd_harm = (traj_pred == masked_labels) & (rd_pred != masked_labels)
        if bool(rd_rescue.any()):
            target[row_indices[rd_rescue], true_cols[rd_rescue]] = 0.72
        if bool(rd_harm.any()):
            target[rd_harm] = 0.12
        hard_pairs = torch.tensor(
            [
                LOW_ALTITUDE_CLASSES.index("DroneTarget"),
                LOW_ALTITUDE_CLASSES.index("BirdTarget"),
                LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
                LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
            ],
            dtype=torch.long,
            device=labels.device,
        )
        hard_pair_mask = torch.isin(masked_labels, hard_pairs)
        hard_pair_high = hard_pair_mask & rd_rescue & (rd_conf >= 0.45)
        if bool(hard_pair_high.any()):
            target[row_indices[hard_pair_high], true_cols[hard_pair_high]] = torch.maximum(
                target[row_indices[hard_pair_high], true_cols[hard_pair_high]],
                torch.full_like(target[row_indices[hard_pair_high], true_cols[hard_pair_high]], 0.65),
            )
        uncertain = ((rd_conf < 0.40) | (traj_conf > rd_conf + 0.15)) & (~rd_rescue)
        if bool(uncertain.any()):
            target[uncertain] = torch.minimum(target[uncertain], torch.full_like(target[uncertain], float(rare_floor)))
    probs = torch.sigmoid(gate_logits[mask])
    return torch.nn.functional.mse_loss(probs, target)


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
        "by_source_accuracy": metrics.get("by_source_accuracy", {}),
        "traj_by_source_accuracy": metrics.get("traj_by_source_accuracy", {}),
        "rd_by_source_accuracy_on_rd": metrics.get("rd_by_source_accuracy_on_rd", {}),
        "rd_source_balanced_accuracy_on_rd": metrics.get("rd_source_balanced_accuracy_on_rd", 0.0),
        "rd_min_source_accuracy_on_rd": metrics.get("rd_min_source_accuracy_on_rd", 0.0),
        "raw_tr_rd_accuracy": metrics.get("raw_tr_rd_accuracy", 0.0),
        "source_balanced_accuracy": metrics.get("source_balanced_accuracy", 0.0),
        "min_source_accuracy": metrics.get("min_source_accuracy", 0.0),
        "with_rd_accuracy": metrics.get("with_rd_accuracy", 0.0),
        "without_rd_accuracy": metrics.get("without_rd_accuracy", 0.0),
        "selection_score": metrics.get("selection_score", 0.0),
        "selection_eligible": metrics.get("selection_eligible", False),
        "selection_components": metrics.get("selection_components", {}),
        "rd_activation_rate": metrics.get("rd_activation_rate", 0.0),
        "rd_activation_rate_on_rd": metrics.get("rd_activation_rate_on_rd", 0.0),
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
    accuracy = float(metrics.get("accuracy", 0.0))
    source_balanced_accuracy = float(metrics.get("source_balanced_accuracy", 0.0))
    min_source_accuracy = float(metrics.get("min_source_accuracy", 0.0))
    raw_source_accuracy = float(metrics.get("raw_tr_rd_accuracy", 0.0))
    rd_raw_source_accuracy = float(metrics.get("rd_raw_tr_rd_accuracy", 0.0))
    rd_source_balanced_accuracy = float(metrics.get("rd_source_balanced_accuracy_on_rd", 0.0))
    rd_min_source_accuracy = float(metrics.get("rd_min_source_accuracy_on_rd", 0.0))
    without_rd_accuracy = float(metrics.get("without_rd_accuracy", 0.0))
    rd_harm_rate = float(metrics.get("rd_harm_rate_on_rd", 0.0))
    rd_net_gain_rate = float(metrics.get("rd_net_gain_rate_on_rd", 0.0))
    final_harm_rate = float(metrics.get("final_harm_rate_on_rd", 0.0))
    final_net_gain_rate = float(metrics.get("final_net_gain_rate_on_rd", 0.0))
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
        0.55 * final_macro
        + 0.25 * accuracy
        + 0.35 * source_balanced_accuracy
        + 0.25 * min_source_accuracy
        + 0.20 * raw_source_accuracy
        + 0.20 * without_rd_accuracy
        + 0.80 * rd_macro
        + 0.18 * rd_source_balanced_accuracy
        + 0.12 * rd_min_source_accuracy
        + 0.15 * rd_raw_source_accuracy
        + 0.28 * drone_bird
        + 0.32 * unknown_clutter
        + 0.20 * min_rare_rd_recall
        + 0.18 * rd_net_gain_rate
        + 0.18 * final_net_gain_rate
        - 0.35 * rd_harm_rate
        - 0.20 * final_harm_rate
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
        eligible = eligible and rd_macro >= config.min_rd_macro_f1_for_best
        eligible = eligible and drone_bird >= config.min_drone_bird_rd_recall_for_best
        if enough_bird:
            eligible = eligible and min_drone_bird_recall >= config.min_drone_bird_single_rd_recall_for_best
        if enough_unknown and enough_clutter:
            eligible = eligible and unknown_clutter >= config.min_unknown_clutter_rd_recall_for_best
        if enough_unknown and enough_clutter and enough_bird:
            eligible = eligible and min_rare_rd_recall >= config.min_rare_rd_recall_for_best
        eligible = eligible and without_rd_accuracy >= config.min_without_rd_accuracy_for_best
        eligible = eligible and source_balanced_accuracy >= config.min_source_balanced_accuracy_for_best
        if enough_raw_source:
            eligible = eligible and raw_source_accuracy >= config.min_raw_source_accuracy_for_best
            eligible = eligible and min_source_accuracy >= config.min_source_accuracy_for_best
        if enough_rd_raw_source:
            eligible = eligible and rd_raw_source_accuracy >= config.min_rd_raw_source_accuracy_for_best
        eligible = eligible and rd_harm_rate <= config.max_rd_harm_rate_for_best
        eligible = eligible and rd_net_gain_rate >= config.min_rd_net_gain_rate_for_best
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
        "rd_raw_tr_rd_accuracy": rd_raw_source_accuracy,
        "rd_source_balanced_accuracy_on_rd": rd_source_balanced_accuracy,
        "rd_min_source_accuracy_on_rd": rd_min_source_accuracy,
        "without_rd_accuracy": without_rd_accuracy,
        "traj_macro_f1": traj_macro,
        "final_minus_traj_macro_f1": final_delta,
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
        "source_support": source_support,
        "rd_source_support_on_rd": rd_source_support,
        "rd_class_support_on_rd": rd_class_support,
        "raw_source_gate_active": enough_raw_source if stage2_enabled else False,
        "rd_raw_source_gate_active": enough_rd_raw_source if stage2_enabled else False,
        "unknown_clutter_gate_active": bool(enough_unknown and enough_clutter) if stage2_enabled else False,
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
    rd_class_weights: torch.Tensor,
    source_feature_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    anchor_traj_logits: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    representation_builder.train(mode=train)
    rd_proj.train(mode=train)
    model.train(mode=train)
    logits_all: list[torch.Tensor] = []
    traj_logits_all: list[torch.Tensor] = []
    rd_logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    sample_ids_all: list[str] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    has_rd_all: list[bool] = []
    rd_activated_all: list[bool] = []
    fusion_weights_all: list[torch.Tensor] = []
    loss_values: list[float] = []
    skipped_non_finite_batches = 0
    if anchor_traj_logits is None:
        anchor_traj_logits = {}

    for batch in loader:
        features = batch.features.to(device)
        if config.source_normalization and source_feature_stats is not None:
            features = _apply_source_feature_norm(features, batch.source_names, source_feature_stats)
        physical = batch.physical_features.to(device)
        rd_features = batch.rd_features.to(device)
        rd_patch_features = batch.rd_patch_features.to(device)
        rd_is_multimodal = batch.rd_is_multimodal.to(device)
        rd_padding_mask = batch.rd_padding_mask.to(device)
        rd_patch_padding_mask = batch.rd_patch_padding_mask.to(device)
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
                rd_patch_features=rd_patch_features,
                rd_context=rd_embedding,
                low_altitude_targets=labels if train else None,
                rd_is_multimodal=rd_is_multimodal,
                rd_padding_mask=rd_padding_mask,
                rd_patch_padding_mask=rd_patch_padding_mask,
                enable_reject=(config.enable_reject_in_train if train else config.enable_reject_in_eval) and stage2_enabled,
                force_rd_activation=bool(train and stage2_enabled),
                disable_rd_activation=bool(train and not stage2_enabled),
            )
            unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
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
            anchor_loss = torch.tensor(0.0, device=device)
            if train and config.low_altitude_traj_anchor_weight > 0.0:
                if not stage2_enabled:
                    anchor_traj_logits.update(
                        {
                            str(sample_id): logits_row
                            for sample_id, logits_row in zip(batch.sample_ids, output.traj_logits.detach().cpu())
                        }
                    )
                elif anchor_traj_logits:
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
            if stage2_enabled and has_rd_mask.any():
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
                main_loss = main_loss + config.low_altitude_rd_forced_fusion_weight * forced_fusion_loss
                gate_loss = _fusion_gate_aux_loss(
                    output,
                    labels,
                    has_rd_mask,
                    rare_indices={unknown_index, clutter_index},
                    temperature=config.low_altitude_rd_gate_temperature,
                    rare_gate_floor=config.low_altitude_rd_rare_gate_floor,
                )
                main_loss = main_loss + config.low_altitude_rd_gate_aux_weight * gate_loss
                rd_gate_target_loss = _rd_gate_target_loss(
                    output,
                    labels,
                    has_rd_mask,
                    temperature=max(config.low_altitude_rd_gate_temperature, 0.1),
                    rare_floor=config.low_altitude_rd_rare_gate_floor,
                )
                main_loss = main_loss + config.low_altitude_rd_gate_aux_weight * rd_gate_target_loss
                rd_class_gate_loss = _rd_class_gate_target_loss(
                    output,
                    labels,
                    has_rd_mask,
                    output.traj_logits,
                    output.rd_logits,
                    config.low_altitude_rd_rare_gate_floor,
                )
                main_loss = main_loss + config.low_altitude_rd_class_gate_aux_weight * rd_class_gate_loss
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
            else:
                mutual_loss = torch.tensor(0.0, device=device)
                contrastive_loss = torch.tensor(0.0, device=device)
            loss = (
                main_loss
                + (config.low_altitude_aux_weight * stage2_scale) * aux_loss
                + (config.low_altitude_traj_anchor_weight * stage2_scale) * anchor_loss
                + (config.low_altitude_mutual_weight * stage2_scale) * mutual_loss
                + (config.low_altitude_contrastive_weight * stage2_scale) * contrastive_loss
            )

            if train:
                prev_state = _clone_bundle_state_to_cpu(representation_builder, rd_proj, model)
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
                if not _parameters_are_finite_bundle(representation_builder, rd_proj, model):
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
        labels_all.append(labels.detach().cpu())
        sample_ids_all.extend(batch.sample_ids)
        source_names_all.extend(batch.source_names)
        label_names_all.extend(batch.label_names)
        has_rd_all.extend([bool(x) for x in batch.rd_is_multimodal.tolist()])
        rd_activated_all.extend([bool(x) for x in output.rd_activated.detach().cpu().tolist()])
        fusion_weights_all.append(output.fusion_weights.detach().cpu())

    if not logits_all:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "skipped_non_finite_batches": skipped_non_finite_batches,
            "has_valid_batches": False,
            "logits": torch.empty((0, len(LOW_ALTITUDE_CLASSES)), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
            "sample_ids": sample_ids_all,
            "source_names": source_names_all,
            "label_names": label_names_all,
            "has_rd": has_rd_all,
            "rd_activated": rd_activated_all,
            "rd_fusion_weight_mean": 0.0,
            "rd_fusion_weight_on_rd_mean": 0.0,
        }

    logits_cat = torch.cat(logits_all, dim=0)
    traj_logits_cat = torch.cat(traj_logits_all, dim=0)
    rd_logits_cat = torch.cat(rd_logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    metrics = compute_classification_metrics(logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    traj_metrics = compute_classification_metrics(traj_logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    rd_metrics = compute_classification_metrics(rd_logits_cat, labels_cat, num_classes=len(LOW_ALTITUDE_CLASSES))
    has_rd_tensor = torch.tensor(has_rd_all, dtype=torch.bool)
    rd_activated_tensor = torch.tensor(rd_activated_all, dtype=torch.bool)
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
    rd_teacher_effect = _rd_teacher_effect_metrics(
        logits_cat,
        traj_logits_cat,
        rd_logits_cat,
        labels_cat,
        has_rd_tensor,
        source_names_all,
    )
    group_metrics = _group_accuracy_metrics(logits_cat, labels_cat, source_names_all, has_rd_all)
    rd_activation_rate = float(rd_activated_tensor.float().mean().item()) if rd_activated_tensor.numel() else 0.0
    rd_activation_rate_on_rd = (
        float(rd_activated_tensor[has_rd_tensor].float().mean().item())
        if bool(has_rd_tensor.any())
        else 0.0
    )
    fusion_weights_cat = torch.cat(fusion_weights_all, dim=0) if fusion_weights_all else torch.empty((0, 2), dtype=torch.float32)
    rd_fusion_weight_mean = float(fusion_weights_cat[:, 1].mean().item()) if fusion_weights_cat.numel() else 0.0
    rd_fusion_weight_on_rd_mean = (
        float(fusion_weights_cat[has_rd_tensor, 1].mean().item())
        if bool(has_rd_tensor.any()) and fusion_weights_cat.numel()
        else 0.0
    )
    rd_drone_bird = diagnostics["rd_on_rd_drone_bird"]
    rd_unknown_clutter = float(diagnostics["rd_on_rd_unknown_clutter_recall_mean"])
    traj_by_source_accuracy = _source_accuracy_dict(traj_logits_cat, labels_cat, source_names_all)
    rd_by_source_accuracy_on_rd = _source_accuracy_dict(
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
    return {
        "loss": float(np.mean(loss_values)) if loss_values else float("nan"),
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "traj_accuracy": traj_metrics.accuracy,
        "traj_macro_f1": traj_metrics.macro_f1,
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
        **rd_teacher_effect,
        "traj_per_class_recall": diagnostics["traj_per_class_recall"],
        "traj_drone_bird": diagnostics["traj_drone_bird"],
        "final_per_class_recall": diagnostics["final_per_class_recall"],
        "final_drone_bird": diagnostics["final_drone_bird"],
        "final_unknown_clutter_recall_mean": diagnostics["final_unknown_clutter_recall_mean"],
        "rd_activation_rate": rd_activation_rate,
        "rd_activation_rate_on_rd": rd_activation_rate_on_rd,
        "rd_fusion_weight_mean": rd_fusion_weight_mean,
        "rd_fusion_weight_on_rd_mean": rd_fusion_weight_on_rd_mean,
        "by_source_accuracy": group_metrics["by_source_accuracy"],
        "traj_by_source_accuracy": traj_by_source_accuracy,
        "rd_by_source_accuracy_on_rd": rd_by_source_accuracy_on_rd,
        "source_support": source_support,
        "rd_source_support_on_rd": rd_source_support_on_rd,
        "rd_class_support_on_rd": rd_class_support_on_rd,
        "rd_source_balanced_accuracy_on_rd": rd_by_source_summary["source_balanced_accuracy"],
        "rd_min_source_accuracy_on_rd": rd_by_source_summary["min_source_accuracy"],
        "raw_tr_rd_accuracy": group_metrics["raw_tr_rd_accuracy"],
        "source_balanced_accuracy": group_metrics["source_balanced_accuracy"],
        "min_source_accuracy": group_metrics["min_source_accuracy"],
        "with_rd_accuracy": group_metrics["with_rd_accuracy"],
        "without_rd_accuracy": group_metrics["without_rd_accuracy"],
        "skipped_non_finite_batches": skipped_non_finite_batches,
        "has_valid_batches": True,
        "logits": logits_cat,
        "traj_logits": traj_logits_cat,
        "rd_logits": rd_logits_cat,
        "labels": labels_cat,
        "sample_ids": sample_ids_all,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "has_rd": has_rd_all,
        "rd_activated": rd_activated_all,
    }


def train_expert1_v1(config: Expert1TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    loaders = build_expert1_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        rd_sample_boost=config.rd_sample_boost,
        raw_source_boost=config.raw_source_boost,
        excel_source_boost=config.excel_source_boost,
        cq08_source_boost=config.cq08_source_boost,
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
        use_rd_patch_cnn=config.use_rd_patch_cnn,
        dropout=config.dropout,
        cosface_scale=config.cosface_scale,
        cosface_margin=config.cosface_margin,
        reject_threshold=config.reject_threshold,
        rd_activation_threshold=config.rd_activation_threshold,
        rd_bird_logit_scale=config.rd_bird_logit_scale,
        rd_unknown_logit_scale=config.rd_unknown_logit_scale,
        rd_clutter_logit_scale=config.rd_clutter_logit_scale,
        initial_rd_gate_bias=config.low_altitude_initial_rd_gate_bias,
        rd_class_activation_threshold=config.low_altitude_rd_class_activation_threshold,
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
        _set_track_encoder_trainable(representation_builder, epoch > config.freeze_track_encoder_epochs)
        train_metrics = _run_epoch(representation_builder, rd_proj, model, loaders.train_loader, criterion, optimizer, device, True, config, stage2_enabled, stage2_scale, rd_class_weights, source_feature_stats, anchor_traj_logits)
        if not train_metrics["has_valid_batches"]:
            break
        val_metrics = _run_epoch(representation_builder, rd_proj, model, loaders.val_loader, criterion, optimizer, device, False, config, stage2_enabled, stage2_scale, rd_class_weights, source_feature_stats)
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
                "train": _epoch_metric_summary(train_metrics),
                "val": _epoch_metric_summary(val_metrics),
            }
        )
        print(
            f"[*] epoch={epoch:02d} "
            f"stage2={'on' if stage2_enabled else 'off'} "
            f"scale={stage2_scale:.2f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
        can_replace_best = val_selection_eligible and val_selection_score > best_val
        selection_reason = "stage2_rd_composite" if stage2_enabled else "traj_composite"
        if val_selection_score > fallback_candidate_score:
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
        best_selection_reason = "fallback_no_rd_quality_checkpoint"
        best_path = fallback_candidate_path

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=True)
        representation_builder.load_state_dict(state["representation_builder_state_dict"])
        rd_proj.load_state_dict(state["rd_proj_state_dict"])
        model.load_state_dict(state["model_state_dict"])
    test_metrics = _run_epoch(representation_builder, rd_proj, model, loaders.test_loader, criterion, optimizer, device, False, config, True, 1.0, rd_class_weights, source_feature_stats)

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
            "rd_accuracy": test_metrics["rd_accuracy"],
            "rd_macro_f1": test_metrics["rd_macro_f1"],
            "rd_accuracy_on_rd": test_metrics["rd_accuracy_on_rd"],
            "rd_macro_f1_on_rd": test_metrics["rd_macro_f1_on_rd"],
            "rd_drone_bird_recall_mean": test_metrics["rd_drone_bird_recall_mean"],
            "rd_unknown_clutter_recall_mean": test_metrics["rd_unknown_clutter_recall_mean"],
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
            "raw_tr_rd_accuracy": test_metrics["raw_tr_rd_accuracy"],
            "source_balanced_accuracy": test_metrics["source_balanced_accuracy"],
            "min_source_accuracy": test_metrics["min_source_accuracy"],
            "final_unknown_clutter_recall_mean": test_metrics["final_unknown_clutter_recall_mean"],
            "route_accuracy": 1.0,
            "skipped_non_finite_batches": test_metrics["skipped_non_finite_batches"],
        },
        "label_names": LOW_ALTITUDE_CLASSES,
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
                "accuracy": 0.0,
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
            "rd": {
                "accuracy": test_metrics["rd_accuracy"],
                "macro_f1": test_metrics["rd_macro_f1"],
                "accuracy_on_rd": test_metrics["rd_accuracy_on_rd"],
                "macro_f1_on_rd": test_metrics["rd_macro_f1_on_rd"],
                "per_class_recall": test_metrics["rd_per_class_recall"],
                "per_class_recall_on_rd": test_metrics["rd_per_class_recall_on_rd"],
                "drone_bird": test_metrics["rd_drone_bird"],
                "unknown_clutter_recall_mean": test_metrics["rd_unknown_clutter_recall_mean"],
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
                "source_balanced_accuracy_on_rd": test_metrics["rd_source_balanced_accuracy_on_rd"],
                "min_source_accuracy_on_rd": test_metrics["rd_min_source_accuracy_on_rd"],
                "teacher_by_source": test_metrics["rd_teacher_by_source"],
            },
        },
        "device": device_info,
        "predictions": [
            {
                "sample_id": sid,
                "source_name": src,
                "label_name_true": true_label,
                "label_name_pred": LOW_ALTITUDE_CLASSES[int(test_metrics["logits"][idx].argmax().item())],
                "label_name_traj_pred": LOW_ALTITUDE_CLASSES[int(test_metrics["traj_logits"][idx].argmax().item())],
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
            correct = 0
            for idx in source_indices:
                pred_idx = int(test_metrics["logits"][idx].argmax().item())
                if LOW_ALTITUDE_CLASSES[pred_idx] == test_metrics["label_names"][idx]:
                    correct += 1
            evaluation_report["by_source"][source_name]["accuracy"] = correct / len(source_indices)

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
            "rd_rescue_count": test_metrics["rd_rescue_count"],
            "rd_harm_count": test_metrics["rd_harm_count"],
            "rd_rescue_rate_on_rd": test_metrics["rd_rescue_rate_on_rd"],
            "rd_harm_rate_on_rd": test_metrics["rd_harm_rate_on_rd"],
            "final_rescue_count": test_metrics["final_rescue_count"],
            "final_harm_count": test_metrics["final_harm_count"],
            "final_rescue_rate_on_rd": test_metrics["final_rescue_rate_on_rd"],
            "final_harm_rate_on_rd": test_metrics["final_harm_rate_on_rd"],
            "rd_raw_tr_rd_accuracy": test_metrics["rd_raw_tr_rd_accuracy"],
            "rd_teacher_by_source": test_metrics["rd_teacher_by_source"],
            "final_unknown_clutter_recall_mean": test_metrics["final_unknown_clutter_recall_mean"],
            "rd_activation_rate": test_metrics["rd_activation_rate"],
            "rd_activation_rate_on_rd": test_metrics["rd_activation_rate_on_rd"],
            "rd_fusion_weight_mean": test_metrics["rd_fusion_weight_mean"],
            "rd_fusion_weight_on_rd_mean": test_metrics["rd_fusion_weight_on_rd_mean"],
            "by_source_accuracy": test_metrics["by_source_accuracy"],
            "traj_by_source_accuracy": test_metrics["traj_by_source_accuracy"],
            "rd_by_source_accuracy_on_rd": test_metrics["rd_by_source_accuracy_on_rd"],
            "rd_source_balanced_accuracy_on_rd": test_metrics["rd_source_balanced_accuracy_on_rd"],
            "rd_min_source_accuracy_on_rd": test_metrics["rd_min_source_accuracy_on_rd"],
            "raw_tr_rd_accuracy": test_metrics["raw_tr_rd_accuracy"],
            "source_balanced_accuracy": test_metrics["source_balanced_accuracy"],
            "min_source_accuracy": test_metrics["min_source_accuracy"],
            "with_rd_accuracy": test_metrics["with_rd_accuracy"],
            "without_rd_accuracy": test_metrics["without_rd_accuracy"],
            "skipped_non_finite_batches": test_metrics["skipped_non_finite_batches"],
            "has_valid_batches": test_metrics["has_valid_batches"],
        },
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "taxonomy": {label: idx for idx, label in enumerate(LOW_ALTITUDE_CLASSES)},
        "trainable_parameters": _count_trainable_parameters(representation_builder) + _count_trainable_parameters(rd_proj) + _count_trainable_parameters(model),
        "class_weights": class_weights.detach().cpu().tolist(),
        "rd_class_weights": rd_class_weights.detach().cpu().tolist(),
        "unknown_loss_weight": config.unknown_loss_weight,
        "unknown_sample_boost": config.unknown_sample_boost,
        "rd_activation_threshold": config.rd_activation_threshold,
        "rd_activation_rate": test_metrics["rd_activation_rate"],
        "rd_activation_rate_on_rd": test_metrics["rd_activation_rate_on_rd"],
        "low_altitude_rd_class_activation_threshold": config.low_altitude_rd_class_activation_threshold,
        "low_altitude_rd_class_gate_aux_weight": config.low_altitude_rd_class_gate_aux_weight,
        "source_normalization": bool(config.source_normalization),
        "source_feature_stats_saved": bool(source_feature_stats is not None),
        "prefer_stage2_checkpoint": bool(config.prefer_stage2_checkpoint),
        "sequence_encoder_lr_multiplier": config.sequence_encoder_lr_multiplier,
        "pretrained_track_checkpoint": config.pretrained_track_checkpoint,
        "pretrained_track_info": pretrained_track_info,
        "use_rd_patch_cnn": bool(config.use_rd_patch_cnn),
        "low_altitude_traj_aux_ratio": config.low_altitude_traj_aux_ratio,
        "low_altitude_rd_aux_ratio": config.low_altitude_rd_aux_ratio,
        "low_altitude_rd_hard_aux_weight": config.low_altitude_rd_hard_aux_weight,
        "low_altitude_rd_focal_gamma": config.low_altitude_rd_focal_gamma,
        "low_altitude_rd_binary_aux_weight": config.low_altitude_rd_binary_aux_weight,
        "low_altitude_rd_pairwise_aux_weight": config.low_altitude_rd_pairwise_aux_weight,
        "low_altitude_rd_unknown_pairwise_aux_weight": config.low_altitude_rd_unknown_pairwise_aux_weight,
        "low_altitude_rd_clutter_pairwise_aux_weight": config.low_altitude_rd_clutter_pairwise_aux_weight,
        "low_altitude_rd_unknown_clutter_pairwise_aux_weight": config.low_altitude_rd_unknown_clutter_pairwise_aux_weight,
        "low_altitude_rd_rare_margin_aux_weight": config.low_altitude_rd_rare_margin_aux_weight,
        "low_altitude_rd_rare_margin": config.low_altitude_rd_rare_margin,
        "low_altitude_rd_forced_fusion_weight": config.low_altitude_rd_forced_fusion_weight,
        "low_altitude_rd_gate_aux_weight": config.low_altitude_rd_gate_aux_weight,
        "low_altitude_rd_gate_temperature": config.low_altitude_rd_gate_temperature,
        "low_altitude_rd_rare_gate_floor": config.low_altitude_rd_rare_gate_floor,
        "low_altitude_initial_rd_gate_bias": config.low_altitude_initial_rd_gate_bias,
        "low_altitude_traj_anchor_weight": config.low_altitude_traj_anchor_weight,
        "low_altitude_traj_anchor_start_epoch": config.low_altitude_traj_anchor_start_epoch,
        "rd_bird_logit_scale": config.rd_bird_logit_scale,
        "rd_unknown_logit_scale": config.rd_unknown_logit_scale,
        "rd_clutter_logit_scale": config.rd_clutter_logit_scale,
        "rd_bird_positive_weight": config.rd_bird_positive_weight,
        "rd_unknown_positive_weight": config.rd_unknown_positive_weight,
        "rd_clutter_positive_weight": config.rd_clutter_positive_weight,
        "low_altitude_mutual_exclude_rare_classes": bool(config.low_altitude_mutual_exclude_rare_classes),
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
        "rd_quality_gate": {
            "min_rd_macro_f1_for_best": config.min_rd_macro_f1_for_best,
            "min_drone_bird_rd_recall_for_best": config.min_drone_bird_rd_recall_for_best,
            "min_drone_bird_single_rd_recall_for_best": config.min_drone_bird_single_rd_recall_for_best,
            "min_unknown_clutter_rd_recall_for_best": config.min_unknown_clutter_rd_recall_for_best,
            "min_rare_rd_recall_for_best": config.min_rare_rd_recall_for_best,
            "min_without_rd_accuracy_for_best": config.min_without_rd_accuracy_for_best,
            "min_source_accuracy_for_best": config.min_source_accuracy_for_best,
            "min_source_balanced_accuracy_for_best": config.min_source_balanced_accuracy_for_best,
            "min_raw_source_accuracy_for_best": config.min_raw_source_accuracy_for_best,
            "min_rd_raw_source_accuracy_for_best": config.min_rd_raw_source_accuracy_for_best,
            "min_final_minus_traj_macro_f1_for_best": config.min_final_minus_traj_macro_f1_for_best,
            "min_source_support_for_hard_gate": config.min_source_support_for_hard_gate,
            "min_rd_source_support_for_hard_gate": config.min_rd_source_support_for_hard_gate,
            "min_rd_class_support_for_hard_gate": config.min_rd_class_support_for_hard_gate,
            "max_rd_harm_rate_for_best": config.max_rd_harm_rate_for_best,
            "min_rd_net_gain_rate_for_best": config.min_rd_net_gain_rate_for_best,
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
