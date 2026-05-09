from __future__ import annotations

import json
import random
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .device import resolve_cuda_device
from .forwardlayer.dataloaders import build_dataloaders
from .forwardlayer.dataset import TrackDataset
from .forwardlayer.labels import DEFAULT_VOCAB
from .experts import LOW_ALTITUDE_CLASSES
from .router import ROUTE_NAMES
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .system import RadarClassificationSystem, RerouteConfig
from .system_config import SystemTrainConfig
from .system_metrics import compute_confusion_matrix, compute_per_class_metrics, compute_system_metrics
from .physical import PHYSICAL_FEATURE_COLUMNS


SAFETY_CRITICAL_PAIRS = [
    ("DroneTarget", "BirdTarget"),
    ("BirdTarget", "DroneTarget"),
    ("BirdTarget", "BalloonTarget"),
    ("DroneTarget", "UnknownTarget"),
    ("AircraftTarget", "DroneTarget"),
    ("AircraftTarget", "BirdTarget"),
    ("PeopleTarget", "CarTarget"),
    ("PeopleTarget", "UnknownTarget"),
    ("CarTarget", "PeopleTarget"),
    ("ShipTarget", "CarTarget"),
    ("UnknownTarget", "ClutterTarget"),
    ("ClutterTarget", "UnknownTarget"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> tuple[torch.device, dict[str, object]]:
    return resolve_cuda_device(requested_device)


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _set_track_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.representation_builder.sequence_encoder.parameters():
        param.requires_grad = trainable


def _set_ground_feature_stack_trainable(model: nn.Module, trainable: bool) -> None:
    modules = [
        model.ground_expert.deep_proj,
        model.ground_expert.physical_input_norm,
        model.ground_expert.physical_proj,
        model.ground_expert.gate,
    ]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = trainable


def _build_optimizer(model: nn.Module, config: SystemTrainConfig) -> torch.optim.Optimizer:
    sequence_encoder_param_ids = {
        id(param)
        for param in model.representation_builder.sequence_encoder.parameters()
    }
    ground_backbone_param_ids = {
        id(param)
        for module in (
            model.ground_expert.deep_proj,
            model.ground_expert.physical_input_norm,
            model.ground_expert.physical_proj,
            model.ground_expert.gate,
        )
        for param in module.parameters()
    }
    ground_classifier_param_ids = {
        id(param)
        for param in model.ground_expert.classifier.parameters()
    }
    groundness_head_param_ids = {
        id(param)
        for param in model.ground_expert.groundness_head.parameters()
    }
    sequence_encoder_params = [
        param
        for param in model.representation_builder.sequence_encoder.parameters()
        if param.requires_grad
    ]
    ground_backbone_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) in ground_backbone_param_ids
    ]
    ground_classifier_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) in ground_classifier_param_ids
    ]
    other_params = [
        param
        for param in model.parameters()
        if param.requires_grad
        and id(param) not in sequence_encoder_param_ids
        and id(param) not in ground_backbone_param_ids
        and id(param) not in ground_classifier_param_ids
        and id(param) not in groundness_head_param_ids
    ]
    param_groups: list[dict[str, object]] = []
    if sequence_encoder_params:
        param_groups.append(
            {
                "params": sequence_encoder_params,
                "lr": config.learning_rate * config.sequence_encoder_lr_multiplier,
                "weight_decay": config.weight_decay,
            }
        )
    if ground_backbone_params:
        param_groups.append(
            {
                "params": ground_backbone_params,
                "lr": config.learning_rate * config.ground_expert_lr_multiplier,
                "weight_decay": config.weight_decay,
                "group_name": "ground_backbone",
            }
        )
    if ground_classifier_params:
        param_groups.append(
            {
                "params": ground_classifier_params,
                "lr": config.learning_rate * config.ground_expert_lr_multiplier,
                "weight_decay": config.weight_decay,
                "group_name": "ground_classifier",
            }
        )
    groundness_head_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) in groundness_head_param_ids
    ]
    if groundness_head_params:
        param_groups.append(
            {
                "params": groundness_head_params,
                "lr": config.learning_rate * config.ground_expert_lr_multiplier,
                "weight_decay": config.weight_decay,
                "group_name": "groundness_head",
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "group_name": "other",
            }
        )
    return torch.optim.AdamW(param_groups)


def _set_ground_expert_learning_rates(optimizer: torch.optim.Optimizer, config: SystemTrainConfig, epoch: int) -> None:
    backbone_lr = 0.0 if epoch <= config.freeze_ground_feature_epochs else config.learning_rate * config.ground_expert_lr_multiplier
    classifier_lr = config.learning_rate * config.ground_expert_lr_multiplier
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        if name == "ground_backbone":
            group["lr"] = backbone_lr
        elif name in {"ground_classifier", "groundness_head"}:
            group["lr"] = classifier_lr


def _set_router_only_warmup(model: nn.Module, router_only: bool) -> None:
    if router_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.router.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True


def _set_expert_warmup_trainable(model: nn.Module, freeze_router: bool, freeze_track_encoder: bool) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if freeze_router:
        for param in model.router.parameters():
            param.requires_grad = False
    if freeze_track_encoder:
        for param in model.representation_builder.sequence_encoder.parameters():
            param.requires_grad = False


def _set_low_altitude_only_expert_warmup(model: nn.Module, freeze_router: bool, freeze_track_encoder: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.low_altitude_expert.parameters():
        param.requires_grad = True
    for param in model.rd_proj.parameters():
        param.requires_grad = True
    for param in model.representation_builder.physical_proj.parameters():
        param.requires_grad = True
    for param in model.representation_builder.physical_norm.parameters():
        param.requires_grad = True
    for param in model.representation_builder.fusion.parameters():
        param.requires_grad = True
    if not freeze_track_encoder:
        for param in model.representation_builder.sequence_encoder.parameters():
            param.requires_grad = True
    if not freeze_router:
        for param in model.router.parameters():
            param.requires_grad = True


def _set_router_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.router.parameters():
        param.requires_grad = trainable


def _load_pretrained_track_encoder(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = {
        key[len("encoder."):] if key.startswith("encoder.") else key: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"no encoder.* weights found in pretrain checkpoint: {checkpoint_path}")
    missing, unexpected = model.representation_builder.sequence_encoder.load_state_dict(encoder_state, strict=False)
    return {
        "loaded_keys": len(encoder_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _load_submodule_checkpoint(submodule: nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    current_state = submodule.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
    }
    skipped_shape_mismatch = {
        key: {
            "checkpoint_shape": list(value.shape),
            "model_shape": list(current_state[key].shape),
        }
        for key, value in state_dict.items()
        if key in current_state and tuple(value.shape) != tuple(current_state[key].shape)
    }
    unexpected_keys = [key for key in state_dict if key not in current_state]
    missing, unexpected = submodule.load_state_dict(compatible_state, strict=False)
    return {
        "loaded_keys": len(compatible_state),
        "missing_keys": list(missing),
        "unexpected_keys": sorted(set(unexpected_keys + list(unexpected))),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def _load_compatible_state_dict(submodule: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    current_state = submodule.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
    }
    skipped_shape_mismatch = {
        key: {
            "checkpoint_shape": list(value.shape),
            "model_shape": list(current_state[key].shape),
        }
        for key, value in state_dict.items()
        if key in current_state and tuple(value.shape) != tuple(current_state[key].shape)
    }
    unexpected_keys = [key for key in state_dict if key not in current_state]
    missing, unexpected = submodule.load_state_dict(compatible_state, strict=False)
    return {
        "loaded_keys": len(compatible_state),
        "missing_keys": list(missing),
        "unexpected_keys": sorted(set(unexpected_keys + list(unexpected))),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def _load_expert_bundle(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    expert_name: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    info: dict[str, object] = {"expert_name": expert_name}
    info["load_track_encoder"] = False
    if checkpoint.get("train_representation_builder"):
        raise ValueError(
            f"{expert_name} checkpoint was trained with a private representation builder; "
            "retrain it with --no-train-representation-builder and a shared representation checkpoint."
        )
    if "representation_builder_state_dict" in checkpoint:
        info["representation_builder"] = {
            "loaded_keys": 0,
            "skipped": "auxiliary expert encoders are not loaded into the shared system encoder",
        }
    model_key = checkpoint.get("model_state_dict", {})
    if expert_name == "expert2_air":
        missing, unexpected = model.conventional_air_expert.load_state_dict(model_key, strict=False)
        source_thresholds = {
            str(key): float(value)
            for key, value in dict(checkpoint.get("source_thresholds", {})).items()
        }
        if source_thresholds and hasattr(model.conventional_air_expert, "set_source_thresholds"):
            model.conventional_air_expert.set_source_thresholds(source_thresholds)
        info["source_thresholds"] = source_thresholds
    elif expert_name == "expert4_ground":
        missing, unexpected = model.ground_expert.load_state_dict(model_key, strict=False)
    else:
        raise ValueError(f"unsupported expert bundle: {expert_name}")
    info["expert_module"] = {
        "loaded_keys": len(model_key),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return info


def _load_expert1_bundle(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    info: dict[str, object] = {"expert_name": "expert1_v1"}
    if "representation_builder_state_dict" in checkpoint:
        info["representation_builder"] = _load_compatible_state_dict(
            model.representation_builder,
            checkpoint["representation_builder_state_dict"],
        )
    if "rd_proj_state_dict" in checkpoint:
        info["rd_proj"] = _load_compatible_state_dict(model.rd_proj, checkpoint["rd_proj_state_dict"])
    model_key = checkpoint.get("model_state_dict", {})
    info["expert_module"] = _load_compatible_state_dict(model.low_altitude_expert, model_key)
    return info


def _build_class_weights(loaders) -> torch.Tensor:
    counts = torch.zeros((len(DEFAULT_VOCAB.labels),), dtype=torch.float32)
    for batch in loaders.train_loader:
        labels = batch.labels
        bincount = torch.bincount(labels, minlength=len(DEFAULT_VOCAB.labels)).to(torch.float32)
        counts += bincount
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.25, max=2.0)
    return weights


def _build_route_weights(loaders) -> torch.Tensor:
    counts = torch.zeros((len(ROUTE_NAMES),), dtype=torch.float32)
    for batch in loaders.train_loader:
        bincount = torch.bincount(batch.route_labels, minlength=len(ROUTE_NAMES)).to(torch.float32)
        counts += bincount
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=3.0)


def _build_low_altitude_weights(loaders) -> torch.Tensor:
    counts = torch.zeros((len(LOW_ALTITUDE_CLASSES),), dtype=torch.float32)
    for batch in loaders.train_loader:
        local_labels, low_mask = _low_altitude_local_targets(batch.labels)
        if not bool(low_mask.any()):
            continue
        bincount = torch.bincount(
            local_labels[low_mask],
            minlength=len(LOW_ALTITUDE_CLASSES),
        ).to(torch.float32)
        counts += bincount
    counts = torch.clamp(counts, min=1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=2.0)


def _build_groundness_weights(loaders) -> torch.Tensor:
    ground_indices = {
        DEFAULT_VOCAB.label_to_index["PeopleTarget"],
        DEFAULT_VOCAB.label_to_index["CarTarget"],
        DEFAULT_VOCAB.label_to_index["ShipTarget"],
    }
    counts = torch.ones((2,), dtype=torch.float32)
    for batch in loaders.train_loader:
        labels = batch.labels
        ground_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label_idx in ground_indices:
            ground_mask |= labels == label_idx
        counts[1] += ground_mask.sum().to(torch.float32)
        counts[0] += (~ground_mask).sum().to(torch.float32)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=4.0)


def _build_groundness_weights_from_labels(labels: torch.Tensor) -> torch.Tensor:
    ground_targets = _groundness_targets(labels)
    counts = torch.tensor(
        [
            float((~ground_targets.bool()).sum().item()),
            float(ground_targets.bool().sum().item()),
        ],
        dtype=torch.float32,
        device=labels.device,
    ).clamp_min(1.0)
    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=0.25, max=4.0)


def _groundness_targets(labels: torch.Tensor) -> torch.Tensor:
    ground_indices = {
        DEFAULT_VOCAB.label_to_index["PeopleTarget"],
        DEFAULT_VOCAB.label_to_index["CarTarget"],
        DEFAULT_VOCAB.label_to_index["ShipTarget"],
    }
    target = torch.zeros_like(labels, dtype=torch.long)
    for label_idx in ground_indices:
        target = torch.where(labels == label_idx, torch.ones_like(target), target)
    return target


def _groundness_separation_loss(
    groundness_probability: torch.Tensor,
    labels: torch.Tensor,
    positive_target: float,
    negative_target: float,
) -> torch.Tensor:
    ground_target = _groundness_targets(labels).bool()
    if not bool(ground_target.any()) and not bool((~ground_target).any()):
        return groundness_probability.new_tensor(0.0)
    loss = groundness_probability.new_tensor(0.0)
    if bool(ground_target.any()):
        loss = loss + torch.nn.functional.relu(float(positive_target) - groundness_probability[ground_target]).mean()
    if bool((~ground_target).any()):
        loss = loss + torch.nn.functional.relu(groundness_probability[~ground_target] - float(negative_target)).mean()
    return loss


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


def _source_feature_stats_from_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, dict[str, torch.Tensor]] | None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stats = checkpoint.get("source_feature_stats")
    if not stats:
        return None
    return stats


def _merge_source_feature_stats(*stats_items: dict[str, dict[str, torch.Tensor]] | None) -> dict[str, dict[str, torch.Tensor]]:
    merged: dict[str, dict[str, torch.Tensor]] = {}
    for stats in stats_items:
        if not stats:
            continue
        for source_name, source_stats in stats.items():
            merged[str(source_name)] = {
                "center": source_stats["center"].detach().cpu()
                if isinstance(source_stats["center"], torch.Tensor)
                else torch.tensor(source_stats["center"], dtype=torch.float32),
                "scale": source_stats["scale"].detach().cpu()
                if isinstance(source_stats["scale"], torch.Tensor)
                else torch.tensor(source_stats["scale"], dtype=torch.float32),
            }
    return merged


def _parameters_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def _gradients_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def _rd_coverage_from_ids(dataset_frame, sample_ids: list[str]) -> dict[str, int]:
    subset = dataset_frame[dataset_frame["sample_id"].astype(str).isin({str(x) for x in sample_ids})].copy()
    if "rd_tensor_path" in subset.columns:
        has_rd = subset["rd_tensor_path"].fillna("").astype(str).str.strip()
        has_rd = has_rd.map(lambda x: x.lower() not in {"", "nan", "none"})
        return {
            "with_rd": int(has_rd.sum()),
            "without_rd": int((~has_rd).sum()),
        }
    if "rd_summary_csv_path" not in subset.columns:
        return {"with_rd": 0, "without_rd": int(len(subset))}
    has_rd = subset["rd_summary_csv_path"].fillna("").astype(str).str.strip()
    has_rd = has_rd.map(lambda x: x.lower() not in {"", "nan", "none"})
    return {
        "with_rd": int(has_rd.sum()),
        "without_rd": int((~has_rd).sum()),
    }


def _route_name_for_label(label_name: str) -> str:
    if label_name in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}:
        return "low_altitude_suspect"
    if label_name in {"AircraftTarget"}:
        return "conventional_air"
    if label_name in {"PeopleTarget", "CarTarget", "ShipTarget"}:
        return "ground_target"
    return "high_altitude_filter"


def _system_reroute_config(config: SystemTrainConfig) -> RerouteConfig:
    return RerouteConfig(
        max_passes=int(config.reroute_max_passes),
        penalty=float(config.reroute_penalty),
        max_total_penalty=float(config.reroute_max_total_penalty),
        low_altitude_conf_threshold=float(config.reroute_low_altitude_conf_threshold),
        conventional_air_conf_threshold=float(config.reroute_conventional_air_conf_threshold),
        ground_conf_threshold=float(config.reroute_ground_conf_threshold),
    )


def _project_to_global_logits(
    batch_labels: torch.Tensor,
    system_output,
    num_classes: int,
    device: torch.device,
    config: SystemTrainConfig,
    return_diagnostics: bool = False,
    source_names: list[str] | None = None,
    apply_source_aircraft_policy: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = torch.full((batch_labels.shape[0], num_classes), fill_value=-6.0, device=device)
    global_index = DEFAULT_VOCAB.label_to_index
    fusion_mode = config.fusion_mode
    if fusion_mode not in {"ungated", "hard_route", "soft_log_weighted", "soft_top2", "reroute", "calibrated_candidates"}:
        raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
    if fusion_mode == "reroute":
        raise ValueError("fusion_mode='reroute' is inference-only; use _project_to_global_logits_reroute()")
    route_weights = system_output.routing.route_weights
    route_log_prior = float(config.calibrated_route_prior_weight) * torch.log(route_weights.clamp_min(1e-6))
    if fusion_mode == "calibrated_candidates":
        low_gate = route_log_prior[:, ROUTE_NAMES.index("low_altitude_suspect")].unsqueeze(-1)
        conventional_gate = route_log_prior[:, ROUTE_NAMES.index("conventional_air")].unsqueeze(-1)
        high_gate = route_log_prior[:, ROUTE_NAMES.index("high_altitude_filter")].unsqueeze(-1)
        ground_gate = route_log_prior[:, ROUTE_NAMES.index("ground_target")].unsqueeze(-1)
    elif fusion_mode == "ungated":
        low_gate = torch.zeros((batch_labels.shape[0], 1), dtype=torch.float32, device=device)
        conventional_gate = torch.zeros_like(low_gate)
        high_gate = torch.zeros_like(low_gate)
        ground_gate = torch.zeros_like(low_gate)
    elif fusion_mode in {"soft_log_weighted", "soft_top2"}:
        if fusion_mode == "soft_top2":
            route_weights = system_output.routing.route_weights
            topk_count = min(2, route_weights.shape[-1])
            topk_indices = route_weights.topk(k=topk_count, dim=-1).indices
            topk_mask = torch.zeros_like(route_weights, dtype=torch.bool)
            topk_mask.scatter_(1, topk_indices, True)
            route_weights = torch.where(topk_mask, route_weights, torch.zeros_like(route_weights))
            route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            route_log_weights = torch.log(route_weights.clamp_min(1e-6))
        else:
            route_log_weights = torch.log(system_output.routing.route_weights.clamp_min(1e-6))
        low_gate = route_log_weights[:, ROUTE_NAMES.index("low_altitude_suspect")].unsqueeze(-1)
        conventional_gate = route_log_weights[:, ROUTE_NAMES.index("conventional_air")].unsqueeze(-1)
        high_gate = route_log_weights[:, ROUTE_NAMES.index("high_altitude_filter")].unsqueeze(-1)
        ground_gate = route_log_weights[:, ROUTE_NAMES.index("ground_target")].unsqueeze(-1)
    else:
        route_indices = system_output.routing.route_indices
        low_gate = torch.where(
            route_indices == ROUTE_NAMES.index("low_altitude_suspect"),
            torch.zeros_like(route_indices, dtype=torch.float32),
            torch.full_like(route_indices, fill_value=-30, dtype=torch.float32),
        ).to(device).unsqueeze(-1)
        conventional_gate = torch.where(
            route_indices == ROUTE_NAMES.index("conventional_air"),
            torch.zeros_like(route_indices, dtype=torch.float32),
            torch.full_like(route_indices, fill_value=-30, dtype=torch.float32),
        ).to(device).unsqueeze(-1)
        high_gate = torch.where(
            route_indices == ROUTE_NAMES.index("high_altitude_filter"),
            torch.zeros_like(route_indices, dtype=torch.float32),
            torch.full_like(route_indices, fill_value=-30, dtype=torch.float32),
        ).to(device).unsqueeze(-1)
        ground_gate = torch.where(
            route_indices == ROUTE_NAMES.index("ground_target"),
            torch.zeros_like(route_indices, dtype=torch.float32),
            torch.full_like(route_indices, fill_value=-30, dtype=torch.float32),
        ).to(device).unsqueeze(-1)
    conventional_route_weight = route_weights[:, ROUTE_NAMES.index("conventional_air")]

    if fusion_mode == "calibrated_candidates":
        low_confidence = system_output.low_altitude_detail.traj_cosine.max(dim=-1).values
        low_membership = torch.clamp(low_confidence - float(config.low_altitude_reject_threshold), min=-1.0, max=1.0)
        low = (
            torch.log_softmax(system_output.low_altitude_logits, dim=-1)
            / max(float(config.calibrated_low_temperature), 1e-6)
            + low_gate
            + float(config.calibrated_low_membership_beta) * low_membership.unsqueeze(-1)
        )
    else:
        low = system_output.low_altitude_logits + low_gate
    logits[:, global_index["DroneTarget"]] = low[:, 0]
    logits[:, global_index["BirdTarget"]] = low[:, 1]
    logits[:, global_index["BalloonTarget"]] = low[:, 2]
    logits[:, global_index["ClutterTarget"]] = low[:, 3]
    logits[:, global_index["UnknownTarget"]] = low[:, 4]

    conv_logits = system_output.conventional_air_logits
    aircraft_column = 1 if conv_logits.shape[1] > 1 else 0
    non_aircraft_column = 0
    conv_positive = conv_logits[:, aircraft_column]
    if conv_logits.shape[1] > 1:
        aircraft_margin = conv_logits[:, aircraft_column] - conv_logits[:, non_aircraft_column]
    else:
        aircraft_margin = conv_positive
    auxiliary = getattr(system_output.conventional_air_detail, "auxiliary", None) or {}
    aircraft_similarity = auxiliary.get("aircraft_similarity")
    if aircraft_similarity is None:
        aircraft_similarity = aircraft_margin.detach()
    aircraft_similarity = aircraft_similarity.to(device=device, dtype=conv_logits.dtype)
    aircraft_threshold = auxiliary.get("aircraft_threshold")
    if aircraft_threshold is None:
        aircraft_threshold = torch.zeros_like(aircraft_similarity)
    aircraft_threshold = aircraft_threshold.to(device=device, dtype=conv_logits.dtype)
    aircraft_confidence = aircraft_similarity - aircraft_threshold
    source_tau_low = torch.full_like(aircraft_confidence, float(config.aircraft_confidence_tau_low))
    source_override_mask = torch.zeros_like(aircraft_confidence, dtype=torch.bool)
    source_overrides = dict(config.aircraft_source_confidence_tau_low or {})
    if apply_source_aircraft_policy and source_names and source_overrides:
        override_values = [
            float(source_overrides.get(str(source_name), float(config.aircraft_confidence_tau_low)))
            for source_name in source_names
        ]
        source_tau_low = torch.tensor(override_values, device=device, dtype=conv_logits.dtype)
        source_override_mask = torch.tensor(
            [str(source_name) in source_overrides for source_name in source_names],
            device=device,
            dtype=torch.bool,
        )
    conventional_gate_flat = conventional_gate.squeeze(-1)
    adjusted_conventional_gate = conventional_gate_flat
    high_conf = aircraft_confidence >= float(config.aircraft_confidence_tau_high)
    mid_conf = (
        (aircraft_confidence >= source_tau_low)
        & ~high_conf
    )
    source_reject = source_override_mask & (aircraft_confidence < source_tau_low)
    if fusion_mode == "calibrated_candidates":
        aircraft_log_prob = torch.log_softmax(conv_logits, dim=-1)[:, aircraft_column]
        floor_weight = max(float(config.aircraft_conventional_gate_floor), 1e-6)
        conventional_gate_floor = float(np.log(floor_weight))
        floor_tensor = torch.full_like(adjusted_conventional_gate, conventional_gate_floor)
        adjusted_conventional_gate = torch.where(
            high_conf,
            torch.zeros_like(adjusted_conventional_gate),
            torch.where(mid_conf, torch.maximum(adjusted_conventional_gate, floor_tensor), adjusted_conventional_gate),
        )
        aircraft_membership = torch.clamp(aircraft_confidence, min=-1.0, max=1.0)
        aircraft_score = (
            aircraft_log_prob / max(float(config.aircraft_temperature), 1e-6)
            + adjusted_conventional_gate
            + float(config.calibrated_aircraft_membership_beta) * aircraft_membership
        )
        calibrated_aircraft_allowed = (
            (aircraft_confidence >= float(config.calibrated_aircraft_min_confidence))
            & (conventional_route_weight >= float(config.calibrated_aircraft_min_route_weight))
        )
        aircraft_score = torch.where(
            calibrated_aircraft_allowed,
            aircraft_score,
            torch.full_like(aircraft_score, float(config.calibrated_aircraft_suppress_logit)),
        )
        if apply_source_aircraft_policy and config.aircraft_source_suppress_below_tau:
            aircraft_score = torch.where(
                source_reject,
                torch.full_like(aircraft_score, float(config.aircraft_source_suppress_logit)),
                aircraft_score,
            )
    elif config.aircraft_protection_enabled:
        floor_weight = max(float(config.aircraft_conventional_gate_floor), 1e-6)
        conventional_gate_floor = float(np.log(floor_weight))
        floor_tensor = torch.full_like(adjusted_conventional_gate, conventional_gate_floor)
        adjusted_conventional_gate = torch.where(
            high_conf,
            torch.zeros_like(adjusted_conventional_gate),
            torch.where(mid_conf, torch.maximum(adjusted_conventional_gate, floor_tensor), adjusted_conventional_gate),
        )
        aircraft_score = aircraft_margin / max(float(config.aircraft_temperature), 1e-6) + adjusted_conventional_gate
        if apply_source_aircraft_policy and config.aircraft_source_suppress_below_tau:
            aircraft_score = torch.where(
                source_reject,
                torch.full_like(aircraft_score, float(config.aircraft_source_suppress_logit)),
                aircraft_score,
            )
    else:
        aircraft_score = conv_positive + conventional_gate_flat

    high_logits = system_output.high_altitude_logits
    high_aircraft_column = 1 if high_logits.shape[1] > 1 else 0
    high_positive = high_logits[:, high_aircraft_column] > high_logits[:, 0]
    high_route_selected = system_output.routing.route_indices == ROUTE_NAMES.index("high_altitude_filter")
    high_allowed = high_positive & high_route_selected & system_output.routing.override_mask
    high_aircraft_score = torch.log_softmax(high_logits, dim=-1)[:, high_aircraft_column] + high_gate.squeeze(-1)
    high_aircraft_score = torch.where(
        high_allowed,
        high_aircraft_score,
        torch.full_like(high_aircraft_score, -6.0),
    )
    logits[:, global_index["AircraftTarget"]] = torch.maximum(aircraft_score, high_aircraft_score)

    ground_logits = system_output.ground_logits
    ground_auxiliary = getattr(system_output.ground_detail, "auxiliary", None) or {}
    groundness_probability = ground_auxiliary.get("groundness_probability")
    if groundness_probability is None:
        groundness_probability = torch.ones((batch_labels.shape[0],), device=device, dtype=ground_logits.dtype)
    groundness_probability = groundness_probability.to(device=device, dtype=ground_logits.dtype)
    groundness_threshold = ground_auxiliary.get("groundness_threshold")
    if groundness_threshold is None:
        groundness_threshold = torch.full_like(groundness_probability, 0.5)
    groundness_threshold = groundness_threshold.to(device=device, dtype=ground_logits.dtype)
    groundness_confidence = groundness_probability - groundness_threshold
    ground_gate_flat = ground_gate.squeeze(-1)
    adjusted_ground_gate = ground_gate_flat
    ground_high_conf = groundness_confidence >= float(config.groundness_confidence_tau_high)
    ground_mid_conf = (
        (groundness_confidence >= float(config.groundness_confidence_tau_low))
        & ~ground_high_conf
    )
    if fusion_mode == "calibrated_candidates":
        floor_weight = max(float(config.ground_route_gate_floor), 1e-6)
        ground_gate_floor = float(np.log(floor_weight))
        ground_floor_tensor = torch.full_like(adjusted_ground_gate, ground_gate_floor)
        adjusted_ground_gate = torch.where(
            ground_high_conf,
            torch.zeros_like(adjusted_ground_gate),
            torch.where(
                ground_mid_conf,
                torch.maximum(adjusted_ground_gate, ground_floor_tensor),
                adjusted_ground_gate - float(config.calibrated_ground_suppression),
            ),
        )
        ground_membership = torch.clamp(groundness_confidence, min=-1.0, max=1.0)
        ground = (
            torch.log_softmax(ground_logits, dim=-1) / max(float(config.groundness_temperature), 1e-6)
            + adjusted_ground_gate.unsqueeze(-1)
            + float(config.calibrated_ground_membership_beta) * ground_membership.unsqueeze(-1)
        )
    elif config.groundness_protection_enabled:
        floor_weight = max(float(config.ground_route_gate_floor), 1e-6)
        ground_gate_floor = float(np.log(floor_weight))
        ground_floor_tensor = torch.full_like(adjusted_ground_gate, ground_gate_floor)
        adjusted_ground_gate = torch.where(
            ground_high_conf,
            torch.zeros_like(adjusted_ground_gate),
            torch.where(
                ground_mid_conf,
                torch.maximum(adjusted_ground_gate, ground_floor_tensor),
                adjusted_ground_gate - 1.5,
            ),
        )
        ground = ground_logits / max(float(config.groundness_temperature), 1e-6) + adjusted_ground_gate.unsqueeze(-1)
    else:
        ground = ground_logits + ground_gate
    logits[:, global_index["PeopleTarget"]] = ground[:, 0]
    logits[:, global_index["CarTarget"]] = ground[:, 1]
    logits[:, global_index["ShipTarget"]] = ground[:, 2]
    if not return_diagnostics:
        return logits
    diagnostics = {
        "aircraft_similarity": aircraft_similarity.detach(),
        "aircraft_threshold": aircraft_threshold.detach(),
        "aircraft_margin": aircraft_margin.detach(),
        "aircraft_confidence": aircraft_confidence.detach(),
        "aircraft_source_tau_low": source_tau_low.detach(),
        "aircraft_source_override": source_override_mask.detach(),
        "aircraft_source_reject": source_reject.detach(),
        "aircraft_score": logits[:, global_index["AircraftTarget"]].detach(),
        "conventional_route_weight": conventional_route_weight.detach(),
        "conventional_gate": conventional_gate_flat.detach(),
        "adjusted_conventional_gate": adjusted_conventional_gate.detach(),
        "expert2_positive": (aircraft_confidence >= 0.0).detach(),
        "aircraft_high_confidence": high_conf.detach(),
        "aircraft_mid_confidence": mid_conf.detach(),
        "calibrated_aircraft_allowed": (
            (
                (aircraft_confidence >= float(config.calibrated_aircraft_min_confidence))
                & (conventional_route_weight >= float(config.calibrated_aircraft_min_route_weight))
            )
            if fusion_mode == "calibrated_candidates"
            else torch.zeros_like(aircraft_confidence, dtype=torch.bool)
        ).detach(),
        "high_altitude_rule_positive": high_positive.detach(),
        "high_altitude_allowed": high_allowed.detach(),
        "groundness_probability": groundness_probability.detach(),
        "groundness_threshold": groundness_threshold.detach(),
        "groundness_confidence": groundness_confidence.detach(),
        "ground_route_weight": route_weights[:, ROUTE_NAMES.index("ground_target")].detach(),
        "ground_gate": ground_gate_flat.detach(),
        "adjusted_ground_gate": adjusted_ground_gate.detach(),
        "ground_high_confidence": ground_high_conf.detach(),
        "ground_mid_confidence": ground_mid_conf.detach(),
    }
    return logits, diagnostics


def _project_to_global_logits_reroute(
    batch_labels: torch.Tensor,
    system_output,
    num_classes: int,
    device: torch.device,
    config: SystemTrainConfig,
    source_names: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if system_output.reroute_logits is None or system_output.reroute_diagnostics is None:
        raise RuntimeError(
            "reroute logits are now produced by RadarClassificationSystem.forward("
            "inference_mode='reroute'); trainer-side reroute post-processing is disabled"
        )
    _, base_diagnostics = _project_to_global_logits(
        batch_labels,
        system_output,
        num_classes,
        device,
        replace(config, fusion_mode="soft_log_weighted"),
        return_diagnostics=True,
        source_names=source_names,
        apply_source_aircraft_policy=True,
    )
    diagnostics = {
        **base_diagnostics,
        **{
            key: value.detach()
            for key, value in system_output.reroute_diagnostics.items()
        },
    }
    return system_output.reroute_logits, diagnostics


def _low_altitude_local_targets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local = torch.full_like(labels, fill_value=-1)
    low_mask = torch.zeros_like(labels, dtype=torch.bool)
    mapping = {
        DEFAULT_VOCAB.label_to_index["DroneTarget"]: 0,
        DEFAULT_VOCAB.label_to_index["BirdTarget"]: 1,
        DEFAULT_VOCAB.label_to_index["BalloonTarget"]: 2,
        DEFAULT_VOCAB.label_to_index["ClutterTarget"]: 3,
        DEFAULT_VOCAB.label_to_index["UnknownTarget"]: 4,
    }
    for global_idx, local_idx in mapping.items():
        mask = labels == global_idx
        if mask.any():
            local[mask] = local_idx
            low_mask |= mask
    return local, low_mask


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


def _stage_loss_scales(stage_name: str, config: SystemTrainConfig, stage_epoch: int | None = None) -> dict[str, float]:
    if stage_name == "warmup":
        return {
            "route": 1.0,
            "low_aux": 0.0,
            "low_mutual": 0.0,
            "low_contrastive": 0.0,
            "consistency": 0.0,
        }
    if stage_name == "expert_warmup":
        return {
            "route": config.stage_expert_warmup_route_loss_scale,
            "low_aux": config.stage_expert_warmup_low_altitude_aux_scale,
            "low_mutual": config.stage_expert_warmup_low_altitude_mutual_scale,
            "low_contrastive": config.stage_expert_warmup_low_altitude_contrastive_scale,
            "consistency": config.stage_expert_warmup_consistency_scale,
        }
    if stage_name == "joint":
        ramp = 1.0
        if stage_epoch is not None and config.joint_loss_ramp_epochs > 0:
            ramp = min(1.0, max(0.0, stage_epoch / float(config.joint_loss_ramp_epochs)))
        return {
            "route": 1.0,
            "low_aux": ramp,
            "low_mutual": ramp,
            "low_contrastive": ramp,
            "consistency": ramp,
        }
    ramp = 1.0
    if stage_name == "finetune" and stage_epoch is not None and config.joint_loss_ramp_epochs > 0:
        ramp = min(1.0, max(0.0, stage_epoch / float(config.joint_loss_ramp_epochs)))
    return {
        "route": config.stage_finetune_route_loss_scale,
        "low_aux": config.stage_finetune_low_altitude_aux_scale * ramp,
        "low_mutual": config.stage_finetune_low_altitude_mutual_scale * ramp,
        "low_contrastive": config.stage_finetune_low_altitude_contrastive_scale * ramp,
        "consistency": config.stage_finetune_consistency_scale * ramp,
    }


def _run_epoch(
    model,
    loader,
    class_criterion,
    route_criterion,
    low_criterion,
    optimizer,
    device: torch.device,
    train: bool,
    skip_non_finite_batch: bool,
    config: SystemTrainConfig,
    stage_name: str,
    stage_epoch: int | None = None,
    source_feature_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    groundness_class_weights: torch.Tensor | None = None,
    detach_ground_features: bool = False,
) -> dict[str, object]:
    model.train(mode=train)
    class_logits_all: list[torch.Tensor] = []
    class_logits_without_aircraft_protection_all: list[torch.Tensor] = []
    class_labels_all: list[torch.Tensor] = []
    route_logits_all: list[torch.Tensor] = []
    route_labels_all: list[torch.Tensor] = []
    aircraft_diagnostics_all: dict[str, list[torch.Tensor]] = {}
    reroute_diagnostics_all: dict[str, list[torch.Tensor]] = {}
    expert_output_all: dict[str, list[torch.Tensor]] = {}
    loss_values: list[float] = []
    sample_ids_all: list[str] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    coarse_label_names_all: list[str] = []
    fine_label_names_all: list[str] = []
    route_names_all: list[str] = []
    skipped_non_finite_batches = 0
    runtime_config = config
    if train and config.fusion_mode == "calibrated_candidates":
        runtime_config = replace(config, fusion_mode="soft_log_weighted")
    unprotected_config = replace(runtime_config, aircraft_protection_enabled=False)
    if unprotected_config.fusion_mode == "reroute":
        unprotected_config = replace(unprotected_config, fusion_mode="soft_log_weighted")

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
        labels = batch.labels.to(device)
        route_labels = batch.route_labels.to(device)
        padding_mask = batch.padding_mask.to(device)
        with torch.set_grad_enabled(train):
            low_targets, low_mask = _low_altitude_local_targets(labels)
            output = model(
                features,
                physical,
                rd_features=rd_features,
                rd_patch_features=rd_patch_features,
                low_altitude_targets=low_targets if train else None,
                rd_is_multimodal=rd_is_multimodal,
                padding_mask=padding_mask,
                rd_padding_mask=rd_padding_mask,
                rd_patch_padding_mask=rd_patch_padding_mask,
                source_names=batch.source_names,
                enable_low_altitude_reject=(
                    config.enable_low_altitude_reject_in_train
                    if train
                    else config.enable_low_altitude_reject_in_eval
                ),
                detach_ground_features=train and detach_ground_features,
                inference_mode="reroute" if (not train and runtime_config.fusion_mode == "reroute") else "parallel",
                reroute_config=_system_reroute_config(runtime_config)
                if (not train and runtime_config.fusion_mode == "reroute")
                else None,
                num_global_classes=len(DEFAULT_VOCAB.labels),
            )
            if not train and config.fusion_mode == "reroute":
                class_logits, reroute_diagnostics = _project_to_global_logits_reroute(
                    labels,
                    output,
                    len(DEFAULT_VOCAB.labels),
                    device,
                    runtime_config,
                    source_names=batch.source_names,
                )
                _, aircraft_diagnostics = _project_to_global_logits(
                    labels,
                    output,
                    len(DEFAULT_VOCAB.labels),
                    device,
                    replace(runtime_config, fusion_mode="soft_log_weighted"),
                    return_diagnostics=True,
                    source_names=batch.source_names,
                    apply_source_aircraft_policy=True,
                )
            else:
                class_logits_result = _project_to_global_logits(
                    labels,
                    output,
                    len(DEFAULT_VOCAB.labels),
                    device,
                    runtime_config,
                    return_diagnostics=True,
                    source_names=batch.source_names,
                    apply_source_aircraft_policy=not train,
                )
                class_logits, aircraft_diagnostics = class_logits_result
                reroute_diagnostics = {}
            with torch.no_grad():
                class_logits_without_aircraft_protection = _project_to_global_logits(
                    labels,
                    output,
                    len(DEFAULT_VOCAB.labels),
                    device,
                    unprotected_config,
                    return_diagnostics=False,
                    source_names=batch.source_names,
                    apply_source_aircraft_policy=False,
                )
            class_loss = class_criterion(class_logits, labels)
            route_loss = route_criterion(output.routing.route_logits, route_labels)
            route_coarse_loss = torch.nn.functional.cross_entropy(output.routing.coarse_logits, labels)
            scales = _stage_loss_scales(stage_name, config, stage_epoch=stage_epoch)
            route_total_loss = route_loss + config.route_coarse_loss_weight * route_coarse_loss
            loss = class_loss + (config.route_loss_weight * scales["route"]) * route_total_loss

            low_aux_loss = torch.tensor(0.0, device=device)
            low_mutual_loss = torch.tensor(0.0, device=device)
            low_contrastive_loss = torch.tensor(0.0, device=device)
            consistency_loss = torch.tensor(0.0, device=device)
            groundness_loss = torch.tensor(0.0, device=device)
            groundness_separation_loss = torch.tensor(0.0, device=device)
            ground_auxiliary = getattr(output.ground_detail, "auxiliary", None) or {}
            groundness_logits = ground_auxiliary.get("groundness_logits")
            groundness_probability = ground_auxiliary.get("groundness_probability")
            if groundness_logits is not None and config.system_groundness_loss_weight > 0.0:
                groundness_labels = _groundness_targets(labels)
                if groundness_class_weights is None:
                    groundness_weights = torch.tensor(
                        [1.0, 1.0],
                        dtype=groundness_logits.dtype,
                        device=device,
                    )
                else:
                    groundness_weights = groundness_class_weights.to(device=device, dtype=groundness_logits.dtype)
                groundness_loss = torch.nn.functional.cross_entropy(
                    groundness_logits,
                    groundness_labels,
                    weight=groundness_weights,
                )
            if groundness_probability is not None and config.system_groundness_separation_loss_weight > 0.0:
                groundness_separation_loss = _groundness_separation_loss(
                    groundness_probability,
                    labels,
                    positive_target=config.system_groundness_positive_target,
                    negative_target=config.system_groundness_negative_target,
                )
            if scales["low_aux"] > 0.0 and low_mask.any():
                low_local_targets = low_targets[low_mask]
                traj_logits = output.low_altitude_detail.traj_logits[low_mask]
                low_aux_loss = low_criterion(traj_logits, low_local_targets)
                low_rd_mask = low_mask & rd_is_multimodal.bool()
                if low_rd_mask.any():
                    rd_logits = output.low_altitude_detail.rd_logits[low_rd_mask]
                    low_aux_loss = 0.5 * (
                        low_aux_loss
                        + low_criterion(rd_logits, low_targets[low_rd_mask])
                    )
                    low_mutual_loss = _symmetric_kl(
                        output.low_altitude_detail.traj_logits[low_rd_mask],
                        rd_logits,
                    )
                    low_embeddings = torch.cat(
                        [
                            output.low_altitude_detail.traj_features[low_rd_mask],
                            output.low_altitude_detail.rd_features[low_rd_mask],
                        ],
                        dim=-1,
                    )
                    low_contrastive_loss = _supervised_contrastive_loss(
                        low_embeddings,
                        low_targets[low_rd_mask],
                        temperature=config.low_altitude_temperature,
                    )
                    traj_embed = output.low_altitude_detail.traj_features[low_rd_mask]
                    rd_embed = output.low_altitude_detail.rd_features[low_rd_mask]
                    consistency_loss = torch.nn.functional.mse_loss(traj_embed, rd_embed)

            loss = (
                loss
                + (config.low_altitude_aux_weight * scales["low_aux"]) * low_aux_loss
                + (config.low_altitude_mutual_weight * scales["low_mutual"]) * low_mutual_loss
                + (config.low_altitude_contrastive_weight * scales["low_contrastive"]) * low_contrastive_loss
                + (config.consistency_loss_weight * scales["consistency"]) * consistency_loss
                + config.system_groundness_loss_weight * groundness_loss
                + config.system_groundness_separation_loss_weight * groundness_separation_loss
            )
            if train:
                if not torch.isfinite(loss):
                    if skip_non_finite_batch:
                        skipped_non_finite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise RuntimeError("non-finite loss detected during system training")
                optimizer.zero_grad()
                loss.backward()
                if not _gradients_are_finite(model):
                    if skip_non_finite_batch:
                        skipped_non_finite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise RuntimeError("non-finite gradient detected during system training")
                sequence_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.representation_builder.sequence_encoder.parameters(),
                        max_norm=config.sequence_encoder_grad_clip_norm,
                    ).detach().cpu().item()
                )
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=config.global_grad_clip_norm,
                    ).detach().cpu().item()
                )
                if not np.isfinite(sequence_grad_norm) or not np.isfinite(grad_norm) or not _gradients_are_finite(model):
                    if skip_non_finite_batch:
                        skipped_non_finite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise RuntimeError("non-finite gradient norm detected during system training")
                optimizer.step()
                if not _parameters_are_finite(model):
                    raise RuntimeError("model parameters became non-finite during system training")
            else:
                grad_norm = 0.0
                sequence_grad_norm = 0.0
        loss_values.append(float(loss.detach().cpu().item()))
        if "class_loss_values" not in locals():
            class_loss_values = []
            route_loss_values = []
            route_coarse_loss_values = []
            groundness_loss_values = []
            groundness_separation_loss_values = []
            low_aux_loss_values = []
            low_mutual_loss_values = []
            low_contrastive_loss_values = []
            consistency_loss_values = []
            grad_norm_values = []
            sequence_grad_norm_values = []
            low_aux_scale_values = []
            low_mutual_scale_values = []
            low_contrastive_scale_values = []
            consistency_scale_values = []
            class_logit_min_values = []
            class_logit_max_values = []
            route_logit_min_values = []
            route_logit_max_values = []
            route_network_logit_min_values = []
            route_network_logit_max_values = []
            route_override_rate_values = []
        class_loss_values.append(float(class_loss.detach().cpu().item()))
        route_loss_values.append(float(route_loss.detach().cpu().item()))
        route_coarse_loss_values.append(float(route_coarse_loss.detach().cpu().item()))
        groundness_loss_values.append(float(groundness_loss.detach().cpu().item()))
        groundness_separation_loss_values.append(float(groundness_separation_loss.detach().cpu().item()))
        low_aux_loss_values.append(float(low_aux_loss.detach().cpu().item()))
        low_mutual_loss_values.append(float(low_mutual_loss.detach().cpu().item()))
        low_contrastive_loss_values.append(float(low_contrastive_loss.detach().cpu().item()))
        consistency_loss_values.append(float(consistency_loss.detach().cpu().item()))
        grad_norm_values.append(float(grad_norm))
        sequence_grad_norm_values.append(float(sequence_grad_norm))
        low_aux_scale_values.append(float(scales["low_aux"]))
        low_mutual_scale_values.append(float(scales["low_mutual"]))
        low_contrastive_scale_values.append(float(scales["low_contrastive"]))
        consistency_scale_values.append(float(scales["consistency"]))
        class_logit_min_values.append(float(class_logits.detach().amin().cpu().item()))
        class_logit_max_values.append(float(class_logits.detach().amax().cpu().item()))
        route_logit_min_values.append(float(output.routing.route_logits.detach().amin().cpu().item()))
        route_logit_max_values.append(float(output.routing.route_logits.detach().amax().cpu().item()))
        route_network_logit_min_values.append(float(output.routing.network_logits.detach().amin().cpu().item()))
        route_network_logit_max_values.append(float(output.routing.network_logits.detach().amax().cpu().item()))
        route_override_rate_values.append(float(output.routing.override_mask.detach().float().mean().cpu().item()))
        class_logits_all.append(class_logits.detach().cpu())
        class_logits_without_aircraft_protection_all.append(
            class_logits_without_aircraft_protection.detach().cpu()
        )
        class_labels_all.append(labels.detach().cpu())
        route_logits_all.append(output.routing.route_logits.detach().cpu())
        route_labels_all.append(route_labels.detach().cpu())
        for diag_name, diag_value in aircraft_diagnostics.items():
            aircraft_diagnostics_all.setdefault(diag_name, []).append(diag_value.detach().cpu())
        for diag_name, diag_value in reroute_diagnostics.items():
            reroute_diagnostics_all.setdefault(diag_name, []).append(diag_value.detach().cpu())
        expert_output_all.setdefault("expert2_logits", []).append(output.conventional_air_logits.detach().cpu())
        expert_output_all.setdefault("expert3_logits", []).append(output.high_altitude_logits.detach().cpu())
        expert_output_all.setdefault("expert4_logits", []).append(output.ground_logits.detach().cpu())
        expert_output_all.setdefault("expert1_traj_logits", []).append(output.low_altitude_detail.traj_logits.detach().cpu())
        expert_output_all.setdefault("expert1_rd_logits", []).append(output.low_altitude_detail.rd_logits.detach().cpu())
        expert_output_all.setdefault("expert1_traj_confidence", []).append(
            output.low_altitude_detail.traj_cosine.max(dim=-1).values.detach().cpu()
        )
        expert_output_all.setdefault("expert1_traj_entropy", []).append(
            torch.distributions.Categorical(logits=output.low_altitude_detail.traj_logits.detach()).entropy().cpu()
        )
        expert_output_all.setdefault("expert1_fusion_weights", []).append(output.low_altitude_detail.fusion_weights.detach().cpu())
        expert_output_all.setdefault("rd_is_multimodal", []).append(rd_is_multimodal.detach().cpu())
        sample_ids_all.extend(batch.sample_ids)
        source_names_all.extend(batch.source_names)
        label_names_all.extend(batch.label_names)
        coarse_label_names_all.extend(batch.coarse_label_names)
        fine_label_names_all.extend(batch.fine_label_names)
        route_names_all.extend(batch.route_names)

    if not class_logits_all:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "route_accuracy": 0.0,
            "route_macro_recall": 0.0,
            "class_logits": torch.empty((0, len(DEFAULT_VOCAB.labels)), dtype=torch.float32),
            "class_labels": torch.empty((0,), dtype=torch.long),
            "route_logits": torch.empty((0, len(ROUTE_NAMES)), dtype=torch.float32),
            "route_labels": torch.empty((0,), dtype=torch.long),
            "class_logits_without_aircraft_protection": torch.empty((0, len(DEFAULT_VOCAB.labels)), dtype=torch.float32),
            "aircraft_diagnostics": {},
            "reroute_diagnostics": {},
            "expert_outputs": {},
            "sample_ids": [],
            "source_names": [],
            "label_names": [],
            "coarse_label_names": [],
            "fine_label_names": [],
            "route_names": [],
            "skipped_non_finite_batches": skipped_non_finite_batches,
            "has_valid_batches": False,
            "loss_components": {},
            "critical_metrics": {},
        }

    class_logits_cat = torch.cat(class_logits_all, dim=0)
    class_logits_without_aircraft_protection_cat = torch.cat(
        class_logits_without_aircraft_protection_all,
        dim=0,
    )
    class_labels_cat = torch.cat(class_labels_all, dim=0)
    route_logits_cat = torch.cat(route_logits_all, dim=0)
    route_labels_cat = torch.cat(route_labels_all, dim=0)
    aircraft_diagnostics_cat = {
        name: torch.cat(values, dim=0)
        for name, values in aircraft_diagnostics_all.items()
    }
    reroute_diagnostics_cat = {
        name: torch.cat(values, dim=0)
        for name, values in reroute_diagnostics_all.items()
    }
    expert_outputs_cat = {
        name: torch.cat(values, dim=0)
        for name, values in expert_output_all.items()
    }
    metrics = compute_system_metrics(
        class_logits=class_logits_cat,
        class_labels=class_labels_cat,
        route_logits=route_logits_cat,
        route_labels=route_labels_cat,
        num_classes=len(DEFAULT_VOCAB.labels),
    )
    source_report = _source_report(class_logits_cat, class_labels_cat, source_names_all)
    critical_metrics = _critical_metric_summary(
        class_logits_cat,
        class_labels_cat,
        route_logits_cat,
        route_labels_cat,
        aircraft_diagnostics=aircraft_diagnostics_cat,
        reroute_diagnostics=reroute_diagnostics_cat,
        class_logits_without_aircraft_protection=class_logits_without_aircraft_protection_cat,
        source_report=source_report,
        source_names=source_names_all,
    )
    return {
        "loss": float(sum(loss_values) / len(loss_values)),
        "accuracy": metrics.classification.accuracy,
        "macro_precision": metrics.classification.macro_precision,
        "macro_recall": metrics.classification.macro_recall,
        "macro_f1": metrics.classification.macro_f1,
        "route_accuracy": metrics.route_accuracy,
        "route_macro_recall": metrics.route_macro_recall,
        "class_logits": class_logits_cat,
        "class_logits_without_aircraft_protection": class_logits_without_aircraft_protection_cat,
        "class_labels": class_labels_cat,
        "route_logits": route_logits_cat,
        "route_labels": route_labels_cat,
        "aircraft_diagnostics": aircraft_diagnostics_cat,
        "reroute_diagnostics": reroute_diagnostics_cat,
        "expert_outputs": expert_outputs_cat,
        "sample_ids": sample_ids_all,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "coarse_label_names": coarse_label_names_all,
        "fine_label_names": fine_label_names_all,
        "route_names": route_names_all,
        "source_report": source_report,
        "skipped_non_finite_batches": skipped_non_finite_batches,
        "has_valid_batches": True,
        "critical_metrics": critical_metrics,
        "loss_components": {
            "class_loss": float(sum(class_loss_values) / len(class_loss_values)),
            "route_loss": float(sum(route_loss_values) / len(route_loss_values)),
            "route_coarse_loss": float(sum(route_coarse_loss_values) / len(route_coarse_loss_values)),
            "groundness_loss": float(sum(groundness_loss_values) / len(groundness_loss_values)),
            "groundness_separation_loss": float(sum(groundness_separation_loss_values) / len(groundness_separation_loss_values)),
            "low_aux_loss": float(sum(low_aux_loss_values) / len(low_aux_loss_values)),
            "low_mutual_loss": float(sum(low_mutual_loss_values) / len(low_mutual_loss_values)),
            "low_contrastive_loss": float(sum(low_contrastive_loss_values) / len(low_contrastive_loss_values)),
            "consistency_loss": float(sum(consistency_loss_values) / len(consistency_loss_values)),
            "grad_norm": float(sum(grad_norm_values) / len(grad_norm_values)),
            "sequence_grad_norm": float(sum(sequence_grad_norm_values) / len(sequence_grad_norm_values)),
            "low_aux_scale": float(sum(low_aux_scale_values) / len(low_aux_scale_values)),
            "low_mutual_scale": float(sum(low_mutual_scale_values) / len(low_mutual_scale_values)),
            "low_contrastive_scale": float(sum(low_contrastive_scale_values) / len(low_contrastive_scale_values)),
            "consistency_scale": float(sum(consistency_scale_values) / len(consistency_scale_values)),
            "class_logit_min": float(sum(class_logit_min_values) / len(class_logit_min_values)),
            "class_logit_max": float(sum(class_logit_max_values) / len(class_logit_max_values)),
            "route_logit_min": float(sum(route_logit_min_values) / len(route_logit_min_values)),
            "route_logit_max": float(sum(route_logit_max_values) / len(route_logit_max_values)),
            "route_network_logit_min": float(sum(route_network_logit_min_values) / len(route_network_logit_min_values)),
            "route_network_logit_max": float(sum(route_network_logit_max_values) / len(route_network_logit_max_values)),
            "route_override_rate": float(sum(route_override_rate_values) / len(route_override_rate_values)),
        },
    }


def _metric_summary(epoch_metrics: dict[str, object]) -> dict[str, float]:
    summary = {
        "loss": float(epoch_metrics["loss"]),
        "accuracy": float(epoch_metrics["accuracy"]),
        "macro_precision": float(epoch_metrics["macro_precision"]),
        "macro_recall": float(epoch_metrics["macro_recall"]),
        "macro_f1": float(epoch_metrics["macro_f1"]),
        "route_accuracy": float(epoch_metrics["route_accuracy"]),
        "route_macro_recall": float(epoch_metrics.get("route_macro_recall", 0.0)),
        "selection_score": float(epoch_metrics.get("selection_score", 0.0)),
        "skipped_non_finite_batches": int(epoch_metrics.get("skipped_non_finite_batches", 0)),
        "has_valid_batches": bool(epoch_metrics.get("has_valid_batches", True)),
    }
    if epoch_metrics.get("critical_metrics"):
        summary["critical_metrics"] = epoch_metrics["critical_metrics"]
    if epoch_metrics.get("selection_components"):
        summary["selection_components"] = epoch_metrics["selection_components"]
    if epoch_metrics.get("loss_components"):
        summary["loss_components"] = epoch_metrics["loss_components"]
    return summary


def _route_report(route_logits: torch.Tensor, route_labels: torch.Tensor) -> dict[str, object]:
    route_preds = route_logits.argmax(dim=-1)
    matrix = torch.zeros((len(ROUTE_NAMES), len(ROUTE_NAMES)), dtype=torch.long)
    for true_idx, pred_idx in zip(route_labels.tolist(), route_preds.tolist()):
        matrix[int(true_idx), int(pred_idx)] += 1
    per_route = []
    recalls = []
    for idx, route_name in enumerate(ROUTE_NAMES):
        support = int((route_labels == idx).sum().item())
        correct = int(((route_labels == idx) & (route_preds == idx)).sum().item())
        recall = correct / support if support > 0 else 0.0
        if support > 0:
            recalls.append(recall)
        per_route.append(
            {
                "route_name": route_name,
                "support": support,
                "correct": correct,
                "recall": recall,
            }
        )
    macro_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    collapsed_route = None
    if route_preds.numel() > 0:
        pred_counts = torch.bincount(route_preds, minlength=len(ROUTE_NAMES))
        if int(pred_counts.max().item()) == int(route_preds.numel()) and int((route_labels.unique()).numel()) > 1:
            collapsed_route = ROUTE_NAMES[int(pred_counts.argmax().item())]
    return {
        "route_names": ROUTE_NAMES,
        "confusion_matrix": matrix.tolist(),
        "per_route": per_route,
        "macro_recall": macro_recall,
        "collapsed_route": collapsed_route,
    }


def _source_report(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    source_names: list[str],
) -> dict[str, object]:
    preds = class_logits.argmax(dim=-1)
    grouped: dict[str, list[int]] = {}
    for idx, source_name in enumerate(source_names):
        grouped.setdefault(source_name, []).append(idx)
    report: dict[str, object] = {}
    for source_name, indices in grouped.items():
        true = class_labels[indices]
        pred = preds[indices]
        accuracy = float((true == pred).float().mean().item()) if len(indices) > 0 else 0.0
        per_label: dict[str, dict[str, float | int]] = {}
        for label_idx, label_name in enumerate(DEFAULT_VOCAB.labels):
            label_mask = true == label_idx
            support = int(label_mask.sum().item())
            pred_mask = pred == label_idx
            tp = int((pred_mask & label_mask).sum().item())
            fp = int((pred_mask & ~label_mask).sum().item())
            fn = int((~pred_mask & label_mask).sum().item())
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            if support > 0 or int(pred_mask.sum().item()) > 0:
                per_label[label_name] = {
                    "support": support,
                    "predicted": int(pred_mask.sum().item()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
        report[source_name] = {
            "count": len(indices),
            "accuracy": accuracy,
            "macro_f1": float(sum(float(item["f1"]) for item in per_label.values()) / len(per_label)) if per_label else 0.0,
            "macro_recall": float(sum(float(item["recall"]) for item in per_label.values()) / len(per_label)) if per_label else 0.0,
            "per_label": per_label,
        }
    return report


def _source_summary(source_report: dict[str, object]) -> dict[str, float]:
    source_items = [
        item
        for item in source_report.values()
        if isinstance(item, dict) and "accuracy" in item
    ]
    if not source_items:
        return {
            "source_balanced_accuracy": 0.0,
            "min_source_accuracy": 0.0,
            "source_balanced_macro_f1": 0.0,
            "min_source_macro_f1": 0.0,
        }
    accuracies = [float(item["accuracy"]) for item in source_items]
    macro_f1s = [float(item.get("macro_f1", 0.0)) for item in source_items]
    return {
        "source_balanced_accuracy": float(sum(accuracies) / len(accuracies)),
        "min_source_accuracy": float(min(accuracies)),
        "source_balanced_macro_f1": float(sum(macro_f1s) / len(macro_f1s)) if macro_f1s else 0.0,
        "min_source_macro_f1": float(min(macro_f1s)) if macro_f1s else 0.0,
    }


def _label_recall(class_logits: torch.Tensor, class_labels: torch.Tensor, label_name: str) -> float:
    label_idx = DEFAULT_VOCAB.label_to_index[label_name]
    mask = class_labels == label_idx
    if not bool(mask.any()):
        return 0.0
    preds = class_logits.argmax(dim=-1)
    return float((preds[mask] == label_idx).float().mean().item())


def _binary_label_metrics(class_logits: torch.Tensor, class_labels: torch.Tensor, label_name: str) -> dict[str, float | int]:
    label_idx = DEFAULT_VOCAB.label_to_index[label_name]
    preds = class_logits.argmax(dim=-1)
    pred_pos = preds == label_idx
    true_pos = class_labels == label_idx
    tp = int((pred_pos & true_pos).sum().item())
    fp = int((pred_pos & ~true_pos).sum().item())
    fn = int((~pred_pos & true_pos).sum().item())
    support = int(true_pos.sum().item())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].float().mean().item())


def _safe_diagnostic(
    diagnostics: dict[str, torch.Tensor] | None,
    name: str,
    reference: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    if diagnostics and name in diagnostics:
        return diagnostics[name].to(dtype=torch.float32)
    return torch.full((reference.shape[0],), fill_value=fill_value, dtype=torch.float32)


def _mask_for_indices(preds: torch.Tensor, indices: list[int]) -> torch.Tensor:
    mask = torch.zeros_like(preds, dtype=torch.bool)
    for index in indices:
        mask |= preds == int(index)
    return mask


def _label_count_dict(preds: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    if not bool(mask.any()):
        return {label_name: 0 for label_name in DEFAULT_VOCAB.labels}
    return {
        label_name: int(((preds == label_idx) & mask).sum().item())
        for label_idx, label_name in enumerate(DEFAULT_VOCAB.labels)
    }


def _local_expert_report(
    local_logits: torch.Tensor,
    global_labels: torch.Tensor,
    label_names: list[str],
) -> dict[str, object]:
    local_to_global = [DEFAULT_VOCAB.label_to_index[label_name] for label_name in label_names]
    local_labels = torch.full_like(global_labels, fill_value=-1)
    in_scope = torch.zeros_like(global_labels, dtype=torch.bool)
    for local_idx, global_idx in enumerate(local_to_global):
        mask = global_labels == global_idx
        local_labels[mask] = local_idx
        in_scope |= mask
    if not bool(in_scope.any()):
        return {
            "scope_label_names": label_names,
            "in_scope_count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [[0 for _ in label_names] for _ in label_names],
            "per_class": [
                {
                    "label_name": label_name,
                    "support": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                for label_name in label_names
            ],
        }
    scoped_logits = local_logits[in_scope]
    scoped_labels = local_labels[in_scope]
    metrics = compute_system_metrics(
        class_logits=scoped_logits,
        class_labels=scoped_labels,
        route_logits=torch.zeros((scoped_logits.shape[0], 1)),
        route_labels=torch.zeros((scoped_logits.shape[0],), dtype=torch.long),
        num_classes=len(label_names),
    ).classification
    per_class = compute_per_class_metrics(scoped_logits, scoped_labels, label_names)
    confusion = compute_confusion_matrix(scoped_logits, scoped_labels, len(label_names))
    return {
        "scope_label_names": label_names,
        "in_scope_count": int(in_scope.sum().item()),
        "accuracy": float(metrics.accuracy),
        "macro_f1": float(metrics.macro_f1),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "label_name": item.label_name,
                "support": item.support,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in per_class
        ],
    }


def _expert2_internal_report(expert_outputs: dict[str, torch.Tensor], class_labels: torch.Tensor) -> dict[str, object]:
    logits = expert_outputs.get("expert2_logits")
    if logits is None:
        return {}
    aircraft_idx = DEFAULT_VOCAB.label_to_index["AircraftTarget"]
    local_preds = logits.argmax(dim=-1)
    local_labels = (class_labels == aircraft_idx).to(dtype=torch.long)
    metrics = compute_system_metrics(
        class_logits=logits,
        class_labels=local_labels,
        route_logits=torch.zeros((logits.shape[0], 1)),
        route_labels=torch.zeros((logits.shape[0],), dtype=torch.long),
        num_classes=2,
    ).classification
    confusion = compute_confusion_matrix(logits, local_labels, 2)
    per_class = compute_per_class_metrics(logits, local_labels, ["NonAircraft", "Aircraft"])
    aircraft_mask = class_labels == aircraft_idx
    non_aircraft_mask = ~aircraft_mask
    false_aircraft = (local_preds == 1) & non_aircraft_mask
    return {
        "label_names": ["NonAircraft", "Aircraft"],
        "accuracy": float(metrics.accuracy),
        "macro_f1": float(metrics.macro_f1),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "label_name": item.label_name,
                "support": item.support,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in per_class
        ],
        "aircraft_precision": float(per_class[1].precision),
        "aircraft_recall": float(per_class[1].recall),
        "false_aircraft_rate": float(false_aircraft.float().sum().item() / non_aircraft_mask.float().sum().item())
        if bool(non_aircraft_mask.any())
        else 0.0,
    }


def _expert3_rule_report(
    expert_outputs: dict[str, torch.Tensor],
    class_labels: torch.Tensor,
    route_labels: torch.Tensor,
) -> dict[str, object]:
    logits = expert_outputs.get("expert3_logits")
    if logits is None:
        return {}
    high_idx = ROUTE_NAMES.index("high_altitude_filter")
    selected = logits.argmax(dim=-1) == 1
    selected_count = int(selected.sum().item())
    expected = route_labels == high_idx
    correct = selected & expected
    selected_label_distribution = {
        label_name: int((selected & (class_labels == label_idx)).sum().item())
        for label_idx, label_name in enumerate(DEFAULT_VOCAB.labels)
        if int((selected & (class_labels == label_idx)).sum().item()) > 0
    }
    return {
        "selected_count": selected_count,
        "expected_count": int(expected.sum().item()),
        "correct_count": int(correct.sum().item()),
        "precision": int(correct.sum().item()) / selected_count if selected_count > 0 else 0.0,
        "recall": int(correct.sum().item()) / int(expected.sum().item()) if bool(expected.any()) else 0.0,
        "selected_label_distribution": selected_label_distribution,
    }


def _expert4_internal_report(expert_outputs: dict[str, torch.Tensor], class_labels: torch.Tensor) -> dict[str, object]:
    logits = expert_outputs.get("expert4_logits")
    if logits is None:
        return {}
    report = _local_expert_report(logits, class_labels, ["PeopleTarget", "CarTarget", "ShipTarget"])
    people_idx = DEFAULT_VOCAB.label_to_index["PeopleTarget"]
    car_idx_local = 1
    local_preds = logits.argmax(dim=-1)
    people_mask = class_labels == people_idx
    report["people_as_car_count"] = int(((local_preds == car_idx_local) & people_mask).sum().item())
    report["people_as_car_rate"] = (
        int(((local_preds == car_idx_local) & people_mask).sum().item()) / int(people_mask.sum().item())
        if bool(people_mask.any())
        else 0.0
    )
    return report


def _expert1_branch_report(expert_outputs: dict[str, torch.Tensor], class_labels: torch.Tensor) -> dict[str, object]:
    traj_logits = expert_outputs.get("expert1_traj_logits")
    rd_logits = expert_outputs.get("expert1_rd_logits")
    if traj_logits is None or rd_logits is None:
        return {}
    low_indices = [DEFAULT_VOCAB.label_to_index[label_name] for label_name in LOW_ALTITUDE_CLASSES]
    low_mask = torch.zeros_like(class_labels, dtype=torch.bool)
    local_labels = torch.full_like(class_labels, fill_value=-1)
    for local_idx, global_idx in enumerate(low_indices):
        mask = class_labels == global_idx
        low_mask |= mask
        local_labels[mask] = local_idx
    rd_mask = expert_outputs.get("rd_is_multimodal")
    rd_mask = rd_mask.bool() if rd_mask is not None else torch.zeros_like(class_labels, dtype=torch.bool)
    trajectory = _local_expert_report(traj_logits, class_labels, LOW_ALTITUDE_CLASSES)
    rd_all = _local_expert_report(rd_logits, class_labels, LOW_ALTITUDE_CLASSES)
    rd_on_rd = _local_expert_report(rd_logits[rd_mask], class_labels[rd_mask], LOW_ALTITUDE_CLASSES) if bool(rd_mask.any()) else _local_expert_report(rd_logits[:0], class_labels[:0], LOW_ALTITUDE_CLASSES)
    traj_confidence = expert_outputs.get("expert1_traj_confidence")
    traj_entropy = expert_outputs.get("expert1_traj_entropy")
    fusion_weights = expert_outputs.get("expert1_fusion_weights")
    high_conf_correct = high_conf_wrong = 0
    if traj_confidence is not None and bool(low_mask.any()):
        traj_preds = traj_logits.argmax(dim=-1)
        high_conf = low_mask & (traj_confidence > 0.8)
        high_conf_correct = int(((traj_preds == local_labels) & high_conf).sum().item())
        high_conf_wrong = int(((traj_preds != local_labels) & high_conf).sum().item())
    return {
        "trajectory": trajectory,
        "rd": rd_all,
        "rd_on_rd": rd_on_rd,
        "traj_confidence_mean": _mean_or_zero(traj_confidence, low_mask) if traj_confidence is not None else 0.0,
        "traj_entropy_mean": _mean_or_zero(traj_entropy, low_mask) if traj_entropy is not None else 0.0,
        "traj_confidence_gt_0_8_correct_count": high_conf_correct,
        "traj_confidence_gt_0_8_wrong_count": high_conf_wrong,
        "fusion_weight_mean": fusion_weights.float().mean(dim=0).tolist() if fusion_weights is not None and fusion_weights.numel() > 0 else [],
    }


def _rd_prototype_similarity(model: nn.Module | None) -> dict[str, object]:
    if model is None or not hasattr(model, "low_altitude_expert"):
        return {}
    rd_head = getattr(model.low_altitude_expert, "rd_head", None)
    if rd_head is None or not hasattr(rd_head, "weight"):
        return {}
    with torch.no_grad():
        weight = torch.nn.functional.normalize(rd_head.weight.detach().cpu(), dim=-1)
        matrix = weight @ weight.T
    return {
        "label_names": LOW_ALTITUDE_CLASSES,
        "cosine_matrix": matrix.tolist(),
    }


def _ground_report(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    aircraft_diagnostics: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    ground_indices = [
        DEFAULT_VOCAB.label_to_index["PeopleTarget"],
        DEFAULT_VOCAB.label_to_index["CarTarget"],
        DEFAULT_VOCAB.label_to_index["ShipTarget"],
    ]
    preds = class_logits.argmax(dim=-1)
    ground_mask = torch.zeros_like(class_labels, dtype=torch.bool)
    for label_idx in ground_indices:
        ground_mask |= class_labels == label_idx
    pred_ground_mask = torch.zeros_like(preds, dtype=torch.bool)
    for label_idx in ground_indices:
        pred_ground_mask |= preds == label_idx
    groundness_probability = _safe_diagnostic(
        aircraft_diagnostics,
        "groundness_probability",
        class_labels,
        fill_value=1.0,
    )
    groundness_confidence = _safe_diagnostic(
        aircraft_diagnostics,
        "groundness_confidence",
        class_labels,
        fill_value=0.5,
    )
    ground_high_conf = _safe_diagnostic(
        aircraft_diagnostics,
        "ground_high_confidence",
        class_labels,
    ).bool()
    tp = int((pred_ground_mask & ground_mask).sum().item())
    fp = int((pred_ground_mask & ~ground_mask).sum().item())
    fn = int((~pred_ground_mask & ground_mask).sum().item())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "ground_group_precision": precision,
        "ground_group_recall": recall,
        "ground_group_f1": f1,
        "pred_ground_on_non_ground_count": fp,
        "groundness_probability_mean_true_ground": _mean_or_zero(groundness_probability, ground_mask),
        "groundness_probability_mean_non_ground": _mean_or_zero(groundness_probability, ~ground_mask),
        "groundness_confidence_mean_true_ground": _mean_or_zero(groundness_confidence, ground_mask),
        "groundness_confidence_mean_non_ground": _mean_or_zero(groundness_confidence, ~ground_mask),
        "ground_high_confidence_count": int(ground_high_conf.sum().item()),
    }


def _reroute_report(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    reroute_diagnostics: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    if not reroute_diagnostics:
        return {
            "reroute_rate": 0.0,
            "reroute_avg_count": 0.0,
            "reroute_recovery_rate": 0.0,
            "reroute_fallback_rate": 0.0,
            "reroute_worsen_rate": 0.0,
            "low_altitude_deep_used_count": 0,
            "low_altitude_deep_used_rate": 0.0,
            "low_altitude_deep_accepted_count": 0,
            "low_altitude_deep_accept_rate": 0.0,
            "reroute_count_distribution": {},
        }
    preds = class_logits.argmax(dim=-1)
    correct = preds == class_labels
    reroute_count = reroute_diagnostics.get("reroute_count")
    reroute_triggered = reroute_diagnostics.get("reroute_triggered")
    reroute_fallback = reroute_diagnostics.get("reroute_fallback")
    if reroute_count is None or reroute_triggered is None or reroute_fallback is None:
        return {
            "reroute_rate": 0.0,
            "reroute_avg_count": 0.0,
            "reroute_recovery_rate": 0.0,
            "reroute_fallback_rate": 0.0,
            "reroute_worsen_rate": 0.0,
            "low_altitude_deep_used_count": 0,
            "low_altitude_deep_used_rate": 0.0,
            "low_altitude_deep_accepted_count": 0,
            "low_altitude_deep_accept_rate": 0.0,
            "reroute_count_distribution": {},
        }
    reroute_triggered = reroute_triggered.bool()
    reroute_fallback = reroute_fallback.bool()
    total = int(reroute_count.shape[0])
    triggered_count = int(reroute_triggered.sum().item())
    fallback_count = int(reroute_fallback.sum().item())
    reroute_rate = triggered_count / total if total > 0 else 0.0
    reroute_avg_count = float(reroute_count.float().mean().item()) if total > 0 else 0.0
    reroute_recovery_rate = float(correct[reroute_triggered].float().mean().item()) if triggered_count > 0 else 0.0
    reroute_worsen_mask = reroute_triggered & ~correct & ~reroute_fallback
    reroute_worsen_rate = float(reroute_worsen_mask.float().sum().item() / triggered_count) if triggered_count > 0 else 0.0
    distribution = torch.bincount(reroute_count, minlength=int(reroute_count.max().item()) + 1 if reroute_count.numel() > 0 else 1)
    low_deep_used = reroute_diagnostics.get("reroute_low_altitude_deep_used")
    low_deep_accepted = reroute_diagnostics.get("reroute_low_altitude_deep_accepted")
    low_deep_used_count = int(low_deep_used.bool().sum().item()) if low_deep_used is not None else 0
    low_deep_accepted_count = int(low_deep_accepted.bool().sum().item()) if low_deep_accepted is not None else 0
    return {
        "reroute_rate": reroute_rate,
        "reroute_avg_count": reroute_avg_count,
        "reroute_recovery_rate": reroute_recovery_rate,
        "reroute_fallback_rate": fallback_count / total if total > 0 else 0.0,
        "reroute_worsen_rate": reroute_worsen_rate,
        "low_altitude_deep_used_count": low_deep_used_count,
        "low_altitude_deep_used_rate": low_deep_used_count / total if total > 0 else 0.0,
        "low_altitude_deep_accepted_count": low_deep_accepted_count,
        "low_altitude_deep_accept_rate": low_deep_accepted_count / low_deep_used_count if low_deep_used_count > 0 else 0.0,
        "reroute_count_distribution": {
            str(index): int(value.item())
            for index, value in enumerate(distribution)
        },
    }


def _aircraft_report(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    aircraft_diagnostics: dict[str, torch.Tensor] | None = None,
    class_logits_without_aircraft_protection: torch.Tensor | None = None,
    source_names: list[str] | None = None,
) -> dict[str, object]:
    aircraft_idx = DEFAULT_VOCAB.label_to_index["AircraftTarget"]
    low_indices = [DEFAULT_VOCAB.label_to_index[label_name] for label_name in LOW_ALTITUDE_CLASSES]
    ground_indices = [
        DEFAULT_VOCAB.label_to_index["PeopleTarget"],
        DEFAULT_VOCAB.label_to_index["CarTarget"],
        DEFAULT_VOCAB.label_to_index["ShipTarget"],
    ]
    if class_logits_without_aircraft_protection is None:
        class_logits_without_aircraft_protection = class_logits
    preds = class_logits.argmax(dim=-1)
    unprotected_preds = class_logits_without_aircraft_protection.argmax(dim=-1)
    true_aircraft = class_labels == aircraft_idx
    non_aircraft = ~true_aircraft
    expert2_positive = _safe_diagnostic(
        aircraft_diagnostics,
        "expert2_positive",
        class_labels,
    ).bool()
    aircraft_margin = _safe_diagnostic(aircraft_diagnostics, "aircraft_margin", class_labels)
    aircraft_confidence = _safe_diagnostic(aircraft_diagnostics, "aircraft_confidence", class_labels)
    conventional_route_weight = _safe_diagnostic(
        aircraft_diagnostics,
        "conventional_route_weight",
        class_labels,
    )
    conventional_gate = _safe_diagnostic(aircraft_diagnostics, "conventional_gate", class_labels)
    high_confidence = _safe_diagnostic(
        aircraft_diagnostics,
        "aircraft_high_confidence",
        class_labels,
    ).bool()
    mid_confidence = _safe_diagnostic(
        aircraft_diagnostics,
        "aircraft_mid_confidence",
        class_labels,
    ).bool()
    source_override = _safe_diagnostic(
        aircraft_diagnostics,
        "aircraft_source_override",
        class_labels,
    ).bool()
    source_reject = _safe_diagnostic(
        aircraft_diagnostics,
        "aircraft_source_reject",
        class_labels,
    ).bool()
    final_aircraft = preds == aircraft_idx
    unprotected_aircraft = unprotected_preds == aircraft_idx
    final_low = _mask_for_indices(preds, low_indices)
    final_ground = _mask_for_indices(preds, ground_indices)
    unprotected_low = _mask_for_indices(unprotected_preds, low_indices)
    unprotected_ground = _mask_for_indices(unprotected_preds, ground_indices)
    protected_recovered = true_aircraft & ~unprotected_aircraft & final_aircraft
    protected_lost = true_aircraft & unprotected_aircraft & ~final_aircraft
    false_positive_introduced = non_aircraft & ~unprotected_aircraft & final_aircraft
    expert2_positive_count = int(expert2_positive.sum().item())
    expert2_positive_system_not_aircraft = expert2_positive & ~final_aircraft
    final_aircraft_expert2_not_positive = final_aircraft & ~expert2_positive
    non_aircraft_count = int(non_aircraft.sum().item())
    aircraft_false_positive_by_source: dict[str, int] = {}
    if source_names is not None:
        fp_mask = non_aircraft & final_aircraft
        for idx, source_name in enumerate(source_names):
            if bool(fp_mask[idx].item()):
                source_key = str(source_name)
                aircraft_false_positive_by_source[source_key] = aircraft_false_positive_by_source.get(source_key, 0) + 1
    metrics = _binary_label_metrics(class_logits, class_labels, "AircraftTarget")
    unprotected_metrics = _binary_label_metrics(
        class_logits_without_aircraft_protection,
        class_labels,
        "AircraftTarget",
    )
    return {
        "system": metrics,
        "without_aircraft_protection": unprotected_metrics,
        "aircraft_system_recall": float(metrics["recall"]),
        "aircraft_system_precision": float(metrics["precision"]),
        "aircraft_system_f1": float(metrics["f1"]),
        "aircraft_recall_without_aircraft_protection": float(unprotected_metrics["recall"]),
        "aircraft_precision_without_aircraft_protection": float(unprotected_metrics["precision"]),
        "aircraft_f1_without_aircraft_protection": float(unprotected_metrics["f1"]),
        "aircraft_margin_mean_true_aircraft": _mean_or_zero(aircraft_margin, true_aircraft),
        "aircraft_margin_mean_non_aircraft": _mean_or_zero(aircraft_margin, non_aircraft),
        "aircraft_confidence_mean_true_aircraft": _mean_or_zero(aircraft_confidence, true_aircraft),
        "aircraft_confidence_mean_non_aircraft": _mean_or_zero(aircraft_confidence, non_aircraft),
        "aircraft_route_weight_mean_true_aircraft": _mean_or_zero(conventional_route_weight, true_aircraft),
        "aircraft_route_weight_mean_non_aircraft": _mean_or_zero(conventional_route_weight, non_aircraft),
        "aircraft_conventional_gate_mean_true_aircraft": _mean_or_zero(conventional_gate, true_aircraft),
        "expert2_positive_count": expert2_positive_count,
        "expert2_positive_system_not_aircraft_count": int(expert2_positive_system_not_aircraft.sum().item()),
        "expert2_positive_system_not_aircraft_rate": (
            int(expert2_positive_system_not_aircraft.sum().item()) / expert2_positive_count
            if expert2_positive_count > 0
            else 0.0
        ),
        "expert2_positive_true_aircraft_count": int((expert2_positive & true_aircraft).sum().item()),
        "expert2_positive_false_aircraft_count": int((expert2_positive & non_aircraft).sum().item()),
        "final_aircraft_expert2_not_positive_count": int(final_aircraft_expert2_not_positive.sum().item()),
        "final_aircraft_expert2_not_positive_true_aircraft_count": int((final_aircraft_expert2_not_positive & true_aircraft).sum().item()),
        "final_aircraft_expert2_not_positive_false_aircraft_count": int((final_aircraft_expert2_not_positive & non_aircraft).sum().item()),
        "aircraft_false_positive_by_source": aircraft_false_positive_by_source,
        "aircraft_high_confidence_count": int(high_confidence.sum().item()),
        "aircraft_mid_confidence_count": int(mid_confidence.sum().item()),
        "aircraft_source_override_count": int(source_override.sum().item()),
        "aircraft_source_reject_count": int(source_reject.sum().item()),
        "aircraft_source_reject_true_aircraft_count": int((source_reject & true_aircraft).sum().item()),
        "aircraft_source_reject_false_aircraft_count": int((source_reject & non_aircraft).sum().item()),
        "aircraft_protection_recovered_count": int(protected_recovered.sum().item()),
        "aircraft_protection_lost_count": int(protected_lost.sum().item()),
        "aircraft_false_positive_introduced_count": int(false_positive_introduced.sum().item()),
        "non_aircraft_as_aircraft_count": int((non_aircraft & final_aircraft).sum().item()),
        "non_aircraft_as_aircraft_rate": (
            int((non_aircraft & final_aircraft).sum().item()) / non_aircraft_count
            if non_aircraft_count > 0
            else 0.0
        ),
        "aircraft_overridden_by_low_altitude_count": int((true_aircraft & final_low).sum().item()),
        "aircraft_overridden_by_ground_count": int((true_aircraft & final_ground).sum().item()),
        "aircraft_overridden_by_low_altitude_without_protection_count": int((true_aircraft & unprotected_low).sum().item()),
        "aircraft_overridden_by_ground_without_protection_count": int((true_aircraft & unprotected_ground).sum().item()),
        "true_aircraft_prediction_counts": _label_count_dict(preds, true_aircraft),
        "true_aircraft_prediction_counts_without_protection": _label_count_dict(unprotected_preds, true_aircraft),
    }


def _low_altitude_report(class_logits: torch.Tensor, class_labels: torch.Tensor) -> dict[str, object]:
    low_indices = [
        DEFAULT_VOCAB.label_to_index[label_name]
        for label_name in LOW_ALTITUDE_CLASSES
    ]
    low_mask = torch.zeros_like(class_labels, dtype=torch.bool)
    for label_idx in low_indices:
        low_mask |= class_labels == label_idx
    if not bool(low_mask.any()):
        return {
            "support": 0,
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "per_class_recall": {label_name: 0.0 for label_name in LOW_ALTITUDE_CLASSES},
            "drone_bird": {
                "drone_recall": 0.0,
                "bird_recall": 0.0,
                "drone_bird_recall_mean": 0.0,
                "drone_as_bird_count": 0,
                "bird_as_drone_count": 0,
                "drone_as_bird_rate": 0.0,
                "bird_as_drone_rate": 0.0,
            },
            "unknown_clutter_recall_mean": 0.0,
        }
    local_logits = class_logits[low_mask][:, low_indices]
    local_labels = class_labels[low_mask].clone()
    for local_idx, global_idx in enumerate(low_indices):
        local_labels[class_labels[low_mask] == global_idx] = local_idx
    low_metrics = compute_system_metrics(
        class_logits=local_logits,
        class_labels=local_labels,
        route_logits=torch.zeros((local_logits.shape[0], 1)),
        route_labels=torch.zeros((local_logits.shape[0],), dtype=torch.long),
        num_classes=len(LOW_ALTITUDE_CLASSES),
    ).classification
    preds = local_logits.argmax(dim=-1)
    recalls: dict[str, float] = {}
    for local_idx, label_name in enumerate(LOW_ALTITUDE_CLASSES):
        mask = local_labels == local_idx
        recalls[label_name] = float((preds[mask] == local_idx).float().mean().item()) if bool(mask.any()) else 0.0
    drone_idx = LOW_ALTITUDE_CLASSES.index("DroneTarget")
    bird_idx = LOW_ALTITUDE_CLASSES.index("BirdTarget")
    drone_mask = local_labels == drone_idx
    bird_mask = local_labels == bird_idx
    drone_support = int(drone_mask.sum().item())
    bird_support = int(bird_mask.sum().item())
    drone_as_bird = int(((preds == bird_idx) & drone_mask).sum().item())
    bird_as_drone = int(((preds == drone_idx) & bird_mask).sum().item())
    drone_bird = {
        "drone_recall": recalls["DroneTarget"],
        "bird_recall": recalls["BirdTarget"],
        "drone_bird_recall_mean": 0.5 * (recalls["DroneTarget"] + recalls["BirdTarget"]) if (drone_support or bird_support) else 0.0,
        "drone_as_bird_count": drone_as_bird,
        "bird_as_drone_count": bird_as_drone,
        "drone_as_bird_rate": drone_as_bird / drone_support if drone_support else 0.0,
        "bird_as_drone_rate": bird_as_drone / bird_support if bird_support else 0.0,
    }
    unknown_clutter = 0.5 * (recalls["UnknownTarget"] + recalls["ClutterTarget"])
    return {
        "support": int(low_mask.sum().item()),
        "macro_f1": low_metrics.macro_f1,
        "accuracy": low_metrics.accuracy,
        "per_class_recall": recalls,
        "drone_bird": drone_bird,
        "unknown_clutter_recall_mean": unknown_clutter,
    }


def _fine_label_distribution(fine_label_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label_name in fine_label_names:
        key = str(label_name)
        counts[key] = counts.get(key, 0) + 1
    return counts


REROUTE_REASON_NAMES = {
    0: "none",
    1: "low_unknown",
    2: "low_confusion_pair",
    3: "low_evidence_conflict",
    4: "low_margin",
    5: "low_other_reject",
    6: "aircraft_reject",
    7: "ground_reject",
    8: "high_altitude_reject",
}


def _reroute_reason_distribution(reroute_diagnostics: dict[str, torch.Tensor] | None) -> dict[str, int]:
    if not reroute_diagnostics or "reroute_reject_reason_code" not in reroute_diagnostics:
        return {}
    codes = reroute_diagnostics["reroute_reject_reason_code"].to(dtype=torch.long)
    triggered = reroute_diagnostics.get("reroute_triggered")
    if triggered is not None:
        codes = codes[triggered.bool()]
    counts: dict[str, int] = {}
    for code in codes.tolist():
        name = REROUTE_REASON_NAMES.get(int(code), f"unknown_{int(code)}")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _route_path_names(reroute_path: torch.Tensor | None, index: int) -> list[str]:
    if reroute_path is None:
        return []
    path_names: list[str] = []
    for route_idx_value in reroute_path[index].tolist():
        route_idx_int = int(route_idx_value)
        if 0 <= route_idx_int < len(ROUTE_NAMES):
            path_names.append(ROUTE_NAMES[route_idx_int])
    return path_names


def _per_class_before_after_recall(
    labels: torch.Tensor,
    without_preds: torch.Tensor,
    with_preds: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label_idx, label_name in enumerate(DEFAULT_VOCAB.labels):
        label_mask = labels == label_idx
        support = int(label_mask.sum().item())
        before_correct = (without_preds == label_idx) & label_mask
        after_correct = (with_preds == label_idx) & label_mask
        helped = (~before_correct) & after_correct
        harmed = before_correct & (~after_correct)
        changed = (without_preds != with_preds) & label_mask
        before_recall = float(before_correct.sum().item() / support) if support > 0 else 0.0
        after_recall = float(after_correct.sum().item() / support) if support > 0 else 0.0
        result[label_name] = {
            "support": support,
            "without_recall": before_recall,
            "with_recall": after_recall,
            "recall_delta": after_recall - before_recall,
            "changed_count": int(changed.sum().item()),
            "helped_count": int(helped.sum().item()),
            "harmed_count": int(harmed.sum().item()),
        }
    return result


def _safety_critical_pair_changes(
    labels: torch.Tensor,
    without_preds: torch.Tensor,
    with_preds: torch.Tensor,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for true_label, pred_label in SAFETY_CRITICAL_PAIRS:
        true_idx = DEFAULT_VOCAB.label_to_index.get(true_label)
        pred_idx = DEFAULT_VOCAB.label_to_index.get(pred_label)
        if true_idx is None or pred_idx is None:
            continue
        before = (labels == true_idx) & (without_preds == pred_idx)
        after = (labels == true_idx) & (with_preds == pred_idx)
        reduced = before & ~after
        introduced = ~before & after
        key = f"{true_label}->{pred_label}"
        result[key] = {
            "before_count": int(before.sum().item()),
            "after_count": int(after.sum().item()),
            "delta": int(after.sum().item() - before.sum().item()),
            "reduced_count": int(reduced.sum().item()),
            "introduced_count": int(introduced.sum().item()),
        }
    return result


def _expert1_scope_report(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    route_logits: torch.Tensor,
    route_labels: torch.Tensor,
) -> dict[str, object]:
    low_route_idx = ROUTE_NAMES.index("low_altitude_suspect")
    routed_mask = route_logits.argmax(dim=-1) == low_route_idx
    true_mask = route_labels == low_route_idx
    return {
        "routed": _low_altitude_report(class_logits[routed_mask], class_labels[routed_mask]) if bool(routed_mask.any()) else _low_altitude_report(class_logits[:0], class_labels[:0]),
        "true_low_altitude": _low_altitude_report(class_logits[true_mask], class_labels[true_mask]) if bool(true_mask.any()) else _low_altitude_report(class_logits[:0], class_labels[:0]),
        "routed_count": int(routed_mask.sum().item()),
        "true_low_altitude_count": int(true_mask.sum().item()),
    }


def _reroute_comparison(without_metrics: dict[str, object], with_metrics: dict[str, object]) -> dict[str, object]:
    without_logits = without_metrics["class_logits"]
    with_logits = with_metrics["class_logits"]
    labels = with_metrics["class_labels"]
    without_preds = without_logits.argmax(dim=-1)
    with_preds = with_logits.argmax(dim=-1)
    before_correct = without_preds == labels
    after_correct = with_preds == labels
    helped = (~before_correct) & after_correct
    harmed = before_correct & (~after_correct)
    changed = without_preds != with_preds
    reroute_diagnostics = with_metrics.get("reroute_diagnostics", {})
    reroute_triggered = reroute_diagnostics.get("reroute_triggered") if isinstance(reroute_diagnostics, dict) else None
    reroute_fallback = reroute_diagnostics.get("reroute_fallback") if isinstance(reroute_diagnostics, dict) else None
    reroute_count = reroute_diagnostics.get("reroute_count") if isinstance(reroute_diagnostics, dict) else None
    reroute_path = reroute_diagnostics.get("reroute_path") if isinstance(reroute_diagnostics, dict) else None
    reject_reason_code = reroute_diagnostics.get("reroute_reject_reason_code") if isinstance(reroute_diagnostics, dict) else None
    sample_ids = list(with_metrics.get("sample_ids", []))
    source_names = list(with_metrics.get("source_names", []))
    sample_outcomes: list[dict[str, object]] = []
    for idx in range(int(labels.shape[0])):
        was_triggered = bool(reroute_triggered[idx].item()) if reroute_triggered is not None else False
        if not (was_triggered or bool(changed[idx].item()) or bool(helped[idx].item()) or bool(harmed[idx].item())):
            continue
        reason_code = int(reject_reason_code[idx].item()) if reject_reason_code is not None else 0
        sample_outcomes.append(
            {
                "sample_id": str(sample_ids[idx]) if idx < len(sample_ids) else str(idx),
                "source_name": str(source_names[idx]) if idx < len(source_names) else "",
                "true_label": DEFAULT_VOCAB.decode(int(labels[idx].item())),
                "without_pred": DEFAULT_VOCAB.decode(int(without_preds[idx].item())),
                "with_pred": DEFAULT_VOCAB.decode(int(with_preds[idx].item())),
                "reroute_triggered": was_triggered,
                "reroute_count": int(reroute_count[idx].item()) if reroute_count is not None else 0,
                "fallback_to_unknown": bool(reroute_fallback[idx].item()) if reroute_fallback is not None else False,
                "reroute_path": _route_path_names(reroute_path, idx),
                "reject_reason": REROUTE_REASON_NAMES.get(reason_code, f"unknown_{reason_code}"),
                "reroute_helped": bool(helped[idx].item()),
                "reroute_harmed": bool(harmed[idx].item()),
                "prediction_changed": bool(changed[idx].item()),
            }
        )
    return {
        "without_reroute": _metric_summary(without_metrics),
        "with_reroute": _metric_summary(with_metrics),
        "accuracy_delta": float(with_metrics.get("accuracy", 0.0)) - float(without_metrics.get("accuracy", 0.0)),
        "macro_f1_delta": float(with_metrics.get("macro_f1", 0.0)) - float(without_metrics.get("macro_f1", 0.0)),
        "changed_count": int(changed.sum().item()),
        "helped_count": int(helped.sum().item()),
        "harmed_count": int(harmed.sum().item()),
        "unchanged_count": int((~changed).sum().item()),
        "reason_distribution": _reroute_reason_distribution(with_metrics.get("reroute_diagnostics", {})),
        "per_class_before_after_recall": _per_class_before_after_recall(labels, without_preds, with_preds),
        "safety_critical_pairs": _safety_critical_pair_changes(labels, without_preds, with_preds),
        "sample_outcomes": sample_outcomes,
    }


def _critical_metric_summary(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    route_logits: torch.Tensor,
    route_labels: torch.Tensor,
    aircraft_diagnostics: dict[str, torch.Tensor] | None = None,
    reroute_diagnostics: dict[str, torch.Tensor] | None = None,
    class_logits_without_aircraft_protection: torch.Tensor | None = None,
    source_report: dict[str, object] | None = None,
    source_names: list[str] | None = None,
) -> dict[str, object]:
    route = _route_report(route_logits, route_labels)
    per_route_recall = {
        item["route_name"]: item["recall"]
        for item in route["per_route"]
    }
    low = _low_altitude_report(class_logits, class_labels)
    ground = _ground_report(
        class_logits,
        class_labels,
        aircraft_diagnostics=aircraft_diagnostics,
    )
    reroute = _reroute_report(
        class_logits,
        class_labels,
        reroute_diagnostics=reroute_diagnostics,
    )
    aircraft = _aircraft_report(
        class_logits,
        class_labels,
        aircraft_diagnostics=aircraft_diagnostics,
        class_logits_without_aircraft_protection=class_logits_without_aircraft_protection,
        source_names=source_names,
    )
    source_summary = _source_summary(source_report or {})
    return {
        "route_macro_recall": route["macro_recall"],
        "per_route_recall": per_route_recall,
        "ground_route_recall": per_route_recall.get("ground_target", 0.0),
        "expert1_low_altitude_macro_f1": low["macro_f1"],
        "expert1_low_altitude_accuracy": low["accuracy"],
        "low_altitude_per_class_recall": low["per_class_recall"],
        "drone_bird": low["drone_bird"],
        "unknown_recall": low["per_class_recall"].get("UnknownTarget", 0.0),
        "clutter_recall": low["per_class_recall"].get("ClutterTarget", 0.0),
        "unknown_clutter_recall_mean": low["unknown_clutter_recall_mean"],
        "ground": ground,
        "ground_group_recall": ground["ground_group_recall"],
        "ground_group_precision": ground["ground_group_precision"],
        "ground_group_f1": ground["ground_group_f1"],
        "pred_ground_on_non_ground_count": ground["pred_ground_on_non_ground_count"],
        "reroute": reroute,
        "reroute_rate": reroute["reroute_rate"],
        "reroute_avg_count": reroute["reroute_avg_count"],
        "reroute_recovery_rate": reroute["reroute_recovery_rate"],
        "reroute_fallback_rate": reroute["reroute_fallback_rate"],
        "reroute_worsen_rate": reroute["reroute_worsen_rate"],
        "reroute_reason_distribution": _reroute_reason_distribution(reroute_diagnostics),
        "aircraft": aircraft,
        "aircraft_system_recall": aircraft["aircraft_system_recall"],
        "aircraft_system_precision": aircraft["aircraft_system_precision"],
        "aircraft_system_f1": aircraft["aircraft_system_f1"],
        "aircraft_recall_without_aircraft_protection": aircraft["aircraft_recall_without_aircraft_protection"],
        "aircraft_source_override_count": aircraft["aircraft_source_override_count"],
        "aircraft_source_reject_count": aircraft["aircraft_source_reject_count"],
        "aircraft_source_reject_true_aircraft_count": aircraft["aircraft_source_reject_true_aircraft_count"],
        "aircraft_source_reject_false_aircraft_count": aircraft["aircraft_source_reject_false_aircraft_count"],
        "expert2_positive_system_not_aircraft_count": aircraft["expert2_positive_system_not_aircraft_count"],
        "expert2_positive_system_not_aircraft_rate": aircraft["expert2_positive_system_not_aircraft_rate"],
        "final_aircraft_expert2_not_positive_count": aircraft["final_aircraft_expert2_not_positive_count"],
        "aircraft_false_positive_by_source": aircraft["aircraft_false_positive_by_source"],
        "non_aircraft_as_aircraft_rate": aircraft["non_aircraft_as_aircraft_rate"],
        "source_report": source_report or {},
        "source_balanced_accuracy": source_summary["source_balanced_accuracy"],
        "min_source_accuracy": source_summary["min_source_accuracy"],
        "source_balanced_macro_f1": source_summary["source_balanced_macro_f1"],
        "min_source_macro_f1": source_summary["min_source_macro_f1"],
    }


def _system_selection_score(epoch_metrics: dict[str, object]) -> tuple[float, dict[str, float]]:
    critical = epoch_metrics.get("critical_metrics", {})
    macro_f1 = float(epoch_metrics.get("macro_f1", 0.0))
    route_macro = float(critical.get("route_macro_recall", epoch_metrics.get("route_macro_recall", 0.0)))
    low_macro = float(critical.get("expert1_low_altitude_macro_f1", 0.0))
    unknown_recall = float(critical.get("unknown_recall", 0.0))
    clutter_recall = float(critical.get("clutter_recall", 0.0))
    ground_route = float(critical.get("ground_route_recall", 0.0))
    drone_bird = float(critical.get("drone_bird", {}).get("drone_bird_recall_mean", 0.0))
    aircraft_recall = float(critical.get("aircraft_system_recall", 0.0))
    aircraft_precision = float(critical.get("aircraft_system_precision", 0.0))
    aircraft_f1 = float(critical.get("aircraft_system_f1", 0.0))
    aircraft_support = int(critical.get("aircraft", {}).get("system", {}).get("support", 0))
    non_aircraft_as_aircraft_rate = float(critical.get("non_aircraft_as_aircraft_rate", 0.0))
    ground_group_f1 = float(critical.get("ground_group_f1", 0.0))
    pred_ground_on_non_ground = float(critical.get("pred_ground_on_non_ground_count", 0.0))
    source_balanced_accuracy = float(critical.get("source_balanced_accuracy", 0.0))
    min_source_accuracy = float(critical.get("min_source_accuracy", 0.0))
    source_balanced_macro_f1 = float(critical.get("source_balanced_macro_f1", 0.0))
    min_source_macro_f1 = float(critical.get("min_source_macro_f1", 0.0))
    score = (
        macro_f1
        + 0.20 * route_macro
        + 0.25 * low_macro
        + 0.15 * drone_bird
        + 0.10 * (unknown_recall + clutter_recall)
        + 0.10 * ground_route
        + 0.24 * ground_group_f1
        + 0.14 * aircraft_recall
        + 0.12 * aircraft_f1
        + 0.05 * aircraft_precision
        + 0.10 * source_balanced_accuracy
        + 0.08 * min_source_accuracy
        + 0.10 * source_balanced_macro_f1
        + 0.08 * min_source_macro_f1
        - 1.50 * non_aircraft_as_aircraft_rate
        - 0.01 * pred_ground_on_non_ground
    )
    if ground_route == 0.0:
        score -= 0.10
    if ground_group_f1 == 0.0:
        score -= 0.35
    if unknown_recall == 0.0:
        score -= 0.05
    if clutter_recall == 0.0:
        score -= 0.05
    if aircraft_precision < 0.20 and aircraft_recall > 0.0:
        score -= 0.25
    if aircraft_support > 0 and aircraft_recall == 0.0:
        score -= 0.60
    elif aircraft_support > 0 and aircraft_recall < 0.40:
        score -= 0.15
    if aircraft_support > 0 and aircraft_f1 == 0.0:
        score -= 0.20
    if non_aircraft_as_aircraft_rate > 0.01:
        score -= 0.20
    if min_source_accuracy < 0.50:
        score -= 0.25
    return score, {
        "macro_f1": macro_f1,
        "route_macro_recall": route_macro,
        "expert1_low_altitude_macro_f1": low_macro,
        "drone_bird_recall_mean": drone_bird,
        "unknown_recall": unknown_recall,
        "clutter_recall": clutter_recall,
        "ground_route_recall": ground_route,
        "ground_group_f1": ground_group_f1,
        "pred_ground_on_non_ground_count": pred_ground_on_non_ground,
        "aircraft_system_recall": aircraft_recall,
        "aircraft_system_precision": aircraft_precision,
        "aircraft_system_f1": aircraft_f1,
        "aircraft_support": float(aircraft_support),
        "non_aircraft_as_aircraft_rate": non_aircraft_as_aircraft_rate,
        "source_balanced_accuracy": source_balanced_accuracy,
        "min_source_accuracy": min_source_accuracy,
        "source_balanced_macro_f1": source_balanced_macro_f1,
        "min_source_macro_f1": min_source_macro_f1,
    }


def _prediction_records(epoch_metrics: dict[str, object]) -> list[dict[str, object]]:
    class_logits = epoch_metrics["class_logits"]
    class_logits_without_aircraft_protection = epoch_metrics.get(
        "class_logits_without_aircraft_protection",
        class_logits,
    )
    route_logits = epoch_metrics["route_logits"]
    reroute_diagnostics = epoch_metrics.get("reroute_diagnostics", {})
    class_preds = class_logits.argmax(dim=-1).tolist()
    class_preds_without_aircraft_protection = class_logits_without_aircraft_protection.argmax(dim=-1).tolist()
    route_preds = route_logits.argmax(dim=-1).tolist()
    reroute_selected_route = reroute_diagnostics.get("reroute_selected_route") if isinstance(reroute_diagnostics, dict) else None
    reroute_fallback = reroute_diagnostics.get("reroute_fallback") if isinstance(reroute_diagnostics, dict) else None
    reroute_path = reroute_diagnostics.get("reroute_path") if isinstance(reroute_diagnostics, dict) else None
    diagnostics = epoch_metrics.get("aircraft_diagnostics", {})
    records: list[dict[str, object]] = []
    for idx, sample_id in enumerate(epoch_metrics["sample_ids"]):
        true_label = str(epoch_metrics["label_names"][idx])
        pred_label = DEFAULT_VOCAB.decode(int(class_preds[idx]))
        pred_without_protection = DEFAULT_VOCAB.decode(int(class_preds_without_aircraft_protection[idx]))
        true_route = str(epoch_metrics["route_names"][idx])
        router_route = ROUTE_NAMES[int(route_preds[idx])]
        final_route = router_route
        is_fallback = bool(reroute_fallback[idx].item()) if reroute_fallback is not None else False
        if reroute_selected_route is not None:
            reroute_route_idx = int(reroute_selected_route[idx].item())
            if is_fallback:
                final_route = "fallback_unknown"
            elif 0 <= reroute_route_idx < len(ROUTE_NAMES):
                final_route = ROUTE_NAMES[reroute_route_idx]
        elif is_fallback:
            final_route = "fallback_unknown"
        path_names = _route_path_names(reroute_path, idx)
        record = {
            "sample_id": sample_id,
            "source_name": str(epoch_metrics["source_names"][idx]),
            "label_name_true": true_label,
            "coarse_label_true": str(epoch_metrics.get("coarse_label_names", epoch_metrics["label_names"])[idx]),
            "fine_label_true": str(epoch_metrics.get("fine_label_names", epoch_metrics["label_names"])[idx]),
            "label_name_pred": pred_label,
            "label_name_pred_without_aircraft_protection": pred_without_protection,
            "route_name_true": true_route,
            "route_name_router_pred": router_route,
            "route_name_pred": router_route,
            "route_name_final": final_route,
            "reroute_path": path_names,
            "is_correct": pred_label == true_label,
            "is_route_correct": router_route == true_route,
            "is_router_route_correct": router_route == true_route,
            "is_final_route_correct": final_route == true_route,
            "aircraft_protection_changed_prediction": pred_label != pred_without_protection,
        }
        for diag_name in (
            "aircraft_similarity",
            "aircraft_threshold",
            "aircraft_margin",
            "aircraft_confidence",
            "aircraft_source_tau_low",
            "aircraft_source_override",
            "aircraft_source_reject",
            "conventional_route_weight",
            "conventional_gate",
            "adjusted_conventional_gate",
            "expert2_positive",
            "aircraft_high_confidence",
            "aircraft_mid_confidence",
            "groundness_probability",
            "groundness_threshold",
            "groundness_confidence",
            "ground_gate",
            "adjusted_ground_gate",
            "ground_high_confidence",
            "ground_mid_confidence",
        ):
            if isinstance(diagnostics, dict) and diag_name in diagnostics:
                value = diagnostics[diag_name][idx]
                if value.dtype == torch.bool:
                    record[diag_name] = bool(value.item())
                else:
                    record[diag_name] = float(value.item())
        for diag_name in (
            "reroute_count",
            "reroute_triggered",
            "reroute_fallback",
            "reroute_selected_route",
            "reroute_initial_route",
            "reroute_total_penalty",
            "reroute_low_altitude_deep_used",
            "reroute_low_altitude_deep_accepted",
            "reroute_accept_confidence",
            "reroute_reject_reason_code",
            "low_reject_score",
            "low_unknown_bonus",
            "low_confusion_pair_bonus",
            "low_evidence_conflict_bonus",
            "low_margin_bonus",
            "low_agreement_confidence_credit",
            "low_high_risk_pair",
            "low_final_margin",
            "low_traj_pred",
            "low_rd_pred",
        ):
            if isinstance(reroute_diagnostics, dict) and diag_name in reroute_diagnostics:
                value = reroute_diagnostics[diag_name][idx]
                if value.dtype == torch.bool:
                    record[diag_name] = bool(value.item())
                else:
                    record[diag_name] = (
                        int(value.item())
                        if "count" in diag_name or "route" in diag_name or "code" in diag_name
                        else float(value.item())
                    )
                if diag_name == "reroute_reject_reason_code":
                    record["reroute_reject_reason"] = REROUTE_REASON_NAMES.get(int(value.item()), f"unknown_{int(value.item())}")
        records.append(record)
    return records


def _expert_internal_report(epoch_metrics: dict[str, object], model: nn.Module | None = None) -> dict[str, object]:
    class_labels = epoch_metrics["class_labels"]
    route_labels = epoch_metrics["route_labels"]
    expert_outputs = epoch_metrics.get("expert_outputs", {})
    if not isinstance(expert_outputs, dict):
        expert_outputs = {}
    return {
        "expert1_low_altitude": _expert1_branch_report(expert_outputs, class_labels),
        "expert2_conventional_air": _expert2_internal_report(expert_outputs, class_labels),
        "expert3_high_altitude_rule": _expert3_rule_report(expert_outputs, class_labels, route_labels),
        "expert4_ground": _expert4_internal_report(expert_outputs, class_labels),
        "rd_prototype_similarity": _rd_prototype_similarity(model),
    }


def _attach_reroute_outcomes_to_predictions(
    evaluation_report: dict[str, object],
    reroute_comparison: dict[str, object],
) -> None:
    outcomes = {
        str(item.get("sample_id")): item
        for item in reroute_comparison.get("sample_outcomes", [])
        if isinstance(item, dict) and item.get("sample_id") is not None
    }
    predictions = evaluation_report.get("predictions", [])
    if not isinstance(predictions, list):
        return
    for record in predictions:
        if not isinstance(record, dict):
            continue
        outcome = outcomes.get(str(record.get("sample_id")))
        record["reroute_helped"] = bool(outcome.get("reroute_helped", False)) if outcome else False
        record["reroute_harmed"] = bool(outcome.get("reroute_harmed", False)) if outcome else False
        record["reroute_prediction_changed"] = bool(outcome.get("prediction_changed", False)) if outcome else False
        if outcome:
            record["without_reroute_pred"] = outcome.get("without_pred")
            record["with_reroute_pred"] = outcome.get("with_pred")


def _build_evaluation_report(epoch_metrics: dict[str, object], model: nn.Module | None = None) -> dict[str, object]:
    class_logits = epoch_metrics["class_logits"]
    class_labels = epoch_metrics["class_labels"]
    route_logits = epoch_metrics["route_logits"]
    route_labels = epoch_metrics["route_labels"]
    aircraft_diagnostics = epoch_metrics.get("aircraft_diagnostics", {})
    class_logits_without_aircraft_protection = epoch_metrics.get(
        "class_logits_without_aircraft_protection",
        class_logits,
    )
    per_class = compute_per_class_metrics(class_logits, class_labels, DEFAULT_VOCAB.labels)
    confusion = compute_confusion_matrix(class_logits, class_labels, len(DEFAULT_VOCAB.labels))
    critical_metrics = _critical_metric_summary(
        class_logits,
        class_labels,
        route_logits,
        route_labels,
        aircraft_diagnostics=aircraft_diagnostics,
        reroute_diagnostics=epoch_metrics.get("reroute_diagnostics", {}),
        class_logits_without_aircraft_protection=class_logits_without_aircraft_protection,
        source_report=epoch_metrics.get("source_report"),
        source_names=epoch_metrics.get("source_names"),
    )
    return {
        "summary": _metric_summary(epoch_metrics),
        "label_names": DEFAULT_VOCAB.labels,
        "classification_confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "label_name": item.label_name,
                "support": item.support,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in per_class
        ],
        "route": _route_report(route_logits, route_labels),
        "critical_metrics": critical_metrics,
        "aircraft_arbitration": critical_metrics["aircraft"],
        "expert1_low_altitude": _expert1_scope_report(class_logits, class_labels, route_logits, route_labels),
        "expert_internal": _expert_internal_report(epoch_metrics, model=model),
        "fine_label_distribution": _fine_label_distribution(epoch_metrics.get("fine_label_names", [])),
        "by_source": epoch_metrics.get("source_report") or _source_report(class_logits, class_labels, epoch_metrics["source_names"]),
        "predictions": _prediction_records(epoch_metrics),
    }


def _evaluate_with_fusion_mode(
    model: nn.Module,
    loader,
    class_criterion: nn.Module,
    route_criterion: nn.Module,
    low_criterion: nn.Module,
    optimizer,
    device: torch.device,
    config: SystemTrainConfig,
    fusion_mode: str,
    source_feature_stats: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, object]:
    mode_config = replace(config, fusion_mode=fusion_mode)
    metrics = _run_epoch(
        model,
        loader,
        class_criterion,
        route_criterion,
        low_criterion,
        optimizer,
        device,
        False,
        config.skip_non_finite_batch,
        mode_config,
        stage_name="finetune",
        stage_epoch=max(1, config.joint_loss_ramp_epochs),
        source_feature_stats=source_feature_stats,
    )
    selection_score, selection_components = _system_selection_score(metrics)
    metrics["selection_score"] = selection_score
    metrics["selection_components"] = selection_components
    metrics["fusion_mode"] = fusion_mode
    return metrics


def _fusion_probe_summary(metrics: dict[str, object]) -> dict[str, object]:
    critical = metrics.get("critical_metrics", {})
    aircraft = critical.get("aircraft", {}) if isinstance(critical, dict) else {}
    ground = critical.get("ground", {}) if isinstance(critical, dict) else {}
    return {
        "fusion_mode": metrics.get("fusion_mode"),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "selection_score": float(metrics.get("selection_score", 0.0)),
        "route_macro_recall": float(metrics.get("route_macro_recall", 0.0)),
        "ground_route_recall": float(critical.get("ground_route_recall", 0.0)),
        "expert1_low_altitude_macro_f1": float(critical.get("expert1_low_altitude_macro_f1", 0.0)),
        "drone_bird_recall_mean": float(
            critical.get("drone_bird", {}).get("drone_bird_recall_mean", 0.0)
        ),
        "unknown_recall": float(critical.get("unknown_recall", 0.0)),
        "clutter_recall": float(critical.get("clutter_recall", 0.0)),
        "aircraft_system_recall": float(critical.get("aircraft_system_recall", 0.0)),
        "aircraft_system_precision": float(critical.get("aircraft_system_precision", 0.0)),
        "aircraft_system_f1": float(critical.get("aircraft_system_f1", 0.0)),
        "aircraft_recall_without_aircraft_protection": float(
            critical.get("aircraft_recall_without_aircraft_protection", 0.0)
        ),
        "expert2_positive_system_not_aircraft_count": int(
            critical.get("expert2_positive_system_not_aircraft_count", 0)
        ),
        "non_aircraft_as_aircraft_rate": float(critical.get("non_aircraft_as_aircraft_rate", 0.0)),
        "source_balanced_macro_f1": float(critical.get("source_balanced_macro_f1", 0.0)),
        "min_source_macro_f1": float(critical.get("min_source_macro_f1", 0.0)),
        "ground_group_f1": float(critical.get("ground_group_f1", 0.0)),
        "pred_ground_on_non_ground_count": int(critical.get("pred_ground_on_non_ground_count", 0)),
        "groundness_probability_mean_true_ground": float(
            ground.get("groundness_probability_mean_true_ground", 0.0)
        ) if isinstance(ground, dict) else 0.0,
        "groundness_probability_mean_non_ground": float(
            ground.get("groundness_probability_mean_non_ground", 0.0)
        ) if isinstance(ground, dict) else 0.0,
        "groundness_confidence_mean_true_ground": float(
            ground.get("groundness_confidence_mean_true_ground", 0.0)
        ) if isinstance(ground, dict) else 0.0,
        "groundness_confidence_mean_non_ground": float(
            ground.get("groundness_confidence_mean_non_ground", 0.0)
        ) if isinstance(ground, dict) else 0.0,
        "ground_high_confidence_count": int(
            ground.get("ground_high_confidence_count", 0)
        ) if isinstance(ground, dict) else 0,
        "aircraft_protection_recovered_count": int(
            aircraft.get("aircraft_protection_recovered_count", 0)
        ) if isinstance(aircraft, dict) else 0,
        "reroute_rate": float(critical.get("reroute_rate", 0.0)),
        "reroute_recovery_rate": float(critical.get("reroute_recovery_rate", 0.0)),
        "reroute_fallback_rate": float(critical.get("reroute_fallback_rate", 0.0)),
        "low_altitude_deep_used_rate": float(
            critical.get("reroute", {}).get("low_altitude_deep_used_rate", 0.0)
        ) if isinstance(critical.get("reroute", {}), dict) else 0.0,
    }


def train_system_v1(config: SystemTrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    run_paths = prepare_run_output_dir(config.output_dir)
    output_dir = run_paths.output_dir
    device, device_info = resolve_device(config.device)

    loaders = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    dataset_frame = TrackDataset(config.data_root, trainable_only=False, index_name="system_trainable_track_index.csv").to_frame()
    model = RadarClassificationSystem(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=config.hidden_dim,
        dropout=config.dropout,
        low_altitude_cosface_scale=config.low_altitude_cosface_scale,
        low_altitude_cosface_margin=config.low_altitude_cosface_margin,
        low_altitude_reject_threshold=config.low_altitude_reject_threshold,
        low_altitude_rd_activation_threshold=config.low_altitude_rd_activation_threshold,
        low_altitude_use_rd_patch_cnn=config.low_altitude_use_rd_patch_cnn,
        low_altitude_initial_rd_gate_bias=config.low_altitude_initial_rd_gate_bias,
        low_altitude_rd_class_activation_threshold=config.low_altitude_rd_class_activation_threshold,
        rd_bird_logit_scale=config.rd_bird_logit_scale,
        rd_unknown_logit_scale=config.rd_unknown_logit_scale,
        rd_clutter_logit_scale=config.rd_clutter_logit_scale,
        rd_residual_logit_scale=config.rd_residual_logit_scale,
        rd_unknown_clutter_logit_scale=config.rd_unknown_clutter_logit_scale,
        rd_clutter_morphology_logit_scale=config.rd_clutter_morphology_logit_scale,
    ).to(device)
    pretrained_track_info = None
    pretrained_router_info = None
    expert1_info = None
    expert2_info = None
    expert4_info = None
    if config.pretrained_track_checkpoint:
        pretrained_track_info = _load_pretrained_track_encoder(model, config.pretrained_track_checkpoint, device)
    if config.pretrained_router_checkpoint:
        pretrained_router_info = _load_submodule_checkpoint(model.router, config.pretrained_router_checkpoint, device)
    if config.expert1_checkpoint:
        expert1_info = _load_expert1_bundle(model, config.expert1_checkpoint, device)
    if config.expert2_checkpoint:
        expert2_info = _load_expert_bundle(
            model,
            config.expert2_checkpoint,
            device,
            "expert2_air",
        )
    if config.expert4_checkpoint:
        expert4_info = _load_expert_bundle(
            model,
            config.expert4_checkpoint,
            device,
            "expert4_ground",
        )
    if config.freeze_router:
        _set_router_trainable(model, trainable=False)
    trainable_parameters = _count_trainable_parameters(model)
    class_weights = _build_class_weights(loaders).to(device)
    route_weights = _build_route_weights(loaders).to(device)
    low_class_weights = _build_low_altitude_weights(loaders).to(device)
    groundness_class_weights = _build_groundness_weights(loaders).to(device)
    source_feature_stats = None
    checkpoint_source_feature_stats = None
    if config.source_normalization:
        fitted_source_feature_stats = _fit_source_feature_stats(loaders.train_loader)
        checkpoint_source_feature_stats = _merge_source_feature_stats(
            _source_feature_stats_from_checkpoint(config.expert1_checkpoint, device) if config.expert1_checkpoint else None,
            _source_feature_stats_from_checkpoint(config.expert2_checkpoint, device) if config.expert2_checkpoint else None,
            _source_feature_stats_from_checkpoint(config.expert4_checkpoint, device) if config.expert4_checkpoint else None,
            _source_feature_stats_from_checkpoint(config.pretrained_track_checkpoint, device) if config.pretrained_track_checkpoint else None,
        )
        source_feature_stats = _merge_source_feature_stats(fitted_source_feature_stats, checkpoint_source_feature_stats)
    optimizer = _build_optimizer(model, config)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    route_criterion = nn.CrossEntropyLoss(weight=route_weights)
    low_criterion = nn.CrossEntropyLoss(weight=low_class_weights)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {trainable_parameters:,}")

    history: list[dict[str, object]] = []
    best_metric = float("-inf")
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        if epoch <= config.stage_warmup_epochs:
            stage_name = "warmup"
            stage_epoch = epoch
        elif config.stage_expert_warmup_epochs > 0 and epoch <= config.stage_warmup_epochs + config.stage_expert_warmup_epochs:
            stage_name = "expert_warmup"
            stage_epoch = epoch - config.stage_warmup_epochs
        elif config.stage_joint_epochs > 0 and epoch <= config.stage_warmup_epochs + config.stage_expert_warmup_epochs + config.stage_joint_epochs:
            stage_name = "joint"
            stage_epoch = epoch - config.stage_warmup_epochs - config.stage_expert_warmup_epochs
        else:
            stage_name = "finetune"
            stage_epoch = epoch - config.stage_warmup_epochs - config.stage_expert_warmup_epochs - config.stage_joint_epochs
        if stage_name == "warmup":
            _set_router_only_warmup(model, router_only=config.freeze_non_router_in_warmup)
        elif stage_name == "expert_warmup":
            if config.freeze_non_low_altitude_in_expert_warmup:
                _set_low_altitude_only_expert_warmup(
                    model,
                    freeze_router=config.freeze_router_in_expert_warmup,
                    freeze_track_encoder=config.freeze_track_encoder_in_expert_warmup,
                )
            else:
                _set_expert_warmup_trainable(
                    model,
                    freeze_router=config.freeze_router_in_expert_warmup,
                    freeze_track_encoder=config.freeze_track_encoder_in_expert_warmup,
                )
        else:
            _set_router_only_warmup(model, router_only=False)
        if config.freeze_router:
            _set_router_trainable(model, trainable=False)
        ground_feature_trainable = epoch > config.freeze_ground_feature_epochs
        _set_ground_feature_stack_trainable(model, trainable=ground_feature_trainable)
        _set_ground_expert_learning_rates(optimizer, config, epoch)
        track_encoder_trainable = epoch > config.freeze_track_encoder_epochs
        if stage_name == "warmup" and config.freeze_non_router_in_warmup:
            track_encoder_trainable = False
        if stage_name == "expert_warmup" and config.freeze_track_encoder_in_expert_warmup:
            track_encoder_trainable = False
        _set_track_encoder_trainable(
            model,
            trainable=track_encoder_trainable,
        )
        train_metrics = _run_epoch(
            model,
            loaders.train_loader,
            class_criterion,
            route_criterion,
            low_criterion,
            optimizer,
            device,
            True,
            config.skip_non_finite_batch,
            config,
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            source_feature_stats=source_feature_stats,
            groundness_class_weights=groundness_class_weights,
            detach_ground_features=not ground_feature_trainable,
        )
        val_metrics = _run_epoch(
            model,
            loaders.val_loader,
            class_criterion,
            route_criterion,
            low_criterion,
            optimizer,
            device,
            False,
            config.skip_non_finite_batch,
            config,
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            source_feature_stats=source_feature_stats,
            groundness_class_weights=groundness_class_weights,
            detach_ground_features=False,
        )
        selection_score, selection_components = _system_selection_score(val_metrics)
        val_metrics["selection_score"] = selection_score
        val_metrics["selection_components"] = selection_components
        record = {
            "epoch": epoch,
            "stage": stage_name,
            "train": _metric_summary(train_metrics),
            "val": _metric_summary(val_metrics),
        }
        history.append(record)
        if not train_metrics.get("has_valid_batches", True) or not val_metrics.get("has_valid_batches", True):
            break
        if selection_score > best_metric:
            best_metric = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "labels": DEFAULT_VOCAB.labels,
                    "routes": ROUTE_NAMES,
                    "source_feature_stats": source_feature_stats,
                },
                best_path,
            )

    if not best_path.exists():
        raise RuntimeError("training finished without a valid checkpoint; all epochs became unstable")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    candidate_modes = list(dict.fromkeys([config.fusion_mode] + list(config.eval_fusion_candidate_modes)))
    val_fusion_metrics = [
        _evaluate_with_fusion_mode(
            model,
            loaders.val_loader,
            class_criterion,
            route_criterion,
            low_criterion,
            optimizer,
            device,
            config,
            fusion_mode,
            source_feature_stats,
        )
        for fusion_mode in candidate_modes
    ]
    if config.fusion_mode_selection_policy == "validation_score":
        selected_val_metrics = max(val_fusion_metrics, key=lambda item: float(item["selection_score"]))
        selected_fusion_mode = str(selected_val_metrics["fusion_mode"])
    elif config.fusion_mode_selection_policy == "configured":
        selected_fusion_mode = str(config.fusion_mode)
        selected_val_metrics = next(
            (metrics for metrics in val_fusion_metrics if str(metrics["fusion_mode"]) == selected_fusion_mode),
            val_fusion_metrics[0],
        )
    else:
        raise ValueError(f"unsupported fusion_mode_selection_policy: {config.fusion_mode_selection_policy}")
    test_fusion_metrics = [
        _evaluate_with_fusion_mode(
            model,
            loaders.test_loader,
            class_criterion,
            route_criterion,
            low_criterion,
            optimizer,
            device,
            config,
            fusion_mode,
            source_feature_stats,
        )
        for fusion_mode in candidate_modes
    ]
    test_metrics_by_mode = {
        str(metrics["fusion_mode"]): metrics
        for metrics in test_fusion_metrics
    }
    test_metrics = test_metrics_by_mode[selected_fusion_mode]
    best_test_probe_metrics = max(
        test_fusion_metrics,
        key=lambda item: (float(item.get("macro_f1", 0.0)), float(item.get("accuracy", 0.0))),
    )
    without_reroute_mode = "soft_log_weighted" if "soft_log_weighted" in test_metrics_by_mode else selected_fusion_mode
    with_reroute_mode = "reroute" if "reroute" in test_metrics_by_mode else selected_fusion_mode
    reroute_comparison = _reroute_comparison(
        test_metrics_by_mode[without_reroute_mode],
        test_metrics_by_mode[with_reroute_mode],
    )
    reroute_comparison["without_reroute_mode"] = without_reroute_mode
    reroute_comparison["with_reroute_mode"] = with_reroute_mode
    reroute_config = asdict(_system_reroute_config(config))
    evaluation_report = _build_evaluation_report(test_metrics, model=model)
    evaluation_report["device"] = device_info
    evaluation_report["selected_fusion_mode"] = selected_fusion_mode
    evaluation_report["fusion_mode_selection_policy"] = config.fusion_mode_selection_policy
    evaluation_report["reroute_config"] = reroute_config
    evaluation_report["fusion_validation_probe"] = [
        _fusion_probe_summary(metrics)
        for metrics in val_fusion_metrics
    ]
    evaluation_report["fusion_test_probe"] = [
        _fusion_probe_summary(metrics)
        for metrics in test_fusion_metrics
    ]
    evaluation_report["best_test_probe"] = _fusion_probe_summary(best_test_probe_metrics)
    evaluation_report["reroute_comparison"] = reroute_comparison
    _attach_reroute_outcomes_to_predictions(evaluation_report, reroute_comparison)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = output_dir / "training_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_ids": loaders.split.train_ids,
                "val_ids": loaders.split.val_ids,
                "test_ids": loaders.split.test_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "best_checkpoint": best_path.as_posix(),
        "history_file": history_path.as_posix(),
        "manifest_file": manifest_path.as_posix(),
        "evaluation_report": report_path.as_posix(),
        "test_metrics": _metric_summary(test_metrics),
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "taxonomy": DEFAULT_VOCAB.label_to_index,
        "trainable_parameters": trainable_parameters,
        "source_normalization": bool(config.source_normalization),
        "source_feature_stats_saved": bool(source_feature_stats is not None),
        "source_feature_stats_from_checkpoint": bool(checkpoint_source_feature_stats),
        "pretrained_track_checkpoint": config.pretrained_track_checkpoint,
        "pretrained_track_info": pretrained_track_info,
        "pretrained_router_checkpoint": config.pretrained_router_checkpoint,
        "pretrained_router_info": pretrained_router_info,
        "expert1_checkpoint": config.expert1_checkpoint,
        "expert1_info": expert1_info,
        "expert2_checkpoint": config.expert2_checkpoint,
        "expert2_info": expert2_info,
        "expert4_checkpoint": config.expert4_checkpoint,
        "expert4_info": expert4_info,
        "fusion_mode": config.fusion_mode,
        "fusion_mode_selection_policy": config.fusion_mode_selection_policy,
        "selected_fusion_mode": selected_fusion_mode,
        "fusion_validation_probe": [
            _fusion_probe_summary(metrics)
            for metrics in val_fusion_metrics
        ],
        "fusion_test_probe": [
            _fusion_probe_summary(metrics)
            for metrics in test_fusion_metrics
        ],
        "best_test_probe": _fusion_probe_summary(best_test_probe_metrics),
        "reroute_comparison": reroute_comparison,
        "reroute_config": reroute_config,
        "freeze_router": bool(config.freeze_router),
        "low_altitude_use_rd_patch_cnn": bool(config.low_altitude_use_rd_patch_cnn),
        "best_selection_score": best_metric,
        "best_selection_policy": "macro_f1 + route/low-altitude/Drone-Bird/Unknown-Clutter/Ground/Aircraft arbitration penalties",
        "aircraft_protection_enabled": bool(config.aircraft_protection_enabled),
        "aircraft_confidence_tau_high": config.aircraft_confidence_tau_high,
        "aircraft_confidence_tau_low": config.aircraft_confidence_tau_low,
        "aircraft_source_confidence_tau_low": dict(config.aircraft_source_confidence_tau_low),
        "aircraft_source_suppress_below_tau": bool(config.aircraft_source_suppress_below_tau),
        "aircraft_source_suppress_logit": config.aircraft_source_suppress_logit,
        "aircraft_conventional_gate_floor": config.aircraft_conventional_gate_floor,
        "aircraft_temperature": config.aircraft_temperature,
        "groundness_protection_enabled": bool(config.groundness_protection_enabled),
        "groundness_confidence_tau_high": config.groundness_confidence_tau_high,
        "groundness_confidence_tau_low": config.groundness_confidence_tau_low,
        "ground_route_gate_floor": config.ground_route_gate_floor,
        "groundness_temperature": config.groundness_temperature,
        "calibrated_route_prior_weight": config.calibrated_route_prior_weight,
        "calibrated_low_membership_beta": config.calibrated_low_membership_beta,
        "calibrated_aircraft_membership_beta": config.calibrated_aircraft_membership_beta,
        "calibrated_ground_membership_beta": config.calibrated_ground_membership_beta,
        "calibrated_low_temperature": config.calibrated_low_temperature,
        "calibrated_ground_suppression": config.calibrated_ground_suppression,
        "calibrated_aircraft_min_confidence": config.calibrated_aircraft_min_confidence,
        "calibrated_aircraft_min_route_weight": config.calibrated_aircraft_min_route_weight,
        "calibrated_aircraft_suppress_logit": config.calibrated_aircraft_suppress_logit,
        "ground_expert_lr_multiplier": config.ground_expert_lr_multiplier,
        "freeze_ground_feature_epochs": config.freeze_ground_feature_epochs,
        "system_groundness_loss_weight": config.system_groundness_loss_weight,
        "system_groundness_separation_loss_weight": config.system_groundness_separation_loss_weight,
        "system_groundness_positive_target": config.system_groundness_positive_target,
        "system_groundness_negative_target": config.system_groundness_negative_target,
        "freeze_track_encoder_epochs": config.freeze_track_encoder_epochs,
        "sequence_encoder_lr_multiplier": config.sequence_encoder_lr_multiplier,
        "sequence_encoder_grad_clip_norm": config.sequence_encoder_grad_clip_norm,
        "global_grad_clip_norm": config.global_grad_clip_norm,
        "freeze_non_router_in_warmup": config.freeze_non_router_in_warmup,
        "stage_warmup_epochs": config.stage_warmup_epochs,
        "stage_expert_warmup_epochs": config.stage_expert_warmup_epochs,
        "stage_joint_epochs": config.stage_joint_epochs,
        "joint_loss_ramp_epochs": config.joint_loss_ramp_epochs,
        "freeze_router_in_expert_warmup": config.freeze_router_in_expert_warmup,
        "freeze_track_encoder_in_expert_warmup": config.freeze_track_encoder_in_expert_warmup,
        "freeze_non_low_altitude_in_expert_warmup": config.freeze_non_low_altitude_in_expert_warmup,
        "stage_expert_warmup_route_loss_scale": config.stage_expert_warmup_route_loss_scale,
        "stage_expert_warmup_low_altitude_aux_scale": config.stage_expert_warmup_low_altitude_aux_scale,
        "stage_expert_warmup_low_altitude_mutual_scale": config.stage_expert_warmup_low_altitude_mutual_scale,
        "stage_expert_warmup_low_altitude_contrastive_scale": config.stage_expert_warmup_low_altitude_contrastive_scale,
        "stage_expert_warmup_consistency_scale": config.stage_expert_warmup_consistency_scale,
        "stage_finetune_route_loss_scale": config.stage_finetune_route_loss_scale,
        "stage_finetune_low_altitude_aux_scale": config.stage_finetune_low_altitude_aux_scale,
        "stage_finetune_low_altitude_mutual_scale": config.stage_finetune_low_altitude_mutual_scale,
        "stage_finetune_low_altitude_contrastive_scale": config.stage_finetune_low_altitude_contrastive_scale,
        "stage_finetune_consistency_scale": config.stage_finetune_consistency_scale,
        "enable_low_altitude_reject_in_train": bool(config.enable_low_altitude_reject_in_train),
        "enable_low_altitude_reject_in_eval": bool(config.enable_low_altitude_reject_in_eval),
        "low_altitude_rd_activation_threshold": config.low_altitude_rd_activation_threshold,
        "low_altitude_initial_rd_gate_bias": config.low_altitude_initial_rd_gate_bias,
        "consistency_loss_weight": config.consistency_loss_weight,
        "class_weights": class_weights.detach().cpu().tolist(),
        "route_weights": route_weights.detach().cpu().tolist(),
        "route_coarse_loss_weight": config.route_coarse_loss_weight,
        "low_altitude_class_weights": low_class_weights.detach().cpu().tolist(),
        "groundness_class_weights": groundness_class_weights.detach().cpu().tolist(),
        "split_sizes": {
            "train": len(loaders.split.train_ids),
            "val": len(loaders.split.val_ids),
            "test": len(loaders.split.test_ids),
        },
        "rd_coverage": {
            "train": _rd_coverage_from_ids(dataset_frame, loaders.split.train_ids),
            "val": _rd_coverage_from_ids(dataset_frame, loaders.split.val_ids),
            "test": _rd_coverage_from_ids(dataset_frame, loaders.split.test_ids),
        },
    }
    report_paths = write_real_experiment_report(
        output_dir=output_dir,
        evaluation_report=evaluation_report,
        summary=summary,
        timestamp_label=run_paths.timestamp_label,
    )
    summary["real_report_markdown"] = report_paths.markdown_path.as_posix()
    summary["real_report_json"] = report_paths.json_path.as_posix()
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
