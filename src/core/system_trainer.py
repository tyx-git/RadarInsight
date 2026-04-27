from __future__ import annotations

import json
import random
from dataclasses import asdict
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
from .system import RadarClassificationSystem
from .system_config import SystemTrainConfig
from .system_metrics import compute_confusion_matrix, compute_per_class_metrics, compute_system_metrics
from .physical import PHYSICAL_FEATURE_COLUMNS


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


def _build_optimizer(model: nn.Module, config: SystemTrainConfig) -> torch.optim.Optimizer:
    sequence_encoder_param_ids = {
        id(param)
        for param in model.representation_builder.sequence_encoder.parameters()
    }
    sequence_encoder_params = [
        param
        for param in model.representation_builder.sequence_encoder.parameters()
        if param.requires_grad
    ]
    other_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in sequence_encoder_param_ids
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
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)


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


def _project_to_global_logits(
    batch_labels: torch.Tensor,
    system_output,
    num_classes: int,
    device: torch.device,
    fusion_mode: str,
) -> torch.Tensor:
    logits = torch.full((batch_labels.shape[0], num_classes), fill_value=-6.0, device=device)
    global_index = DEFAULT_VOCAB.label_to_index
    if fusion_mode not in {"ungated", "hard_route", "soft_log_weighted", "soft_top2"}:
        raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
    if fusion_mode == "ungated":
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

    low = system_output.low_altitude_logits + low_gate
    logits[:, global_index["DroneTarget"]] = low[:, 0]
    logits[:, global_index["BirdTarget"]] = low[:, 1]
    logits[:, global_index["BalloonTarget"]] = low[:, 2]
    logits[:, global_index["ClutterTarget"]] = low[:, 3]
    logits[:, global_index["UnknownTarget"]] = low[:, 4]

    conv = system_output.conventional_air_logits + conventional_gate
    aircraft_column = 1 if conv.shape[1] > 1 else 0
    high = system_output.high_altitude_logits + high_gate
    high_aircraft_column = 1 if high.shape[1] > 1 else 0
    logits[:, global_index["AircraftTarget"]] = torch.maximum(
        conv[:, aircraft_column],
        high[:, high_aircraft_column],
    )

    ground = system_output.ground_logits + ground_gate
    logits[:, global_index["PeopleTarget"]] = ground[:, 0]
    logits[:, global_index["CarTarget"]] = ground[:, 1]
    logits[:, global_index["ShipTarget"]] = ground[:, 2]
    return logits


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


def _run_epoch(model, loader, class_criterion, route_criterion, low_criterion, optimizer, device: torch.device, train: bool, skip_non_finite_batch: bool, config: SystemTrainConfig, stage_name: str, stage_epoch: int | None = None, source_feature_stats: dict[str, dict[str, torch.Tensor]] | None = None) -> dict[str, object]:
    model.train(mode=train)
    class_logits_all: list[torch.Tensor] = []
    class_labels_all: list[torch.Tensor] = []
    route_logits_all: list[torch.Tensor] = []
    route_labels_all: list[torch.Tensor] = []
    loss_values: list[float] = []
    sample_ids_all: list[str] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    route_names_all: list[str] = []
    skipped_non_finite_batches = 0

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
                enable_low_altitude_reject=(
                    config.enable_low_altitude_reject_in_train
                    if train
                    else config.enable_low_altitude_reject_in_eval
                ),
            )
            class_logits = _project_to_global_logits(labels, output, len(DEFAULT_VOCAB.labels), device, config.fusion_mode)
            class_loss = class_criterion(class_logits, labels)
            route_loss = route_criterion(output.routing.route_logits, route_labels)
            scales = _stage_loss_scales(stage_name, config, stage_epoch=stage_epoch)
            loss = class_loss + (config.route_loss_weight * scales["route"]) * route_loss

            low_aux_loss = torch.tensor(0.0, device=device)
            low_mutual_loss = torch.tensor(0.0, device=device)
            low_contrastive_loss = torch.tensor(0.0, device=device)
            consistency_loss = torch.tensor(0.0, device=device)
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
        class_labels_all.append(labels.detach().cpu())
        route_logits_all.append(output.routing.route_logits.detach().cpu())
        route_labels_all.append(route_labels.detach().cpu())
        sample_ids_all.extend(batch.sample_ids)
        source_names_all.extend(batch.source_names)
        label_names_all.extend(batch.label_names)
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
            "sample_ids": [],
            "source_names": [],
            "label_names": [],
            "route_names": [],
            "skipped_non_finite_batches": skipped_non_finite_batches,
            "has_valid_batches": False,
            "loss_components": {},
            "critical_metrics": {},
        }

    class_logits_cat = torch.cat(class_logits_all, dim=0)
    class_labels_cat = torch.cat(class_labels_all, dim=0)
    route_logits_cat = torch.cat(route_logits_all, dim=0)
    route_labels_cat = torch.cat(route_labels_all, dim=0)
    metrics = compute_system_metrics(
        class_logits=class_logits_cat,
        class_labels=class_labels_cat,
        route_logits=route_logits_cat,
        route_labels=route_labels_cat,
        num_classes=len(DEFAULT_VOCAB.labels),
    )
    critical_metrics = _critical_metric_summary(
        class_logits_cat,
        class_labels_cat,
        route_logits_cat,
        route_labels_cat,
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
        "class_labels": class_labels_cat,
        "route_logits": route_logits_cat,
        "route_labels": route_labels_cat,
        "sample_ids": sample_ids_all,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "route_names": route_names_all,
        "skipped_non_finite_batches": skipped_non_finite_batches,
        "has_valid_batches": True,
        "critical_metrics": critical_metrics,
        "loss_components": {
            "class_loss": float(sum(class_loss_values) / len(class_loss_values)),
            "route_loss": float(sum(route_loss_values) / len(route_loss_values)),
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
        report[source_name] = {
            "count": len(indices),
            "accuracy": accuracy,
        }
    return report


def _label_recall(class_logits: torch.Tensor, class_labels: torch.Tensor, label_name: str) -> float:
    label_idx = DEFAULT_VOCAB.label_to_index[label_name]
    mask = class_labels == label_idx
    if not bool(mask.any()):
        return 0.0
    preds = class_logits.argmax(dim=-1)
    return float((preds[mask] == label_idx).float().mean().item())


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


def _critical_metric_summary(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    route_logits: torch.Tensor,
    route_labels: torch.Tensor,
) -> dict[str, object]:
    route = _route_report(route_logits, route_labels)
    per_route_recall = {
        item["route_name"]: item["recall"]
        for item in route["per_route"]
    }
    low = _low_altitude_report(class_logits, class_labels)
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
    score = (
        macro_f1
        + 0.20 * route_macro
        + 0.25 * low_macro
        + 0.15 * drone_bird
        + 0.10 * (unknown_recall + clutter_recall)
        + 0.10 * ground_route
    )
    if ground_route == 0.0:
        score -= 0.10
    if unknown_recall == 0.0:
        score -= 0.05
    if clutter_recall == 0.0:
        score -= 0.05
    return score, {
        "macro_f1": macro_f1,
        "route_macro_recall": route_macro,
        "expert1_low_altitude_macro_f1": low_macro,
        "drone_bird_recall_mean": drone_bird,
        "unknown_recall": unknown_recall,
        "clutter_recall": clutter_recall,
        "ground_route_recall": ground_route,
    }


def _prediction_records(epoch_metrics: dict[str, object]) -> list[dict[str, object]]:
    class_logits = epoch_metrics["class_logits"]
    route_logits = epoch_metrics["route_logits"]
    class_preds = class_logits.argmax(dim=-1).tolist()
    route_preds = route_logits.argmax(dim=-1).tolist()
    records: list[dict[str, object]] = []
    for idx, sample_id in enumerate(epoch_metrics["sample_ids"]):
        true_label = str(epoch_metrics["label_names"][idx])
        pred_label = DEFAULT_VOCAB.decode(int(class_preds[idx]))
        true_route = str(epoch_metrics["route_names"][idx])
        pred_route = ROUTE_NAMES[int(route_preds[idx])]
        records.append(
            {
                "sample_id": sample_id,
                "source_name": str(epoch_metrics["source_names"][idx]),
                "label_name_true": true_label,
                "label_name_pred": pred_label,
                "route_name_true": true_route,
                "route_name_pred": pred_route,
                "is_correct": pred_label == true_label,
                "is_route_correct": pred_route == true_route,
            }
        )
    return records


def _build_evaluation_report(epoch_metrics: dict[str, object]) -> dict[str, object]:
    class_logits = epoch_metrics["class_logits"]
    class_labels = epoch_metrics["class_labels"]
    route_logits = epoch_metrics["route_logits"]
    route_labels = epoch_metrics["route_labels"]
    per_class = compute_per_class_metrics(class_logits, class_labels, DEFAULT_VOCAB.labels)
    confusion = compute_confusion_matrix(class_logits, class_labels, len(DEFAULT_VOCAB.labels))
    critical_metrics = _critical_metric_summary(class_logits, class_labels, route_logits, route_labels)
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
        "expert1_low_altitude": _low_altitude_report(class_logits, class_labels),
        "by_source": _source_report(class_logits, class_labels, epoch_metrics["source_names"]),
        "predictions": _prediction_records(epoch_metrics),
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
    source_feature_stats = None
    checkpoint_source_feature_stats = None
    if config.source_normalization:
        fitted_source_feature_stats = _fit_source_feature_stats(loaders.train_loader)
        checkpoint_source_feature_stats = _merge_source_feature_stats(
            _source_feature_stats_from_checkpoint(config.expert1_checkpoint, device) if config.expert1_checkpoint else None,
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
        track_encoder_trainable = epoch > config.freeze_track_encoder_epochs
        if stage_name == "warmup" and config.freeze_non_router_in_warmup:
            track_encoder_trainable = False
        if stage_name == "expert_warmup" and config.freeze_track_encoder_in_expert_warmup:
            track_encoder_trainable = False
        _set_track_encoder_trainable(
            model,
            trainable=track_encoder_trainable,
        )
        train_metrics = _run_epoch(model, loaders.train_loader, class_criterion, route_criterion, low_criterion, optimizer, device, True, config.skip_non_finite_batch, config, stage_name=stage_name, stage_epoch=stage_epoch, source_feature_stats=source_feature_stats)
        val_metrics = _run_epoch(model, loaders.val_loader, class_criterion, route_criterion, low_criterion, optimizer, device, False, config.skip_non_finite_batch, config, stage_name=stage_name, stage_epoch=stage_epoch, source_feature_stats=source_feature_stats)
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
    test_metrics = _run_epoch(
        model,
        loaders.test_loader,
        class_criterion,
        route_criterion,
        low_criterion,
        optimizer,
        device,
        False,
        config.skip_non_finite_batch,
        config,
        stage_name="finetune",
        stage_epoch=max(1, config.joint_loss_ramp_epochs),
        source_feature_stats=source_feature_stats,
    )
    test_selection_score, test_selection_components = _system_selection_score(test_metrics)
    test_metrics["selection_score"] = test_selection_score
    test_metrics["selection_components"] = test_selection_components
    evaluation_report = _build_evaluation_report(test_metrics)
    evaluation_report["device"] = device_info

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
        "freeze_router": bool(config.freeze_router),
        "low_altitude_use_rd_patch_cnn": bool(config.low_altitude_use_rd_patch_cnn),
        "best_selection_score": best_metric,
        "best_selection_policy": "macro_f1 + route/low-altitude/Drone-Bird/Unknown-Clutter/Ground penalties",
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
        "consistency_loss_weight": config.consistency_loss_weight,
        "class_weights": class_weights.detach().cpu().tolist(),
        "route_weights": route_weights.detach().cpu().tolist(),
        "low_altitude_class_weights": low_class_weights.detach().cpu().tolist(),
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
