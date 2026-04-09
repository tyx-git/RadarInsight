import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.models.fusion_classifier import FusionRadarClassifier
from src.training.config import TrainingConfig
from src.training.session_manager import build_dataset, build_session_manifest, build_session_subsets, save_session_manifest
from src.utils.logger import logger


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('weight_tensor', weight if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight_tensor,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        probabilities = torch.softmax(logits, dim=-1)
        pt = probabilities.gather(1, targets.view(-1, 1)).view(-1).clamp_min(1e-6)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * ce).mean()


def compute_classification_metrics(labels: list[int], predictions: list[int], num_classes: int) -> dict[str, float]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for expected, predicted in zip(labels, predictions):
        confusion[expected, predicted] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for index in range(num_classes):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    divisor = max(num_classes, 1)
    return {
        "accuracy": accuracy,
        "macro_precision": precision_sum / divisor,
        "macro_recall": recall_sum / divisor,
        "macro_f1": f1_sum / divisor,
    }


def apply_logit_bias(logits: torch.Tensor, logit_bias: torch.Tensor | None) -> torch.Tensor:
    if logit_bias is None:
        return logits
    return logits + logit_bias.view(1, -1)


def _score_logits_tuple(labels: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    predictions = logits.argmax(axis=1).tolist()
    metrics = compute_classification_metrics(labels.tolist(), predictions, num_classes=logits.shape[1])
    return float(metrics["accuracy"]), float(metrics["macro_f1"])


def optimize_global_logit_bias(
    labels: np.ndarray,
    logits: np.ndarray,
    search_values: np.ndarray | None = None,
    rounds: int = 3,
) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[0] == 0:
        return np.zeros((logits.shape[1] if logits.ndim == 2 else 0,), dtype=np.float32)

    bias = np.zeros((logits.shape[1],), dtype=np.float32)
    candidate_values = search_values if search_values is not None else np.linspace(-1.5, 1.5, num=31, dtype=np.float32)

    for _ in range(max(rounds, 1)):
        improved = False
        for class_index in range(logits.shape[1]):
            best_delta = 0.0
            best_score = _score_logits_tuple(labels, logits + bias.reshape(1, -1))
            for delta in candidate_values:
                candidate_bias = bias.copy()
                candidate_bias[class_index] += float(delta)
                candidate_score = _score_logits_tuple(labels, logits + candidate_bias.reshape(1, -1))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_delta = float(delta)
            if abs(best_delta) > 1e-8:
                bias[class_index] += best_delta
                improved = True
        if not improved:
            break

    return bias.astype(np.float32)


def _score_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    predictions = logits.argmax(axis=1).tolist()
    metrics = compute_classification_metrics(labels.tolist(), predictions, num_classes=logits.shape[1])
    return float(metrics["macro_f1"])


def optimize_group_logit_bias(
    labels: np.ndarray,
    logits: np.ndarray,
    fine_to_coarse_index: np.ndarray,
    search_values: np.ndarray | None = None,
    rounds: int = 2,
) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[0] == 0:
        return np.zeros((logits.shape[1] if logits.ndim == 2 else 0,), dtype=np.float32)

    unique_coarse_indices = sorted(set(int(value) for value in fine_to_coarse_index.tolist()))
    group_sizes = [int(np.sum(fine_to_coarse_index == coarse_index)) for coarse_index in unique_coarse_indices]
    if max(group_sizes, default=0) <= 1:
        return optimize_global_logit_bias(labels=labels, logits=logits, search_values=search_values, rounds=max(rounds, 2))

    bias = np.zeros((logits.shape[1],), dtype=np.float32)
    candidate_values = search_values if search_values is not None else np.linspace(-0.8, 0.8, num=17, dtype=np.float32)

    for _ in range(max(rounds, 1)):
        improved = False
        for coarse_index in unique_coarse_indices:
            member_indices = np.flatnonzero(fine_to_coarse_index == coarse_index)
            if len(member_indices) <= 1:
                continue
            for member_index in member_indices:
                best_delta = 0.0
                best_score = _score_logits(labels, logits + bias.reshape(1, -1))
                for delta in candidate_values:
                    candidate_bias = bias.copy()
                    candidate_bias[member_index] += float(delta)
                    redistribute = float(delta) / max(len(member_indices) - 1, 1)
                    for other_index in member_indices:
                        if other_index == member_index:
                            continue
                        candidate_bias[other_index] -= redistribute
                    candidate_score = _score_logits(labels, logits + candidate_bias.reshape(1, -1))
                    if candidate_score > best_score + 1e-8:
                        best_score = candidate_score
                        best_delta = float(delta)
                if abs(best_delta) > 1e-8:
                    bias[member_index] += best_delta
                    redistribute = best_delta / max(len(member_indices) - 1, 1)
                    for other_index in member_indices:
                        if other_index == member_index:
                            continue
                        bias[other_index] -= redistribute
                    improved = True
        if not improved:
            break

    return bias.astype(np.float32)


def _resolve_subset_labels(data_source: Any) -> list[int]:
    if isinstance(data_source, Subset):
        parent = data_source.dataset
        indices = data_source.indices
        return [parent.label_to_index[parent.samples[index].label_name] for index in indices]

    if isinstance(data_source, RadarFusionDataset):
        return [data_source.label_to_index[sample.label_name] for sample in data_source.samples]

    raise TypeError("Unsupported dataset type for label extraction.")


def build_class_balanced_weights(data_source: Any, num_classes: int, beta: float) -> torch.Tensor:
    labels = _resolve_subset_labels(data_source)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num[effective_num <= 0] = 1.0
    weights = (1.0 - beta) / effective_num
    positive_mask = counts > 0
    if positive_mask.any():
        fallback_weight = float(np.median(weights[positive_mask]))
    else:
        fallback_weight = 1.0
    weights[~positive_mask] = fallback_weight
    weights = weights / max(weights.sum(), 1e-8) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if epochs <= 1:
        return None

    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    cosine_epochs = max(epochs - warmup_epochs, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if warmup_epochs == 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def build_dataloaders(
    config: TrainingConfig,
    session_ids: Sequence[str] | None = None,
) -> tuple[Any, DataLoader, DataLoader | None, DataLoader | None, dict[str, list[str]]]:
    config.include_track = _env_flag("TRACK", bool(getattr(config, "include_track", True)))
    config.include_rd = _env_flag("RD", bool(config.include_rd))
    dataset = build_dataset(data_root=config.data_root, config=config, session_ids=session_ids)

    if len(dataset) == 1:
        dataset.fit_track_normalization_for_sessions(sorted(dataset.sessions.keys()))
        dataset.fit_track_global_normalization_for_sessions(sorted(dataset.sessions.keys()))
        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.num_workers,
        )
        split_map = {"train": sorted(dataset.sessions.keys()), "val": [], "test": []}
        return dataset, train_loader, None, None, split_map

    train_dataset, val_dataset, test_dataset, split_map = build_session_subsets(
        dataset=dataset,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        session_ids=list(session_ids) if session_ids is not None else None,
    )
    dataset.fit_track_normalization_for_sessions(split_map["train"])
    dataset.fit_track_global_normalization_for_sessions(split_map["train"])
    if hasattr(dataset, "window_mix_probability"):
        dataset.window_mix_probability = float(getattr(config, "track_window_mix_prob", 0.0)) if bool(getattr(config, "use_track_window_mix", False)) else 0.0

    train_sampler = None
    train_shuffle = True
    sampling_boost = float(getattr(config, "bird_surface_sampling_boost", 1.0))
    if sampling_boost > 1.0 and hasattr(train_dataset, "indices") and hasattr(dataset, "samples"):
        sample_weights = []
        for subset_index in train_dataset.indices:
            sample = dataset.samples[subset_index]
            label_name = getattr(sample, "label_name", "")
            weight = sampling_boost if label_name in {"BirdTarget", "ShipTarget"} else 1.0
            sample_weights.append(weight)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )
    if hasattr(getattr(val_dataset, "dataset", None), "window_mix_probability"):
        getattr(val_dataset, "dataset").window_mix_probability = 0.0
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    if hasattr(getattr(test_dataset, "dataset", None), "window_mix_probability"):
        getattr(test_dataset, "dataset").window_mix_probability = 0.0
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    return dataset, train_loader, val_loader, test_loader, split_map


def build_coarse_taxonomy(
    config: TrainingConfig,
    ordered_labels: Sequence[str],
) -> tuple[dict[str, int], torch.Tensor]:
    coarse_names: list[str] = []
    fine_to_coarse: list[int] = []
    coarse_name_to_index: dict[str, int] = {}
    coarse_label_map = dict(config.coarse_label_map)

    for label in ordered_labels:
        coarse_name = str(coarse_label_map.get(label, label))
        if coarse_name not in coarse_name_to_index:
            coarse_name_to_index[coarse_name] = len(coarse_names)
            coarse_names.append(coarse_name)
        fine_to_coarse.append(coarse_name_to_index[coarse_name])

    return coarse_name_to_index, torch.tensor(fine_to_coarse, dtype=torch.long)


def _build_detail_boundary_loss(
    detail_logits: torch.Tensor,
    labels: torch.Tensor,
    fine_to_coarse_index: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if detail_logits.ndim != 2 or detail_logits.size(0) == 0:
        return detail_logits.new_zeros(())

    coarse_targets = fine_to_coarse_index[labels]
    class_coarse = fine_to_coarse_index.to(device=detail_logits.device)
    same_coarse_mask = class_coarse.unsqueeze(0) == coarse_targets.unsqueeze(1)
    same_coarse_mask.scatter_(1, labels.unsqueeze(1), False)
    valid_rows = same_coarse_mask.any(dim=1)
    if not valid_rows.any():
        return detail_logits.new_zeros(())

    target_logits = detail_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    negative_logits = detail_logits.masked_fill(~same_coarse_mask, float('-inf')).max(dim=1).values
    boundary_loss = torch.relu(float(margin) + negative_logits - target_logits)
    return boundary_loss[valid_rows].mean()


def _bird_surface_prototype_loss(
    rd_feature: torch.Tensor | None,
    bird_surface_target: torch.Tensor | None,
    margin: float,
) -> torch.Tensor:
    if rd_feature is None or bird_surface_target is None or rd_feature.ndim != 2:
        return rd_feature.new_zeros(()) if isinstance(rd_feature, torch.Tensor) else torch.tensor(0.0)
    valid_mask = bird_surface_target >= 0.0
    if valid_mask.sum() < 2:
        return rd_feature.new_zeros(())
    valid_feature = rd_feature[valid_mask]
    valid_target = bird_surface_target[valid_mask]
    bird_mask = valid_target > 0.5
    surface_mask = valid_target <= 0.5
    if bird_mask.sum() == 0 or surface_mask.sum() == 0:
        return rd_feature.new_zeros(())
    bird_proto = valid_feature[bird_mask].mean(dim=0, keepdim=True)
    surface_proto = valid_feature[surface_mask].mean(dim=0, keepdim=True)
    bird_pos = torch.norm(valid_feature[bird_mask] - bird_proto, dim=-1).mean()
    surface_pos = torch.norm(valid_feature[surface_mask] - surface_proto, dim=-1).mean()
    bird_neg = torch.norm(valid_feature[bird_mask] - surface_proto, dim=-1).mean()
    surface_neg = torch.norm(valid_feature[surface_mask] - bird_proto, dim=-1).mean()
    separation = torch.relu(float(margin) + bird_pos + surface_pos - bird_neg - surface_neg)
    compactness = 0.5 * (bird_pos + surface_pos)
    return compactness + separation


def _bird_surface_triplet_loss(
    rd_feature: torch.Tensor | None,
    bird_surface_target: torch.Tensor | None,
    margin: float,
) -> torch.Tensor:
    if rd_feature is None or bird_surface_target is None or rd_feature.ndim != 2:
        return rd_feature.new_zeros(()) if isinstance(rd_feature, torch.Tensor) else torch.tensor(0.0)
    valid_mask = bird_surface_target >= 0.0
    if valid_mask.sum() < 2:
        return rd_feature.new_zeros(())
    features = torch.nn.functional.normalize(rd_feature[valid_mask], dim=-1)
    targets = bird_surface_target[valid_mask]
    bird_mask = targets > 0.5
    surface_mask = targets <= 0.5
    if bird_mask.sum() == 0 or surface_mask.sum() == 0:
        return rd_feature.new_zeros(())
    bird_features = features[bird_mask]
    surface_features = features[surface_mask]
    bird_proto = bird_features.mean(dim=0, keepdim=True)
    surface_proto = surface_features.mean(dim=0, keepdim=True)
    bird_positive = 1.0 - (bird_features * bird_proto).sum(dim=-1)
    bird_negative = 1.0 - (bird_features * surface_proto).sum(dim=-1)
    surface_positive = 1.0 - (surface_features * surface_proto).sum(dim=-1)
    surface_negative = 1.0 - (surface_features * bird_proto).sum(dim=-1)
    bird_triplet = torch.relu(bird_positive - bird_negative + margin)
    surface_triplet = torch.relu(surface_positive - surface_negative + margin)
    return torch.cat([bird_triplet, surface_triplet], dim=0).mean()


def compute_losses(
    outputs: dict[str, Any] | torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    auxiliary_criterion: nn.Module,
    notes_criterion: nn.Module | None,
    bird_drone_criterion: nn.Module | None,
    bird_surface_criterion: nn.Module | None,
    config: TrainingConfig,
    fine_to_coarse_index: torch.Tensor,
    rd_mask: torch.Tensor | None = None,
    notes_target: torch.Tensor | None = None,
    bird_drone_target: torch.Tensor | None = None,
    bird_surface_target: torch.Tensor | None = None,
    track_router_target: torch.Tensor | None = None,
    ship_drone_target: torch.Tensor | None = None,
    ship_bird_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        logits = outputs
        return logits, criterion(logits, labels)

    logits = outputs["logits"]
    loss = criterion(logits, labels)

    track_logits = outputs.get("track_logits")
    if track_logits is not None and config.track_aux_loss_weight > 0:
        loss = loss + config.track_aux_loss_weight * criterion(track_logits, labels)

    rd_logits = outputs.get("rd_logits")
    if _env_flag("RD", bool(config.include_rd)) and rd_logits is not None and config.rd_aux_loss_weight > 0:
        if rd_mask is None:
            loss = loss + config.rd_aux_loss_weight * criterion(rd_logits, labels)
        else:
            valid_rd = rd_mask.view(-1) > 0.5
            if valid_rd.any():
                loss = loss + config.rd_aux_loss_weight * criterion(rd_logits[valid_rd], labels[valid_rd])

    coarse_targets = fine_to_coarse_index[labels]
    coarse_logits = outputs.get("coarse_logits")
    if coarse_logits is not None and config.hierarchical_aux_loss_weight > 0:
        loss = loss + config.hierarchical_aux_loss_weight * auxiliary_criterion(coarse_logits, coarse_targets)

    detail_logits = outputs.get("detail_logits")
    if detail_logits is not None and config.detail_aux_loss_weight > 0:
        loss = loss + config.detail_aux_loss_weight * criterion(detail_logits, labels)
    if detail_logits is not None and config.detail_boundary_loss_weight > 0:
        boundary_loss = _build_detail_boundary_loss(
            detail_logits=detail_logits,
            labels=labels,
            fine_to_coarse_index=fine_to_coarse_index,
            margin=config.detail_boundary_margin,
        )
        loss = loss + config.detail_boundary_loss_weight * boundary_loss

    notes_logits = outputs.get("notes_logits") if isinstance(outputs, dict) else None
    if notes_criterion is not None and notes_logits is not None and notes_target is not None and getattr(config, "notes_aux_loss_weight", 0.0) > 0:
        loss = loss + float(config.notes_aux_loss_weight) * notes_criterion(notes_logits, notes_target)
    bird_drone_logits = outputs.get("bird_drone_logits") if isinstance(outputs, dict) else None
    if bird_drone_criterion is not None and bird_drone_logits is not None and bird_drone_target is not None and getattr(config, "bird_drone_aux_loss_weight", 0.0) > 0:
        valid_mask = bird_drone_target >= 0.0
        if valid_mask.any():
            loss = loss + float(config.bird_drone_aux_loss_weight) * bird_drone_criterion(bird_drone_logits.view(-1)[valid_mask], bird_drone_target[valid_mask])
    bird_surface_logits = outputs.get("bird_surface_logits") if isinstance(outputs, dict) else None
    if bird_surface_criterion is not None and bird_surface_logits is not None and bird_surface_target is not None and getattr(config, "bird_surface_aux_loss_weight", 0.0) > 0:
        valid_mask = bird_surface_target >= 0.0
        if valid_mask.any():
            loss = loss + float(config.bird_surface_aux_loss_weight) * bird_surface_criterion(bird_surface_logits.view(-1)[valid_mask], bird_surface_target[valid_mask])

    track_router_logits = outputs.get("track_router_logits") if isinstance(outputs, dict) else None
    if track_router_logits is not None and track_router_target is not None and getattr(config, "track_router_aux_loss_weight", 0.0) > 0:
        valid_mask = track_router_target >= 0
        if valid_mask.any():
            router_loss = auxiliary_criterion(track_router_logits[valid_mask], track_router_target[valid_mask])
            loss = loss + float(getattr(config, "track_router_aux_loss_weight", 0.0)) * router_loss

    rd_bird_drone_logits = outputs.get("rd_bird_drone_logits") if isinstance(outputs, dict) else None
    if rd_bird_drone_logits is not None and bird_drone_target is not None and getattr(config, "rd_bird_drone_aux_loss_weight", 0.0) > 0:
        valid_mask = bird_drone_target >= 0.0
        if valid_mask.any():
            bird_drone_targets_long = bird_drone_target[valid_mask].long()
            rd_specialist_loss = auxiliary_criterion(rd_bird_drone_logits[valid_mask], bird_drone_targets_long)
            loss = loss + float(getattr(config, "rd_bird_drone_aux_loss_weight", 0.0)) * rd_specialist_loss

    rd_ship_drone_logits = outputs.get("rd_ship_drone_logits") if isinstance(outputs, dict) else None
    if rd_ship_drone_logits is not None and ship_drone_target is not None and getattr(config, "rd_ship_drone_aux_loss_weight", 0.0) > 0:
        valid_mask = ship_drone_target >= 0.0
        if valid_mask.any():
            ship_drone_targets_long = ship_drone_target[valid_mask].long()
            ship_drone_loss = auxiliary_criterion(rd_ship_drone_logits[valid_mask], ship_drone_targets_long)
            loss = loss + float(getattr(config, "rd_ship_drone_aux_loss_weight", 0.0)) * ship_drone_loss

    rd_ship_bird_logits = outputs.get("rd_ship_bird_logits") if isinstance(outputs, dict) else None
    if rd_ship_bird_logits is not None and ship_bird_target is not None and getattr(config, "rd_ship_bird_aux_loss_weight", 0.0) > 0:
        valid_mask = ship_bird_target >= 0.0
        if valid_mask.any():
            ship_bird_targets_long = ship_bird_target[valid_mask].long()
            ship_bird_loss = auxiliary_criterion(rd_ship_bird_logits[valid_mask], ship_bird_targets_long)
            loss = loss + float(getattr(config, "rd_ship_bird_aux_loss_weight", 0.0)) * ship_bird_loss

    rd_feature = outputs.get("rd_feature") if isinstance(outputs, dict) else None
    if getattr(config, "bird_surface_prototype_loss_weight", 0.0) > 0:
        prototype_loss = _bird_surface_prototype_loss(
            rd_feature=rd_feature,
            bird_surface_target=bird_surface_target,
            margin=float(getattr(config, "bird_surface_prototype_margin", 0.35)),
        )
        loss = loss + float(getattr(config, "bird_surface_prototype_loss_weight", 0.0)) * prototype_loss

    if getattr(config, "bird_surface_triplet_loss_weight", 0.0) > 0:
        triplet_loss = _bird_surface_triplet_loss(
            rd_feature=rd_feature,
            bird_surface_target=bird_surface_target,
            margin=float(getattr(config, "bird_surface_triplet_margin", 0.4)),
        )
        loss = loss + float(getattr(config, "bird_surface_triplet_loss_weight", 0.0)) * triplet_loss
    return logits, loss


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    auxiliary_criterion: nn.Module,
    notes_criterion: nn.Module | None,
    bird_drone_criterion: nn.Module | None,
    bird_surface_criterion: nn.Module | None,
    device: torch.device,
    config: TrainingConfig,
    num_classes: int,
    fine_to_coarse_index: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for batch in data_loader:
        track_features = batch["track_features"].to(device=device, dtype=torch.float32)
        track_global_features = batch["track_global_features"].to(device=device, dtype=torch.float32)
        track_short_features = batch.get("track_short_features")
        if track_short_features is not None:
            track_short_features = track_short_features.to(device=device, dtype=torch.float32)
        track_long_features = batch.get("track_long_features")
        if track_long_features is not None:
            track_long_features = track_long_features.to(device=device, dtype=torch.float32)
        notes_target = batch.get("notes_target")
        if notes_target is not None:
            notes_target = notes_target.to(device=device, dtype=torch.float32)
        bird_drone_target = batch.get("bird_drone_target")
        if bird_drone_target is not None:
            bird_drone_target = bird_drone_target.to(device=device, dtype=torch.float32)
        bird_surface_target = batch.get("bird_surface_target")
        if bird_surface_target is not None:
            bird_surface_target = bird_surface_target.to(device=device, dtype=torch.float32)
        track_router_target = batch.get("track_router_target")
        if track_router_target is not None:
            track_router_target = track_router_target.to(device=device, dtype=torch.long)
        ship_drone_target = batch.get("ship_drone_target")
        if ship_drone_target is not None:
            ship_drone_target = ship_drone_target.to(device=device, dtype=torch.float32)
        ship_bird_target = batch.get("ship_bird_target")
        if ship_bird_target is not None:
            ship_bird_target = ship_bird_target.to(device=device, dtype=torch.float32)
        rd_features = None
        rd_mask = None
        if _env_flag("RD", bool(config.include_rd)):
            rd_features = batch["rd_features"].to(device=device, dtype=torch.float32)
            rd_mask = batch["has_rd"].to(device=device, dtype=torch.float32).view(-1, 1)
        labels = batch["label"].to(device=device, dtype=torch.long)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(track_features, track_global_features, track_short_features, track_long_features, rd_features, rd_mask)
            logits, loss = compute_losses(outputs, labels, criterion, auxiliary_criterion, notes_criterion, bird_drone_criterion, bird_surface_criterion, config, fine_to_coarse_index, rd_mask, notes_target, bird_drone_target, bird_surface_target, track_router_target, ship_drone_target, ship_bird_target)
            if training:
                loss.backward()
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

        predictions = logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_predictions, num_classes=max(num_classes, 1))
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics




def _compute_selection_metric(config: TrainingConfig, val_metrics: dict[str, float]) -> float:
    metric_name = str(getattr(config, "best_metric_name", "val_composite"))
    if metric_name == "val_macro_f1":
        return float(val_metrics.get("macro_f1", 0.0))
    if metric_name == "val_accuracy":
        return float(val_metrics.get("accuracy", 0.0))
    weights = getattr(config, "best_metric_weights", {"val_macro_f1": 0.7, "val_accuracy": 0.3})
    w_f1 = float(weights.get("val_macro_f1", 0.7))
    w_acc = float(weights.get("val_accuracy", 0.3))
    return w_f1 * float(val_metrics.get("macro_f1", 0.0)) + w_acc * float(val_metrics.get("accuracy", 0.0))

def save_training_artifacts(
    output_dir: Path,
    config: TrainingConfig,
    dataset: Any,
    history: list[dict[str, Any]],
    num_parameters: int,
    split_map: dict[str, list[str]],
    coarse_to_index: dict[str, int],
    fine_to_coarse_index: torch.Tensor,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "training_metadata.json"
    history_path = output_dir / "training_history.json"

    metadata = {
        "config": config.to_dict(),
        "labels": dataset.index_to_label,
        "coarse_to_index": coarse_to_index,
        "fine_to_coarse_index": fine_to_coarse_index.tolist(),
        "num_samples": len(dataset),
        "num_parameters": num_parameters,
        "session_splits": split_map,
        "track_feature_center": dataset.track_feature_center.tolist(),
        "track_feature_scale": dataset.track_feature_scale.tolist(),
        "track_global_center": dataset.track_global_center.tolist(),
        "track_global_scale": dataset.track_global_scale.tolist(),
        "num_notes": getattr(dataset, "note_dim", 0),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    return history_path


def run_training(
    config_values: dict[str, Any] | None = None,
    session_ids: Sequence[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = TrainingConfig.from_dict(config_values)
    set_random_seed(config.random_seed)

    dataset, train_loader, val_loader, test_loader, split_map = build_dataloaders(config, session_ids=session_ids)
    resolved_device = torch.device(device or "cpu")
    ordered_labels = [dataset.index_to_label[index] for index in sorted(dataset.index_to_label)]
    coarse_training_mode = str(getattr(config, 'label_granularity', 'fine')).lower() == 'coarse'
    if coarse_training_mode:
        config.hierarchical_aux_loss_weight = 0.0
        config.detail_aux_loss_weight = 0.0
        config.detail_boundary_loss_weight = 0.0

    coarse_to_index, fine_to_coarse_index = build_coarse_taxonomy(config, ordered_labels)
    fine_to_coarse_index = fine_to_coarse_index.to(resolved_device)

    model = FusionRadarClassifier(
        track_input_dim=config.feature_dim,
        track_global_feature_dim=config.global_feature_dim,
        num_classes=len(dataset.label_to_index),
        num_coarse_classes=len(coarse_to_index),
        include_track=_env_flag("TRACK", bool(getattr(config, "include_track", True))),
        include_rd=_env_flag("RD", bool(config.include_rd)),
        track_projection_dim=config.track_projection_dim,
        tcn_channels=config.tcn_channels,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_dropout=config.transformer_dropout,
        max_sequence_length=config.track_long_length,
        short_sequence_length=config.track_short_length,
        mid_sequence_length=config.track_mid_length,
        long_sequence_length=config.track_long_length,
        x3d_in_channels=config.rd_input_channels,
        x3d_stem_channels=config.x3d_stem_channels,
        x3d_stage_channels=config.x3d_stage_channels,
        x3d_stage_depths=config.x3d_stage_depths,
        x3d_spatial_strides=config.x3d_spatial_strides,
        x3d_temporal_strides=config.x3d_temporal_strides,
        x3d_bottleneck_expansion=config.x3d_bottleneck_expansion,
        x3d_se_ratio=config.x3d_se_ratio,
        cross_attention_layers=config.cross_attention_layers,
        cross_attention_heads=config.cross_attention_heads,
        cross_attention_dropout=config.cross_attention_dropout,
        fusion_hidden_size=config.fusion_hidden_size,
        detail_logit_residual_weight=config.detail_logit_residual_weight,
        detail_cosine_scale=config.detail_cosine_scale,
        num_notes=getattr(dataset, "note_dim", 0),
        track_backbone_type=config.track_backbone_type,
        temporal_2d_branch_channels=getattr(config, "temporal_2d_branch_channels", 96),
        bird_class_index=dataset.label_to_index.get("BirdTarget"),
        surface_class_index=dataset.label_to_index.get("ShipTarget"),
        aircraft_class_index=dataset.label_to_index.get("AircraftTarget"),
        drone_class_index=dataset.label_to_index.get("DroneTarget"),
        bird_surface_logit_residual_weight=float(getattr(config, "bird_surface_logit_residual_weight", 0.2)),
        ship_specialist_logit_residual_weight=float(getattr(config, "ship_specialist_logit_residual_weight", 0.08)),
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights = None
    if config.use_class_balanced_loss:
        class_weights = build_class_balanced_weights(
            train_loader.dataset,
            num_classes=len(dataset.label_to_index),
            beta=config.class_balance_beta,
        ).to(resolved_device)
    if getattr(config, 'use_focal_loss', False):
        criterion = FocalCrossEntropyLoss(weight=class_weights, gamma=float(getattr(config, 'focal_gamma', 2.0)), label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    auxiliary_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    notes_criterion = nn.BCEWithLogitsLoss() if getattr(dataset, "note_dim", 0) > 0 else None
    bird_drone_criterion = nn.BCEWithLogitsLoss()
    bird_surface_criterion = nn.BCEWithLogitsLoss()
    scheduler = build_scheduler(optimizer, epochs=config.epochs, warmup_epochs=config.warmup_epochs)
    output_dir = Path(config.output_dir)
    best_checkpoint = output_dir / "best_fusion_model.pt"
    best_macro_f1 = -1.0
    best_selection_metric = float("-inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    num_parameters = count_trainable_parameters(model)

    logger.info(
        f"[*] Training samples: {len(train_loader.dataset)}, "
        f"validation samples: {0 if val_loader is None else len(val_loader.dataset)}, "
        f"classes: {dataset.index_to_label}"
    )
    logger.info(
        f"[*] Session split | train={split_map['train']} | val={split_map['val']} | test={split_map['test']}"
    )
    logger.info(f"[*] Using device: {resolved_device}")
    logger.info(f"[*] Coarse taxonomy: {coarse_to_index}")
    logger.info(f"[*] Fine-to-coarse index: {fine_to_coarse_index.detach().cpu().tolist()}")
    logger.info(f"[*] Trainable parameters: {num_parameters:,}")
    if class_weights is not None:
        logger.info(f"[*] Class weights: {class_weights.detach().cpu().tolist()}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            auxiliary_criterion=auxiliary_criterion,
            notes_criterion=notes_criterion,
            bird_drone_criterion=bird_drone_criterion,
            bird_surface_criterion=bird_surface_criterion,
            device=resolved_device,
            config=config,
            num_classes=len(dataset.label_to_index),
            fine_to_coarse_index=fine_to_coarse_index,
            optimizer=optimizer,
        )

        if val_loader is None:
            val_metrics = train_metrics
        else:
            val_metrics = run_epoch(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                auxiliary_criterion=auxiliary_criterion,
                notes_criterion=notes_criterion,
                bird_drone_criterion=bird_drone_criterion,
            bird_surface_criterion=bird_surface_criterion,
                device=resolved_device,
                config=config,
                num_classes=len(dataset.label_to_index),
                fine_to_coarse_index=fine_to_coarse_index,
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_selection_metric": _compute_selection_metric(config, val_metrics),
        }
        history.append(epoch_record)

        logger.info(
            f"[*] Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if scheduler is not None:
            scheduler.step()

        selection_metric = _compute_selection_metric(config, val_metrics)
        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]

        if selection_metric > best_selection_metric + float(getattr(config, "early_stopping_min_delta", 0.0)):
            best_selection_metric = selection_metric
            stale_epochs = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "label_to_index": dataset.label_to_index,
                    "coarse_to_index": coarse_to_index,
                    "fine_to_coarse_index": fine_to_coarse_index.detach().cpu().tolist(),
                    "best_macro_f1": best_macro_f1,
                    "best_selection_metric": best_selection_metric,
                    "num_parameters": num_parameters,
                    "track_feature_center": dataset.track_feature_center.tolist(),
                    "track_feature_scale": dataset.track_feature_scale.tolist(),
                    "track_global_center": dataset.track_global_center.tolist(),
                    "track_global_scale": dataset.track_global_scale.tolist(),
                    "num_notes": getattr(dataset, "note_dim", 0),
                },
                best_checkpoint,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= int(getattr(config, "early_stopping_patience", 999999)):
            logger.info(f"[*] Early stopping triggered at epoch {epoch:03d} | stale_epochs={stale_epochs}")
            break

    history_path = save_training_artifacts(output_dir, config, dataset, history, num_parameters, split_map, coarse_to_index, fine_to_coarse_index.detach().cpu())
    manifest = build_session_manifest(data_root=config.data_root, include_rd=_env_flag("RD", bool(config.include_rd)), split_map=split_map)
    manifest_path = save_session_manifest(manifest, output_dir)

    if best_checkpoint.exists():
        checkpoint_payload = torch.load(best_checkpoint, map_location=resolved_device)
        model.load_state_dict(checkpoint_payload["model_state_dict"])

    test_metrics = None
    if test_loader is not None:
        test_metrics = run_epoch(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            auxiliary_criterion=auxiliary_criterion,
            notes_criterion=notes_criterion,
            bird_drone_criterion=bird_drone_criterion,
            bird_surface_criterion=bird_surface_criterion,
            device=resolved_device,
            config=config,
            num_classes=len(dataset.label_to_index),
            fine_to_coarse_index=fine_to_coarse_index,
        )
        logger.info(
            f"[*] Test metrics | loss={test_metrics['loss']:.4f} | acc={test_metrics['accuracy']:.4f} | f1={test_metrics['macro_f1']:.4f}"
        )

    return {
        "best_checkpoint": str(best_checkpoint),
        "history_path": str(history_path),
        "manifest_path": str(manifest_path),
        "num_samples": len(dataset),
        "labels": dataset.index_to_label,
        "num_parameters": num_parameters,
        "session_splits": split_map,
        "test_metrics": test_metrics,
    }
