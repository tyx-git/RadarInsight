import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.models.fusion_classifier import FusionRadarClassifier
from src.models.session_bag_classifier import SessionBagRadarClassifier
from src.training.session_bag import SessionBagDataset, session_bag_collate_fn
from src.training.curated_manifest import build_manifest_mask, load_curated_clip_manifest
from src.training.config import TrainingConfig
from src.training.session_manager import build_dataset, build_session_manifest, build_session_subsets, save_session_manifest
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import logger


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_track_global_feature_lookup(dataset) -> dict[str, int]:
    columns = tuple(getattr(dataset, "global_feature_columns", ()))
    return {str(name): index for index, name in enumerate(columns)}


def _is_suspicious_ship_pattern(dataset, sample, config: TrainingConfig) -> bool:
    if getattr(sample, "label_name", "") != "ShipTarget":
        return False
    suspicious_session_ids = {str(item) for item in getattr(config, "suspicious_ship_session_ids", ())}
    session_id = str(getattr(sample, "session_id", ""))
    if suspicious_session_ids and session_id in suspicious_session_ids:
        return True
    extract_features = getattr(dataset, "_extract_track_global_features", None)
    if extract_features is None:
        return False
    feature_lookup = _build_track_global_feature_lookup(dataset)
    if not feature_lookup:
        return False
    try:
        feature_vector = np.asarray(extract_features(sample), dtype=np.float32)
    except Exception:
        return False
    speed_mean_threshold = float(getattr(config, "suspicious_ship_speed_mean_threshold", 0.0))
    direction_consistency_threshold = float(getattr(config, "suspicious_ship_direction_consistency_threshold", 0.0))
    surface_bias_threshold = float(getattr(config, "suspicious_ship_surface_bias_threshold", 0.0))
    snr_mean_threshold = float(getattr(config, "suspicious_ship_snr_mean_threshold", 0.0))

    def feature(name: str, default: float = 0.0) -> float:
        index = feature_lookup.get(name)
        if index is None or index >= feature_vector.shape[0]:
            return default
        return float(feature_vector[index])

    clip_speed_mean = abs(feature("clip_speed_mean_mps"))
    clip_speed_direction_consistency = feature("clip_speed_direction_consistency")
    clip_range_monotonicity = feature("clip_range_monotonicity")
    clip_snr_mean = feature("clip_snr_mean_db")

    return (
        clip_speed_mean >= speed_mean_threshold
        and clip_speed_direction_consistency >= direction_consistency_threshold
        and clip_range_monotonicity >= surface_bias_threshold
        and clip_snr_mean <= snr_mean_threshold
    )


def _build_pair_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_index: int | None,
    negative_index: int | None,
    positive_labels: set[int],
    negative_labels: set[int],
    margin: float,
) -> torch.Tensor:
    if positive_index is None or negative_index is None or logits.ndim != 2 or logits.size(0) == 0:
        return logits.new_zeros(())
    positive_mask = torch.zeros_like(labels, dtype=torch.bool)
    negative_mask = torch.zeros_like(labels, dtype=torch.bool)
    for value in positive_labels:
        positive_mask |= labels == int(value)
    for value in negative_labels:
        negative_mask |= labels == int(value)
    valid_mask = positive_mask | negative_mask
    if not valid_mask.any():
        return logits.new_zeros(())
    signed_target = torch.where(positive_mask[valid_mask], 1.0, -1.0).to(logits.dtype)
    pair_margin = logits[valid_mask, positive_index] - logits[valid_mask, negative_index]
    return torch.relu(float(margin) - signed_target * pair_margin).mean()


def _build_masked_pair_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_mask: torch.Tensor | None,
    positive_index: int | None,
    negative_index: int | None,
    positive_labels: set[int],
    negative_labels: set[int],
    margin: float,
) -> torch.Tensor:
    if sample_mask is None:
        return _build_pair_margin_loss(logits, labels, positive_index, negative_index, positive_labels, negative_labels, margin)
    if positive_index is None or negative_index is None or logits.ndim != 2 or logits.size(0) == 0:
        return logits.new_zeros(())
    positive_mask = torch.zeros_like(labels, dtype=torch.bool)
    negative_mask = torch.zeros_like(labels, dtype=torch.bool)
    for value in positive_labels:
        positive_mask |= labels == int(value)
    for value in negative_labels:
        negative_mask |= labels == int(value)
    valid_mask = (positive_mask | negative_mask) & (sample_mask.view(-1) > 0.5)
    if not valid_mask.any():
        return logits.new_zeros(())
    signed_target = torch.where(positive_mask[valid_mask], 1.0, -1.0).to(logits.dtype)
    pair_margin = logits[valid_mask, positive_index] - logits[valid_mask, negative_index]
    return torch.relu(float(margin) - signed_target * pair_margin).mean()


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
        if hasattr(parent, "samples") and hasattr(parent, "label_to_index"):
            return [parent.label_to_index[parent.samples[index].label_name] for index in indices]
        if hasattr(parent, "base_dataset") and hasattr(parent, "session_ids"):
            return [int(parent[index]["label"]) for index in indices]

    if hasattr(data_source, "samples") and hasattr(data_source, "label_to_index"):
        return [data_source.label_to_index[sample.label_name] for sample in data_source.samples]

    if hasattr(data_source, "session_ids") and hasattr(data_source, "__getitem__"):
        return [int(data_source[index]["label"]) for index in range(len(data_source))]

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

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return 0.2 + 0.8 * (float(epoch_index) / float(warmup_epochs))
        cosine_index = max(epoch_index - warmup_epochs, 0)
        progress = float(cosine_index) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _linear_warm_schedule(epoch: int, start_epoch: int, full_epoch: int, start_value: float, target_value: float) -> float:
    if epoch < start_epoch:
        return float(start_value)
    if full_epoch <= start_epoch:
        return float(target_value)
    if epoch >= full_epoch:
        return float(target_value)
    progress = float(epoch - start_epoch) / float(full_epoch - start_epoch)
    return float(start_value + progress * (target_value - start_value))


def _resolve_epoch_fusion_controls(config: TrainingConfig, epoch: int) -> dict[str, float]:
    if not bool(getattr(config, "rd_fusion_curriculum_enabled", False)):
        return {
            "min_rd_fusion_weight": float(getattr(config, "min_rd_fusion_weight", 0.15)),
            "ambiguous_rd_fusion_boost": float(getattr(config, "ambiguous_rd_fusion_boost", 0.15)),
            "surface_rd_fusion_boost": float(getattr(config, "surface_rd_fusion_boost", 0.1)),
        }

    start_epoch = int(getattr(config, "rd_fusion_curriculum_start_epoch", 3))
    full_epoch = int(getattr(config, "rd_fusion_curriculum_full_epoch", 6))
    return {
        "min_rd_fusion_weight": _linear_warm_schedule(
            epoch=epoch,
            start_epoch=start_epoch,
            full_epoch=full_epoch,
            start_value=float(getattr(config, "min_rd_fusion_weight_start", 0.0)),
            target_value=float(getattr(config, "min_rd_fusion_weight", 0.15)),
        ),
        "ambiguous_rd_fusion_boost": _linear_warm_schedule(
            epoch=epoch,
            start_epoch=start_epoch,
            full_epoch=full_epoch,
            start_value=float(getattr(config, "ambiguous_rd_fusion_boost_start", 0.0)),
            target_value=float(getattr(config, "ambiguous_rd_fusion_boost", 0.15)),
        ),
        "surface_rd_fusion_boost": _linear_warm_schedule(
            epoch=epoch,
            start_epoch=start_epoch,
            full_epoch=full_epoch,
            start_value=float(getattr(config, "surface_rd_fusion_boost_start", 0.0)),
            target_value=float(getattr(config, "surface_rd_fusion_boost", 0.1)),
        ),
    }


def build_dataloaders(
    config: TrainingConfig,
    session_ids: Sequence[str] | None = None,
) -> tuple[Any, DataLoader, DataLoader | None, DataLoader | None, dict[str, list[str]]]:
    config.include_track = _env_flag("TRACK", bool(getattr(config, "include_track", True)))
    config.include_rd = _env_flag("RD", bool(config.include_rd))
    dataset = build_dataset(data_root=config.data_root, config=config, session_ids=session_ids)
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(int(config.random_seed))

    if len(dataset) == 1:
        dataset.fit_track_normalization_for_sessions(sorted(dataset.sessions.keys()))
        dataset.fit_track_global_normalization_for_sessions(sorted(dataset.sessions.keys()))
        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
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
    if hasattr(dataset, "set_window_mix_enabled_indices"):
        dataset.set_window_mix_enabled_indices(getattr(train_dataset, "indices", None))

    if bool(getattr(config, "use_session_bag_training", False)):
        bag_batch_size = max(1, int(getattr(config, 'session_bag_batch_size', 1)))
        bag_shuffle = bool(getattr(config, 'session_bag_shuffle', True))
        bag_sort_by_length = bool(getattr(config, 'session_bag_sort_by_length', True))
        train_bag_dataset = SessionBagDataset(
            base_dataset=dataset,
            indices=getattr(train_dataset, 'indices', []),
            max_clips=int(getattr(config, 'session_bag_max_clips', 32)),
            random_sampling=bool(getattr(config, 'session_bag_random_sampling', True)),
            sort_by_length=bag_sort_by_length,
        )
        val_bag_dataset = None if val_dataset is None else SessionBagDataset(
            base_dataset=dataset,
            indices=getattr(val_dataset, 'indices', []),
            max_clips=int(getattr(config, 'session_bag_max_clips', 32)),
            random_sampling=False,
            sort_by_length=bag_sort_by_length,
        )
        test_bag_dataset = None if test_dataset is None else SessionBagDataset(
            base_dataset=dataset,
            indices=getattr(test_dataset, 'indices', []),
            max_clips=int(getattr(config, 'session_bag_max_clips', 32)),
            random_sampling=False,
            sort_by_length=bag_sort_by_length,
        )
        train_bag_sampler = None
        if len(train_bag_dataset) > 0:
            if bool(getattr(config, 'session_bag_source_class_balanced_sampling', False)):
                session_bag_labels = [int(train_bag_dataset[index]["label"]) for index in range(len(train_bag_dataset))]
                session_bag_sources = [str(train_bag_dataset[index]["source_name"]) for index in range(len(train_bag_dataset))]
                source_counts: dict[str, int] = {}
                class_counts = np.bincount(session_bag_labels, minlength=len(dataset.label_to_index)).astype(np.float64)
                for source_name in session_bag_sources:
                    source_counts[source_name] = source_counts.get(source_name, 0) + 1
                source_temperature = max(1e-6, float(getattr(config, 'session_bag_source_balance_temperature', 1.0)))
                class_temperature = max(1e-6, float(getattr(config, 'session_bag_class_balance_temperature', 0.5)))
                people_cyclist_boost = max(1.0, float(getattr(config, 'session_bag_people_cyclist_sampling_boost', 1.0)))
                bird_clutter_other_boost = max(1.0, float(getattr(config, 'session_bag_bird_clutter_other_sampling_boost', 1.0)))
                sample_weights = []
                for label_value, source_name in zip(session_bag_labels, session_bag_sources):
                    source_count = max(float(source_counts.get(source_name, 1)), 1.0)
                    class_count = max(float(class_counts[int(label_value)]), 1.0)
                    source_weight = 1.0 / (source_count ** source_temperature)
                    class_weight = 1.0 / (class_count ** class_temperature)
                    label_name = dataset.index_to_label.get(int(label_value), '')
                    boost = 1.0
                    if source_name == 'Carrada' and label_name in {'PeopleTarget', 'CyclistTarget'}:
                        boost *= people_cyclist_boost
                    if source_name == 'CQ-08-MAT' and label_name in {'BirdTarget', 'ClutterTarget', 'OtherTarget'}:
                        boost *= bird_clutter_other_boost
                    sample_weights.append(source_weight * class_weight * boost)
                train_bag_sampler = WeightedRandomSampler(
                    weights=torch.tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                bag_shuffle = False
            elif bool(getattr(config, 'session_bag_class_balanced_sampling', False)):
                session_bag_labels = [int(train_bag_dataset[index]["label"]) for index in range(len(train_bag_dataset))]
                label_counts = np.bincount(session_bag_labels, minlength=len(dataset.label_to_index)).astype(np.float64)
                sample_weights = []
                for label_value in session_bag_labels:
                    count = max(float(label_counts[int(label_value)]), 1.0)
                    sample_weights.append(1.0 / count)
                train_bag_sampler = WeightedRandomSampler(
                    weights=torch.tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                bag_shuffle = False
        train_loader = DataLoader(
            train_bag_dataset,
            batch_size=bag_batch_size,
            shuffle=bag_shuffle if train_bag_sampler is None else False,
            sampler=train_bag_sampler,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
            collate_fn=session_bag_collate_fn,
        )
        val_loader = None if val_bag_dataset is None else DataLoader(
            val_bag_dataset,
            batch_size=bag_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
            collate_fn=session_bag_collate_fn,
        )
        test_loader = None if test_bag_dataset is None else DataLoader(
            test_bag_dataset,
            batch_size=bag_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
            collate_fn=session_bag_collate_fn,
        )
        return dataset, train_loader, val_loader, test_loader, split_map

    train_sampler = None
    train_batch_sampler = None
    train_shuffle = True
    sampling_boost = float(getattr(config, "bird_surface_sampling_boost", 1.0))
    hard_ship_session_sampling_boost = float(getattr(config, "hard_ship_session_sampling_boost", 1.0))
    hard_ship_session_ids = {str(item) for item in getattr(config, "hard_ship_session_ids", ())}
    hard_bird_session_sampling_boost = float(getattr(config, "hard_bird_session_sampling_boost", 1.0))
    hard_bird_session_ids = {str(item) for item in getattr(config, "hard_bird_session_ids", ())}
    suspicious_ship_downweight_enabled = bool(getattr(config, "suspicious_ship_downweight_enabled", False))
    suspicious_ship_downweight_factor = float(getattr(config, "suspicious_ship_downweight_factor", 1.0))
    if (
        sampling_boost > 1.0
        or (hard_ship_session_sampling_boost > 1.0 and hard_ship_session_ids)
        or (hard_bird_session_sampling_boost > 1.0 and hard_bird_session_ids)
        or (suspicious_ship_downweight_enabled and suspicious_ship_downweight_factor < 1.0)
        or bool(getattr(config, "confusion_pair_sampling_enabled", False))
    ) and hasattr(train_dataset, "indices") and hasattr(dataset, "samples"):
        confusion_pair_sampling_enabled = bool(getattr(config, "confusion_pair_sampling_enabled", False))
        confusion_pair_sampling_boost = float(getattr(config, "confusion_pair_sampling_boost", 1.0))
        confusion_pair_bird_ship_session_ids = {str(item) for item in getattr(config, "confusion_pair_bird_ship_session_ids", ())}
        confusion_pair_ship_drone_session_ids = {str(item) for item in getattr(config, "confusion_pair_ship_drone_session_ids", ())}
        sample_weights = []
        for subset_index in train_dataset.indices:
            sample = dataset.samples[subset_index]
            label_name = getattr(sample, "label_name", "")
            session_id = str(getattr(sample, "session_id", ""))
            weight = sampling_boost if label_name in {"BirdTarget", "ShipTarget"} else 1.0
            if hard_ship_session_ids and session_id in hard_ship_session_ids:
                weight *= hard_ship_session_sampling_boost
            if hard_bird_session_ids and session_id in hard_bird_session_ids:
                weight *= hard_bird_session_sampling_boost
            if suspicious_ship_downweight_enabled and suspicious_ship_downweight_factor < 1.0:
                if _is_suspicious_ship_pattern(dataset, sample, config):
                    weight *= suspicious_ship_downweight_factor
            if confusion_pair_sampling_enabled and confusion_pair_sampling_boost > 1.0:
                if label_name in {"BirdTarget", "ShipTarget"} and session_id in confusion_pair_bird_ship_session_ids:
                    weight *= confusion_pair_sampling_boost
                if label_name in {"ShipTarget", "DroneTarget"} and session_id in confusion_pair_ship_drone_session_ids:
                    weight *= confusion_pair_sampling_boost
            sample_weights.append(weight)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=data_loader_generator,
        )
        train_shuffle = False

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
        )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
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
        if '__' in str(label):
            coarse_name = str(label).split('__', 1)[0]
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

    if not torch.is_tensor(fine_to_coarse_index):
        fine_to_coarse_index = torch.as_tensor(fine_to_coarse_index, dtype=torch.long, device=labels.device)
    else:
        fine_to_coarse_index = fine_to_coarse_index.to(device=labels.device, dtype=torch.long)
    if not torch.is_tensor(fine_to_coarse_index):
        fine_to_coarse_index = torch.as_tensor(fine_to_coarse_index, dtype=torch.long, device=labels.device)
    else:
        fine_to_coarse_index = fine_to_coarse_index.to(device=labels.device, dtype=torch.long)
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
    label_to_index: dict[str, int] | None = None,
    rd_mask: torch.Tensor | None = None,
    notes_target: torch.Tensor | None = None,
    bird_rd_target: torch.Tensor | None = None,
    bird_drone_target: torch.Tensor | None = None,
    bird_surface_target: torch.Tensor | None = None,
    track_router_target: torch.Tensor | None = None,
    ship_drone_target: torch.Tensor | None = None,
    ship_bird_target: torch.Tensor | None = None,
    ship_air_target: torch.Tensor | None = None,
    suspicious_ship_hard_pattern: torch.Tensor | None = None,
    curated_ship_drone_mask: torch.Tensor | None = None,
    curated_bird_ship_mask: torch.Tensor | None = None,
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

    if not torch.is_tensor(fine_to_coarse_index):
        fine_to_coarse_index = torch.as_tensor(fine_to_coarse_index, dtype=torch.long, device=labels.device)
    else:
        fine_to_coarse_index = fine_to_coarse_index.to(device=labels.device, dtype=torch.long)
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

    bird_rd_logits = outputs.get("bird_rd_logits") if isinstance(outputs, dict) else None
    if bird_surface_criterion is not None and bird_rd_logits is not None and bird_rd_target is not None and getattr(config, "bird_rd_aux_loss_weight", 0.0) > 0:
        valid_mask = bird_rd_target >= 0.0
        if rd_mask is not None:
            valid_mask = valid_mask & (rd_mask.view(-1) > 0.5)
        if valid_mask.any():
            loss = loss + float(getattr(config, "bird_rd_aux_loss_weight", 0.0)) * bird_surface_criterion(bird_rd_logits.view(-1)[valid_mask], bird_rd_target[valid_mask])

    bird_surface_rd_logits = outputs.get("bird_surface_rd_logits") if isinstance(outputs, dict) else None
    if bird_surface_criterion is not None and bird_surface_rd_logits is not None and bird_surface_target is not None and getattr(config, "bird_surface_rd_aux_loss_weight", 0.0) > 0:
        valid_mask = bird_surface_target >= 0.0
        if rd_mask is not None:
            valid_mask = valid_mask & (rd_mask.view(-1) > 0.5)
        if valid_mask.any():
            loss = loss + float(getattr(config, "bird_surface_rd_aux_loss_weight", 0.0)) * bird_surface_criterion(bird_surface_rd_logits.view(-1)[valid_mask], bird_surface_target[valid_mask])

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

    rd_ship_air_logits = outputs.get("rd_ship_air_logits") if isinstance(outputs, dict) else None
    if rd_ship_air_logits is not None and ship_air_target is not None and getattr(config, "rd_ship_air_aux_loss_weight", 0.0) > 0:
        valid_mask = ship_air_target >= 0.0
        if valid_mask.any():
            ship_air_targets_long = ship_air_target[valid_mask].long()
            ship_air_loss = auxiliary_criterion(rd_ship_air_logits[valid_mask], ship_air_targets_long)
            loss = loss + float(getattr(config, "rd_ship_air_aux_loss_weight", 0.0)) * ship_air_loss

    ship_drone_handler_logits = outputs.get("ship_drone_handler_logits") if isinstance(outputs, dict) else None
    ship_drone_handler_weight = float(getattr(config, "confusion_ship_drone_handler_weight", 0.0) or getattr(config, "confusion_pair_handler_weight", 0.0))
    if ship_drone_handler_logits is not None and ship_drone_target is not None and ship_drone_handler_weight > 0:
        valid_mask = ship_drone_target >= 0.0
        if valid_mask.any():
            handler_targets = ship_drone_target[valid_mask].long()
            handler_loss = auxiliary_criterion(ship_drone_handler_logits[valid_mask], handler_targets)
            loss = loss + 0.5 * ship_drone_handler_weight * handler_loss

    bird_ship_handler_logits = outputs.get("bird_ship_handler_logits") if isinstance(outputs, dict) else None
    bird_ship_handler_weight = float(getattr(config, "confusion_bird_ship_handler_weight", 0.0) or getattr(config, "confusion_pair_handler_weight", 0.0))
    if bird_ship_handler_logits is not None and ship_bird_target is not None and bird_ship_handler_weight > 0:
        valid_mask = ship_bird_target >= 0.0
        if valid_mask.any():
            handler_targets = 1 - ship_bird_target[valid_mask].long()
            handler_loss = auxiliary_criterion(bird_ship_handler_logits[valid_mask], handler_targets)
            loss = loss + 0.5 * bird_ship_handler_weight * handler_loss

    ship_drone_refiner_logits = outputs.get("ship_drone_refiner_logits") if isinstance(outputs, dict) else None
    ship_drone_refiner_aux_loss_weight = float(getattr(config, "ship_drone_refiner_aux_loss_weight", 0.0))
    if ship_drone_refiner_logits is not None and ship_drone_target is not None and ship_drone_refiner_aux_loss_weight > 0:
        valid_mask = ship_drone_target >= 0.0
        if valid_mask.any():
            refiner_targets = ship_drone_target[valid_mask].long()
            refiner_loss = auxiliary_criterion(ship_drone_refiner_logits[valid_mask], refiner_targets)
            loss = loss + ship_drone_refiner_aux_loss_weight * refiner_loss

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

    label_lookup = {str(name): int(index) for name, index in (label_to_index or {}).items()}
    bird_index = label_lookup.get("BirdTarget")
    drone_index = label_lookup.get("DroneTarget")
    ship_index = label_lookup.get("ShipTarget")

    ship_drone_margin_loss_weight = float(getattr(config, "ship_drone_margin_loss_weight", 0.0))
    if ship_drone_margin_loss_weight > 0 and ship_index is not None and drone_index is not None:
        ship_drone_margin_loss = _build_pair_margin_loss(
            logits=logits,
            labels=labels,
            positive_index=ship_index,
            negative_index=drone_index,
            positive_labels={ship_index},
            negative_labels={drone_index},
            margin=float(getattr(config, "ship_drone_margin", 0.25)),
        )
        loss = loss + ship_drone_margin_loss_weight * ship_drone_margin_loss

    conditional_ship_drone_margin_weight = float(getattr(config, "conditional_ship_drone_margin_weight", 0.0))
    if bool(getattr(config, "conditional_ship_drone_margin_enabled", False)) and conditional_ship_drone_margin_weight > 0 and ship_index is not None and drone_index is not None:
        conditional_ship_drone_margin_loss = _build_masked_pair_margin_loss(
            logits=logits,
            labels=labels,
            sample_mask=suspicious_ship_hard_pattern,
            positive_index=ship_index,
            negative_index=drone_index,
            positive_labels={ship_index},
            negative_labels={drone_index},
            margin=float(getattr(config, "conditional_ship_drone_margin", 0.15)),
        )
        loss = loss + conditional_ship_drone_margin_weight * conditional_ship_drone_margin_loss

    curated_ship_drone_margin_weight = float(getattr(config, "curated_ship_drone_margin_weight", 0.0))
    if bool(getattr(config, "curated_ship_drone_margin_enabled", False)) and curated_ship_drone_margin_weight > 0 and ship_index is not None and drone_index is not None:
        curated_ship_drone_margin_loss = _build_masked_pair_margin_loss(
            logits=logits,
            labels=labels,
            sample_mask=curated_ship_drone_mask,
            positive_index=ship_index,
            negative_index=drone_index,
            positive_labels={ship_index},
            negative_labels={drone_index},
            margin=float(getattr(config, "curated_ship_drone_margin", 0.15)),
        )
        loss = loss + curated_ship_drone_margin_weight * curated_ship_drone_margin_loss

    curated_bird_ship_margin_weight = float(getattr(config, "curated_bird_ship_margin_weight", 0.0))
    if bool(getattr(config, "curated_bird_ship_margin_enabled", False)) and curated_bird_ship_margin_weight > 0 and bird_index is not None and ship_index is not None:
        curated_bird_ship_margin_loss = _build_masked_pair_margin_loss(
            logits=logits,
            labels=labels,
            sample_mask=curated_bird_ship_mask,
            positive_index=bird_index,
            negative_index=ship_index,
            positive_labels={bird_index},
            negative_labels={ship_index},
            margin=float(getattr(config, "curated_bird_ship_margin", 0.15)),
        )
        loss = loss + curated_bird_ship_margin_weight * curated_bird_ship_margin_loss

    bird_ship_margin_loss_weight = float(getattr(config, "bird_ship_margin_loss_weight", 0.0))
    if bird_ship_margin_loss_weight > 0 and bird_index is not None and ship_index is not None:
        bird_ship_margin_loss = _build_pair_margin_loss(
            logits=logits,
            labels=labels,
            positive_index=bird_index,
            negative_index=ship_index,
            positive_labels={bird_index},
            negative_labels={ship_index},
            margin=float(getattr(config, "bird_ship_margin", 0.25)),
        )
        loss = loss + bird_ship_margin_loss_weight * bird_ship_margin_loss
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
    label_to_index: dict[str, int],
    curated_ship_drone_manifest = None,
    curated_bird_ship_manifest = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    accumulation_steps = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(data_loader, start=1):
        if bool(getattr(config, "use_session_bag_training", False)):
            clip_mask = batch["clip_mask"].to(device=device, dtype=torch.float32)
            track_features = batch["track_features"].to(device=device, dtype=torch.float32)
            track_global_features = batch["track_global_features"].to(device=device, dtype=torch.float32)
            track_short_features = batch.get("track_short_features")
            if track_short_features is not None:
                track_short_features = track_short_features.to(device=device, dtype=torch.float32)
            track_long_features = batch.get("track_long_features")
            if track_long_features is not None:
                track_long_features = track_long_features.to(device=device, dtype=torch.float32)
            rd_features = None
            rd_mask = None
            if _env_flag("RD", bool(config.include_rd)):
                rd_features = batch["rd_features"].to(device=device, dtype=torch.float32)
                rd_mask = batch["has_rd"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
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

            with torch.set_grad_enabled(training):
                outputs = model(track_features, track_global_features, track_short_features, track_long_features, rd_features, rd_mask, clip_mask)
                session_logits = outputs["session_logits"]
                loss = float(getattr(config, "session_loss_weight", 1.0)) * criterion(session_logits, labels)
                clip_logits = outputs.get("clip_logits")
                clip_loss_weight = float(getattr(config, "clip_loss_weight", 0.0))
                if clip_loss_weight > 0 and clip_logits is not None:
                    repeated_labels = labels.unsqueeze(1).expand(-1, clip_logits.size(1)).reshape(-1)
                    flat_clip_logits = clip_logits.reshape(-1, clip_logits.size(-1))
                    flat_clip_mask = clip_mask.reshape(-1) > 0.5
                    if flat_clip_mask.any():
                        loss = loss + clip_loss_weight * criterion(flat_clip_logits[flat_clip_mask], repeated_labels[flat_clip_mask])

                attention_weights = outputs.get("session_attention_weights")
                attention_entropy_loss_weight = float(getattr(config, "session_attention_entropy_loss_weight", 0.0))
                if attention_weights is not None and attention_entropy_loss_weight > 0:
                    safe_weights = attention_weights.clamp_min(1e-8)
                    entropy = -(safe_weights * safe_weights.log()).sum(dim=1)
                    valid_counts = clip_mask.sum(dim=1).clamp_min(1.0)
                    max_entropy = valid_counts.log()
                    normalized_entropy = entropy / max_entropy.clamp_min(1e-6)
                    loss = loss - attention_entropy_loss_weight * normalized_entropy.mean()

                session_track_router_logits = outputs.get("session_track_router_logits")
                session_track_router_weight = float(getattr(config, "session_track_router_aux_loss_weight", 0.0))
                if session_track_router_logits is not None and track_router_target is not None and session_track_router_weight > 0:
                    valid_mask = track_router_target >= 0
                    if valid_mask.any():
                        router_loss = auxiliary_criterion(session_track_router_logits[valid_mask], track_router_target[valid_mask])
                        loss = loss + session_track_router_weight * router_loss

                session_ship_drone_logits = outputs.get("session_ship_drone_logits")
                session_ship_drone_weight = float(getattr(config, "session_ship_drone_aux_loss_weight", 0.0))
                if session_ship_drone_logits is not None and ship_drone_target is not None and session_ship_drone_weight > 0:
                    valid_mask = ship_drone_target >= 0.0
                    if valid_mask.any():
                        ship_drone_loss = auxiliary_criterion(session_ship_drone_logits[valid_mask], ship_drone_target[valid_mask].long())
                        loss = loss + session_ship_drone_weight * ship_drone_loss

                session_ship_bird_logits = outputs.get("session_ship_bird_logits")
                session_ship_bird_weight = float(getattr(config, "session_ship_bird_aux_loss_weight", 0.0))
                if session_ship_bird_logits is not None and ship_bird_target is not None and session_ship_bird_weight > 0:
                    valid_mask = ship_bird_target >= 0.0
                    if valid_mask.any():
                        ship_bird_loss = auxiliary_criterion(session_ship_bird_logits[valid_mask], ship_bird_target[valid_mask].long())
                        loss = loss + session_ship_bird_weight * ship_bird_loss

                session_bird_drone_logits = outputs.get("session_bird_drone_logits")
                session_bird_drone_weight = float(getattr(config, "session_bird_drone_aux_loss_weight", 0.0))
                if session_bird_drone_logits is not None and bird_drone_target is not None and session_bird_drone_weight > 0:
                    valid_mask = bird_drone_target >= 0.0
                    if valid_mask.any():
                        bird_drone_loss = auxiliary_criterion(session_bird_drone_logits[valid_mask], bird_drone_target[valid_mask].long())
                        loss = loss + session_bird_drone_weight * bird_drone_loss

                session_bird_surface_logits = outputs.get("session_bird_surface_logits")
                session_bird_surface_weight = float(getattr(config, "session_bird_surface_aux_loss_weight", 0.0))
                if session_bird_surface_logits is not None and bird_surface_target is not None and session_bird_surface_weight > 0:
                    valid_mask = bird_surface_target >= 0.0
                    if valid_mask.any():
                        bird_surface_loss = bird_surface_criterion(session_bird_surface_logits[valid_mask], bird_surface_target[valid_mask])
                        loss = loss + session_bird_surface_weight * bird_surface_loss

                session_people_cyclist_logits = outputs.get("session_people_cyclist_logits")
                session_people_cyclist_weight = float(getattr(config, "session_people_cyclist_aux_loss_weight", 0.0))
                if session_people_cyclist_logits is not None and session_people_cyclist_weight > 0:
                    people_index = label_to_index.get("PeopleTarget")
                    cyclist_index = label_to_index.get("CyclistTarget")
                    if people_index is not None and cyclist_index is not None:
                        valid_mask = (labels == people_index) | (labels == cyclist_index)
                        if valid_mask.any():
                            pair_target = (labels[valid_mask] == cyclist_index).long()
                            people_cyclist_loss = auxiliary_criterion(session_people_cyclist_logits[valid_mask], pair_target)
                            loss = loss + session_people_cyclist_weight * people_cyclist_loss

                session_bird_clutter_other_logits = outputs.get("session_bird_clutter_other_logits")
                session_bird_clutter_other_weight = float(getattr(config, "session_bird_clutter_other_aux_loss_weight", 0.0))
                if session_bird_clutter_other_logits is not None and session_bird_clutter_other_weight > 0:
                    bird_index = label_to_index.get("BirdTarget")
                    clutter_index = label_to_index.get("ClutterTarget")
                    other_index = label_to_index.get("OtherTarget")
                    if bird_index is not None and clutter_index is not None and other_index is not None:
                        valid_mask = (labels == bird_index) | (labels == clutter_index) | (labels == other_index)
                        if valid_mask.any():
                            subset_labels = labels[valid_mask]
                            remapped = torch.full_like(subset_labels, fill_value=0)
                            remapped[subset_labels == bird_index] = 0
                            remapped[subset_labels == clutter_index] = 1
                            remapped[subset_labels == other_index] = 2
                            bird_clutter_other_loss = auxiliary_criterion(session_bird_clutter_other_logits[valid_mask], remapped)
                            loss = loss + session_bird_clutter_other_weight * bird_clutter_other_loss

                logits = session_logits
                if training:
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()
                    should_step = (batch_index % accumulation_steps == 0) or (batch_index == len(data_loader))
                    if should_step:
                        if config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            predictions = logits.argmax(dim=-1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            continue

        sample_ids = [str(item) for item in batch["sample_id"]]
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
        bird_rd_target = batch.get("bird_rd_target")
        if bird_rd_target is not None:
            bird_rd_target = bird_rd_target.to(device=device, dtype=torch.float32)
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
        ship_air_target = batch.get("ship_air_target")
        if ship_air_target is not None:
            ship_air_target = ship_air_target.to(device=device, dtype=torch.float32)
        suspicious_ship_hard_pattern = batch.get("suspicious_ship_hard_pattern")
        if suspicious_ship_hard_pattern is not None:
            suspicious_ship_hard_pattern = suspicious_ship_hard_pattern.to(device=device, dtype=torch.float32)
        curated_ship_drone_mask = None
        if curated_ship_drone_manifest is not None:
            curated_ship_drone_mask = torch.tensor(
                build_manifest_mask(sample_ids, curated_ship_drone_manifest),
                device=device,
                dtype=torch.float32,
            )
        curated_bird_ship_mask = None
        if curated_bird_ship_manifest is not None:
            curated_bird_ship_mask = torch.tensor(
                build_manifest_mask(sample_ids, curated_bird_ship_manifest),
                device=device,
                dtype=torch.float32,
            )
        rd_features = None
        rd_mask = None
        if _env_flag("RD", bool(config.include_rd)):
            rd_features = batch["rd_features"].to(device=device, dtype=torch.float32)
            rd_mask = batch["has_rd"].to(device=device, dtype=torch.float32).view(-1, 1)
        labels = batch["label"].to(device=device, dtype=torch.long)

        with torch.set_grad_enabled(training):
            outputs = model(track_features, track_global_features, track_short_features, track_long_features, rd_features, rd_mask)
            logits, loss = compute_losses(outputs, labels, criterion, auxiliary_criterion, notes_criterion, bird_drone_criterion, bird_surface_criterion, config, fine_to_coarse_index, label_to_index, rd_mask, notes_target, bird_rd_target, bird_drone_target, bird_surface_target, track_router_target, ship_drone_target, ship_bird_target, ship_air_target, suspicious_ship_hard_pattern, curated_ship_drone_mask, curated_bird_ship_mask)
            if training:
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                should_step = (batch_index % accumulation_steps == 0) or (batch_index == len(data_loader))
                if should_step:
                    if config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

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
    min_val_accuracy = float(getattr(config, "best_metric_min_val_accuracy", 0.0))
    min_val_macro_f1 = float(getattr(config, "best_metric_min_val_macro_f1", 0.0))
    if float(val_metrics.get("accuracy", 0.0)) < min_val_accuracy:
        return float("-inf")
    if float(val_metrics.get("macro_f1", 0.0)) < min_val_macro_f1:
        return float("-inf")
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
        "track_feature_by_source": dataset.get_track_normalization_by_source(),
        "track_global_center": dataset.track_global_center.tolist(),
        "track_global_scale": dataset.track_global_scale.tolist(),
        "track_global_by_source": dataset.get_track_global_normalization_by_source(),
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
    set_random_seed(config.random_seed, deterministic=bool(getattr(config, "deterministic", False)))

    dataset, train_loader, val_loader, test_loader, split_map = build_dataloaders(config, session_ids=session_ids)
    curated_ship_drone_manifest = load_curated_clip_manifest(getattr(config, "curated_ship_drone_manifest_path", ""))
    curated_bird_ship_manifest = load_curated_clip_manifest(getattr(config, "curated_bird_ship_manifest_path", ""))
    resolved_device = torch.device(device or "cpu")
    ordered_labels = [dataset.index_to_label[index] for index in sorted(dataset.index_to_label)]
    coarse_training_mode = str(getattr(config, 'label_granularity', 'fine')).lower() == 'coarse'
    if coarse_training_mode:
        config.hierarchical_aux_loss_weight = 0.0
        config.detail_aux_loss_weight = 0.0
        config.detail_boundary_loss_weight = 0.0

    coarse_to_index, fine_to_coarse_index = build_coarse_taxonomy(config, ordered_labels)
    fine_to_coarse_index = fine_to_coarse_index.to(resolved_device)

    clip_model = FusionRadarClassifier(
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
        confusion_pair_handler_enabled=bool(getattr(config, "confusion_pair_handler_enabled", False)),
        confusion_pair_handler_hidden_dim=int(getattr(config, "confusion_pair_handler_hidden_dim", 64)),
        confusion_pair_handler_dropout=float(getattr(config, "confusion_pair_handler_dropout", 0.1)),
        confusion_pair_handler_weight=float(getattr(config, "confusion_pair_handler_weight", 0.0)),
        confusion_pair_handler_gate_threshold=float(getattr(config, "confusion_pair_handler_gate_threshold", 0.55)),
        confusion_ship_drone_handler_weight=float(getattr(config, "confusion_ship_drone_handler_weight", 0.0)),
        confusion_bird_ship_handler_weight=float(getattr(config, "confusion_bird_ship_handler_weight", 0.0)),
        ship_drone_refiner_enabled=bool(getattr(config, "ship_drone_refiner_enabled", False)),
        ship_drone_refiner_hidden_dim=int(getattr(config, "ship_drone_refiner_hidden_dim", 128)),
        ship_drone_refiner_dropout=float(getattr(config, "ship_drone_refiner_dropout", 0.1)),
        ship_drone_refiner_logit_weight=float(getattr(config, "ship_drone_refiner_logit_weight", 0.0)),
        ship_drone_refiner_gate_threshold=float(getattr(config, "ship_drone_refiner_gate_threshold", 0.6)),
        rd_dominant_fusion_bias=float(getattr(config, "rd_dominant_fusion_bias", 0.35)),
        rd_unimodal_logit_residual_weight=float(getattr(config, "rd_unimodal_logit_residual_weight", 0.18)),
        min_rd_fusion_weight=float(getattr(config, "min_rd_fusion_weight", 0.15)),
        ambiguous_rd_fusion_boost=float(getattr(config, "ambiguous_rd_fusion_boost", 0.15)),
        surface_rd_fusion_boost=float(getattr(config, "surface_rd_fusion_boost", 0.1)),
        use_weighted_fusion_feature=bool(getattr(config, "use_weighted_fusion_feature", False)),
    ).to(resolved_device)

    if bool(getattr(config, "use_session_bag_training", False)):
        model = SessionBagRadarClassifier(
            clip_model=clip_model,
            num_classes=len(dataset.label_to_index),
            embedding_dim=config.fusion_hidden_size * 2,
            bag_hidden_dim=int(getattr(config, "session_bag_hidden_dim", config.fusion_hidden_size)),
            dropout=float(getattr(config, "session_bag_dropout", 0.1)),
            encoder_layers=int(getattr(config, "session_bag_encoder_layers", 2)),
            attention_heads=int(getattr(config, "session_bag_attention_heads", 4)),
            use_positional_encoding=bool(getattr(config, "session_bag_use_positional_encoding", True)),
            use_stat_pooling=bool(getattr(config, "session_bag_use_stat_pooling", True)),
        ).to(resolved_device)
    else:
        model = clip_model

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
    best_epoch = 0
    stale_epochs = 0
    fallback_checkpoint = output_dir / "last_fusion_model.pt"
    history: list[dict[str, Any]] = []
    num_parameters = count_trainable_parameters(model)

    logger.info(
        f"[*] Training samples: {len(train_loader.dataset)}, "
        f"validation samples: {0 if val_loader is None else len(val_loader.dataset)}, "
        f"classes: {dataset.index_to_label}"
    )
    if bool(getattr(config, "use_session_bag_training", False)):
        logger.info(
            "[*] Session bag mode | "
            f"bag_batch_size={int(getattr(config, 'session_bag_batch_size', 1))} | "
            f"max_clips={int(getattr(config, 'session_bag_max_clips', 32))} | "
            f"random_sampling={bool(getattr(config, 'session_bag_random_sampling', True))} | "
            f"shuffle={bool(getattr(config, 'session_bag_shuffle', True))} | "
            f"sort_by_length={bool(getattr(config, 'session_bag_sort_by_length', True))} | "
            f"grad_accum={max(1, int(getattr(config, 'gradient_accumulation_steps', 1)))}"
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
        if hasattr(model, "set_runtime_fusion_controls"):
            fusion_controls = _resolve_epoch_fusion_controls(config, epoch)
            model.set_runtime_fusion_controls(**fusion_controls)
        else:
            fusion_controls = {
                "min_rd_fusion_weight": float(getattr(config, "min_rd_fusion_weight", 0.15)),
                "ambiguous_rd_fusion_boost": float(getattr(config, "ambiguous_rd_fusion_boost", 0.15)),
                "surface_rd_fusion_boost": float(getattr(config, "surface_rd_fusion_boost", 0.1)),
            }
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
            label_to_index=dataset.label_to_index,
            curated_ship_drone_manifest=curated_ship_drone_manifest,
            curated_bird_ship_manifest=curated_bird_ship_manifest,
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
                label_to_index=dataset.label_to_index,
                curated_ship_drone_manifest=curated_ship_drone_manifest,
                curated_bird_ship_manifest=curated_bird_ship_manifest,
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
            "min_rd_fusion_weight": fusion_controls["min_rd_fusion_weight"],
            "ambiguous_rd_fusion_boost": fusion_controls["ambiguous_rd_fusion_boost"],
            "surface_rd_fusion_boost": fusion_controls["surface_rd_fusion_boost"],
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
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"rd_floor={fusion_controls['min_rd_fusion_weight']:.3f} | "
            f"rd_amb={fusion_controls['ambiguous_rd_fusion_boost']:.3f} | "
            f"rd_surface={fusion_controls['surface_rd_fusion_boost']:.3f}"
        )

        if scheduler is not None:
            scheduler.step()

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "label_to_index": dataset.label_to_index,
                "coarse_to_index": coarse_to_index,
                "fine_to_coarse_index": fine_to_coarse_index.detach().cpu().tolist(),
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "best_selection_metric": best_selection_metric,
                "num_parameters": num_parameters,
                "track_feature_center": dataset.track_feature_center.tolist(),
                "track_feature_scale": dataset.track_feature_scale.tolist(),
                "track_feature_by_source": dataset.get_track_normalization_by_source(),
                "track_global_center": dataset.track_global_center.tolist(),
                "track_global_scale": dataset.track_global_scale.tolist(),
                "track_global_by_source": dataset.get_track_global_normalization_by_source(),
                "num_notes": getattr(dataset, "note_dim", 0),
                "runtime_fusion_controls": fusion_controls,
            },
            fallback_checkpoint,
        )

        selection_metric = _compute_selection_metric(config, val_metrics)
        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]

        if selection_metric > best_selection_metric + float(getattr(config, "early_stopping_min_delta", 0.0)):
            best_selection_metric = selection_metric
            best_epoch = epoch
            stale_epochs = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "label_to_index": dataset.label_to_index,
                    "coarse_to_index": coarse_to_index,
                    "fine_to_coarse_index": fine_to_coarse_index.detach().cpu().tolist(),
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_macro_f1,
                    "best_selection_metric": best_selection_metric,
                    "num_parameters": num_parameters,
                    "track_feature_center": dataset.track_feature_center.tolist(),
                    "track_feature_scale": dataset.track_feature_scale.tolist(),
                    "track_feature_by_source": dataset.get_track_normalization_by_source(),
                    "track_global_center": dataset.track_global_center.tolist(),
                    "track_global_scale": dataset.track_global_scale.tolist(),
                    "track_global_by_source": dataset.get_track_global_normalization_by_source(),
                    "num_notes": getattr(dataset, "note_dim", 0),
                    "runtime_fusion_controls": fusion_controls,
                },
                best_checkpoint,
            )
            logger.info(
                f"[*] New best checkpoint | epoch={epoch:03d} | "
                f"selection={selection_metric:.4f} | stale_epochs={stale_epochs}"
            )
        else:
            stale_epochs += 1
            logger.info(
                f"[*] Early-stop monitor | epoch={epoch:03d} | "
                f"selection={selection_metric:.4f} | best_epoch={best_epoch:03d} | "
                f"best_selection={best_selection_metric:.4f} | stale_epochs={stale_epochs}"
            )

        if stale_epochs >= int(getattr(config, "early_stopping_patience", 999999)):
            logger.info(f"[*] Early stopping triggered at epoch {epoch:03d} | stale_epochs={stale_epochs}")
            break

    history_path = save_training_artifacts(output_dir, config, dataset, history, num_parameters, split_map, coarse_to_index, fine_to_coarse_index.detach().cpu())
    manifest = build_session_manifest(data_root=config.data_root, include_rd=_env_flag("RD", bool(config.include_rd)), split_map=split_map)
    manifest_path = save_session_manifest(manifest, output_dir)

    if not best_checkpoint.exists() and fallback_checkpoint.exists():
        logger.info("[*] No checkpoint satisfied the selection floor; falling back to the last epoch checkpoint.")
        best_checkpoint = fallback_checkpoint

    if best_checkpoint.exists():
        checkpoint_payload = load_checkpoint(best_checkpoint, map_location=resolved_device)
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        runtime_fusion_controls = checkpoint_payload.get("runtime_fusion_controls")
        if runtime_fusion_controls and hasattr(model, "set_runtime_fusion_controls"):
            model.set_runtime_fusion_controls(**runtime_fusion_controls)

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
            label_to_index=dataset.label_to_index,
            curated_ship_drone_manifest=curated_ship_drone_manifest,
            curated_bird_ship_manifest=curated_bird_ship_manifest,
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
        "best_epoch": best_epoch,
        "session_splits": split_map,
        "test_metrics": test_metrics,
    }
