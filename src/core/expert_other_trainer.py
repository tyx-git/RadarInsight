from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .device import resolve_cuda_device
from .experts import ConventionalAirExpert, GroundTargetExpert
from .expert_other_config import ExpertOtherTrainConfig
from .expert_other_data import build_subset_dataloaders
from .metrics import compute_classification_metrics
from .physical import PHYSICAL_FEATURE_COLUMNS
from .reporting import write_real_experiment_report
from .run_paths import prepare_run_output_dir
from .trajectory import TrackRepresentationBuilder


EXPERT_LABELS = {
    "expert2_air": ["AircraftTarget"],
    "expert4_ground": ["PeopleTarget", "CarTarget", "ShipTarget"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> tuple[torch.device, dict[str, object]]:
    return resolve_cuda_device(requested_device)


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(max(0.0, gamma))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        if self.gamma <= 0:
            return ce.mean()
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def _open_set_logits_from_similarity(similarity: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    centered = similarity - float(threshold)
    return torch.stack([-centered, centered], dim=-1) * float(scale)


def _thresholds_for_sources(
    source_names: list[str],
    default_threshold: float,
    source_thresholds: dict[str, float] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    thresholds = source_thresholds or {}
    values = [
        float(thresholds.get(str(source_name), float(default_threshold)))
        for source_name in source_names
    ]
    return torch.tensor(values, dtype=dtype, device=device)


def _open_set_logits_from_similarity_by_source(
    similarity: torch.Tensor,
    source_names: list[str],
    default_threshold: float,
    source_thresholds: dict[str, float] | None,
    scale: float,
) -> torch.Tensor:
    thresholds = _thresholds_for_sources(
        source_names=source_names,
        default_threshold=default_threshold,
        source_thresholds=source_thresholds,
        device=similarity.device,
        dtype=similarity.dtype,
    )
    centered = similarity - thresholds
    return torch.stack([-centered, centered], dim=-1) * float(scale)


def _binary_details(logits: torch.Tensor, labels: torch.Tensor, label_names: list[str]) -> dict[str, object]:
    preds = logits.argmax(dim=-1)
    details: dict[str, object] = {
        "prediction_distribution": {
            label_names[idx]: int((preds == idx).sum().item())
            for idx in range(len(label_names))
        },
        "per_class": [],
    }
    recalls: list[float] = []
    for idx, label_name in enumerate(label_names):
        pred_pos = preds == idx
        true_pos = labels == idx
        tp = int((pred_pos & true_pos).sum().item())
        fp = int((pred_pos & ~true_pos).sum().item())
        fn = int((~pred_pos & true_pos).sum().item())
        support = int(true_pos.sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        details["per_class"].append(
            {
                "label_name": label_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        if label_name == "AircraftTarget":
            details["aircraft_precision"] = precision
            details["aircraft_recall"] = recall
            details["aircraft_f1"] = f1
            details["aircraft_support"] = support
        elif label_name == "NonAircraftTarget":
            details["non_aircraft_precision"] = precision
            details["non_aircraft_recall"] = recall
            details["non_aircraft_f1"] = f1
            details["non_aircraft_support"] = support
    details["balanced_accuracy"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
    return details


def _calibrate_open_set_threshold(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    source_names: list[str],
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    logit_scale: float,
    min_aircraft_recall_for_best: float,
    min_aircraft_precision_for_best: float,
    max_raw_false_aircraft_rate_for_best: float,
    raw_false_aircraft_penalty: float,
    source_focus_name: str,
) -> dict[str, object]:
    feasible_best: dict[str, object] | None = None
    fallback_best: dict[str, object] | None = None
    steps = max(2, int(threshold_steps))
    aircraft_idx = label_names.index("AircraftTarget") if "AircraftTarget" in label_names else None
    raw_focus_mask = torch.tensor([str(source_name) == str(source_focus_name) for source_name in source_names], dtype=torch.bool)
    for threshold in torch.linspace(float(threshold_min), float(threshold_max), steps=steps):
        threshold_value = float(threshold.item())
        logits = _open_set_logits_from_similarity(similarities, threshold_value, logit_scale)
        metrics = compute_classification_metrics(logits, labels, num_classes=len(label_names))
        details = _binary_details(logits, labels, label_names)
        aircraft_recall = float(details.get("aircraft_recall", 0.0))
        aircraft_precision = float(details.get("aircraft_precision", 0.0))
        aircraft_f1 = float(details.get("aircraft_f1", 0.0))
        non_aircraft_recall = float(details.get("non_aircraft_recall", 0.0))
        raw_false_aircraft_rate = 0.0
        if aircraft_idx is not None and bool(raw_focus_mask.any()):
            raw_focus_labels = labels[raw_focus_mask]
            raw_focus_preds = logits.argmax(dim=-1)[raw_focus_mask]
            raw_focus_non_aircraft = raw_focus_labels != aircraft_idx
            raw_false_aircraft = int(((raw_focus_preds == aircraft_idx) & raw_focus_non_aircraft).sum().item())
            raw_focus_non_aircraft_count = int(raw_focus_non_aircraft.sum().item())
            raw_false_aircraft_rate = raw_false_aircraft / raw_focus_non_aircraft_count if raw_focus_non_aircraft_count > 0 else 0.0
        penalty = 0.0
        if aircraft_recall < float(min_aircraft_recall_for_best):
            penalty += 1.0 + float(min_aircraft_recall_for_best) - aircraft_recall
        if aircraft_precision < float(min_aircraft_precision_for_best):
            penalty += 1.0 + float(min_aircraft_precision_for_best) - aircraft_precision
        if raw_false_aircraft_rate > float(max_raw_false_aircraft_rate_for_best):
            penalty += 1.0 + raw_false_aircraft_rate - float(max_raw_false_aircraft_rate_for_best)
        score = (
            0.45 * metrics.macro_f1
            + 0.20 * aircraft_f1
            + 0.15 * aircraft_precision
            + 0.15 * aircraft_recall
            + 0.05 * non_aircraft_recall
            - float(raw_false_aircraft_penalty) * raw_false_aircraft_rate
            - penalty
        )
        candidate = {
            "threshold": threshold_value,
            "score": score,
            "accuracy": metrics.accuracy,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "macro_f1": metrics.macro_f1,
            "aircraft_precision": aircraft_precision,
            **details,
            "raw_false_aircraft_rate": raw_false_aircraft_rate,
        }
        feasible = (
            aircraft_recall >= float(min_aircraft_recall_for_best)
            and aircraft_precision >= float(min_aircraft_precision_for_best)
            and raw_false_aircraft_rate <= float(max_raw_false_aircraft_rate_for_best)
        )
        if feasible and (feasible_best is None or float(candidate["score"]) > float(feasible_best["score"])):
            feasible_best = candidate
        if fallback_best is None or float(candidate["score"]) > float(fallback_best["score"]):
            fallback_best = candidate
    if feasible_best is not None:
        feasible_best["feasible"] = True
        return feasible_best
    if fallback_best is None:
        raise ValueError("failed to calibrate open-set threshold")
    fallback_best["feasible"] = False
    return fallback_best


def _calibrate_source_open_set_thresholds(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    source_names: list[str],
    default_threshold: float,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    logit_scale: float,
    min_aircraft_recall_for_best: float,
    min_aircraft_precision_for_best: float,
    max_raw_false_aircraft_rate_for_best: float,
    raw_false_aircraft_penalty: float,
    source_focus_name: str,
    min_source_positive_support_for_threshold: int,
) -> dict[str, object]:
    aircraft_idx = label_names.index("AircraftTarget") if "AircraftTarget" in label_names else None
    source_thresholds: dict[str, float] = {}
    source_details: dict[str, object] = {}
    unique_sources = sorted({str(source_name) for source_name in source_names})
    step = (float(threshold_max) - float(threshold_min)) / max(1, int(threshold_steps) - 1)
    for source_name in unique_sources:
        source_mask = torch.tensor([str(item) == source_name for item in source_names], dtype=torch.bool)
        source_similarities = similarities[source_mask]
        source_labels = labels[source_mask]
        if source_similarities.numel() == 0:
            continue
        positive_count = int((source_labels == aircraft_idx).sum().item()) if aircraft_idx is not None else 0
        negative_count = int((source_labels != aircraft_idx).sum().item()) if aircraft_idx is not None else int(source_labels.numel())
        if source_name != str(source_focus_name) and positive_count < int(min_source_positive_support_for_threshold):
            threshold = float(default_threshold)
            info = {
                "threshold": threshold,
                "source": "insufficient_positive_support_default_threshold",
                "support": int(source_similarities.numel()),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "source_false_aircraft_rate": 0.0,
            }
            source_details[source_name] = info
            continue
        if positive_count == 0:
            threshold = min(float(threshold_max), max(float(threshold_min), float(source_similarities.max().item()) + max(step, 1e-3)))
            info = {
                "threshold": threshold,
                "source": "negative_only_reject_all",
                "support": int(source_similarities.numel()),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "source_false_aircraft_rate": 0.0,
            }
        elif negative_count == 0:
            threshold = min(float(threshold_max), max(float(threshold_min), float(source_similarities.min().item()) - max(step, 1e-3)))
            info = {
                "threshold": threshold,
                "source": "positive_only_accept_all",
                "support": int(source_similarities.numel()),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "source_false_aircraft_rate": 0.0,
            }
        elif source_name == str(source_focus_name):
            info = _calibrate_open_set_threshold(
                source_similarities,
                source_labels,
                label_names,
                [source_name for _ in range(int(source_similarities.numel()))],
                threshold_min,
                threshold_max,
                threshold_steps,
                logit_scale,
                min_aircraft_recall_for_best,
                min_aircraft_precision_for_best,
                max_raw_false_aircraft_rate_for_best,
                raw_false_aircraft_penalty,
                source_focus_name=source_name,
            )
            threshold = float(info["threshold"])
            info = {
                **info,
                "support": int(source_similarities.numel()),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "source_false_aircraft_rate": float(info.get("raw_false_aircraft_rate", 0.0)),
            }
        else:
            feasible_candidate: dict[str, object] | None = None
            fallback_candidate: dict[str, object] | None = None
            focus_rate = float(max_raw_false_aircraft_rate_for_best)
            for threshold in torch.linspace(float(threshold_min), float(threshold_max), steps=max(2, int(threshold_steps))):
                threshold_value = float(threshold.item())
                logits = _open_set_logits_from_similarity(source_similarities, threshold_value, logit_scale)
                metrics = compute_classification_metrics(logits, source_labels, num_classes=len(label_names))
                details = _binary_details(logits, source_labels, label_names)
                aircraft_recall = float(details.get("aircraft_recall", 0.0))
                aircraft_precision = float(details.get("aircraft_precision", 0.0))
                aircraft_f1 = float(details.get("aircraft_f1", 0.0))
                non_aircraft_recall = float(details.get("non_aircraft_recall", 0.0))
                source_false_aircraft_rate = 0.0
                source_non_aircraft_count = 0
                if aircraft_idx is not None:
                    source_preds = logits.argmax(dim=-1)
                    source_non_aircraft = source_labels != aircraft_idx
                    source_false_aircraft = int(((source_preds == aircraft_idx) & source_non_aircraft).sum().item())
                    source_non_aircraft_count = int(source_non_aircraft.sum().item())
                    source_false_aircraft_rate = (
                        source_false_aircraft / source_non_aircraft_count if source_non_aircraft_count > 0 else 0.0
                    )
                precision_shortfall = max(0.0, float(min_aircraft_precision_for_best) - aircraft_precision)
                recall_shortfall = max(0.0, float(min_aircraft_recall_for_best) - aircraft_recall)
                source_fp_shortfall = max(0.0, source_false_aircraft_rate - focus_rate)
                penalty = (
                    1.25 * precision_shortfall
                    + 0.75 * recall_shortfall
                    + 2.50 * source_fp_shortfall
                )
                score = (
                    0.18 * metrics.macro_f1
                    + 0.14 * aircraft_f1
                    + 0.34 * aircraft_precision
                    + 0.14 * aircraft_recall
                    + 0.10 * non_aircraft_recall
                    - 3.00 * source_false_aircraft_rate
                    - penalty
                )
                candidate = {
                    "threshold": threshold_value,
                    "score": score,
                    "accuracy": metrics.accuracy,
                    "macro_precision": metrics.macro_precision,
                    "macro_recall": metrics.macro_recall,
                    "macro_f1": metrics.macro_f1,
                    "aircraft_precision": aircraft_precision,
                    "source_false_aircraft_rate": source_false_aircraft_rate,
                    "source_non_aircraft_count": source_non_aircraft_count,
                    **details,
                }
                feasible = (
                    aircraft_recall >= float(min_aircraft_recall_for_best)
                    and aircraft_precision >= float(min_aircraft_precision_for_best)
                    and source_false_aircraft_rate <= focus_rate
                )
                if feasible and (feasible_candidate is None or float(candidate["score"]) > float(feasible_candidate["score"])):
                    feasible_candidate = {**candidate, "feasible": True}
                if fallback_candidate is None or float(candidate["score"]) > float(fallback_candidate["score"]):
                    fallback_candidate = {**candidate, "feasible": False}
            best_candidate = feasible_candidate or fallback_candidate
            if best_candidate is None:
                continue
            info = {
                **best_candidate,
                "support": int(source_similarities.numel()),
                "positive_count": positive_count,
                "negative_count": negative_count,
            }
            threshold = float(info["threshold"])
        source_thresholds[source_name] = float(threshold)
        source_details[source_name] = info
    if source_thresholds:
        default_threshold = float(sum(source_thresholds.values()) / len(source_thresholds))
    logits = _open_set_logits_from_similarity_by_source(
        similarities,
        source_names,
        default_threshold=default_threshold,
        source_thresholds=source_thresholds,
        scale=logit_scale,
    )
    metrics = compute_classification_metrics(logits, labels, num_classes=len(label_names))
    details = _binary_details(logits, labels, label_names)
    raw_false_aircraft_rate = 0.0
    max_source_false_aircraft_rate = 0.0
    if aircraft_idx is not None:
        raw_focus_mask = torch.tensor([str(source_name) == str(source_focus_name) for source_name in source_names], dtype=torch.bool)
        if bool(raw_focus_mask.any()):
            raw_focus_labels = labels[raw_focus_mask]
            raw_focus_preds = logits.argmax(dim=-1)[raw_focus_mask]
            raw_focus_non_aircraft = raw_focus_labels != aircraft_idx
            raw_false_aircraft = int(((raw_focus_preds == aircraft_idx) & raw_focus_non_aircraft).sum().item())
            raw_focus_non_aircraft_count = int(raw_focus_non_aircraft.sum().item())
            raw_false_aircraft_rate = raw_false_aircraft / raw_focus_non_aircraft_count if raw_focus_non_aircraft_count > 0 else 0.0
        preds = logits.argmax(dim=-1)
        for source_name in unique_sources:
            source_mask = torch.tensor([str(item) == source_name for item in source_names], dtype=torch.bool)
            if not bool(source_mask.any()):
                continue
            source_labels = labels[source_mask]
            source_preds = preds[source_mask]
            source_non_aircraft = source_labels != aircraft_idx
            source_non_aircraft_count = int(source_non_aircraft.sum().item())
            if source_non_aircraft_count <= 0:
                continue
            source_false_aircraft = int(((source_preds == aircraft_idx) & source_non_aircraft).sum().item())
            source_false_aircraft_rate = source_false_aircraft / source_non_aircraft_count
            max_source_false_aircraft_rate = max(max_source_false_aircraft_rate, source_false_aircraft_rate)
    aircraft_recall = float(details.get("aircraft_recall", 0.0))
    aircraft_precision = float(details.get("aircraft_precision", 0.0))
    feasible = (
        aircraft_recall >= float(min_aircraft_recall_for_best)
        and aircraft_precision >= float(min_aircraft_precision_for_best)
        and raw_false_aircraft_rate <= float(max_raw_false_aircraft_rate_for_best)
        and max_source_false_aircraft_rate <= float(max_raw_false_aircraft_rate_for_best)
    )
    score = (
        0.45 * metrics.macro_f1
        + 0.20 * float(details.get("aircraft_f1", 0.0))
        + 0.15 * aircraft_precision
        + 0.15 * aircraft_recall
        + 0.05 * float(details.get("non_aircraft_recall", 0.0))
        - float(raw_false_aircraft_penalty) * raw_false_aircraft_rate
        - 0.75 * max_source_false_aircraft_rate
    )
    if not feasible:
        score -= 0.5
    return {
        "threshold": float(default_threshold),
        "source": "validation_source_calibrated",
        "source_thresholds": source_thresholds,
        "source_details": source_details,
        "score": score,
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "raw_false_aircraft_rate": raw_false_aircraft_rate,
        "max_source_false_aircraft_rate": max_source_false_aircraft_rate,
        "feasible": feasible,
        **details,
    }


def _calibrate_groundness_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    steps = max(2, int(threshold_steps))
    for threshold in torch.linspace(float(threshold_min), float(threshold_max), steps=steps):
        threshold_value = float(threshold.item())
        preds = probabilities >= threshold_value
        positives = labels.bool()
        tp = int((preds & positives).sum().item())
        fp = int((preds & ~positives).sum().item())
        fn = int((~preds & positives).sum().item())
        tn = int((~preds & ~positives).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        score = f1 + 0.05 * min(recall, specificity)
        candidate = {
            "threshold": threshold_value,
            "score": score,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate
    if best is None:
        raise ValueError("failed to calibrate groundness threshold")
    return best


def _groundness_details(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, object]:
    preds = probabilities >= float(threshold)
    positives = labels.bool()
    tp = int((preds & positives).sum().item())
    fp = int((preds & ~positives).sum().item())
    fn = int((~preds & positives).sum().item())
    tn = int((~preds & ~positives).sum().item())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "groundness_threshold": float(threshold),
        "groundness_precision": precision,
        "groundness_recall": recall,
        "groundness_specificity": specificity,
        "groundness_f1": f1,
        "groundness_tp": tp,
        "groundness_fp": fp,
        "groundness_fn": fn,
        "groundness_tn": tn,
    }


def _calibration_separation_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    positive_target: float,
    negative_target: float,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    positive_mask = labels.bool()
    negative_mask = ~positive_mask
    loss = scores.new_tensor(0.0)
    if bool(positive_mask.any()):
        positive_scores = scores[positive_mask]
        loss = loss + float(positive_weight) * torch.relu(float(positive_target) - positive_scores).pow(2).mean()
    if bool(negative_mask.any()):
        negative_scores = scores[negative_mask]
        loss = loss + float(negative_weight) * torch.relu(negative_scores - float(negative_target)).pow(2).mean()
    return loss


def _set_track_encoder_trainable(representation_builder: TrackRepresentationBuilder, trainable: bool) -> None:
    for param in representation_builder.sequence_encoder.parameters():
        param.requires_grad = trainable


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


def _load_shared_representation_builder(
    representation_builder: TrackRepresentationBuilder,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "representation_builder_state_dict" in checkpoint:
        state_dict = checkpoint["representation_builder_state_dict"]
    else:
        raw_state = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            key[len("representation_builder."):] if key.startswith("representation_builder.") else key: value
            for key, value in raw_state.items()
            if key.startswith("representation_builder.")
        }
    if not state_dict:
        encoder_state = {
            key[len("encoder."):] if key.startswith("encoder.") else key: value
            for key, value in checkpoint.get("model_state_dict", checkpoint).items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(f"no representation_builder or encoder.* weights found in shared checkpoint: {checkpoint_path}")
        missing, unexpected = representation_builder.sequence_encoder.load_state_dict(encoder_state, strict=False)
        return {
            "loaded_keys": len(encoder_state),
            "loaded_scope": "sequence_encoder",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    missing, unexpected = representation_builder.load_state_dict(state_dict, strict=False)
    return {
        "loaded_keys": len(state_dict),
        "loaded_scope": "representation_builder",
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _fit_source_feature_stats(loader) -> dict[str, dict[str, torch.Tensor]]:
    buckets: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        for idx, source_name in enumerate(batch.source_names):
            buckets.setdefault(str(source_name), []).append(batch.features[idx, : batch.lengths[idx]].float())
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for source_name, tensors in buckets.items():
        cat = torch.cat(tensors, dim=0)
        stats[source_name] = {
            "center": cat.mean(dim=0),
            "scale": cat.std(dim=0).clamp_min(1e-3),
        }
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


def _run_epoch(
    representation_builder,
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    global_to_local: dict[int, int],
    labels: list[str],
    expert_name: str,
    source_feature_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    positive_noise_std: float = 0.0,
    config: ExpertOtherTrainConfig | None = None,
) -> dict[str, object]:
    representation_builder.train(mode=train)
    model.train(mode=train)
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    source_names_all: list[str] = []
    label_names_all: list[str] = []
    sample_ids_all: list[str] = []
    loss_values: list[float] = []
    aircraft_similarity_all: list[torch.Tensor] = []
    groundness_probabilities_all: list[torch.Tensor] = []
    groundness_labels_all: list[torch.Tensor] = []
    groundness_threshold = float(getattr(model, "groundness_threshold", torch.tensor(0.5)).detach().cpu().item()) if hasattr(model, "groundness_threshold") else 0.5

    for batch in loader:
        features = batch.features.to(device)
        if source_feature_stats is not None:
            features = _apply_source_feature_norm(features, batch.source_names, source_feature_stats)
        physical = batch.physical_features.to(device)
        if expert_name == "expert2_air":
            labels_local = torch.tensor(
                [1 if name == "AircraftTarget" else 0 for name in batch.label_names],
                dtype=torch.long,
                device=device,
            )
        else:
            labels_local = torch.tensor(
                [global_to_local.get(int(label), 0) for label in batch.labels.tolist()],
                dtype=torch.long,
                device=device,
            )
            groundness_labels = torch.tensor(
                [1 if name in EXPERT_LABELS["expert4_ground"] else 0 for name in batch.label_names],
                dtype=torch.long,
                device=device,
            )
        with torch.set_grad_enabled(train):
            if train and any(param.requires_grad for param in representation_builder.parameters()):
                rep = representation_builder(
                    sequence_features=features,
                    physical_features=physical,
                    padding_mask=batch.padding_mask.to(device),
                )
            else:
                with torch.no_grad():
                    rep = representation_builder(
                        sequence_features=features,
                        physical_features=physical,
                        padding_mask=batch.padding_mask.to(device),
                    )
            if expert_name == "expert2_air":
                output = model(
                    rep.sequence_features,
                    rep.raw_physical_features,
                    padding_mask=batch.padding_mask.to(device),
                    source_names=batch.source_names,
                )
            else:
                output = model(rep.deep_features, rep.raw_physical_features)
            if expert_name == "expert4_ground" and output.auxiliary:
                ground_mask = groundness_labels.bool()
                if ground_mask.any():
                    loss = criterion(output.logits[ground_mask], labels_local[ground_mask])
                else:
                    loss = output.logits.sum() * 0.0
                groundness_logits = output.auxiliary["groundness_logits"]
                groundness_weight = getattr(criterion, "groundness_weight", None)
                groundness_loss = F.cross_entropy(
                    groundness_logits,
                    groundness_labels,
                    weight=groundness_weight,
                )
                loss = loss + float(getattr(criterion, "groundness_loss_weight", 0.0)) * groundness_loss
                if config is not None and float(config.groundness_separation_loss_weight) > 0:
                    loss = loss + float(config.groundness_separation_loss_weight) * _calibration_separation_loss(
                        output.auxiliary["groundness_probability"],
                        groundness_labels,
                        positive_target=config.groundness_positive_target,
                        negative_target=config.groundness_negative_target,
                        positive_weight=1.0,
                        negative_weight=1.0,
                    )
            else:
                loss = criterion(output.logits, labels_local)
                if (
                    expert_name == "expert2_air"
                    and config is not None
                    and output.auxiliary
                    and "aircraft_similarity" in output.auxiliary
                ):
                    similarity = output.auxiliary["aircraft_similarity"]
                    positive_mask = labels_local == 1
                    negative_mask = ~positive_mask
                    if float(config.open_set_similarity_loss_weight) > 0:
                        negative_weights = torch.ones_like(similarity)
                        if config.open_set_hard_negative_source_name:
                            source_hard = torch.tensor(
                                [
                                    str(source_name) == str(config.open_set_hard_negative_source_name)
                                    for source_name in batch.source_names
                                ],
                                dtype=torch.bool,
                                device=device,
                            )
                            negative_weights = torch.where(
                                negative_mask & source_hard,
                                torch.full_like(negative_weights, float(config.open_set_hard_negative_weight)),
                                negative_weights,
                            )
                        if config.open_set_hard_negative_label_names:
                            label_hard = torch.tensor(
                                [
                                    str(label_name) in set(config.open_set_hard_negative_label_names)
                                    for label_name in batch.label_names
                                ],
                                dtype=torch.bool,
                                device=device,
                            )
                            negative_weights = torch.where(
                                negative_mask & label_hard,
                                negative_weights * float(config.open_set_hard_negative_label_weight),
                                negative_weights,
                            )
                        positive_loss = similarity.new_tensor(0.0)
                        negative_loss = similarity.new_tensor(0.0)
                        if bool(positive_mask.any()):
                            positive_loss = torch.relu(
                                float(config.open_set_positive_similarity_target) - similarity[positive_mask]
                            ).pow(2).mean()
                        if bool(negative_mask.any()):
                            raw_negative_loss = torch.relu(
                                similarity[negative_mask] - float(config.open_set_negative_similarity_target)
                            ).pow(2)
                            negative_loss = (
                                raw_negative_loss * negative_weights[negative_mask]
                            ).sum() / negative_weights[negative_mask].sum().clamp_min(1.0)
                        loss = loss + float(config.open_set_similarity_loss_weight) * (positive_loss + negative_loss)
                    if (
                        float(config.open_set_pairwise_loss_weight) > 0
                        and bool(positive_mask.any())
                        and bool(negative_mask.any())
                    ):
                        positive_scores = similarity[positive_mask]
                        negative_scores = similarity[negative_mask]
                        pairwise_loss = torch.relu(
                            float(config.open_set_pairwise_margin)
                            - (positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0))
                        ).pow(2).mean()
                        loss = loss + float(config.open_set_pairwise_loss_weight) * pairwise_loss
            if train and expert_name == "expert2_air" and positive_noise_std > 0 and labels_local.eq(1).any():
                positive_mask = labels_local == 1
                noised_features = features.clone()
                noise = torch.randn_like(noised_features[positive_mask]) * float(positive_noise_std)
                noised_features[positive_mask] = noised_features[positive_mask] + noise
                if any(param.requires_grad for param in representation_builder.parameters()):
                    aug_rep = representation_builder(
                        sequence_features=noised_features,
                        physical_features=physical,
                        padding_mask=batch.padding_mask.to(device),
                    )
                else:
                    with torch.no_grad():
                        aug_rep = representation_builder(
                            sequence_features=noised_features,
                            physical_features=physical,
                            padding_mask=batch.padding_mask.to(device),
                        )
                aug_output = model(
                    aug_rep.sequence_features,
                    aug_rep.raw_physical_features,
                    padding_mask=batch.padding_mask.to(device),
                    source_names=batch.source_names,
                )
                aug_loss = criterion(aug_output.logits[positive_mask], labels_local[positive_mask])
                if (
                    config is not None
                    and float(config.open_set_similarity_loss_weight) > 0
                    and aug_output.auxiliary
                    and "aircraft_similarity" in aug_output.auxiliary
                ):
                    aug_similarity = aug_output.auxiliary["aircraft_similarity"][positive_mask]
                    aug_loss = aug_loss + float(config.open_set_similarity_loss_weight) * torch.relu(
                        float(config.open_set_positive_similarity_target) - aug_similarity
                    ).pow(2).mean()
                loss = loss + 0.5 * aug_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if expert_name == "expert4_ground":
            ground_mask_cpu = groundness_labels.detach().cpu().bool()
            if bool(ground_mask_cpu.any()):
                logits_all.append(output.logits.detach().cpu()[ground_mask_cpu])
                labels_all.append(labels_local.detach().cpu()[ground_mask_cpu])
                source_names_all.extend(
                    source_name
                    for source_name, keep in zip(batch.source_names, ground_mask_cpu.tolist())
                    if keep
                )
                label_names_all.extend(
                    label_name
                    for label_name, keep in zip(batch.label_names, ground_mask_cpu.tolist())
                    if keep
                )
                sample_ids_all.extend(
                    sample_id
                    for sample_id, keep in zip(batch.sample_ids, ground_mask_cpu.tolist())
                    if keep
                )
        else:
            logits_all.append(output.logits.detach().cpu())
            labels_all.append(labels_local.detach().cpu())
            source_names_all.extend(batch.source_names)
            label_names_all.extend(batch.label_names)
            sample_ids_all.extend(batch.sample_ids)
        if output.auxiliary and "aircraft_similarity" in output.auxiliary:
            aircraft_similarity_all.append(output.auxiliary["aircraft_similarity"].detach().cpu())
        if output.auxiliary and "groundness_probability" in output.auxiliary:
            groundness_probabilities_all.append(output.auxiliary["groundness_probability"].detach().cpu())
            groundness_labels_all.append(groundness_labels.detach().cpu())
        loss_values.append(float(loss.detach().cpu().item()))

    logits_cat = torch.cat(logits_all, dim=0) if logits_all else torch.empty((0, len(labels)), dtype=torch.float32)
    labels_cat = torch.cat(labels_all, dim=0) if labels_all else torch.empty((0,), dtype=torch.long)
    metrics = compute_classification_metrics(logits_cat, labels_cat, num_classes=len(labels))
    binary_details = _binary_details(logits_cat, labels_cat, labels)
    groundness_probabilities = torch.cat(groundness_probabilities_all, dim=0) if groundness_probabilities_all else None
    groundness_labels_cat = torch.cat(groundness_labels_all, dim=0) if groundness_labels_all else None
    groundness_metrics = (
        _groundness_details(groundness_probabilities, groundness_labels_cat, groundness_threshold)
        if groundness_probabilities is not None and groundness_labels_cat is not None
        else {}
    )
    return {
        "loss": float(np.mean(loss_values)),
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        **binary_details,
        **groundness_metrics,
        "logits": logits_cat,
        "labels": labels_cat,
        "aircraft_similarity": torch.cat(aircraft_similarity_all, dim=0) if aircraft_similarity_all else None,
        "groundness_probability": groundness_probabilities,
        "groundness_labels": groundness_labels_cat,
        "source_names": source_names_all,
        "label_names": label_names_all,
        "sample_ids": sample_ids_all,
    }


def train_other_expert(config: ExpertOtherTrainConfig) -> dict[str, object]:
    if config.expert_name not in EXPERT_LABELS:
        raise ValueError(f"unsupported expert_name: {config.expert_name}")
    requested_labels = EXPERT_LABELS[config.expert_name]
    set_seed(config.seed)
    device, device_info = resolve_device(config.device)
    run_paths = prepare_run_output_dir(config.output_dir)
    loaders = build_subset_dataloaders(
        data_root=config.data_root,
        labels=requested_labels,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        balanced_sampling=bool(config.balanced_sampling and config.expert_name in {"expert2_air", "expert4_ground"}),
        balanced_positive_multiplier=config.balanced_positive_multiplier,
        include_ground_hard_negatives=bool(
            config.include_ground_hard_negatives and config.expert_name == "expert4_ground"
        ),
        source_sampling_name=(
            config.open_set_source_sampling_name if config.expert_name == "expert2_air" else ""
        ),
        source_sampling_multiplier=(
            config.open_set_source_sampling_multiplier if config.expert_name == "expert2_air" else 1.0
        ),
    )
    labels = loaders.labels

    if config.expert_name == "expert2_air":
        model = ConventionalAirExpert(
            sequence_dim=128,
            physical_dim=len(PHYSICAL_FEATURE_COLUMNS),
            dropout=config.dropout,
            open_set=bool(config.open_set_aircraft),
            logit_scale=config.open_set_logit_scale,
            decision_threshold=config.open_set_initial_threshold,
        ).to(device)
    else:
        model = GroundTargetExpert(
            deep_dim=128,
            raw_physical_dim=10,
            dropout=config.dropout,
            groundness_threshold=config.groundness_initial_threshold,
        ).to(device)
    representation_builder = TrackRepresentationBuilder(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=128,
        dropout=config.dropout,
    ).to(device)
    pretrained_track_info = None
    shared_representation_info = None
    if config.shared_representation_checkpoint:
        shared_representation_info = _load_shared_representation_builder(
            representation_builder,
            config.shared_representation_checkpoint,
            device,
        )
    if config.pretrained_track_checkpoint and not config.shared_representation_checkpoint:
        pretrained_track_info = _load_pretrained_track_encoder(
            representation_builder,
            config.pretrained_track_checkpoint,
            device,
        )
    if not config.train_representation_builder:
        for param in representation_builder.parameters():
            param.requires_grad = False

    optimizer_params = list(model.parameters())
    if config.train_representation_builder:
        optimizer_params = list(representation_builder.parameters()) + optimizer_params
    optimizer = torch.optim.AdamW(optimizer_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.expert_name == "expert2_air":
        class_weight = torch.tensor(
            [float(config.negative_class_weight), float(config.positive_class_weight)],
            dtype=torch.float32,
            device=device,
        )
        criterion = FocalCrossEntropyLoss(weight=class_weight, gamma=config.focal_loss_gamma)
    else:
        class_weight = None
        if config.use_ground_class_weighting:
            train_label_counts = loaders.train_loader.dataset.index_df["label_name"].astype(str).value_counts().to_dict()
            max_count = max(1, max((int(count) for count in train_label_counts.values()), default=1))
            class_weight_values = [
                float(max_count) / max(1.0, float(train_label_counts.get(label_name, 1)))
                for label_name in labels
            ]
            class_weight = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        groundness_weight = torch.tensor(
            [float(config.groundness_negative_weight), float(config.groundness_positive_weight)],
            dtype=torch.float32,
            device=device,
        )
        criterion.groundness_weight = groundness_weight
        criterion.groundness_loss_weight = float(config.groundness_loss_weight)
        criterion.groundness_separation_loss_weight = float(config.groundness_separation_loss_weight)
    source_feature_stats = None
    checkpoint_source_feature_stats = None
    if config.source_normalization:
        fitted_source_feature_stats = _fit_source_feature_stats(loaders.train_loader)
        checkpoint_source_feature_stats = _merge_source_feature_stats(
            _source_feature_stats_from_checkpoint(config.shared_representation_checkpoint, device)
            if config.shared_representation_checkpoint
            else None,
            _source_feature_stats_from_checkpoint(config.pretrained_track_checkpoint, device)
            if config.pretrained_track_checkpoint
            else None,
        )
        source_feature_stats = _merge_source_feature_stats(fitted_source_feature_stats, checkpoint_source_feature_stats)

    print(f"[*] Using device: {device}")
    print(f"[*] Trainable parameters: {_count_trainable_parameters(representation_builder) + _count_trainable_parameters(model):,}")

    history: list[dict[str, object]] = []
    best_val = float("-inf")
    best_threshold = float(config.open_set_initial_threshold)
    best_source_thresholds: dict[str, float] = {}
    best_groundness_threshold = float(config.groundness_initial_threshold)
    best_selection: dict[str, object] = {}
    best_path = run_paths.output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        _set_track_encoder_trainable(representation_builder, bool(config.train_representation_builder))
        train_metrics = _run_epoch(
            representation_builder,
            model,
            loaders.train_loader,
            criterion,
            optimizer,
            device,
            True,
            loaders.global_to_local,
            labels,
            config.expert_name,
            source_feature_stats,
            positive_noise_std=config.aircraft_positive_noise_std if config.expert_name == "expert2_air" else 0.0,
            config=config,
        )
        val_metrics = _run_epoch(
            representation_builder,
            model,
            loaders.val_loader,
            criterion,
            optimizer,
            device,
            False,
            loaders.global_to_local,
            labels,
            config.expert_name,
            source_feature_stats,
            config=config,
        )
        selection_metric = float(val_metrics["macro_f1"])
        threshold_info: dict[str, object] | None = None
        if config.expert_name == "expert2_air" and config.open_set_aircraft and val_metrics.get("aircraft_similarity") is not None:
            global_threshold_info = _calibrate_open_set_threshold(
                val_metrics["aircraft_similarity"],
                val_metrics["labels"],
                labels,
                val_metrics["source_names"],
                config.threshold_min,
                config.threshold_max,
                config.threshold_steps,
                config.open_set_logit_scale,
                config.open_set_min_aircraft_recall_for_best,
                config.open_set_min_aircraft_precision_for_best,
                config.open_set_max_raw_false_aircraft_rate_for_best,
                config.open_set_raw_false_aircraft_penalty,
                config.open_set_source_focus_name,
            )
            source_threshold_info = _calibrate_source_open_set_thresholds(
                val_metrics["aircraft_similarity"],
                val_metrics["labels"],
                labels,
                val_metrics["source_names"],
                float(global_threshold_info["threshold"]),
                config.threshold_min,
                config.threshold_max,
                config.threshold_steps,
                config.open_set_logit_scale,
                config.open_set_min_aircraft_recall_for_best,
                config.open_set_min_aircraft_precision_for_best,
                config.open_set_max_raw_false_aircraft_rate_for_best,
                config.open_set_raw_false_aircraft_penalty,
                config.open_set_source_focus_name,
                config.open_set_min_source_positive_support_for_threshold,
            )
            threshold_info = (
                source_threshold_info
                if float(source_threshold_info["score"]) >= float(global_threshold_info["score"])
                else global_threshold_info
            )
            calibrated_logits = _open_set_logits_from_similarity_by_source(
                val_metrics["aircraft_similarity"],
                val_metrics["source_names"],
                default_threshold=float(threshold_info["threshold"]),
                source_thresholds=threshold_info.get("source_thresholds") if isinstance(threshold_info, dict) else None,
                scale=config.open_set_logit_scale,
            )
            calibrated_metrics = compute_classification_metrics(calibrated_logits, val_metrics["labels"], num_classes=len(labels))
            calibrated_details = _binary_details(calibrated_logits, val_metrics["labels"], labels)
            val_metrics = {
                **val_metrics,
                "calibrated_threshold": threshold_info["threshold"],
                "calibrated_accuracy": calibrated_metrics.accuracy,
                "calibrated_macro_precision": calibrated_metrics.macro_precision,
                "calibrated_macro_recall": calibrated_metrics.macro_recall,
                "calibrated_macro_f1": calibrated_metrics.macro_f1,
                "calibrated_details": calibrated_details,
            }
            selection_metric = float(calibrated_metrics.macro_f1)
            selection_metric += 0.20 * float(calibrated_details.get("aircraft_f1", 0.0))
            selection_metric += 0.10 * float(calibrated_details.get("aircraft_precision", 0.0))
            selection_metric += 0.10 * float(calibrated_details.get("aircraft_recall", 0.0))
            selection_metric += 0.05 * float(calibrated_details.get("non_aircraft_recall", 0.0))
            selection_metric -= 0.60 * float(threshold_info.get("raw_false_aircraft_rate", 0.0))
            selection_metric -= 0.40 * float(threshold_info.get("max_source_false_aircraft_rate", 0.0))
            if not bool(threshold_info.get("feasible", False)):
                selection_metric -= 0.25
            if float(calibrated_details.get("aircraft_precision", 0.0)) < float(config.open_set_min_aircraft_precision_for_best):
                selection_metric -= 0.75
            if float(calibrated_details.get("aircraft_recall", 0.0)) < float(config.open_set_min_aircraft_recall_for_best):
                selection_metric -= 0.50
            if float(threshold_info.get("raw_false_aircraft_rate", 0.0)) > float(config.open_set_max_raw_false_aircraft_rate_for_best):
                selection_metric -= 0.50
            val_metrics["calibrated_selection_metric"] = selection_metric
        elif config.expert_name == "expert4_ground" and val_metrics.get("groundness_probability") is not None:
            threshold_info = _calibrate_groundness_threshold(
                val_metrics["groundness_probability"],
                val_metrics["groundness_labels"],
                config.groundness_threshold_min,
                config.groundness_threshold_max,
                config.groundness_threshold_steps,
            )
            val_metrics = {
                **val_metrics,
                **threshold_info,
                "calibrated_groundness_threshold": threshold_info["threshold"],
            }
            selection_metric = float(val_metrics["macro_f1"]) + 0.25 * float(threshold_info["f1"]) + 0.10 * float(threshold_info["specificity"])
            if float(threshold_info["f1"]) < 0.40:
                selection_metric -= 0.25
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if selection_metric > best_val:
            best_val = selection_metric
            if threshold_info is not None:
                if config.expert_name == "expert2_air":
                    best_threshold = float(threshold_info["threshold"])
                    model.set_decision_threshold(best_threshold)
                    best_source_thresholds = {
                        str(key): float(value)
                        for key, value in dict(threshold_info.get("source_thresholds", {})).items()
                    }
                    if hasattr(model, "set_source_thresholds"):
                        model.set_source_thresholds(best_source_thresholds)
                elif config.expert_name == "expert4_ground":
                    best_groundness_threshold = float(threshold_info["threshold"])
                    model.set_groundness_threshold(best_groundness_threshold)
                best_selection = threshold_info
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "labels": labels,
                "source_feature_stats": source_feature_stats,
                "shared_representation_checkpoint": config.shared_representation_checkpoint,
                "train_representation_builder": bool(config.train_representation_builder),
                "best_selection_metric": best_val,
                "best_selection": best_selection,
                "decision_threshold": best_threshold,
                "source_thresholds": best_source_thresholds,
                "groundness_threshold": best_groundness_threshold,
            }
            if config.train_representation_builder:
                checkpoint["representation_builder_state_dict"] = representation_builder.state_dict()
            torch.save(checkpoint, best_path)

    state = torch.load(best_path, map_location=device, weights_only=True)
    if "representation_builder_state_dict" in state:
        representation_builder.load_state_dict(state["representation_builder_state_dict"])
    model.load_state_dict(state["model_state_dict"])
    best_threshold = float(state.get("decision_threshold", best_threshold))
    best_source_thresholds = {
        str(key): float(value)
        for key, value in dict(state.get("source_thresholds", {})).items()
    }
    best_groundness_threshold = float(state.get("groundness_threshold", best_groundness_threshold))
    if hasattr(model, "set_decision_threshold"):
        model.set_decision_threshold(best_threshold)
    if hasattr(model, "set_source_thresholds"):
        model.set_source_thresholds(best_source_thresholds)
    if hasattr(model, "set_groundness_threshold"):
        model.set_groundness_threshold(best_groundness_threshold)
    test_metrics = _run_epoch(
        representation_builder,
        model,
        loaders.test_loader,
        criterion,
        optimizer,
        device,
        False,
        loaders.global_to_local,
        labels,
        config.expert_name,
        source_feature_stats,
        config=config,
    )
    test_threshold_info: dict[str, object] | None = None
    if config.expert_name == "expert2_air" and config.open_set_aircraft and test_metrics.get("aircraft_similarity") is not None:
        calibrated_logits = _open_set_logits_from_similarity_by_source(
            test_metrics["aircraft_similarity"],
            test_metrics["source_names"],
            default_threshold=best_threshold,
            source_thresholds=best_source_thresholds,
            scale=config.open_set_logit_scale,
        )
        calibrated_metrics = compute_classification_metrics(calibrated_logits, test_metrics["labels"], num_classes=len(labels))
        calibrated_details = _binary_details(calibrated_logits, test_metrics["labels"], labels)
        calibrated_selection_metric = float(calibrated_metrics.macro_f1)
        calibrated_selection_metric += 0.20 * float(calibrated_details.get("aircraft_f1", 0.0))
        calibrated_selection_metric += 0.10 * float(calibrated_details.get("aircraft_precision", 0.0))
        calibrated_selection_metric += 0.10 * float(calibrated_details.get("aircraft_recall", 0.0))
        calibrated_selection_metric += 0.05 * float(calibrated_details.get("non_aircraft_recall", 0.0))
        test_metrics = {
            **test_metrics,
            "logits": calibrated_logits,
            "accuracy": calibrated_metrics.accuracy,
            "macro_precision": calibrated_metrics.macro_precision,
            "macro_recall": calibrated_metrics.macro_recall,
            "macro_f1": calibrated_metrics.macro_f1,
            **calibrated_details,
        }
        test_threshold_info = {
            "threshold": best_threshold,
            "source": "validation_source_calibrated" if best_source_thresholds else "validation_calibrated",
            "source_thresholds": best_source_thresholds,
            "feasible": bool(state.get("best_selection", {}).get("feasible", False)),
            "calibrated_selection_metric": calibrated_selection_metric,
            **calibrated_details,
        }
    elif config.expert_name == "expert4_ground" and test_metrics.get("groundness_probability") is not None:
        groundness_details = _groundness_details(
            test_metrics["groundness_probability"],
            test_metrics["groundness_labels"],
            best_groundness_threshold,
        )
        test_metrics = {
            **test_metrics,
            **groundness_details,
        }
        test_threshold_info = {
            "threshold": best_groundness_threshold,
            "source": "validation_calibrated_groundness",
            **groundness_details,
        }

    history_path = run_paths.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {
        "labels": labels,
        "train_ids": loaders.split.train_ids,
        "val_ids": loaders.split.val_ids,
        "test_ids": loaders.split.test_ids,
    }
    manifest_path = run_paths.output_dir / "training_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    confusion = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    preds = test_metrics["logits"].argmax(dim=-1)
    for true_idx, pred_idx in zip(test_metrics["labels"].tolist(), preds.tolist()):
        confusion[int(true_idx), int(pred_idx)] += 1

    evaluation_report = {
        "summary": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "route_accuracy": 1.0,
            "aircraft_recall": test_metrics.get("aircraft_recall"),
            "non_aircraft_recall": test_metrics.get("non_aircraft_recall"),
            "balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "decision_threshold": best_threshold if config.expert_name == "expert2_air" else None,
            "source_thresholds": best_source_thresholds if config.expert_name == "expert2_air" else {},
            "groundness_threshold": best_groundness_threshold if config.expert_name == "expert4_ground" else None,
            "groundness_precision": test_metrics.get("groundness_precision"),
            "groundness_recall": test_metrics.get("groundness_recall"),
            "groundness_specificity": test_metrics.get("groundness_specificity"),
            "groundness_f1": test_metrics.get("groundness_f1"),
        },
        "label_names": labels,
        "classification_confusion_matrix": confusion.tolist(),
        "per_class": [],
        "route": {
            "label_names": [config.expert_name],
            "confusion_matrix": [[len(test_metrics["sample_ids"])]],
            "per_route": [{"route_name": config.expert_name, "support": len(test_metrics["sample_ids"]), "correct": len(test_metrics["sample_ids"]), "recall": 1.0}],
        },
        "by_source": {},
        "device": device_info,
        "expert_details": {
            "open_set_aircraft": bool(config.expert_name == "expert2_air" and config.open_set_aircraft),
            "decision_threshold": best_threshold if config.expert_name == "expert2_air" else None,
            "source_thresholds": best_source_thresholds if config.expert_name == "expert2_air" else {},
            "groundness_threshold": best_groundness_threshold if config.expert_name == "expert4_ground" else None,
            "threshold_info": test_threshold_info,
            "best_selection": state.get("best_selection", {}),
            "prediction_distribution": test_metrics.get("prediction_distribution", {}),
        },
        "predictions": [],
    }
    for idx, label in enumerate(labels):
        pred_pos = preds == idx
        true_pos = test_metrics["labels"] == idx
        tp = int((pred_pos & true_pos).sum().item())
        fp = int((pred_pos & ~true_pos).sum().item())
        fn = int((~pred_pos & true_pos).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        evaluation_report["per_class"].append({"label_name": label, "support": int(true_pos.sum().item()), "precision": precision, "recall": recall, "f1": f1})

    for source_name in sorted(set(test_metrics["source_names"])):
        indices = [i for i, s in enumerate(test_metrics["source_names"]) if s == source_name]
        if indices:
            acc = float((preds[indices] == test_metrics["labels"][indices]).float().mean().item())
            evaluation_report["by_source"][source_name] = {"count": len(indices), "accuracy": acc}

    for idx, sample_id in enumerate(test_metrics["sample_ids"]):
        evaluation_report["predictions"].append(
            {
                "sample_id": sample_id,
                "source_name": test_metrics["source_names"][idx],
                "label_name_true": test_metrics["label_names"][idx],
                "label_name_pred": labels[int(preds[idx].item())],
                "route_name_true": config.expert_name,
                "route_name_pred": config.expert_name,
                "is_correct": labels[int(preds[idx].item())] == test_metrics["label_names"][idx],
                "is_route_correct": True,
            }
        )

    evaluation_report_path = run_paths.output_dir / "evaluation_report.json"
    evaluation_report_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "best_checkpoint": str(best_path.as_posix()),
        "history_file": str(history_path.as_posix()),
        "manifest_file": str(manifest_path.as_posix()),
        "evaluation_report": str(evaluation_report_path.as_posix()),
        "test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "aircraft_precision": test_metrics.get("aircraft_precision"),
            "aircraft_recall": test_metrics.get("aircraft_recall"),
            "aircraft_f1": test_metrics.get("aircraft_f1"),
            "non_aircraft_precision": test_metrics.get("non_aircraft_precision"),
            "non_aircraft_recall": test_metrics.get("non_aircraft_recall"),
            "non_aircraft_f1": test_metrics.get("non_aircraft_f1"),
            "groundness_precision": test_metrics.get("groundness_precision"),
            "groundness_recall": test_metrics.get("groundness_recall"),
            "groundness_specificity": test_metrics.get("groundness_specificity"),
            "groundness_f1": test_metrics.get("groundness_f1"),
            "prediction_distribution": test_metrics.get("prediction_distribution", {}),
        },
        "run_timestamp": run_paths.timestamp_label,
        "experiment_name": run_paths.experiment_name,
        "device": device_info,
        "taxonomy": {label: idx for idx, label in enumerate(labels)},
        "trainable_parameters": _count_trainable_parameters(representation_builder) + _count_trainable_parameters(model),
        "pretrained_track_checkpoint": config.pretrained_track_checkpoint,
        "pretrained_track_info": pretrained_track_info,
        "shared_representation_checkpoint": config.shared_representation_checkpoint,
        "shared_representation_info": shared_representation_info,
        "train_representation_builder": bool(config.train_representation_builder),
        "source_normalization": bool(config.source_normalization),
        "source_feature_stats_saved": bool(source_feature_stats is not None),
        "source_feature_stats_from_checkpoint": bool(checkpoint_source_feature_stats),
        "open_set_aircraft": bool(config.expert_name == "expert2_air" and config.open_set_aircraft),
        "decision_threshold": best_threshold if config.expert_name == "expert2_air" else None,
        "source_thresholds": best_source_thresholds if config.expert_name == "expert2_air" else {},
        "include_ground_hard_negatives": bool(config.include_ground_hard_negatives and config.expert_name == "expert4_ground"),
        "groundness_threshold": best_groundness_threshold if config.expert_name == "expert4_ground" else None,
        "best_selection_metric": state.get("best_selection_metric"),
        "best_selection": state.get("best_selection", {}),
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
