from __future__ import annotations

from dataclasses import dataclass

import torch

from .metrics import ClassificationMetrics, compute_classification_metrics


@dataclass(frozen=True)
class SystemMetrics:
    classification: ClassificationMetrics
    route_accuracy: float
    route_macro_recall: float


@dataclass(frozen=True)
class PerClassMetrics:
    label_name: str
    support: int
    precision: float
    recall: float
    f1: float


def compute_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
        matrix[int(true_idx), int(pred_idx)] += 1
    return matrix


def compute_per_class_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
) -> list[PerClassMetrics]:
    preds = logits.argmax(dim=-1)
    results: list[PerClassMetrics] = []
    for class_idx, label_name in enumerate(label_names):
        pred_pos = preds == class_idx
        true_pos = labels == class_idx
        tp = int((pred_pos & true_pos).sum().item())
        fp = int((pred_pos & ~true_pos).sum().item())
        fn = int((~pred_pos & true_pos).sum().item())
        support = int(true_pos.sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        results.append(
            PerClassMetrics(
                label_name=label_name,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return results


def compute_system_metrics(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    route_logits: torch.Tensor,
    route_labels: torch.Tensor,
    num_classes: int,
) -> SystemMetrics:
    classification = compute_classification_metrics(class_logits, class_labels, num_classes=num_classes)
    route_preds = route_logits.argmax(dim=-1)
    route_accuracy = float((route_preds == route_labels).float().mean().item())
    recalls: list[float] = []
    for route_idx in range(route_logits.shape[-1]):
        route_mask = route_labels == route_idx
        if not bool(route_mask.any()):
            continue
        recalls.append(float((route_preds[route_mask] == route_idx).float().mean().item()))
    route_macro_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    return SystemMetrics(
        classification=classification,
        route_accuracy=route_accuracy,
        route_macro_recall=route_macro_recall,
    )
