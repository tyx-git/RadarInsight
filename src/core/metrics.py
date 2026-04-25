from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float().mean().item()

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for class_idx in range(num_classes):
        pred_pos = preds == class_idx
        true_pos = labels == class_idx
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return ClassificationMetrics(
        accuracy=correct,
        macro_precision=sum(precisions) / num_classes,
        macro_recall=sum(recalls) / num_classes,
        macro_f1=sum(f1s) / num_classes,
    )
