"""r02 的 parent-level Bird→Drone 混淆感知损失。"""

from __future__ import annotations

import torch
from torch import nn


@torch.jit.script
def bird_drone_margin_loss(
    final_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    parent_weights: torch.Tensor,
) -> torch.Tensor:
    """约束真实 Bird 的 Bird logit 至少超过 Drone margin，反之亦然。

    仅接受每个完整 parent 的最终 causal state，因此不会把事件当成独立
    训练样本。非 Bird/Drone parent 的 margin 项为零。
    """
    if final_logits.ndim != 2 or final_logits.shape[1] < 2:
        raise ValueError("r02 margin logits must be [B,C] with C>=2")
    if labels.ndim != 1 or labels.shape[0] != final_logits.shape[0]:
        raise ValueError("r02 margin labels must be [B]")
    if parent_weights.shape != labels.shape:
        raise ValueError("r02 parent weights must be [B]")
    if not torch.isfinite(final_logits).all() or not torch.isfinite(parent_weights).all():
        raise ValueError("r02 margin inputs must be finite")
    if bool((parent_weights <= 0.0).any()):
        raise ValueError("r02 parent weights must be positive")
    drone = final_logits[:, 0]
    bird = final_logits[:, 1]
    bird_loss = torch.relu(float(margin) - (bird - drone))
    drone_loss = torch.relu(float(margin) - (drone - bird))
    bird_mask = labels == 1
    drone_mask = labels == 0
    selected = torch.where(bird_mask, bird_loss, torch.where(drone_mask, drone_loss, torch.zeros_like(bird_loss)))
    return (selected * parent_weights).sum() / parent_weights.sum()


def reliability_bce_loss(
    reliability_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    parent_weights: torch.Tensor,
) -> torch.Tensor:
    """Bird reliability 的完整 parent prefix 辅助损失。"""
    if reliability_logits.ndim != 2 or padding_mask.shape != reliability_logits.shape:
        raise ValueError("r02 reliability tensors must be [B,T]")
    if labels.ndim != 1 or labels.shape[0] != reliability_logits.shape[0]:
        raise ValueError("r02 reliability labels must be [B]")
    if parent_weights.shape != labels.shape:
        raise ValueError("r02 reliability parent weights must be [B]")
    target = (labels == 1).to(dtype=reliability_logits.dtype).unsqueeze(1).expand_as(reliability_logits)
    per_event = nn.functional.binary_cross_entropy_with_logits(reliability_logits, target, reduction="none")
    valid = (~padding_mask).to(dtype=per_event.dtype)
    per_parent = (per_event * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return (per_parent * parent_weights).sum() / parent_weights.sum()


def maturity_bce_loss(
    maturity_logits: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    parent_weights: torch.Tensor,
) -> torch.Tensor:
    """观测成熟度目标为 causal prefix / parent 完整长度。"""
    if maturity_logits.ndim != 2 or padding_mask.shape != maturity_logits.shape:
        raise ValueError("r02 maturity tensors must be [B,T]")
    if parent_weights.shape != (maturity_logits.shape[0],):
        raise ValueError("r02 maturity parent weights must be [B]")
    valid = ~padding_mask
    lengths = valid.long().sum(dim=1)
    steps = torch.arange(maturity_logits.shape[1], device=maturity_logits.device).unsqueeze(0) + 1
    target = (steps.to(dtype=maturity_logits.dtype) / lengths.unsqueeze(1).to(dtype=maturity_logits.dtype)).clamp(0.0, 1.0)
    per_event = nn.functional.binary_cross_entropy_with_logits(maturity_logits, target, reduction="none")
    valid_float = valid.to(dtype=per_event.dtype)
    per_parent = (per_event * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
    return (per_parent * parent_weights).sum() / parent_weights.sum()


def combined_loss(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    bird_reliability_logits: torch.Tensor,
    maturity_logits: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    parent_weights: torch.Tensor,
    margin: float,
    margin_weight: float,
    reliability_weight: float,
    maturity_weight: float,
) -> dict[str, torch.Tensor]:
    """r02 总损失：parent CE + Bird/Drone margin + 两个辅助头。"""
    lengths = (~padding_mask).long().sum(dim=1)
    indexes = torch.arange(sequence_logits.shape[0], device=sequence_logits.device)
    final_logits = sequence_logits[indexes, lengths - 1]
    ce = nn.functional.cross_entropy(final_logits, labels.long(), weight=class_weights, reduction="none")
    ce = (ce * parent_weights).sum() / parent_weights.sum()
    margin_value = bird_drone_margin_loss(final_logits, labels, margin=margin, parent_weights=parent_weights)
    reliability = reliability_bce_loss(bird_reliability_logits, labels, padding_mask, parent_weights=parent_weights)
    maturity = maturity_bce_loss(maturity_logits, padding_mask, parent_weights=parent_weights)
    total = ce + margin_weight * margin_value + reliability_weight * reliability + maturity_weight * maturity
    return {
        "loss": total,
        "parent_ce": ce,
        "bird_drone_margin": margin_value,
        "bird_reliability": reliability,
        "maturity": maturity,
        "final_logits": final_logits,
    }


__all__ = ["bird_drone_margin_loss", "combined_loss", "maturity_bce_loss", "reliability_bce_loss"]
