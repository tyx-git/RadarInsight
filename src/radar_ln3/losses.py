"""L1/L2 分类训练使用的通用损失。"""

from __future__ import annotations

import math

import torch
from torch import nn


def parent_final_fixed_parent_count_ce(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    prefix_weight: float = 0.0,
    prefix_events: int = 0,
) -> dict[str, torch.Tensor | int]:
    """按 parent 计算最终事件 CE，可加末尾因果前缀辅助项。

    最终事件项固定以 parent 数量为分母；前缀项先在每个 parent 内对
    最后若干个有效非终点事件取均值，再按 parent 平均。
    """

    if sequence_logits.ndim != 3:
        raise ValueError("sequence_logits 必须是 [B,T,C]")
    batch, length, class_count = sequence_logits.shape
    if batch <= 0 or length <= 0 or class_count < 2:
        raise ValueError("sequence_logits 的 batch、时间和类别维度无效")
    if labels.shape != (batch,):
        raise ValueError("labels 必须是 [B]")
    if padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask 必须是 [B,T] bool")
    if class_weights.shape != (class_count,):
        raise ValueError("class_weights 与类别数量不一致")
    prefix_weight = float(prefix_weight)
    if not math.isfinite(prefix_weight) or prefix_weight < 0.0:
        raise ValueError("prefix_weight 必须是有限非负值")
    if isinstance(prefix_events, bool) or not isinstance(prefix_events, int) or prefix_events < 0:
        raise ValueError("prefix_events 必须是非负整数")
    if prefix_weight > 0.0 and prefix_events <= 0:
        raise ValueError("启用前缀损失时 prefix_events 必须为正整数")
    if not torch.isfinite(sequence_logits).all():
        raise ValueError("sequence_logits 不能包含非有限值")
    if not torch.isfinite(class_weights).all() or bool((class_weights <= 0).any()):
        raise ValueError("class_weights 必须是正有限值")
    targets = labels.to(device=sequence_logits.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= class_count)).any()):
        raise ValueError("labels 超出类别范围")
    lengths = (~padding_mask).long().sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("损失不能接收空 parent")
    rows = torch.arange(batch, device=sequence_logits.device)
    final_logits = sequence_logits[rows, lengths - 1]
    per_parent = nn.functional.cross_entropy(final_logits, targets, reduction="none")
    weights = class_weights.to(
        device=sequence_logits.device,
        dtype=sequence_logits.dtype,
    )
    denominator = final_logits.new_tensor(float(batch))
    final_state_loss = (per_parent * weights[targets]).sum() / denominator
    prefix_auxiliary_loss = final_state_loss.new_zeros(())
    prefix_event_count = 0
    if prefix_weight > 0.0:
        prefix_event_count = min(prefix_events, max(0, int(lengths.max().item()) - 1))
        if prefix_event_count > 0:
            offsets = torch.arange(
                1, prefix_event_count + 1, device=sequence_logits.device
            )
            positions = lengths.unsqueeze(1) - 1 - offsets.unsqueeze(0)
            valid = positions >= 0
            selected_logits = sequence_logits[
                rows.unsqueeze(1), positions.clamp_min(0)
            ]
            selected_targets = targets.unsqueeze(1).expand(-1, prefix_event_count)
            per_prefix_event = nn.functional.cross_entropy(
                selected_logits.reshape(-1, class_count),
                selected_targets.reshape(-1),
                reduction="none",
            ).reshape(batch, prefix_event_count)
            per_prefix_event = per_prefix_event * weights[selected_targets]
            per_prefix_event = per_prefix_event.masked_fill(~valid, 0.0)
            per_parent_prefix = per_prefix_event.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            prefix_auxiliary_loss = per_parent_prefix.mean()
    loss = final_state_loss + prefix_weight * prefix_auxiliary_loss
    effective_mass = final_logits.new_zeros((class_count,))
    effective_mass.scatter_add_(0, targets, weights[targets])
    if not torch.isfinite(loss).item():
        raise ValueError("分类损失产生非有限值")
    return {
        "loss": loss,
        "final_state_loss": final_state_loss,
        "prefix_auxiliary_loss": prefix_auxiliary_loss,
        "prefix_event_count": prefix_event_count,
        "fixed_parent_count": denominator,
        "effective_class_mass": effective_mass,
        "final_logits": final_logits,
    }


__all__ = ["parent_final_fixed_parent_count_ce"]
