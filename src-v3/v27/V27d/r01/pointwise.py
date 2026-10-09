"""r01 逐点输入与逐点 soft-gate 适配器。

r20 的评估对象仍是 parent-final 分类；本模块只把每个 event row 当作一个
独立点输入，并把同一长度的 keep_weight 对齐到 batch 的时间轴，不改变 padding、
dt 或 parent 顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .denoising import DenoisingCalibration, SequenceDenoisingResult, score_sequence


@dataclass(frozen=True)
class PointwiseSequenceInput:
    """一条序列的逐点输入视图。"""

    parent_sample_id: str
    values: np.ndarray
    keep_weight: np.ndarray
    valid_mask: np.ndarray
    result: SequenceDenoisingResult

    @property
    def point_count(self) -> int:
        return int(self.values.shape[0])


def build_pointwise_input(
    calibration: DenoisingCalibration,
    sequence: Any,
) -> PointwiseSequenceInput:
    """按 event row 顺序生成逐点 gate，不改变既有 r20 分数定义。"""

    rows = list(getattr(sequence, "event_rows", ()) or ())
    result = score_sequence(calibration, sequence)
    if len(rows) != len(result.keep_weight):
        raise ValueError("逐点输入与降噪结果长度不一致")
    # 逐点视图只保留稳定索引；物理特征仍由 r20 的原始输入张量承载。
    values = np.arange(len(rows), dtype=np.float32)[:, None]
    valid_mask = np.ones(len(rows), dtype=bool)
    return PointwiseSequenceInput(
        parent_sample_id=str(getattr(sequence, "parent_sample_id", "")),
        values=values,
        keep_weight=result.keep_weight.astype(np.float32, copy=True),
        valid_mask=valid_mask,
        result=result,
    )


def _gate_tensor(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    gate = weights
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    return value * gate


def apply_pointwise_event_gate(
    inputs: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """把逐点权重对齐到 r20 batch；padding 和时间语义保持不变。"""

    if weights.ndim != 2 or weights.dtype not in {torch.float16, torch.float32, torch.float64}:
        raise ValueError("逐点权重必须是二维浮点张量")
    if "tr_features" not in inputs or inputs["tr_features"].shape[:2] != weights.shape:
        raise ValueError("逐点权重与 tr_features 时间轴不一致")
    result = dict(inputs)
    gate = weights.to(device=inputs["tr_features"].device, dtype=inputs["tr_features"].dtype)
    for key in (
        "tr_features",
        "rd_frames",
        "rd_quality",
        "rd_physical_features",
        "micro_doppler_features",
        "radial_sidecar_values",
    ):
        value = result.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[:2] == gate.shape:
            result[key] = _gate_tensor(value, gate)
    return result


def build_weight_tensor(
    pointwise_inputs: list[PointwiseSequenceInput],
    width: int,
) -> torch.Tensor:
    """把变长逐点权重安全地 padding 到 batch 时间宽度。"""

    if width <= 0:
        raise ValueError("逐点 batch width 必须为正")
    result = torch.zeros((len(pointwise_inputs), width), dtype=torch.float32)
    for row, item in enumerate(pointwise_inputs):
        length = min(width, item.point_count)
        result[row, :length] = torch.from_numpy(item.keep_weight[:length])
    return result


__all__ = [
    "PointwiseSequenceInput",
    "apply_pointwise_event_gate",
    "build_pointwise_input",
    "build_weight_tensor",
]
