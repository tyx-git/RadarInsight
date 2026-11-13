"""模型输入 / ONNX 导出契约的唯一真源。

输入分四组：

- ``BASE_SEQUENCE_INPUTS``：TR/RD 序列的公共输入（L1、L2 都消费）；
- ``L1_TASK_INPUTS`` / ``L2_TASK_INPUTS``：任务专属分支输入；
- ``RESERVED_EXPORT_INPUTS``：全链导出的**保留位**，不属于任一模型的
  ``base_inputs``，由级联子分支消费（观测增量、legacy reference TR、
  stable8 侧车与 motion）。它们按固定顺序挂在导出签名末尾，使
  ``residual / direct / direct_grouped`` 与 L1/L2 的不同导出共享同一 ABI。

``FULL_CASCADE_INPUTS`` 的顺序是对外 ABI：已导出的 ONNX artifact 按位置绑定
输入，因此只允许追加，不允许重排。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BASE_SEQUENCE_INPUTS: tuple[str, ...] = (
    "tr_features",
    "tr_missing_mask",
    "tr_dt_seconds",
    "tr_dt_valid",
    "tr_dt_reason",
    "tr_padding_mask",
    "rd_frames",
    "rd_frame_mask",
    "rd_quality",
)
L2_TASK_INPUTS: tuple[str, ...] = (
    "rd_physical_features",
    "rd_physical_valid",
    "micro_doppler_features",
    "micro_doppler_valid",
)
L1_TASK_INPUTS: tuple[str, ...] = (
    "course_features",
    "course_valid",
    "course_support",
)
RESERVED_EXPORT_INPUTS: tuple[str, ...] = (
    "observation_increment_values",
    "reference_tr_features",
    "reference_tr_missing_mask",
    "stable_summary_tokens",
    "stable_summary_valid",
    "stable_frame_values",
    "stable_frame_mask",
    "motion_values",
)

L1_MODEL_INPUTS: tuple[str, ...] = BASE_SEQUENCE_INPUTS + L1_TASK_INPUTS
L2_MODEL_INPUTS: tuple[str, ...] = BASE_SEQUENCE_INPUTS + L2_TASK_INPUTS
#: Evaluate 合并批时，从 L1 batch 取回的输入（其余取 L2 batch）。
L1_SOURCE_ONLY_INPUTS: tuple[str, ...] = L1_TASK_INPUTS + (
    "observation_increment_values",
)
FULL_CASCADE_INPUTS: tuple[str, ...] = (
    BASE_SEQUENCE_INPUTS
    + L2_TASK_INPUTS
    + L1_TASK_INPUTS
    + RESERVED_EXPORT_INPUTS
)

LV28_INPUT_NAMES: tuple[str, ...] = (
    "l1_tr_features",
    "l2_tr_features",
    "tr_missing_mask",
    "tr_dt_seconds",
    "tr_dt_valid",
    "tr_dt_reason",
    "tr_padding_mask",
    "l1_rd_frames",
    "l2_rd_frames",
    "rd_frame_mask",
    "rd_quality",
    "course_features",
    "course_valid",
    "course_support",
    "rd_physical_features",
    "rd_physical_valid",
    "micro_doppler_features",
    "micro_doppler_valid",
    "observation_increment",
)
#: LV28 是早期实验图：L1/L2 各自持有 TR/RD 张量键，只有观测增量键名不同。
#: 键 = ONNX 输入名，值 = 产品 batch 中对应的 canonical 键。
LEGACY_LV28_BATCH_ALIASES: dict[str, str] = {
    "l1_tr_features": "l1_tr_features",
    "l2_tr_features": "l2_tr_features",
    "tr_missing_mask": "tr_missing_mask",
    "tr_dt_seconds": "tr_dt_seconds",
    "tr_dt_valid": "tr_dt_valid",
    "tr_dt_reason": "tr_dt_reason",
    "tr_padding_mask": "tr_padding_mask",
    "l1_rd_frames": "l1_rd_frames",
    "l2_rd_frames": "l2_rd_frames",
    "rd_frame_mask": "rd_frame_mask",
    "rd_quality": "rd_quality",
    "course_features": "course_features",
    "course_valid": "course_valid",
    "course_support": "course_support",
    "rd_physical_features": "rd_physical_features",
    "rd_physical_valid": "rd_physical_valid",
    "micro_doppler_features": "micro_doppler_features",
    "micro_doppler_valid": "micro_doppler_valid",
    "observation_increment": "observation_increment_values",
}


def select_inputs(
    values: Mapping[str, Any],
    names: tuple[str, ...],
) -> dict[str, Any]:
    """按契约顺序挑选输入；缺键抛 ``ValueError``。"""

    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"模型 batch 缺少输入：{missing}")
    return {name: values[name] for name in names}


__all__ = [
    "BASE_SEQUENCE_INPUTS",
    "FULL_CASCADE_INPUTS",
    "L1_MODEL_INPUTS",
    "L1_SOURCE_ONLY_INPUTS",
    "L1_TASK_INPUTS",
    "L2_MODEL_INPUTS",
    "L2_TASK_INPUTS",
    "LEGACY_LV28_BATCH_ALIASES",
    "LV28_INPUT_NAMES",
    "RESERVED_EXPORT_INPUTS",
    "select_inputs",
]
