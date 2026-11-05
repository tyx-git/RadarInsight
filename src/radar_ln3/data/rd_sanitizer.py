"""通用 RD 幅值输入清洗与审计。

该模块只处理 RD 输入的数值完整性，不解释任何数据集的字段或文件名。
对于 dB 幅值，复数 I/Q 为零时得到 ``-Inf`` 是零功率的可解释表示；只有
这一种情况允许在数据规范化边界转换为有限的零值。其他非有限值必须失败，
避免把损坏或语义不明的数据静默带入后续特征链路。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import DataFormatError, DataIntegrityError


RD_AMPLITUDE_REPAIR_SCHEMA = "RADAR-LN3-RD-AMPLITUDE-REPAIR-V1"
RD_AMPLITUDE_REPAIR_POLICY = "zero_complex_power_neg_inf_to_zero"


@dataclass(frozen=True)
class RDAmplitudeSanitization:
    """一次 RD dB 幅值清洗的有限结果和审计信息。"""

    values: np.ndarray
    audit: dict[str, Any]


def _require_2d_array(value: Any, *, name: str, dtype: np.dtype | type) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise DataFormatError(f"{name} 无法转换为数值数组") from exc
    if array.ndim != 2 or any(int(size) <= 0 for size in array.shape):
        raise DataFormatError(f"{name} 必须是非空二维数组：shape={array.shape}")
    return array


def sanitize_rd_amplitude_db(
    amplitude_db: Any,
    complex_values: Any | None,
    *,
    name: str,
) -> RDAmplitudeSanitization:
    """校验并有限化一张 dB RD 幅值平面。

    ``-Inf`` 只有在对应复数 I/Q 幅值严格为零时才会被转换为 ``0``，与既有
    canonical RD 的零功率表示保持一致。
    ``complex_values`` 缺失时不能证明非有限幅值是零功率，因此该情况失败。
    """

    amplitude = _require_2d_array(
        amplitude_db,
        name=f"{name}:amplitude",
        dtype=np.float32,
    )
    complex_array: np.ndarray | None = None
    if complex_values is not None:
        try:
            complex_array = np.asarray(complex_values)
        except (TypeError, ValueError) as exc:
            raise DataFormatError(f"{name}:complex 无法转换为数组") from exc
        if complex_array.shape != amplitude.shape:
            raise DataFormatError(
                f"{name}:complex 与 amplitude shape 不一致："
                f"{complex_array.shape} != {amplitude.shape}"
            )
        if not np.issubdtype(complex_array.dtype, np.number):
            raise DataFormatError(f"{name}:complex 必须是数值数组")
        if not np.isfinite(complex_array).all():
            raise DataIntegrityError(f"{name}:complex 包含非有限值")

    finite_mask = np.isfinite(amplitude)
    negative_inf_mask = np.isneginf(amplitude)
    positive_inf_mask = np.isposinf(amplitude)
    nan_mask = np.isnan(amplitude)
    input_nonfinite_count = int((~finite_mask).sum())
    negative_inf_count = int(negative_inf_mask.sum())
    positive_inf_count = int(positive_inf_mask.sum())
    nan_count = int(nan_mask.sum())

    replace_mask = np.zeros(amplitude.shape, dtype=bool)
    if negative_inf_count:
        if complex_array is None:
            raise DataIntegrityError(
                f"{name}:amplitude 包含 -Inf，但缺少复数 I/Q，无法确认零功率"
            )
        replace_mask = negative_inf_mask & (np.abs(complex_array) == 0)
        if np.any(negative_inf_mask & ~replace_mask):
            raise DataIntegrityError(
                f"{name}:amplitude 包含与非零复数功率对应的 -Inf"
            )
    if nan_count or positive_inf_count:
        raise DataIntegrityError(
            f"{name}:amplitude 包含不允许的非有限值："
            f"nan={nan_count}, positive_inf={positive_inf_count}"
        )

    result = amplitude.copy()
    if np.any(replace_mask):
        result[replace_mask] = 0.0
    if not np.isfinite(result).all():
        raise DataIntegrityError(f"{name}:amplitude 清洗后仍包含非有限值")

    repaired_rows = sorted(
        int(index) for index in np.flatnonzero(replace_mask.any(axis=1))
    )
    audit = {
        "schema_version": RD_AMPLITUDE_REPAIR_SCHEMA,
        "repair_policy": RD_AMPLITUDE_REPAIR_POLICY,
        "input_nonfinite_count": input_nonfinite_count,
        "negative_inf_count": negative_inf_count,
        "positive_inf_count": positive_inf_count,
        "nan_count": nan_count,
        "zero_power_replaced_count": int(replace_mask.sum()),
        "repaired_rows": repaired_rows,
        "output_finite": True,
    }
    return RDAmplitudeSanitization(values=result, audit=audit)


__all__ = [
    "RD_AMPLITUDE_REPAIR_POLICY",
    "RD_AMPLITUDE_REPAIR_SCHEMA",
    "RDAmplitudeSanitization",
    "sanitize_rd_amplitude_db",
]
