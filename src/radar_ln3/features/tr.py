"""通用 TR 观测编码。

TR 编码只处理当前观测本身。角度使用 sin/cos 表示，缺失或非有限值在
值向量中编码为零，同时由独立 missing mask 保留缺失事实。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import (
    FeatureInputError,
    FeatureNumericalError,
    FeatureSchema,
    event_values,
)


TR_SCHEMA = "causal-tr-observation-v1"
TR_VALUE_NAMES = (
    "range_m_scaled",
    "radial_speed_mps_scaled",
    "elevation_sin",
    "elevation_cos",
    "azimuth_sin",
    "azimuth_cos",
    "course_sin",
    "course_cos",
    "speed_mps_scaled",
    "amplitude_signed_log_scaled",
    "snr_db_scaled",
    "height_m_scaled",
    "latitude_scaled",
    "longitude_scaled",
)
TR_MISSING_NAMES = (
    "range_m_missing",
    "radial_speed_mps_missing",
    "elevation_deg_missing",
    "azimuth_deg_missing",
    "course_deg_missing",
    "speed_mps_missing",
    "amplitude_db_missing",
    "snr_db_missing",
    "height_m_missing",
    "latitude_missing",
    "longitude_missing",
)
RAW_TO_MISSING_INDEX = {
    "range_m": 0,
    "radial_speed_mps": 1,
    "elevation_deg": 2,
    "azimuth_deg": 3,
    "course_deg": 4,
    "speed_mps": 5,
    "amplitude_db": 6,
    "snr_db": 7,
    "height_m": 8,
    "latitude": 9,
    "longitude": 10,
}

TR_VALUE_DIM = len(TR_VALUE_NAMES)
TR_MISSING_DIM = len(TR_MISSING_NAMES)


def tr_schema() -> FeatureSchema:
    """返回 TR 值向量和缺失事实的稳定描述。"""

    return FeatureSchema(
        name="tr_observation",
        schema_version=TR_SCHEMA,
        value_names=TR_VALUE_NAMES,
        valid_names=TR_MISSING_NAMES,
        mask_mode="explicit_input",
        value_bounds=None,
    )


def _finite_number(values: Mapping[str, Any], name: str) -> tuple[float, bool]:
    raw = values.get(name)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, True
    if not math.isfinite(value):
        return 0.0, True
    return value, False


def _angle_pair(values: Mapping[str, Any], name: str) -> tuple[float, float, bool]:
    value, missing = _finite_number(values, name)
    if missing:
        return 0.0, 0.0, True
    radians = math.radians(value)
    return math.sin(radians), math.cos(radians), False


def encode_tr_values(values: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """编码一个 TR 观测，返回 ``(values, missing_mask)``。"""

    if not isinstance(values, Mapping):
        raise FeatureInputError("TR values 必须是 mapping")
    range_m, range_missing = _finite_number(values, "range_m")
    radial, radial_missing = _finite_number(values, "radial_speed_mps")
    elevation_sin, elevation_cos, elevation_missing = _angle_pair(
        values, "elevation_deg"
    )
    azimuth_sin, azimuth_cos, azimuth_missing = _angle_pair(values, "azimuth_deg")
    course_sin, course_cos, course_missing = _angle_pair(values, "course_deg")
    speed, speed_missing = _finite_number(values, "speed_mps")
    amplitude, amplitude_missing = _finite_number(values, "amplitude_db")
    snr, snr_missing = _finite_number(values, "snr_db")
    height, height_missing = _finite_number(values, "height_m")
    latitude, latitude_missing = _finite_number(values, "latitude")
    longitude, longitude_missing = _finite_number(values, "longitude")
    amplitude_log = math.copysign(math.log1p(abs(amplitude)), amplitude) / 20.0
    encoded = np.asarray(
        [
            range_m / 5000.0,
            radial / 100.0,
            elevation_sin,
            elevation_cos,
            azimuth_sin,
            azimuth_cos,
            course_sin,
            course_cos,
            speed / 100.0,
            amplitude_log,
            snr / 80.0,
            height / 1000.0,
            latitude / 90.0,
            longitude / 180.0,
        ],
        dtype=np.float32,
    )
    # 模型把 missing mask 作为显式 float32 输入；1.0 表示原始字段缺失。
    missing = np.asarray(
        [
            range_missing,
            radial_missing,
            elevation_missing,
            azimuth_missing,
            course_missing,
            speed_missing,
            amplitude_missing,
            snr_missing,
            height_missing,
            latitude_missing,
            longitude_missing,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(encoded).all():
        raise FeatureNumericalError("TR 编码产生非有限值")
    return encoded, missing


def encode_tr_feature_values(values: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """兼容历史调用方的明确别名。"""

    return encode_tr_values(values)


def encode_tr_sequence(
    values: Sequence[Mapping[str, Any] | Any],
) -> tuple[np.ndarray, np.ndarray]:
    """按输入因果顺序编码 TR 序列，不按时间戳重新排序。"""

    if not values:
        raise FeatureInputError("TR 序列不能为空")
    encoded = [encode_tr_values(event_values(item)) for item in values]
    result = np.stack([item[0] for item in encoded], axis=0).astype(
        np.float32, copy=False
    )
    missing = np.stack([item[1] for item in encoded], axis=0).astype(
        np.float32, copy=False
    )
    return result, missing


def parse_feature_json(value: str) -> dict[str, Any]:
    """解析历史 JSON 字段，但不改变其字段语义。"""

    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise FeatureInputError("TR feature JSON 无法解析") from exc
    if not isinstance(payload, dict):
        raise FeatureInputError("TR feature JSON 必须解码为对象")
    return payload


__all__ = [
    "RAW_TO_MISSING_INDEX",
    "TR_MISSING_DIM",
    "TR_MISSING_NAMES",
    "TR_SCHEMA",
    "TR_VALUE_DIM",
    "TR_VALUE_NAMES",
    "encode_tr_feature_values",
    "encode_tr_sequence",
    "encode_tr_values",
    "parse_feature_json",
    "tr_schema",
]
