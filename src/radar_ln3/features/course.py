"""因果航向圆形离散度。

该特征来自历史 V21e 单因素实验。它不是替换 TR 的逐点角度编码，而是对
非首事件的 course 建立一个只使用当前及历史观测的 circular prefix 统计。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import FeatureInputError, FeatureNumericalError, FeatureSchema


COURSE_SCHEMA = "causal-prefix-course-circular-incoherence-v1"
COURSE_FEATURE_NAME = "causal_prefix_course_circular_incoherence"
COURSE_MINIMUM_VALID_NONFIRST_POINTS = 3


def course_schema() -> FeatureSchema:
    return FeatureSchema(
        name=COURSE_FEATURE_NAME,
        schema_version=COURSE_SCHEMA,
        value_names=(COURSE_FEATURE_NAME,),
        valid_names=(f"{COURSE_FEATURE_NAME}_valid",),
        # L1 将 value*valid 和 valid 一并送入 Fusion；valid 是显式
        # 的质量事实，而不是隐藏的默认填充值。
        mask_mode="explicit_input",
        value_bounds=(0.0, 1.0),
    )


def course_contract() -> dict[str, Any]:
    """返回该候选特征的计算和历史证据边界。"""

    return {
        "schema": COURSE_SCHEMA,
        "feature_name": COURSE_FEATURE_NAME,
        "value_dim": 1,
        "value_bounds": [0.0, 1.0],
        "valid_mask_mode": "explicit_input",
        "minimum_valid_nonfirst_points": COURSE_MINIMUM_VALID_NONFIRST_POINTS,
        "first_event_policy": "always excluded from prefix statistic",
        "causal_scope": "current and prior non-first course observations",
        "history": {
            "experiment": "V21e",
            "role": "KnownGate single-factor candidate",
            "development_result": "parent-final local validation positive; direct feature-zeroing affected Clutter recall",
            "not_proven": [
                "cross-source generalization",
                "formal open-set rejection",
                "final L1/L2/L3 deployment benefit",
            ],
        },
    }


def _course_value(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("course_deg")
    tr_values = getattr(item, "tr_values", None)
    if isinstance(tr_values, dict):
        return tr_values.get("course_deg")
    if item is None:
        return None
    return item


def _finite_or_missing(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureInputError("course_deg 无法解析") from exc
    if not math.isfinite(parsed):
        raise FeatureNumericalError("course_deg 包含非有限值")
    return parsed


def extract_causal_course_circular_incoherence(
    course_degrees: Sequence[float | None] | Sequence[Any],
    *,
    minimum_valid_nonfirst_points: int = COURSE_MINIMUM_VALID_NONFIRST_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``[event,1]`` 的因果 circular incoherence 和有效性。"""

    minimum = int(minimum_valid_nonfirst_points)
    if minimum < 1:
        raise FeatureInputError("course circular incoherence 最小 support 必须为正")
    courses = tuple(course_degrees)
    if not courses:
        raise FeatureInputError("course 序列不能为空")
    values = np.zeros((len(courses), 1), dtype=np.float32)
    valid = np.zeros((len(courses), 1), dtype=np.bool_)
    sine_sum = 0.0
    cosine_sum = 0.0
    count = 0
    for event_index, item in enumerate(courses):
        current = _finite_or_missing(_course_value(item))
        # 首事件只保留为逐点观测，不参与长期方向统计。
        if event_index > 0 and current is not None:
            radians = math.radians(current)
            sine_sum += math.sin(radians)
            cosine_sum += math.cos(radians)
            count += 1
        if count >= minimum:
            resultant_length = math.hypot(sine_sum, cosine_sum) / float(count)
            values[event_index, 0] = np.float32(
                np.clip(1.0 - resultant_length, 0.0, 1.0)
            )
            valid[event_index, 0] = True
    if not np.isfinite(values).all():
        raise FeatureNumericalError("course circular incoherence 产生非有限值")
    return values, valid


def encode_causal_course_sequence(
    events: Sequence[Any],
    *,
    minimum_valid_nonfirst_points: int = COURSE_MINIMUM_VALID_NONFIRST_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """编码 course，并额外返回每个 prefix 的非首有效 support 数。"""

    values, valid = extract_causal_course_circular_incoherence(
        events,
        minimum_valid_nonfirst_points=minimum_valid_nonfirst_points,
    )
    support = np.zeros((len(events),), dtype=np.int64)
    count = 0
    for index, item in enumerate(events):
        current = _finite_or_missing(_course_value(item))
        if index > 0 and current is not None:
            count += 1
        support[index] = count
    return values, valid, support


__all__ = [
    "COURSE_FEATURE_NAME",
    "COURSE_MINIMUM_VALID_NONFIRST_POINTS",
    "COURSE_SCHEMA",
    "course_contract",
    "course_schema",
    "encode_causal_course_sequence",
    "extract_causal_course_circular_incoherence",
]
