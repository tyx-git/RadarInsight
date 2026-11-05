"""TR 的因果运动与逐观测增量特征。

运动特征只在当前事件已经提供合法正 ``dt`` 时计算时间导数。首事件、同秒
事件和来源时间倒退事件都不会被伪造为正时间间隔；它们仍然更新可用的原始
历史值，便于后续真实正间隔继续工作。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import FeatureInputError, FeatureNumericalError, FeatureSchema


TR_MOTION_SCHEMA = "causal-tr-motion-v1"
TR_MOTION_MASK_MODE = "gating_only"
TR_MOTION_NAMES = (
    "reported_speed_acceleration_scaled",
    "radial_acceleration_scaled",
    "course_turn_rate_scaled",
    "geometric_speed_scaled",
    "reported_geometric_speed_abs_error_scaled",
    "range_rate_scaled",
    "radial_range_rate_abs_error_scaled",
    "radial_sign_flip",
    "causal_path_straightness",
)
TR_MOTION_VALID_NAMES = tuple(f"{name}_valid" for name in TR_MOTION_NAMES)
TR_MOTION_SCALES = {
    "reported_speed_acceleration_scaled": 10.0,
    "radial_acceleration_scaled": 10.0,
    "course_turn_rate_scaled": 45.0,
    "geometric_speed_scaled": 100.0,
    "reported_geometric_speed_abs_error_scaled": 100.0,
    "range_rate_scaled": 100.0,
    "radial_range_rate_abs_error_scaled": 100.0,
    "radial_sign_flip": 1.0,
    "causal_path_straightness": 1.0,
}
TR_MOTION_CLIP = 5.0

TR_OBSERVATION_INCREMENT_SCHEMA = "causal-tr-observation-increment-v1"
TR_OBSERVATION_INCREMENT_MASK_MODE = "gating_only"
TR_OBSERVATION_INCREMENT_NAMES = (
    "reported_speed_delta_scaled",
    "radial_speed_delta_scaled",
    "course_delta_scaled",
    "geometric_step_distance_scaled",
    "range_delta_scaled",
    "radial_range_direction_agreement",
    "radial_sign_flip",
    "consecutive_step_direction_cosine",
    "causal_path_straightness",
)
TR_OBSERVATION_INCREMENT_VALID_NAMES = tuple(
    f"{name}_valid" for name in TR_OBSERVATION_INCREMENT_NAMES
)
TR_OBSERVATION_INCREMENT_SCALES = {
    "reported_speed_delta_scaled": 10.0,
    "radial_speed_delta_scaled": 10.0,
    "course_delta_scaled": 45.0,
    "geometric_step_distance_scaled": 100.0,
    "range_delta_scaled": 100.0,
    "radial_range_direction_agreement": 1.0,
    "radial_sign_flip": 1.0,
    "consecutive_step_direction_cosine": 1.0,
    "causal_path_straightness": 1.0,
}
TR_OBSERVATION_INCREMENT_CLIP = 5.0


@dataclass(frozen=True)
class CausalTRFeatureContract:
    """一个因果 TR 动态变体的版本化描述。"""

    schema: str
    value_names: tuple[str, ...]
    valid_names: tuple[str, ...]
    mask_mode: str
    scales: dict[str, float]
    clip_absolute: float

    @property
    def value_dim(self) -> int:
        return len(self.value_names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "value_dim": self.value_dim,
            "value_names": list(self.value_names),
            "valid_mask_dim": len(self.valid_names),
            "valid_mask_names": list(self.valid_names),
            "valid_mask_mode": self.mask_mode,
            "scales": dict(self.scales),
            "clip_absolute": float(self.clip_absolute),
        }


_CONTRACTS = {
    TR_MOTION_SCHEMA: CausalTRFeatureContract(
        schema=TR_MOTION_SCHEMA,
        value_names=TR_MOTION_NAMES,
        valid_names=TR_MOTION_VALID_NAMES,
        mask_mode=TR_MOTION_MASK_MODE,
        scales=TR_MOTION_SCALES,
        clip_absolute=TR_MOTION_CLIP,
    ),
    TR_OBSERVATION_INCREMENT_SCHEMA: CausalTRFeatureContract(
        schema=TR_OBSERVATION_INCREMENT_SCHEMA,
        value_names=TR_OBSERVATION_INCREMENT_NAMES,
        valid_names=TR_OBSERVATION_INCREMENT_VALID_NAMES,
        mask_mode=TR_OBSERVATION_INCREMENT_MASK_MODE,
        scales=TR_OBSERVATION_INCREMENT_SCALES,
        clip_absolute=TR_OBSERVATION_INCREMENT_CLIP,
    ),
}


def resolve_causal_tr_feature_contract(
    schema: str | None,
) -> CausalTRFeatureContract:
    try:
        return _CONTRACTS[str(schema)]
    except KeyError as exc:
        raise FeatureInputError(f"不支持的因果 TR 特征 schema：{schema!r}") from exc


def require_causal_tr_motion_schema(schema: str | None) -> None:
    if schema != TR_MOTION_SCHEMA:
        raise FeatureInputError(
            "当前配置没有声明 causal TR motion schema："
            f"expected={TR_MOTION_SCHEMA!r}, actual={schema!r}"
        )


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _number(values: Mapping[str, Any], name: str) -> tuple[float, bool]:
    value = values.get(name)
    if not _finite(value):
        return 0.0, True
    return float(value), False


def _event_values(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item
    values = getattr(item, "tr_values", None)
    if not isinstance(values, Mapping):
        raise FeatureInputError("TR motion event 缺少 mapping 类型的 tr_values")
    return values


def _spherical_position_m(values: Mapping[str, Any]) -> np.ndarray | None:
    range_m, range_missing = _number(values, "range_m")
    azimuth_deg, azimuth_missing = _number(values, "azimuth_deg")
    elevation_deg, elevation_missing = _number(values, "elevation_deg")
    if range_missing or azimuth_missing or elevation_missing or range_m < 0.0:
        return None
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = range_m * math.cos(elevation)
    position = np.asarray(
        [
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            range_m * math.sin(elevation),
        ],
        dtype=np.float64,
    )
    return position if np.isfinite(position).all() else None


def _wrapped_delta_degrees(current: float, previous: float) -> float:
    return float((current - previous + 180.0) % 360.0 - 180.0)


def _scaled_motion(value: float, scale: float, *, nonnegative: bool = False) -> float:
    lower = 0.0 if nonnegative else -TR_MOTION_CLIP
    return float(np.clip(float(value) / float(scale), lower, TR_MOTION_CLIP))


@dataclass
class CausalTRMotionState:
    """每个 parent 独立的因果运动状态。"""

    previous_speed_mps: float | None = None
    previous_radial_speed_mps: float | None = None
    previous_course_deg: float | None = None
    previous_range_m: float | None = None
    previous_position_m: np.ndarray | None = None
    path_anchor_m: np.ndarray | None = None
    cumulative_path_m: float = 0.0
    event_count: int = 0
    ignore_first_speed_course: bool = True

    def reset(self) -> None:
        self.previous_speed_mps = None
        self.previous_radial_speed_mps = None
        self.previous_course_deg = None
        self.previous_range_m = None
        self.previous_position_m = None
        self.path_anchor_m = None
        self.cumulative_path_m = 0.0
        self.event_count = 0

    def update(
        self,
        values: Mapping[str, Any],
        *,
        dt_seconds: float,
        dt_valid: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((len(TR_MOTION_NAMES),), dtype=np.float32)
        valid = np.zeros((len(TR_MOTION_NAMES),), dtype=np.bool_)
        dt = float(dt_seconds) if _finite(dt_seconds) else 0.0
        interval_valid = bool(dt_valid) and dt > 0.0

        speed, speed_missing = _number(values, "speed_mps")
        radial, radial_missing = _number(values, "radial_speed_mps")
        course, course_missing = _number(values, "course_deg")
        range_m, range_missing = _number(values, "range_m")
        position = _spherical_position_m(values)

        if interval_valid and not speed_missing and self.previous_speed_mps is not None:
            features[0] = _scaled_motion(
                (speed - self.previous_speed_mps) / dt,
                TR_MOTION_SCALES[TR_MOTION_NAMES[0]],
            )
            valid[0] = True
        if (
            interval_valid
            and not radial_missing
            and self.previous_radial_speed_mps is not None
        ):
            features[1] = _scaled_motion(
                (radial - self.previous_radial_speed_mps) / dt,
                TR_MOTION_SCALES[TR_MOTION_NAMES[1]],
            )
            valid[1] = True
            features[7] = float(radial * self.previous_radial_speed_mps < 0.0)
            valid[7] = True
        if interval_valid and not course_missing and self.previous_course_deg is not None:
            features[2] = _scaled_motion(
                _wrapped_delta_degrees(course, self.previous_course_deg) / dt,
                TR_MOTION_SCALES[TR_MOTION_NAMES[2]],
            )
            valid[2] = True

        range_rate: float | None = None
        if interval_valid and not range_missing and self.previous_range_m is not None:
            range_rate = (range_m - self.previous_range_m) / dt
            features[5] = _scaled_motion(
                range_rate, TR_MOTION_SCALES[TR_MOTION_NAMES[5]]
            )
            valid[5] = True
            if not radial_missing and self.previous_radial_speed_mps is not None:
                interval_radial = 0.5 * (radial + self.previous_radial_speed_mps)
                features[6] = _scaled_motion(
                    abs(interval_radial - range_rate),
                    TR_MOTION_SCALES[TR_MOTION_NAMES[6]],
                    nonnegative=True,
                )
                valid[6] = True

        if interval_valid and position is not None and self.previous_position_m is not None:
            step_distance = float(np.linalg.norm(position - self.previous_position_m))
            geometric_speed = step_distance / dt
            features[3] = _scaled_motion(
                geometric_speed,
                TR_MOTION_SCALES[TR_MOTION_NAMES[3]],
                nonnegative=True,
            )
            valid[3] = True
            if not speed_missing:
                features[4] = _scaled_motion(
                    abs(speed - geometric_speed),
                    TR_MOTION_SCALES[TR_MOTION_NAMES[4]],
                    nonnegative=True,
                )
                valid[4] = True
            if self.path_anchor_m is None:
                self.path_anchor_m = self.previous_position_m.copy()
            self.cumulative_path_m += step_distance
            if self.cumulative_path_m > 1.0e-6:
                displacement = float(np.linalg.norm(position - self.path_anchor_m))
                features[8] = float(
                    np.clip(displacement / self.cumulative_path_m, 0.0, 1.0)
                )
                valid[8] = True
        elif position is not None:
            self.path_anchor_m = position.copy()
            self.cumulative_path_m = 0.0
        else:
            self.path_anchor_m = None
            self.cumulative_path_m = 0.0

        first_event = self.event_count == 0
        self.previous_speed_mps = (
            None
            if speed_missing or (first_event and self.ignore_first_speed_course)
            else speed
        )
        self.previous_radial_speed_mps = None if radial_missing else radial
        self.previous_course_deg = (
            None
            if course_missing or (first_event and self.ignore_first_speed_course)
            else course
        )
        self.previous_range_m = None if range_missing else range_m
        self.previous_position_m = None if position is None else position.copy()
        self.event_count += 1

        features[~valid] = 0.0
        if not np.isfinite(features).all():
            raise FeatureNumericalError("TR motion 编码产生非有限值")
        return features, valid


@dataclass
class CausalTRObservationIncrementState:
    """不使用时间的逐观测变化状态。"""

    previous_speed_mps: float | None = None
    previous_radial_speed_mps: float | None = None
    previous_course_deg: float | None = None
    previous_range_m: float | None = None
    previous_position_m: np.ndarray | None = None
    previous_step_m: np.ndarray | None = None
    path_anchor_m: np.ndarray | None = None
    cumulative_path_m: float = 0.0
    event_count: int = 0
    ignore_first_speed_course: bool = True

    def reset(self) -> None:
        self.previous_speed_mps = None
        self.previous_radial_speed_mps = None
        self.previous_course_deg = None
        self.previous_range_m = None
        self.previous_position_m = None
        self.previous_step_m = None
        self.path_anchor_m = None
        self.cumulative_path_m = 0.0
        self.event_count = 0

    def update(self, values: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        contract = resolve_causal_tr_feature_contract(
            TR_OBSERVATION_INCREMENT_SCHEMA
        )
        features = np.zeros((contract.value_dim,), dtype=np.float32)
        valid = np.zeros((contract.value_dim,), dtype=np.bool_)
        speed, speed_missing = _number(values, "speed_mps")
        radial, radial_missing = _number(values, "radial_speed_mps")
        course, course_missing = _number(values, "course_deg")
        range_m, range_missing = _number(values, "range_m")
        position = _spherical_position_m(values)

        if not speed_missing and self.previous_speed_mps is not None:
            features[0] = _scaled_increment(
                speed - self.previous_speed_mps,
                contract.scales[contract.value_names[0]],
            )
            valid[0] = True
        if not radial_missing and self.previous_radial_speed_mps is not None:
            features[1] = _scaled_increment(
                radial - self.previous_radial_speed_mps,
                contract.scales[contract.value_names[1]],
            )
            valid[1] = True
            features[6] = float(radial * self.previous_radial_speed_mps < 0.0)
            valid[6] = True
        if not course_missing and self.previous_course_deg is not None:
            features[2] = _scaled_increment(
                _wrapped_delta_degrees(course, self.previous_course_deg),
                contract.scales[contract.value_names[2]],
            )
            valid[2] = True

        range_delta: float | None = None
        if not range_missing and self.previous_range_m is not None:
            range_delta = range_m - self.previous_range_m
            features[4] = _scaled_increment(
                range_delta, contract.scales[contract.value_names[4]]
            )
            valid[4] = True
            if not radial_missing and self.previous_radial_speed_mps is not None:
                interval_radial = 0.5 * (radial + self.previous_radial_speed_mps)
                features[5] = float(np.sign(interval_radial) * np.sign(range_delta))
                valid[5] = True

        if position is not None and self.previous_position_m is not None:
            step = position - self.previous_position_m
            step_distance = float(np.linalg.norm(step))
            features[3] = _scaled_increment(
                step_distance,
                contract.scales[contract.value_names[3]],
                nonnegative=True,
            )
            valid[3] = True
            if self.previous_step_m is not None:
                previous_distance = float(np.linalg.norm(self.previous_step_m))
                if step_distance > 1.0e-6 and previous_distance > 1.0e-6:
                    cosine = float(
                        np.dot(step, self.previous_step_m)
                        / (step_distance * previous_distance)
                    )
                    features[7] = float(np.clip(cosine, -1.0, 1.0))
                    valid[7] = True
            if self.path_anchor_m is None:
                self.path_anchor_m = self.previous_position_m.copy()
            self.cumulative_path_m += step_distance
            if self.cumulative_path_m > 1.0e-6:
                displacement = float(np.linalg.norm(position - self.path_anchor_m))
                features[8] = float(
                    np.clip(displacement / self.cumulative_path_m, 0.0, 1.0)
                )
                valid[8] = True
            self.previous_step_m = step.copy()
        elif position is not None:
            self.path_anchor_m = position.copy()
            self.cumulative_path_m = 0.0
            self.previous_step_m = None
        else:
            self.path_anchor_m = None
            self.cumulative_path_m = 0.0
            self.previous_step_m = None

        first_event = self.event_count == 0
        self.previous_speed_mps = (
            None
            if speed_missing or (first_event and self.ignore_first_speed_course)
            else speed
        )
        self.previous_radial_speed_mps = None if radial_missing else radial
        self.previous_course_deg = (
            None
            if course_missing or (first_event and self.ignore_first_speed_course)
            else course
        )
        self.previous_range_m = None if range_missing else range_m
        self.previous_position_m = None if position is None else position.copy()
        self.event_count += 1

        features[~valid] = 0.0
        if not np.isfinite(features).all():
            raise FeatureNumericalError("TR observation increment 产生非有限值")
        return features, valid


def _scaled_increment(value: float, scale: float, *, nonnegative: bool = False) -> float:
    lower = 0.0 if nonnegative else -TR_OBSERVATION_INCREMENT_CLIP
    return float(
        np.clip(float(value) / float(scale), lower, TR_OBSERVATION_INCREMENT_CLIP)
    )


def _timing(item: Any, values: Mapping[str, Any]) -> tuple[float, bool]:
    dt = getattr(item, "tr_dt_seconds", None)
    valid = getattr(item, "tr_dt_valid", None)
    if dt is None:
        dt = values.get("dt_seconds", 0.0)
    if valid is None:
        valid = values.get("dt_valid", False)
    try:
        return float(dt), bool(valid)
    except (TypeError, ValueError) as exc:
        raise FeatureInputError("TR motion dt 无法解析") from exc


def encode_causal_tr_motion_sequence(
    values: Sequence[Mapping[str, Any] | Any],
    dt_seconds: Sequence[float] | np.ndarray | None = None,
    dt_valid: Sequence[bool] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        raise FeatureInputError("TR motion 序列不能为空")
    if (dt_seconds is None) != (dt_valid is None):
        raise FeatureInputError("dt_seconds 与 dt_valid 必须同时提供")
    if dt_seconds is None:
        timing = [_timing(item, _event_values(item)) for item in values]
        dt_values = [item[0] for item in timing]
        valid_values = [item[1] for item in timing]
    else:
        dt_values = list(dt_seconds)
        valid_values = list(dt_valid)  # type: ignore[arg-type]
        if len(dt_values) != len(values) or len(valid_values) != len(values):
            raise FeatureInputError("TR motion values、dt 和 validity 长度不一致")
    state = CausalTRMotionState()
    encoded = [
        state.update(
            _event_values(point),
            dt_seconds=float(dt),
            dt_valid=bool(is_valid),
        )
        for point, dt, is_valid in zip(values, dt_values, valid_values, strict=True)
    ]
    return (
        np.stack([item[0] for item in encoded], axis=0).astype(np.float32, copy=False),
        np.stack([item[1] for item in encoded], axis=0).astype(np.bool_, copy=False),
    )


def encode_causal_tr_observation_increment_sequence(
    values: Sequence[Mapping[str, Any] | Any],
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        raise FeatureInputError("TR observation increment 序列不能为空")
    state = CausalTRObservationIncrementState()
    encoded = [state.update(_event_values(point)) for point in values]
    return (
        np.stack([item[0] for item in encoded], axis=0).astype(np.float32, copy=False),
        np.stack([item[1] for item in encoded], axis=0).astype(np.bool_, copy=False),
    )


def causal_tr_motion_schema() -> FeatureSchema:
    contract = resolve_causal_tr_feature_contract(TR_MOTION_SCHEMA)
    return FeatureSchema(
        name="tr_motion",
        schema_version=contract.schema,
        value_names=contract.value_names,
        valid_names=contract.valid_names,
        mask_mode=contract.mask_mode,
        value_bounds=(-TR_MOTION_CLIP, TR_MOTION_CLIP),
    )


def causal_tr_motion_contract(schema: str = TR_MOTION_SCHEMA) -> dict[str, Any]:
    """返回指定因果 TR 动态变体的 schema 字典。"""

    return resolve_causal_tr_feature_contract(schema).as_dict()


__all__ = [
    "CausalTRFeatureContract",
    "CausalTRMotionState",
    "CausalTRObservationIncrementState",
    "TR_MOTION_CLIP",
    "TR_MOTION_MASK_MODE",
    "TR_MOTION_NAMES",
    "TR_MOTION_SCALES",
    "TR_MOTION_SCHEMA",
    "TR_MOTION_VALID_NAMES",
    "TR_OBSERVATION_INCREMENT_CLIP",
    "TR_OBSERVATION_INCREMENT_MASK_MODE",
    "TR_OBSERVATION_INCREMENT_NAMES",
    "TR_OBSERVATION_INCREMENT_SCALES",
    "TR_OBSERVATION_INCREMENT_SCHEMA",
    "TR_OBSERVATION_INCREMENT_VALID_NAMES",
    "causal_tr_motion_schema",
    "causal_tr_motion_contract",
    "encode_causal_tr_motion_sequence",
    "encode_causal_tr_observation_increment_sequence",
    "require_causal_tr_motion_schema",
    "resolve_causal_tr_feature_contract",
]
