from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np



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
            "clip_absolute": self.clip_absolute,
        }


_CAUSAL_TR_FEATURE_CONTRACTS = {
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


def resolve_causal_tr_feature_contract(schema: str | None) -> CausalTRFeatureContract:
    try:
        return _CAUSAL_TR_FEATURE_CONTRACTS[str(schema)]
    except KeyError as exc:
        raise ValueError(f"unsupported causal TR feature schema: {schema!r}") from exc


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _number(values: dict[str, Any], name: str) -> tuple[float, bool]:
    value = values.get(name)
    if not _finite(value):
        return 0.0, True
    return float(value), False


def _angle_pair(values: dict[str, Any], name: str) -> tuple[float, float, bool]:
    value, missing = _number(values, name)
    if missing:
        return 0.0, 0.0, True
    radians = math.radians(value)
    return math.sin(radians), math.cos(radians), False




def _scaled_motion(value: float, scale: float, *, nonnegative: bool = False) -> float:
    lower = 0.0 if nonnegative else -TR_MOTION_CLIP
    return float(np.clip(float(value) / float(scale), lower, TR_MOTION_CLIP))


def _wrapped_delta_degrees(current: float, previous: float) -> float:
    return float((current - previous + 180.0) % 360.0 - 180.0)


def _spherical_position_m(values: dict[str, Any]) -> np.ndarray | None:
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


@dataclass
class CausalTRMotionState:
    """Minimal per-target history for causal TR motion encoding.

    The state is reset at every parent/session boundary.  Invalid or missing
    intervals never receive a fabricated duration: all nine values remain
    zero with ``valid=False`` for that event.  The validity mask is intended
    only for numerical gating and must not be concatenated into classifier
    inputs.
    """

    previous_speed_mps: float | None = None
    previous_radial_speed_mps: float | None = None
    previous_course_deg: float | None = None
    previous_range_m: float | None = None
    previous_position_m: np.ndarray | None = None
    path_anchor_m: np.ndarray | None = None
    cumulative_path_m: float = 0.0
    event_count: int = 0

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
        values: dict[str, Any],
        *,
        dt_seconds: float,
        dt_valid: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((len(TR_MOTION_NAMES),), dtype=np.float32)
        valid = np.zeros((len(TR_MOTION_NAMES),), dtype=bool)
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

        if interval_valid and not radial_missing and self.previous_radial_speed_mps is not None:
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
            features[5] = _scaled_motion(range_rate, TR_MOTION_SCALES[TR_MOTION_NAMES[5]])
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
                features[8] = float(np.clip(displacement / self.cumulative_path_m, 0.0, 1.0))
                valid[8] = True
        elif position is not None:
            self.path_anchor_m = position.copy()
            self.cumulative_path_m = 0.0
        else:
            self.path_anchor_m = None
            self.cumulative_path_m = 0.0

        # CQ08 first-point speed/course are frequently exact zero while radial
        # speed is non-zero.  Their upstream initialization semantics are not
        # confirmed, so V9 does not amplify them into a second-event derivative.
        first_event = self.event_count == 0
        self.previous_speed_mps = None if speed_missing or first_event else speed
        self.previous_radial_speed_mps = None if radial_missing else radial
        self.previous_course_deg = None if course_missing or first_event else course
        self.previous_range_m = None if range_missing else range_m
        self.previous_position_m = None if position is None else position.copy()
        self.event_count += 1

        features[~valid] = 0.0
        if features.shape != (len(TR_MOTION_NAMES),) or valid.shape != features.shape:
            raise ValueError("causal TR motion encoder produced an invalid shape")
        if not np.isfinite(features).all():
            raise ValueError("causal TR motion encoder produced a non-finite value")
        return features, valid


@dataclass
class CausalTRObservationIncrementState:
    """Causal per-observation changes that never consume time information.

    Values describe change per radar report, not change per second.  The state
    receives only current TR fields, so timestamp precision, ``dt`` and
    ``dt_valid`` cannot control either a value or its validity.  State must be
    reset at every parent/session/target boundary.
    """

    previous_speed_mps: float | None = None
    previous_radial_speed_mps: float | None = None
    previous_course_deg: float | None = None
    previous_range_m: float | None = None
    previous_position_m: np.ndarray | None = None
    previous_step_m: np.ndarray | None = None
    path_anchor_m: np.ndarray | None = None
    cumulative_path_m: float = 0.0
    event_count: int = 0

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

    def update(self, values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        contract = resolve_causal_tr_feature_contract(TR_OBSERVATION_INCREMENT_SCHEMA)
        features = np.zeros((contract.value_dim,), dtype=np.float32)
        valid = np.zeros((contract.value_dim,), dtype=bool)

        speed, speed_missing = _number(values, "speed_mps")
        radial, radial_missing = _number(values, "radial_speed_mps")
        course, course_missing = _number(values, "course_deg")
        range_m, range_missing = _number(values, "range_m")
        position = _spherical_position_m(values)

        if not speed_missing and self.previous_speed_mps is not None:
            features[0] = _scaled_observation_increment(
                speed - self.previous_speed_mps,
                contract.scales[contract.value_names[0]],
            )
            valid[0] = True

        if not radial_missing and self.previous_radial_speed_mps is not None:
            features[1] = _scaled_observation_increment(
                radial - self.previous_radial_speed_mps,
                contract.scales[contract.value_names[1]],
            )
            valid[1] = True
            features[6] = float(radial * self.previous_radial_speed_mps < 0.0)
            valid[6] = True

        if not course_missing and self.previous_course_deg is not None:
            features[2] = _scaled_observation_increment(
                _wrapped_delta_degrees(course, self.previous_course_deg),
                contract.scales[contract.value_names[2]],
            )
            valid[2] = True

        range_delta: float | None = None
        if not range_missing and self.previous_range_m is not None:
            range_delta = range_m - self.previous_range_m
            features[4] = _scaled_observation_increment(
                range_delta,
                contract.scales[contract.value_names[4]],
            )
            valid[4] = True
            if not radial_missing and self.previous_radial_speed_mps is not None:
                interval_radial = 0.5 * (radial + self.previous_radial_speed_mps)
                features[5] = float(np.sign(interval_radial) * np.sign(range_delta))
                valid[5] = True

        if position is not None and self.previous_position_m is not None:
            step = position - self.previous_position_m
            step_distance = float(np.linalg.norm(step))
            features[3] = _scaled_observation_increment(
                step_distance,
                contract.scales[contract.value_names[3]],
                nonnegative=True,
            )
            valid[3] = True
            if self.previous_step_m is not None:
                previous_distance = float(np.linalg.norm(self.previous_step_m))
                if step_distance > 1.0e-6 and previous_distance > 1.0e-6:
                    cosine = float(np.dot(step, self.previous_step_m) / (step_distance * previous_distance))
                    features[7] = float(np.clip(cosine, -1.0, 1.0))
                    valid[7] = True
            if self.path_anchor_m is None:
                self.path_anchor_m = self.previous_position_m.copy()
            self.cumulative_path_m += step_distance
            if self.cumulative_path_m > 1.0e-6:
                displacement = float(np.linalg.norm(position - self.path_anchor_m))
                features[8] = float(np.clip(displacement / self.cumulative_path_m, 0.0, 1.0))
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

        # CQ08 first-event speed/course can be tracker initialization zeros.
        # Do not amplify that placeholder into a second-event delta; both
        # channels become available from the next observed baseline onward.
        first_event = self.event_count == 0
        self.previous_speed_mps = None if speed_missing or first_event else speed
        self.previous_radial_speed_mps = None if radial_missing else radial
        self.previous_course_deg = None if course_missing or first_event else course
        self.previous_range_m = None if range_missing else range_m
        self.previous_position_m = None if position is None else position.copy()
        self.event_count += 1

        features[~valid] = 0.0
        if features.shape != (contract.value_dim,) or valid.shape != features.shape:
            raise ValueError("causal TR observation-increment encoder produced an invalid shape")
        if not np.isfinite(features).all():
            raise ValueError("causal TR observation-increment encoder produced a non-finite value")
        return features, valid


def _scaled_observation_increment(value: float, scale: float, *, nonnegative: bool = False) -> float:
    lower = 0.0 if nonnegative else -TR_OBSERVATION_INCREMENT_CLIP
    return float(
        np.clip(
            float(value) / float(scale),
            lower,
            TR_OBSERVATION_INCREMENT_CLIP,
        )
    )




def encode_causal_tr_observation_increment_sequence(
    values: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise ValueError("TR observation-increment sequence cannot be empty")
    state = CausalTRObservationIncrementState()
    encoded = [state.update(point) for point in values]
    return (
        np.stack([item[0] for item in encoded], axis=0).astype(np.float32, copy=False),
        np.stack([item[1] for item in encoded], axis=0).astype(bool, copy=False),
    )


def encode_tr_feature_values(values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Encode one TR point without using IDs, paths, labels or future points.

    Values are physically scaled and circular angles use sin/cos. Missing
    values become zero while the returned mask preserves the missingness.
    ``amplitude_db`` is a legacy field containing CQ08 linear magnitude, so a
    signed log transform is applied rather than treating it as calibrated dB.
    """

    range_m, range_missing = _number(values, "range_m")
    radial, radial_missing = _number(values, "radial_speed_mps")
    elevation_sin, elevation_cos, elevation_missing = _angle_pair(values, "elevation_deg")
    azimuth_sin, azimuth_cos, azimuth_missing = _angle_pair(values, "azimuth_deg")
    course_sin, course_cos, course_missing = _angle_pair(values, "course_deg")
    speed, speed_missing = _number(values, "speed_mps")
    amplitude, amplitude_missing = _number(values, "amplitude_db")
    snr, snr_missing = _number(values, "snr_db")
    height, height_missing = _number(values, "height_m")
    latitude, latitude_missing = _number(values, "latitude")
    longitude, longitude_missing = _number(values, "longitude")
    amplitude_log = math.copysign(math.log1p(abs(amplitude)), amplitude) / 20.0
    feature_values = np.asarray(
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
    if not np.isfinite(feature_values).all() or not np.isfinite(missing).all():
        raise ValueError("TR feature encoder produced a non-finite value")
    return feature_values, missing


def parse_feature_json(value: str) -> dict[str, Any]:
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError("tr_feature_values_json must decode to an object")
    return payload




__all__ = ["CausalTRFeatureContract", "CausalTRMotionState", "CausalTRObservationIncrementState", "TR_MOTION_CLIP", "TR_MOTION_MASK_MODE", "TR_MOTION_NAMES", "TR_MOTION_SCALES", "TR_MOTION_SCHEMA", "TR_MOTION_VALID_NAMES", "TR_OBSERVATION_INCREMENT_CLIP", "TR_OBSERVATION_INCREMENT_MASK_MODE", "TR_OBSERVATION_INCREMENT_NAMES", "TR_OBSERVATION_INCREMENT_SCALES", "TR_OBSERVATION_INCREMENT_SCHEMA", "TR_OBSERVATION_INCREMENT_VALID_NAMES", "encode_causal_tr_observation_increment_sequence", "encode_tr_feature_values", "parse_feature_json", "resolve_causal_tr_feature_contract"]