from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np



RD_PHYSICAL_NAMES = (
    "velocity_bandwidth_mps_over_85",
    "zero_velocity_energy_fraction_abs_le_2_mps",
    "velocity_spectral_entropy_over_log_384",
    "top10_velocity_energy_fraction",
)
RD_PHYSICAL_VALUE_DIM = len(RD_PHYSICAL_NAMES)
RD_VELOCITY_BIN_COUNT = 384
RD_VELOCITY_MIN_MPS = -85.0
RD_VELOCITY_MAX_MPS = 85.0
RD_VELOCITY_AXIS_SHA256_FLOAT64 = "5912f04e9119f420c668074b3f73dbfeabe1d466e4ad701f9824a1e0598b03fe"
RD_ZERO_VELOCITY_HALF_WIDTH_MPS = 2.0
RD_TOP_VELOCITY_BIN_COUNT = 10


@dataclass(frozen=True)
class RDPhysicalEvent:
    values: np.ndarray
    valid: bool

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.shape != (RD_PHYSICAL_VALUE_DIM,) or values.dtype != np.float32:
            raise ValueError("RD physical event values must be float32 with shape [4]")
        if not np.isfinite(values).all():
            raise ValueError("RD physical event values must be finite")
        if not self.valid and np.any(values != 0.0):
            raise ValueError("an invalid RD physical event must be exactly zero")


def velocity_axis_sha256(velocity_mps: np.ndarray) -> str:
    """Hash the physical axis using the frozen float64 byte representation."""

    axis = np.ascontiguousarray(np.asarray(velocity_mps, dtype=np.float64))
    return hashlib.sha256(axis.tobytes(order="C")).hexdigest()


def validate_velocity_axis(velocity_mps: np.ndarray) -> np.ndarray:
    axis = np.asarray(velocity_mps, dtype=np.float64)
    if axis.shape != (RD_VELOCITY_BIN_COUNT,) or not np.isfinite(axis).all():
        raise ValueError("RD velocity axis must contain 384 finite values")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("RD velocity axis must be strictly increasing")
    if not math.isclose(float(axis[0]), RD_VELOCITY_MIN_MPS, abs_tol=1.0e-6):
        raise ValueError("RD velocity axis minimum is outside the frozen contract")
    if not math.isclose(float(axis[-1]), RD_VELOCITY_MAX_MPS, abs_tol=1.0e-6):
        raise ValueError("RD velocity axis maximum is outside the frozen contract")
    actual_hash = velocity_axis_sha256(axis)
    if actual_hash != RD_VELOCITY_AXIS_SHA256_FLOAT64:
        raise ValueError(
            "RD velocity axis hash differs from the frozen CQ08 axis: "
            f"expected={RD_VELOCITY_AXIS_SHA256_FLOAT64}, actual={actual_hash}"
        )
    return axis


def extract_current_event_rd_physical(
    amplitude_db: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    frame_mask: np.ndarray | Iterable[bool] | None = None,
) -> RDPhysicalEvent:
    """Extract four causal physical-spectrum values from one RD event.

    ``amplitude_db`` must be the canonical, normalization-free RD amplitude
    with shape ``[frames, 21, 384]``. Each valid frame is independently
    peak-normalized in logarithmic power before valid frames and range bins are
    summed into one velocity profile. No previous/future event is consumed.
    """

    axis = validate_velocity_axis(velocity_mps)
    amplitude = np.asarray(amplitude_db)
    if amplitude.ndim != 3 or amplitude.shape[1:] != (21, RD_VELOCITY_BIN_COUNT):
        raise ValueError("current-event RD amplitude must have shape [frames,21,384]")
    if not np.issubdtype(amplitude.dtype, np.number) or not np.isfinite(amplitude).all():
        raise ValueError("current-event RD amplitude must be finite numeric data")
    if frame_mask is None:
        active = np.ones((amplitude.shape[0],), dtype=bool)
    else:
        active = np.asarray(list(frame_mask) if not isinstance(frame_mask, np.ndarray) else frame_mask, dtype=bool)
        if active.shape != (amplitude.shape[0],):
            raise ValueError("current-event RD frame mask must have shape [frames]")
    selected = np.asarray(amplitude[active], dtype=np.float64)
    if selected.shape[0] == 0:
        return RDPhysicalEvent(np.zeros((RD_PHYSICAL_VALUE_DIM,), dtype=np.float32), False)

    frame_maximum = selected.max(axis=(1, 2), keepdims=True)
    relative_power = np.power(10.0, (selected - frame_maximum) / 10.0)
    velocity_profile = relative_power.sum(axis=(0, 1), dtype=np.float64)
    total = float(velocity_profile.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("current-event RD physical extraction produced invalid total power")
    probability = velocity_profile / total
    mean_velocity = float(np.sum(probability * axis, dtype=np.float64))
    variance = float(np.sum(probability * np.square(axis - mean_velocity), dtype=np.float64))
    bandwidth_scaled = math.sqrt(max(0.0, variance)) / RD_VELOCITY_MAX_MPS
    zero_velocity_fraction = float(
        probability[np.abs(axis) <= RD_ZERO_VELOCITY_HALF_WIDTH_MPS].sum(dtype=np.float64)
    )
    nonzero = probability[probability > 0.0]
    entropy_scaled = float(-np.sum(nonzero * np.log(nonzero), dtype=np.float64) / math.log(RD_VELOCITY_BIN_COUNT))
    top_count = min(RD_TOP_VELOCITY_BIN_COUNT, probability.size)
    top_fraction = float(np.partition(probability, -top_count)[-top_count:].sum(dtype=np.float64))
    values = np.asarray(
        [bandwidth_scaled, zero_velocity_fraction, entropy_scaled, top_fraction],
        dtype=np.float32,
    )
    tolerance = 2.0e-6
    if not np.isfinite(values).all() or np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError(f"current-event RD physical values violate [0,1] bounds: {values.tolist()}")
    values = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    return RDPhysicalEvent(values, True)






__all__ = ["RDPhysicalEvent", "RD_PHYSICAL_NAMES", "RD_PHYSICAL_VALUE_DIM", "RD_VELOCITY_AXIS_SHA256_FLOAT64", "extract_current_event_rd_physical", "validate_velocity_axis", "velocity_axis_sha256"]