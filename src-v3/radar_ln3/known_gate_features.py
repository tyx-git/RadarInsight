"""Causal, label-free feature contracts for the V21 KnownGate experiments."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .micro_doppler import RawEchoFrame, _target_location, _target_record
from .rd_physical import validate_velocity_axis


KNOWN_GATE_FEATURE_SCHEMA = "cq08-current-event-known-gate-evidence-v1"
MAX_KNOWN_GATE_FEATURE_CACHE_BYTES = 64 * 1024 * 1024
INTERFRAME_SPECTRAL_FEATURE_SCHEMA = (
    "cq08-current-event-interframe-spectral-consistency-v1"
)
D_RANGE_PREFIX_FEATURE_SCHEMA = (
    "cq08-causal-prefix-mtd-minus-tr-range-stability-v1"
)
COURSE_PREFIX_FEATURE_SCHEMA = (
    "cq08-causal-prefix-course-circular-incoherence-v1"
)
KNOWN_GATE_FEATURE_MASK_MODE = "gating_only_per_value"
SPECTRAL_TEMPORAL_CONSISTENCY = "spectral_temporal_consistency"
MODAL_CONSISTENCY = "modal_consistency"
TARGET_ROW_ISO = "target_row_iso"
D_RANGE_STABILITY = "d_range_stability"
BASELINE_VARIANT = "baseline"
V21A_VARIANT = "v21a"
V21A_INTERFRAME_VARIANT = "v21a_interframe"
V21B_VARIANT = "v21b"
V21C_VARIANT = "v21c"
V21D_VARIANT = "v21d"
V21D_PREFIX_VARIANT = "v21d_prefix"
V21E_COURSE_PREFIX_VARIANT = "v21e_course_prefix"
V21F_COURSE_INTERFRAME_VARIANT = "v21f_course_interframe"
V21H_COURSE_ISO_VARIANT = "v21h_course_iso"
V21I_COURSE_D_RANGE_VARIANT = "v21i_course_d_range"
V22A_RD_TEMPORAL_RESIDUAL_VARIANT = "v22a_rd_temporal_residual"
# V24 deliberately keeps the only adopted V21e auxiliary input while creating
# a fresh, independently trained control for the next RD counter-evidence
# series.  It must remain a distinct variant so its checkpoints and reports
# cannot be mistaken for the historical V21e run.
V24_COURSE_REASON_AWARE_BASELINE_VARIANT = "v24_course_reason_aware_baseline"
# V24a appends one independent, current-event-only RD counter-evidence
# feature to the V24 course baseline.  It intentionally does not reuse V21a's
# optional preceding-event frame bridge: a one-frame current event is invalid.
V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT = "v24a_course_rd_peak_drift_mean"
# V24c deliberately reuses V24's only adopted causal-course input.  Its sole
# variable is the reason-aware liquid update at same-second observations.
V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT = "v24c_course_strict_same_second_skip"
# V25a is a fresh V24 replication whose only added behavior is read-only
# pre-clip gradient telemetry in the trainer.  It intentionally keeps the
# one adopted causal-course column and does not introduce a model feature.
V25A_GRADIENT_TELEMETRY_VARIANT = "v25a_course_preclip_gradient_telemetry"
# V25d is a loss-only follow-up to the completed V25a audit.  It reuses the
# exact selected V21e causal-course cache; a distinct variant keeps its cache
# identity explicit rather than relying on a fallback lookup.
V25D_PRESENT_CLASS_MEAN_VARIANT = "v25d_course_present_class_mean"
# V25e keeps the exact V25a causal-course cache.  Its sole variable is a
# mixed-batch shared-gradient projection, so no new feature column or cache
# schema is allowed.
V25E_SHARED_GRADIENT_PROJECTION_VARIANT = "v25e_shared_gradient_conflict_projection"
# V26 keeps the frozen V21e causal-course input for a fresh Known-only
# baseline.  The variant is intentionally distinct because its mixed
# validation/test open-set audit is a separate, post-checkpoint artifact.
V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT = "v26_known_only_open_set_baseline"

SPECTRAL_FEATURE_NAMES = (
    "native_stft_adjacent_spectral_affinity_median",
    "native_stft_adjacent_spectral_affinity_minimum",
)
INTERFRAME_SPECTRAL_FEATURE_NAMES = (
    "current_event_adjacent_rd_velocity_profile_cosine_mean",
    "current_event_adjacent_rd_velocity_profile_cosine_minimum",
    "current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
    "current_event_adjacent_rd_peak_velocity_drift_max_fraction",
    "current_event_zero_velocity_energy_fraction_std",
    "current_event_zero_velocity_energy_fraction_minimum",
)
MODAL_FEATURE_NAMES = (
    "target_row_rd_peak_vs_tr_radial_abs_deviation_median_over_85mps",
    "target_row_rd_peak_vs_tr_radial_abs_deviation_max_over_85mps",
)
ISO_FEATURE_NAMES = ("target_row_linear_energy_fraction_mean",)
D_RANGE_FEATURE_NAMES = ("mtd_minus_track_range_std_m_over_75m",)
D_RANGE_PREFIX_FEATURE_NAMES = (
    "causal_prefix_mtd_minus_tr_range_sample_std_m_over_75m",
)
COURSE_PREFIX_FEATURE_NAMES = (
    "causal_prefix_course_circular_incoherence",
)
COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    "current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
)
COURSE_INTERFRAME_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    *INTERFRAME_SPECTRAL_FEATURE_NAMES,
)
COURSE_ISO_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    *ISO_FEATURE_NAMES,
)
COURSE_D_RANGE_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    *D_RANGE_PREFIX_FEATURE_NAMES,
)
KNOWN_GATE_FEATURE_NAMES = (
    *SPECTRAL_FEATURE_NAMES,
    *MODAL_FEATURE_NAMES,
    *ISO_FEATURE_NAMES,
    *D_RANGE_FEATURE_NAMES,
)
KNOWN_GATE_FEATURE_DIM = len(KNOWN_GATE_FEATURE_NAMES)
FEATURE_NAMES_BY_VARIANT: Mapping[str, tuple[str, ...]] = {
    BASELINE_VARIANT: (),
    V21A_VARIANT: SPECTRAL_FEATURE_NAMES,
    V21A_INTERFRAME_VARIANT: INTERFRAME_SPECTRAL_FEATURE_NAMES,
    V21B_VARIANT: MODAL_FEATURE_NAMES,
    V21C_VARIANT: ISO_FEATURE_NAMES,
    V21D_VARIANT: D_RANGE_FEATURE_NAMES,
    V21D_PREFIX_VARIANT: D_RANGE_PREFIX_FEATURE_NAMES,
    V21E_COURSE_PREFIX_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V21F_COURSE_INTERFRAME_VARIANT: COURSE_INTERFRAME_FEATURE_NAMES,
    V21H_COURSE_ISO_VARIANT: COURSE_ISO_FEATURE_NAMES,
    V21I_COURSE_D_RANGE_VARIANT: COURSE_D_RANGE_FEATURE_NAMES,
    # V22a reuses the frozen V21e course cache.  Its new RD evidence is an
    # internal model branch, deliberately not an external feature-cache field.
    V22A_RD_TEMPORAL_RESIDUAL_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT: (
        COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES
    ),
    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V25A_GRADIENT_TELEMETRY_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V25D_PRESENT_CLASS_MEAN_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V25E_SHARED_GRADIENT_PROJECTION_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
}
# These variants store an already-selected feature matrix rather than the
# original canonical KnownGate feature matrix.  They must therefore use
# cache-local column positions (0..width-1), not canonical feature indices.
# Keep this set shared by index construction and cache loading: a mismatch
# silently turns a reuse-only variant into a KeyError at Dataset construction.
SELECTED_CACHE_VARIANTS = frozenset(
    {
        V21A_INTERFRAME_VARIANT,
        V21D_PREFIX_VARIANT,
        V21E_COURSE_PREFIX_VARIANT,
        V21F_COURSE_INTERFRAME_VARIANT,
        V21H_COURSE_ISO_VARIANT,
        V21I_COURSE_D_RANGE_VARIANT,
        V22A_RD_TEMPORAL_RESIDUAL_VARIANT,
        V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
        V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
        V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
        V25A_GRADIENT_TELEMETRY_VARIANT,
        V25D_PRESENT_CLASS_MEAN_VARIANT,
        V25E_SHARED_GRADIENT_PROJECTION_VARIANT,
        V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
    }
)
FEATURE_INDICES_BY_VARIANT: Mapping[str, tuple[int, ...]] = {
    variant: tuple(KNOWN_GATE_FEATURE_NAMES.index(name) for name in names)
    for variant, names in FEATURE_NAMES_BY_VARIANT.items()
    if variant not in SELECTED_CACHE_VARIANTS
}
MODAL_DEVIATION_CLIP_MPS = 85.0
D_RANGE_STD_CLIP_M = 75.0


def extract_causal_course_circular_incoherence(
    course_degrees: Sequence[float | None],
    *,
    minimum_valid_nonfirst_points: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a causal prefix circular-dispersion sequence for TR course.

    CQ08's first reported course is unconditionally excluded because it is
    frequently an initialization zero.  A later finite 0-degree course is a
    valid physical direction.  At event ``t`` the statistic uses only valid
    non-first courses from events ``1..t`` and is
    ``1 - abs(mean(exp(1j * radians(course))))``.  A missing current course
    does not erase an already-valid causal prefix.
    """

    minimum = int(minimum_valid_nonfirst_points)
    if minimum < 1:
        raise ValueError("course circular incoherence minimum support must be positive")
    courses = tuple(course_degrees)
    values = np.zeros((len(courses), 1), dtype=np.float32)
    valid = np.zeros((len(courses), 1), dtype=np.bool_)
    sine_sum = 0.0
    cosine_sum = 0.0
    count = 0
    for event_index, raw_value in enumerate(courses):
        if event_index > 0 and raw_value is not None:
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("course circular incoherence received non-finite course")
            radians = math.radians(value)
            sine_sum += math.sin(radians)
            cosine_sum += math.cos(radians)
            count += 1
        if count >= minimum:
            resultant_length = math.hypot(sine_sum, cosine_sum) / float(count)
            values[event_index, 0] = np.float32(
                np.clip(1.0 - resultant_length, 0.0, 1.0)
            )
            valid[event_index, 0] = True
    return values, valid


@dataclass(frozen=True)
class KnownGateFeatureEvent:
    values: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        valid = np.asarray(self.valid)
        if values.shape != (KNOWN_GATE_FEATURE_DIM,) or values.dtype != np.float32:
            raise ValueError("KnownGate feature values must be float32 with shape [6]")
        if valid.shape != (KNOWN_GATE_FEATURE_DIM,) or valid.dtype != np.bool_:
            raise ValueError("KnownGate feature validity must be bool with shape [6]")
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("KnownGate feature values must be finite and bounded to [0,1]")
        if np.any(values[~valid] != 0.0):
            raise ValueError("invalid KnownGate feature values must be exactly zero")


def _native_target_row_stft_power(
    frame: RawEchoFrame,
    *,
    track_point_index: int,
) -> np.ndarray:
    """Return full-native-Doppler STFT power without a velocity-band crop."""

    record = _target_record(frame, int(track_point_index))
    target_range_index, _ = _target_location(frame, record)
    signal = np.asarray(frame.iq[target_range_index], dtype=np.complex128)
    segment_length = 32 if signal.size < 192 else 64
    segment_length = min(segment_length, int(signal.size))
    if segment_length < 4:
        raise ValueError("KnownGate spectral consistency requires at least four slow-time samples")
    hop = max(1, segment_length // 2)
    starts = list(range(0, int(signal.size) - segment_length + 1, hop))
    if not starts:
        starts = [0]
    window = np.hanning(segment_length).astype(np.float64)
    spectra = [
        np.square(
            np.abs(np.fft.fftshift(np.fft.fft(signal[start : start + segment_length] * window, n=128))),
            dtype=np.float64,
        )
        for start in starts
    ]
    power = np.stack(spectra, axis=1)
    if power.shape[1] < 2 or not np.isfinite(power).all() or float(power.max(initial=0.0)) <= 0.0:
        raise ValueError("KnownGate native STFT does not contain two finite positive time spectra")
    return power


def extract_spectral_temporal_consistency(
    frames: Sequence[RawEchoFrame],
    raw_frame_indices: Iterable[int],
    *,
    track_point_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure adjacent-time spectral-shape affinity in the current event.

    Each native STFT column is converted to a probability distribution and
    adjacent columns are compared with the Bhattacharyya affinity.  The median
    describes typical consistency; the minimum retains an occasional abrupt
    spectral-shape change.  No label, future event, PRT mode or path identifier
    enters the returned values.
    """

    indices = [int(value) for value in raw_frame_indices]
    if not indices:
        return np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.bool_)
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("KnownGate raw frame indices must be ordered and unique")
    if min(indices) < 0 or max(indices) >= len(frames):
        raise ValueError("KnownGate raw frame index is outside the parsed echo")
    affinities: list[float] = []
    for raw_index in indices:
        power = _native_target_row_stft_power(
            frames[raw_index], track_point_index=int(track_point_index)
        )
        column_total = power.sum(axis=0, dtype=np.float64, keepdims=True)
        probability = power / np.maximum(column_total, np.finfo(np.float64).tiny)
        adjacent = np.sum(
            np.sqrt(np.maximum(probability[:, :-1] * probability[:, 1:], 0.0)),
            axis=0,
            dtype=np.float64,
        )
        affinities.extend(np.clip(adjacent, 0.0, 1.0).astype(float).tolist())
    if not affinities:
        return np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.bool_)
    values = np.asarray(
        [np.median(affinities), np.min(affinities)], dtype=np.float32
    )
    return np.clip(values, 0.0, 1.0), np.ones((2,), dtype=np.bool_)


def extract_interframe_spectral_temporal_consistency(
    amplitude_db: np.ndarray,
    velocity_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure current-event spectral-shape consistency between adjacent RD frames.

    The original train-only V21 feature probe defined spectral temporal
    consistency on adjacent *RD frames*: each canonical frame is collapsed over
    range into a full velocity profile, then adjacent profiles are compared by
    cosine similarity. Peak-velocity drift and the variability of the near-zero
    velocity energy fraction retain complementary shape-motion evidence from
    the original train-only diagnostic. This differs materially from an
    intra-frame STFT-window statistic.

    The caller may prepend the immediately preceding valid frame to the current
    event. That bridge makes one-frame events comparable without reading a
    future frame. Missing-RD gaps must reset the bridge. Inputs with fewer than
    two frames have no temporal comparison and are invalid rather than assigned
    a fabricated consistency value.
    """

    amplitude = np.asarray(amplitude_db)
    if amplitude.ndim != 3 or amplitude.shape[1:] != (21, 384):
        raise ValueError(
            "V21a inter-frame spectral consistency requires amplitude [frames,21,384]"
        )
    if not np.isfinite(amplitude).all():
        raise ValueError("V21a inter-frame spectral consistency received non-finite amplitude")
    if amplitude.shape[0] < 2:
        return np.zeros((6,), dtype=np.float32), np.zeros((6,), dtype=np.bool_)

    selected = amplitude.astype(np.float64, copy=False)
    velocity = validate_velocity_axis(velocity_mps)
    if velocity.shape != (384,):
        raise ValueError("V21a inter-frame velocity axis must contain 384 bins")
    frame_maximum = selected.max(axis=(1, 2), keepdims=True)
    relative_power = np.power(10.0, (selected - frame_maximum) / 10.0)
    profiles = relative_power.sum(axis=1, dtype=np.float64)
    profile_norm = np.linalg.norm(profiles, axis=1)
    if np.any(~np.isfinite(profile_norm)) or np.any(profile_norm <= np.finfo(np.float64).tiny):
        raise ValueError("V21a inter-frame RD velocity profile has no finite positive norm")
    adjacent = np.sum(profiles[:-1] * profiles[1:], axis=1, dtype=np.float64) / (
        profile_norm[:-1] * profile_norm[1:]
    )
    adjacent = np.clip(adjacent, 0.0, 1.0)
    peak_velocity = velocity[np.argmax(profiles, axis=1)]
    velocity_span = float(velocity[-1] - velocity[0])
    if not math.isfinite(velocity_span) or velocity_span <= 0.0:
        raise ValueError("V21a inter-frame velocity span must be finite and positive")
    peak_drift_fraction = np.clip(
        np.abs(np.diff(peak_velocity)) / velocity_span, 0.0, 1.0
    )
    probability = profiles / np.maximum(
        profiles.sum(axis=1, dtype=np.float64, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    zero_velocity = probability[:, np.abs(velocity) <= 2.0].sum(
        axis=1, dtype=np.float64
    )
    values = np.asarray(
        [
            np.mean(adjacent),
            np.min(adjacent),
            np.mean(peak_drift_fraction),
            np.max(peak_drift_fraction),
            np.std(zero_velocity, ddof=0),
            np.min(zero_velocity),
        ],
        dtype=np.float32,
    )
    return np.clip(values, 0.0, 1.0), np.ones((6,), dtype=np.bool_)


def extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction(
    amplitude_db: np.ndarray,
    velocity_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only current-event adjacent RD peak-drift evidence.

    This narrow V24a extractor deliberately does *not* accept an optional
    preceding-event bridge.  An event must itself contain at least two ordered
    canonical RD frames to contribute one adjacent-frame velocity comparison.
    The invalid representation is the repository-wide zero sentinel paired
    with ``valid=False``.  The cache retains that distinction for alignment
    and audit: a physically zero drift with two or more frames has
    ``valid=True``.  The current KnownGate contract deliberately uses a
    gating-only validity mask, however, so validity is *not* an additional
    learned input and both cases contribute numeric zero at fusion.  V24a
    therefore tests drift magnitude, not RD-frame-availability as a second
    experimental factor.
    """

    amplitude = np.asarray(amplitude_db)
    if amplitude.ndim != 3 or amplitude.shape[1:] != (21, 384):
        raise ValueError(
            "V24a current-event RD peak drift requires amplitude [frames,21,384]"
        )
    if not np.isfinite(amplitude).all():
        raise ValueError("V24a current-event RD peak drift received non-finite amplitude")
    if amplitude.shape[0] < 2:
        return np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.bool_)
    velocity = validate_velocity_axis(velocity_mps)
    if velocity.shape != (384,):
        raise ValueError("V24a current-event velocity axis must contain 384 bins")
    velocity_span = float(velocity[-1] - velocity[0])
    if not math.isfinite(velocity_span) or velocity_span <= 0.0:
        raise ValueError("V24a current-event velocity span must be finite and positive")
    selected = amplitude.astype(np.float64, copy=False)
    frame_maximum = selected.max(axis=(1, 2), keepdims=True)
    relative_power = np.power(10.0, (selected - frame_maximum) / 10.0)
    profiles = relative_power.sum(axis=1, dtype=np.float64)
    if not np.isfinite(profiles).all() or np.any(
        profiles.sum(axis=1, dtype=np.float64) <= np.finfo(np.float64).tiny
    ):
        raise ValueError("V24a current-event RD velocity profile is not positive")
    peak_velocity = velocity[np.argmax(profiles, axis=1)]
    drift = np.abs(np.diff(peak_velocity)) / velocity_span
    value = np.asarray([np.clip(np.mean(drift), 0.0, 1.0)], dtype=np.float32)
    return value, np.ones((1,), dtype=np.bool_)


def extract_canonical_known_gate_features(
    amplitude_db: np.ndarray,
    velocity_mps: np.ndarray,
    range_offset_m: np.ndarray,
    *,
    tr_radial_speed_mps: float | None,
    tr_range_m: float | None,
    mtd_detected_range_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract V21b/c/d evidence from one current canonical RD event."""

    amplitude = np.asarray(amplitude_db)
    velocity = validate_velocity_axis(velocity_mps)
    offsets = np.asarray(range_offset_m, dtype=np.float64)
    detected_range = np.asarray(tuple(float(value) for value in mtd_detected_range_m), dtype=np.float64)
    if amplitude.ndim != 3 or amplitude.shape[1:] != (21, 384):
        raise ValueError("KnownGate canonical amplitude must have shape [frames,21,384]")
    if offsets.shape != (21,) or not np.isfinite(offsets).all():
        raise ValueError("KnownGate canonical range offset must contain 21 finite values")
    if detected_range.shape != (amplitude.shape[0],):
        raise ValueError("KnownGate MTD range references must align with current-event frames")
    if not np.isfinite(amplitude).all() or not np.isfinite(detected_range).all():
        raise ValueError("KnownGate canonical evidence contains a non-finite value")
    radial_speed = None if tr_radial_speed_mps is None else float(tr_radial_speed_mps)
    track_range = None if tr_range_m is None else float(tr_range_m)
    if radial_speed is not None and not math.isfinite(radial_speed):
        raise ValueError("KnownGate TR radial speed must be finite when present")
    if track_range is not None and not math.isfinite(track_range):
        raise ValueError("KnownGate TR range must be finite when present")
    if amplitude.shape[0] == 0:
        return np.zeros((4,), dtype=np.float32), np.zeros((4,), dtype=np.bool_)

    target_row = int(np.argmin(np.abs(offsets)))
    if abs(float(offsets[target_row])) > 1.0e-6:
        raise ValueError("KnownGate canonical range axis does not contain the zero-offset target row")
    selected = np.asarray(amplitude, dtype=np.float64)
    frame_maximum = selected.max(axis=(1, 2), keepdims=True)
    relative_power = np.power(10.0, (selected - frame_maximum) / 10.0)

    target_row_power = relative_power[:, target_row, :]
    peak_velocity = velocity[np.argmax(target_row_power, axis=1)]
    if radial_speed is None:
        modal = np.zeros((2,), dtype=np.float32)
        modal_valid = np.zeros((2,), dtype=np.bool_)
    else:
        deviation = np.abs(peak_velocity - radial_speed)
        modal = np.asarray(
            [
                np.clip(np.median(deviation) / MODAL_DEVIATION_CLIP_MPS, 0.0, 1.0),
                np.clip(np.max(deviation) / MODAL_DEVIATION_CLIP_MPS, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        modal_valid = np.ones((2,), dtype=np.bool_)

    total_power = relative_power.sum(axis=(1, 2), dtype=np.float64)
    iso_per_frame = target_row_power.sum(axis=1, dtype=np.float64) / np.maximum(
        total_power, np.finfo(np.float64).tiny
    )
    iso = np.asarray([np.clip(np.mean(iso_per_frame), 0.0, 1.0)], dtype=np.float32)

    if track_range is not None and detected_range.size >= 2:
        d_range = detected_range - track_range
        d_range_value = np.asarray(
            [np.clip(np.std(d_range, ddof=0) / D_RANGE_STD_CLIP_M, 0.0, 1.0)],
            dtype=np.float32,
        )
        d_range_valid = np.ones((1,), dtype=np.bool_)
    else:
        d_range_value = np.zeros((1,), dtype=np.float32)
        d_range_valid = np.zeros((1,), dtype=np.bool_)
    values = np.concatenate([modal, iso, d_range_value]).astype(np.float32, copy=False)
    valid = np.concatenate(
        [modal_valid, np.ones((1,), dtype=np.bool_), d_range_valid]
    )
    return values, valid


def extract_known_gate_feature_event(
    *,
    raw_frames: Sequence[RawEchoFrame],
    raw_frame_indices: Iterable[int],
    track_point_index: int,
    amplitude_db: np.ndarray,
    velocity_mps: np.ndarray,
    range_offset_m: np.ndarray,
    tr_radial_speed_mps: float | None,
    tr_range_m: float | None,
    mtd_detected_range_m: Sequence[float],
) -> KnownGateFeatureEvent:
    spectral_values, spectral_valid = extract_spectral_temporal_consistency(
        raw_frames,
        raw_frame_indices,
        track_point_index=int(track_point_index),
    )
    canonical_values, canonical_valid = extract_canonical_known_gate_features(
        amplitude_db,
        velocity_mps,
        range_offset_m,
        tr_radial_speed_mps=tr_radial_speed_mps,
        tr_range_m=tr_range_m,
        mtd_detected_range_m=mtd_detected_range_m,
    )
    return KnownGateFeatureEvent(
        values=np.concatenate([spectral_values, canonical_values]).astype(np.float32),
        valid=np.concatenate([spectral_valid, canonical_valid]).astype(np.bool_),
    )


def feature_contract() -> dict[str, Any]:
    return {
        "schema": KNOWN_GATE_FEATURE_SCHEMA,
        "value_names": list(KNOWN_GATE_FEATURE_NAMES),
        "value_dim": KNOWN_GATE_FEATURE_DIM,
        "value_dtype": "float32",
        "value_bounds": [0.0, 1.0],
        "valid_dtype": "bool_per_value",
        "valid_mask_mode": KNOWN_GATE_FEATURE_MASK_MODE,
        "variant_features": {
            variant: list(FEATURE_NAMES_BY_VARIANT[variant])
            for variant in (
                BASELINE_VARIANT,
                V21A_VARIANT,
                V21B_VARIANT,
                V21C_VARIANT,
                V21D_VARIANT,
                V21D_PREFIX_VARIANT,
                V21E_COURSE_PREFIX_VARIANT,
                V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
                V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
                V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
                V21F_COURSE_INTERFRAME_VARIANT,
                V21H_COURSE_ISO_VARIANT,
                V21I_COURSE_D_RANGE_VARIANT,
                V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
            )
        },
        "causal_scope": "current event only; no future event or parent-final statistic",
        "label_policy": "feature extraction is label-free; labels are used only by train/evaluation grouping",
        "identifier_policy": "sample/parent/path/frame identifiers are cache alignment only and never model inputs",
        "features": {
            SPECTRAL_TEMPORAL_CONSISTENCY: {
                "names": list(SPECTRAL_FEATURE_NAMES),
                "source": "current CQ08 raw complex I/Q target row",
                "formula": "native full-Doppler STFT adjacent-column Bhattacharyya affinity; event median and minimum",
            },
            MODAL_CONSISTENCY: {
                "names": list(MODAL_FEATURE_NAMES),
                "source": "current canonical RD zero-offset target row plus current TR radial speed",
                "formula": "median/max abs(RD target-row peak velocity - TR radial speed), divided by 85 m/s and clipped",
                "missing_policy": "TR radial speed missing -> both modal values zero and invalid; other canonical evidence remains independently usable",
            },
            TARGET_ROW_ISO: {
                "names": list(ISO_FEATURE_NAMES),
                "source": "current canonical RD amplitude before model normalization",
                "formula": "mean per-frame zero-offset target-row linear power / full 21x384 linear power",
            },
            D_RANGE_STABILITY: {
                "names": list(D_RANGE_FEATURE_NAMES),
                "source": "current canonical frame debug_info detected range plus current TR filtered range",
                "formula": "std(MTD detected range - TR range) across current-event frames, divided by 75 m and clipped",
                "missing_policy": "TR range missing or fewer than two current-event frames -> value zero and invalid; spectral/modal/iso evidence remains independently usable",
            },
        },
    }


def _preflight_known_gate_feature_cache_archive(path: Path) -> None:
    cache_size = path.stat().st_size
    if cache_size > MAX_KNOWN_GATE_FEATURE_CACHE_BYTES:
        raise ValueError(
            "KnownGate feature cache exceeds the fail-closed size limit: "
            f"{cache_size} > {MAX_KNOWN_GATE_FEATURE_CACHE_BYTES} bytes"
        )
    if path.suffix.lower() != ".npz":
        raise ValueError(f"KnownGate feature cache must be an .npz archive: {path}")
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            if len(members) > 16:
                raise ValueError("KnownGate feature cache contains too many archive members")
            declared_uncompressed = 0
            for member in members:
                if member.is_dir():
                    raise ValueError("KnownGate feature cache must not contain directory members")
                if member.file_size > MAX_KNOWN_GATE_FEATURE_CACHE_BYTES:
                    raise ValueError(
                        "KnownGate feature cache member exceeds the fail-closed size limit: "
                        f"{member.filename}={member.file_size} bytes"
                    )
                declared_uncompressed += int(member.file_size)
            if declared_uncompressed > MAX_KNOWN_GATE_FEATURE_CACHE_BYTES:
                raise ValueError(
                    "KnownGate feature cache declared uncompressed size exceeds the "
                    f"fail-closed limit: {declared_uncompressed} > "
                    f"{MAX_KNOWN_GATE_FEATURE_CACHE_BYTES} bytes"
                )
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"KnownGate feature cache is not a valid .npz archive: {path}"
        ) from exc


def load_known_gate_feature_cache(
    path: str | Path,
    *,
    expected_sample_ids: Iterable[str],
    variant: str,
    allow_superset: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    variant_name = str(variant)
    try:
        feature_names = FEATURE_NAMES_BY_VARIANT[variant_name]
    except KeyError as exc:
        raise ValueError(f"unsupported KnownGate feature variant: {variant!r}") from exc
    if not feature_names:
        raise ValueError("baseline KnownGate does not load a feature cache")
    if variant_name in SELECTED_CACHE_VARIANTS:
        cache_width = len(feature_names)
        indices = tuple(range(cache_width))
    else:
        cache_width = KNOWN_GATE_FEATURE_DIM
        indices = FEATURE_INDICES_BY_VARIANT[variant_name]
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"KnownGate feature cache is missing: {cache_path}")
    _preflight_known_gate_feature_cache_archive(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "values", "valid"}:
            raise ValueError("KnownGate feature cache fields must be sample_ids/values/valid")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("KnownGate cache sample_ids must be a one-dimensional string array")
    identifiers = [str(value) for value in sample_ids.tolist()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("KnownGate cache contains duplicate sample_ids")
    if values.shape != (len(identifiers), cache_width) or values.dtype != np.float32:
        raise ValueError(
            f"KnownGate cache values must be float32 with shape [N,{cache_width}]"
        )
    if valid.shape != values.shape or valid.dtype != np.bool_:
        raise ValueError(
            f"KnownGate cache valid must be bool with shape [N,{cache_width}]"
        )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("KnownGate cache values must be finite and bounded to [0,1]")
    if np.any(values[~valid] != 0.0):
        raise ValueError("invalid KnownGate cache values must be exactly zero")
    expected = [str(value) for value in expected_sample_ids]
    if allow_superset:
        if len(expected) != len(set(expected)):
            raise ValueError("KnownGate expected subset contains duplicate sample_ids")
        expected_set = set(expected)
        selected_indices = [
            index for index, sample_id in enumerate(identifiers) if sample_id in expected_set
        ]
        selected_identifiers = [identifiers[index] for index in selected_indices]
        if selected_identifiers != expected:
            missing = sorted(expected_set - set(identifiers))
            raise ValueError(
                "KnownGate cache does not preserve the requested Dataset subset ordering: "
                f"missing={missing[:3]}"
            )
        values = values[selected_indices]
        valid = valid[selected_indices]
        identifiers = selected_identifiers
    elif identifiers != expected:
        missing = sorted(set(expected) - set(identifiers))
        extra = sorted(set(identifiers) - set(expected))
        raise ValueError(
            "KnownGate cache ordering/sample set differs from the Dataset split: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    selected_values = values[:, indices]
    selected_valid = valid[:, indices]
    value_mapping = {
        sample_id: selected_values[index] for index, sample_id in enumerate(identifiers)
    }
    valid_mapping = {
        sample_id: selected_valid[index] for index, sample_id in enumerate(identifiers)
    }
    return value_mapping, valid_mapping, {
        "path": str(cache_path),
        "variant": variant_name,
        "feature_names": list(feature_names),
        "event_count": len(identifiers),
        "source_cache_event_count": len(sample_ids),
        "subset_from_superset": bool(allow_superset),
        "valid_value_count": int(selected_valid.sum()),
        "invalid_value_count": int((~selected_valid).sum()),
    }


__all__ = [
    "BASELINE_VARIANT",
    "COURSE_PREFIX_FEATURE_NAMES",
    "COURSE_PREFIX_FEATURE_SCHEMA",
    "COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES",
    "COURSE_INTERFRAME_FEATURE_NAMES",
    "COURSE_ISO_FEATURE_NAMES",
    "COURSE_D_RANGE_FEATURE_NAMES",
    "D_RANGE_FEATURE_NAMES",
    "D_RANGE_PREFIX_FEATURE_NAMES",
    "D_RANGE_PREFIX_FEATURE_SCHEMA",
    "D_RANGE_STD_CLIP_M",
    "D_RANGE_STABILITY",
    "FEATURE_INDICES_BY_VARIANT",
    "FEATURE_NAMES_BY_VARIANT",
    "INTERFRAME_SPECTRAL_FEATURE_NAMES",
    "INTERFRAME_SPECTRAL_FEATURE_SCHEMA",
    "ISO_FEATURE_NAMES",
    "KNOWN_GATE_FEATURE_DIM",
    "KNOWN_GATE_FEATURE_NAMES",
    "KNOWN_GATE_FEATURE_SCHEMA",
    "KnownGateFeatureEvent",
    "MODAL_CONSISTENCY",
    "MODAL_FEATURE_NAMES",
    "SPECTRAL_FEATURE_NAMES",
    "SPECTRAL_TEMPORAL_CONSISTENCY",
    "SELECTED_CACHE_VARIANTS",
    "TARGET_ROW_ISO",
    "V21A_VARIANT",
    "V21A_INTERFRAME_VARIANT",
    "V21B_VARIANT",
    "V21C_VARIANT",
    "V21D_VARIANT",
    "V21D_PREFIX_VARIANT",
    "V21E_COURSE_PREFIX_VARIANT",
    "V21F_COURSE_INTERFRAME_VARIANT",
    "V21H_COURSE_ISO_VARIANT",
    "V21I_COURSE_D_RANGE_VARIANT",
    "V22A_RD_TEMPORAL_RESIDUAL_VARIANT",
    "V24_COURSE_REASON_AWARE_BASELINE_VARIANT",
    "V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT",
    "V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT",
    "V25A_GRADIENT_TELEMETRY_VARIANT",
    "V25D_PRESENT_CLASS_MEAN_VARIANT",
    "V25E_SHARED_GRADIENT_PROJECTION_VARIANT",
    "V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT",
    "extract_canonical_known_gate_features",
    "extract_causal_course_circular_incoherence",
    "extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
    "extract_interframe_spectral_temporal_consistency",
    "extract_known_gate_feature_event",
    "extract_spectral_temporal_consistency",
    "feature_contract",
    "load_known_gate_feature_cache",
]
