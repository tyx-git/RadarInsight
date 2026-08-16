"""Causal, label-free feature contracts for the V21 KnownGate experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .micro_doppler import RawEchoFrame, _target_location, _target_record
from .rd_physical import validate_velocity_axis


KNOWN_GATE_FEATURE_SCHEMA = "cq08-current-event-known-gate-evidence-v1"
KNOWN_GATE_FEATURE_MASK_MODE = "gating_only_per_value"
SPECTRAL_TEMPORAL_CONSISTENCY = "spectral_temporal_consistency"
MODAL_CONSISTENCY = "modal_consistency"
TARGET_ROW_ISO = "target_row_iso"
D_RANGE_STABILITY = "d_range_stability"
BASELINE_VARIANT = "baseline"
V21A_VARIANT = "v21a"
V21B_VARIANT = "v21b"
V21C_VARIANT = "v21c"
V21D_VARIANT = "v21d"

SPECTRAL_FEATURE_NAMES = (
    "native_stft_adjacent_spectral_affinity_median",
    "native_stft_adjacent_spectral_affinity_minimum",
)
MODAL_FEATURE_NAMES = (
    "target_row_rd_peak_vs_tr_radial_abs_deviation_median_over_85mps",
    "target_row_rd_peak_vs_tr_radial_abs_deviation_max_over_85mps",
)
ISO_FEATURE_NAMES = ("target_row_linear_energy_fraction_mean",)
D_RANGE_FEATURE_NAMES = ("mtd_minus_track_range_std_m_over_75m",)
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
    V21B_VARIANT: MODAL_FEATURE_NAMES,
    V21C_VARIANT: ISO_FEATURE_NAMES,
    V21D_VARIANT: D_RANGE_FEATURE_NAMES,
}
FEATURE_INDICES_BY_VARIANT: Mapping[str, tuple[int, ...]] = {
    variant: tuple(KNOWN_GATE_FEATURE_NAMES.index(name) for name in names)
    for variant, names in FEATURE_NAMES_BY_VARIANT.items()
}
MODAL_DEVIATION_CLIP_MPS = 85.0
D_RANGE_STD_CLIP_M = 75.0


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
            variant: list(names) for variant, names in FEATURE_NAMES_BY_VARIANT.items()
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


def load_known_gate_feature_cache(
    path: str | Path,
    *,
    expected_sample_ids: Iterable[str],
    variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    try:
        indices = FEATURE_INDICES_BY_VARIANT[str(variant)]
    except KeyError as exc:
        raise ValueError(f"unsupported KnownGate feature variant: {variant!r}") from exc
    if not indices:
        raise ValueError("baseline KnownGate does not load a feature cache")
    cache_path = Path(path)
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
    if values.shape != (len(identifiers), KNOWN_GATE_FEATURE_DIM) or values.dtype != np.float32:
        raise ValueError("KnownGate cache values must be float32 with shape [N,6]")
    if valid.shape != values.shape or valid.dtype != np.bool_:
        raise ValueError("KnownGate cache valid must be bool with shape [N,6]")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("KnownGate cache values must be finite and bounded to [0,1]")
    if np.any(values[~valid] != 0.0):
        raise ValueError("invalid KnownGate cache values must be exactly zero")
    expected = [str(value) for value in expected_sample_ids]
    if identifiers != expected:
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
        "variant": str(variant),
        "feature_names": list(FEATURE_NAMES_BY_VARIANT[str(variant)]),
        "event_count": len(identifiers),
        "valid_value_count": int(selected_valid.sum()),
        "invalid_value_count": int((~selected_valid).sum()),
    }


__all__ = [
    "BASELINE_VARIANT",
    "D_RANGE_FEATURE_NAMES",
    "D_RANGE_STABILITY",
    "FEATURE_INDICES_BY_VARIANT",
    "FEATURE_NAMES_BY_VARIANT",
    "ISO_FEATURE_NAMES",
    "KNOWN_GATE_FEATURE_DIM",
    "KNOWN_GATE_FEATURE_NAMES",
    "KNOWN_GATE_FEATURE_SCHEMA",
    "KnownGateFeatureEvent",
    "MODAL_CONSISTENCY",
    "MODAL_FEATURE_NAMES",
    "SPECTRAL_FEATURE_NAMES",
    "SPECTRAL_TEMPORAL_CONSISTENCY",
    "TARGET_ROW_ISO",
    "V21A_VARIANT",
    "V21B_VARIANT",
    "V21C_VARIANT",
    "V21D_VARIANT",
    "extract_canonical_known_gate_features",
    "extract_known_gate_feature_event",
    "extract_spectral_temporal_consistency",
    "feature_contract",
    "load_known_gate_feature_cache",
]
