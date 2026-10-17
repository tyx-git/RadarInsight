from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .runtime import preflight_npz_archive
from scipy.signal.windows import taylor


MICRO_DOPPLER_SCHEMA = "cq08-current-event-raw-iq-micro-doppler-summary-v3"
MICRO_DOPPLER_MASK_MODE = "gating_only"
MICRO_DOPPLER_NAMES = (
    "peak_band_energy_time_cv_clipped_0_2_over_2",
    "minus_10db_velocity_spread_ratio",
    "modulation_period_seconds_clipped_0_1s_over_0_1s",
)
MICRO_DOPPLER_VALUE_DIM = len(MICRO_DOPPLER_NAMES)

RANGE_BIN_COUNT = 31
STFT_VELOCITY_BIN_COUNT = 64
STFT_TIME_BIN_COUNT = 16
STFT_FRAME_CAP = 12
STFT_VELOCITY_MIN_MPS = -85.0
STFT_VELOCITY_MAX_MPS = 85.0
STFT_LOG_POWER_FLOOR_DB = -60.0
PEAK_BAND_HALF_WIDTH_MPS = 8.0
FLICKER_CV_CLIP = 2.0
MODULATION_PERIOD_CLIP_SECONDS = 0.1
MODULATION_MIN_AUTOCORRELATION = 0.25
SPEED_OF_LIGHT_MPS = 299_792_458.0






@dataclass(frozen=True)
class RawTargetRecord:
    raw_target_identifier: int
    track_point_index: int
    range_bin_raw: int
    doppler_bin_raw: int


@dataclass(frozen=True)
class RawEchoFrame:
    raw_frame_index: int
    frequency_hz: float
    cpi_count: int
    prt_num: int
    prt_seconds: float
    target_records: tuple[RawTargetRecord, ...]
    iq: np.ndarray


@dataclass(frozen=True)
class MicroDopplerFrame:
    stft_tensor: np.ndarray
    raw_summary: np.ndarray
    model_summary: np.ndarray
    modulation_period_estimable: bool
    target_range_index_zero_based: int
    target_doppler_index_zero_based: int
    target_doppler_mps: float
    target_local_stft_peak_velocity_mps: float
    global_stft_peak_velocity_mps: float
    target_local_peak_power_ratio_to_global: float
    stft_native_time_bin_count: int

    def __post_init__(self) -> None:
        if self.stft_tensor.shape != (STFT_VELOCITY_BIN_COUNT, STFT_TIME_BIN_COUNT):
            raise ValueError("micro-Doppler STFT tensor has an invalid shape")
        if self.stft_tensor.dtype != np.float32 or not np.isfinite(self.stft_tensor).all():
            raise ValueError("micro-Doppler STFT tensor must be finite float32")
        if np.any(self.stft_tensor < 0.0) or np.any(self.stft_tensor > 1.0):
            raise ValueError("micro-Doppler STFT tensor must be bounded to [0,1]")
        for value, name in ((self.raw_summary, "raw"), (self.model_summary, "model")):
            if value.shape != (MICRO_DOPPLER_VALUE_DIM,) or value.dtype != np.float32:
                raise ValueError(f"micro-Doppler {name} summary must be float32 with shape [3]")
            if not np.isfinite(value).all() or np.any(value < 0.0):
                raise ValueError(f"micro-Doppler {name} summary must be finite and non-negative")
        if np.any(self.model_summary > 1.0):
            raise ValueError("micro-Doppler model summary must be bounded to [0,1]")
        if (
            self.target_range_index_zero_based < 0
            or self.target_range_index_zero_based >= RANGE_BIN_COUNT
            or self.target_doppler_index_zero_based < 0
            or not np.isfinite(self.target_doppler_mps)
            or not np.isfinite(self.target_local_stft_peak_velocity_mps)
            or not np.isfinite(self.global_stft_peak_velocity_mps)
            or not np.isfinite(self.target_local_peak_power_ratio_to_global)
            or self.target_local_peak_power_ratio_to_global < 0.0
            or self.target_local_peak_power_ratio_to_global > 1.0
        ):
            raise ValueError("micro-Doppler target/STFT peak coordinates are invalid")


@dataclass(frozen=True)
class MicroDopplerEvent:
    values: np.ndarray
    valid: bool
    stft_tensors: np.ndarray
    frame_mask: np.ndarray
    raw_frame_indices: np.ndarray
    frame_raw_summaries: np.ndarray
    frame_period_estimable: np.ndarray
    target_range_indices: np.ndarray
    target_doppler_indices: np.ndarray
    target_doppler_mps: np.ndarray
    target_local_stft_peak_velocity_mps: np.ndarray
    global_stft_peak_velocity_mps: np.ndarray
    target_local_peak_power_ratio_to_global: np.ndarray
    native_time_bin_counts: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (MICRO_DOPPLER_VALUE_DIM,) or self.values.dtype != np.float32:
            raise ValueError("micro-Doppler event values must be float32 with shape [3]")
        if not np.isfinite(self.values).all() or np.any(self.values < 0.0) or np.any(self.values > 1.0):
            raise ValueError("micro-Doppler event values must be finite and bounded to [0,1]")
        if self.stft_tensors.shape != (
            STFT_FRAME_CAP,
            STFT_VELOCITY_BIN_COUNT,
            STFT_TIME_BIN_COUNT,
        ):
            raise ValueError("micro-Doppler event STFT frame tensor has an invalid shape")
        if self.stft_tensors.dtype != np.float16:
            raise ValueError("micro-Doppler stored STFT tensors must use float16")
        expected_vector = (STFT_FRAME_CAP,)
        if (
            self.frame_mask.shape != expected_vector
            or self.raw_frame_indices.shape != expected_vector
            or self.frame_period_estimable.shape != expected_vector
            or self.target_range_indices.shape != expected_vector
            or self.target_doppler_indices.shape != expected_vector
            or self.target_doppler_mps.shape != expected_vector
            or self.target_local_stft_peak_velocity_mps.shape != expected_vector
            or self.global_stft_peak_velocity_mps.shape != expected_vector
            or self.target_local_peak_power_ratio_to_global.shape != expected_vector
            or self.native_time_bin_counts.shape != expected_vector
            or self.frame_raw_summaries.shape != (STFT_FRAME_CAP, MICRO_DOPPLER_VALUE_DIM)
        ):
            raise ValueError("micro-Doppler event frame metadata shapes are invalid")
        if self.frame_mask.dtype != np.bool_ or self.frame_period_estimable.dtype != np.bool_:
            raise ValueError("micro-Doppler event masks must be bool")
        if (
            self.raw_frame_indices.dtype != np.int32
            or self.frame_raw_summaries.dtype != np.float32
            or self.target_range_indices.dtype != np.int16
            or self.target_doppler_indices.dtype != np.int32
            or self.target_doppler_mps.dtype != np.float32
            or self.target_local_stft_peak_velocity_mps.dtype != np.float32
            or self.global_stft_peak_velocity_mps.dtype != np.float32
            or self.target_local_peak_power_ratio_to_global.dtype != np.float32
            or self.native_time_bin_counts.dtype != np.int16
        ):
            raise ValueError("micro-Doppler event frame metadata dtypes are invalid")
        if (
            not np.isfinite(self.frame_raw_summaries).all()
            or not np.isfinite(self.target_doppler_mps).all()
            or not np.isfinite(self.target_local_stft_peak_velocity_mps).all()
            or not np.isfinite(self.global_stft_peak_velocity_mps).all()
            or not np.isfinite(self.target_local_peak_power_ratio_to_global).all()
            or np.any(self.target_local_peak_power_ratio_to_global < 0.0)
            or np.any(self.target_local_peak_power_ratio_to_global > 1.0)
        ):
            raise ValueError("micro-Doppler event frame metadata must be finite")
        if not self.valid:
            if (
                self.frame_mask.any()
                or np.any(self.values != 0.0)
                or np.any(self.stft_tensors != 0.0)
                or np.any(self.target_doppler_mps != 0.0)
                or np.any(self.target_local_stft_peak_velocity_mps != 0.0)
                or np.any(self.global_stft_peak_velocity_mps != 0.0)
                or np.any(self.target_local_peak_power_ratio_to_global != 0.0)
            ):
                raise ValueError("invalid micro-Doppler event must be exactly zero with no active frame")
        elif not self.frame_mask.any():
            raise ValueError("valid micro-Doppler event must contain at least one active frame")






def _target_record(frame: RawEchoFrame, track_point_index: int) -> RawTargetRecord:
    matches = [value for value in frame.target_records if value.track_point_index == int(track_point_index)]
    if len(matches) != 1:
        raise ValueError(
            "current raw frame must contain exactly one target record for the selected track point: "
            f"raw_frame_index={frame.raw_frame_index}, track_point_index={track_point_index}, matches={len(matches)}"
        )
    return matches[0]


def _target_location(frame: RawEchoFrame, record: RawTargetRecord) -> tuple[int, int]:
    """Reproduce the existing CQ08 local MTD target search without a label."""

    pulse_count = int(frame.prt_num)
    window = taylor(pulse_count, nbar=4, sll=30.0, norm=True, sym=True).astype(np.float64)
    rd = np.fft.fftshift(
        np.fft.fft(np.asarray(frame.iq, dtype=np.complex128) * window[None, :], axis=1),
        axes=1,
    )
    # The raw Doppler record and MATLAB processing are one-based. Preserve the
    # frozen mapping used by batch_process_all.m before the +/-5 local search.
    raw_bin = int(record.doppler_bin_raw)
    mapped_one_based = raw_bin - pulse_count // 2 if raw_bin > pulse_count / 2 else raw_bin + pulse_count // 2
    mapped_zero_based = int(np.clip(mapped_one_based - 1, 0, pulse_count - 1))
    range_start = max(0, 15 - 5)
    range_stop = min(RANGE_BIN_COUNT, 15 + 5 + 1)
    doppler_start = max(0, mapped_zero_based - 5)
    doppler_stop = min(pulse_count, mapped_zero_based + 5 + 1)
    local = np.abs(rd[range_start:range_stop, doppler_start:doppler_stop])
    if local.size == 0 or not np.isfinite(local).all():
        raise ValueError("CQ08 target-local MTD search produced an invalid plane")
    local_row, local_column = np.unravel_index(int(np.argmax(local)), local.shape)
    return range_start + int(local_row), doppler_start + int(local_column)


def _stft_linear_power(frame: RawEchoFrame, target_range_index: int) -> tuple[np.ndarray, np.ndarray, float]:
    signal = np.asarray(frame.iq[target_range_index], dtype=np.complex128)
    segment_length = 32 if signal.size < 192 else 64
    segment_length = min(segment_length, int(signal.size))
    hop = max(1, segment_length // 2)
    starts = list(range(0, int(signal.size) - segment_length + 1, hop))
    if not starts:
        starts = [0]
    nfft = 128
    window = np.hanning(segment_length).astype(np.float64)
    spectra = []
    for start in starts:
        segment = signal[start : start + segment_length]
        spectrum = np.fft.fftshift(np.fft.fft(segment * window, n=nfft))
        spectra.append(np.square(np.abs(spectrum), dtype=np.float64))
    native_power = np.stack(spectra, axis=1)
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=float(frame.prt_seconds)))
    source_velocity = frequency_hz * SPEED_OF_LIGHT_MPS / (2.0 * float(frame.frequency_hz))
    target_velocity = np.linspace(
        STFT_VELOCITY_MIN_MPS,
        STFT_VELOCITY_MAX_MPS,
        STFT_VELOCITY_BIN_COUNT,
        dtype=np.float64,
    )
    physical_power = np.stack(
        [
            np.interp(target_velocity, source_velocity, native_power[:, index], left=0.0, right=0.0)
            for index in range(native_power.shape[1])
        ],
        axis=1,
    )
    if not np.isfinite(physical_power).all() or float(physical_power.max(initial=0.0)) <= 0.0:
        raise ValueError("micro-Doppler STFT produced no finite positive power")
    return physical_power, target_velocity, float(hop) * float(frame.prt_seconds)


def _canonical_stft_tensor(power: np.ndarray) -> np.ndarray:
    old_time = np.linspace(0.0, 1.0, power.shape[1], dtype=np.float64)
    new_time = np.linspace(0.0, 1.0, STFT_TIME_BIN_COUNT, dtype=np.float64)
    resampled = np.stack(
        [np.interp(new_time, old_time, row) for row in power],
        axis=0,
    )
    maximum = max(float(resampled.max(initial=0.0)), np.finfo(np.float64).tiny)
    relative_db = 10.0 * np.log10(np.maximum(resampled / maximum, np.finfo(np.float64).tiny))
    bounded = np.clip(
        (relative_db - STFT_LOG_POWER_FLOOR_DB) / -STFT_LOG_POWER_FLOOR_DB,
        0.0,
        1.0,
    )
    return np.asarray(bounded, dtype=np.float32)


def _modulation_period(series: np.ndarray, step_seconds: float) -> tuple[float, bool]:
    centered = np.asarray(series, dtype=np.float64) - float(np.mean(series))
    if centered.size < 4 or float(np.std(centered)) <= 1.0e-12:
        return 0.0, False
    autocorrelation = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if float(autocorrelation[0]) <= 0.0:
        return 0.0, False
    autocorrelation = autocorrelation / autocorrelation[0]
    maximum_lag = max(2, centered.size // 2)
    candidates = [
        lag
        for lag in range(2, min(maximum_lag + 1, centered.size - 1))
        if autocorrelation[lag] > autocorrelation[lag - 1]
        and autocorrelation[lag] >= autocorrelation[lag + 1]
        and autocorrelation[lag] >= MODULATION_MIN_AUTOCORRELATION
    ]
    if not candidates:
        return 0.0, False
    return float(candidates[0]) * float(step_seconds), True


def extract_micro_doppler_frame(
    frame: RawEchoFrame,
    *,
    track_point_index: int,
) -> MicroDopplerFrame:
    record = _target_record(frame, track_point_index)
    target_range_index, target_doppler_index = _target_location(frame, record)
    power, velocity, time_step_seconds = _stft_linear_power(frame, target_range_index)
    target_doppler_hz = np.fft.fftshift(
        np.fft.fftfreq(int(frame.prt_num), d=float(frame.prt_seconds))
    )[target_doppler_index]
    target_doppler_mps = float(
        target_doppler_hz * SPEED_OF_LIGHT_MPS / (2.0 * float(frame.frequency_hz))
    )
    integrated = power.sum(axis=1, dtype=np.float64)
    global_peak_index = int(np.argmax(integrated))
    global_peak_velocity = float(velocity[global_peak_index])
    global_peak_power = max(
        float(integrated[global_peak_index]), np.finfo(np.float64).tiny
    )
    target_neighborhood = (
        np.abs(velocity - target_doppler_mps) <= PEAK_BAND_HALF_WIDTH_MPS
    )
    if target_neighborhood.any():
        candidate_indices = np.flatnonzero(target_neighborhood)
        target_local_peak_index = int(
            candidate_indices[int(np.argmax(integrated[candidate_indices]))]
        )
    else:
        # A very small number of CQ08 target records lie outside the frozen
        # [-85, 85] m/s evidence axis. Preserve a finite diagnostic at the
        # nearest boundary; readiness reports these as unaligned.
        target_local_peak_index = int(np.argmin(np.abs(velocity - target_doppler_mps)))
    target_local_peak_velocity = float(velocity[target_local_peak_index])
    target_local_peak_power_ratio_to_global = float(
        np.clip(integrated[target_local_peak_index] / global_peak_power, 0.0, 1.0)
    )
    peak_band = (
        np.abs(velocity - target_local_peak_velocity) <= PEAK_BAND_HALF_WIDTH_MPS
    )
    if not peak_band.any():
        peak_band[target_local_peak_index] = True
    peak_band_series = power[peak_band].sum(axis=0, dtype=np.float64)
    mean_energy = max(float(np.mean(peak_band_series)), np.finfo(np.float64).tiny)
    flicker_cv = float(np.std(peak_band_series, ddof=0) / mean_energy)
    target_peak_power = max(
        float(integrated[target_local_peak_index]), np.finfo(np.float64).tiny
    )
    above_threshold = integrated >= target_peak_power * 0.1
    spread_start = target_local_peak_index
    spread_stop = target_local_peak_index
    while spread_start > 0 and above_threshold[spread_start - 1]:
        spread_start -= 1
    while spread_stop + 1 < integrated.size and above_threshold[spread_stop + 1]:
        spread_stop += 1
    spread_ratio = float(spread_stop - spread_start + 1) / float(
        STFT_VELOCITY_BIN_COUNT
    )
    period_seconds, period_estimable = _modulation_period(peak_band_series, time_step_seconds)
    raw_summary = np.asarray([flicker_cv, spread_ratio, period_seconds], dtype=np.float32)
    model_summary = np.asarray(
        [
            np.clip(flicker_cv / FLICKER_CV_CLIP, 0.0, 1.0),
            np.clip(spread_ratio, 0.0, 1.0),
            np.clip(period_seconds / MODULATION_PERIOD_CLIP_SECONDS, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return MicroDopplerFrame(
        stft_tensor=_canonical_stft_tensor(power),
        raw_summary=raw_summary,
        model_summary=model_summary,
        modulation_period_estimable=period_estimable,
        target_range_index_zero_based=target_range_index,
        target_doppler_index_zero_based=target_doppler_index,
        target_doppler_mps=target_doppler_mps,
        target_local_stft_peak_velocity_mps=target_local_peak_velocity,
        global_stft_peak_velocity_mps=global_peak_velocity,
        target_local_peak_power_ratio_to_global=(
            target_local_peak_power_ratio_to_global
        ),
        stft_native_time_bin_count=int(power.shape[1]),
    )


def empty_micro_doppler_event() -> MicroDopplerEvent:
    return MicroDopplerEvent(
        values=np.zeros((MICRO_DOPPLER_VALUE_DIM,), dtype=np.float32),
        valid=False,
        stft_tensors=np.zeros(
            (STFT_FRAME_CAP, STFT_VELOCITY_BIN_COUNT, STFT_TIME_BIN_COUNT), dtype=np.float16
        ),
        frame_mask=np.zeros((STFT_FRAME_CAP,), dtype=np.bool_),
        raw_frame_indices=np.full((STFT_FRAME_CAP,), -1, dtype=np.int32),
        frame_raw_summaries=np.zeros((STFT_FRAME_CAP, MICRO_DOPPLER_VALUE_DIM), dtype=np.float32),
        frame_period_estimable=np.zeros((STFT_FRAME_CAP,), dtype=np.bool_),
        target_range_indices=np.full((STFT_FRAME_CAP,), -1, dtype=np.int16),
        target_doppler_indices=np.full((STFT_FRAME_CAP,), -1, dtype=np.int32),
        target_doppler_mps=np.zeros((STFT_FRAME_CAP,), dtype=np.float32),
        target_local_stft_peak_velocity_mps=np.zeros(
            (STFT_FRAME_CAP,), dtype=np.float32
        ),
        global_stft_peak_velocity_mps=np.zeros((STFT_FRAME_CAP,), dtype=np.float32),
        target_local_peak_power_ratio_to_global=np.zeros(
            (STFT_FRAME_CAP,), dtype=np.float32
        ),
        native_time_bin_counts=np.zeros((STFT_FRAME_CAP,), dtype=np.int16),
    )


def extract_current_event_micro_doppler(
    frames: Sequence[RawEchoFrame],
    raw_frame_indices: Iterable[int],
    *,
    track_point_index: int,
) -> MicroDopplerEvent:
    indices = [int(value) for value in raw_frame_indices]
    if not indices:
        return empty_micro_doppler_event()
    if len(indices) > STFT_FRAME_CAP:
        raise ValueError(
            f"current event exceeds the frozen {STFT_FRAME_CAP}-frame micro-Doppler cap"
        )
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("current-event raw frame indices must be unique and ordered")
    if min(indices) < 0 or max(indices) >= len(frames):
        raise ValueError("current-event raw frame index is outside the parsed echo")
    extracted = [
        extract_micro_doppler_frame(frames[index], track_point_index=track_point_index)
        for index in indices
    ]
    tensors = np.zeros(
        (STFT_FRAME_CAP, STFT_VELOCITY_BIN_COUNT, STFT_TIME_BIN_COUNT), dtype=np.float16
    )
    mask = np.zeros((STFT_FRAME_CAP,), dtype=np.bool_)
    frame_indices = np.full((STFT_FRAME_CAP,), -1, dtype=np.int32)
    raw_summaries = np.zeros((STFT_FRAME_CAP, MICRO_DOPPLER_VALUE_DIM), dtype=np.float32)
    period_estimable = np.zeros((STFT_FRAME_CAP,), dtype=np.bool_)
    range_indices = np.full((STFT_FRAME_CAP,), -1, dtype=np.int16)
    doppler_indices = np.full((STFT_FRAME_CAP,), -1, dtype=np.int32)
    target_doppler_mps = np.zeros((STFT_FRAME_CAP,), dtype=np.float32)
    target_local_peak_velocity_mps = np.zeros((STFT_FRAME_CAP,), dtype=np.float32)
    global_peak_velocity_mps = np.zeros((STFT_FRAME_CAP,), dtype=np.float32)
    target_local_peak_power_ratio = np.zeros((STFT_FRAME_CAP,), dtype=np.float32)
    time_counts = np.zeros((STFT_FRAME_CAP,), dtype=np.int16)
    for position, (raw_index, value) in enumerate(zip(indices, extracted, strict=True)):
        tensors[position] = value.stft_tensor.astype(np.float16)
        mask[position] = True
        frame_indices[position] = raw_index
        raw_summaries[position] = value.raw_summary
        period_estimable[position] = value.modulation_period_estimable
        range_indices[position] = value.target_range_index_zero_based
        doppler_indices[position] = value.target_doppler_index_zero_based
        target_doppler_mps[position] = value.target_doppler_mps
        target_local_peak_velocity_mps[position] = (
            value.target_local_stft_peak_velocity_mps
        )
        global_peak_velocity_mps[position] = value.global_stft_peak_velocity_mps
        target_local_peak_power_ratio[position] = (
            value.target_local_peak_power_ratio_to_global
        )
        time_counts[position] = value.stft_native_time_bin_count
    values = np.median(
        np.stack([value.model_summary for value in extracted], axis=0),
        axis=0,
    ).astype(np.float32)
    # A missing modulation peak is represented by zero. If at least one frame
    # has an estimable period, aggregate only estimable periods so missingness
    # cannot pull a valid physical period toward zero.
    estimable_periods = [value.model_summary[2] for value in extracted if value.modulation_period_estimable]
    values[2] = np.float32(np.median(estimable_periods)) if estimable_periods else np.float32(0.0)
    return MicroDopplerEvent(
        values=values,
        valid=True,
        stft_tensors=tensors,
        frame_mask=mask,
        raw_frame_indices=frame_indices,
        frame_raw_summaries=raw_summaries,
        frame_period_estimable=period_estimable,
        target_range_indices=range_indices,
        target_doppler_indices=doppler_indices,
        target_doppler_mps=target_doppler_mps,
        target_local_stft_peak_velocity_mps=target_local_peak_velocity_mps,
        global_stft_peak_velocity_mps=global_peak_velocity_mps,
        target_local_peak_power_ratio_to_global=target_local_peak_power_ratio,
        native_time_bin_counts=time_counts,
    )


def load_micro_doppler_cache(
    path: str | Path,
    *,
    expected_sample_ids: Iterable[str] | None = None,
    expected_value_dim: int = MICRO_DOPPLER_VALUE_DIM,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = Path(path)
    preflight_npz_archive(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {"sample_ids", "values", "valid"}
        if set(payload.files) != required:
            raise ValueError(f"micro-Doppler cache fields must be exactly {sorted(required)}")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("micro-Doppler cache sample_ids must be a one-dimensional string array")
    identifiers = [str(value) for value in sample_ids.tolist()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("micro-Doppler cache contains duplicate sample_ids")
    expected_dim = int(expected_value_dim)
    if expected_dim <= 0:
        raise ValueError("micro-Doppler cache expected value dimension must be positive")
    if values.shape != (len(identifiers), expected_dim) or values.dtype != np.float32:
        raise ValueError(
            "micro-Doppler cache values must be float32 with the configured shape: "
            f"expected={[len(identifiers), expected_dim]}, actual={list(values.shape)}"
        )
    if valid.shape != (len(identifiers),) or valid.dtype != np.bool_:
        raise ValueError("micro-Doppler cache valid must be bool with shape [N]")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("micro-Doppler cache values must be finite and bounded to [0,1]")
    if np.any(values[~valid] != 0.0):
        raise ValueError("invalid micro-Doppler cache rows must be exactly zero")
    if expected_sample_ids is not None:
        expected = [str(value) for value in expected_sample_ids]
        if len(expected) != len(set(expected)):
            raise ValueError("expected micro-Doppler sample_ids contain duplicates")
        missing = sorted(set(expected) - set(identifiers))
        extra = sorted(set(identifiers) - set(expected))
        if missing or extra or len(expected) != len(identifiers):
            raise ValueError(
                "micro-Doppler cache sample set does not exactly match the Dataset split: "
                f"missing={missing[:5]}, extra={extra[:5]}, expected={len(expected)}, actual={len(identifiers)}"
            )
    mapping = {sample_id: values[index] for index, sample_id in enumerate(identifiers)}
    valid_mapping = {sample_id: bool(valid[index]) for index, sample_id in enumerate(identifiers)}
    return mapping, {
        "path": str(cache_path),
        "event_count": len(identifiers),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_by_sample_id": valid_mapping,
    }




__all__ = ["FLICKER_CV_CLIP", "MICRO_DOPPLER_MASK_MODE", "MICRO_DOPPLER_NAMES", "MICRO_DOPPLER_SCHEMA", "MICRO_DOPPLER_VALUE_DIM", "MicroDopplerEvent", "MicroDopplerFrame", "RawEchoFrame", "RawTargetRecord", "STFT_FRAME_CAP", "STFT_TIME_BIN_COUNT", "STFT_VELOCITY_BIN_COUNT", "empty_micro_doppler_event", "extract_current_event_micro_doppler", "extract_micro_doppler_frame", "load_micro_doppler_cache"]