from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .micro_doppler import (
    RANGE_BIN_COUNT,
    SPEED_OF_LIGHT_MPS,
    RawEchoFrame,
    _target_location,
    _target_record,
)



# PRT=142 has only four Drone, three Bird and zero Balloon train parents in the
# frozen V16 grouped split.  Its physical resolution is retained in evidence,
# but it is not allowed to create a class contribution.  Candidate modes all
# have at least eight train parents per DBB class; readiness independently
# verifies the frozen support before formal training is allowed.  Reliability
# also requires frame-level velocity/modulation Nyquist support.
# (pulse count, PRI in integer nanoseconds, carrier in integer MHz).  PRTnum
# alone is not a sampling contract: each pulse count occurs with two different
# PRI/carrier combinations in frozen V16.  Treating them as one mode could hide
# the same conditional direction reversal that invalidated V17.
BROADBAND_CANDIDATE_SAMPLING_MODES = (
    (226, 20_000, 15_600),
    (226, 24_000, 15_900),
    (360, 48_000, 15_600),
    (360, 51_000, 15_900),
    (570, 48_000, 15_600),
    (570, 51_000, 15_900),
    (900, 48_000, 15_600),
    (900, 51_000, 15_900),
)

BROADBAND_FRAME_CAP = 12
BROADBAND_MAX_OFFSET_MPS = 90.0
BROADBAND_BODY_SEARCH_HALF_WIDTH_MPS = 8.0
BROADBAND_MIN_INNER_OFFSET_MPS = 8.0
BROADBAND_RAYLEIGH_EXCLUSION_MULTIPLIER = 1.5
BROADBAND_DOPPLER_PROFILE_BIN_COUNT = 181
BROADBAND_MODULATION_PROFILE_BIN_COUNT = 64
BROADBAND_MODULATION_MIN_HZ = 300.0
BROADBAND_MODULATION_MAX_HZ = 2000.0
BROADBAND_SNR_CLIP_DB = 30.0
BROADBAND_MIN_MODULATION_NYQUIST_HZ = 2000.0

BROADBAND_FRAME_NAMES = (
    "residual_negative_8_30mps_energy_fraction",
    "residual_positive_8_30mps_energy_fraction",
    "residual_negative_30_60mps_energy_fraction",
    "residual_positive_30_60mps_energy_fraction",
    "residual_negative_60_90mps_energy_fraction",
    "residual_positive_60_90mps_energy_fraction",
    "residual_total_8_90mps_energy_fraction",
    "residual_rms_offset_mps_over_90",
    "residual_q90_abs_offset_mps_over_90",
    "residual_positive_negative_symmetry",
    "residual_resolution_cell_entropy",
    "residual_top4_resolution_cell_concentration",
    "residual_resolution_cell_peak_count_over_12",
    "residual_sideband_snr_db_over_30",
    "pulse_envelope_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_300_600hz_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_600_1200hz_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_1200_2000hz_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_600_2000hz_fraction",
    "pulse_envelope_modulation_centroid_hz_over_2000",
    "pulse_envelope_modulation_top3_concentration",
    "pulse_envelope_modulation_entropy",
    "pulse_envelope_modulation_snr_db_over_30",
)
BROADBAND_FRAME_VALUE_DIM = len(BROADBAND_FRAME_NAMES)

# Keep the complete 23-dimensional frame evidence above for physical audit, but
# expose only train-only transferable event summaries to the classifier.  Each
# retained descriptor clears the frozen global AUC/effect gate and keeps its
# Drone-minus-Bird effect direction in all eight calibrated pulse/PRI/carrier
# modes.  Potentially useful but mode-reversing descriptors remain evidence-only
# for a future explicitly mode-aware model; they cannot enter this global head.
BROADBAND_MEDIAN_FRAME_INDICES = (
    1,
    2,
    3,
    7,
    11,
    13,
    14,
    22,
)

# The original q90 descriptors either reversed direction by sampling mode or
# were near-duplicates of retained medians.  The complete frame evidence still
# permits future q90/flash research without exposing those unstable columns now.
BROADBAND_Q90_FRAME_INDICES: tuple[int, ...] = ()
BROADBAND_MICRO_DOPPLER_NAMES = tuple(
    f"median_{BROADBAND_FRAME_NAMES[index]}"
    for index in BROADBAND_MEDIAN_FRAME_INDICES
) + tuple(
    f"q90_{BROADBAND_FRAME_NAMES[index]}" for index in BROADBAND_Q90_FRAME_INDICES
)
BROADBAND_MICRO_DOPPLER_VALUE_DIM = len(BROADBAND_MICRO_DOPPLER_NAMES)
BROADBAND_MODE_STABLE_INFORMATIVE_NAMES = (
    "median_residual_positive_8_30mps_energy_fraction",
    "median_residual_negative_30_60mps_energy_fraction",
    "median_residual_positive_30_60mps_energy_fraction",
    "median_residual_rms_offset_mps_over_90",
    "median_residual_q90_abs_offset_mps_over_90",
    "median_residual_top4_resolution_cell_concentration",
    "median_residual_resolution_cell_peak_count_over_12",
    "median_residual_sideband_snr_db_over_30",
    "median_pulse_envelope_rms_clipped_0_2_over_2",
    "median_pulse_envelope_modulation_snr_db_over_30",
)
BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES = (
    "median_residual_q90_abs_offset_mps_over_90",
    "median_residual_resolution_cell_peak_count_over_12",
)

if len(BROADBAND_MEDIAN_FRAME_INDICES) != 8:
    raise RuntimeError("broadband micro-Doppler stable median contract must contain 8 columns")
if BROADBAND_Q90_FRAME_INDICES:
    raise RuntimeError("broadband micro-Doppler stable contract must not expose q90 columns")
if BROADBAND_MICRO_DOPPLER_VALUE_DIM != 8:
    raise RuntimeError("broadband micro-Doppler stable model contract must contain 8 columns")
if set(BROADBAND_MICRO_DOPPLER_NAMES) != (
    set(BROADBAND_MODE_STABLE_INFORMATIVE_NAMES)
    - set(BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES)
):
    raise RuntimeError("broadband stable model columns differ from the train-only selection")

BROADBAND_DOPPLER_PROFILE_AXIS_MPS = np.linspace(
    -BROADBAND_MAX_OFFSET_MPS,
    BROADBAND_MAX_OFFSET_MPS,
    BROADBAND_DOPPLER_PROFILE_BIN_COUNT,
    dtype=np.float64,
)
BROADBAND_MODULATION_PROFILE_AXIS_HZ = np.linspace(
    BROADBAND_MODULATION_MIN_HZ,
    BROADBAND_MODULATION_MAX_HZ,
    BROADBAND_MODULATION_PROFILE_BIN_COUNT,
    dtype=np.float64,
)


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < int(value):
        result <<= 1
    return result


def broadband_sampling_mode(
    prt_num: int,
    prt_seconds: float,
    frequency_hz: float,
) -> tuple[int, int, int]:
    values = (
        int(prt_num),
        int(round(float(prt_seconds) * 1.0e9)),
        int(round(float(frequency_hz) / 1.0e6)),
    )
    if values[0] <= 0 or values[1] <= 0 or values[2] <= 0:
        raise ValueError("broadband micro-Doppler sampling mode is invalid")
    return values


def broadband_sampling_mode_name(mode: tuple[int, int, int]) -> str:
    prt_num, pri_ns, carrier_mhz = (int(value) for value in mode)
    return f"prt_num={prt_num}|pri_ns={pri_ns}|carrier_mhz={carrier_mhz}"


def _wrapped_frequency_difference_hz(
    frequency_hz: np.ndarray,
    reference_frequency_hz: float,
    prf_hz: float,
) -> np.ndarray:
    """Return signed sampled-frequency offsets on ``[-PRF/2, PRF/2)``.

    Slow-time Doppler is periodic at the pulse-repetition frequency.  A rotor
    sideband can cross a Nyquist edge while its offset from the body remains
    unambiguous.  Ordinary subtraction would discard that physically valid
    sideband; the circular difference preserves it exactly once.
    """

    period = float(prf_hz)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("broadband micro-Doppler PRF must be finite and positive")
    values = np.asarray(frequency_hz, dtype=np.float64)
    return np.mod(values - float(reference_frequency_hz) + 0.5 * period, period) - (
        0.5 * period
    )


def _detrended_normalized_envelope(
    signal: np.ndarray,
    *,
    normalization_amplitude: float | None = None,
) -> np.ndarray:
    envelope = np.abs(np.asarray(signal, dtype=np.complex128)).astype(np.float64)
    scale = (
        float(envelope.mean())
        if normalization_amplitude is None
        else float(normalization_amplitude)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("broadband envelope normalization amplitude is invalid")
    envelope = envelope / max(scale, np.finfo(np.float64).tiny)
    sample_index = np.arange(envelope.size, dtype=np.float64)
    centered_index = sample_index - float(sample_index.mean())
    centered_envelope = envelope - float(envelope.mean())
    denominator = float(np.square(centered_index).sum(dtype=np.float64))
    slope = (
        float(
            np.sum(centered_index * centered_envelope, dtype=np.float64)
        )
        / denominator
        if denominator > 0.0
        else 0.0
    )
    return np.asarray(centered_envelope - slope * centered_index, dtype=np.float64)


def _neighbor_range_noise_spectrum(
    frame: RawEchoFrame,
    *,
    target_range_index: int,
    window: np.ndarray,
    nfft: int,
) -> np.ndarray:
    """Estimate a frequency-resolved noise floor outside the target range row.

    Estimating the floor from the target row's 30--90 m/s region would subtract
    the very broadband rotor sidebands that V18 is intended to retain.  The
    local-MTD search already limits the relevant range neighborhood to rows
    10--20; use its non-target rows as independent noise/clutter observations
    and convert their per-frequency median to an exponential-noise mean.
    """

    start = max(0, int(target_range_index) - 5)
    stop = min(RANGE_BIN_COUNT, int(target_range_index) + 6)
    rows = [value for value in range(start, stop) if value != int(target_range_index)]
    if not rows:
        raise ValueError("broadband micro-Doppler noise estimate has no neighbor range row")
    neighbor = np.asarray(frame.iq[rows], dtype=np.complex128)
    spectra = np.square(
        np.abs(
            np.fft.fftshift(
                np.fft.fft(neighbor * window[None, :], n=nfft, axis=1),
                axes=1,
            )
        ),
        dtype=np.float64,
    )
    noise = np.median(spectra, axis=0) / math.log(2.0)
    if noise.shape != (nfft,) or not np.isfinite(noise).all() or np.any(noise < 0.0):
        raise ValueError("broadband micro-Doppler neighbor-range noise floor is invalid")
    return np.asarray(noise, dtype=np.float64)


def _neighbor_range_envelope_modulation_noise(
    frame: RawEchoFrame,
    *,
    target_range_index: int,
    window: np.ndarray,
    nfft: int,
    target_envelope_mean: float,
) -> np.ndarray:
    """Estimate modulation noise without using the target's own broad band."""

    start = max(0, int(target_range_index) - 5)
    stop = min(RANGE_BIN_COUNT, int(target_range_index) + 6)
    rows = [value for value in range(start, stop) if value != int(target_range_index)]
    if not rows:
        raise ValueError("broadband envelope modulation has no neighbor range row")
    envelopes = np.stack(
        [
            _detrended_normalized_envelope(
                frame.iq[row], normalization_amplitude=target_envelope_mean
            )
            for row in rows
        ],
        axis=0,
    )
    spectra = np.square(
        np.abs(np.fft.rfft(envelopes * window[None, :], n=nfft, axis=1)),
        dtype=np.float64,
    )
    noise = np.median(spectra, axis=0) / math.log(2.0)
    if not np.isfinite(noise).all() or np.any(noise < 0.0):
        raise ValueError("broadband envelope modulation noise floor is invalid")
    return np.asarray(noise, dtype=np.float64)


def _bounded_profile(axis: np.ndarray, values: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    order = np.argsort(axis, kind="stable")
    ordered_axis = np.asarray(axis, dtype=np.float64)[order]
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    unique_axis, unique_indices = np.unique(ordered_axis, return_index=True)
    profile = np.interp(
        np.asarray(target_axis, dtype=np.float64),
        unique_axis,
        ordered_values[unique_indices],
        left=0.0,
        right=0.0,
    )
    maximum = float(profile.max(initial=0.0))
    if maximum > 0.0:
        profile = profile / maximum
    return np.asarray(np.clip(profile, 0.0, 1.0), dtype=np.float32)


def _normalized_entropy(probability: np.ndarray) -> float:
    selected = np.asarray(probability, dtype=np.float64)
    selected = selected[selected > 0.0]
    if selected.size <= 1:
        return 0.0
    return float(-np.sum(selected * np.log(selected), dtype=np.float64) / math.log(selected.size))


def _resolution_cell_statistics(
    relative_velocity_mps: np.ndarray,
    excess_power: np.ndarray,
    *,
    inner_offset_mps: float,
    rayleigh_velocity_mps: float,
) -> tuple[float, float, float]:
    width = max(2.0, float(rayleigh_velocity_mps))
    edges = np.arange(
        -BROADBAND_MAX_OFFSET_MPS,
        BROADBAND_MAX_OFFSET_MPS + width,
        width,
        dtype=np.float64,
    )
    if edges[-1] < BROADBAND_MAX_OFFSET_MPS:
        edges = np.append(edges, BROADBAND_MAX_OFFSET_MPS)
    selected = (
        (np.abs(relative_velocity_mps) >= float(inner_offset_mps))
        & (np.abs(relative_velocity_mps) <= BROADBAND_MAX_OFFSET_MPS)
    )
    cell_power, _ = np.histogram(
        relative_velocity_mps[selected], bins=edges, weights=excess_power[selected]
    )
    total = float(cell_power.sum(dtype=np.float64))
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    probability = np.asarray(cell_power / total, dtype=np.float64)
    entropy = _normalized_entropy(probability)
    top_count = min(4, probability.size)
    concentration = float(
        np.partition(probability, -top_count)[-top_count:].sum(dtype=np.float64)
    )
    threshold = max(0.02, 1.0 / max(8.0, float(probability.size) * 4.0))
    peaks = 0
    for index, value in enumerate(probability):
        left = probability[index - 1] if index > 0 else -1.0
        right = probability[index + 1] if index + 1 < probability.size else -1.0
        if value >= threshold and value > left and value >= right:
            peaks += 1
    return entropy, concentration, min(12, peaks) / 12.0


@dataclass(frozen=True)
class BroadbandMicroDopplerFrame:
    values: np.ndarray
    residual_doppler_profile: np.ndarray
    envelope_modulation_profile: np.ndarray
    feature_reliable: bool
    target_range_index_zero_based: int
    target_doppler_index_zero_based: int
    target_doppler_mps: float
    refined_body_doppler_mps: float
    velocity_nyquist_mps: float
    rayleigh_velocity_mps: float
    residual_sideband_snr_db: float
    envelope_modulation_snr_db: float

    def __post_init__(self) -> None:
        if self.values.shape != (BROADBAND_FRAME_VALUE_DIM,) or self.values.dtype != np.float32:
            raise ValueError("broadband micro-Doppler frame values must be float32 with shape [23]")
        if not np.isfinite(self.values).all() or np.any(self.values < 0.0) or np.any(self.values > 1.0):
            raise ValueError("broadband micro-Doppler frame values must be finite in [0,1]")
        expected_profiles = (
            (self.residual_doppler_profile, (BROADBAND_DOPPLER_PROFILE_BIN_COUNT,)),
            (self.envelope_modulation_profile, (BROADBAND_MODULATION_PROFILE_BIN_COUNT,)),
        )
        for profile, shape in expected_profiles:
            if profile.shape != shape or profile.dtype != np.float32:
                raise ValueError("broadband micro-Doppler evidence profile has an invalid shape/dtype")
            if not np.isfinite(profile).all() or np.any(profile < 0.0) or np.any(profile > 1.0):
                raise ValueError("broadband micro-Doppler evidence profile must be finite in [0,1]")
        scalars = (
            self.target_doppler_mps,
            self.refined_body_doppler_mps,
            self.velocity_nyquist_mps,
            self.rayleigh_velocity_mps,
            self.residual_sideband_snr_db,
            self.envelope_modulation_snr_db,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("broadband micro-Doppler frame diagnostics must be finite")
        if self.target_range_index_zero_based < 0 or self.target_range_index_zero_based >= RANGE_BIN_COUNT:
            raise ValueError("broadband micro-Doppler target range index is invalid")


@dataclass(frozen=True)
class BroadbandMicroDopplerEvent:
    values: np.ndarray
    valid: bool
    frame_mask: np.ndarray
    frame_feature_reliable: np.ndarray
    raw_frame_indices: np.ndarray
    frame_values: np.ndarray
    residual_doppler_profiles: np.ndarray
    envelope_modulation_profiles: np.ndarray
    target_range_indices: np.ndarray
    target_doppler_indices: np.ndarray
    target_doppler_mps: np.ndarray
    refined_body_doppler_mps: np.ndarray
    velocity_nyquist_mps: np.ndarray
    rayleigh_velocity_mps: np.ndarray
    residual_sideband_snr_db: np.ndarray
    envelope_modulation_snr_db: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (BROADBAND_MICRO_DOPPLER_VALUE_DIM,) or self.values.dtype != np.float32:
            raise ValueError("broadband micro-Doppler event values have an invalid shape/dtype")
        if not np.isfinite(self.values).all() or np.any(self.values < 0.0) or np.any(self.values > 1.0):
            raise ValueError("broadband micro-Doppler event values must be finite in [0,1]")
        vector = (BROADBAND_FRAME_CAP,)
        if (
            self.frame_mask.shape != vector
            or self.frame_feature_reliable.shape != vector
            or self.raw_frame_indices.shape != vector
            or self.target_range_indices.shape != vector
            or self.target_doppler_indices.shape != vector
            or self.target_doppler_mps.shape != vector
            or self.refined_body_doppler_mps.shape != vector
            or self.velocity_nyquist_mps.shape != vector
            or self.rayleigh_velocity_mps.shape != vector
            or self.residual_sideband_snr_db.shape != vector
            or self.envelope_modulation_snr_db.shape != vector
            or self.frame_values.shape != (BROADBAND_FRAME_CAP, BROADBAND_FRAME_VALUE_DIM)
            or self.residual_doppler_profiles.shape
            != (BROADBAND_FRAME_CAP, BROADBAND_DOPPLER_PROFILE_BIN_COUNT)
            or self.envelope_modulation_profiles.shape
            != (BROADBAND_FRAME_CAP, BROADBAND_MODULATION_PROFILE_BIN_COUNT)
        ):
            raise ValueError("broadband micro-Doppler event evidence shapes are invalid")
        if self.frame_mask.dtype != np.bool_ or self.frame_feature_reliable.dtype != np.bool_:
            raise ValueError("broadband micro-Doppler event masks must be bool")
        if (
            self.raw_frame_indices.dtype != np.int32
            or self.frame_values.dtype != np.float32
            or self.residual_doppler_profiles.dtype != np.float32
            or self.envelope_modulation_profiles.dtype != np.float32
            or self.target_range_indices.dtype != np.int16
            or self.target_doppler_indices.dtype != np.int32
            or self.target_doppler_mps.dtype != np.float32
            or self.refined_body_doppler_mps.dtype != np.float32
            or self.velocity_nyquist_mps.dtype != np.float32
            or self.rayleigh_velocity_mps.dtype != np.float32
            or self.residual_sideband_snr_db.dtype != np.float32
            or self.envelope_modulation_snr_db.dtype != np.float32
        ):
            raise ValueError("broadband micro-Doppler event evidence dtypes are invalid")
        if (
            not np.isfinite(self.frame_values).all()
            or not np.isfinite(self.residual_doppler_profiles).all()
            or not np.isfinite(self.envelope_modulation_profiles).all()
            or not np.isfinite(self.target_doppler_mps).all()
            or not np.isfinite(self.refined_body_doppler_mps).all()
            or not np.isfinite(self.velocity_nyquist_mps).all()
            or not np.isfinite(self.rayleigh_velocity_mps).all()
            or not np.isfinite(self.residual_sideband_snr_db).all()
            or not np.isfinite(self.envelope_modulation_snr_db).all()
        ):
            raise ValueError("broadband micro-Doppler event evidence must be finite")
        if np.any(self.frame_feature_reliable & ~self.frame_mask):
            raise ValueError("inactive broadband frames cannot be feature-reliable")
        if not self.valid and (self.frame_mask.any() or np.any(self.values != 0.0)):
            raise ValueError("invalid broadband micro-Doppler events must be exactly empty/zero")
        if self.valid and not self.frame_mask.any():
            raise ValueError("valid broadband micro-Doppler events require a raw frame")


def extract_broadband_micro_doppler_frame(
    frame: RawEchoFrame,
    *,
    track_point_index: int,
) -> BroadbandMicroDopplerFrame:
    record = _target_record(frame, track_point_index)
    target_range_index, target_doppler_index = _target_location(frame, record)
    signal = np.asarray(frame.iq[target_range_index], dtype=np.complex128)
    pulse_count = int(signal.size)
    pri_seconds = float(frame.prt_seconds)
    prf_hz = 1.0 / pri_seconds
    carrier_hz = float(frame.frequency_hz)
    velocity_scale = SPEED_OF_LIGHT_MPS / (2.0 * carrier_hz)
    native_frequency = np.fft.fftshift(np.fft.fftfreq(pulse_count, d=pri_seconds))
    target_frequency_hz = float(native_frequency[target_doppler_index])
    target_doppler_mps = target_frequency_hz * velocity_scale

    nfft = max(1024, _next_power_of_two(2 * pulse_count))
    window = np.hanning(pulse_count).astype(np.float64)
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=pri_seconds))
    original_spectrum = np.square(
        np.abs(np.fft.fftshift(np.fft.fft(signal * window, n=nfft))), dtype=np.float64
    )
    # Slow-time frequency is periodic at PRF.  Circular distance keeps a target
    # body or sideband continuous when it crosses a Nyquist edge.  Each FFT bin
    # still appears exactly once; only its signed offset from the target/body is
    # represented on the principal interval.
    target_delta_velocity = _wrapped_frequency_difference_hz(
        frequency_hz, target_frequency_hz, prf_hz
    ) * velocity_scale
    body_candidates = np.abs(target_delta_velocity) <= BROADBAND_BODY_SEARCH_HALF_WIDTH_MPS
    if not body_candidates.any():
        raise ValueError("broadband body search has no candidate frequency")
    candidate_indices = np.flatnonzero(body_candidates)
    body_index = int(candidate_indices[int(np.argmax(original_spectrum[candidate_indices]))])
    body_frequency_hz = float(frequency_hz[body_index])
    body_doppler_mps = body_frequency_hz * velocity_scale

    time_seconds = np.arange(pulse_count, dtype=np.float64) * pri_seconds
    body_carrier = np.exp(2j * np.pi * body_frequency_hz * time_seconds)
    body_amplitude = np.mean(signal * np.conj(body_carrier), dtype=np.complex128)
    residual = signal - body_amplitude * body_carrier
    residual_spectrum = np.square(
        np.abs(np.fft.fftshift(np.fft.fft(residual * window, n=nfft))), dtype=np.float64
    )
    relative_velocity = _wrapped_frequency_difference_hz(
        frequency_hz, body_frequency_hz, prf_hz
    ) * velocity_scale
    velocity_nyquist_mps = prf_hz * SPEED_OF_LIGHT_MPS / (4.0 * carrier_hz)
    rayleigh_velocity_mps = SPEED_OF_LIGHT_MPS / (
        2.0 * carrier_hz * pulse_count * pri_seconds
    )
    inner_offset_mps = max(
        BROADBAND_MIN_INNER_OFFSET_MPS,
        BROADBAND_RAYLEIGH_EXCLUSION_MULTIPLIER * rayleigh_velocity_mps,
    )
    sideband_domain = (
        (np.abs(relative_velocity) >= inner_offset_mps)
        & (np.abs(relative_velocity) <= BROADBAND_MAX_OFFSET_MPS)
    )
    noise_spectrum = _neighbor_range_noise_spectrum(
        frame,
        target_range_index=target_range_index,
        window=window,
        nfft=nfft,
    )
    excess_power = np.maximum(residual_spectrum - noise_spectrum, 0.0)
    excess_power[~sideband_domain] = 0.0
    window_energy = max(float(np.square(window).sum()), np.finfo(np.float64).tiny)
    residual_sideband_power = float(excess_power.sum(dtype=np.float64)) / (
        float(nfft) * window_energy
    )
    coherent_body_power = float(np.square(np.abs(body_amplitude)))
    energy_denominator = max(
        coherent_body_power + residual_sideband_power, np.finfo(np.float64).tiny
    )

    def fraction(lower: float, upper: float, *, positive: bool) -> float:
        signed = relative_velocity > 0.0 if positive else relative_velocity < 0.0
        selected = signed & (np.abs(relative_velocity) >= lower) & (
            np.abs(relative_velocity) < upper
        )
        value = float(excess_power[selected].sum(dtype=np.float64)) / (
            float(nfft) * window_energy
        )
        return value / energy_denominator

    directional = [
        fraction(8.0, 30.0, positive=False),
        fraction(8.0, 30.0, positive=True),
        fraction(30.0, 60.0, positive=False),
        fraction(30.0, 60.0, positive=True),
        fraction(60.0, BROADBAND_MAX_OFFSET_MPS + 1.0e-9, positive=False),
        fraction(60.0, BROADBAND_MAX_OFFSET_MPS + 1.0e-9, positive=True),
    ]
    total_sideband_fraction = residual_sideband_power / energy_denominator
    sideband_sum = float(excess_power.sum(dtype=np.float64))
    if sideband_sum > 0.0:
        probability = excess_power / sideband_sum
        rms_offset = math.sqrt(
            max(
                0.0,
                float(
                    np.sum(
                        probability * np.square(relative_velocity), dtype=np.float64
                    )
                ),
            )
        )
        order = np.argsort(np.abs(relative_velocity), kind="stable")
        cumulative = np.cumsum(probability[order], dtype=np.float64)
        q90_index = min(int(np.searchsorted(cumulative, 0.90)), len(order) - 1)
        q90_offset = float(np.abs(relative_velocity[order[q90_index]]))
        positive_power = float(excess_power[relative_velocity > 0.0].sum(dtype=np.float64))
        negative_power = float(excess_power[relative_velocity < 0.0].sum(dtype=np.float64))
        symmetry = 2.0 * min(positive_power, negative_power) / max(
            positive_power + negative_power, np.finfo(np.float64).tiny
        )
    else:
        rms_offset = q90_offset = symmetry = 0.0
    entropy, concentration, peak_count = _resolution_cell_statistics(
        relative_velocity,
        excess_power,
        inner_offset_mps=inner_offset_mps,
        rayleigh_velocity_mps=rayleigh_velocity_mps,
    )
    sideband_noise_total = float(noise_spectrum[sideband_domain].sum(dtype=np.float64))
    sideband_snr_db = 10.0 * math.log10(
        1.0 + sideband_sum / max(sideband_noise_total, np.finfo(np.float64).tiny)
    )

    target_envelope_mean = max(
        float(np.abs(signal).mean()), np.finfo(np.float64).tiny
    )
    envelope = _detrended_normalized_envelope(
        signal, normalization_amplitude=target_envelope_mean
    )
    envelope_rms = float(np.std(envelope, ddof=0))
    modulation_frequency_hz = np.fft.rfftfreq(nfft, d=pri_seconds)
    modulation_spectrum = np.square(
        np.abs(np.fft.rfft(envelope * window, n=nfft)), dtype=np.float64
    )
    modulation_domain = (
        (modulation_frequency_hz >= BROADBAND_MODULATION_MIN_HZ)
        & (modulation_frequency_hz <= BROADBAND_MODULATION_MAX_HZ)
    )
    modulation_noise = _neighbor_range_envelope_modulation_noise(
        frame,
        target_range_index=target_range_index,
        window=window,
        nfft=nfft,
        target_envelope_mean=target_envelope_mean,
    )
    modulation_excess = np.maximum(modulation_spectrum - modulation_noise, 0.0)
    modulation_excess[~modulation_domain] = 0.0
    modulation_total = float(modulation_excess.sum(dtype=np.float64))

    def modulation_band_power(lower_hz: float, upper_hz: float) -> float:
        selected = (modulation_frequency_hz >= lower_hz) & (
            modulation_frequency_hz < upper_hz
        )
        return 2.0 * float(modulation_excess[selected].sum(dtype=np.float64)) / (
            float(nfft) * window_energy
        )

    modulation_band_powers = (
        modulation_band_power(300.0, 600.0),
        modulation_band_power(600.0, 1200.0),
        modulation_band_power(1200.0, BROADBAND_MODULATION_MAX_HZ + 1.0e-9),
    )
    modulation_amplitudes = [
        float(np.clip(math.sqrt(max(0.0, value)) / 2.0, 0.0, 1.0))
        for value in modulation_band_powers
    ]
    if modulation_total > 0.0:
        modulation_probability = modulation_excess[modulation_domain] / modulation_total
        modulation_axis = modulation_frequency_hz[modulation_domain]
        modulation_centroid = float(
            np.sum(modulation_probability * modulation_axis, dtype=np.float64)
        )
        top_count = min(3, modulation_probability.size)
        modulation_concentration = float(
            np.partition(modulation_probability, -top_count)[-top_count:].sum(
                dtype=np.float64
            )
        )
        modulation_entropy = _normalized_entropy(modulation_probability)
    else:
        modulation_centroid = modulation_concentration = modulation_entropy = 0.0
    modulation_band_total = sum(modulation_band_powers)
    high_modulation_fraction = (
        (modulation_band_powers[1] + modulation_band_powers[2])
        / max(modulation_band_total, np.finfo(np.float64).tiny)
    )
    modulation_noise_total = float(
        modulation_noise[modulation_domain].sum(dtype=np.float64)
    )
    modulation_snr_db = 10.0 * math.log10(
        1.0
        + modulation_total
        / max(modulation_noise_total, np.finfo(np.float64).tiny)
    )

    values = np.asarray(
        [
            *directional,
            total_sideband_fraction,
            rms_offset / BROADBAND_MAX_OFFSET_MPS,
            q90_offset / BROADBAND_MAX_OFFSET_MPS,
            symmetry,
            entropy,
            concentration,
            peak_count,
            sideband_snr_db / BROADBAND_SNR_CLIP_DB,
            envelope_rms / 2.0,
            *modulation_amplitudes,
            high_modulation_fraction,
            modulation_centroid / BROADBAND_MODULATION_MAX_HZ,
            modulation_concentration,
            modulation_entropy,
            modulation_snr_db / BROADBAND_SNR_CLIP_DB,
        ],
        dtype=np.float32,
    )
    values = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    return BroadbandMicroDopplerFrame(
        values=values,
        residual_doppler_profile=_bounded_profile(
            relative_velocity, excess_power, BROADBAND_DOPPLER_PROFILE_AXIS_MPS
        ),
        envelope_modulation_profile=_bounded_profile(
            modulation_frequency_hz,
            modulation_excess,
            BROADBAND_MODULATION_PROFILE_AXIS_HZ,
        ),
        feature_reliable=(
            broadband_sampling_mode(
                frame.prt_num, frame.prt_seconds, frame.frequency_hz
            )
            in BROADBAND_CANDIDATE_SAMPLING_MODES
            and velocity_nyquist_mps >= BROADBAND_MAX_OFFSET_MPS
            and 0.5 / pri_seconds >= BROADBAND_MIN_MODULATION_NYQUIST_HZ
        ),
        target_range_index_zero_based=target_range_index,
        target_doppler_index_zero_based=target_doppler_index,
        target_doppler_mps=float(target_doppler_mps),
        refined_body_doppler_mps=float(body_doppler_mps),
        velocity_nyquist_mps=float(velocity_nyquist_mps),
        rayleigh_velocity_mps=float(rayleigh_velocity_mps),
        residual_sideband_snr_db=float(sideband_snr_db),
        envelope_modulation_snr_db=float(modulation_snr_db),
    )


def _empty_event() -> BroadbandMicroDopplerEvent:
    frame_vector = (BROADBAND_FRAME_CAP,)
    return BroadbandMicroDopplerEvent(
        values=np.zeros((BROADBAND_MICRO_DOPPLER_VALUE_DIM,), dtype=np.float32),
        valid=False,
        frame_mask=np.zeros(frame_vector, dtype=np.bool_),
        frame_feature_reliable=np.zeros(frame_vector, dtype=np.bool_),
        raw_frame_indices=np.full(frame_vector, -1, dtype=np.int32),
        frame_values=np.zeros(
            (BROADBAND_FRAME_CAP, BROADBAND_FRAME_VALUE_DIM), dtype=np.float32
        ),
        residual_doppler_profiles=np.zeros(
            (BROADBAND_FRAME_CAP, BROADBAND_DOPPLER_PROFILE_BIN_COUNT),
            dtype=np.float32,
        ),
        envelope_modulation_profiles=np.zeros(
            (BROADBAND_FRAME_CAP, BROADBAND_MODULATION_PROFILE_BIN_COUNT),
            dtype=np.float32,
        ),
        target_range_indices=np.full(frame_vector, -1, dtype=np.int16),
        target_doppler_indices=np.full(frame_vector, -1, dtype=np.int32),
        target_doppler_mps=np.zeros(frame_vector, dtype=np.float32),
        refined_body_doppler_mps=np.zeros(frame_vector, dtype=np.float32),
        velocity_nyquist_mps=np.zeros(frame_vector, dtype=np.float32),
        rayleigh_velocity_mps=np.zeros(frame_vector, dtype=np.float32),
        residual_sideband_snr_db=np.zeros(frame_vector, dtype=np.float32),
        envelope_modulation_snr_db=np.zeros(frame_vector, dtype=np.float32),
    )


def aggregate_broadband_frame_values(
    frame_values: np.ndarray,
    frame_feature_reliable: np.ndarray,
) -> np.ndarray:
    values = np.asarray(frame_values, dtype=np.float32)
    reliable = np.asarray(frame_feature_reliable, dtype=bool)
    if values.ndim != 2 or values.shape[1] != BROADBAND_FRAME_VALUE_DIM:
        raise ValueError("broadband frame aggregation requires shape [frames,23]")
    if reliable.shape != (values.shape[0],):
        raise ValueError("broadband frame reliability mask has an invalid shape")
    if not reliable.any():
        return np.zeros((BROADBAND_MICRO_DOPPLER_VALUE_DIM,), dtype=np.float32)
    selected = values[reliable]
    medians = np.median(selected[:, BROADBAND_MEDIAN_FRAME_INDICES], axis=0)
    q90 = (
        np.quantile(selected[:, BROADBAND_Q90_FRAME_INDICES], 0.90, axis=0)
        if BROADBAND_Q90_FRAME_INDICES
        else np.zeros((0,), dtype=np.float32)
    )
    result = np.concatenate([medians, q90]).astype(np.float32, copy=False)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def extract_current_event_broadband_micro_doppler(
    frames: Sequence[RawEchoFrame],
    raw_frame_indices: Iterable[int],
    *,
    track_point_index: int,
) -> BroadbandMicroDopplerEvent:
    indices = [int(value) for value in raw_frame_indices]
    if not indices:
        return _empty_event()
    if len(indices) > BROADBAND_FRAME_CAP:
        raise ValueError("broadband micro-Doppler event exceeds the frozen raw-frame cap")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("broadband raw frame indices must be ordered and unique")
    if min(indices) < 0 or max(indices) >= len(frames):
        raise ValueError("broadband raw frame index is outside the parsed echo")
    extracted = [
        extract_broadband_micro_doppler_frame(
            frames[index], track_point_index=track_point_index
        )
        for index in indices
    ]
    count = len(extracted)
    mask = np.zeros((BROADBAND_FRAME_CAP,), dtype=np.bool_)
    reliable = np.zeros((BROADBAND_FRAME_CAP,), dtype=np.bool_)
    raw_indices = np.full((BROADBAND_FRAME_CAP,), -1, dtype=np.int32)
    frame_values = np.zeros(
        (BROADBAND_FRAME_CAP, BROADBAND_FRAME_VALUE_DIM), dtype=np.float32
    )
    doppler_profiles = np.zeros(
        (BROADBAND_FRAME_CAP, BROADBAND_DOPPLER_PROFILE_BIN_COUNT), dtype=np.float32
    )
    modulation_profiles = np.zeros(
        (BROADBAND_FRAME_CAP, BROADBAND_MODULATION_PROFILE_BIN_COUNT), dtype=np.float32
    )
    target_range = np.full((BROADBAND_FRAME_CAP,), -1, dtype=np.int16)
    target_doppler_index = np.full((BROADBAND_FRAME_CAP,), -1, dtype=np.int32)
    float_fields = [np.zeros((BROADBAND_FRAME_CAP,), dtype=np.float32) for _ in range(6)]
    for position, (raw_index, value) in enumerate(zip(indices, extracted, strict=True)):
        mask[position] = True
        reliable[position] = bool(value.feature_reliable)
        raw_indices[position] = raw_index
        frame_values[position] = value.values
        doppler_profiles[position] = value.residual_doppler_profile
        modulation_profiles[position] = value.envelope_modulation_profile
        target_range[position] = value.target_range_index_zero_based
        target_doppler_index[position] = value.target_doppler_index_zero_based
        for field, scalar in zip(
            float_fields,
            (
                value.target_doppler_mps,
                value.refined_body_doppler_mps,
                value.velocity_nyquist_mps,
                value.rayleigh_velocity_mps,
                value.residual_sideband_snr_db,
                value.envelope_modulation_snr_db,
            ),
            strict=True,
        ):
            field[position] = np.float32(scalar)
    return BroadbandMicroDopplerEvent(
        values=aggregate_broadband_frame_values(
            frame_values[:count], reliable[:count]
        ),
        valid=True,
        frame_mask=mask,
        frame_feature_reliable=reliable,
        raw_frame_indices=raw_indices,
        frame_values=frame_values,
        residual_doppler_profiles=doppler_profiles,
        envelope_modulation_profiles=modulation_profiles,
        target_range_indices=target_range,
        target_doppler_indices=target_doppler_index,
        target_doppler_mps=float_fields[0],
        refined_body_doppler_mps=float_fields[1],
        velocity_nyquist_mps=float_fields[2],
        rayleigh_velocity_mps=float_fields[3],
        residual_sideband_snr_db=float_fields[4],
        envelope_modulation_snr_db=float_fields[5],
    )




__all__ = ["BROADBAND_CANDIDATE_SAMPLING_MODES", "BROADBAND_DOPPLER_PROFILE_AXIS_MPS", "BROADBAND_DOPPLER_PROFILE_BIN_COUNT", "BROADBAND_FRAME_CAP", "BROADBAND_FRAME_NAMES", "BROADBAND_FRAME_VALUE_DIM", "BROADBAND_MEDIAN_FRAME_INDICES", "BROADBAND_MICRO_DOPPLER_NAMES", "BROADBAND_MICRO_DOPPLER_VALUE_DIM", "BROADBAND_MODE_STABLE_INFORMATIVE_NAMES", "BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES", "BROADBAND_MODULATION_PROFILE_AXIS_HZ", "BROADBAND_MODULATION_PROFILE_BIN_COUNT", "BROADBAND_MIN_MODULATION_NYQUIST_HZ", "BROADBAND_Q90_FRAME_INDICES", "BroadbandMicroDopplerEvent", "BroadbandMicroDopplerFrame", "aggregate_broadband_frame_values", "broadband_sampling_mode", "broadband_sampling_mode_name", "extract_broadband_micro_doppler_frame", "extract_current_event_broadband_micro_doppler"]