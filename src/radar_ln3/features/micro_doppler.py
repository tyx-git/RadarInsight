"""raw-IQ 微多普勒的通用接入边界与 stable8 聚合。

raw-IQ 的文件解析、目标定位和采样参数恢复属于数据源适配器或专用
``RawEchoProvider``。本模块只规定一个可替换的 frame extractor 接口，并
实现经过历史实验验证的 23 维 frame evidence 到 stable8 的因果聚合。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .contracts import (
    FeatureInputError,
    FeatureNormalization,
    FeatureNumericalError,
    FeatureShapeError,
    RawEchoFrame,
    RawEchoProvider,
)


MICRO_DOPPLER_SCHEMA = (
    "raw-iq-broadband-multiscale-micro-doppler-summary-v1"
)
MICRO_DOPPLER_EVIDENCE_SCHEMA = (
    "raw-iq-broadband-multiscale-micro-doppler-evidence-v1"
)
MICRO_DOPPLER_MASK_MODE = "gating_only"
MICRO_DOPPLER_FRAME_CAP = 12

MICRO_DOPPLER_FRAME_NAMES = (
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
MICRO_DOPPLER_FRAME_VALUE_DIM = len(MICRO_DOPPLER_FRAME_NAMES)
STABLE8_INDICES = (1, 2, 3, 7, 11, 13, 14, 22)
STABLE8_RAW_NAMES = tuple(MICRO_DOPPLER_FRAME_NAMES[index] for index in STABLE8_INDICES)
STABLE8_NAMES = tuple(f"median_{name}" for name in STABLE8_RAW_NAMES)
MICRO_DOPPLER_NAMES = STABLE8_NAMES
MICRO_DOPPLER_VALUE_DIM = len(MICRO_DOPPLER_NAMES)
SPEED_OF_LIGHT_MPS = 299_792_458.0
MICRO_DOPPLER_MAX_OFFSET_MPS = 90.0
MICRO_DOPPLER_BODY_SEARCH_HALF_WIDTH_MPS = 8.0
MICRO_DOPPLER_MIN_INNER_OFFSET_MPS = 8.0
MICRO_DOPPLER_RAYLEIGH_EXCLUSION_MULTIPLIER = 1.5
MICRO_DOPPLER_DOPPLER_PROFILE_BIN_COUNT = 181
MICRO_DOPPLER_MODULATION_PROFILE_BIN_COUNT = 64
MICRO_DOPPLER_MODULATION_MIN_HZ = 300.0
MICRO_DOPPLER_MODULATION_MAX_HZ = 2000.0
MICRO_DOPPLER_SNR_CLIP_DB = 30.0
MICRO_DOPPLER_MIN_MODULATION_NYQUIST_HZ = 2000.0

# Modes whose physical velocity and modulation support were calibrated for the
# CQ08 DBB classifier.  Other modes remain available in raw evidence but do
# not contribute to the stable event summary.
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

# broad-band 命名只描述算法默认值，不包含任何 CQ08 文件路径或文件名规则。
BROADBAND_MICRO_DOPPLER_SCHEMA = MICRO_DOPPLER_SCHEMA
BROADBAND_MICRO_DOPPLER_EVIDENCE_SCHEMA = MICRO_DOPPLER_EVIDENCE_SCHEMA
BROADBAND_MICRO_DOPPLER_MASK_MODE = MICRO_DOPPLER_MASK_MODE
BROADBAND_FRAME_CAP = MICRO_DOPPLER_FRAME_CAP
BROADBAND_FRAME_NAMES = MICRO_DOPPLER_FRAME_NAMES
BROADBAND_FRAME_VALUE_DIM = MICRO_DOPPLER_FRAME_VALUE_DIM
BROADBAND_MEDIAN_FRAME_INDICES = STABLE8_INDICES
BROADBAND_Q90_FRAME_INDICES: tuple[int, ...] = ()
BROADBAND_MICRO_DOPPLER_NAMES = MICRO_DOPPLER_NAMES
BROADBAND_MICRO_DOPPLER_VALUE_DIM = MICRO_DOPPLER_VALUE_DIM

if MICRO_DOPPLER_VALUE_DIM != 8:
    raise RuntimeError("stable8 contract 必须包含 8 个值")


class RawIQFrameExtractor(Protocol):
    """把一个 raw-IQ frame 转换成完整 23 维证据。"""

    def __call__(self, frame: RawEchoFrame) -> "MicroDopplerFrameEvidence":
        """
        当前调用只能读取传入 frame。目标定位、MAT/DAT 解析等数据源逻辑
        必须在该 callable 或其 provider 内部完成。
        """


@dataclass(frozen=True)
class BroadbandMicroDopplerConfig:
    """可复用的 broadband 微多普勒物理参数。

    ``candidate_sampling_modes`` 是可选的训练期校准注册表。数据集若没有
    经过采样模式校准，应传 ``None`` 并仅使用物理可观测性门控，不能把另一
    个数据集的模式注册表隐式带入当前数据。
    """

    max_offset_mps: float = MICRO_DOPPLER_MAX_OFFSET_MPS
    body_search_half_width_mps: float = MICRO_DOPPLER_BODY_SEARCH_HALF_WIDTH_MPS
    min_inner_offset_mps: float = MICRO_DOPPLER_MIN_INNER_OFFSET_MPS
    rayleigh_exclusion_multiplier: float = MICRO_DOPPLER_RAYLEIGH_EXCLUSION_MULTIPLIER
    neighbor_range_radius: int = 5
    modulation_min_hz: float = MICRO_DOPPLER_MODULATION_MIN_HZ
    modulation_max_hz: float = MICRO_DOPPLER_MODULATION_MAX_HZ
    snr_clip_db: float = MICRO_DOPPLER_SNR_CLIP_DB
    minimum_modulation_nyquist_hz: float = MICRO_DOPPLER_MIN_MODULATION_NYQUIST_HZ
    candidate_sampling_modes: tuple[tuple[int, int, int], ...] | None = (
        BROADBAND_CANDIDATE_SAMPLING_MODES
    )

    def __post_init__(self) -> None:
        positive = (
            self.max_offset_mps,
            self.body_search_half_width_mps,
            self.min_inner_offset_mps,
            self.rayleigh_exclusion_multiplier,
            self.modulation_min_hz,
            self.modulation_max_hz,
            self.snr_clip_db,
            self.minimum_modulation_nyquist_hz,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise FeatureInputError("broadband 微多普勒物理参数必须是正有限值")
        if self.modulation_min_hz >= self.modulation_max_hz:
            raise FeatureInputError("调制频率下界必须小于上界")
        if int(self.neighbor_range_radius) < 1:
            raise FeatureInputError("neighbor_range_radius 必须为正")
        if self.candidate_sampling_modes is not None:
            modes = tuple(
                tuple(int(component) for component in mode)
                for mode in self.candidate_sampling_modes
            )
            if any(len(mode) != 3 or any(component <= 0 for component in mode) for mode in modes):
                raise FeatureInputError("candidate_sampling_modes 格式无效")
            object.__setattr__(self, "candidate_sampling_modes", modes)


@dataclass(frozen=True)
class MicroDopplerFrameEvidence:
    values: np.ndarray
    feature_reliable: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.shape != (MICRO_DOPPLER_FRAME_VALUE_DIM,) or values.dtype != np.float32:
            raise FeatureShapeError("micro-Doppler frame evidence 必须是 float32 [23]")
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise FeatureNumericalError("micro-Doppler frame evidence 必须有限且位于 [0,1]")


@dataclass(frozen=True)
class MicroDopplerEvent:
    values: np.ndarray
    valid: bool
    frame_mask: np.ndarray
    frame_feature_reliable: np.ndarray
    raw_frame_indices: np.ndarray
    frame_values: np.ndarray
    present: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        frame_mask = np.asarray(self.frame_mask)
        reliable = np.asarray(self.frame_feature_reliable)
        raw_indices = np.asarray(self.raw_frame_indices)
        frame_values = np.asarray(self.frame_values)
        if values.shape != (MICRO_DOPPLER_VALUE_DIM,) or values.dtype != np.float32:
            raise FeatureShapeError("micro-Doppler event values shape/dtype 无效")
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise FeatureNumericalError("micro-Doppler event values 必须有限且位于 [0,1]")
        vector = (MICRO_DOPPLER_FRAME_CAP,)
        if (
            frame_mask.shape != vector
            or reliable.shape != vector
            or raw_indices.shape != vector
            or frame_values.shape != (MICRO_DOPPLER_FRAME_CAP, MICRO_DOPPLER_FRAME_VALUE_DIM)
        ):
            raise FeatureShapeError("micro-Doppler event frame evidence shape 无效")
        if frame_mask.dtype != np.bool_ or reliable.dtype != np.bool_:
            raise FeatureShapeError("micro-Doppler frame mask 必须是 bool")
        if raw_indices.dtype != np.int32 or frame_values.dtype != np.float32:
            raise FeatureShapeError("micro-Doppler frame index/value dtype 无效")
        if np.any(reliable & ~frame_mask):
            raise FeatureInputError("inactive frame 不能是 reliable")
        if not np.isfinite(frame_values).all() or np.any(frame_values < 0.0) or np.any(frame_values > 1.0):
            raise FeatureNumericalError("micro-Doppler frame evidence 包含非法值")
        if np.any(frame_values[~frame_mask] != 0.0):
            raise FeatureInputError("inactive micro-Doppler frame evidence 必须为零")
        if not self.valid and np.any(values != 0.0):
            raise FeatureInputError("invalid micro-Doppler event values 必须为零")
        if self.valid and not reliable.any():
            raise FeatureInputError("valid micro-Doppler event 必须至少有一个 reliable frame")


def _coerce_evidence(value: Any) -> MicroDopplerFrameEvidence:
    if isinstance(value, MicroDopplerFrameEvidence):
        return value
    if isinstance(value, Mapping):
        raw_values = value.get("values")
        reliable = value.get("feature_reliable", value.get("reliable", False))
    else:
        raw_values = value
        reliable = False
    values = np.asarray(raw_values, dtype=np.float32)
    return MicroDopplerFrameEvidence(values, bool(reliable))


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < int(value):
        result <<= 1
    return result


def _sampling_mode(frame: RawEchoFrame) -> tuple[int, int, int]:
    return (
        int(frame.prt_num),
        int(round(float(frame.prt_seconds) * 1.0e9)),
        int(round(float(frame.frequency_hz) / 1.0e6)),
    )


def _wrapped_frequency_difference_hz(
    frequency_hz: np.ndarray,
    reference_frequency_hz: float,
    prf_hz: float,
) -> np.ndarray:
    period = float(prf_hz)
    if not math.isfinite(period) or period <= 0.0:
        raise FeatureInputError("raw-IQ PRF 必须为正有限值")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    return np.mod(
        frequency - float(reference_frequency_hz) + 0.5 * period,
        period,
    ) - 0.5 * period


def _detrended_normalized_envelope(
    signal: np.ndarray,
    *,
    normalization_amplitude: float | None = None,
) -> np.ndarray:
    envelope = np.abs(np.asarray(signal, dtype=np.complex128)).astype(np.float64)
    scale = float(envelope.mean()) if normalization_amplitude is None else float(normalization_amplitude)
    if not math.isfinite(scale) or scale <= 0.0:
        raise FeatureNumericalError("raw-IQ envelope normalization 无效")
    envelope = envelope / max(scale, np.finfo(np.float64).tiny)
    sample_index = np.arange(envelope.size, dtype=np.float64)
    centered_index = sample_index - float(sample_index.mean())
    centered_envelope = envelope - float(envelope.mean())
    denominator = float(np.square(centered_index).sum(dtype=np.float64))
    slope = (
        float(np.sum(centered_index * centered_envelope, dtype=np.float64)) / denominator
        if denominator > 0.0
        else 0.0
    )
    return np.asarray(centered_envelope - slope * centered_index, dtype=np.float64)


def _neighbor_range_noise_spectrum(
    iq: np.ndarray,
    *,
    target_range_index: int,
    window: np.ndarray,
    nfft: int,
    radius: int,
) -> np.ndarray:
    row_count = int(iq.shape[0])
    start = max(0, int(target_range_index) - int(radius))
    stop = min(row_count, int(target_range_index) + int(radius) + 1)
    rows = [index for index in range(start, stop) if index != int(target_range_index)]
    if not rows:
        raise FeatureInputError("raw-IQ noise estimation 没有邻近 range row")
    neighbor = np.asarray(iq[rows], dtype=np.complex128)
    spectra = np.square(
        np.abs(
            np.fft.fftshift(
                np.fft.fft(neighbor * window[None, :], n=nfft, axis=1), axes=1
            )
        ),
        dtype=np.float64,
    )
    noise = np.median(spectra, axis=0) / math.log(2.0)
    if not np.isfinite(noise).all() or np.any(noise < 0.0):
        raise FeatureNumericalError("raw-IQ Doppler noise floor 无效")
    return np.asarray(noise, dtype=np.float64)


def _neighbor_range_envelope_noise(
    iq: np.ndarray,
    *,
    target_range_index: int,
    window: np.ndarray,
    nfft: int,
    radius: int,
    target_envelope_mean: float,
) -> np.ndarray:
    row_count = int(iq.shape[0])
    start = max(0, int(target_range_index) - int(radius))
    stop = min(row_count, int(target_range_index) + int(radius) + 1)
    rows = [index for index in range(start, stop) if index != int(target_range_index)]
    if not rows:
        raise FeatureInputError("raw-IQ modulation noise estimation 没有邻近 range row")
    envelopes = np.stack(
        [
            _detrended_normalized_envelope(
                iq[index], normalization_amplitude=target_envelope_mean
            )
            for index in rows
        ],
        axis=0,
    )
    spectra = np.square(
        np.abs(np.fft.rfft(envelopes * window[None, :], n=nfft, axis=1)),
        dtype=np.float64,
    )
    noise = np.median(spectra, axis=0) / math.log(2.0)
    if not np.isfinite(noise).all() or np.any(noise < 0.0):
        raise FeatureNumericalError("raw-IQ modulation noise floor 无效")
    return np.asarray(noise, dtype=np.float64)


def _normalized_entropy(probability: np.ndarray) -> float:
    selected = np.asarray(probability, dtype=np.float64)
    selected = selected[selected > 0.0]
    if selected.size <= 1:
        return 0.0
    return float(
        -np.sum(selected * np.log(selected), dtype=np.float64)
        / math.log(float(selected.size))
    )


def _resolution_cell_statistics(
    relative_velocity_mps: np.ndarray,
    excess_power: np.ndarray,
    *,
    inner_offset_mps: float,
    rayleigh_velocity_mps: float,
    max_offset_mps: float,
) -> tuple[float, float, float]:
    width = max(2.0, float(rayleigh_velocity_mps))
    edges = np.arange(
        -float(max_offset_mps),
        float(max_offset_mps) + width,
        width,
        dtype=np.float64,
    )
    if edges[-1] < float(max_offset_mps):
        edges = np.append(edges, float(max_offset_mps))
    selected = (
        (np.abs(relative_velocity_mps) >= float(inner_offset_mps))
        & (np.abs(relative_velocity_mps) <= float(max_offset_mps))
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


def _bounded_profile(
    axis: np.ndarray,
    values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
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


def _target_record(frame: RawEchoFrame, event_index: int) -> Any:
    matches = [
        record
        for record in frame.target_records
        if int(record.event_index) == int(event_index)
    ]
    if len(matches) != 1:
        raise FeatureInputError(
            "当前 raw-IQ frame 必须为事件提供唯一 target record："
            f"frame={frame.raw_frame_index}, event={event_index}, matches={len(matches)}"
        )
    return matches[0]


def extract_broadband_micro_doppler_frame(
    frame: RawEchoFrame,
    *,
    event_index: int | None = None,
    track_point_index: int | None = None,
    config: BroadbandMicroDopplerConfig | None = None,
) -> MicroDopplerFrameEvidence:
    """从单个通用 raw-IQ frame 生成完整 23 维 broadband evidence。

    ``RawTargetRecord`` 的 range/doppler bin 必须已经由 provider 转成零基、
    fftshift 后的 canonical 坐标。这样 features 不需要了解任何数据集的
    原始一基索引、MAT 字段或文件名。
    """

    config = BroadbandMicroDopplerConfig() if config is None else config
    if event_index is None:
        event_index = track_point_index
    elif track_point_index is not None and int(event_index) != int(track_point_index):
        raise FeatureInputError("event_index 与 track_point_index 不一致")
    if event_index is None:
        raise FeatureInputError("broadband extractor 必须提供 event_index")
    record = _target_record(frame, event_index)
    iq = np.asarray(frame.iq, dtype=np.complex128)
    range_index = int(record.range_bin)
    doppler_index = int(record.doppler_bin)
    if not 0 <= range_index < iq.shape[0]:
        raise FeatureInputError("raw-IQ target range bin 越界")
    signal = np.asarray(iq[range_index], dtype=np.complex128)
    pulse_count = int(signal.size)
    if pulse_count < 4 or not 0 <= doppler_index < pulse_count:
        raise FeatureInputError("raw-IQ target Doppler bin或 pulse 数量无效")
    pri_seconds = float(frame.prt_seconds)
    prf_hz = 1.0 / pri_seconds
    carrier_hz = float(frame.frequency_hz)
    velocity_scale = SPEED_OF_LIGHT_MPS / (2.0 * carrier_hz)
    native_frequency = np.fft.fftshift(np.fft.fftfreq(pulse_count, d=pri_seconds))
    target_frequency_hz = float(native_frequency[doppler_index])
    target_doppler_mps = target_frequency_hz * velocity_scale

    nfft = max(1024, _next_power_of_two(2 * pulse_count))
    window = np.hanning(pulse_count).astype(np.float64)
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=pri_seconds))
    original_spectrum = np.square(
        np.abs(np.fft.fftshift(np.fft.fft(signal * window, n=nfft))),
        dtype=np.float64,
    )
    target_delta_velocity = _wrapped_frequency_difference_hz(
        frequency_hz, target_frequency_hz, prf_hz
    ) * velocity_scale
    body_candidates = np.abs(target_delta_velocity) <= config.body_search_half_width_mps
    if not body_candidates.any():
        raise FeatureInputError("raw-IQ body search 没有 candidate frequency")
    candidate_indices = np.flatnonzero(body_candidates)
    body_index = int(candidate_indices[int(np.argmax(original_spectrum[candidate_indices]))])
    body_frequency_hz = float(frequency_hz[body_index])
    body_doppler_mps = body_frequency_hz * velocity_scale

    time_seconds = np.arange(pulse_count, dtype=np.float64) * pri_seconds
    body_carrier = np.exp(2j * np.pi * body_frequency_hz * time_seconds)
    body_amplitude = np.mean(signal * np.conj(body_carrier), dtype=np.complex128)
    residual = signal - body_amplitude * body_carrier
    residual_spectrum = np.square(
        np.abs(np.fft.fftshift(np.fft.fft(residual * window, n=nfft))),
        dtype=np.float64,
    )
    relative_velocity = _wrapped_frequency_difference_hz(
        frequency_hz, body_frequency_hz, prf_hz
    ) * velocity_scale
    velocity_nyquist_mps = prf_hz * SPEED_OF_LIGHT_MPS / (4.0 * carrier_hz)
    rayleigh_velocity_mps = SPEED_OF_LIGHT_MPS / (
        2.0 * carrier_hz * pulse_count * pri_seconds
    )
    inner_offset_mps = max(
        config.min_inner_offset_mps,
        config.rayleigh_exclusion_multiplier * rayleigh_velocity_mps,
    )
    sideband_domain = (
        (np.abs(relative_velocity) >= inner_offset_mps)
        & (np.abs(relative_velocity) <= config.max_offset_mps)
    )
    noise_spectrum = _neighbor_range_noise_spectrum(
        iq,
        target_range_index=range_index,
        window=window,
        nfft=nfft,
        radius=config.neighbor_range_radius,
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
        power = float(excess_power[selected].sum(dtype=np.float64)) / (
            float(nfft) * window_energy
        )
        return power / energy_denominator

    directional = [
        fraction(8.0, 30.0, positive=False),
        fraction(8.0, 30.0, positive=True),
        fraction(30.0, 60.0, positive=False),
        fraction(30.0, 60.0, positive=True),
        fraction(60.0, config.max_offset_mps + 1.0e-9, positive=False),
        fraction(60.0, config.max_offset_mps + 1.0e-9, positive=True),
    ]
    total_sideband_fraction = residual_sideband_power / energy_denominator
    sideband_sum = float(excess_power.sum(dtype=np.float64))
    if sideband_sum > 0.0:
        probability = excess_power / sideband_sum
        rms_offset = math.sqrt(
            max(
                0.0,
                float(np.sum(probability * np.square(relative_velocity), dtype=np.float64)),
            )
        )
        order = np.argsort(np.abs(relative_velocity), kind="stable")
        cumulative = np.cumsum(probability[order], dtype=np.float64)
        q90_index = min(int(np.searchsorted(cumulative, 0.90)), len(order) - 1)
        q90_offset = float(np.abs(relative_velocity[order[q90_index]]))
        positive_power = float(
            excess_power[relative_velocity > 0.0].sum(dtype=np.float64)
        )
        negative_power = float(
            excess_power[relative_velocity < 0.0].sum(dtype=np.float64)
        )
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
        max_offset_mps=config.max_offset_mps,
    )
    sideband_noise_total = float(noise_spectrum[sideband_domain].sum(dtype=np.float64))
    sideband_snr_db = 10.0 * math.log10(
        1.0 + sideband_sum / max(sideband_noise_total, np.finfo(np.float64).tiny)
    )

    target_envelope_mean = max(float(np.abs(signal).mean()), np.finfo(np.float64).tiny)
    envelope = _detrended_normalized_envelope(
        signal, normalization_amplitude=target_envelope_mean
    )
    envelope_rms = float(np.std(envelope, ddof=0))
    modulation_frequency_hz = np.fft.rfftfreq(nfft, d=pri_seconds)
    modulation_spectrum = np.square(
        np.abs(np.fft.rfft(envelope * window, n=nfft)), dtype=np.float64
    )
    modulation_domain = (
        (modulation_frequency_hz >= config.modulation_min_hz)
        & (modulation_frequency_hz <= config.modulation_max_hz)
    )
    modulation_noise = _neighbor_range_envelope_noise(
        iq,
        target_range_index=range_index,
        window=window,
        nfft=nfft,
        radius=config.neighbor_range_radius,
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
        modulation_band_power(config.modulation_min_hz, 600.0),
        modulation_band_power(600.0, 1200.0),
        modulation_band_power(1200.0, config.modulation_max_hz + 1.0e-9),
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
    modulation_noise_total = float(modulation_noise[modulation_domain].sum(dtype=np.float64))
    modulation_snr_db = 10.0 * math.log10(
        1.0 + modulation_total / max(modulation_noise_total, np.finfo(np.float64).tiny)
    )
    values = np.asarray(
        [
            *directional,
            total_sideband_fraction,
            rms_offset / config.max_offset_mps,
            q90_offset / config.max_offset_mps,
            symmetry,
            entropy,
            concentration,
            peak_count,
            sideband_snr_db / config.snr_clip_db,
            envelope_rms / 2.0,
            *modulation_amplitudes,
            high_modulation_fraction,
            modulation_centroid / config.modulation_max_hz,
            modulation_concentration,
            modulation_entropy,
            modulation_snr_db / config.snr_clip_db,
        ],
        dtype=np.float32,
    )
    if values.shape != (MICRO_DOPPLER_FRAME_VALUE_DIM,) or not np.isfinite(values).all():
        raise FeatureNumericalError("broadband micro-Doppler frame values 无效")
    values = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    mode = _sampling_mode(frame)
    mode_ok = (
        config.candidate_sampling_modes is None
        or mode in config.candidate_sampling_modes
    )
    reliable = bool(
        mode_ok
        and velocity_nyquist_mps >= config.max_offset_mps
        and 0.5 / pri_seconds >= config.minimum_modulation_nyquist_hz
    )
    metadata = {
        "event_index": int(event_index),
        "target_range_index": range_index,
        "target_doppler_index": doppler_index,
        "target_doppler_mps": float(target_doppler_mps),
        "refined_body_doppler_mps": float(body_doppler_mps),
        "velocity_nyquist_mps": float(velocity_nyquist_mps),
        "rayleigh_velocity_mps": float(rayleigh_velocity_mps),
        "residual_sideband_snr_db": float(sideband_snr_db),
        "envelope_modulation_snr_db": float(modulation_snr_db),
        "sampling_mode": list(mode),
        "sampling_mode_reliable": mode_ok,
        "residual_doppler_profile": _bounded_profile(
            relative_velocity,
            excess_power,
            np.linspace(
                -config.max_offset_mps,
                config.max_offset_mps,
                MICRO_DOPPLER_DOPPLER_PROFILE_BIN_COUNT,
                dtype=np.float64,
            ),
        ),
        "envelope_modulation_profile": _bounded_profile(
            modulation_frequency_hz,
            modulation_excess,
            np.linspace(
                config.modulation_min_hz,
                config.modulation_max_hz,
                MICRO_DOPPLER_MODULATION_PROFILE_BIN_COUNT,
                dtype=np.float64,
            ),
        ),
    }
    return MicroDopplerFrameEvidence(values, reliable, metadata)


def aggregate_stable8_frame_values(
    frame_values: np.ndarray,
    frame_feature_reliable: np.ndarray,
) -> np.ndarray:
    """对 reliable frame 做逐维 median，并只选择稳定的 8 维。"""

    values = np.asarray(frame_values, dtype=np.float32)
    reliable = np.asarray(frame_feature_reliable, dtype=np.bool_)
    if values.ndim != 2 or values.shape[1] != MICRO_DOPPLER_FRAME_VALUE_DIM:
        raise FeatureShapeError("stable8 frame aggregation 需要 [frame,23]")
    if reliable.shape != (values.shape[0],):
        raise FeatureShapeError("stable8 reliable mask shape 无效")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise FeatureNumericalError("stable8 frame values 必须有限且位于 [0,1]")
    if not reliable.any():
        return np.zeros((MICRO_DOPPLER_VALUE_DIM,), dtype=np.float32)
    result = np.median(values[reliable][:, STABLE8_INDICES], axis=0).astype(
        np.float32, copy=False
    )
    if not np.isfinite(result).all():
        raise FeatureNumericalError("stable8 聚合产生非有限值")
    return result


def extract_current_event_micro_doppler(
    frames: Sequence[RawEchoFrame],
    *,
    frame_extractor: RawIQFrameExtractor,
    raw_frame_indices: Sequence[int] | None = None,
    frame_cap: int = MICRO_DOPPLER_FRAME_CAP,
) -> MicroDopplerEvent:
    """通过 provider 注入的 extractor 构建当前事件 stable8 证据。"""

    cap = int(frame_cap)
    if cap != MICRO_DOPPLER_FRAME_CAP:
        raise FeatureInputError(
            f"当前 stable8 contract 的 frame cap 固定为 {MICRO_DOPPLER_FRAME_CAP}"
        )
    if len(frames) == 0:
        return _empty_micro_doppler_event()
    indices = list(range(len(frames))) if raw_frame_indices is None else [int(value) for value in raw_frame_indices]
    if not indices:
        return _empty_micro_doppler_event()
    if len(indices) > cap:
        raise FeatureInputError("当前事件 raw-IQ frame 数超过 stable8 cap")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise FeatureInputError("raw-IQ frame indices 必须有序且唯一")
    if min(indices) < 0 or max(indices) >= len(frames):
        raise FeatureInputError("raw-IQ frame index 越界")

    frame_mask = np.zeros((cap,), dtype=np.bool_)
    reliable = np.zeros((cap,), dtype=np.bool_)
    raw_indices = np.full((cap,), -1, dtype=np.int32)
    frame_values = np.zeros((cap, MICRO_DOPPLER_FRAME_VALUE_DIM), dtype=np.float32)
    for position, index in enumerate(indices):
        evidence = _coerce_evidence(frame_extractor(frames[index]))
        frame_mask[position] = True
        reliable[position] = bool(evidence.feature_reliable)
        raw_indices[position] = int(frames[index].raw_frame_index)
        frame_values[position] = evidence.values
    values = aggregate_stable8_frame_values(
        frame_values[: len(indices)], reliable[: len(indices)]
    )
    return MicroDopplerEvent(
        values=values,
        valid=bool(reliable[: len(indices)].any()),
        frame_mask=frame_mask,
        frame_feature_reliable=reliable,
        raw_frame_indices=raw_indices,
        frame_values=frame_values,
        present=True,
    )


def extract_current_event_broadband_micro_doppler(
    frames: Sequence[RawEchoFrame],
    raw_frame_indices: Sequence[int] | None = None,
    *,
    event_index: int | None = None,
    track_point_index: int | None = None,
    config: BroadbandMicroDopplerConfig | None = None,
) -> MicroDopplerEvent:
    """直接使用通用 broadband extractor 生成当前事件 stable8。"""

    if event_index is None:
        event_index = track_point_index
    elif track_point_index is not None and int(event_index) != int(track_point_index):
        raise FeatureInputError("event_index 与 track_point_index 不一致")
    if event_index is None:
        raise FeatureInputError("broadband event extractor 必须提供 event_index")
    return extract_current_event_micro_doppler(
        frames,
        frame_extractor=lambda frame: extract_broadband_micro_doppler_frame(
            frame, event_index=event_index, config=config
        ),
        raw_frame_indices=raw_frame_indices,
    )


def extract_micro_doppler_from_provider(
    event: Any,
    *,
    provider: RawEchoProvider,
    frame_extractor: RawIQFrameExtractor,
    frame_cap: int = MICRO_DOPPLER_FRAME_CAP,
) -> MicroDopplerEvent:
    """从可替换 provider 读取当前事件并进行 stable8 聚合。"""

    # TODO: 后续由前端/统一日志系统接管 provider 读取耗时和可靠帧统计。
    frames = provider.read_event_frames(event)
    return extract_current_event_micro_doppler(
        frames,
        frame_extractor=frame_extractor,
        frame_cap=frame_cap,
    )


def normalize_stable8(
    values: np.ndarray,
    valid: np.ndarray | bool,
    *,
    median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """应用 train-only parent-equal median/IQR 统计量。"""

    array = np.asarray(values, dtype=np.float32)
    center = np.asarray(median, dtype=np.float32)
    divisor = np.asarray(scale, dtype=np.float32)
    if array.shape[-1] != MICRO_DOPPLER_VALUE_DIM:
        raise FeatureShapeError("stable8 values 宽度必须为 8")
    if center.shape != (MICRO_DOPPLER_VALUE_DIM,) or divisor.shape != center.shape:
        raise FeatureShapeError("stable8 normalization shape 无效")
    if not np.isfinite(center).all() or not np.isfinite(divisor).all() or np.any(divisor <= 0.0):
        raise FeatureInputError("stable8 normalization 统计量无效")
    mask = np.asarray(valid, dtype=np.bool_)
    if mask.shape not in {array.shape[:-1], array.shape}:
        raise FeatureShapeError("stable8 valid shape 无效")
    if mask.shape == array.shape:
        event_valid = mask.all(axis=-1)
    else:
        event_valid = mask
    result = (array - center) / divisor
    result = np.where(event_valid[..., None], result, 0.0).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise FeatureNumericalError("stable8 normalization 产生非有限值")
    return result


def stable8_normalization_hash(
    median: np.ndarray | Sequence[float],
    scale: np.ndarray | Sequence[float],
) -> str:
    """返回与 feature cache descriptor 相同的 stable8 normalization hash。"""

    center = np.asarray(median, dtype=np.float32)
    divisor = np.asarray(scale, dtype=np.float32)
    if (
        center.shape != (len(STABLE8_NAMES),)
        or divisor.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(divisor).all()
        or np.any(divisor <= 0.0)
    ):
        raise FeatureInputError("stable8 normalization 统计量无效")
    payload = {
        "feature_names": list(STABLE8_NAMES),
        "median": center.tolist(),
        "scale": divisor.tolist(),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def fit_parent_equal_median_iqr(
    values: np.ndarray,
    reliable: np.ndarray,
    parent_sample_ids: Sequence[str],
    *,
    feature_names: Sequence[str] = STABLE8_NAMES,
    fit_split: str = "train",
    scale_floor: float = 1.0e-6,
) -> FeatureNormalization:
    """以 parent 等权方式拟合 train-only median/IQR。"""

    array = np.asarray(values, dtype=np.float32)
    mask = np.asarray(reliable, dtype=np.bool_)
    parent_ids = tuple(str(value) for value in parent_sample_ids)
    names = tuple(str(value) for value in feature_names)
    parent_array = np.asarray(parent_sample_ids, dtype=str)
    if array.ndim != 2 or array.shape[1] != len(names):
        raise FeatureShapeError("normalization values 必须是 [row, feature]")
    if mask.shape != (array.shape[0],) or len(parent_array) != array.shape[0]:
        raise FeatureShapeError("normalization rows 未对齐")
    if not np.isfinite(array).all() or scale_floor <= 0.0:
        raise FeatureInputError("normalization 输入无效")
    parent_medians: list[np.ndarray] = []
    for parent_id in sorted(set(parent_array.tolist())):
        parent_mask = parent_array == parent_id
        rows = array[parent_mask]
        row_mask = mask[parent_mask]
        if row_mask.any():
            parent_medians.append(np.median(rows[row_mask], axis=0))
    if not parent_medians:
        raise FeatureInputError("normalization 没有 reliable train parent")
    matrix = np.asarray(parent_medians, dtype=np.float64)
    center = np.median(matrix, axis=0)
    scale = np.maximum(
        np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0),
        float(scale_floor),
    )
    return FeatureNormalization(
        feature_names=names,
        median=center.astype(np.float32),
        scale=scale.astype(np.float32),
        fit_split=fit_split,
        parent_count_used=len(parent_medians),
        scale_floor=float(scale_floor),
        metadata={
            "parent_equal_weighting": True,
            "reliable_row_count": int(mask.sum()),
            "validation_not_used": fit_split == "train",
        },
    )


def _empty_micro_doppler_event() -> MicroDopplerEvent:
    return MicroDopplerEvent(
        values=np.zeros((MICRO_DOPPLER_VALUE_DIM,), dtype=np.float32),
        valid=False,
        frame_mask=np.zeros((MICRO_DOPPLER_FRAME_CAP,), dtype=np.bool_),
        frame_feature_reliable=np.zeros((MICRO_DOPPLER_FRAME_CAP,), dtype=np.bool_),
        raw_frame_indices=np.full((MICRO_DOPPLER_FRAME_CAP,), -1, dtype=np.int32),
        frame_values=np.zeros(
            (MICRO_DOPPLER_FRAME_CAP, MICRO_DOPPLER_FRAME_VALUE_DIM),
            dtype=np.float32,
        ),
        present=False,
    )


def micro_doppler_contract() -> dict[str, Any]:
    return {
        "summary_schema": MICRO_DOPPLER_SCHEMA,
        "evidence_schema": MICRO_DOPPLER_EVIDENCE_SCHEMA,
        "value_dim": MICRO_DOPPLER_VALUE_DIM,
        "value_names": list(MICRO_DOPPLER_NAMES),
        "frame_value_dim": MICRO_DOPPLER_FRAME_VALUE_DIM,
        "frame_value_names": list(MICRO_DOPPLER_FRAME_NAMES),
        "stable8_indices": list(STABLE8_INDICES),
        "frame_cap": MICRO_DOPPLER_FRAME_CAP,
        "event_aggregation": {
            "method": "median_over_reliable_current_event_frames",
            "selected_frame_indices": list(STABLE8_INDICES),
            "q90_frame_indices": list(BROADBAND_Q90_FRAME_INDICES),
            "all_frame_evidence_retained": True,
        },
        "valid_mask_mode": MICRO_DOPPLER_MASK_MODE,
        "invalid_value": [0.0] * MICRO_DOPPLER_VALUE_DIM,
        "causal_scope": "current event raw-IQ frames only",
        "normalization": "train-only parent-equal median/IQR",
        "raw_parser": "injected RawEchoProvider and RawIQFrameExtractor",
    }


__all__ = [
    "MICRO_DOPPLER_EVIDENCE_SCHEMA",
    "MICRO_DOPPLER_FRAME_CAP",
    "MICRO_DOPPLER_FRAME_NAMES",
    "MICRO_DOPPLER_FRAME_VALUE_DIM",
    "MICRO_DOPPLER_MASK_MODE",
    "MICRO_DOPPLER_NAMES",
    "MICRO_DOPPLER_SCHEMA",
    "MICRO_DOPPLER_VALUE_DIM",
    "MICRO_DOPPLER_BODY_SEARCH_HALF_WIDTH_MPS",
    "MICRO_DOPPLER_DOPPLER_PROFILE_BIN_COUNT",
    "MICRO_DOPPLER_MAX_OFFSET_MPS",
    "MICRO_DOPPLER_MIN_INNER_OFFSET_MPS",
    "MICRO_DOPPLER_MIN_MODULATION_NYQUIST_HZ",
    "MICRO_DOPPLER_MODULATION_MAX_HZ",
    "MICRO_DOPPLER_MODULATION_MIN_HZ",
    "MICRO_DOPPLER_MODULATION_PROFILE_BIN_COUNT",
    "MICRO_DOPPLER_RAYLEIGH_EXCLUSION_MULTIPLIER",
    "MICRO_DOPPLER_SNR_CLIP_DB",
    "BROADBAND_FRAME_CAP",
    "BROADBAND_CANDIDATE_SAMPLING_MODES",
    "BROADBAND_FRAME_NAMES",
    "BROADBAND_FRAME_VALUE_DIM",
    "BROADBAND_MEDIAN_FRAME_INDICES",
    "BROADBAND_MICRO_DOPPLER_EVIDENCE_SCHEMA",
    "BROADBAND_MICRO_DOPPLER_MASK_MODE",
    "BROADBAND_MICRO_DOPPLER_NAMES",
    "BROADBAND_MICRO_DOPPLER_SCHEMA",
    "BROADBAND_MICRO_DOPPLER_VALUE_DIM",
    "BROADBAND_Q90_FRAME_INDICES",
    "BroadbandMicroDopplerConfig",
    "MicroDopplerEvent",
    "MicroDopplerFrameEvidence",
    "RawEchoProvider",
    "RawIQFrameExtractor",
    "STABLE8_INDICES",
    "STABLE8_NAMES",
    "STABLE8_RAW_NAMES",
    "aggregate_stable8_frame_values",
    "extract_current_event_micro_doppler",
    "extract_current_event_broadband_micro_doppler",
    "extract_broadband_micro_doppler_frame",
    "extract_micro_doppler_from_provider",
    "fit_parent_equal_median_iqr",
    "micro_doppler_contract",
    "normalize_stable8",
    "stable8_normalization_hash",
]
