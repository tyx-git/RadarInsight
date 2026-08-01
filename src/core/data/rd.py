from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EncodedRDSequence:
    features: torch.Tensor
    valid_length: int
    patch_features: torch.Tensor | None = None
    patch_valid_length: int = 0
    deep_patch_features: torch.Tensor | None = None
    deep_patch_valid_length: int = 0
    time_seconds: torch.Tensor | None = None
    patch_time_seconds: torch.Tensor | None = None
    deep_patch_time_seconds: torch.Tensor | None = None


MULTIMODAL_RD_COLUMNS = [
    "amplitude_range",
    "phase_range",
    "amplitude_doppler",
    "phase_doppler",
]

MULTIMODAL_RD_RANGE_BINS = 512
MULTIMODAL_RD_DOPPLER_BINS = 64
MULTIMODAL_RD_TIME_BINS = 128
MULTIMODAL_RD_PATCH_RANGE_BINS = 64
MULTIMODAL_RD_PATCH_DOPPLER_BINS = 64
MULTIMODAL_RD_DEEP_PATCH_TIME_BINS = 160

RD_SUMMARY_COLUMNS = [
    "frame_energy_mean",
    "frame_energy_max",
    "frame_energy_min",
]


def _prepare_amplitude(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    high = float(np.percentile(finite, 99.0))
    low = float(np.percentile(finite, 1.0))
    if high > 1.0e4:
        values = np.clip(values, a_min=0.0, a_max=None)
        return np.log1p(values).astype(np.float32)
    values = np.clip(values, a_min=low, a_max=high)
    return values.astype(np.float32)


def _temporal_delta(values: torch.Tensor, time_seconds: torch.Tensor | None = None) -> torch.Tensor:
    delta = torch.zeros_like(values)
    if values.shape[0] > 1:
        differences = values[1:] - values[:-1]
        if time_seconds is None:
            delta[1:] = differences
        else:
            times = time_seconds.to(device=values.device, dtype=values.dtype).reshape(-1)
            if int(times.numel()) != int(values.shape[0]):
                raise ValueError("time_seconds length must match the RD frame count")
            intervals = times[1:] - times[:-1]
            positive = intervals[intervals > 1.0e-6]
            fallback = positive.median() if positive.numel() else intervals.new_tensor(1.0)
            intervals = torch.where(intervals > 1.0e-6, intervals, fallback).clamp_min(1.0e-6)
            view_shape = (intervals.shape[0],) + (1,) * (values.ndim - 1)
            delta[1:] = differences / intervals.reshape(view_shape)
    return delta


def _time_axis(rd_tensor: dict[str, np.ndarray], frame_count: int) -> torch.Tensor:
    for key in ("time_seconds", "time_raw"):
        raw = rd_tensor.get(key)
        if raw is None:
            continue
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.size != frame_count or not np.isfinite(values).all():
            continue
        values = values - values[0]
        if np.all(np.diff(values) >= 0.0):
            return torch.tensor(values, dtype=torch.float32)
    return torch.arange(frame_count, dtype=torch.float32)


def _circular_mean(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.atan2(torch.sin(values).mean(dim=dim), torch.cos(values).mean(dim=dim))


def _circular_interpolate(
    values: torch.Tensor,
    *,
    size: int | tuple[int, ...],
    mode: str,
) -> torch.Tensor:
    sine = F.interpolate(torch.sin(values), size=size, mode=mode, align_corners=False)
    cosine = F.interpolate(torch.cos(values), size=size, mode=mode, align_corners=False)
    return torch.atan2(sine, cosine)


def _standardize_over_time(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (values - mean) / std


def _resample_indices(frame_count: int, target_count: int) -> torch.Tensor:
    if frame_count <= 1:
        return torch.zeros((max(1, target_count),), dtype=torch.long)
    return torch.linspace(0, frame_count - 1, steps=max(1, target_count)).round().long().clamp(0, frame_count - 1)


def _topk_window_indices(scores: torch.Tensor, count: int, frame_count: int, radius: int = 2) -> list[int]:
    if count <= 0 or frame_count <= 0:
        return []
    clean_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    topk = min(max(1, count), int(clean_scores.numel()))
    anchors = torch.topk(clean_scores, k=topk).indices.tolist()
    selected: list[int] = []
    for anchor in anchors:
        start = max(0, int(anchor) - radius)
        end = min(frame_count - 1, int(anchor) + radius)
        selected.extend(range(start, end + 1))
        if len(selected) >= count:
            break
    return selected[:count]


def _hybrid_deep_frame_indices(amplitude_frames: torch.Tensor, target_count: int = MULTIMODAL_RD_DEEP_PATCH_TIME_BINS) -> torch.Tensor:
    if amplitude_frames.ndim == 2:
        frames = amplitude_frames.unsqueeze(-1)
    elif amplitude_frames.ndim == 3:
        frames = amplitude_frames
    else:
        raise ValueError("amplitude_frames must be [frame, range] or [frame, range, doppler]")
    frame_count = int(frames.shape[0])
    if frame_count <= 0:
        return torch.zeros((max(1, target_count),), dtype=torch.long)
    if frame_count <= target_count:
        return torch.arange(frame_count, dtype=torch.long)

    energy = frames.abs().mean(dim=(1, 2))
    if frame_count > 1:
        delta = torch.zeros_like(energy)
        delta[1:] = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2))
    else:
        delta = torch.zeros_like(energy)
    doppler_energy = frames.abs().mean(dim=1).clamp_min(1.0e-6)
    doppler_prob = doppler_energy / doppler_energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    doppler_entropy = -(doppler_prob * doppler_prob.log()).sum(dim=-1)
    mean_energy = energy.mean().clamp_min(1.0e-6)
    roughness = delta / mean_energy

    uniform_count = int(round(target_count * 0.30))
    energy_count = int(round(target_count * 0.25))
    delta_count = int(round(target_count * 0.20))
    entropy_count = int(round(target_count * 0.15))
    context_count = max(0, target_count - uniform_count - energy_count - delta_count - entropy_count)

    selected: list[int] = []
    selected.extend(_resample_indices(frame_count, uniform_count).tolist())
    selected.extend(_topk_window_indices(energy, energy_count, frame_count, radius=2))
    selected.extend(_topk_window_indices(delta, delta_count, frame_count, radius=2))
    selected.extend(_topk_window_indices(doppler_entropy + roughness, entropy_count, frame_count, radius=2))
    selected.extend(torch.topk(-energy, k=min(context_count, frame_count)).indices.tolist())

    if not selected:
        return _resample_indices(frame_count, target_count)
    unique_sorted = sorted(set(int(idx) for idx in selected if 0 <= int(idx) < frame_count))
    if len(unique_sorted) >= target_count:
        positions = _resample_indices(len(unique_sorted), target_count)
        return torch.tensor([unique_sorted[int(pos)] for pos in positions.tolist()], dtype=torch.long)
    filler = _resample_indices(frame_count, target_count - len(unique_sorted)).tolist()
    merged = sorted(unique_sorted + [int(idx) for idx in filler])
    return torch.tensor(merged[:target_count], dtype=torch.long)


def load_rd_summary_frame(data_root: Path | str, summary_csv_path: str) -> pd.DataFrame:
    data_root = Path(data_root)
    csv_path = Path(summary_csv_path)
    if not csv_path.is_absolute():
        csv_path = data_root.parent / csv_path
    return pd.read_csv(csv_path)


def load_rd_tensor(data_root: Path | str, tensor_path: str) -> dict[str, np.ndarray]:
    data_root = Path(data_root)
    npz_path = Path(tensor_path)
    if not npz_path.is_absolute():
        npz_path = data_root.parent / npz_path
    with np.load(npz_path) as handle:
        payload = {
            "amplitude": handle["amplitude"].astype(np.float32),
            "phase": handle["phase"].astype(np.float32),
            "time_raw": handle["time_raw"].astype(np.float32),
        }
        if "velocity_mps" in handle.files:
            payload["velocity_mps"] = handle["velocity_mps"].astype(np.float32)
        if "range_offset_m" in handle.files:
            payload["range_offset_m"] = handle["range_offset_m"].astype(np.float32)
        if "time_seconds" in handle.files:
            payload["time_seconds"] = handle["time_seconds"].astype(np.float32)
        return payload


def encode_rd_summary(frame: pd.DataFrame) -> EncodedRDSequence:
    features = frame[RD_SUMMARY_COLUMNS].copy()
    features = features.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    values = features.to_numpy(dtype="float32")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, a_min=0.0, a_max=None)
    values = np.log1p(values)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    values = (values - mean) / std
    rd_presence = np.ones((values.shape[0], 1), dtype=np.float32)
    values = np.concatenate([values.astype(np.float32), rd_presence], axis=1)
    tensor = torch.tensor(values, dtype=torch.float32)
    patch = torch.zeros(
        (
            int(tensor.shape[0]),
            MULTIMODAL_RD_PATCH_RANGE_BINS,
            MULTIMODAL_RD_PATCH_DOPPLER_BINS,
            3,
        ),
        dtype=torch.float32,
    )
    return EncodedRDSequence(
        features=tensor,
        valid_length=int(tensor.shape[0]),
        patch_features=patch,
        patch_valid_length=int(patch.shape[0]),
        deep_patch_features=patch,
        deep_patch_valid_length=int(patch.shape[0]),
    )


def encode_multimodal_rd(rd_tensor: dict[str, np.ndarray], *, legacy: bool = False) -> EncodedRDSequence:
    if not legacy:
        raise RuntimeError(
            "encode_multimodal_rd is legacy-only and cannot feed the active Physical RD-only model; "
            "use PhysicalRDWindowDataset via the physical model factory"
        )
    amplitude = np.asarray(rd_tensor["amplitude"], dtype=np.float32)
    phase = np.asarray(rd_tensor["phase"], dtype=np.float32)
    amplitude = _prepare_amplitude(amplitude.reshape(amplitude.shape[0], -1)).reshape(amplitude.shape)
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

    if amplitude.ndim == 2 and phase.ndim == 2:
        frame_times = _time_axis(rd_tensor, int(amplitude.shape[1]))
        amplitude_t = torch.tensor(amplitude.T, dtype=torch.float32).unsqueeze(1)
        phase_t = torch.tensor(phase.T, dtype=torch.float32).unsqueeze(1)
        if amplitude_t.shape[-1] != MULTIMODAL_RD_RANGE_BINS:
            amplitude_t = F.interpolate(amplitude_t, size=MULTIMODAL_RD_RANGE_BINS, mode="linear", align_corners=False)
        if phase_t.shape[-1] != MULTIMODAL_RD_RANGE_BINS:
            phase_t = _circular_interpolate(
                phase_t,
                size=MULTIMODAL_RD_RANGE_BINS,
                mode="linear",
            )
        amplitude_profile = _standardize_over_time(amplitude_t.squeeze(1))
        phase_profile = phase_t.squeeze(1) / np.pi
        zeros = torch.zeros_like(amplitude_profile)
        stacked = torch.stack([amplitude_profile, phase_profile, zeros, zeros], dim=-1)
        patch_amp = amplitude_profile.unsqueeze(1)
        patch_phase = phase_t
        patch_amp = F.interpolate(
            patch_amp,
            size=MULTIMODAL_RD_PATCH_RANGE_BINS,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        patch_phase = _circular_interpolate(
            patch_phase,
            size=MULTIMODAL_RD_PATCH_RANGE_BINS,
            mode="linear",
        ).squeeze(1) / np.pi
        patch_amp = patch_amp.unsqueeze(-1).repeat(1, 1, MULTIMODAL_RD_PATCH_DOPPLER_BINS)
        patch_phase = patch_phase.unsqueeze(-1).repeat(1, 1, MULTIMODAL_RD_PATCH_DOPPLER_BINS)
        patch_delta = _temporal_delta(patch_amp, frame_times)
        patch_tensor = torch.stack([patch_amp, patch_phase, patch_delta], dim=-1).to(torch.float32)
        deep_indices = _hybrid_deep_frame_indices(amplitude_profile, MULTIMODAL_RD_DEEP_PATCH_TIME_BINS)
        deep_times = frame_times.index_select(0, deep_indices)
        deep_patch_amp = patch_amp.index_select(0, deep_indices)
        deep_patch_phase = patch_phase.index_select(0, deep_indices)
        deep_patch_delta = _temporal_delta(deep_patch_amp, deep_times)
        deep_patch_tensor = torch.stack([deep_patch_amp, deep_patch_phase, deep_patch_delta], dim=-1).to(torch.float32)
    elif amplitude.ndim == 3 and phase.ndim == 3:
        frame_count, range_bins, doppler_bins = amplitude.shape
        frame_times = _time_axis(rd_tensor, frame_count)
        physical_velocity_axis = np.asarray(rd_tensor.get("velocity_mps", []), dtype=np.float32).reshape(-1)
        target_doppler_bins = (
            int(physical_velocity_axis.size)
            if physical_velocity_axis.size == doppler_bins and physical_velocity_axis.size > 0
            else MULTIMODAL_RD_DOPPLER_BINS
        )
        amplitude_t = torch.tensor(amplitude, dtype=torch.float32).reshape(frame_count, 1, range_bins, doppler_bins)
        phase_t = torch.tensor(phase, dtype=torch.float32).reshape(frame_count, 1, range_bins, doppler_bins)
        amplitude_t = F.interpolate(
            amplitude_t,
            size=(MULTIMODAL_RD_RANGE_BINS, target_doppler_bins),
            mode="bilinear",
            align_corners=False,
        )
        phase_t = _circular_interpolate(
            phase_t,
            size=(MULTIMODAL_RD_RANGE_BINS, target_doppler_bins),
            mode="bilinear",
        )
        amplitude_plane = amplitude_t.squeeze(1)
        phase_plane = phase_t.squeeze(1)
        amp_range = _standardize_over_time(amplitude_plane.mean(dim=-1))
        ph_range = _circular_mean(phase_plane, dim=-1) / np.pi
        amp_dopp = amplitude_plane.mean(dim=1).unsqueeze(1)
        amp_dopp = F.interpolate(
            amp_dopp,
            size=MULTIMODAL_RD_RANGE_BINS,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        amp_dopp = _standardize_over_time(amp_dopp)
        ph_dopp = _circular_mean(phase_plane, dim=1).unsqueeze(1)
        ph_dopp = _circular_interpolate(
            ph_dopp,
            size=MULTIMODAL_RD_RANGE_BINS,
            mode="linear",
        ).squeeze(1) / np.pi
        stacked = torch.stack([amp_range, ph_range, amp_dopp, ph_dopp], dim=-1)
        amp_patch = F.interpolate(
            amplitude_t,
            size=(MULTIMODAL_RD_PATCH_RANGE_BINS, MULTIMODAL_RD_PATCH_DOPPLER_BINS),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        phase_patch = _circular_interpolate(
            phase_t,
            size=(MULTIMODAL_RD_PATCH_RANGE_BINS, MULTIMODAL_RD_PATCH_DOPPLER_BINS),
            mode="bilinear",
        ).squeeze(1)
        amp_patch = _standardize_over_time(amp_patch)
        phase_patch = phase_patch / np.pi
        patch_delta = _temporal_delta(amp_patch, frame_times)
        patch_tensor = torch.stack([amp_patch, phase_patch, patch_delta], dim=-1).to(torch.float32)
        deep_indices = _hybrid_deep_frame_indices(amplitude_plane, MULTIMODAL_RD_DEEP_PATCH_TIME_BINS)
        deep_times = frame_times.index_select(0, deep_indices)
        deep_amp = amplitude_plane.index_select(0, deep_indices)
        deep_phase = phase_plane.index_select(0, deep_indices)
        deep_amp = F.interpolate(
            deep_amp.unsqueeze(1),
            size=(MULTIMODAL_RD_PATCH_RANGE_BINS, MULTIMODAL_RD_PATCH_DOPPLER_BINS),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        deep_phase = _circular_interpolate(
            deep_phase.unsqueeze(1),
            size=(MULTIMODAL_RD_PATCH_RANGE_BINS, MULTIMODAL_RD_PATCH_DOPPLER_BINS),
            mode="bilinear",
        ).squeeze(1)
        deep_amp = _standardize_over_time(deep_amp)
        deep_phase = deep_phase / np.pi
        deep_delta = _temporal_delta(deep_amp, deep_times)
        deep_patch_tensor = torch.stack([deep_amp, deep_phase, deep_delta], dim=-1).to(torch.float32)
    else:
        raise ValueError("rd tensor must be 2D [range_bin, frame] or 3D [frame, range_bin, doppler_bin]")

    tensor = torch.as_tensor(stacked, dtype=torch.float32)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    patch_tensor = torch.nan_to_num(patch_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    deep_patch_tensor = torch.nan_to_num(deep_patch_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return EncodedRDSequence(
        features=tensor,
        valid_length=int(tensor.shape[0]),
        patch_features=patch_tensor,
        patch_valid_length=int(patch_tensor.shape[0]),
        deep_patch_features=deep_patch_tensor,
        deep_patch_valid_length=int(deep_patch_tensor.shape[0]),
        time_seconds=frame_times,
        patch_time_seconds=frame_times,
        deep_patch_time_seconds=deep_times,
    )


__all__ = [
    "EncodedRDSequence",
    "MULTIMODAL_RD_COLUMNS",
    "MULTIMODAL_RD_RANGE_BINS",
    "MULTIMODAL_RD_PATCH_RANGE_BINS",
    "MULTIMODAL_RD_PATCH_DOPPLER_BINS",
    "MULTIMODAL_RD_DEEP_PATCH_TIME_BINS",
    "RD_SUMMARY_COLUMNS",
    "load_rd_summary_frame",
    "load_rd_tensor",
    "encode_rd_summary",
    "encode_multimodal_rd",
]
