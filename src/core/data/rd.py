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


def _temporal_delta(values: torch.Tensor) -> torch.Tensor:
    delta = torch.zeros_like(values)
    if values.shape[0] > 1:
        delta[1:] = values[1:] - values[:-1]
    return delta


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
        return {
            "amplitude": handle["amplitude"].astype(np.float32),
            "phase": handle["phase"].astype(np.float32),
            "time_raw": handle["time_raw"].astype(np.float32),
        }


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
    )


def encode_multimodal_rd(rd_tensor: dict[str, np.ndarray]) -> EncodedRDSequence:
    amplitude = np.asarray(rd_tensor["amplitude"], dtype=np.float32)
    phase = np.asarray(rd_tensor["phase"], dtype=np.float32)
    amplitude = _prepare_amplitude(amplitude)
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

    if amplitude.ndim == 2 and phase.ndim == 2:
        amplitude_t = torch.tensor(amplitude.T, dtype=torch.float32).unsqueeze(1)
        phase_t = torch.tensor(phase.T, dtype=torch.float32).unsqueeze(1)
        if amplitude_t.shape[-1] != MULTIMODAL_RD_RANGE_BINS:
            amplitude_t = F.interpolate(amplitude_t, size=MULTIMODAL_RD_RANGE_BINS, mode="linear", align_corners=False)
        if phase_t.shape[-1] != MULTIMODAL_RD_RANGE_BINS:
            phase_t = F.interpolate(phase_t, size=MULTIMODAL_RD_RANGE_BINS, mode="linear", align_corners=False)
        amplitude = amplitude_t.squeeze(1).cpu().numpy()
        phase = phase_t.squeeze(1).cpu().numpy()
        amp_mean = amplitude.mean(axis=0, keepdims=True)
        amp_std = amplitude.std(axis=0, keepdims=True)
        amp_std = np.where(amp_std < 1e-6, 1.0, amp_std)
        amplitude = (amplitude - amp_mean) / amp_std
        phase = phase / np.pi
        zeros = np.zeros_like(amplitude, dtype=np.float32)
        stacked = np.stack([amplitude, phase, zeros, zeros], axis=-1)
        patch_amp = torch.tensor(amplitude, dtype=torch.float32).unsqueeze(1)
        patch_phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(1)
        patch_amp = F.interpolate(
            patch_amp,
            size=MULTIMODAL_RD_PATCH_RANGE_BINS,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        patch_phase = F.interpolate(
            patch_phase,
            size=MULTIMODAL_RD_PATCH_RANGE_BINS,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        patch_amp = patch_amp.unsqueeze(-1).repeat(1, 1, MULTIMODAL_RD_PATCH_DOPPLER_BINS)
        patch_phase = patch_phase.unsqueeze(-1).repeat(1, 1, MULTIMODAL_RD_PATCH_DOPPLER_BINS)
        patch_delta = _temporal_delta(patch_amp)
        patch_tensor = torch.stack([patch_amp, patch_phase, patch_delta], dim=-1).to(torch.float32)
    elif amplitude.ndim == 3 and phase.ndim == 3:
        frame_count, range_bins, doppler_bins = amplitude.shape
        amplitude_t = torch.tensor(amplitude, dtype=torch.float32).reshape(frame_count, 1, range_bins, doppler_bins)
        phase_t = torch.tensor(phase, dtype=torch.float32).reshape(frame_count, 1, range_bins, doppler_bins)
        amplitude_t = F.interpolate(
            amplitude_t,
            size=(MULTIMODAL_RD_RANGE_BINS, MULTIMODAL_RD_DOPPLER_BINS),
            mode="bilinear",
            align_corners=False,
        )
        phase_t = F.interpolate(
            phase_t,
            size=(MULTIMODAL_RD_RANGE_BINS, MULTIMODAL_RD_DOPPLER_BINS),
            mode="bilinear",
            align_corners=False,
        )
        amp_range = amplitude_t.squeeze(1).mean(dim=-1).transpose(0, 1).unsqueeze(0)
        ph_range = phase_t.squeeze(1).mean(dim=-1).transpose(0, 1).unsqueeze(0)
        amp_dopp = amplitude_t.squeeze(1).mean(dim=1).transpose(0, 1).unsqueeze(0)
        ph_dopp = phase_t.squeeze(1).mean(dim=1).transpose(0, 1).unsqueeze(0)
        amp_range = F.interpolate(amp_range, size=MULTIMODAL_RD_TIME_BINS, mode="linear", align_corners=False).squeeze(0).transpose(0, 1).cpu().numpy()
        ph_range = F.interpolate(ph_range, size=MULTIMODAL_RD_TIME_BINS, mode="linear", align_corners=False).squeeze(0).transpose(0, 1).cpu().numpy()
        amp_dopp = F.interpolate(amp_dopp, size=MULTIMODAL_RD_TIME_BINS, mode="linear", align_corners=False).squeeze(0).transpose(0, 1).cpu().numpy()
        ph_dopp = F.interpolate(ph_dopp, size=MULTIMODAL_RD_TIME_BINS, mode="linear", align_corners=False).squeeze(0).transpose(0, 1).cpu().numpy()
        amp_range_mean = amp_range.mean(axis=0, keepdims=True)
        amp_range_std = np.where(amp_range.std(axis=0, keepdims=True) < 1e-6, 1.0, amp_range.std(axis=0, keepdims=True))
        amp_dopp_mean = amp_dopp.mean(axis=0, keepdims=True)
        amp_dopp_std = np.where(amp_dopp.std(axis=0, keepdims=True) < 1e-6, 1.0, amp_dopp.std(axis=0, keepdims=True))
        amp_range = (amp_range - amp_range_mean) / amp_range_std
        amp_dopp = (amp_dopp - amp_dopp_mean) / amp_dopp_std
        ph_range = ph_range / np.pi
        ph_dopp = ph_dopp / np.pi
        if amp_dopp.shape[1] != MULTIMODAL_RD_RANGE_BINS:
            amp_dopp_t = torch.tensor(amp_dopp[:, None, :], dtype=torch.float32)
            ph_dopp_t = torch.tensor(ph_dopp[:, None, :], dtype=torch.float32)
            amp_dopp = F.interpolate(amp_dopp_t, size=MULTIMODAL_RD_RANGE_BINS, mode="linear", align_corners=False).squeeze(1).cpu().numpy()
            ph_dopp = F.interpolate(ph_dopp_t, size=MULTIMODAL_RD_RANGE_BINS, mode="linear", align_corners=False).squeeze(1).cpu().numpy()
        stacked = np.stack([amp_range, ph_range, amp_dopp, ph_dopp], axis=-1)
        amp_volume = amplitude_t.permute(1, 0, 2, 3).unsqueeze(0)
        phase_volume = phase_t.permute(1, 0, 2, 3).unsqueeze(0)
        amp_patch = F.interpolate(
            amp_volume,
            size=(
                MULTIMODAL_RD_TIME_BINS,
                MULTIMODAL_RD_PATCH_RANGE_BINS,
                MULTIMODAL_RD_PATCH_DOPPLER_BINS,
            ),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        phase_patch = F.interpolate(
            phase_volume,
            size=(
                MULTIMODAL_RD_TIME_BINS,
                MULTIMODAL_RD_PATCH_RANGE_BINS,
                MULTIMODAL_RD_PATCH_DOPPLER_BINS,
            ),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        amp_patch_mean = amp_patch.mean()
        amp_patch_std = amp_patch.std().clamp_min(1e-6)
        amp_patch = (amp_patch - amp_patch_mean) / amp_patch_std
        phase_patch = phase_patch / np.pi
        patch_delta = _temporal_delta(amp_patch)
        patch_tensor = torch.stack([amp_patch, phase_patch, patch_delta], dim=-1).to(torch.float32)
    else:
        raise ValueError("rd tensor must be 2D [range_bin, frame] or 3D [frame, range_bin, doppler_bin]")

    tensor = torch.tensor(stacked, dtype=torch.float32)
    patch_tensor = torch.nan_to_num(patch_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return EncodedRDSequence(
        features=tensor,
        valid_length=int(tensor.shape[0]),
        patch_features=patch_tensor,
        patch_valid_length=int(patch_tensor.shape[0]),
    )


__all__ = [
    "EncodedRDSequence",
    "MULTIMODAL_RD_COLUMNS",
    "MULTIMODAL_RD_RANGE_BINS",
    "MULTIMODAL_RD_PATCH_RANGE_BINS",
    "MULTIMODAL_RD_PATCH_DOPPLER_BINS",
    "RD_SUMMARY_COLUMNS",
    "load_rd_summary_frame",
    "load_rd_tensor",
    "encode_rd_summary",
    "encode_multimodal_rd",
]
