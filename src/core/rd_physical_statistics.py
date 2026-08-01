from __future__ import annotations

import math

import torch


PHYSICAL_RD_STAT_DIM = 14
PHYSICAL_RD_STATISTIC_NAMES = (
    "log_relative_power_mean",
    "log_relative_power_std",
    "range_centroid_mean_m",
    "range_centroid_std_m",
    "range_bandwidth_mean_m",
    "velocity_centroid_mean_mps",
    "velocity_centroid_std_mps",
    "velocity_bandwidth_mean_mps",
    "velocity_entropy_mean",
    "velocity_top10_concentration_mean",
    "log_power_lag1_correlation",
    "dynamic_range_mean_db",
    "phase_coherence_mean",
    "phase_coherence_std",
)


def _masked_time_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_time_std(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mean = _masked_time_mean(values, valid)
    weights = valid.to(values.dtype)
    variance = ((values - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return variance.clamp_min(0.0).sqrt()


def _masked_lag_one_correlation(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape[1] < 2:
        return values.new_zeros((values.shape[0],))
    pair_valid = valid[:, 1:] & valid[:, :-1]
    centered = values - _masked_time_mean(values, valid).unsqueeze(1)
    numerator = (centered[:, 1:] * centered[:, :-1] * pair_valid.to(values.dtype)).sum(dim=1)
    denominator = (
        centered[:, 1:].pow(2) * pair_valid.to(values.dtype)
    ).sum(dim=1).sqrt() * (
        centered[:, :-1].pow(2) * pair_valid.to(values.dtype)
    ).sum(dim=1).sqrt()
    return numerator / denominator.clamp_min(1.0e-6)


def physical_rd_statistics(
    raw_amplitude_db: torch.Tensor,
    phase_sin: torch.Tensor,
    phase_cos: torch.Tensor,
    padding_mask: torch.Tensor,
    range_offset_m: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    amplitude_clip_low_db: float,
    amplitude_clip_high_db: float,
) -> torch.Tensor:
    if raw_amplitude_db.ndim != 4:
        raise ValueError("raw physical RD amplitude must be [batch, time, range, doppler]")
    if phase_sin.shape != raw_amplitude_db.shape or phase_cos.shape != raw_amplitude_db.shape:
        raise ValueError("physical RD phase channels must match amplitude shape")
    batch_size, frame_count, range_count, doppler_count = raw_amplitude_db.shape
    if padding_mask.shape != (batch_size, frame_count):
        raise ValueError("physical RD padding mask must match batch/time dimensions")
    if range_offset_m.shape != (range_count,) or velocity_mps.shape != (doppler_count,):
        raise ValueError("physical RD axes must match range/doppler dimensions")
    if amplitude_clip_high_db <= amplitude_clip_low_db:
        raise ValueError("physical RD amplitude clip bounds are invalid")
    if not torch.isfinite(raw_amplitude_db).all() or not torch.isfinite(phase_sin).all() or not torch.isfinite(phase_cos).all():
        raise ValueError("physical RD statistics inputs must be finite")
    valid = ~padding_mask.bool()
    clip = raw_amplitude_db.clamp(min=amplitude_clip_low_db, max=amplitude_clip_high_db)
    relative_power = torch.pow(10.0, (clip - amplitude_clip_high_db) / 10.0)
    relative_power = relative_power * valid.unsqueeze(-1).unsqueeze(-1).to(relative_power.dtype)
    range_profile = relative_power.mean(dim=-1)
    velocity_profile = relative_power.mean(dim=-2)
    range_mass = range_profile.sum(dim=-1).clamp_min(1.0e-8)
    velocity_mass = velocity_profile.sum(dim=-1).clamp_min(1.0e-8)
    range_axis = range_offset_m.to(device=raw_amplitude_db.device, dtype=raw_amplitude_db.dtype).view(1, 1, -1)
    velocity_axis = velocity_mps.to(device=raw_amplitude_db.device, dtype=raw_amplitude_db.dtype).view(1, 1, -1)
    range_centroid = (range_profile * range_axis).sum(dim=-1) / range_mass
    velocity_centroid = (velocity_profile * velocity_axis).sum(dim=-1) / velocity_mass
    range_bandwidth = ((range_profile * (range_axis - range_centroid.unsqueeze(-1)).pow(2)).sum(dim=-1) / range_mass).clamp_min(0.0).sqrt()
    velocity_bandwidth = ((velocity_profile * (velocity_axis - velocity_centroid.unsqueeze(-1)).pow(2)).sum(dim=-1) / velocity_mass).clamp_min(0.0).sqrt()
    velocity_probability = velocity_profile / velocity_mass.unsqueeze(-1)
    velocity_entropy = -(
        velocity_probability * velocity_probability.clamp_min(1.0e-8).log()
    ).sum(dim=-1) / math.log(float(doppler_count))
    top_count = max(1, doppler_count // 10)
    velocity_top_concentration = velocity_profile.topk(top_count, dim=-1).values.sum(dim=-1) / velocity_mass
    frame_power = relative_power.mean(dim=(-1, -2)).clamp_min(1.0e-8)
    log_frame_power = frame_power.log()
    valid_cells = valid.unsqueeze(-1).unsqueeze(-1)
    frame_min = raw_amplitude_db.masked_fill(~valid_cells, float("inf")).amin(dim=(-1, -2))
    frame_max = raw_amplitude_db.masked_fill(~valid_cells, float("-inf")).amax(dim=(-1, -2))
    frame_min = torch.where(valid, frame_min, torch.zeros_like(frame_min))
    frame_max = torch.where(valid, frame_max, torch.zeros_like(frame_max))
    dynamic_range = (frame_max - frame_min).clamp_min(0.0)
    phase_weight = relative_power.sum(dim=(-1, -2))
    phase_sin_mean = (phase_sin * relative_power).sum(dim=(-1, -2)) / phase_weight.clamp_min(1.0e-8)
    phase_cos_mean = (phase_cos * relative_power).sum(dim=(-1, -2)) / phase_weight.clamp_min(1.0e-8)
    phase_coherence = torch.where(
        phase_weight > 1.0e-8,
        torch.hypot(phase_sin_mean, phase_cos_mean),
        torch.zeros_like(phase_weight),
    )
    stats = torch.stack(
        [
            _masked_time_mean(log_frame_power, valid),
            _masked_time_std(log_frame_power, valid),
            _masked_time_mean(range_centroid, valid),
            _masked_time_std(range_centroid, valid),
            _masked_time_mean(range_bandwidth, valid),
            _masked_time_mean(velocity_centroid, valid),
            _masked_time_std(velocity_centroid, valid),
            _masked_time_mean(velocity_bandwidth, valid),
            _masked_time_mean(velocity_entropy, valid),
            _masked_time_mean(velocity_top_concentration, valid),
            _masked_lag_one_correlation(log_frame_power, valid),
            _masked_time_mean(dynamic_range, valid),
            _masked_time_mean(phase_coherence, valid),
            _masked_time_std(phase_coherence, valid),
        ],
        dim=-1,
    )
    has_valid_frame = valid.any(dim=1)
    stats = torch.where(has_valid_frame.unsqueeze(-1), stats, torch.zeros_like(stats))
    if not torch.isfinite(stats).all():
        raise FloatingPointError("physical RD statistics generated non-finite values")
    return stats


__all__ = ["PHYSICAL_RD_STAT_DIM", "PHYSICAL_RD_STATISTIC_NAMES", "physical_rd_statistics"]
