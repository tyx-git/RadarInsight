from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data.rd_physical_window_dataset import PHYSICAL_RD_INPUT_CHANNELS, QUALITY_FEATURE_DIM
from .rd_physical_statistics import PHYSICAL_RD_STAT_DIM, physical_rd_statistics


MICRO_TIME_BINS = 8
MICRO_VELOCITY_BINS = 32
SPATIAL_POOL_RANGE_BINS = 6
SPATIAL_POOL_DOPPLER_BINS = 16
TEMPORAL_POOLING_VERSION = "gru_hidden_mean_max_attention_v1"


@dataclass(frozen=True)
class PhysicalRDDroneBirdOutput:
    logits: torch.Tensor
    frame_tokens: torch.Tensor
    statistics: torch.Tensor
    frame_logits: torch.Tensor
    micro_logits: torch.Tensor


class PhysicalRDDroneBirdClassifier(nn.Module):
    label_names = ("DroneTarget", "BirdTarget")
    supports_rejection = False

    def __init__(
        self,
        input_channels: int = PHYSICAL_RD_INPUT_CHANNELS,
        token_dim: int = 256,
        quality_dim: int = QUALITY_FEATURE_DIM,
        dropout: float = 0.1,
        amplitude_clip_low_db: float = 37.125,
        amplitude_clip_high_db: float = 78.0,
        statistics_center: torch.Tensor | None = None,
        statistics_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_channels != PHYSICAL_RD_INPUT_CHANNELS:
            raise ValueError(f"physical RD classifier requires exactly {PHYSICAL_RD_INPUT_CHANNELS} schema channels")
        if quality_dim != QUALITY_FEATURE_DIM:
            raise ValueError(f"physical RD classifier requires exactly {QUALITY_FEATURE_DIM} schema quality features")
        if token_dim < 32:
            raise ValueError("token_dim must be at least 32")
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.quality_dim = int(quality_dim)
        self.amplitude_clip_low_db = float(amplitude_clip_low_db)
        self.amplitude_clip_high_db = float(amplitude_clip_high_db)
        if self.amplitude_clip_high_db <= self.amplitude_clip_low_db:
            raise ValueError("physical RD amplitude clip bounds are invalid")
        center = torch.zeros(PHYSICAL_RD_STAT_DIM, dtype=torch.float32) if statistics_center is None else statistics_center.detach().to(torch.float32)
        scale = torch.ones(PHYSICAL_RD_STAT_DIM, dtype=torch.float32) if statistics_scale is None else statistics_scale.detach().to(torch.float32)
        if tuple(center.shape) != (PHYSICAL_RD_STAT_DIM,) or tuple(scale.shape) != (PHYSICAL_RD_STAT_DIM,):
            raise ValueError("physical RD statistics center/scale dimensions are invalid")
        if (scale <= 1.0e-6).any():
            raise ValueError("physical RD statistics scale must be positive")
        self.register_buffer("statistics_center", center)
        self.register_buffer("statistics_scale", scale)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3), bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3), bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.spatial_average_pool = nn.AdaptiveAvgPool2d((SPATIAL_POOL_RANGE_BINS, SPATIAL_POOL_DOPPLER_BINS))
        self.spatial_max_pool = nn.AdaptiveMaxPool2d((SPATIAL_POOL_RANGE_BINS, SPATIAL_POOL_DOPPLER_BINS))
        self.spatial_projection = nn.Sequential(
            nn.Linear(64 * SPATIAL_POOL_RANGE_BINS * SPATIAL_POOL_DOPPLER_BINS * 2, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.temporal_encoder = nn.GRU(token_dim, token_dim, batch_first=True)
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )
        self.frame_classifier = nn.Linear(token_dim, 2)
        self.quality_projection = (
            nn.Sequential(
                nn.LayerNorm(quality_dim),
                nn.Linear(quality_dim, token_dim),
                nn.GELU(),
                nn.LayerNorm(token_dim),
            )
            if quality_dim
            else None
        )
        self.statistics_projection = nn.Sequential(
            nn.LayerNorm(PHYSICAL_RD_STAT_DIM),
            nn.Linear(PHYSICAL_RD_STAT_DIM, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.micro_projection = nn.Sequential(
            nn.Linear(2 * MICRO_TIME_BINS * MICRO_VELOCITY_BINS, token_dim // 2),
            nn.GELU(),
            nn.LayerNorm(token_dim // 2),
        )
        self.micro_classifier = nn.Linear(token_dim // 2, 2)
        classifier_input_dim = token_dim * 5 + token_dim // 2 + (token_dim if quality_dim else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 2),
        )

    @staticmethod
    def _time_features(time_seconds: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count = time_seconds.shape
        valid = ~padding_mask.bool()
        positions = torch.arange(frame_count, device=time_seconds.device, dtype=time_seconds.dtype).view(1, -1)
        valid_lengths = valid.sum(dim=1).clamp_min(1).to(time_seconds.dtype)
        normalized_order = positions / (valid_lengths.unsqueeze(1) - 1.0).clamp_min(1.0)
        normalized_order = torch.where(valid, normalized_order, torch.zeros_like(normalized_order))
        local_delta = torch.zeros_like(time_seconds)
        if frame_count > 1:
            differences = time_seconds[:, 1:] - time_seconds[:, :-1]
            valid_pairs = valid[:, 1:] & valid[:, :-1]
            local_delta[:, 1:] = torch.where(
                valid_pairs & (differences > 1.0e-6),
                torch.log1p(differences.clamp_min(0.0)),
                torch.zeros_like(differences),
            )
        local_delta = torch.where(valid, local_delta, torch.zeros_like(local_delta))
        return torch.stack([normalized_order, local_delta], dim=-1)

    @staticmethod
    def _micro_velocity_time_profile(
        relative_power: torch.Tensor,
        time_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if relative_power.ndim != 4 or time_seconds.shape != relative_power.shape[:2] or padding_mask.shape != relative_power.shape[:2]:
            raise ValueError("micro-Doppler input must include matching [batch, time] timestamps and padding mask")
        valid_lengths = (~padding_mask.bool()).sum(dim=1)
        velocity_profiles = relative_power.mean(dim=-2)
        pooled_rows: list[torch.Tensor] = []
        for batch_index, length in enumerate(valid_lengths.tolist()):
            profile = velocity_profiles[batch_index, : int(length)]
            observed_times = time_seconds[batch_index, : int(length)]
            if int(length) == 0:
                pooled_rows.append(profile.new_zeros((2 * MICRO_TIME_BINS * MICRO_VELOCITY_BINS,)))
                continue
            if int(length) == 1:
                normalized_times = observed_times.new_zeros((1,))
            else:
                elapsed = observed_times - observed_times[0]
                duration = elapsed[-1]
                if float(duration.detach()) > 1.0e-6:
                    normalized_times = elapsed / duration
                else:
                    normalized_times = torch.arange(
                        int(length), device=profile.device, dtype=profile.dtype
                    ) / float(max(1, int(length) - 1))
            target_times = torch.linspace(
                0.0,
                1.0,
                MICRO_TIME_BINS,
                device=profile.device,
                dtype=profile.dtype,
            )
            if int(length) == 1:
                resampled = profile.expand(MICRO_TIME_BINS, -1)
            else:
                right = torch.searchsorted(normalized_times, target_times, right=True).clamp(
                    min=1,
                    max=int(length) - 1,
                )
                left = right - 1
                left_times = normalized_times[left]
                right_times = normalized_times[right]
                interpolation = ((target_times - left_times) / (right_times - left_times).clamp_min(1.0e-6)).unsqueeze(-1)
                resampled = profile[left] * (1.0 - interpolation) + profile[right] * interpolation
            profile = resampled.unsqueeze(0).unsqueeze(0)
            average = nn.functional.adaptive_avg_pool2d(
                profile,
                (MICRO_TIME_BINS, MICRO_VELOCITY_BINS),
            )
            maximum = nn.functional.adaptive_max_pool2d(
                profile,
                (MICRO_TIME_BINS, MICRO_VELOCITY_BINS),
            )
            pooled_rows.append(torch.cat([average, maximum], dim=1).flatten(1).squeeze(0))
        return torch.stack(pooled_rows, dim=0)

    def forward(
        self,
        features: torch.Tensor,
        time_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
        quality_features: torch.Tensor | None = None,
        range_offset_m: torch.Tensor | None = None,
        velocity_mps: torch.Tensor | None = None,
        raw_amplitude_db: torch.Tensor | None = None,
    ) -> PhysicalRDDroneBirdOutput:
        if features.ndim != 5:
            raise ValueError("physical RD features must be [batch, time, channel, range, doppler]")
        batch_size, frame_count, channels, _, _ = features.shape
        if channels != self.input_channels:
            raise ValueError(f"unexpected physical RD channel count: {channels}")
        if time_seconds.shape != (batch_size, frame_count) or padding_mask.shape != (batch_size, frame_count):
            raise ValueError("physical RD time and padding tensors must match batch/time dimensions")
        if not torch.isfinite(time_seconds).all():
            raise ValueError("physical RD time axis must be finite")
        valid = ~padding_mask.bool()
        if frame_count > 1:
            differences = time_seconds[:, 1:] - time_seconds[:, :-1]
            valid_pairs = valid[:, 1:] & valid[:, :-1]
            if (valid_pairs & (differences <= 1.0e-6)).any():
                raise ValueError("physical RD valid-frame time axis must be strictly increasing")
        lengths = (~padding_mask.bool()).sum(dim=1)
        if quality_features is None:
            raise ValueError("physical RD classifier requires explicit quality features")
        if quality_features.shape != (batch_size, self.quality_dim):
            raise ValueError("physical RD quality features must have shape [batch, quality_dim]")
        if not torch.isfinite(quality_features).all():
            raise ValueError("physical RD quality features must be finite")
        if range_offset_m is None or velocity_mps is None:
            raise ValueError("physical RD classifier requires explicit range and velocity axes")
        if raw_amplitude_db is None or raw_amplitude_db.shape != features.shape[:2] + features.shape[-2:]:
            raise ValueError("physical RD classifier requires raw_amplitude_db shaped [batch, time, range, doppler]")
        statistics = physical_rd_statistics(
            raw_amplitude_db,
            features[:, :, 1],
            features[:, :, 2],
            padding_mask,
            range_offset_m,
            velocity_mps,
            amplitude_clip_low_db=self.amplitude_clip_low_db,
            amplitude_clip_high_db=self.amplitude_clip_high_db,
        )
        flat = features.reshape(batch_size * frame_count, channels, features.shape[-2], features.shape[-1])
        spatial = self.spatial_encoder(flat)
        pooled = torch.cat(
            [self.spatial_average_pool(spatial).flatten(1), self.spatial_max_pool(spatial).flatten(1)],
            dim=-1,
        )
        tokens = self.spatial_projection(pooled).reshape(batch_size, frame_count, self.token_dim)
        tokens = tokens + self.time_projection(self._time_features(time_seconds, padding_mask))
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        frame_tokens = tokens.new_zeros((batch_size, frame_count, self.token_dim))
        hidden = tokens.new_zeros((1, batch_size, self.token_dim))
        nonempty = lengths > 0
        if bool(nonempty.any()):
            packed = nn.utils.rnn.pack_padded_sequence(
                tokens[nonempty],
                lengths[nonempty].detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_tokens, packed_hidden = self.temporal_encoder(packed)
            nonempty_frame_tokens, _ = nn.utils.rnn.pad_packed_sequence(
                packed_tokens,
                batch_first=True,
                total_length=frame_count,
            )
            frame_tokens[nonempty] = nonempty_frame_tokens
            hidden[:, nonempty] = packed_hidden
        frame_tokens = frame_tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        token_weights = valid.to(frame_tokens.dtype).unsqueeze(-1)
        temporal_mean = (frame_tokens * token_weights).sum(dim=1) / token_weights.sum(dim=1).clamp_min(1.0)
        temporal_max = frame_tokens.masked_fill(padding_mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        temporal_max = torch.nan_to_num(temporal_max, nan=0.0, neginf=0.0, posinf=0.0)
        attention_scores = self.temporal_attention(frame_tokens).squeeze(-1)
        attention_scores = attention_scores.masked_fill(padding_mask, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attention_weights = attention_weights * valid.to(attention_weights.dtype)
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        temporal_attention = (frame_tokens * attention_weights.unsqueeze(-1)).sum(dim=1)
        frame_logits = self.frame_classifier(frame_tokens)
        frame_logits = frame_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        quality = (
            self.quality_projection(quality_features.to(device=features.device, dtype=features.dtype))
            if self.quality_projection is not None
            else None
        )
        normalized_statistics = (statistics - self.statistics_center.to(features.dtype)) / self.statistics_scale.to(features.dtype)
        statistics_embedding = self.statistics_projection(normalized_statistics)
        clipped_amplitude = raw_amplitude_db.to(device=features.device, dtype=features.dtype).clamp(
            min=self.amplitude_clip_low_db,
            max=self.amplitude_clip_high_db,
        )
        relative_power = torch.pow(10.0, (clipped_amplitude - self.amplitude_clip_high_db) / 10.0)
        micro_profile = self._micro_velocity_time_profile(relative_power, time_seconds, padding_mask)
        micro_embedding = self.micro_projection(micro_profile)
        micro_logits = self.micro_classifier(micro_embedding)
        classifier_inputs = [hidden[-1], temporal_mean, temporal_max, temporal_attention, statistics_embedding, micro_embedding]
        if quality is not None:
            classifier_inputs.append(quality)
        logits = self.classifier(torch.cat(classifier_inputs, dim=-1))
        return PhysicalRDDroneBirdOutput(
            logits=logits,
            frame_tokens=frame_tokens,
            statistics=statistics,
            frame_logits=frame_logits,
            micro_logits=micro_logits,
        )


__all__ = [
    "PhysicalRDDroneBirdClassifier",
    "PhysicalRDDroneBirdOutput",
    "SPATIAL_POOL_DOPPLER_BINS",
    "SPATIAL_POOL_RANGE_BINS",
    "TEMPORAL_POOLING_VERSION",
]
