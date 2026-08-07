from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES
from .data.rd_physical_window_dataset import PHYSICAL_RD_INPUT_CHANNELS
from .rd_physical_drone_bird_model import (
    MICRO_TIME_BINS,
    MICRO_VELOCITY_BINS,
    SPATIAL_POOL_DOPPLER_BINS,
    SPATIAL_POOL_RANGE_BINS,
    TEMPORAL_POOLING_VERSION,
)
from .rd_physical_statistics import PHYSICAL_RD_STAT_DIM, physical_rd_statistics


FIVE_RD_MODEL_SCHEMA_VERSION = "physical_rd_five_model_v1"


@dataclass(frozen=True)
class PhysicalRDFiveOutput:
    flat_logits: torch.Tensor
    cascade_logits: torch.Tensor
    cascade_probabilities: torch.Tensor
    parent_logits: torch.Tensor
    frame_logits: torch.Tensor
    micro_logits: torch.Tensor

    @property
    def logits(self) -> torch.Tensor:
        return self.flat_logits


class PhysicalRDFiveClassifier(nn.Module):
    label_names = RD_FIVE_LABELS
    quality_names = RD_FIVE_QUALITY_NAMES
    supports_rejection = False
    temporal_pooling_version = TEMPORAL_POOLING_VERSION

    def __init__(
        self,
        *,
        input_channels: int = PHYSICAL_RD_INPUT_CHANNELS,
        token_dim: int = 256,
        quality_dim: int = len(RD_FIVE_QUALITY_NAMES),
        dropout: float = 0.1,
        amplitude_clip_low_db: float = 37.125,
        amplitude_clip_high_db: float = 78.0,
        statistics_center: torch.Tensor | None = None,
        statistics_scale: torch.Tensor | None = None,
        parent_partition: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4)),
        head_mode: str = "five",
    ) -> None:
        super().__init__()
        if input_channels != PHYSICAL_RD_INPUT_CHANNELS or quality_dim != len(RD_FIVE_QUALITY_NAMES):
            raise ValueError("five-class RD input channel/quality dimensions do not match schema")
        if token_dim < 32 or amplitude_clip_high_db <= amplitude_clip_low_db:
            raise ValueError("invalid five-class RD model dimensions or amplitude bounds")
        self.input_channels = input_channels
        self.token_dim = token_dim
        self.quality_dim = quality_dim
        self.amplitude_clip_low_db = float(amplitude_clip_low_db)
        self.amplitude_clip_high_db = float(amplitude_clip_high_db)
        self.parent_partition = tuple(tuple(int(index) for index in group) for group in parent_partition)
        if head_mode not in {"five", "parent_gate", "dbb", "cu"}:
            raise ValueError("unsupported RD head mode")
        self.head_mode = str(head_mode)
        flattened = [index for group in self.parent_partition for index in group]
        if sorted(flattened) != list(range(len(RD_FIVE_LABELS))):
            raise ValueError("five-class soft cascade partition must cover all labels exactly once")
        center = torch.zeros(PHYSICAL_RD_STAT_DIM) if statistics_center is None else statistics_center.detach().float()
        scale = torch.ones(PHYSICAL_RD_STAT_DIM) if statistics_scale is None else statistics_scale.detach().float()
        if center.shape != (PHYSICAL_RD_STAT_DIM,) or scale.shape != (PHYSICAL_RD_STAT_DIM,) or (scale <= 1.0e-6).any():
            raise ValueError("five-class RD statistics center/scale are invalid")
        self.register_buffer("statistics_center", center)
        self.register_buffer("statistics_scale", scale)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3), bias=False),
            nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3), bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((SPATIAL_POOL_RANGE_BINS, SPATIAL_POOL_DOPPLER_BINS))
        self.max_pool = nn.AdaptiveMaxPool2d((SPATIAL_POOL_RANGE_BINS, SPATIAL_POOL_DOPPLER_BINS))
        self.spatial_projection = nn.Sequential(
            nn.Linear(64 * SPATIAL_POOL_RANGE_BINS * SPATIAL_POOL_DOPPLER_BINS * 2, token_dim), nn.GELU(), nn.LayerNorm(token_dim)
        )
        self.time_projection = nn.Sequential(nn.Linear(2, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.temporal_encoder = nn.GRU(token_dim, token_dim, batch_first=True)
        self.temporal_attention = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.quality_projection = nn.Sequential(nn.LayerNorm(quality_dim), nn.Linear(quality_dim, token_dim), nn.GELU(), nn.LayerNorm(token_dim))
        self.statistics_projection = nn.Sequential(nn.LayerNorm(PHYSICAL_RD_STAT_DIM), nn.Linear(PHYSICAL_RD_STAT_DIM, token_dim), nn.GELU(), nn.LayerNorm(token_dim))
        self.micro_projection = nn.Sequential(nn.Linear(2 * MICRO_TIME_BINS * MICRO_VELOCITY_BINS, token_dim // 2), nn.GELU(), nn.LayerNorm(token_dim // 2))
        embedding_dim = token_dim * 5 + token_dim // 2 + token_dim
        local_classes = {
            "five": len(RD_FIVE_LABELS),
            "parent_gate": len(self.parent_partition),
            "dbb": len(self.parent_partition[0]),
            "cu": len(self.parent_partition[1]),
        }[self.head_mode]
        self.flat_head = (
            nn.Sequential(nn.LayerNorm(embedding_dim), nn.Dropout(dropout), nn.Linear(embedding_dim, local_classes))
            if self.head_mode != "parent_gate" else None
        )
        self.parent_head = nn.Linear(embedding_dim, len(self.parent_partition)) if self.head_mode in {"five", "parent_gate"} else None
        self.child_heads = (
            nn.ModuleList(nn.Linear(embedding_dim, len(group)) for group in self.parent_partition)
            if self.head_mode == "five" else None
        )
        self.frame_head = nn.Linear(token_dim, local_classes)
        self.micro_head = nn.Linear(token_dim // 2, local_classes)

    @staticmethod
    def _time_features(time_seconds: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = ~padding_mask.bool()
        if bool((padding_mask.bool()[:, :-1] & ~padding_mask.bool()[:, 1:]).any()):
            raise ValueError("five-class RD padding mask must be suffix-only")
        positions = torch.arange(time_seconds.shape[1], device=time_seconds.device, dtype=time_seconds.dtype).view(1, -1)
        lengths = valid.sum(dim=1).clamp_min(1).to(time_seconds.dtype)
        order = torch.where(valid, positions / (lengths[:, None] - 1.0).clamp_min(1.0), torch.zeros_like(time_seconds))
        delta = torch.zeros_like(time_seconds)
        if time_seconds.shape[1] > 1:
            diff = time_seconds[:, 1:] - time_seconds[:, :-1]
            pair = valid[:, 1:] & valid[:, :-1] & (diff > 1.0e-6)
            delta[:, 1:] = torch.where(pair, torch.log1p(diff.clamp_min(0.0)), torch.zeros_like(diff))
        return torch.stack([order, torch.where(valid, delta, torch.zeros_like(delta))], dim=-1)

    @staticmethod
    def _micro_profile(relative_power: torch.Tensor, time_seconds: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid_lengths = (~padding_mask.bool()).sum(dim=1)
        profiles = relative_power.mean(dim=-2)
        rows: list[torch.Tensor] = []
        for batch_index, length_value in enumerate(valid_lengths.tolist()):
            length = int(length_value)
            profile = profiles[batch_index, :length]
            observed = time_seconds[batch_index, :length]
            if length == 0:
                rows.append(relative_power.new_zeros((2 * MICRO_TIME_BINS * MICRO_VELOCITY_BINS,)))
                continue
            if length == 1:
                resampled = profile.expand(MICRO_TIME_BINS, -1)
            else:
                elapsed = observed - observed[0]
                normalized = elapsed / elapsed[-1].clamp_min(1.0e-6)
                target = torch.linspace(0.0, 1.0, MICRO_TIME_BINS, device=profile.device, dtype=profile.dtype)
                right = torch.searchsorted(normalized, target, right=True).clamp(1, length - 1)
                left = right - 1
                ratio = ((target - normalized[left]) / (normalized[right] - normalized[left]).clamp_min(1.0e-6))[:, None]
                resampled = profile[left] * (1.0 - ratio) + profile[right] * ratio
            view = resampled[None, None]
            average = nn.functional.adaptive_avg_pool2d(view, (MICRO_TIME_BINS, MICRO_VELOCITY_BINS))
            maximum = nn.functional.adaptive_max_pool2d(view, (MICRO_TIME_BINS, MICRO_VELOCITY_BINS))
            rows.append(torch.cat([average, maximum], dim=1).flatten())
        return torch.stack(rows, dim=0)

    def forward(self, features: torch.Tensor, time_seconds: torch.Tensor, padding_mask: torch.Tensor, quality_features: torch.Tensor, range_offset_m: torch.Tensor, velocity_mps: torch.Tensor, raw_amplitude_db: torch.Tensor) -> PhysicalRDFiveOutput:
        expected_shape = (features.shape[0], features.shape[1], self.input_channels, 21, 384) if features.ndim == 5 else None
        if features.ndim != 5 or tuple(features.shape) != expected_shape or time_seconds.shape != features.shape[:2] or padding_mask.shape != features.shape[:2]:
            raise ValueError("five-class RD forward shape contract failed")
        if not torch.isfinite(features).all() or not torch.isfinite(time_seconds).all():
            raise ValueError("five-class RD features/time must be finite")
        if quality_features.shape != (features.shape[0], self.quality_dim) or not torch.isfinite(quality_features).all():
            raise ValueError("five-class RD quality contract failed")
        if range_offset_m.shape != (features.shape[-2],) or velocity_mps.shape != (features.shape[-1],):
            raise ValueError("five-class RD axis shape contract failed")
        if not torch.isfinite(range_offset_m).all() or not torch.isfinite(velocity_mps).all():
            raise ValueError("five-class RD axes must be finite")
        if (torch.diff(range_offset_m) <= 0.0).any() or (torch.diff(velocity_mps) <= 0.0).any():
            raise ValueError("five-class RD range/doppler axes must be strictly increasing")
        if raw_amplitude_db.shape != features.shape[:2] + features.shape[-2:] or not torch.isfinite(raw_amplitude_db).all():
            raise ValueError("five-class RD raw amplitude shape contract failed")
        valid = ~padding_mask.bool()
        if bool((padding_mask.bool()[:, :-1] & ~padding_mask.bool()[:, 1:]).any()):
            raise ValueError("five-class RD padding mask must be suffix-only")
        if features.shape[1] > 1:
            diff = time_seconds[:, 1:] - time_seconds[:, :-1]
            if (valid[:, 1:] & valid[:, :-1] & (diff <= 1.0e-6)).any():
                raise ValueError("five-class RD time axis must be strictly increasing")
        lengths = valid.sum(dim=1)
        if (lengths <= 0).any():
            raise ValueError("five-class RD does not accept all-padding windows")
        statistics = physical_rd_statistics(raw_amplitude_db, features[:, :, 1], features[:, :, 2], padding_mask, range_offset_m, velocity_mps, amplitude_clip_low_db=self.amplitude_clip_low_db, amplitude_clip_high_db=self.amplitude_clip_high_db)
        flat = features.reshape(features.shape[0] * features.shape[1], self.input_channels, features.shape[-2], features.shape[-1])
        spatial = self.spatial_encoder(flat)
        pooled = torch.cat([self.avg_pool(spatial).flatten(1), self.max_pool(spatial).flatten(1)], dim=-1)
        tokens = self.spatial_projection(pooled).reshape(features.shape[0], features.shape[1], self.token_dim) + self.time_projection(self._time_features(time_seconds, padding_mask))
        tokens = tokens.masked_fill(padding_mask[..., None], 0.0)
        frame_tokens = tokens.new_zeros(tokens.shape)
        hidden = tokens.new_zeros((1, features.shape[0], self.token_dim))
        nonempty = lengths > 0
        if nonempty.any():
            packed = nn.utils.rnn.pack_padded_sequence(tokens[nonempty], lengths[nonempty].detach().cpu(), batch_first=True, enforce_sorted=False)
            packed_out, packed_hidden = self.temporal_encoder(packed)
            frame_tokens[nonempty], _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=features.shape[1])
            hidden[:, nonempty] = packed_hidden
        frame_tokens = frame_tokens.masked_fill(padding_mask[..., None], 0.0)
        weights = valid.to(frame_tokens.dtype)[..., None]
        mean = (frame_tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        maximum = torch.nan_to_num(frame_tokens.masked_fill(padding_mask[..., None], float("-inf")).amax(1), neginf=0.0)
        attention_scores = self.temporal_attention(frame_tokens).squeeze(-1).masked_fill(padding_mask, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=1).nan_to_num(0.0) * valid.to(frame_tokens.dtype)
        attention = (frame_tokens * (attention_weights / attention_weights.sum(1, keepdim=True).clamp_min(1.0))[..., None]).sum(1)
        quality = self.quality_projection(quality_features)
        statistic_embedding = self.statistics_projection((statistics - self.statistics_center) / self.statistics_scale)
        clipped = raw_amplitude_db.clamp(self.amplitude_clip_low_db, self.amplitude_clip_high_db)
        relative = torch.pow(10.0, (clipped - self.amplitude_clip_high_db) / 10.0)
        micro_embedding = self.micro_projection(self._micro_profile(relative, time_seconds, padding_mask))
        embedding = torch.cat([hidden[-1], mean, maximum, attention, statistic_embedding, micro_embedding, quality], dim=-1)
        if self.head_mode == "parent_gate":
            assert self.parent_head is not None
            parent_logits = self.parent_head(embedding)
            flat_logits = parent_logits
            cascade_logits = parent_logits
        elif self.head_mode in {"dbb", "cu"}:
            assert self.flat_head is not None
            flat_logits = self.flat_head(embedding)
            parent_logits = embedding.new_zeros((embedding.shape[0], len(self.parent_partition)))
            cascade_logits = flat_logits
        else:
            assert self.flat_head is not None and self.parent_head is not None and self.child_heads is not None
            flat_logits = self.flat_head(embedding)
            parent_logits = self.parent_head(embedding)
            # Every class belongs to exactly one configured parent group, so the
            # cascade tensor remains finite while preserving the exact
            # parent-log-probability + conditional-child-log-probability contract.
            cascade_logits = embedding.new_zeros((embedding.shape[0], len(RD_FIVE_LABELS)))
            parent_log_prob = torch.log_softmax(parent_logits, dim=-1)
            for parent_index, (group, child_head) in enumerate(zip(self.parent_partition, self.child_heads)):
                indices = torch.tensor(group, device=embedding.device, dtype=torch.long)
                cascade_logits[:, indices] = parent_log_prob[:, parent_index, None] + torch.log_softmax(child_head(embedding), dim=-1)
        return PhysicalRDFiveOutput(
            flat_logits=flat_logits,
            cascade_logits=cascade_logits,
            cascade_probabilities=torch.softmax(cascade_logits, dim=-1),
            parent_logits=parent_logits,
            frame_logits=self.frame_head(frame_tokens),
            micro_logits=self.micro_head(micro_embedding),
        )


__all__ = ["FIVE_RD_MODEL_SCHEMA_VERSION", "PhysicalRDFiveClassifier", "PhysicalRDFiveOutput"]
