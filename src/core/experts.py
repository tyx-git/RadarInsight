from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


LOW_ALTITUDE_CLASSES = ["DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"]
CONVENTIONAL_AIR_CLASSES = ["AircraftTarget"]
GROUND_TARGET_CLASSES = ["PeopleTarget", "CarTarget", "ShipTarget"]


@dataclass(frozen=True)
class ExpertOutput:
    """
    专家输出类，包含分类 logits 和特征表示。
    """
    logits: torch.Tensor
    features: torch.Tensor | None = None
    auxiliary: dict[str, torch.Tensor] | None = None



class TemporalAttentionPool(nn.Module):
    """
    时间注意力池化层，用于对序列数据进行加权平均，生成固定长度的表示。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.score(sequence)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(sequence * weights, dim=1)



@dataclass(frozen=True)
class LowAltitudeFusionOutput:
    """
    低海拔融合输出类，包含轨迹特征、径向速度特征、轨迹余弦相似度、径向速度余弦相似度、轨迹分类 logits、径向速度分类 logits、融合 logits、融合权重、径向速度是否激活、是否拒绝未知目标等信息。
    """
    traj_features: torch.Tensor
    rd_features: torch.Tensor
    traj_cosine: torch.Tensor
    rd_cosine: torch.Tensor
    traj_logits: torch.Tensor
    rd_logits: torch.Tensor
    rd_bird_logits: torch.Tensor
    rd_unknown_logits: torch.Tensor
    rd_clutter_logits: torch.Tensor
    logits: torch.Tensor
    fusion_weights: torch.Tensor
    rd_activated: torch.Tensor
    rejected_unknown: torch.Tensor



class CosineMarginHead(nn.Module):
    """
    余弦边距头模型，用于低海拔目标分类。

    参数:
        input_dim (int): 输入特征维度。
        num_classes (int): 输出类别数量。
        scale (float, 可选): 缩放因子，默认值为 16.0。
        margin (float, 可选): 边距值，默认值为 0.2。
    """
    def __init__(self, input_dim: int, num_classes: int, scale: float = 16.0, margin: float = 0.2) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def _cosine(self, features: torch.Tensor) -> torch.Tensor:
        norm_features = F.normalize(features, dim=-1)
        norm_weight = F.normalize(self.weight, dim=-1)
        return norm_features @ norm_weight.transpose(0, 1)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = self._cosine(features)
        logits = cosine
        if labels is not None:
            valid = labels >= 0
            if valid.any():
                one_hot = torch.zeros_like(cosine)
                one_hot[valid, labels[valid]] = 1.0
                logits = cosine - one_hot * self.margin
        return cosine, logits * self.scale


class MLPExpert(nn.Module):
    """
    多层感知机专家模型，用于低海拔目标分类。

    参数:
        input_dim (int): 输入特征维度。
        num_classes (int): 输出类别数量。
        hidden_dim (int, 可选): 隐藏层维度，默认值为 128。
        dropout (float, 可选): Dropout 概率，默认值为 0.1。
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> ExpertOutput:
        return ExpertOutput(logits=self.network(fused_features))



class LowAltitudeSuspectExpert(nn.Module):
    """
    低空目标专家模型
    """
    def __init__(
        self,
        input_dim: int,
        rd_input_dim: int = 4,
        rd_patch_input_dim: int = 2,
        rd_context_dim: int = 0,
        use_rd_patch_cnn: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        cosface_scale: float = 16.0,
        cosface_margin: float = 0.2,
        reject_threshold: float = 0.35,
        rd_activation_threshold: float = 0.8,
        rd_binary_logit_scale: float = 0.75,
        rd_bird_logit_scale: float | None = None,
        rd_unknown_logit_scale: float | None = None,
        rd_clutter_logit_scale: float | None = None,
        initial_rd_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.rd_activation_threshold = rd_activation_threshold
        self.reject_threshold = reject_threshold
        self.rd_binary_logit_scale = rd_binary_logit_scale
        self.rd_bird_logit_scale = float(rd_binary_logit_scale if rd_bird_logit_scale is None else rd_bird_logit_scale)
        self.rd_unknown_logit_scale = float(rd_binary_logit_scale if rd_unknown_logit_scale is None else rd_unknown_logit_scale)
        self.rd_clutter_logit_scale = float(rd_binary_logit_scale if rd_clutter_logit_scale is None else rd_clutter_logit_scale)
        self.rd_model_dim = 512
        self.rd_stats_dim = 64
        self.rd_context_dim = int(max(0, rd_context_dim))
        self.rd_context_hidden_dim = 32 if self.rd_context_dim > 0 else 0
        self.rd_patch_dim = 64 if use_rd_patch_cnn else 0
        self.rd_patch_input_dim = int(max(1, rd_patch_input_dim))
        self.traj_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.traj_head = CosineMarginHead(
            input_dim=hidden_dim,
            num_classes=len(LOW_ALTITUDE_CLASSES),
            scale=cosface_scale,
            margin=cosface_margin,
        )
        self.range_amplitude_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        self.range_phase_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        self.doppler_amplitude_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        self.doppler_phase_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        self.rd_token_proj = nn.Linear(2048, self.rd_model_dim)
        rd_layer = nn.TransformerEncoderLayer(
            d_model=self.rd_model_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.rd_encoder = nn.TransformerEncoder(rd_layer, num_layers=3)
        self.rd_norm = nn.LayerNorm(self.rd_model_dim)
        self.stats_proj = nn.Sequential(
            nn.Linear(4, self.rd_stats_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.rd_stats_dim, self.rd_stats_dim),
            nn.LayerNorm(self.rd_stats_dim),
        )
        if self.rd_context_dim > 0:
            self.rd_context_proj = nn.Sequential(
                nn.Linear(self.rd_context_dim, self.rd_context_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.rd_context_hidden_dim),
            )
        else:
            self.rd_context_proj = None
        if use_rd_patch_cnn:
            self.rd_patch_cnn = nn.Sequential(
                nn.Conv2d(self.rd_patch_input_dim, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, self.rd_patch_dim),
                nn.GELU(),
                nn.LayerNorm(self.rd_patch_dim),
            )
            self.rd_patch_temporal_pool = TemporalAttentionPool(self.rd_patch_dim, hidden_dim=64)
        else:
            self.rd_patch_cnn = None
            self.rd_patch_temporal_pool = None
        self.rd_backbone = nn.Sequential(
            nn.Linear(self.rd_model_dim + self.rd_stats_dim + self.rd_context_hidden_dim + self.rd_patch_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
        )
        self.rd_head = CosineMarginHead(
            input_dim=512,
            num_classes=len(LOW_ALTITUDE_CLASSES),
            scale=cosface_scale,
            margin=cosface_margin,
        )
        self.rd_bird_head = nn.Linear(512, 2)
        self.rd_unknown_head = nn.Linear(512, 2)
        self.rd_clutter_head = nn.Linear(512, 2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(len(LOW_ALTITUDE_CLASSES) * 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        with torch.no_grad():
            bias = float(initial_rd_gate_bias)
            self.fusion_gate[-1].bias.copy_(torch.tensor([-bias, bias], dtype=self.fusion_gate[-1].bias.dtype))

    def _rd_statistics(self, amp_range: torch.Tensor, phase_range: torch.Tensor, amp_dopp: torch.Tensor, phase_dopp: torch.Tensor) -> torch.Tensor:
        amp_mean = amp_range.mean(dim=-1, keepdim=True)
        amp_std = amp_range.std(dim=-1, keepdim=True).clamp_min(1e-6)
        centered = amp_range - amp_mean
        amp_skew = (centered.pow(3).mean(dim=-1) / amp_std.squeeze(-1).pow(3)).mean(dim=-1, keepdim=True)
        amp_kurt = (centered.pow(4).mean(dim=-1) / amp_std.squeeze(-1).pow(4)).mean(dim=-1, keepdim=True)
        phase_abs_mean = phase_dopp.abs().mean(dim=-1).mean(dim=-1, keepdim=True)
        phase_std = phase_dopp.std(dim=-1).mean(dim=-1, keepdim=True)
        stats = torch.cat([amp_skew, amp_kurt, phase_abs_mean, phase_std], dim=-1)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return stats

    def _encode_rd(self, rd_features: torch.Tensor, rd_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        amp_range = rd_features[:, :, :, 0]
        phase_range = rd_features[:, :, :, 1]
        amp_dopp = rd_features[:, :, :, 2]
        phase_dopp = rd_features[:, :, :, 3]
        amp_range_embed = self.range_amplitude_proj(amp_range)
        phase_range_embed = self.range_phase_proj(phase_range)
        amp_dopp_embed = self.doppler_amplitude_proj(amp_dopp)
        phase_dopp_embed = self.doppler_phase_proj(phase_dopp)
        hidden = self.rd_token_proj(torch.cat([amp_range_embed, phase_range_embed, amp_dopp_embed, phase_dopp_embed], dim=-1))
        hidden = self.rd_encoder(hidden, src_key_padding_mask=rd_padding_mask)
        hidden = self.rd_norm(hidden)
        if rd_padding_mask is None:
            encoded = hidden.mean(dim=1)
        else:
            valid = (~rd_padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            encoded = (hidden * valid).sum(dim=1) / denom
        stats = self.stats_proj(self._rd_statistics(amp_range, phase_range, amp_dopp, phase_dopp))
        return encoded, stats

    def _encode_rd_context(self, rd_context: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.rd_context_proj is None:
            return torch.zeros((batch_size, 0), dtype=dtype, device=device)
        if rd_context is None:
            rd_context = torch.zeros((batch_size, self.rd_context_dim), dtype=dtype, device=device)
        rd_context = torch.nan_to_num(rd_context.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        return self.rd_context_proj(rd_context)

    def _encode_rd_patch(self, rd_features: torch.Tensor, rd_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.rd_patch_cnn is None:
            return torch.zeros((rd_features.shape[0], 0), dtype=rd_features.dtype, device=rd_features.device)
        patch_input = torch.nan_to_num(rd_features, nan=0.0, posinf=0.0, neginf=0.0)
        if rd_padding_mask is not None:
            while rd_padding_mask.shape[1] != patch_input.shape[1]:
                if rd_padding_mask.shape[1] > patch_input.shape[1]:
                    rd_padding_mask = rd_padding_mask[:, : patch_input.shape[1]]
                else:
                    pad = torch.ones(
                        (rd_padding_mask.shape[0], patch_input.shape[1] - rd_padding_mask.shape[1]),
                        dtype=rd_padding_mask.dtype,
                        device=rd_padding_mask.device,
                    )
                    rd_padding_mask = torch.cat([rd_padding_mask, pad], dim=1)
            patch_input = patch_input.masked_fill(rd_padding_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)
        if patch_input.ndim == 5:
            batch_size, time_steps, range_bins, doppler_bins, channels = patch_input.shape
            frames = patch_input.reshape(batch_size * time_steps, range_bins, doppler_bins, channels)
            frames = frames.permute(0, 3, 1, 2).contiguous()
            if frames.shape[1] > self.rd_patch_input_dim:
                frames = frames[:, : self.rd_patch_input_dim]
            elif frames.shape[1] < self.rd_patch_input_dim:
                pad = frames.new_zeros(
                    (
                        frames.shape[0],
                        self.rd_patch_input_dim - frames.shape[1],
                        frames.shape[2],
                        frames.shape[3],
                    )
                )
                frames = torch.cat([frames, pad], dim=1)
            encoded_chunks = [self.rd_patch_cnn(chunk) for chunk in frames.split(64, dim=0)]
            frame_features = torch.cat(encoded_chunks, dim=0).view(batch_size, time_steps, self.rd_patch_dim)
            if self.rd_patch_temporal_pool is None:
                return frame_features.mean(dim=1)
            return self.rd_patch_temporal_pool(frame_features, padding_mask=rd_padding_mask)
        elif patch_input.ndim == 4:
            patch_input = patch_input.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("rd patch features must be [B,T,R,D,C] or [B,T,R,C]")
        if patch_input.shape[1] > self.rd_patch_input_dim:
            patch_input = patch_input[:, : self.rd_patch_input_dim]
        elif patch_input.shape[1] < self.rd_patch_input_dim:
            pad = patch_input.new_zeros(
                (
                    patch_input.shape[0],
                    self.rd_patch_input_dim - patch_input.shape[1],
                    patch_input.shape[2],
                    patch_input.shape[3],
                )
            )
            patch_input = torch.cat([patch_input, pad], dim=1)
        return self.rd_patch_cnn(patch_input)

    def _apply_reject(self, logits: torch.Tensor, cosine: torch.Tensor, activated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_cosine = cosine.max(dim=-1).values
        reject_mask = activated & (max_cosine < self.reject_threshold)
        corrected = logits.clone()
        if reject_mask.any():
            corrected[reject_mask] = -6.0
            corrected[reject_mask, LOW_ALTITUDE_CLASSES.index("UnknownTarget")] = 6.0
        return corrected, reject_mask

    def forward(
        self,
        traj_input: torch.Tensor,
        rd_features: torch.Tensor,
        rd_patch_features: torch.Tensor | None = None,
        rd_context: torch.Tensor | None = None,
        low_altitude_targets: torch.Tensor | None = None,
        rd_is_multimodal: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        enable_reject: bool = True,
        force_rd_activation: bool = False,
    ) -> LowAltitudeFusionOutput:
        traj_features = self.traj_backbone(traj_input)
        traj_cosine, traj_logits = self.traj_head(traj_features, labels=low_altitude_targets)
        rd_context_features = self._encode_rd_context(
            rd_context,
            batch_size=traj_input.shape[0],
            device=traj_input.device,
            dtype=traj_input.dtype,
        )
        if rd_features.ndim == 4:
            rd_encoded, rd_stats = self._encode_rd(rd_features, rd_padding_mask=rd_padding_mask)
            patch_source = rd_patch_features if rd_patch_features is not None else rd_features
            patch_mask = rd_patch_padding_mask if rd_patch_features is not None else rd_padding_mask
            rd_patch = self._encode_rd_patch(patch_source, rd_padding_mask=patch_mask)
            rd_features_vec = self.rd_backbone(torch.cat([rd_encoded, rd_stats, rd_context_features, rd_patch], dim=-1))
            rd_cosine, rd_logits = self.rd_head(rd_features_vec, labels=low_altitude_targets)
            rd_bird_logits = self.rd_bird_head(rd_features_vec)
            rd_unknown_logits = self.rd_unknown_head(rd_features_vec)
            rd_clutter_logits = self.rd_clutter_head(rd_features_vec)
            bird_delta = rd_bird_logits[:, 1] - rd_bird_logits[:, 0]
            unknown_delta = rd_unknown_logits[:, 1] - rd_unknown_logits[:, 0]
            clutter_delta = rd_clutter_logits[:, 1] - rd_clutter_logits[:, 0]
            rd_logits = rd_logits.clone()
            rd_logits[:, LOW_ALTITUDE_CLASSES.index("BirdTarget")] = (
                rd_logits[:, LOW_ALTITUDE_CLASSES.index("BirdTarget")]
                + self.rd_bird_logit_scale * bird_delta
            )
            rd_logits[:, LOW_ALTITUDE_CLASSES.index("UnknownTarget")] = (
                rd_logits[:, LOW_ALTITUDE_CLASSES.index("UnknownTarget")]
                + self.rd_unknown_logit_scale * unknown_delta
            )
            rd_logits[:, LOW_ALTITUDE_CLASSES.index("ClutterTarget")] = (
                rd_logits[:, LOW_ALTITUDE_CLASSES.index("ClutterTarget")]
                + self.rd_clutter_logit_scale * clutter_delta
            )
            if rd_is_multimodal is not None and not bool(rd_is_multimodal.all()):
                no_rd = ~rd_is_multimodal.bool()
                if no_rd.any():
                    neutral_logits = torch.zeros_like(rd_logits)
                    neutral_cosine = torch.zeros_like(rd_cosine)
                    rd_logits = torch.where(no_rd.unsqueeze(-1), neutral_logits, rd_logits)
                    rd_cosine = torch.where(no_rd.unsqueeze(-1), neutral_cosine, rd_cosine)
                    rd_bird_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(rd_bird_logits), rd_bird_logits)
                    rd_unknown_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(rd_unknown_logits), rd_unknown_logits)
                    rd_clutter_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(rd_clutter_logits), rd_clutter_logits)
        else:
            rd_encoded = traj_input.new_zeros((traj_input.shape[0], self.rd_model_dim))
            rd_stats = traj_input.new_zeros((traj_input.shape[0], self.rd_stats_dim))
            rd_patch = traj_input.new_zeros((traj_input.shape[0], self.rd_patch_dim))
            rd_features_vec = self.rd_backbone(torch.cat([rd_encoded, rd_stats, rd_context_features, rd_patch], dim=-1))
            rd_cosine, rd_logits = self.rd_head(rd_features_vec, labels=low_altitude_targets)
            rd_bird_logits = self.rd_bird_head(rd_features_vec)
            rd_unknown_logits = self.rd_unknown_head(rd_features_vec)
            rd_clutter_logits = self.rd_clutter_head(rd_features_vec)

        traj_probs = torch.softmax(traj_logits, dim=-1)
        rd_probs = torch.softmax(rd_logits, dim=-1)
        gate_logits = self.fusion_gate(torch.cat([traj_probs, rd_probs], dim=-1))
        fusion_weights = torch.softmax(gate_logits, dim=-1)
        fused_logits = (
            fusion_weights[:, :1] * traj_logits
            + fusion_weights[:, 1:2] * rd_logits
        )

        traj_confidence = traj_probs.max(dim=-1).values
        if rd_is_multimodal is not None:
            has_rd = rd_is_multimodal
        elif rd_features.ndim == 4:
            has_rd = torch.ones((rd_features.shape[0],), dtype=torch.bool, device=rd_features.device)
        else:
            has_rd = rd_features[..., -1].amax(dim=1) > 0.5
        forced_rd = torch.full_like(has_rd, bool(force_rd_activation), dtype=torch.bool)
        rd_activated = has_rd & (forced_rd | (traj_confidence < self.rd_activation_threshold))
        fused_cosine = 0.5 * (traj_cosine + rd_cosine)
        selected_logits = torch.where(rd_activated.unsqueeze(-1), fused_logits, traj_logits)
        selected_cosine = torch.where(rd_activated.unsqueeze(-1), fused_cosine, traj_cosine)
        if enable_reject:
            final_logits, rejected_unknown = self._apply_reject(selected_logits, selected_cosine, rd_activated)
        else:
            final_logits = selected_logits
            rejected_unknown = torch.zeros_like(has_rd, dtype=torch.bool)
        return LowAltitudeFusionOutput(
            traj_features=traj_features,
            rd_features=rd_features_vec,
            traj_cosine=traj_cosine,
            rd_cosine=rd_cosine,
            traj_logits=traj_logits,
            rd_logits=rd_logits,
            rd_bird_logits=rd_bird_logits,
            rd_unknown_logits=rd_unknown_logits,
            rd_clutter_logits=rd_clutter_logits,
            logits=final_logits,
            fusion_weights=fusion_weights,
            rd_activated=rd_activated,
            rejected_unknown=rejected_unknown,
        )



class ConventionalAirExpert(nn.Module):
    """
    传统航空专家类，用于处理传统航空目标的分类任务。
    """
    def __init__(
        self,
        sequence_dim: int,
        physical_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        open_set: bool = True,
        logit_scale: float = 10.0,
        decision_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        self.open_set = bool(open_set)
        self.logit_scale = float(logit_scale)
        self.physical_input_norm = nn.LayerNorm(physical_dim)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(sequence_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal_pool = TemporalAttentionPool(128, hidden_dim=64)
        self.physical_proj = nn.Sequential(
            nn.Linear(physical_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.aircraft_prototype = nn.Parameter(torch.randn(128))
        nn.init.normal_(self.aircraft_prototype, mean=0.0, std=0.02)
        self.register_buffer("decision_threshold", torch.tensor(float(decision_threshold), dtype=torch.float32))

    def set_decision_threshold(self, value: float) -> None:
        self.decision_threshold.fill_(float(value))

    def _open_set_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_features = F.normalize(features, dim=-1)
        normalized_prototype = F.normalize(self.aircraft_prototype, dim=0)
        similarity = normalized_features @ normalized_prototype
        centered = similarity - self.decision_threshold.to(device=features.device, dtype=features.dtype)
        logits = torch.stack([-centered, centered], dim=-1) * self.logit_scale
        return logits, similarity

    def forward(self, sequence_features: torch.Tensor, physical_features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> ExpertOutput:
        seq = sequence_features.transpose(1, 2)
        hidden = self.temporal_conv(seq).transpose(1, 2)
        temporal = self.temporal_pool(hidden, padding_mask=padding_mask)
        physical = self.physical_proj(self.physical_input_norm(physical_features))
        fused = self.fuse(torch.cat([temporal, physical], dim=-1))
        if self.open_set:
            logits, similarity = self._open_set_logits(fused)
            return ExpertOutput(
                logits=logits,
                features=fused,
                auxiliary={
                    "aircraft_similarity": similarity,
                    "aircraft_threshold": self.decision_threshold.expand_as(similarity),
                },
            )
        logits = self.classifier(fused)
        return ExpertOutput(logits=logits, features=fused)


class GroundTargetExpert(nn.Module):
    """
    地面目标专家类，用于分类地面目标。
    """
    def __init__(
        self,
        deep_dim: int,
        raw_physical_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        groundness_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.physical_input_norm = nn.LayerNorm(raw_physical_dim)
        self.deep_proj = nn.Sequential(
            nn.Linear(deep_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        self.physical_proj = nn.Sequential(
            nn.Linear(raw_physical_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )
        self.gate = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(GROUND_TARGET_CLASSES)),
        )
        self.groundness_head = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.register_buffer("groundness_threshold", torch.tensor(float(groundness_threshold), dtype=torch.float32))

    def set_groundness_threshold(self, value: float) -> None:
        self.groundness_threshold.fill_(float(value))

    def forward(self, deep_features: torch.Tensor, raw_physical_features: torch.Tensor) -> ExpertOutput:
        selected = torch.stack(
            [
                raw_physical_features[:, 0],
                raw_physical_features[:, 3],
                raw_physical_features[:, 4],
                raw_physical_features[:, 5],
                raw_physical_features[:, 7],
                raw_physical_features[:, 8],
                raw_physical_features[:, 9],
                raw_physical_features[:, 10],
                raw_physical_features[:, 13],
                raw_physical_features[:, 14],
            ],
            dim=-1,
        )
        deep_hidden = self.deep_proj(deep_features)
        physical_hidden = self.physical_proj(self.physical_input_norm(selected))
        weights = torch.softmax(self.gate(torch.cat([deep_hidden, physical_hidden], dim=-1)), dim=-1)
        fused = torch.cat(
            [
                weights[:, :1] * deep_hidden,
                weights[:, 1:2] * physical_hidden,
            ],
            dim=-1,
        )
        logits = self.classifier(fused)
        groundness_logits = self.groundness_head(fused)
        groundness_probability = torch.softmax(groundness_logits, dim=-1)[:, 1]
        threshold = self.groundness_threshold.to(device=groundness_probability.device, dtype=groundness_probability.dtype)
        return ExpertOutput(
            logits=logits,
            features=fused,
            auxiliary={
                "groundness_logits": groundness_logits,
                "groundness_probability": groundness_probability,
                "groundness_threshold": threshold.expand_as(groundness_probability),
                "groundness_margin": groundness_probability - threshold,
            },
        )


class RuleBasedHighAltitudeExpert(nn.Module):
    def __init__(self, altitude_threshold_m: float = 5000.0, speed_threshold_mps: float = 100.0) -> None:
        super().__init__()
        self.altitude_threshold_m = altitude_threshold_m
        self.speed_threshold_mps = speed_threshold_mps

    def forward(self, height_mean: torch.Tensor, speed_mean: torch.Tensor) -> ExpertOutput:
        positive = (height_mean > self.altitude_threshold_m) & (speed_mean > self.speed_threshold_mps)
        logits = torch.zeros((height_mean.shape[0], 2), dtype=height_mean.dtype, device=height_mean.device)
        logits[:, 0] = torch.where(positive, torch.tensor(-1.0, device=height_mean.device), torch.tensor(1.0, device=height_mean.device))
        logits[:, 1] = torch.where(positive, torch.tensor(1.0, device=height_mean.device), torch.tensor(-1.0, device=height_mean.device))
        return ExpertOutput(logits=logits)
