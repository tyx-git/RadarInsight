from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import nn
import torch.nn.functional as F


LOW_ALTITUDE_CLASSES = ["DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"]
CONVENTIONAL_AIR_CLASSES = ["AircraftTarget"]
GROUND_TARGET_CLASSES = ["PeopleTarget", "CarTarget", "ShipTarget"]


def apply_low_altitude_reject(
    logits: torch.Tensor,
    cosine: torch.Tensor,
    activated: torch.Tensor,
    reject_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
    clutter_index = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
    known_indices = [
        LOW_ALTITUDE_CLASSES.index("DroneTarget"),
        LOW_ALTITUDE_CLASSES.index("BirdTarget"),
        LOW_ALTITUDE_CLASSES.index("BalloonTarget"),
    ]
    known_cosine = cosine[:, known_indices]
    reject_mask = activated.bool() & (known_cosine.max(dim=-1).values < float(reject_threshold))
    corrected = logits.clone()
    if bool(reject_mask.any()):
        corrected[reject_mask] = -6.0
        residual_logits = logits[reject_mask][:, [unknown_index, clutter_index]]
        residual_pred = residual_logits.argmax(dim=-1)
        reject_indices = reject_mask.nonzero(as_tuple=False).flatten()
        unknown_rows = reject_indices[residual_pred == 0]
        clutter_rows = reject_indices[residual_pred == 1]
        if unknown_rows.numel():
            corrected[unknown_rows, unknown_index] = 6.0
        if clutter_rows.numel():
            corrected[clutter_rows, clutter_index] = 6.0
    return corrected, reject_mask


def apply_low_altitude_attribution_reject(
    logits: torch.Tensor,
    cosine: torch.Tensor,
    activated: torch.Tensor,
    reject_threshold: float,
    *,
    track_quality: torch.Tensor | None = None,
    min_known_confidence: float = 0.0,
    min_known_margin: float = 0.0,
    min_track_quality: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Route non-attributable low-altitude samples to UnknownTarget.

    The route uses only model evidence and provenance-free observable track
    quality.  It never turns a sample into ClutterTarget: Clutter remains a
    separately supervised business class.
    """
    if logits.ndim != 2 or cosine.shape != logits.shape:
        raise ValueError("logits and cosine must both have shape [batch_size, class_count]")
    if activated.ndim != 1 or activated.shape[0] != logits.shape[0]:
        raise ValueError("activated must have shape [batch_size]")
    if logits.shape[-1] != len(LOW_ALTITUDE_CLASSES):
        raise ValueError("low-altitude logits must match LOW_ALTITUDE_CLASSES")

    attribution_indices = [
        LOW_ALTITUDE_CLASSES.index("DroneTarget"),
        LOW_ALTITUDE_CLASSES.index("BirdTarget"),
        LOW_ALTITUDE_CLASSES.index("BalloonTarget"),
        LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
    ]
    unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
    attribution_cosine = cosine[:, attribution_indices]
    full_probs = torch.softmax(logits, dim=-1)
    attribution_probs = full_probs[:, attribution_indices]
    attribution_top2 = attribution_probs.topk(k=2, dim=-1)
    known_confidence = attribution_top2.values[:, 0]
    known_margin = attribution_top2.values[:, 0] - attribution_top2.values[:, 1]
    prototype_distance_trigger = attribution_cosine.max(dim=-1).values < float(reject_threshold)
    known_confidence_trigger = known_confidence < float(min_known_confidence)
    known_margin_trigger = known_margin < float(min_known_margin)

    if track_quality is None:
        quality = torch.ones_like(known_confidence)
    else:
        quality = track_quality.to(device=logits.device, dtype=logits.dtype)
        if quality.ndim != 1 or quality.shape[0] != logits.shape[0]:
            raise ValueError("track_quality must have shape [batch_size]")
        quality = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    quality_low = quality < float(min_track_quality)
    quality_evidence_trigger = quality_low & (prototype_distance_trigger | known_confidence_trigger | known_margin_trigger)
    uncertain_known_trigger = known_confidence_trigger & known_margin_trigger
    reject_mask = activated.bool() & (
        prototype_distance_trigger | uncertain_known_trigger | quality_evidence_trigger
    )

    corrected = logits.clone()
    if bool(reject_mask.any()):
        corrected[reject_mask] = -6.0
        corrected[reject_mask, unknown_index] = 6.0
    reason_payload = {
        "prototype_distance_trigger": prototype_distance_trigger.detach(),
        "known_confidence_trigger": known_confidence_trigger.detach(),
        "known_margin_trigger": known_margin_trigger.detach(),
        "quality_evidence_trigger": quality_evidence_trigger.detach(),
        "known_confidence": known_confidence.detach(),
        "known_margin": known_margin.detach(),
        "track_quality": quality.detach(),
    }
    reject_score = (
        prototype_distance_trigger.float()
        + 0.5 * known_confidence_trigger.float()
        + 0.5 * known_margin_trigger.float()
        + 0.5 * quality_evidence_trigger.float()
    )
    return corrected, reject_mask, reject_score, reason_payload


def compose_low_altitude_final_logits(
    traj_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    rd_logits: torch.Tensor,
    traj_cosine: torch.Tensor,
    rd_cosine: torch.Tensor,
    rd_pair_fusion_weights: torch.Tensor,
    tr_rd_active: torch.Tensor,
    rd_alignment_quality: torch.Tensor | None,
    *,
    joint_logit_delta_scale: float,
    rd_specialist_residual_scale: float,
    residual_scale_multiplier: float,
    enable_reject: bool,
    reject_threshold: float,
    force_pair_active: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_mask = tr_rd_active.bool()
    if rd_alignment_quality is None:
        alignment_quality = active_mask.to(dtype=traj_logits.dtype)
    else:
        alignment_quality = rd_alignment_quality.to(device=traj_logits.device, dtype=traj_logits.dtype)
        if alignment_quality.ndim != 1 or alignment_quality.shape[0] != traj_logits.shape[0]:
            raise ValueError("rd_alignment_quality must have shape [batch_size]")
        alignment_quality = alignment_quality.clamp(min=0.0, max=1.0)
    alignment_quality = alignment_quality * active_mask.to(dtype=traj_logits.dtype)
    joint_delta_scale = max(0.0, float(joint_logit_delta_scale))
    joint_selected_logits = traj_logits + (
        joint_delta_scale * alignment_quality.unsqueeze(-1) * (joint_logits - traj_logits)
    )
    pair_selected_logits = torch.where(
        active_mask.unsqueeze(-1),
        joint_selected_logits,
        traj_logits,
    ).clone()
    drone_index = LOW_ALTITUDE_CLASSES.index("DroneTarget")
    bird_index = LOW_ALTITUDE_CLASSES.index("BirdTarget")
    balloon_index = LOW_ALTITUDE_CLASSES.index("BalloonTarget")
    unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
    clutter_index = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
    pair_specs = [
        (0, [drone_index, bird_index]),
        (1, [unknown_index, clutter_index]),
        (2, [balloon_index]),
    ]
    residual_scale = float(residual_scale_multiplier) * max(0.0, float(rd_specialist_residual_scale))
    for pair_index, class_indices in pair_specs:
        current_pair_logits = pair_selected_logits[:, class_indices]
        rd_pair_logits = rd_logits[:, class_indices]
        pair_weight = rd_pair_fusion_weights[:, pair_index].unsqueeze(-1)
        pair_rd_margin = rd_pair_logits.max(dim=-1).values - current_pair_logits.max(dim=-1).values
        pair_active = active_mask & (pair_rd_margin > 0.05)
        if force_pair_active:
            pair_active = active_mask
        proposed_pair_logits = current_pair_logits + residual_scale * pair_weight * (rd_pair_logits - current_pair_logits)
        pair_selected_logits[:, class_indices] = torch.where(
            pair_active.unsqueeze(-1),
            proposed_pair_logits,
            current_pair_logits,
        )
    selected_cosine = traj_cosine + 0.5 * alignment_quality.unsqueeze(-1) * (rd_cosine - traj_cosine)
    selected_cosine = torch.where(active_mask.unsqueeze(-1), selected_cosine, traj_cosine)
    if enable_reject:
        return apply_low_altitude_reject(
            pair_selected_logits,
            selected_cosine,
            active_mask,
            reject_threshold,
        )
    return pair_selected_logits, torch.zeros_like(active_mask, dtype=torch.bool)


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


class CrossModalFusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.track_to_rd = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.rd_to_track = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.track_norm1 = nn.LayerNorm(dim)
        self.rd_norm1 = nn.LayerNorm(dim)
        self.track_norm2 = nn.LayerNorm(dim)
        self.rd_norm2 = nn.LayerNorm(dim)
        self.track_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.rd_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _safe_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        safe = mask.bool()
        all_masked = safe.all(dim=1)
        if bool(all_masked.any()):
            safe = safe.clone()
            safe[all_masked] = False
        return safe

    def forward(
        self,
        track_tokens: torch.Tensor,
        rd_tokens: torch.Tensor,
        track_padding_mask: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_track_mask = self._safe_mask(track_padding_mask)
        safe_rd_mask = self._safe_mask(rd_padding_mask)
        track_cross, _ = self.track_to_rd(
            track_tokens,
            rd_tokens,
            rd_tokens,
            key_padding_mask=safe_rd_mask,
            need_weights=False,
        )
        rd_cross, _ = self.rd_to_track(
            rd_tokens,
            track_tokens,
            track_tokens,
            key_padding_mask=safe_track_mask,
            need_weights=False,
        )
        track_tokens = self.track_norm1(track_tokens + track_cross)
        rd_tokens = self.rd_norm1(rd_tokens + rd_cross)
        track_tokens = self.track_norm2(track_tokens + self.track_ffn(track_tokens))
        rd_tokens = self.rd_norm2(rd_tokens + self.rd_ffn(rd_tokens))
        if track_padding_mask is not None:
            track_tokens = track_tokens.masked_fill(track_padding_mask.bool().unsqueeze(-1), 0.0)
        if rd_padding_mask is not None:
            rd_tokens = rd_tokens.masked_fill(rd_padding_mask.bool().unsqueeze(-1), 0.0)
        return track_tokens, rd_tokens



@dataclass(frozen=True)
class RDDroneBirdDiagnosticOutput:
    """RD-only Drone/Bird diagnostic output; never a fusion decision."""

    rd_features: torch.Tensor
    bird_logits: torch.Tensor
    micro_doppler_logits: torch.Tensor


@dataclass(frozen=True)
class LowAltitudeFusionOutput:
    """
    Expert1 V2 输出：无 RD 样本走 trajectory-only，有 RD 样本走 TR+RD joint；RD-only heads 仅作为 evidence/diagnostic。
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
    traj_residual_logits: torch.Tensor
    rd_residual_logits: torch.Tensor
    rd_unknown_clutter_logits: torch.Tensor
    rd_clutter_morphology_logits: torch.Tensor
    rd_class_fusion_weights: torch.Tensor | None
    rd_pair_fusion_weights: torch.Tensor | None
    known_confidence: torch.Tensor
    residual_score: torch.Tensor
    logits: torch.Tensor
    fusion_weights: torch.Tensor
    tr_rd_active: torch.Tensor
    rejected_unknown: torch.Tensor
    reject_score: torch.Tensor | None = None
    reject_reasons: dict[str, torch.Tensor] | None = None
    # Final TR/TR+RD fusion logits before the attribution-reject action and
    # before precision logit bias.  ``logits`` remains the historical
    # post-reject output for backward compatibility; consumers that evaluate
    # accepted classification must use this field together with
    # ``rejected_unknown`` instead of treating UnknownTarget as rejection.
    pre_reject_logits: torch.Tensor | None = None
    inference_mode: str = "v2_joint"
    joint_logits: torch.Tensor | None = None
    joint_reliability_logits: torch.Tensor | None = None
    joint_rd_weight: torch.Tensor | None = None
    joint_track_weight: torch.Tensor | None = None
    drone_bird_micro_doppler_logits: torch.Tensor | None = None
    traj_drone_bird_logits: torch.Tensor | None = None

    @property
    def rd_activated(self) -> torch.Tensor:
        """Backward-compatible alias; V2 semantics are TR+RD path presence."""
        return self.tr_rd_active



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
        traj_sequence_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        cosface_scale: float = 16.0,
        cosface_margin: float = 0.2,
        reject_threshold: float = 0.35,
        attribution_reject_min_known_confidence: float = 0.0,
        attribution_reject_min_known_margin: float = 0.0,
        attribution_reject_min_track_quality: float = 0.0,
        trajectory_main_sequence_enabled: bool = False,
        traj_physical_dim: int = 32,
        rd_binary_logit_scale: float = 0.75,
        rd_bird_logit_scale: float | None = None,
        rd_unknown_logit_scale: float | None = None,
        rd_clutter_logit_scale: float | None = None,
        rd_residual_logit_scale: float = 0.9,
        rd_unknown_clutter_logit_scale: float = 1.1,
        rd_clutter_morphology_logit_scale: float = 0.9,
        joint_logit_delta_scale: float = 1.0,
        rd_specialist_residual_scale: float = 0.35,
        joint_agreement_prior_scale: float = 0.25,
        rd_joint_segment_count: int = 5,
        enable_rd_segment_joint_tokens: bool = True,
        micro_doppler_feature_clip: float = 8.0,
        use_rd_continuous_time_encoding: bool = False,
        rd_time_encoding_mode: str = "absolute_elapsed",
        rd_local_delta_time_clip_seconds: float = 1.0,
        rd_local_delta_time_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.reject_threshold = reject_threshold
        self.attribution_reject_min_known_confidence = float(max(0.0, attribution_reject_min_known_confidence))
        self.attribution_reject_min_known_margin = float(max(0.0, attribution_reject_min_known_margin))
        self.attribution_reject_min_track_quality = float(max(0.0, attribution_reject_min_track_quality))
        self.trajectory_main_sequence_enabled = bool(trajectory_main_sequence_enabled)
        self.traj_physical_dim = int(max(1, traj_physical_dim))
        self.rd_binary_logit_scale = rd_binary_logit_scale
        self.rd_bird_logit_scale = float(rd_binary_logit_scale if rd_bird_logit_scale is None else rd_bird_logit_scale)
        self.rd_unknown_logit_scale = float(rd_binary_logit_scale if rd_unknown_logit_scale is None else rd_unknown_logit_scale)
        self.rd_clutter_logit_scale = float(rd_binary_logit_scale if rd_clutter_logit_scale is None else rd_clutter_logit_scale)
        self.rd_residual_logit_scale = float(rd_residual_logit_scale)
        self.rd_unknown_clutter_logit_scale = float(rd_unknown_clutter_logit_scale)
        self.rd_clutter_morphology_logit_scale = float(rd_clutter_morphology_logit_scale)
        self.joint_logit_delta_scale = float(joint_logit_delta_scale)
        self.rd_specialist_residual_scale = float(rd_specialist_residual_scale)
        self.joint_agreement_prior_scale = float(joint_agreement_prior_scale)
        self.precision_logit_bias: list[float] | None = None
        self.rd_joint_segment_count = int(max(1, rd_joint_segment_count))
        self.enable_rd_segment_joint_tokens = bool(enable_rd_segment_joint_tokens)
        self.micro_doppler_feature_clip = float(max(1.0e-3, micro_doppler_feature_clip))
        self.use_rd_continuous_time_encoding = bool(use_rd_continuous_time_encoding)
        supported_rd_time_encoding_modes = {"absolute_elapsed", "normalized_order_local_delta"}
        if rd_time_encoding_mode not in supported_rd_time_encoding_modes:
            raise ValueError(
                "unsupported rd_time_encoding_mode: "
                f"{rd_time_encoding_mode}; expected one of {sorted(supported_rd_time_encoding_modes)}"
            )
        self.rd_time_encoding_mode = str(rd_time_encoding_mode)
        self.rd_local_delta_time_clip_seconds = float(max(1.0e-6, rd_local_delta_time_clip_seconds))
        self.rd_local_delta_time_scale = float(max(0.0, rd_local_delta_time_scale))
        self.known_indices = [
            LOW_ALTITUDE_CLASSES.index("DroneTarget"),
            LOW_ALTITUDE_CLASSES.index("BirdTarget"),
            LOW_ALTITUDE_CLASSES.index("BalloonTarget"),
        ]
        self.residual_indices = [
            LOW_ALTITUDE_CLASSES.index("UnknownTarget"),
            LOW_ALTITUDE_CLASSES.index("ClutterTarget"),
        ]
        self.rd_model_dim = 512
        self.rd_stats_input_dim = 16
        self.rd_stats_dim = 64
        self.rd_context_dim = int(max(0, rd_context_dim))
        self.rd_context_hidden_dim = 32 if self.rd_context_dim > 0 else 0
        self.rd_patch_dim = 64 if use_rd_patch_cnn else 0
        self.rd_patch_input_dim = int(max(1, rd_patch_input_dim))
        self.traj_sequence_dim = int(max(1, traj_sequence_dim))
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
        if self.trajectory_main_sequence_enabled:
            self.traj_sequence_main_pool = TemporalAttentionPool(
                self.traj_sequence_dim,
                hidden_dim=max(32, self.traj_sequence_dim // 2),
            )
            self.traj_sequence_main_proj = nn.Sequential(
                nn.Linear(self.traj_sequence_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.traj_physical_main_gate = nn.Sequential(
                nn.Linear(self.traj_physical_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.traj_main_norm = nn.Identity()
            self.traj_drone_bird_head = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(self.traj_sequence_main_proj[-1].weight)
            nn.init.zeros_(self.traj_sequence_main_proj[-1].bias)
        else:
            self.traj_sequence_main_pool = None
            self.traj_sequence_main_proj = None
            self.traj_physical_main_gate = None
            self.traj_main_norm = None
            self.traj_drone_bird_head = None
        self.traj_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
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
            nn.Linear(self.rd_stats_input_dim, self.rd_stats_dim),
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
            self.deep_rd_patch_dim = 96
            self.deep_rd_patch_cnn = nn.Sequential(
                nn.Conv2d(self.rd_patch_input_dim, 24, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(48, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, self.deep_rd_patch_dim),
                nn.GELU(),
                nn.LayerNorm(self.deep_rd_patch_dim),
            )
            self.deep_rd_patch_temporal_pool = TemporalAttentionPool(self.deep_rd_patch_dim, hidden_dim=96)
            self.deep_rd_patch_fuse = nn.Sequential(
                nn.Linear(self.deep_rd_patch_dim * 3, self.deep_rd_patch_dim),
                nn.GELU(),
                nn.LayerNorm(self.deep_rd_patch_dim),
            )
        else:
            self.rd_patch_cnn = None
            self.rd_patch_temporal_pool = None
            self.deep_rd_patch_dim = 0
            self.deep_rd_patch_cnn = None
            self.deep_rd_patch_temporal_pool = None
            self.deep_rd_patch_fuse = None
        self.rd_backbone = nn.Sequential(
            nn.Linear(self.rd_model_dim + self.rd_stats_dim + self.rd_context_hidden_dim + self.rd_patch_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
        )
        self.deep_rd_refine = nn.Sequential(
            nn.Linear(512 + self.deep_rd_patch_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
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
        self.rd_residual_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.rd_unknown_clutter_head = nn.Linear(512, 2)
        self.rd_clutter_morphology_head = nn.Sequential(
            nn.Linear(self.rd_stats_dim + self.rd_patch_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 2),
        )
        # Drone/Bird micro-Doppler auxiliary head (6 features → 2-class: Drone vs Bird)
        self.drone_bird_micro_doppler_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self.joint_dim = 512
        if hidden_dim == self.joint_dim:
            self.joint_traj_proj = nn.LayerNorm(self.joint_dim)
        else:
            self.joint_traj_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.joint_dim),
                nn.GELU(),
                nn.LayerNorm(self.joint_dim),
            )
        self.joint_rd_proj = nn.LayerNorm(self.joint_dim)
        self.joint_track_token_proj = nn.Sequential(
            nn.LayerNorm(self.traj_sequence_dim),
            nn.Linear(self.traj_sequence_dim, self.joint_dim),
            nn.GELU(),
            nn.LayerNorm(self.joint_dim),
        )
        self.joint_fusion = CrossModalFusionBlock(self.joint_dim, num_heads=8, dropout=dropout)
        self.joint_quality_dim = 12
        self.joint_quality_proj = nn.Sequential(
            nn.Linear(self.joint_quality_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        joint_input_dim = self.joint_dim * 6 + self.rd_stats_dim + self.rd_patch_dim + 64
        self.joint_head = nn.Sequential(
            nn.LayerNorm(joint_input_dim),
            nn.Linear(joint_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(LOW_ALTITUDE_CLASSES)),
        )
        self.joint_reliability_head = nn.Sequential(
            nn.LayerNorm(joint_input_dim),
            nn.Linear(joint_input_dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 2),
        )
        self.deep_pair_correction_head = nn.Sequential(
            nn.Linear(512 + hidden_dim + self.deep_rd_patch_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(LOW_ALTITUDE_CLASSES)),
        )
        self.deep_unknown_clutter_residual_head = nn.Sequential(
            nn.Linear(512 + self.rd_stats_dim + self.deep_rd_patch_dim, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 3),
        )
        with torch.no_grad():
            nn.init.zeros_(self.joint_reliability_head[-1].weight)
            self.joint_reliability_head[-1].bias.copy_(
                torch.tensor([1.0, -1.0], dtype=self.joint_reliability_head[-1].bias.dtype)
            )

    def _rd_statistics(self, amp_range: torch.Tensor, phase_range: torch.Tensor, amp_dopp: torch.Tensor, phase_dopp: torch.Tensor) -> torch.Tensor:
        # amp_range: [B, T, R], phase_range: [B, T, R], amp_dopp: [B, T, D], phase_dopp: [B, T, D]
        B = amp_range.shape[0]

        # --- Doppler validity detection: 2D data has amp_dopp all zeros ---
        doppler_valid = (amp_dopp.abs().sum(dim=(1, 2)) > 0.5).float().unsqueeze(-1)  # [B, 1]

        amp_mean = amp_range.mean(dim=-1, keepdim=True)
        amp_std = amp_range.std(dim=-1, keepdim=True).clamp_min(1e-6)
        centered = amp_range - amp_mean
        amp_skew = (centered.pow(3).mean(dim=-1) / amp_std.squeeze(-1).pow(3)).mean(dim=-1, keepdim=True)
        amp_kurt = (centered.pow(4).mean(dim=-1) / amp_std.squeeze(-1).pow(4)).mean(dim=-1, keepdim=True)
        phase_abs_mean = phase_dopp.abs().mean(dim=-1).mean(dim=-1, keepdim=True)
        phase_std = phase_dopp.std(dim=-1).mean(dim=-1, keepdim=True)
        range_energy = amp_range.abs().clamp_min(0.0) + 1e-6
        doppler_energy = amp_dopp.abs().clamp_min(0.0) + 1e-6
        range_prob = range_energy / range_energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        doppler_prob = doppler_energy / doppler_energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        range_entropy = (-(range_prob * range_prob.log()).sum(dim=-1) / math.log(float(range_prob.shape[-1]))).mean(dim=-1, keepdim=True)
        doppler_entropy = (-(doppler_prob * doppler_prob.log()).sum(dim=-1) / math.log(float(doppler_prob.shape[-1]))).mean(dim=-1, keepdim=True)
        peak_to_mean = (range_energy.max(dim=-1).values / range_energy.mean(dim=-1).clamp_min(1e-6)).mean(dim=-1, keepdim=True)
        if amp_range.shape[1] > 1:
            temporal_roughness = (amp_range[:, 1:] - amp_range[:, :-1]).abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        else:
            temporal_roughness = amp_range.new_zeros((amp_range.shape[0], 1))

        # --- New features: range/doppler centroid dynamics ---
        range_bins = torch.arange(1, range_energy.shape[-1] + 1, device=range_energy.device, dtype=range_energy.dtype)
        doppler_bins = torch.arange(1, doppler_energy.shape[-1] + 1, device=doppler_energy.device, dtype=doppler_energy.dtype)
        range_centroid = (range_energy * range_bins).sum(dim=-1) / range_energy.sum(dim=-1).clamp_min(1e-6)  # [B, T]
        doppler_centroid = (doppler_energy * doppler_bins).sum(dim=-1) / doppler_energy.sum(dim=-1).clamp_min(1e-6)  # [B, T]
        range_centroid_std = range_centroid.std(dim=-1, keepdim=True, unbiased=False)  # [B, 1]
        doppler_centroid_std = doppler_centroid.std(dim=-1, keepdim=True, unbiased=False)  # [B, 1]

        # --- Temporal autocorrelation of frame energy ---
        frame_energy = amp_range.mean(dim=-1)  # [B, T]
        if frame_energy.shape[1] > 2:
            fe_mean = frame_energy.mean(dim=-1, keepdim=True)
            fe_centered = frame_energy - fe_mean
            fe_var = fe_centered.pow(2).mean(dim=-1).clamp_min(1e-8)
            autocorr_lag1 = (fe_centered[:, 1:] * fe_centered[:, :-1]).mean(dim=-1) / fe_var  # [B]
            temporal_autocorr = autocorr_lag1.unsqueeze(-1)
        else:
            temporal_autocorr = amp_range.new_zeros((amp_range.shape[0], 1))

        # --- Spectral bandwidth of range energy ---
        range_energy_mean = range_energy.mean(dim=1)  # [B, R]
        re_norm = range_energy_mean / range_energy_mean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        range_centroid_global = (re_norm * range_bins).sum(dim=-1, keepdim=True)
        spectral_bandwidth = ((re_norm * (range_bins - range_centroid_global).pow(2)).sum(dim=-1)).sqrt().unsqueeze(-1)

        # --- Energy temporal burstiness (std/mean of frame energy) ---
        energy_burstiness = (frame_energy.std(dim=-1, unbiased=False) / frame_energy.mean(dim=-1).clamp_min(1e-6)).unsqueeze(-1)

        # --- Top-bin concentration: fraction of energy in top 10% bins ---
        range_sorted, _ = range_energy.mean(dim=1).sort(dim=-1, descending=True)
        top_k_r = max(1, range_sorted.shape[-1] // 10)
        range_top10_conc = range_sorted[:, :top_k_r].sum(dim=-1) / range_sorted.sum(dim=-1).clamp_min(1e-6)
        range_top10_conc = range_top10_conc.unsqueeze(-1)

        doppler_sorted, _ = doppler_energy.mean(dim=1).sort(dim=-1, descending=True)
        top_k_d = max(1, doppler_sorted.shape[-1] // 10)
        doppler_top10_conc = doppler_sorted[:, :top_k_d].sum(dim=-1) / doppler_sorted.sum(dim=-1).clamp_min(1e-6)
        doppler_top10_conc = doppler_top10_conc.unsqueeze(-1)

        # --- Zero out doppler features for 2D data (no doppler dimension) ---
        doppler_entropy = doppler_entropy * doppler_valid
        doppler_centroid_std = doppler_centroid_std * doppler_valid
        doppler_top10_conc = doppler_top10_conc * doppler_valid

        stats = torch.cat(
            [
                amp_skew,               # 0: amplitude skewness
                amp_kurt,               # 1: amplitude kurtosis
                phase_abs_mean,         # 2: phase coherence
                phase_std,              # 3: phase stability
                range_entropy,          # 4: range energy spread
                doppler_entropy,        # 5: doppler energy spread
                peak_to_mean,           # 6: range peak spikiness
                temporal_roughness,     # 7: frame-to-frame amplitude change
                range_centroid_std,     # 8: range centroid temporal variation
                doppler_centroid_std,   # 9: doppler centroid temporal variation
                temporal_autocorr,      # 10: frame energy lag-1 autocorrelation
                spectral_bandwidth,     # 11: range spectral bandwidth
                energy_burstiness,      # 12: temporal energy burstiness
                range_top10_conc,       # 13: range top-10% energy concentration
                doppler_top10_conc,     # 14: doppler top-10% energy concentration
                amp_range.new_zeros((amp_range.shape[0], 1)),  # 15: reserved
            ],
            dim=-1,
        )
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return stats

    def _micro_doppler_features(self, rd_features: torch.Tensor) -> torch.Tensor:
        """Extract 6 micro-Doppler features for Drone/Bird discrimination.
        Only meaningful for 3D RD data (CQ08). Returns [B, 6] features."""
        amp_range = rd_features[:, :, :, 0]   # [B, T, R]
        amp_dopp = rd_features[:, :, :, 2]    # [B, T, D]

        B = amp_range.shape[0]
        doppler_energy = amp_dopp.abs().clamp_min(0.0) + 1e-6

        # Doppler validity: 2D data has amp_dopp all zeros
        doppler_valid = (amp_dopp.abs().sum(dim=(1, 2)) > 0.5).float().unsqueeze(-1)  # [B, 1]

        # 1. Doppler spectral skewness
        dopp_mean = doppler_energy.mean(dim=-1, keepdim=True)
        dopp_std = doppler_energy.std(dim=-1, keepdim=True).clamp_min(1e-6)
        dopp_centered = doppler_energy - dopp_mean
        dopp_skew = (dopp_centered.pow(3).mean(dim=-1) / dopp_std.squeeze(-1).pow(3)).mean(dim=-1, keepdim=True)

        # 2. Doppler spectral kurtosis
        dopp_kurt = (dopp_centered.pow(4).mean(dim=-1) / dopp_std.squeeze(-1).pow(4)).mean(dim=-1, keepdim=True)

        # 3. Mean 3dB bandwidth (bins)
        dopp_peak = doppler_energy.max(dim=-1, keepdim=True).values
        dopp_above_half = (doppler_energy >= dopp_peak * 0.5).float()
        dopp_bw = dopp_above_half.sum(dim=-1).float()  # [B, T]
        dopp_bw_mean = dopp_bw.mean(dim=-1, keepdim=True)

        # 4. Bandwidth temporal variation
        if dopp_bw.shape[1] > 1:
            dopp_bw_std = dopp_bw.std(dim=-1, keepdim=True, unbiased=False)
        else:
            dopp_bw_std = amp_range.new_zeros((B, 1))

        # 5. RCS scintillation (amplitude CV)
        frame_amp = amp_range.mean(dim=-1)  # [B, T]
        rcs_scint = (frame_amp.std(dim=-1, unbiased=False) / frame_amp.mean(dim=-1).clamp_min(1e-6)).unsqueeze(-1)

        # 6. Inter-frame RD correlation
        if amp_range.shape[1] > 1:
            amp_norm = amp_range - amp_range.mean(dim=-1, keepdim=True)
            amp_norm = amp_norm / amp_norm.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            frame_corr = (amp_norm[:, 1:] * amp_norm[:, :-1]).sum(dim=-1)  # [B, T-1]
            rd_corr = frame_corr.mean(dim=-1, keepdim=True)
        else:
            rd_corr = amp_range.new_zeros((B, 1))

        features = torch.cat([dopp_skew, dopp_kurt, dopp_bw_mean, dopp_bw_std, rcs_scint, rd_corr], dim=-1)
        features = features * doppler_valid  # zero out for 2D data
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.clamp(
            min=-self.micro_doppler_feature_clip,
            max=self.micro_doppler_feature_clip,
        )  # [B, 6]

    def _rd_continuous_time_encoding(
        self,
        rd_features: torch.Tensor,
        rd_time_seconds: torch.Tensor | None,
        rd_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, frame_count = rd_features.shape[:2]
        if rd_padding_mask is None:
            valid = torch.ones((batch_size, frame_count), device=rd_features.device, dtype=torch.bool)
        else:
            valid = ~rd_padding_mask
        if rd_time_seconds is None:
            times = torch.arange(frame_count, device=rd_features.device, dtype=rd_features.dtype).unsqueeze(0)
            times = times.expand(batch_size, -1)
        else:
            times = rd_time_seconds.to(device=rd_features.device, dtype=rd_features.dtype)
            if times.shape != (batch_size, frame_count):
                raise ValueError("rd_time_seconds must have shape [batch_size, time]")
        if self.rd_time_encoding_mode == "normalized_order_local_delta":
            valid_count = valid.sum(dim=1, keepdim=True).to(rd_features.dtype)
            order = valid.to(rd_features.dtype).cumsum(dim=1) - 1.0
            order_denom = (valid_count - 1.0).clamp_min(1.0)
            normalized_order = torch.where(
                valid,
                order / order_denom,
                torch.zeros_like(order),
            )
            local_delta = torch.zeros_like(times)
            if frame_count > 1:
                adjacent_delta = (times[:, 1:] - times[:, :-1]).clamp_min(0.0)
                adjacent_valid = valid[:, 1:] & valid[:, :-1]
                local_delta[:, 1:] = torch.where(adjacent_valid, adjacent_delta, torch.zeros_like(adjacent_delta))
            normalized_delta = local_delta / self.rd_local_delta_time_clip_seconds
            positions = normalized_order + torch.log1p(normalized_delta.clamp_max(1.0)).mul(
                self.rd_local_delta_time_scale
            )
        else:
            first_index = valid.to(torch.int64).argmax(dim=1, keepdim=True)
            first_time = times.gather(1, first_index)
            relative_time = torch.where(valid, (times - first_time).clamp_min(0.0), torch.zeros_like(times))
            positions = torch.log1p(relative_time)
        half_dim = self.rd_model_dim // 2
        frequency_index = torch.arange(half_dim, device=rd_features.device, dtype=rd_features.dtype)
        denominator = max(1, half_dim - 1)
        frequencies = torch.exp(-math.log(10_000.0) * frequency_index / denominator)
        angles = positions.unsqueeze(-1) * frequencies
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if encoding.shape[-1] < self.rd_model_dim:
            encoding = F.pad(encoding, (0, self.rd_model_dim - encoding.shape[-1]))
        return encoding * valid.unsqueeze(-1).to(encoding.dtype)

    def _encode_rd(
        self,
        rd_features: torch.Tensor,
        rd_padding_mask: torch.Tensor | None = None,
        rd_time_seconds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        amp_range = rd_features[:, :, :, 0]
        phase_range = rd_features[:, :, :, 1]
        amp_dopp = rd_features[:, :, :, 2]
        phase_dopp = rd_features[:, :, :, 3]
        amp_range_embed = self.range_amplitude_proj(amp_range)
        phase_range_embed = self.range_phase_proj(phase_range)
        amp_dopp_embed = self.doppler_amplitude_proj(amp_dopp)
        phase_dopp_embed = self.doppler_phase_proj(phase_dopp)
        hidden = self.rd_token_proj(torch.cat([amp_range_embed, phase_range_embed, amp_dopp_embed, phase_dopp_embed], dim=-1))
        if self.use_rd_continuous_time_encoding:
            hidden = hidden + self._rd_continuous_time_encoding(rd_features, rd_time_seconds, rd_padding_mask)
        hidden = self.rd_encoder(hidden, src_key_padding_mask=rd_padding_mask)
        hidden = self.rd_norm(hidden)
        if rd_padding_mask is None:
            encoded = hidden.mean(dim=1)
        else:
            valid = (~rd_padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            encoded = (hidden * valid).sum(dim=1) / denom
        stats = self.stats_proj(self._rd_statistics(amp_range, phase_range, amp_dopp, phase_dopp))
        return encoded, stats, hidden

    def rd_drone_bird_diagnostic(
        self,
        rd_features: torch.Tensor,
        rd_patch_features: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        rd_time_seconds: torch.Tensor | None = None,
    ) -> RDDroneBirdDiagnosticOutput:
        """Run the RD trunk for a diagnostic-only Drone/Bird experiment.

        This method deliberately does not construct trajectory logits, Joint logits,
        reliability weights, or final fusion logits.  Callers must keep its outputs
        separate from deployment fusion until timing evidence independently passes.
        """
        if rd_features.ndim != 4:
            raise ValueError("rd_features must have shape [batch_size, time, range, channel]")
        rd_encoded, rd_stats, _ = self._encode_rd(
            rd_features,
            rd_padding_mask=rd_padding_mask,
            rd_time_seconds=rd_time_seconds,
        )
        patch_source = rd_patch_features if rd_patch_features is not None else rd_features
        patch_mask = rd_patch_padding_mask if rd_patch_features is not None else rd_padding_mask
        rd_patch = self._encode_rd_patch(patch_source, rd_padding_mask=patch_mask)
        context = self._encode_rd_context(
            None,
            batch_size=int(rd_features.shape[0]),
            device=rd_features.device,
            dtype=rd_features.dtype,
        )
        rd_features_vec = self.rd_backbone(torch.cat([rd_encoded, rd_stats, context, rd_patch], dim=-1))
        bird_logits = self.rd_bird_head(rd_features_vec)
        micro_doppler_logits = self.drone_bird_micro_doppler_head(
            self._micro_doppler_features(rd_features)
        )
        return RDDroneBirdDiagnosticOutput(
            rd_features=rd_features_vec,
            bird_logits=bird_logits,
            micro_doppler_logits=micro_doppler_logits,
        )

    def _traj_joint_tokens(
        self,
        traj_features: torch.Tensor,
        traj_sequence_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if traj_sequence_features is None:
            return self.joint_traj_proj(traj_features).unsqueeze(1)
        return self.joint_track_token_proj(traj_sequence_features)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is None:
            return tokens.mean(dim=1)
        valid = (~padding_mask.bool()).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (tokens * valid).sum(dim=1) / denom

    def _segment_rd_tokens(
        self,
        rd_tokens: torch.Tensor,
        rd_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.enable_rd_segment_joint_tokens or self.rd_joint_segment_count <= 1:
            return rd_tokens, rd_padding_mask
        batch_size, time_steps, _ = rd_tokens.shape
        segment_count = min(self.rd_joint_segment_count, max(1, time_steps))
        if segment_count >= time_steps and rd_padding_mask is None:
            return rd_tokens, rd_padding_mask
        if rd_padding_mask is None:
            valid = torch.ones((batch_size, time_steps), dtype=torch.bool, device=rd_tokens.device)
        else:
            valid = ~rd_padding_mask.bool()
            if valid.shape[1] != time_steps:
                if valid.shape[1] > time_steps:
                    valid = valid[:, :time_steps]
                else:
                    pad = torch.zeros(
                        (batch_size, time_steps - valid.shape[1]),
                        dtype=torch.bool,
                        device=valid.device,
                    )
                    valid = torch.cat([valid, pad], dim=1)
        valid_lengths = valid.sum(dim=1).clamp_min(1)
        positions = torch.arange(time_steps, dtype=torch.long, device=rd_tokens.device).view(1, time_steps)
        pooled_segments: list[torch.Tensor] = []
        segment_masks: list[torch.Tensor] = []
        for segment_index in range(segment_count):
            start = torch.div(valid_lengths * segment_index, segment_count, rounding_mode="floor")
            end = torch.div(valid_lengths * (segment_index + 1), segment_count, rounding_mode="floor")
            segment_active = end > start
            segment_selector = valid & (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1))
            weights = segment_selector.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled = (rd_tokens * weights).sum(dim=1) / denom
            pooled = torch.where(segment_active.unsqueeze(-1), pooled, torch.zeros_like(pooled))
            pooled_segments.append(pooled)
            segment_masks.append(~segment_active)
        return torch.stack(pooled_segments, dim=1), torch.stack(segment_masks, dim=1)

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

    def _encode_deep_rd_patch(self, rd_features: torch.Tensor, rd_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.deep_rd_patch_cnn is None:
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
            encoded = torch.cat([self.deep_rd_patch_cnn(chunk) for chunk in frames.split(32, dim=0)], dim=0)
            frame_features = encoded.view(batch_size, time_steps, self.deep_rd_patch_dim)
            if rd_padding_mask is not None:
                valid = ~rd_padding_mask.bool()
            else:
                valid = torch.ones((batch_size, time_steps), dtype=torch.bool, device=frame_features.device)
            valid_float = valid.float().unsqueeze(-1)
            mean_pool = (frame_features * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
            max_pool = frame_features.masked_fill(~valid.unsqueeze(-1), -1.0e4).max(dim=1).values
            max_pool = torch.where(valid.any(dim=1).unsqueeze(-1), max_pool, torch.zeros_like(max_pool))
            if self.deep_rd_patch_temporal_pool is None:
                attn_pool = mean_pool
            else:
                attn_pool = self.deep_rd_patch_temporal_pool(frame_features, padding_mask=~valid)
            if self.deep_rd_patch_fuse is None:
                return attn_pool
            return self.deep_rd_patch_fuse(torch.cat([attn_pool, mean_pool, max_pool], dim=-1))
        if patch_input.ndim == 4:
            patch_input = patch_input.permute(0, 3, 1, 2).contiguous()
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
            return self.deep_rd_patch_cnn(patch_input)
        raise ValueError("rd patch features must be [B,T,R,D,C] or [B,T,R,C]")

    def _apply_reject(
        self,
        logits: torch.Tensor,
        cosine: torch.Tensor,
        activated: torch.Tensor,
        track_quality: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return apply_low_altitude_attribution_reject(
            logits,
            cosine,
            activated,
            self.reject_threshold,
            track_quality=track_quality,
            min_known_confidence=self.attribution_reject_min_known_confidence,
            min_known_margin=self.attribution_reject_min_known_margin,
            min_track_quality=self.attribution_reject_min_track_quality,
        )

    def set_precision_logit_bias(self, bias: object) -> None:
        if bias is None:
            self.precision_logit_bias = None
            return
        if isinstance(bias, dict):
            values = [float(bias.get(label_name, 0.0)) for label_name in LOW_ALTITUDE_CLASSES]
        else:
            values = [float(item) for item in bias]  # type: ignore[arg-type]
        if len(values) != len(LOW_ALTITUDE_CLASSES):
            raise ValueError(f"precision_logit_bias must have {len(LOW_ALTITUDE_CLASSES)} values")
        self.precision_logit_bias = values

    def _apply_precision_logit_bias(self, logits: torch.Tensor) -> torch.Tensor:
        bias = getattr(self, "precision_logit_bias", None)
        if not bias:
            return logits
        bias_tensor = torch.tensor(bias, dtype=logits.dtype, device=logits.device)
        if bias_tensor.numel() != logits.shape[-1] or bool((bias_tensor == 0).all().item()):
            return logits
        return logits + bias_tensor.unsqueeze(0)

    def acceptance(self, output: LowAltitudeFusionOutput, threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        traj_confidence = output.traj_cosine.max(dim=-1).values
        final_confidence = torch.softmax(output.logits, dim=-1).max(dim=-1).values
        confidence = torch.where(output.tr_rd_active, torch.maximum(traj_confidence, final_confidence), traj_confidence)
        threshold_value = float(self.reject_threshold if threshold is None else threshold)
        unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
        predicted_unknown = output.logits.argmax(dim=-1) == unknown_index
        reject_score = output.reject_score
        if reject_score is None:
            reject_score = torch.zeros_like(confidence)
        if output.inference_mode == "deep_reroute":
            residual_accept = predicted_unknown & (output.residual_score >= 0.55)
            known_accept = (~predicted_unknown) & (confidence >= threshold_value) & (reject_score < 1.25)
            return confidence, known_accept | residual_accept
        accepted = (confidence >= threshold_value) & (~predicted_unknown) & (reject_score < 0.75)
        return confidence, accepted

    def forward(
        self,
        traj_input: torch.Tensor,
        rd_features: torch.Tensor,
        traj_sequence_features: torch.Tensor | None = None,
        rd_patch_features: torch.Tensor | None = None,
        rd_deep_patch_features: torch.Tensor | None = None,
        rd_context: torch.Tensor | None = None,
        low_altitude_targets: torch.Tensor | None = None,
        rd_is_multimodal: torch.Tensor | None = None,
        rd_alignment_quality: torch.Tensor | None = None,
        traj_padding_mask: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_time_seconds: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        rd_deep_patch_padding_mask: torch.Tensor | None = None,
        traj_physical_features: torch.Tensor | None = None,
        track_quality: torch.Tensor | None = None,
        enable_reject: bool = True,
        disable_tr_rd_fusion: bool = False,
        inference_mode: str = "v2_joint",
    ) -> LowAltitudeFusionOutput:
        if inference_mode not in {"v2_joint", "deep_reroute"}:
            raise ValueError(f"unsupported low-altitude inference mode: {inference_mode}")
        deep_reroute = inference_mode == "deep_reroute"
        traj_features = self.traj_backbone(traj_input)
        if self.trajectory_main_sequence_enabled:
            if traj_sequence_features is None:
                sequence_main = traj_features.new_zeros(traj_features.shape)
            else:
                if traj_sequence_features.ndim != 3 or traj_sequence_features.shape[0] != traj_features.shape[0]:
                    raise ValueError("traj_sequence_features must have shape [batch_size, time, feature_dim]")
                if traj_sequence_features.shape[-1] != self.traj_sequence_dim:
                    raise ValueError("traj_sequence_features feature dimension does not match traj_sequence_dim")
                if self.traj_sequence_main_pool is None or self.traj_sequence_main_proj is None:
                    raise RuntimeError("trajectory main sequence modules are unavailable")
                sequence_main = self.traj_sequence_main_proj(
                    self.traj_sequence_main_pool(traj_sequence_features, padding_mask=traj_padding_mask)
                )
            if traj_physical_features is None:
                physical_main = traj_features.new_zeros((traj_features.shape[0], self.traj_physical_dim))
            else:
                physical_main = traj_physical_features.to(device=traj_features.device, dtype=traj_features.dtype)
                if physical_main.ndim != 2 or physical_main.shape[0] != traj_features.shape[0]:
                    raise ValueError("traj_physical_features must have shape [batch_size, feature_dim]")
                if physical_main.shape[-1] != self.traj_physical_dim:
                    raise ValueError("traj_physical_features dimension does not match traj_physical_dim")
            if self.traj_physical_main_gate is None or self.traj_main_norm is None:
                raise RuntimeError("trajectory main physical modules are unavailable")
            traj_features = self.traj_main_norm(
                traj_features + self.traj_physical_main_gate(physical_main) * sequence_main
            )
        traj_cosine, traj_logits = self.traj_head(traj_features, labels=low_altitude_targets)
        traj_drone_bird_logits = (
            self.traj_drone_bird_head(traj_features)
            if self.traj_drone_bird_head is not None
            else None
        )
        if rd_is_multimodal is not None:
            has_rd = rd_is_multimodal.bool()
        else:
            has_rd = torch.zeros((traj_input.shape[0],), dtype=torch.bool, device=traj_input.device)
        has_rd = has_rd.bool()
        if rd_alignment_quality is None:
            alignment_quality = has_rd.float()
        else:
            alignment_quality = rd_alignment_quality.to(device=traj_input.device, dtype=traj_input.dtype)
            if alignment_quality.ndim != 1 or alignment_quality.shape[0] != traj_input.shape[0]:
                raise ValueError("rd_alignment_quality must have shape [batch_size]")
            alignment_quality = alignment_quality.clamp(min=0.0, max=1.0)
        alignment_quality = alignment_quality * has_rd.float()
        traj_residual_logits = self.traj_residual_head(traj_features)
        if disable_tr_rd_fusion or not bool(has_rd.any()):
            batch_size = traj_input.shape[0]
            num_classes = len(LOW_ALTITUDE_CLASSES)
            rd_features_vec = traj_input.new_zeros((batch_size, 512))
            rd_cosine = traj_input.new_zeros((batch_size, num_classes))
            rd_logits = traj_input.new_zeros((batch_size, num_classes))
            rd_bird_logits = traj_input.new_zeros((batch_size, 2))
            rd_unknown_logits = traj_input.new_zeros((batch_size, 2))
            rd_clutter_logits = traj_input.new_zeros((batch_size, 2))
            rd_residual_logits = traj_input.new_zeros((batch_size, 2))
            rd_unknown_clutter_logits = traj_input.new_zeros((batch_size, 2))
            rd_clutter_morphology_logits = traj_input.new_zeros((batch_size, 2))
            joint_logits = traj_logits
            joint_reliability_logits = traj_input.new_zeros((batch_size, 2))
            joint_rd_weight = traj_input.new_zeros((batch_size,))
            joint_track_weight = traj_input.new_ones((batch_size,))
            fusion_weights = torch.stack([joint_track_weight, joint_rd_weight], dim=-1)
            tr_rd_active = torch.zeros_like(has_rd, dtype=torch.bool)
            attribution_reasons: dict[str, torch.Tensor] = {}
            pre_reject_logits = traj_logits
            if enable_reject:
                final_logits, rejected_unknown, attribution_reject_score, attribution_reasons = self._apply_reject(
                    traj_logits,
                    traj_cosine,
                    torch.ones_like(has_rd, dtype=torch.bool),
                    track_quality=track_quality,
                )
            else:
                final_logits = traj_logits
                rejected_unknown = torch.zeros_like(has_rd, dtype=torch.bool)
                attribution_reject_score = traj_input.new_zeros((batch_size,))
            final_logits = self._apply_precision_logit_bias(final_logits)
            final_probs = torch.softmax(final_logits, dim=-1)
            top2 = final_probs.topk(k=2, dim=-1)
            final_margin = top2.values[:, 0] - top2.values[:, 1]
            unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
            known_confidence = torch.softmax(traj_logits, dim=-1)[:, self.known_indices].max(dim=-1).values
            residual_score = torch.softmax(traj_residual_logits, dim=-1)[:, 1]
            reject_score = attribution_reject_score + (
                (top2.indices[:, 0] == unknown_index).float()
                + (final_margin < 0.20).float() * 0.35
            )
            reject_reasons = {
                **attribution_reasons,
                "unknown_trigger": (top2.indices[:, 0] == unknown_index).detach(),
                "attribution_reject_trigger": rejected_unknown.detach(),
                "confusion_pair_trigger": torch.zeros_like(has_rd).detach(),
                "evidence_conflict_trigger": torch.zeros_like(has_rd).detach(),
                "low_margin_trigger": (final_margin < 0.20).detach(),
                "agreement_confidence_credit": torch.zeros_like(has_rd).detach(),
                "high_risk_pair": torch.zeros_like(has_rd).detach(),
                "traj_pred": torch.softmax(traj_logits, dim=-1).argmax(dim=-1).detach(),
                "rd_pred": torch.zeros_like(has_rd, dtype=torch.long).detach(),
                "final_margin": final_margin.detach(),
                "deep_pair_correction_max": traj_input.new_zeros((batch_size,)).detach(),
                "deep_unknown_clutter_pred": torch.zeros_like(has_rd, dtype=torch.long).detach(),
                "deep_unknown_clutter_confidence": traj_input.new_zeros((batch_size,)).detach(),
            }
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
                traj_residual_logits=traj_residual_logits,
                rd_residual_logits=rd_residual_logits,
                rd_unknown_clutter_logits=rd_unknown_clutter_logits,
                rd_clutter_morphology_logits=rd_clutter_morphology_logits,
                drone_bird_micro_doppler_logits=traj_input.new_zeros((batch_size, 2)),
                traj_drone_bird_logits=traj_drone_bird_logits,
                rd_class_fusion_weights=joint_rd_weight.unsqueeze(-1).expand_as(rd_logits),
                rd_pair_fusion_weights=joint_rd_weight.unsqueeze(-1).expand(-1, 3),
                known_confidence=known_confidence,
                residual_score=residual_score,
                logits=final_logits,
                fusion_weights=fusion_weights,
                tr_rd_active=tr_rd_active,
                rejected_unknown=rejected_unknown,
                reject_score=reject_score.detach(),
                reject_reasons=reject_reasons,
                pre_reject_logits=pre_reject_logits,
                inference_mode=inference_mode,
                joint_logits=joint_logits,
                joint_reliability_logits=joint_reliability_logits,
                joint_rd_weight=joint_rd_weight,
                joint_track_weight=joint_track_weight,
            )
        rd_context_features = self._encode_rd_context(
            rd_context,
            batch_size=traj_input.shape[0],
            device=traj_input.device,
            dtype=traj_input.dtype,
        )
        if rd_features.ndim == 4:
            rd_encoded, rd_stats, rd_tokens = self._encode_rd(
                rd_features,
                rd_padding_mask=rd_padding_mask,
                rd_time_seconds=rd_time_seconds,
            )
            patch_source = rd_patch_features if rd_patch_features is not None else rd_features
            patch_mask = rd_patch_padding_mask if rd_patch_features is not None else rd_padding_mask
            rd_patch = self._encode_rd_patch(patch_source, rd_padding_mask=patch_mask)
            if deep_reroute:
                deep_patch_source = rd_deep_patch_features if rd_deep_patch_features is not None else patch_source
                deep_patch_mask = rd_deep_patch_padding_mask if rd_deep_patch_features is not None else patch_mask
                deep_rd_patch = self._encode_deep_rd_patch(deep_patch_source, rd_padding_mask=deep_patch_mask)
            else:
                deep_rd_patch = traj_input.new_zeros((traj_input.shape[0], self.deep_rd_patch_dim))
            rd_features_vec = self.rd_backbone(torch.cat([rd_encoded, rd_stats, rd_context_features, rd_patch], dim=-1))
            if deep_reroute:
                rd_features_vec = self.deep_rd_refine(torch.cat([rd_features_vec, deep_rd_patch], dim=-1))
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
                    deep_rd_patch = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(deep_rd_patch), deep_rd_patch)
                    rd_tokens = torch.where(no_rd.view(-1, 1, 1), torch.zeros_like(rd_tokens), rd_tokens)
        else:
            rd_encoded = traj_input.new_zeros((traj_input.shape[0], self.rd_model_dim))
            rd_stats = traj_input.new_zeros((traj_input.shape[0], self.rd_stats_dim))
            rd_tokens = traj_input.new_zeros((traj_input.shape[0], 1, self.rd_model_dim))
            rd_patch = traj_input.new_zeros((traj_input.shape[0], self.rd_patch_dim))
            deep_rd_patch = traj_input.new_zeros((traj_input.shape[0], self.deep_rd_patch_dim))
            rd_features_vec = self.rd_backbone(torch.cat([rd_encoded, rd_stats, rd_context_features, rd_patch], dim=-1))
            if deep_reroute:
                rd_features_vec = self.deep_rd_refine(torch.cat([rd_features_vec, deep_rd_patch], dim=-1))
            rd_cosine, rd_logits = self.rd_head(rd_features_vec, labels=low_altitude_targets)
            rd_bird_logits = self.rd_bird_head(rd_features_vec)
            rd_unknown_logits = self.rd_unknown_head(rd_features_vec)
            rd_clutter_logits = self.rd_clutter_head(rd_features_vec)

        rd_residual_logits = self.rd_residual_head(rd_features_vec)
        rd_unknown_clutter_logits = self.rd_unknown_clutter_head(rd_features_vec)
        rd_clutter_morphology_logits = self.rd_clutter_morphology_head(torch.cat([rd_stats, rd_patch], dim=-1))
        # Drone/Bird micro-Doppler auxiliary head (only meaningful for 3D RD data)
        micro_doppler_features = self._micro_doppler_features(rd_features)  # [B, 6]
        drone_bird_micro_doppler_logits = self.drone_bird_micro_doppler_head(micro_doppler_features)  # [B, 2]
        if rd_is_multimodal is not None and not bool(rd_is_multimodal.all()):
            no_rd = ~rd_is_multimodal.bool()
            if no_rd.any():
                drone_bird_micro_doppler_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(drone_bird_micro_doppler_logits), drone_bird_micro_doppler_logits)
        deep_pair_correction_logits = self.deep_pair_correction_head(torch.cat([rd_features_vec, traj_features, deep_rd_patch], dim=-1))
        deep_unknown_clutter_residual_logits = self.deep_unknown_clutter_residual_head(torch.cat([rd_features_vec, rd_stats, deep_rd_patch], dim=-1))
        if rd_is_multimodal is not None and not bool(rd_is_multimodal.all()):
            no_rd = ~rd_is_multimodal.bool()
            if no_rd.any():
                rd_residual_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(rd_residual_logits), rd_residual_logits)
                rd_unknown_clutter_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(rd_unknown_clutter_logits), rd_unknown_clutter_logits)
                rd_clutter_morphology_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(rd_clutter_morphology_logits), rd_clutter_morphology_logits)
                deep_pair_correction_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(deep_pair_correction_logits), deep_pair_correction_logits)
                deep_unknown_clutter_residual_logits = torch.where(no_rd.unsqueeze(-1), torch.zeros_like(deep_unknown_clutter_residual_logits), deep_unknown_clutter_residual_logits)

        unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
        clutter_index = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
        traj_residual_prob = torch.softmax(traj_residual_logits, dim=-1)[:, 1]
        rd_residual_prob = torch.softmax(rd_residual_logits, dim=-1)[:, 1]
        residual_score = traj_residual_prob + alignment_quality * 0.55 * (rd_residual_prob - traj_residual_prob)
        known_confidence = torch.softmax(traj_logits, dim=-1)[:, self.known_indices].max(dim=-1).values
        residual_delta = torch.logit(residual_score.clamp(1e-4, 1.0 - 1e-4))
        rd_unknown_clutter_delta = rd_unknown_clutter_logits[:, 1] - rd_unknown_clutter_logits[:, 0]
        rd_clutter_morphology_delta = rd_clutter_morphology_logits[:, 1] - rd_clutter_morphology_logits[:, 0]
        rd_logits = rd_logits.clone()
        rd_logits[:, unknown_index] = (
            rd_logits[:, unknown_index]
            + self.rd_residual_logit_scale * residual_delta
            - 0.50 * self.rd_unknown_clutter_logit_scale * rd_unknown_clutter_delta
        )
        rd_logits[:, clutter_index] = (
            rd_logits[:, clutter_index]
            + self.rd_residual_logit_scale * residual_delta
            + self.rd_unknown_clutter_logit_scale * rd_unknown_clutter_delta
            + self.rd_clutter_morphology_logit_scale * rd_clutter_morphology_delta
        )
        if deep_reroute:
            deep_known_delta = deep_unknown_clutter_residual_logits[:, 0]
            deep_unknown_delta = deep_unknown_clutter_residual_logits[:, 1] - deep_known_delta
            deep_clutter_delta = deep_unknown_clutter_residual_logits[:, 2] - deep_known_delta
            rd_logits = rd_logits + 0.35 * deep_pair_correction_logits
            rd_logits[:, unknown_index] = rd_logits[:, unknown_index] + 0.55 * deep_unknown_delta
            rd_logits[:, clutter_index] = rd_logits[:, clutter_index] + 0.65 * deep_clutter_delta

        traj_probs = torch.softmax(traj_logits, dim=-1)
        rd_probs = torch.softmax(rd_logits, dim=-1)
        traj_confidence = traj_probs.max(dim=-1).values
        rd_confidence = rd_probs.max(dim=-1).values
        traj_pred = traj_probs.argmax(dim=-1)
        rd_pred = rd_probs.argmax(dim=-1)
        evidence_agreement = traj_pred == rd_pred
        rd_cosine_confidence = rd_cosine.max(dim=-1).values
        traj_entropy = -(traj_probs.clamp_min(1e-6) * traj_probs.clamp_min(1e-6).log()).sum(dim=-1) / math.log(float(traj_probs.shape[-1]))
        rd_entropy = -(rd_probs.clamp_min(1e-6) * rd_probs.clamp_min(1e-6).log()).sum(dim=-1) / math.log(float(rd_probs.shape[-1]))
        patch_norm = rd_patch.norm(dim=-1) if rd_patch.numel() else torch.zeros_like(traj_confidence)
        quality_features = torch.stack(
            [
                alignment_quality,
                traj_confidence,
                rd_confidence,
                rd_cosine_confidence,
                torch.abs(traj_confidence - rd_confidence),
                evidence_agreement.float(),
                residual_score,
                traj_entropy,
                rd_entropy,
                rd_stats.mean(dim=-1),
                rd_stats.std(dim=-1),
                patch_norm,
            ],
            dim=-1,
        )
        quality_features = torch.nan_to_num(quality_features, nan=0.0, posinf=0.0, neginf=0.0)
        joint_quality = self.joint_quality_proj(quality_features)
        joint_traj_anchor = self.joint_traj_proj(traj_features)
        joint_rd_anchor = self.joint_rd_proj(rd_features_vec)
        joint_track_tokens = self._traj_joint_tokens(traj_features, traj_sequence_features)
        if traj_sequence_features is None:
            joint_track_padding_mask = None
        else:
            joint_track_padding_mask = traj_padding_mask
        joint_rd_tokens = rd_tokens
        joint_rd_padding_mask = rd_padding_mask if rd_features.ndim == 4 else None
        joint_rd_tokens, joint_rd_padding_mask = self._segment_rd_tokens(joint_rd_tokens, joint_rd_padding_mask)
        fused_track_tokens, fused_rd_tokens = self.joint_fusion(
            joint_track_tokens,
            joint_rd_tokens,
            track_padding_mask=joint_track_padding_mask,
            rd_padding_mask=joint_rd_padding_mask,
        )
        joint_traj = self._masked_mean(fused_track_tokens, joint_track_padding_mask)
        joint_rd = self._masked_mean(fused_rd_tokens, joint_rd_padding_mask)
        joint_rd = torch.where(has_rd.unsqueeze(-1), joint_rd, torch.zeros_like(joint_rd))
        joint_input = torch.cat(
            [
                joint_traj_anchor,
                joint_rd_anchor,
                joint_traj,
                joint_rd,
                torch.abs(joint_traj - joint_rd),
                joint_traj * joint_rd,
                rd_stats,
                rd_patch,
                joint_quality,
            ],
            dim=-1,
        )
        joint_logits = self.joint_head(joint_input)
        if self.joint_agreement_prior_scale > 0.0:
            agreement_strength = (
                self.joint_agreement_prior_scale
                * alignment_quality
                * evidence_agreement.float()
                * torch.minimum(traj_confidence, rd_confidence)
            )
            agreement_prior = torch.zeros_like(joint_logits)
            agreement_prior.scatter_(1, traj_pred.unsqueeze(-1), agreement_strength.unsqueeze(-1))
            joint_logits = joint_logits + agreement_prior
        joint_reliability_logits = self.joint_reliability_head(joint_input)
        if deep_reroute:
            joint_reliability_logits = joint_reliability_logits.clone()
            joint_reliability_logits[:, 1] = joint_reliability_logits[:, 1] + 0.35
        joint_weights = torch.softmax(joint_reliability_logits, dim=-1)
        joint_track_weight = joint_weights[:, 0]
        joint_rd_weight = joint_weights[:, 1]

        if disable_tr_rd_fusion:
            tr_rd_active = torch.zeros_like(has_rd, dtype=torch.bool)
        else:
            tr_rd_active = has_rd
        if disable_tr_rd_fusion:
            joint_rd_weight = torch.zeros_like(joint_rd_weight)
        joint_rd_cap = 0.65 if deep_reroute else 0.45
        joint_rd_weight = torch.clamp(joint_rd_weight, min=0.0, max=joint_rd_cap)
        if rd_is_multimodal is not None:
            joint_rd_weight = torch.where(
                rd_is_multimodal.bool(),
                joint_rd_weight,
                torch.zeros_like(joint_rd_weight),
            )
        joint_rd_weight = joint_rd_weight * alignment_quality
        joint_rd_weight = torch.where(tr_rd_active, joint_rd_weight, torch.zeros_like(joint_rd_weight))
        joint_track_weight = 1.0 - joint_rd_weight
        fusion_weights = torch.stack([joint_track_weight, joint_rd_weight], dim=-1)
        rd_class_fusion_weights = joint_rd_weight.unsqueeze(-1).expand_as(rd_logits)
        rd_pair_fusion_weights = joint_rd_weight.unsqueeze(-1).expand(-1, 3)
        final_logits, rejected_unknown = compose_low_altitude_final_logits(
            traj_logits,
            joint_logits,
            rd_logits,
            traj_cosine,
            rd_cosine,
            rd_pair_fusion_weights,
            tr_rd_active,
            alignment_quality,
            joint_logit_delta_scale=self.joint_logit_delta_scale,
            rd_specialist_residual_scale=self.rd_specialist_residual_scale,
            residual_scale_multiplier=0.16 if deep_reroute else 0.10,
            enable_reject=False,
            reject_threshold=self.reject_threshold,
            force_pair_active=deep_reroute,
        )
        pre_reject_logits = final_logits
        attribution_reasons: dict[str, torch.Tensor] = {}
        if enable_reject:
            final_logits, rejected_unknown, attribution_reject_score, attribution_reasons = self._apply_reject(
                final_logits,
                traj_cosine,
                torch.ones_like(has_rd, dtype=torch.bool),
                track_quality=track_quality,
            )
        else:
            attribution_reject_score = traj_input.new_zeros((traj_input.shape[0],))
        final_logits = self._apply_precision_logit_bias(final_logits)
        final_probs = torch.softmax(final_logits, dim=-1)
        top2 = final_probs.topk(k=2, dim=-1)
        top1_idx = top2.indices[:, 0]
        top2_idx = top2.indices[:, 1]
        final_margin = top2.values[:, 0] - top2.values[:, 1]
        drone_bird = {LOW_ALTITUDE_CLASSES.index("DroneTarget"), LOW_ALTITUDE_CLASSES.index("BirdTarget")}
        unknown_clutter = {unknown_index, clutter_index}
        bird_clutter = {LOW_ALTITUDE_CLASSES.index("BirdTarget"), clutter_index}
        drone_unknown = {LOW_ALTITUDE_CLASSES.index("DroneTarget"), unknown_index}
        pair_ids = torch.stack([top1_idx, top2_idx], dim=-1)
        high_risk_pair = torch.zeros_like(has_rd, dtype=torch.bool)
        for pair in (drone_bird, unknown_clutter, bird_clutter, drone_unknown):
            pair_tensor = torch.tensor(sorted(pair), dtype=torch.long, device=traj_input.device)
            high_risk_pair |= torch.isin(pair_ids, pair_tensor).all(dim=-1)
        unknown_trigger = top1_idx == unknown_index
        evidence_conflict = has_rd & (traj_pred != rd_pred) & (rd_cosine.max(dim=-1).values >= 0.45)
        low_margin = final_margin < 0.20
        confusion_trigger = high_risk_pair & low_margin
        agreement_credit = has_rd & (traj_pred == rd_pred) & (final_margin >= 0.35)
        reject_score = attribution_reject_score + (
            unknown_trigger.float() * 1.00
            + confusion_trigger.float() * 0.75
            + evidence_conflict.float() * 0.55
            + low_margin.float() * 0.35
            - agreement_credit.float() * 0.50
        )
        reject_reasons = {
            **attribution_reasons,
            "unknown_trigger": unknown_trigger.detach(),
            "attribution_reject_trigger": rejected_unknown.detach(),
            "confusion_pair_trigger": confusion_trigger.detach(),
            "evidence_conflict_trigger": evidence_conflict.detach(),
            "low_margin_trigger": low_margin.detach(),
            "agreement_confidence_credit": agreement_credit.detach(),
            "high_risk_pair": high_risk_pair.detach(),
            "traj_pred": traj_pred.detach(),
            "rd_pred": rd_pred.detach(),
            "final_margin": final_margin.detach(),
            "deep_pair_correction_max": deep_pair_correction_logits.abs().max(dim=-1).values.detach(),
            "deep_unknown_clutter_pred": deep_unknown_clutter_residual_logits.argmax(dim=-1).detach(),
            "deep_unknown_clutter_confidence": torch.softmax(deep_unknown_clutter_residual_logits, dim=-1).max(dim=-1).values.detach(),
        }
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
            traj_residual_logits=traj_residual_logits,
            rd_residual_logits=rd_residual_logits,
            rd_unknown_clutter_logits=rd_unknown_clutter_logits,
            rd_clutter_morphology_logits=rd_clutter_morphology_logits,
            drone_bird_micro_doppler_logits=drone_bird_micro_doppler_logits,
            traj_drone_bird_logits=traj_drone_bird_logits,
            rd_class_fusion_weights=rd_class_fusion_weights,
            rd_pair_fusion_weights=rd_pair_fusion_weights,
            known_confidence=known_confidence,
            residual_score=residual_score,
            logits=final_logits,
            fusion_weights=fusion_weights,
            tr_rd_active=tr_rd_active,
            rejected_unknown=rejected_unknown,
            reject_score=reject_score.detach(),
            reject_reasons=reject_reasons,
            pre_reject_logits=pre_reject_logits,
            inference_mode=inference_mode,
            joint_logits=joint_logits,
            joint_reliability_logits=joint_reliability_logits,
            joint_rd_weight=joint_rd_weight,
            joint_track_weight=joint_track_weight,
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
        self.source_thresholds: dict[str, float] = {}

    def set_decision_threshold(self, value: float) -> None:
        self.decision_threshold.fill_(float(value))

    def set_source_thresholds(self, thresholds: dict[str, float] | None) -> None:
        self.source_thresholds = {
            str(key): float(value)
            for key, value in (thresholds or {}).items()
        }

    def _threshold_for_sources(self, batch_size: int, device: torch.device, dtype: torch.dtype, source_names: list[str] | None = None) -> torch.Tensor:
        default_threshold = self.decision_threshold.to(device=device, dtype=dtype)
        if not self.source_thresholds or not source_names:
            return default_threshold.expand(batch_size)
        values = [
            float(self.source_thresholds.get(str(source_name), float(default_threshold.item())))
            for source_name in source_names
        ]
        return torch.tensor(values, dtype=dtype, device=device)

    def _open_set_logits(self, features: torch.Tensor, threshold: torch.Tensor | float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_features = F.normalize(features, dim=-1)
        normalized_prototype = F.normalize(self.aircraft_prototype, dim=0)
        similarity = normalized_features @ normalized_prototype
        if threshold is None:
            threshold = self.decision_threshold.to(device=features.device, dtype=features.dtype)
        elif not torch.is_tensor(threshold):
            threshold = torch.tensor(float(threshold), device=features.device, dtype=features.dtype)
        centered = similarity - threshold
        logits = torch.stack([-centered, centered], dim=-1) * self.logit_scale
        return logits, similarity

    def acceptance(self, output: ExpertOutput, threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = output.auxiliary or {}
        similarity = auxiliary.get("aircraft_similarity")
        if similarity is None:
            similarity = output.logits[:, 1] if output.logits.shape[1] > 1 else output.logits[:, 0]
        threshold_value = float(self.decision_threshold.item() if threshold is None else threshold)
        confidence = similarity.to(dtype=torch.float32, device=output.logits.device)
        return confidence, confidence >= threshold_value

    def forward(self, sequence_features: torch.Tensor, physical_features: torch.Tensor, padding_mask: torch.Tensor | None = None, source_names: list[str] | None = None) -> ExpertOutput:
        seq = sequence_features.transpose(1, 2)
        hidden = self.temporal_conv(seq).transpose(1, 2)
        temporal = self.temporal_pool(hidden, padding_mask=padding_mask)
        physical = self.physical_proj(self.physical_input_norm(physical_features))
        fused = self.fuse(torch.cat([temporal, physical], dim=-1))
        if self.open_set:
            threshold = self._threshold_for_sources(
                batch_size=fused.shape[0],
                device=fused.device,
                dtype=fused.dtype,
                source_names=source_names,
            )
            logits, similarity = self._open_set_logits(fused, threshold=threshold)
            return ExpertOutput(
                logits=logits,
                features=fused,
                auxiliary={
                    "aircraft_similarity": similarity,
                    "aircraft_threshold": threshold,
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

    def acceptance(self, output: ExpertOutput, threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = output.auxiliary or {}
        probability = auxiliary.get("groundness_probability")
        if probability is None:
            probability = torch.ones((output.logits.shape[0],), dtype=output.logits.dtype, device=output.logits.device)
        threshold_value = float(self.groundness_threshold.item() if threshold is None else threshold)
        confidence = probability.to(dtype=torch.float32, device=output.logits.device)
        return confidence, confidence >= threshold_value

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

    def acceptance(self, output: ExpertOutput, threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        accepted = output.logits[:, 1] > output.logits[:, 0]
        confidence = (output.logits[:, 1] - output.logits[:, 0]).to(dtype=torch.float32)
        return confidence, accepted

    def forward(self, height_mean: torch.Tensor, speed_mean: torch.Tensor) -> ExpertOutput:
        positive = (height_mean > self.altitude_threshold_m) & (speed_mean > self.speed_threshold_mps)
        logits = torch.zeros((height_mean.shape[0], 2), dtype=height_mean.dtype, device=height_mean.device)
        logits[:, 0] = torch.where(positive, torch.tensor(-1.0, device=height_mean.device), torch.tensor(1.0, device=height_mean.device))
        logits[:, 1] = torch.where(positive, torch.tensor(1.0, device=height_mean.device), torch.tensor(-1.0, device=height_mean.device))
        return ExpertOutput(logits=logits)
