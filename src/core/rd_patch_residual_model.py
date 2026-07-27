from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .experts import LowAltitudeSuspectExpert, TemporalAttentionPool


@dataclass(frozen=True)
class RDPatchResidualOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    residual_delta: torch.Tensor
    residual_gate: torch.Tensor


class RDPatchResidualBranch(nn.Module):
    def __init__(self, base_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.temporal_pool = TemporalAttentionPool(64, hidden_dim=64)
        self.residual_head = nn.Sequential(
            nn.Linear(base_feature_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(base_feature_dim + 64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        final_layer = self.residual_head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def forward(self, patch_features: torch.Tensor, patch_padding_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_features.ndim != 5:
            raise ValueError("patch_features must be [batch, time, range, doppler, channel]")
        batch_size, time_steps, range_bins, doppler_bins, channels = patch_features.shape
        if channels < 3:
            raise ValueError("patch_features must include amplitude, phase, and temporal delta")
        frames = torch.nan_to_num(patch_features[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
        frames = frames.reshape(batch_size * time_steps, range_bins, doppler_bins, 3).permute(0, 3, 1, 2).contiguous()
        frame_features = torch.cat([self.frame_cnn(chunk) for chunk in frames.split(64, dim=0)], dim=0)
        frame_features = frame_features.view(batch_size, time_steps, -1)
        return self.temporal_pool(frame_features, padding_mask=patch_padding_mask), frame_features

    def delta(self, base_features: torch.Tensor, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([base_features.detach(), patch_features], dim=-1)
        return self.residual_head(fused), torch.sigmoid(self.gate_head(fused))


class FrozenSequencePatchResidual(nn.Module):
    def __init__(self, base_checkpoint: Path | str, *, dropout: float, residual_scale: float) -> None:
        super().__init__()
        checkpoint = torch.load(Path(base_checkpoint), map_location="cpu", weights_only=False)
        base_config = dict(checkpoint.get("config", {}))
        if bool(base_config.get("use_rd_patch_cnn", False)):
            raise ValueError("base checkpoint must be a non-patch sequence RD diagnostic")
        self.base_checkpoint = str(Path(base_checkpoint))
        self.residual_scale = float(residual_scale)
        self.base_model = LowAltitudeSuspectExpert(
            input_dim=160,
            rd_input_dim=4,
            rd_patch_input_dim=3,
            rd_context_dim=0,
            use_rd_patch_cnn=False,
            dropout=float(base_config.get("dropout", dropout)),
        )
        self.base_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.patch_branch = RDPatchResidualBranch(base_feature_dim=512, dropout=dropout)

    def train(self, mode: bool = True) -> FrozenSequencePatchResidual:
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        rd_features: torch.Tensor,
        rd_patch_features: torch.Tensor | None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        *,
        force_zero_residual: bool = False,
    ) -> RDPatchResidualOutput:
        self.base_model.eval()
        with torch.no_grad():
            base = self.base_model.rd_drone_bird_diagnostic(rd_features, rd_padding_mask=rd_padding_mask)
        base_logits = base.bird_logits
        if force_zero_residual or rd_patch_features is None:
            zeros = torch.zeros_like(base_logits)
            return RDPatchResidualOutput(base_logits, base_logits, zeros, zeros[:, :1])
        patch_vector, _ = self.patch_branch(rd_patch_features, rd_patch_padding_mask)
        raw_delta, gate = self.patch_branch.delta(base.rd_features, patch_vector)
        delta = torch.tanh(raw_delta) * gate * self.residual_scale
        if rd_patch_padding_mask is not None:
            invalid_patch = rd_patch_padding_mask.bool().all(dim=1, keepdim=True)
            delta = torch.where(invalid_patch, torch.zeros_like(delta), delta)
            gate = torch.where(invalid_patch, torch.zeros_like(gate), gate)
        return RDPatchResidualOutput(base_logits + delta, base_logits, delta, gate)
