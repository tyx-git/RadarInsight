"""V22b KnownGate readout-only RD temporal residual diagnostic.

V22a appended the current-event RD temporal residual to the recurrent fusion
input.  V22b keeps exactly the same residual definition, but applies a
zero-initialized, signed binary-logit correction after the V21e readout.  The
complete shared backbone is retrained for the frozen 20-epoch protocol; the
single structural variable is that the residual cannot alter the liquid state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .known_gate import KnownGateClassifier, KnownGateConfig


V22B_VARIANT = "v22b_rd_temporal_late_readout"
V22B_SCHEMA_VERSION = "RADAR-LN3-V22b"
V22B_PROTOCOL_SCHEMA = "RADAR-LN3-V22b-rd-temporal-late-readout-protocol-v1"
V22B_CONTRACT_SCHEMA = "RADAR-LN3-V22b-rd-temporal-late-readout-contract-v1"
V22B_READINESS_SCHEMA = (
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-training-readiness-v1"
)
V22B_CHECKPOINT_SCHEMA = (
    "RADAR-LN3-V22b-v22b_rd_temporal_late_readout-seed43-known-gate-checkpoint-v1"
)

V22B_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V22b-cq08-liquid-known-gate-rd-temporal-late-readout-seed43-config.json"
)
V22B_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V22b-cq08-known-gate-rd-temporal-late-readout-contract.json"
)
V22B_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-training-readiness.json"
)
V22B_ARTIFACT = Path(
    "artifacts/"
    "RADAR-LN3-V22b-cq08-liquid-known-gate-rd-temporal-late-readout-seed43-formal"
)
V22B_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-paired-validation.json"
)

V21E_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-seed43-config.json"
)
V21E_SUMMARY = Path(
    "artifacts/RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal/training-summary.json"
)
V21E_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal/best_model.pt"
)


class V22BLateReadoutKnownGate(nn.Module):
    """V21e KnownGate plus a zero-init signed late-logit correction."""

    def __init__(self, backbone: KnownGateClassifier) -> None:
        super().__init__()
        if backbone.config.model.rd_frame_aggregation != "masked_mean":
            raise ValueError("V22b requires the frozen masked_mean RD path")
        if backbone.config.model.rd_temporal_residual_dim != 0:
            raise ValueError("V22b backbone must be the non-residual V21e model")
        self.backbone = backbone
        self.config: KnownGateConfig = backbone.config
        self.class_names = backbone.class_names
        self.task_name = backbone.task_name
        self.rd_temporal_residual_dim = int(backbone.config.model.rd_embedding_dim)
        self.late_residual_weight = nn.Parameter(
            torch.zeros(self.rd_temporal_residual_dim, dtype=torch.float32)
        )

    def freeze_backbone(self) -> None:
        raise RuntimeError("V22b is a full-parameter experiment; freezing is forbidden")

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def late_residual_parameter_count(self) -> int:
        return int(self.late_residual_weight.numel())

    def _readout_correction(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-1] != self.rd_temporal_residual_dim:
            raise ValueError("V22b residual width differs from the frozen RD embedding")
        delta = torch.matmul(residual, self.late_residual_weight)
        # A signed binary correction preserves the baseline logit midpoint and
        # cannot learn a useless common-mode class offset.
        return torch.stack((-0.5 * delta, 0.5 * delta), dim=-1)

    @staticmethod
    def _window_from_sequence(
        sequence_logits: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        lengths = (~padding_mask).long().sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("V22b received an empty sequence")
        indices = lengths - 1
        batch_indices = torch.arange(
            sequence_logits.shape[0], device=sequence_logits.device
        )
        return sequence_logits[batch_indices, indices]

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        base_output = self.backbone(
            **kwargs, return_rd_temporal_residual=True
        )
        residual = base_output["rd_temporal_residual"]
        correction = self._readout_correction(residual)
        sequence_logits = base_output["sequence_logits"] + correction
        output = dict(base_output)
        output["sequence_logits"] = sequence_logits
        output["window_logits"] = self._window_from_sequence(
            sequence_logits, kwargs["tr_padding_mask"]
        )
        output["rd_temporal_residual"] = residual
        output["late_residual_logit_delta"] = correction
        return output

    def forward_online_point(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        base_output = self.backbone.forward_online_point(
            **kwargs, return_rd_temporal_residual=True
        )
        residual = base_output["rd_temporal_residual"]
        correction = self._readout_correction(residual)
        output = dict(base_output)
        output["window_logits"] = base_output["window_logits"] + correction
        output["rd_temporal_residual"] = residual
        output["late_residual_logit_delta"] = correction
        return output


def build_v22b_model(backbone: KnownGateClassifier) -> V22BLateReadoutKnownGate:
    model = V22BLateReadoutKnownGate(backbone)
    if model.late_residual_parameter_count != model.rd_temporal_residual_dim:
        raise AssertionError("V22b late readout must add exactly 24 trainable weights")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("V22b must retrain every shared and added parameter")
    return model


def protocol_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def finite_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return dict(mapping)


__all__ = [
    "V21E_CHECKPOINT",
    "V21E_CONFIG",
    "V21E_SUMMARY",
    "V22B_ARTIFACT",
    "V22B_CHECKPOINT_SCHEMA",
    "V22B_CONFIG",
    "V22B_CONTRACT",
    "V22B_CONTRACT_SCHEMA",
    "V22B_PROTOCOL_SCHEMA",
    "V22B_READINESS",
    "V22B_READINESS_SCHEMA",
    "V22B_REPORT",
    "V22B_SCHEMA_VERSION",
    "V22B_VARIANT",
    "V22BLateReadoutKnownGate",
    "build_v22b_model",
    "protocol_relative",
]
