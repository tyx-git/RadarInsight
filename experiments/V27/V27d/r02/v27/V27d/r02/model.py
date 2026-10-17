"""V27d-r02 Bird→Drone 混淆感知模型。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from v27.V27d.r20.model import R20L2Model


class R02BirdDroneModel(nn.Module):
    """在 r20 液态序列状态上增加 Bird 可靠性、成熟度和二分类校正头。

    基模型仍然负责 TR/RD/液态状态编码；r02 只在完整 parent 序列的每个
    causal prefix 上产生辅助信号，禁止把事件打散成独立样本。
    """

    def __init__(
        self,
        base_model: R20L2Model,
        *,
        reliability_hidden_dim: int = 24,
        maturity_hidden_dim: int = 24,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, R20L2Model):
            raise TypeError("r02 base_model must be R20L2Model")
        hidden_dim = int(base_model.base_model.liquid.cell.input_dim)
        if hidden_dim <= 0:
            raise ValueError("r02 liquid hidden dimension must be positive")
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, int(reliability_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(reliability_hidden_dim), 1),
        )
        self.maturity_head = nn.Sequential(
            nn.Linear(hidden_dim, int(maturity_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(maturity_hidden_dim), 1),
        )
        self.correction_head = nn.Linear(2, 2, bias=True)
        # 新增分支从零开始，使 r02 的初始输出严格等价于 r20。
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)
        nn.init.zeros_(self.maturity_head[-1].weight)
        nn.init.zeros_(self.maturity_head[-1].bias)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-r02-bird-drone-model-contract-v1",
            "experiment_id": "V27d-r02",
            "task": "dbb_specialist",
            "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
            "base_model_contract": self.base_model.model_contract(),
            "liquid_hidden_dim": self.hidden_dim,
            "reliability_head": {
                "target": "BirdTarget=1; DroneTarget/BalloonTarget=0",
                "input": "r20 liquid_sequence causal hidden state",
                "output": "one logit per event prefix",
            },
            "maturity_head": {
                "target": "causal prefix length / complete parent length",
                "input": "r20 liquid_sequence causal hidden state",
                "output": "one logit per event prefix",
            },
            "correction_head": {
                "input": "sigmoid(reliability_logit), sigmoid(maturity_logit)",
                "output": "Drone/Bird logit deltas only",
                "zero_initialized": True,
            },
            "sequence_policy": "complete_parent_sequence_no_event_random_split",
            "validation_audit_only_hard_negatives": True,
        }

    @staticmethod
    def _validate_prefix_shapes(
        sequence: torch.Tensor,
        padding_mask: torch.Tensor,
        sequence_logits: torch.Tensor,
    ) -> None:
        if sequence.ndim != 3 or padding_mask.ndim != 2 or sequence_logits.ndim != 3:
            raise ValueError("r02 prefix tensors must be [B,T,D], [B,T], [B,T,C]")
        if sequence.shape[:2] != padding_mask.shape or sequence.shape[:2] != sequence_logits.shape[:2]:
            raise ValueError("r02 prefix tensors have inconsistent parent sequence shapes")
        if sequence_logits.shape[-1] != 3:
            raise ValueError("r02 expects three DBB logits")
        if padding_mask.dtype != torch.bool:
            raise ValueError("r02 padding mask must be boolean")
        if bool((~padding_mask).long().sum(dim=1).le(0).any()):
            raise ValueError("r02 received an empty parent")

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.base_model(**inputs)
        liquid_sequence = base["liquid_sequence"]
        sequence_logits = base["sequence_logits"]
        padding_mask = inputs["tr_padding_mask"]
        self._validate_prefix_shapes(liquid_sequence, padding_mask, sequence_logits)
        reliability_logits = self.reliability_head(liquid_sequence).squeeze(-1)
        maturity_logits = self.maturity_head(liquid_sequence).squeeze(-1)
        reliability = torch.sigmoid(reliability_logits)
        maturity = torch.sigmoid(maturity_logits)
        correction = self.correction_head(torch.stack((reliability, maturity), dim=-1))
        corrected_sequence_logits = sequence_logits.clone()
        corrected_sequence_logits[..., 0:2] = corrected_sequence_logits[..., 0:2] + correction
        corrected_sequence_logits = corrected_sequence_logits.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        lengths = (~padding_mask).long().sum(dim=1)
        batch_indices = torch.arange(sequence_logits.shape[0], device=sequence_logits.device)
        corrected_window_logits = corrected_sequence_logits[batch_indices, lengths - 1]
        values = (
            reliability_logits,
            maturity_logits,
            corrected_sequence_logits,
            corrected_window_logits,
        )
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("r02 auxiliary heads produced non-finite values")
        return {
            **base,
            "base_window_logits": base["window_logits"],
            "base_sequence_logits": base["sequence_logits"],
            "window_logits": corrected_window_logits,
            "sequence_logits": corrected_sequence_logits,
            "bird_reliability_logits": reliability_logits,
            "bird_reliability": reliability,
            "maturity_logits": maturity_logits,
            "maturity": maturity,
            "bird_drone_correction": correction,
        }


__all__ = ["R02BirdDroneModel"]
