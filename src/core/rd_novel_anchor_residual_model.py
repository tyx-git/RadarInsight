from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .rd_novel_parent_data import NovelParentBatch
from .rd_novel_parent_model import FrozenPhysicalV2NovelParentState, NovelParentStateOutput


class FrozenPhysicalV2AnchorResidualState(FrozenPhysicalV2NovelParentState):
    def __init__(
        self,
        base_checkpoint: str | Path,
        *,
        adapter_dim: int = 96,
        state_dim: int = 96,
        dropout: float = 0.0,
        max_residual_logit_delta: float = 0.75,
    ) -> None:
        super().__init__(
            base_checkpoint,
            adapter_dim=adapter_dim,
            state_dim=state_dim,
            dropout=dropout,
        )
        del self.parent_head
        self.max_residual_logit_delta = float(max_residual_logit_delta)
        self.residual_head = nn.Linear(state_dim, 2)
        self.residual_gate = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, -4.0)

    def trainable_state_dict(self) -> dict[str, Mapping[str, torch.Tensor]]:
        return {
            "adapter": self.adapter.state_dict(),
            "state_cell": self.state_cell.state_dict(),
            "residual_head": self.residual_head.state_dict(),
            "residual_gate": self.residual_gate.state_dict(),
        }

    def load_trainable_state_dict(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.adapter.load_state_dict(state["adapter"], strict=True)
        self.state_cell.load_state_dict(state["state_cell"], strict=True)
        self.residual_head.load_state_dict(state["residual_head"], strict=True)
        if "residual_gate" in state:
            self.residual_gate.load_state_dict(state["residual_gate"], strict=True)

    def forward(self, batch: NovelParentBatch) -> NovelParentStateOutput:
        if batch.anchor_logits is None:
            raise ValueError("anchor residual state requires frozen cumulative anchor logits")
        observations = self._encode_observations(batch)
        batch_size, chunk_count = observations.shape[:2]
        state = observations.new_zeros((batch_size, self.state_cell.hidden_size))
        logits = observations.new_zeros((batch_size, chunk_count, 2))
        final_logits = observations.new_zeros((batch_size, 2))
        for chunk_position in range(chunk_count):
            active = ~batch.chunk_padding_mask[:, chunk_position]
            has_observation = active & ~batch.is_zero_novel_update[:, chunk_position]
            elapsed = batch.delta_time_seconds[:, chunk_position].clamp_min(0.0)
            decay = torch.exp(-elapsed / 30.0).unsqueeze(-1)
            state = torch.where(active.unsqueeze(-1), state * decay, state)
            timed_observation = observations[:, chunk_position] + self._elapsed_time_encoding(elapsed)
            candidate_state = self.state_cell(timed_observation, state)
            state = torch.where(has_observation.unsqueeze(-1), candidate_state, state)
            gate = torch.sigmoid(self.residual_gate(state))
            residual = self.max_residual_logit_delta * gate * torch.tanh(self.residual_head(state))
            residual = torch.where(has_observation.unsqueeze(-1), residual, torch.zeros_like(residual))
            combined_logits = batch.anchor_logits[:, chunk_position] + residual
            logits[:, chunk_position] = combined_logits
            final_logits = torch.where(active.unsqueeze(-1), combined_logits, final_logits)
        return NovelParentStateOutput(logits=logits, final_logits=final_logits, state=state)


__all__ = ["FrozenPhysicalV2AnchorResidualState"]
