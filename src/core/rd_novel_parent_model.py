from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .experts import LowAltitudeSuspectExpert
from .rd_novel_parent_data import NovelParentBatch


@dataclass(frozen=True)
class NovelParentStateOutput:
    logits: torch.Tensor
    final_logits: torch.Tensor
    state: torch.Tensor


class FrozenPhysicalV2NovelParentState(nn.Module):
    def __init__(
        self,
        base_checkpoint: str | Path,
        *,
        adapter_dim: int = 96,
        state_dim: int = 96,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_checkpoint = str(base_checkpoint)
        self.adapter_dim = int(adapter_dim)
        self.frozen_trunk = LowAltitudeSuspectExpert(
            input_dim=160,
            rd_input_dim=4,
            rd_patch_input_dim=3,
            rd_context_dim=0,
            use_rd_patch_cnn=False,
            dropout=0.1,
        )
        checkpoint = torch.load(Path(base_checkpoint), map_location="cpu", weights_only=False)
        self.frozen_trunk.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for parameter in self.frozen_trunk.parameters():
            parameter.requires_grad = False
        self.frozen_trunk.eval()
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.frozen_trunk.rd_model_dim),
            nn.Linear(self.frozen_trunk.rd_model_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(adapter_dim),
        )
        self.state_cell = nn.GRUCell(adapter_dim, state_dim)
        self.parent_head = nn.Linear(state_dim, 2)

    def train(self, mode: bool = True) -> "FrozenPhysicalV2NovelParentState":
        super().train(mode)
        self.frozen_trunk.eval()
        return self

    def trainable_state_dict(self) -> dict[str, Mapping[str, torch.Tensor]]:
        return {
            "adapter": self.adapter.state_dict(),
            "state_cell": self.state_cell.state_dict(),
            "parent_head": self.parent_head.state_dict(),
        }

    def load_trainable_state_dict(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.adapter.load_state_dict(state["adapter"], strict=True)
        self.state_cell.load_state_dict(state["state_cell"], strict=True)
        self.parent_head.load_state_dict(state["parent_head"], strict=True)

    def _encode_observations(self, batch: NovelParentBatch) -> torch.Tensor:
        batch_size, chunk_count, frame_count = batch.features.shape[:3]
        flat_features = batch.features.reshape(batch_size * chunk_count, frame_count, *batch.features.shape[3:])
        flat_frame_padding = batch.frame_padding_mask.reshape(batch_size * chunk_count, frame_count)
        flat_frame_times = batch.frame_time_seconds.reshape(batch_size * chunk_count, frame_count)
        active_chunks = (~batch.chunk_padding_mask & ~batch.is_zero_novel_update).reshape(-1)
        observations = batch.features.new_zeros((batch_size * chunk_count, self.adapter_dim))
        if bool(active_chunks.any()):
            with torch.no_grad():
                encoded, _, _ = self.frozen_trunk._encode_rd(
                    flat_features[active_chunks],
                    rd_padding_mask=flat_frame_padding[active_chunks],
                    rd_time_seconds=flat_frame_times[active_chunks],
                )
            observations[active_chunks] = self.adapter(encoded)
        return observations.reshape(batch_size, chunk_count, -1)

    def _elapsed_time_encoding(self, delta_time_seconds: torch.Tensor) -> torch.Tensor:
        half_dim = self.adapter_dim // 2
        frequency_index = torch.arange(
            half_dim,
            device=delta_time_seconds.device,
            dtype=delta_time_seconds.dtype,
        )
        denominator = max(1, half_dim - 1)
        frequencies = torch.exp(-torch.log(delta_time_seconds.new_tensor(10_000.0)) * frequency_index / denominator)
        angles = torch.log1p(delta_time_seconds.clamp_min(0.0)).unsqueeze(-1) * frequencies
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if encoding.shape[-1] < self.adapter_dim:
            encoding = torch.nn.functional.pad(encoding, (0, self.adapter_dim - encoding.shape[-1]))
        return encoding

    def forward(self, batch: NovelParentBatch) -> NovelParentStateOutput:
        observations = self._encode_observations(batch)
        batch_size, chunk_count = observations.shape[:2]
        state = observations.new_zeros((batch_size, self.state_cell.hidden_size))
        logits = observations.new_zeros((batch_size, chunk_count, 2))
        for chunk_position in range(chunk_count):
            active = ~batch.chunk_padding_mask[:, chunk_position]
            has_observation = active & ~batch.is_zero_novel_update[:, chunk_position]
            elapsed = batch.delta_time_seconds[:, chunk_position].clamp_min(0.0)
            decay = torch.exp(-elapsed / 30.0).unsqueeze(-1)
            decayed_state = state * decay
            state = torch.where(active.unsqueeze(-1), decayed_state, state)
            timed_observation = observations[:, chunk_position] + self._elapsed_time_encoding(elapsed)
            candidate_state = self.state_cell(timed_observation, state)
            state = torch.where(has_observation.unsqueeze(-1), candidate_state, state)
            logits[:, chunk_position] = self.parent_head(state)
        return NovelParentStateOutput(logits=logits, final_logits=self.parent_head(state), state=state)


__all__ = ["FrozenPhysicalV2NovelParentState", "NovelParentStateOutput"]
