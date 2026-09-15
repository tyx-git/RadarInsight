"""Independent V27d knownness and quality/maturity heads."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .protocol import V27DContext


class V27dHeads(nn.Module):
    """Two output-only task heads over a frozen causal representation.

    The knownness branch consumes the complete prefix trajectory and explicit
    quality facts.  The quality branch consumes only deterministic causal facts
    and never feeds back into either the trajectory encoder or classification.
    """

    def __init__(self, context: V27DContext, *, task: str, variant: str) -> None:
        super().__init__()
        if task not in {"known_gate", "dbb_specialist"}:
            raise ValueError("V27d head task must be known_gate or dbb_specialist")
        if variant not in {"r1", "r2", "r3", "r4"}:
            raise ValueError("V27d head variant is invalid")
        self.task = task
        self.variant = variant
        self.raw_hidden_dim = int(context.config.data.liquid_hidden_dim)
        self.hidden_dim = int(context.config.model.trajectory_hidden_dim)
        self.projection_dim = int(context.config.model.knownness_projection_dim)
        self.task_fact_dim = int(
            context.config.data.l1_task_fact_dim
            if task == "known_gate"
            else context.config.data.l2_task_fact_dim
        )
        self.event_fact_dim = int(context.config.data.event_fact_dim)
        self.raw_dim = int(context.config.model.raw_scale_feature_dim)
        self.stable_dim = int(context.config.model.stable8_feature_dim)
        self.event_trajectory_dim = self.raw_hidden_dim * 7 + self.event_fact_dim
        self.summary_feature_dim = self.raw_hidden_dim * 14
        self.use_raw_scale = variant in {"r2", "r3", "r4"}
        self.use_stable8 = variant in {"r3", "r4"}
        # ``event_trajectory`` already ends with event facts in the frozen
        # V27 representation.  Feeding ``event_facts_sequence`` again would
        # double-weight the same observation in the recurrent encoder.
        trajectory_input_dim = self.event_trajectory_dim + self.task_fact_dim
        if self.use_raw_scale:
            trajectory_input_dim += self.raw_dim
        if self.use_stable8:
            trajectory_input_dim += self.stable_dim
        self.trajectory_encoder = nn.GRU(trajectory_input_dim, self.hidden_dim, batch_first=True)
        projection_input_dim = self.summary_feature_dim + self.hidden_dim + self.task_fact_dim
        if self.use_raw_scale:
            projection_input_dim += self.raw_dim
        if self.use_stable8:
            projection_input_dim += self.stable_dim
        self.knownness_projection = nn.Sequential(
            nn.Linear(projection_input_dim, self.projection_dim),
            nn.SiLU(),
        )
        self.knownness_head = nn.Linear(self.projection_dim, 1)
        quality_input_dim = self.task_fact_dim + self.event_fact_dim
        self.quality_head = nn.Sequential(
            nn.Linear(quality_input_dim, max(8, self.hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(8, self.hidden_dim // 2), 1),
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-head-contract-v1",
            "experiment_id": "V27d",
            "task": self.task,
            "variant": self.variant,
            "trajectory_input_dim": self.trajectory_encoder.input_size,
            "raw_liquid_hidden_dim": self.raw_hidden_dim,
            "summary_feature_dim": self.summary_feature_dim,
            "event_trajectory_dim": self.event_trajectory_dim,
            "trajectory_hidden_dim": self.hidden_dim,
            "knownness_projection_dim": self.projection_dim,
            "knownness_projection_normalization": "none",
            "event_facts_in_trajectory": True,
            "event_facts_duplicated_in_trajectory_encoder": False,
            "quality_head_input": "deterministic_task_and_event_facts_only",
            "quality_output_feedback": "forbidden",
            "classifier_logits_input_to_head": False,
            "raw_scale_enabled": self.use_raw_scale,
            "stable8_enabled": self.use_stable8,
        }

    @staticmethod
    def _validate_tensor(value: torch.Tensor, name: str) -> None:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"V27d head input is invalid: {name}")

    def forward(
        self,
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        event_facts_sequence: torch.Tensor,
        raw_scale_features: torch.Tensor,
        stable8_causal_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("V27d head padding mask is invalid")
        batch, length = padding_mask.shape
        expected = {
            "summary_features": (batch, length, self.summary_feature_dim),
            "event_trajectory": (batch, length, self.event_trajectory_dim),
            "task_facts_sequence": (batch, length, self.task_fact_dim),
            "event_facts_sequence": (batch, length, self.event_fact_dim),
            "raw_scale_features": (batch, length, self.raw_dim),
            "stable8_causal_features": (batch, length, self.stable_dim),
        }
        values = {
            "summary_features": summary_features,
            "event_trajectory": event_trajectory,
            "task_facts_sequence": task_facts_sequence,
            "event_facts_sequence": event_facts_sequence,
            "raw_scale_features": raw_scale_features,
            "stable8_causal_features": stable8_causal_features,
        }
        for name, shape in expected.items():
            value = values[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"V27d head input shape differs: {name}")
            self._validate_tensor(value, name)
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError(f"V27d head padded input is non-zero: {name}")
        for row in range(batch):
            padded = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded.numel() and bool((~padding_mask[row, int(padded[0].item()) :]).any()):
                raise ValueError("V27d head padding must be a suffix")
        lengths = (~padding_mask).long().sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("V27d head received an empty sequence")
        pieces = [event_trajectory, task_facts_sequence]
        if self.use_raw_scale:
            pieces.append(raw_scale_features)
        if self.use_stable8:
            pieces.append(stable8_causal_features)
        trajectory_input = torch.cat(pieces, dim=-1)
        packed = pack_padded_sequence(
            trajectory_input,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.trajectory_encoder(packed)
        trajectory_sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=length,
        )
        projection_pieces = [summary_features, trajectory_sequence, task_facts_sequence]
        if self.use_raw_scale:
            projection_pieces.append(raw_scale_features)
        if self.use_stable8:
            projection_pieces.append(stable8_causal_features)
        representation = torch.cat(projection_pieces, dim=-1)
        projection = self.knownness_projection(representation)
        knownness_logits = self.knownness_head(projection).squeeze(-1)
        quality_logits = self.quality_head(torch.cat((task_facts_sequence, event_facts_sequence), dim=-1)).squeeze(-1)
        projection = projection.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        trajectory_sequence = trajectory_sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        knownness_logits = knownness_logits.masked_fill(padding_mask, 0.0)
        quality_logits = quality_logits.masked_fill(padding_mask, 0.0)
        if not all(torch.isfinite(value).all() for value in (projection, trajectory_sequence, knownness_logits, quality_logits)):
            raise FloatingPointError("V27d heads produced non-finite output")
        rows = torch.arange(batch, device=padding_mask.device)
        return {
            "knownness_projection_sequence": projection,
            "knownness_logit_sequence": knownness_logits,
            "quality_logit_sequence": quality_logits,
            "trajectory_encoder_sequence": trajectory_sequence,
            "knownness_projection_final": projection[rows, lengths - 1],
            "knownness_logit_final": knownness_logits[rows, lengths - 1],
            "quality_logit_final": quality_logits[rows, lengths - 1],
        }


def build_heads(context: V27DContext, *, task: str, variant: str, seed: int) -> V27dHeads:
    if int(seed) < 0:
        raise ValueError("V27d seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return V27dHeads(context, task=task, variant=variant)


__all__ = ["V27dHeads", "build_heads"]
