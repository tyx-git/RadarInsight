"""Parameter-matched GRU and Liquid V2 Bird-route verifier cores."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper


class BirdVerifier(nn.Module):
    """A task-owned verifier whose recurrent core is the only arm variable."""

    def __init__(
        self,
        *,
        arm: str,
        event_trajectory_dim: int = 355,
        task_fact_dim: int = 7,
        hidden_dim: int = 64,
        summary_dim: int = 672,
        projection_dim: int = 32,
        liquid_tau_min_seconds: float = 0.05,
        liquid_tau_max_seconds: float = 30.0,
        liquid_zero_dt_event_floor_seconds: float = 0.001,
        liquid_dt_max_seconds: float = 120.0,
        liquid_invalid_dt_update_mode: str = "bounded_observation",
        liquid_invalid_dt_observation_gain: float = 0.25,
    ) -> None:
        super().__init__()
        if arm not in {"gru_control", "liquid_treatment"}:
            raise ValueError("V27d-r6 verifier arm is invalid")
        self.arm = str(arm)
        self.event_trajectory_dim = int(event_trajectory_dim)
        self.task_fact_dim = int(task_fact_dim)
        self.input_dim = self.event_trajectory_dim + self.task_fact_dim
        self.hidden_dim = int(hidden_dim)
        self.summary_dim = int(summary_dim)
        self.projection_dim = int(projection_dim)
        if min(self.event_trajectory_dim, self.task_fact_dim, self.hidden_dim, self.summary_dim, self.projection_dim) <= 0:
            raise ValueError("V27d-r6 verifier dimensions must be positive")
        if self.arm == "gru_control":
            self.core = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        else:
            self.core = LiquidV2SequenceWrapper(
                LiquidV2Cell(
                    self.input_dim,
                    self.hidden_dim,
                    tau_min_seconds=float(liquid_tau_min_seconds),
                    tau_max_seconds=float(liquid_tau_max_seconds),
                    zero_dt_event_floor_seconds=float(liquid_zero_dt_event_floor_seconds),
                    dt_max_seconds=float(liquid_dt_max_seconds),
                    invalid_dt_update_mode=str(liquid_invalid_dt_update_mode),
                    invalid_dt_observation_gain=float(liquid_invalid_dt_observation_gain),
                )
            )
        self.readout = nn.Sequential(
            nn.Linear(self.summary_dim + self.hidden_dim + self.task_fact_dim, self.projection_dim),
            nn.SiLU(),
        )
        self.verifier_head = nn.Linear(self.projection_dim, 1)

    @property
    def recurrent_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.core.parameters())

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-r6-verifier-contract-v1",
            "experiment_id": "V27d-r6-bird-liquid-verifier",
            "arm": self.arm,
            "core": "gru" if self.arm == "gru_control" else "liquid_v2",
            "event_trajectory_dim": self.event_trajectory_dim,
            "task_fact_dim": self.task_fact_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "summary_dim": self.summary_dim,
            "projection_dim": self.projection_dim,
            "recurrent_parameter_count": self.recurrent_parameter_count,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
            "padding_policy": "suffix_masked_no_recursive_update",
            "liquid_equation": "RADAR-LN3-V27-liquid-v2-raw-memory-pre-norm-v1" if self.arm == "liquid_treatment" else None,
        }

    @staticmethod
    def _validate_common(
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if summary_features.ndim != 3 or event_trajectory.ndim != 3 or task_facts_sequence.ndim != 3:
            raise ValueError("V27d-r6 verifier tensors must be three-dimensional")
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("V27d-r6 padding mask is invalid")
        if summary_features.shape[:2] != padding_mask.shape or event_trajectory.shape[:2] != padding_mask.shape or task_facts_sequence.shape[:2] != padding_mask.shape:
            raise ValueError("V27d-r6 verifier sequence shapes differ")
        values = (summary_features, event_trajectory, task_facts_sequence)
        if not all(value.is_floating_point() and torch.isfinite(value).all() for value in values):
            raise ValueError("V27d-r6 verifier inputs must be finite floating tensors")
        for value in values:
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError("V27d-r6 padded verifier inputs must be zero")
        lengths = (~padding_mask).long().sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("V27d-r6 verifier received an empty sequence")
        return int(padding_mask.shape[0]), int(padding_mask.shape[1])

    @staticmethod
    def _time_contract(event_facts: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode only deterministic time facts needed by Liquid V2."""

        if event_facts.shape[-1] != 19:
            raise ValueError("V27d-r6 event facts must have width 19")
        dt_seconds = (event_facts[..., 7].clamp_min(0.0) * 120.0).masked_fill(padding_mask, 0.0)
        dt_valid = event_facts[..., 8].ge(0.5) & ~padding_mask
        first = event_facts[..., 9].ge(0.5)
        zero = event_facts[..., 10].ge(0.5)
        positive = event_facts[..., 11].ge(0.5)
        reason = torch.zeros_like(event_facts[..., 8], dtype=torch.long)
        reason = reason.masked_fill(first, 1).masked_fill(zero, 2).masked_fill(positive, 3).masked_fill(padding_mask, 0)
        if bool((reason.eq(3) != dt_valid).masked_select(~padding_mask).any()):
            raise ValueError("V27d-r6 event time reason and validity differ")
        return dt_seconds, dt_valid, reason

    def _recurrent_sequence(
        self,
        event_input: torch.Tensor,
        padding_mask: torch.Tensor,
        event_facts: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        lengths = (~padding_mask).long().sum(dim=1)
        if self.arm == "gru_control":
            packed = pack_padded_sequence(event_input, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.core(packed)
            sequence, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=event_input.shape[1])
            return sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0), None
        dt_seconds, dt_valid, dt_reason = self._time_contract(event_facts, padding_mask)
        output = self.core(
            event_input,
            dt_seconds,
            padding_mask,
            dt_valid=dt_valid,
            dt_reason=dt_reason,
            return_trace=True,
        )
        return output.sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0), output.trace

    def forward(
        self,
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        event_facts_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        batch, length = self._validate_common(
            summary_features=summary_features,
            event_trajectory=event_trajectory,
            task_facts_sequence=task_facts_sequence,
            padding_mask=padding_mask,
        )
        if event_trajectory.shape[-1] != self.event_trajectory_dim or task_facts_sequence.shape[-1] != self.task_fact_dim:
            raise ValueError("V27d-r6 verifier input width differs")
        if event_facts_sequence.shape != (*padding_mask.shape, 19) or not event_facts_sequence.is_floating_point() or not torch.isfinite(event_facts_sequence).all():
            raise ValueError("V27d-r6 event facts shape/value differs")
        if bool(event_facts_sequence.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r6 padded event facts must be zero")
        event_input = torch.cat((event_trajectory, task_facts_sequence), dim=-1)
        recurrent_sequence, trace = self._recurrent_sequence(event_input, padding_mask, event_facts_sequence)
        representation = torch.cat((summary_features, recurrent_sequence, task_facts_sequence), dim=-1)
        projection_sequence = self.readout(representation)
        logits = self.verifier_head(projection_sequence).squeeze(-1)
        projection_sequence = projection_sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        logits = logits.masked_fill(padding_mask, 0.0)
        rows = torch.arange(batch, device=padding_mask.device)
        lengths = (~padding_mask).long().sum(dim=1)
        final = lengths - 1
        values = (recurrent_sequence, projection_sequence, logits)
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("V27d-r6 verifier produced non-finite outputs")
        return {
            "recurrent_sequence": recurrent_sequence,
            "projection_sequence": projection_sequence,
            "verifier_logit_sequence": logits,
            "verifier_score_sequence": torch.sigmoid(logits),
            "recurrent_final": recurrent_sequence[rows, final],
            "projection_final": projection_sequence[rows, final],
            "verifier_logit_final": logits[rows, final],
            "verifier_score_final": torch.sigmoid(logits[rows, final]),
            "liquid_trace": trace,
        }


def build_verifier(config: Mapping[str, Any], *, arm: str, seed: int) -> BirdVerifier:
    if int(seed) < 0:
        raise ValueError("V27d-r6 seed must be non-negative")
    model = config["model"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return BirdVerifier(
            arm=arm,
            event_trajectory_dim=int(model["event_trajectory_dim"]),
            task_fact_dim=int(model["task_fact_dim"]),
            hidden_dim=int(model["hidden_dim"]),
            summary_dim=int(model["summary_dim"]),
            projection_dim=int(model["projection_dim"]),
            liquid_tau_min_seconds=float(model["liquid_tau_min_seconds"]),
            liquid_tau_max_seconds=float(model["liquid_tau_max_seconds"]),
            liquid_zero_dt_event_floor_seconds=float(model["liquid_zero_dt_event_floor_seconds"]),
            liquid_dt_max_seconds=float(model["liquid_dt_max_seconds"]),
            liquid_invalid_dt_update_mode=str(model["liquid_invalid_dt_update_mode"]),
            liquid_invalid_dt_observation_gain=float(model["liquid_invalid_dt_observation_gain"]),
        )


__all__ = ["BirdVerifier", "build_verifier"]
