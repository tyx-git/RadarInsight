"""Independent causal late-residual models for V27a."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .protocol import V27A_ARMS, V27A_L2_CLASS_NAMES, V27AConfig


class V27AEvidenceResidual(nn.Module):
    """A shared-capacity causal evidence encoder with a constrained residual.

    ``mechanism`` consumes one reliable event-summary token and ``main``
    consumes all reliable frame tokens.  The module graph is identical for
    both arms; only the token tensor presented to the frame GRU differs.
    """

    def __init__(self, config: V27AConfig, *, arm: str) -> None:
        super().__init__()
        if arm not in V27A_ARMS:
            raise ValueError(f"unsupported V27a arm: {arm}")
        self.config = config
        self.arm = str(arm)
        model = config.model
        if model.evidence_dim != len(config.data.stable8_indices):
            raise ValueError("V27a model/data evidence widths differ")
        if model.class_count != len(V27A_L2_CLASS_NAMES):
            raise ValueError("V27a model class count differs")
        self.evidence_dim = int(model.evidence_dim)
        self.hidden_dim = int(model.hidden_dim)
        self.class_count = int(model.class_count)
        self.residual_bound = float(model.residual_bound)
        self.token_projection = nn.Linear(self.evidence_dim, self.hidden_dim)
        self.frame_encoder = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=int(model.frame_gru_layers),
            batch_first=True,
        )
        self.parent_encoder = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=int(model.parent_gru_layers),
            batch_first=True,
        )
        self.residual_projection = nn.Linear(
            self.hidden_dim,
            self.class_count,
            bias=False,
        )
        nn.init.zeros_(self.residual_projection.weight)

    def model_contract(self) -> dict[str, object]:
        """Return the architecture contract stored with every V27a run."""

        return {
            "schema_version": "RADAR-LN3-V27a-residual-model-contract-v1",
            "experiment_id": "V27a",
            "arm": self.arm,
            "evidence_dim": self.evidence_dim,
            "hidden_dim": self.hidden_dim,
            "class_count": self.class_count,
            "frame_gru_layers": int(self.config.model.frame_gru_layers),
            "parent_gru_layers": int(self.config.model.parent_gru_layers),
            "token_activation": self.config.model.token_activation,
            "residual_bound": self.residual_bound,
            "residual_projection_bias": False,
            "residual_zero_initialization": True,
            "residual_zero_sum": True,
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @property
    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(value.shape) for name, value in self.named_parameters()
        }

    def _validate_inputs(
        self,
        baseline_logits: torch.Tensor,
        summary_tokens: torch.Tensor,
        summary_valid: torch.Tensor,
        frame_values: torch.Tensor,
        frame_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int, int]:
        if baseline_logits.ndim != 3 or baseline_logits.shape[-1] != self.class_count:
            raise ValueError("V27a baseline logits must have shape [B,T,3]")
        batch, length = baseline_logits.shape[:2]
        if batch <= 0 or length <= 0:
            raise ValueError("V27a model requires a non-empty batch and time axis")
        if summary_tokens.shape != (batch, length, self.evidence_dim):
            raise ValueError("V27a summary token shape differs")
        if summary_valid.shape != (batch, length) or summary_valid.dtype != torch.bool:
            raise ValueError("V27a summary validity shape/dtype differs")
        if frame_values.ndim != 4 or frame_values.shape[:2] != (batch, length):
            raise ValueError("V27a frame values must have shape [B,T,K,D]")
        frame_count = int(frame_values.shape[2])
        if frame_values.shape[-1] != self.evidence_dim:
            raise ValueError("V27a frame value width differs")
        if frame_mask.shape != (batch, length, frame_count) or frame_mask.dtype != torch.bool:
            raise ValueError("V27a frame mask shape/dtype differs")
        if padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
            raise ValueError("V27a padding mask shape/dtype differs")
        values = (
            baseline_logits,
            summary_tokens,
            frame_values,
        )
        if not all(value.is_floating_point() for value in values):
            raise TypeError("V27a model values must be floating point")
        if any(value.device != baseline_logits.device for value in values):
            raise ValueError("V27a model values must share a device")
        if any(value.dtype != baseline_logits.dtype for value in values):
            raise ValueError("V27a model values must share a dtype")
        if any(value.device != baseline_logits.device for value in (summary_valid, frame_mask, padding_mask)):
            raise ValueError("V27a model masks must share the value device")
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("V27a model values must be finite")
        if bool(baseline_logits.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27a padded baseline logits must be zero")
        if bool(summary_valid.masked_select(padding_mask).any()):
            raise ValueError("V27a padded summary tokens cannot be valid")
        invalid_summary = (~summary_valid).unsqueeze(-1).expand_as(summary_tokens)
        if bool(summary_tokens.masked_select(invalid_summary).ne(0).any()):
            raise ValueError("V27a invalid summary tokens must be zero")
        if bool(frame_mask.masked_select(padding_mask.unsqueeze(-1)).any()):
            raise ValueError("V27a padded events cannot contain frame tokens")
        invalid_frames = (~frame_mask).unsqueeze(-1).expand_as(frame_values)
        if bool(frame_values.masked_select(invalid_frames).ne(0).any()):
            raise ValueError("V27a inactive frame token values must be zero")
        for row in range(batch):
            active = ~padding_mask[row]
            padded = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded.numel() and bool(active[int(padded[0]) :].any()):
                raise ValueError("V27a padding must be a suffix")
            for event in torch.nonzero(active, as_tuple=False).flatten().tolist():
                mask = frame_mask[row, event]
                first_inactive = torch.nonzero(~mask, as_tuple=False).flatten()
                if first_inactive.numel() and bool(mask[int(first_inactive[0]) :].any()):
                    raise ValueError("V27a frame padding must be a suffix")
        if not torch.equal(summary_valid, frame_mask.any(dim=-1) & ~padding_mask):
            raise ValueError(
                "V27a summary validity must equal the reliable frame presence contract"
            )
        return batch, length, frame_count

    def _token_hidden(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a non-empty batch of event-local token sequences."""

        if tokens.ndim != 3 or lengths.ndim != 1 or tokens.shape[0] != lengths.shape[0]:
            raise ValueError("V27a token batch shape differs")
        if bool((lengths <= 0).any()) or bool((lengths > tokens.shape[1]).any()):
            raise ValueError("V27a token lengths must be within the token tensor")
        projected = self.token_projection(tokens)
        if self.config.model.token_activation == "silu":
            projected = torch.nn.functional.silu(projected)
        else:
            raise ValueError("unsupported V27a token activation")
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.frame_encoder(packed)
        return hidden[-1]

    def _bounded_residual(self, raw: torch.Tensor) -> torch.Tensor:
        centered = raw - raw.mean(dim=-1, keepdim=True)
        maximum = centered.abs().amax(dim=-1, keepdim=True)
        multiplier = torch.clamp(
            self.residual_bound / maximum.clamp_min(torch.finfo(centered.dtype).tiny),
            max=1.0,
        )
        return centered * multiplier

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if int(batch_size) <= 0:
            raise ValueError("V27a initial state batch size must be positive")
        return {
            "parent_state": torch.zeros(
                (int(batch_size), self.hidden_dim), device=device, dtype=dtype
            ),
            "previous_residual": torch.zeros(
                (int(batch_size), self.class_count), device=device, dtype=dtype
            ),
        }

    def _validate_state(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(state, Mapping):
            raise TypeError("V27a continuation state must be a mapping")
        parent_state = state.get("parent_state")
        previous_residual = state.get("previous_residual")
        if parent_state is None or previous_residual is None:
            raise ValueError("V27a continuation state is incomplete")
        if parent_state.shape != (batch_size, self.hidden_dim):
            raise ValueError("V27a continuation parent state shape differs")
        if previous_residual.shape != (batch_size, self.class_count):
            raise ValueError("V27a continuation residual state shape differs")
        if parent_state.device != device or previous_residual.device != device:
            raise ValueError("V27a continuation state device differs")
        if parent_state.dtype != dtype or previous_residual.dtype != dtype:
            raise ValueError("V27a continuation state dtype differs")
        if not parent_state.is_floating_point() or not previous_residual.is_floating_point():
            raise TypeError("V27a continuation state must be floating point")
        if not torch.isfinite(parent_state).all() or not torch.isfinite(previous_residual).all():
            raise ValueError("V27a continuation state must be finite")
        return parent_state, previous_residual

    def step_event(
        self,
        *,
        summary_tokens: torch.Tensor,
        summary_valid: torch.Tensor,
        frame_values: torch.Tensor,
        frame_mask: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Process one event and return the updated explicit continuation state."""

        if summary_tokens.ndim != 2 or summary_tokens.shape[-1] != self.evidence_dim:
            raise ValueError("V27a event summary token shape differs")
        batch = int(summary_tokens.shape[0])
        if summary_valid.shape != (batch,) or summary_valid.dtype != torch.bool:
            raise ValueError("V27a event summary validity differs")
        if frame_values.ndim != 3 or frame_values.shape[0] != batch or frame_values.shape[-1] != self.evidence_dim:
            raise ValueError("V27a event frame values differ")
        if frame_mask.shape != frame_values.shape[:2] or frame_mask.dtype != torch.bool:
            raise ValueError("V27a event frame mask differs")
        parent_state, previous_residual = self._validate_state(
            state,
            batch_size=batch,
            device=summary_tokens.device,
            dtype=summary_tokens.dtype,
        )
        if not torch.equal(summary_valid, frame_mask.any(dim=-1)):
            raise ValueError("V27a event validity fields are inconsistent")
        selected_rows = torch.nonzero(
            summary_valid if self.arm == "mechanism" else frame_mask.any(dim=-1),
            as_tuple=False,
        ).flatten()
        event_representation = parent_state.new_zeros((batch, self.hidden_dim))
        next_parent = parent_state
        next_residual = previous_residual
        increment = previous_residual.new_zeros((batch, self.class_count))
        if selected_rows.numel():
            if self.arm == "mechanism":
                tokens = summary_tokens[selected_rows].unsqueeze(1)
                lengths = torch.ones(
                    (selected_rows.numel(),), dtype=torch.long, device=summary_tokens.device
                )
            else:
                counts = frame_mask[selected_rows].sum(dim=-1).to(dtype=torch.long)
                max_tokens = int(counts.max().item())
                tokens = frame_values[selected_rows, :max_tokens]
                lengths = counts
            event_hidden = self._token_hidden(tokens, lengths)
            event_representation = event_representation.index_copy(
                0, selected_rows, event_hidden
            )
            _, hidden = self.parent_encoder(
                event_representation[selected_rows].unsqueeze(1),
                parent_state[selected_rows].unsqueeze(0),
            )
            next_parent = parent_state.index_copy(0, selected_rows, hidden[-1])
            bounded = self._bounded_residual(
                self.residual_projection(next_parent[selected_rows])
            )
            next_residual = previous_residual.index_copy(0, selected_rows, bounded)
            increment = increment.index_copy(
                0, selected_rows, bounded - previous_residual[selected_rows]
            )
        return {
            "parent_state": next_parent,
            "previous_residual": next_residual,
            "event_representation": event_representation,
            "evidence_update": event_representation,
            "residual": next_residual,
            "residual_increment": increment,
        }

    def forward(
        self,
        baseline_logits: torch.Tensor,
        summary_tokens: torch.Tensor,
        summary_valid: torch.Tensor,
        frame_values: torch.Tensor,
        frame_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        initial_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, length, frame_count = self._validate_inputs(
            baseline_logits,
            summary_tokens,
            summary_valid,
            frame_values,
            frame_mask,
            padding_mask,
        )
        dtype = baseline_logits.dtype
        device = baseline_logits.device
        continuation = (
            self.initial_state(batch, device=device, dtype=dtype)
            if initial_state is None
            else dict(initial_state)
        )
        state, previous_residual = self._validate_state(
            continuation,
            batch_size=batch,
            device=device,
            dtype=dtype,
        )
        parent_rows: list[torch.Tensor] = []
        event_rows: list[torch.Tensor] = []
        evidence_update_rows: list[torch.Tensor] = []
        residual_rows: list[torch.Tensor] = []
        residual_increment_rows: list[torch.Tensor] = []

        for event_index in range(length):
            active = ~padding_mask[:, event_index]
            step = self.step_event(
                summary_tokens=summary_tokens[:, event_index],
                summary_valid=summary_valid[:, event_index] & active,
                frame_values=frame_values[:, event_index],
                frame_mask=frame_mask[:, event_index] & active.unsqueeze(-1),
                state={"parent_state": state, "previous_residual": previous_residual},
            )
            state = step["parent_state"]
            previous_residual = step["previous_residual"]
            parent_rows.append(state)
            event_rows.append(step["event_representation"])
            evidence_update_rows.append(step["evidence_update"])
            residual_rows.append(step["residual"])
            residual_increment_rows.append(step["residual_increment"])

        parent_sequence = torch.stack(parent_rows, dim=1)
        event_representation = torch.stack(event_rows, dim=1)
        evidence_update = torch.stack(evidence_update_rows, dim=1)
        residual_sequence = torch.stack(residual_rows, dim=1)
        residual_increment = torch.stack(residual_increment_rows, dim=1)
        # Every global event index contributes exactly one row.  Padding is
        # represented by zero output rows and never invokes a recurrent update.
        parent_sequence = parent_sequence.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        residual_sequence = residual_sequence.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        residual_increment = residual_increment.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        event_representation = event_representation.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        evidence_update = evidence_update.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        combined = (baseline_logits + residual_sequence).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        if not all(
            torch.isfinite(value).all()
            for value in (
                parent_sequence,
                event_representation,
                residual_sequence,
                residual_increment,
                combined,
            )
        ):
            raise FloatingPointError("V27a evidence residual produced non-finite values")
        return {
            "sequence_logits": combined,
            "baseline_logits": baseline_logits,
            "residual_logits": residual_sequence,
            "residual_increment": residual_increment,
            "parent_sequence": parent_sequence,
            "event_representation": event_representation,
            "evidence_update": evidence_update,
            "frame_count": frame_mask.sum(dim=-1),
            "final_state": {
                "parent_state": state,
                "previous_residual": previous_residual,
            },
        }




__all__ = ["V27AEvidenceResidual"]