"""The trainable, conditional Drone--Bird correction head for V27b."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .protocol import V27BConfig, V27B_L2_CLASS_NAMES, V27B_TREATMENT_ARM


class V27BConditionalPairResidual(nn.Module):
    """A causal correction head conditioned on the frozen V27a decision.

    The V27a representation and classifier are materialized before this
    module is constructed, so every parameter here belongs to the treatment
    head.  A scalar pairwise delta is converted to a zero-sum three-class
    residual ``(+delta/2, -delta/2, 0)``.  Consequently the head can only
    move the Drone--Bird boundary and cannot alter the Balloon logit.
    """

    def __init__(self, config: V27BConfig) -> None:
        super().__init__()
        self.config = config
        model = config.model
        if model.class_count != len(V27B_L2_CLASS_NAMES):
            raise ValueError("V27b conditional head class count differs")
        expected_input_dim = (
            model.parent_state_dim
            + model.event_representation_dim
            + model.decision_state_dim
            + (
                model.frame_encoder_hidden_dim
                if model.evidence_carrier == "direct_frame_gru"
                else 0
            )
        )
        if model.input_dim != expected_input_dim:
            raise ValueError("V27b conditional head input dimensions differ")
        self.parent_state_dim = int(model.parent_state_dim)
        self.event_representation_dim = int(model.event_representation_dim)
        self.decision_state_dim = int(model.decision_state_dim)
        self.input_dim = int(model.input_dim)
        self.hidden_dim = int(model.hidden_dim)
        self.class_count = int(model.class_count)
        self.residual_bound = float(model.residual_bound)
        self.pairwise_delta_bound = float(model.pairwise_delta_bound)
        self.activation = str(model.activation)
        self.evidence_carrier = str(model.evidence_carrier)
        self.frame_encoder_hidden_dim = int(model.frame_encoder_hidden_dim)
        self.frame_encoder_layers = int(model.frame_encoder_layers)
        self.stable8_dim = int(config.data.stable8_dim)
        self.max_frames_per_event = int(config.data.max_frames_per_event)

        if self.evidence_carrier == "direct_frame_gru":
            self.frame_token_projection: nn.Linear | None = nn.Linear(
                self.stable8_dim,
                self.frame_encoder_hidden_dim,
            )
            self.frame_encoder: nn.GRU | None = nn.GRU(
                self.frame_encoder_hidden_dim,
                self.frame_encoder_hidden_dim,
                num_layers=self.frame_encoder_layers,
                batch_first=True,
            )
        else:
            self.frame_token_projection = None
            self.frame_encoder = None

        self.correction_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        # The treatment is exactly the frozen V27a control before training.
        nn.init.zeros_(self.correction_projection[-1].weight)

    def model_contract(self) -> dict[str, object]:
        contract = {
            "schema_version": (
                "RADAR-LN3-V27b-conditional-pair-residual-model-v3"
                if self.evidence_carrier == "direct_frame_gru"
                else "RADAR-LN3-V27b-conditional-pair-residual-model-v2"
            ),
            "experiment_id": "V27b",
            "arm": V27B_TREATMENT_ARM,
            "parent_state_dim": self.parent_state_dim,
            "event_representation_dim": self.event_representation_dim,
            "decision_state_dim": self.decision_state_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "class_count": self.class_count,
            "activation": self.activation,
            "residual_bound": self.residual_bound,
            "pairwise_delta_bound": self.pairwise_delta_bound,
            "residual_projection_bias": False,
            "residual_zero_initialization": True,
            "residual_zero_sum": True,
            "correction_pair": "DroneTarget__vs__BirdTarget",
            "balloon_correction": "exact_zero",
            "no_reliable_event": "zero_output_correction_and_increment_preserve_continuation_state",
            "padding": "zero_output_and_no_state_update",
        }
        if self.evidence_carrier == "direct_frame_gru":
            contract.update(
                {
                    "evidence_carrier": self.evidence_carrier,
                    "stable8_dim": self.stable8_dim,
                    "max_frames_per_event": self.max_frames_per_event,
                    "frame_encoder_hidden_dim": self.frame_encoder_hidden_dim,
                    "frame_encoder_layers": self.frame_encoder_layers,
                    "frame_encoder": "mask_aware_event_local_gru",
                    "frame_sequence_scope": "within_event_only",
                }
            )
        return contract

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(value.shape) for name, value in self.named_parameters()}

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("V27b initial state batch size must be positive")
        return {
            "previous_delta": torch.zeros(
                (int(batch_size),), device=device, dtype=dtype
            )
        }

    def _validate_state(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(state, Mapping):
            raise TypeError("V27b continuation state must be a mapping")
        previous = state.get("previous_delta")
        if not isinstance(previous, torch.Tensor):
            raise ValueError("V27b continuation state lacks previous_delta")
        if previous.shape != (batch_size,):
            raise ValueError("V27b continuation previous_delta shape differs")
        if previous.device != device or previous.dtype != dtype:
            raise ValueError("V27b continuation previous_delta device/dtype differs")
        if not previous.is_floating_point() or not torch.isfinite(previous).all():
            raise ValueError("V27b continuation previous_delta must be finite float")
        if bool(previous.abs().gt(self.pairwise_delta_bound + 1.0e-6).any()):
            raise ValueError("V27b continuation previous_delta exceeds its bound")
        return previous

    @staticmethod
    def _suffix_padding(padding_mask: torch.Tensor) -> None:
        for row in range(int(padding_mask.shape[0])):
            padded = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded.numel() and bool((~padding_mask[row, int(padded[0]) :]).any()):
                raise ValueError("V27b padding must be a suffix")

    def _validate_inputs(
        self,
        parent_state: torch.Tensor,
        event_representation: torch.Tensor,
        frozen_l2_logits: torch.Tensor,
        v27a_residual_logits: torch.Tensor,
        event_summary_valid: torch.Tensor,
        padding_mask: torch.Tensor,
        event_frame_values: torch.Tensor | None,
        event_frame_mask: torch.Tensor | None,
    ) -> tuple[int, int]:
        if parent_state.ndim != 3:
            raise ValueError("V27b parent_state must have shape [B,T,H]")
        batch, length = parent_state.shape[:2]
        expected = {
            "parent_state": (batch, length, self.parent_state_dim),
            "event_representation": (batch, length, self.event_representation_dim),
            "frozen_l2_logits": (batch, length, self.class_count),
            "v27a_residual_logits": (batch, length, self.class_count),
            "event_summary_valid": (batch, length),
            "padding_mask": (batch, length),
        }
        values = {
            "parent_state": parent_state,
            "event_representation": event_representation,
            "frozen_l2_logits": frozen_l2_logits,
            "v27a_residual_logits": v27a_residual_logits,
            "event_summary_valid": event_summary_valid,
            "padding_mask": padding_mask,
        }
        for name, shape in expected.items():
            value = values[name]
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"V27b {name} shape differs: expected={shape}, actual={tuple(value.shape)}"
                )
        if batch <= 0 or length <= 0:
            raise ValueError("V27b requires a non-empty batch and sequence")
        if event_summary_valid.dtype != torch.bool or padding_mask.dtype != torch.bool:
            raise ValueError("V27b validity and padding masks must be boolean")
        float_values = (
            parent_state,
            event_representation,
            frozen_l2_logits,
            v27a_residual_logits,
        )
        if any(not value.is_floating_point() for value in float_values):
            raise TypeError("V27b feature inputs must be floating point")
        if any(value.device != parent_state.device for value in values.values()):
            raise ValueError("V27b feature inputs must share a device")
        if any(value.dtype != parent_state.dtype for value in float_values):
            raise ValueError("V27b feature inputs must share a dtype")
        if not all(torch.isfinite(value).all() for value in float_values):
            raise ValueError("V27b feature inputs must be finite")
        if (event_frame_values is None) != (event_frame_mask is None):
            raise ValueError("V27b direct frame values and mask must be provided together")
        if self.evidence_carrier == "direct_frame_gru":
            if event_frame_values is None or event_frame_mask is None:
                raise ValueError("V27b direct frame carrier requires frame values and mask")
            frame_count = (
                int(event_frame_values.shape[2])
                if event_frame_values.ndim == 4
                else -1
            )
            if event_frame_values.shape != (
                batch,
                length,
                frame_count,
                self.stable8_dim,
            ):
                raise ValueError("V27b direct frame values shape differs")
            if frame_count != self.max_frames_per_event:
                raise ValueError("V27b direct frame width differs from the data contract")
            if event_frame_mask.shape != (batch, length, frame_count):
                raise ValueError("V27b direct frame mask shape differs")
            if event_frame_mask.dtype != torch.bool:
                raise ValueError("V27b direct frame mask must be boolean")
            if event_frame_values.device != parent_state.device or event_frame_mask.device != parent_state.device:
                raise ValueError("V27b direct frame values and mask must share the feature device")
            if event_frame_values.dtype != parent_state.dtype:
                raise ValueError("V27b direct frame values must share the feature dtype")
            if not event_frame_values.is_floating_point() or not torch.isfinite(event_frame_values).all():
                raise ValueError("V27b direct frame values must be finite floating point")
            if bool(event_frame_mask.masked_select(padding_mask.unsqueeze(-1)).any()):
                raise ValueError("V27b padded events cannot contain direct frame tokens")
            inactive_frames = (~event_frame_mask).unsqueeze(-1).expand_as(event_frame_values)
            if bool(event_frame_values.masked_select(inactive_frames).ne(0).any()):
                raise ValueError("V27b inactive direct frame values must be zero")
            for row in range(batch):
                for event in range(length):
                    mask = event_frame_mask[row, event]
                    first_inactive = torch.nonzero(~mask, as_tuple=False).flatten()
                    if first_inactive.numel() and bool(mask[int(first_inactive[0]) :].any()):
                        raise ValueError("V27b direct frame masks must be suffix-padded")
            expected_valid = event_frame_mask.any(dim=-1) & ~padding_mask
            if not torch.equal(event_summary_valid, expected_valid):
                raise ValueError("V27b event validity and direct frame mask are inconsistent")
        elif event_frame_values is not None or event_frame_mask is not None:
            raise ValueError("V27b compressed carrier cannot receive direct frame tensors")
        self._suffix_padding(padding_mask)
        if bool(event_summary_valid.masked_select(padding_mask).any()):
            raise ValueError("V27b padded events cannot be reliable")
        for name, value in (
            ("parent_state", parent_state),
            ("event_representation", event_representation),
            ("frozen_l2_logits", frozen_l2_logits),
            ("v27a_residual_logits", v27a_residual_logits),
        ):
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError(f"V27b padded {name} values must be zero")
        if bool(
            v27a_residual_logits.abs().amax(dim=-1).gt(self.residual_bound + 1.0e-5).any()
        ):
            raise ValueError("V27b frozen V27a residual exceeds its registered bound")
        return batch, length

    def _encode_frame_event(
        self,
        event_frame_values: torch.Tensor | None,
        event_frame_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode reliable stable8 frames belonging to one event only."""

        if self.evidence_carrier != "direct_frame_gru":
            if event_frame_values is not None or event_frame_mask is not None:
                raise ValueError("V27b compressed carrier cannot encode direct frames")
            raise RuntimeError("V27b compressed carrier has no frame representation")
        if event_frame_values is None or event_frame_mask is None:
            raise ValueError("V27b direct frame event inputs are missing")
        if event_frame_values.ndim != 3 or event_frame_mask.ndim != 2:
            raise ValueError("V27b direct frame event tensors have invalid rank")
        result = self._encode_frame_sequence(
            event_frame_values.unsqueeze(1), event_frame_mask.unsqueeze(1)
        )
        return result[:, 0]

    def _encode_frame_sequence(
        self,
        event_frame_values: torch.Tensor,
        event_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-encode all event-local frame sequences without cross-event state."""

        if event_frame_values.ndim != 4 or event_frame_mask.ndim != 3:
            raise ValueError("V27b direct frame sequence tensors have invalid rank")
        batch, length, frame_count, width = event_frame_values.shape
        if width != self.stable8_dim or frame_count != self.max_frames_per_event:
            raise ValueError("V27b direct frame sequence dimensions differ")
        if event_frame_mask.shape != (batch, length, frame_count):
            raise ValueError("V27b direct frame sequence mask dimensions differ")
        result = event_frame_values.new_zeros(
            (batch, length, self.frame_encoder_hidden_dim)
        )
        flat_values = event_frame_values.reshape(batch * length, frame_count, width)
        flat_mask = event_frame_mask.reshape(batch * length, frame_count)
        selected_rows = torch.nonzero(flat_mask.any(dim=-1), as_tuple=False).flatten()
        if not selected_rows.numel():
            return result
        counts = flat_mask[selected_rows].sum(dim=-1).to(dtype=torch.long)
        max_tokens = int(counts.max().item())
        tokens = flat_values[selected_rows, :max_tokens]
        lengths = counts
        if self.frame_token_projection is None or self.frame_encoder is None:
            raise RuntimeError("V27b direct frame encoder is not constructed")
        projected = self.frame_token_projection(tokens)
        if self.activation == "silu":
            projected = torch.nn.functional.silu(projected)
        else:
            raise ValueError("unsupported V27b activation")
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.frame_encoder(packed)
        flat_result = result.reshape(batch * length, self.frame_encoder_hidden_dim)
        flat_result = flat_result.index_copy(0, selected_rows, hidden[-1])
        return flat_result.reshape(batch, length, self.frame_encoder_hidden_dim)

    def _decision_state(
        self,
        frozen_l2_logits: torch.Tensor,
        v27a_residual_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Build the nine-dimensional frozen decision condition."""

        margins = torch.stack(
            (
                frozen_l2_logits[..., 0] - frozen_l2_logits[..., 1],
                frozen_l2_logits[..., 0] - frozen_l2_logits[..., 2],
                frozen_l2_logits[..., 1] - frozen_l2_logits[..., 2],
            ),
            dim=-1,
        )
        result = torch.cat((frozen_l2_logits, margins, v27a_residual_logits), dim=-1)
        if result.shape[-1] != self.decision_state_dim:
            raise RuntimeError("V27b decision state width differs")
        return result

    def _raw_delta(
        self,
        parent_state: torch.Tensor,
        event_representation: torch.Tensor,
        frozen_l2_logits: torch.Tensor,
        v27a_residual_logits: torch.Tensor,
        frame_representation: torch.Tensor | None,
    ) -> torch.Tensor:
        features = self._feature_sequence(
            parent_state,
            event_representation,
            frozen_l2_logits,
            v27a_residual_logits,
            frame_representation,
        )
        return self.correction_projection(features).squeeze(-1)

    def _feature_sequence(
        self,
        parent_state: torch.Tensor,
        event_representation: torch.Tensor,
        frozen_l2_logits: torch.Tensor,
        v27a_residual_logits: torch.Tensor,
        frame_representation: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build correction features for a complete causal batch."""

        decision = self._decision_state(frozen_l2_logits, v27a_residual_logits)
        components = [parent_state, event_representation]
        if self.evidence_carrier == "direct_frame_gru":
            if frame_representation is None:
                raise ValueError("V27b direct frame representation is missing")
            components.append(frame_representation)
        components.append(decision)
        features = torch.cat(components, dim=-1)
        if features.shape[-1] != self.input_dim:
            raise RuntimeError("V27b conditional feature width differs")
        return features

    def _bounded_delta(self, raw_delta: torch.Tensor) -> torch.Tensor:
        return self.pairwise_delta_bound * torch.tanh(raw_delta)

    @staticmethod
    def _correction_from_delta(delta: torch.Tensor) -> torch.Tensor:
        return torch.stack((delta * 0.5, -delta * 0.5, torch.zeros_like(delta)), dim=-1)

    def step_event(
        self,
        *,
        parent_state: torch.Tensor,
        event_representation: torch.Tensor,
        frozen_l2_logits: torch.Tensor,
        v27a_residual_logits: torch.Tensor,
        event_summary_valid: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        event_frame_values: torch.Tensor | None = None,
        event_frame_mask: torch.Tensor | None = None,
        frame_representation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process one active/padded event using the same online contract."""

        if parent_state.ndim != 2 or parent_state.shape[-1] != self.parent_state_dim:
            raise ValueError("V27b step parent_state must have shape [B,H]")
        batch = int(parent_state.shape[0])
        for name, value, width in (
            ("event_representation", event_representation, self.event_representation_dim),
            ("frozen_l2_logits", frozen_l2_logits, self.class_count),
            ("v27a_residual_logits", v27a_residual_logits, self.class_count),
        ):
            if value.shape != (batch, width):
                raise ValueError(f"V27b step {name} shape differs")
        if event_summary_valid.shape != (batch,) or event_summary_valid.dtype != torch.bool:
            raise ValueError("V27b step event_summary_valid differs")
        values = (
            parent_state,
            event_representation,
            frozen_l2_logits,
            v27a_residual_logits,
        )
        if any(value.device != parent_state.device for value in values):
            raise ValueError("V27b step values must share a device")
        if any(value.dtype != parent_state.dtype for value in values):
            raise ValueError("V27b step values must share a dtype")
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("V27b step values must be finite")
        if (event_frame_values is None) != (event_frame_mask is None):
            raise ValueError("V27b step direct frame values and mask must be paired")
        if self.evidence_carrier == "direct_frame_gru":
            if event_frame_values is None or event_frame_mask is None:
                raise ValueError("V27b step direct frame inputs are missing")
            if event_frame_values.ndim != 3 or event_frame_values.shape[0] != batch:
                raise ValueError("V27b step direct frame values shape differs")
            if event_frame_values.shape[1] != self.max_frames_per_event:
                raise ValueError("V27b step direct frame width differs")
            if event_frame_values.shape[-1] != self.stable8_dim:
                raise ValueError("V27b step stable8 width differs")
            if event_frame_mask.shape != event_frame_values.shape[:2] or event_frame_mask.dtype != torch.bool:
                raise ValueError("V27b step direct frame mask differs")
            if event_frame_values.device != parent_state.device or event_frame_mask.device != parent_state.device:
                raise ValueError("V27b step direct frame inputs must share a device")
            if event_frame_values.dtype != parent_state.dtype:
                raise ValueError("V27b step direct frame values must share a dtype")
            if not torch.isfinite(event_frame_values).all():
                raise ValueError("V27b step direct frame values must be finite")
            inactive = (~event_frame_mask).unsqueeze(-1).expand_as(event_frame_values)
            if bool(event_frame_values.masked_select(inactive).ne(0).any()):
                raise ValueError("V27b step inactive direct frames must be zero")
            for row in range(batch):
                row_mask = event_frame_mask[row]
                first_inactive = torch.nonzero(~row_mask, as_tuple=False).flatten()
                if first_inactive.numel() and bool(
                    row_mask[int(first_inactive[0]) :].any()
                ):
                    raise ValueError("V27b step direct frame mask must be suffix-padded")
            if not torch.equal(event_summary_valid, event_frame_mask.any(dim=-1)):
                raise ValueError("V27b step validity and direct frame mask are inconsistent")
        elif event_frame_values is not None or event_frame_mask is not None:
            raise ValueError("V27b compressed step cannot receive direct frames")
        previous = self._validate_state(
            state,
            batch_size=batch,
            device=parent_state.device,
            dtype=parent_state.dtype,
        )
        reliable = event_summary_valid
        if self.evidence_carrier == "direct_frame_gru":
            if frame_representation is None:
                frame_representation = self._encode_frame_event(
                    event_frame_values, event_frame_mask
                )
            elif frame_representation.shape != (
                batch,
                self.frame_encoder_hidden_dim,
            ):
                raise ValueError("V27b direct frame representation shape differs")
            elif frame_representation.device != parent_state.device or frame_representation.dtype != parent_state.dtype:
                raise ValueError("V27b direct frame representation device/dtype differs")
            elif not torch.isfinite(frame_representation).all():
                raise ValueError("V27b direct frame representation must be finite")
        elif frame_representation is not None:
            raise ValueError("V27b compressed step cannot receive a frame representation")
        raw = self._raw_delta(
            parent_state,
            event_representation,
            frozen_l2_logits,
            v27a_residual_logits,
            frame_representation,
        )
        proposed = self._bounded_delta(raw)
        next_previous = torch.where(reliable, proposed, previous)
        # A missing/invalid stable8 event may not contribute a correction.  We
        # still retain the prior continuation state internally, so a later
        # reliable event starts from the same online state as an offline pass.
        applied_delta = torch.where(reliable, proposed, torch.zeros_like(proposed))
        increment = torch.where(reliable, proposed - previous, torch.zeros_like(proposed))
        correction = self._correction_from_delta(applied_delta)
        return {
            "previous_delta": next_previous,
            "delta": applied_delta,
            "delta_increment": increment,
            "correction": correction,
            "raw_delta": raw,
            "frame_representation": (
                frame_representation
                if frame_representation is not None
                else parent_state.new_zeros((batch, 0))
            ),
        }

    def forward(
        self,
        *,
        parent_state: torch.Tensor,
        event_representation: torch.Tensor,
        frozen_l2_logits: torch.Tensor,
        v27a_residual_logits: torch.Tensor,
        event_summary_valid: torch.Tensor,
        padding_mask: torch.Tensor,
        event_frame_values: torch.Tensor | None = None,
        event_frame_mask: torch.Tensor | None = None,
        initial_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch, length = self._validate_inputs(
            parent_state,
            event_representation,
            frozen_l2_logits,
            v27a_residual_logits,
            event_summary_valid,
            padding_mask,
            event_frame_values,
            event_frame_mask,
        )
        state = self.initial_state(
            batch,
            device=parent_state.device,
            dtype=parent_state.dtype,
        ) if initial_state is None else dict(initial_state)
        previous = self._validate_state(
            state,
            batch_size=batch,
            device=parent_state.device,
            dtype=parent_state.dtype,
        )
        deltas: list[torch.Tensor] = []
        increments: list[torch.Tensor] = []
        frame_sequence = (
            self._encode_frame_sequence(event_frame_values, event_frame_mask)
            if self.evidence_carrier == "direct_frame_gru"
            else None
        )
        raw_delta_sequence = self.correction_projection(
            self._feature_sequence(
                parent_state,
                event_representation,
                frozen_l2_logits,
                v27a_residual_logits,
                frame_sequence,
            )
        ).squeeze(-1)
        proposed_sequence = self._bounded_delta(raw_delta_sequence)
        active_sequence = ~padding_mask
        reliable_sequence = event_summary_valid & active_sequence
        for event_index in range(length):
            active = active_sequence[:, event_index]
            reliable = reliable_sequence[:, event_index]
            proposed = proposed_sequence[:, event_index]
            next_previous = torch.where(reliable, proposed, previous)
            delta = torch.where(reliable, proposed, torch.zeros_like(proposed))
            increment = torch.where(
                reliable, proposed - previous, torch.zeros_like(proposed)
            )
            # Padding is not an event: it must not alter continuation state.
            previous = torch.where(active, next_previous, previous)
            deltas.append(delta.masked_fill(~active, 0.0))
            increments.append(increment.masked_fill(~active, 0.0))
        delta_sequence = torch.stack(deltas, dim=1)
        delta_increment = torch.stack(increments, dim=1)
        correction_sequence = self._correction_from_delta(delta_sequence)
        raw_delta_sequence = raw_delta_sequence.masked_fill(
            ~active_sequence, 0.0
        )
        sequence_logits = frozen_l2_logits + correction_sequence
        if not torch.isfinite(sequence_logits).all():
            raise FloatingPointError("V27b conditional residual produced non-finite logits")
        if bool(correction_sequence[..., 2].ne(0.0).any()):
            raise AssertionError("V27b correction changed Balloon logit")
        if bool(correction_sequence.sum(dim=-1).abs().gt(1.0e-6).any()):
            raise AssertionError("V27b correction is not zero-sum")
        if bool(correction_sequence.abs().gt(self.residual_bound + 1.0e-5).any()):
            raise AssertionError("V27b correction exceeds residual bound")
        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "sequence_logits": sequence_logits,
            "correction_logits": correction_sequence,
            "pairwise_delta_sequence": delta_sequence,
            "pairwise_delta_increment": delta_increment,
            "raw_pairwise_delta_sequence": raw_delta_sequence,
            "frozen_l2_logits": frozen_l2_logits,
            "final_state": {"previous_delta": previous},
        }
        if self.evidence_carrier == "direct_frame_gru":
            result["frame_representation_sequence"] = frame_sequence.masked_fill(
                ~active_sequence.unsqueeze(-1), 0.0
            )
        return result


def build_v27b_model(config: V27BConfig, *, seed: int) -> V27BConditionalPairResidual:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27b initialization seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return V27BConditionalPairResidual(config)


__all__ = ["V27BConditionalPairResidual", "build_v27b_model"]
