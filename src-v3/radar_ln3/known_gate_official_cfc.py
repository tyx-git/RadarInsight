"""V23 KnownGate classifier using the official ``ncps.torch.CfC`` backbone."""

from __future__ import annotations

import hashlib
import numpy as np
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KnownGateConfig,
    KnownGateFusionLinear,
    build_known_gate_model_for_config,
    seed_everything,
)
from .model import RDSpatialEncoder
from .official_cfc import OfficialCfCCausalEncoder
from .official_cfc_state_adapters import (
    ADAPTER_NONE,
    ExternalOfficialCfCStateAdapter,
    ExternalStateAdapterSpec,
)


class OfficialCfCKnownGateClassifier(nn.Module):
    """V23's pure official-CfC KnownGate baseline.

    V23 deliberately reuses V21's ordinary TR/RD front end and output heads,
    but replaces the complete project-local recurrent core with dense official
    ``ncps.torch.CfC``.  The baseline has no appended experimental evidence.
    """

    def __init__(
        self,
        config: KnownGateConfig,
        *,
        state_adapter_spec: ExternalStateAdapterSpec | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        model = config.model
        if model.rd_temporal_residual_dim != 0:
            raise ValueError("V23 official CfC excludes the rejected V22a residual branch")
        if model.evidence_dim != 0:
            raise ValueError("V23 pure official CfC baseline forbids appended evidence")
        self.task_name = "known_gate"
        self.class_names = KNOWN_GATE_CLASS_NAMES
        self.evidence_dim = model.evidence_dim
        self.rd_temporal_residual_dim = 0
        self.tr_encoder = nn.Sequential(
            nn.Linear(model.tr_value_dim + model.tr_missing_dim, model.hidden_dim),
            nn.LayerNorm(model.hidden_dim),
            nn.SiLU(),
            nn.Linear(model.hidden_dim, model.hidden_dim),
            nn.SiLU(),
        )
        self.rd_encoder = RDSpatialEncoder(
            model.rd_embedding_dim, aggregation_mode=model.rd_frame_aggregation
        )
        self.base_fusion_input_dim = model.hidden_dim + model.rd_embedding_dim + model.rd_quality_dim
        self.fusion_input_dim = self.base_fusion_input_dim + self.evidence_dim
        self.fusion = nn.Sequential(
            KnownGateFusionLinear(
                self.fusion_input_dim,
                model.liquid_hidden_dim,
                base_input_dim=self.base_fusion_input_dim,
                zero_input_dim=0,
            ),
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.SiLU(),
        )
        self.liquid = OfficialCfCCausalEncoder(
            model.liquid_hidden_dim,
            model.liquid_hidden_dim,
            backbone_units=model.liquid_hidden_dim,
            backbone_layers=1,
            backbone_dropout=0.0,
            mixed_memory=False,
        )
        self.state_adapter_spec = state_adapter_spec or ExternalStateAdapterSpec(
            kind=ADAPTER_NONE,
            hidden_dim=model.liquid_hidden_dim,
        )
        if self.state_adapter_spec.hidden_dim != model.liquid_hidden_dim:
            raise ValueError("V23 official-CfC state adapter hidden dimension differs from model")
        self.state_adapter = ExternalOfficialCfCStateAdapter(self.state_adapter_spec)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.Linear(model.liquid_hidden_dim, len(KNOWN_GATE_CLASS_NAMES)),
        )

    def backbone_specification(self) -> dict[str, Any]:
        return self.liquid.specification()

    def external_state_adapter_specification(self) -> dict[str, Any]:
        return self.state_adapter.specification()

    def _evidence_input(
        self,
        features: torch.Tensor | None,
        valid: torch.Tensor | None,
        *,
        leading_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.evidence_dim == 0:
            if features is not None or valid is not None:
                raise ValueError("V23 baseline-shaped input must not include evidence")
            return None
        expected = (*leading_shape, self.evidence_dim)
        if features is None or valid is None or features.shape != expected or valid.shape != expected:
            raise ValueError("V23 official CfC evidence value/valid shape is invalid")
        if valid.dtype != torch.bool or not torch.isfinite(features).all():
            raise ValueError("V23 official CfC evidence must be finite with boolean validity")
        if bool((features < 0.0).any()) or bool((features > 1.0).any()):
            raise ValueError("V23 official CfC evidence must stay in [0,1]")
        if bool((features.masked_select(~valid) != 0).any()):
            raise ValueError("V23 invalid evidence must be exactly zero")
        return features * valid.to(dtype=dtype)

    def _point_events(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        known_gate_features: torch.Tensor | None,
        known_gate_feature_valid: torch.Tensor | None,
        leading_shape: tuple[int, ...],
    ) -> torch.Tensor:
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        if tr_features.ndim == 3:
            rd_embedding = self.rd_encoder(rd_frames, rd_frame_mask)
        elif tr_features.ndim == 2:
            rd_embedding = self.rd_encoder(
                rd_frames.unsqueeze(1), rd_frame_mask.unsqueeze(1)
            ).squeeze(1)
        else:
            raise ValueError("V23 official CfC TR features must be rank 2 or 3")
        inputs = [tr_embedding, rd_embedding, rd_quality]
        evidence = self._evidence_input(
            known_gate_features,
            known_gate_feature_valid,
            leading_shape=leading_shape,
            dtype=tr_features.dtype,
        )
        if evidence is not None:
            inputs.append(evidence)
        return self.fusion(torch.cat(inputs, dim=-1))

    def forward(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        tr_padding_mask: torch.Tensor,
        known_gate_features: torch.Tensor | None = None,
        known_gate_feature_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model = self.config.model
        if tr_features.ndim != 3 or tr_features.shape[-1] != model.tr_value_dim:
            raise ValueError("V23 TR features must have shape [batch,sequence,14]")
        if tr_missing_mask.shape != (*tr_features.shape[:2], model.tr_missing_dim):
            raise ValueError("V23 TR missing-mask shape is invalid")
        if (
            tr_dt_seconds.shape != tr_features.shape[:2]
            or tr_dt_valid.shape != tr_features.shape[:2]
            or tr_dt_reason.shape != tr_features.shape[:2]
            or tr_padding_mask.shape != tr_features.shape[:2]
        ):
            raise ValueError("V23 dt/padding tensors must align with the sequence")
        if rd_quality.shape != (*tr_features.shape[:2], model.rd_quality_dim):
            raise ValueError("V23 RD-quality shape is invalid")
        if not all(
            torch.isfinite(value).all()
            for value in (tr_features, tr_missing_mask, tr_dt_seconds, rd_frames, rd_quality)
        ):
            raise ValueError("V23 official CfC model received a non-finite base input")
        if known_gate_feature_valid is not None and bool(
            known_gate_feature_valid.masked_select(tr_padding_mask.unsqueeze(-1)).any()
        ):
            raise ValueError("V23 padded events contain valid course evidence")
        point_events = self._point_events(
            tr_features=tr_features,
            tr_missing_mask=tr_missing_mask,
            rd_frames=rd_frames,
            rd_frame_mask=rd_frame_mask,
            rd_quality=rd_quality,
            known_gate_features=known_gate_features,
            known_gate_feature_valid=known_gate_feature_valid,
            leading_shape=tuple(tr_features.shape[:2]),
        )
        state = self.liquid(
            point_events,
            tr_dt_seconds,
            tr_padding_mask,
            dt_valid=tr_dt_valid,
            dt_reason=tr_dt_reason,
            # Preserve frozen V23 exactly when adaptation is disabled: the
            # original parent-wise official-CfC path remains the baseline.
            # V23a/V23b/V23c alone opt into the explicit per-event external
            # adapter path below.
            state_adapter=(
                None
                if self.state_adapter_spec.kind == ADAPTER_NONE
                else self.state_adapter
            ),
        )
        logits = self.classifier(state.sequence)
        # Padding never reaches the official recurrence.  Keep the readout's
        # padded bias values untouched so V23 has the same observable tensor
        # contract as the frozen V21e/V22 controls; loss and evaluation use
        # only true causal lengths and therefore never supervise/read them.
        lengths = (~tr_padding_mask).long().sum(dim=1)
        indices = lengths - 1
        batch_indices = torch.arange(tr_features.shape[0], device=tr_features.device)
        return {
            "sequence_logits": logits,
            "window_logits": logits[batch_indices, indices],
            "liquid_sequence": state.sequence,
            "liquid_final_state": state.final_state,
            "dt_clamped_mask": state.dt_clamped_mask,
            "dt_zero_mask": state.dt_zero_mask,
            "dt_invalid_mask": state.dt_invalid_mask,
            "dt_first_event_mask": state.dt_first_event_mask,
            "dt_zero_delta_mask": state.dt_zero_delta_mask,
            "rd_temporal_residual": logits.new_zeros((*logits.shape[:-1], 0)),
        }

    def forward_online_point(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        liquid_state: torch.Tensor | None = None,
        known_gate_features: torch.Tensor | None = None,
        known_gate_feature_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 2 or tr_missing_mask.ndim != 2 or rd_quality.ndim != 2:
            raise ValueError("V23 online TR/RD-quality tensors must be rank 2")
        point_event = self._point_events(
            tr_features=tr_features,
            tr_missing_mask=tr_missing_mask,
            rd_frames=rd_frames,
            rd_frame_mask=rd_frame_mask,
            rd_quality=rd_quality,
            known_gate_features=known_gate_features,
            known_gate_feature_valid=known_gate_feature_valid,
            leading_shape=(tr_features.shape[0],),
        )
        state, dt_clamped, dt_zero = self.liquid.step(
            point_event,
            tr_dt_seconds,
            liquid_state,
            tr_dt_valid,
            tr_dt_reason,
            (
                None
                if self.state_adapter_spec.kind == ADAPTER_NONE
                else self.state_adapter
            ),
        )
        logits = self.classifier(self.liquid.readout(state))
        return {
            "window_logits": logits,
            "liquid_state": state,
            "dt_clamped_mask": dt_clamped,
            "dt_zero_mask": dt_zero,
            "dt_invalid_mask": ~tr_dt_valid,
            "rd_temporal_residual": logits.new_zeros((*logits.shape[:-1], 0)),
        }


def _state_subset_sha256(
    state: dict[str, torch.Tensor], *, names: set[str]
) -> str:
    """Hash a named state subset without importing a training entry point."""

    digest = hashlib.sha256()
    for name in sorted(names):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash complete initialization state with the same stable tensor ordering."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_official_cfc_model_from_frozen_source(
    project_root: Path,
    source_config: KnownGateConfig,
    *,
    state_adapter_spec: ExternalStateAdapterSpec | None = None,
) -> tuple[OfficialCfCKnownGateClassifier, dict[str, Any]]:
    """Construct V23 with every non-recurrent tensor copied from its Control.

    The recurrent state equations cannot share weights across the historical
    project-local encoder and the official CfC.  All compatible TR/RD front
    end, fusion and classifier parameters/buffers *can* and must be copied
    exactly.  This prevents differences in random-number consumption during
    CfC construction from becoming an unregistered second experiment factor.
    """

    source_model, source_reference, source_pairing = build_known_gate_model_for_config(
        project_root, source_config
    )
    del source_reference
    source_state = dict(source_model.state_dict())
    source_initial_hash = _state_dict_sha256(source_state)
    source_parameter_names = {name for name, _value in source_model.named_parameters()}
    source_buffer_names = {name for name, _value in source_model.named_buffers()}

    # Deliberately give the newly introduced official recurrent module a
    # reproducible seed of its own, then overwrite every compatible tensor.
    # The training entry point resets all RNGs again before optimization.
    seed_everything(source_config.train.seed)
    official_model = OfficialCfCKnownGateClassifier(
        source_config,
        state_adapter_spec=state_adapter_spec,
    )
    official_state = official_model.state_dict()
    source_names = {name for name in source_state if not name.startswith("liquid.")}
    official_names = {
        name
        for name in official_state
        if not name.startswith("liquid.") and not name.startswith("state_adapter.")
    }
    if source_names != official_names:
        missing = sorted(source_names - official_names)
        unexpected = sorted(official_names - source_names)
        raise ValueError(
            "V23 non-recurrent source/official state namespaces differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name in sorted(source_names):
            source_value = source_state[name]
            official_value = official_state[name]
            if source_value.shape != official_value.shape or source_value.dtype != official_value.dtype:
                raise ValueError(f"V23 non-recurrent source/official state differs at {name}")
            official_value.copy_(source_value)
    copied_state = official_model.state_dict()
    if any(not torch.equal(source_state[name], copied_state[name]) for name in source_names):
        raise AssertionError("V23 non-recurrent initialization copy is not tensor-exact")
    parameter_names = {name for name, _value in official_model.named_parameters()}
    buffer_names = {name for name, _value in official_model.named_buffers()}
    audit = {
        "strategy": "copy_frozen_source_non_recurrent_state_then_seed_official_cfc",
        "source_initial_model_state_sha256": source_initial_hash,
        "source_initialization_pairing": dict(source_pairing),
        "shared_non_recurrent_state_sha256": _state_subset_sha256(
            copied_state, names=source_names
        ),
        "shared_non_recurrent_tensor_count": len(source_names),
        "shared_non_recurrent_parameter_count": len(source_names & parameter_names),
        "shared_non_recurrent_buffer_count": len(source_names & buffer_names),
        "source_non_recurrent_parameter_count": len(source_names & source_parameter_names),
        "source_non_recurrent_buffer_count": len(source_names & source_buffer_names),
        "shared_non_recurrent_state_exactly_equal": True,
        "official_recurrent_state_is_distinct_by_architecture": True,
        "external_state_adapter": official_model.external_state_adapter_specification(),
    }
    # Keep the complete source model out of the training graph and CUDA memory.
    del source_model
    return official_model, audit


__all__ = [
    "OfficialCfCKnownGateClassifier",
    "build_official_cfc_model_from_frozen_source",
]
