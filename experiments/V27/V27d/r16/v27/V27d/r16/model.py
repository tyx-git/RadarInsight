"""Liquid V2 recurrence over the local time axis of each RD event."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from radar_ln3.data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)
from radar_ln3.liquid_v2 import LiquidV2Cell
from v27.v27_baseline.model import V27LiquidV2TaskModel
from v27.v27_baseline.protocol import V27_L2_TASK_NAME
from v27.v27a.model import V27AEvidenceResidual
from v27.v27a.protocol import load_v27a_config

from .protocol import (
    R16_CHECKPOINT_SCHEMA,
    R16_CLASS_NAMES,
    R16Context,
    R16_EXPERIMENT_ID,
    R16_ARMS,
    project_path,
)


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    from radar_ln3.runtime import canonical_sha256

    parts: list[Any] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r16 state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((name, str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


class R16ContinuousRDEncoder(nn.Module):
    """Encode RD frames spatially, then evolve a raw Liquid V2 frame state.

    The state is reset for every TR event.  The masked mean remains the base
    evidence and the learned residual is active only when at least two
    monotonic frame timestamps are available.  This preserves a clean
    missing/single-frame boundary and makes the intervention zero at start.
    """

    def __init__(
        self,
        spatial_encoder: nn.Module,
        *,
        embedding_dim: int,
        rd_height: int,
        rd_width: int,
        frame_tau_min_seconds: float,
        frame_tau_max_seconds: float,
        frame_zero_dt_floor_seconds: float,
        frame_dt_max_seconds: float,
        frame_invalid_observation_gain: float,
        residual_scale: float,
        time_tolerance_seconds: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.rd_height = int(rd_height)
        self.rd_width = int(rd_width)
        self.time_tolerance_seconds = float(time_tolerance_seconds)
        self.residual_scale = float(residual_scale)
        if self.embedding_dim <= 0 or self.rd_height <= 0 or self.rd_width <= 0:
            raise ValueError("V27d-r16 RD encoder dimensions must be positive")
        if self.residual_scale <= 0.0 or self.time_tolerance_seconds < 0.0:
            raise ValueError("V27d-r16 RD encoder bounds are invalid")
        self.network = copy.deepcopy(spatial_encoder.network)
        self.projection = copy.deepcopy(spatial_encoder.projection)
        self.frame_cell = LiquidV2Cell(
            self.embedding_dim,
            self.embedding_dim,
            tau_min_seconds=float(frame_tau_min_seconds),
            tau_max_seconds=float(frame_tau_max_seconds),
            zero_dt_event_floor_seconds=float(frame_zero_dt_floor_seconds),
            dt_max_seconds=float(frame_dt_max_seconds),
            invalid_dt_update_mode="bounded_observation",
            invalid_dt_observation_gain=float(frame_invalid_observation_gain),
        )
        self.residual_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        nn.init.zeros_(self.residual_projection.weight)
        self._frame_times: torch.Tensor | None = None
        self._event_time_valid: torch.Tensor | None = None
        self.last_diagnostics: dict[str, int | float] = {}

    @classmethod
    def from_spatial(
        cls,
        spatial_encoder: nn.Module,
        *,
        config: Mapping[str, Any],
    ) -> "R16ContinuousRDEncoder":
        return cls(
            spatial_encoder,
            embedding_dim=int(config["spatial_embedding_dim"]),
            rd_height=int(config["rd_height"]),
            rd_width=int(config["rd_width"]),
            frame_tau_min_seconds=float(config["frame_tau_min_seconds"]),
            frame_tau_max_seconds=float(config["frame_tau_max_seconds"]),
            frame_zero_dt_floor_seconds=float(config["frame_zero_dt_floor_seconds"]),
            frame_dt_max_seconds=float(config["frame_dt_max_seconds"]),
            frame_invalid_observation_gain=float(config["frame_invalid_observation_gain"]),
            residual_scale=float(config["residual_scale"]),
            time_tolerance_seconds=float(config["frame_time_tolerance_seconds"]),
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "class": "R16ContinuousRDEncoder",
            "spatial_frontend": "V27RDSpatialEncoder.network_and_projection",
            "embedding_dim": self.embedding_dim,
            "frame_cell": "radar_ln3.liquid_v2.LiquidV2Cell",
            "frame_memory": "raw_frame_state",
            "frame_compute_state": "LayerNorm(raw_previous_frame_state)",
            "time_input": "canonical per-frame time_seconds; adjacent delta only",
            "frame_state_reset": "each_current_TR_event",
            "aggregation": "masked_mean_plus_zero_initialized_liquid_residual",
            "residual_scale": self.residual_scale,
            "time_tolerance_seconds": self.time_tolerance_seconds,
            "single_or_missing_policy": "masked_mean_and_zero_residual",
            "nonmonotonic_policy": "masked_mean_and_zero_residual;diagnostic_count_recorded",
            "gru_used": False,
        }

    def set_frame_times(
        self,
        frame_times: torch.Tensor,
        event_time_valid: torch.Tensor,
    ) -> None:
        self._frame_times = frame_times
        self._event_time_valid = event_time_valid

    def clear_frame_times(self) -> None:
        self._frame_times = None
        self._event_time_valid = None

    def _encode_frames(
        self,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        if rd_frames.ndim != 6 or tuple(rd_frames.shape[3:]) != (2, self.rd_height, self.rd_width):
            raise ValueError("V27d-r16 RD frames have an invalid shape")
        if rd_frame_mask.shape != rd_frames.shape[:3] or rd_frame_mask.dtype != torch.bool:
            raise ValueError("V27d-r16 RD frame mask has an invalid shape/dtype")
        if not rd_frames.is_floating_point() or not torch.isfinite(rd_frames).all():
            raise ValueError("V27d-r16 RD frames must be finite floating tensors")
        batch, length, frame_count = rd_frames.shape[:3]
        flattened = rd_frames.reshape(batch * length * frame_count, 2, self.rd_height, self.rd_width)
        return self.projection(self.network(flattened)).reshape(
            batch, length, frame_count, self.embedding_dim
        )

    @staticmethod
    def _masked_mean(
        frame_embeddings: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = frame_mask.to(dtype=frame_embeddings.dtype).unsqueeze(-1)
        return (frame_embeddings * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

    def _validate_frame_times(
        self,
        frame_times: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> None:
        if frame_times.shape != frame_mask.shape or not frame_times.is_floating_point():
            raise ValueError("V27d-r16 frame timestamps must match the frame mask")
        if frame_times.device != frame_mask.device or not torch.isfinite(frame_times).all():
            raise ValueError("V27d-r16 frame timestamps must be finite and colocated")
        if bool(frame_times.masked_select(~frame_mask).ne(0.0).any()):
            raise ValueError("V27d-r16 inactive frame timestamps must be zero")
        for row in range(frame_mask.shape[0]):
            for event in range(frame_mask.shape[1]):
                mask = frame_mask[row, event]
                inactive = torch.nonzero(~mask, as_tuple=False).flatten()
                if inactive.numel() and bool(mask[int(inactive[0]) :].any()):
                    raise ValueError("V27d-r16 frame masks must be suffix-active")

    def _temporal_aggregate(
        self,
        frame_embeddings: torch.Tensor,
        frame_mask: torch.Tensor,
        frame_times: torch.Tensor,
        event_time_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        batch, length, frame_count, _ = frame_embeddings.shape
        if event_time_valid.shape != (batch, length) or event_time_valid.dtype != torch.bool:
            raise ValueError("V27d-r16 event frame-time validity has an invalid shape/dtype")
        active_count = frame_mask.to(dtype=torch.long).sum(dim=-1)
        adjacent = frame_mask[..., 1:] & frame_mask[..., :-1]
        deltas = frame_times[..., 1:] - frame_times[..., :-1]
        negative = adjacent & deltas.lt(-self.time_tolerance_seconds)
        present = active_count.gt(0)
        derived_valid = present & active_count.ge(2) & ~negative.any(dim=-1)
        if not torch.equal(derived_valid, event_time_valid):
            raise ValueError("V27d-r16 frame-time validity differs from canonical timestamps")
        eligible = event_time_valid

        flat_count = batch * length
        embeddings = frame_embeddings.reshape(flat_count, frame_count, self.embedding_dim)
        masks = frame_mask.reshape(flat_count, frame_count)
        times = frame_times.reshape(flat_count, frame_count)
        eligible_flat = eligible.reshape(flat_count)
        state = frame_embeddings.new_zeros((flat_count, self.embedding_dim))
        seen = torch.zeros((flat_count,), dtype=torch.bool, device=frame_embeddings.device)
        zero_ordinal = torch.ones((flat_count,), dtype=torch.long, device=frame_embeddings.device)
        positive_delta_count = 0
        zero_delta_count = 0

        for frame_index in range(frame_count):
            active = masks[:, frame_index] & eligible_flat
            indices = torch.nonzero(active, as_tuple=False).flatten()
            if not indices.numel():
                continue
            first = ~seen.index_select(0, indices)
            if frame_index == 0 and bool((~first).any()):
                raise AssertionError("V27d-r16 first frame state boundary differs")
            if frame_index > 0 and bool(first.any()):
                raise AssertionError("V27d-r16 frame mask is not prefix-active")
            current_times = times.index_select(0, indices)[:, frame_index]
            if frame_index == 0:
                dt = torch.zeros_like(current_times)
                reason = torch.full_like(current_times, DT_REASON_FIRST_EVENT, dtype=torch.long)
                valid = torch.zeros_like(first)
                ordinal = torch.ones_like(zero_ordinal.index_select(0, indices))
            else:
                previous_times = times.index_select(0, indices)[:, frame_index - 1]
                raw_delta = current_times - previous_times
                dt = raw_delta.clamp_min(0.0)
                is_zero = raw_delta.abs().le(self.time_tolerance_seconds)
                reason = torch.where(
                    is_zero,
                    torch.full_like(raw_delta, DT_REASON_ZERO_DELTA, dtype=torch.long),
                    torch.full_like(raw_delta, DT_REASON_POSITIVE_DELTA, dtype=torch.long),
                )
                valid = reason.eq(DT_REASON_POSITIVE_DELTA)
                previous_ordinal = zero_ordinal.index_select(0, indices)
                ordinal = torch.where(
                    is_zero,
                    previous_ordinal + 1,
                    torch.ones_like(previous_ordinal),
                )
                positive_delta_count += int(valid.sum().item())
                zero_delta_count += int((~valid).sum().item())
            step = self.frame_cell.step(
                embeddings.index_select(0, indices)[:, frame_index],
                state.index_select(0, indices),
                dt,
                valid,
                reason,
                ordinal,
            )
            state = state.index_copy(0, indices, step.next_state)
            seen = seen.index_fill(0, indices, True)
            zero_ordinal = zero_ordinal.index_copy(0, indices, ordinal)

        if bool((eligible_flat & ~seen).any()):
            raise AssertionError("V27d-r16 eligible frame event did not receive all frames")
        continuous = state.reshape(batch, length, self.embedding_dim)
        diagnostics: dict[str, int | float] = {
            "rd_event_count": int(batch * length),
            "rd_present_event_count": int(present.sum().item()),
            "temporal_eligible_event_count": int(eligible.sum().item()),
            "single_or_missing_event_count": int((present & active_count.lt(2)).sum().item()),
            "nonmonotonic_event_count": int((present & negative.any(dim=-1)).sum().item()),
            "positive_delta_count": positive_delta_count,
            "zero_delta_count": zero_delta_count,
            "used_delta_seconds_sum": float(deltas.masked_select(adjacent & eligible.unsqueeze(-1)).clamp_min(0.0).sum().item()),
        }
        return continuous, diagnostics

    def forward(
        self,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        frame_times = self._frame_times
        event_time_valid = self._event_time_valid
        if frame_times is None or event_time_valid is None:
            raise RuntimeError("V27d-r16 RD encoder requires an explicit frame-time sidecar")
        self._validate_frame_times(frame_times, rd_frame_mask)
        if event_time_valid.shape != rd_frame_mask.shape[:2] or event_time_valid.dtype != torch.bool:
            raise ValueError("V27d-r16 event frame-time validity has an invalid shape/dtype")
        embeddings = self._encode_frames(rd_frames, rd_frame_mask)
        mean = self._masked_mean(embeddings, rd_frame_mask)
        continuous, diagnostics = self._temporal_aggregate(
            embeddings,
            rd_frame_mask,
            frame_times,
            event_time_valid,
        )
        active_count = rd_frame_mask.to(dtype=torch.long).sum(dim=-1)
        adjacent = rd_frame_mask[..., 1:] & rd_frame_mask[..., :-1]
        deltas = frame_times[..., 1:] - frame_times[..., :-1]
        nonmonotonic = (adjacent & deltas.lt(-self.time_tolerance_seconds)).any(dim=-1)
        eligible = event_time_valid
        residual = self.residual_scale * torch.tanh(
            self.residual_projection(continuous - mean)
        )
        output = mean + residual * eligible.unsqueeze(-1).to(dtype=mean.dtype)
        output = output.masked_fill(active_count.eq(0).unsqueeze(-1), 0.0)
        if not torch.isfinite(output).all():
            raise FloatingPointError("V27d-r16 RD temporal encoder produced non-finite output")
        self.last_diagnostics = diagnostics
        return output


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _source_checkpoint(context: R16Context, seed: int) -> tuple[Path, dict[str, Any]]:
    path = project_path(
        context.project_root,
        context.config.source["paths"][f"r14_control_seed{seed}_checkpoint"],
        must_exist=True,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r16 r14 source checkpoint is not a mapping")
    if payload.get("experiment_id") != "V27d-r14" or payload.get("arm") != "control" or int(payload.get("seed", -1)) != int(seed) or int(payload.get("epoch", -1)) != 20:
        raise ValueError("V27d-r16 r14 source checkpoint identity differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27d-r16 r14 source checkpoint state hash differs")
    return path, dict(payload)


def _load_stable_model(context: R16Context) -> V27AEvidenceResidual:
    source = context.config.source
    config = load_v27a_config(context.project_root, source["paths"]["v27a_config"])
    model = V27AEvidenceResidual(config, arm="main")
    payload = torch.load(
        project_path(context.project_root, source["paths"]["v27a_main_checkpoint"], must_exist=True),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping) or payload.get("experiment_id") != "V27a" or payload.get("arm") != "main" or int(payload.get("epoch", -1)) != 20:
        raise ValueError("V27d-r16 V27a Main checkpoint identity differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27d-r16 V27a Main checkpoint lacks model state")
    model.load_state_dict(dict(state), strict=True)
    return model


class R16IntegratedL2Model(nn.Module):
    """Frozen r14 control endpoint with an optional RD frame-time branch."""

    def __init__(self, context: R16Context, *, arm: str, seed: int) -> None:
        super().__init__()
        if arm not in R16_ARMS or int(seed) not in (42, 44):
            raise ValueError("V27d-r16 arm or seed is invalid")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            trainable = V27LiquidV2TaskModel(context.baseline_config, task=V27_L2_TASK_NAME)
            frozen_reference = V27LiquidV2TaskModel(context.baseline_config, task=V27_L2_TASK_NAME)
            stable = _load_stable_model(context)
            _, source_payload = _source_checkpoint(context, int(seed))
            state = source_payload["model_state"]
            plain = nn.ModuleDict(
                {
                    "trainable_base_model": trainable,
                    "frozen_reference_base_model": frozen_reference,
                    "stable_model": stable,
                }
            )
            plain.load_state_dict(dict(state), strict=True)
        self.context = context
        self.arm = str(arm)
        self.seed = int(seed)
        self.trainable_base_model = plain["trainable_base_model"]
        self.frozen_reference_base_model = plain["frozen_reference_base_model"]
        self.stable_model = plain["stable_model"]
        if self.arm == "treatment":
            self.trainable_base_model.rd_encoder = R16ContinuousRDEncoder.from_spatial(
                self.trainable_base_model.rd_encoder,
                config={
                    **dict(context.config.model),
                    "rd_height": context.baseline_config.model.rd_height,
                    "rd_width": context.baseline_config.model.rd_width,
                    "frame_time_tolerance_seconds": context.config.data["frame_time_tolerance_seconds"],
                },
            )
        _freeze(self.frozen_reference_base_model)
        _freeze(self.stable_model)
        if self.arm == "control":
            _freeze(self.trainable_base_model)
        else:
            for parameter in self.trainable_base_model.parameters():
                parameter.requires_grad_(False)
            rd_encoder = self.trainable_base_model.rd_encoder
            if not isinstance(rd_encoder, R16ContinuousRDEncoder):
                raise AssertionError("V27d-r16 treatment RD encoder was not installed")
            for parameter in rd_encoder.frame_cell.parameters():
                parameter.requires_grad_(True)
            for parameter in rd_encoder.residual_projection.parameters():
                parameter.requires_grad_(True)
            rd_encoder.network.eval()
            rd_encoder.projection.eval()
            rd_encoder.frame_cell.train()
            rd_encoder.residual_projection.train()

    @property
    def class_names(self) -> tuple[str, ...]:
        return R16_CLASS_NAMES

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": R16_CHECKPOINT_SCHEMA,
            "experiment_id": R16_EXPERIMENT_ID,
            "arm": self.arm,
            "seed": self.seed,
            "class_names": list(R16_CLASS_NAMES),
            "direct_baseline": "V27d-r14 control epoch020",
            "outer_model": "V27 Liquid V2 L2 unchanged",
            "rd_aggregation": "masked_mean" if self.arm == "control" else self.trainable_base_model.rd_encoder.model_contract(),
            "treatment_trainable": "none" if self.arm == "control" else [
                "trainable_base_model.rd_encoder.frame_cell",
                "trainable_base_model.rd_encoder.residual_projection",
            ],
            "unknown_is_l2_classifier_class": False,
            "knownness_quality_l3": "not_in_scope",
            "loss": "parent_final_fixed_parent_count_ce",
        }

    def _validate_time_sidecars(
        self,
        base_inputs: Mapping[str, torch.Tensor],
        rd_frame_times: torch.Tensor,
        rd_frame_time_valid: torch.Tensor,
        rd_frame_time_fallback: torch.Tensor,
    ) -> None:
        expected = base_inputs["rd_frame_mask"].shape
        if rd_frame_times.shape != expected or rd_frame_time_valid.shape != expected[:2] or rd_frame_time_fallback.shape != expected[:2]:
            raise ValueError("V27d-r16 frame-time sidecar shapes differ")
        if rd_frame_times.device != base_inputs["rd_frames"].device or rd_frame_time_valid.device != rd_frame_times.device or rd_frame_time_fallback.device != rd_frame_times.device:
            raise ValueError("V27d-r16 frame-time sidecars are on different devices")
        if rd_frame_times.dtype != base_inputs["rd_frames"].dtype or rd_frame_time_valid.dtype != torch.bool or rd_frame_time_fallback.dtype != torch.bool:
            raise ValueError("V27d-r16 frame-time sidecar dtypes differ")
        if not torch.isfinite(rd_frame_times).all():
            raise ValueError("V27d-r16 frame timestamps are non-finite")
        mask = base_inputs["rd_frame_mask"]
        padding = base_inputs["tr_padding_mask"]
        if bool(rd_frame_times.masked_select(~mask).ne(0.0).any()):
            raise ValueError("V27d-r16 inactive frame timestamps are non-zero")
        if bool(rd_frame_times.masked_select(padding.unsqueeze(-1).expand_as(rd_frame_times)).ne(0.0).any()) or bool(rd_frame_time_valid.masked_select(padding).any()):
            raise ValueError("V27d-r16 padded frame-time sidecars are active")
        if bool((rd_frame_time_valid & rd_frame_time_fallback).any()):
            raise ValueError("V27d-r16 valid and fallback frame-time flags overlap")

    def forward(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        cached_v27_base_logits: torch.Tensor,
        stable_summary_tokens: torch.Tensor,
        stable_summary_valid: torch.Tensor,
        stable_frame_values: torch.Tensor,
        stable_frame_mask: torch.Tensor,
        rd_frame_times: torch.Tensor,
        rd_frame_time_valid: torch.Tensor,
        rd_frame_time_fallback: torch.Tensor,
        return_trace: bool = False,
    ) -> dict[str, Any]:
        self._validate_time_sidecars(
            base_inputs,
            rd_frame_times,
            rd_frame_time_valid,
            rd_frame_time_fallback,
        )
        encoder = self.trainable_base_model.rd_encoder
        if self.arm == "treatment":
            if not isinstance(encoder, R16ContinuousRDEncoder):
                raise AssertionError("V27d-r16 treatment encoder is missing")
            encoder.set_frame_times(rd_frame_times, rd_frame_time_valid)
        try:
            current = self.trainable_base_model(
                **dict(base_inputs),
                return_trace=return_trace,
            )
            rd_diagnostics = dict(getattr(encoder, "last_diagnostics", {})) if self.arm == "treatment" else {
                "rd_event_count": int(base_inputs["rd_frame_mask"].shape[0] * base_inputs["rd_frame_mask"].shape[1]),
                "temporal_eligible_event_count": 0,
                "positive_delta_count": 0,
                "zero_delta_count": 0,
            }
        finally:
            if self.arm == "treatment":
                assert isinstance(encoder, R16ContinuousRDEncoder)
                encoder.clear_frame_times()
        with torch.inference_mode():
            self.frozen_reference_base_model.eval()
            reference = self.frozen_reference_base_model(
                **dict(base_inputs),
                return_trace=False,
            )["sequence_logits"]
        stable = self.stable_model(
            baseline_logits=cached_v27_base_logits,
            summary_tokens=stable_summary_tokens,
            summary_valid=stable_summary_valid,
            frame_values=stable_frame_values,
            frame_mask=stable_frame_mask,
            padding_mask=base_inputs["tr_padding_mask"],
        )
        padding = base_inputs["tr_padding_mask"]
        if reference.shape != current["sequence_logits"].shape:
            raise ValueError("V27d-r16 current/reference logits shapes differ")
        combined = (
            cached_v27_base_logits
            + current["sequence_logits"]
            - reference
            + stable["residual_logits"]
        ).masked_fill(padding.unsqueeze(-1), 0.0)
        if not torch.isfinite(combined).all():
            raise FloatingPointError("V27d-r16 integrated logits are non-finite")
        return {
            "sequence_logits": combined,
            "native_sequence_logits": current["sequence_logits"],
            "frozen_reference_sequence_logits": reference,
            "stable_residual_logits": stable["residual_logits"],
            "rd_time_diagnostics": rd_diagnostics,
            "base_output": current,
        }


def build_r16_model(
    context: R16Context,
    *,
    arm: str,
    seed: int,
    device: torch.device,
) -> R16IntegratedL2Model:
    return R16IntegratedL2Model(context, arm=arm, seed=int(seed)).to(device)


def trainable_parameter_groups(model: R16IntegratedL2Model) -> list[nn.Parameter]:
    values = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if model.arm == "control" and values:
        raise AssertionError("V27d-r16 control must have no trainable parameters")
    if model.arm == "treatment" and not values:
        raise ValueError("V27d-r16 treatment has no trainable temporal parameters")
    if model.arm == "treatment":
        allowed_prefixes = (
            "trainable_base_model.rd_encoder.frame_cell.",
            "trainable_base_model.rd_encoder.residual_projection.",
        )
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if any(not name.startswith(allowed_prefixes) for name in names):
            raise AssertionError("V27d-r16 treatment unfroze an existing model parameter")
    if any(parameter.requires_grad for parameter in model.frozen_reference_base_model.parameters()) or any(parameter.requires_grad for parameter in model.stable_model.parameters()):
        raise AssertionError("V27d-r16 frozen endpoint received trainable parameters")
    return values


__all__ = ["R16ContinuousRDEncoder", "R16IntegratedL2Model", "build_r16_model", "tensor_state_hash", "trainable_parameter_groups"]