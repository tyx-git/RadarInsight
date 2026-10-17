"""Integrated V27 L2, V27a stable8, and causal-motion model for r23."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from v27.v27_baseline.base_binding import classifier_state_hash, mutable_trajectory_head_keys
from v27.v27_baseline.model import V27LiquidV2TaskModel
from v27.v27_baseline.protocol import V27_L2_TASK_NAME
from v27.v27a.model import V27AEvidenceResidual

from .protocol import R23_ARM_SPECS, R23_ARMS, R23_L2_CLASSES, R23Config


class R23MotionAdapter(nn.Module):
    """Bias-free causal-motion residual injected between fusion and Liquid V2.

    The last projection starts at exact zero.  Consequently, legacy arms have
    exactly the same pre-Liquid event stream at initialization, while training
    can still make a nonzero treatment alter every subsequent raw-memory step.
    The all-zero control remains a capacity-matched no-op for any adapter
    weights because neither layer has a bias.
    """

    def __init__(self, config: R23Config, *, output_dim: int) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            config.data.motion_dim,
            config.model.motion_projection_dim,
            bias=config.model.motion_projection_bias,
        )
        self.hidden_projection = nn.Linear(
            config.model.motion_projection_dim,
            config.model.motion_adapter_dim,
            bias=False,
        )
        self.output_projection = nn.Linear(
            config.model.motion_adapter_dim,
            int(output_dim),
            bias=config.model.motion_output_bias,
        )
        if int(output_dim) <= 0:
            raise ValueError("V27c-r23 motion adapter output width must be positive")
        if config.model.motion_output_zero_initialization:
            nn.init.zeros_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)

    def model_contract(self) -> dict[str, Any]:
        return {
            "class": "R23MotionAdapter",
            "input_dim": self.config.data.motion_dim,
            "projection_dim": self.config.model.motion_projection_dim,
            "hidden_dim": self.config.model.motion_adapter_dim,
            "output_dim": self.output_projection.out_features,
            "activation": "SiLU",
            "input_bias": self.input_projection.bias is not None,
            "output_bias": self.output_projection.bias is not None,
            "output_zero_initialization": self.config.model.motion_output_zero_initialization,
            "validity_mask_input": False,
            "zero_control_is_weight_independent_noop": True,
            "injection": "post_v27_fusion_pre_liquid_v2",
            "hidden_projection": True,
        }

    def forward(self, motion_values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if (
            motion_values.ndim != 3
            or motion_values.shape[-1] != self.config.data.motion_dim
            or padding_mask.shape != motion_values.shape[:2]
            or padding_mask.dtype != torch.bool
            or not motion_values.is_floating_point()
            or not torch.isfinite(motion_values).all()
        ):
            raise ValueError("V27c-r23 motion adapter input shape/value differs")
        if bool(
            motion_values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()
        ):
            raise ValueError("V27c-r23 padded motion values must be zero")
        if bool(
            (motion_values.abs() > self.config.data.motion_clip_absolute + 1.0e-6).any()
        ):
            raise ValueError("V27c-r23 motion values exceed their registered clip")
        projected = torch.nn.functional.silu(self.input_projection(motion_values))
        projected = torch.nn.functional.silu(self.hidden_projection(projected))
        output = self.output_projection(projected)
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(output).all():
            raise FloatingPointError("V27c-r23 motion adapter produced a non-finite value")
        return output


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _freeze_trajectory_heads(model: V27LiquidV2TaskModel) -> None:
    _freeze(model.trajectory_heads)
    if any(parameter.requires_grad for parameter in model.trajectory_heads.parameters()):
        raise AssertionError("V27c-r23 trajectory heads were not frozen")


class R23IntegratedL2Model(nn.Module):
    """One matched r23 arm.

    The output is anchored to the historical cache rather than depending on a
    bitwise re-execution of legacy CUDA kernels:

    ``cached V27 base + (current L2 - frozen-reference L2) + current stable8``.

    At initialization the middle term is exactly zero because both L2 modules
    own the same epoch-11 state, and the stable branch has the V27a Main state.
    This reproduces V27a Main while allowing base L2, stable8 and (treatment
    only) motion to be optimized jointly thereafter.
    """

    def __init__(
        self,
        config: R23Config,
        *,
        arm: str,
        trainable_base_model: V27LiquidV2TaskModel,
        frozen_reference_base_model: V27LiquidV2TaskModel,
        stable_model: V27AEvidenceResidual,
        motion_adapter: R23MotionAdapter,
    ) -> None:
        super().__init__()
        if arm not in R23_ARMS:
            raise ValueError("V27c-r23 model arm is invalid")
        if (
            trainable_base_model.task != V27_L2_TASK_NAME
            or frozen_reference_base_model.task != V27_L2_TASK_NAME
            or stable_model.arm != "main"
        ):
            raise ValueError("V27c-r23 model components do not match L2/Main contracts")
        self.config = config
        self.arm = arm
        self.trainable_base_model = trainable_base_model
        self.frozen_reference_base_model = frozen_reference_base_model
        self.stable_model = stable_model
        self.motion_adapter = motion_adapter
        _freeze(self.frozen_reference_base_model)
        _freeze_trajectory_heads(self.trainable_base_model)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27c-r23-integrated-l2-model-contract-v1",
            "experiment_id": self.config.experiment_id,
            "arm": self.arm,
            "class_names": list(R23_L2_CLASSES),
            "base_model": self.trainable_base_model.model_contract(),
            "frozen_reference_base": {
                "same_initial_state_as_trainable_base": True,
                "requires_grad": False,
                "trajectory_heads": "frozen_unused",
            },
            "stable8_model": self.stable_model.model_contract(),
            "motion_adapter": self.motion_adapter.model_contract(),
            "motion_input": R23_ARM_SPECS[self.arm]["motion_mode"],
            "input_mode": R23_ARM_SPECS[self.arm]["input_mode"],
            "output_equation": "cached_v27a_logits + (current_arm_input_l2_logits - frozen_legacy_reference_l2_logits) + trainable_v27a_main_stable8_residual",
            "unknown_classifier_class": False,
            "trajectory_heads_trainable": False,
        }

    def v27a_main_endpoint(
        self,
        *,
        cached_v27_base_logits: torch.Tensor,
        stable_summary_tokens: torch.Tensor,
        stable_summary_valid: torch.Tensor,
        stable_frame_values: torch.Tensor,
        stable_frame_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the archived V27a Main endpoint from its cache fields.

        ``event_baseline_logits`` is only the input to the V27a residual.  The
        Main endpoint is the baseline plus the residual produced by the
        frozen Main model, so this helper keeps that distinction explicit for
        smoke checks and evaluation.
        """

        output = self.stable_model(
            baseline_logits=cached_v27_base_logits,
            summary_tokens=stable_summary_tokens,
            summary_valid=stable_summary_valid,
            frame_values=stable_frame_values,
            frame_mask=stable_frame_mask,
            padding_mask=padding_mask,
        )
        return output["sequence_logits"]

    def _validate_sidecars(
        self,
        *,
        cached_v27_base_logits: torch.Tensor,
        stable_summary_tokens: torch.Tensor,
        stable_summary_valid: torch.Tensor,
        stable_frame_values: torch.Tensor,
        stable_frame_mask: torch.Tensor,
        motion_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if cached_v27_base_logits.ndim != 3:
            raise ValueError("V27c-r23 cached V27 logits must have [B,T,C]")
        batch, length, classes = cached_v27_base_logits.shape
        if (
            batch <= 0
            or length <= 0
            or classes != len(R23_L2_CLASSES)
            or stable_summary_tokens.shape
            != (batch, length, self.config.data.stable8_dim)
            or stable_summary_valid.shape != (batch, length)
            or stable_summary_valid.dtype != torch.bool
            or stable_frame_values.ndim != 4
            or stable_frame_values.shape[:2] != (batch, length)
            or stable_frame_values.shape[-1] != self.config.data.stable8_dim
            or stable_frame_mask.shape != stable_frame_values.shape[:3]
            or stable_frame_mask.dtype != torch.bool
            or motion_values.shape
            != (batch, length, self.config.data.motion_dim)
            or padding_mask.shape != (batch, length)
            or padding_mask.dtype != torch.bool
        ):
            raise ValueError("V27c-r23 sidecar shapes differ")
        floating = (
            cached_v27_base_logits,
            stable_summary_tokens,
            stable_frame_values,
            motion_values,
        )
        if not all(value.is_floating_point() and torch.isfinite(value).all() for value in floating):
            raise ValueError("V27c-r23 sidecar values must be finite floats")
        if any(value.device != cached_v27_base_logits.device for value in floating):
            raise ValueError("V27c-r23 sidecar values must share a device")
        if any(value.dtype != cached_v27_base_logits.dtype for value in floating):
            raise ValueError("V27c-r23 sidecar values must share a dtype")
        if any(
            value.device != cached_v27_base_logits.device
            for value in (stable_summary_valid, stable_frame_mask, padding_mask)
        ):
            raise ValueError("V27c-r23 sidecar masks must share a device")
        if bool(
            cached_v27_base_logits.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()
        ):
            raise ValueError("V27c-r23 padded cached V27 logits must be zero")
        if bool(
            motion_values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()
        ):
            raise ValueError("V27c-r23 padded motion values must be zero")
        if not torch.equal(stable_summary_valid, stable_frame_mask.any(dim=-1) & ~padding_mask):
            raise ValueError("V27c-r23 stable8 summary/reliable-frame contract differs")
        return batch, length

    @staticmethod
    def _native_events(
        model: V27LiquidV2TaskModel,
        *,
        base_inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reproduce the native V27 L2 fusion immediately before recurrence."""

        tr_features = base_inputs["tr_features"]
        tr_missing = base_inputs["tr_missing_mask"]
        rd_frames = base_inputs["rd_frames"]
        rd_frame_mask = base_inputs["rd_frame_mask"]
        rd_quality = base_inputs["rd_quality"]
        physical = base_inputs["rd_physical_features"]
        physical_valid = base_inputs["rd_physical_valid"]
        micro = base_inputs["micro_doppler_features"]
        micro_valid = base_inputs["micro_doppler_valid"]
        return model.fusion(
            torch.cat(
                (
                    model.tr_encoder(torch.cat((tr_features, tr_missing), dim=-1)),
                    model.rd_encoder(rd_frames, rd_frame_mask),
                    rd_quality,
                    physical * physical_valid.to(dtype=tr_features.dtype).unsqueeze(-1),
                    micro * micro_valid.to(dtype=tr_features.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
        )

    @staticmethod
    def _liquid_logits(
        model: V27LiquidV2TaskModel,
        *,
        events: torch.Tensor,
        base_inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        padding = base_inputs["tr_padding_mask"]
        output = model.liquid(
            events,
            base_inputs["tr_dt_seconds"],
            padding,
            dt_valid=base_inputs["tr_dt_valid"],
            dt_reason=base_inputs["tr_dt_reason"],
            return_trace=False,
        )
        return model.classifier(model.classifier_norm(output.sequence)).masked_fill(
            padding.unsqueeze(-1), 0.0
        )

    def _events_with_motion(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        motion_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.trainable_base_model
        # Reuse the native V27 L2 feature encoders and fusion verbatim.  The
        # only new learned transform occurs after this native fusion result.
        native_events = self._native_events(model, base_inputs=base_inputs)
        selected_motion = (
            motion_values
            if R23_ARM_SPECS[self.arm]["motion_mode"] == "causal"
            else torch.zeros_like(motion_values)
        )
        adapter = self.motion_adapter(selected_motion, padding_mask)
        events = (native_events + adapter).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(events).all():
            raise FloatingPointError("V27c-r23 fused events are non-finite")
        return events, adapter

    def _current_l2_logits(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        motion_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding = base_inputs["tr_padding_mask"]
        events, adapter = self._events_with_motion(
            base_inputs=base_inputs,
            motion_values=motion_values,
            padding_mask=padding,
        )
        logits = self._liquid_logits(
            self.trainable_base_model,
            events=events,
            base_inputs=base_inputs,
        )
        return logits, adapter

    def _reference_l2_logits(
        self, *, base_inputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        # Use precisely the same pre-Liquid decomposition as the trainable
        # path, with the frozen module and no adapter.  This makes the delta
        # exact at initialization; the native V27 full forward is separately
        # called in ``forward`` as an input-contract sentinel.
        with torch.no_grad():
            events = self._native_events(
                self.frozen_reference_base_model,
                base_inputs=base_inputs,
            )
            return self._liquid_logits(
                self.frozen_reference_base_model,
                events=events,
                base_inputs=base_inputs,
            )

    def forward(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        reference_base_inputs: Mapping[str, torch.Tensor],
        cached_v27_base_logits: torch.Tensor,
        stable_summary_tokens: torch.Tensor,
        stable_summary_valid: torch.Tensor,
        stable_frame_values: torch.Tensor,
        stable_frame_mask: torch.Tensor,
        motion_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        padding = base_inputs.get("tr_padding_mask")
        if not isinstance(padding, torch.Tensor):
            raise TypeError("V27c-r23 base inputs lack a padding tensor")
        batch, length = self._validate_sidecars(
            cached_v27_base_logits=cached_v27_base_logits,
            stable_summary_tokens=stable_summary_tokens,
            stable_summary_valid=stable_summary_valid,
            stable_frame_values=stable_frame_values,
            stable_frame_mask=stable_frame_mask,
            motion_values=motion_values,
            padding_mask=padding,
        )
        if batch != padding.shape[0] or length != padding.shape[1]:
            raise ValueError("V27c-r23 native L2/sidecar time dimensions differ")
        reference_l2_logits = self._reference_l2_logits(
            base_inputs=reference_base_inputs
        )
        current_l2_logits, motion_adapter_output = self._current_l2_logits(
            base_inputs=base_inputs, motion_values=motion_values
        )
        stable_output = self.stable_model(
            baseline_logits=cached_v27_base_logits,
            summary_tokens=stable_summary_tokens,
            summary_valid=stable_summary_valid,
            frame_values=stable_frame_values,
            frame_mask=stable_frame_mask,
            padding_mask=padding,
        )
        stable_residual = stable_output["residual_logits"]
        l2_delta = current_l2_logits - reference_l2_logits
        sequence_logits = (
            cached_v27_base_logits + l2_delta + stable_residual
        ).masked_fill(padding.unsqueeze(-1), 0.0)
        if not all(
            torch.isfinite(value).all()
            for value in (
                current_l2_logits,
                reference_l2_logits,
                stable_residual,
                l2_delta,
                sequence_logits,
            )
        ):
            raise FloatingPointError("V27c-r23 integrated L2 output is non-finite")
        return {
            "sequence_logits": sequence_logits,
            "cached_v27_base_logits": cached_v27_base_logits,
            "current_l2_logits": current_l2_logits,
            "frozen_reference_l2_logits": reference_l2_logits,
            "l2_delta_logits": l2_delta,
            "stable_residual_logits": stable_residual,
            "stable_sequence_logits": stable_output["sequence_logits"],
            "motion_adapter_output": motion_adapter_output,
        }


def build_r23_model(
    config: R23Config,
    *,
    arm: str,
    v27_config: Any,
    v27_l2_state: Mapping[str, torch.Tensor],
    v27a_config: Any,
    v27a_main_state: Mapping[str, torch.Tensor],
    seed: int,
) -> R23IntegratedL2Model:
    """Build a deterministic arm from the immutable V27/V27a states."""

    if arm not in R23_ARMS or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27c-r23 model arm or seed is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        trainable_base = V27LiquidV2TaskModel(v27_config, task=V27_L2_TASK_NAME)
        # Current V27 trajectory/quality heads intentionally evolved after the
        # historical epoch-11 classifier was frozen.  They are outside r23's
        # trainable graph, so load only the verified classifier/recurrent state
        # and require *exactly* the declared head keys to remain missing.
        checkpoint_heads = mutable_trajectory_head_keys(v27_l2_state)
        current_heads = mutable_trajectory_head_keys(trainable_base.state_dict())
        checkpoint_base = {
            name: value
            for name, value in v27_l2_state.items()
            if name not in checkpoint_heads
        }
        incompatible = trainable_base.load_state_dict(checkpoint_base, strict=False)
        if (
            set(incompatible.missing_keys) != set(current_heads)
            or incompatible.unexpected_keys
        ):
            raise ValueError(
                "V27c-r23 L2 classification-state load boundary differs: "
                f"missing={sorted(incompatible.missing_keys)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        if classifier_state_hash(trainable_base.state_dict()) != classifier_state_hash(
            v27_l2_state
        ):
            raise ValueError("V27c-r23 L2 classifier state changed during partial load")
        reference_base = copy.deepcopy(trainable_base)
        stable = V27AEvidenceResidual(v27a_config, arm="main")
        stable.load_state_dict(dict(v27a_main_state), strict=True)
        adapter = R23MotionAdapter(
            config,
            output_dim=trainable_base.liquid.cell.input_dim,
        )
        model = R23IntegratedL2Model(
            config,
            arm=arm,
            trainable_base_model=trainable_base,
            frozen_reference_base_model=reference_base,
            stable_model=stable,
            motion_adapter=adapter,
        )
    return model


def trainable_parameter_groups(
    model: R23IntegratedL2Model,
) -> dict[str, list[nn.Parameter]]:
    """Return explicit non-overlapping optimizer groups for r23."""

    base = [
        parameter
        for name, parameter in model.trainable_base_model.named_parameters()
        if parameter.requires_grad and not name.startswith("trajectory_heads.")
    ]
    stable = [parameter for parameter in model.stable_model.parameters() if parameter.requires_grad]
    motion = [parameter for parameter in model.motion_adapter.parameters() if parameter.requires_grad]
    if not base or not stable or not motion:
        raise ValueError("V27c-r23 optimizer group is unexpectedly empty")
    all_ids = [id(parameter) for values in (base, stable, motion) for parameter in values]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("V27c-r23 optimizer groups overlap")
    if any(parameter.requires_grad for parameter in model.frozen_reference_base_model.parameters()):
        raise AssertionError("V27c-r23 frozen reference received trainable parameters")
    if any(parameter.requires_grad for parameter in model.trainable_base_model.trajectory_heads.parameters()):
        raise AssertionError("V27c-r23 trajectory heads received trainable parameters")
    return {"base": base, "stable": stable, "motion": motion}


__all__ = ["R23IntegratedL2Model", "R23MotionAdapter", "build_r23_model", "trainable_parameter_groups"]