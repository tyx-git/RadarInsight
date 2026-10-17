"""r20 L2 model with a structured radial-speed sidecar."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .protocol import R20_ARMS, R20_RADIAL_VALUE_DIM, R20Config


class R20RadialAdapter(nn.Module):
    """Bias-free post-fusion adapter for four radial values plus validity flags."""

    def __init__(self, config: R20Config, *, output_dim: int) -> None:
        super().__init__()
        input_dim = int(config.data["radial_sidecar_input_dim"])
        projection_dim = int(config.model["radial_projection_dim"])
        adapter_dim = int(config.model["radial_adapter_dim"])
        if input_dim != 2 * R20_RADIAL_VALUE_DIM or min(projection_dim, adapter_dim, int(output_dim)) <= 0:
            raise ValueError("r20 radial adapter dimensions are invalid")
        self.input_projection = nn.Linear(input_dim, projection_dim, bias=False)
        self.hidden_projection = nn.Linear(projection_dim, adapter_dim, bias=False)
        self.output_projection = nn.Linear(adapter_dim, int(output_dim), bias=False)
        if bool(config.model["output_zero_initialization"]):
            nn.init.zeros_(self.output_projection.weight)

    def model_contract(self) -> dict[str, Any]:
        return {
            "class": "R20RadialAdapter",
            "input_dim": self.input_projection.in_features,
            "projection_dim": self.input_projection.out_features,
            "adapter_dim": self.hidden_projection.out_features,
            "output_dim": self.output_projection.out_features,
            "activation": "SiLU",
            "bias_free": True,
            "output_zero_initialization": True,
            "input_layout": {
                "values": [
                    "abs_radial_speed_scaled",
                    "radial_speed_sign",
                    "radial_speed_rate_scaled",
                    "radial_range_direction_consistency",
                ],
                "validity": [
                    "abs_radial_speed_scaled_valid",
                    "radial_speed_sign_valid",
                    "radial_speed_rate_scaled_valid",
                    "radial_range_direction_consistency_valid",
                ],
            },
            "injection": "post_fusion_pre_liquid",
            "control_zero_input_is_exact_noop": True,
        }

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        clip_absolute: float,
    ) -> torch.Tensor:
        if (
            values.ndim != 3
            or values.shape[-1] != R20_RADIAL_VALUE_DIM
            or valid.shape != values.shape
            or valid.dtype != torch.bool
            or padding_mask.shape != values.shape[:2]
            or padding_mask.dtype != torch.bool
            or not values.is_floating_point()
            or not torch.isfinite(values).all()
        ):
            raise ValueError("r20 radial adapter input shape or dtype differs")
        if bool(values.masked_select(~valid).ne(0).any()):
            raise ValueError("r20 invalid radial sidecar values must be zero")
        if bool(values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("r20 padded radial sidecar values must be zero")
        if bool(valid.masked_select(padding_mask.unsqueeze(-1)).any()):
            raise ValueError("r20 padded radial sidecar validity must be false")
        if bool((values.abs() > float(clip_absolute) + 1.0e-6).any()):
            raise ValueError("r20 radial sidecar exceeds its registered clip")
        if bool(values[..., 0].lt(0.0).any()):
            raise ValueError("r20 absolute radial sidecar cannot be negative")
        selected = values * valid.to(dtype=values.dtype)
        selected = torch.cat((selected, valid.to(dtype=values.dtype)), dim=-1)
        output = self.output_projection(
            torch.nn.functional.silu(
                self.hidden_projection(
                    torch.nn.functional.silu(self.input_projection(selected))
                )
            )
        )
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(output).all():
            raise FloatingPointError("r20 radial adapter produced a non-finite value")
        return output


class R20L2Model(nn.Module):
    """V27 L2 classifier with an explicit causal radial decomposition input."""

    def __init__(self, config: R20Config, *, base_model: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in R20_ARMS:
            raise ValueError("r20 model arm is invalid")
        if not hasattr(base_model, "task") or str(base_model.task) != "dbb_specialist":
            raise ValueError("r20 base model is not the V27 L2 task model")
        if not all(hasattr(base_model, name) for name in ("fusion", "tr_encoder", "rd_encoder", "liquid", "classifier_norm", "classifier")):
            raise ValueError("r20 base model lacks the V27 L2 runtime components")
        self.config = config
        self.arm = arm
        self.base_model = base_model
        self.radial_adapter = R20RadialAdapter(
            config,
            output_dim=int(base_model.liquid.cell.input_dim),
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-r20-l2-model-contract-v1",
            "experiment_id": self.config["experiment_id"],
            "arm": self.arm,
            "task": "dbb_specialist",
            "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
            "base_model": self.base_model.model_contract(),
            "radial_adapter": self.radial_adapter.model_contract(),
            "raw_radial_scalar": "removed from tr_features and represented only by sidecar",
            "radial_validity_flags": "explicit model inputs to adapter",
            "unknown_classifier_class": False,
            "l3": "not_executed",
        }

    def _validate_l2_inputs(self, inputs: Mapping[str, torch.Tensor]) -> tuple[int, int]:
        required = (
            "tr_features",
            "tr_missing_mask",
            "tr_dt_seconds",
            "tr_dt_valid",
            "tr_dt_reason",
            "tr_padding_mask",
            "rd_frames",
            "rd_frame_mask",
            "rd_quality",
            "rd_physical_features",
            "rd_physical_valid",
            "micro_doppler_features",
            "micro_doppler_valid",
            "radial_sidecar_values",
            "radial_sidecar_valid",
        )
        missing = [name for name in required if name not in inputs]
        if missing:
            raise ValueError(f"r20 model inputs are missing: {missing}")
        batch, length = self.base_model._validate_common(
            tr_features=inputs["tr_features"],
            tr_missing_mask=inputs["tr_missing_mask"],
            tr_dt_seconds=inputs["tr_dt_seconds"],
            tr_dt_valid=inputs["tr_dt_valid"],
            tr_dt_reason=inputs["tr_dt_reason"],
            tr_padding_mask=inputs["tr_padding_mask"],
            rd_frames=inputs["rd_frames"],
            rd_frame_mask=inputs["rd_frame_mask"],
            rd_quality=inputs["rd_quality"],
        )
        tr = inputs["tr_features"]
        physical = inputs["rd_physical_features"]
        physical_valid = inputs["rd_physical_valid"]
        micro = inputs["micro_doppler_features"]
        micro_valid = inputs["micro_doppler_valid"]
        self.base_model._require_shape(physical, (batch, length, self.base_model.physical_dim), "rd_physical_features")
        self.base_model._require_shape(physical_valid, (batch, length), "rd_physical_valid")
        self.base_model._require_shape(micro, (batch, length, self.base_model.micro_dim), "micro_doppler_features")
        self.base_model._require_shape(micro_valid, (batch, length), "micro_doppler_valid")
        if physical_valid.dtype != torch.bool or micro_valid.dtype != torch.bool:
            raise ValueError("r20 physical/micro validity masks must be boolean")
        if not all(
            value.device == tr.device and value.dtype == tr.dtype and value.is_floating_point() and torch.isfinite(value).all()
            for value in (physical, micro)
        ):
            raise ValueError("r20 physical/micro inputs must be finite and match TR dtype/device")
        padding = inputs["tr_padding_mask"]
        if bool(physical.masked_select(~physical_valid.unsqueeze(-1)).ne(0).any()) or bool(micro.masked_select(~micro_valid.unsqueeze(-1)).ne(0).any()):
            raise ValueError("r20 invalid physical/micro values must be zero")
        if bool(physical_valid.masked_select(padding).any()) or bool(micro_valid.masked_select(padding).any()):
            raise ValueError("r20 padded physical/micro validity must be false")
        radial = inputs["radial_sidecar_values"]
        radial_valid = inputs["radial_sidecar_valid"]
        if radial.shape != (batch, length, R20_RADIAL_VALUE_DIM) or radial_valid.shape != radial.shape:
            raise ValueError("r20 radial sidecar shape differs")
        if radial.device != tr.device or radial_valid.device != tr.device or radial.dtype != tr.dtype:
            raise ValueError("r20 radial sidecar device/dtype differs from TR")
        if radial_valid.dtype != torch.bool:
            raise ValueError("r20 radial sidecar validity must be boolean")
        return batch, length

    def _native_events(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tr = inputs["tr_features"]
        physical = inputs["rd_physical_features"]
        physical_valid = inputs["rd_physical_valid"]
        micro = inputs["micro_doppler_features"]
        micro_valid = inputs["micro_doppler_valid"]
        return self.base_model.fusion(
            torch.cat(
                (
                    self.base_model.tr_encoder(torch.cat((tr, inputs["tr_missing_mask"]), dim=-1)),
                    self.base_model.rd_encoder(inputs["rd_frames"], inputs["rd_frame_mask"]),
                    inputs["rd_quality"],
                    physical * physical_valid.to(dtype=tr.dtype).unsqueeze(-1),
                    micro * micro_valid.to(dtype=tr.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
        )

    def forward(self, **inputs: torch.Tensor) -> dict[str, Any]:
        batch, length = self._validate_l2_inputs(inputs)
        padding = inputs["tr_padding_mask"]
        adapter_output = self.radial_adapter(
            inputs["radial_sidecar_values"],
            inputs["radial_sidecar_valid"],
            padding,
            clip_absolute=float(self.config.data["sidecar_clip_absolute"]),
        )
        events = (self._native_events(inputs) + adapter_output).masked_fill(
            padding.unsqueeze(-1), 0.0
        )
        liquid_output = self.base_model.liquid(
            events,
            inputs["tr_dt_seconds"],
            padding,
            dt_valid=inputs["tr_dt_valid"],
            dt_reason=inputs["tr_dt_reason"],
            return_trace=False,
        )
        sequence_logits = self.base_model.classifier(
            self.base_model.classifier_norm(liquid_output.sequence)
        ).masked_fill(padding.unsqueeze(-1), 0.0)
        lengths = (~padding).long().sum(dim=1)
        indices = torch.arange(batch, device=events.device)
        window_logits = sequence_logits[indices, lengths - 1]
        values = (events, liquid_output.sequence, sequence_logits, window_logits, adapter_output)
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("r20 model output is non-finite")
        return {
            "task": "dbb_specialist",
            "window_logits": window_logits,
            "sequence_logits": sequence_logits,
            "liquid_sequence": liquid_output.sequence,
            "liquid_final_state": liquid_output.final_state,
            "radial_adapter_output": adapter_output,
            "dt_clamped_mask": liquid_output.dt_clamped_mask,
            "dt_zero_mask": liquid_output.dt_zero_mask,
            "dt_invalid_mask": liquid_output.dt_invalid_mask,
            "dt_first_event_mask": liquid_output.dt_first_event_mask,
            "dt_zero_delta_mask": liquid_output.dt_zero_delta_mask,
        }


__all__ = ["R20L2Model", "R20RadialAdapter"]
