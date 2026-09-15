"""V27d-r14 anchored L2 model.

At initialization the output is exactly the frozen V27a-r3 Main endpoint:

    cached_v27_logits
    + (current_liquid_v2_l2_logits - frozen_epoch11_l2_logits)
    + v27a_main_stable8_residual

Both arms share all trainable tensors.  The only arm difference is the time
policy in the current Liquid V2 model.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from radar_ln3.runtime import sha256_file
from v27.v27_baseline.base_binding import classifier_state_hash, mutable_trajectory_head_keys
from v27.v27_baseline.model import V27LiquidV2TaskModel
from v27.v27_baseline.protocol import V27_L2_TASK_NAME
from v27.v27a.model import V27AEvidenceResidual
from v27.v27a.protocol import load_v27a_config

from .data import r14_sidecar_kwargs
from .protocol import R14Context, R14_MODES, mode_config, project_path


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _partial_load_l2(model: V27LiquidV2TaskModel, state: Mapping[str, torch.Tensor]) -> None:
    checkpoint_heads = mutable_trajectory_head_keys(state)
    current_heads = mutable_trajectory_head_keys(model.state_dict())
    base_state = {name: value for name, value in state.items() if name not in checkpoint_heads}
    incompatible = model.load_state_dict(base_state, strict=False)
    if set(incompatible.missing_keys) != set(current_heads) or incompatible.unexpected_keys:
        raise ValueError(
            "V27d-r14 L2 initial-state load boundary differs: "
            f"missing={sorted(incompatible.missing_keys)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if classifier_state_hash(model.state_dict()) != classifier_state_hash(state):
        raise ValueError("V27d-r14 initial L2 classifier state changed during partial load")


def _load_initial_l2_state(context: R14Context) -> dict[str, torch.Tensor]:
    path = project_path(context.project_root, context.config.source["paths"]["v27_l2_epoch11_checkpoint"], must_exist=True)
    if sha256_file(path) != context.config.source["expected_hashes"]["v27_l2_epoch11_checkpoint"]:
        raise ValueError("V27d-r14 frozen epoch-11 L2 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("task") != V27_L2_TASK_NAME or int(payload.get("epoch", -1)) != 11:
        raise ValueError("V27d-r14 frozen epoch-11 L2 checkpoint identity differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27d-r14 frozen epoch-11 L2 checkpoint lacks model state")
    return dict(state)


def _load_v27a_state(context: R14Context) -> tuple[Any, dict[str, torch.Tensor]]:
    path = project_path(context.project_root, context.config.source["paths"]["v27a_main_checkpoint"], must_exist=True)
    if sha256_file(path) != context.config.source["expected_hashes"]["v27a_main_checkpoint"]:
        raise ValueError("V27d-r14 V27a Main checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("experiment_id") != "V27a" or payload.get("arm") != "main" or int(payload.get("epoch", -1)) != 20:
        raise ValueError("V27d-r14 V27a Main checkpoint identity differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27d-r14 V27a Main checkpoint lacks model state")
    config = load_v27a_config(context.project_root, context.config.source["paths"]["v27a_config"])
    model = V27AEvidenceResidual(config, arm="main")
    model.load_state_dict(dict(state), strict=True)
    return config, dict(state)


class R14IntegratedL2Model(nn.Module):
    def __init__(self, context: R14Context, *, arm: str, seed: int) -> None:
        super().__init__()
        if arm not in R14_MODES:
            raise ValueError("V27d-r14 arm is invalid")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            train_config = mode_config(context, mode=R14_MODES[arm], seed=seed)
            reference_config = mode_config(context, mode="bounded_observation", seed=seed)
            trainable_base = V27LiquidV2TaskModel(train_config, task=V27_L2_TASK_NAME)
            frozen_reference = V27LiquidV2TaskModel(reference_config, task=V27_L2_TASK_NAME)
            l2_state = _load_initial_l2_state(context)
            _partial_load_l2(trainable_base, l2_state)
            _partial_load_l2(frozen_reference, l2_state)
            v27a_config, v27a_state = _load_v27a_state(context)
            stable = V27AEvidenceResidual(v27a_config, arm="main")
            stable.load_state_dict(v27a_state, strict=True)
        self.context = context
        self.arm = arm
        self.seed = int(seed)
        self.trainable_base_model = trainable_base
        self.frozen_reference_base_model = frozen_reference
        self.stable_model = stable
        _freeze(self.frozen_reference_base_model)
        for parameter in self.trainable_base_model.trajectory_heads.parameters():
            parameter.requires_grad_(False)

    @property
    def task(self) -> str:
        return V27_L2_TASK_NAME

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-r14-anchored-l2-model-v1",
            "experiment_id": "V27d-r14",
            "arm": self.arm,
            "seed": self.seed,
            "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
            "liquid_mode": R14_MODES[self.arm],
            "output_equation": "cached_v27_epoch11_logits+(current_l2-frozen_epoch11_l2)+v27a_main_stable8_residual",
            "frozen_reference_base": "V27_epoch11_bounded_observation",
            "stable8_branch": "V27a-r3-main-epoch-020",
            "stable8_trainable": True,
            "trajectory_heads_trainable": False,
            "unknown_is_classifier_class": False,
            "loss": "parent_final_fixed_parent_count_ce",
        }

    def forward(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        cached_v27_base_logits: torch.Tensor,
        stable_summary_tokens: torch.Tensor,
        stable_summary_valid: torch.Tensor,
        stable_frame_values: torch.Tensor,
        stable_frame_mask: torch.Tensor,
        reference_sequence_logits: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> dict[str, Any]:
        current = self.trainable_base_model(**dict(base_inputs), return_trace=return_trace)
        if reference_sequence_logits is None:
            with torch.no_grad():
                self.frozen_reference_base_model.eval()
                reference_logits = self.frozen_reference_base_model(**dict(base_inputs), return_trace=False)["sequence_logits"]
        else:
            reference_logits = reference_sequence_logits
            expected_shape = current["sequence_logits"].shape
            if reference_logits.shape != expected_shape or reference_logits.device != current["sequence_logits"].device or reference_logits.dtype != current["sequence_logits"].dtype:
                raise ValueError("V27d-r14 reference logits shape/device/dtype differs")
            if not torch.isfinite(reference_logits).all():
                raise ValueError("V27d-r14 reference logits are non-finite")
        stable = self.stable_model(
            baseline_logits=cached_v27_base_logits,
            summary_tokens=stable_summary_tokens,
            summary_valid=stable_summary_valid,
            frame_values=stable_frame_values,
            frame_mask=stable_frame_mask,
            padding_mask=base_inputs["tr_padding_mask"],
        )
        padding = base_inputs["tr_padding_mask"]
        delta = current["sequence_logits"] - reference_logits
        combined = (cached_v27_base_logits + delta + stable["residual_logits"]).masked_fill(padding.unsqueeze(-1), 0.0)
        if not torch.isfinite(combined).all():
            raise FloatingPointError("V27d-r14 integrated L2 output is non-finite")
        return {
            "sequence_logits": combined,
            "native_sequence_logits": current["sequence_logits"],
            "frozen_reference_sequence_logits": reference_logits,
            "l2_delta_logits": delta,
            "stable_residual_logits": stable["residual_logits"],
            "stable_sequence_logits": stable["sequence_logits"],
            "base_output": current,
        }


def build_r14_model(context: R14Context, *, arm: str, seed: int, device: torch.device) -> R14IntegratedL2Model:
    return R14IntegratedL2Model(context, arm=arm, seed=seed).to(device)


def trainable_parameter_groups(model: R14IntegratedL2Model) -> list[nn.Parameter]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("V27d-r14 optimizer has no trainable parameters")
    if any(parameter.requires_grad for parameter in model.frozen_reference_base_model.parameters()):
        raise AssertionError("V27d-r14 frozen reference received trainable parameters")
    if any(parameter.requires_grad for parameter in model.trainable_base_model.trajectory_heads.parameters()):
        raise AssertionError("V27d-r14 trajectory heads received trainable parameters")
    return trainable


__all__ = ["R14IntegratedL2Model", "build_r14_model", "trainable_parameter_groups"]
