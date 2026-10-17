"""V27 L1 与 V10 observation-increment 的匹配迁移模型。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from radar_ln3.runtime import canonical_sha256
from v27.v27_baseline.model import V27LiquidV2TaskModel
from v27.v27_baseline.protocol import V27_L1_TASK_NAME, load_v27_config

from .protocol import R15_ARMS, R15_CLASS_NAMES, R15_CHECKPOINT_SCHEMA, R15Context


class R15ObservationIncrementAdapter(nn.Module):
    """V10 的 9 维无 dt observation-increment encoder。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def model_contract(self) -> dict[str, Any]:
        return {
            "class": "R15ObservationIncrementAdapter",
            "input_dim": self.input_projection.in_features,
            "hidden_dim": self.input_projection.out_features,
            "output_dim": self.output_projection.out_features,
            "activation": "SiLU",
            "normalization": "LayerNorm_affine_false",
            "input_bias": False,
            "output_bias": False,
            "output_zero_initialization": True,
            "validity_mask_input": False,
            "semantic_source": "V10_causal_per_observation_increment",
        }

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if (
            values.ndim != 3
            or values.shape[-1] != self.input_projection.in_features
            or padding_mask.shape != values.shape[:2]
            or padding_mask.dtype != torch.bool
            or not values.is_floating_point()
            or not torch.isfinite(values).all()
        ):
            raise ValueError("V27d-r15 observation-increment adapter input differs")
        if bool(values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r15 padded observation-increment values must be zero")
        encoded = self.output_projection(
            torch.nn.functional.silu(self.normalization(self.input_projection(values)))
        )
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("V27d-r15 observation-increment adapter is non-finite")
        return encoded


class R15L1Model(nn.Module):
    """一个匹配的 control/treatment V27 L1 训练臂。"""

    def __init__(self, context: R15Context, *, arm: str, seed: int) -> None:
        super().__init__()
        if arm not in R15_ARMS:
            raise ValueError(f"V27d-r15 arm is invalid: {arm}")
        v27_config = load_v27_config(
            context.project_root,
            context.config.source["v27_config_path"],
        )
        self.base = V27LiquidV2TaskModel(v27_config, task=V27_L1_TASK_NAME)
        checkpoint_path = context.config.source["v27_l1_checkpoint_path"]
        checkpoint = torch.load(
            context.project_root / checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping) or checkpoint.get("task") != V27_L1_TASK_NAME:
            raise ValueError("V27d-r15 frozen V27 L1 checkpoint identity differs")
        state = checkpoint.get("model_state")
        if not isinstance(state, Mapping):
            raise ValueError("V27d-r15 frozen V27 L1 checkpoint state is missing")
        # V27 epoch-14 的分类 state 可复用；历史 checkpoint 中的
        # trajectory_heads 属于已冻结的旧 knownness/quality 旁路，当前
        # V27 代码曾扩展过其事实宽度。r15 不训练、不前向该旁路，因此
        # 只允许这一明确的兼容差异，其余分类参数必须严格匹配。
        current_state = self.base.state_dict()
        filtered_state = {
            name: value
            for name, value in state.items()
            if not str(name).startswith("trajectory_heads.")
        }
        expected_filtered = {
            name: value
            for name, value in current_state.items()
            if not str(name).startswith("trajectory_heads.")
        }
        if set(filtered_state) != set(expected_filtered):
            raise ValueError("V27d-r15 frozen V27 L1 classification state keys differ")
        for name, value in filtered_state.items():
            if tuple(value.shape) != tuple(expected_filtered[name].shape) or value.dtype != expected_filtered[name].dtype:
                raise ValueError(f"V27d-r15 frozen V27 L1 classification tensor differs: {name}")
        self.base.load_state_dict(filtered_state, strict=False)
        if checkpoint.get("model_state_sha256") != context.config.source["expected_v27_l1_checkpoint_state_sha256"]:
            raise ValueError("V27d-r15 V27 L1 checkpoint state hash differs")
        self.arm = arm
        self.seed = int(seed)
        torch.manual_seed(int(seed))
        self.observation_adapter = R15ObservationIncrementAdapter(
            int(context.config.data["observation_increment_dim"]),
            int(context.config.model["motion_hidden_dim"]),
            int(context.config.model["motion_output_dim"]),
        )
        if int(context.config.model["motion_output_dim"]) != int(v27_config.model.liquid_input_dim):
            raise ValueError("V27d-r15 residual width must equal V27 Liquid input width")
        # 原始 V27 fusion 保持完整，V10 分支只在其输出后提供 residual。
        for parameter in self.base.fusion.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base.trajectory_heads.parameters():
            parameter.requires_grad_(False)
        self.base.trajectory_heads.eval()

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(R15_CLASS_NAMES)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": R15_CHECKPOINT_SCHEMA,
            "experiment_id": "V27d-r15",
            "arm": self.arm,
            "seed": self.seed,
            "class_names": list(self.class_names),
            "base_task": "known_gate",
            "base_checkpoint": "V27-known_gate-seed42-epoch14",
            "observation_increment_adapter": self.observation_adapter.model_contract(),
            "fusion_injection": "V27_fusion_then_zero_initialized_V10_residual_before_LiquidV2",
            "control_input": "all_zero_observation_increment",
            "treatment_input": "V10_numeric_values_after_gating_only",
            "unknown_is_classifier_class": False,
            "trajectory_heads_trainable": False,
            "padding_policy": "suffix_zero_no_state_update",
        }

    @staticmethod
    def _base_fusion_input(base: V27LiquidV2TaskModel, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tr_features = inputs["tr_features"]
        padding = inputs["tr_padding_mask"]
        tr_embedding = base.tr_encoder(torch.cat((tr_features, inputs["tr_missing_mask"]), dim=-1))
        rd_embedding = base.rd_encoder(inputs["rd_frames"], inputs["rd_frame_mask"])
        course_features = inputs["course_features"]
        course_valid = inputs["course_valid"]
        course_value = course_features[..., 0] * course_valid[..., 0].to(dtype=tr_features.dtype)
        course_is_valid = course_valid[..., 0].to(dtype=tr_features.dtype)
        result = torch.cat(
            (
                tr_embedding,
                rd_embedding,
                inputs["rd_quality"],
                course_value.unsqueeze(-1),
                course_is_valid.unsqueeze(-1),
            ),
            dim=-1,
        )
        return result.masked_fill(padding.unsqueeze(-1), 0.0)

    def _selected_observation(self, values: torch.Tensor) -> torch.Tensor:
        if self.arm == "control":
            return torch.zeros_like(values)
        return values

    def forward(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        observation_increment_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        padding = base_inputs["tr_padding_mask"]
        if observation_increment_values.shape != (*padding.shape, 9):
            raise ValueError("V27d-r15 observation-increment shape differs")
        if bool(observation_increment_values.abs().gt(5.0 + 1.0e-6).any()):
            raise ValueError("V27d-r15 observation-increment value exceeds V10 clip")
        selected = self._selected_observation(observation_increment_values)
        encoded = self.observation_adapter(selected, padding)
        fusion_input = self._base_fusion_input(self.base, base_inputs)
        base_events = self.base.fusion(fusion_input)
        events = (base_events + encoded).masked_fill(padding.unsqueeze(-1), 0.0)
        liquid = self.base.liquid(
            events,
            base_inputs["tr_dt_seconds"],
            padding,
            dt_valid=base_inputs["tr_dt_valid"],
            dt_reason=base_inputs["tr_dt_reason"],
            return_trace=False,
        )
        sequence_logits = self.base.classifier(self.base.classifier_norm(liquid.sequence)).masked_fill(
            padding.unsqueeze(-1), 0.0
        )
        lengths = (~padding).long().sum(dim=1)
        rows = torch.arange(sequence_logits.shape[0], device=sequence_logits.device)
        final_logits = sequence_logits[rows, lengths - 1]
        if not torch.isfinite(sequence_logits).all():
            raise FloatingPointError("V27d-r15 logits are non-finite")
        return {
            "sequence_logits": sequence_logits,
            "final_logits": final_logits,
            "liquid_sequence": liquid.sequence,
            "observation_embedding": encoded,
        }


def build_r15_model(context: R15Context, *, arm: str, seed: int, device: torch.device) -> R15L1Model:
    model = R15L1Model(context, arm=arm, seed=seed)
    return model.to(device)


def model_state_hash(model: nn.Module) -> str:
    parts: list[Any] = []
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        parts.extend((name, str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


__all__ = ["R15L1Model", "R15ObservationIncrementAdapter", "build_r15_model", "model_state_hash"]