"""L1 已知目标/背景路由模型。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..backbone import L1_TASK_NAME, KnownGateClassifier
from ..config import ModelConfig, task_class_names


class ObservationIncrementAdapter(nn.Module):
    """将逐观测变化编码为 Liquid V2 residual。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def model_contract(self) -> dict[str, Any]:
        return {
            "class": "ObservationIncrementAdapter",
            "input_dim": self.input_projection.in_features,
            "hidden_dim": self.input_projection.out_features,
            "output_dim": self.output_projection.out_features,
            "activation": "SiLU",
            "normalization": "LayerNorm_affine_false",
            "input_bias": False,
            "output_bias": False,
            "output_zero_initialization": True,
            "validity_mask_input": False,
            "semantic_source": "causal_per_observation_increment",
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
            raise ValueError("observation increment adapter 输入 shape 或数值不一致")
        if bool(values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("padding 的 observation increment 值必须为零")
        encoded = self.output_projection(
            torch.nn.functional.silu(
                self.normalization(self.input_projection(values))
            )
        )
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("observation increment adapter 产生非有限值")
        return encoded


class L1Model(nn.Module):
    """L1 路由模型；control 和 observation_increment 共用容量。"""

    ARMS = ("control", "observation_increment")

    def __init__(
        self,
        config: ModelConfig,
        *,
        arm: str = "observation_increment",
        seed: int = 42,
        base_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"不支持的 L1 arm：{arm}")
        self.config = config
        self.arm = str(arm)
        self.seed = int(seed)
        self.base = KnownGateClassifier(config)
        if base_state is not None:
            base_state = dict(base_state)
            classifier_state = {
                name: value
                for name, value in base_state.items()
                if not str(name).startswith("trajectory_heads.")
            }
            expected_state = {
                name: value
                for name, value in self.base.state_dict().items()
                if not str(name).startswith("trajectory_heads.")
            }
            if set(classifier_state) != set(expected_state):
                raise ValueError("L1 基础分类 state key 不一致")
            for name, value in classifier_state.items():
                if tuple(value.shape) != tuple(expected_state[name].shape):
                    raise ValueError(f"L1 基础 tensor shape 不一致：{name}")
            incompatible = self.base.load_state_dict(classifier_state, strict=False)
            expected_missing = {
                name
                for name in self.base.state_dict()
                if name.startswith("trajectory_heads.")
            }
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError("L1 基础 state key 不一致")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.observation_adapter = ObservationIncrementAdapter(
                config.data.motion_dim,
                config.model.observation_adapter_hidden_dim,
                config.model.liquid_input_dim,
            )
        # 分类主干和 fusion 参与正式训练；trajectory heads 仍是独立的只读旁路。
        for parameter in self.base.trajectory_heads.parameters():
            parameter.requires_grad_(False)
        self.base.trajectory_heads.eval()

    @property
    def class_names(self) -> tuple[str, ...]:
        return task_class_names(self.config, L1_TASK_NAME)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-l1-model-contract-v1",
            "task": L1_TASK_NAME,
            "arm": self.arm,
            "seed": self.seed,
            "class_names": list(self.class_names),
            "base_model": self.base.model_contract(),
            "observation_increment_adapter": self.observation_adapter.model_contract(),
            "fusion_injection": "fusion_then_residual_before_liquid",
            "control_input": "all_zero_observation_increment",
            "treatment_input": "causal_per_observation_numeric_values_after_gating",
            "unknown_is_classifier_class": False,
            "trajectory_heads_trainable": False,
            "padding_policy": "suffix_zero_no_state_update",
        }

    @staticmethod
    def _fusion_input(
        base: KnownGateClassifier,
        inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        tr_features = inputs["tr_features"]
        padding = inputs["tr_padding_mask"]
        tr_embedding = base.tr_encoder(
            torch.cat((tr_features, inputs["tr_missing_mask"]), dim=-1)
        )
        rd_embedding = base.rd_encoder(
            inputs["rd_frames"], inputs["rd_frame_mask"]
        )
        course_features = inputs["course_features"]
        course_valid = inputs["course_valid"]
        course_value = course_features[..., 0] * course_valid[..., 0].to(
            dtype=tr_features.dtype
        )
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
        return torch.zeros_like(values) if self.arm == "control" else values

    def forward(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        observation_increment_values: torch.Tensor,
        return_representation: bool = False,
    ) -> dict[str, torch.Tensor]:
        padding = base_inputs.get("tr_padding_mask")
        if not isinstance(padding, torch.Tensor):
            raise TypeError("L1 base_inputs 缺少 padding tensor")
        if observation_increment_values.shape != (*padding.shape, 9):
            raise ValueError("observation increment shape 不一致")
        if bool(observation_increment_values.abs().gt(5.0 + 1.0e-6).any()):
            raise ValueError("observation increment 超出登记的 clip 范围")
        selected = self._selected_observation(observation_increment_values)
        encoded = self.observation_adapter(selected, padding)
        fusion_input = self._fusion_input(self.base, base_inputs)
        base_events = self.base.fusion(fusion_input)
        events = (base_events + encoded).masked_fill(padding.unsqueeze(-1), 0.0)
        liquid = self.base.liquid(
            events,
            base_inputs["tr_dt_seconds"],
            padding,
            dt_valid=base_inputs["tr_dt_valid"],
            dt_reason=base_inputs["tr_dt_reason"],
            return_trace=bool(return_representation),
        )
        sequence_logits = self.base.classifier(
            self.base.classifier_norm(liquid.sequence)
        ).masked_fill(padding.unsqueeze(-1), 0.0)
        lengths = (~padding).long().sum(dim=1)
        rows = torch.arange(sequence_logits.shape[0], device=sequence_logits.device)
        final_logits = sequence_logits[rows, lengths - 1]
        if not torch.isfinite(sequence_logits).all():
            raise FloatingPointError("L1 logits 包含非有限值")
        result: dict[str, Any] = {
            "sequence_logits": sequence_logits,
            "final_logits": final_logits,
            "liquid_sequence": liquid.sequence,
            "observation_embedding": encoded,
        }
        if return_representation:
            if liquid.trace is None:
                raise RuntimeError("L1 representation requires raw liquid trace")
            task_facts = self.base._task_facts_sequence(
                padding=padding,
                rd_frame_mask=base_inputs["rd_frame_mask"],
                rd_quality=base_inputs["rd_quality"],
                dt_valid=base_inputs["tr_dt_valid"],
                dt_reason=base_inputs["tr_dt_reason"],
                physical_valid=None,
                micro_valid=None,
                course_valid=base_inputs["course_valid"],
            )
            event_facts = self.base._trajectory_event_facts(
                padding=padding,
                rd_frame_mask=base_inputs["rd_frame_mask"],
                rd_quality=base_inputs["rd_quality"],
                dt_seconds=base_inputs["tr_dt_seconds"],
                dt_valid=base_inputs["tr_dt_valid"],
                dt_reason=base_inputs["tr_dt_reason"],
                physical_valid=None,
                micro_valid=None,
                course_features=base_inputs["course_features"],
                course_valid=base_inputs["course_valid"],
                course_support=base_inputs["course_support"],
                zero_ordinal=liquid.trace["zero_ordinal"],
            )
            features = self.base.trajectory_heads.build_representation_features(
                raw_sequence=liquid.sequence,
                trace=liquid.trace,
                padding_mask=padding,
                task_facts_sequence=task_facts,
                event_facts=event_facts,
            )
            result.update(
                {
                    "summary_features": features["summary_features"],
                    "event_trajectory": features["event_trajectory"],
                    "task_facts_sequence": features["task_facts_sequence"],
                    "event_facts_sequence": features["event_facts_sequence"],
                }
            )
        return result


def build_l1_model(
    config: ModelConfig,
    *,
    arm: str = "observation_increment",
    seed: int = 42,
    base_state: Mapping[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> L1Model:
    model = L1Model(config, arm=arm, seed=seed, base_state=base_state)
    return model if device is None else model.to(device)


__all__ = ["L1Model", "ObservationIncrementAdapter", "build_l1_model"]
