"""L2 子类分类模型及 stable8、motion residual。"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..backbone import L2_TASK_NAME, LiquidTaskModel
from ..config import ModelConfig, task_class_names
from .evidence import EvidenceResidual

ARMS = ("standard_control", "standard_motion", "robust_control", "robust_motion")
ARM_SPECS = {
    "standard_control": {"input_mode": "standard", "motion_mode": "zero"},
    "standard_motion": {"input_mode": "standard", "motion_mode": "causal"},
    "robust_control": {"input_mode": "robust", "motion_mode": "zero"},
    "robust_motion": {"input_mode": "robust", "motion_mode": "causal"},
}


class MotionAdapter(nn.Module):
    """注入 Fusion 与 Liquid V2 之间的无 bias 因果 motion residual。

    最后投影以零初始化，因此 control 在任意 adapter 权重下都保持零输入；
    motion arm 则可在训练后改变后续 raw-memory 状态。
    """

    def __init__(self, config: ModelConfig, *, output_dim: int) -> None:
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
            raise ValueError("motion adapter 输出宽度必须为正")
        if config.model.motion_output_zero_initialization:
            nn.init.zeros_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)

    def model_contract(self) -> dict[str, Any]:
        return {
            "class": "MotionAdapter",
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
            "injection": "post_fusion_pre_liquid_v2",
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
            raise ValueError("motion adapter 输入 shape 或数值无效")
        if bool(
            motion_values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()
        ):
            raise ValueError("padding 的 motion 值必须为零")
        if bool(
            (motion_values.abs() > self.config.data.motion_clip_absolute + 1.0e-6).any()
        ):
            raise ValueError("motion 值超过注册 clip 范围")
        projected = torch.nn.functional.silu(self.input_projection(motion_values))
        projected = torch.nn.functional.silu(self.hidden_projection(projected))
        output = self.output_projection(projected)
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(output).all():
            raise FloatingPointError("motion adapter 产生非有限值")
        return output


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _freeze_trajectory_heads(model: LiquidTaskModel) -> None:
    _freeze(model.trajectory_heads)
    if any(parameter.requires_grad for parameter in model.trajectory_heads.parameters()):
        raise AssertionError("trajectory heads 未冻结")


class L2Model(nn.Module):
    """一个与配置匹配的 L2 arm。

    输出以参考主干的当前 logits 为锚点，不依赖旧 CUDA kernel 的逐位重放：

    ``参考 logits + (当前 L2 - 冻结参考 L2) + 当前 stable8 residual``。

    初始化时中间差值为零，之后允许主干、stable8 和 motion 分支按 arm 参与训练。
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        arm: str,
        mode: str = "residual",
        trainable_base_model: LiquidTaskModel,
        frozen_reference_base_model: LiquidTaskModel,
        stable_model: EvidenceResidual,
        motion_adapter: MotionAdapter,
    ) -> None:
        super().__init__()
        if arm not in ARMS or mode not in {"residual", "direct", "direct_grouped"}:
            raise ValueError("L2 model arm 无效")
        if (
            trainable_base_model.task != L2_TASK_NAME
            or frozen_reference_base_model.task != L2_TASK_NAME
            or stable_model.arm != "main"
        ):
            raise ValueError("L2 模型组件不满足 contract")
        self.config = config
        self.arm = arm
        self.mode = str(mode)
        self.trainable_base_model = trainable_base_model
        self.frozen_reference_base_model = frozen_reference_base_model
        self.stable_model = stable_model
        self.motion_adapter = motion_adapter
        self.trainable_base_model.trajectory_heads.eval()
        _freeze(self.frozen_reference_base_model)
        _freeze_trajectory_heads(self.trainable_base_model)
        if self.mode in {"direct", "direct_grouped"}:
            # direct 模式只训练完整 L2 主干；stable residual 和 motion
            # adapter 不参与该路径，保留模块是为了复用 checkpoint 结构。
            _freeze(self.stable_model)
            _freeze(self.motion_adapter)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-integrated-l2-model-contract-v1",
            "arm": self.arm,
            "mode": self.mode,
            "class_names": list(task_class_names(self.config, L2_TASK_NAME)),
            "base_model": self.trainable_base_model.model_contract(),
            "frozen_reference_base": {
                "same_initial_state_as_trainable_base": True,
                "requires_grad": False,
                "trajectory_heads": "frozen_unused",
            },
            "stable8_model": self.stable_model.model_contract(),
            "motion_adapter": self.motion_adapter.model_contract(),
            "motion_input": ARM_SPECS[self.arm]["motion_mode"],
            "input_mode": ARM_SPECS[self.arm]["input_mode"],
            "output_equation": (
                "current_logits"
                if self.mode in {"direct", "direct_grouped"}
                else "baseline_logits + (current_logits - frozen_reference_logits) + stable8_residual"
            ),
            "baseline_source": "none" if self.mode in {"direct", "direct_grouped"} else "event_baseline_logits_anchor",
            "unknown_classifier_class": False,
            "trajectory_heads_trainable": False,
        }

    def direct_logits(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        return_representation: bool = False,
    ) -> dict[str, torch.Tensor]:
        """执行完整主干分类路径，不注入 baseline/stable residual。"""

        if self.mode not in {"direct", "direct_grouped"}:
            raise ValueError("当前 L2 model 不是 direct 模式")
        output = self.trainable_base_model(
            **base_inputs,
            return_trace=bool(return_representation),
        )
        result: dict[str, Any] = {
            "sequence_logits": output["sequence_logits"],
            "final_logits": output["window_logits"],
        }
        if return_representation:
            result.update(
                {
                    "summary_features": output["summary_features"],
                    "event_trajectory": output["event_trajectory"],
                    "task_facts_sequence": output["task_facts_sequence"],
                    "event_facts_sequence": output["event_facts_sequence"],
                }
            )
        return result

    def _validate_sidecars(
        self,
        *,
        baseline_logits: torch.Tensor,
        stable_summary_tokens: torch.Tensor,
        stable_summary_valid: torch.Tensor,
        stable_frame_values: torch.Tensor,
        stable_frame_mask: torch.Tensor,
        motion_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if baseline_logits.ndim != 3:
            raise ValueError("baseline logits 必须是 [B,T,C]")
        batch, length, classes = baseline_logits.shape
        if (
            batch <= 0
            or length <= 0
            or classes != len(task_class_names(self.config, L2_TASK_NAME))
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
            raise ValueError("L2 sidecar shape 不一致")
        floating = (
            baseline_logits,
            stable_summary_tokens,
            stable_frame_values,
            motion_values,
        )
        if not all(value.is_floating_point() and torch.isfinite(value).all() for value in floating):
            raise ValueError("L2 sidecar 必须是有限浮点值")
        if any(value.device != baseline_logits.device for value in floating):
            raise ValueError("L2 sidecar 必须共用 device")
        if any(value.dtype != baseline_logits.dtype for value in floating):
            raise ValueError("L2 sidecar 必须共用 dtype")
        if any(
            value.device != baseline_logits.device
            for value in (stable_summary_valid, stable_frame_mask, padding_mask)
        ):
            raise ValueError("L2 sidecar mask 必须共用 device")
        if bool(
            baseline_logits.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()
        ):
            raise ValueError("padding 的 baseline logits 必须为零")
        if bool(
            motion_values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()
        ):
            raise ValueError("padding 的 motion 值必须为零")
        if not torch.equal(stable_summary_valid, stable_frame_mask.any(dim=-1) & ~padding_mask):
            raise ValueError("stable8 summary 与 reliable-frame contract 不一致")
        return batch, length

    @staticmethod
    def _native_events(
        model: LiquidTaskModel,
        *,
        base_inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """复用 L2 主干在 recurrence 前的 Fusion 计算。"""

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
        model: LiquidTaskModel,
        *,
        events: torch.Tensor,
        base_inputs: Mapping[str, torch.Tensor],
        return_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        padding = base_inputs["tr_padding_mask"]
        output = model.liquid(
            events,
            base_inputs["tr_dt_seconds"],
            padding,
            dt_valid=base_inputs["tr_dt_valid"],
            dt_reason=base_inputs["tr_dt_reason"],
            return_trace=bool(return_output),
        )
        logits = model.classifier(model.classifier_norm(output.sequence)).masked_fill(
            padding.unsqueeze(-1), 0.0
        )
        return (logits, output) if return_output else logits

    def _events_with_motion(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        motion_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.trainable_base_model
        # motion adapter 只作用于既有 Fusion 输出之后。
        native_events = self._native_events(model, base_inputs=base_inputs)
        selected_motion = (
            motion_values
            if ARM_SPECS[self.arm]["motion_mode"] == "causal"
            else torch.zeros_like(motion_values)
        )
        adapter = self.motion_adapter(selected_motion, padding_mask)
        events = (native_events + adapter).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(events).all():
            raise FloatingPointError("L2 融合事件包含非有限值")
        return events, adapter

    def _current_l2_logits(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        motion_values: torch.Tensor,
        return_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
        padding = base_inputs["tr_padding_mask"]
        events, adapter = self._events_with_motion(
            base_inputs=base_inputs,
            motion_values=motion_values,
            padding_mask=padding,
        )
        liquid_result = self._liquid_logits(
            self.trainable_base_model,
            events=events,
            base_inputs=base_inputs,
            return_output=return_output,
        )
        if return_output:
            logits, output = liquid_result
            return logits, adapter, output
        return liquid_result, adapter, None

    def _reference_l2_logits(
        self, *, base_inputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        # 冻结参考路径采用相同的 Fusion 分解且不注入 adapter，确保初始化时
        # trainable/reference 差值严格为零。
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

    def reference_logits(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """兼容旧调用方；请使用 ``frozen_reference_logits``。"""

        return self.frozen_reference_logits(base_inputs=base_inputs)

    def frozen_reference_logits(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """计算冻结参考主干的 logits，供构造事件 anchor。"""

        return self._reference_l2_logits(base_inputs=base_inputs)

    def forward(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        reference_base_inputs: Mapping[str, torch.Tensor] | None,
        baseline_logits: torch.Tensor,
        stable_summary_tokens: torch.Tensor,
        stable_summary_valid: torch.Tensor,
        stable_frame_values: torch.Tensor,
        stable_frame_mask: torch.Tensor,
        motion_values: torch.Tensor,
        frozen_reference_logits_override: torch.Tensor | None = None,
        return_representation: bool = False,
    ) -> dict[str, torch.Tensor]:
        padding = base_inputs.get("tr_padding_mask")
        if not isinstance(padding, torch.Tensor):
            raise TypeError("L2 base_inputs 缺少 padding tensor")
        batch, length = self._validate_sidecars(
            baseline_logits=baseline_logits,
            stable_summary_tokens=stable_summary_tokens,
            stable_summary_valid=stable_summary_valid,
            stable_frame_values=stable_frame_values,
            stable_frame_mask=stable_frame_mask,
            motion_values=motion_values,
            padding_mask=padding,
        )
        if batch != padding.shape[0] or length != padding.shape[1]:
            raise ValueError("L2 主干与 sidecar 时间维度不一致")
        if frozen_reference_logits_override is None:
            if reference_base_inputs is None:
                raise ValueError(
                    "L2 必须提供 reference_base_inputs 或已校验的 frozen reference logits"
                )
            reference_l2_logits = self._reference_l2_logits(
                base_inputs=reference_base_inputs
            )
        else:
            reference_l2_logits = frozen_reference_logits_override
            if (
                reference_l2_logits.shape != baseline_logits.shape
                or reference_l2_logits.device != baseline_logits.device
                or reference_l2_logits.dtype != baseline_logits.dtype
                or not torch.isfinite(reference_l2_logits).all()
            ):
                raise ValueError("L2 frozen reference logits override contract 不一致")
        current_l2_logits, motion_adapter_output, liquid_output = self._current_l2_logits(
            base_inputs=base_inputs,
            motion_values=motion_values,
            return_output=bool(return_representation),
        )
        stable_output = self.stable_model(
            baseline_logits=baseline_logits,
            summary_tokens=stable_summary_tokens,
            summary_valid=stable_summary_valid,
            frame_values=stable_frame_values,
            frame_mask=stable_frame_mask,
            padding_mask=padding,
        )
        stable_residual = stable_output["residual_logits"]
        l2_delta = current_l2_logits - reference_l2_logits
        sequence_logits = (
            baseline_logits + l2_delta + stable_residual
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
            raise FloatingPointError("L2 输出包含非有限值")
        result: dict[str, Any] = {
            "sequence_logits": sequence_logits,
            "baseline_logits": baseline_logits,
            "current_l2_logits": current_l2_logits,
            "frozen_reference_l2_logits": reference_l2_logits,
            "l2_delta_logits": l2_delta,
            "stable_residual_logits": stable_residual,
            "stable_sequence_logits": stable_output["sequence_logits"],
            "motion_adapter_output": motion_adapter_output,
        }
        if return_representation:
            if liquid_output is None or liquid_output.trace is None:
                raise RuntimeError("L2 representation requires raw liquid trace")
            physical_valid = base_inputs["rd_physical_valid"]
            micro_valid = base_inputs["micro_doppler_valid"]
            task_facts = self.trainable_base_model._task_facts_sequence(
                padding=padding,
                rd_frame_mask=base_inputs["rd_frame_mask"],
                rd_quality=base_inputs["rd_quality"],
                dt_valid=base_inputs["tr_dt_valid"],
                dt_reason=base_inputs["tr_dt_reason"],
                physical_valid=physical_valid,
                micro_valid=micro_valid,
                course_valid=None,
            )
            event_facts = self.trainable_base_model._trajectory_event_facts(
                padding=padding,
                rd_frame_mask=base_inputs["rd_frame_mask"],
                rd_quality=base_inputs["rd_quality"],
                dt_seconds=base_inputs["tr_dt_seconds"],
                dt_valid=base_inputs["tr_dt_valid"],
                dt_reason=base_inputs["tr_dt_reason"],
                physical_valid=physical_valid,
                micro_valid=micro_valid,
                course_features=None,
                course_valid=None,
                course_support=None,
                zero_ordinal=liquid_output.trace["zero_ordinal"],
            )
            features = self.trainable_base_model.trajectory_heads.build_representation_features(
                raw_sequence=liquid_output.sequence,
                trace=liquid_output.trace,
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


def build_l2_model(
    config: ModelConfig,
    *,
    arm: str,
    mode: str = "residual",
    base_state: Mapping[str, torch.Tensor],
    stable_state: Mapping[str, torch.Tensor],
    seed: int,
) -> L2Model:
    """从基础权重构造 L2 arm。"""

    if arm not in ARMS or mode not in {"residual", "direct", "direct_grouped"} or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("L2 model arm 或 seed 无效")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        trainable_base = LiquidTaskModel(config, task=L2_TASK_NAME)
        # trajectory heads 不参与分类训练；只加载已验证的分类/递归状态，且
        # 只允许这组 head key 缺失。
        base_state = dict(base_state)
        if mode in {"direct", "direct_grouped"}:
            # 迁移语义：只复用历史 TR/RD encoder，Fusion、Liquid 和
            # classifier 从当前配置重新初始化后训练。
            source_keys = {
                name: value
                for name, value in base_state.items()
                if str(name).startswith(("tr_encoder.", "rd_encoder."))
            }
            target_state = trainable_base.state_dict()
            expected_keys = {
                name for name in target_state
                if name.startswith(("tr_encoder.", "rd_encoder."))
            }
            if set(source_keys) != expected_keys:
                raise ValueError("L2 direct 初始化缺少 TR/RD encoder state")
            for name, value in source_keys.items():
                if tuple(value.shape) != tuple(target_state[name].shape):
                    raise ValueError(f"L2 direct 初始化 tensor shape 不一致：{name}")
                target_state[name].copy_(value)
            classifier_state = {}
            expected_state = {}
            incompatible = None
        else:
            classifier_state = {
                name: value
                for name, value in base_state.items()
                if not str(name).startswith("trajectory_heads.")
            }
            expected_state = {
                name: value
                for name, value in trainable_base.state_dict().items()
                if not str(name).startswith("trajectory_heads.")
            }
            if set(classifier_state) != set(expected_state):
                raise ValueError("L2 基础分类 state key 不一致")
            for name, value in classifier_state.items():
                if tuple(value.shape) != tuple(expected_state[name].shape):
                    raise ValueError(f"L2 基础 tensor shape 不一致：{name}")
            incompatible = trainable_base.load_state_dict(classifier_state, strict=False)
        if mode not in {"direct", "direct_grouped"}:
            expected_missing = {
                name
                for name in trainable_base.state_dict()
                if name.startswith("trajectory_heads.")
            }
            if incompatible is None or set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError("L2 基础 state key 不一致")
        if any(
            name.startswith("trajectory_heads.")
            for name in classifier_state
            if name not in trainable_base.state_dict()
        ):
            raise ValueError("L2 基础 state key 不一致")
        reference_base = copy.deepcopy(trainable_base)
        stable = EvidenceResidual(config, arm="main")
        stable.load_state_dict(dict(stable_state), strict=True)
        adapter = MotionAdapter(
            config,
            output_dim=trainable_base.liquid.cell.input_dim,
        )
        model = L2Model(
            config,
            arm=arm,
            mode=mode,
            trainable_base_model=trainable_base,
            frozen_reference_base_model=reference_base,
            stable_model=stable,
            motion_adapter=adapter,
        )
    return model


def trainable_parameter_groups(
    model: L2Model,
) -> dict[str, list[nn.Parameter]]:
    """返回显式且互不重叠的 L2 optimizer 参数组。"""

    base = [
        parameter
        for name, parameter in model.trainable_base_model.named_parameters()
        if parameter.requires_grad and not name.startswith("trajectory_heads.")
    ]
    stable = [parameter for parameter in model.stable_model.parameters() if parameter.requires_grad]
    motion = [parameter for parameter in model.motion_adapter.parameters() if parameter.requires_grad]
    if not base or (model.mode == "residual" and (not stable or not motion)):
        raise ValueError("L2 optimizer 参数组不能为空")
    all_ids = [id(parameter) for values in (base, stable, motion) for parameter in values]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("L2 optimizer 参数组发生重叠")
    if any(parameter.requires_grad for parameter in model.frozen_reference_base_model.parameters()):
        raise AssertionError("冻结参考主干包含可训练参数")
    if any(parameter.requires_grad for parameter in model.trainable_base_model.trajectory_heads.parameters()):
        raise AssertionError("trajectory heads 包含可训练参数")
    return {"base": base, "stable": stable, "motion": motion}


__all__ = [
    "ARM_SPECS",
    "ARMS",
    "L2Model",
    "MotionAdapter",
    "build_l2_model",
    "trainable_parameter_groups",
]
