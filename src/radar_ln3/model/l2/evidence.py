"""独立的因果 stable8 late residual 模型。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ..config import ModelConfig, task_class_names

EVIDENCE_ARMS = ("mechanism", "main")


class EvidenceResidual(nn.Module):
    """共享容量的 causal evidence encoder 与受约束 residual。

    ``mechanism`` 消费一个 reliable event-summary token，``main`` 消费当前
    事件全部 reliable frame token。两个 arm 的模块图一致，只改变输入 frame GRU
    的 token tensor。
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        arm: str,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if arm not in EVIDENCE_ARMS:
            raise ValueError(f"不支持的 evidence arm：{arm}")
        self.config = config
        self.arm = str(arm)
        model = config.model
        if model.evidence_dim != len(config.data.l2.stable8_indices):
            raise ValueError("模型与 stable8 数据的 evidence 宽度不一致")
        self.class_names = task_class_names(config, "l2")
        if len(self.class_names) != 3:
            raise ValueError("evidence 模型类别数量不一致")
        self.evidence_dim = int(model.evidence_dim)
        self.hidden_dim = int(
            config.model.stable8_hidden_dim if hidden_dim is None else hidden_dim
        )
        self.class_count = len(self.class_names)
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
        """返回每个正式模型都应保存的 architecture contract。"""

        return {
            "schema_version": "RADAR-LN3-evidence-residual-model-contract-v1",
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
            raise ValueError("baseline logits 必须是 [B,T,3]")
        batch, length = baseline_logits.shape[:2]
        if batch <= 0 or length <= 0:
            raise ValueError("evidence 模型需要非空 batch 和时间维")
        if summary_tokens.shape != (batch, length, self.evidence_dim):
            raise ValueError("summary token shape 不一致")
        if summary_valid.shape != (batch, length) or summary_valid.dtype != torch.bool:
            raise ValueError("summary validity shape 或 dtype 不一致")
        if frame_values.ndim != 4 or frame_values.shape[:2] != (batch, length):
            raise ValueError("frame values 必须是 [B,T,K,D]")
        frame_count = int(frame_values.shape[2])
        if frame_values.shape[-1] != self.evidence_dim:
            raise ValueError("frame value 宽度不一致")
        if frame_mask.shape != (batch, length, frame_count) or frame_mask.dtype != torch.bool:
            raise ValueError("frame mask shape 或 dtype 不一致")
        if padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
            raise ValueError("padding mask shape 或 dtype 不一致")
        values = (
            baseline_logits,
            summary_tokens,
            frame_values,
        )
        if not all(value.is_floating_point() for value in values):
            raise TypeError("evidence 模型值必须是浮点数")
        if any(value.device != baseline_logits.device for value in values):
            raise ValueError("evidence 模型值必须共用 device")
        if any(value.dtype != baseline_logits.dtype for value in values):
            raise ValueError("evidence 模型值必须共用 dtype")
        if any(value.device != baseline_logits.device for value in (summary_valid, frame_mask, padding_mask)):
            raise ValueError("evidence mask 必须与值共用 device")
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("evidence 模型值必须有限")
        if bool(baseline_logits.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("padding 的 baseline logits 必须为零")
        if bool(summary_valid.masked_select(padding_mask).any()):
            raise ValueError("padding 的 summary token 不能有效")
        invalid_summary = (~summary_valid).unsqueeze(-1).expand_as(summary_tokens)
        if bool(summary_tokens.masked_select(invalid_summary).ne(0).any()):
            raise ValueError("无效 summary token 必须为零")
        if bool(frame_mask.masked_select(padding_mask.unsqueeze(-1)).any()):
            raise ValueError("padding event 不能包含 frame token")
        invalid_frames = (~frame_mask).unsqueeze(-1).expand_as(frame_values)
        if bool(frame_values.masked_select(invalid_frames).ne(0).any()):
            raise ValueError("无效 frame token 必须为零")
        for row in range(batch):
            active = ~padding_mask[row]
            padded = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded.numel() and bool(active[int(padded[0]) :].any()):
                raise ValueError("padding 必须位于序列后缀")
            for event in torch.nonzero(active, as_tuple=False).flatten().tolist():
                mask = frame_mask[row, event]
                first_inactive = torch.nonzero(~mask, as_tuple=False).flatten()
                if first_inactive.numel() and bool(mask[int(first_inactive[0]) :].any()):
                    raise ValueError("frame padding 必须位于后缀")
        if not torch.equal(summary_valid, frame_mask.any(dim=-1) & ~padding_mask):
            raise ValueError(
                "summary validity 必须等于 reliable frame presence contract"
            )
        return batch, length, frame_count

    def _token_hidden(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """编码非空的 event-local token 序列。"""

        if tokens.ndim != 3 or lengths.ndim != 1 or tokens.shape[0] != lengths.shape[0]:
            raise ValueError("token batch shape 不一致")
        if bool((lengths <= 0).any()) or bool((lengths > tokens.shape[1]).any()):
            raise ValueError("token length 超出 token tensor")
        projected = self.token_projection(tokens)
        if self.config.model.token_activation == "silu":
            projected = torch.nn.functional.silu(projected)
        else:
            raise ValueError("不支持的 token activation")
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
            raise ValueError("stable8 初始状态 batch size 必须为正")
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
            raise TypeError("stable8 continuation state 必须是对象")
        parent_state = state.get("parent_state")
        previous_residual = state.get("previous_residual")
        if parent_state is None or previous_residual is None:
            raise ValueError("stable8 continuation state 不完整")
        if parent_state.shape != (batch_size, self.hidden_dim):
            raise ValueError("stable8 continuation parent state shape 不一致")
        if previous_residual.shape != (batch_size, self.class_count):
            raise ValueError("stable8 continuation residual state shape 不一致")
        if parent_state.device != device or previous_residual.device != device:
            raise ValueError("stable8 continuation state device 不一致")
        if parent_state.dtype != dtype or previous_residual.dtype != dtype:
            raise ValueError("stable8 continuation state dtype 不一致")
        if not parent_state.is_floating_point() or not previous_residual.is_floating_point():
            raise TypeError("stable8 continuation state 必须是浮点数")
        if not torch.isfinite(parent_state).all() or not torch.isfinite(previous_residual).all():
            raise ValueError("stable8 continuation state 必须有限")
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
        """处理一个事件，并返回更新后的显式 continuation state。"""

        if summary_tokens.ndim != 2 or summary_tokens.shape[-1] != self.evidence_dim:
            raise ValueError("stable8 event summary token shape 不一致")
        batch = int(summary_tokens.shape[0])
        if summary_valid.shape != (batch,) or summary_valid.dtype != torch.bool:
            raise ValueError("stable8 event summary validity 不一致")
        if frame_values.ndim != 3 or frame_values.shape[0] != batch or frame_values.shape[-1] != self.evidence_dim:
            raise ValueError("stable8 event frame values 不一致")
        if frame_mask.shape != frame_values.shape[:2] or frame_mask.dtype != torch.bool:
            raise ValueError("stable8 event frame mask 不一致")
        parent_state, previous_residual = self._validate_state(
            state,
            batch_size=batch,
            device=summary_tokens.device,
            dtype=summary_tokens.dtype,
        )
        if not torch.equal(summary_valid, frame_mask.any(dim=-1)):
            raise ValueError("stable8 event validity 字段不一致")
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
        # 每个全局事件索引都对应一行；padding 使用零输出行表示，且不会调用
        # recurrent update。
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
            raise FloatingPointError("stable8 evidence residual 产生非有限值")
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


def build_evidence_model(
    config: ModelConfig,
    *,
    arm: str,
    seed: int,
) -> EvidenceResidual:
    """以确定性方式构造一个 stable8 arm。"""

    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("stable8 初始化 seed 必须是非负整数")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return EvidenceResidual(config, arm=arm)


__all__ = ["EVIDENCE_ARMS", "EvidenceResidual", "build_evidence_model"]
