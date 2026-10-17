"""Independent V28 Liquid V2 classifiers with explicit r15 residual boundary."""
from __future__ import annotations

from typing import Any, Mapping
import torch
from torch import nn
from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper, liquid_v2_contract
from .protocol import V28Context, V28_L1_TASK, V28_L1_CLASSES, V28_L2_TASK, V28_L2_CLASSES


class V28RDEncoder(nn.Module):
    def __init__(self, dim: int, height: int, width: int) -> None:
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.output_dim = int(dim)
        self.net = nn.Sequential(nn.Conv2d(2, 8, 3, stride=(1, 4), padding=1), nn.GroupNorm(4, 8), nn.SiLU(), nn.Conv2d(8, 16, 3, stride=(2, 4), padding=1), nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.SiLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(24, int(dim)), nn.LayerNorm(int(dim)), nn.SiLU())

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 6 or tuple(frames.shape[3:]) != (2, self.height, self.width) or mask.shape != frames.shape[:3] or mask.dtype != torch.bool:
            raise ValueError("V28 RD input shape differs")
        if not torch.isfinite(frames).all():
            raise ValueError("V28 RD frames are non-finite")
        batch, length, count = frames.shape[:3]
        flat = frames.reshape(batch * length * count, 2, self.height, self.width)
        valid = mask.reshape(-1)
        encoded_flat = flat.new_zeros((flat.shape[0], self.output_dim))
        if bool(valid.any()):
            # GroupNorm and the convolutional encoder operate independently per
            # frame.  Invalid frames are masked out immediately after this
            # encoder, so skipping them is numerically equivalent while
            # avoiding a large convolution over zero padding.
            encoded_flat[valid] = self.proj(self.net(flat[valid]))
        encoded = encoded_flat.reshape(batch, length, count, self.output_dim)
        weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        return (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


class V28ObservationResidual(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(9, int(hidden_dim), bias=False)
        self.norm = nn.LayerNorm(int(hidden_dim), elementwise_affine=False)
        self.output = nn.Linear(int(hidden_dim), int(output_dim), bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, values: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        if values.shape[:2] != padding.shape or values.shape[-1] != 9 or not torch.isfinite(values).all() or bool(values.masked_select(padding.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V28 observation residual input differs")
        return self.output(torch.nn.functional.silu(self.norm(self.input(values)))).masked_fill(padding.unsqueeze(-1), 0.0)


class V28TaskModel(nn.Module):
    def __init__(self, context: V28Context, *, task: str) -> None:
        super().__init__()
        if task not in {V28_L1_TASK, V28_L2_TASK}:
            raise ValueError("V28 task is invalid")
        self.context, self.task = context, task
        model = context.config.model
        self.tr_dim, self.missing_dim, self.quality_dim = int(model["tr_value_dim"]), int(model["tr_missing_dim"]), int(model["rd_quality_dim"])
        self.hidden_dim, self.input_dim, self.rd_dim = int(model["hidden_dim"]), int(model["liquid_input_dim"]), int(model["rd_embedding_dim"])
        self.physical_dim, self.micro_dim = (0, 0) if task == V28_L1_TASK else (4, 8)
        self.class_names = V28_L1_CLASSES if task == V28_L1_TASK else V28_L2_CLASSES
        self.tr_encoder = nn.Sequential(nn.Linear(self.tr_dim + self.missing_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU())
        self.rd_encoder = V28RDEncoder(self.rd_dim, int(model["rd_height"]), int(model["rd_width"]))
        optional_dim = self.physical_dim + self.micro_dim + (2 if task == V28_L1_TASK else 0)
        self.fusion = nn.Sequential(nn.Linear(self.hidden_dim + self.rd_dim + self.quality_dim + optional_dim, self.input_dim), nn.LayerNorm(self.input_dim), nn.SiLU())
        cell = LiquidV2Cell(self.input_dim, int(model["liquid_hidden_dim"]), tau_min_seconds=float(model["tau_min_seconds"]), tau_max_seconds=float(model["tau_max_seconds"]), zero_dt_event_floor_seconds=float(model["zero_dt_event_floor_seconds"]), dt_max_seconds=float(model["dt_max_seconds"]), invalid_dt_update_mode=str(model["invalid_dt_update_mode"]), invalid_dt_observation_gain=float(model["invalid_dt_observation_gain"]))
        self.liquid = LiquidV2SequenceWrapper(cell)
        self.classifier_norm = nn.LayerNorm(int(model["liquid_hidden_dim"]))
        self.classifier = nn.Linear(int(model["liquid_hidden_dim"]), len(self.class_names))
        self.observation_residual = V28ObservationResidual(self.input_dim, int(model["observation_residual_hidden_dim"])) if task == V28_L1_TASK else None

    def model_contract(self) -> dict[str, Any]:
        return {"schema_version": "RADAR-LN3-V28-task-model-contract-v3", "task": self.task, "class_names": list(self.class_names), "l1_l2_recurrent_state_shared": False, "all_valid_events_update_task_state": True, "unknown_is_classifier_class": False, "l2_clutter_classifier_training": False, "l1_observation_increment_boundary": "post_fusion_pre_liquid_v2" if self.task == V28_L1_TASK else None, "classifier_readout": "independent_layer_norm_on_raw_state", "liquid_v2_contract": liquid_v2_contract(self.liquid.cell)}

    def _validate(self, inputs: Mapping[str, torch.Tensor]) -> tuple[int, int]:
        tr = inputs["tr_features"]
        if tr.ndim != 3 or tr.shape[-1] != self.tr_dim:
            raise ValueError("V28 TR shape differs")
        batch, length = tr.shape[:2]
        for key, shape in (("tr_missing_mask", (batch, length, self.missing_dim)), ("tr_dt_seconds", (batch, length)), ("tr_dt_valid", (batch, length)), ("tr_dt_reason", (batch, length)), ("tr_padding_mask", (batch, length)), ("rd_quality", (batch, length, self.quality_dim))):
            if tuple(inputs[key].shape) != shape:
                raise ValueError(f"V28 input shape differs: {key}")
        padding = inputs["tr_padding_mask"]
        if inputs["tr_dt_valid"].dtype != torch.bool or padding.dtype != torch.bool or inputs["tr_dt_reason"].dtype not in (torch.int64, torch.long):
            raise ValueError("V28 time masks have invalid dtypes")
        frames = inputs["rd_frames"]
        if frames.ndim != 6 or tuple(frames.shape[3:]) != (2, int(self.context.config.model["rd_height"]), int(self.context.config.model["rd_width"])) or inputs["rd_frame_mask"].shape != frames.shape[:3] or inputs["rd_frame_mask"].dtype != torch.bool:
            raise ValueError("V28 RD shape/mask differs")
        if bool(padding.all(dim=1).any()):
            raise ValueError("V28 empty parent")
        if not all(torch.isfinite(value).all() for value in (tr, inputs["tr_missing_mask"], inputs["tr_dt_seconds"], inputs["rd_quality"], frames)):
            raise ValueError("V28 input contains non-finite values")
        if bool(inputs["tr_missing_mask"].lt(0).any()) or bool(inputs["tr_missing_mask"].gt(1).any()) or bool(inputs["rd_quality"].lt(0).any()) or bool(inputs["rd_quality"].gt(1).any()):
            raise ValueError("V28 missing masks and RD quality facts must be within [0,1]")
        if bool(tr.masked_select(padding.unsqueeze(-1)).ne(0).any()) or bool(inputs["tr_missing_mask"].masked_select(padding.unsqueeze(-1)).ne(0).any()) or bool(inputs["tr_dt_seconds"].masked_select(padding).ne(0).any()) or bool(inputs["rd_quality"].masked_select(padding.unsqueeze(-1)).ne(0).any()) or bool(frames.masked_select(padding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).ne(0).any()):
            raise ValueError("V28 padded values must be zero")
        if bool(inputs["rd_frame_mask"].masked_select(padding.unsqueeze(-1)).any()) or bool(frames.masked_select((~inputs["rd_frame_mask"]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).ne(0).any()):
            raise ValueError("V28 RD padding/mask values differ")
        if bool(((inputs["tr_dt_reason"] == 3) != inputs["tr_dt_valid"]).masked_select(~padding).any()):
            raise ValueError("V28 dt reason/validity differs")
        if "observation_increment" not in inputs or inputs["observation_increment"].shape != (batch, length, 9):
            raise ValueError("V28 observation increment shape differs")
        if not torch.isfinite(inputs["observation_increment"]).all() or bool(inputs["observation_increment"].masked_select(padding.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V28 observation increment contains invalid padded values")
        if self.task == V28_L1_TASK:
            for key in ("course_features", "course_valid", "course_support"):
                if key not in inputs:
                    raise ValueError(f"V28 L1 input is missing {key}")
            if inputs["course_features"].shape != (batch, length, 1) or inputs["course_valid"].shape != (batch, length, 1) or inputs["course_support"].shape != (batch, length):
                raise ValueError("V28 L1 course input shape differs")
            if inputs["course_valid"].dtype != torch.bool or not torch.isfinite(inputs["course_features"]).all() or bool(inputs["course_features"].masked_select(~inputs["course_valid"]).ne(0).any()) or bool(inputs["course_features"].masked_select(padding.unsqueeze(-1)).ne(0).any()) or bool(inputs["course_valid"].masked_select(padding.unsqueeze(-1)).any()) or bool(inputs["course_support"].masked_select(padding).ne(0).any()) or bool(inputs["course_support"].lt(0).any()):
                raise ValueError("V28 L1 course facts violate the causal/padding contract")
        else:
            for key, shape in (("rd_physical_features", (batch, length, self.physical_dim)), ("rd_physical_valid", (batch, length)), ("micro_doppler_features", (batch, length, self.micro_dim)), ("micro_doppler_valid", (batch, length))):
                if tuple(inputs[key].shape) != shape:
                    raise ValueError(f"V28 L2 input shape differs: {key}")
            if inputs["rd_physical_valid"].dtype != torch.bool or inputs["micro_doppler_valid"].dtype != torch.bool:
                raise ValueError("V28 L2 feature validity dtype differs")
            if not torch.isfinite(inputs["rd_physical_features"]).all() or not torch.isfinite(inputs["micro_doppler_features"]).all() or bool(inputs["rd_physical_features"].lt(0).any()) or bool(inputs["rd_physical_features"].gt(1).any()) or bool(inputs["micro_doppler_features"].lt(0).any()) or bool(inputs["micro_doppler_features"].gt(1).any()) or bool(inputs["rd_physical_features"].masked_select(~inputs["rd_physical_valid"].unsqueeze(-1)).ne(0).any()) or bool(inputs["micro_doppler_features"].masked_select(~inputs["micro_doppler_valid"].unsqueeze(-1)).ne(0).any()):
                raise ValueError("V28 L2 feature values violate validity/bounds")
        return batch, length

    def _facts(self, inputs: Mapping[str, torch.Tensor], physical_valid: torch.Tensor | None, micro_valid: torch.Tensor | None, zero_ordinal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = inputs["tr_padding_mask"]
        dtype = inputs["rd_quality"].dtype
        active = ~padding
        count = active.to(dtype=dtype).cumsum(dim=1).clamp_min(1.0)
        def frac(value: torch.Tensor) -> torch.Tensor:
            return (value.to(dtype=dtype) * active.to(dtype=dtype)).cumsum(dim=1) / count
        facts = [
            (active.to(dtype=dtype).cumsum(dim=1) / float(self.context.config.data["max_sequence_length"])).clamp(max=1.0),
            frac(inputs["rd_frame_mask"].any(dim=-1)),
            frac(inputs["rd_quality"][..., 1]),
            frac(inputs["tr_dt_valid"]),
            frac(inputs["tr_dt_reason"].eq(2)),
            frac(micro_valid if micro_valid is not None else torch.zeros_like(active)),
            frac(physical_valid if physical_valid is not None else torch.zeros_like(active)),
        ]
        if self.task == V28_L1_TASK:
            facts.append(frac(inputs["course_valid"][..., 0]))
        task_facts = torch.stack(facts, dim=-1).masked_fill(padding.unsqueeze(-1), 0.0)
        course_value = course_ok = course_count = torch.zeros_like(inputs["tr_dt_seconds"], dtype=dtype)
        if self.task == V28_L1_TASK:
            course_value = inputs["course_features"][..., 0] * inputs["course_valid"][..., 0].to(dtype=dtype)
            course_ok = inputs["course_valid"][..., 0].to(dtype=dtype)
            course_count = (inputs["course_support"].to(dtype=dtype) / float(max(1, self.context.config.data["max_sequence_length"] - 1))).clamp(max=1.0)
        event_facts = torch.cat((inputs["rd_quality"], (inputs["tr_dt_seconds"].clamp(min=0.0, max=float(self.context.config.model["dt_max_seconds"])) / float(self.context.config.model["dt_max_seconds"])).unsqueeze(-1), inputs["tr_dt_valid"].to(dtype=dtype).unsqueeze(-1), torch.stack((inputs["tr_dt_reason"].eq(1).to(dtype=dtype), inputs["tr_dt_reason"].eq(2).to(dtype=dtype), inputs["tr_dt_reason"].eq(3).to(dtype=dtype)), dim=-1), inputs["rd_frame_mask"].any(dim=-1).to(dtype=dtype).unsqueeze(-1), (physical_valid if physical_valid is not None else torch.zeros_like(active)).to(dtype=dtype).unsqueeze(-1), (micro_valid if micro_valid is not None else torch.zeros_like(active)).to(dtype=dtype).unsqueeze(-1), course_value.unsqueeze(-1), course_ok.unsqueeze(-1), course_count.unsqueeze(-1), zero_ordinal.to(dtype=dtype).unsqueeze(-1)), dim=-1).masked_fill(padding.unsqueeze(-1), 0.0)
        return task_facts, event_facts

    def forward(self, **inputs: torch.Tensor) -> dict[str, Any]:
        batch, length = self._validate(inputs)
        padding = inputs["tr_padding_mask"]
        tr = self.tr_encoder(torch.cat((inputs["tr_features"], inputs["tr_missing_mask"]), dim=-1))
        rd = self.rd_encoder(inputs["rd_frames"], inputs["rd_frame_mask"])
        optional: list[torch.Tensor] = []
        physical_valid = micro_valid = None
        if self.task == V28_L1_TASK:
            optional.extend((inputs["course_features"] * inputs["course_valid"].to(dtype=tr.dtype), inputs["course_valid"].to(dtype=tr.dtype)))
        else:
            physical_valid, micro_valid = inputs["rd_physical_valid"], inputs["micro_doppler_valid"]
            optional.extend((inputs["rd_physical_features"] * physical_valid.to(dtype=tr.dtype).unsqueeze(-1), inputs["micro_doppler_features"] * micro_valid.to(dtype=tr.dtype).unsqueeze(-1)))
        events = self.fusion(torch.cat((tr, rd, inputs["rd_quality"], *optional), dim=-1))
        if self.task == V28_L1_TASK:
            residual = self.observation_residual(inputs["observation_increment"], padding)
            events = (events + residual).masked_fill(padding.unsqueeze(-1), 0.0)
        liquid = self.liquid(events, inputs["tr_dt_seconds"], padding, dt_valid=inputs["tr_dt_valid"], dt_reason=inputs["tr_dt_reason"], return_trace=True)
        if liquid.trace is None:
            raise RuntimeError("V28 Liquid V2 trace is missing")
        trace = dict(liquid.trace)
        trace["zero_ordinal"] = trace["zero_ordinal"].to(dtype=torch.long)
        logits = self.classifier(self.classifier_norm(liquid.sequence)).masked_fill(padding.unsqueeze(-1), 0.0)
        lengths = (~padding).long().sum(dim=1)
        rows = torch.arange(batch, device=logits.device)
        task_facts, event_facts = self._facts(inputs, physical_valid, micro_valid, trace["zero_ordinal"])
        return {"task": self.task, "class_names": list(self.class_names), "sequence_logits": logits, "final_logits": logits[rows, lengths - 1], "raw_sequence": liquid.sequence, "trace": trace, "task_facts_sequence": task_facts, "event_facts_sequence": event_facts, "lengths": lengths, "padding_mask": padding, "final_raw_state": liquid.final_state}

    def forward_encoded_chunk(
        self,
        events: torch.Tensor,
        dt_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run an already encoded chunk through the same offline Liquid V2 cell."""

        liquid = self.liquid(
            events,
            dt_seconds,
            padding_mask,
            dt_valid=dt_valid,
            dt_reason=dt_reason,
            initial_state=initial_state,
            zero_ordinal=zero_ordinal,
            return_trace=True,
        )
        logits = self.classifier(self.classifier_norm(liquid.sequence)).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return {"sequence_logits": logits, "raw_sequence": liquid.sequence, "final_raw_state": liquid.final_state, "trace": liquid.trace}

    def online_encoded_step(
        self,
        event: torch.Tensor,
        dt_seconds: torch.Tensor,
        *,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        state: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance one encoded event through the exact online Liquid V2 cell."""

        step = self.liquid.step(event, dt_seconds, dt_valid=dt_valid, dt_reason=dt_reason, state=state, zero_ordinal=zero_ordinal)
        logits = self.classifier(self.classifier_norm(step.next_state))
        return {"logits": logits, "raw_state": step.next_state, "step": step}


__all__ = ["V28ObservationResidual", "V28RDEncoder", "V28TaskModel"]
