"""Canonical L2 data views for the V27c-r23 factorial arms."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.features import CausalTRMotionState, encode_tr_feature_values, parse_feature_json
from v27.v27_baseline.data import (
    V27TaskDataset,
    V27TaskSequence,
    collate_v27,
    model_input_kwargs,
)
from v27.v27_baseline.protocol import V27_L2_TASK_NAME

from .preprocessing import clip_tr_values
from .protocol import R23_ARM_SPECS, R23_L2_CLASSES, R23Context, load_verified_v27a_cache


@dataclass(frozen=True)
class R23L2Sequence:
    """One L2 parent with an arm-specific input and a legacy reference input."""

    sequence: V27TaskSequence
    reference_sequence: V27TaskSequence
    cached_v27_base_logits: torch.Tensor
    stable_summary_tokens: torch.Tensor
    stable_summary_valid: torch.Tensor
    stable_frame_values: torch.Tensor
    stable_frame_mask: torch.Tensor
    motion_values: torch.Tensor


def _materialize_tr_variant(
    dataset: V27TaskDataset,
    sequence: V27TaskSequence,
    *,
    mode: str,
    clipping: Mapping[str, Any],
) -> V27TaskSequence:
    if mode not in {"legacy", "robust"}:
        raise ValueError("V27c-r23 TR input mode is invalid")
    base = sequence.base
    rows = dataset.base._groups[base.parent_sample_id].iloc[: base.event_count]
    state = CausalTRMotionState()
    tr_values: list[torch.Tensor] = []
    tr_missing: list[torch.Tensor] = []
    motion_values: list[torch.Tensor] = []
    motion_valid: list[torch.Tensor] = []
    for _, row in rows.iterrows():
        values = parse_feature_json(str(row["tr_feature_values_json"]))
        if mode == "robust":
            values = clip_tr_values(values, clipping)
        encoded, missing = encode_tr_feature_values(values)
        event_index = len(tr_values)
        dt_seconds = float(base.tr_dt_seconds[event_index].item())
        dt_valid = bool(base.tr_dt_valid[event_index].item())
        motion, valid = state.update(
            values,
            dt_seconds=dt_seconds,
            dt_valid=dt_valid,
        )
        tr_values.append(torch.from_numpy(encoded).float())
        tr_missing.append(torch.from_numpy(missing).float())
        motion_values.append(torch.from_numpy(motion).float())
        motion_valid.append(torch.from_numpy(valid).bool())
    if len(tr_values) != base.event_count:
        raise ValueError("V27c-r23 TR materialization length differs from the parent")
    updated = replace(
        base,
        tr_features=torch.stack(tr_values, dim=0),
        tr_missing_mask=torch.stack(tr_missing, dim=0),
        tr_motion_features=torch.stack(motion_values, dim=0),
        tr_motion_valid_mask=torch.stack(motion_valid, dim=0),
    )
    return V27TaskSequence(base=updated, task=sequence.task)


class R23L2Dataset(Dataset[R23L2Sequence]):
    """Known-only L2 parents with arm-specific TR and causal-motion materialization."""

    def __init__(self, context: R23Context, *, split: str, arm: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("V27c-r23 dataset split must be train or val")
        if arm not in R23_ARM_SPECS:
            raise ValueError("V27c-r23 dataset arm is invalid")
        self.context = context
        self.split = split
        self.arm = arm
        self.input_mode = R23_ARM_SPECS[arm]["input_mode"]
        split_name = (
            context.config.data.train_split_name
            if split == "train"
            else context.config.data.validation_split_name
        )
        self.base = V27TaskDataset(
            context.project_root,
            context.v27_config,
            task=V27_L2_TASK_NAME,
            split_name=split_name,
        )
        self.stable = load_verified_v27a_cache(context, split=split)
        self.parent_ids = [str(value) for value in self.stable["parent_sample_ids"]]
        self.label_names = [str(value) for value in self.stable["parent_labels"]]
        expected_count = (
            context.config.data.train_parent_count
            if split == "train"
            else context.config.data.validation_parent_count
        )
        if len(self.base) != expected_count or len(self.parent_ids) != expected_count:
            raise ValueError("V27c-r23 L2 parent support differs")
        if set(self.base.parent_ids) != set(self.parent_ids):
            raise ValueError("V27c-r23 L2/V27a parent populations differ")
        self._base_position = {
            parent_id: position for position, parent_id in enumerate(self.base.parent_ids)
        }
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("V27c-r23 V27a parent ids are not unique")
        self._clipping = context.input_binding["tr_clipping"]
        self._assert_event_alignment()

    def _assert_event_alignment(self) -> None:
        event_ids: list[str] = []
        expected_event_parents: list[str] = []
        for parent_id, label, length in zip(
            self.parent_ids,
            self.label_names,
            self.stable["lengths"],
            strict=True,
        ):
            position = self._base_position.get(parent_id)
            if position is None:
                raise ValueError("V27c-r23 stable parent lacks a V27 L2 position")
            item = self.base[position]
            if (
                item.base.parent_sample_id != parent_id
                or item.base.label_name != label
                or item.base.event_count != int(length)
            ):
                raise ValueError("V27c-r23 L2/V27a parent label or length differs")
            rows = self.base.base._groups[parent_id].iloc[: item.base.event_count]
            indices = rows["causal_event_index"].tolist()
            if indices != list(range(item.base.event_count)):
                raise ValueError("V27c-r23 processed index causal order differs")
            event_ids.extend(rows["sample_id"].astype(str).tolist())
            expected_event_parents.extend([parent_id] * item.base.event_count)
        if expected_event_parents != [str(value) for value in self.stable["event_parent_ids"]]:
            raise ValueError("V27c-r23 V27a event parent order differs")
        # The stable cache does not expose event ids, so the processed index is
        # the single event-order authority for the reconstructed motion.
        if len(event_ids) != len(self.stable["event_parent_ids"]):
            raise ValueError("V27c-r23 event population differs")

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> R23L2Sequence:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("V27c-r23 dataset index must be an integer")
        position = int(index)
        if not 0 <= position < len(self):
            raise IndexError("V27c-r23 dataset index is out of range")
        row = position
        parent_id = self.parent_ids[row]
        item = self.base[self._base_position[parent_id]]
        legacy = _materialize_tr_variant(
            self.base, item, mode="legacy", clipping=self._clipping
        )
        primary = (
            legacy
            if self.input_mode == "legacy"
            else _materialize_tr_variant(
                self.base, item, mode="robust", clipping=self._clipping
            )
        )
        length = int(item.base.event_count)
        width = self.context.config.data.max_sequence_length
        stable_padding = np.asarray(self.stable["padding_mask"][row], dtype=bool)
        if (
            int(self.stable["lengths"][row]) != length
            or stable_padding.shape != (width,)
            or not np.array_equal(stable_padding, np.arange(width) >= length)
        ):
            raise ValueError("V27c-r23 stable cache length/padding differs")
        return R23L2Sequence(
            sequence=primary,
            reference_sequence=legacy,
            cached_v27_base_logits=torch.from_numpy(
                np.array(self.stable["event_baseline_logits"][row], copy=True)
            ),
            stable_summary_tokens=torch.from_numpy(
                np.array(self.stable["event_summary_tokens"][row], copy=True)
            ),
            stable_summary_valid=torch.from_numpy(
                np.array(self.stable["event_summary_valid"][row], copy=True)
            ),
            stable_frame_values=torch.from_numpy(
                np.array(self.stable["event_frame_values"][row], copy=True)
            ),
            stable_frame_mask=torch.from_numpy(
                np.array(self.stable["event_frame_mask"][row], copy=True)
            ),
            motion_values=primary.base.tr_motion_features.detach().clone(),
        )


def _trim_sidecar(
    value: torch.Tensor,
    *,
    length: int,
    width: int,
    padding: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if value.ndim < 1 or value.shape[0] < width or value.shape[0] < length:
        raise ValueError(f"V27c-r23 {name} sidecar time axis differs")
    if value[width:].numel() and bool(value[width:].ne(0).any()):
        raise ValueError(f"V27c-r23 {name} has non-zero values beyond batch width")
    result = value[:width]
    mask = padding
    if result.dtype == torch.bool:
        expanded = mask
        while expanded.ndim < result.ndim:
            expanded = expanded.unsqueeze(-1)
        if bool(result.masked_select(expanded.expand_as(result)).any()):
            raise ValueError(f"V27c-r23 padded {name} mask is active")
    else:
        expanded = mask
        while expanded.ndim < result.ndim:
            expanded = expanded.unsqueeze(-1)
        if bool(result.masked_select(expanded.expand_as(result)).ne(0).any()):
            raise ValueError(f"V27c-r23 padded {name} values must be zero")
    return result


def collate_r23(items: list[R23L2Sequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("V27c-r23 cannot collate an empty batch")
    primary = collate_v27([item.sequence for item in items])
    reference = collate_v27([item.reference_sequence for item in items])
    if primary.get("task") != V27_L2_TASK_NAME or reference.get("task") != V27_L2_TASK_NAME:
        raise ValueError("V27c-r23 collate received a non-L2 sequence")
    raw_motion = primary.pop("tr_motion_features")
    raw_motion_valid = primary.pop("tr_motion_valid_mask")
    reference.pop("tr_motion_features")
    reference.pop("tr_motion_valid_mask")
    if (
        raw_motion.shape != raw_motion_valid.shape
        or raw_motion.shape[-1] != 9
        or raw_motion_valid.dtype != torch.bool
        or bool(raw_motion.masked_select(~raw_motion_valid).ne(0).any())
    ):
        raise ValueError("V27c-r23 causal motion materialization shape differs")
    if primary["parent_sample_ids"] != reference["parent_sample_ids"]:
        raise ValueError("V27c-r23 primary/reference parent order differs")
    reference_keys = (
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
    )
    for key in reference_keys:
        reference_key = f"reference_{key}"
        if key not in reference:
            raise ValueError(f"V27c-r23 reference batch lacks {key}")
        primary[reference_key] = reference[key]
    primary["motion_values"] = raw_motion
    width = int(primary["tr_features"].shape[1])
    padding = primary["tr_padding_mask"]
    if padding.dtype != torch.bool:
        raise ValueError("V27c-r23 padding must be boolean")
    fields: dict[str, list[torch.Tensor]] = {
        "cached_v27_base_logits": [],
        "stable_summary_tokens": [],
        "stable_summary_valid": [],
        "stable_frame_values": [],
        "stable_frame_mask": [],
    }
    names = {
        "cached_v27_base_logits": "cached V27a logits",
        "stable_summary_tokens": "stable8 summary tokens",
        "stable_summary_valid": "stable8 summary validity",
        "stable_frame_values": "stable8 frame values",
        "stable_frame_mask": "stable8 frame mask",
    }
    for row, item in enumerate(items):
        length = int(item.sequence.base.event_count)
        expected_padding = torch.arange(width) >= length
        if not torch.equal(padding[row].cpu(), expected_padding):
            raise ValueError("V27c-r23 collated padding differs from parent length")
        for field, values in fields.items():
            values.append(
                _trim_sidecar(
                    getattr(item, field),
                    length=length,
                    width=width,
                    padding=expected_padding,
                    name=names[field],
                )
            )
    primary.update({field: torch.stack(values) for field, values in fields.items()})
    if (
        primary["cached_v27_base_logits"].shape
        != (len(items), width, len(R23_L2_CLASSES))
        or primary["motion_values"].shape != (len(items), width, 9)
        or primary["stable_summary_tokens"].shape[-1]
        != primary["stable_frame_values"].shape[-1]
        or primary["stable_summary_valid"].dtype != torch.bool
        or primary["stable_frame_mask"].dtype != torch.bool
    ):
        raise ValueError("V27c-r23 collated sidecar shapes differ")
    if bool(primary["motion_values"].masked_select(padding.unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27c-r23 padded motion values must be zero")
    return primary


def build_r23_dataloader(
    dataset: R23L2Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("V27c-r23 batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r23,
        generator=generator,
    )


def r23_model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return model_input_kwargs(batch, device)


def r23_reference_model_input_kwargs(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    reference_keys = (
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
    )
    missing = [f"reference_{key}" for key in reference_keys if f"reference_{key}" not in batch]
    if missing:
        raise ValueError(f"V27c-r23 batch is missing reference inputs: {missing}")
    result: dict[str, Any] = {}
    for key in reference_keys:
        value = batch[f"reference_{key}"]
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"V27c-r23 reference input {key} must be a non-scalar tensor")
        result[key] = value.to(device)
    return result


def r23_sidecar_kwargs(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    required = (
        "cached_v27_base_logits",
        "stable_summary_tokens",
        "stable_summary_valid",
        "stable_frame_values",
        "stable_frame_mask",
        "motion_values",
    )
    result: dict[str, torch.Tensor] = {}
    for name in required:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27c-r23 sidecar field must be a tensor: {name}")
        result[name] = value.to(device=device, non_blocking=True)
    if "tr_motion_features" in batch or "tr_motion_valid_mask" in batch:
        raise AssertionError("V27c-r23 raw motion fields leaked into model sidecars")
    return result


__all__ = ["R23L2Dataset", "R23L2Sequence", "build_r23_dataloader", "collate_r23", "r23_model_input_kwargs", "r23_reference_model_input_kwargs", "r23_sidecar_kwargs"]