"""Full L2 inputs plus independently materialized motion/stable8 sidecars.

The r22 dataset deliberately rebuilds the V27 L2 classifier inputs from the
canonical processed index.  V27a supplies only its already-normalized stable8
evidence and V27c-r21 supplies only numerically gated motion values.  This
keeps the treatment at the intended pre-Liquid boundary without allowing a
validity mask, cached endpoint logit, Unknown label, or test row into the
classifier input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.runtime import sequence_sha256
from v27.v27_baseline.data import (
    V27TaskDataset,
    V27TaskSequence,
    collate_v27,
    model_input_kwargs,
)
from v27.v27_baseline.protocol import V27_L2_TASK_NAME

from .protocol import (
    R22_L2_CLASSES,
    R22Context,
    load_verified_r21_motion_cache,
    load_verified_v27a_cache,
)


@dataclass(frozen=True)
class R22L2Sequence:
    """One canonical L2 sequence with aligned stable8/motion sidecars."""

    sequence: V27TaskSequence
    cached_v27_base_logits: torch.Tensor
    stable_summary_tokens: torch.Tensor
    stable_summary_valid: torch.Tensor
    stable_frame_values: torch.Tensor
    stable_frame_mask: torch.Tensor
    motion_values: torch.Tensor
    v27a_main_reference_logits: torch.Tensor


class R22L2Dataset(Dataset[R22L2Sequence]):
    """Known-only L2 parents in the V27a cache's canonical parent order."""

    def __init__(self, context: R22Context, *, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("V27c-r22 dataset split must be train or val")
        self.context = context
        self.split = split
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
        self.motion = load_verified_r21_motion_cache(context, split=split)
        self.parent_ids = [str(value) for value in self.stable["parent_sample_ids"]]
        self.label_names = [str(value) for value in self.stable["parent_labels"]]
        expected_count = (
            context.config.data.train_parent_count
            if split == "train"
            else context.config.data.validation_parent_count
        )
        if len(self.base) != expected_count or len(self.parent_ids) != expected_count:
            raise ValueError("V27c-r22 L2 dataset parent support differs")
        if set(self.base.parent_ids) != set(self.parent_ids):
            raise ValueError("V27c-r22 L2/V27a parent populations differ")
        self._base_position = {
            parent_id: position for position, parent_id in enumerate(self.base.parent_ids)
        }
        self._stable_position = {
            parent_id: position for position, parent_id in enumerate(self.parent_ids)
        }
        if len(self._stable_position) != len(self.parent_ids):
            raise ValueError("V27c-r22 V27a parent ids are not unique")
        motion_parents = [str(value) for value in self.motion["parent_ids"]]
        motion_labels = [str(value) for value in self.motion["label_names"]]
        if motion_parents != self.parent_ids or motion_labels != self.label_names:
            raise ValueError("V27c-r22 r21 motion sidecar parent/label order differs")
        self._assert_event_alignment()

    def _assert_event_alignment(self) -> None:
        """Prove r21 motion and V27 L2 have the same causal event sequence."""

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
                raise ValueError("V27c-r22 canonical parent lacks a V27 L2 position")
            item = self.base[position]
            if (
                item.base.parent_sample_id != parent_id
                or item.base.label_name != label
                or item.base.event_count != int(length)
            ):
                raise ValueError("V27c-r22 V27 L2 parent/label/length differs")
            rows = self.base.base._groups[parent_id].iloc[: item.base.event_count]
            indices = rows["causal_event_index"].tolist()
            if indices != list(range(item.base.event_count)):
                raise ValueError("V27c-r22 processed index causal event order differs")
            event_ids.extend(rows["sample_id"].astype(str).tolist())
            expected_event_parents.extend([parent_id] * item.base.event_count)
        if sequence_sha256(event_ids) != self.motion.get("event_ids_order_sha256"):
            raise ValueError("V27c-r22 r21 motion event IDs do not align with V27 L2")
        if expected_event_parents != [str(value) for value in self.stable["event_parent_ids"]]:
            raise ValueError("V27c-r22 V27a stable8 event parent order differs")

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> R22L2Sequence:
        if isinstance(index, bool) or not 0 <= int(index) < len(self):
            raise IndexError("V27c-r22 dataset index is out of range")
        row = int(index)
        parent_id = self.parent_ids[row]
        item = self.base[self._base_position[parent_id]]
        length = int(item.base.event_count)
        width = self.context.config.data.max_sequence_length
        tensors = self.motion["tensors"]
        motion_padding = tensors["padding_mask"][row]
        stable_padding = self.stable["padding_mask"][row]
        if (
            int(tensors["lengths"][row].item()) != length
            or int(self.stable["lengths"][row]) != length
            or not torch.equal(motion_padding, torch.from_numpy(stable_padding.copy()))
            or motion_padding.shape != (width,)
        ):
            raise ValueError("V27c-r22 sidecar length/padding differs from V27 L2")
        if bool(motion_padding[:length].any()) or not bool(motion_padding[length:].all()):
            raise ValueError("V27c-r22 sidecar padding is not a suffix")
        return R22L2Sequence(
            sequence=item,
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
            # This tensor was gated once by r21 materialization.  Its validity
            # mask intentionally is not retained in R22L2Sequence or collation.
            motion_values=tensors["motion_values"][row].detach().clone(),
            # Audit-only cached V27a Main endpoint.  It is never passed to the
            # r22 model or loss, but makes initial replay checks explicit.
            v27a_main_reference_logits=tensors["frozen_endpoint_logits"][
                row
            ].detach().clone(),
        )


def _trim_sidecar(
    value: torch.Tensor,
    *,
    length: int,
    width: int,
    padding: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if value.ndim < 1 or value.shape[0] < width:
        raise ValueError(f"V27c-r22 {name} sidecar time axis differs")
    suffix = value[width:]
    if suffix.numel() and bool(suffix.ne(0).any()):
        raise ValueError(f"V27c-r22 {name} has non-zero values beyond collated width")
    result = value[:width]
    if result.shape[0] != width:
        raise ValueError(f"V27c-r22 {name} sidecar trim differs")
    if value.shape[0] < length:
        raise ValueError(f"V27c-r22 {name} sidecar is shorter than its parent")
    if result.dtype == torch.bool:
        # Bool sidecars use true only for reliable active observations.  A
        # padded event must never carry an active frame or summary marker.
        if result.ndim > 1 and bool(result.masked_select(padding.unsqueeze(-1)).any()):
            raise ValueError(f"V27c-r22 padded {name} mask is active")
        if result.ndim == 1 and bool(result.masked_select(padding).any()):
            raise ValueError(f"V27c-r22 padded {name} mask is active")
    else:
        mask = padding.unsqueeze(-1)
        while mask.ndim < result.ndim:
            mask = mask.unsqueeze(-1)
        if bool(result.masked_select(mask.expand_as(result)).ne(0).any()):
            raise ValueError(f"V27c-r22 padded {name} values must be zero")
    return result


def collate_r22(items: list[R22L2Sequence]) -> dict[str, Any]:
    """Collate native V27 L2 inputs and sidecars without a motion mask."""

    if not items:
        raise ValueError("V27c-r22 cannot collate an empty batch")
    batch = collate_v27([item.sequence for item in items])
    if batch.get("task") != V27_L2_TASK_NAME:
        raise ValueError("V27c-r22 collate received a non-L2 sequence")
    # `collate_point_sequences` exposes a generic raw-motion pair for every
    # dataset, including V27's zero-width default.  R22 must not retain either
    # field because only the r21 numerically gated sidecar may reach the model.
    raw_motion = batch.pop("tr_motion_features", None)
    raw_motion_valid = batch.pop("tr_motion_valid_mask", None)
    if raw_motion is None or raw_motion_valid is None:
        raise ValueError("V27c-r22 base collate unexpectedly omitted raw motion fields")
    if raw_motion.shape != raw_motion_valid.shape or raw_motion.shape[-1] != 0:
        raise ValueError("V27c-r22 native V27 L2 must not supply raw motion features")
    width = int(batch["tr_features"].shape[1])
    padding = batch["tr_padding_mask"]
    if padding.dtype != torch.bool or padding.shape != (len(items), width):
        raise ValueError("V27c-r22 collated padding shape/dtype differs")
    tensors: dict[str, list[torch.Tensor]] = {
        "cached_v27_base_logits": [],
        "stable_summary_tokens": [],
        "stable_summary_valid": [],
        "stable_frame_values": [],
        "stable_frame_mask": [],
        "motion_values": [],
        "v27a_main_reference_logits": [],
    }
    names = {
        "cached_v27_base_logits": "cached V27 base logits",
        "stable_summary_tokens": "stable8 summary tokens",
        "stable_summary_valid": "stable8 summary validity",
        "stable_frame_values": "stable8 frame values",
        "stable_frame_mask": "stable8 frame mask",
        "motion_values": "gated causal motion values",
        "v27a_main_reference_logits": "V27a Main audit logits",
    }
    for row, item in enumerate(items):
        length = int(item.sequence.base.event_count)
        expected_padding = torch.arange(width) >= length
        if not torch.equal(padding[row].cpu(), expected_padding):
            raise ValueError("V27c-r22 collated V27 padding differs from sidecar length")
        for field in tensors:
            tensors[field].append(
                _trim_sidecar(
                    getattr(item, field),
                    length=length,
                    width=width,
                    padding=expected_padding,
                    name=names[field],
                )
            )
    result = {**batch, **{field: torch.stack(values) for field, values in tensors.items()}}
    if (
        result["cached_v27_base_logits"].shape != (len(items), width, len(R22_L2_CLASSES))
        or result["motion_values"].shape[-1] != 9
        or result["stable_summary_tokens"].shape[-1] != 8
        or result["stable_frame_values"].shape[-1] != 8
        or result["stable_summary_valid"].dtype != torch.bool
        or result["stable_frame_mask"].dtype != torch.bool
    ):
        raise ValueError("V27c-r22 collated sidecar shapes differ")
    if "tr_motion_features" in result or "tr_motion_valid_mask" in result:
        raise AssertionError("V27c-r22 validity mask leaked into the model batch")
    return result


def build_r22_dataloader(
    dataset: R22L2Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("V27c-r22 batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r22,
        generator=generator,
    )


def r22_model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move exactly the original V27 L2 classifier inputs to ``device``."""

    values = model_input_kwargs(batch, device)
    if "tr_motion_features" in values or "tr_motion_valid_mask" in values:
        raise AssertionError("V27c-r22 raw motion mask reached V27 model inputs")
    return values


def r22_sidecar_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
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
            raise TypeError(f"V27c-r22 sidecar field must be a tensor: {name}")
        result[name] = value.to(device=device, non_blocking=True)
    if "tr_motion_valid_mask" in batch or "tr_motion_features" in batch:
        raise AssertionError("V27c-r22 raw motion fields leaked into sidecar inputs")
    return result


__all__ = [
    "R22L2Dataset",
    "R22L2Sequence",
    "build_r22_dataloader",
    "collate_r22",
    "r22_model_input_kwargs",
    "r22_sidecar_kwargs",
]
