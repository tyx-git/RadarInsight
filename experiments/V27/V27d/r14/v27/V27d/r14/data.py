"""Canonical known-only V27 L2 inputs plus the frozen V27a Main sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from v27.v27_baseline.data import (
    V27TaskDataset,
    V27TaskSequence,
    collate_v27,
    model_input_kwargs,
)
from v27.v27_baseline.protocol import V27_L2_TASK_NAME

from .protocol import R14Context, project_path


@dataclass(frozen=True)
class R14L2Sequence:
    sequence: V27TaskSequence
    cached_v27_base_logits: torch.Tensor
    stable_summary_tokens: torch.Tensor
    stable_summary_valid: torch.Tensor
    stable_frame_values: torch.Tensor
    stable_frame_mask: torch.Tensor
    v27a_main_reference_logits: torch.Tensor


def _load_v27a_cache(context: R14Context, *, split: str) -> dict[str, Any]:
    key = "v27a_train_cache" if split == "train" else "v27a_validation_cache"
    path = project_path(context.project_root, context.config.source["paths"][key], must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r14 V27a cache must be a mapping")
    checked = dict(payload)
    if checked.get("cache_payload_sha256") is None:
        raise ValueError("V27d-r14 V27a cache lacks payload digest")
    expected_count = 953 if split == "train" else 223
    if len(checked.get("parent_sample_ids", ())) != expected_count:
        raise ValueError("V27d-r14 V27a cache parent support differs")
    required = (
        "parent_sample_ids",
        "parent_labels",
        "lengths",
        "padding_mask",
        "event_baseline_logits",
        "event_summary_tokens",
        "event_summary_valid",
        "event_frame_values",
        "event_frame_mask",
    )
    if any(name not in checked for name in required):
        raise ValueError("V27d-r14 V27a cache fields are incomplete")
    if checked["event_baseline_logits"].shape != (expected_count, 29, 3):
        raise ValueError("V27d-r14 V27a baseline-logit cache shape differs")
    if checked["event_summary_tokens"].shape != (expected_count, 29, 8):
        raise ValueError("V27d-r14 V27a stable8 summary shape differs")
    if checked["event_frame_values"].shape[:2] != (expected_count, 29) or checked["event_frame_values"].shape[-1] != 8:
        raise ValueError("V27d-r14 V27a stable8 frame shape differs")
    return checked


class R14L2Dataset(Dataset[R14L2Sequence]):
    """Known DBB parents in the V27a Main cache order."""

    def __init__(self, context: R14Context, *, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("V27d-r14 dataset split must be train or val")
        split_name = (
            context.baseline_config.data.train_split_name
            if split == "train"
            else context.baseline_config.data.validation_split_name
        )
        self.base = V27TaskDataset(
            context.project_root,
            context.baseline_config,
            task=V27_L2_TASK_NAME,
            split_name=split_name,
        )
        self.stable = _load_v27a_cache(context, split=split)
        self.parent_ids = [str(value) for value in self.stable["parent_sample_ids"]]
        self.labels = [str(value) for value in self.stable["parent_labels"]]
        if set(self.base.parent_ids) != set(self.parent_ids) or len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("V27d-r14 V27/V27a parent populations differ")
        self._base_position = {parent_id: index for index, parent_id in enumerate(self.base.parent_ids)}
        self._stable_position = {parent_id: index for index, parent_id in enumerate(self.parent_ids)}
        for parent_id, label, length in zip(self.parent_ids, self.labels, self.stable["lengths"], strict=True):
            item = self.base[self._base_position[parent_id]]
            if item.base.label_name != label or item.base.event_count != int(length):
                raise ValueError("V27d-r14 V27/V27a label or length alignment differs")

    def reference_cache_payload(self) -> dict[str, Any]:
        """Return the immutable inputs needed to materialize reference logits."""

        return self.stable

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> R14L2Sequence:
        row = int(index)
        if not 0 <= row < len(self):
            raise IndexError("V27d-r14 dataset index is out of range")
        parent_id = self.parent_ids[row]
        item = self.base[self._base_position[parent_id]]
        length = int(item.base.event_count)
        padding = torch.from_numpy(np.array(self.stable["padding_mask"][row], copy=True))
        expected_padding = torch.arange(29) >= length
        if not torch.equal(padding, expected_padding):
            raise ValueError("V27d-r14 V27a cache padding differs from the parent length")
        return R14L2Sequence(
            sequence=item,
            cached_v27_base_logits=torch.from_numpy(np.array(self.stable["event_baseline_logits"][row], copy=True)),
            stable_summary_tokens=torch.from_numpy(np.array(self.stable["event_summary_tokens"][row], copy=True)),
            stable_summary_valid=torch.from_numpy(np.array(self.stable["event_summary_valid"][row], copy=True)),
            stable_frame_values=torch.from_numpy(np.array(self.stable["event_frame_values"][row], copy=True)),
            stable_frame_mask=torch.from_numpy(np.array(self.stable["event_frame_mask"][row], copy=True)),
            v27a_main_reference_logits=torch.from_numpy(np.array(self.stable["event_baseline_logits"][row], copy=True)),
        )


def _trim(value: torch.Tensor, *, width: int, padding: torch.Tensor, name: str) -> torch.Tensor:
    result = value[:width]
    if result.shape[0] != width:
        raise ValueError(f"V27d-r14 {name} sidecar width differs")
    if result.dtype == torch.bool:
        mask = padding
        if result.ndim > 1:
            mask = mask.unsqueeze(-1)
        if bool(result.masked_select(mask.expand_as(result)).any()):
            raise ValueError(f"V27d-r14 padded {name} is active")
    else:
        mask = padding
        while mask.ndim < result.ndim:
            mask = mask.unsqueeze(-1)
        if bool(result.masked_select(mask.expand_as(result)).ne(0).any()):
            raise ValueError(f"V27d-r14 padded {name} is non-zero")
    return result


def collate_r14(items: list[R14L2Sequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("V27d-r14 cannot collate an empty batch")
    batch = collate_v27([item.sequence for item in items])
    raw_motion = batch.pop("tr_motion_features", None)
    raw_valid = batch.pop("tr_motion_valid_mask", None)
    if raw_motion is None or raw_valid is None or raw_motion.shape[-1] != 0:
        raise ValueError("V27d-r14 native V27 L2 unexpectedly exposes TR motion")
    width = int(batch["tr_features"].shape[1])
    padding = batch["tr_padding_mask"]
    sidecar_names = (
        "cached_v27_base_logits",
        "stable_summary_tokens",
        "stable_summary_valid",
        "stable_frame_values",
        "stable_frame_mask",
        "v27a_main_reference_logits",
    )
    values: dict[str, list[torch.Tensor]] = {name: [] for name in sidecar_names}
    for row, item in enumerate(items):
        for name in sidecar_names:
            values[name].append(
                _trim(
                    getattr(item, name),
                    width=width,
                    padding=padding[row],
                    name=name,
                )
            )
    result = {**batch, **{name: torch.stack(rows) for name, rows in values.items()}}
    if result["cached_v27_base_logits"].shape[-1] != 3 or result["stable_frame_values"].shape[-1] != 8:
        raise ValueError("V27d-r14 sidecar feature widths differ")
    return result


def build_r14_dataloader(dataset: R14L2Dataset, *, batch_size: int, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("V27d-r14 batch size must be positive")
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate_r14, generator=generator)


def r14_model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    values = model_input_kwargs(batch, device)
    if "tr_motion_features" in values or "tr_motion_valid_mask" in values:
        raise AssertionError("V27d-r14 raw motion leaked into native L2 inputs")
    return values


def r14_sidecar_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "cached_v27_base_logits",
        "stable_summary_tokens",
        "stable_summary_valid",
        "stable_frame_values",
        "stable_frame_mask",
    )
    result: dict[str, torch.Tensor] = {}
    for name in names:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r14 sidecar field is not a tensor: {name}")
        result[name] = value.to(device=device, non_blocking=True)
    return result


__all__ = ["R14L2Dataset", "R14L2Sequence", "build_r14_dataloader", "collate_r14", "r14_model_input_kwargs", "r14_sidecar_kwargs"]