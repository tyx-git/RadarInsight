"""R16 L2 data view with canonical RD-frame timestamps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from v27.V27d.r14.data import R14L2Dataset, R14L2Sequence, collate_r14, r14_model_input_kwargs

from .protocol import R16Context


FRAME_TIME_TOLERANCE_SECONDS = 1.0e-8


@dataclass(frozen=True)
class R16L2Sequence:
    """One r14 control sample plus its unmodified canonical frame times."""

    r14: R14L2Sequence
    frame_times: torch.Tensor
    frame_time_valid: torch.Tensor
    frame_time_fallback: torch.Tensor


def _event_frame_times(
    context: R16Context,
    dataset: R14L2Dataset,
    sequence: R14L2Sequence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = sequence.sequence.base
    base_dataset = dataset.base.base
    rows = base_dataset._groups[base.parent_sample_id].iloc[: base.event_count]
    max_frames = int(base.rd_frames.shape[1])
    times = np.zeros((base.event_count, max_frames), dtype=np.float32)
    valid = np.zeros((base.event_count,), dtype=bool)
    fallback = np.ones((base.event_count,), dtype=bool)

    for event_index, (_, row) in enumerate(rows.iterrows()):
        if not bool(base.rd_frame_mask[event_index].any().item()):
            continue
        raw_indices = json.loads(str(row["rd_tensor_frame_indices_json"]))
        if not isinstance(raw_indices, list):
            raise ValueError("V27d-r16 RD frame indices must decode to a list")
        selected = [int(value) for value in raw_indices[:max_frames]]
        if len(selected) != int(base.rd_frame_mask[event_index].sum().item()):
            raise ValueError(
                "V27d-r16 frame-time count differs from the frozen RD frame mask"
            )
        _, _, source_times = base_dataset._tensor_cache.load(str(row["rd_tensor_path"]))
        if selected and (min(selected) < 0 or max(selected) >= len(source_times)):
            raise ValueError("V27d-r16 RD frame-time index is out of bounds")
        selected_times = np.asarray(source_times[np.asarray(selected, dtype=np.int64)], dtype=np.float32)
        if not np.isfinite(selected_times).all():
            raise ValueError("V27d-r16 RD frame times contain non-finite values")
        times[event_index, : len(selected_times)] = selected_times
        if len(selected_times) >= 2:
            deltas = np.diff(selected_times.astype(np.float64))
            valid[event_index] = bool(np.all(deltas >= -FRAME_TIME_TOLERANCE_SECONDS))
        fallback[event_index] = not bool(valid[event_index])

    result = (
        torch.from_numpy(times),
        torch.from_numpy(valid),
        torch.from_numpy(fallback),
    )
    if not torch.isfinite(result[0]).all() or bool(result[0].masked_select(~base.rd_frame_mask).ne(0).any()):
        raise ValueError("V27d-r16 inactive frame timestamps must be finite zero")
    if bool((result[1] & result[2]).any()):
        raise ValueError("V27d-r16 valid frame-time events cannot be fallback events")
    return result


class R16L2Dataset(Dataset[R16L2Sequence]):
    """The exact r14 L2 population with a timestamp sidecar."""

    def __init__(self, context: R16Context, *, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("V27d-r16 dataset split must be train or val")
        self.r14_context = context.r14_context
        self.r14 = R14L2Dataset(self.r14_context, split=split)
        self.context = context
        self.split = split
        self.parent_ids = list(self.r14.parent_ids)
        self._frame_time_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.r14)

    def __getitem__(self, index: int) -> R16L2Sequence:
        if not 0 <= int(index) < len(self):
            raise IndexError("V27d-r16 dataset index is out of range")
        sequence = self.r14[int(index)]
        cached = self._frame_time_cache.get(sequence.sequence.base.parent_sample_id)
        if cached is None:
            cached = _event_frame_times(self.context, self.r14, sequence)
            self._frame_time_cache[sequence.sequence.base.parent_sample_id] = cached
        return R16L2Sequence(
            r14=sequence,
            frame_times=cached[0].clone(),
            frame_time_valid=cached[1].clone(),
            frame_time_fallback=cached[2].clone(),
        )


def _trim(value: torch.Tensor, *, width: int, name: str) -> torch.Tensor:
    if value.shape[0] != width:
        raise ValueError(f"V27d-r16 {name} width differs from the batch")
    return value


def collate_r16(items: list[R16L2Sequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("V27d-r16 cannot collate an empty batch")
    r14_batch = collate_r14([item.r14 for item in items])
    width = int(r14_batch["tr_features"].shape[1])
    max_frames = int(r14_batch["rd_frames"].shape[2])
    times: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    fallback: list[torch.Tensor] = []
    for item in items:
        length = int(item.r14.sequence.base.event_count)
        if item.frame_times.shape != (length, max_frames) or item.frame_time_valid.shape != (length,) or item.frame_time_fallback.shape != (length,):
            raise ValueError("V27d-r16 frame-time sidecar shape differs")
        padded_times = torch.zeros((width, max_frames), dtype=torch.float32)
        padded_valid = torch.zeros((width,), dtype=torch.bool)
        padded_fallback = torch.ones((width,), dtype=torch.bool)
        padded_times[:length] = item.frame_times
        padded_valid[:length] = item.frame_time_valid
        padded_fallback[:length] = item.frame_time_fallback
        times.append(padded_times)
        valid.append(padded_valid)
        fallback.append(padded_fallback)
    result = {
        **r14_batch,
        "rd_frame_times": torch.stack(times),
        "rd_frame_time_valid": torch.stack(valid),
        "rd_frame_time_fallback": torch.stack(fallback),
    }
    padding = result["tr_padding_mask"]
    if bool(result["rd_frame_time_valid"].masked_select(padding).any()):
        raise ValueError("V27d-r16 padded frame-time events cannot be valid")
    if bool(result["rd_frame_times"].masked_select(padding.unsqueeze(-1).expand_as(result["rd_frame_times"])).ne(0).any()):
        raise ValueError("V27d-r16 padded frame timestamps must be zero")
    if bool(result["rd_frame_times"].masked_select(~result["rd_frame_mask"]).ne(0).any()):
        raise ValueError("V27d-r16 inactive frame timestamps must be zero")
    return result


def build_r16_dataloader(
    dataset: R16L2Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("V27d-r16 batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r16,
        generator=generator,
    )


def r16_model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    values = r14_model_input_kwargs(batch, device)
    forbidden = {"rd_frame_times", "rd_frame_time_valid", "rd_frame_time_fallback"}
    if forbidden & set(values):
        raise AssertionError("V27d-r16 frame-time sidecars leaked into the r14 base input")
    return values


def r16_frame_time_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    names = ("rd_frame_times", "rd_frame_time_valid", "rd_frame_time_fallback")
    result: dict[str, torch.Tensor] = {}
    for name in names:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r16 frame-time sidecar is missing: {name}")
        result[name] = value.to(device=device, non_blocking=True)
    return result


__all__ = [
    "FRAME_TIME_TOLERANCE_SECONDS",
    "R16L2Dataset",
    "R16L2Sequence",
    "build_r16_dataloader",
    "collate_r16",
    "r16_frame_time_kwargs",
    "r16_model_input_kwargs",
]
