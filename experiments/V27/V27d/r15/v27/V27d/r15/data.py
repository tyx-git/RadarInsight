"""V27d-r15 的 L1 数据视图与 observation-increment 对齐逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from v27.v27_baseline.data import (
    V27OpenSetDataset,
    V27TaskDataset,
    V27TaskSequence,
    collate_v27,
    collate_v27_open_set,
    model_input_kwargs,
)
from v27.v27_baseline.protocol import V27_L1_TASK_NAME, load_v27_config

from .features import load_observation_cache
from .protocol import R15Context, project_path


def _course_cache_path(context: R15Context, v27_config: Any, *, view: str) -> Path:
    """按数据视图选择与其样本集合完全一致的 course cache。"""

    l1_data = v27_config.data.l1
    if view in {"representation_development", "calibration"}:
        return project_path(
            context.project_root,
            context.config.source["role_course_cache_paths"][view],
            must_exist=True,
        )
    if view == "train":
        return project_path(
            context.project_root,
            l1_data.train_course_cache_path,
            must_exist=True,
        )
    if view == "validation":
        return project_path(
            context.project_root,
            l1_data.validation_course_cache_path,
            must_exist=True,
        )
    raise ValueError(f"V27d-r15 course cache view is unsupported: {view}")


@dataclass(frozen=True)
class R15Sequence:
    sequence: V27TaskSequence
    observation_values: torch.Tensor
    include_classifier_labels: bool


class R15L1Dataset(Dataset[R15Sequence]):
    """一个固定角色的 V27 L1 序列与 V10 sidecar。"""

    def __init__(self, context: R15Context, *, view: str) -> None:
        if view not in {"train", "validation", "representation_development", "calibration"}:
            raise ValueError(f"V27d-r15 view is unsupported: {view}")
        self.context = context
        self.view = view
        self.cache = load_observation_cache(context)["views"][view]
        v27_config = load_v27_config(
            context.project_root,
            context.config.source["v27_config_path"],
        )
        if view in {"train", "validation"}:
            split_name = (
                context.config.data["train_split_name"]
                if view == "train"
                else context.config.data["validation_split_name"]
            )
            self.base: Dataset[V27TaskSequence] = V27TaskDataset(
                context.project_root,
                v27_config,
                task=V27_L1_TASK_NAME,
                split_name=split_name,
            )
        else:
            split_name = (
                context.config.data["train_split_name"]
                if view == "representation_development"
                else context.config.data["validation_split_name"]
            )
            l1_data = v27_config.data.l1
            course_cache = _course_cache_path(context, v27_config, view=view)
            self.base = V27OpenSetDataset(
                context.project_root,
                v27_config,
                task=V27_L1_TASK_NAME,
                split_name=split_name,
                parent_ids=[str(value) for value in self.cache["parent_ids"]],
                course_cache_path=course_cache,
            )
        expected_ids = [str(value) for value in self.cache["parent_ids"]]
        actual_ids = [str(value) for value in self.base.parent_ids]
        if actual_ids != expected_ids:
            raise ValueError(f"V27d-r15 {view} parent order differs from observation cache")
        if len(self.base) != len(expected_ids):
            raise ValueError(f"V27d-r15 {view} parent count differs")

    @property
    def include_classifier_labels(self) -> bool:
        return self.view in {"train", "validation"}

    @property
    def parent_ids(self) -> list[str]:
        return [str(value) for value in self.cache["parent_ids"]]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> R15Sequence:
        if isinstance(index, bool) or not 0 <= int(index) < len(self):
            raise IndexError("V27d-r15 dataset index is out of range")
        position = int(index)
        item = self.base[position]
        length = int(item.base.event_count)
        expected_length = int(self.cache["lengths"][position].item())
        if length != expected_length:
            raise ValueError(f"V27d-r15 parent length differs: {item.base.parent_sample_id}")
        values = self.cache["values"][position, :length].detach().clone()
        if values.shape != (length, 9) or not torch.isfinite(values).all():
            raise ValueError("V27d-r15 observation values have an invalid shape/value")
        return R15Sequence(
            sequence=item,
            observation_values=values,
            include_classifier_labels=self.include_classifier_labels,
        )


def collate_r15(items: list[R15Sequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("V27d-r15 cannot collate an empty batch")
    include_labels = bool(items[0].include_classifier_labels)
    if any(bool(item.include_classifier_labels) != include_labels for item in items):
        raise ValueError("V27d-r15 batch cannot mix known-only and open-set views")
    batch = collate_v27([item.sequence for item in items]) if include_labels else collate_v27_open_set([item.sequence for item in items])
    width = int(batch["tr_features"].shape[1])
    motion = torch.zeros((len(items), width, 9), dtype=torch.float32)
    padding = batch["tr_padding_mask"]
    for row, item in enumerate(items):
        length = int(item.sequence.base.event_count)
        if item.observation_values.shape != (length, 9):
            raise ValueError("V27d-r15 observation sidecar shape differs")
        motion[row, :length] = item.observation_values
    if bool(motion.masked_select(padding.unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d-r15 padded observation values must be zero")
    batch["observation_increment_values"] = motion
    return batch


def build_r15_dataloader(
    dataset: R15L1Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("V27d-r15 batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r15,
        generator=generator,
    )


def r15_model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    values = model_input_kwargs(batch, device)
    if "tr_motion_features" in values or "tr_motion_valid_mask" in values:
        raise AssertionError("V27d-r15 legacy motion fields leaked into model inputs")
    return values


def r15_sidecar_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    value = batch.get("observation_increment_values")
    if not isinstance(value, torch.Tensor):
        raise TypeError("V27d-r15 observation sidecar is missing")
    return {"observation_increment_values": value.to(device=device, non_blocking=True)}


__all__ = ["R15L1Dataset", "R15Sequence", "build_r15_dataloader", "collate_r15", "r15_model_input_kwargs", "r15_sidecar_kwargs"]