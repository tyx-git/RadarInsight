from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SessionBagDataset(Dataset):
    def __init__(
        self,
        base_dataset: Any,
        indices: Sequence[int],
        max_clips: int = 0,
        random_sampling: bool = False,
        sort_by_length: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.max_clips = max(int(max_clips), 0)
        self.random_sampling = bool(random_sampling)
        session_to_indices: dict[str, list[int]] = {}
        for dataset_index in self.indices:
            sample = self.base_dataset.samples[dataset_index]
            session_to_indices.setdefault(str(sample.session_id), []).append(int(dataset_index))
        self.sort_by_length = bool(sort_by_length)
        ordered_session_ids = sorted(session_to_indices.keys())
        if self.sort_by_length:
            ordered_session_ids = sorted(
                ordered_session_ids,
                key=lambda item: (len(session_to_indices[item]), str(item)),
            )
        self.session_ids = ordered_session_ids
        self.session_to_indices = session_to_indices

    def __len__(self) -> int:
        return len(self.session_ids)

    def _resolve_indices(self, session_id: str) -> list[int]:
        resolved = list(self.session_to_indices[session_id])
        if self.max_clips <= 0 or len(resolved) <= self.max_clips:
            return resolved
        if self.random_sampling:
            sampled = random.sample(resolved, self.max_clips)
            sampled.sort()
            return sampled
        positions = np.linspace(0, len(resolved) - 1, num=self.max_clips, dtype=np.int64)
        return [resolved[int(position)] for position in positions.tolist()]

    def __getitem__(self, index: int) -> dict[str, Any]:
        session_id = self.session_ids[int(index)]
        resolved_indices = self._resolve_indices(session_id)
        items = [self.base_dataset[dataset_index] for dataset_index in resolved_indices]
        if not items:
            raise IndexError(f"Empty bag for session: {session_id}")

        def stack_array(key: str) -> np.ndarray:
            values = [np.asarray(item[key]) for item in items]
            return np.stack(values, axis=0)

        first_item = items[0]
        base_sample = self.base_dataset.samples[resolved_indices[0]]
        source_name = str(getattr(base_sample, 'source_name', first_item.get('source_name', 'Unknown')))
        group_id = str(getattr(base_sample, 'group_id', session_id))
        return {
            "sample_id": session_id,
            "session_id": session_id,
            "clip_sample_ids": [str(item["sample_id"]) for item in items],
            "clip_mask": np.ones((len(items),), dtype=np.float32),
            "track_features": stack_array("track_features"),
            "track_short_features": stack_array("track_short_features"),
            "track_long_features": stack_array("track_long_features"),
            "track_global_features": stack_array("track_global_features"),
            "rd_features": stack_array("rd_features"),
            "label": np.int64(first_item["label"]),
            "label_name": str(first_item.get("label_name", "")),
            "source_name": source_name,
            "group_id": group_id,
            "has_rd": stack_array("has_rd").reshape(len(items)),
            "bird_rd_target": np.float32(first_item.get("bird_rd_target", -1.0)),
            "bird_surface_target": np.float32(first_item.get("bird_surface_target", -1.0)),
            "ship_drone_target": np.float32(first_item.get("ship_drone_target", -1.0)),
            "ship_bird_target": np.float32(first_item.get("ship_bird_target", -1.0)),
            "ship_air_target": np.float32(first_item.get("ship_air_target", -1.0)),
            "track_router_target": np.int64(first_item.get("track_router_target", -1)),
        }


def _pad_stack(batch: list[np.ndarray], fill_value: float = 0.0) -> torch.Tensor:
    max_clips = max(array.shape[0] for array in batch)
    template_shape = batch[0].shape[1:]
    padded = np.full((len(batch), max_clips, *template_shape), fill_value, dtype=batch[0].dtype)
    for row_index, array in enumerate(batch):
        padded[row_index, : array.shape[0]] = array
    return torch.from_numpy(padded)


def session_bag_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch is not supported")
    result: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in batch],
        "session_id": [item["session_id"] for item in batch],
        "clip_sample_ids": [item["clip_sample_ids"] for item in batch],
        "clip_mask": _pad_stack([np.asarray(item["clip_mask"], dtype=np.float32).reshape(-1, 1) for item in batch], fill_value=0.0).squeeze(-1),
        "track_features": _pad_stack([item["track_features"].astype(np.float32) for item in batch], fill_value=0.0),
        "track_short_features": _pad_stack([item["track_short_features"].astype(np.float32) for item in batch], fill_value=0.0),
        "track_long_features": _pad_stack([item["track_long_features"].astype(np.float32) for item in batch], fill_value=0.0),
        "track_global_features": _pad_stack([item["track_global_features"].astype(np.float32) for item in batch], fill_value=0.0),
        "rd_features": _pad_stack([item["rd_features"].astype(np.float32) for item in batch], fill_value=0.0),
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "label_name": [item["label_name"] for item in batch],
        "source_name": [item["source_name"] for item in batch],
        "group_id": [item["group_id"] for item in batch],
        "has_rd": _pad_stack([np.asarray(item["has_rd"], dtype=np.float32).reshape(-1, 1) for item in batch], fill_value=0.0).squeeze(-1),
        "bird_rd_target": torch.tensor([float(item["bird_rd_target"]) for item in batch], dtype=torch.float32),
        "bird_surface_target": torch.tensor([float(item["bird_surface_target"]) for item in batch], dtype=torch.float32),
        "ship_drone_target": torch.tensor([float(item["ship_drone_target"]) for item in batch], dtype=torch.float32),
        "ship_bird_target": torch.tensor([float(item["ship_bird_target"]) for item in batch], dtype=torch.float32),
        "ship_air_target": torch.tensor([float(item["ship_air_target"]) for item in batch], dtype=torch.float32),
        "track_router_target": torch.tensor([int(item["track_router_target"]) for item in batch], dtype=torch.long),
    }
    return result
