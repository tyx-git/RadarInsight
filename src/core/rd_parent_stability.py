from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from .data.collate import TrackBatch, collate_track_batch
from .data.torch_dataset import TorchTrackDataset, TorchTrackItem


REQUIRED_INDEX_COLUMNS = {
    "sample_id", "parent_sample_id", "split_name", "label_name",
    "window_start_seconds", "window_end_seconds",
}


@dataclass(frozen=True)
class ParentPredecessorLookup:
    predecessor_by_sample_id: Mapping[str, str]
    report: dict[str, object]


@dataclass(frozen=True)
class ParentStabilityItem:
    current: TorchTrackItem
    predecessor: TorchTrackItem | None
    predecessor_sample_id: str


@dataclass(frozen=True)
class ParentStabilityBatch:
    current: TrackBatch
    predecessor: TrackBatch | None
    predecessor_positions: torch.Tensor
    predecessor_sample_ids: list[str]

    @property
    def pair_count(self) -> int:
        return int(self.predecessor_positions.numel())


def build_parent_predecessor_lookup(index: pd.DataFrame, sample_ids: Sequence[str]) -> ParentPredecessorLookup:
    missing = REQUIRED_INDEX_COLUMNS.difference(index.columns)
    if missing:
        raise ValueError(f"parent predecessor index missing columns: {sorted(missing)}")
    selected_ids = {str(sample_id) for sample_id in sample_ids}
    selected = index[index["sample_id"].astype(str).isin(selected_ids)].copy()
    if len(selected) != len(selected_ids):
        present = set(selected["sample_id"].astype(str))
        raise ValueError(f"selected sample ids missing from index: {sorted(selected_ids.difference(present))[:5]}")
    if selected["sample_id"].astype(str).duplicated().any():
        raise ValueError("selected parent predecessor index contains duplicate sample_id")
    for column in ("window_start_seconds", "window_end_seconds"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    starts = selected["window_start_seconds"].to_numpy(dtype=np.float64)
    ends = selected["window_end_seconds"].to_numpy(dtype=np.float64)
    invalid_time = ~np.isfinite(starts) | ~np.isfinite(ends) | (ends < starts)
    if bool(invalid_time.any()):
        raise ValueError(f"parent predecessor index has {int(invalid_time.sum())} invalid windows")

    predecessor_by_sample_id: dict[str, str] = {}
    parent_sizes: list[int] = []
    for _, group in selected.groupby("parent_sample_id", sort=False):
        parent_sizes.append(len(group))
        if group["split_name"].astype(str).nunique() != 1:
            raise ValueError("parent predecessor index has cross-split parent")
        if group["label_name"].astype(str).nunique() != 1:
            raise ValueError("parent predecessor index has label-conflicted parent")
        ordered = group.sort_values(["window_end_seconds", "window_start_seconds", "sample_id"], kind="stable")
        ordered_ends = ordered["window_end_seconds"].to_numpy(dtype=np.float64)
        if np.any(np.diff(ordered_ends) <= 0.0):
            raise ValueError("parent predecessor index has non-strict window end order")
        ordered_ids = ordered["sample_id"].astype(str).tolist()
        predecessor_by_sample_id.update(dict(zip(ordered_ids[1:], ordered_ids[:-1])))
    report = {
        "selected_window_count": int(len(selected)), "parent_count": int(len(parent_sizes)),
        "pair_count": int(len(predecessor_by_sample_id)),
        "first_window_count": int(len(selected) - len(predecessor_by_sample_id)),
        "pair_coverage": float(len(predecessor_by_sample_id) / len(selected)) if len(selected) else 0.0,
        "single_window_parent_count": int(sum(size == 1 for size in parent_sizes)),
        "parent_window_count_min": int(min(parent_sizes)) if parent_sizes else 0,
        "parent_window_count_max": int(max(parent_sizes)) if parent_sizes else 0,
        "label_counts": dict(sorted(Counter(selected["label_name"].astype(str)).items())),
        "split_counts": dict(sorted(Counter(selected["split_name"].astype(str)).items())),
        "contract": {"same_parent_only": True, "same_split_only": True, "same_label_only": True, "strictly_historical_window_end": True},
    }
    return ParentPredecessorLookup(predecessor_by_sample_id=predecessor_by_sample_id, report=report)


class ParentStabilityDataset(Dataset[ParentStabilityItem]):
    def __init__(
        self,
        dataset: TorchTrackDataset,
        predecessor_by_sample_id: Mapping[str, str],
        current_sample_ids: Sequence[str],
    ) -> None:
        self.dataset = dataset
        self.predecessor_by_sample_id = dict(predecessor_by_sample_id)
        self.index_by_sample_id = {
            str(sample_id): position
            for position, sample_id in enumerate(dataset.index_df["sample_id"].astype(str))
        }
        unknown = set(self.predecessor_by_sample_id).union(self.predecessor_by_sample_id.values())
        missing = sorted(unknown.difference(self.index_by_sample_id))
        if missing:
            raise ValueError(f"parent predecessor samples missing from dataset: {missing[:5]}")
        self.current_positions = [
            self.index_by_sample_id[str(sample_id)]
            for sample_id in current_sample_ids
        ]
        if len(self.current_positions) != len(set(self.current_positions)):
            raise ValueError("parent stability current sample ids contain duplicates")

    def __len__(self) -> int:
        return len(self.current_positions)

    def __getitem__(self, position: int) -> ParentStabilityItem:
        current = self.dataset[self.current_positions[position]]
        predecessor_id = self.predecessor_by_sample_id.get(current.sample_id, "")
        if not predecessor_id:
            return ParentStabilityItem(current=current, predecessor=None, predecessor_sample_id="")
        predecessor = self.dataset[self.index_by_sample_id[predecessor_id]]
        if predecessor.parent_sample_id != current.parent_sample_id:
            raise ValueError("parent predecessor dataset returned cross-parent pair")
        if predecessor.label_name != current.label_name:
            raise ValueError("parent predecessor dataset returned label-mismatched pair")
        if predecessor.window_end_seconds >= current.window_end_seconds:
            raise ValueError("parent predecessor dataset returned non-historical pair")
        return ParentStabilityItem(
            current=current,
            predecessor=predecessor,
            predecessor_sample_id=predecessor_id,
        )


def collate_parent_stability_batch(items: list[ParentStabilityItem]) -> ParentStabilityBatch:
    if not items:
        raise ValueError("cannot collate empty parent stability batch")
    current = collate_track_batch([item.current for item in items])
    positions = [position for position, item in enumerate(items) if item.predecessor is not None]
    if not positions:
        return ParentStabilityBatch(
            current=current,
            predecessor=None,
            predecessor_positions=torch.zeros((0,), dtype=torch.long),
            predecessor_sample_ids=[],
        )
    predecessor_items = [items[position].predecessor for position in positions]
    if any(item is None for item in predecessor_items):
        raise RuntimeError("parent predecessor collation received invalid predecessor")
    predecessor = collate_track_batch([item for item in predecessor_items if item is not None])
    return ParentStabilityBatch(
        current=current,
        predecessor=predecessor,
        predecessor_positions=torch.tensor(positions, dtype=torch.long),
        predecessor_sample_ids=[items[position].predecessor_sample_id for position in positions],
    )


def parent_stability_aggregate_loss(
    current_logits: torch.Tensor,
    labels: torch.Tensor,
    predecessor_logits: torch.Tensor | None,
    predecessor_positions: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if predecessor_logits is None or predecessor_positions.numel() == 0:
        return current_logits.new_zeros(()), 0
    positions = predecessor_positions.to(device=current_logits.device, dtype=torch.long)
    if predecessor_logits.shape[0] != positions.numel():
        raise ValueError("predecessor logits and positions have different batch sizes")
    if positions.numel() and (int(positions.min()) < 0 or int(positions.max()) >= current_logits.shape[0]):
        raise ValueError("parent predecessor positions are outside current batch")
    aggregate_logits = 0.5 * (predecessor_logits.detach() + current_logits.index_select(0, positions))
    pair_labels = labels.index_select(0, positions)
    return nn.functional.cross_entropy(aggregate_logits, pair_labels), int(positions.numel())


__all__ = [
    "ParentPredecessorLookup", "ParentStabilityBatch", "ParentStabilityDataset",
    "ParentStabilityItem", "build_parent_predecessor_lookup",
    "collate_parent_stability_batch", "parent_stability_aggregate_loss",
]
