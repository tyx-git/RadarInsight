from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from torch.utils.data import Dataset

from .rd_physical_five_dataset import (
    RD_FIVE_LABELS,
    PhysicalRDFiveWindowDataset,
    collate_physical_rd_five_windows,
)
from .rd_physical_window_dataset import PhysicalRDWindowBatch, PhysicalRDWindowItem


SPECIALIST_LABELS = {
    "parent_gate": ("DBBGroup", "CUGroup"),
    "dbb": ("DroneTarget", "BirdTarget", "BalloonTarget"),
    "cu": ("ClutterTarget", "UnknownTarget"),
}
SPECIALIST_GLOBAL_INDICES = {
    "parent_gate": (0, 1),
    "dbb": (0, 1, 2),
    "cu": (3, 4),
}


def specialist_label_names(kind: str) -> tuple[str, ...]:
    try:
        return SPECIALIST_LABELS[str(kind)]
    except KeyError as error:
        raise ValueError(f"unknown RD specialist kind: {kind!r}") from error


def specialist_global_indices(kind: str) -> tuple[int, ...]:
    try:
        return SPECIALIST_GLOBAL_INDICES[str(kind)]
    except KeyError as error:
        raise ValueError(f"unknown RD specialist kind: {kind!r}") from error


def remap_specialist_label(kind: str, label_name: str) -> int:
    name = str(label_name)
    if kind == "parent_gate":
        if name in {"DroneTarget", "BirdTarget", "BalloonTarget"}:
            return 0
        if name in {"ClutterTarget", "UnknownTarget"}:
            return 1
    else:
        labels = specialist_label_names(kind)
        if name in labels:
            return labels.index(name)
    raise ValueError(f"label {name!r} is outside {kind} specialist scope")


class PhysicalRDSpecialistWindowDataset(Dataset[PhysicalRDWindowItem]):
    """A read-only label view over the strict five-class RD dataset.

    The underlying dataset still validates canonical RD completeness, split and
    no-fusion rows.  Only the target integer is remapped; IDs and provenance
    remain metadata for grouping and are never passed to a model.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        kind: str,
        index_name: str,
        split_name: str,
        normalization_stats_name: str,
        tensor_cache_size: int = 8,
    ) -> None:
        self.kind = str(kind)
        specialist_label_names(self.kind)
        self.base = PhysicalRDFiveWindowDataset(
            data_root,
            index_name=index_name,
            split_name=split_name,
            trainable_only=True,
            tensor_cache_size=tensor_cache_size,
            normalization_stats_name=normalization_stats_name,
            include_missing_rd=False,
            present_only=False,
        )
        if self.kind == "parent_gate":
            allowed = set(RD_FIVE_LABELS)
        else:
            allowed = set(specialist_label_names(self.kind))
        self.base.frame = self.base.frame[self.base.frame["label_name"].astype(str).isin(allowed)].reset_index(drop=True)
        if self.base.frame.empty:
            raise ValueError(f"RD specialist {self.kind} selected no rows")
        if self.base.frame["label_name"].astype(str).map(lambda value: remap_specialist_label(self.kind, value)).isna().any():
            raise ValueError(f"RD specialist {self.kind} contains an unmappable label")
        self.frame = self.base.frame
        self.label_names = specialist_label_names(self.kind)
        self.global_indices = specialist_global_indices(self.kind)

    @property
    def normalization_stats(self):
        return self.base.normalization_stats

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> PhysicalRDWindowItem:
        item = self.base[index]
        return replace(item, label=remap_specialist_label(self.kind, item.label_name))


def collate_physical_rd_specialist_windows(items: list[PhysicalRDWindowItem]) -> PhysicalRDWindowBatch:
    return collate_physical_rd_five_windows(items)


__all__ = [
    "SPECIALIST_GLOBAL_INDICES",
    "SPECIALIST_LABELS",
    "PhysicalRDSpecialistWindowDataset",
    "collate_physical_rd_specialist_windows",
    "remap_specialist_label",
    "specialist_global_indices",
    "specialist_label_names",
]
