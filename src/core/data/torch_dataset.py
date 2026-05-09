from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .track_dataset import TrackDataset
from .features import encode_track_sequence, extract_physical_features, DEFAULT_VOCAB, FINE_VOCAB, load_rd_tensor, encode_multimodal_rd, load_rd_summary_frame, encode_rd_summary
from ..router import ROUTE_NAMES


@dataclass(frozen=True)
class TorchTrackItem:
    sample_id: str
    source_name: str
    label_name: str
    coarse_label: str
    fine_label: str
    label_index: int
    fine_label_index: int
    features: torch.Tensor
    rd_features: torch.Tensor
    rd_patch_features: torch.Tensor
    rd_deep_patch_features: torch.Tensor
    rd_valid_length: int
    rd_patch_valid_length: int
    rd_deep_patch_valid_length: int
    rd_is_multimodal: bool
    rd_phase_missing: bool
    height_missing: bool
    physical_features: torch.Tensor
    route_name: str
    route_index: int
    valid_length: int


def infer_route_name(label_name: str) -> str:
    if label_name in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}:
        return "low_altitude_suspect"
    if label_name in {"AircraftTarget"}:
        return "conventional_air"
    if label_name in {"PeopleTarget", "CarTarget", "ShipTarget"}:
        return "ground_target"
    return "high_altitude_filter"


class TorchTrackDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        sample_ids: Iterable[str] | None = None,
        trainable_only: bool = True,
        index_name: str | None = None,
        cache_size: int = 64,
    ) -> None:
        base_dataset = TrackDataset(data_root, trainable_only=trainable_only, index_name=index_name)
        index_df = base_dataset.to_frame()
        if sample_ids is not None:
            sample_id_set = {str(sample_id) for sample_id in sample_ids}
            index_df = index_df[index_df["sample_id"].astype(str).isin(sample_id_set)].copy()
        self.track_dataset = base_dataset
        self.index_df = index_df.reset_index(drop=True)
        self.samples_by_id = {sample.sample_id: sample for sample in self.track_dataset.iter_samples()}
        self.cache_size = int(max(0, cache_size))
        self._cache: OrderedDict[str, TorchTrackItem] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index_df)

    def _get_cached(self, sample_id: str) -> TorchTrackItem | None:
        if sample_id not in self._cache:
            return None
        item = self._cache.pop(sample_id)
        self._cache[sample_id] = item
        return item

    def _put_cached(self, sample_id: str, item: TorchTrackItem) -> None:
        if self.cache_size <= 0:
            return
        if sample_id in self._cache:
            self._cache.pop(sample_id)
        self._cache[sample_id] = item
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def __getitem__(self, index: int) -> TorchTrackItem:
        row = self.index_df.iloc[index]
        sample_id = str(row["sample_id"])
        cached = self._get_cached(sample_id)
        if cached is not None:
            return cached
        sample = self.samples_by_id.get(sample_id)
        if sample is None:
            raise KeyError(f"sample not found in dataset: {sample_id}")
        frame = sample.load_frame()
        encoded = encode_track_sequence(frame)
        physical = extract_physical_features(
            frame,
            height_missing=sample.height_missing,
            phase_missing=sample.rd_phase_missing,
        )
        route_name = infer_route_name(sample.label_name)
        rd_features = torch.zeros((1, 512, 4), dtype=torch.float32)
        rd_patch_features = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        rd_deep_patch_features = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        rd_valid_length = 1
        rd_patch_valid_length = 1
        rd_deep_patch_valid_length = 1
        rd_is_multimodal = False
        rd_tensor_path = str(sample.rd_tensor_path).strip()
        rd_summary_csv_path = str(sample.rd_summary_csv_path).strip()
        if rd_tensor_path and rd_tensor_path.lower() not in {"nan", "none", ""}:
            rd_tensor = load_rd_tensor(self.track_dataset.data_root, rd_tensor_path)
            rd_encoded = encode_multimodal_rd(rd_tensor)
            rd_features = rd_encoded.features
            rd_patch_features = (
                rd_encoded.patch_features
                if rd_encoded.patch_features is not None
                else torch.zeros((rd_encoded.valid_length, 64, 64, 3), dtype=torch.float32)
            )
            rd_deep_patch_features = (
                rd_encoded.deep_patch_features
                if rd_encoded.deep_patch_features is not None
                else rd_patch_features
            )
            rd_valid_length = rd_encoded.valid_length
            rd_patch_valid_length = rd_encoded.patch_valid_length or int(rd_patch_features.shape[0])
            rd_deep_patch_valid_length = rd_encoded.deep_patch_valid_length or int(rd_deep_patch_features.shape[0])
            rd_is_multimodal = True
        elif rd_summary_csv_path and rd_summary_csv_path.lower() not in {"nan", "none", ""}:
            rd_frame = load_rd_summary_frame(self.track_dataset.data_root, rd_summary_csv_path)
            rd_encoded = encode_rd_summary(rd_frame)
            summary_values = rd_encoded.features[:, :3]
            width = 512
            expanded = torch.zeros((summary_values.shape[0], width, 4), dtype=torch.float32)
            expanded[:, :, 0] = summary_values[:, :1].repeat(1, width)
            expanded[:, :, 1] = summary_values[:, 1:2].repeat(1, width)
            expanded[:, :, 2] = summary_values[:, 2:3].repeat(1, width)
            rd_features = expanded
            rd_patch_features = (
                rd_encoded.patch_features
                if rd_encoded.patch_features is not None
                else torch.zeros((rd_encoded.valid_length, 64, 64, 3), dtype=torch.float32)
            )
            rd_deep_patch_features = (
                rd_encoded.deep_patch_features
                if rd_encoded.deep_patch_features is not None
                else rd_patch_features
            )
            rd_valid_length = int(expanded.shape[0])
            rd_patch_valid_length = int(rd_patch_features.shape[0])
            rd_deep_patch_valid_length = int(rd_deep_patch_features.shape[0])
        item = TorchTrackItem(
            sample_id=sample.sample_id,
            source_name=sample.source_name,
            label_name=sample.label_name,
            coarse_label=sample.coarse_label,
            fine_label=sample.fine_label,
            label_index=DEFAULT_VOCAB.encode(sample.label_name),
            fine_label_index=FINE_VOCAB.encode(sample.fine_label),
            features=torch.from_numpy(encoded.features),
            rd_features=rd_features,
            rd_patch_features=rd_patch_features,
            rd_deep_patch_features=rd_deep_patch_features,
            rd_valid_length=rd_valid_length,
            rd_patch_valid_length=rd_patch_valid_length,
            rd_deep_patch_valid_length=rd_deep_patch_valid_length,
            rd_is_multimodal=rd_is_multimodal,
            rd_phase_missing=sample.rd_phase_missing,
            height_missing=sample.height_missing,
            physical_features=torch.from_numpy(physical.values),
            route_name=route_name,
            route_index=ROUTE_NAMES.index(route_name),
            valid_length=encoded.valid_length,
        )
        self._put_cached(sample_id, item)
        return item


__all__ = [
    "TorchTrackItem",
    "TorchTrackDataset",
    "infer_route_name",
]
