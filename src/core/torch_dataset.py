from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .dataset import TrackDataset
from .encoding import encode_track_sequence
from .labels import DEFAULT_VOCAB
from .physical import extract_physical_features
from .rd import encode_rd_summary, load_rd_summary_frame
from .router import ROUTE_NAMES


@dataclass(frozen=True)
class TorchTrackItem:
    sample_id: str
    source_name: str
    label_name: str
    label_index: int
    features: torch.Tensor
    rd_features: torch.Tensor
    rd_valid_length: int
    physical_features: torch.Tensor
    route_name: str
    route_index: int
    valid_length: int


def infer_route_name(label_name: str) -> str:
    if label_name in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"}:
        return "low_altitude_suspect"
    if label_name in {"AircraftTarget"}:
        return "conventional_air"
    if label_name in {"PeopleTarget", "CarTarget", "ShipTarget"}:
        return "ground_target"
    return "high_altitude_filter"


class TorchTrackDataset(Dataset):
    def __init__(self, data_root: Path | str, sample_ids: Iterable[str] | None = None, trainable_only: bool = True) -> None:
        base_dataset = TrackDataset(data_root, trainable_only=trainable_only)
        index_df = base_dataset.to_frame()
        if sample_ids is not None:
            sample_id_set = {str(sample_id) for sample_id in sample_ids}
            index_df = index_df[index_df["sample_id"].astype(str).isin(sample_id_set)].copy()
        self.track_dataset = base_dataset
        self.index_df = index_df.reset_index(drop=True)
        self.samples_by_id = {sample.sample_id: sample for sample in self.track_dataset.iter_samples()}

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, index: int) -> TorchTrackItem:
        row = self.index_df.iloc[index]
        sample_id = str(row["sample_id"])
        sample = self.samples_by_id.get(sample_id)
        if sample is None:
            raise KeyError(f"sample not found in dataset: {sample_id}")
        frame = sample.load_frame()
        encoded = encode_track_sequence(frame)
        physical = extract_physical_features(frame)
        route_name = infer_route_name(sample.label_name)
        rd_features = torch.zeros((1, 4), dtype=torch.float32)
        rd_valid_length = 1
        rd_summary_csv_path = str(sample.rd_summary_csv_path).strip()
        if rd_summary_csv_path and rd_summary_csv_path.lower() not in {"nan", "none", ""}:
            rd_frame = load_rd_summary_frame(self.track_dataset.data_root, rd_summary_csv_path)
            rd_encoded = encode_rd_summary(rd_frame)
            rd_features = rd_encoded.features
            rd_valid_length = rd_encoded.valid_length
        return TorchTrackItem(
            sample_id=sample.sample_id,
            source_name=sample.source_name,
            label_name=sample.label_name,
            label_index=DEFAULT_VOCAB.encode(sample.label_name),
            features=torch.from_numpy(encoded.features),
            rd_features=rd_features,
            rd_valid_length=rd_valid_length,
            physical_features=torch.from_numpy(physical.values),
            route_name=route_name,
            route_index=ROUTE_NAMES.index(route_name),
            valid_length=encoded.valid_length,
        )
