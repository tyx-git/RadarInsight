from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from .forwardlayer.collate import collate_track_batch
from .forwardlayer.dataset import TrackDataset
from .forwardlayer.labels import DEFAULT_VOCAB
from .forwardlayer.split import SplitResult, stratified_split
from .forwardlayer.torch_dataset import TorchTrackDataset


@dataclass(frozen=True)
class ExpertOtherLoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split: SplitResult
    labels: list[str]
    global_to_local: dict[int, int]


def build_subset_dataloaders(
    data_root: Path | str,
    labels: list[str],
    batch_size: int = 16,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
) -> ExpertOtherLoaderBundle:
    dataset = TrackDataset(data_root, trainable_only=False, index_name="system_trainable_track_index.csv")
    frame = dataset.to_frame().copy()
    if labels == ["AircraftTarget"]:
        frame = frame[
            frame["label_name"].astype(str).isin(
                [
                    "AircraftTarget",
                    "DroneTarget",
                    "BirdTarget",
                    "BalloonTarget",
                    "ClutterTarget",
                    "UnknownTarget",
                    "PeopleTarget",
                    "CarTarget",
                    "ShipTarget",
                ]
            )
        ].copy()
    else:
        frame = frame[frame["label_name"].astype(str).isin(labels)].copy()
    project_root = Path(data_root).parent
    csv_exists = frame["csv_path"].astype(str).map(lambda rel: (project_root / rel).exists())
    frame = frame[csv_exists].copy()
    split = stratified_split(
        frame,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_ds = TorchTrackDataset(data_root, sample_ids=split.train_ids, trainable_only=False, index_name="system_trainable_track_index.csv")
    val_ds = TorchTrackDataset(data_root, sample_ids=split.val_ids, trainable_only=False, index_name="system_trainable_track_index.csv")
    test_ds = TorchTrackDataset(data_root, sample_ids=split.test_ids, trainable_only=False, index_name="system_trainable_track_index.csv")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_track_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_track_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_track_batch,
    )
    if labels == ["AircraftTarget"]:
        global_to_local = {
            DEFAULT_VOCAB.label_to_index["AircraftTarget"]: 1,
        }
        output_labels = ["NonAircraftTarget", "AircraftTarget"]
    else:
        global_to_local = {
            DEFAULT_VOCAB.label_to_index[label]: idx
            for idx, label in enumerate(labels)
        }
        output_labels = labels
    return ExpertOtherLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split=split,
        labels=output_labels,
        global_to_local=global_to_local,
    )
