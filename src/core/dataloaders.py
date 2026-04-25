from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from .collate import collate_track_batch
from .dataset import TrackDataset
from .split import SplitResult, stratified_split
from .torch_dataset import TorchTrackDataset
from pathlib import Path


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split: SplitResult


def build_dataloaders(
    data_root: Path | str,
    batch_size: int = 16,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
) -> LoaderBundle:
    dataset = TrackDataset(data_root, trainable_only=True)
    project_root = Path(data_root).parent
    frame = dataset.to_frame()
    csv_exists = frame["csv_path"].astype(str).map(lambda rel: (project_root / rel).exists())
    if not bool(csv_exists.all()):
        frame = frame[csv_exists].copy()
    split = stratified_split(
        frame,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_ds = TorchTrackDataset(data_root, sample_ids=split.train_ids, trainable_only=True)
    val_ds = TorchTrackDataset(data_root, sample_ids=split.val_ids, trainable_only=True)
    test_ds = TorchTrackDataset(data_root, sample_ids=split.test_ids, trainable_only=True)

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
    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split=split,
    )
