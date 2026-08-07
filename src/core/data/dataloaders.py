from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from torch.utils.data import DataLoader

from .collate import collate_track_batch
from .track_dataset import TrackDataset
from .split import SplitResult, stratified_split
from .torch_dataset import TorchTrackDataset


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
    trajectory_feature_schema: str = "v1",
    physical_representation_contract_version: str = "legacy_v2",
) -> LoaderBundle:
    dataset = TrackDataset(data_root, trainable_only=False, index_name="system_trainable_track_index.csv")
    project_root = Path(data_root).parent
    frame = dataset.to_frame()
    csv_exists = frame["csv_path"].astype(str).map(lambda rel: (project_root / rel).exists())
    if not bool(csv_exists.all()):
        frame = frame[csv_exists].copy()
    cq08_split_path = Path(data_root) / "manifests" / "cq08_grouped_split.json"
    if cq08_split_path.exists():
        payload = json.loads(cq08_split_path.read_text(encoding="utf-8"))
        cq08_df = frame[frame["source_name"].astype(str) == "cq08_track"].copy()
        other_df = frame[frame["source_name"].astype(str) != "cq08_track"].copy()
        cq08_train = cq08_df[cq08_df["sample_id"].astype(str).map(lambda x: x.split("|")[-1]).isin(payload["train_group_ids"])]["sample_id"].astype(str).tolist()
        cq08_val = cq08_df[cq08_df["sample_id"].astype(str).map(lambda x: x.split("|")[-1]).isin(payload["val_group_ids"])]["sample_id"].astype(str).tolist()
        cq08_test = cq08_df[cq08_df["sample_id"].astype(str).map(lambda x: x.split("|")[-1]).isin(payload["test_group_ids"])]["sample_id"].astype(str).tolist()
        other_split = stratified_split(
            other_df,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        split = SplitResult(
            train_ids=sorted(cq08_train + other_split.train_ids),
            val_ids=sorted(cq08_val + other_split.val_ids),
            test_ids=sorted(cq08_test + other_split.test_ids),
        )
    else:
        split = stratified_split(
            frame,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    train_ds = TorchTrackDataset(
        data_root,
        sample_ids=split.train_ids,
        trainable_only=False,
        index_name="system_trainable_track_index.csv",
        trajectory_feature_schema=trajectory_feature_schema,
        physical_representation_contract_version=physical_representation_contract_version,
    )
    val_ds = TorchTrackDataset(
        data_root,
        sample_ids=split.val_ids,
        trainable_only=False,
        index_name="system_trainable_track_index.csv",
        trajectory_feature_schema=trajectory_feature_schema,
        physical_representation_contract_version=physical_representation_contract_version,
    )
    test_ds = TorchTrackDataset(
        data_root,
        sample_ids=split.test_ids,
        trainable_only=False,
        index_name="system_trainable_track_index.csv",
        trajectory_feature_schema=trajectory_feature_schema,
        physical_representation_contract_version=physical_representation_contract_version,
    )

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


__all__ = [
    "LoaderBundle",
    "build_dataloaders",
]
