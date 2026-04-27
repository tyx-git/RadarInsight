from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from .experts import LOW_ALTITUDE_CLASSES
from .forwardlayer.collate import collate_track_batch
from .forwardlayer.dataset import TrackDataset
from .forwardlayer.split import SplitResult, stratified_split
from .forwardlayer.torch_dataset import TorchTrackDataset
from .forwardlayer.labels import DEFAULT_VOCAB


@dataclass(frozen=True)
class Expert1LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split: SplitResult


LOW_ALTITUDE_GLOBAL_TO_LOCAL = {
    DEFAULT_VOCAB.label_to_index[label]: idx
    for idx, label in enumerate(LOW_ALTITUDE_CLASSES)
}


def build_expert1_dataloaders(
    data_root: Path | str,
    batch_size: int = 8,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    rd_sample_boost: float = 4.0,
    raw_source_boost: float = 2.0,
    excel_source_boost: float = 1.0,
    cq08_source_boost: float = 1.0,
    unknown_sample_boost: float = 1.0,
    clutter_sample_boost: float = 2.0,
) -> Expert1LoaderBundle:
    dataset = TrackDataset(data_root, trainable_only=False, index_name="expert1_track_index.csv")
    frame = dataset.to_frame().copy()
    frame = frame[frame["label_name"].astype(str).isin(LOW_ALTITUDE_CLASSES)].copy()
    project_root = Path(data_root).parent
    csv_exists = frame["csv_path"].astype(str).map(lambda rel: (project_root / rel).exists())
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
    train_ds = TorchTrackDataset(data_root, sample_ids=split.train_ids, trainable_only=False, index_name="expert1_track_index.csv")
    val_ds = TorchTrackDataset(data_root, sample_ids=split.val_ids, trainable_only=False, index_name="expert1_track_index.csv")
    test_ds = TorchTrackDataset(data_root, sample_ids=split.test_ids, trainable_only=False, index_name="expert1_track_index.csv")

    train_weights = []
    train_meta = train_ds.index_df.reset_index(drop=True)
    for row in train_meta.to_dict(orient="records"):
        weight = 1.0
        source_name = str(row.get("source_name", ""))
        if source_name == "raw_tr_rd":
            weight *= raw_source_boost
        elif source_name == "excel_track":
            weight *= excel_source_boost
        elif source_name == "cq08_track":
            weight *= cq08_source_boost
        rd_tensor_path = str(row.get("rd_tensor_path", "")).strip().lower()
        if rd_tensor_path not in {"", "nan", "none"}:
            weight *= rd_sample_boost
        if str(row.get("label_name", "")) == "UnknownTarget":
            weight *= unknown_sample_boost
        if str(row.get("label_name", "")) == "ClutterTarget":
            weight *= clutter_sample_boost
        train_weights.append(weight)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(train_weights, dtype=torch.double),
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
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
    return Expert1LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split=split,
    )
