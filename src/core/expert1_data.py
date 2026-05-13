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
    index_name: str = "expert1_track_index.csv",
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
    source_balance_power: float = 0.5,
    source_label_balance_power: float = 0.25,
    parent_window_balance_power: float = 0.5,
    max_sampler_weight: float = 12.0,
    unknown_sample_boost: float = 1.0,
    clutter_sample_boost: float = 2.0,
    bird_sample_boost: float = 1.0,
) -> Expert1LoaderBundle:
    dataset = TrackDataset(data_root, trainable_only=False, index_name=index_name)
    frame = dataset.to_frame().copy()
    frame = frame[frame["label_name"].astype(str).isin(LOW_ALTITUDE_CLASSES)].copy()
    if "is_trainable" in frame.columns:
        trainable_mask = frame["is_trainable"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        frame = frame[trainable_mask].copy()
    project_root = Path(data_root).parent
    csv_exists = frame["csv_path"].astype(str).map(lambda rel: (project_root / rel).exists())
    frame = frame[csv_exists].copy()
    cq08_split_path = Path(data_root) / "manifests" / "cq08_grouped_split.json"
    if "split_name" in frame.columns:
        split_frame = frame[frame["split_name"].astype(str).isin({"train", "val", "test"})].copy()
        split = SplitResult(
            train_ids=sorted(split_frame[split_frame["split_name"].astype(str) == "train"]["sample_id"].astype(str).tolist()),
            val_ids=sorted(split_frame[split_frame["split_name"].astype(str) == "val"]["sample_id"].astype(str).tolist()),
            test_ids=sorted(split_frame[split_frame["split_name"].astype(str) == "test"]["sample_id"].astype(str).tolist()),
        )
    elif index_name == "expert1_track_index.csv" and cq08_split_path.exists():
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
    train_ds = TorchTrackDataset(data_root, sample_ids=split.train_ids, trainable_only=False, index_name=index_name)
    val_ds = TorchTrackDataset(data_root, sample_ids=split.val_ids, trainable_only=False, index_name=index_name)
    test_ds = TorchTrackDataset(data_root, sample_ids=split.test_ids, trainable_only=False, index_name=index_name)

    train_weights = []
    train_meta = train_ds.index_df.reset_index(drop=True)
    source_counts = train_meta["source_name"].astype(str).value_counts().to_dict()
    source_label_counts = (
        train_meta.assign(
            source_name=train_meta["source_name"].astype(str),
            label_name=train_meta["label_name"].astype(str),
        )
        .groupby(["source_name", "label_name"])
        .size()
        .to_dict()
    )
    max_source_count = max(source_counts.values()) if source_counts else 1
    max_source_label_count = max(source_label_counts.values()) if source_label_counts else 1
    parent_window_counts = (
        train_meta["parent_sample_id"].astype(str).value_counts().to_dict()
        if "parent_sample_id" in train_meta.columns
        else {}
    )
    for row in train_meta.to_dict(orient="records"):
        weight = 1.0
        source_name = str(row.get("source_name", ""))
        label_name = str(row.get("label_name", ""))
        if source_name == "raw_tr_rd":
            weight *= raw_source_boost
        elif source_name == "excel_track":
            weight *= excel_source_boost
        elif source_name == "cq08_track":
            weight *= cq08_source_boost
        if source_balance_power > 0.0:
            source_count = max(1, int(source_counts.get(source_name, 1)))
            weight *= (max_source_count / source_count) ** float(source_balance_power)
        if source_label_balance_power > 0.0:
            source_label_count = max(1, int(source_label_counts.get((source_name, label_name), 1)))
            weight *= (max_source_label_count / source_label_count) ** float(source_label_balance_power)
        if parent_window_balance_power > 0.0 and parent_window_counts:
            parent_id = str(row.get("parent_sample_id", row.get("sample_id", "")))
            parent_count = max(1, int(parent_window_counts.get(parent_id, 1)))
            weight *= (1.0 / parent_count) ** float(parent_window_balance_power)
        rd_tensor_path = str(row.get("rd_tensor_path", "")).strip().lower()
        if rd_tensor_path not in {"", "nan", "none"}:
            weight *= rd_sample_boost
        if label_name == "UnknownTarget":
            weight *= unknown_sample_boost
        if label_name == "ClutterTarget":
            weight *= clutter_sample_boost
        if label_name == "BirdTarget":
            weight *= bird_sample_boost
        train_weights.append(min(float(weight), float(max_sampler_weight)))
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
