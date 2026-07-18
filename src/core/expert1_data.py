from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
import functools
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler

from .experts import LOW_ALTITUDE_CLASSES
from .forwardlayer.collate import collate_track_batch
from .forwardlayer.dataset import TrackDataset
from .forwardlayer.split import (
    SplitResult,
    canonical_group_id,
    stratified_group_split,
    validate_grouped_split,
)
from .forwardlayer.torch_dataset import TorchTrackDataset
from .forwardlayer.labels import DEFAULT_VOCAB


@dataclass(frozen=True)
class Expert1LoaderBundle:
    train_loader: DataLoader
    stage2_train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split: SplitResult
    train_sample_weights: list[float]


LOW_ALTITUDE_GLOBAL_TO_LOCAL = {
    DEFAULT_VOCAB.label_to_index[label]: idx
    for idx, label in enumerate(LOW_ALTITUDE_CLASSES)
}


class RDAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        rd_indices: list[int],
        no_rd_indices: list[int],
        sample_weights: list[float],
        batch_size: int,
        rd_fraction: float = 0.55,
        seed: int = 42,
    ) -> None:
        self.rd_indices = [int(item) for item in rd_indices]
        self.no_rd_indices = [int(item) for item in no_rd_indices]
        self.sample_weights = [float(item) for item in sample_weights]
        self.batch_size = int(max(1, batch_size))
        self.rd_fraction = float(min(1.0, max(0.0, rd_fraction)))
        self.seed = int(seed)
        if not self.rd_indices or not self.no_rd_indices:
            self.num_batches = 0
            return
        self.rd_weights = [max(1.0e-6, self.sample_weights[item]) for item in self.rd_indices]
        self.no_rd_weights = [max(1.0e-6, self.sample_weights[item]) for item in self.no_rd_indices]
        self.rd_per_batch = int(round(self.batch_size * self.rd_fraction))
        self.rd_per_batch = min(max(1, self.rd_per_batch), self.batch_size - 1)
        self.no_rd_per_batch = self.batch_size - self.rd_per_batch
        self.num_batches = max(
            1,
            int(np.ceil(max(
                len(self.rd_indices) / float(self.rd_per_batch),
                len(self.no_rd_indices) / float(self.no_rd_per_batch),
            ))),
        )

    def __iter__(self):
        if self.num_batches <= 0:
            return
        rng = random.Random(self.seed)
        for _ in range(self.num_batches):
            batch: list[int] = []
            batch.extend(rng.choices(self.rd_indices, weights=self.rd_weights, k=self.rd_per_batch))
            batch.extend(rng.choices(self.no_rd_indices, weights=self.no_rd_weights, k=self.no_rd_per_batch))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return int(self.num_batches)


def _weighted_sample_without_replacement(
    rng: random.Random,
    items: list[int],
    weights: list[float],
    k: int,
) -> list[int]:
    if k <= 0 or not items:
        return []
    pool = [
        (int(item), max(1.0e-6, float(weight)))
        for item, weight in zip(items, weights)
    ]
    selected: list[int] = []
    for _ in range(min(int(k), len(pool))):
        total = sum(weight for _item, weight in pool)
        threshold = rng.random() * total
        cumulative = 0.0
        chosen_index = len(pool) - 1
        for index, (_item, weight) in enumerate(pool):
            cumulative += weight
            if cumulative >= threshold:
                chosen_index = index
                break
        item, _weight = pool.pop(chosen_index)
        selected.append(item)
    return selected


class RDShardLocalBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        rd_indices: list[int],
        no_rd_indices: list[int],
        sample_weights: list[float],
        batch_size: int,
        rd_fraction: float = 0.55,
        seed: int = 42,
        shard_keys_by_index: dict[int, str] | None = None,
        shard_group_batches: int = 4,
    ) -> None:
        self.rd_indices = [int(item) for item in rd_indices]
        self.no_rd_indices = [int(item) for item in no_rd_indices]
        self.sample_weights = [float(item) for item in sample_weights]
        self.batch_size = int(max(1, batch_size))
        self.rd_fraction = float(min(1.0, max(0.0, rd_fraction)))
        self.seed = int(seed)
        self.shard_keys_by_index = dict(shard_keys_by_index or {})
        self.shard_group_batches = int(max(1, shard_group_batches))
        if not self.rd_indices or not self.no_rd_indices:
            self.num_batches = 0
            self._shard_keys: list[str] = []
            self._shard_weights: list[float] = []
            self._rd_by_shard: dict[str, list[int]] = {}
            return
        self.no_rd_weights = [max(1.0e-6, self.sample_weights[item]) for item in self.no_rd_indices]
        self.rd_per_batch = int(round(self.batch_size * self.rd_fraction))
        self.rd_per_batch = min(max(1, self.rd_per_batch), self.batch_size - 1)
        self.no_rd_per_batch = self.batch_size - self.rd_per_batch
        self.num_batches = max(
            1,
            int(np.ceil(max(
                len(self.rd_indices) / float(self.rd_per_batch),
                len(self.no_rd_indices) / float(self.no_rd_per_batch),
            ))),
        )
        rd_by_shard: dict[str, list[int]] = defaultdict(list)
        for index in self.rd_indices:
            shard_key = str(self.shard_keys_by_index.get(index) or f"sample:{index}")
            rd_by_shard[shard_key].append(index)
        self._rd_by_shard = dict(rd_by_shard)
        self._shard_keys = sorted(self._rd_by_shard)
        self._shard_weights = [
            sum(max(1.0e-6, self.sample_weights[index]) for index in self._rd_by_shard[shard_key])
            for shard_key in self._shard_keys
        ]

    def __iter__(self):
        if self.num_batches <= 0:
            return
        rng = random.Random(self.seed)
        # Build shard visit groups: each group keeps one shard active for
        # shard_group_batches consecutive batches to maximise LRU cache hits.
        shard_groups = [
            [shard_key] * self.shard_group_batches
            for shard_key in self._shard_keys
        ]
        yield_count = 0
        while yield_count < self.num_batches:
            groups = list(shard_groups)
            rng.shuffle(groups)
            for group in groups:
                for shard_key in group:
                    shard_indices = self._rd_by_shard[shard_key]
                    rd_weights = [max(1.0e-6, self.sample_weights[item]) for item in shard_indices]
                    batch: list[int] = []
                    if len(shard_indices) >= self.rd_per_batch:
                        batch.extend(_weighted_sample_without_replacement(rng, shard_indices, rd_weights, self.rd_per_batch))
                    else:
                        batch.extend(shard_indices)
                        remaining = self.rd_per_batch - len(batch)
                        if remaining > 0:
                            batch.extend(rng.choices(shard_indices, weights=rd_weights, k=remaining))
                    batch.extend(rng.choices(self.no_rd_indices, weights=self.no_rd_weights, k=self.no_rd_per_batch))
                    rng.shuffle(batch)
                    yield batch
                    yield_count += 1
                    if yield_count >= self.num_batches:
                        return

    def __len__(self) -> int:
        return int(self.num_batches)


def _row_has_rd(row: dict[str, object]) -> bool:
    for key in ("rd_tensor_path", "rd_summary_csv_path"):
        value = str(row.get(key, "")).strip().lower()
        if value not in {"", "nan", "none"}:
            return True
    return False


def _json_first_string(value: object) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, list) and payload:
        return str(payload[0])
    return ""


def _stage2_shard_keys_by_index(train_meta, data_root: Path | str, window_segment_map_name: str) -> dict[int, str]:
    data_root = Path(data_root)
    project_root = data_root.parent
    map_name = str(window_segment_map_name or "").strip()
    if not map_name:
        return {}
    map_path = Path(map_name)
    if not map_path.is_absolute():
        if map_name.replace("\\", "/").startswith("data/"):
            map_path = project_root / map_path
        else:
            map_path = data_root / "processed" / map_path
    if not map_path.exists():
        return {}
    map_df = pd.read_csv(map_path)
    if "sample_id" not in map_df.columns or "shard_keys_json" not in map_df.columns:
        return {}
    shard_by_sample = {
        str(row["sample_id"]): _json_first_string(row.get("shard_keys_json", ""))
        for row in map_df.to_dict(orient="records")
    }
    result: dict[int, str] = {}
    for row_index, row in train_meta.reset_index(drop=True).iterrows():
        sample_id = str(row.get("sample_id", ""))
        shard_key = shard_by_sample.get(sample_id, "")
        if not shard_key:
            shard_key = str(row.get("parent_sample_id") or sample_id)
        result[int(row_index)] = shard_key
    return result


def _sample_weight(
    row: dict[str, object],
    *,
    source_counts: dict[str, int],
    source_label_counts: dict[tuple[str, str], int],
    parent_window_counts: dict[str, int],
    max_source_count: int,
    max_source_label_count: int,
    rd_sample_boost: float,
    raw_source_boost: float,
    excel_source_boost: float,
    cq08_source_boost: float,
    source_balance_power: float,
    source_label_balance_power: float,
    parent_window_balance_power: float,
    unknown_sample_boost: float,
    clutter_sample_boost: float,
    bird_sample_boost: float,
    max_sampler_weight: float,
) -> float:
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
        parent_count = max(1, int(parent_window_counts.get(canonical_group_id(row), 1)))
        weight *= (1.0 / parent_count) ** float(parent_window_balance_power)
    if _row_has_rd(row):
        weight *= rd_sample_boost
    if label_name == "UnknownTarget":
        weight *= unknown_sample_boost
    if label_name == "ClutterTarget":
        weight *= clutter_sample_boost
    if label_name == "BirdTarget":
        weight *= bird_sample_boost
    return min(float(weight), float(max_sampler_weight))


def build_expert1_dataloaders(
    data_root: Path | str,
    index_name: str = "expert1_track_index.csv",
    batch_size: int = 8,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    eval_num_workers: int = -1,
    rd_sample_boost: float = 4.0,
    raw_source_boost: float = 2.0,
    excel_source_boost: float = 1.0,
    cq08_source_boost: float = 1.0,
    source_balance_power: float = 0.5,
    source_label_balance_power: float = 0.25,
    parent_window_balance_power: float = 0.5,
    stage2_rd_aware_sampling: bool = True,
    stage2_rd_batch_fraction: float = 0.55,
    stage2_rd_shard_local_sampling: bool = False,
    max_sampler_weight: float = 12.0,
    unknown_sample_boost: float = 1.0,
    clutter_sample_boost: float = 2.0,
    bird_sample_boost: float = 1.0,
    dataset_cache_size: int = 64,
    rd_encoded_cache_size: int = 256,
    enable_window_rd_time_crop: bool = True,
    rd_window_crop_min_frames: int = 2,
    use_rd_segment_store: bool = False,
    rd_segment_manifest_name: str = "expert1_rd_segment_manifest.csv",
    window_segment_map_name: str = "expert1_window_segment_map.csv",
    rd_segment_store_cache_size: int = 64,
    allow_rd_segment_store_fallback: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    validate_csv_files: bool = True,
) -> Expert1LoaderBundle:
    dataset = TrackDataset(data_root, trainable_only=False, index_name=index_name)
    frame = dataset.to_frame().copy()
    if "split_name" in frame.columns:
        persisted_split_labels = frame["split_name"].astype(str)
        invalid_split_labels = ~persisted_split_labels.isin({"train", "val", "test"})
        if invalid_split_labels.any():
            raise ValueError("persisted split_name contains malformed labels")
    frame = frame[frame["label_name"].astype(str).isin(LOW_ALTITUDE_CLASSES)].copy()
    if "is_trainable" in frame.columns:
        trainable_mask = frame["is_trainable"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        frame = frame[trainable_mask].copy()
    project_root = Path(data_root).parent
    if validate_csv_files:
        csv_exists = frame["csv_path"].astype(str).map(lambda rel: (project_root / rel).exists())
        frame = frame[csv_exists].copy()
    cq08_split_path = Path(data_root) / "manifests" / "cq08_grouped_split.json"
    if "split_name" in frame.columns:
        split_labels = frame["split_name"].astype(str)
        split = SplitResult(
            train_ids=sorted(frame[split_labels == "train"]["sample_id"].astype(str).tolist()),
            val_ids=sorted(frame[split_labels == "val"]["sample_id"].astype(str).tolist()),
            test_ids=sorted(frame[split_labels == "test"]["sample_id"].astype(str).tolist()),
        )
        validate_grouped_split(frame, split)
    elif index_name == "expert1_track_index.csv" and cq08_split_path.exists():
        payload = json.loads(cq08_split_path.read_text(encoding="utf-8"))
        cq08_df = frame[frame["source_name"].astype(str) == "cq08_track"].copy()
        other_df = frame[frame["source_name"].astype(str) != "cq08_track"].copy()
        cq08_train = cq08_df[cq08_df["sample_id"].astype(str).map(lambda x: x.split("|")[-1]).isin(payload["train_group_ids"])]["sample_id"].astype(str).tolist()
        cq08_val = cq08_df[cq08_df["sample_id"].astype(str).map(lambda x: x.split("|")[-1]).isin(payload["val_group_ids"])]["sample_id"].astype(str).tolist()
        cq08_test = cq08_df[cq08_df["sample_id"].astype(str).map(lambda x: x.split("|")[-1]).isin(payload["test_group_ids"])]["sample_id"].astype(str).tolist()
        other_split = stratified_group_split(
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
        split = stratified_group_split(
            frame,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    validate_grouped_split(frame, split)
    dataset_kwargs = {
        "trainable_only": False,
        "index_name": index_name,
        "cache_size": dataset_cache_size,
        "rd_cache_size": rd_encoded_cache_size,
        "enable_window_rd_time_crop": enable_window_rd_time_crop,
        "rd_window_crop_min_frames": rd_window_crop_min_frames,
        "use_rd_segment_store": use_rd_segment_store,
        "rd_segment_manifest_name": rd_segment_manifest_name,
        "window_segment_map_name": window_segment_map_name,
        "rd_segment_store_cache_size": rd_segment_store_cache_size,
        "allow_rd_segment_store_fallback": allow_rd_segment_store_fallback,
        "max_patch_frames": 32,
        "skip_deep_patch": True,
    }
    train_ds = TorchTrackDataset(data_root, sample_ids=split.train_ids, load_rd=False, **dataset_kwargs)
    stage2_train_ds = TorchTrackDataset(data_root, sample_ids=split.train_ids, load_rd=True, **dataset_kwargs)
    val_ds = TorchTrackDataset(data_root, sample_ids=split.val_ids, **dataset_kwargs)
    test_ds = TorchTrackDataset(data_root, sample_ids=split.test_ids, **dataset_kwargs)

    train_weights: list[float] = []
    train_meta = stage2_train_ds.index_df.reset_index(drop=True)
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
    train_rows = train_meta.to_dict(orient="records")
    parent_window_counts = dict(Counter(canonical_group_id(row) for row in train_rows))
    for row in train_rows:
        train_weights.append(
            _sample_weight(
                row,
                source_counts=source_counts,
                source_label_counts=source_label_counts,
                parent_window_counts=parent_window_counts,
                max_source_count=max_source_count,
                max_source_label_count=max_source_label_count,
                rd_sample_boost=rd_sample_boost,
                raw_source_boost=raw_source_boost,
                excel_source_boost=excel_source_boost,
                cq08_source_boost=cq08_source_boost,
                source_balance_power=source_balance_power,
                source_label_balance_power=source_label_balance_power,
                parent_window_balance_power=parent_window_balance_power,
                unknown_sample_boost=unknown_sample_boost,
                clutter_sample_boost=clutter_sample_boost,
                bird_sample_boost=bird_sample_boost,
                max_sampler_weight=max_sampler_weight,
            )
        )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(train_weights, dtype=torch.double),
        num_samples=len(train_weights),
        replacement=True,
    )

    worker_count = int(max(0, num_workers))
    eval_worker_count = worker_count if int(eval_num_workers) < 0 else int(max(0, eval_num_workers))
    _collate = functools.partial(collate_track_batch, max_deep_patch_frames=32)
    train_loader_kwargs = {
        "num_workers": worker_count,
        "collate_fn": _collate,
        "pin_memory": bool(pin_memory),
    }
    if worker_count > 0:
        train_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        train_loader_kwargs["prefetch_factor"] = int(max(1, prefetch_factor))
    eval_loader_kwargs = {
        "num_workers": eval_worker_count,
        "collate_fn": _collate,
        "pin_memory": bool(pin_memory),
    }
    if eval_worker_count > 0:
        eval_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        eval_loader_kwargs["prefetch_factor"] = int(max(1, prefetch_factor))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        **train_loader_kwargs,
    )
    stage2_base_train_loader = DataLoader(
        stage2_train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        **train_loader_kwargs,
    )
    rd_indices: list[int] = []
    no_rd_indices: list[int] = []
    for row_index, row in enumerate(train_rows):
        if _row_has_rd(row):
            rd_indices.append(row_index)
        else:
            no_rd_indices.append(row_index)
    if stage2_rd_aware_sampling and rd_indices and no_rd_indices:
        if stage2_rd_shard_local_sampling and use_rd_segment_store:
            stage2_batch_sampler = RDShardLocalBatchSampler(
                rd_indices=rd_indices,
                no_rd_indices=no_rd_indices,
                sample_weights=train_weights,
                batch_size=batch_size,
                rd_fraction=stage2_rd_batch_fraction,
                seed=seed + 17,
                shard_keys_by_index=_stage2_shard_keys_by_index(
                    train_meta,
                    data_root,
                    window_segment_map_name,
                ),
                shard_group_batches=4,
            )
        else:
            stage2_batch_sampler = RDAwareBatchSampler(
                rd_indices=rd_indices,
                no_rd_indices=no_rd_indices,
                sample_weights=train_weights,
                batch_size=batch_size,
                rd_fraction=stage2_rd_batch_fraction,
                seed=seed + 17,
            )
        stage2_train_loader = DataLoader(
            stage2_train_ds,
            batch_sampler=stage2_batch_sampler,
            **train_loader_kwargs,
        )
    else:
        stage2_train_loader = stage2_base_train_loader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )
    return Expert1LoaderBundle(
        train_loader=train_loader,
        stage2_train_loader=stage2_train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split=split,
        train_sample_weights=train_weights,
    )
