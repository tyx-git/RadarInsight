from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .forwardlayer.dataset import TrackDataset
from .encoding import encode_track_sequence


@dataclass(frozen=True)
class PretrainTrackItem:
    sample_id: str
    source_name: str
    original: torch.Tensor
    masked: torch.Tensor
    mask: torch.Tensor
    view_a: torch.Tensor
    view_b: torch.Tensor
    valid_length: int


def _augment_sequence(features: np.ndarray, noise_std: float = 0.01, dropout_ratio: float = 0.1) -> np.ndarray:
    augmented = features.copy()
    if augmented.size == 0:
        return augmented
    noise = np.random.normal(loc=0.0, scale=noise_std, size=augmented.shape).astype(np.float32)
    augmented = augmented + noise
    if augmented.shape[0] > 1:
        keep = np.random.rand(augmented.shape[0]) > dropout_ratio
        if not keep.any():
            keep[np.random.randint(0, augmented.shape[0])] = True
        augmented = augmented[keep]
    return augmented


class TrackPretrainDataset(Dataset):
    def __init__(self, data_root: Path | str, mask_ratio: float = 0.15) -> None:
        self.track_dataset = TrackDataset(data_root, trainable_only=False, index_name="system_trainable_track_index.csv")
        self.samples = list(self.track_dataset.iter_samples())
        self.mask_ratio = mask_ratio
        self.source_stats = self._fit_source_stats()

    def __len__(self) -> int:
        return len(self.samples)

    def _fit_source_stats(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        grouped: dict[str, list[np.ndarray]] = {}
        for sample in self.samples:
            frame = sample.load_frame()
            encoded = encode_track_sequence(frame)
            grouped.setdefault(sample.source_name, []).append(encoded.features)
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source_name, items in grouped.items():
            cat = np.concatenate(items, axis=0).astype(np.float32, copy=False)
            center = cat.mean(axis=0).astype(np.float32, copy=False)
            scale = cat.std(axis=0).astype(np.float32, copy=False)
            scale = np.clip(scale, a_min=1e-3, a_max=None)
            stats[source_name] = (center, scale)
        return stats

    def __getitem__(self, index: int) -> PretrainTrackItem:
        sample = self.samples[index]
        frame = sample.load_frame()
        encoded = encode_track_sequence(frame)
        features = encoded.features.astype(np.float32, copy=True)
        center, scale = self.source_stats[sample.source_name]
        features = (features - center) / scale
        masked = features.copy()
        mask = np.zeros((features.shape[0],), dtype=bool)
        if features.shape[0] > 0:
            num_mask = max(1, int(round(features.shape[0] * self.mask_ratio)))
            indices = np.random.choice(features.shape[0], size=min(num_mask, features.shape[0]), replace=False)
            mask[indices] = True
            masked[indices] = 0.0
        view_a = _augment_sequence(features)
        view_b = _augment_sequence(features)
        return PretrainTrackItem(
            sample_id=sample.sample_id,
            source_name=sample.source_name,
            original=torch.from_numpy(features),
            masked=torch.from_numpy(masked),
            mask=torch.from_numpy(mask),
            view_a=torch.from_numpy(view_a),
            view_b=torch.from_numpy(view_b),
            valid_length=int(features.shape[0]),
        )


@dataclass(frozen=True)
class PretrainTrackBatch:
    sample_ids: list[str]
    source_names: list[str]
    original: torch.Tensor
    masked: torch.Tensor
    mask: torch.Tensor
    view_a: torch.Tensor
    view_b: torch.Tensor
    lengths: torch.Tensor
    view_a_lengths: torch.Tensor
    view_b_lengths: torch.Tensor
    padding_mask: torch.Tensor
    view_a_padding_mask: torch.Tensor
    view_b_padding_mask: torch.Tensor


def collate_pretrain_track_batch(items: list[PretrainTrackItem]) -> PretrainTrackBatch:
    max_len = max(item.valid_length for item in items)
    feat_dim = items[0].original.shape[1]
    max_a = max(item.view_a.shape[0] for item in items)
    max_b = max(item.view_b.shape[0] for item in items)
    batch = len(items)

    original = torch.zeros((batch, max_len, feat_dim), dtype=torch.float32)
    masked = torch.zeros((batch, max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch, max_len), dtype=torch.bool)
    padding_mask = torch.ones((batch, max_len), dtype=torch.bool)
    lengths = torch.zeros((batch,), dtype=torch.long)

    view_a = torch.zeros((batch, max_a, feat_dim), dtype=torch.float32)
    view_b = torch.zeros((batch, max_b, feat_dim), dtype=torch.float32)
    view_a_padding_mask = torch.ones((batch, max_a), dtype=torch.bool)
    view_b_padding_mask = torch.ones((batch, max_b), dtype=torch.bool)
    view_a_lengths = torch.zeros((batch,), dtype=torch.long)
    view_b_lengths = torch.zeros((batch,), dtype=torch.long)

    sample_ids: list[str] = []
    source_names: list[str] = []

    for idx, item in enumerate(items):
        l = item.valid_length
        original[idx, :l] = item.original
        masked[idx, :l] = item.masked
        mask[idx, :l] = item.mask
        padding_mask[idx, :l] = False
        lengths[idx] = l

        la = item.view_a.shape[0]
        lb = item.view_b.shape[0]
        view_a[idx, :la] = item.view_a
        view_b[idx, :lb] = item.view_b
        view_a_padding_mask[idx, :la] = False
        view_b_padding_mask[idx, :lb] = False
        view_a_lengths[idx] = la
        view_b_lengths[idx] = lb
        sample_ids.append(item.sample_id)
        source_names.append(item.source_name)

    return PretrainTrackBatch(
        sample_ids=sample_ids,
        source_names=source_names,
        original=original,
        masked=masked,
        mask=mask,
        view_a=view_a,
        view_b=view_b,
        lengths=lengths,
        view_a_lengths=view_a_lengths,
        view_b_lengths=view_b_lengths,
        padding_mask=padding_mask,
        view_a_padding_mask=view_a_padding_mask,
        view_b_padding_mask=view_b_padding_mask,
    )


def build_pretrain_track_loader(
    data_root: Path | str,
    batch_size: int = 32,
    num_workers: int = 0,
    mask_ratio: float = 0.15,
) -> DataLoader:
    dataset = TrackPretrainDataset(data_root, mask_ratio=mask_ratio)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pretrain_track_batch,
    )
