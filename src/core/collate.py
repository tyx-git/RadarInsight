from __future__ import annotations

from dataclasses import dataclass

import torch

from .torch_dataset import TorchTrackItem


@dataclass(frozen=True)
class TrackBatch:
    sample_ids: list[str]
    source_names: list[str]
    label_names: list[str]
    route_names: list[str]
    labels: torch.Tensor
    route_labels: torch.Tensor
    features: torch.Tensor
    rd_features: torch.Tensor
    rd_lengths: torch.Tensor
    rd_padding_mask: torch.Tensor
    physical_features: torch.Tensor
    lengths: torch.Tensor
    padding_mask: torch.Tensor


def collate_track_batch(items: list[TorchTrackItem]) -> TrackBatch:
    if not items:
        raise ValueError("cannot collate empty batch")
    max_len = max(item.valid_length for item in items)
    feature_dim = items[0].features.shape[1]
    max_rd_len = max(item.rd_valid_length for item in items)
    rd_feature_dim = items[0].rd_features.shape[1]
    batch_size = len(items)

    features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    rd_features = torch.zeros((batch_size, max_rd_len, rd_feature_dim), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    rd_padding_mask = torch.ones((batch_size, max_rd_len), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    route_labels = torch.zeros((batch_size,), dtype=torch.long)
    lengths = torch.zeros((batch_size,), dtype=torch.long)
    rd_lengths = torch.zeros((batch_size,), dtype=torch.long)
    physical_dim = int(items[0].physical_features.shape[0])
    physical_features = torch.zeros((batch_size, physical_dim), dtype=torch.float32)

    sample_ids: list[str] = []
    source_names: list[str] = []
    label_names: list[str] = []
    route_names: list[str] = []

    for idx, item in enumerate(items):
        seq_len = item.valid_length
        rd_len = item.rd_valid_length
        features[idx, :seq_len] = item.features
        rd_features[idx, :rd_len] = item.rd_features
        padding_mask[idx, :seq_len] = False
        rd_padding_mask[idx, :rd_len] = False
        labels[idx] = item.label_index
        route_labels[idx] = item.route_index
        physical_features[idx] = item.physical_features
        lengths[idx] = seq_len
        rd_lengths[idx] = rd_len
        sample_ids.append(item.sample_id)
        source_names.append(item.source_name)
        label_names.append(item.label_name)
        route_names.append(item.route_name)

    return TrackBatch(
        sample_ids=sample_ids,
        source_names=source_names,
        label_names=label_names,
        route_names=route_names,
        labels=labels,
        route_labels=route_labels,
        features=features,
        rd_features=rd_features,
        rd_lengths=rd_lengths,
        rd_padding_mask=rd_padding_mask,
        physical_features=physical_features,
        lengths=lengths,
        padding_mask=padding_mask,
    )
