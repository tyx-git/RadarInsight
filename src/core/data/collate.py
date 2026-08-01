from __future__ import annotations

from dataclasses import dataclass

import torch

from .torch_dataset import TorchTrackItem


@dataclass(frozen=True)
class TrackBatch:
    sample_ids: list[str]
    parent_sample_ids: list[str]
    source_names: list[str]
    label_names: list[str]
    coarse_label_names: list[str]
    fine_label_names: list[str]
    route_names: list[str]
    labels: torch.Tensor
    fine_labels: torch.Tensor
    route_labels: torch.Tensor
    features: torch.Tensor
    rd_features: torch.Tensor
    rd_patch_features: torch.Tensor
    rd_deep_patch_features: torch.Tensor
    rd_time_seconds: torch.Tensor
    rd_patch_time_seconds: torch.Tensor
    rd_deep_patch_time_seconds: torch.Tensor
    rd_lengths: torch.Tensor
    rd_patch_lengths: torch.Tensor
    rd_deep_patch_lengths: torch.Tensor
    rd_padding_mask: torch.Tensor
    rd_patch_padding_mask: torch.Tensor
    rd_deep_patch_padding_mask: torch.Tensor
    rd_is_multimodal: torch.Tensor
    rd_phase_missing: torch.Tensor
    rd_time_sync_status: list[str]
    window_start_seconds: torch.Tensor
    window_end_seconds: torch.Tensor
    track_max_time_gap_seconds: torch.Tensor
    track_numeric_finite_ratio: torch.Tensor
    height_missing: torch.Tensor
    physical_features: torch.Tensor
    lengths: torch.Tensor
    padding_mask: torch.Tensor


def collate_track_batch(
    items: list[TorchTrackItem],
    max_deep_patch_frames: int = 32,
) -> TrackBatch:
    if not items:
        raise ValueError("cannot collate empty batch")
    max_len = max(item.valid_length for item in items)
    feature_dim = items[0].features.shape[1]
    max_rd_len = max(item.rd_valid_length for item in items)
    max_rd_patch_len = max(item.rd_patch_valid_length for item in items)
    max_rd_deep_patch_len = max(item.rd_deep_patch_valid_length for item in items)
    if max_deep_patch_frames > 0 and max_rd_deep_patch_len > max_deep_patch_frames:
        max_rd_deep_patch_len = max_deep_patch_frames
    batch_size = len(items)

    features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    if items[0].rd_features.ndim == 3:
        rd_shape = items[0].rd_features.shape[1:]
        rd_features = torch.zeros((batch_size, max_rd_len, rd_shape[0], rd_shape[1]), dtype=torch.float32)
    else:
        raise ValueError("rd_features must be 3D per sample: [T, R, C]")
    if items[0].rd_patch_features.ndim == 4:
        rd_patch_shape = items[0].rd_patch_features.shape[1:]
        rd_patch_features = torch.zeros(
            (batch_size, max_rd_patch_len, rd_patch_shape[0], rd_patch_shape[1], rd_patch_shape[2]),
            dtype=torch.float32,
        )
    else:
        raise ValueError("rd_patch_features must be 4D per sample: [T, R, D, C]")
    if items[0].rd_deep_patch_features.ndim == 4:
        rd_deep_patch_shape = items[0].rd_deep_patch_features.shape[1:]
        rd_deep_patch_features = torch.zeros(
            (
                batch_size,
                max_rd_deep_patch_len,
                rd_deep_patch_shape[0],
                rd_deep_patch_shape[1],
                rd_deep_patch_shape[2],
            ),
            dtype=torch.float32,
        )
    else:
        raise ValueError("rd_deep_patch_features must be 4D per sample: [T, R, D, C]")
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    rd_padding_mask = torch.ones((batch_size, max_rd_len), dtype=torch.bool)
    rd_patch_padding_mask = torch.ones((batch_size, max_rd_patch_len), dtype=torch.bool)
    rd_deep_patch_padding_mask = torch.ones((batch_size, max_rd_deep_patch_len), dtype=torch.bool)
    rd_time_seconds = torch.zeros((batch_size, max_rd_len), dtype=torch.float32)
    rd_patch_time_seconds = torch.zeros((batch_size, max_rd_patch_len), dtype=torch.float32)
    rd_deep_patch_time_seconds = torch.zeros((batch_size, max_rd_deep_patch_len), dtype=torch.float32)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    fine_labels = torch.zeros((batch_size,), dtype=torch.long)
    route_labels = torch.zeros((batch_size,), dtype=torch.long)
    lengths = torch.zeros((batch_size,), dtype=torch.long)
    rd_lengths = torch.zeros((batch_size,), dtype=torch.long)
    rd_patch_lengths = torch.zeros((batch_size,), dtype=torch.long)
    rd_deep_patch_lengths = torch.zeros((batch_size,), dtype=torch.long)
    rd_is_multimodal = torch.zeros((batch_size,), dtype=torch.bool)
    rd_phase_missing = torch.zeros((batch_size,), dtype=torch.bool)
    height_missing = torch.zeros((batch_size,), dtype=torch.bool)
    physical_dim = int(items[0].physical_features.shape[0])
    physical_features = torch.zeros((batch_size, physical_dim), dtype=torch.float32)

    sample_ids: list[str] = []
    source_names: list[str] = []
    label_names: list[str] = []
    coarse_label_names: list[str] = []
    fine_label_names: list[str] = []
    route_names: list[str] = []
    rd_time_sync_status: list[str] = []
    parent_sample_ids: list[str] = []
    window_start_seconds = torch.zeros((batch_size,), dtype=torch.float32)
    window_end_seconds = torch.zeros((batch_size,), dtype=torch.float32)
    track_max_time_gap_seconds = torch.zeros((batch_size,), dtype=torch.float32)
    track_numeric_finite_ratio = torch.zeros((batch_size,), dtype=torch.float32)

    for idx, item in enumerate(items):
        seq_len = item.valid_length
        rd_len = item.rd_valid_length
        rd_patch_len = item.rd_patch_valid_length
        rd_deep_patch_len = item.rd_deep_patch_valid_length
        features[idx, :seq_len] = item.features
        rd_features[idx, :rd_len] = item.rd_features
        rd_patch_features[idx, :rd_patch_len] = item.rd_patch_features
        item_rd_times = item.rd_time_seconds if int(item.rd_time_seconds.numel()) == rd_len else torch.arange(rd_len, dtype=torch.float32)
        item_patch_times = (
            item.rd_patch_time_seconds
            if int(item.rd_patch_time_seconds.numel()) == rd_patch_len
            else torch.arange(rd_patch_len, dtype=torch.float32)
        )
        rd_time_seconds[idx, :rd_len] = item_rd_times
        rd_patch_time_seconds[idx, :rd_patch_len] = item_patch_times
        # Deep patch: subsample frames if source has more than max_deep_patch_frames
        if max_deep_patch_frames > 0 and rd_deep_patch_len > max_deep_patch_frames:
            step = max(1, rd_deep_patch_len // max_deep_patch_frames)
            src = item.rd_deep_patch_features[:rd_deep_patch_len:step][:max_deep_patch_frames]
            item_deep_times = (
                item.rd_deep_patch_time_seconds
                if int(item.rd_deep_patch_time_seconds.numel()) == rd_deep_patch_len
                else torch.arange(rd_deep_patch_len, dtype=torch.float32)
            )
            sub_len = src.shape[0]
            rd_deep_patch_features[idx, :sub_len] = src
            rd_deep_patch_time_seconds[idx, :sub_len] = item_deep_times[:rd_deep_patch_len:step][:max_deep_patch_frames]
            rd_deep_patch_lengths[idx] = sub_len
        else:
            rd_deep_patch_features[idx, :rd_deep_patch_len] = item.rd_deep_patch_features
            item_deep_times = (
                item.rd_deep_patch_time_seconds
                if int(item.rd_deep_patch_time_seconds.numel()) == rd_deep_patch_len
                else torch.arange(rd_deep_patch_len, dtype=torch.float32)
            )
            rd_deep_patch_time_seconds[idx, :rd_deep_patch_len] = item_deep_times
            rd_deep_patch_lengths[idx] = rd_deep_patch_len
        padding_mask[idx, :seq_len] = False
        rd_padding_mask[idx, :rd_len] = False
        rd_patch_padding_mask[idx, :rd_patch_len] = False
        rd_deep_patch_padding_mask[idx, :rd_deep_patch_lengths[idx]] = False
        labels[idx] = item.label_index
        fine_labels[idx] = item.fine_label_index
        route_labels[idx] = item.route_index
        rd_is_multimodal[idx] = item.rd_is_multimodal
        rd_phase_missing[idx] = item.rd_phase_missing
        height_missing[idx] = item.height_missing
        physical_features[idx] = item.physical_features
        lengths[idx] = seq_len
        rd_lengths[idx] = rd_len
        rd_patch_lengths[idx] = rd_patch_len
        sample_ids.append(item.sample_id)
        parent_sample_ids.append(item.parent_sample_id)
        source_names.append(item.source_name)
        label_names.append(item.label_name)
        coarse_label_names.append(item.coarse_label)
        fine_label_names.append(item.fine_label)
        route_names.append(item.route_name)
        rd_time_sync_status.append(item.rd_time_sync_status)
        window_start_seconds[idx] = item.window_start_seconds
        window_end_seconds[idx] = item.window_end_seconds
        track_max_time_gap_seconds[idx] = item.track_max_time_gap_seconds
        track_numeric_finite_ratio[idx] = item.track_numeric_finite_ratio

    return TrackBatch(
        sample_ids=sample_ids,
        parent_sample_ids=parent_sample_ids,
        source_names=source_names,
        label_names=label_names,
        coarse_label_names=coarse_label_names,
        fine_label_names=fine_label_names,
        route_names=route_names,
        labels=labels,
        fine_labels=fine_labels,
        route_labels=route_labels,
        features=features,
        rd_features=rd_features,
        rd_patch_features=rd_patch_features,
        rd_deep_patch_features=rd_deep_patch_features,
        rd_time_seconds=rd_time_seconds,
        rd_patch_time_seconds=rd_patch_time_seconds,
        rd_deep_patch_time_seconds=rd_deep_patch_time_seconds,
        rd_lengths=rd_lengths,
        rd_patch_lengths=rd_patch_lengths,
        rd_deep_patch_lengths=rd_deep_patch_lengths,
        rd_padding_mask=rd_padding_mask,
        rd_patch_padding_mask=rd_patch_padding_mask,
        rd_deep_patch_padding_mask=rd_deep_patch_padding_mask,
        rd_is_multimodal=rd_is_multimodal,
        rd_phase_missing=rd_phase_missing,
        rd_time_sync_status=rd_time_sync_status,
        window_start_seconds=window_start_seconds,
        window_end_seconds=window_end_seconds,
        track_max_time_gap_seconds=track_max_time_gap_seconds,
        track_numeric_finite_ratio=track_numeric_finite_ratio,
        height_missing=height_missing,
        physical_features=physical_features,
        lengths=lengths,
        padding_mask=padding_mask,
    )


__all__ = [
    "TrackBatch",
    "collate_track_batch",
]
