from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data.rd import MULTIMODAL_RD_RANGE_BINS, EncodedRDSequence, encode_multimodal_rd
from .data.rd_physical_window_dataset import encode_rd_drone_bird_label


RD_ONLY_LABELS = ("DroneTarget", "BirdTarget")
NOVEL_UPDATE_SCHEMA_VERSION = "expert1_cq08_physical_novel_update_v2_exact_once"
REQUIRED_INDEX_COLUMNS = {
    "sample_id",
    "parent_sample_id",
    "split_name",
    "label_name",
    "update_index",
    "chunk_index",
    "chunk_count",
    "is_final_chunk",
    "is_zero_novel_update",
    "delta_time_seconds",
    "chunk_frame_count",
    "rd_tensor_path",
    "chunk_tensor_frame_indices_json",
}


@dataclass(frozen=True)
class NovelParentChunk:
    update_index: int
    chunk_index: int
    is_final_chunk: bool
    is_zero_novel_update: bool
    delta_time_seconds: float
    features: torch.Tensor
    frame_time_seconds: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.float32))
    anchor_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class NovelParentItem:
    parent_sample_id: str
    label_name: str
    chunks: tuple[NovelParentChunk, ...]


@dataclass(frozen=True)
class NovelParentBatch:
    features: torch.Tensor
    frame_padding_mask: torch.Tensor
    chunk_padding_mask: torch.Tensor
    is_final_chunk: torch.Tensor
    is_zero_novel_update: torch.Tensor
    delta_time_seconds: torch.Tensor
    frame_time_seconds: torch.Tensor
    labels: torch.Tensor
    parent_sample_ids: list[str]
    anchor_logits: torch.Tensor | None = None


def _parse_indices(value: object, *, sample_id: str) -> list[int]:
    if not isinstance(value, str):
        raise ValueError(f"chunk_tensor_frame_indices_json is not text for {sample_id}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid chunk_tensor_frame_indices_json for {sample_id}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"chunk_tensor_frame_indices_json is not a list for {sample_id}")
    indices = [int(item) for item in parsed]
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices) or any(
        current <= previous for previous, current in zip(indices, indices[1:])
    ):
        raise ValueError(f"chunk tensor frame indices must be unique and strictly increasing for {sample_id}")
    return indices


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _encode_incremental_physical_sequence(
    amplitude: np.ndarray,
    phase: np.ndarray,
    time_seconds: np.ndarray | None = None,
) -> EncodedRDSequence:
    amplitude_array = np.asarray(amplitude, dtype=np.float32)
    phase_array = np.asarray(phase, dtype=np.float32)
    if amplitude_array.ndim != 3 or phase_array.ndim != 3 or amplitude_array.shape != phase_array.shape:
        raise ValueError("physical RD chunk must contain matching [frame, range, doppler] amplitude and phase")
    if amplitude_array.shape[0] == 0:
        empty = torch.zeros((0, MULTIMODAL_RD_RANGE_BINS, 4), dtype=torch.float32)
        return EncodedRDSequence(features=empty, valid_length=0, time_seconds=torch.zeros((0,), dtype=torch.float32))
    payload: dict[str, np.ndarray] = {
        "amplitude": amplitude_array,
        "phase": phase_array,
        "time_raw": (
            np.asarray(time_seconds, dtype=np.float32)
            if time_seconds is not None
            else np.arange(amplitude_array.shape[0], dtype=np.float32)
        ),
    }
    return encode_multimodal_rd(payload, legacy=True)


def encode_incremental_physical_frames(
    amplitude: np.ndarray,
    phase: np.ndarray,
    time_seconds: np.ndarray | None = None,
) -> torch.Tensor:
    return _encode_incremental_physical_sequence(amplitude, phase, time_seconds).features


class NovelParentSequenceDataset(Dataset[NovelParentItem]):
    def __init__(
        self,
        data_root: str | Path,
        *,
        index_name: str = "expert1_cq08_rd_physical_novel_update_index_v2.csv",
        split_name: str,
        anchor_cache_name: str | None = None,
        tensor_cache_size: int = 8,
    ) -> None:
        self.data_root = Path(data_root)
        self.processed_root = self.data_root / "data" / "processed" if (self.data_root / "data" / "processed").exists() else self.data_root / "processed"
        self.index_path = self.processed_root / index_name
        self.index_frame = pd.read_csv(self.index_path, low_memory=False)
        if set(self.index_frame["schema_version"].astype(str).unique()) != {NOVEL_UPDATE_SCHEMA_VERSION}:
            raise ValueError("novel parent dataset requires the exact-once novel update index schema")
        missing_columns = sorted(REQUIRED_INDEX_COLUMNS.difference(self.index_frame.columns))
        if missing_columns:
            raise ValueError(f"novel update index missing columns: {missing_columns}")
        if split_name not in {"train", "val"}:
            raise ValueError("novel parent dataset only permits train or val")
        self.index_frame = self.index_frame[
            self.index_frame["split_name"].astype(str).eq(split_name)
            & self.index_frame["label_name"].astype(str).isin(RD_ONLY_LABELS)
        ].copy()
        if self.index_frame.empty:
            raise ValueError(f"novel parent dataset has no rows for split={split_name}")
        self.parent_sample_ids = sorted(self.index_frame["parent_sample_id"].astype(str).unique().tolist())
        self.rows_by_parent = {
            parent_sample_id: self.index_frame[
                self.index_frame["parent_sample_id"].astype(str).eq(parent_sample_id)
            ].sort_values(["update_index", "chunk_index"], kind="stable")
            for parent_sample_id in self.parent_sample_ids
        }
        self.tensor_cache_size = max(1, int(tensor_cache_size))
        self.tensor_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self.anchor_logits_by_parent_update = self._load_anchor_cache(anchor_cache_name)
        self._validate_parent_rows()

    def _load_anchor_cache(self, anchor_cache_name: str | None) -> dict[tuple[str, int], torch.Tensor] | None:
        if not anchor_cache_name:
            return None
        raise ValueError(
            "legacy RD anchor logit caches are retired and incompatible with the exact-once Physical RD contract"
        )

    def _validate_parent_rows(self) -> None:
        for parent_sample_id, parent_rows in self.rows_by_parent.items():
            if parent_rows["label_name"].astype(str).nunique() != 1:
                raise ValueError(f"parent has label conflict: {parent_sample_id}")
            update_indices = sorted(parent_rows["update_index"].astype(int).unique().tolist())
            if update_indices != list(range(len(update_indices))):
                raise ValueError(f"parent updates are non-contiguous: {parent_sample_id}")
            for update_index, update_rows in parent_rows.groupby("update_index", sort=False):
                ordered_rows = update_rows.sort_values("chunk_index", kind="stable")
                chunk_indices = ordered_rows["chunk_index"].astype(int).tolist()
                if chunk_indices != list(range(len(chunk_indices))):
                    raise ValueError(f"parent update has non-contiguous chunks: {parent_sample_id}:{update_index}")
                declared_chunk_counts = ordered_rows["chunk_count"].astype(int).unique().tolist()
                if declared_chunk_counts != [len(ordered_rows)]:
                    raise ValueError(f"parent update has invalid chunk_count: {parent_sample_id}:{update_index}")
                final_count = int(ordered_rows["is_final_chunk"].map(_as_bool).sum())
                if final_count != 1 or not _as_bool(ordered_rows.iloc[-1]["is_final_chunk"]):
                    raise ValueError(f"parent update has invalid final chunk: {parent_sample_id}:{update_index}")
                delta_values = [float(value) for value in ordered_rows["delta_time_seconds"].tolist()]
                if not np.isfinite(delta_values).all() or any(value < 0.0 for value in delta_values):
                    raise ValueError(f"novel update delta time is invalid: {parent_sample_id}:{update_index}")
                if any(abs(value) > 1.0e-9 for value in delta_values[1:]):
                    raise ValueError(f"novel update delta time repeats across chunks: {parent_sample_id}:{update_index}")
                frame_counts = ordered_rows["chunk_frame_count"].astype(int).tolist()
                if any(count < 0 for count in frame_counts):
                    raise ValueError(f"parent update has negative chunk frame count: {parent_sample_id}:{update_index}")
                zero_markers = ordered_rows["is_zero_novel_update"].map(_as_bool).tolist()
                if any(marker != (count == 0) for marker, count in zip(zero_markers, frame_counts)):
                    raise ValueError(f"parent update zero-novel marker is inconsistent: {parent_sample_id}:{update_index}")
            observed_frames: set[int] = set()
            for row in parent_rows.itertuples(index=False):
                frame_indices = _parse_indices(row.chunk_tensor_frame_indices_json, sample_id=str(row.sample_id))
                repeated = observed_frames.intersection(frame_indices)
                if repeated:
                    raise ValueError(f"novel parent reuses physical frames: {parent_sample_id}:{sorted(repeated)[:3]}")
                observed_frames.update(frame_indices)

    def __len__(self) -> int:
        return len(self.parent_sample_ids)

    def _load_tensor(self, tensor_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self.tensor_cache.get(tensor_path)
        if cached is not None:
            self.tensor_cache.move_to_end(tensor_path)
            return cached
        resolved_path = Path(tensor_path)
        if not resolved_path.is_absolute():
            resolved_path = self.data_root / resolved_path
        with np.load(resolved_path, allow_pickle=False) as payload:
            amplitude = np.asarray(payload["amplitude"], dtype=np.float32)
            phase = np.asarray(payload["phase"], dtype=np.float32)
            if "time_seconds" in payload.files:
                time_seconds = np.asarray(payload["time_seconds"], dtype=np.float32)
            elif "time_raw" in payload.files:
                time_seconds = np.asarray(payload["time_raw"], dtype=np.float32)
            else:
                time_seconds = np.arange(amplitude.shape[0], dtype=np.float32)
        if amplitude.ndim != 3 or phase.shape != amplitude.shape:
            raise ValueError(f"invalid novel parent physical tensor shape: {resolved_path}")
        if time_seconds.ndim != 1 or int(time_seconds.size) != int(amplitude.shape[0]):
            raise ValueError(f"invalid novel parent physical tensor time axis: {resolved_path}")
        if not np.isfinite(amplitude).all() or not np.isfinite(phase).all() or not np.isfinite(time_seconds).all():
            raise ValueError(f"novel parent physical tensor contains non-finite values: {resolved_path}")
        if (np.diff(time_seconds) <= 1.0e-9).any():
            raise ValueError(f"novel parent physical tensor time axis is not strictly increasing: {resolved_path}")
        self.tensor_cache[tensor_path] = (amplitude, phase, time_seconds)
        while len(self.tensor_cache) > self.tensor_cache_size:
            self.tensor_cache.popitem(last=False)
        return amplitude, phase, time_seconds

    def __getitem__(self, item_index: int) -> NovelParentItem:
        parent_sample_id = self.parent_sample_ids[item_index]
        parent_rows = self.rows_by_parent[parent_sample_id]
        label_name = str(parent_rows.iloc[0]["label_name"])
        chunks: list[NovelParentChunk] = []
        parent_tensor_path = ""
        for row in parent_rows.itertuples(index=False):
            tensor_indices = _parse_indices(row.chunk_tensor_frame_indices_json, sample_id=str(row.sample_id))
            expected_count = int(row.chunk_frame_count)
            if len(tensor_indices) != expected_count:
                raise ValueError(f"chunk frame count mismatch for {row.sample_id}")
            if tensor_indices:
                tensor_path = str(row.rd_tensor_path)
                if parent_tensor_path and tensor_path != parent_tensor_path:
                    raise ValueError(f"parent novel updates span multiple tensors: {parent_sample_id}")
                parent_tensor_path = tensor_path
                amplitude, phase, time_seconds = self._load_tensor(tensor_path)
                if min(tensor_indices) < 0 or max(tensor_indices) >= amplitude.shape[0]:
                    raise ValueError(f"chunk tensor frame index out of range for {row.sample_id}")
                encoded = _encode_incremental_physical_sequence(
                    amplitude[tensor_indices],
                    phase[tensor_indices],
                    time_seconds[tensor_indices],
                )
                features = encoded.features
                frame_time_seconds = (
                    encoded.time_seconds
                    if encoded.time_seconds is not None
                    else torch.arange(encoded.valid_length, dtype=torch.float32)
                )
            else:
                features = torch.zeros((0, MULTIMODAL_RD_RANGE_BINS, 4), dtype=torch.float32)
                frame_time_seconds = torch.zeros((0,), dtype=torch.float32)
            chunks.append(
                NovelParentChunk(
                    update_index=int(row.update_index),
                    chunk_index=int(row.chunk_index),
                    is_final_chunk=_as_bool(row.is_final_chunk),
                    is_zero_novel_update=_as_bool(row.is_zero_novel_update),
                    delta_time_seconds=float(row.delta_time_seconds),
                    features=features,
                    frame_time_seconds=frame_time_seconds,
                    anchor_logits=(
                        self.anchor_logits_by_parent_update[(parent_sample_id, int(row.update_index))].clone()
                        if self.anchor_logits_by_parent_update is not None
                        else None
                    ),
                )
            )
        return NovelParentItem(parent_sample_id=parent_sample_id, label_name=label_name, chunks=tuple(chunks))


def collate_novel_parent_batch(items: list[NovelParentItem]) -> NovelParentBatch:
    if not items:
        raise ValueError("cannot collate empty novel parent batch")
    batch_size = len(items)
    max_chunks = max(len(item.chunks) for item in items)
    max_frames = max(1, max(chunk.features.shape[0] for item in items for chunk in item.chunks))
    features = torch.zeros((batch_size, max_chunks, max_frames, MULTIMODAL_RD_RANGE_BINS, 4), dtype=torch.float32)
    frame_padding_mask = torch.ones((batch_size, max_chunks, max_frames), dtype=torch.bool)
    chunk_padding_mask = torch.ones((batch_size, max_chunks), dtype=torch.bool)
    is_final_chunk = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
    is_zero_novel_update = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
    delta_time_seconds = torch.zeros((batch_size, max_chunks), dtype=torch.float32)
    frame_time_seconds = torch.zeros((batch_size, max_chunks, max_frames), dtype=torch.float32)
    labels = torch.empty((batch_size,), dtype=torch.long)
    parent_sample_ids: list[str] = []
    any_anchor = any(chunk.anchor_logits is not None for item in items for chunk in item.chunks)
    if any_anchor and any(chunk.anchor_logits is None for item in items for chunk in item.chunks):
        raise ValueError("cannot collate mixed anchored and unanchored novel parent chunks")
    anchor_logits = torch.zeros((batch_size, max_chunks, 2), dtype=torch.float32) if any_anchor else None
    for batch_index, item in enumerate(items):
        labels[batch_index] = encode_rd_drone_bird_label(item.label_name)
        parent_sample_ids.append(item.parent_sample_id)
        for chunk_position, chunk in enumerate(item.chunks):
            chunk_padding_mask[batch_index, chunk_position] = False
            frame_count = int(chunk.features.shape[0])
            if frame_count:
                features[batch_index, chunk_position, :frame_count] = chunk.features
                chunk_times = chunk.frame_time_seconds
                if int(chunk_times.numel()) != frame_count:
                    chunk_times = torch.arange(frame_count, dtype=torch.float32)
                frame_time_seconds[batch_index, chunk_position, :frame_count] = chunk_times
                frame_padding_mask[batch_index, chunk_position, :frame_count] = False
            is_final_chunk[batch_index, chunk_position] = chunk.is_final_chunk
            is_zero_novel_update[batch_index, chunk_position] = chunk.is_zero_novel_update
            delta_time_seconds[batch_index, chunk_position] = chunk.delta_time_seconds
            if anchor_logits is not None and chunk.anchor_logits is not None:
                anchor_logits[batch_index, chunk_position] = chunk.anchor_logits
    return NovelParentBatch(
        features=features,
        frame_padding_mask=frame_padding_mask,
        chunk_padding_mask=chunk_padding_mask,
        is_final_chunk=is_final_chunk,
        is_zero_novel_update=is_zero_novel_update,
        delta_time_seconds=delta_time_seconds,
        frame_time_seconds=frame_time_seconds,
        labels=labels,
        parent_sample_ids=parent_sample_ids,
        anchor_logits=anchor_logits,
    )


__all__ = [
    "NovelParentBatch",
    "NovelParentSequenceDataset",
    "collate_novel_parent_batch",
    "encode_incremental_physical_frames",
]
