from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data.track_dataset import TrackDataset
from .encoding import (
    encode_track_sequence,
    normalize_track_sequence_features,
    prepare_track_sequence_features,
    track_sequence_feature_missing_mask,
)
from .features import (
    TRACK_RELATIVE_TIME_REFERENCE,
    track_sequence_binary_feature_indices,
    track_sequence_feature_dim,
)


_PRETRAIN_SAMPLING_MODES = {"full_track", "causal_window"}
_RELATIVE_TIME_REFERENCE = TRACK_RELATIVE_TIME_REFERENCE


def _parse_index_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError(f"missing boolean index field: {field_name}")
    try:
        if pd.isna(value):
            raise ValueError(f"missing boolean index field: {field_name}")
    except TypeError:
        pass
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean index field {field_name}: {value!r}")


@dataclass(frozen=True)
class PretrainTrackItem:
    sample_id: str
    source_name: str
    original: torch.Tensor
    masked: torch.Tensor
    mask: torch.Tensor
    feature_missing_mask: torch.Tensor
    view_a: torch.Tensor
    view_b: torch.Tensor
    valid_length: int


def _augment_sequence(
    features: np.ndarray,
    noise_std: float = 0.01,
    dropout_ratio: float = 0.1,
    binary_indices: list[int] | None = None,
) -> np.ndarray:
    augmented = features.copy()
    if augmented.size == 0:
        return augmented
    noise = np.random.normal(loc=0.0, scale=noise_std, size=augmented.shape).astype(np.float32)
    if binary_indices:
        noise[:, binary_indices] = 0.0
    augmented = augmented + noise
    if augmented.shape[0] > 1:
        keep = np.random.rand(augmented.shape[0]) > dropout_ratio
        if not keep.any():
            keep[np.random.randint(0, augmented.shape[0])] = True
        augmented = augmented[keep]
    return augmented


class TrackPretrainDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        mask_ratio: float = 0.15,
        trajectory_feature_schema: str = "v1",
        index_name: str = "system_trainable_track_index.csv",
        allowed_parent_ids: set[str] | None = None,
        sampling_mode: str = "full_track",
    ) -> None:
        self.sampling_mode = str(sampling_mode).strip().lower()
        if self.sampling_mode not in _PRETRAIN_SAMPLING_MODES:
            raise ValueError(
                "unsupported pretraining sampling_mode: "
                f"{sampling_mode!r}; expected one of {sorted(_PRETRAIN_SAMPLING_MODES)}"
            )
        self.index_name = str(index_name)
        self.track_dataset = TrackDataset(data_root, trainable_only=False, index_name=index_name)
        self.allowed_parent_ids = set(allowed_parent_ids or set())
        self._parent_id_by_sample_id: dict[str, str] = {}
        self.window_metadata_by_sample_id: dict[str, dict[str, object]] = {}

        if self.sampling_mode == "full_track":
            all_samples = list(self.track_dataset.iter_samples())
            self.samples = (
                [sample for sample in all_samples if sample.sample_id in self.allowed_parent_ids]
                if self.allowed_parent_ids
                else all_samples
            )
            self._parent_id_by_sample_id = {
                str(sample.sample_id): str(sample.sample_id)
                for sample in self.samples
            }
        else:
            index_df = getattr(self.track_dataset, "index_df", None)
            if not isinstance(index_df, pd.DataFrame):
                raise ValueError("causal_window pretraining requires an indexed TrackDataset")
            required_columns = {
                "sample_id",
                "parent_sample_id",
                "split_name",
                "is_trainable",
                "window_start_seconds",
                "window_end_seconds",
                "window_mode",
            }
            missing_columns = sorted(required_columns - set(index_df.columns))
            if missing_columns:
                raise ValueError(
                    "causal_window pretraining index is missing required columns: "
                    f"{missing_columns}"
                )
            split_values = index_df["split_name"].astype(str).str.strip().str.lower()
            trainable_values = index_df["is_trainable"].map(
                lambda value: _parse_index_bool(value, field_name="is_trainable")
            )
            selected = index_df[(split_values == "train") & trainable_values].copy()
            if selected.empty:
                raise ValueError("causal_window pretraining index has no trainable train rows")
            if "is_causal_window" in index_df.columns:
                causal_values = index_df["is_causal_window"].map(
                    lambda value: _parse_index_bool(value, field_name="is_causal_window")
                )
                if not bool(causal_values.loc[selected.index].all()):
                    raise ValueError("causal_window pretraining index contains a non-causal selected row")
            if selected["sample_id"].astype(str).duplicated().any():
                raise ValueError("causal_window pretraining index contains duplicate sample_id values")
            if selected["parent_sample_id"].isna().any():
                raise ValueError("causal_window pretraining index contains missing parent_sample_id")
            selected["sample_id"] = selected["sample_id"].astype(str)
            selected["parent_sample_id"] = selected["parent_sample_id"].astype(str)
            start_values = pd.to_numeric(selected["window_start_seconds"], errors="coerce")
            end_values = pd.to_numeric(selected["window_end_seconds"], errors="coerce")
            if not bool(np.isfinite(start_values.to_numpy()).all()) or not bool(
                np.isfinite(end_values.to_numpy()).all()
            ):
                raise ValueError("causal_window pretraining index has non-finite window bounds")
            if bool((end_values < start_values).any()):
                raise ValueError("causal_window pretraining index has an inverted window")
            parent_split_counts = (
                index_df[index_df["parent_sample_id"].astype(str).isin(self.allowed_parent_ids)]
                .assign(_split=split_values)
                .groupby("parent_sample_id", sort=True)["_split"]
                .nunique()
                if self.allowed_parent_ids
                else pd.Series(dtype=np.int64)
            )
            if bool((parent_split_counts > 1).any()):
                raise ValueError("causal_window pretraining parent appears in multiple persisted splits")
            if self.allowed_parent_ids:
                selected = selected[
                    selected["parent_sample_id"].isin(self.allowed_parent_ids)
                ].copy()
            if selected.empty:
                raise ValueError(
                    "causal_window pretraining index is empty after applying allowed_parent_ids"
                )
            source_counts = selected.groupby("parent_sample_id")["source_name"].nunique()
            if bool((source_counts > 1).any()):
                raise ValueError("causal_window pretraining parent contains multiple source_name values")
            self.track_dataset.index_df = selected.reset_index(drop=True)
            self.samples = list(self.track_dataset.iter_samples())
            self._parent_id_by_sample_id = {
                str(row["sample_id"]): str(row["parent_sample_id"])
                for row in selected.to_dict(orient="records")
            }
            self.window_metadata_by_sample_id = {
                str(row["sample_id"]): {
                    "parent_sample_id": str(row["parent_sample_id"]),
                    "window_start_seconds": float(row["window_start_seconds"]),
                    "window_end_seconds": float(row["window_end_seconds"]),
                    "window_mode": str(row["window_mode"]),
                }
                for row in selected.to_dict(orient="records")
            }

        if not self.samples:
            raise ValueError("pretraining dataset is empty after applying allowed_parent_ids")
        self.actual_parent_ids = {
            self._parent_id_by_sample_id[str(sample.sample_id)]
            for sample in self.samples
        }
        if self.allowed_parent_ids and self.actual_parent_ids != self.allowed_parent_ids:
            missing = sorted(self.allowed_parent_ids - self.actual_parent_ids)
            unexpected = sorted(self.actual_parent_ids - self.allowed_parent_ids)
            raise ValueError(
                "pretraining manifest/dataset parent mismatch: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        self.mask_ratio = mask_ratio
        self.trajectory_feature_schema = str(trajectory_feature_schema).strip().lower()
        self.source_stats = self._fit_source_stats()
        if self.sampling_mode == "causal_window":
            index_df = self.track_dataset.index_df
            self.sampling_metadata = {
                "mode": self.sampling_mode,
                "index_name": self.index_name,
                "split_name": "train",
                "sample_count": int(len(self.samples)),
                "parent_count": int(len(self.actual_parent_ids)),
                "window_mode_counts": {
                    str(key): int(value)
                    for key, value in index_df["window_mode"].astype(str).value_counts().to_dict().items()
                },
                "window_duration_seconds": {
                    "min": float(pd.to_numeric(index_df["window_end_seconds"], errors="coerce").sub(
                        pd.to_numeric(index_df["window_start_seconds"], errors="coerce")
                    ).min()),
                    "max": float(pd.to_numeric(index_df["window_end_seconds"], errors="coerce").sub(
                        pd.to_numeric(index_df["window_start_seconds"], errors="coerce")
                    ).max()),
                },
                "relative_time_reference": _RELATIVE_TIME_REFERENCE,
            }
        else:
            self.sampling_metadata = {
                "mode": self.sampling_mode,
                "index_name": self.index_name,
                "split_name": "manifest_parent_set" if self.allowed_parent_ids else "index_rows",
                "sample_count": int(len(self.samples)),
                "parent_count": int(len(self.actual_parent_ids)),
                "relative_time_reference": _RELATIVE_TIME_REFERENCE,
            }

    def __len__(self) -> int:
        return len(self.samples)

    def _fit_source_stats(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        grouped: dict[str, list[np.ndarray]] = {}
        for sample in self.samples:
            frame = sample.load_frame()
            encoded = encode_track_sequence(frame, feature_schema=self.trajectory_feature_schema)
            grouped.setdefault(sample.source_name, []).append(
                prepare_track_sequence_features(
                    encoded.features,
                    source_name=sample.source_name,
                    feature_schema=self.trajectory_feature_schema,
                )
            )
        feature_dim = track_sequence_feature_dim(self.trajectory_feature_schema)
        continuous_indices = np.asarray(
            [
                index
                for index in range(feature_dim)
                if index not in set(track_sequence_binary_feature_indices(self.trajectory_feature_schema))
            ],
            dtype=np.int64,
        )
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source_name, items in grouped.items():
            cat = np.concatenate(items, axis=0).astype(np.float32, copy=False)
            center = np.zeros((feature_dim,), dtype=np.float32)
            scale = np.ones((feature_dim,), dtype=np.float32)
            if continuous_indices.size:
                center[continuous_indices] = cat[:, continuous_indices].mean(axis=0).astype(np.float32, copy=False)
                scale[continuous_indices] = cat[:, continuous_indices].std(axis=0).astype(np.float32, copy=False)
                scale[continuous_indices] = np.clip(scale[continuous_indices], a_min=1e-3, a_max=None)
            stats[source_name] = (center, scale)
        return stats

    def __getitem__(self, index: int) -> PretrainTrackItem:
        sample = self.samples[index]
        frame = sample.load_frame()
        encoded = encode_track_sequence(frame, feature_schema=self.trajectory_feature_schema)
        feature_missing_mask = track_sequence_feature_missing_mask(
            frame,
            feature_schema=self.trajectory_feature_schema,
        )
        features = prepare_track_sequence_features(
            encoded.features,
            source_name=sample.source_name,
            feature_schema=self.trajectory_feature_schema,
        )
        center, scale = self.source_stats[sample.source_name]
        features = normalize_track_sequence_features(
            features,
            center,
            scale,
            feature_schema=self.trajectory_feature_schema,
        )
        masked = features.copy()
        mask = np.zeros((features.shape[0],), dtype=bool)
        valid_token_indices = np.flatnonzero(~feature_missing_mask.all(axis=1))
        if valid_token_indices.size > 0:
            num_mask = max(1, int(round(valid_token_indices.size * self.mask_ratio)))
            indices = np.random.choice(
                valid_token_indices,
                size=min(num_mask, valid_token_indices.size),
                replace=False,
            )
            mask[indices] = True
            masked[indices] = 0.0
        binary_indices = track_sequence_binary_feature_indices(self.trajectory_feature_schema)
        view_a = _augment_sequence(features, binary_indices=binary_indices)
        view_b = _augment_sequence(features, binary_indices=binary_indices)
        return PretrainTrackItem(
            sample_id=sample.sample_id,
            source_name=sample.source_name,
            original=torch.from_numpy(features),
            masked=torch.from_numpy(masked),
            mask=torch.from_numpy(mask),
            feature_missing_mask=torch.from_numpy(feature_missing_mask),
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
    feature_missing_mask: torch.Tensor
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
    feature_missing_mask = torch.zeros((batch, max_len, feat_dim), dtype=torch.bool)
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
        if item.feature_missing_mask.shape != item.original.shape:
            raise ValueError("feature_missing_mask must match the unpadded sequence shape")
        feature_missing_mask[idx, :l] = item.feature_missing_mask
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
        feature_missing_mask=feature_missing_mask,
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
    trajectory_feature_schema: str = "v1",
    index_name: str = "system_trainable_track_index.csv",
    allowed_parent_ids: set[str] | None = None,
    sampling_mode: str = "full_track",
) -> DataLoader:
    dataset = TrackPretrainDataset(
        data_root,
        mask_ratio=mask_ratio,
        trajectory_feature_schema=trajectory_feature_schema,
        index_name=index_name,
        allowed_parent_ids=allowed_parent_ids,
        sampling_mode=sampling_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pretrain_track_batch,
    )
