from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data.track_dataset import TrackDataset
from .tr_only_contract import TR_ONLY_ALLOWED_SOURCES, validate_tr_only_sources
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
_SOURCE_STATISTICS_WEIGHTING_MODES = {"point_weighted_v1", "parent_equal_v1"}
_PRETRAIN_RANDOMIZATION_MODES = {"stochastic", "deterministic_v1"}
_PRETRAIN_DATASET_ROLES = {"pretrain_train", "pretrain_validation"}
_PRETRAIN_PARENT_SAMPLING_MODES = {"window_shuffle_v1", "parent_equal_v1"}
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
    parent_sample_id: str = ""


def _augment_sequence(
    features: np.ndarray,
    noise_std: float = 0.01,
    dropout_ratio: float = 0.1,
    binary_indices: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    augmented = features.copy()
    if augmented.size == 0:
        return augmented
    noise = (
        rng.normal(loc=0.0, scale=noise_std, size=augmented.shape)
        if rng is not None
        else np.random.normal(loc=0.0, scale=noise_std, size=augmented.shape)
    ).astype(np.float32)
    if binary_indices:
        noise[:, binary_indices] = 0.0
    augmented = augmented + noise
    if augmented.shape[0] > 1:
        keep = (
            rng.random(augmented.shape[0])
            if rng is not None
            else np.random.rand(augmented.shape[0])
        ) > dropout_ratio
        if not keep.any():
            selected_index = (
                int(rng.integers(0, augmented.shape[0]))
                if rng is not None
                else int(np.random.randint(0, augmented.shape[0]))
            )
            keep[selected_index] = True
        augmented = augmented[keep]
    return augmented


def _fit_grouped_source_stats(
    grouped: dict[str, dict[str, list[np.ndarray]]],
    *,
    feature_dim: int,
    continuous_indices: np.ndarray,
    weighting: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    normalized_weighting = str(weighting).strip().lower()
    if normalized_weighting not in _SOURCE_STATISTICS_WEIGHTING_MODES:
        raise ValueError(
            "unsupported source statistics weighting: "
            f"{weighting!r}; expected one of {sorted(_SOURCE_STATISTICS_WEIGHTING_MODES)}"
        )
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source_name, parent_items in grouped.items():
        center = np.zeros((feature_dim,), dtype=np.float32)
        scale = np.ones((feature_dim,), dtype=np.float32)
        if continuous_indices.size:
            if normalized_weighting == "point_weighted_v1":
                all_items = [item for items in parent_items.values() for item in items]
                values = np.concatenate(all_items, axis=0).astype(np.float32, copy=False)
                if not np.isfinite(values[:, continuous_indices]).all():
                    raise ValueError(f"non-finite source statistics input for source={source_name}")
                source_center = values[:, continuous_indices].mean(axis=0)
                source_scale = values[:, continuous_indices].std(axis=0)
            else:
                parent_means: list[np.ndarray] = []
                parent_second_moments: list[np.ndarray] = []
                for items in parent_items.values():
                    values = np.concatenate(items, axis=0).astype(np.float64, copy=False)
                    continuous = values[:, continuous_indices]
                    if not np.isfinite(continuous).all():
                        raise ValueError(
                            f"non-finite parent-equal source statistics input for source={source_name}"
                        )
                    parent_means.append(continuous.mean(axis=0))
                    parent_second_moments.append(np.square(continuous).mean(axis=0))
                source_center = np.stack(parent_means, axis=0).mean(axis=0)
                source_second_moment = np.stack(parent_second_moments, axis=0).mean(axis=0)
                source_variance = np.maximum(
                    source_second_moment - np.square(source_center),
                    0.0,
                )
                source_scale = np.sqrt(source_variance)
            center[continuous_indices] = source_center.astype(np.float32, copy=False)
            scale[continuous_indices] = np.maximum(
                source_scale,
                1.0e-3,
            ).astype(np.float32, copy=False)
        stats[source_name] = (center, scale)
    return stats


class TrackPretrainDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        mask_ratio: float = 0.15,
        trajectory_feature_schema: str = "v1",
        index_name: str = "system_trainable_track_index.csv",
        allowed_parent_ids: set[str] | None = None,
        sampling_mode: str = "full_track",
        source_statistics_weighting: str = "point_weighted_v1",
        source_stats_override: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        randomization_mode: str = "stochastic",
        random_seed: int = 0,
        dataset_role: str = "pretrain_train",
        allowed_source_names: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.sampling_mode = str(sampling_mode).strip().lower()
        if self.sampling_mode not in _PRETRAIN_SAMPLING_MODES:
            raise ValueError(
                "unsupported pretraining sampling_mode: "
                f"{sampling_mode!r}; expected one of {sorted(_PRETRAIN_SAMPLING_MODES)}"
            )
        self.source_statistics_weighting = str(source_statistics_weighting).strip().lower()
        if self.source_statistics_weighting not in _SOURCE_STATISTICS_WEIGHTING_MODES:
            raise ValueError(
                "unsupported source statistics weighting: "
                f"{source_statistics_weighting!r}; expected one of "
                f"{sorted(_SOURCE_STATISTICS_WEIGHTING_MODES)}"
            )
        self.randomization_mode = str(randomization_mode).strip().lower()
        if self.randomization_mode not in _PRETRAIN_RANDOMIZATION_MODES:
            raise ValueError(
                "unsupported pretraining randomization_mode: "
                f"{randomization_mode!r}; expected one of {sorted(_PRETRAIN_RANDOMIZATION_MODES)}"
            )
        self.dataset_role = str(dataset_role).strip().lower()
        if self.dataset_role not in _PRETRAIN_DATASET_ROLES:
            raise ValueError(
                "unsupported pretraining dataset_role: "
                f"{dataset_role!r}; expected one of {sorted(_PRETRAIN_DATASET_ROLES)}"
            )
        if self.dataset_role == "pretrain_validation":
            if self.randomization_mode != "deterministic_v1":
                raise ValueError("pretrain_validation requires randomization_mode='deterministic_v1'")
            if source_stats_override is None:
                raise ValueError("pretrain_validation requires source_stats_override fitted on pretrain_train")
        self.random_seed = int(random_seed)
        self.index_name = str(index_name)
        self.allowed_source_names = (
            frozenset(str(value).strip() for value in allowed_source_names)
            if allowed_source_names is not None
            else None
        )
        self.track_dataset = TrackDataset(data_root, trainable_only=False, index_name=index_name)
        if self.allowed_source_names is not None:
            validate_tr_only_sources(
                self.track_dataset.index_df["source_name"].tolist()
                if "source_name" in self.track_dataset.index_df.columns
                else [],
                context="TrackPretrainDataset",
            )
            unexpected = sorted(
                set(self.track_dataset.index_df["source_name"].astype(str).str.strip())
                - set(self.allowed_source_names)
            )
            if unexpected:
                raise ValueError(
                    "TrackPretrainDataset source allowlist mismatch: "
                    f"allowed={sorted(self.allowed_source_names)}, unexpected={unexpected}"
                )
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
        if self.randomization_mode == "deterministic_v1":
            self.samples = sorted(
                self.samples,
                key=lambda sample: hashlib.sha256(
                    (
                        f"{self.random_seed}:{self.dataset_role}:sample_order:"
                        f"{sample.sample_id}"
                    ).encode("utf-8")
                ).digest(),
            )
            self.sample_ordering = "seeded_sample_id_sha256_v1"
        else:
            self.sample_ordering = "dataset_index_order"
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
        if source_stats_override is None:
            self.source_stats = self._fit_source_stats()
            self.source_statistics_origin = self.dataset_role
        else:
            self.source_stats = self._validated_source_stats_override(source_stats_override)
            self.source_statistics_origin = "override"
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
                "source_statistics_weighting": self.source_statistics_weighting,
                "dataset_role": self.dataset_role,
                "randomization_mode": self.randomization_mode,
                "random_seed": self.random_seed,
                "source_statistics_origin": self.source_statistics_origin,
                "sample_ordering": self.sample_ordering,
            }
        else:
            self.sampling_metadata = {
                "mode": self.sampling_mode,
                "index_name": self.index_name,
                "split_name": "manifest_parent_set" if self.allowed_parent_ids else "index_rows",
                "sample_count": int(len(self.samples)),
                "parent_count": int(len(self.actual_parent_ids)),
                "relative_time_reference": _RELATIVE_TIME_REFERENCE,
                "source_statistics_weighting": self.source_statistics_weighting,
                "dataset_role": self.dataset_role,
                "randomization_mode": self.randomization_mode,
                "random_seed": self.random_seed,
                "source_statistics_origin": self.source_statistics_origin,
                "sample_ordering": self.sample_ordering,
            }

    def __len__(self) -> int:
        return len(self.samples)

    def _fit_source_stats(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        grouped: dict[str, dict[str, list[np.ndarray]]] = {}
        for sample in self.samples:
            frame = sample.load_frame()
            encoded = encode_track_sequence(frame, feature_schema=self.trajectory_feature_schema)
            parent_id = self._parent_id_by_sample_id[str(sample.sample_id)]
            grouped.setdefault(sample.source_name, {}).setdefault(parent_id, []).append(
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
        return _fit_grouped_source_stats(
            grouped,
            feature_dim=feature_dim,
            continuous_indices=continuous_indices,
            weighting=self.source_statistics_weighting,
        )

    def _validated_source_stats_override(
        self,
        source_stats_override: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not isinstance(source_stats_override, dict) or not source_stats_override:
            raise ValueError("source_stats_override must be a non-empty source mapping")
        expected_dim = track_sequence_feature_dim(self.trajectory_feature_schema)
        required_sources = {str(sample.source_name) for sample in self.samples}
        missing_sources = sorted(required_sources - set(source_stats_override))
        if missing_sources:
            raise ValueError(
                "source_stats_override is missing validation sources: "
                f"{missing_sources}"
            )
        validated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source_name, values in source_stats_override.items():
            if not isinstance(values, tuple) or len(values) != 2:
                raise ValueError(
                    f"source_stats_override[{source_name!r}] must be a (center, scale) tuple"
                )
            center = np.asarray(values[0], dtype=np.float32).reshape(-1).copy()
            scale = np.asarray(values[1], dtype=np.float32).reshape(-1).copy()
            if center.shape != (expected_dim,) or scale.shape != (expected_dim,):
                raise ValueError(
                    "source_stats_override feature dimension mismatch for "
                    f"source={source_name}: center={center.size}, scale={scale.size}, "
                    f"expected={expected_dim}"
                )
            if not np.isfinite(center).all() or not np.isfinite(scale).all():
                raise ValueError(f"source_stats_override contains non-finite values for source={source_name}")
            if bool((scale <= 0.0).any()):
                raise ValueError(f"source_stats_override contains non-positive scale for source={source_name}")
            validated[str(source_name)] = (center, scale)
        return validated

    def _sample_rng(self, sample_id: str) -> np.random.Generator | None:
        if self.randomization_mode == "stochastic":
            return None
        seed_material = f"{self.random_seed}:{self.dataset_role}:{sample_id}".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        return np.random.default_rng(derived_seed)

    def __getitem__(self, index: int) -> PretrainTrackItem:
        sample = self.samples[index]
        sample_id = str(sample.sample_id)
        rng = self._sample_rng(sample_id)
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
            mask_count = min(num_mask, valid_token_indices.size)
            indices = (
                rng.choice(valid_token_indices, size=mask_count, replace=False)
                if rng is not None
                else np.random.choice(valid_token_indices, size=mask_count, replace=False)
            )
            mask[indices] = True
            masked[indices] = 0.0
        binary_indices = track_sequence_binary_feature_indices(self.trajectory_feature_schema)
        view_a = _augment_sequence(features, binary_indices=binary_indices, rng=rng)
        view_b = _augment_sequence(features, binary_indices=binary_indices, rng=rng)
        return PretrainTrackItem(
            sample_id=sample_id,
            source_name=sample.source_name,
            original=torch.from_numpy(features),
            masked=torch.from_numpy(masked),
            mask=torch.from_numpy(mask),
            feature_missing_mask=torch.from_numpy(feature_missing_mask),
            view_a=torch.from_numpy(view_a),
            view_b=torch.from_numpy(view_b),
            valid_length=int(features.shape[0]),
            parent_sample_id=self._parent_id_by_sample_id[sample_id],
        )


@dataclass(frozen=True)
class PretrainTrackBatch:
    sample_ids: list[str]
    parent_sample_ids: list[str]
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
    parent_sample_ids: list[str] = []
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
        parent_sample_ids.append(item.parent_sample_id or item.sample_id)
        source_names.append(item.source_name)

    return PretrainTrackBatch(
        sample_ids=sample_ids,
        parent_sample_ids=parent_sample_ids,
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
    source_statistics_weighting: str = "point_weighted_v1",
    source_stats_override: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    randomization_mode: str = "stochastic",
    random_seed: int = 0,
    dataset_role: str = "pretrain_train",
    shuffle: bool = True,
    parent_sampling_mode: str = "window_shuffle_v1",
    allowed_source_names: set[str] | frozenset[str] | None = None,
) -> DataLoader:
    normalized_parent_sampling = str(parent_sampling_mode).strip().lower()
    if normalized_parent_sampling not in _PRETRAIN_PARENT_SAMPLING_MODES:
        raise ValueError(
            "unsupported parent_sampling_mode: "
            f"{parent_sampling_mode!r}; expected one of "
            f"{sorted(_PRETRAIN_PARENT_SAMPLING_MODES)}"
        )
    dataset = TrackPretrainDataset(
        data_root,
        mask_ratio=mask_ratio,
        trajectory_feature_schema=trajectory_feature_schema,
        index_name=index_name,
        allowed_parent_ids=allowed_parent_ids,
        sampling_mode=sampling_mode,
        source_statistics_weighting=source_statistics_weighting,
        source_stats_override=source_stats_override,
        randomization_mode=randomization_mode,
        random_seed=random_seed,
        dataset_role=dataset_role,
        allowed_source_names=allowed_source_names,
    )
    dataset.sampling_metadata["parent_sampling_mode"] = normalized_parent_sampling
    sampler = None
    if normalized_parent_sampling == "parent_equal_v1":
        if str(dataset_role).strip().lower() != "pretrain_train":
            raise ValueError("parent_equal_v1 sampling is only valid for pretrain_train")
        parent_counts: dict[str, int] = {}
        sample_parent_ids = [
            str(dataset._parent_id_by_sample_id[str(sample.sample_id)])
            for sample in dataset.samples
        ]
        for parent_id in sample_parent_ids:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        weights = torch.tensor(
            [1.0 / float(parent_counts[parent_id]) for parent_id in sample_parent_ids],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(int(random_seed)),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_pretrain_track_batch,
    )
