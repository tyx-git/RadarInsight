from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .track_dataset import TrackDataset
from .features import encode_track_sequence, extract_physical_features, DEFAULT_VOCAB, FINE_VOCAB, load_rd_tensor, encode_multimodal_rd, load_rd_summary_frame, encode_rd_summary
from .rd import EncodedRDSequence
from .rd_segment_store import WindowAssembler
from ..router import ROUTE_NAMES


@dataclass(frozen=True)
class TorchTrackItem:
    sample_id: str
    source_name: str
    label_name: str
    coarse_label: str
    fine_label: str
    label_index: int
    fine_label_index: int
    features: torch.Tensor
    rd_features: torch.Tensor
    rd_patch_features: torch.Tensor
    rd_deep_patch_features: torch.Tensor
    rd_valid_length: int
    rd_patch_valid_length: int
    rd_deep_patch_valid_length: int
    rd_is_multimodal: bool
    rd_phase_missing: bool
    rd_time_sync_status: str
    height_missing: bool
    physical_features: torch.Tensor
    route_name: str
    route_index: int
    valid_length: int


def infer_route_name(label_name: str) -> str:
    if label_name in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}:
        return "low_altitude_suspect"
    if label_name in {"AircraftTarget"}:
        return "conventional_air"
    if label_name in {"PeopleTarget", "CarTarget", "ShipTarget"}:
        return "ground_target"
    return "high_altitude_filter"


class TorchTrackDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        sample_ids: Iterable[str] | None = None,
        trainable_only: bool = True,
        index_name: str | None = None,
        cache_size: int = 64,
        rd_cache_size: int = 32,
        enable_window_rd_time_crop: bool = True,
        rd_window_crop_min_frames: int = 2,
        rd_segment_manifest_name: str = "",
        window_segment_map_name: str = "",
        use_rd_segment_store: bool = False,
        rd_segment_store_cache_size: int = 64,
        allow_rd_segment_store_fallback: bool = False,
        load_rd: bool = True,
    ) -> None:
        base_dataset = TrackDataset(data_root, trainable_only=trainable_only, index_name=index_name)
        index_df = base_dataset.to_frame()
        if sample_ids is not None:
            sample_id_set = {str(sample_id) for sample_id in sample_ids}
            index_df = index_df[index_df["sample_id"].astype(str).isin(sample_id_set)].copy()
        self.track_dataset = base_dataset
        self.index_df = index_df.reset_index(drop=True)
        self.samples_by_id = {sample.sample_id: sample for sample in self.track_dataset.iter_samples()}
        self.cache_size = int(max(0, cache_size))
        self.rd_cache_size = int(max(0, rd_cache_size))
        self.enable_window_rd_time_crop = bool(enable_window_rd_time_crop)
        self.rd_window_crop_min_frames = int(max(1, rd_window_crop_min_frames))
        self.load_rd = bool(load_rd)
        self.use_rd_segment_store = bool(use_rd_segment_store)
        self.allow_rd_segment_store_fallback = bool(allow_rd_segment_store_fallback)
        self._cache: OrderedDict[str, TorchTrackItem] = OrderedDict()
        self._rd_cache: OrderedDict[str, EncodedRDSequence] = OrderedDict()
        self._parent_time_bounds = self._build_parent_time_bounds(self.index_df)
        self._window_assembler: WindowAssembler | None = None
        if self.load_rd and self.use_rd_segment_store:
            manifest_name = str(rd_segment_manifest_name or "expert1_rd_segment_manifest.csv")
            try:
                self._window_assembler = WindowAssembler(
                    self.track_dataset.data_root,
                    segment_manifest_name=manifest_name,
                    window_segment_map_name=str(window_segment_map_name or ""),
                    cache_size=rd_segment_store_cache_size,
                    strict_contained=True,
                )
            except FileNotFoundError:
                if not self.allow_rd_segment_store_fallback:
                    raise
                self._window_assembler = None

    def __len__(self) -> int:
        return len(self.index_df)

    def _get_cached(self, sample_id: str) -> TorchTrackItem | None:
        if sample_id not in self._cache:
            return None
        item = self._cache.pop(sample_id)
        self._cache[sample_id] = item
        return item

    def _put_cached(self, sample_id: str, item: TorchTrackItem) -> None:
        if self.cache_size <= 0:
            return
        if sample_id in self._cache:
            self._cache.pop(sample_id)
        self._cache[sample_id] = item
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _is_present(value: object) -> bool:
        text = str(value if value is not None else "").strip().lower()
        return text not in {"", "nan", "none"}

    @staticmethod
    def _finite_float(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    @classmethod
    def _build_parent_time_bounds(cls, index_df) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        if "window_start_seconds" not in index_df.columns or "window_end_seconds" not in index_df.columns:
            return bounds
        for row in index_df.to_dict(orient="records"):
            parent_id = str(row.get("parent_sample_id") or row.get("sample_id") or "")
            if not parent_id:
                continue
            start = cls._finite_float(row.get("window_start_seconds"))
            end = cls._finite_float(row.get("window_end_seconds"))
            if start is None:
                start = cls._finite_float(row.get("time_seconds_min"))
            if end is None:
                end = cls._finite_float(row.get("time_seconds_max"))
            if start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            if parent_id not in bounds:
                bounds[parent_id] = (start, end)
            else:
                current_start, current_end = bounds[parent_id]
                bounds[parent_id] = (min(current_start, start), max(current_end, end))
        return bounds

    def _get_rd_cached(self, cache_key: str) -> EncodedRDSequence | None:
        if cache_key not in self._rd_cache:
            return None
        encoded = self._rd_cache.pop(cache_key)
        self._rd_cache[cache_key] = encoded
        return encoded

    def _put_rd_cached(self, cache_key: str, encoded: EncodedRDSequence) -> None:
        if self.rd_cache_size <= 0:
            return
        if cache_key in self._rd_cache:
            self._rd_cache.pop(cache_key)
        self._rd_cache[cache_key] = encoded
        while len(self._rd_cache) > self.rd_cache_size:
            self._rd_cache.popitem(last=False)

    @staticmethod
    def _expand_summary_rd(encoded: EncodedRDSequence) -> EncodedRDSequence:
        summary_values = encoded.features[:, :3]
        width = 512
        expanded = torch.zeros((summary_values.shape[0], width, 4), dtype=torch.float32)
        expanded[:, :, 0] = summary_values[:, :1].repeat(1, width)
        expanded[:, :, 1] = summary_values[:, 1:2].repeat(1, width)
        expanded[:, :, 2] = summary_values[:, 2:3].repeat(1, width)
        patch_features = (
            encoded.patch_features
            if encoded.patch_features is not None
            else torch.zeros((encoded.valid_length, 64, 64, 3), dtype=torch.float32)
        )
        deep_patch_features = (
            encoded.deep_patch_features
            if encoded.deep_patch_features is not None
            else patch_features
        )
        return EncodedRDSequence(
            features=expanded,
            valid_length=int(expanded.shape[0]),
            patch_features=patch_features,
            patch_valid_length=int(patch_features.shape[0]),
            deep_patch_features=deep_patch_features,
            deep_patch_valid_length=int(deep_patch_features.shape[0]),
        )

    def _load_encoded_rd(self, sample) -> tuple[EncodedRDSequence | None, bool]:
        rd_tensor_path = str(sample.rd_tensor_path).strip()
        if self._is_present(rd_tensor_path):
            cache_key = f"tensor:{rd_tensor_path}"
            cached = self._get_rd_cached(cache_key)
            if cached is not None:
                return cached, True
            rd_tensor = load_rd_tensor(self.track_dataset.data_root, rd_tensor_path)
            encoded = encode_multimodal_rd(rd_tensor)
            self._put_rd_cached(cache_key, encoded)
            return encoded, True

        rd_summary_csv_path = str(sample.rd_summary_csv_path).strip()
        if self._is_present(rd_summary_csv_path):
            cache_key = f"summary:{rd_summary_csv_path}"
            cached = self._get_rd_cached(cache_key)
            if cached is not None:
                return cached, False
            rd_frame = load_rd_summary_frame(self.track_dataset.data_root, rd_summary_csv_path)
            encoded = self._expand_summary_rd(encode_rd_summary(rd_frame))
            self._put_rd_cached(cache_key, encoded)
            return encoded, False

        return None, False

    def _load_segment_store_rd(self, row) -> tuple[EncodedRDSequence | None, bool, str]:
        if self._window_assembler is None:
            return None, False, ""
        encoded, rd_is_multimodal, assembly = self._window_assembler.assemble(row)
        if encoded is None:
            return None, False, ""
        status_counts = assembly.rd_time_sync_status_counts if assembly is not None else {}
        if status_counts:
            rd_time_sync_status = max(
                status_counts.items(),
                key=lambda item: (int(item[1]), str(item[0])),
            )[0]
        else:
            rd_time_sync_status = "unavailable"
        return encoded, rd_is_multimodal, rd_time_sync_status

    def _window_fraction(self, row) -> tuple[float, float] | None:
        if not self.enable_window_rd_time_crop:
            return None
        if "window_start_seconds" not in row.index or "window_end_seconds" not in row.index:
            return None
        parent_id = str(row.get("parent_sample_id") or row.get("sample_id") or "")
        parent_bounds = self._parent_time_bounds.get(parent_id)
        if parent_bounds is None:
            return None
        window_start = self._finite_float(row.get("window_start_seconds"))
        window_end = self._finite_float(row.get("window_end_seconds"))
        if window_start is None or window_end is None:
            return None
        parent_start, parent_end = parent_bounds
        parent_span = parent_end - parent_start
        window_span = window_end - window_start
        if parent_span <= 1.0e-6 or window_span <= 0.0:
            return None
        start_fraction = (window_start - parent_start) / parent_span
        end_fraction = (window_end - parent_start) / parent_span
        start_fraction = min(max(start_fraction, 0.0), 1.0)
        end_fraction = min(max(end_fraction, 0.0), 1.0)
        if end_fraction - start_fraction >= 1.0 - 1.0e-6:
            return None
        if end_fraction <= start_fraction:
            return None
        return start_fraction, end_fraction

    def _slice_time_tensor(
        self,
        tensor: torch.Tensor,
        valid_length: int,
        start_fraction: float,
        end_fraction: float,
    ) -> tuple[torch.Tensor, int]:
        length = min(max(1, int(valid_length)), int(tensor.shape[0]))
        start_index = int(math.floor(start_fraction * length))
        end_index = int(math.ceil(end_fraction * length))
        start_index = min(max(0, start_index), length - 1)
        end_index = min(max(start_index + 1, end_index), length)
        min_frames = min(self.rd_window_crop_min_frames, length)
        if end_index - start_index < min_frames:
            center = 0.5 * (start_index + end_index)
            start_index = int(round(center - 0.5 * min_frames))
            start_index = min(max(0, start_index), length - min_frames)
            end_index = start_index + min_frames
        sliced = tensor[:length][start_index:end_index].contiguous()
        return sliced, int(sliced.shape[0])

    def _slice_encoded_rd(self, encoded: EncodedRDSequence, row) -> EncodedRDSequence:
        fraction = self._window_fraction(row)
        if fraction is None:
            return encoded
        start_fraction, end_fraction = fraction
        features, valid_length = self._slice_time_tensor(
            encoded.features,
            encoded.valid_length,
            start_fraction,
            end_fraction,
        )
        patch_features = None
        patch_valid_length = 0
        if encoded.patch_features is not None:
            patch_features, patch_valid_length = self._slice_time_tensor(
                encoded.patch_features,
                encoded.patch_valid_length or int(encoded.patch_features.shape[0]),
                start_fraction,
                end_fraction,
            )
        deep_patch_features = None
        deep_patch_valid_length = 0
        if encoded.deep_patch_features is not None:
            deep_patch_features, deep_patch_valid_length = self._slice_time_tensor(
                encoded.deep_patch_features,
                encoded.deep_patch_valid_length or int(encoded.deep_patch_features.shape[0]),
                start_fraction,
                end_fraction,
            )
        return EncodedRDSequence(
            features=features,
            valid_length=valid_length,
            patch_features=patch_features,
            patch_valid_length=patch_valid_length,
            deep_patch_features=deep_patch_features,
            deep_patch_valid_length=deep_patch_valid_length,
        )

    def __getitem__(self, index: int) -> TorchTrackItem:
        row = self.index_df.iloc[index]
        sample_id = str(row["sample_id"])
        cached = self._get_cached(sample_id)
        if cached is not None:
            return cached
        sample = self.samples_by_id.get(sample_id)
        if sample is None:
            raise KeyError(f"sample not found in dataset: {sample_id}")
        frame = sample.load_frame()
        encoded = encode_track_sequence(frame)
        physical = extract_physical_features(
            frame,
            height_missing=sample.height_missing,
            phase_missing=sample.rd_phase_missing,
        )
        route_name = infer_route_name(sample.label_name)
        rd_features = torch.zeros((1, 512, 4), dtype=torch.float32)
        rd_patch_features = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        rd_deep_patch_features = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        rd_valid_length = 1
        rd_patch_valid_length = 1
        rd_deep_patch_valid_length = 1
        rd_is_multimodal = False
        rd_time_sync_status = "unavailable"
        rd_encoded = None
        if self.load_rd:
            rd_encoded, rd_is_multimodal, segment_status = self._load_segment_store_rd(row)
            if rd_encoded is None:
                rd_encoded, rd_is_multimodal = self._load_encoded_rd(sample)
                if rd_encoded is not None:
                    rd_encoded = self._slice_encoded_rd(rd_encoded, row)
                    rd_time_sync_status = "proportional_fallback"
            else:
                rd_time_sync_status = segment_status
        if rd_encoded is not None:
            rd_features = rd_encoded.features
            rd_patch_features = (
                rd_encoded.patch_features
                if rd_encoded.patch_features is not None
                else torch.zeros((rd_encoded.valid_length, 64, 64, 3), dtype=torch.float32)
            )
            rd_deep_patch_features = (
                rd_encoded.deep_patch_features
                if rd_encoded.deep_patch_features is not None
                else rd_patch_features
            )
            rd_valid_length = rd_encoded.valid_length
            rd_patch_valid_length = rd_encoded.patch_valid_length or int(rd_patch_features.shape[0])
            rd_deep_patch_valid_length = rd_encoded.deep_patch_valid_length or int(rd_deep_patch_features.shape[0])
        item = TorchTrackItem(
            sample_id=sample.sample_id,
            source_name=sample.source_name,
            label_name=sample.label_name,
            coarse_label=sample.coarse_label,
            fine_label=sample.fine_label,
            label_index=DEFAULT_VOCAB.encode(sample.label_name),
            fine_label_index=FINE_VOCAB.encode(sample.fine_label),
            features=torch.from_numpy(encoded.features),
            rd_features=rd_features,
            rd_patch_features=rd_patch_features,
            rd_deep_patch_features=rd_deep_patch_features,
            rd_valid_length=rd_valid_length,
            rd_patch_valid_length=rd_patch_valid_length,
            rd_deep_patch_valid_length=rd_deep_patch_valid_length,
            rd_is_multimodal=rd_is_multimodal,
            rd_phase_missing=sample.rd_phase_missing,
            rd_time_sync_status=rd_time_sync_status,
            height_missing=sample.height_missing,
            physical_features=torch.from_numpy(physical.values),
            route_name=route_name,
            route_index=ROUTE_NAMES.index(route_name),
            valid_length=encoded.valid_length,
        )
        self._put_cached(sample_id, item)
        return item


__all__ = [
    "TorchTrackItem",
    "TorchTrackDataset",
    "infer_route_name",
]
