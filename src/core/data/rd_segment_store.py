from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .rd import EncodedRDSequence


@dataclass(frozen=True)
class WindowSegmentAssembly:
    parent_sample_id: str
    sample_id: str
    segment_ids: list[str]
    segment_indices: list[int]
    shard_keys: list[str]
    coverage_ratio: float
    rd_time_sync_status_counts: dict[str, int]
    rd_quality_flags: list[str]


def _is_present(value: object) -> bool:
    text = str(value if value is not None else "").strip().lower()
    return text not in {"", "nan", "none"}


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def parent_sample_id_from_row(row: pd.Series | dict[str, object]) -> str:
    parent_id = str(row.get("parent_sample_id", "") or "").strip()
    if parent_id:
        return parent_id
    sample_id = str(row.get("sample_id", "") or "").strip()
    if "|win" in sample_id:
        return sample_id.split("|win", 1)[0]
    return sample_id


def resolve_processed_path(data_root: Path | str, path_name: str) -> Path:
    data_root = Path(data_root)
    project_root = data_root.parent
    candidate = Path(path_name)
    if candidate.is_absolute():
        return candidate
    if str(path_name).replace("\\", "/").startswith("data/"):
        return project_root / candidate
    return data_root / "processed" / candidate


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", mmap=True)


def _json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value if value is not None else "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _window_bounds(row: pd.Series | dict[str, object]) -> tuple[float, float] | None:
    start = _finite_float(row.get("window_start_seconds"))
    end = _finite_float(row.get("window_end_seconds"))
    if start is None:
        start = _finite_float(row.get("time_seconds_min"))
    if end is None:
        end = _finite_float(row.get("time_seconds_max"))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def _coverage_ratio(segment_rows: list[dict[str, object]], window_start: float, window_end: float) -> float:
    window_duration = max(0.0, float(window_end) - float(window_start))
    if window_duration <= 1.0e-6:
        return 0.0
    covered = 0.0
    for row in segment_rows:
        segment_start = _finite_float(row.get("segment_start_seconds"))
        segment_end = _finite_float(row.get("segment_end_seconds"))
        if segment_start is None or segment_end is None:
            continue
        overlap_start = max(float(window_start), segment_start)
        overlap_end = min(float(window_end), segment_end)
        if overlap_end > overlap_start:
            covered += overlap_end - overlap_start
    return float(min(1.0, max(0.0, covered / window_duration)))


class WindowAssembler:
    def __init__(
        self,
        data_root: Path | str,
        segment_manifest_name: str = "expert1_rd_segment_manifest.csv",
        window_segment_map_name: str | None = None,
        cache_size: int = 64,
        strict_contained: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.project_root = self.data_root.parent
        self.segment_manifest_path = resolve_processed_path(self.data_root, segment_manifest_name)
        if not self.segment_manifest_path.exists():
            raise FileNotFoundError(self.segment_manifest_path)
        self.segment_df = pd.read_csv(self.segment_manifest_path)
        self.segment_df["parent_sample_id"] = self.segment_df["parent_sample_id"].astype(str)
        self.segment_df["segment_start_seconds"] = pd.to_numeric(
            self.segment_df["segment_start_seconds"],
            errors="coerce",
        )
        self.segment_df["segment_end_seconds"] = pd.to_numeric(
            self.segment_df["segment_end_seconds"],
            errors="coerce",
        )
        self._segments_by_parent: dict[str, pd.DataFrame] = {
            str(parent_id): group.sort_values(["segment_start_seconds", "segment_end_seconds"]).reset_index(drop=True)
            for parent_id, group in self.segment_df.groupby("parent_sample_id", sort=False)
        }
        self.strict_contained = bool(strict_contained)
        self.cache_size = int(max(0, cache_size))
        self._payload_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._map_by_sample: dict[str, list[str]] = {}
        if window_segment_map_name:
            map_path = resolve_processed_path(self.data_root, window_segment_map_name)
            if map_path.exists():
                map_df = pd.read_csv(map_path)
                if "sample_id" in map_df.columns and "segment_ids_json" in map_df.columns:
                    self._map_by_sample = {
                        str(row["sample_id"]): [str(item) for item in _json_list(row.get("segment_ids_json"))]
                        for row in map_df.to_dict(orient="records")
                    }

    def _get_payload(self, shard_key: str) -> dict[str, Any]:
        if shard_key in self._payload_cache:
            payload = self._payload_cache.pop(shard_key)
            self._payload_cache[shard_key] = payload
            return payload
        path = Path(shard_key)
        if not path.is_absolute():
            path = self.project_root / path
        payload = _load_torch_payload(path)
        if self.cache_size > 0:
            self._payload_cache[shard_key] = payload
            while len(self._payload_cache) > self.cache_size:
                self._payload_cache.popitem(last=False)
        return payload

    def select_segment_rows(self, row: pd.Series | dict[str, object]) -> tuple[list[dict[str, object]], WindowSegmentAssembly | None]:
        sample_id = str(row.get("sample_id", "") or "")
        parent_id = parent_sample_id_from_row(row)
        bounds = _window_bounds(row)
        if bounds is None:
            return [], None
        window_start, window_end = bounds
        parent_segments = self._segments_by_parent.get(parent_id)
        if parent_segments is None or parent_segments.empty:
            return [], None

        selected_df: pd.DataFrame
        mapped_segment_ids = self._map_by_sample.get(sample_id, [])
        if mapped_segment_ids:
            mapped = set(mapped_segment_ids)
            selected_df = parent_segments[parent_segments["segment_id"].astype(str).isin(mapped)].copy()
        else:
            epsilon = 1.0e-6
            if self.strict_contained:
                mask = (
                    (parent_segments["segment_start_seconds"] >= window_start - epsilon)
                    & (parent_segments["segment_end_seconds"] <= window_end + epsilon)
                )
            else:
                mask = (
                    (parent_segments["segment_end_seconds"] > window_start + epsilon)
                    & (parent_segments["segment_start_seconds"] < window_end - epsilon)
                )
            selected_df = parent_segments[mask].copy()
        if selected_df.empty:
            return [], None
        selected_df = selected_df.sort_values(["segment_start_seconds", "segment_end_seconds"])
        selected_rows = selected_df.to_dict(orient="records")
        coverage = _coverage_ratio(selected_rows, window_start, window_end)
        time_sync_counts = Counter(str(item.get("rd_time_sync_status", "unavailable")) for item in selected_rows)
        quality_flags = sorted(
            {
                flag
                for item in selected_rows
                for flag in str(item.get("rd_quality_flags", "")).split(";")
                if flag
            }
        )
        assembly = WindowSegmentAssembly(
            parent_sample_id=parent_id,
            sample_id=sample_id,
            segment_ids=[str(item.get("segment_id", "")) for item in selected_rows],
            segment_indices=[int(item.get("segment_index", 0)) for item in selected_rows],
            shard_keys=[str(item.get("shard_key", "")) for item in selected_rows],
            coverage_ratio=coverage,
            rd_time_sync_status_counts=dict(time_sync_counts),
            rd_quality_flags=quality_flags,
        )
        return selected_rows, assembly

    def assemble(
        self,
        row: pd.Series | dict[str, object],
        max_patch_frames: int = 0,
        skip_deep_patch: bool = False,
    ) -> tuple[EncodedRDSequence | None, bool, WindowSegmentAssembly | None]:
        selected_rows, assembly = self.select_segment_rows(row)
        if not selected_rows or assembly is None:
            return None, False, None

        features: list[torch.Tensor] = []
        patch_features: list[torch.Tensor] = []
        deep_patch_features: list[torch.Tensor] = []
        rd_is_multimodal = False
        for segment_row in selected_rows:
            shard_key = str(segment_row.get("shard_key", ""))
            if not _is_present(shard_key):
                continue
            payload = self._get_payload(shard_key)
            segment_index = int(segment_row.get("segment_index", 0))
            segments = payload.get("segments", [])
            if segment_index < 0 or segment_index >= len(segments):
                continue
            segment = segments[segment_index]
            rd_is_multimodal = (
                rd_is_multimodal
                or _as_bool(payload.get("rd_is_multimodal", False))
                or _as_bool(segment_row.get("rd_is_multimodal", False))
            )
            rd_len = int(segment.get("rd_valid_length", 0))
            patch_len = int(segment.get("rd_patch_valid_length", 0))
            deep_len = int(segment.get("rd_deep_patch_valid_length", 0))
            if rd_len > 0:
                features.append(segment["rd_features"][:rd_len].to(torch.float32).contiguous())
            if patch_len > 0:
                # Limit frames loaded from disk: subsample BEFORE contiguous()
                # to minimise mmap page reads
                if max_patch_frames > 0 and patch_len > max_patch_frames:
                    step = max(1, patch_len // max_patch_frames)
                    patch_src = segment["rd_patch_features"][:patch_len:step][:max_patch_frames]
                else:
                    patch_src = segment["rd_patch_features"][:patch_len]
                patch_features.append(patch_src.to(torch.float32).contiguous())
            if not skip_deep_patch and deep_len > 0:
                deep_patch_features.append(segment["rd_deep_patch_features"][:deep_len].to(torch.float32).contiguous())
        if not features:
            return None, False, None
        feature_tensor = torch.cat(features, dim=0)
        patch_tensor = torch.cat(patch_features, dim=0) if patch_features else None
        deep_patch_tensor = torch.cat(deep_patch_features, dim=0) if deep_patch_features else patch_tensor
        encoded = EncodedRDSequence(
            features=feature_tensor,
            valid_length=int(feature_tensor.shape[0]),
            patch_features=patch_tensor,
            patch_valid_length=int(patch_tensor.shape[0]) if patch_tensor is not None else 0,
            deep_patch_features=deep_patch_tensor,
            deep_patch_valid_length=int(deep_patch_tensor.shape[0]) if deep_patch_tensor is not None else 0,
        )
        return encoded, rd_is_multimodal, assembly


def build_window_segment_map(
    data_root: Path | str,
    window_index_name: str = "expert1_window_index.csv",
    segment_manifest_name: str = "expert1_rd_segment_manifest.csv",
    output_name: str = "expert1_window_segment_map.csv",
    max_windows: int = 0,
    strict_contained: bool = True,
) -> dict[str, object]:
    data_root = Path(data_root)
    window_index_path = resolve_processed_path(data_root, window_index_name)
    output_path = resolve_processed_path(data_root, output_name)
    if not window_index_path.exists():
        raise FileNotFoundError(window_index_path)
    assembler = WindowAssembler(
        data_root,
        segment_manifest_name=segment_manifest_name,
        window_segment_map_name=None,
        cache_size=0,
        strict_contained=strict_contained,
    )
    window_df = pd.read_csv(window_index_path)
    if "is_trainable" in window_df.columns:
        trainable_mask = window_df["is_trainable"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        window_df = window_df[trainable_mask].copy()
    if {"rd_tensor_path", "rd_summary_csv_path"}.intersection(window_df.columns):
        has_rd_mask = window_df.apply(
            lambda row: _is_present(row.get("rd_tensor_path", "")) or _is_present(row.get("rd_summary_csv_path", "")),
            axis=1,
        )
        window_df = window_df[has_rd_mask].copy()
    if max_windows > 0:
        window_df = window_df.head(max_windows).copy()

    rows: list[dict[str, object]] = []
    missing_count = 0
    coverage_values: list[float] = []
    segment_counts: list[int] = []
    status_counts: Counter[str] = Counter()
    for row in window_df.to_dict(orient="records"):
        selected_rows, assembly = assembler.select_segment_rows(row)
        if assembly is None:
            missing_count += 1
            continue
        bounds = _window_bounds(row)
        if bounds is None:
            missing_count += 1
            continue
        window_start, window_end = bounds
        segment_counts.append(len(assembly.segment_ids))
        coverage_values.append(assembly.coverage_ratio)
        status_counts.update(assembly.rd_time_sync_status_counts)
        rows.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "parent_sample_id": assembly.parent_sample_id,
                "source_name": str(row.get("source_name", "")),
                "label_name": str(row.get("label_name", "")),
                "window_start_seconds": float(window_start),
                "window_end_seconds": float(window_end),
                "window_duration_seconds": float(max(0.0, window_end - window_start)),
                "segment_ids_json": json.dumps(assembly.segment_ids, ensure_ascii=False),
                "segment_indices_json": json.dumps(assembly.segment_indices, ensure_ascii=False),
                "shard_keys_json": json.dumps(assembly.shard_keys, ensure_ascii=False),
                "segment_count": int(len(assembly.segment_ids)),
                "rd_coverage_ratio": float(assembly.coverage_ratio),
                "rd_time_sync_status_counts_json": json.dumps(assembly.rd_time_sync_status_counts, ensure_ascii=False),
                "rd_quality_flags_json": json.dumps(assembly.rd_quality_flags, ensure_ascii=False),
            }
        )

    output_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    summary = {
        "window_index_name": window_index_name,
        "segment_manifest_name": segment_manifest_name,
        "output_name": output_name,
        "window_count": int(len(window_df)),
        "mapped_window_count": int(len(rows)),
        "missing_window_count": int(missing_count),
        "strict_contained": bool(strict_contained),
        "segment_count_mean": float(sum(segment_counts) / len(segment_counts)) if segment_counts else 0.0,
        "rd_coverage_ratio_mean": float(sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0,
        "rd_time_sync_status_counts": dict(status_counts),
        "output_path": output_path.relative_to(data_root.parent).as_posix(),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "WindowAssembler",
    "WindowSegmentAssembly",
    "build_window_segment_map",
    "parent_sample_id_from_row",
    "resolve_processed_path",
]
