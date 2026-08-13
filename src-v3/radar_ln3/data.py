from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .features import (
    CausalTRMotionState,
    CausalTRObservationIncrementState,
    TR_MOTION_SCHEMA,
    TR_OBSERVATION_INCREMENT_SCHEMA,
    encode_tr_feature_values,
    parse_feature_json,
    resolve_causal_tr_feature_contract,
)
from .micro_doppler import MICRO_DOPPLER_VALUE_DIM, load_micro_doppler_cache
from .micro_doppler_broadband import BROADBAND_MICRO_DOPPLER_VALUE_DIM
from .rd_physical import RD_PHYSICAL_VALUE_DIM, load_rd_physical_cache


RD_QUALITY_SCHEMA = "causal-current-event-v1"
RD_QUALITY_NAMES = (
    "rd_present",
    "rd_frame_count_scaled",
    "rd_local_duration_scaled",
    "rd_missing",
    "rd_local_time_non_monotonic",
    "rd_multiple_frames",
    "rd_frames_truncated",
)

DT_REASON_PADDING = 0
DT_REASON_FIRST_EVENT = 1
DT_REASON_ZERO_DELTA = 2
DT_REASON_POSITIVE_DELTA = 3
DT_REASON_NAMES = (
    "padding",
    "first_event_no_predecessor",
    "zero_delta_at_second_precision",
    "positive_delta_at_second_precision",
)
DT_REASON_TO_ID = {name: index for index, name in enumerate(DT_REASON_NAMES)}
REASON_AWARE_INDEX_SCHEMA_VERSIONS = frozenset(
    {"RADAR-LN3-V3", "RADAR-LN3-V16"}
)


def encode_dt_reason(value: Any, *, dt_valid: bool) -> int:
    """Encode the frozen V3 time reason for dynamics-only consumption."""

    reason = str(value)
    try:
        reason_id = DT_REASON_TO_ID[reason]
    except KeyError as exc:
        raise ValueError(f"unsupported tr_dt_reason: {reason!r}") from exc
    if reason_id == DT_REASON_PADDING:
        raise ValueError("padding is a collate-only dt reason and cannot appear in the index")
    expected_valid = reason_id == DT_REASON_POSITIVE_DELTA
    if bool(dt_valid) != expected_valid:
        raise ValueError(
            f"tr_dt_reason/tr_dt_valid mismatch: reason={reason!r}, dt_valid={bool(dt_valid)}"
        )
    return reason_id


def require_causal_rd_quality_schema(schema: str | None) -> None:
    if schema != RD_QUALITY_SCHEMA:
        raise ValueError(
            "configuration/checkpoint does not declare the causal RD quality schema; "
            f"expected={RD_QUALITY_SCHEMA!r}, actual={schema!r}. "
            "Legacy V3/V4/V5 artifacts must not run against the repaired input semantics."
        )


REQUIRED_INDEX_COLUMNS = {
    "schema_version",
    "sample_id",
    "parent_sample_id",
    "track_id",
    "source_name",
    "split_group_id",
    "split_name",
    "label_name",
    "label_id",
    "causal_event_index",
    "tr_time_seconds",
    "tr_dt_seconds",
    "tr_dt_is_negative",
    "tr_feature_values_json",
    "tr_missing_count",
    "rd_present",
    "rd_frame_count",
    "rd_tensor_path",
    "rd_tensor_frame_indices_json",
    "rd_local_duration_seconds",
    "rd_parent_coverage_status",
    "rd_parent_order_status",
    "rd_quality_missing",
    "rd_quality_partial_parent",
    "rd_quality_parent_order_anomaly",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_indices(value: Any) -> list[int]:
    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("RD frame index field must decode to a list")
    return [int(item) for item in payload]


def encode_causal_rd_quality(
    *,
    present: bool,
    original_count: int,
    duration_seconds: float,
    local_time_non_monotonic: bool,
    truncated: bool,
    max_rd_frames_per_point: int,
) -> np.ndarray:
    """Encode only information observable at the current TR/RD event.

    Parent-level coverage and order summaries are intentionally excluded: they
    require future events and would leak target-completion information into an
    online causal prefix. The vector remains seven-dimensional for the current
    model interface, but its schema is explicitly versioned.
    """

    count = max(0, int(original_count))
    max_frames = max(1, int(max_rd_frames_per_point))
    duration = max(0.0, float(duration_seconds))
    values = np.asarray(
        [
            float(bool(present)),
            float(np.log1p(count) / np.log1p(max_frames)),
            float(np.clip(duration / 0.5, 0.0, 1.0)),
            float(not bool(present)),
            float(bool(local_time_non_monotonic)),
            float(count > 1),
            float(bool(truncated)),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(RD_QUALITY_NAMES),) or not np.isfinite(values).all():
        raise ValueError("causal RD quality encoder produced an invalid vector")
    return values


@dataclass
class PointSequence:
    parent_sample_id: str
    track_id: str
    label_name: str
    split_name: str
    tr_features: torch.Tensor
    tr_missing_mask: torch.Tensor
    tr_dt_seconds: torch.Tensor
    tr_dt_valid: torch.Tensor
    tr_dt_reason: torch.Tensor
    tr_motion_features: torch.Tensor
    tr_motion_valid_mask: torch.Tensor
    rd_frames: torch.Tensor
    rd_frame_mask: torch.Tensor
    rd_quality: torch.Tensor
    rd_physical_features: torch.Tensor
    rd_physical_valid: torch.Tensor
    micro_doppler_features: torch.Tensor
    micro_doppler_valid: torch.Tensor
    label: torch.Tensor
    event_count: int
    source_name: str


class _TensorCache:
    def __init__(self, project_root: Path, max_size: int = 4) -> None:
        self.project_root = project_root
        self.max_size = max(1, int(max_size))
        self._items: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def load(self, relative_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not relative_path:
            raise ValueError("present RD row has an empty tensor path")
        path = Path(relative_path.replace("\\", "/"))
        if not path.is_absolute():
            path = self.project_root / path
        key = str(path.resolve())
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            return cached
        if not path.exists():
            raise FileNotFoundError(f"RD tensor does not exist: {path}")
        with np.load(path, allow_pickle=False) as payload:
            required = {"amplitude", "phase", "time_seconds"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"RD tensor missing keys: {path}: {missing}")
            amplitude = np.asarray(payload["amplitude"], dtype=np.float32)
            phase = np.asarray(payload["phase"], dtype=np.float32)
            times = np.asarray(payload["time_seconds"], dtype=np.float32)
        if amplitude.ndim != 3 or tuple(amplitude.shape[1:]) != (21, 384):
            raise ValueError(f"invalid RD tensor shape: {path}: {amplitude.shape}")
        if phase.shape != amplitude.shape or times.shape != (amplitude.shape[0],):
            raise ValueError(f"invalid RD tensor array shapes: {path}")
        if not np.isfinite(amplitude).all() or not np.isfinite(phase).all() or not np.isfinite(times).all():
            raise ValueError(f"RD tensor contains non-finite values: {path}")
        value = (amplitude, phase, times)
        self._items[key] = value
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
        return value


class CQ08PointSequenceDataset(Dataset[PointSequence]):
    """Parent-level sequences built from the V0 one-row-per-TR-point index."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        index_path: str | Path,
        split_name: str,
        max_sequence_length: int,
        max_rd_frames_per_point: int,
        rd_amplitude_center: float,
        rd_amplitude_scale: float,
        exclude_anomalous_parents: bool = True,
        sample_limit: int | None = None,
        tensor_cache_size: int = 4,
        expected_schema_version: str = "RADAR-LN3-V0",
        enable_tr_motion: bool = False,
        tr_motion_schema: str | None = None,
        rd_physical_cache_path: str | Path | None = None,
        micro_doppler_cache_path: str | Path | None = None,
        micro_doppler_value_dim: int = MICRO_DOPPLER_VALUE_DIM,
        allowed_labels: frozenset[str] | set[str] | tuple[str, ...] | None = None,
        forbidden_labels: frozenset[str] | set[str] | tuple[str, ...] | None = None,
        allowed_parent_ids: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        path = Path(str(index_path).replace("\\", "/"))
        self.index_path = path if path.is_absolute() else self.project_root / path
        self.max_sequence_length = max(1, int(max_sequence_length))
        self.max_rd_frames_per_point = max(1, int(max_rd_frames_per_point))
        self.rd_amplitude_center = float(rd_amplitude_center)
        self.rd_amplitude_scale = float(rd_amplitude_scale)
        if tr_motion_schema is None and enable_tr_motion:
            tr_motion_schema = TR_MOTION_SCHEMA
        self.tr_motion_schema = None if tr_motion_schema is None else str(tr_motion_schema)
        self.enable_tr_motion = self.tr_motion_schema is not None
        if self.tr_motion_schema is not None:
            resolve_causal_tr_feature_contract(self.tr_motion_schema)
        if self.rd_amplitude_scale <= 0.0:
            raise ValueError("rd_amplitude_scale must be positive")
        frame = pd.read_csv(self.index_path, low_memory=False)
        missing_columns = sorted(REQUIRED_INDEX_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(f"V0 point index is missing columns: {missing_columns}")
        self.expected_schema_version = str(expected_schema_version)
        if frame["schema_version"].astype(str).ne(self.expected_schema_version).any():
            raise ValueError(f"dataset contains a row outside expected schema {self.expected_schema_version}")
        if self.expected_schema_version in REASON_AWARE_INDEX_SCHEMA_VERSIONS:
            missing_reason_aware = sorted({"tr_dt_valid", "tr_dt_reason"} - set(frame.columns))
            if missing_reason_aware:
                raise ValueError(
                    f"{self.expected_schema_version} point index is missing dt validity columns: "
                    f"{missing_reason_aware}"
                )
        frame = frame[frame["split_name"].astype(str).eq(str(split_name))].copy()
        if frame.empty:
            raise ValueError(f"V0 point index has no rows for split={split_name}")
        allowed = None if allowed_labels is None else frozenset(str(value) for value in allowed_labels)
        forbidden = frozenset(str(value) for value in (forbidden_labels or ()))
        if allowed is not None:
            if not allowed or allowed & forbidden:
                raise ValueError("Dataset allowed_labels must be non-empty and disjoint from forbidden_labels")
            frame = frame[frame["label_name"].astype(str).isin(allowed)].copy()
            if frame.empty:
                raise ValueError(f"Dataset has no rows after allowed-label filtering for split={split_name}")
        observed_forbidden = sorted(set(frame["label_name"].astype(str)) & forbidden)
        if observed_forbidden:
            raise ValueError(f"Dataset contains forbidden labels after filtering: {observed_forbidden}")
        if frame["source_name"].astype(str).ne("CQ08").any():
            raise ValueError("CQ08 dataset received a non-CQ08 source")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("V0 point index contains duplicate sample_id")
        groups = frame.groupby("parent_sample_id", sort=True)
        selected: list[str] = []
        for parent_id, group in groups:
            if group["label_name"].astype(str).nunique() != 1:
                raise ValueError(f"parent has multiple labels: {parent_id}")
            if group["split_name"].astype(str).nunique() != 1:
                raise ValueError(f"parent crosses splits: {parent_id}")
            if exclude_anomalous_parents and (
                group["tr_dt_is_negative"].map(_as_bool).any()
                or group["rd_parent_order_status"].astype(str).eq("non_monotonic").any()
            ):
                continue
            selected.append(str(parent_id))
        selected = sorted(selected)
        requested_parents = (
            None
            if allowed_parent_ids is None
            else frozenset(str(value) for value in allowed_parent_ids)
        )
        if requested_parents is not None:
            if not requested_parents:
                raise ValueError("Dataset allowed_parent_ids must be non-empty")
            unavailable = sorted(requested_parents - set(selected))
            if unavailable:
                raise ValueError(
                    "Dataset allowed_parent_ids contain unavailable parents: "
                    f"count={len(unavailable)}, first={unavailable[:3]}"
                )
            selected = [parent_id for parent_id in selected if parent_id in requested_parents]
        if sample_limit is not None:
            selected = selected[: max(1, int(sample_limit))]
        if not selected:
            raise ValueError("V0 dataset has no selected parent sequences")
        self.frame = frame[frame["parent_sample_id"].astype(str).isin(selected)].copy()
        self.frame["parent_sample_id"] = self.frame["parent_sample_id"].astype(str)
        self.frame["causal_event_index"] = pd.to_numeric(self.frame["causal_event_index"], errors="raise").astype(int)
        self.frame = self.frame.sort_values(["parent_sample_id", "causal_event_index"], kind="stable").reset_index(drop=True)
        if self.expected_schema_version in REASON_AWARE_INDEX_SCHEMA_VERSIONS:
            reasons = self.frame["tr_dt_reason"].astype(str)
            unsupported_reasons = sorted(set(reasons) - set(DT_REASON_NAMES[1:]))
            if unsupported_reasons:
                raise ValueError(
                    f"{self.expected_schema_version} point index contains unsupported dt reasons: "
                    f"{unsupported_reasons}"
                )
            valid = self.frame["tr_dt_valid"].map(_as_bool)
            expected_valid = reasons.eq("positive_delta_at_second_precision")
            if not valid.equals(expected_valid):
                raise ValueError(
                    f"{self.expected_schema_version} point index has a tr_dt_reason/tr_dt_valid mismatch"
                )
            raw_dt = pd.to_numeric(self.frame["tr_dt_seconds"], errors="raise")
            if not np.isfinite(raw_dt.to_numpy(dtype=np.float64)).all():
                raise ValueError(f"{self.expected_schema_version} point index contains a non-finite raw dt")
            zero_reason = reasons.isin(("first_event_no_predecessor", "zero_delta_at_second_precision"))
            if bool(raw_dt[zero_reason].abs().gt(1.0e-9).any()) or bool(raw_dt[expected_valid].le(0.0).any()):
                raise ValueError(
                    f"{self.expected_schema_version} point index raw dt does not match its dt reason"
                )
            for parent_id, parent in self.frame.groupby("parent_sample_id", sort=False):
                event_indices = parent["causal_event_index"].tolist()
                if event_indices != list(range(len(parent))):
                    raise ValueError(
                        f"{self.expected_schema_version} parent causal event indices are not contiguous "
                        f"from zero: {parent_id}"
                    )
                parent_reasons = parent["tr_dt_reason"].astype(str).tolist()
                if (
                    not parent_reasons
                    or parent_reasons[0] != "first_event_no_predecessor"
                    or "first_event_no_predecessor" in parent_reasons[1:]
                ):
                    raise ValueError(
                        f"{self.expected_schema_version} parent dt reason does not mark exactly one first "
                        f"event: {parent_id}"
                    )
        self.parent_ids = selected
        self.allowed_labels = allowed
        self.forbidden_labels = forbidden
        self.allowed_parent_ids = requested_parents
        self._groups = {parent_id: group.reset_index(drop=True) for parent_id, group in self.frame.groupby("parent_sample_id", sort=False)}
        self._tensor_cache = _TensorCache(self.project_root, max_size=tensor_cache_size)
        self.rd_physical_cache_path: Path | None = None
        self._rd_physical_values: dict[str, np.ndarray] | None = None
        self._rd_physical_valid: dict[str, bool] | None = None
        self.rd_physical_cache_metadata: dict[str, Any] | None = None
        if rd_physical_cache_path is not None:
            cache_path = Path(str(rd_physical_cache_path).replace("\\", "/"))
            self.rd_physical_cache_path = cache_path if cache_path.is_absolute() else self.project_root / cache_path
            expected_ids = self.frame["sample_id"].astype(str).tolist()
            mapping, metadata = load_rd_physical_cache(
                self.rd_physical_cache_path,
                expected_sample_ids=expected_ids,
            )
            self._rd_physical_values = mapping
            self._rd_physical_valid = dict(metadata.pop("valid_by_sample_id"))
            self.rd_physical_cache_metadata = metadata
        self.micro_doppler_cache_path: Path | None = None
        self._micro_doppler_values: dict[str, np.ndarray] | None = None
        self._micro_doppler_valid: dict[str, bool] | None = None
        self.micro_doppler_cache_metadata: dict[str, Any] | None = None
        if micro_doppler_cache_path is not None:
            expected_micro_doppler_dim = int(micro_doppler_value_dim)
            if expected_micro_doppler_dim not in {
                MICRO_DOPPLER_VALUE_DIM,
                BROADBAND_MICRO_DOPPLER_VALUE_DIM,
            }:
                raise ValueError(
                    "Dataset micro-Doppler cache dimension does not match a registered contract"
                )
            cache_path = Path(str(micro_doppler_cache_path).replace("\\", "/"))
            self.micro_doppler_cache_path = (
                cache_path if cache_path.is_absolute() else self.project_root / cache_path
            )
            expected_ids = self.frame["sample_id"].astype(str).tolist()
            mapping, metadata = load_micro_doppler_cache(
                self.micro_doppler_cache_path,
                expected_sample_ids=expected_ids,
                expected_value_dim=expected_micro_doppler_dim,
            )
            self._micro_doppler_values = mapping
            self._micro_doppler_valid = dict(metadata.pop("valid_by_sample_id"))
            self.micro_doppler_cache_metadata = metadata

    def __len__(self) -> int:
        return len(self.parent_ids)

    def _load_rd_event(self, row: pd.Series) -> tuple[np.ndarray, bool, float, float, bool, bool]:
        present = _as_bool(row["rd_present"])
        if not present:
            empty = np.zeros((self.max_rd_frames_per_point, 2, 21, 384), dtype=np.float32)
            return empty, False, 0.0, 0.0, False, False
        indices = _parse_indices(row["rd_tensor_frame_indices_json"])
        if not indices:
            raise ValueError(f"present RD row has no frame indices: {row['sample_id']}")
        amplitude, phase, times = self._tensor_cache.load(str(row["rd_tensor_path"]))
        if min(indices) < 0 or max(indices) >= amplitude.shape[0]:
            raise ValueError(f"RD frame index out of bounds: {row['sample_id']}")
        original_count = len(indices)
        selected = indices[: self.max_rd_frames_per_point]
        amp = amplitude[np.asarray(selected, dtype=np.int64)]
        ph = phase[np.asarray(selected, dtype=np.int64)]
        amp = np.clip((amp - self.rd_amplitude_center) / self.rd_amplitude_scale, -6.0, 6.0)
        ph = np.clip(ph / np.pi, -1.0, 1.0)
        frames = np.zeros((self.max_rd_frames_per_point, 2, 21, 384), dtype=np.float32)
        frames[: len(selected), 0] = amp
        frames[: len(selected), 1] = ph
        selected_times = times[np.asarray(selected, dtype=np.int64)]
        duration = float(max(0.0, float(selected_times[-1] - selected_times[0]))) if len(selected) > 1 else 0.0
        local_time_non_monotonic = bool(len(selected_times) > 1 and (np.diff(selected_times) < -1.0e-8).any())
        truncated = original_count > len(selected)
        return frames, True, duration, float(original_count), local_time_non_monotonic, truncated

    def __getitem__(self, index: int) -> PointSequence:
        parent_id = self.parent_ids[int(index)]
        group = self._groups[parent_id].iloc[: self.max_sequence_length].copy()
        tr_values: list[torch.Tensor] = []
        tr_masks: list[torch.Tensor] = []
        dt_values: list[float] = []
        dt_valid_values: list[bool] = []
        dt_reason_values: list[int] = []
        motion_values: list[torch.Tensor] = []
        motion_masks: list[torch.Tensor] = []
        rd_values: list[np.ndarray] = []
        rd_masks: list[np.ndarray] = []
        quality_values: list[np.ndarray] = []
        rd_physical_values: list[torch.Tensor] = []
        rd_physical_valid_values: list[bool] = []
        micro_doppler_values: list[torch.Tensor] = []
        micro_doppler_valid_values: list[bool] = []
        labels = set()
        if self.tr_motion_schema == TR_MOTION_SCHEMA:
            motion_state: CausalTRMotionState | CausalTRObservationIncrementState | None = CausalTRMotionState()
        elif self.tr_motion_schema == TR_OBSERVATION_INCREMENT_SCHEMA:
            motion_state = CausalTRObservationIncrementState()
        else:
            motion_state = None
        for _, row in group.iterrows():
            values = parse_feature_json(str(row["tr_feature_values_json"]))
            encoded, missing = encode_tr_feature_values(values)
            tr_values.append(torch.from_numpy(encoded))
            tr_masks.append(torch.from_numpy(missing))
            dt_seconds = float(row["tr_dt_seconds"])
            dt_valid = _as_bool(row["tr_dt_valid"]) if "tr_dt_valid" in row.index else True
            dt_values.append(dt_seconds)
            dt_valid_values.append(dt_valid)
            if "tr_dt_reason" in row.index:
                dt_reason_values.append(encode_dt_reason(row["tr_dt_reason"], dt_valid=dt_valid))
            else:
                dt_reason_values.append(DT_REASON_POSITIVE_DELTA if dt_valid else DT_REASON_ZERO_DELTA)
            if motion_state is None:
                motion = np.zeros((0,), dtype=np.float32)
                motion_valid = np.zeros((0,), dtype=bool)
            elif isinstance(motion_state, CausalTRMotionState):
                motion, motion_valid = motion_state.update(
                    values,
                    dt_seconds=dt_seconds,
                    dt_valid=dt_valid,
                )
            else:
                motion, motion_valid = motion_state.update(values)
            motion_values.append(torch.from_numpy(motion))
            motion_masks.append(torch.from_numpy(motion_valid))
            frames, present, duration, original_count, local_time_non_monotonic, truncated = self._load_rd_event(row)
            rd_values.append(frames)
            frame_mask = np.zeros((self.max_rd_frames_per_point,), dtype=bool)
            frame_mask[: min(int(original_count), self.max_rd_frames_per_point)] = present
            rd_masks.append(frame_mask)
            quality_values.append(
                encode_causal_rd_quality(
                    present=present,
                    original_count=int(original_count),
                    duration_seconds=duration,
                    local_time_non_monotonic=local_time_non_monotonic,
                    truncated=truncated,
                    max_rd_frames_per_point=self.max_rd_frames_per_point,
                )
            )
            sample_id = str(row["sample_id"])
            if self._rd_physical_values is None or self._rd_physical_valid is None:
                rd_physical_values.append(torch.zeros((0,), dtype=torch.float32))
                rd_physical_valid_values.append(False)
            else:
                try:
                    physical_value = self._rd_physical_values[sample_id]
                    physical_valid = self._rd_physical_valid[sample_id]
                except KeyError as exc:
                    raise ValueError(f"RD physical cache lacks Dataset event: {sample_id}") from exc
                rd_physical_values.append(torch.from_numpy(np.asarray(physical_value, dtype=np.float32)))
                rd_physical_valid_values.append(bool(physical_valid))
            if self._micro_doppler_values is None or self._micro_doppler_valid is None:
                micro_doppler_values.append(torch.zeros((0,), dtype=torch.float32))
                micro_doppler_valid_values.append(False)
            else:
                try:
                    micro_value = self._micro_doppler_values[sample_id]
                    micro_valid = self._micro_doppler_valid[sample_id]
                except KeyError as exc:
                    raise ValueError(f"micro-Doppler cache lacks Dataset event: {sample_id}") from exc
                micro_doppler_values.append(
                    torch.from_numpy(np.asarray(micro_value, dtype=np.float32))
                )
                micro_doppler_valid_values.append(bool(micro_valid))
            labels.add(str(row["label_name"]))
        if len(labels) != 1:
            raise ValueError(f"parent sequence has inconsistent labels: {parent_id}")
        label_name = next(iter(labels))
        label_id = int(group.iloc[0]["label_id"])
        if not 0 <= label_id < 5:
            raise ValueError(f"invalid V0 label id: {label_id}")
        return PointSequence(
            parent_sample_id=parent_id,
            track_id=str(group.iloc[0]["track_id"]),
            label_name=label_name,
            split_name=str(group.iloc[0]["split_name"]),
            tr_features=torch.stack(tr_values, dim=0).float(),
            tr_missing_mask=torch.stack(tr_masks, dim=0).float(),
            tr_dt_seconds=torch.tensor(dt_values, dtype=torch.float32),
            tr_dt_valid=torch.tensor(dt_valid_values, dtype=torch.bool),
            tr_dt_reason=torch.tensor(dt_reason_values, dtype=torch.long),
            tr_motion_features=torch.stack(motion_values, dim=0).float(),
            tr_motion_valid_mask=torch.stack(motion_masks, dim=0).bool(),
            rd_frames=torch.from_numpy(np.stack(rd_values, axis=0)).float(),
            rd_frame_mask=torch.from_numpy(np.stack(rd_masks, axis=0)),
            rd_quality=torch.from_numpy(np.stack(quality_values, axis=0)).float(),
            rd_physical_features=torch.stack(rd_physical_values, dim=0).float(),
            rd_physical_valid=torch.tensor(rd_physical_valid_values, dtype=torch.bool),
            micro_doppler_features=torch.stack(micro_doppler_values, dim=0).float(),
            micro_doppler_valid=torch.tensor(micro_doppler_valid_values, dtype=torch.bool),
            label=torch.tensor(label_id, dtype=torch.long),
            event_count=len(group),
            source_name=str(group.iloc[0]["source_name"]),
        )


def collate_point_sequences(items: list[PointSequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    lengths = [int(item.event_count) for item in items]
    max_length = max(lengths)
    value_dim = int(items[0].tr_features.shape[-1])
    missing_dim = int(items[0].tr_missing_mask.shape[-1])
    motion_dim = int(items[0].tr_motion_features.shape[-1])
    rd_physical_dim = int(items[0].rd_physical_features.shape[-1])
    micro_doppler_dim = int(items[0].micro_doppler_features.shape[-1])
    max_frames = int(items[0].rd_frames.shape[1])
    rd_shape = tuple(items[0].rd_frames.shape[2:])
    batch_size = len(items)
    tr_features = torch.zeros((batch_size, max_length, value_dim), dtype=torch.float32)
    tr_missing = torch.zeros((batch_size, max_length, missing_dim), dtype=torch.float32)
    dt = torch.zeros((batch_size, max_length), dtype=torch.float32)
    dt_valid = torch.zeros((batch_size, max_length), dtype=torch.bool)
    dt_reason = torch.full((batch_size, max_length), DT_REASON_PADDING, dtype=torch.long)
    tr_motion = torch.zeros((batch_size, max_length, motion_dim), dtype=torch.float32)
    tr_motion_valid = torch.zeros((batch_size, max_length, motion_dim), dtype=torch.bool)
    rd_frames = torch.zeros((batch_size, max_length, max_frames, *rd_shape), dtype=torch.float32)
    rd_frame_mask = torch.zeros((batch_size, max_length, max_frames), dtype=torch.bool)
    rd_quality = torch.zeros((batch_size, max_length, len(RD_QUALITY_NAMES)), dtype=torch.float32)
    rd_physical = torch.zeros((batch_size, max_length, rd_physical_dim), dtype=torch.float32)
    rd_physical_valid = torch.zeros((batch_size, max_length), dtype=torch.bool)
    micro_doppler = torch.zeros((batch_size, max_length, micro_doppler_dim), dtype=torch.float32)
    micro_doppler_valid = torch.zeros((batch_size, max_length), dtype=torch.bool)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
    labels = torch.empty((batch_size,), dtype=torch.long)
    for batch_index, item in enumerate(items):
        length = int(item.event_count)
        if (
            item.tr_features.shape[-1] != value_dim
            or item.tr_missing_mask.shape[-1] != missing_dim
            or item.tr_motion_features.shape[-1] != motion_dim
            or item.tr_motion_valid_mask.shape != item.tr_motion_features.shape
            or item.rd_physical_features.shape[-1] != rd_physical_dim
            or item.rd_physical_valid.shape != (length,)
            or item.micro_doppler_features.shape[-1] != micro_doppler_dim
            or item.micro_doppler_valid.shape != (length,)
            or item.tr_dt_reason.shape != (length,)
            or tuple(item.rd_frames.shape[2:]) != rd_shape
        ):
            raise ValueError("incompatible point sequence shapes")
        tr_features[batch_index, :length] = item.tr_features
        tr_missing[batch_index, :length] = item.tr_missing_mask
        dt[batch_index, :length] = item.tr_dt_seconds
        dt_valid[batch_index, :length] = item.tr_dt_valid
        dt_reason[batch_index, :length] = item.tr_dt_reason
        tr_motion[batch_index, :length] = item.tr_motion_features
        tr_motion_valid[batch_index, :length] = item.tr_motion_valid_mask
        rd_frames[batch_index, :length] = item.rd_frames
        rd_frame_mask[batch_index, :length] = item.rd_frame_mask
        rd_quality[batch_index, :length] = item.rd_quality
        rd_physical[batch_index, :length] = item.rd_physical_features
        rd_physical_valid[batch_index, :length] = item.rd_physical_valid
        micro_doppler[batch_index, :length] = item.micro_doppler_features
        micro_doppler_valid[batch_index, :length] = item.micro_doppler_valid
        padding_mask[batch_index, :length] = False
        labels[batch_index] = item.label
    result = {
        "tr_features": tr_features,
        "tr_missing_mask": tr_missing,
        "tr_dt_seconds": dt,
        "tr_dt_valid": dt_valid,
        "tr_dt_reason": dt_reason,
        "tr_motion_features": tr_motion,
        "tr_motion_valid_mask": tr_motion_valid,
        "rd_frames": rd_frames,
        "rd_frame_mask": rd_frame_mask,
        "rd_quality": rd_quality,
        "tr_padding_mask": padding_mask,
        "labels": labels,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "parent_sample_ids": [item.parent_sample_id for item in items],
        "track_ids": [item.track_id for item in items],
        "label_names": [item.label_name for item in items],
        "source_names": [item.source_name for item in items],
    }
    if rd_physical_dim:
        if rd_physical_dim != RD_PHYSICAL_VALUE_DIM:
            raise ValueError(f"RD physical feature dimension must be {RD_PHYSICAL_VALUE_DIM}")
        result["rd_physical_features"] = rd_physical
        result["rd_physical_valid"] = rd_physical_valid
    if micro_doppler_dim:
        if micro_doppler_dim not in {
            MICRO_DOPPLER_VALUE_DIM,
            BROADBAND_MICRO_DOPPLER_VALUE_DIM,
        }:
            raise ValueError(
                "micro-Doppler feature dimension does not match a registered contract: "
                f"actual={micro_doppler_dim}, allowed="
                f"{[MICRO_DOPPLER_VALUE_DIM, BROADBAND_MICRO_DOPPLER_VALUE_DIM]}"
            )
        result["micro_doppler_features"] = micro_doppler
        result["micro_doppler_valid"] = micro_doppler_valid
    return result


def build_point_dataloader(
    dataset: CQ08PointSequenceDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build the v3 parent-sequence DataLoader with the isolated collate."""

    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=collate_point_sequences,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


__all__ = [
    "CQ08PointSequenceDataset",
    "DT_REASON_FIRST_EVENT",
    "DT_REASON_NAMES",
    "DT_REASON_PADDING",
    "DT_REASON_POSITIVE_DELTA",
    "DT_REASON_TO_ID",
    "DT_REASON_ZERO_DELTA",
    "PointSequence",
    "RD_QUALITY_NAMES",
    "RD_QUALITY_SCHEMA",
    "REQUIRED_INDEX_COLUMNS",
    "build_point_dataloader",
    "collate_point_sequences",
    "encode_causal_rd_quality",
    "encode_dt_reason",
    "require_causal_rd_quality_schema",
]
