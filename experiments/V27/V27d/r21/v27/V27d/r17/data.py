"""V27d/r17 的 OriData Excel TR-only 数据处理。

``bz`` 的列结构提供 parent 标注，``tck`` 的列结构提供逐点 TR。
parent 身份严格使用 ``source_suffix + StartTimer + Batch``；不把同一
Batch 下的不同 StartTimer 轨迹拼接。Excel Noise 不进入模型，SNR 始终
按缺失处理。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from radar_ln3.features import (
    encode_causal_tr_observation_increment_sequence,
    encode_tr_feature_values,
)
from radar_ln3.data import encode_causal_rd_quality
from radar_ln3.known_gate_features import extract_causal_course_circular_incoherence

from .protocol import R17_DT_REASON, R17Config, canonical_sha256, data_path, sha256_file


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"(null)", "null", "none", "nan"}:
        return None
    return text


def _float(value: Any) -> float | None:
    text = _text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


























def load_r17_data_manifest(project_root: str | Path, config: R17Config) -> dict[str, Any]:
    path = data_path(project_root, config.artifacts["data_manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("r17 processed data manifest is not an object")
    declared = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if declared != canonical_sha256(body):
        raise ValueError("r17 processed data manifest hash differs")
    for key in ("parents_path", "events_path"):
        if not payload.get(f"{key.split('_')[0]}_sha256") == sha256_file(data_path(project_root, payload[key])):
            raise ValueError(f"r17 {key} hash differs")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"r17 JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _event_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"range_m": row.get("range_m"), "radial_speed_mps": row.get("radial_speed_mps"), "elevation_deg": row.get("elevation_deg"), "azimuth_deg": row.get("azimuth_deg"), "course_deg": row.get("course_deg"), "speed_mps": row.get("speed_mps"), "amplitude_db": row.get("amplitude_db"), "snr_db": None, "height_m": row.get("height_m"), "latitude": row.get("latitude"), "longitude": row.get("longitude")}


def _l2_robust_values(
    values: Mapping[str, Any],
    config: R17Config,
) -> tuple[dict[str, Any], bool, bool]:
    """Apply the frozen V27c-r23 robust-control input contract.

    The r23 endpoint was trained after clipping the physical ``range_m`` and
    reported ``speed_mps`` fields.  The clipping is deliberately applied to
    the raw physical values before both the ordinary TR encoder and the
    causal observation-increment encoder.  Radial speed is not clipped here;
    it was not one of r23's registered clipped fields.
    """

    clipping = config.data.get("l2_robust_clipping")
    if not isinstance(clipping, Mapping):
        raise ValueError("r17 L2 robust clipping provenance is missing")
    result = dict(values)
    changed_range = False
    changed_speed = False
    for name, marker in (("range_m", "range"), ("speed_mps", "speed")):
        bounds = clipping.get(name)
        if not isinstance(bounds, Mapping):
            raise ValueError(f"r17 L2 robust clipping lacks {name} bounds")
        value = _float(result.get(name))
        if value is None:
            continue
        lower = _float(bounds.get("lower"))
        upper = _float(bounds.get("upper"))
        if lower is None or upper is None or lower < 0.0 or upper < lower:
            raise ValueError(f"r17 L2 robust clipping bounds are invalid: {name}")
        clipped = float(np.clip(value, lower, upper))
        result[name] = clipped
        if clipped != value:
            if marker == "range":
                changed_range = True
            else:
                changed_speed = True
    return result, changed_range, changed_speed


@dataclass(frozen=True)
class R17Sequence:
    parent_sample_id: str
    source_suffix: str
    label_name: str
    label_name_raw: str
    task: str
    split_name: str
    event_rows: tuple[dict[str, Any], ...]
    tr_features: torch.Tensor
    tr_missing_mask: torch.Tensor
    tr_dt_seconds: torch.Tensor
    tr_dt_valid: torch.Tensor
    tr_dt_reason: torch.Tensor
    observation_increment_values: torch.Tensor
    course_features: torch.Tensor
    course_valid: torch.Tensor
    course_support: torch.Tensor
    rd_quality: torch.Tensor
    rd_placeholder_frames: int
    l2_range_clipped_count: int
    l2_speed_clipped_count: int
    l1_label: int
    l2_label: int
    raw_event_count: int
    model_truncation_applied: bool

    @property
    def event_count(self) -> int:
        return len(self.event_rows)


class R17Dataset(Dataset[R17Sequence]):
    def __init__(self, project_root: str | Path, config: R17Config, *, split_name: str | None = None, task: str = "l1", parent_ids: Sequence[str] | None = None) -> None:
        if task not in {"l1", "l2"}:
            raise ValueError("r17 task must be l1 or l2")
        manifest = load_r17_data_manifest(project_root, config)
        root = Path(project_root).resolve()
        parents = _read_jsonl(data_path(root, manifest["parents_path"]))
        events = _read_jsonl(data_path(root, manifest["events_path"]))
        by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in events:
            by_parent[str(row["parent_sample_id"])].append(row)
        requested = None if parent_ids is None else {str(value) for value in parent_ids}
        selected: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        max_length = int(config.data["max_sequence_length"])
        for parent in sorted(parents, key=lambda row: str(row["parent_sample_id"])):
            parent_id = str(parent["parent_sample_id"])
            if split_name is not None and str(parent["split_name"]) != str(split_name):
                continue
            if requested is not None and parent_id not in requested:
                continue
            if task == "l2" and int(parent["l2_label_id"]) < 0:
                continue
            raw_rows = sorted(by_parent.get(parent_id, []), key=lambda row: int(row["causal_event_index"]))
            if len(raw_rows) != int(parent["raw_event_count"]):
                raise ValueError(f"r17 raw event count differs: {parent_id}")
            selected.append((parent, raw_rows[:max_length]))
        if requested is not None and requested - {str(item[0]["parent_sample_id"]) for item in selected}:
            raise ValueError("r17 requested parent ids are unavailable")
        if not selected:
            raise ValueError(f"r17 dataset is empty: task={task}, split={split_name}")
        self.config = config
        self.task = task
        self.split_name = None if split_name is None else str(split_name)
        self.parent_ids = [str(parent["parent_sample_id"]) for parent, _ in selected]
        self._items = [self._build(parent, rows) for parent, rows in selected]

    def _build(self, parent: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> R17Sequence:
        values: list[np.ndarray] = []
        missing: list[np.ndarray] = []
        raw_values: list[dict[str, Any]] = []
        courses: list[float | None] = []
        rd_quality_values: list[np.ndarray] = []
        range_clipped_count = 0
        speed_clipped_count = 0
        for row in rows:
            feature_values = _event_values(row)
            if self.task == "l2":
                feature_values, range_changed, speed_changed = _l2_robust_values(
                    feature_values, self.config
                )
                range_clipped_count += int(range_changed)
                speed_clipped_count += int(speed_changed)
            encoded, mask = encode_tr_feature_values(feature_values)
            values.append(encoded)
            missing.append(mask)
            raw_values.append(feature_values)
            courses.append(_float(row.get("course_deg")))
            rd_quality_values.append(
                encode_causal_rd_quality(
                    present=False,
                    original_count=0,
                    duration_seconds=0.0,
                    local_time_non_monotonic=False,
                    truncated=False,
                    max_rd_frames_per_point=int(self.config.data["rd_placeholder_frames"]),
                )
            )
        observation, valid = encode_causal_tr_observation_increment_sequence(raw_values)
        observation = np.where(valid, observation, 0.0).astype(np.float32, copy=False)
        course_values, course_valid = extract_causal_course_circular_incoherence(courses)
        support = np.zeros(len(rows), dtype=np.int64)
        count = 0
        for index, course in enumerate(courses):
            if index > 0 and course is not None:
                count += 1
            support[index] = count
        label = str(parent["label_name"])
        l1_label = int(parent["l1_label_id"])
        l2_label = int(parent["l2_label_id"])
        return R17Sequence(
            parent_sample_id=str(parent["parent_sample_id"]),
            source_suffix=str(parent["source_suffix"]),
            label_name=label,
            label_name_raw=str(parent["label_name_raw"]),
            task=self.task,
            split_name=str(parent["split_name"]),
            event_rows=tuple(dict(row) for row in rows),
            tr_features=torch.from_numpy(np.stack(values).astype(np.float32)),
            tr_missing_mask=torch.from_numpy(np.stack(missing).astype(np.float32)),
            tr_dt_seconds=torch.tensor(
                [float(row["dt_seconds"]) for row in rows], dtype=torch.float32
            ),
            tr_dt_valid=torch.tensor(
                [bool(row["dt_valid"]) for row in rows], dtype=torch.bool
            ),
            tr_dt_reason=torch.tensor(
                [R17_DT_REASON[str(row["dt_reason"])] for row in rows], dtype=torch.long
            ),
            observation_increment_values=torch.from_numpy(observation),
            course_features=torch.from_numpy(course_values.astype(np.float32)),
            course_valid=torch.from_numpy(course_valid.astype(np.bool_)),
            course_support=torch.from_numpy(support),
            rd_quality=torch.from_numpy(np.stack(rd_quality_values).astype(np.float32)),
            rd_placeholder_frames=int(self.config.data["rd_placeholder_frames"]),
            l2_range_clipped_count=range_clipped_count,
            l2_speed_clipped_count=speed_clipped_count,
            l1_label=l1_label,
            l2_label=l2_label,
            raw_event_count=int(parent["raw_event_count"]),
            model_truncation_applied=bool(parent["model_truncation_applied"]),
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> R17Sequence:
        if isinstance(index, bool) or not 0 <= int(index) < len(self):
            raise IndexError("r17 dataset index is out of range")
        return self._items[int(index)]


def collate_r17(items: list[R17Sequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("r17 cannot collate an empty batch")
    task = items[0].task
    if task not in {"l1", "l2"} or any(item.task != task for item in items):
        raise ValueError("r17 cannot mix L1 and L2 sequences")
    frame_counts = {int(item.rd_placeholder_frames) for item in items}
    if frame_counts != {1}:
        raise ValueError("r17 TR-only RD placeholder frame count must be exactly one")
    batch_size = len(items)
    width = max(item.event_count for item in items)
    tr = torch.zeros((batch_size, width, 14), dtype=torch.float32)
    missing = torch.zeros((batch_size, width, 11), dtype=torch.float32)
    dt = torch.zeros((batch_size, width), dtype=torch.float32)
    dt_valid = torch.zeros((batch_size, width), dtype=torch.bool)
    reason = torch.zeros((batch_size, width), dtype=torch.long)
    observation = torch.zeros((batch_size, width, 9), dtype=torch.float32)
    course = torch.zeros((batch_size, width, 1), dtype=torch.float32)
    course_valid = torch.zeros((batch_size, width, 1), dtype=torch.bool)
    course_support = torch.zeros((batch_size, width), dtype=torch.long)
    quality = torch.zeros((batch_size, width, 7), dtype=torch.float32)
    padding = torch.ones((batch_size, width), dtype=torch.bool)
    l1_labels = torch.empty(batch_size, dtype=torch.long)
    l2_labels = torch.empty(batch_size, dtype=torch.long)
    for index, item in enumerate(items):
        length = item.event_count
        tr[index, :length] = item.tr_features
        missing[index, :length] = item.tr_missing_mask
        dt[index, :length] = item.tr_dt_seconds
        dt_valid[index, :length] = item.tr_dt_valid
        reason[index, :length] = item.tr_dt_reason
        observation[index, :length] = item.observation_increment_values
        course[index, :length] = item.course_features
        course_valid[index, :length] = item.course_valid
        course_support[index, :length] = item.course_support
        quality[index, :length] = item.rd_quality
        padding[index, :length] = False
        l1_labels[index] = item.l1_label
        l2_labels[index] = item.l2_label
    rd_frames = torch.zeros((batch_size, width, 1, 2, 21, 384), dtype=torch.float32)
    rd_mask = torch.zeros((batch_size, width, 1), dtype=torch.bool)
    if bool(quality.masked_select(padding.unsqueeze(-1)).ne(0).any()):
        raise ValueError("r17 padded RD quality must be zero")
    active_quality = quality[~padding]
    expected_missing = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=quality.dtype
    )
    if active_quality.numel() and not torch.equal(
        active_quality, expected_missing.expand_as(active_quality)
    ):
        raise ValueError("r17 active TR-only RD quality must encode rd_missing=1")
    return {"task": task, "tr_features": tr, "tr_missing_mask": missing, "tr_dt_seconds": dt, "tr_dt_valid": dt_valid, "tr_dt_reason": reason, "tr_padding_mask": padding, "rd_frames": rd_frames, "rd_frame_mask": rd_mask, "rd_quality": quality, "course_features": course, "course_valid": course_valid, "course_support": course_support, "observation_increment_values": observation, "rd_physical_features": torch.zeros((batch_size, width, 4)), "rd_physical_valid": torch.zeros((batch_size, width), dtype=torch.bool), "micro_doppler_features": torch.zeros((batch_size, width, 8)), "micro_doppler_valid": torch.zeros((batch_size, width), dtype=torch.bool), "l1_labels": l1_labels, "l2_labels": l2_labels, "labels": l1_labels if task == "l1" else l2_labels, "lengths": torch.tensor([item.event_count for item in items], dtype=torch.long), "parent_sample_ids": [item.parent_sample_id for item in items], "label_names": [item.label_name for item in items], "label_names_raw": [item.label_name_raw for item in items], "raw_event_counts": torch.tensor([item.raw_event_count for item in items], dtype=torch.long), "model_truncation_applied": [item.model_truncation_applied for item in items], "l2_range_clipped_count": torch.tensor([item.l2_range_clipped_count for item in items], dtype=torch.long), "l2_speed_clipped_count": torch.tensor([item.l2_speed_clipped_count for item in items], dtype=torch.long)}








__all__ = ["R17Dataset", "R17Sequence", "collate_r17", "load_r17_data_manifest"]