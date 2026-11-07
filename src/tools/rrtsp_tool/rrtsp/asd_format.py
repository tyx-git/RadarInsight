# -*- coding: utf-8 -*-
"""AeroSight Data Package (.asd) format.

An ``.asd`` file is a ZIP container with one causal track/event sequence.
The package stores raw physical RD frames. Consumers canonicalize those
frames with their own registered physical-axis converter before model input.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
import re
import zipfile

import numpy as np


PACKAGE_SCHEMA = "RADAR-LN3-ASD-PACKAGE-V1"
TR_SCHEMA = "RADAR-LN3-ASD-TR-V1"
RD_SCHEMA = "RADAR-LN3-ASD-RD-RAW-PHYSICAL-V1"
CANONICALIZATION_VERSION = "RADAR-LN3-CANONICAL-RD-INTERP-V1"
EXTENSION = ".asd"
MAX_RD_FRAMES_PER_EVENT = 12
CANONICAL_RD_SHAPE = [21, 384]
MICRO_DOPPLER_EVIDENCE_SCHEMA = "RADAR-LN3-ASD-MICRO-DOPPLER-EVIDENCE-V1"
MICRO_DOPPLER_FRAME_DIM = 23
STABLE8_DIM = 8
STABLE8_INDICES = [1, 2, 3, 7, 11, 13, 14, 22]

TR_FIELDS = (
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_speed_mps",
    "amplitude_db",
    "snr_db",
    "speed_mps",
    "course_deg",
)
FORBIDDEN_DT_FIELDS = ("dt_seconds", "dt_valid", "dt_reason")

COORDINATE_SYSTEM = {
    "frame": "radar_relative",
    "angle_unit": "degree_signed",
    "azimuth": {
        "zero": "radar_forward",
        "positive": "clockwise_right",
        "range": [-180.0, 180.0],
    },
    "course": {
        "zero": "velocity_direction",
        "positive": "clockwise_right",
        "range": [-180.0, 180.0],
    },
    "cartesian": {"x": "forward", "y": "left", "z": "up"},
}

_EVENT_RE = re.compile(r"(?:^|/)events/(?P<index>\d{4})(?:/|$)")
_RD_RE = re.compile(
    r"(?:^|/)events/(?P<event>\d{4})/rd/(?P<frame>\d{4})(?:/|$)"
)


class AsdError(Exception):
    """Invalid ASD package or package input."""


def _finite_float(value, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AsdError(f"{name} 必须是数值: {value!r}") from exc
    if not math.isfinite(number):
        raise AsdError(f"{name} 必须是有限数值: {value!r}")
    return number


def _axis_values(value, *, name: str, expected: int | None = None) -> list[float]:
    if value is None:
        raise AsdError(f"{name} 不能为空")
    try:
        values = [_finite_float(item, name=name) for item in value]
    except TypeError as exc:
        raise AsdError(f"{name} 必须是数值数组") from exc
    if len(values) < 2:
        raise AsdError(f"{name} 至少需要两个点")
    if expected is not None and len(values) != int(expected):
        raise AsdError(f"{name} 长度 {len(values)} 与 shape 不一致：{expected}")
    differences = np.diff(np.asarray(values, dtype=np.float64))
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise AsdError(f"{name} 必须严格递增或严格递减")
    return values


def _check_2d(name: str, value) -> tuple[int, int]:
    array = np.asarray(value)
    shape = tuple(int(item) for item in array.shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise AsdError(f"{name} 必须是非空二维矩阵，实际为 {shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise AsdError(f"{name} 必须是有限数值矩阵")
    return shape


def _json_mapping(value) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AsdError("source/metadata 必须是 JSON 对象")
    return dict(value)


def _frame_item(item):
    """Normalize the tuple and mapping forms accepted by ``AsdWriter``."""

    if isinstance(item, Mapping):
        frame_time = item.get("time_seconds", item.get("time"))
        amplitude = item.get("amplitude", item.get("amp"))
        phase = item.get("phase")
        axes = dict(item.get("axes") or {})
        for key in (
            "range_axis",
            "range_axis_m",
            "velocity_axis",
            "velocity_axis_mps",
            "target_range_m",
            "source_path",
            "raw_frame_index",
            "mat_frame_number",
            "track_point_index",
            "rd_observation_duration_seconds",
        ):
            if key in item and key not in axes:
                axes[key] = item[key]
        return frame_time, amplitude, phase, axes
    try:
        size = len(item)
    except TypeError as exc:
        raise AsdError("RD 帧必须是 mapping 或 3/4 元素序列") from exc
    if size == 3:
        frame_time, amplitude, phase = item
        axes = {}
    elif size == 4:
        frame_time, amplitude, phase, axes = item
        axes = dict(axes or {})
    else:
        raise AsdError("RD 帧必须是 3 或 4 元素序列")
    return frame_time, amplitude, phase, axes


def _validate_stable_evidence(
    value: Mapping | None,
    *,
    event_index: int,
    frame_count: int,
) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AsdError("stable_evidence 必须是 JSON 对象")

    def array(name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        if name not in value:
            raise AsdError(f"stable_evidence 缺少字段: {name}")
        result = np.asarray(value[name], dtype=dtype)
        if result.shape != shape:
            raise AsdError(
                f"stable_evidence.{name} shape 无效: "
                f"expected={shape}, actual={result.shape}"
            )
        return result

    stable_values = array("stable8_values", (STABLE8_DIM,), np.float32)
    stable_frame_values = array(
        "stable8_frame_values",
        (MAX_RD_FRAMES_PER_EVENT, STABLE8_DIM),
        np.float32,
    )
    stable_frame_mask = array(
        "stable8_frame_mask", (MAX_RD_FRAMES_PER_EVENT,), np.bool_
    )
    micro_values = array(
        "micro_doppler_frame_values",
        (MAX_RD_FRAMES_PER_EVENT, MICRO_DOPPLER_FRAME_DIM),
        np.float32,
    )
    micro_frame_mask = array(
        "micro_doppler_frame_mask", (MAX_RD_FRAMES_PER_EVENT,), np.bool_
    )
    micro_reliable = array(
        "micro_doppler_reliable", (MAX_RD_FRAMES_PER_EVENT,), np.bool_
    )
    raw_indices = array(
        "raw_frame_indices", (MAX_RD_FRAMES_PER_EVENT,), np.int32
    )
    stable_valid = bool(value.get("stable8_valid", False))
    present = bool(value.get("present", frame_count > 0))
    if frame_count > MAX_RD_FRAMES_PER_EVENT:
        raise AsdError("stable_evidence frame 数超过 12")
    if not np.isfinite(stable_values).all() or not np.isfinite(stable_frame_values).all():
        raise AsdError("stable_evidence 数值必须有限")
    if not np.isfinite(micro_values).all():
        raise AsdError("micro-Doppler evidence 数值必须有限")
    if (
        np.any(stable_values < 0.0)
        or np.any(stable_values > 1.0)
        or np.any(stable_frame_values < 0.0)
        or np.any(stable_frame_values > 1.0)
        or np.any(micro_values < 0.0)
        or np.any(micro_values > 1.0)
    ):
        raise AsdError("stable_evidence 数值必须位于 [0,1]")
    if np.any(micro_reliable & ~micro_frame_mask):
        raise AsdError("reliable frame 必须属于 active frame")
    if not np.array_equal(stable_frame_mask, micro_reliable):
        raise AsdError("stable8_frame_mask 必须与 micro_doppler_reliable 一致")
    if stable_valid != bool(stable_frame_mask.any()):
        raise AsdError("stable8_valid 与 stable8_frame_mask 不一致")
    if np.any(stable_frame_values[~stable_frame_mask] != 0.0):
        raise AsdError("非 reliable Stable8 frame 必须为零")
    if np.any(micro_values[~micro_reliable] != 0.0):
        raise AsdError("非 reliable Micro-Doppler frame 必须为零")
    if np.any(raw_indices[~micro_frame_mask] != -1):
        raise AsdError("inactive raw frame index 必须为 -1")
    active_indices = raw_indices[micro_frame_mask]
    if active_indices.size and np.any(active_indices < 0):
        raise AsdError("active raw frame index 不能为负数")
    if int(micro_frame_mask.sum()) != int(frame_count):
        raise AsdError(
            "stable_evidence active frame 数与 RD frame 数不一致: "
            f"evidence={int(micro_frame_mask.sum())}, rd={frame_count}"
        )
    return {
        "schema_version": MICRO_DOPPLER_EVIDENCE_SCHEMA,
        "event_index": int(value.get("event_index", event_index)),
        "frame_cap": MAX_RD_FRAMES_PER_EVENT,
        "stable8_indices": list(STABLE8_INDICES),
        "stable8_input_mode": "raw_01",
        "present": present,
        "stable8_values": stable_values.tolist(),
        "stable8_valid": stable_valid,
        "stable8_frame_values": stable_frame_values.tolist(),
        "stable8_frame_mask": stable_frame_mask.tolist(),
        "micro_doppler_frame_values": micro_values.tolist(),
        "micro_doppler_frame_mask": micro_frame_mask.tolist(),
        "micro_doppler_reliable": micro_reliable.tolist(),
        "raw_frame_indices": raw_indices.tolist(),
    }


class AsdWriter:
    """Write the RadarInsight-compatible ASD package."""

    def __init__(
        self,
        path: str,
        kind: str = "track",
        label=None,
        label_name: str = "UnknownTarget",
        source: dict | None = None,
        velocity_axis_id: str | None = None,
        *,
        parent_sample_id: str | None = None,
        label_id: int | None = None,
        tr_amplitude_semantics: str = "linear_magnitude_legacy_cq08",
        rd_amplitude_mode: str = "db",
    ):
        self.path = str(path)
        self.kind = str(kind)
        self.label = label
        self.label_name = str(label_name)
        self.source = _json_mapping(source)
        self.parent_sample_id = str(
            parent_sample_id
            or self.source.get("parent_sample_id")
            or f"{self.source.get('session', 'aerosight')}|track|"
            f"{self.source.get('track_id', '0')}"
        )
        self.label_id = label_id
        if self.label_id is None and isinstance(label, int) and not isinstance(label, bool):
            self.label_id = int(label)
        self.tr_amplitude_semantics = str(tr_amplitude_semantics)
        self.rd_amplitude_mode = str(rd_amplitude_mode).strip().lower()
        if self.rd_amplitude_mode not in {"db", "magnitude_db", "power_db", "raw"}:
            raise AsdError(f"不支持的 rd_amplitude_mode: {rd_amplitude_mode!r}")
        if velocity_axis_id not in (None, ""):
            raise AsdError(
                "新 ASD 包不使用 velocity_axis_id；请在每个 RD frame 提供完整物理轴"
            )
        self._events: list[dict] = []

    def add_event(
        self,
        timestamp: float | None = None,
        tr: dict | None = None,
        rd_frames: list | None = None,
        *,
        timestamp_seconds: float | None = None,
        event_index: int | None = None,
        track_point_index: int | None = None,
        source: dict | None = None,
        stable_evidence: Mapping | None = None,
    ) -> None:
        if timestamp_seconds is None:
            timestamp_seconds = timestamp
        elif timestamp is not None and not math.isclose(
            float(timestamp), float(timestamp_seconds), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise AsdError("timestamp 与 timestamp_seconds 不一致")
        timestamp_value = _finite_float(timestamp_seconds, name="timestamp_seconds")
        if not isinstance(tr, Mapping):
            raise AsdError("tr 必须是 JSON 对象")
        tr_values = dict(tr)
        unknown = set(tr_values) - set(TR_FIELDS) - set(FORBIDDEN_DT_FIELDS)
        if unknown:
            raise AsdError(f"未知 TR 字段: {sorted(unknown)}")
        forbidden = sorted(set(tr_values) & set(FORBIDDEN_DT_FIELDS))
        if forbidden:
            raise AsdError(f"TR 不允许由数据提供端写入 dt 字段: {forbidden}")
        missing = [name for name in TR_FIELDS if name not in tr_values]
        if missing:
            raise AsdError(f"TR 缺少必需字段: {missing}")
        tr_values = {
            name: _finite_float(tr_values[name], name=f"tr.{name}")
            for name in TR_FIELDS
        }

        index = len(self._events) if event_index is None else int(event_index)
        if index != len(self._events):
            raise AsdError(
                f"event_index 必须从 0 连续递增: expected={len(self._events)}, actual={index}"
            )
        point_index = index + 1 if track_point_index is None else int(track_point_index)
        if point_index <= 0:
            raise AsdError("track_point_index 必须从 1 开始")

        frames: list[dict] = []
        for position, item in enumerate(rd_frames or []):
            frame_time, amplitude, phase, axes = _frame_item(item)
            frame_time_value = _finite_float(frame_time, name="RD time_seconds")
            amp_array = np.asarray(amplitude, dtype=np.float32)
            phase_array = np.asarray(phase, dtype=np.float32)
            shape = _check_2d("RD amplitude", amp_array)
            if tuple(phase_array.shape) != shape:
                raise AsdError("RD amplitude 和 phase shape 不一致")
            if not np.isfinite(phase_array).all():
                raise AsdError("RD phase 必须是有限数值")

            range_axis = axes.get("range_axis_m", axes.get("range_axis"))
            velocity_axis = axes.get(
                "velocity_axis_mps", axes.get("velocity_axis")
            )
            range_axis = _axis_values(
                range_axis, name="range_axis_m", expected=shape[0]
            )
            velocity_axis = _axis_values(
                velocity_axis, name="velocity_axis_mps", expected=shape[1]
            )
            target_range = _finite_float(
                axes.get("target_range_m"), name="target_range_m"
            )
            metadata = {
                "time_seconds": frame_time_value,
                "shape": list(shape),
                "dtype": "float32",
                "byte_order": "little-endian",
                "layout": "C-order",
                "amplitude_unit": "dB"
                if self.rd_amplitude_mode == "db"
                else "linear",
                "amplitude_mode": self.rd_amplitude_mode,
                "phase_unit": "rad",
                "range_axis_m": range_axis,
                "velocity_axis_mps": velocity_axis,
                "target_range_m": target_range,
                "canonicalization_version": CANONICALIZATION_VERSION,
                "event_index": index,
                "frame_position": position,
                "track_point_index": point_index,
            }
            for key in (
                "source_path",
                "raw_frame_index",
                "mat_frame_number",
                "rd_observation_duration_seconds",
            ):
                if key in axes and axes[key] is not None:
                    metadata[key] = axes[key]
            frames.append(
                {
                    "metadata": metadata,
                    "amplitude": amp_array,
                    "phase": phase_array,
                }
            )
        if len(frames) > MAX_RD_FRAMES_PER_EVENT:
            raise AsdError(
                f"单事件 RD frame 数超过 {MAX_RD_FRAMES_PER_EVENT}，请先按协议截断"
            )
        evidence = _validate_stable_evidence(
            stable_evidence,
            event_index=index,
            frame_count=len(frames),
        )
        self._events.append(
            {
                "event_index": index,
                "track_point_index": point_index,
                "timestamp_seconds": timestamp_value,
                "tr": tr_values,
                "rd": frames,
                "source": _json_mapping(source),
                "stable_evidence": evidence,
            }
        )

    def finish(self) -> str:
        if not self._events:
            raise AsdError("没有任何事件，无法封包")
        shapes = {
            tuple(frame["metadata"]["shape"])
            for event in self._events
            for frame in event["rd"]
        }
        shape_counts: dict[str, int] = {}
        for shape in shapes:
            shape_counts["x".join(str(value) for value in shape)] = sum(
                1
                for event in self._events
                for frame in event["rd"]
                if tuple(frame["metadata"]["shape"]) == shape
            )
        source = dict(self.source)
        source.setdefault("parent_sample_id", self.parent_sample_id)
        source.setdefault("dataset", "AeroSight")
        source.setdefault("session", "aerosight")
        manifest = {
            "schema_version": PACKAGE_SCHEMA,
            "package_type": "AeroSight ASD raw physical RD",
            "parent_sample_id": self.parent_sample_id,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "timestamp_unit": "s",
            "event_index_basis": "zero_based_causal_order",
            "source": source,
            "coordinate_system": COORDINATE_SYSTEM,
            "tr": {
                "schema_version": TR_SCHEMA,
                "fields": list(TR_FIELDS),
                "numeric_only": True,
                "amplitude_semantics": self.tr_amplitude_semantics,
                "dt_fields_forbidden": list(FORBIDDEN_DT_FIELDS),
            },
            "rd": {
                "schema_version": RD_SCHEMA,
                "representation": "raw_physical_frame",
                "pairing": "event/rd/frame",
                "frame_order": "frame_position_ascending",
                "time_semantics": "parent_relative_cpi_elapsed_seconds",
                "max_frames_per_event": MAX_RD_FRAMES_PER_EVENT,
                "amplitude_mode": self.rd_amplitude_mode,
                "amplitude_unit": "dB"
                if self.rd_amplitude_mode == "db"
                else "linear",
                "phase_unit": "rad",
                "dtype": "float32",
                "byte_order": "little-endian",
                "layout": "C-order",
                "canonicalization_version": CANONICALIZATION_VERSION,
                "canonical_shape": CANONICAL_RD_SHAPE,
            },
            "stable_evidence": {
                "available": any(event["stable_evidence"] for event in self._events),
                "schema_version": MICRO_DOPPLER_EVIDENCE_SCHEMA,
                "frame_value_dim": MICRO_DOPPLER_FRAME_DIM,
                "stable8_dim": STABLE8_DIM,
                "stable8_indices": list(STABLE8_INDICES),
                "frame_cap": MAX_RD_FRAMES_PER_EVENT,
                "value_mode": "raw_01",
                "reliable_frame_zero_policy": True,
            },
            "counts": {
                "event_count": len(self._events),
                "rd_event_count": sum(bool(event["rd"]) for event in self._events),
                "rd_frame_count": sum(len(event["rd"]) for event in self._events),
                "rd_frame_shape_counts": shape_counts,
            },
            "provenance": {"generated_by": "AeroSight AsdWriter"},
        }
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with zipfile.ZipFile(
            self.path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            for event in self._events:
                event_index = int(event["event_index"])
                base = f"events/{event_index:04d}"
                tr_payload = {
                    "schema_version": TR_SCHEMA,
                    "event_index": event_index,
                    "track_point_index": int(event["track_point_index"]),
                    "timestamp_seconds": event["timestamp_seconds"],
                    "tr": event["tr"],
                }
                if event["source"]:
                    tr_payload["source"] = event["source"]
                archive.writestr(
                    f"{base}/tr.json",
                    json.dumps(tr_payload, ensure_ascii=False, indent=2) + "\n",
                )
                if event["stable_evidence"] is not None:
                    archive.writestr(
                        f"{base}/evidence.json",
                        json.dumps(
                            event["stable_evidence"],
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                    )
                for frame in event["rd"]:
                    metadata = dict(frame["metadata"])
                    position = int(metadata["frame_position"])
                    frame_root = f"{base}/rd/{position:04d}"
                    archive.writestr(
                        f"{frame_root}/meta.json",
                        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    )
                    archive.writestr(
                        f"{frame_root}/amplitude.bin",
                        frame["amplitude"].astype("<f4", copy=False).tobytes(
                            order="C"
                        ),
                    )
                    archive.writestr(
                        f"{frame_root}/phase.bin",
                        frame["phase"].astype("<f4", copy=False).tobytes(
                            order="C"
                        ),
                    )
        return self.path


class AsdData:
    """Read both the current ASD schema and the previous local format."""

    def __init__(self, path: str):
        self.path = str(path)
        if not zipfile.is_zipfile(path):
            raise AsdError(f"不是 .asd 数据包: {path}")
        self._zip = zipfile.ZipFile(path, "r")
        try:
            self.manifest = json.loads(
                self._zip.read("manifest.json").decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._zip.close()
            raise AsdError("manifest.json 无效或缺失") from exc
        self.schema_version = self.manifest.get("schema_version")
        self.legacy = self.schema_version != PACKAGE_SCHEMA
        if self.legacy and self.manifest.get("format") != "AeroSight Data":
            self._zip.close()
            raise AsdError("manifest 格式标识不匹配")
        if self.legacy:
            self.n_events = int(self.manifest["n_events"])
        else:
            counts = self.manifest.get("counts") or {}
            self.n_events = int(counts.get("event_count", 0))
            if self.n_events <= 0:
                raise AsdError("manifest counts.event_count 无效")
            if self.manifest.get("timestamp_unit") != "s":
                raise AsdError("当前 ASD timestamp_unit 必须为 s")
        self._rd_indices = self._discover_rd_indices()

    def _discover_rd_indices(self) -> dict[int, list[int]]:
        events: set[int] = set()
        frames: dict[int, set[int]] = {}
        for name in self._zip.namelist():
            event_match = _EVENT_RE.search(name)
            if event_match:
                events.add(int(event_match.group("index")))
            rd_match = _RD_RE.search(name)
            if rd_match:
                event = int(rd_match.group("event"))
                frames.setdefault(event, set()).add(int(rd_match.group("frame")))
        if not self.legacy and sorted(events) != list(range(self.n_events)):
            raise AsdError("events index 必须从 0000 连续递增")
        result: dict[int, list[int]] = {}
        for event, values in frames.items():
            ordered = sorted(values)
            if ordered != list(range(len(ordered))):
                raise AsdError(f"event {event:04d} RD frame index 不连续")
            if len(ordered) > MAX_RD_FRAMES_PER_EVENT:
                raise AsdError(
                    f"event {event:04d} RD frame 数超过 {MAX_RD_FRAMES_PER_EVENT}"
                )
            result[event] = ordered
        return result

    @property
    def label_name(self) -> str:
        return str(self.manifest.get("label_name", "-"))

    @property
    def kind(self) -> str:
        return str(
            self.manifest.get("package_type", self.manifest.get("kind", "-"))
        )

    @property
    def coordinate_system(self) -> dict:
        return dict(self.manifest.get("coordinate_system", COORDINATE_SYSTEM))

    @property
    def source(self) -> dict:
        value = self.manifest.get("source", {})
        return dict(value) if isinstance(value, Mapping) else {}

    def summary(self) -> str:
        counts = self.manifest.get("counts") or {}
        nrd = int(
            counts.get(
                "rd_frame_count",
                self.manifest.get(
                    "n_rd_frames",
                    sum(self.rd_count(i) for i in range(self.n_events)),
                ),
            )
        )
        rd_text = f" / {nrd} RD帧" if nrd else ""
        return f"{os.path.basename(self.path)}  |  {self.label_name}  |  {self.n_events} 事件{rd_text}"

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def read_event(self, event_i: int) -> dict:
        if int(event_i) < 0 or int(event_i) >= self.n_events:
            raise AsdError(f"event index 越界: {event_i}")
        base = f"events/{int(event_i):04d}"
        try:
            tr_payload = json.loads(
                self._zip.read(f"{base}/tr.json").decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AsdError(f"TR JSON 无效: {base}/tr.json") from exc
        timestamp = tr_payload.get(
            "timestamp_seconds", tr_payload.get("timestamp")
        )
        timestamp_value = _finite_float(timestamp, name="timestamp_seconds")
        raw_tr = tr_payload.get("tr")
        if not isinstance(raw_tr, Mapping):
            raise AsdError(f"tr.json 缺少 tr 对象: {base}/tr.json")
        tr = {
            key: _finite_float(value, name=f"tr.{key}")
            for key, value in raw_tr.items()
            if key in TR_FIELDS
        }
        if not self.legacy:
            missing = [key for key in TR_FIELDS if key not in tr]
            if missing:
                raise AsdError(f"TR 缺少字段: {missing}")
            if any(key in raw_tr for key in FORBIDDEN_DT_FIELDS):
                raise AsdError("TR 不允许包含 dt_seconds/dt_valid/dt_reason")

        rd = []
        rd_metadata = []
        for frame_index in self._rd_indices.get(int(event_i), []):
            frame = self._read_rd_frame(base, frame_index)
            rd.append((frame["time_seconds"], frame["amplitude"], frame["phase"]))
            rd_metadata.append(frame)
        evidence = None
        try:
            evidence_payload = json.loads(
                self._zip.read(f"{base}/evidence.json").decode("utf-8")
            )
        except KeyError:
            evidence_payload = None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AsdError(f"stable evidence JSON 无效: {base}/evidence.json") from exc
        if evidence_payload is not None:
            evidence = _validate_stable_evidence(
                evidence_payload,
                event_index=int(event_i),
                frame_count=len(rd),
            )
        return {
            "timestamp_seconds": timestamp_value,
            "timestamp": timestamp_value,
            "event_index": int(event_i),
            "track_point_index": int(
                tr_payload.get("track_point_index", int(event_i) + 1)
            ),
            "tr": tr,
            "rd": rd,
            "rd_metadata": rd_metadata,
            "stable_evidence": evidence,
        }

    def read_frame_meta(self, event_i: int, frame_j: int) -> dict:
        if int(frame_j) not in self._rd_indices.get(int(event_i), []):
            raise AsdError(f"RD frame 不存在: event={event_i}, frame={frame_j}")
        return self._read_rd_frame(
            f"events/{int(event_i):04d}", int(frame_j), tensors=False
        )

    def _read_rd_frame(
        self, base: str, frame_index: int, tensors: bool = True
    ) -> dict:
        frame_root = f"{base}/rd/{int(frame_index):04d}"
        try:
            meta = json.loads(
                self._zip.read(f"{frame_root}/meta.json").decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AsdError(f"RD meta 无效: {frame_root}/meta.json") from exc
        try:
            shape = tuple(int(item) for item in meta["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AsdError(f"RD shape 无效: {frame_root}") from exc
        if len(shape) != 2 or min(shape) <= 0:
            raise AsdError(f"RD shape 无效: {frame_root}: {shape}")
        range_axis = meta.get("range_axis_m", meta.get("range_axis"))
        velocity_axis = meta.get("velocity_axis_mps", meta.get("velocity_axis"))
        if not self.legacy:
            range_axis = _axis_values(
                range_axis, name="range_axis_m", expected=shape[0]
            )
            velocity_axis = _axis_values(
                velocity_axis, name="velocity_axis_mps", expected=shape[1]
            )
            _finite_float(meta.get("target_range_m"), name="target_range_m")
            if meta.get("phase_unit") != "rad":
                raise AsdError(f"phase_unit 必须是 rad: {frame_root}")
        result = {
            "time_seconds": _finite_float(
                meta.get("time_seconds"), name="RD time_seconds"
            ),
            "shape": shape,
            "range_axis_m": range_axis,
            "velocity_axis_mps": velocity_axis,
            "range_axis": range_axis,
            "velocity_axis": velocity_axis,
            "target_range_m": meta.get("target_range_m"),
            "meta": dict(meta),
        }
        if tensors:
            count = shape[0] * shape[1]
            try:
                amp_raw = self._zip.read(f"{frame_root}/amplitude.bin")
                phase_raw = self._zip.read(f"{frame_root}/phase.bin")
            except KeyError as exc:
                raise AsdError(f"RD 二进制缺失: {frame_root}") from exc
            expected_bytes = count * 4
            if len(amp_raw) != expected_bytes or len(phase_raw) != expected_bytes:
                raise AsdError(f"RD 张量字节数错误: {frame_root}")
            amplitude = np.frombuffer(amp_raw, dtype="<f4").reshape(shape)
            phase = np.frombuffer(phase_raw, dtype="<f4").reshape(shape)
            if not np.isfinite(amplitude).all() or not np.isfinite(phase).all():
                raise AsdError(f"RD 张量包含非有限值: {frame_root}")
            result["amplitude"] = amplitude
            result["phase"] = phase
        return result

    def rd_count(self, event_i: int) -> int:
        return len(self._rd_indices.get(int(event_i), []))


open_asd = AsdData
