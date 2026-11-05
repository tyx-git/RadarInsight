"""AeroSight ``.asd`` package adapter.

The adapter is intentionally independent from CQ08. It accepts either an
``.asd`` zip file or an already extracted package, converts string-valued TR
fields to finite numeric values, computes seconds timestamps, and materializes
canonical RD frames for the shared feature pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import zipfile
from typing import Any

import numpy as np

from .contracts import DataFormatError, DataIntegrityError, EventData, EventSequence
from .rd_canonical import CANONICAL_VELOCITY_MPS, CANONICAL_RANGE_OFFSETS_M, canonicalize_rd_frame
from .temporal import annotate_event_data


ASD_TR_FIELDS = (
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_speed_mps",
    "amplitude_db",
    "snr_db",
    "speed_mps",
    "course_deg",
)
_EVENT_RE = re.compile(r"(?:^|/)events/(?P<index>\d+)(?:/|$)")
_LABELS = {
    "drone": (0, "DroneTarget"),
    "dronetarget": (0, "DroneTarget"),
    "bird": (1, "BirdTarget"),
    "birdtarget": (1, "BirdTarget"),
    "balloon": (2, "BalloonTarget"),
    "balloontarget": (2, "BalloonTarget"),
    "clutter": (3, "ClutterTarget"),
    "cluttertarget": (3, "ClutterTarget"),
    "unknown": (4, "UnknownTarget"),
    "unknowntarget": (4, "UnknownTarget"),
}
_NUMERIC_LABELS = {
    "1": (0, "DroneTarget"),
    "2": (0, "DroneTarget"),
    "3": (1, "BirdTarget"),
    "4": (2, "BalloonTarget"),
}
ASD_EVIDENCE_SCHEMA = "RADAR-LN3-ASD-MICRO-DOPPLER-EVIDENCE-V1"
ASD_EVIDENCE_FRAME_CAP = 12
ASD_EVIDENCE_FRAME_DIM = 23
ASD_EVIDENCE_STABLE_DIM = 8


class _Package:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._zip: zipfile.ZipFile | None = None
        self._root: Path | None = None
        if self.path.is_dir():
            self._root = self.path
        elif self.path.is_file() and self.path.suffix.lower() == ".asd":
            try:
                self._zip = zipfile.ZipFile(self.path, "r")
            except zipfile.BadZipFile as exc:
                raise DataFormatError(f".asd 不是有效 zip 包：{self.path}") from exc
        else:
            raise DataFormatError(f".asd 路径不存在或类型不支持：{self.path}")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "_Package":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _zip_name(self, relative: str) -> str:
        assert self._zip is not None
        names = [name for name in self._zip.namelist() if not name.endswith("/")]
        normalized = relative.replace("\\", "/").lstrip("/")
        direct = [name for name in names if name.replace("\\", "/") == normalized]
        if direct:
            return direct[0]
        suffix = "/" + normalized
        matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(relative)
        raise DataFormatError(f".asd 包内成员路径不唯一：{relative}")

    def read_bytes(self, relative: str) -> bytes:
        if self._root is not None:
            target = (self._root / relative).resolve()
            try:
                target.relative_to(self._root)
            except ValueError as exc:
                raise DataFormatError(f".asd 成员越过包根目录：{relative}") from exc
            if not target.is_file():
                raise FileNotFoundError(relative)
            return target.read_bytes()
        assert self._zip is not None
        return self._zip.read(self._zip_name(relative))

    def read_json(self, relative: str) -> Any:
        try:
            value = json.loads(self.read_bytes(relative).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataFormatError(f".asd JSON 无效：{relative}") from exc
        return value

    def event_indices(self) -> list[int]:
        if self._root is not None:
            events = self._root / "events"
            if not events.is_dir():
                raise DataFormatError(".asd 缺少 events 目录")
            values = [int(item.name) for item in events.iterdir() if item.is_dir() and item.name.isdigit()]
        else:
            assert self._zip is not None
            values = []
            for name in self._zip.namelist():
                match = _EVENT_RE.search(name.replace("\\", "/"))
                if match is not None:
                    values.append(int(match.group("index")))
        values = sorted(set(values))
        if not values or values != list(range(values[-1] + 1)):
            raise DataFormatError(".asd event index 必须从 0 开始连续递增")
        return values

    def rd_frame_indices(self, event_relative: str) -> list[int]:
        prefix = event_relative.replace("\\", "/").rstrip("/") + "/rd/"
        if self._root is not None:
            target = self._root / event_relative / "rd"
            if not target.is_dir():
                return []
            values = [int(item.name) for item in target.iterdir() if item.is_dir() and item.name.isdigit()]
        else:
            assert self._zip is not None
            values = []
            for name in self._zip.namelist():
                normalized = name.replace("\\", "/")
                if normalized.startswith(prefix):
                    tail = normalized[len(prefix):].split("/", 1)[0]
                    if tail.isdigit():
                        values.append(int(tail))
        return sorted(set(values))


def _finite_number(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DataFormatError(f".asd TR 字段无法转换为数值：{field}={value!r}") from exc
    if not np.isfinite(parsed):
        raise DataIntegrityError(f".asd TR 字段包含非有限值：{field}")
    return parsed


def _timestamp_seconds(value: Any, unit: str) -> float:
    number = _finite_number(value, field="timestamp")
    factors = {"s": 1.0, "ms": 1.0e-3, "us": 1.0e-6, "ns": 1.0e-9}
    normalized = str(unit).strip().lower()
    if normalized not in factors:
        raise DataFormatError(f"不支持的 timestamp unit：{unit!r}")
    return float(number * factors[normalized])


def _label_info(manifest: Mapping[str, Any]) -> tuple[int | None, str]:
    source = manifest.get("source")
    source_name = source.get("mat_file_class") if isinstance(source, Mapping) else None
    raw_name = str(manifest.get("label_name") or source_name or manifest.get("label") or "unknown")
    key = raw_name.strip().lower().replace(" ", "")
    if key in _LABELS:
        return _LABELS[key]
    if key in _NUMERIC_LABELS:
        return _NUMERIC_LABELS[key]
    try:
        return int(manifest.get("label")), raw_name
    except (TypeError, ValueError):
        return None, raw_name


def _evidence_metadata(
    value: Any,
    *,
    event_index: int,
    rd_frame_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataFormatError(".asd stable evidence 必须是对象")
    if str(value.get("schema_version", "")) != ASD_EVIDENCE_SCHEMA:
        raise DataFormatError(".asd stable evidence schema_version 不匹配")
    if int(value.get("frame_cap", -1)) != ASD_EVIDENCE_FRAME_CAP:
        raise DataFormatError(".asd stable evidence frame_cap 必须为 12")
    stable_values = np.asarray(value.get("stable8_values"), dtype=np.float32)
    stable_frame_values = np.asarray(
        value.get("stable8_frame_values"), dtype=np.float32
    )
    stable_frame_mask = np.asarray(
        value.get("stable8_frame_mask"), dtype=np.bool_
    )
    micro_values = np.asarray(
        value.get("micro_doppler_frame_values"), dtype=np.float32
    )
    micro_frame_mask = np.asarray(
        value.get("micro_doppler_frame_mask"), dtype=np.bool_
    )
    micro_reliable = np.asarray(
        value.get("micro_doppler_reliable"), dtype=np.bool_
    )
    raw_indices = np.asarray(value.get("raw_frame_indices"), dtype=np.int32)
    if stable_values.shape != (ASD_EVIDENCE_STABLE_DIM,):
        raise DataFormatError(".asd stable8_values shape 无效")
    if stable_frame_values.shape != (
        ASD_EVIDENCE_FRAME_CAP,
        ASD_EVIDENCE_STABLE_DIM,
    ):
        raise DataFormatError(".asd stable8_frame_values shape 无效")
    if micro_values.shape != (
        ASD_EVIDENCE_FRAME_CAP,
        ASD_EVIDENCE_FRAME_DIM,
    ):
        raise DataFormatError(".asd micro_doppler_frame_values shape 无效")
    for name, array in (
        ("stable8_frame_mask", stable_frame_mask),
        ("micro_doppler_frame_mask", micro_frame_mask),
        ("micro_doppler_reliable", micro_reliable),
        ("raw_frame_indices", raw_indices),
    ):
        if array.shape != (ASD_EVIDENCE_FRAME_CAP,):
            raise DataFormatError(f".asd {name} shape 无效")
    if (
        not np.isfinite(stable_values).all()
        or not np.isfinite(stable_frame_values).all()
        or not np.isfinite(micro_values).all()
    ):
        raise DataIntegrityError(".asd stable evidence 包含非有限值")
    if (
        np.any(stable_values < 0.0)
        or np.any(stable_values > 1.0)
        or np.any(stable_frame_values < 0.0)
        or np.any(stable_frame_values > 1.0)
        or np.any(micro_values < 0.0)
        or np.any(micro_values > 1.0)
    ):
        raise DataIntegrityError(".asd stable evidence 值必须位于 [0,1]")
    if not np.array_equal(stable_frame_mask, micro_reliable):
        raise DataIntegrityError(".asd stable8_frame_mask 与 reliable mask 不一致")
    if bool(value.get("stable8_valid", False)) != bool(stable_frame_mask.any()):
        raise DataIntegrityError(".asd stable8_valid 与 frame mask 不一致")
    if np.any(stable_frame_values[~stable_frame_mask] != 0.0):
        raise DataIntegrityError(".asd 非 reliable Stable8 frame 必须为零")
    if np.any(micro_values[~micro_reliable] != 0.0):
        raise DataIntegrityError(".asd 非 reliable Micro-Doppler frame 必须为零")
    if np.any(raw_indices[~micro_frame_mask] != -1):
        raise DataIntegrityError(".asd inactive raw frame index 必须为 -1")
    if int(micro_frame_mask.sum()) != int(rd_frame_count):
        raise DataIntegrityError(
            ".asd stable evidence frame count 与 RD 不一致："
            f"evidence={int(micro_frame_mask.sum())}, rd={rd_frame_count}"
        )
    if int(value.get("event_index", event_index)) != int(event_index):
        raise DataIntegrityError(".asd stable evidence event_index 不一致")
    return {
        "stable8_values": stable_values,
        "stable8_valid": bool(value.get("stable8_valid", False)),
        "stable8_frame_values": stable_frame_values,
        "stable8_frame_mask": stable_frame_mask,
        "micro_doppler_frame_values": micro_values,
        "micro_doppler_frame_mask": micro_frame_mask,
        "micro_doppler_reliable": micro_reliable,
        "micro_doppler_values": stable_values,
        "micro_doppler_valid": bool(value.get("stable8_valid", False)),
        "micro_doppler_raw_frame_indices": raw_indices,
        "stable8_sidecar_mode": "raw",
        "stable8_evidence_schema": ASD_EVIDENCE_SCHEMA,
    }


@dataclass(frozen=True)
class ASDAdapter:
    """读取一个 AeroSight package 的配置。"""

    timestamp_unit: str | None = None
    default_timestamp_unit: str = "ms"
    amplitude_mode: str | None = None
    default_amplitude_mode: str = "magnitude_db"
    require_phase: bool = True
    max_events: int | None = None

    def __post_init__(self) -> None:
        if self.max_events is not None and (
            isinstance(self.max_events, bool) or int(self.max_events) <= 0
        ):
            raise ValueError("max_events must be a positive integer or None")

    def load(self, path: str | Path) -> EventSequence:
        with _Package(path) as package:
            manifest = package.read_json("manifest.json")
            if not isinstance(manifest, Mapping):
                raise DataFormatError(".asd manifest 必须是 JSON 对象")
            return self._load_package(package, manifest)

    def _load_package(self, package: _Package, manifest: Mapping[str, Any]) -> EventSequence:
        source = manifest.get("source")
        source = source if isinstance(source, Mapping) else {}
        track_id = str(source.get("track_id") or package.path.stem)
        parent_id = str(
            manifest.get("parent_sample_id")
            or f"{source.get('session', package.path.stem)}|track|{track_id}"
        )
        label_id, label_name = _label_info(manifest)
        timestamp_unit = str(
            self.timestamp_unit
            or manifest.get("timestamp_unit")
            or manifest.get("time", {}).get("unit")
            or self.default_timestamp_unit
        )
        rd_contract = manifest.get("rd")
        rd_contract = rd_contract if isinstance(rd_contract, Mapping) else {}
        declared_amplitude_mode = (
            rd_contract.get("amplitude_mode")
            or manifest.get("amplitude_mode")
        )
        amplitude_mode = str(
            self.amplitude_mode
            or declared_amplitude_mode
            or self.default_amplitude_mode
        )
        amplitude_mode_assumed = (
            self.amplitude_mode is None and declared_amplitude_mode is None
        )
        assumed_unit = self.timestamp_unit is None and not (
            manifest.get("timestamp_unit") or manifest.get("time", {}).get("unit")
        )
        events: list[EventData] = []
        all_event_indices = package.event_indices()
        selected_event_indices = (
            all_event_indices
            if self.max_events is None
            else all_event_indices[: int(self.max_events)]
        )
        for index in selected_event_indices:
            relative = f"events/{index:04d}"
            payload = package.read_json(f"{relative}/tr.json")
            if not isinstance(payload, Mapping) or not isinstance(payload.get("tr"), Mapping):
                raise DataFormatError(f".asd TR 记录结构无效：{relative}/tr.json")
            tr_values: dict[str, Any] = {}
            for name in ASD_TR_FIELDS:
                if name in payload["tr"] and payload["tr"][name] is not None:
                    tr_values[name] = _finite_number(payload["tr"][name], field=name)
            timestamp_raw = payload.get("timestamp_seconds", payload.get("timestamp"))
            if timestamp_raw is None:
                raise DataFormatError(f".asd 事件缺少 timestamp：{relative}")
            tr_values["time_seconds"] = _timestamp_seconds(timestamp_raw, timestamp_unit)
            rd_values = self._read_rd(
                package,
                relative,
                tr_values.get("range_m"),
                amplitude_mode=amplitude_mode,
            )
            evidence = None
            try:
                evidence_payload = package.read_json(f"{relative}/evidence.json")
            except FileNotFoundError:
                evidence_payload = None
            if evidence_payload is not None:
                evidence = _evidence_metadata(
                    evidence_payload,
                    event_index=index,
                    rd_frame_count=int(
                        (rd_values or {}).get("frame_count", 0)
                    ),
                )
            events.append(
                EventData(
                    sample_id=f"{parent_id}:event-{index}",
                    parent_sample_id=parent_id,
                    track_id=track_id,
                    causal_event_index=index,
                    label_id=label_id,
                    label_name=label_name,
                    split_name="inference",
                    tr_values=tr_values,
                    rd_values=rd_values,
                    tr_available=True,
                    rd_available=rd_values is not None,
                    metadata={
                        "source_path": str(package.path),
                        "asd_timestamp_unit": timestamp_unit,
                        "asd_timestamp_unit_assumed": assumed_unit,
                        "asd_amplitude_mode": amplitude_mode,
                        "asd_amplitude_mode_assumed": amplitude_mode_assumed,
                        **(evidence or {}),
                    },
                )
            )
        events = annotate_event_data(events)
        return EventSequence(
            parent_sample_id=parent_id,
            track_id=track_id,
            label_id=label_id,
            label_name=label_name,
            split_name="inference",
            events=events,
            source_name="asd",
            metadata={
                "source_path": str(package.path),
                "original_event_count": len(all_event_indices),
                "model_input_event_count": len(events),
                "sequence_truncated": len(all_event_indices) > len(events),
                "timestamp_unit": timestamp_unit,
                "timestamp_unit_assumed": assumed_unit,
                "amplitude_mode": amplitude_mode,
                "amplitude_mode_assumed": amplitude_mode_assumed,
                "manifest": dict(manifest),
            },
        )

    def _read_rd(
        self,
        package: _Package,
        event_relative: str,
        target_range_m: float | None,
        *,
        amplitude_mode: str,
    ) -> dict[str, Any] | None:
        frame_indices = package.rd_frame_indices(event_relative)
        if not frame_indices:
            return None
        frames: list[dict[str, Any]] = []
        original_count = len(frame_indices)
        truncated = original_count > 12
        for frame_index in frame_indices[:12]:
            frame_root = f"{event_relative}/rd/{frame_index:04d}"
            meta_relative = f"{frame_root}/meta.json"
            meta = package.read_json(meta_relative)
            if not isinstance(meta, Mapping):
                raise DataFormatError(f"RD meta 必须是对象：{meta_relative}")
            shape = tuple(int(value) for value in meta.get("shape", ()))
            if len(shape) != 2 or min(shape) <= 0:
                raise DataFormatError(f"RD shape 无效：{meta_relative}")
            range_axis = np.asarray(
                meta.get("range_axis", meta.get("range_axis_m")),
                dtype=np.float64,
            )
            velocity_axis = np.asarray(
                meta.get("velocity_axis", meta.get("velocity_axis_mps")),
                dtype=np.float64,
            )
            if range_axis.shape != (shape[0],) or velocity_axis.shape != (shape[1],):
                raise DataFormatError(f"RD 物理轴长度与 shape 不一致：{meta_relative}")
            frame_target_range = meta.get("target_range_m", meta.get("rd_target_range_m"))
            if frame_target_range is None:
                frame_target_range = target_range_m
            if frame_target_range is None:
                raise DataFormatError(f"RD 缺少 target_range_m：{meta_relative}")
            frame_target_range = _finite_number(
                frame_target_range,
                field=f"{meta_relative}:target_range_m",
            )
            amplitude = np.frombuffer(
                package.read_bytes(f"{frame_root}/amplitude.bin"), dtype="<f4"
            )
            if amplitude.size != shape[0] * shape[1]:
                raise DataFormatError("amplitude.bin 长度与 RD shape 不一致")
            amplitude = amplitude.reshape(shape)
            try:
                phase_bytes = package.read_bytes(f"{frame_root}/phase.bin")
            except FileNotFoundError:
                if self.require_phase:
                    return None
                phase = None
            else:
                phase = np.frombuffer(phase_bytes, dtype="<f4")
                if phase.size != shape[0] * shape[1]:
                    raise DataFormatError("phase.bin 长度与 RD shape 不一致")
                phase = phase.reshape(shape)
            frame = canonicalize_rd_frame(
                amplitude,
                phase=phase,
                range_axis_m=range_axis,
                velocity_axis_mps=velocity_axis,
                target_range_m=frame_target_range,
                amplitude_mode=amplitude_mode,
            )
            frames.append(
                {
                    "amplitude": frame.amplitude,
                    "phase": frame.phase,
                    "time_seconds": float(meta.get("time_seconds", frame_index)),
                    "target_range_m": frame_target_range,
                }
            )
        return {
            "frames": frames,
            "velocity_mps": CANONICAL_VELOCITY_MPS.copy(),
            "range_offset_m": CANONICAL_RANGE_OFFSETS_M.copy(),
            "frame_count": len(frames),
            "original_count": original_count,
            "truncated": truncated,
            "canonical": True,
            "canonicalization": dict(frame.metadata),
        }


def load_asd(
    path: str | Path,
    *,
    timestamp_unit: str | None = None,
    amplitude_mode: str | None = None,
    max_events: int | None = None,
) -> EventSequence:
    """加载 `.asd` 并返回统一的 `EventSequence`。"""

    return ASDAdapter(
        timestamp_unit=timestamp_unit,
        amplitude_mode=amplitude_mode,
        max_events=max_events,
    ).load(path)


__all__ = ["ASDAdapter", "ASD_TR_FIELDS", "load_asd"]
