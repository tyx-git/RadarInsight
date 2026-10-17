"""V27d/r17 的 OriData Excel TR-only 数据处理。

``bz`` 的列结构提供 parent 标注，``tck`` 的列结构提供逐点 TR。
parent 身份严格使用 ``source_suffix + StartTimer + Batch``；不把同一
Batch 下的不同 StartTimer 轨迹拼接。Excel Noise 不进入模型，SNR 始终
按缺失处理。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import openpyxl
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.features import (
    encode_causal_tr_observation_increment_sequence,
    encode_tr_feature_values,
)
from radar_ln3.data import encode_causal_rd_quality
from radar_ln3.known_gate_features import extract_causal_course_circular_incoherence

from .protocol import (
    R17_ALLOWED_RAW_LABELS,
    R17_CLASS_NAMES,
    R17_DATA_ROOT,
    R17_DBB_RAW_LABELS,
    R17_DT_REASON,
    R17_EXPERIMENT_ID,
    R17_L2_CLASS_NAMES,
    R17_RAW_LABEL_TO_CLASS,
    R17Config,
    canonical_sha256,
    data_path,
    project_path,
    relative_path,
    sha256_file,
    write_json_if_identical,
    write_text_if_identical,
)


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


def _batch(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _read_xlsx(path: Path) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if not book.worksheets:
            raise ValueError(f"r17 Excel file has no worksheet: {path}")
        rows = book.worksheets[0].iter_rows(values_only=True)
        try:
            raw_header = next(rows)
        except StopIteration as exc:
            raise ValueError(f"r17 Excel file is empty: {path}") from exc
        header = tuple(
            _text(value) or f"__empty_{index}"
            for index, value in enumerate(raw_header)
        )
        return header, [dict(zip(header, row, strict=False)) for row in rows]
    finally:
        book.close()


def _require_columns(header: Sequence[str], required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"r17 columns missing in {path.name}: {missing}")


def _suffix(path: Path, prefix: str) -> str:
    if not path.stem.startswith(prefix):
        raise ValueError(f"r17 file prefix differs: {path.name}")
    return path.stem[len(prefix) :]


def _parent_key(row: Mapping[str, Any], *, source_suffix: str) -> tuple[str, str, str]:
    start_timer = _text(row.get("StartTimer"))
    batch_id = _batch(row.get("Batch"))
    if not source_suffix or start_timer is None or batch_id is None:
        raise ValueError("r17 parent row lacks source suffix, StartTimer or Batch")
    return source_suffix, start_timer, batch_id




def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    write_text_if_identical(path, "".join(_json_line(row) for row in rows))
    return sha256_file(path)


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {name: None for name in ("min", "p01", "p05", "median", "p95", "p99", "max")}
    return {
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _assign_splits(
    parent_ids: Sequence[str],
    labels: Mapping[str, str],
    *,
    seed: int,
    validation_fraction: float,
) -> dict[str, str]:
    result: dict[str, str] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for parent_id in parent_ids:
        grouped[str(labels[parent_id])].append(str(parent_id))
    for label, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda value: hashlib.sha256(f"{seed}|{label}|{value}".encode()).hexdigest(),
        )
        validation_count = 0
        if len(ordered) >= 5:
            validation_count = min(max(1, int(round(len(ordered) * validation_fraction)),), len(ordered) - 1)
        for parent_id in ordered[:validation_count]:
            result[parent_id] = "validation"
        for parent_id in ordered[validation_count:]:
            result[parent_id] = "train"
    if set(result) != set(parent_ids):
        raise RuntimeError("r17 split assignment is incomplete")
    return result


@dataclass(frozen=True)
class R17BuildResult:
    manifest_path: Path
    manifest_sha256: str
    parents_path: Path
    events_path: Path
    parent_count: int
    event_count: int
    split_counts: Mapping[str, int]
    class_counts: Mapping[str, int]


def build_r17_data(project_root: str | Path, config: R17Config) -> R17BuildResult:
    """读取原始 Excel，生成 parent/event JSONL 和数据 manifest。"""

    root = Path(project_root).resolve()
    source_root = project_path(root, config.source["raw_excel_root"])
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    bz_paths = sorted(source_root.glob("bz_*.xlsx"))
    tck_paths = sorted(source_root.glob("tck_*.xlsx"))
    if not bz_paths or not tck_paths:
        raise FileNotFoundError("r17 requires bz_*.xlsx and tck_*.xlsx")

    # 完全重复的 bz 文件只保留排序后的第一份；其余文件仍写入审计。
    bz_hashes: dict[str, list[str]] = defaultdict(list)
    bz_records: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    bz_audit: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    for path in bz_paths:
        header, rows = _read_xlsx(path)
        _require_columns(header, ("StartTimer", "Batch", "MarkType", "MarkTypeID", "Notes"), path)
        suffix = _suffix(path, "bz_")
        digest = sha256_file(path)
        bz_hashes[digest].append(path.name)
        bz_audit.append({"path": relative_path(root, path), "sha256": digest, "row_count": len(rows), "suffix": suffix, "role": "annotation"})
        for row_index, row in enumerate(rows, start=2):
            key = _parent_key(row, source_suffix=suffix)
            raw_label = _text(row.get("MarkType"))
            if raw_label is None:
                raise ValueError(f"r17 empty MarkType: {path.name}:{row_index}")
            item = {
                "source_file": path.name,
                "source_suffix": suffix,
                "source_row_index": row_index,
                "start_timer": key[1],
                "batch_id": key[2],
                "label_name_raw": raw_label,
                "mark_type_id_raw": _text(row.get("MarkTypeID")),
                "notes": _text(row.get("Notes")) or "",
            }
            bz_records[key].append(item)
            if raw_label not in R17_ALLOWED_RAW_LABELS:
                excluded_counts[raw_label] += 1

    duplicate_bz_files = {
        name
        for names in bz_hashes.values() if len(names) > 1
        for name in sorted(names)[1:]
    }
    canonical_suffixes = {
        _suffix(path, "bz_")
        for digest, names in bz_hashes.items()
        for path in [source_root / sorted(names)[0]]
    }

    tck_records: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    tck_audit: list[dict[str, Any]] = []
    for path in tck_paths:
        suffix = _suffix(path, "tck_")
        header, rows = _read_xlsx(path)
        _require_columns(header, ("StartTimer", "Batch", "Azimuth", "Distance", "Latitude", "longitude", "Speed_R", "Elevation", "Course", "Speed", "Height", "Time", "Amplitude", "Noise", "Type"), path)
        digest = sha256_file(path)
        tck_audit.append({"path": relative_path(root, path), "sha256": digest, "row_count": len(rows), "suffix": suffix, "role": "tr_measurement"})
        if suffix not in canonical_suffixes:
            continue
        for row_index, row in enumerate(rows, start=2):
            key = _parent_key(row, source_suffix=suffix)
            tck_records[key].append({"source_file": path.name, "source_suffix": suffix, "source_row_index": row_index, "values": dict(row)})

    selected_annotations: dict[tuple[str, str, str], dict[str, Any]] = {}
    conflict_keys: list[tuple[str, str, str]] = []
    for key, values in bz_records.items():
        if key[0] not in canonical_suffixes:
            continue
        identities = {(item["label_name_raw"], item["mark_type_id_raw"]) for item in values}
        if len(identities) > 1:
            conflict_keys.append(key)
            continue
        selected_annotations[key] = sorted(values, key=lambda item: item["source_row_index"])[0]

    selected_keys = sorted(key for key, item in selected_annotations.items() if item["label_name_raw"] in R17_ALLOWED_RAW_LABELS)
    tck_only = sorted(set(tck_records) - set(selected_annotations))
    missing_tck = sorted(key for key in selected_keys if key not in tck_records)
    if missing_tck:
        raise ValueError(f"r17 selected annotations lack tck rows: {missing_tck[:5]}")

    parent_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    raw_noise: list[float] = []
    raw_amplitude: list[float] = []
    derived_snr: list[float] = []
    max_length = int(config.data["max_sequence_length"])

    for source_suffix, start_timer, batch_id in selected_keys:
        annotation = selected_annotations[(source_suffix, start_timer, batch_id)]
        label_name = R17_RAW_LABEL_TO_CLASS[annotation["label_name_raw"]]
        l1_id, l2_id = (1, -1) if label_name == "ClutterTarget" else (0, R17_L2_CLASS_NAMES.index(label_name))
        parent_id = f"r17|excel|{source_suffix}|start{start_timer}|batch{batch_id}"
        labels[parent_id] = label_name
        rows = sorted(tck_records[(source_suffix, start_timer, batch_id)], key=lambda item: item["source_row_index"])
        first_time = _float(rows[0]["values"].get("Time"))
        if first_time is None:
            raise ValueError(f"r17 first time is missing: {parent_id}")
        times = [_float(item["values"].get("Time")) for item in rows]
        if any(value is None for value in times):
            raise ValueError(f"r17 time is missing: {parent_id}")
        times_float = [float(value) for value in times]
        deltas = [0.0] + [
            (current - previous) / 1000.0
            for previous, current in zip(times_float[:-1], times_float[1:], strict=True)
        ]
        if any(delta < 0.0 for delta in deltas[1:]):
            raise ValueError(f"r17 time decreases within parent: {parent_id}")
        event_parent_rows: list[dict[str, Any]] = []
        source_type_counts: Counter[str] = Counter()
        for event_index, (item, delta) in enumerate(zip(rows, deltas, strict=True)):
            values = item["values"]
            speed_r = _float(values.get("Speed_R"))
            amplitude = _float(values.get("Amplitude"))
            noise = _float(values.get("Noise"))
            if amplitude is not None:
                raw_amplitude.append(amplitude)
            if noise is not None:
                raw_noise.append(noise)
            if amplitude is not None and noise is not None:
                derived_snr.append(amplitude - noise)
            reason = "first_event_no_predecessor" if event_index == 0 else "zero_delta_at_second_precision" if abs(delta) <= 1.0e-12 else "positive_delta_at_second_precision"
            source_type = _text(values.get("Type")) or ""
            source_type_counts[source_type] += 1
            event_parent_rows.append({
                "schema_version": config.data["event_schema_version"],
                "parent_sample_id": parent_id,
                "source_suffix": source_suffix,
                "source_tck_file": item["source_file"],
                "source_row_index": int(item["source_row_index"]),
                "causal_event_index": event_index,
                "time_raw": times_float[event_index],
                "time_seconds": (times_float[event_index] - float(first_time)) / 1000.0,
                "dt_seconds": max(0.0, float(delta)),
                "dt_valid": reason == "positive_delta_at_second_precision",
                "dt_reason": reason,
                "azimuth_deg": _float(values.get("Azimuth")),
                "range_m": _float(values.get("Distance")),
                "radial_speed_mps": None if speed_r is None else -speed_r,
                "elevation_deg": _float(values.get("Elevation")),
                "height_m": _float(values.get("Height")),
                "amplitude_db": amplitude,
                "snr_db": None,
                "noise_raw_untrusted": noise,
                "course_deg": _float(values.get("Course")),
                "speed_mps": _float(values.get("Speed")),
                "latitude": _float(values.get("Latitude")),
                "longitude": _float(values.get("longitude")),
                "source_type_raw": source_type,
                "label_name_raw": annotation["label_name_raw"],
                "label_name": label_name,
                "l1_label_id": l1_id,
                "l2_label_id": l2_id,
            })
        event_rows.extend(event_parent_rows)
        parent_rows.append({
            "schema_version": config.data["parent_schema_version"],
            "parent_sample_id": parent_id,
            "source_suffix": source_suffix,
            "bz_source_file": annotation["source_file"],
            "tck_source_file": event_parent_rows[0]["source_tck_file"],
            "start_timer": start_timer,
            "batch_id": batch_id,
            "label_name_raw": annotation["label_name_raw"],
            "label_name": label_name,
            "l1_label_id": l1_id,
            "l2_label_id": l2_id,
            "split_name": "",
            "split_group_id": parent_id,
            "event_count": min(len(event_parent_rows), max_length),
            "raw_event_count": len(event_parent_rows),
            "model_event_count": min(len(event_parent_rows), max_length),
            "model_truncation_applied": len(event_parent_rows) > max_length,
            "time_seconds_min": float(min(row["time_seconds"] for row in event_parent_rows)),
            "time_seconds_max": float(max(row["time_seconds"] for row in event_parent_rows)),
            "time_monotonic": True,
            "zero_dt_event_count": sum(row["dt_reason"] == "zero_delta_at_second_precision" for row in event_parent_rows),
            "negative_dt_event_count": 0,
            "mixed_source_type": len(source_type_counts) > 1,
            "source_type_counts_json": json.dumps(dict(sorted(source_type_counts.items())), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "notes": annotation["notes"],
            "events_path": "",
        })

    parent_ids = [row["parent_sample_id"] for row in parent_rows]
    label_map = {row["parent_sample_id"]: row["label_name"] for row in parent_rows}
    split_map = _assign_splits(parent_ids, label_map, seed=int(config.data["split_seed"]), validation_fraction=float(config.data["validation_fraction"]))
    for row in parent_rows:
        row["split_name"] = split_map[row["parent_sample_id"]]
    parent_rows.sort(key=lambda row: row["parent_sample_id"])
    event_rows.sort(key=lambda row: (row["parent_sample_id"], row["causal_event_index"]))

    output_root = (root / R17_DATA_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parents_path = data_path(root, output_root / "parents.jsonl")
    events_path = data_path(root, output_root / "events.jsonl")
    for row in parent_rows:
        row["events_path"] = relative_path(root, events_path)
    parents_hash = _write_jsonl(parents_path, parent_rows)
    events_hash = _write_jsonl(events_path, event_rows)

    class_counts = Counter(row["label_name"] for row in parent_rows)
    split_counts = Counter(row["split_name"] for row in parent_rows)
    data_payload = {
        "schema_version": "RADAR-LN3-V27d-r17-processed-data-manifest-v2",
        "experiment_id": R17_EXPERIMENT_ID,
        "dataset": "OriData/excel",
        "semantic_resolution": "column_schema_over_filename_role",
        "source_root": relative_path(root, source_root),
        "parents_path": relative_path(root, parents_path),
        "parents_sha256": parents_hash,
        "events_path": relative_path(root, events_path),
        "events_sha256": events_hash,
        "parent_count": len(parent_rows),
        "event_count": len(event_rows),
        "class_names": list(R17_CLASS_NAMES),
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "raw_label_to_class": dict(R17_RAW_LABEL_TO_CLASS),
        "excluded_label_counts": dict(sorted(excluded_counts.items())),
        "conflict_parent_count": len(conflict_keys),
        "duplicate_bz_hash_groups": [list(names) for names in bz_hashes.values() if len(names) > 1],
        "tck_only_parent_count": len(tck_only),
        "split": {"strategy": "parent_stratified_deterministic", "group_unit": "source_suffix+StartTimer+Batch", "seed": int(config.data["split_seed"]), "validation_fraction": float(config.data["validation_fraction"]), "test": "not_created"},
        "time": {"source_field": "Time", "raw_scale": "milliseconds", "normalized_field": "parent_first_time_relative_seconds", "event_order": "workbook_row_order", "negative_dt_policy": "fail_closed"},
        "radial_speed": {"source_field": "Speed_R", "transform": "radial_speed_mps = -Speed_R"},
        "snr": {"raw_noise_field": "Noise", "model_input": "excluded; snr_db is missing", "raw_noise_quantiles": _quantiles(raw_noise), "derived_snr_quantiles_untrusted_diagnostic_only": _quantiles(derived_snr), "amplitude_quantiles": _quantiles(raw_amplitude)},
        "scope": {"training_raw_labels": sorted(R17_DBB_RAW_LABELS | {"杂波"}), "ships_included": False, "people_included": False, "unknown_payload_present": False, "rd_payload_present": False},
        "source_audit": {"bz_files": bz_audit, "tck_files": tck_audit, "conflict_parent_keys": [list(key) for key in conflict_keys], "tck_only_parent_keys": [list(key) for key in tck_only]},
    }
    data_payload["manifest_sha256"] = canonical_sha256(data_payload)
    manifest_path = data_path(root, output_root / "manifest.json")
    write_json_if_identical(manifest_path, data_payload)
    return R17BuildResult(manifest_path, sha256_file(manifest_path), parents_path, events_path, len(parent_rows), len(event_rows), dict(split_counts), dict(class_counts))


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


def build_r17_loader(dataset: R17Dataset, *, batch_size: int, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("r17 batch size must be positive")
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate_r17, generator=generator)


def r17_l1_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask", "rd_quality", "course_features", "course_valid", "course_support")
    return {key: batch[key].to(device=device) for key in keys}


def r17_l2_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask", "rd_quality", "rd_physical_features", "rd_physical_valid", "micro_doppler_features", "micro_doppler_valid")
    return {key: batch[key].to(device=device) for key in keys}


__all__ = ["R17BuildResult", "R17Dataset", "R17Sequence", "build_r17_data", "build_r17_loader", "collate_r17", "load_r17_data_manifest", "r17_l1_inputs", "r17_l2_inputs"]