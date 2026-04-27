from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


TRACK_PATTERN = re.compile(r"^Tracks_(?P<track_id>\d+)_(?P<label_id>\d+)_(?P<track_length>\d+)\.txt$")
POINT_PATTERN = re.compile(r"^PointTracks_(?P<track_id>\d+)_(?P<label_id>\d+)_(?P<track_length>\d+)\.txt$")
RAW_ECHO_PATTERN = re.compile(r"^(?P<track_id>\d+)_Label_(?P<label_id>\d+)\.dat$")
MAT_PATTERN = re.compile(
    r"^(?P<capture_id>\d+)_DAUR_RD_(?P<label_text>[^_]+)_(?P<frame_index>\d+)_(?P<track_id>\d+)\.mat$",
    re.IGNORECASE,
)

CQ08_LABEL_MAP = {
    1: "DroneTarget",
    2: "DroneTarget",
    3: "BirdTarget",
    4: "BalloonTarget",
    5: "ClutterTarget",
    6: "UnknownTarget",
}

CQ08_TRACK_COLUMNS = [
    "点时间",
    "批号",
    "滤波距离",
    "滤波方位",
    "滤波俯仰",
    "全速度",
    "X向速度",
    "Y向速度",
    "Z向速度",
    "航向",
]

CQ08_POINT_COLUMNS = [
    "点时间",
    "批号",
    "距离",
    "方位",
    "俯仰",
    "多普勒速度",
    "和幅度",
    "信噪比",
    "原始点数量",
]


@dataclass
class Cq08ManifestRecord:
    session_id: str
    source_name: str
    group_unit: str
    split_group_id: str
    frame_level_split_forbidden: bool
    track_id: int
    label_id: int
    label_name: str
    track_length_hint: int
    capture_id: str
    track_file: str
    point_file: str
    raw_echo_file: str
    rd_mat_files: List[str]
    num_track_rows: int
    num_point_rows: int
    num_rd_mat_frames: int
    raw_echo_bytes: int
    has_track: bool
    has_points: bool
    has_raw_echo: bool
    has_rd_mat: bool
    is_usable: bool
    unusable_reason: str


def _read_text_meta(path: Path, expected_columns: Iterable[str]) -> tuple[int, list[str]]:
    with path.open("r", encoding="gbk", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CQ-08 text file: {path}")
    header = [str(item).strip() for item in rows[0]]
    if header != list(expected_columns):
        raise ValueError(f"unexpected columns in {path.name}: {header}")
    return max(0, len(rows) - 1), header


def _collect_track_files(track_dir: Path) -> dict[tuple[int, int], dict[str, object]]:
    track_map: dict[tuple[int, int], dict[str, object]] = {}
    for path in sorted(track_dir.glob("Tracks_*.txt")):
        matched = TRACK_PATTERN.match(path.name)
        if matched is None:
            continue
        track_id = int(matched.group("track_id"))
        label_id = int(matched.group("label_id"))
        track_length = int(matched.group("track_length"))
        row_count, columns = _read_text_meta(path, CQ08_TRACK_COLUMNS)
        track_map[(track_id, label_id)] = {
            "path": path,
            "track_length_hint": track_length,
            "row_count": row_count,
            "columns": columns,
        }
    return track_map


def _collect_point_files(point_dir: Path) -> dict[tuple[int, int], dict[str, object]]:
    point_map: dict[tuple[int, int], dict[str, object]] = {}
    for path in sorted(point_dir.glob("PointTracks_*.txt")):
        matched = POINT_PATTERN.match(path.name)
        if matched is None:
            continue
        track_id = int(matched.group("track_id"))
        label_id = int(matched.group("label_id"))
        row_count, columns = _read_text_meta(path, CQ08_POINT_COLUMNS)
        point_map[(track_id, label_id)] = {
            "path": path,
            "row_count": row_count,
            "columns": columns,
        }
    return point_map


def _collect_raw_echo_files(raw_echo_dir: Path) -> dict[tuple[int, int], dict[str, object]]:
    raw_map: dict[tuple[int, int], dict[str, object]] = {}
    for path in sorted(raw_echo_dir.glob("*_Label_*.dat")):
        matched = RAW_ECHO_PATTERN.match(path.name)
        if matched is None:
            continue
        track_id = int(matched.group("track_id"))
        label_id = int(matched.group("label_id"))
        raw_map[(track_id, label_id)] = {
            "path": path,
            "raw_echo_bytes": int(path.stat().st_size),
        }
    return raw_map


def _collect_mat_files(mat_dir: Path) -> dict[int, dict[str, object]]:
    mat_map: dict[int, dict[str, object]] = {}
    for path in sorted(mat_dir.glob("*.mat")):
        matched = MAT_PATTERN.match(path.name)
        if matched is None:
            continue
        track_id = int(matched.group("track_id"))
        capture_id = matched.group("capture_id")
        label_text = str(matched.group("label_text")).strip()
        frame_index = int(matched.group("frame_index"))
        payload = mat_map.setdefault(
            track_id,
            {
                "capture_ids": set(),
                "label_texts": set(),
                "frames": [],
            },
        )
        payload["capture_ids"].add(capture_id)
        payload["label_texts"].add(label_text)
        payload["frames"].append((frame_index, path))
    for track_id, payload in mat_map.items():
        payload["frames"] = [frame_path for _, frame_path in sorted(payload["frames"], key=lambda item: item[0])]
    return mat_map


def build_cq08_manifest(cq08_root: Path) -> tuple[list[Cq08ManifestRecord], dict[str, object]]:
    track_map = _collect_track_files(cq08_root / "航迹")
    point_map = _collect_point_files(cq08_root / "点迹")
    raw_map = _collect_raw_echo_files(cq08_root / "原始回波")
    mat_map = _collect_mat_files(cq08_root / "MAT")

    records: list[Cq08ManifestRecord] = []
    label_distribution: dict[str, int] = {}
    usable_count = 0
    unusable_count = 0
    reasons: dict[str, int] = {}

    for key in sorted(track_map.keys()):
        track_id, label_id = key
        label_name = CQ08_LABEL_MAP.get(label_id)
        if label_name is None:
            continue
        track_item = track_map[key]
        point_item = point_map.get(key)
        raw_item = raw_map.get(key)
        mat_item = mat_map.get(track_id)

        capture_ids = sorted(list(mat_item["capture_ids"])) if mat_item else []
        label_texts = sorted(list(mat_item["label_texts"])) if mat_item else []
        rd_mat_files = [frame_path.as_posix() for frame_path in mat_item["frames"]] if mat_item else []

        issues: list[str] = []
        if point_item is None:
            issues.append("missing_point_track")
        if raw_item is None:
            issues.append("missing_raw_echo")
        if mat_item is None or not rd_mat_files:
            issues.append("missing_rd_mat")
        if mat_item is not None and len(capture_ids) != 1:
            issues.append("multi_capture_id")
        if mat_item is not None and len(label_texts) != 1:
            issues.append("multi_mat_label_text")
        if point_item is not None and int(point_item["row_count"]) != int(track_item["row_count"]):
            issues.append("track_point_row_mismatch")

        is_usable = len(issues) == 0
        unusable_reason = ";".join(issues)
        if is_usable:
            usable_count += 1
        else:
            unusable_count += 1
            reasons[unusable_reason] = reasons.get(unusable_reason, 0) + 1

        label_distribution[label_name] = label_distribution.get(label_name, 0) + 1
        records.append(
            Cq08ManifestRecord(
                session_id=f"cq08_{label_name.lower()}_{track_id}_{label_id}_{track_item['track_length_hint']}",
                source_name="CQ-08",
                group_unit="track_id",
                split_group_id=str(track_id),
                frame_level_split_forbidden=True,
                track_id=int(track_id),
                label_id=int(label_id),
                label_name=label_name,
                track_length_hint=int(track_item["track_length_hint"]),
                capture_id=capture_ids[0] if len(capture_ids) == 1 else "",
                track_file=track_item["path"].as_posix(),
                point_file=point_item["path"].as_posix() if point_item is not None else "",
                raw_echo_file=raw_item["path"].as_posix() if raw_item is not None else "",
                rd_mat_files=rd_mat_files,
                num_track_rows=int(track_item["row_count"]),
                num_point_rows=int(point_item["row_count"]) if point_item is not None else 0,
                num_rd_mat_frames=len(rd_mat_files),
                raw_echo_bytes=int(raw_item["raw_echo_bytes"]) if raw_item is not None else 0,
                has_track=True,
                has_points=point_item is not None,
                has_raw_echo=raw_item is not None,
                has_rd_mat=bool(rd_mat_files),
                is_usable=is_usable,
                unusable_reason=unusable_reason,
            )
        )

    summary = {
        "source_name": "CQ-08",
        "group_unit": "track_id",
        "frame_level_split_forbidden": True,
        "total_track_groups": len(records),
        "usable_track_groups": usable_count,
        "unusable_track_groups": unusable_count,
        "label_distribution": label_distribution,
        "unusable_reasons": reasons,
    }
    return records, summary


def write_cq08_manifest(project_root: Path) -> dict[str, object]:
    cq08_root = project_root / "OriData" / "23" / "CQ-08"
    output_root = project_root / "data" / "manifests"
    output_root.mkdir(parents=True, exist_ok=True)

    records, summary = build_cq08_manifest(cq08_root)
    manifest_path = output_root / "cq08_unified_manifest.json"
    summary_path = output_root / "cq08_availability_summary.json"
    split_policy_path = output_root / "cq08_split_policy.json"

    manifest_path.write_text(
        json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    split_policy_path.write_text(
        json.dumps(
            {
                "source_name": "CQ-08",
                "split_group_unit": "track_id",
                "split_group_field": "split_group_id",
                "frame_level_split_forbidden": True,
                "policy": "All track rows, point rows, raw echo files, and RD MAT frames that belong to the same track_id must stay in the same split.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
