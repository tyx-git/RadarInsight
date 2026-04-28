from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import pandas as pd

from .cq08_manifest import write_cq08_manifest


RAW_PATTERN = re.compile(
    r"^(?P<timestamp>\d+)_DAUR_(?P<kind>TR|RD)_(?P<label>.+)_(?P<segment>\d+)_(?P<sample>\d+)\.mat$"
)
EXCEL_PATTERN = re.compile(r"^(?P<prefix>tck|bz)_(?P<timestamp>[0-9_]+)\.xlsx$")


@dataclass
class RawPairRecord:
    pair_id: str
    timestamp: str
    label: str
    segment: str
    sample: str
    tr_file: str
    rd_file: str
    frames: int
    tr_points: int
    rd_bins: int


@dataclass
class ExcelPairRecord:
    pair_id: str
    timestamp: str
    track_file: str
    mark_file: str
    track_rows: int
    mark_rows: int
    track_columns: List[str]
    mark_columns: List[str]
    track_semantics: Dict[str, str]
    mark_semantics: Dict[str, str]


def _read_raw_meta(path: Path) -> Dict[str, int]:
    with h5py.File(path, "r") as handle:
        data_time = handle["DATA_time"]
        frames = int(data_time.shape[0])
        if "R" in handle:
            tr_points = int(handle["R"].shape[0])
        else:
            tr_points = 0
        if "DPL" in handle:
            rd_bins = int(handle["DPL"].shape[0])
        else:
            rd_bins = 0
    return {"frames": frames, "tr_points": tr_points, "rd_bins": rd_bins}


def build_raw_manifest(raw_root: Path) -> List[RawPairRecord]:
    tr_dir = raw_root / "TR"
    rd_dir = raw_root / "RD"
    rd_map: Dict[str, Path] = {}
    for rd_path in sorted(rd_dir.glob("*.mat")):
        match = RAW_PATTERN.match(rd_path.name)
        if not match:
            continue
        key = "{timestamp}|{label}|{segment}|{sample}".format(**match.groupdict())
        rd_map[key] = rd_path
    records: List[RawPairRecord] = []
    for tr_path in sorted(tr_dir.glob("*.mat")):
        match = RAW_PATTERN.match(tr_path.name)
        if not match:
            continue
        key = "{timestamp}|{label}|{segment}|{sample}".format(**match.groupdict())
        rd_path = rd_map.get(key)
        if rd_path is None:
            continue
        tr_meta = _read_raw_meta(tr_path)
        rd_meta = _read_raw_meta(rd_path)
        records.append(
            RawPairRecord(
                pair_id=key,
                timestamp=match.group("timestamp"),
                label=match.group("label"),
                segment=match.group("segment"),
                sample=match.group("sample"),
                tr_file=str(tr_path.as_posix()),
                rd_file=str(rd_path.as_posix()),
                frames=min(tr_meta["frames"], rd_meta["frames"]),
                tr_points=tr_meta["tr_points"],
                rd_bins=rd_meta["rd_bins"],
            )
        )
    return records


def build_excel_manifest(excel_root: Path) -> List[ExcelPairRecord]:
    tck_map: Dict[str, Path] = {}
    bz_map: Dict[str, Path] = {}
    for path in sorted(excel_root.glob("*.xlsx")):
        match = EXCEL_PATTERN.match(path.name)
        if not match:
            continue
        timestamp = match.group("timestamp")
        prefix = match.group("prefix")
        if prefix == "tck":
            tck_map[timestamp] = path
        else:
            bz_map[timestamp] = path
    records: List[ExcelPairRecord] = []
    track_semantics = {
        "StartTimer": "原始起始时间标识",
        "Batch": "批次号",
        "Azimuth": "方位角，度",
        "Distance": "距离，米",
        "Latitude": "纬度",
        "longitude": "经度",
        "Speed_R": "径向速度，米每秒",
        "Level": "层级/级别字段",
        "Elevation": "仰角，度",
        "Course": "航向角，度",
        "Speed": "目标速度，米每秒",
        "Height": "高度，米",
        "Time": "顺序时间/计数时间",
        "Amplitude": "幅度",
        "Type": "区域/类型字段",
        "Area": "区域",
        "Noise": "噪声值",
        "sPictuerUrl": "图片链接",
        "sVideoUrl": "视频链接",
    }
    mark_semantics = {
        "StartTimer": "原始起始时间标识",
        "Batch": "批次号",
        "MarkTypeID": "标注目标类型编号",
        "NotesID": "备注类型编号",
        "MarkType": "标注目标类型名称",
        "Notes": "备注文本",
        "Unnamed: 6": "人工记录时间文本",
    }
    for timestamp, tck_path in sorted(tck_map.items()):
        bz_path = bz_map.get(timestamp)
        if bz_path is None:
            continue
        tck_df = pd.read_excel(tck_path)
        bz_df = pd.read_excel(bz_path)
        records.append(
            ExcelPairRecord(
                pair_id=timestamp,
                timestamp=timestamp,
                track_file=str(tck_path.as_posix()),
                mark_file=str(bz_path.as_posix()),
                track_rows=int(len(tck_df)),
                mark_rows=int(len(bz_df)),
                track_columns=[str(col) for col in tck_df.columns],
                mark_columns=[str(col) for col in bz_df.columns],
                track_semantics=track_semantics,
                mark_semantics=mark_semantics,
            )
        )
    return records


def write_manifests(project_root: Path) -> Dict[str, int]:
    raw_root = project_root / "OriData" / "raw"
    excel_root = project_root / "OriData" / "excel"
    output_root = project_root / "data" / "manifests"
    output_root.mkdir(parents=True, exist_ok=True)

    raw_records = build_raw_manifest(raw_root)
    excel_records = build_excel_manifest(excel_root)

    raw_json = output_root / "raw_tr_rd_manifest.json"
    excel_json = output_root / "excel_track_manifest.json"
    summary_json = output_root / "manifest_summary.json"

    raw_json.write_text(
        json.dumps([asdict(record) for record in raw_records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    excel_json.write_text(
        json.dumps([asdict(record) for record in excel_records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "raw_pairs": len(raw_records),
        "excel_pairs": len(excel_records),
        "raw_labels": sorted({record.label for record in raw_records}),
        "excel_timestamps": [record.timestamp for record in excel_records],
    }
    cq08_summary = write_cq08_manifest(project_root)
    summary["cq08_track_groups"] = int(cq08_summary["total_track_groups"])
    summary["cq08_usable_track_groups"] = int(cq08_summary["usable_track_groups"])
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary = write_manifests(project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
