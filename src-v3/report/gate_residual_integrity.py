from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "RADAR-LN3-V10-gate-residual-integrity-audit-v1"
DEFAULT_INDEX = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
DEFAULT_RD_INDEX = Path("data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv")
DEFAULT_MANIFEST = Path("data/manifests/cq08_unified_manifest.json")
DEFAULT_RESIDUAL_REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-feature-audit.json"
)
DEFAULT_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/best_model.pt"
)
DEFAULT_OUTPUT = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-integrity-audit.json"
)
EXPECTED_SOURCE_LABEL_IDS = {"BirdTarget": 3, "UnknownTarget": 6}


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _data_row_count(path: Path) -> int:
    with path.open("rb") as stream:
        line_count = sum(1 for _ in stream)
    return max(0, line_count - 1)


def _json_indices(value: Any) -> list[int]:
    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("RD frame-index field must decode to a list")
    return [int(item) for item in payload]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _contiguous(values: Iterable[int], start: int) -> bool:
    ordered = list(values)
    return ordered == list(range(start, start + len(ordered)))


def _filename_contract(path: Path, kind: str) -> tuple[int, int, int | None]:
    patterns = {
        "track": r"^Tracks_(\d+)_(\d+)_(\d+)\.txt$",
        "point": r"^PointTracks_(\d+)_(\d+)_(\d+)\.txt$",
        "raw": r"^(\d+)_Label_(\d+)\.dat$",
    }
    match = re.fullmatch(patterns[kind], path.name)
    if match is None:
        raise ValueError(f"unexpected CQ08 {kind} filename: {path.name}")
    values = tuple(int(item) for item in match.groups())
    return values[0], values[1], values[2] if len(values) == 3 else None


def _load_rd_rows(path: Path, track_ids: set[str]) -> pd.DataFrame:
    columns = [
        "parent_sample_id",
        "track_id",
        "label_name",
        "raw_frame_index",
        "tensor_frame_index",
        "is_tensor_frame_kept",
        "track_point_index",
        "rd_tensor_path",
    ]
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=50_000, low_memory=False):
        part = chunk[chunk["track_id"].astype(str).isin(track_ids)].copy()
        if not part.empty:
            selected.append(part)
    if not selected:
        raise ValueError("physical CQ08 RD index has no residual-parent rows")
    frame = pd.concat(selected, ignore_index=True)
    for name in ("raw_frame_index", "tensor_frame_index", "track_point_index"):
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(int)
    return frame


def run(
    project_root: Path,
    index_path: Path,
    rd_index_path: Path,
    manifest_path: Path,
    residual_report_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    residual = _load_json(residual_report_path)
    wrong_rows = [
        dict(row)
        for row in residual.get("parents", [])
        if str(row.get("prediction_status")) == "wrong"
    ]
    expected_counts = Counter({"BirdTarget": 6, "UnknownTarget": 6})
    actual_counts = Counter(str(row["label_name"]) for row in wrong_rows)
    if len(wrong_rows) != 12 or actual_counts != expected_counts:
        raise ValueError(f"unexpected V10 residual-parent set: {dict(actual_counts)}")

    parent_ids = {str(row["parent_sample_id"]) for row in wrong_rows}
    track_ids = {parent.rsplit("|", 1)[-1] for parent in parent_ids}
    if len(parent_ids) != len(track_ids):
        raise ValueError("residual parent IDs do not map one-to-one to CQ08 track IDs")

    index_columns = [
        "parent_sample_id",
        "track_id",
        "split_name",
        "label_name",
        "causal_event_index",
        "track_point_index",
        "tr_time_seconds",
        "tr_time_order_status",
        "rd_present",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
        "rd_raw_frame_indices_json",
        "rd_parent_coverage_status",
        "rd_parent_order_status",
    ]
    index = pd.read_csv(index_path, usecols=index_columns, low_memory=False)
    index = index[index["parent_sample_id"].astype(str).isin(parent_ids)].copy()
    if set(index["parent_sample_id"].astype(str)) != parent_ids:
        raise ValueError("V3 index does not contain every V10 residual parent")
    for name in ("causal_event_index", "track_point_index"):
        index[name] = pd.to_numeric(index[name], errors="raise").astype(int)
    index["tr_time_seconds"] = pd.to_numeric(index["tr_time_seconds"], errors="raise")

    rd_index = _load_rd_rows(rd_index_path, track_ids)
    manifest_payload = _load_json(manifest_path)
    manifest = {
        str(row["track_id"]): dict(row)
        for row in manifest_payload
        if str(row.get("track_id")) in track_ids
    }
    if set(manifest) != track_ids:
        raise ValueError("CQ08 manifest does not contain every V10 residual track")

    output_rows: list[dict[str, Any]] = []
    for residual_row in sorted(wrong_rows, key=lambda row: int(str(row["parent_sample_id"]).rsplit("|", 1)[-1])):
        parent_id = str(residual_row["parent_sample_id"])
        track_id = parent_id.rsplit("|", 1)[-1]
        label_name = str(residual_row["label_name"])
        expected_label_id = EXPECTED_SOURCE_LABEL_IDS[label_name]
        manifest_row = manifest[track_id]

        event_rows = index[index["parent_sample_id"].astype(str).eq(parent_id)].copy()
        event_rows = event_rows.sort_values("causal_event_index", kind="stable")
        source_rows = rd_index[rd_index["track_id"].astype(str).eq(track_id)].copy()
        source_rows = source_rows.sort_values("raw_frame_index", kind="stable")

        track_path = _manifest_path(project_root, str(manifest_row["track_file"]))
        point_path = _manifest_path(project_root, str(manifest_row["point_file"]))
        raw_path = _manifest_path(project_root, str(manifest_row["raw_echo_file"]))
        mat_paths = [_manifest_path(project_root, str(value)) for value in manifest_row.get("rd_mat_files", [])]
        required_paths = [track_path, point_path, raw_path, *mat_paths]
        all_source_files_exist = all(path.exists() for path in required_paths)
        if not all_source_files_exist:
            missing = [str(path) for path in required_paths if not path.exists()]
            raise FileNotFoundError(f"residual parent has missing CQ08 files: {missing}")

        track_file_id, track_file_label, track_file_length = _filename_contract(track_path, "track")
        point_file_id, point_file_label, point_file_length = _filename_contract(point_path, "point")
        raw_file_id, raw_file_label, _ = _filename_contract(raw_path, "raw")
        manifest_length = int(manifest_row["track_length_hint"])
        event_count = len(event_rows)
        filename_label_contract = (
            {track_file_id, point_file_id, raw_file_id} == {int(track_id)}
            and {track_file_label, point_file_label, raw_file_label, int(manifest_row["label_id"])}
            == {expected_label_id}
            and str(manifest_row["label_name"]) == label_name
            and set(event_rows["label_name"].astype(str)) == {label_name}
            and set(source_rows["label_name"].astype(str)) == {label_name}
        )
        length_contract = (
            track_file_length == point_file_length == manifest_length == event_count
            and _data_row_count(track_path) == manifest_length
            and _data_row_count(point_path) == manifest_length
            and int(residual_row["event_count"]) == manifest_length
        )

        event_indices = event_rows["causal_event_index"].tolist()
        point_indices = event_rows["track_point_index"].tolist()
        event_sequence_contract = (
            _contiguous(event_indices, start=0)
            and _contiguous(point_indices, start=1)
            and set(event_rows["split_name"].astype(str)) == {"val"}
            and set(event_rows["tr_time_order_status"].astype(str)) == {"monotonic"}
            and bool((np.diff(event_rows["tr_time_seconds"].to_numpy(dtype=np.float64)) >= 0.0).all())
        )

        source_raw = source_rows["raw_frame_index"].tolist()
        source_tensor = source_rows["tensor_frame_index"].tolist()
        source_points = source_rows["track_point_index"].tolist()
        source_unique_points = sorted(set(source_points))
        source_order_contract = (
            len(source_raw) == len(set(source_raw))
            and len(source_tensor) == len(set(source_tensor))
            and _contiguous(source_raw, start=0)
            and _contiguous(source_tensor, start=0)
            and bool((np.diff(np.asarray(source_points, dtype=np.int64)) >= 0).all())
            and all(_as_bool(value) for value in source_rows["is_tensor_frame_kept"])
            and set(event_rows["rd_parent_order_status"].astype(str)) == {"monotonic"}
        )

        mapped_raw: list[int] = []
        mapped_tensor: list[int] = []
        mapping_exact = True
        present_points: list[int] = []
        for row in event_rows.itertuples(index=False):
            point_index = int(row.track_point_index)
            raw_indices = _json_indices(row.rd_raw_frame_indices_json)
            tensor_indices = _json_indices(row.rd_tensor_frame_indices_json)
            point_source = source_rows[source_rows["track_point_index"].eq(point_index)]
            source_raw_at_point = point_source["raw_frame_index"].tolist()
            source_tensor_at_point = point_source["tensor_frame_index"].tolist()
            present = _as_bool(row.rd_present)
            if present:
                present_points.append(point_index)
            mapping_exact = mapping_exact and (
                present == bool(len(point_source))
                and raw_indices == source_raw_at_point
                and tensor_indices == source_tensor_at_point
            )
            mapped_raw.extend(raw_indices)
            mapped_tensor.extend(tensor_indices)
        mapping_exact = mapping_exact and (
            mapped_raw == source_raw
            and mapped_tensor == source_tensor
            and len(mapped_raw) == len(set(mapped_raw))
            and len(mapped_tensor) == len(set(mapped_tensor))
        )

        rd_tensor_paths = {
            _manifest_path(project_root, str(value))
            for value in source_rows["rd_tensor_path"].astype(str)
        }
        tensor_contract = len(rd_tensor_paths) == 1 and all(path.exists() for path in rd_tensor_paths)
        coverage = "full" if source_unique_points == list(range(1, event_count + 1)) else "partial"
        coverage_contract = (
            set(event_rows["rd_parent_coverage_status"].astype(str)) == {coverage}
            and present_points == source_unique_points
        )

        checks = {
            "all_source_files_exist": all_source_files_exist,
            "filename_and_manifest_label_codes_consistent": filename_label_contract,
            "track_point_lengths_consistent": length_contract,
            "causal_event_sequence_contiguous_and_monotonic": event_sequence_contract,
            "source_rd_frame_order_unique_and_monotonic": source_order_contract,
            "v3_rd_mapping_exactly_matches_source_frame_index": mapping_exact,
            "rd_tensor_reference_exists": tensor_contract,
            "rd_coverage_status_matches_source_points": coverage_contract,
            "manifest_mat_count_matches_source_frame_count": len(mat_paths) == len(source_rows),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"residual integrity checks failed for {parent_id}: {failed}")

        output_rows.append(
            {
                "parent_sample_id": parent_id,
                "track_id": int(track_id),
                "label_name": label_name,
                "source_label_id": expected_label_id,
                "V10_pred_id": int(residual_row["pred_id"]),
                "V10_final_confidence": float(residual_row["confidence"]),
                "event_count": event_count,
                "track_length_hint": manifest_length,
                "rd_coverage": coverage,
                "rd_present_event_count": len(present_points),
                "rd_missing_event_count": event_count - len(present_points),
                "rd_source_point_min": min(source_unique_points),
                "rd_source_point_max": max(source_unique_points),
                "rd_source_point_count": len(source_unique_points),
                "rd_source_frame_count": len(source_rows),
                "rd_mapped_frame_count": len(mapped_raw),
                "rd_mat_count": len(mat_paths),
                "raw_echo_bytes": raw_path.stat().st_size,
                "checks": checks,
            }
        )

    partial_rows = [row for row in output_rows if row["rd_coverage"] == "partial"]
    result = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "CQ08 V3 validation only; structural audit of the 12 V10 final Gate residual parents; "
            "test excluded; no training, threshold fitting or label mutation"
        ),
        "inputs": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
            "index": {"path": str(index_path), "sha256": _sha256(index_path)},
            "rd_frame_index": {"path": str(rd_index_path), "sha256": _sha256(rd_index_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "residual_feature_report": {
                "path": str(residual_report_path),
                "sha256": _sha256(residual_report_path),
            },
        },
        "summary": {
            "residual_parent_count": len(output_rows),
            "label_counts": dict(actual_counts),
            "full_rd_coverage_parent_count": len(output_rows) - len(partial_rows),
            "partial_rd_coverage_parent_count": len(partial_rows),
            "all_structural_checks_passed": True,
            "manual_semantic_label_review": "未执行",
            "open_set_unknown_review": "未执行",
        },
        "partial_rd_coverage": partial_rows,
        "parents": output_rows,
        "interpretation": [
            "all 12 residual parents have matching manifest/file label codes and complete declared source files",
            "the audit verifies structural label consistency only; it cannot prove that the human semantic class is correct",
            "track 333 has 22 RD MAT frames mapped exactly to source track points 16-22, so its 7/22 event coverage is an upstream late-start/partial acquisition rather than a V3 pairing omission",
            "the other 11 residual parents have full event-level RD coverage and exact one-use frame mappings",
            "UnknownTarget remains a heterogeneous closed-set label here; open-set reject/abstain is not evaluated",
        ],
        "test": "未执行",
        "formal_training": "V10正式训练已由用户完成；本结构审计未启动训练",
        "output_path": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit structural integrity of V10 Gate residual parents.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--rd-index", type=Path, default=DEFAULT_RD_INDEX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--residual-report", type=Path, default=DEFAULT_RESIDUAL_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(
        root,
        _resolve(root, args.index),
        _resolve(root, args.rd_index),
        _resolve(root, args.manifest),
        _resolve(root, args.residual_report),
        _resolve(root, args.checkpoint),
        _resolve(root, args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
