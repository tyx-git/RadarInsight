from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_rd_novel_evidence import FrameReference, WindowEvidence, load_window_evidence


SCHEMA_VERSION = "expert1_cq08_physical_novel_update_v2_exact_once"
DEFAULT_MAX_FRAMES_PER_CHUNK = 32
REQUIRED_FRAME_INDEX_COLUMNS = {
    "parent_sample_id",
    "rd_tensor_path",
    "tensor_frame_index",
    "is_tensor_frame_kept",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parent_tensor_paths(frame_index: pd.DataFrame) -> dict[str, str]:
    missing_columns = sorted(REQUIRED_FRAME_INDEX_COLUMNS.difference(frame_index.columns))
    if missing_columns:
        raise ValueError(f"physical canonical frame index missing columns: {missing_columns}")
    kept = frame_index[frame_index["is_tensor_frame_kept"].map(_as_bool)].copy()
    parent_paths: dict[str, str] = {}
    for parent_sample_id, group in kept.groupby("parent_sample_id", sort=False):
        paths = set(group["rd_tensor_path"].astype(str))
        if len(paths) != 1:
            raise ValueError(f"parent has multiple physical tensor paths: {parent_sample_id}")
        parent_paths[str(parent_sample_id)] = paths.pop()
    return parent_paths


def _chunked_references(references: list[FrameReference], max_frames_per_chunk: int) -> list[list[FrameReference]]:
    if max_frames_per_chunk <= 0:
        raise ValueError("max_frames_per_chunk must be positive")
    if not references:
        return [[]]
    return [
        references[start_index:start_index + max_frames_per_chunk]
        for start_index in range(0, len(references), max_frames_per_chunk)
    ]


def build_novel_update_rows(
    evidence_rows: list[WindowEvidence],
    parent_tensor_paths: dict[str, str],
    *,
    max_frames_per_chunk: int,
) -> list[dict[str, object]]:
    rows_by_parent: dict[str, list[WindowEvidence]] = {}
    for evidence in evidence_rows:
        rows_by_parent.setdefault(evidence.parent_sample_id, []).append(evidence)
    index_rows: list[dict[str, object]] = []
    for parent_sample_id in sorted(rows_by_parent):
        parent_windows = sorted(
            rows_by_parent[parent_sample_id],
            key=lambda item: (item.window_end_seconds, item.window_start_seconds, item.sample_id),
        )
        labels = {item.label_name for item in parent_windows}
        splits = {item.split_name for item in parent_windows}
        if len(labels) != 1 or len(splits) != 1:
            raise ValueError(f"parent identity contract failed for {parent_sample_id}")
        tensor_path = parent_tensor_paths.get(parent_sample_id)
        if tensor_path is None:
            raise ValueError(f"parent has no physical tensor path: {parent_sample_id}")
        seen_tensor_indices: set[int] = set()
        previous_window_end_seconds: float | None = None
        for update_index, evidence in enumerate(parent_windows):
            current_tensor_indices = [reference.tensor_frame_index for reference in evidence.frame_references]
            if len(current_tensor_indices) != len(set(current_tensor_indices)):
                raise ValueError(f"window evidence has duplicate tensor reference: {evidence.sample_id}")
            novel_references = [
                reference
                for reference in evidence.frame_references
                if reference.tensor_frame_index not in seen_tensor_indices
            ]
            seen_tensor_indices.update(reference.tensor_frame_index for reference in novel_references)
            chunks = _chunked_references(novel_references, max_frames_per_chunk)
            delta_time_seconds = (
                0.0
                if previous_window_end_seconds is None
                else float(evidence.window_end_seconds - previous_window_end_seconds)
            )
            if delta_time_seconds < 0.0:
                raise ValueError(f"non-causal update order for {evidence.sample_id}")
            for chunk_index, chunk_references in enumerate(chunks):
                index_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": evidence.sample_id,
                        "parent_sample_id": parent_sample_id,
                        "split_name": evidence.split_name,
                        "label_name": evidence.label_name,
                        "update_index": int(update_index),
                        "window_start_seconds": float(evidence.window_start_seconds),
                        "window_end_seconds": float(evidence.window_end_seconds),
                        "delta_time_seconds": delta_time_seconds if chunk_index == 0 else 0.0,
                        "current_frame_count": int(len(current_tensor_indices)),
                        "novel_frame_count": int(len(novel_references)),
                        "chunk_index": int(chunk_index),
                        "chunk_count": int(len(chunks)),
                        "is_final_chunk": bool(chunk_index == len(chunks) - 1),
                        "is_zero_novel_update": bool(not novel_references),
                        "chunk_frame_count": int(len(chunk_references)),
                        "rd_tensor_path": tensor_path,
                        "chunk_raw_frame_indices_json": json.dumps(
                            [reference.raw_frame_index for reference in chunk_references],
                            separators=(",", ":"),
                        ),
                        "chunk_tensor_frame_indices_json": json.dumps(
                            [reference.tensor_frame_index for reference in chunk_references],
                            separators=(",", ":"),
                        ),
                        "chunk_anchor_times_seconds_json": json.dumps(
                            [reference.anchor_time_seconds for reference in chunk_references],
                            separators=(",", ":"),
                        ),
                    }
                )
            previous_window_end_seconds = evidence.window_end_seconds
    return index_rows


def _validate_index_rows(index_rows: list[dict[str, object]], evidence_rows: list[WindowEvidence]) -> dict[str, object]:
    expected_by_parent: dict[str, set[int]] = {}
    for evidence in evidence_rows:
        expected_by_parent.setdefault(evidence.parent_sample_id, set()).update(
            reference.tensor_frame_index for reference in evidence.frame_references
        )
    observed_by_parent: dict[str, set[int]] = {}
    update_chunks: dict[tuple[str, int], list[dict[str, object]]] = {}
    zero_novel_count = 0
    for row in index_rows:
        parent_sample_id = str(row["parent_sample_id"])
        update_key = (parent_sample_id, int(row["update_index"]))
        update_chunks.setdefault(update_key, []).append(row)
        tensor_indices = json.loads(str(row["chunk_tensor_frame_indices_json"]))
        if int(row["chunk_frame_count"]) != len(tensor_indices):
            raise ValueError(f"chunk count mismatch for {update_key}")
        observed = observed_by_parent.setdefault(parent_sample_id, set())
        repeated = observed.intersection(tensor_indices)
        if repeated:
            raise ValueError(f"novel index emits duplicate frames for {update_key}: {sorted(repeated)[:3]}")
        observed.update(int(value) for value in tensor_indices)
        zero_novel_count += int(bool(row["is_zero_novel_update"]))
    for update_key, chunks in update_chunks.items():
        ordered = sorted(chunks, key=lambda row: int(row["chunk_index"]))
        expected_chunk_indices = list(range(len(ordered)))
        actual_chunk_indices = [int(row["chunk_index"]) for row in ordered]
        if actual_chunk_indices != expected_chunk_indices:
            raise ValueError(f"non-contiguous chunk indices for {update_key}")
        if int(sum(bool(row["is_final_chunk"]) for row in ordered)) != 1:
            raise ValueError(f"invalid final chunk count for {update_key}")
        if not bool(ordered[-1]["is_final_chunk"]):
            raise ValueError(f"last chunk is not final for {update_key}")
        if any(abs(float(row["delta_time_seconds"])) > 1.0e-9 for row in ordered[1:]):
            raise ValueError(f"novel update delta time repeats across chunks for {update_key}")
    reconstruction_failure_count = int(
        sum(observed_by_parent.get(parent_sample_id, set()) != expected_frames for parent_sample_id, expected_frames in expected_by_parent.items())
    )
    return {
        "parent_count": int(len(expected_by_parent)),
        "update_count": int(len(update_chunks)),
        "chunk_count": int(len(index_rows)),
        "zero_novel_chunk_count": int(zero_novel_count),
        "reconstruction_failure_count": reconstruction_failure_count,
    }


def _build_summary(index_rows: list[dict[str, object]], validation: dict[str, object], max_frames_per_chunk: int) -> dict[str, object]:
    frame_counts = np.asarray([int(row["chunk_frame_count"]) for row in index_rows], dtype=np.int64)
    split_counts = Counter(str(row["split_name"]) for row in index_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "max_frames_per_chunk": int(max_frames_per_chunk),
        "split_chunk_counts": dict(sorted(split_counts.items())),
        "chunk_frame_count": {
            "min": int(frame_counts.min()) if frame_counts.size else 0,
            "max": int(frame_counts.max()) if frame_counts.size else 0,
            "mean": float(frame_counts.mean()) if frame_counts.size else 0.0,
            "p95": float(np.quantile(frame_counts, 0.95)) if frame_counts.size else 0.0,
        },
        "validation": validation,
    }


def build_novel_update_index(
    project_root: Path,
    *,
    index_name: str,
    window_map_name: str,
    manifest_name: str,
    frame_index_name: str,
    output_name: str,
    summary_name: str,
    max_frames_per_chunk: int,
) -> dict[str, object]:
    evidence_rows, _, _ = load_window_evidence(
        project_root,
        index_name=index_name,
        window_map_name=window_map_name,
        manifest_name=manifest_name,
        frame_index_name=frame_index_name,
    )
    frame_index = pd.read_csv(project_root / "data" / "processed" / frame_index_name, low_memory=False)
    parent_tensor_paths = _parent_tensor_paths(frame_index)
    index_rows = build_novel_update_rows(
        evidence_rows,
        parent_tensor_paths,
        max_frames_per_chunk=max_frames_per_chunk,
    )
    repeated_rows = build_novel_update_rows(
        evidence_rows,
        parent_tensor_paths,
        max_frames_per_chunk=max_frames_per_chunk,
    )
    if index_rows != repeated_rows:
        raise ValueError("novel update index generation is not deterministic")
    validation = _validate_index_rows(index_rows, evidence_rows)
    if int(validation["reconstruction_failure_count"]) != 0:
        raise ValueError("novel update index does not reconstruct physical evidence")
    output_path = project_root / "data" / "processed" / output_name
    frame = pd.DataFrame(index_rows)
    frame.to_csv(output_path, index=False)
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    summary = _build_summary(index_rows, validation, max_frames_per_chunk)
    summary.update(
        {
            "scope": "deterministic physical-v2 novel-update index; no tensor rebuild, model training, fusion, threshold tuning, or test evaluation",
            "inputs": {
                "index": index_name,
                "window_map": window_map_name,
                "manifest": manifest_name,
                "frame_index": frame_index_name,
            },
            "output": output_name,
            "sha256": output_hash,
            "test_evaluation_executed": False,
            "fusion_enabled": False,
        }
    )
    summary_path = project_root / "tmp" / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic non-overlapping physical-v2 RD update index.")
    parser.add_argument("--index-name", default="expert1_realtime_causal_window_index_v7.csv")
    parser.add_argument("--window-map-name", default="expert1_realtime_causal_window_cq08_physical_anchor_map_v4.csv")
    parser.add_argument("--manifest-name", default="expert1_cq08_rd_physical_anchor_manifest_v4.csv")
    parser.add_argument("--frame-index-name", default="cq08_rd_physical_canonical_frame_index_v2.csv")
    parser.add_argument("--output-name", default="expert1_cq08_rd_physical_novel_update_index_v2.csv")
    parser.add_argument("--summary-name", default="20260727-stage009q-novel-update-index-summary.json")
    parser.add_argument("--max-frames-per-chunk", type=int, default=DEFAULT_MAX_FRAMES_PER_CHUNK)
    args = parser.parse_args()
    payload = build_novel_update_index(
        Path.cwd(),
        index_name=args.index_name,
        window_map_name=args.window_map_name,
        manifest_name=args.manifest_name,
        frame_index_name=args.frame_index_name,
        output_name=args.output_name,
        summary_name=args.summary_name,
        max_frames_per_chunk=args.max_frames_per_chunk,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
