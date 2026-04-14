from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(slots=True)
class CuratedClipManifest:
    sample_ids: set[str]
    spans: list[tuple[str, int, int]]


def split_training_sample_id(sample_id: str) -> tuple[str, int | None]:
    text = str(sample_id)
    if ":" in text:
        parts = text.split(":")
        if len(parts) >= 2:
            try:
                return parts[0], int(parts[-1])
            except ValueError:
                return parts[0], None
        return text, None
    parts = text.split("_")
    if len(parts) >= 3:
        session_id = "_".join(parts[:-2])
        try:
            return session_id, int(parts[-1])
        except ValueError:
            return session_id, None
    return text, None


def load_curated_clip_manifest(manifest_path: str | Path | None) -> CuratedClipManifest | None:
    if manifest_path in {None, "", "None"}:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_ids = {str(item) for item in payload.get("sample_ids", [])}
        spans = [
            (str(item.get("session_id", "")), int(item.get("start_clip_token", 0)), int(item.get("end_clip_token", 0)))
            for item in payload.get("spans", [])
            if str(item.get("session_id", ""))
        ]
        return CuratedClipManifest(sample_ids=sample_ids, spans=spans)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        sample_ids = {str(row["sample_id"]) for row in rows if row.get("sample_id")}
        spans: list[tuple[str, int, int]] = []
        for row in rows:
            session_id = str(row.get("session_id", "") or "")
            start_clip_token = row.get("start_clip_token")
            end_clip_token = row.get("end_clip_token")
            if not session_id or start_clip_token in {None, ""} or end_clip_token in {None, ""}:
                continue
            spans.append((session_id, int(float(start_clip_token)), int(float(end_clip_token))))
        return CuratedClipManifest(sample_ids=sample_ids, spans=spans)
    return None


def build_manifest_mask(sample_ids: Sequence[str], manifest: CuratedClipManifest | None) -> list[float]:
    if manifest is None:
        return [0.0 for _ in sample_ids]
    mask: list[float] = []
    for sample_id in sample_ids:
        sample_id_text = str(sample_id)
        if sample_id_text in manifest.sample_ids:
            mask.append(1.0)
            continue
        session_id, clip_token = split_training_sample_id(sample_id_text)
        if clip_token is None:
            mask.append(0.0)
            continue
        matched = any(
            session_id == span_session_id and start_clip_token <= clip_token <= end_clip_token
            for span_session_id, start_clip_token, end_clip_token in manifest.spans
        )
        mask.append(1.0 if matched else 0.0)
    return mask


def build_curated_manifest_from_spans_csv(
    source_csv: str | Path,
    output_path: str | Path,
    true_label: str | None = None,
    predicted_label: str | None = None,
    min_clips: int = 1,
) -> Path:
    source_path = Path(source_csv)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    spans: list[dict[str, Any]] = []
    for row in rows:
        row_true_label = str(row.get("true_label", "") or "")
        row_predicted_label = str(row.get("dominant_predicted_label", "") or row.get("predicted_label", "") or "")
        row_num_clips = int(float(row.get("num_clips", 0) or 0))
        if true_label is not None and row_true_label != str(true_label):
            continue
        if predicted_label is not None and row_predicted_label != str(predicted_label):
            continue
        if row_num_clips < int(min_clips):
            continue
        session_id = str(row.get("session_id", "") or "")
        if not session_id:
            continue
        spans.append(
            {
                "session_id": session_id,
                "start_clip_token": int(float(row.get("start_clip_token", 0) or 0)),
                "end_clip_token": int(float(row.get("end_clip_token", 0) or 0)),
                "true_label": row_true_label,
                "predicted_label": row_predicted_label,
                "num_clips": row_num_clips,
            }
        )

    payload = {
        "version": 1,
        "source_csv": str(source_path),
        "sample_ids": [],
        "spans": spans,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination
