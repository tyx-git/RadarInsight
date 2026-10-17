"""Prepare strict capture/content-grouped paired OOF support for V21e KnownGate."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from radar_ln3.known_gate import KNOWN_LABELS
from radar_ln3.known_gate_v21e_oof import ASSIGNMENT_SCHEMA, ASSIGNMENT_SEED, FOLD_COUNT, canonical_sha256, sequence_sha256


LABELS = tuple(KNOWN_LABELS)
EXPECTED_COUNTS = {
    "DroneTarget": 476,
    "BirdTarget": 239,
    "BalloonTarget": 238,
    "ClutterTarget": 33,
}


def _tie_hash(component_id: str) -> str:
    return hashlib.sha256(
        f"{ASSIGNMENT_SEED}|{component_id}".encode("utf-8")
    ).hexdigest()


def _parent_records(index_path: Path) -> list[dict[str, Any]]:
    required = {
        "schema_version",
        "parent_sample_id",
        "split_name",
        "label_name",
        "provenance_component_id",
        "capture_id",
    }
    frame = pd.read_csv(index_path, usecols=lambda name: name in required, low_memory=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V21e OOF index lacks columns: {missing}")
    if set(frame["schema_version"].astype(str)) != {"RADAR-LN3-V16"}:
        raise ValueError("V21e OOF source index schema changed")
    selected = frame[
        frame["split_name"].astype(str).eq("train")
        & frame["label_name"].astype(str).isin(LABELS)
    ].copy()
    grouped = selected.groupby("parent_sample_id", sort=False)
    records: list[dict[str, Any]] = []
    for parent_id, group in grouped:
        labels = set(group["label_name"].astype(str))
        components = set(group["provenance_component_id"].astype(str))
        captures = set(group["capture_id"].astype(str))
        if len(labels) != 1 or len(components) != 1 or len(captures) != 1:
            raise ValueError(f"V21e OOF parent provenance is inconsistent: {parent_id}")
        records.append(
            {
                "parent_sample_id": str(parent_id),
                "label_name": labels.pop(),
                "provenance_component_id": components.pop(),
                "capture_id": captures.pop(),
                "event_count": int(len(group)),
            }
        )
    records.sort(key=lambda row: row["parent_sample_id"])
    counts = Counter(str(row["label_name"]) for row in records)
    if len(records) != 986 or dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"V21e OOF known train support changed: {dict(counts)}")
    if len({row["parent_sample_id"] for row in records}) != len(records):
        raise ValueError("V21e OOF parent records repeat an identifier")
    return records


def build_assignments(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["provenance_component_id"])].append(dict(row))
    components: list[dict[str, Any]] = []
    for component_id, rows in grouped.items():
        counts = Counter(str(row["label_name"]) for row in rows)
        components.append(
            {
                "component_id": component_id,
                "rows": sorted(rows, key=lambda row: row["parent_sample_id"]),
                "counts": {label: int(counts.get(label, 0)) for label in LABELS},
                "parent_count": len(rows),
                "label_signature": "+".join(
                    f"{label}:{int(counts.get(label, 0))}"
                    for label in LABELS
                    if counts.get(label, 0)
                ),
            }
        )
    if len(components) != 975 or max(row["parent_count"] for row in components) != 2:
        raise ValueError("V21e OOF provenance component support changed")
    targets = {label: EXPECTED_COUNTS[label] / FOLD_COUNT for label in LABELS}
    total_target = len(records) / FOLD_COUNT
    fold_counts = [{label: 0 for label in LABELS} for _ in range(FOLD_COUNT)]
    fold_totals = [0 for _ in range(FOLD_COUNT)]
    component_fold: dict[str, int] = {}
    components.sort(
        key=lambda row: (
            -int(row["counts"]["ClutterTarget"]),
            -sum(value > 0 for value in row["counts"].values()),
            -int(row["parent_count"]),
            _tie_hash(str(row["component_id"])),
        )
    )
    for component in components:
        candidates: list[tuple[float, int, int]] = []
        for fold_id in range(FOLD_COUNT):
            label_penalty = 0.0
            for label in LABELS:
                if int(component["counts"][label]) == 0:
                    continue
                after = fold_counts[fold_id][label] + int(component["counts"][label])
                label_penalty += (after / max(1.0, targets[label])) ** 2
            after_total = fold_totals[fold_id] + int(component["parent_count"])
            total_penalty = (after_total / total_target) ** 2
            candidates.append((label_penalty * 10.0 + total_penalty, fold_totals[fold_id], fold_id))
        _, _, selected_fold = min(candidates)
        component_fold[str(component["component_id"])] = selected_fold
        fold_totals[selected_fold] += int(component["parent_count"])
        for label in LABELS:
            fold_counts[selected_fold][label] += int(component["counts"][label])
    assignments: list[dict[str, Any]] = []
    for row in records:
        assignments.append(
            {
                **row,
                "fold_id": int(component_fold[str(row["provenance_component_id"])]),
            }
        )
    assignments.sort(key=lambda row: row["parent_sample_id"])
    component_folds: dict[str, set[int]] = defaultdict(set)
    for row in assignments:
        component_folds[str(row["provenance_component_id"])].add(int(row["fold_id"]))
    if any(len(values) != 1 for values in component_folds.values()):
        raise ValueError("V21e OOF split a provenance component across folds")
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        selected = [row for row in assignments if int(row["fold_id"]) == fold_id]
        counts = Counter(str(row["label_name"]) for row in selected)
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "held_out_parent_count": len(selected),
                "held_out_component_count": len(
                    {str(row["provenance_component_id"]) for row in selected}
                ),
                "held_out_label_parent_counts": {
                    label: int(counts.get(label, 0)) for label in LABELS
                },
                "held_out_parent_ids_sha256": sequence_sha256(
                    sorted(str(row["parent_sample_id"]) for row in selected)
                ),
            }
        )
    for label in LABELS:
        values = [row["held_out_label_parent_counts"][label] for row in fold_summaries]
        if max(values) - min(values) > 1:
            raise ValueError(f"V21e OOF {label} support is not balanced across folds: {values}")
    return {
        "schema_version": ASSIGNMENT_SCHEMA,
        "fit_split": "train",
        "fit_population": "known_only",
        "prediction_protocol": "out_of_fold",
        "grouping": "capture_content_provenance_component",
        "assignment_seed": ASSIGNMENT_SEED,
        "fold_count": FOLD_COUNT,
        "known_parent_count": len(assignments),
        "provenance_component_count": len(component_folds),
        "max_component_parent_count": max(
            sum(1 for row in assignments if row["provenance_component_id"] == component)
            for component in component_folds
        ),
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
        "known_parent_ids_sha256": sequence_sha256(
            sorted(str(row["parent_sample_id"]) for row in assignments)
        ),
        "assignments_sha256": canonical_sha256(assignments),
        "folds": fold_summaries,
        "assignments": assignments,
    }









