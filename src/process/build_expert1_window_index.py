from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.core.data.content_fingerprint import (
    FILE_CONTENT_FINGERPRINT_ALGORITHM,
    TRACK_CONTENT_FINGERPRINT_ALGORITHM,
    file_content_fingerprint,
    track_content_fingerprint,
)
from src.core.data.labels import two_layer_label_fields
from src.process.schema import TRACK_INDEX_COLUMNS, TRACK_SCHEMA_COLUMNS


EXPERT1_LABELS = {
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
}

WINDOW_EXTRA_COLUMNS = [
    "is_trainable",
    "exclude_reason",
    "parent_sample_id",
    "window_id",
    "window_start_seconds",
    "window_end_seconds",
    "window_duration_seconds",
    "window_point_count",
    "window_unique_time_count",
    "is_causal_window",
    "window_mode",
    "track_content_fingerprint",
    "track_content_fingerprint_status",
    "raw_echo_content_fingerprint",
    "raw_echo_content_fingerprint_status",
    "rd_content_fingerprint",
    "rd_content_fingerprint_status",
    "content_group_id",
    "content_group_size",
    "content_canonical_parent_id",
    "is_content_canonical",
    "window_content_fingerprint",
    "split_group_id",
]

WINDOW_INDEX_COLUMNS = TRACK_INDEX_COLUMNS + WINDOW_EXTRA_COLUMNS

RD_CONTENT_FINGERPRINT_ALGORITHM = "sha256-npz-arrays-v1"
CONTENT_COMPONENT_ALGORITHM = "sha256-content-component-v1"
CONTENT_COMPONENT_GROUP_PREFIX = "content-component:"
WINDOW_CONTENT_FINGERPRINT_ALGORITHM = "sha256-window-model-inputs-v1"
WINDOW_CONTENT_GROUP_ALGORITHM = "sha256-window-split-component-v1"
WINDOW_CONTENT_GROUP_PREFIX = "window-content-component:"
SESSION_SPLIT_GROUP_ALGORITHM = "sha256-session-split-component-v1"
SESSION_SPLIT_GROUP_PREFIX = "session-split-component:"
PARENT_FALLBACK_GROUP_PREFIX = "parent:"
SPLIT_EVIDENCE_CONTRACTS = {"multimodal_v1", "tr_only_v1"}

RD_CONTENT_FINGERPRINT_AUDIT_COLUMNS = [
    "parent_sample_id",
    "source_name",
    "label_name",
    "rd_tensor_path",
    "track_content_fingerprint",
    "track_content_fingerprint_status",
    "raw_echo_content_fingerprint",
    "raw_echo_content_fingerprint_status",
    "rd_content_fingerprint",
    "rd_content_fingerprint_status",
    "content_group_id",
    "content_group_size",
    "content_canonical_parent_id",
    "is_content_canonical",
    "split_group_id",
    "split_name",
]


def _safe_name(value: str) -> str:
    safe = str(value).replace("|", "__").replace("/", "_").replace("\\", "_").replace(":", "_")
    return safe.replace(" ", "_")


def _is_present(value: object) -> bool:
    text = str(value if value is not None else "").strip().lower()
    return text not in {"", "nan", "none"}


def _metadata_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan", "none"}:
        return False
    raise ValueError(f"invalid boolean metadata for {field_name}: {value!r}")


def _window_content_fingerprint(
    frame: pd.DataFrame,
    *,
    source_name: str,
    height_missing: bool,
    rd_phase_missing: bool,
    include_rd_phase_missing: bool = True,
) -> str:
    track_fingerprint = track_content_fingerprint(frame)
    digest = hashlib.sha256()
    digest.update(WINDOW_CONTENT_FINGERPRINT_ALGORITHM.encode("utf-8"))
    metadata_values = [
        str(source_name),
        f"height_missing={str(bool(height_missing)).lower()}",
    ]
    if include_rd_phase_missing:
        metadata_values.append(f"rd_phase_missing={str(bool(rd_phase_missing)).lower()}")
    for value in metadata_values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    digest.update(track_fingerprint.encode("ascii"))
    return digest.hexdigest()


def _merge_window_split_groups(window_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Make parent groups atomic when any generated model-input window matches."""
    required = {"split_group_id", "window_content_fingerprint", "label_name"}
    missing = sorted(required - set(window_df.columns))
    if missing:
        raise ValueError(f"window content grouping requires columns: {missing}")
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for _, group in window_df.groupby("window_content_fingerprint", sort=True):
        original_groups = sorted(set(group["split_group_id"].astype(str)))
        labels = sorted(set(group["label_name"].astype(str)))
        if len(labels) != 1:
            raise ValueError(
                "identical window model inputs have conflicting labels: "
                f"labels={labels}, groups={original_groups}"
            )
        for group_id in original_groups:
            adjacency[group_id].update(other for other in original_groups if other != group_id)
    all_groups = sorted(set(window_df["split_group_id"].astype(str)))
    mapping: dict[str, str] = {}
    members_by_merged: dict[str, set[str]] = {}
    visited: set[str] = set()
    for group_id in all_groups:
        if group_id in visited:
            continue
        stack = [group_id]
        component: set[str] = set()
        fingerprints: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            fingerprints.update(
                window_df.loc[
                    window_df["split_group_id"].astype(str).eq(current),
                    "window_content_fingerprint",
                ].astype(str).tolist()
            )
            stack.extend(sorted(adjacency[current] - visited))
        digest = hashlib.sha256()
        digest.update(WINDOW_CONTENT_GROUP_ALGORITHM.encode("utf-8"))
        for value in sorted(component | fingerprints):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
        merged_id = f"{WINDOW_CONTENT_GROUP_PREFIX}{digest.hexdigest()}"
        for original in component:
            mapping[original] = merged_id
        members_by_merged[merged_id] = component
    result = window_df.copy()
    result["split_group_id"] = result["split_group_id"].astype(str).map(mapping)
    return result, members_by_merged


def _merge_session_split_groups(window_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Make capture/session groups atomic without using labels or RD metadata."""
    required = {"split_group_id", "source_name", "timestamp"}
    missing = sorted(required - set(window_df.columns))
    if missing:
        raise ValueError(f"session grouping requires columns: {missing}")
    session_values = (
        window_df["source_name"].astype(str).str.strip()
        + "|"
        + window_df["timestamp"].astype(str).str.strip()
    )
    malformed = session_values.str.endswith("|") | session_values.str.contains(
        r"\|(?:nan|none)$",
        case=False,
        regex=True,
    )
    if bool(malformed.any()):
        examples = window_df.loc[malformed, ["source_name", "timestamp"]].head(5).to_dict(orient="records")
        raise ValueError(f"TR-only session grouping requires source_name and timestamp: examples={examples}")

    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    session_keys_by_group: defaultdict[str, set[str]] = defaultdict(set)
    grouped = window_df.assign(_session_key=session_values).groupby("_session_key", sort=True)
    for session_key, session_df in grouped:
        group_ids = sorted(set(session_df["split_group_id"].astype(str)))
        for group_id in group_ids:
            session_keys_by_group[group_id].add(str(session_key))
            adjacency[group_id].update(other for other in group_ids if other != group_id)

    all_groups = sorted(set(window_df["split_group_id"].astype(str)))
    mapping: dict[str, str] = {}
    members_by_merged: dict[str, set[str]] = {}
    visited: set[str] = set()
    for group_id in all_groups:
        if group_id in visited:
            continue
        stack = [group_id]
        component: set[str] = set()
        session_keys: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            session_keys.update(session_keys_by_group[current])
            stack.extend(sorted(adjacency[current] - visited))
        digest = hashlib.sha256()
        digest.update(SESSION_SPLIT_GROUP_ALGORITHM.encode("utf-8"))
        for value in sorted(component | session_keys):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
        merged_id = f"{SESSION_SPLIT_GROUP_PREFIX}{digest.hexdigest()}"
        for original in component:
            mapping[original] = merged_id
        members_by_merged[merged_id] = component
    result = window_df.copy()
    result["split_group_id"] = result["split_group_id"].astype(str).map(mapping)
    return result, members_by_merged


def _resolve_project_path(project_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else project_root / path


def _canonical_array_for_fingerprint(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("RD tensor content fingerprint does not support object arrays")
    if array.dtype.byteorder not in {"|", "<"}:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def _rd_tensor_content_fingerprint(tensor_path: Path) -> str:
    if tensor_path.suffix.lower() != ".npz":
        raise ValueError(f"RD content fingerprint requires .npz tensor: {tensor_path}")
    digest = hashlib.sha256()
    digest.update(RD_CONTENT_FINGERPRINT_ALGORITHM.encode("utf-8"))
    try:
        with np.load(tensor_path, allow_pickle=False) as payload:
            keys = sorted(payload.files)
            if not keys:
                raise ValueError("RD tensor archive contains no arrays")
            for key in keys:
                array = _canonical_array_for_fingerprint(np.asarray(payload[key]))
                dtype_text = array.dtype.str.encode("ascii")
                shape_text = ",".join(str(int(value)) for value in array.shape).encode("ascii")
                digest.update(len(key).to_bytes(4, "little"))
                digest.update(key.encode("utf-8"))
                digest.update(len(dtype_text).to_bytes(4, "little"))
                digest.update(dtype_text)
                digest.update(len(shape_text).to_bytes(4, "little"))
                digest.update(shape_text)
                digest.update(array.tobytes(order="C"))
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"failed to fingerprint RD tensor: {tensor_path}") from error
    return digest.hexdigest()


def _content_component_id(evidence_keys: set[str], parent_sample_id: str) -> str:
    """Return a stable content-component key without using labels or split names."""
    digest = hashlib.sha256()
    digest.update(CONTENT_COMPONENT_ALGORITHM.encode("utf-8"))
    if evidence_keys:
        for key in sorted(evidence_keys):
            encoded = key.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    else:
        encoded = str(parent_sample_id).encode("utf-8")
        digest.update(b"parent-fallback")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return f"{CONTENT_COMPONENT_GROUP_PREFIX}{digest.hexdigest()}"


def _build_content_fingerprint_records(
    index_df: pd.DataFrame,
    project_root: Path,
    *,
    split_evidence_contract: str = "multimodal_v1",
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Build source-local semantic duplicate components for Expert1 parents.

    Track-model-input equality is always checked.  ``multimodal_v1`` also uses
    CQ08 raw-echo and processed-RD equality.  ``tr_only_v1`` neither reads nor
    hashes those RD assets.  Labels are excluded from hashes and a conflicting-
    label component is rejected rather than silently trained.
    """
    normalized_contract = str(split_evidence_contract).strip().lower()
    if normalized_contract not in SPLIT_EVIDENCE_CONTRACTS:
        raise ValueError(
            "unsupported split evidence contract: "
            f"{split_evidence_contract!r}; expected one of {sorted(SPLIT_EVIDENCE_CONTRACTS)}"
        )
    use_rd_evidence = normalized_contract == "multimodal_v1"
    required_columns = {"sample_id", "source_name", "label_name", "csv_path"}
    missing_columns = sorted(required_columns - set(index_df.columns))
    if missing_columns:
        raise ValueError(f"content fingerprint requires columns: {missing_columns}")

    parent_records: dict[str, dict[str, str]] = {}
    parent_evidence: dict[str, set[str]] = {}
    evidence_parents: defaultdict[str, list[str]] = defaultdict(list)
    parent_ids: list[str] = []
    for row in index_df.to_dict(orient="records"):
        parent_sample_id = str(row["sample_id"])
        source_name = str(row["source_name"])
        label_name = str(row["label_name"])
        csv_path = _resolve_project_path(project_root, row.get("csv_path", ""))
        if not csv_path.exists():
            raise FileNotFoundError(f"track CSV for content fingerprint does not exist: {csv_path}")
        track_fingerprint = track_content_fingerprint(pd.read_csv(csv_path))

        raw_echo_fingerprint = ""
        raw_echo_status = "not_applicable" if use_rd_evidence else "not_applicable_tr_only"
        raw_echo_path = str(row.get("rd_path", ""))
        if use_rd_evidence and source_name == "cq08_track" and _is_present(raw_echo_path):
            resolved_raw_echo = _resolve_project_path(project_root, raw_echo_path)
            if not resolved_raw_echo.exists():
                raise FileNotFoundError(f"CQ08 raw echo for content fingerprint does not exist: {resolved_raw_echo}")
            raw_echo_fingerprint = file_content_fingerprint(resolved_raw_echo)
            raw_echo_status = "fingerprinted"
        elif use_rd_evidence and source_name == "cq08_track":
            raw_echo_status = "missing_raw_echo"

        rd_fingerprint = ""
        rd_status = "no_rd_tensor" if use_rd_evidence else "not_applicable_tr_only"
        rd_tensor_path = str(row.get("rd_tensor_path", ""))
        if use_rd_evidence and _is_present(rd_tensor_path):
            resolved_tensor = _resolve_project_path(project_root, rd_tensor_path)
            if not resolved_tensor.exists():
                raise FileNotFoundError(f"RD tensor for content fingerprint does not exist: {resolved_tensor}")
            rd_fingerprint = _rd_tensor_content_fingerprint(resolved_tensor)
            rd_status = "fingerprinted"

        height_missing = _metadata_bool(row.get("height_missing", False), field_name="height_missing")
        if use_rd_evidence:
            rd_phase_missing = _metadata_bool(row.get("rd_phase_missing", False), field_name="rd_phase_missing")
            track_evidence = (
                f"track:{source_name}:height_missing={str(height_missing).lower()}:"
                f"rd_phase_missing={str(rd_phase_missing).lower()}:{track_fingerprint}"
            )
        else:
            track_evidence = (
                f"track:{source_name}:height_missing={str(height_missing).lower()}:"
                f"{track_fingerprint}"
            )
        evidence = {track_evidence}
        if raw_echo_fingerprint:
            evidence.add(f"raw_echo:{source_name}:{raw_echo_fingerprint}")
        if rd_fingerprint:
            evidence.add(f"rd:{source_name}:{rd_fingerprint}")
        parent_evidence[parent_sample_id] = evidence
        parent_ids.append(parent_sample_id)
        for key in evidence:
            evidence_parents[key].append(parent_sample_id)
        parent_records[parent_sample_id] = {
            "parent_sample_id": parent_sample_id,
            "source_name": source_name,
            "label_name": label_name,
            "csv_path": str(row.get("csv_path", "")),
            "rd_path": raw_echo_path,
            "rd_tensor_path": rd_tensor_path,
            "track_content_fingerprint": track_fingerprint,
            "track_content_fingerprint_status": "fingerprinted",
            "raw_echo_content_fingerprint": raw_echo_fingerprint,
            "raw_echo_content_fingerprint_status": raw_echo_status,
            "rd_content_fingerprint": rd_fingerprint,
            "rd_content_fingerprint_status": rd_status,
        }

    parent_set = set(parent_ids)
    adjacent: defaultdict[str, set[str]] = defaultdict(set)
    for members in evidence_parents.values():
        unique_members = sorted(set(members))
        for parent_id in unique_members:
            adjacent[parent_id].update(member for member in unique_members if member != parent_id)

    visited: set[str] = set()
    for parent_id in sorted(parent_set):
        if parent_id in visited:
            continue
        stack = [parent_id]
        component: set[str] = set()
        component_evidence: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            component_evidence.update(parent_evidence[current])
            stack.extend(sorted(adjacent[current] - visited))
        labels = sorted({parent_records[item]["label_name"] for item in component})
        if len(labels) != 1:
            raise ValueError(
                "identical or transitively linked content has conflicting labels: "
                f"labels={labels}, parents={sorted(component)}"
            )
        canonical_parent_id = min(component)
        group_id = _content_component_id(component_evidence, canonical_parent_id)
        for item in component:
            parent_records[item]["content_group_id"] = group_id
            parent_records[item]["content_group_size"] = str(len(component))
            parent_records[item]["content_canonical_parent_id"] = canonical_parent_id
            parent_records[item]["is_content_canonical"] = str(item == canonical_parent_id).lower()
            parent_records[item]["split_group_id"] = group_id

    audit_rows = [parent_records[parent_id] for parent_id in sorted(parent_records)]
    return parent_records, audit_rows


# Compatibility name retained for callers of the old RD-only implementation.
def _build_rd_content_fingerprint_records(
    index_df: pd.DataFrame,
    project_root: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    return _build_content_fingerprint_records(
        index_df,
        project_root,
        split_evidence_contract="multimodal_v1",
    )

def _frozen_group_assignments(
    frozen_parent_assignments: dict[str, str],
    fingerprint_records_by_parent: dict[str, dict[str, str]],
) -> dict[str, str]:
    grouped_splits: defaultdict[str, set[str]] = defaultdict(set)
    for parent_sample_id, split_name in frozen_parent_assignments.items():
        record = fingerprint_records_by_parent.get(parent_sample_id)
        if record is None:
            raise ValueError(f"frozen split parent missing content-fingerprint record: {parent_sample_id}")
        grouped_splits[record["split_group_id"]].add(split_name)
    conflicts = {
        group_id: sorted(split_names)
        for group_id, split_names in grouped_splits.items()
        if len(split_names) > 1
    }
    if conflicts:
        examples = list(sorted(conflicts.items()))[:5]
        raise ValueError(
            "frozen parent split manifest separates identical RD content or track content across splits: "
            f"examples={examples}"
        )
    return {
        group_id: next(iter(split_names))
        for group_id, split_names in grouped_splits.items()
    }


def _majority(values: Iterable[object], fallback: str = "UnknownTarget") -> str:
    cleaned = [str(value) for value in values if str(value).strip() and str(value).lower() != "nan"]
    if not cleaned:
        return fallback
    counts = Counter(cleaned)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _window_starts(span_start: float, span_end: float, window_seconds: float, stride_seconds: float) -> list[float]:
    if span_end < span_start:
        return []
    span = span_end - span_start
    if span <= window_seconds:
        return [span_start]
    starts: list[float] = []
    current = span_start
    epsilon = 1e-6
    while current + window_seconds <= span_end + epsilon:
        starts.append(float(current))
        current += stride_seconds
    tail_start = span_end - window_seconds
    if starts and tail_start - starts[-1] > stride_seconds * 0.5:
        starts.append(float(tail_start))
    elif not starts:
        starts.append(float(tail_start))
    return sorted({round(value, 6) for value in starts})


def _assign_group_splits(
    window_df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    group_label_rows = (
        window_df.groupby("split_group_id", sort=True)
        .agg(
            label_name=("label_name", lambda values: _majority(values, fallback="UnknownTarget")),
            source_name=("source_name", lambda values: _majority(values, fallback="")),
            window_count=("sample_id", "count"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for _, label_group in group_label_rows.groupby(["source_name", "label_name"], sort=True):
        group_ids = label_group["split_group_id"].astype(str).to_numpy(dtype=object)
        rng.shuffle(group_ids)
        n = len(group_ids)
        if n == 1:
            train_n, val_n, test_n = 1, 0, 0
        elif n == 2:
            train_n, val_n, test_n = 1, 0, 1
        else:
            train_n = int(round(n * train_ratio))
            val_n = int(round(n * val_ratio))
            train_n = max(1, train_n)
            val_n = max(1, val_n)
            if train_n + val_n >= n:
                if train_n >= val_n and train_n > 1:
                    train_n -= 1
                else:
                    val_n = max(1, val_n - 1)
            test_n = n - train_n - val_n
            if test_n < 1:
                test_n = 1
                if train_n >= val_n and train_n > 1:
                    train_n -= 1
                elif val_n > 1:
                    val_n -= 1
        for group_id in group_ids[:train_n]:
            assignments[str(group_id)] = "train"
        for group_id in group_ids[train_n : train_n + val_n]:
            assignments[str(group_id)] = "val"
        for group_id in group_ids[train_n + val_n : train_n + val_n + test_n]:
            assignments[str(group_id)] = "test"
    return assignments


def _load_frozen_parent_assignments(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_ids"
        values = payload.get(key)
        if not isinstance(values, list):
            raise ValueError(f"frozen parent split manifest is missing list field: {key}")
        for value in values:
            parent_sample_id = str(value).strip()
            if not parent_sample_id:
                raise ValueError(f"frozen parent split manifest contains empty ID in {key}")
            previous = assignments.get(parent_sample_id)
            if previous is not None:
                raise ValueError(
                    "frozen parent split manifest assigns a parent to multiple splits: "
                    f"{parent_sample_id} ({previous}, {split_name})"
                )
            assignments[parent_sample_id] = split_name
    if not assignments:
        raise ValueError("frozen parent split manifest contains no parent IDs")
    return assignments


def _summary(window_df: pd.DataFrame, assignments: dict[str, str]) -> dict[str, object]:
    split_by_sample = window_df["split_group_id"].astype(str).map(assignments).fillna("unassigned")
    trainable_mask = (
        window_df["is_trainable"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        if "is_trainable" in window_df.columns
        else pd.Series(True, index=window_df.index)
    )
    payload: dict[str, object] = {
        "total_windows": int(len(window_df)),
        "trainable_windows": int(trainable_mask.sum()),
        "diagnostic_windows": int((~trainable_mask).sum()),
        "parent_tracks": int(window_df["parent_sample_id"].nunique()),
        "source_counts": window_df["source_name"].value_counts(dropna=False).to_dict(),
        "label_counts": window_df["label_name"].value_counts(dropna=False).to_dict(),
        "window_mode_counts": window_df["window_mode"].value_counts(dropna=False).to_dict(),
        "rd_coverage": {
            "with_rd": int(window_df["rd_tensor_path"].map(_is_present).sum()),
            "without_rd": int((~window_df["rd_tensor_path"].map(_is_present)).sum()),
        },
        "split_counts": split_by_sample.value_counts(dropna=False).to_dict(),
        "trainable_split_counts": split_by_sample[trainable_mask].value_counts(dropna=False).to_dict(),
        "split_source_counts": {},
        "split_label_counts": {},
        "exclude_reason_counts": window_df.loc[~trainable_mask, "exclude_reason"].value_counts(dropna=False).to_dict()
        if "exclude_reason" in window_df.columns
        else {},
        "window_duration_seconds": {
            "min": float(window_df["window_duration_seconds"].min()) if len(window_df) else 0.0,
            "p50": float(window_df["window_duration_seconds"].quantile(0.50)) if len(window_df) else 0.0,
            "p90": float(window_df["window_duration_seconds"].quantile(0.90)) if len(window_df) else 0.0,
            "max": float(window_df["window_duration_seconds"].max()) if len(window_df) else 0.0,
        },
        "window_point_count": {
            "min": int(window_df["window_point_count"].min()) if len(window_df) else 0,
            "p50": float(window_df["window_point_count"].quantile(0.50)) if len(window_df) else 0.0,
            "p90": float(window_df["window_point_count"].quantile(0.90)) if len(window_df) else 0.0,
            "max": int(window_df["window_point_count"].max()) if len(window_df) else 0,
        },
    }
    for split_name in ["train", "val", "test", "unassigned"]:
        split_df = window_df[split_by_sample == split_name]
        if split_df.empty:
            continue
        payload["split_source_counts"][split_name] = split_df["source_name"].value_counts(dropna=False).to_dict()
        payload["split_label_counts"][split_name] = split_df["label_name"].value_counts(dropna=False).to_dict()
    parent_splits: defaultdict[str, set[str]] = defaultdict(set)
    for parent_sample_id, split_name in zip(window_df["parent_sample_id"].astype(str), split_by_sample.astype(str)):
        parent_splits[parent_sample_id].add(split_name)
    payload["parent_leakage_count"] = int(sum(1 for values in parent_splits.values() if len(values) > 1))
    if "rd_content_fingerprint" in window_df.columns:
        fingerprint_df = window_df[window_df["rd_content_fingerprint"].astype(str).str.len() > 0].copy()
        if fingerprint_df.empty:
            payload["rd_content_fingerprint"] = {
                "algorithm": RD_CONTENT_FINGERPRINT_ALGORITHM,
                "fingerprinted_parent_count": 0,
                "unique_fingerprint_count": 0,
                "duplicate_fingerprint_count": 0,
                "duplicate_parent_count": 0,
                "cross_split_leakage_count": 0,
            }
        else:
            fingerprint_parents = fingerprint_df[["rd_content_fingerprint", "parent_sample_id"]].drop_duplicates()
            parent_counts = fingerprint_parents.groupby("rd_content_fingerprint", sort=True)["parent_sample_id"].nunique()
            fingerprint_splits = (
                pd.DataFrame(
                    {
                        "rd_content_fingerprint": fingerprint_df["rd_content_fingerprint"].astype(str),
                        "split_name": split_by_sample.loc[fingerprint_df.index].astype(str),
                    }
                )
                .drop_duplicates()
                .groupby("rd_content_fingerprint", sort=True)["split_name"]
                .nunique()
            )
            payload["rd_content_fingerprint"] = {
                "algorithm": RD_CONTENT_FINGERPRINT_ALGORITHM,
                "fingerprinted_parent_count": int(fingerprint_parents["parent_sample_id"].nunique()),
                "unique_fingerprint_count": int(parent_counts.shape[0]),
                "duplicate_fingerprint_count": int((parent_counts > 1).sum()),
                "duplicate_parent_count": int(parent_counts[parent_counts > 1].sum()),
                "cross_split_leakage_count": int((fingerprint_splits > 1).sum()),
            }
    return payload


def build_expert1_window_index(
    project_root: Path,
    window_seconds: float = 10.0,
    stride_seconds: float = 2.0,
    min_points: int = 5,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    window_index_name: str = "expert1_window_index.csv",
    split_manifest_name: str = "expert1_window_split_manifest.json",
    summary_name: str = "expert1_window_summary.json",
    window_output_dir_name: str = "expert1_windows",
    frozen_parent_split_manifest: Path | str | None = None,
    include_prefix_windows: bool = False,
    prefix_stride_points: int = 5,
    min_unique_time_points: int = 2,
    rd_fingerprint_audit_name: str | None = None,
    split_evidence_contract: str = "multimodal_v1",
    source_index_name: str = "expert1_track_index.csv",
) -> dict[str, object]:
    normalized_split_contract = str(split_evidence_contract).strip().lower()
    if normalized_split_contract not in SPLIT_EVIDENCE_CONTRACTS:
        raise ValueError(
            "unsupported split evidence contract: "
            f"{split_evidence_contract!r}; expected one of {sorted(SPLIT_EVIDENCE_CONTRACTS)}"
        )
    is_tr_only_split = normalized_split_contract == "tr_only_v1"
    processed_root = project_root / "data" / "processed"
    output_root = processed_root / window_output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = processed_root / source_index_name
    index_df = pd.read_csv(index_path)
    index_df = index_df[index_df["label_name"].astype(str).isin(EXPERT1_LABELS)].copy()
    fingerprint_records_by_parent, fingerprint_audit_rows = _build_content_fingerprint_records(
        index_df,
        project_root,
        split_evidence_contract=normalized_split_contract,
    )
    frozen_manifest_path = Path(frozen_parent_split_manifest) if frozen_parent_split_manifest else None
    frozen_parent_assignments = (
        _load_frozen_parent_assignments(frozen_manifest_path)
        if frozen_manifest_path is not None
        else None
    )
    if frozen_parent_assignments is not None:
        index_parent_ids = set(index_df["sample_id"].astype(str))
        missing_from_index = sorted(set(frozen_parent_assignments) - index_parent_ids)
        missing_from_manifest = sorted(index_parent_ids - set(frozen_parent_assignments))
        if missing_from_index or missing_from_manifest:
            raise ValueError(
                "frozen parent split manifest must cover exactly the Expert1 parent IDs; "
                f"missing_from_index={missing_from_index[:5]}, "
                f"missing_from_manifest={missing_from_manifest[:5]}"
            )
    rows: list[dict[str, object]] = []
    manifest_windows: list[dict[str, object]] = []

    for row in index_df.to_dict(orient="records"):
        parent_sample_id = str(row["sample_id"])
        source_name = str(row["source_name"])
        fingerprint_record = fingerprint_records_by_parent.get(parent_sample_id)
        if fingerprint_record is None:
            raise ValueError(f"parent is missing RD content fingerprint record: {parent_sample_id}")
        csv_path = project_root / str(row["csv_path"])
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        if frame.empty or "time_seconds" not in frame.columns:
            continue
        frame = frame.sort_values("time_seconds").reset_index(drop=True)
        time_values = pd.to_numeric(frame["time_seconds"], errors="coerce")
        frame = frame[time_values.notna()].copy()
        if frame.empty:
            continue
        frame["time_seconds"] = pd.to_numeric(frame["time_seconds"], errors="coerce").astype(float)
        start_time = float(frame["time_seconds"].min())
        end_time = float(frame["time_seconds"].max())
        parent_duration = float(end_time - start_time)
        starts = _window_starts(start_time, end_time, window_seconds, stride_seconds)
        parent_safe = _safe_name(parent_sample_id)

        def emit_window(
            window_index: int,
            window_df: pd.DataFrame,
            window_mode: str,
            effective_start: float,
            effective_end: float,
            force_exclude_reasons: list[str] | None = None,
        ) -> None:
            label_name = _majority(window_df.get("label_name", pd.Series(dtype=str)), fallback=str(row["label_name"]))
            raw_label_name = _majority(
                window_df.get("label_name_raw", pd.Series(dtype=str)),
                fallback=str(row.get("label_name_raw", label_name)),
            )
            label_fields = two_layer_label_fields(
                label_name,
                source_name=source_name,
                batch_id=row.get("batch_id", ""),
                label_name_raw=raw_label_name,
            )
            window_id = f"{parent_sample_id}|win{window_index:04d}"
            window_df = window_df.copy()
            window_df["record_id"] = [f"{window_id}#{idx:04d}" for idx in range(len(window_df))]
            window_df["pair_id"] = window_id
            window_df["label_name"] = label_name
            window_df["label_name_raw"] = raw_label_name
            window_df["coarse_label"] = label_fields["coarse_label"]
            window_df["fine_label"] = label_fields["fine_label"]
            for column in TRACK_SCHEMA_COLUMNS:
                if column not in window_df.columns:
                    window_df[column] = np.nan if column not in {"note_text"} else ""
            window_df = window_df[TRACK_SCHEMA_COLUMNS]
            source_dir = output_root / _safe_name(source_name)
            source_dir.mkdir(parents=True, exist_ok=True)
            window_csv_path = source_dir / f"{parent_safe}__win_{window_index:04d}.csv"
            serialized_window = window_df.to_csv(index=False)
            window_csv_path.write_text(serialized_window, encoding="utf-8")
            # Fingerprint the representation consumers actually reload.  The
            # default CSV parser can round high-precision floats differently
            # from the in-memory frame used before serialization.
            window_df = pd.read_csv(StringIO(serialized_window))
            split_group_id = fingerprint_record["split_group_id"]
            window_content_fingerprint = _window_content_fingerprint(
                window_df,
                source_name=source_name,
                height_missing=_metadata_bool(row.get("height_missing", False), field_name="height_missing"),
                rd_phase_missing=_metadata_bool(row.get("rd_phase_missing", False), field_name="rd_phase_missing"),
                include_rd_phase_missing=not is_tr_only_split,
            )
            effective_duration = float(effective_end - effective_start)
            exclude_reasons: list[str] = []
            if len(window_df) < min_points:
                exclude_reasons.append("too_few_points_for_window_training")
            unique_time_count = int(window_df["time_seconds"].nunique())
            if unique_time_count < max(2, int(min_unique_time_points)):
                exclude_reasons.append("too_few_unique_timestamps_for_causal_window")
            if effective_duration > window_seconds + 1e-6:
                exclude_reasons.append("window_duration_exceeds_target")
            if label_name not in EXPERT1_LABELS:
                exclude_reasons.append(f"unsupported_label:{label_name}")
            if str(fingerprint_record["is_content_canonical"]).lower() != "true":
                exclude_reasons.append("duplicate_content_noncanonical")
            if force_exclude_reasons:
                exclude_reasons.extend(force_exclude_reasons)
            exclude_reasons = sorted(set(exclude_reasons))
            is_trainable = len(exclude_reasons) == 0
            row_payload = {
                "dataset_name": str(row.get("dataset_name", "")),
                "source_name": source_name,
                "sample_id": window_id,
                "sample_kind": "expert1_window",
                "timestamp": str(row.get("timestamp", "")),
                "batch_id": str(row.get("batch_id", "")),
                "label_name_raw": raw_label_name,
                "label_name": label_name,
                "coarse_label": label_fields["coarse_label"],
                "fine_label": label_fields["fine_label"],
                "record_count": int(len(window_df)),
                "time_seconds_min": float(window_df["time_seconds"].min()),
                "time_seconds_max": float(window_df["time_seconds"].max()),
                "csv_path": window_csv_path.relative_to(project_root).as_posix(),
                "rd_path": str(row.get("rd_path", "")),
                "rd_summary_csv_path": str(row.get("rd_summary_csv_path", "")),
                "rd_tensor_path": str(row.get("rd_tensor_path", "")),
                "rd_phase_missing": bool(row.get("rd_phase_missing", False)),
                "height_missing": bool(row.get("height_missing", False)),
                "is_trainable": is_trainable,
                "exclude_reason": ";".join(exclude_reasons),
                "parent_sample_id": parent_sample_id,
                "window_id": f"win{window_index:04d}",
                "window_start_seconds": float(effective_start),
                "window_end_seconds": float(effective_end),
                "window_duration_seconds": effective_duration,
                "window_point_count": int(len(window_df)),
                "window_unique_time_count": unique_time_count,
                "is_causal_window": True,
                "window_mode": window_mode,
                "track_content_fingerprint": fingerprint_record["track_content_fingerprint"],
                "track_content_fingerprint_status": fingerprint_record["track_content_fingerprint_status"],
                "raw_echo_content_fingerprint": fingerprint_record["raw_echo_content_fingerprint"],
                "raw_echo_content_fingerprint_status": fingerprint_record["raw_echo_content_fingerprint_status"],
                "rd_content_fingerprint": fingerprint_record["rd_content_fingerprint"],
                "rd_content_fingerprint_status": fingerprint_record["rd_content_fingerprint_status"],
                "content_group_id": fingerprint_record["content_group_id"],
                "content_group_size": int(fingerprint_record["content_group_size"]),
                "content_canonical_parent_id": fingerprint_record["content_canonical_parent_id"],
                "is_content_canonical": str(fingerprint_record["is_content_canonical"]).lower() == "true",
                "window_content_fingerprint": window_content_fingerprint,
                "split_group_id": split_group_id,
            }
            rows.append(row_payload)
            manifest_windows.append(
                {
                    "sample_id": window_id,
                    "parent_sample_id": parent_sample_id,
                    "split_group_id": split_group_id,
                    "source_name": source_name,
                    "label_name": label_name,
                    "window_start_seconds": float(effective_start),
                    "window_end_seconds": float(effective_end),
                    "window_duration_seconds": effective_duration,
                    "window_point_count": int(len(window_df)),
                    "window_unique_time_count": unique_time_count,
                    "is_causal_window": True,
                    "window_mode": window_mode,
                    "track_content_fingerprint": fingerprint_record["track_content_fingerprint"],
                    "raw_echo_content_fingerprint": fingerprint_record["raw_echo_content_fingerprint"],
                    "rd_content_fingerprint": fingerprint_record["rd_content_fingerprint"],
                    "content_group_id": fingerprint_record["content_group_id"],
                    "content_group_size": int(fingerprint_record["content_group_size"]),
                    "content_canonical_parent_id": fingerprint_record["content_canonical_parent_id"],
                    "is_content_canonical": str(fingerprint_record["is_content_canonical"]).lower() == "true",
                    "window_content_fingerprint": window_content_fingerprint,
                    "is_trainable": is_trainable,
                    "exclude_reason": ";".join(exclude_reasons),
                }
            )

        candidates: list[tuple[float, float, str, bool]] = []
        if include_prefix_windows:
            prefix_end = start_time + stride_seconds
            while prefix_end < min(end_time, start_time + window_seconds) - 1e-6:
                candidates.append((start_time, float(prefix_end), "prefix", True))
                prefix_end += stride_seconds
            point_stride = max(1, int(prefix_stride_points))
            warmup_end = min(end_time, start_time + window_seconds)
            for end_index in range(min_points - 1, len(frame) - 1, point_stride):
                prefix_end = float(frame.iloc[end_index]["time_seconds"])
                if prefix_end >= warmup_end - 1e-6:
                    break
                candidates.append((start_time, prefix_end, "prefix", True))
        if parent_duration <= window_seconds:
            mode = "short_full" if parent_duration < window_seconds else "exact_full"
            candidates.append((start_time, end_time, mode, True))
        else:
            for window_index, window_start in enumerate(starts):
                window_end = min(float(window_start + window_seconds), end_time)
                candidates.append(
                    (
                        float(window_start),
                        float(window_end),
                        "sliding",
                        True,
                    )
                )

        deduplicated_candidates: dict[tuple[float, float, str, bool], tuple[float, float, str, bool]] = {}
        for start, end, mode, include_end in candidates:
            if end >= start:
                key = (float(start), float(end), mode, bool(include_end))
                deduplicated_candidates.setdefault(key, key)
        candidates = sorted(
            deduplicated_candidates.values(),
            key=lambda item: (item[1], item[0], item[2], item[3]),
        )
        emitted_window = False
        emitted_count = 0
        for effective_start, effective_end, window_mode, include_end in candidates:
            mask = frame["time_seconds"] >= effective_start
            if include_end:
                mask &= frame["time_seconds"] <= effective_end
            else:
                mask &= frame["time_seconds"] < effective_end
            window_df = frame[mask].copy()
            if len(window_df) < min_points:
                if parent_duration <= window_seconds and not emitted_window and window_mode != "prefix":
                    emit_window(0, frame, "short_full_below_min_points", start_time, end_time)
                    emitted_window = True
                continue
            emit_window(
                emitted_count,
                window_df,
                window_mode,
                effective_start,
                effective_end,
            )
            emitted_window = True
            emitted_count += 1
        if not emitted_window:
            emit_window(
                0,
                frame,
                "diagnostic_sparse_full",
                start_time,
                end_time,
                force_exclude_reasons=["no_window_meets_min_points"],
            )

    window_df = pd.DataFrame(rows)
    if window_df.empty:
        raise RuntimeError("no expert1 windows were generated")
    window_df = window_df[WINDOW_INDEX_COLUMNS]
    window_df, merged_window_groups = _merge_window_split_groups(window_df)
    if is_tr_only_split:
        window_df, _session_merged_groups = _merge_session_split_groups(window_df)
    if frozen_parent_assignments is None:
        assignments = _assign_group_splits(window_df, seed, train_ratio, val_ratio, test_ratio)
    elif is_tr_only_split:
        assignments = {}
        for final_group_id, group_df in window_df.groupby("split_group_id", sort=True):
            parent_ids = sorted(set(group_df["parent_sample_id"].astype(str)))
            split_names = {frozen_parent_assignments[parent_id] for parent_id in parent_ids}
            if len(split_names) != 1:
                raise ValueError(
                    "frozen parent split manifest is not session-atomic for TR-only: "
                    f"group={final_group_id}, parents={parent_ids[:5]}, splits={sorted(split_names)}"
                )
            assignments[str(final_group_id)] = next(iter(split_names))
    else:
        parent_assignments = _frozen_group_assignments(
            frozen_parent_assignments,
            fingerprint_records_by_parent,
        )
        assignments = {}
        for merged_group_id, original_groups in merged_window_groups.items():
            split_names = {parent_assignments[group_id] for group_id in original_groups}
            if len(split_names) != 1:
                raise ValueError(
                    "frozen parent split manifest separates identical generated windows across splits: "
                    f"group={merged_group_id}, splits={sorted(split_names)}"
                )
            assignments[merged_group_id] = next(iter(split_names))
    window_df["split_name"] = window_df["split_group_id"].astype(str).map(assignments).fillna("unassigned")
    final_columns = WINDOW_INDEX_COLUMNS + ["split_name"]
    window_index_path = processed_root / window_index_name
    window_df[final_columns].to_csv(window_index_path, index=False, encoding="utf-8")

    audit_name = rd_fingerprint_audit_name or (
        f"{Path(window_index_name).stem}_content_fingerprint_audit.csv"
        if is_tr_only_split
        else f"{Path(window_index_name).stem}_rd_content_fingerprint_audit.csv"
    )
    audit_path = processed_root / audit_name
    audit_df = pd.DataFrame(fingerprint_audit_rows)
    parent_split_counts = window_df.groupby("parent_sample_id", sort=True)["split_name"].nunique()
    crossing_parents = parent_split_counts[parent_split_counts > 1]
    if not crossing_parents.empty:
        raise ValueError(
            "generated windows from one parent cross persisted splits: "
            f"parents={crossing_parents.index.astype(str).tolist()[:5]}"
        )
    parent_split_names = (
        window_df.groupby("parent_sample_id", sort=True)["split_name"].first().astype(str).to_dict()
    )
    audit_df["split_name"] = (
        audit_df["parent_sample_id"].astype(str).map(parent_split_names).fillna("not_generated")
    )
    audit_df[RD_CONTENT_FINGERPRINT_AUDIT_COLUMNS].to_csv(audit_path, index=False, encoding="utf-8")

    split_manifest = {
        "window_seconds": float(window_seconds),
        "stride_seconds": float(stride_seconds),
        "min_points": int(min_points),
        "split_evidence_contract": normalized_split_contract,
        "split_group_key": (
            "session_component_v1(source_name|timestamp|track_model_inputs)"
            if is_tr_only_split
            else "content_component_v1(track_model_inputs|cq08_raw_echo|processed_rd)"
        ),
        "content_component_algorithm": CONTENT_COMPONENT_ALGORITHM,
        "window_content_fingerprint_algorithm": WINDOW_CONTENT_FINGERPRINT_ALGORITHM,
        "window_content_group_algorithm": WINDOW_CONTENT_GROUP_ALGORITHM,
        "track_content_fingerprint_algorithm": TRACK_CONTENT_FINGERPRINT_ALGORITHM,
        "session_group_key": "source_name|timestamp" if is_tr_only_split else None,
        "session_split_group_algorithm": SESSION_SPLIT_GROUP_ALGORITHM if is_tr_only_split else None,
        "raw_echo_content_fingerprint_algorithm": (
            "not_applicable_tr_only" if is_tr_only_split else FILE_CONTENT_FINGERPRINT_ALGORITHM
        ),
        "rd_content_fingerprint_algorithm": (
            "not_applicable_tr_only" if is_tr_only_split else RD_CONTENT_FINGERPRINT_ALGORITHM
        ),
        "content_fingerprint_audit": audit_path.relative_to(project_root).as_posix(),
        "rd_content_fingerprint_audit": (
            None if is_tr_only_split else audit_path.relative_to(project_root).as_posix()
        ),
        "train_ids": sorted(window_df[(window_df["split_name"] == "train") & (window_df["is_trainable"])]["sample_id"].astype(str).tolist()),
        "val_ids": sorted(window_df[(window_df["split_name"] == "val") & (window_df["is_trainable"])]["sample_id"].astype(str).tolist()),
        "test_ids": sorted(window_df[(window_df["split_name"] == "test") & (window_df["is_trainable"])]["sample_id"].astype(str).tolist()),
        "diagnostic_ids": sorted(window_df[~window_df["is_trainable"]]["sample_id"].astype(str).tolist()),
        "group_assignments": assignments,
        "frozen_parent_split_manifest": (
            frozen_manifest_path.relative_to(project_root).as_posix()
            if frozen_manifest_path is not None and frozen_manifest_path.is_relative_to(project_root)
            else (str(frozen_manifest_path) if frozen_manifest_path is not None else None)
        ),
        "include_prefix_windows": bool(include_prefix_windows),
        "prefix_stride_points": int(prefix_stride_points),
        "min_unique_time_points": int(max(2, min_unique_time_points)),
        "source_parent_index": index_path.relative_to(project_root).as_posix(),
    }
    manifest_path = project_root / "data" / "manifests" / split_manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _summary(window_df, assignments)
    if is_tr_only_split:
        summary["rd_coverage"] = {"status": "not_applicable_tr_only"}
        summary["rd_content_fingerprint"] = {"status": "not_applicable_tr_only"}
    summary.update(
        {
            "window_seconds": float(window_seconds),
            "stride_seconds": float(stride_seconds),
            "min_points": int(min_points),
            "include_prefix_windows": bool(include_prefix_windows),
            "prefix_stride_points": int(prefix_stride_points),
            "min_unique_time_points": int(max(2, min_unique_time_points)),
            "window_end_inclusive": True,
            "window_index": window_index_path.relative_to(project_root).as_posix(),
            "split_manifest": manifest_path.relative_to(project_root).as_posix(),
            "split_evidence_contract": normalized_split_contract,
            "source_parent_index": index_path.relative_to(project_root).as_posix(),
            "session_group_key": "source_name|timestamp" if is_tr_only_split else None,
            "content_fingerprint_audit": audit_path.relative_to(project_root).as_posix(),
            "rd_content_fingerprint_audit": (
                None if is_tr_only_split else audit_path.relative_to(project_root).as_posix()
            ),
        }
    )
    summary_path = processed_root / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deployment-style sliding window index for Expert1.")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-index-name", default="expert1_window_index.csv")
    parser.add_argument("--split-manifest-name", default="expert1_window_split_manifest.json")
    parser.add_argument("--summary-name", default="expert1_window_summary.json")
    parser.add_argument("--window-output-dir-name", default="expert1_windows")
    parser.add_argument("--frozen-parent-split-manifest")
    parser.add_argument("--include-prefix-windows", action="store_true")
    parser.add_argument("--prefix-stride-points", type=int, default=5)
    parser.add_argument("--min-unique-time-points", type=int, default=2)
    parser.add_argument("--rd-fingerprint-audit-name")
    parser.add_argument("--source-index-name", default="expert1_track_index.csv")
    parser.add_argument(
        "--split-evidence-contract",
        choices=sorted(SPLIT_EVIDENCE_CONTRACTS),
        default="multimodal_v1",
        help=(
            "Use tr_only_v1 to exclude every RD-derived split signal and make "
            "source_name+timestamp sessions atomic."
        ),
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    summary = build_expert1_window_index(
        project_root,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_points=args.min_points,
        seed=args.seed,
        window_index_name=args.window_index_name,
        split_manifest_name=args.split_manifest_name,
        summary_name=args.summary_name,
        window_output_dir_name=args.window_output_dir_name,
        frozen_parent_split_manifest=args.frozen_parent_split_manifest,
        include_prefix_windows=args.include_prefix_windows,
        prefix_stride_points=args.prefix_stride_points,
        min_unique_time_points=args.min_unique_time_points,
        rd_fingerprint_audit_name=args.rd_fingerprint_audit_name,
        split_evidence_contract=args.split_evidence_contract,
        source_index_name=args.source_index_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
