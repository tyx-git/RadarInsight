from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from build.radar_ln3_normalization import build as build_normalization


VERSION = "RADAR-LN3-V16"
SOURCE_INDEX_RELATIVE = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
SOURCE_MANIFEST_RELATIVE = Path("data/manifests/cq08_unified_manifest.json")
PROVENANCE_RELATIVE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
SPLIT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-split.json"
)
INDEX_RELATIVE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
STATS_RELATIVE = Path(
    "data/processed/RADAR-LN3-V16-cq08-tr-rd-train-normalization-stats.json"
)
CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V16-cq08-product-local-data-contract.json"
)
CONFIG_DIR_RELATIVE = Path("config/RADAR-LN3")

BASE_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
TASKS: dict[str, dict[str, Any]] = {
    "dbb_cu_gate": {
        "slug": "dbb-cu-gate",
        "class_names": ["DBB", "CU"],
        "label_to_id": {
            "DroneTarget": 0,
            "BirdTarget": 0,
            "BalloonTarget": 0,
            "ClutterTarget": 1,
            "UnknownTarget": 1,
        },
        "allowed_labels": list(BASE_LABELS),
        "target": 0.95,
    },
    "dbb_specialist": {
        "slug": "dbb-specialist",
        "class_names": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        "label_to_id": {"DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2},
        "allowed_labels": ["DroneTarget", "BirdTarget", "BalloonTarget"],
        "target": 0.95,
    },
    "cu_specialist": {
        "slug": "cu-specialist",
        "class_names": ["ClutterTarget", "UnknownTarget"],
        "label_to_id": {"ClutterTarget": 0, "UnknownTarget": 1},
        "allowed_labels": ["ClutterTarget", "UnknownTarget"],
        "target": 0.90,
    },
}

CONTENT_KEYS = (
    "raw_sha256",
    "rd_sha256",
    "normalized_track_sha256",
    "normalized_point_sha256",
    "model_input_semantics_sha256",
)
GROUP_KEYS = ("capture_id", *CONTENT_KEYS)
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(project_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def _resolve(project_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _canonical_text_table_sha256(path: Path) -> str:
    """Hash numeric rows while removing the per-file track-id wrapper column."""

    lines = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
    normalized: list[bytes] = []
    for line_number, line in enumerate(lines):
        value = line.strip()
        if not value or line_number == 0:
            continue
        fields = [field.strip() for field in value.split(b",")]
        if len(fields) < 3:
            raise ValueError(f"CQ08 table row has fewer than three columns: {path}: line={line_number + 1}")
        fields[1] = b"<TRACK_ID>"
        normalized.append(b",".join(fields))
    if not normalized:
        raise ValueError(f"CQ08 table has no data rows: {path}")
    return _sha256_bytes(b"\n".join(normalized) + b"\n")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V16 model-input provenance received a non-finite float: {value!r}")
    if result == 0.0:
        return 0.0
    return result


def _assert_same_path(left: Path, right_value: Any, *, field_name: str, track_id: int) -> None:
    right = Path(str(right_value).replace("\\", "/")).resolve()
    if left.resolve() != right:
        raise ValueError(
            f"V16 manifest/index {field_name} mismatch for track_id={track_id}: "
            f"index={left.resolve()}, manifest={right}"
        )


def _model_input_semantics_sha256(parent: pd.DataFrame, *, rd_sha256: str | None) -> str:
    rows: list[dict[str, Any]] = []
    ordered = parent.sort_values("causal_event_index", kind="stable")
    for _, row in ordered.iterrows():
        rows.append(
            {
                "causal_event_index": int(row["causal_event_index"]),
                "tr_time_seconds": _finite_float(row["tr_time_seconds"]),
                "tr_dt_seconds": _finite_float(row["tr_dt_seconds"]),
                "tr_dt_valid": _as_bool(row["tr_dt_valid"]),
                "tr_dt_reason": str(row["tr_dt_reason"]),
                "tr_feature_values": json.loads(str(row["tr_feature_values_json"])),
                "tr_missing_count": int(row["tr_missing_count"]),
                "rd_present": _as_bool(row["rd_present"]),
                "rd_frame_count": int(row["rd_frame_count"]),
                "rd_tensor_frame_indices": json.loads(str(row["rd_tensor_frame_indices_json"])),
                "rd_content_sha256": rd_sha256,
                "rd_local_duration_seconds": _finite_float(row["rd_local_duration_seconds"]),
                "rd_local_time_order_status": str(row.get("rd_local_time_order_status", "")),
                "rd_quality_missing": _as_bool(row["rd_quality_missing"]),
            }
        )
    return _sha256_bytes(_canonical_json(rows).encode("utf-8"))


def _write_json_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V16 JSON artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _write_csv_new_or_identical(path: Path, frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V16 CSV artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return "created"


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _component_id(parent_ids: list[str]) -> str:
    payload = "\n".join(sorted(parent_ids)).encode("utf-8")
    return f"v16-component-{hashlib.sha256(payload).hexdigest()}"


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    parents = frame[["parent_sample_id", "label_name"]].drop_duplicates("parent_sample_id")
    values = parents.groupby("label_name").size().to_dict()
    return {label: int(values.get(label, 0)) for label in BASE_LABELS}


def _point_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.groupby("label_name").size().to_dict()
    return {label: int(values.get(label, 0)) for label in BASE_LABELS}


def _inverse_frequency_weights(counts: list[int]) -> list[float]:
    if not counts or any(value <= 0 for value in counts):
        raise ValueError(f"cannot build class weights from non-positive counts: {counts}")
    total = float(sum(counts))
    classes = float(len(counts))
    return [total / (classes * float(value)) for value in counts]


def _config_payload(
    *,
    task_name: str,
    index_sha256: str,
    contract_sha256: str,
    stats_sha256: str,
    amplitude_center: float,
    amplitude_scale: float,
    train_parent_counts: dict[str, int],
) -> dict[str, Any]:
    task = TASKS[task_name]
    encoded_counts = [
        sum(train_parent_counts[label] for label, class_id in task["label_to_id"].items() if class_id == index)
        for index in range(len(task["class_names"]))
    ]
    slug = str(task["slug"])
    artifact = f"artifacts/RADAR-LN3-V16-cq08-liquid-product-local-{slug}-formal"
    return {
        "schema_version": VERSION,
        "config_name": f"RADAR-LN3-V16-cq08-liquid-product-local-{slug}",
        "task": {
            "name": task_name,
            "class_names": task["class_names"],
            "label_to_id": task["label_to_id"],
        },
        "data": {
            "index_path": str(INDEX_RELATIVE).replace("\\", "/"),
            "contract_path": str(CONTRACT_RELATIVE).replace("\\", "/"),
            "normalization_stats_path": str(STATS_RELATIVE).replace("\\", "/"),
            "train_split_name": "train",
            "validation_split_name": "val",
            "test_split_name": "test",
            "index_schema_version": VERSION,
            "contract_schema_version": VERSION,
            "rd_quality_schema": "causal-current-event-v1",
            "expected_index_sha256": index_sha256,
            "expected_contract_sha256": contract_sha256,
            "expected_stats_sha256": stats_sha256,
            "max_sequence_length": 29,
            "max_rd_frames_per_point": 12,
            "rd_amplitude_center": amplitude_center,
            "rd_amplitude_scale": amplitude_scale,
            "exclude_anomalous_parents": False,
        },
        "model": {
            "tr_value_dim": 14,
            "tr_missing_dim": 11,
            "rd_quality_dim": 7,
            "hidden_dim": 48,
            "rd_embedding_dim": 24,
            "liquid_hidden_dim": 48,
            "rd_height": 21,
            "rd_width": 384,
            "liquid_tau_min_seconds": 0.05,
            "liquid_tau_max_seconds": 30.0,
            "liquid_zero_dt_event_floor_seconds": 0.001,
            "liquid_dt_max_seconds": 120.0,
            "dt_valid_dim": 1,
            "dt_valid_feature_mode": "dynamics_only",
            "dt_reason_mode": "v3_reason_aware_dynamics_only",
            "liquid_invalid_dt_update_mode": "bounded_observation",
            "liquid_invalid_dt_observation_gain": 0.25,
            "num_classes": len(task["class_names"]),
            "liquid_state_normalization": "recurrent",
            "rd_frame_aggregation": "masked_mean",
        },
        "train": {
            "batch_size": 2,
            "learning_rate": 0.0001,
            "seed": 42,
            "device": "cuda",
            "epochs": 20,
            "early_stopping_patience": 0,
            "gradient_clip_norm": 100.0,
            "checkpoint_selection_metric": "parent_min_precision_recall",
            "task_min_precision_recall_target": float(task["target"]),
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "class_weights": _inverse_frequency_weights(encoded_counts),
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": 0.1,
            "prefix_auxiliary_min_events": 3,
            "checkpoint_schema": f"RADAR-LN3-V16-product-local-{slug}-checkpoint-v1",
            "allow_parameter_updates": True,
            "dataloader_seed": 42,
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": False,
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    source_index_path = project_root / SOURCE_INDEX_RELATIVE
    source_manifest_path = project_root / SOURCE_MANIFEST_RELATIVE
    provenance_path = project_root / PROVENANCE_RELATIVE
    split_path = project_root / SPLIT_RELATIVE
    index_path = project_root / INDEX_RELATIVE
    stats_path = project_root / STATS_RELATIVE
    contract_path = project_root / CONTRACT_RELATIVE

    source = pd.read_csv(source_index_path, low_memory=False)
    if source["schema_version"].astype(str).ne("RADAR-LN3-V3").any():
        raise ValueError("V16 source index must contain only RADAR-LN3-V3 rows")
    if source["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("V16 source index contains a non-CQ08 row")
    if source["sample_id"].astype(str).duplicated().any():
        raise ValueError("V16 source index contains duplicate sample IDs")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("CQ08 unified manifest must be a JSON list")
    manifest_by_track: dict[int, dict[str, Any]] = {}
    for value in manifest:
        track_id = int(value["track_id"])
        if track_id in manifest_by_track:
            raise ValueError(f"CQ08 unified manifest repeats track_id={track_id}")
        manifest_by_track[track_id] = dict(value)

    parent_groups = list(source.groupby("parent_sample_id", sort=True))
    print(
        f"[V16 build] parents={len(parent_groups)} rows={len(source)} "
        f"raw_and_rd_hashing=started",
        flush=True,
    )
    file_hash_cache: dict[str, str] = {}
    canonical_hash_cache: dict[str, str] = {}

    def file_hash(path: Path) -> str:
        key = str(path.resolve())
        if key not in file_hash_cache:
            file_hash_cache[key] = _sha256_file(path)
        return file_hash_cache[key]

    def canonical_hash(path: Path) -> str:
        key = str(path.resolve())
        if key not in canonical_hash_cache:
            canonical_hash_cache[key] = _canonical_text_table_sha256(path)
        return canonical_hash_cache[key]

    records: list[dict[str, Any]] = []
    for number, (parent_id, parent) in enumerate(parent_groups, start=1):
        if parent["label_name"].astype(str).nunique() != 1:
            raise ValueError(f"V16 source parent has multiple labels: {parent_id}")
        if parent["split_name"].astype(str).nunique() != 1:
            raise ValueError(f"V16 source parent crosses V3 splits: {parent_id}")
        first = parent.sort_values("causal_event_index", kind="stable").iloc[0]
        track_id = int(first["track_id"])
        try:
            manifest_record = manifest_by_track[track_id]
        except KeyError as exc:
            raise ValueError(f"V16 source parent lacks CQ08 manifest provenance: track_id={track_id}") from exc
        raw_path = _resolve(project_root, str(first["cq08_raw_echo_file"]))
        track_path = _resolve(project_root, str(first["cq08_track_file"]))
        point_path = _resolve(project_root, str(first["cq08_point_file"]))
        if str(manifest_record.get("label_name")) != str(first["label_name"]):
            raise ValueError(
                f"V16 manifest/index label mismatch for track_id={track_id}: "
                f"index={first['label_name']}, manifest={manifest_record.get('label_name')}"
            )
        _assert_same_path(
            raw_path,
            manifest_record.get("raw_echo_file"),
            field_name="raw_echo_file",
            track_id=track_id,
        )
        _assert_same_path(
            track_path,
            manifest_record.get("track_file"),
            field_name="track_file",
            track_id=track_id,
        )
        _assert_same_path(
            point_path,
            manifest_record.get("point_file"),
            field_name="point_file",
            track_id=track_id,
        )
        for path in (raw_path, track_path, point_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        rd_values = [
            str(value)
            for value in parent.loc[parent["rd_present"].map(_as_bool), "rd_tensor_path"].dropna().unique()
            if str(value).strip()
        ]
        if len(rd_values) > 1:
            raise ValueError(f"V16 parent references multiple RD tensors: {parent_id}: {rd_values}")
        rd_path = _resolve(project_root, rd_values[0]) if rd_values else None
        if rd_path is not None and not rd_path.is_file():
            raise FileNotFoundError(rd_path)
        rd_sha256 = None if rd_path is None else file_hash(rd_path)
        record = {
            "parent_sample_id": str(parent_id),
            "track_id": track_id,
            "label_name": str(first["label_name"]),
            "v3_split_name": str(first["split_name"]),
            "capture_id": str(manifest_record.get("capture_id", "")).strip(),
            "event_count": int(len(parent)),
            "raw_echo_file": str(raw_path),
            "track_file": str(track_path),
            "point_file": str(point_path),
            "rd_tensor_path": None if rd_path is None else str(rd_path),
            "raw_sha256": file_hash(raw_path),
            "track_sha256": file_hash(track_path),
            "point_sha256": file_hash(point_path),
            "rd_sha256": rd_sha256,
            "normalized_track_sha256": canonical_hash(track_path),
            "normalized_point_sha256": canonical_hash(point_path),
            "model_input_semantics_sha256": _model_input_semantics_sha256(
                parent,
                rd_sha256=rd_sha256,
            ),
        }
        if not record["capture_id"]:
            raise ValueError(f"V16 parent has an empty capture_id: {parent_id}")
        records.append(record)
        if number == 1 or number % 100 == 0 or number == len(parent_groups):
            print(
                f"[V16 build] hashed_parent={number}/{len(parent_groups)} "
                f"raw_cache={len(file_hash_cache)} canonical_cache={len(canonical_hash_cache)}",
                flush=True,
            )

    parent_ids = [str(value["parent_sample_id"]) for value in records]
    union = _UnionFind(parent_ids)
    groups_by_key: dict[str, dict[str, list[str]]] = {
        key: defaultdict(list) for key in GROUP_KEYS
    }
    records_by_parent = {str(value["parent_sample_id"]): value for value in records}
    for record in records:
        parent_id = str(record["parent_sample_id"])
        for key in GROUP_KEYS:
            value = record.get(key)
            if value is None or str(value).strip() in {"", "MISSING"}:
                continue
            groups_by_key[key][str(value)].append(parent_id)
    for key in GROUP_KEYS:
        for members in groups_by_key[key].values():
            anchor = members[0]
            for member in members[1:]:
                union.union(anchor, member)

    content_label_conflicts: list[dict[str, Any]] = []
    equivalence_groups: list[dict[str, Any]] = []
    for key in GROUP_KEYS:
        for value, members in sorted(groups_by_key[key].items()):
            if len(members) <= 1:
                continue
            labels = sorted({str(records_by_parent[parent]["label_name"]) for parent in members})
            splits = sorted({str(records_by_parent[parent]["v3_split_name"]) for parent in members})
            group_record = {
                "key_type": key,
                "key_sha256_or_id": value,
                "parent_count": len(members),
                "parent_sample_ids": sorted(members),
                "labels": labels,
                "v3_splits": splits,
            }
            equivalence_groups.append(group_record)
            if key in CONTENT_KEYS and len(labels) > 1:
                content_label_conflicts.append(group_record)
    if content_label_conflicts:
        preview = content_label_conflicts[:5]
        raise ValueError(
            "V16 found cross-label content-equivalent parents; manual label-policy adjudication is required: "
            + _canonical_json(preview)
        )

    components_by_root: dict[str, list[str]] = defaultdict(list)
    for parent_id in parent_ids:
        components_by_root[union.find(parent_id)].append(parent_id)
    components: list[dict[str, Any]] = []
    split_by_parent: dict[str, str] = {}
    component_by_parent: dict[str, str] = {}
    for members in sorted((sorted(value) for value in components_by_root.values()), key=lambda value: value[0]):
        component_id = _component_id(members)
        old_splits = sorted({str(records_by_parent[parent]["v3_split_name"]) for parent in members})
        new_split = max(old_splits, key=lambda value: SPLIT_PRIORITY[value])
        labels = sorted({str(records_by_parent[parent]["label_name"]) for parent in members})
        captures = sorted({str(records_by_parent[parent]["capture_id"]) for parent in members})
        components.append(
            {
                "component_id": component_id,
                "parent_count": len(members),
                "parent_sample_ids": members,
                "track_ids": sorted(int(records_by_parent[parent]["track_id"]) for parent in members),
                "labels": labels,
                "capture_ids": captures,
                "v3_splits": old_splits,
                "v16_split": new_split,
                "migration_reason": (
                    "unchanged_single_v3_split"
                    if len(old_splits) == 1
                    else "conservative_component_priority_test_then_val_then_train"
                ),
            }
        )
        for parent in members:
            split_by_parent[parent] = new_split
            component_by_parent[parent] = component_id

    for record in records:
        parent_id = str(record["parent_sample_id"])
        record["provenance_component_id"] = component_by_parent[parent_id]
        record["v16_split_name"] = split_by_parent[parent_id]
        record["moved_from_v3_split"] = record["v3_split_name"] != record["v16_split_name"]

    new_index = source.copy()
    new_index["v3_split_name"] = new_index["split_name"].astype(str)
    new_index["provenance_component_id"] = new_index["parent_sample_id"].astype(str).map(component_by_parent)
    new_index["capture_id"] = new_index["parent_sample_id"].astype(str).map(
        {parent: str(value["capture_id"]) for parent, value in records_by_parent.items()}
    )
    new_index["split_group_id"] = new_index["provenance_component_id"]
    new_index["split_name"] = new_index["parent_sample_id"].astype(str).map(split_by_parent)
    new_index["schema_version"] = VERSION
    if new_index[["provenance_component_id", "capture_id", "split_name"]].isna().any().any():
        raise ValueError("V16 index mapping produced missing provenance or split fields")
    index_write_status = _write_csv_new_or_identical(index_path, new_index)
    index_sha256 = _sha256_file(index_path)

    split_parent_counts = {
        split: int(sum(value == split for value in split_by_parent.values()))
        for split in SPLIT_PRIORITY
    }
    split_point_counts = {
        split: int(new_index["split_name"].astype(str).eq(split).sum()) for split in SPLIT_PRIORITY
    }
    split_label_parent_counts = {
        split: _label_counts(new_index[new_index["split_name"].astype(str).eq(split)])
        for split in SPLIT_PRIORITY
    }
    for split, counts in split_label_parent_counts.items():
        missing_labels = [label for label, count in counts.items() if count <= 0]
        if missing_labels:
            raise ValueError(f"V16 {split} split lacks labels: {missing_labels}")

    key_split_violations: dict[str, int] = {}
    for key in GROUP_KEYS:
        violations = 0
        for members in groups_by_key[key].values():
            if len({split_by_parent[parent] for parent in members}) > 1:
                violations += 1
        key_split_violations[key] = violations
    if any(key_split_violations.values()):
        raise ValueError(f"V16 split leaves provenance keys across splits: {key_split_violations}")
    protected_to_train = [
        record["parent_sample_id"]
        for record in records
        if record["v3_split_name"] in {"val", "test"} and record["v16_split_name"] == "train"
    ]
    if protected_to_train:
        raise ValueError(f"V16 moved old validation/test parents into train: {protected_to_train[:5]}")

    provenance_payload = {
        "schema_version": VERSION,
        "manifest_name": "cq08_capture_content_provenance_v1",
        "source_index": str(SOURCE_INDEX_RELATIVE).replace("\\", "/"),
        "source_index_sha256": _sha256_file(source_index_path),
        "source_manifest": str(SOURCE_MANIFEST_RELATIVE).replace("\\", "/"),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "parent_count": len(records),
        "component_count": len(components),
        "group_key_semantics": {
            "capture_id": "same acquisition timestamp/group; always forced to one split",
            "raw_sha256": "byte-identical original echo file",
            "rd_sha256": "byte-identical canonical RD NPZ",
            "normalized_track_sha256": "track text rows with track-id wrapper removed",
            "normalized_point_sha256": "point text rows with track-id wrapper removed",
            "model_input_semantics_sha256": "label/path/id-free causal TR/RD event contract",
        },
        "cross_label_content_conflict_count": len(content_label_conflicts),
        "cross_label_capture_group_count": sum(
            1
            for value in equivalence_groups
            if value["key_type"] == "capture_id" and len(value["labels"]) > 1
        ),
        "equivalence_groups": equivalence_groups,
        "components": components,
        "records": records,
        "test_payload_use": {
            "raw_and_rd_read_for_sha256_provenance_only": True,
            "test_values_used_for_normalization": False,
            "test_used_for_model_selection": False,
        },
    }
    provenance_write_status = _write_json_new_or_identical(provenance_path, provenance_payload)
    provenance_sha256 = _sha256_file(provenance_path)

    moved = [value for value in records if value["moved_from_v3_split"]]
    split_payload = {
        "schema_version": VERSION,
        "source_name": "CQ08",
        "group_unit": "capture_content_connected_component",
        "component_keys": list(GROUP_KEYS),
        "split_policy": "retain V3 split unless connected; component priority is test > val > train",
        "old_validation_or_test_to_train_count": len(protected_to_train),
        "parent_counts": split_parent_counts,
        "point_counts": split_point_counts,
        "label_parent_counts": split_label_parent_counts,
        "component_counts": {
            split: int(sum(value["v16_split"] == split for value in components))
            for split in SPLIT_PRIORITY
        },
        "moved_parent_count": len(moved),
        "moved_parents": [
            {
                "parent_sample_id": value["parent_sample_id"],
                "track_id": value["track_id"],
                "label_name": value["label_name"],
                "from": value["v3_split_name"],
                "to": value["v16_split_name"],
                "component_id": value["provenance_component_id"],
            }
            for value in moved
        ],
        "split_purity": {
            "parent_cross_split_count": 0,
            "key_cross_split_group_counts": key_split_violations,
            "all_checked_keys_zero": not any(key_split_violations.values()),
        },
        "paths": {
            "index": str(INDEX_RELATIVE).replace("\\", "/"),
            "provenance": str(PROVENANCE_RELATIVE).replace("\\", "/"),
        },
        "hashes": {
            "index_sha256": index_sha256,
            "provenance_sha256": provenance_sha256,
        },
    }
    split_write_status = _write_json_new_or_identical(split_path, split_payload)
    split_sha256 = _sha256_file(split_path)

    normalization_result = build_normalization(
        project_root,
        index_path,
        stats_path,
        "train",
        allowed_labels=None,
        forbidden_labels=(),
        schema_version=VERSION,
        stats_name="cq08_v16_capture_content_grouped_all_label_train_normalization_v1",
        scope=(
            "V16 capture/content-grouped CQ08 train split, all five labels; "
            "validation and test excluded from normalization"
        ),
    )
    stats_sha256 = _sha256_file(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_counts = split_label_parent_counts["train"]
    dt_counts = {
        str(key): int(value)
        for key, value in new_index.groupby("tr_dt_reason").size().sort_index().items()
    }
    contract_payload = {
        "schema_version": VERSION,
        "contract_name": "cq08_capture_content_grouped_product_local_liquid_tasks_v1",
        "contract_status": "implementation_prepared_formal_training_not_executed",
        "source_index": str(INDEX_RELATIVE).replace("\\", "/"),
        "source_index_sha256": index_sha256,
        "source_provenance": str(PROVENANCE_RELATIVE).replace("\\", "/"),
        "source_provenance_sha256": provenance_sha256,
        "source_split": str(SPLIT_RELATIVE).replace("\\", "/"),
        "source_split_sha256": split_sha256,
        "source_normalization_stats": str(STATS_RELATIVE).replace("\\", "/"),
        "source_normalization_stats_sha256": stats_sha256,
        "source_purity": {
            "allowed_sources": ["CQ08"],
            "excluded_sources": ["raw", "Excel", "RD-only"],
            "raw_echo_use": "SHA-256 provenance only; not a model tensor",
            "synthetic_data_used": False,
        },
        "data_scope": {
            "parent_count": len(records),
            "point_count": int(len(new_index)),
            "split_parent_counts": split_parent_counts,
            "split_point_counts": split_point_counts,
            "split_label_parent_counts": split_label_parent_counts,
            "label_mapping": {label: index for index, label in enumerate(BASE_LABELS)},
            "test_payload_access_for_hash_only": True,
            "test_used_for_normalization": False,
            "test_used_for_checkpoint_selection": False,
            "test_evaluation": "未执行",
        },
        "provenance_contract": {
            "group_unit": "capture_content_connected_component",
            "group_keys": list(GROUP_KEYS),
            "component_count": len(components),
            "cross_label_content_conflict_count": len(content_label_conflicts),
            "old_validation_or_test_to_train_count": len(protected_to_train),
            "key_cross_split_group_counts": key_split_violations,
            "all_split_purity_checks_passed": not any(key_split_violations.values()),
        },
        "normalization_contract": {
            "split": "train",
            "all_five_labels_included": True,
            "parent_count": int(stats["parent_count"]),
            "label_parent_counts": stats["selected_label_parent_counts"],
            "validation_and_test_excluded": True,
            "sha256": stats_sha256,
        },
        "time_contract": {
            "raw_precision_seconds": 1.0,
            "dt_field": "tr_dt_seconds",
            "dt_valid_field": "tr_dt_valid",
            "dt_reason_field": "tr_dt_reason",
            "observed_counts_all_splits": dt_counts,
            "first_event_rule": "initialize from current observation without elapsed-time claim",
            "zero_delta_rule": "bounded observation update; counted as zero_dt, never as a gap",
            "positive_delta_rule": "continuous-time liquid update with retained positive seconds",
            "gap_fraction_rule": "reserved for genuine missing/unavailable/over-threshold intervals; current V16 index has no such reason",
            "padding_rule": "never update and never enter a dt denominator",
        },
        "feature_contract": {
            "tr_value_dim": 14,
            "tr_missing_dim": 11,
            "rd_quality_dim": 7,
            "rd_quality_schema": "causal-current-event-v1",
            "rd_quality_names": [
                "rd_present",
                "rd_frame_count_scaled",
                "rd_local_duration_scaled",
                "rd_missing",
                "rd_local_time_non_monotonic",
                "rd_multiple_frames",
                "rd_frames_truncated",
            ],
            "rd_frame_shape": [2, 21, 384],
            "rd_normalization_stats": str(STATS_RELATIVE).replace("\\", "/"),
            "rd_normalization_stats_sha256": stats_sha256,
            "excluded_model_inputs": [
                "capture_id",
                "provenance_component_id",
                "v3_split_name",
                "split_name",
                "label_name",
                "path",
                "filename",
                "parent_sample_id",
                "track_id",
                "raw/RD/provenance SHA-256",
            ],
            "dt_reason_input_scope": "liquid dynamics control only; never concatenated into classification fusion",
        },
        "task_contracts": {
            task_name: {
                "class_names": task["class_names"],
                "label_to_id": task["label_to_id"],
                "allowed_training_labels": task["allowed_labels"],
                "train_parent_count": int(sum(train_counts[label] for label in task["allowed_labels"])),
                "validation_parent_count": int(
                    sum(split_label_parent_counts["val"][label] for label in task["allowed_labels"])
                ),
                "target_min_precision_recall": float(task["target"]),
                "loss": "parent-final CE plus 0.10 parent-balanced causal-prefix auxiliary from event 3, excluding final",
                "checkpoint_selection": "validation parent minimum per-class precision/recall, parent Macro-F1 tie-break",
            }
            for task_name, task in TASKS.items()
        },
        "training_contract": {
            "entry": "src-v3/train/tasks.py",
            "optimization": "three independent runs, optimizers, checkpoints and best epochs",
            "joint_training": False,
            "main_loss": "one final valid state per parent",
            "prefix_auxiliary_weight": 0.1,
            "prefix_auxiliary_min_events": 3,
            "prefix_final_state_excluded": True,
            "epochs": 20,
            "early_stopping": False,
            "full_cascade_evaluation": "未执行",
            "rejection_calibration": "未执行",
            "stable": "未执行",
            "test": "未执行",
            "p95": "未执行",
        },
        "action_semantics": {
            "unknown_supervised_label": "CU specialist label used for local boundary learning",
            "unknown_rejected_action": "separate future runtime action; not evaluated in this local-task stage",
            "dbb_not_accepted_future_route": "must fall back to CU; direct UNKNOWN_REJECTED is forbidden in V16 product action",
        },
    }
    contract_write_status = _write_json_new_or_identical(contract_path, contract_payload)
    contract_sha256 = _sha256_file(contract_path)

    config_results: dict[str, Any] = {}
    for task_name in TASKS:
        payload = _config_payload(
            task_name=task_name,
            index_sha256=index_sha256,
            contract_sha256=contract_sha256,
            stats_sha256=stats_sha256,
            amplitude_center=float(stats["amplitude"]["center"]),
            amplitude_scale=float(stats["amplitude"]["scale"]),
            train_parent_counts=train_counts,
        )
        slug = TASKS[task_name]["slug"]
        config_relative = CONFIG_DIR_RELATIVE / f"RADAR-LN3-V16-cq08-liquid-product-local-{slug}-config.json"
        config_path = project_root / config_relative
        artifact_path = project_root / payload["train"]["formal_artifact_dir"]
        artifact_nonempty = artifact_path.exists() and any(artifact_path.iterdir())
        if artifact_nonempty and not config_path.is_file():
            raise FileExistsError(f"refusing to prepare over a non-empty V16 artifact: {artifact_path}")
        status = _write_json_new_or_identical(config_path, payload)
        config_results[task_name] = {
            "path": str(config_relative).replace("\\", "/"),
            "sha256": _sha256_file(config_path),
            "write_status": status,
            "existing_artifact_nonempty": artifact_nonempty,
            "train_parent_count": contract_payload["task_contracts"][task_name]["train_parent_count"],
            "validation_parent_count": contract_payload["task_contracts"][task_name]["validation_parent_count"],
            "target_min_precision_recall": TASKS[task_name]["target"],
        }

    return {
        "status": "completed",
        "schema_version": VERSION,
        "write_status": {
            "provenance": provenance_write_status,
            "split": split_write_status,
            "index": index_write_status,
            "normalization": normalization_result["write_status"],
            "contract": contract_write_status,
        },
        "paths": {
            "provenance": str(provenance_path),
            "split": str(split_path),
            "index": str(index_path),
            "normalization": str(stats_path),
            "contract": str(contract_path),
        },
        "hashes": {
            "provenance_sha256": provenance_sha256,
            "split_sha256": split_sha256,
            "index_sha256": index_sha256,
            "normalization_sha256": stats_sha256,
            "contract_sha256": contract_sha256,
        },
        "parent_counts": split_parent_counts,
        "label_parent_counts": split_label_parent_counts,
        "component_count": len(components),
        "moved_parent_count": len(moved),
        "cross_label_content_conflict_count": len(content_label_conflicts),
        "split_purity": key_split_violations,
        "configs": config_results,
        "formal_training": "未执行",
        "full_cascade_evaluation": "未执行",
        "oof_calibration": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "p95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the RADAR-LN3 V16 capture/content-grouped local product-task contract."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
