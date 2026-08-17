from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "RADAR-LN3-V21-clutter-coverage-semantics-audit-v1"
CLUTTER_LABEL = "ClutterTarget"
UNKNOWN_LABEL = "UnknownTarget"
SOURCE_CLUTTER_LABEL_ID = 5
SOURCE_UNKNOWN_LABEL_ID = 6

SOURCE_MANIFEST = Path("data/manifests/cq08_unified_manifest.json")
SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
SOURCE_SPLIT = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-split.json"
)
SOURCE_PROVENANCE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
SOURCE_MATLAB_LABEL_SCRIPT = Path("OriData/23/CQ-08/batch_process_all.m")

PARENT_REGISTRY = Path(
    "data/processed/RADAR-LN3-V21-cq08-clutter-parent-registry.csv"
)
CAPTURE_REGISTRY = Path(
    "data/processed/RADAR-LN3-V21-cq08-clutter-capture-registry.csv"
)
BURST_REGISTRY = Path(
    "data/processed/RADAR-LN3-V21-cq08-clutter-acquisition-burst-registry.csv"
)
AUDIT_CLUSTER_PROFILES = Path(
    "data/processed/RADAR-LN3-V21-cq08-clutter-audit-cluster-profiles.csv"
)
SEMANTIC_ADJUDICATION_TEMPLATE = Path(
    "data/processed/RADAR-LN3-V21-cq08-clutter-semantic-adjudication-template.csv"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V21-clutter-coverage-semantics-audit.json"
)
ADJUDICATION_PROTOCOL = Path("docs/radar_ln3_clutter_semantic_adjudication_protocol.md")

BURST_THRESHOLDS_SECONDS = (30, 60, 120, 300, 600)
AUDIT_CLUSTER_COUNT = 4

# These are descriptive audit dimensions, not model inputs or semantic subtype labels.
AUDIT_CLUSTER_FEATURES = (
    "log1p_event_count",
    "log1p_duration_seconds",
    "rd_present_fraction",
    "log1p_rd_frames_per_event",
    "zero_dt_fraction_nonfirst",
    "log1p_range_median",
    "log1p_range_span",
    "abs_radial_speed_median",
    "radial_speed_iqr",
)

DESCRIPTOR_FIELDS = (
    "event_count",
    "duration_seconds",
    "rd_present_event_count",
    "rd_present_fraction",
    "rd_frame_count_total",
    "rd_frames_per_event",
    "rd_quality_missing_fraction",
    "rd_quality_partial_parent_fraction",
    "zero_dt_event_count",
    "zero_dt_fraction_nonfirst",
    "positive_dt_event_count",
    "positive_dt_fraction_nonfirst",
    "range_valid_event_count",
    "range_median_m",
    "range_iqr_m",
    "range_span_m",
    "radial_speed_valid_event_count",
    "radial_speed_median_mps",
    "abs_radial_speed_median_mps",
    "radial_speed_iqr_mps",
    "radial_speed_std_mps",
    "speed_median_mps",
    "snr_median_db",
)

PARENT_REGISTRY_FIELDS = (
    "schema_version",
    "parent_sample_id",
    "track_id",
    "split_name",
    "v3_split_name",
    "source_numeric_label_id",
    "source_matlab_token",
    "current_label_name",
    "capture_id",
    "capture_token_date",
    "provenance_component_id",
    "provenance_component_parent_count",
    "provenance_component_labels_json",
    "provenance_component_cross_label",
    "moved_from_v3_split",
    "manifest_track_length_hint",
    "manifest_track_row_count",
    "manifest_point_row_count",
    "manifest_rd_mat_frame_count",
    "manifest_has_track",
    "manifest_has_points",
    "manifest_has_raw_echo",
    "manifest_has_rd_mat",
    "manifest_is_usable",
    "raw_sha256",
    "track_sha256",
    "point_sha256",
    "rd_sha256",
    "model_input_semantics_sha256",
    *DESCRIPTOR_FIELDS,
    "audit_cluster_id",
    "audit_cluster_assignment_scope",
    "audit_cluster_distance",
    "burst_030s_id",
    "burst_060s_id",
    "burst_120s_id",
    "burst_300s_id",
    "burst_600s_id",
    "semantic_subtype",
    "semantic_review_status",
    "payload_audit_scope",
)

CAPTURE_REGISTRY_FIELDS = (
    "schema_version",
    "capture_id",
    "capture_token_datetime",
    "capture_token_date",
    "capture_token_provenance",
    "v16_split_name",
    "clutter_parent_count",
    "clutter_parent_sample_ids_json",
    "clutter_track_ids_json",
    "all_v16_parent_count_at_capture",
    "all_v16_parent_sample_ids_json",
    "all_v16_track_ids_json",
    "all_v16_labels_json",
    "cross_label_capture",
    "provenance_component_ids_json",
    "burst_030s_id",
    "burst_060s_id",
    "burst_120s_id",
    "burst_300s_id",
    "burst_600s_id",
    "confirmed_acquisition_group_id",
    "confirmed_environment_id",
    "confirmed_sensor_mode_id",
    "capture_review_status",
)

BURST_REGISTRY_FIELDS = (
    "schema_version",
    "threshold_seconds",
    "burst_id",
    "start_capture_id",
    "end_capture_id",
    "span_seconds",
    "parent_count",
    "capture_count",
    "split_names_json",
    "cross_split",
    "parent_sample_ids_json",
    "track_ids_json",
    "capture_ids_json",
    "provenance_component_ids_json",
    "semantic_status",
)

AUDIT_CLUSTER_PROFILE_FIELDS = (
    "schema_version",
    "audit_cluster_id",
    "fit_population",
    "semantic_status",
    "train_parent_count",
    "validation_nearest_cluster_parent_count",
    "train_parent_sample_ids_json",
    "validation_parent_sample_ids_json",
    *tuple(
        f"train_median_{field}"
        for field in (
            "event_count",
            "duration_seconds",
            "rd_present_fraction",
            "rd_frames_per_event",
            "zero_dt_fraction_nonfirst",
            "range_median_m",
            "range_span_m",
            "abs_radial_speed_median_mps",
            "radial_speed_iqr_mps",
        )
    ),
)

SEMANTIC_TEMPLATE_FIELDS = (
    "schema_version",
    "parent_sample_id",
    "track_id",
    "split_name",
    "capture_id",
    "provenance_component_id",
    "audit_cluster_id",
    "candidate_burst_060s_id",
    "confirmed_acquisition_group_id",
    "confirmed_environment_id",
    "confirmed_sensor_mode_id",
    "source_numeric_label_id",
    "source_matlab_token",
    "current_label_name",
    "review_status",
    "reviewer",
    "reviewed_at_iso8601",
    "acquisition_log_available",
    "operator_note_available",
    "scene_context_available",
    "raw_rd_evidence_reviewed",
    "tr_track_evidence_reviewed",
    "semantic_subtype",
    "secondary_semantic_subtype",
    "subtype_confidence",
    "subtype_evidence_basis",
    "source_label_decision",
    "evidence_references",
    "notes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_source_text(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", payload, 0, 1, f"cannot decode {path}")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _capture_datetime(capture_id: str) -> datetime:
    if not re.fullmatch(r"\d{14}", capture_id):
        raise ValueError(f"invalid CQ08 capture_id: {capture_id!r}")
    return datetime.strptime(capture_id, "%Y%m%d%H%M%S")


def _track_sort_key(parent_sample_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", parent_sample_id)
    return (int(match.group(1)) if match else 2**63 - 1, parent_sample_id)


def _quantile(values: Sequence[float], q: float) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


def _median(values: Sequence[float]) -> float | None:
    return _quantile(values, 0.5)


def _span(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(max(finite) - min(finite)) if finite else None


def _descriptor_rows(index_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_parent_ids: set[str] = set()
    counters = Counter()
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "parent_sample_id",
            "track_id",
            "split_name",
            "label_name",
            "causal_event_index",
            "tr_time_seconds",
            "tr_dt_reason",
            "tr_feature_values_json",
            "rd_present",
            "rd_frame_count",
            "rd_quality_missing",
            "rd_quality_partial_parent",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"V16 index is missing Clutter audit columns: {missing}")
        for row in reader:
            if str(row["label_name"]) != CLUTTER_LABEL:
                continue
            counters["clutter_index_event_rows"] += 1
            split = str(row["split_name"])
            parent_id = str(row["parent_sample_id"])
            if split == "test":
                # The sealed split is registered from metadata only. In particular, do not
                # parse its TR feature JSON or consume any raw/MAT/RD tensor payload here.
                test_parent_ids.add(parent_id)
                counters["test_index_rows_skipped_before_physical_parse"] += 1
                continue
            if split not in {"train", "val"}:
                raise ValueError(f"unexpected Clutter split {split!r}")
            features = json.loads(str(row["tr_feature_values_json"]))
            counters[f"{split}_physical_index_rows_parsed"] += 1
            grouped[parent_id].append(
                {
                    "parent_sample_id": parent_id,
                    "track_id": int(row["track_id"]),
                    "split_name": split,
                    "event_index": int(row["causal_event_index"]),
                    "time_seconds": float(row["tr_time_seconds"]),
                    "dt_reason": str(row["tr_dt_reason"]),
                    "rd_present": _as_bool(row["rd_present"]),
                    "rd_frame_count": int(row["rd_frame_count"]),
                    "rd_quality_missing": _as_bool(row["rd_quality_missing"]),
                    "rd_quality_partial_parent": _as_bool(
                        row["rd_quality_partial_parent"]
                    ),
                    "features": features,
                }
            )

    descriptors: dict[str, dict[str, Any]] = {}
    for parent_id, rows in grouped.items():
        rows.sort(key=lambda item: item["event_index"])
        if [item["event_index"] for item in rows] != list(range(len(rows))):
            raise ValueError(f"Clutter parent {parent_id} has non-contiguous causal events")
        splits = {str(item["split_name"]) for item in rows}
        tracks = {int(item["track_id"]) for item in rows}
        if len(splits) != 1 or len(tracks) != 1:
            raise ValueError(f"Clutter parent {parent_id} is internally inconsistent")
        event_count = len(rows)
        nonfirst = max(1, event_count - 1)
        times = [float(item["time_seconds"]) for item in rows]
        ranges = [
            value
            for item in rows
            if (value := _as_float(item["features"].get("range_m"))) is not None
        ]
        radial = [
            value
            for item in rows
            if (value := _as_float(item["features"].get("radial_speed_mps")))
            is not None
        ]
        speeds = [
            value
            for item in rows
            if (value := _as_float(item["features"].get("speed_mps"))) is not None
        ]
        snr = [
            value
            for item in rows
            if (value := _as_float(item["features"].get("snr_db"))) is not None
        ]
        rd_present_count = sum(bool(item["rd_present"]) for item in rows)
        rd_frames = [int(item["rd_frame_count"]) for item in rows]
        zero_count = sum(
            item["dt_reason"] == "zero_delta_at_second_precision" for item in rows
        )
        positive_count = sum(
            item["dt_reason"] == "positive_delta_at_second_precision" for item in rows
        )
        descriptor = {
            "parent_sample_id": parent_id,
            "track_id": next(iter(tracks)),
            "split_name": next(iter(splits)),
            "event_count": event_count,
            "duration_seconds": float(max(times) - min(times)),
            "rd_present_event_count": int(rd_present_count),
            "rd_present_fraction": float(rd_present_count / event_count),
            "rd_frame_count_total": int(sum(rd_frames)),
            "rd_frames_per_event": float(sum(rd_frames) / event_count),
            "rd_quality_missing_fraction": float(
                sum(bool(item["rd_quality_missing"]) for item in rows) / event_count
            ),
            "rd_quality_partial_parent_fraction": float(
                sum(bool(item["rd_quality_partial_parent"]) for item in rows)
                / event_count
            ),
            "zero_dt_event_count": int(zero_count),
            "zero_dt_fraction_nonfirst": float(zero_count / nonfirst),
            "positive_dt_event_count": int(positive_count),
            "positive_dt_fraction_nonfirst": float(positive_count / nonfirst),
            "range_valid_event_count": len(ranges),
            "range_median_m": _median(ranges),
            "range_iqr_m": (
                float(_quantile(ranges, 0.75) - _quantile(ranges, 0.25))
                if ranges
                else None
            ),
            "range_span_m": _span(ranges),
            "radial_speed_valid_event_count": len(radial),
            "radial_speed_median_mps": _median(radial),
            "abs_radial_speed_median_mps": _median([abs(value) for value in radial]),
            "radial_speed_iqr_mps": (
                float(_quantile(radial, 0.75) - _quantile(radial, 0.25))
                if radial
                else None
            ),
            "radial_speed_std_mps": (
                float(np.std(np.asarray(radial, dtype=np.float64))) if radial else None
            ),
            "speed_median_mps": _median(speeds),
            "snr_median_db": _median(snr),
        }
        descriptors[parent_id] = descriptor
    counters["test_parent_count_registered_without_physical_parse"] = len(test_parent_ids)
    counters["test_tr_feature_json_parse_count"] = 0
    counters["test_raw_file_open_count"] = 0
    counters["test_mat_file_open_count"] = 0
    counters["test_rd_tensor_open_count"] = 0
    return descriptors, {str(key): int(value) for key, value in sorted(counters.items())}


def _cluster_vector(descriptor: Mapping[str, Any]) -> list[float]:
    def nonnegative(name: str) -> float:
        value = _as_float(descriptor.get(name))
        return max(0.0, float(value or 0.0))

    return [
        math.log1p(nonnegative("event_count")),
        math.log1p(nonnegative("duration_seconds")),
        nonnegative("rd_present_fraction"),
        math.log1p(nonnegative("rd_frames_per_event")),
        nonnegative("zero_dt_fraction_nonfirst"),
        math.log1p(nonnegative("range_median_m")),
        math.log1p(nonnegative("range_span_m")),
        nonnegative("abs_radial_speed_median_mps"),
        nonnegative("radial_speed_iqr_mps"),
    ]


def _fit_audit_clusters(
    descriptors: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_ids = sorted(
        (
            parent_id
            for parent_id, descriptor in descriptors.items()
            if descriptor["split_name"] == "train"
        ),
        key=_track_sort_key,
    )
    val_ids = sorted(
        (
            parent_id
            for parent_id, descriptor in descriptors.items()
            if descriptor["split_name"] == "val"
        ),
        key=_track_sort_key,
    )
    if len(train_ids) < AUDIT_CLUSTER_COUNT:
        raise ValueError("too few train Clutter parents for the frozen audit cluster count")
    train_raw = np.asarray(
        [_cluster_vector(descriptors[parent_id]) for parent_id in train_ids],
        dtype=np.float64,
    )
    if not np.isfinite(train_raw).all():
        raise ValueError("non-finite train Clutter audit descriptor")
    median = np.median(train_raw, axis=0)
    q25 = np.quantile(train_raw, 0.25, axis=0)
    q75 = np.quantile(train_raw, 0.75, axis=0)
    scale = q75 - q25
    std = np.std(train_raw, axis=0)
    scale = np.where(scale > 1.0e-9, scale, np.where(std > 1.0e-9, std, 1.0))
    train = (train_raw - median) / scale

    # Deterministic farthest-point initialization avoids a hidden random seed.
    first_dist = np.sum(train * train, axis=1)
    selected = [min(range(len(train_ids)), key=lambda i: (first_dist[i], train_ids[i]))]
    while len(selected) < AUDIT_CLUSTER_COUNT:
        distances = np.min(
            np.stack(
                [np.sum((train - train[index]) ** 2, axis=1) for index in selected],
                axis=1,
            ),
            axis=1,
        )
        candidates = [index for index in range(len(train_ids)) if index not in selected]
        selected.append(max(candidates, key=lambda i: (distances[i], train_ids[i])))
    centroids = train[np.asarray(selected, dtype=np.int64)].copy()
    assignments = np.full(len(train_ids), -1, dtype=np.int64)
    for _ in range(100):
        distances = np.sum((train[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        for cluster_index in range(AUDIT_CLUSTER_COUNT):
            members = train[assignments == cluster_index]
            if members.size == 0:
                nearest = np.min(distances, axis=1)
                replacement = max(
                    range(len(train_ids)), key=lambda i: (nearest[i], train_ids[i])
                )
                centroids[cluster_index] = train[replacement]
            else:
                centroids[cluster_index] = np.mean(members, axis=0)
    else:
        raise RuntimeError("deterministic Clutter audit clustering did not converge")

    # Give clusters stable names based on their minimum train parent id, not centroid order.
    old_clusters = list(range(AUDIT_CLUSTER_COUNT))
    old_clusters.sort(
        key=lambda cluster: min(
            (_track_sort_key(train_ids[index]) for index in np.where(assignments == cluster)[0]),
            default=(2**63 - 1, ""),
        )
    )
    old_to_new = {old: new for new, old in enumerate(old_clusters)}
    centroids = centroids[np.asarray(old_clusters, dtype=np.int64)]
    assignments = np.asarray([old_to_new[int(value)] for value in assignments])

    result: dict[str, dict[str, Any]] = {}
    train_distance = np.sqrt(
        np.sum((train - centroids[assignments]) ** 2, axis=1)
    )
    for index, parent_id in enumerate(train_ids):
        result[parent_id] = {
            "audit_cluster_id": f"audit_cluster_{int(assignments[index]) + 1:02d}",
            "audit_cluster_assignment_scope": "train_fit_member",
            "audit_cluster_distance": float(train_distance[index]),
        }

    if val_ids:
        val_raw = np.asarray(
            [_cluster_vector(descriptors[parent_id]) for parent_id in val_ids],
            dtype=np.float64,
        )
        val = (val_raw - median) / scale
        val_distances = np.sum(
            (val[:, None, :] - centroids[None, :, :]) ** 2, axis=2
        )
        val_assignments = np.argmin(val_distances, axis=1)
        for index, parent_id in enumerate(val_ids):
            cluster = int(val_assignments[index])
            result[parent_id] = {
                "audit_cluster_id": f"audit_cluster_{cluster + 1:02d}",
                "audit_cluster_assignment_scope": "validation_nearest_train_cluster",
                "audit_cluster_distance": float(math.sqrt(val_distances[index, cluster])),
            }

    profiles: list[dict[str, Any]] = []
    for cluster_index in range(AUDIT_CLUSTER_COUNT):
        cluster_id = f"audit_cluster_{cluster_index + 1:02d}"
        train_members = [
            parent_id
            for parent_id in train_ids
            if result[parent_id]["audit_cluster_id"] == cluster_id
        ]
        val_members = [
            parent_id
            for parent_id in val_ids
            if result[parent_id]["audit_cluster_id"] == cluster_id
        ]
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "audit_cluster_id": cluster_id,
            "fit_population": "train_ClutterTarget_parent_only",
            "semantic_status": "descriptive_cluster_not_semantic_subtype",
            "train_parent_count": len(train_members),
            "validation_nearest_cluster_parent_count": len(val_members),
            "train_parent_sample_ids_json": _json_compact(train_members),
            "validation_parent_sample_ids_json": _json_compact(val_members),
        }
        for field in (
            "event_count",
            "duration_seconds",
            "rd_present_fraction",
            "rd_frames_per_event",
            "zero_dt_fraction_nonfirst",
            "range_median_m",
            "range_span_m",
            "abs_radial_speed_median_mps",
            "radial_speed_iqr_mps",
        ):
            values = [
                float(descriptors[parent_id][field])
                for parent_id in train_members
                if descriptors[parent_id].get(field) is not None
            ]
            row[f"train_median_{field}"] = _median(values)
        profiles.append(row)

    method = {
        "status": "descriptive_only_unadjudicated",
        "fit_population": "V16 train ClutterTarget parents only",
        "fit_parent_count": len(train_ids),
        "validation_assignment": "nearest frozen train centroid; no refit or tuning",
        "validation_parent_count": len(val_ids),
        "test_assignment": "sealed_unassigned",
        "test_parent_count_used": 0,
        "cluster_count": AUDIT_CLUSTER_COUNT,
        "cluster_count_policy": (
            "frozen descriptive partition for coverage review; not selected by label or validation performance"
        ),
        "feature_names": list(AUDIT_CLUSTER_FEATURES),
        "scaling": "train median and train IQR; train std fallback for zero-IQR dimensions",
        "initialization": "deterministic farthest-point",
        "semantic_claim": False,
        "model_input": False,
        "training_weight_or_sampler_input": False,
    }
    return result, profiles, method


def _build_bursts(
    clutter_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[int, str]], list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        clutter_records,
        key=lambda row: (_capture_datetime(str(row["capture_id"])), int(row["track_id"])),
    )
    assignments: dict[str, dict[int, str]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for threshold in BURST_THRESHOLDS_SECONDS:
        bursts: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        previous: datetime | None = None
        for record in ordered:
            capture_time = _capture_datetime(str(record["capture_id"]))
            if previous is None or (capture_time - previous).total_seconds() <= threshold:
                current.append(record)
            else:
                bursts.append(current)
                current = [record]
            previous = capture_time
        if current:
            bursts.append(current)
        for burst_index, members in enumerate(bursts, start=1):
            burst_id = f"burst_{threshold:03d}s_{burst_index:03d}"
            parent_ids = sorted(
                (str(member["parent_sample_id"]) for member in members),
                key=_track_sort_key,
            )
            for parent_id in parent_ids:
                assignments[parent_id][threshold] = burst_id
            capture_ids = sorted(str(member["capture_id"]) for member in members)
            starts = _capture_datetime(capture_ids[0])
            ends = _capture_datetime(capture_ids[-1])
            splits = sorted({str(member["v16_split_name"]) for member in members})
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "threshold_seconds": threshold,
                    "burst_id": burst_id,
                    "start_capture_id": capture_ids[0],
                    "end_capture_id": capture_ids[-1],
                    "span_seconds": int((ends - starts).total_seconds()),
                    "parent_count": len(parent_ids),
                    "capture_count": len(set(capture_ids)),
                    "split_names_json": _json_compact(splits),
                    "cross_split": len(splits) > 1,
                    "parent_sample_ids_json": _json_compact(parent_ids),
                    "track_ids_json": _json_compact(
                        sorted(int(member["track_id"]) for member in members)
                    ),
                    "capture_ids_json": _json_compact(capture_ids),
                    "provenance_component_ids_json": _json_compact(
                        sorted(
                            {
                                str(member["provenance_component_id"])
                                for member in members
                            }
                        )
                    ),
                    "semantic_status": "candidate_acquisition_burst_unadjudicated",
                }
            )
        threshold_rows = [row for row in rows if row["threshold_seconds"] == threshold]
        summary[str(threshold)] = {
            "candidate_burst_count": len(bursts),
            "multi_parent_burst_count": sum(row["parent_count"] > 1 for row in threshold_rows),
            "cross_split_candidate_burst_count": sum(
                bool(row["cross_split"]) for row in threshold_rows
            ),
            "maximum_parent_count": max(row["parent_count"] for row in threshold_rows),
            "interpretation": "time-adjacency candidate only; not a confirmed session or semantic subtype",
        }
    return assignments, rows, summary


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _output_path(base: Path, relative: Path) -> Path:
    return base / relative


def build(project_root: Path, *, output_root: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    output_base = (output_root or root).resolve()
    source_paths = {
        "manifest": root / SOURCE_MANIFEST,
        "index": root / SOURCE_INDEX,
        "split": root / SOURCE_SPLIT,
        "provenance": root / SOURCE_PROVENANCE,
        "matlab_label_script": root / SOURCE_MATLAB_LABEL_SCRIPT,
        "adjudication_protocol": root / ADJUDICATION_PROTOCOL,
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Clutter audit source files are missing: {missing}")

    manifest = _load_json(source_paths["manifest"])
    provenance = _load_json(source_paths["provenance"])
    split_contract = _load_json(source_paths["split"])
    matlab = _read_source_text(source_paths["matlab_label_script"])
    if not isinstance(manifest, list) or not isinstance(provenance, dict):
        raise ValueError("unexpected CQ08 manifest/provenance structure")

    clutter_manifest = [
        record for record in manifest if str(record.get("label_name")) == CLUTTER_LABEL
    ]
    unknown_manifest = [
        record for record in manifest if str(record.get("label_name")) == UNKNOWN_LABEL
    ]
    clutter_provenance = [
        record
        for record in provenance.get("records", [])
        if str(record.get("label_name")) == CLUTTER_LABEL
    ]
    all_provenance_track_ids = {
        int(record["track_id"]) for record in provenance.get("records", [])
    }
    all_provenance_by_capture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in provenance.get("records", []):
        all_provenance_by_capture[str(record["capture_id"])].append(record)
    manifest_outside_v16 = [
        {
            "track_id": int(record["track_id"]),
            "label_name": str(record["label_name"]),
            "is_usable": bool(record.get("is_usable")),
            "unusable_reason": str(record.get("unusable_reason", "")),
        }
        for record in manifest
        if int(record["track_id"]) not in all_provenance_track_ids
    ]

    if len(clutter_manifest) != 50 or len(clutter_provenance) != 50:
        raise ValueError("the frozen CQ08/V16 Clutter population is no longer 50 parents")
    manifest_by_track = {int(record["track_id"]): record for record in clutter_manifest}
    provenance_by_track = {
        int(record["track_id"]): record for record in clutter_provenance
    }
    if set(manifest_by_track) != set(provenance_by_track):
        raise ValueError("CQ08 manifest and V16 provenance Clutter parent sets differ")
    if any(int(record.get("label_id", -1)) != SOURCE_CLUTTER_LABEL_ID for record in clutter_manifest):
        raise ValueError("CQ08 Clutter manifest no longer maps to source label 5")
    if any(int(record.get("label_id", -1)) != SOURCE_UNKNOWN_LABEL_ID for record in unknown_manifest):
        raise ValueError("CQ08 Unknown manifest no longer maps to source label 6")
    if any(not bool(record.get("is_usable")) for record in clutter_manifest):
        raise ValueError("an in-scope source-label-5 Clutter parent is not usable")
    if any(
        f"_{SOURCE_CLUTTER_LABEL_ID}.dat" not in str(record.get("raw_echo_file", ""))
        or f"_{SOURCE_CLUTTER_LABEL_ID}_" not in str(record.get("track_file", ""))
        or f"_{SOURCE_CLUTTER_LABEL_ID}_" not in str(record.get("point_file", ""))
        or any("_DAUR_RD_Clutter_" not in str(path) for path in record.get("rd_mat_files", []))
        for record in clutter_manifest
    ):
        raise ValueError("source-label-5 Clutter filename lineage is inconsistent")
    if any(
        any("_DAUR_RD_Other_" not in str(path) for path in record.get("rd_mat_files", []))
        for record in unknown_manifest
    ):
        raise ValueError("source-label-6 Other filename lineage is inconsistent")

    compact_matlab = re.sub(r"\s+", " ", matlab)
    mapping_5_verified = bool(
        re.search(r"case\s+5\s+target_type\s*=\s*'Clutter'", compact_matlab)
    )
    mapping_6_verified = bool(
        re.search(r"case\s+6\s+target_type\s*=\s*'Other'", compact_matlab)
    )
    capture_date_from_point_file_mtime_verified = bool(
        re.search(r"file_info\s*=\s*dir\(point_path\)", compact_matlab)
        and re.search(
            r"date_str\s*=\s*datestr\(file_info\.datenum\s*,\s*'yyyymmdd'\)",
            compact_matlab,
        )
    )
    capture_time_from_first_point_row_verified = bool(
        re.search(r"first_line\s*=\s*fgetl\(fid_pt\)", compact_matlab)
        and re.search(r"time_str\s*=\s*parts_time\{1\}", compact_matlab)
        and re.search(r"timestamp\s*=\s*\[date_str\s*,\s*time_clean\]", compact_matlab)
    )
    if (
        not mapping_5_verified
        or not mapping_6_verified
        or not capture_date_from_point_file_mtime_verified
        or not capture_time_from_first_point_row_verified
    ):
        raise ValueError("MATLAB label/capture-token lineage could not be verified")

    components = {
        str(component["component_id"]): component
        for component in provenance.get("components", [])
    }
    descriptors, access_counters = _descriptor_rows(source_paths["index"])
    descriptor_counts = Counter(
        str(descriptor["split_name"]) for descriptor in descriptors.values()
    )
    if dict(descriptor_counts) != {"train": 33, "val": 9}:
        raise ValueError(f"unexpected train/val Clutter descriptor counts: {descriptor_counts}")

    cluster_assignments, cluster_profiles, cluster_method = _fit_audit_clusters(
        descriptors
    )
    burst_assignments, burst_rows, burst_summary = _build_bursts(clutter_provenance)

    registry_rows: list[dict[str, Any]] = []
    capture_registry_rows: list[dict[str, Any]] = []
    semantic_template_rows: list[dict[str, Any]] = []
    for track_id in sorted(manifest_by_track):
        source = manifest_by_track[track_id]
        record = provenance_by_track[track_id]
        parent_id = str(record["parent_sample_id"])
        split = str(record["v16_split_name"])
        component = components[str(record["provenance_component_id"])]
        descriptor = descriptors.get(parent_id, {})
        cluster = cluster_assignments.get(
            parent_id,
            {
                "audit_cluster_id": "SEALED_UNASSIGNED",
                "audit_cluster_assignment_scope": "test_metadata_only_sealed",
                "audit_cluster_distance": "",
            },
        )
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "parent_sample_id": parent_id,
            "track_id": track_id,
            "split_name": split,
            "v3_split_name": str(record["v3_split_name"]),
            "source_numeric_label_id": SOURCE_CLUTTER_LABEL_ID,
            "source_matlab_token": "Clutter",
            "current_label_name": CLUTTER_LABEL,
            "capture_id": str(record["capture_id"]),
            "capture_token_date": str(record["capture_id"])[:8],
            "provenance_component_id": str(record["provenance_component_id"]),
            "provenance_component_parent_count": int(component["parent_count"]),
            "provenance_component_labels_json": _json_compact(component["labels"]),
            "provenance_component_cross_label": len(component["labels"]) > 1,
            "moved_from_v3_split": bool(record["moved_from_v3_split"]),
            "manifest_track_length_hint": int(source["track_length_hint"]),
            "manifest_track_row_count": int(source["num_track_rows"]),
            "manifest_point_row_count": int(source["num_point_rows"]),
            "manifest_rd_mat_frame_count": int(source["num_rd_mat_frames"]),
            "manifest_has_track": bool(source["has_track"]),
            "manifest_has_points": bool(source["has_points"]),
            "manifest_has_raw_echo": bool(source["has_raw_echo"]),
            "manifest_has_rd_mat": bool(source["has_rd_mat"]),
            "manifest_is_usable": bool(source["is_usable"]),
            "raw_sha256": str(record["raw_sha256"]),
            "track_sha256": str(record["track_sha256"]),
            "point_sha256": str(record["point_sha256"]),
            "rd_sha256": str(record["rd_sha256"]),
            "model_input_semantics_sha256": str(
                record["model_input_semantics_sha256"]
            ),
            **{field: descriptor.get(field, "") for field in DESCRIPTOR_FIELDS},
            **cluster,
            **{
                f"burst_{threshold:03d}s_id": burst_assignments[parent_id][threshold]
                for threshold in BURST_THRESHOLDS_SECONDS
            },
            "semantic_subtype": "UNADJUDICATED",
            "semantic_review_status": (
                "SEALED_NOT_REVIEWED" if split == "test" else "PENDING_HUMAN_REVIEW"
            ),
            "payload_audit_scope": (
                "metadata_only_sealed"
                if split == "test"
                else "index_train_validation_descriptive_audit"
            ),
        }
        registry_rows.append(row)
        if split in {"train", "val"}:
            semantic_template_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "parent_sample_id": parent_id,
                    "track_id": track_id,
                    "split_name": split,
                    "capture_id": str(record["capture_id"]),
                    "provenance_component_id": str(record["provenance_component_id"]),
                    "audit_cluster_id": cluster["audit_cluster_id"],
                    "candidate_burst_060s_id": burst_assignments[parent_id][60],
                    "confirmed_acquisition_group_id": "",
                    "confirmed_environment_id": "",
                    "confirmed_sensor_mode_id": "",
                    "source_numeric_label_id": SOURCE_CLUTTER_LABEL_ID,
                    "source_matlab_token": "Clutter",
                    "current_label_name": CLUTTER_LABEL,
                    "review_status": "PENDING_HUMAN_REVIEW",
                    "reviewer": "",
                    "reviewed_at_iso8601": "",
                    "acquisition_log_available": "",
                    "operator_note_available": "",
                    "scene_context_available": "",
                    "raw_rd_evidence_reviewed": "",
                    "tr_track_evidence_reviewed": "",
                    "semantic_subtype": "",
                    "secondary_semantic_subtype": "",
                    "subtype_confidence": "",
                    "subtype_evidence_basis": "",
                    "source_label_decision": "",
                    "evidence_references": "",
                    "notes": "",
                }
            )

    for capture_id in sorted({str(row["capture_id"]) for row in registry_rows}):
        clutter_members = [
            row for row in registry_rows if str(row["capture_id"]) == capture_id
        ]
        all_members = sorted(
            all_provenance_by_capture[capture_id], key=lambda row: int(row["track_id"])
        )
        splits = sorted({str(row["v16_split_name"]) for row in all_members})
        if len(splits) != 1:
            raise ValueError(f"V16 capture {capture_id} is not split-pure")
        representative_parent = str(clutter_members[0]["parent_sample_id"])
        capture_registry_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "capture_id": capture_id,
                "capture_token_datetime": _capture_datetime(capture_id).isoformat(),
                "capture_token_date": capture_id[:8],
                "capture_token_provenance": (
                    "batch_process_all.m: point-file modification date + first point HHMMSS"
                ),
                "v16_split_name": splits[0],
                "clutter_parent_count": len(clutter_members),
                "clutter_parent_sample_ids_json": _json_compact(
                    sorted(
                        (str(row["parent_sample_id"]) for row in clutter_members),
                        key=_track_sort_key,
                    )
                ),
                "clutter_track_ids_json": _json_compact(
                    sorted(int(row["track_id"]) for row in clutter_members)
                ),
                "all_v16_parent_count_at_capture": len(all_members),
                "all_v16_parent_sample_ids_json": _json_compact(
                    sorted(
                        (str(row["parent_sample_id"]) for row in all_members),
                        key=_track_sort_key,
                    )
                ),
                "all_v16_track_ids_json": _json_compact(
                    [int(row["track_id"]) for row in all_members]
                ),
                "all_v16_labels_json": _json_compact(
                    sorted({str(row["label_name"]) for row in all_members})
                ),
                "cross_label_capture": len(
                    {str(row["label_name"]) for row in all_members}
                )
                > 1,
                "provenance_component_ids_json": _json_compact(
                    sorted(
                        {str(row["provenance_component_id"]) for row in all_members}
                    )
                ),
                **{
                    f"burst_{threshold:03d}s_id": burst_assignments[
                        representative_parent
                    ][threshold]
                    for threshold in BURST_THRESHOLDS_SECONDS
                },
                "confirmed_acquisition_group_id": "",
                "confirmed_environment_id": "",
                "confirmed_sensor_mode_id": "",
                "capture_review_status": (
                    "SEALED_NOT_REVIEWED"
                    if splits[0] == "test"
                    else "PENDING_HUMAN_REVIEW"
                ),
            }
        )

    registry_rows.sort(key=lambda row: _track_sort_key(str(row["parent_sample_id"])))
    capture_registry_rows.sort(key=lambda row: str(row["capture_id"]))
    semantic_template_rows.sort(
        key=lambda row: (str(row["split_name"]), _track_sort_key(str(row["parent_sample_id"])))
    )
    split_parent_counts = Counter(str(row["split_name"]) for row in registry_rows)
    split_capture_counts = {
        split: len(
            {
                str(row["capture_id"])
                for row in registry_rows
                if row["split_name"] == split
            }
        )
        for split in ("train", "val", "test")
    }
    clutter_components = {
        str(row["provenance_component_id"]): components[
            str(row["provenance_component_id"])
        ]
        for row in clutter_provenance
    }
    cross_label_component_count = sum(
        len(component["labels"]) > 1 for component in clutter_components.values()
    )
    multi_parent_component_count = sum(
        int(component["parent_count"]) > 1 for component in clutter_components.values()
    )
    source_clutter_outside_v16 = [
        row for row in manifest_outside_v16 if row["label_name"] == CLUTTER_LABEL
    ]
    exact_split_purity = split_contract.get("split_purity", {})

    outputs = {
        "parent_registry": PARENT_REGISTRY,
        "capture_registry": CAPTURE_REGISTRY,
        "burst_registry": BURST_REGISTRY,
        "audit_cluster_profiles": AUDIT_CLUSTER_PROFILES,
        "semantic_adjudication_template": SEMANTIC_ADJUDICATION_TEMPLATE,
        "report": REPORT,
    }
    _write_csv(
        _output_path(output_base, PARENT_REGISTRY),
        registry_rows,
        PARENT_REGISTRY_FIELDS,
    )
    _write_csv(
        _output_path(output_base, CAPTURE_REGISTRY),
        capture_registry_rows,
        CAPTURE_REGISTRY_FIELDS,
    )
    _write_csv(
        _output_path(output_base, BURST_REGISTRY),
        burst_rows,
        BURST_REGISTRY_FIELDS,
    )
    _write_csv(
        _output_path(output_base, AUDIT_CLUSTER_PROFILES),
        cluster_profiles,
        AUDIT_CLUSTER_PROFILE_FIELDS,
    )
    _write_csv(
        _output_path(output_base, SEMANTIC_ADJUDICATION_TEMPLATE),
        semantic_template_rows,
        SEMANTIC_TEMPLATE_FIELDS,
    )

    report: dict[str, Any] = {
        "status": "completed_with_collection_and_semantic_adjudication_blockers",
        "schema_version": SCHEMA_VERSION,
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-08-17",
            "verification_status": "VERIFIED",
            "verification_scope": (
                "deterministic local inventory, lineage, split, registry and access-boundary reproduction only"
            ),
            "semantic_ground_truth_status": "UNVERIFIED_PENDING_HUMAN_ADJUDICATION",
            "version_label": "clutter_coverage_semantics_audit_v1",
        },
        "scope": (
            "CQ08 source-label-5 Clutter inventory, capture/provenance coverage, "
            "train-only descriptive audit clusters and label-semantic adjudication readiness"
        ),
        "source_integrity": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "legal_parent_definition": {
            "source": "cq08_unified_manifest",
            "criteria": [
                "source numeric label_id is 5",
                "manifest label_name is ClutterTarget",
                "track, point, original echo and MAT availability flags are true",
                "manifest is_usable is true",
                "parent is represented in the V16 capture/content provenance component index",
            ],
            "overlapping_prefixes_or_windows_count_as_new_parents": False,
        },
        "inventory": {
            "cq08_manifest_parent_count": len(manifest),
            "cq08_manifest_clutter_parent_count": len(clutter_manifest),
            "v16_parent_count": int(provenance["parent_count"]),
            "v16_clutter_parent_count": len(clutter_provenance),
            "source_clutter_parent_outside_v16_count": len(source_clutter_outside_v16),
            "source_clutter_parent_outside_v16": source_clutter_outside_v16,
            "new_legal_clutter_parent_recovered_count": 0,
            "manifest_parent_outside_v16": sorted(
                manifest_outside_v16, key=lambda row: int(row["track_id"])
            ),
            "conclusion": (
                "All 50 legal source-label-5 Clutter parents are already represented in V16; "
                "there is no omitted in-repository Clutter parent to recover."
            ),
        },
        "coverage": {
            "parent_counts_by_split": {
                split: int(split_parent_counts[split])
                for split in ("train", "val", "test")
            },
            "unique_capture_counts_by_split": split_capture_counts,
            "unique_capture_count": len({str(row["capture_id"]) for row in registry_rows}),
            "capture_registry_row_count": len(capture_registry_rows),
            "cross_label_capture_count": sum(
                bool(row["cross_label_capture"]) for row in capture_registry_rows
            ),
            "cross_label_capture_ids": [
                str(row["capture_id"])
                for row in capture_registry_rows
                if bool(row["cross_label_capture"])
            ],
            "cross_label_capture_interpretation": (
                "A shared derived capture token is a provenance grouping fact, not proof that either "
                "source label is wrong; these rows require acquisition-log adjudication."
            ),
            "unique_capture_token_dates": sorted(
                {str(row["capture_token_date"]) for row in registry_rows}
            ),
            "unique_capture_token_date_count": len(
                {str(row["capture_token_date"]) for row in registry_rows}
            ),
            "unique_clutter_provenance_component_count": len(clutter_components),
            "multi_parent_provenance_component_count": multi_parent_component_count,
            "cross_label_provenance_component_count": cross_label_component_count,
            "exact_capture_content_split_purity": exact_split_purity,
            "capture_independence_interpretation": (
                "50 distinct capture_ids and 50 distinct Clutter-containing V16 components do not "
                "establish 50 independent environments. Their common date token was generated from "
                "point-file modification dates, so it is not an authoritative acquisition date."
            ),
            "capture_id_time_provenance": {
                "date_part": "point-track file modification date used by batch_process_all.m",
                "time_part": "first point row HH:MM:SS used by batch_process_all.m",
                "authoritative_acquisition_timestamp": False,
                "date_lineage_verified_from_source": capture_date_from_point_file_mtime_verified,
                "time_lineage_verified_from_source": capture_time_from_first_point_row_verified,
            },
        },
        "candidate_acquisition_bursts": {
            "method": (
                "single-link consecutive derived capture-token grouping at frozen 30/60/120/300/600-second gaps"
            ),
            "semantic_or_session_ground_truth": False,
            "authoritative_time_source": False,
            "thresholds": burst_summary,
            "split_risk": (
                "Several time-adjacent candidate bursts cross V16 splits. Exact capture/hash isolation "
                "still passes, but near-time environment independence requires acquisition-log adjudication."
            ),
        },
        "descriptive_subflow_audit": {
            **cluster_method,
            "profiles": cluster_profiles,
            "train_parent_counts_by_audit_cluster": {
                str(row["audit_cluster_id"]): int(row["train_parent_count"])
                for row in cluster_profiles
            },
            "validation_nearest_train_cluster_counts": {
                str(row["audit_cluster_id"]): int(
                    row["validation_nearest_cluster_parent_count"]
                )
                for row in cluster_profiles
            },
            "validation_observed_train_cluster_count": sum(
                int(row["validation_nearest_cluster_parent_count"]) > 0
                for row in cluster_profiles
            ),
            "train_audit_clusters_without_validation_nearest_member": [
                str(row["audit_cluster_id"])
                for row in cluster_profiles
                if int(row["validation_nearest_cluster_parent_count"]) == 0
            ],
            "train_audit_clusters_below_five_parents": [
                str(row["audit_cluster_id"])
                for row in cluster_profiles
                if int(row["train_parent_count"]) < 5
            ],
            "coverage_conclusion": (
                "The descriptive partition is strongly imbalanced and validation has no nearest "
                "member for two train strata; this is a coverage warning, not a subtype discovery."
            ),
            "interpretation": (
                "audit_cluster_id is an unlabeled coverage stratum only. It must not be renamed as a "
                "physical Clutter subtype, used to change labels, or consumed by model training."
            ),
        },
        "label_semantics": {
            "verified_lineage": [
                "CQ08 source filenames encode numeric label 5",
                "the unified manifest maps source label 5 to ClutterTarget",
                "batch_process_all.m maps case 5 to the MAT filename token Clutter",
                "all 50 source-label-5 manifest records preserve that filename/token lineage",
            ],
            "source_label_5_to_matlab_clutter_token_verified": mapping_5_verified,
            "source_label_6_to_matlab_other_token_verified": mapping_6_verified,
            "independent_human_semantic_ground_truth_available": False,
            "scene_or_environment_annotation_available": False,
            "clutter_mechanism_annotation_available": False,
            "allowed_current_claim": (
                "ClutterTarget means CQ08 source numeric label 5 with the inherited Clutter token."
            ),
            "forbidden_unverified_claims": [
                "source label 5 is known to be stationary ground clutter",
                "source label 5 is known to be vegetation, weather, multipath or sidelobe clutter",
                "an audit cluster is a true physical subtype",
                "source label 6/Other is intrinsically a validated open-set rejection population",
            ],
            "unknown_policy": {
                "source_lineage": "CQ08 numeric label 6 was named Other by batch_process_all.m",
                "v21_product_semantics": "Unknown is strict open-set outcome/evaluation truth, not a KnownGate class",
                "used_in_this_clutter_cluster_fit": False,
                "used_in_this_semantic_template": False,
                "physical_payload_access_count": 0,
            },
        },
        "semantic_adjudication": {
            "protocol": ADJUDICATION_PROTOCOL.as_posix(),
            "template": SEMANTIC_ADJUDICATION_TEMPLATE.as_posix(),
            "capture_registry": CAPTURE_REGISTRY.as_posix(),
            "template_parent_count": len(semantic_template_rows),
            "template_parent_counts_by_split": {
                split: sum(row["split_name"] == split for row in semantic_template_rows)
                for split in ("train", "val", "test")
            },
            "test_rows_excluded": True,
            "all_subtypes_initially_unadjudicated": True,
            "automatic_label_changes": 0,
            "label_change_requires_separate_authorization_and_version": True,
        },
        "collection_gap": {
            "current_verified_independent_acquisition_dates": 0,
            "observed_capture_token_date_count": 1,
            "capture_token_date_is_authoritative": False,
            "current_verified_environment_groups": 0,
            "current_verified_sensor_mode_groups": 0,
            "current_human_adjudicated_clutter_mechanisms": 0,
            "provisional_planning_floor_not_power_analysis": {
                "independent_acquisition_dates": {
                    "target_minimum": 3,
                    "current_verified": 0,
                    "minimum_additional_or_source_log_confirmation_required": 3,
                },
                "environment_groups": {
                    "target_minimum": 3,
                    "current_verified": 0,
                    "minimum_additional_or_annotation_required": 3,
                },
                "sensor_modes": (
                    "inventory every deployment mode; numeric gap is not computable from current metadata"
                ),
                "per_human_confirmed_mechanism": (
                    "provisionally seek at least 10 independent train parents and 5 independent "
                    "development parents across at least two dates before mechanism-level claims"
                ),
            },
            "parent_count_gap": (
                "not computable until environment/mode inventory and human subtype adjudication are complete"
            ),
            "legal_way_to_increase_support": "new CQ08-compatible acquisition with provenance and annotation",
            "forbidden_shortcuts": [
                "duplicating parents or overlapping prefixes/windows",
                "moving validation/test parents into training",
                "relabeling difficult known targets as Clutter",
                "using model errors or audit clusters as automatic labels",
            ],
        },
        "access_and_leakage": {
            **access_counters,
            "test_parent_registry_source": "existing manifest and V16 provenance metadata",
            "test_physical_descriptors_emitted": False,
            "test_cluster_assignment_performed": False,
            "validation_used_to_fit_clusters": False,
            "validation_used_for_threshold_or_model_selection": False,
        },
        "methodological_risk_scan": {
            "overall_confidence": "CAUTION",
            "fallacy_coverage": "11/11",
            "items": [
                {
                    "name": "Simpson's paradox",
                    "status": "CAUTION",
                    "detail": "Aggregate Clutter coverage must not replace split, burst, capture or audit-stratum views.",
                },
                {
                    "name": "Ecological fallacy",
                    "status": "CAUTION",
                    "detail": "Capture/burst/audit-cluster membership cannot establish an individual parent physical mechanism.",
                },
                {
                    "name": "Berkson's paradox",
                    "status": "NOTE",
                    "detail": "The registry is conditioned on usable V16 parents, but no source-label-5 Clutter parent is excluded.",
                },
                {
                    "name": "Collider bias",
                    "status": "NOTE",
                    "detail": "No adjusted association or causal model is fitted in this audit.",
                },
                {
                    "name": "Base rate neglect",
                    "status": "CAUTION",
                    "detail": "Clutter support is explicitly only 33/9/8 parents; capture and prefix counts cannot replace parent support.",
                },
                {
                    "name": "Regression to the mean",
                    "status": "NOTE",
                    "detail": "No pre/post or extreme-outcome-selected performance comparison is made.",
                },
                {
                    "name": "Survivorship bias",
                    "status": "NOTE",
                    "detail": "All 50 manifest source-label-5 Clutter parents are retained; none is dropped for difficulty or quality.",
                },
                {
                    "name": "Look-elsewhere effect",
                    "status": "CAUTION",
                    "detail": "Five derived-time burst thresholds are exploratory registries and carry no inferential or semantic claim.",
                },
                {
                    "name": "Garden of forking paths",
                    "status": "CAUTION",
                    "detail": "Four descriptive clusters and all burst thresholds are frozen in code and forbidden from model/performance selection.",
                },
                {
                    "name": "Correlation is not causation",
                    "status": "CAUTION",
                    "detail": "TR/RD descriptor similarity does not identify the cause of a Clutter return.",
                },
                {
                    "name": "Reverse causality",
                    "status": "NOTE",
                    "detail": "No directional causal claim is made from labels or descriptors.",
                },
            ],
        },
        "checks": {
            "all_50_source_clutter_parents_are_in_v16": len(source_clutter_outside_v16)
            == 0,
            "clutter_parent_capture_count_is_50": len(
                {str(row["capture_id"]) for row in registry_rows}
            )
            == 50,
            "capture_registry_has_exactly_50_unique_rows": len(capture_registry_rows)
            == 50
            and len({str(row["capture_id"]) for row in capture_registry_rows}) == 50,
            "clutter_split_counts_are_33_9_8": dict(split_parent_counts)
            == {"train": 33, "val": 9, "test": 8},
            "exact_capture_content_split_keys_are_pure": bool(
                exact_split_purity.get("all_checked_keys_zero")
            ),
            "derived_capture_token_lineage_is_source_verified": (
                capture_date_from_point_file_mtime_verified
                and capture_time_from_first_point_row_verified
            ),
            "train_only_cluster_fit_has_33_parents": cluster_method["fit_parent_count"]
            == 33,
            "validation_nearest_cluster_assignment_has_9_parents": cluster_method[
                "validation_parent_count"
            ]
            == 9,
            "test_is_sealed_and_unassigned": all(
                row["audit_cluster_id"] == "SEALED_UNASSIGNED"
                and row["payload_audit_scope"] == "metadata_only_sealed"
                for row in registry_rows
                if row["split_name"] == "test"
            ),
            "semantic_template_excludes_test": all(
                row["split_name"] != "test" for row in semantic_template_rows
            ),
            "unknown_is_absent_from_clutter_cluster_and_template": True,
            "automatic_label_change_count_is_zero": True,
        },
        "outputs": {
            name: {
                "path": relative.as_posix(),
                "sha256": _sha256(_output_path(output_base, relative)),
            }
            for name, relative in outputs.items()
            if name != "report"
        },
        "training": "未执行",
        "OOF": "未执行",
        "test_model_evaluation": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    if not all(bool(value) for value in report["checks"].values()):
        raise AssertionError(f"Clutter audit invariant failed: {report['checks']}")
    _write_json(_output_path(output_base, REPORT), report)
    return {
        **report,
        "outputs": {
            **report["outputs"],
            "report": {
                "path": REPORT.as_posix(),
                "sha256": _sha256(_output_path(output_base, REPORT)),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all legal CQ08 Clutter parents, capture/provenance coverage and "
            "unadjudicated descriptive subflows without touching sealed test payloads."
        )
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    result = build(args.project_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
