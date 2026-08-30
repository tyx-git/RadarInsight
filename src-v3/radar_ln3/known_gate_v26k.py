"""Immutable protocol and numerical helpers for the V26k L2 knownness audit.

V26k is intentionally inference-only.  It reloads the frozen V26c DBB
specialist checkpoint for each fold, fits a class-conditional reference on
that fold's DBB train parents only, and audits a distance-based novelty score
on the already frozen V26f mixed validation/test population.

The score is deliberately kept outside the product graph.  UnknownTarget is
not used for model fitting, normalization, reference fitting, threshold
fitting, checkpoint selection, or any other choice.  It is attached only
after the score and its train-only diagnostic operating point are frozen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .known_gate_v26c_oof import (
    CONTRACT as V26C_CONTRACT,
    CONTRACT_SCHEMA as V26C_CONTRACT_SCHEMA,
    FOLD_COUNT,
    L2_CHECKPOINT_SCHEMA,
    L2_EPOCH,
    READINESS as V26C_READINESS,
    READINESS_SCHEMA as V26C_READINESS_SCHEMA,
    fold_record,
    l2_checkpoint_path,
    l2_config_path,
    l2_data_contract_path,
    l2_held_out_micro_cache_path,
    l2_held_out_physical_cache_path,
    l2_normalization_path,
    l2_train_micro_cache_path,
    l2_train_physical_cache_path,
    sequence_sha256,
)
from .known_gate_v26f import (
    CONTRACT as V26F_CONTRACT,
    CONTRACT_SCHEMA as V26F_CONTRACT_SCHEMA,
    DBB_LABELS,
    EXPECTED_PARENT_COUNTS,
    FEATURE_CACHE_MAX_BYTES,
    INDEX as V26F_INDEX,
    L1_LABELS,
    L1_TEST_CACHE,
    L1_VALIDATION_CACHE,
    PARENT_DETAIL_CSV as V26F_PARENT_DETAIL_CSV,
    READINESS as V26F_READINESS,
    READINESS_SCHEMA as V26F_READINESS_SCHEMA,
    REPORT as V26F_REPORT,
    REPORT_SCHEMA as V26F_REPORT_SCHEMA,
    bound_path as v26f_bound_path,
    fold_contract_entry,
    require_bound_file as v26f_require_bound_file,
    selected_event_sample_ids,
    sha256_file as v26f_sha256_file,
    validate_feature_cache,
    validate_prepared_contract as validate_v26f_prepared_contract,
)


CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26k-cq08-l2-train-only-mahalanobis-knownness-diagnostic-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic.json"
)
DETAIL_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-parent-details.csv"
)
DOCUMENT = Path("src-v3/docs/radar_ln3_v26k_l2_train_only_mahalanobis_knownness_diagnostic.md")

CONTRACT_SCHEMA = "RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-readiness-v1"
REPORT_SCHEMA = "RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-v1"

INDEX_PATH = V26F_INDEX
UNKNOWN_BASE_LABEL = "UnknownTarget"
MIXED_LABELS = (*L1_LABELS, UNKNOWN_BASE_LABEL)
L1_TARGET_ROUTE = "KNOWN_TARGET_EVIDENCE"
L1_CLUTTER_ROUTE = "KNOWN_CLUTTER_EVIDENCE"
L2_CLASSES = tuple(DBB_LABELS)
REPRESENTATION_NAME = "classifier_layer_norm_liquid_final_state"
SCORE_NAME = "minimum_class_conditional_pooled_diagonal_mahalanobis_distance"
SCORE_ORIENTATION = "higher_means_less_like_train_known"
VARIANCE_FLOOR_MULTIPLIER = 1.0e-6
VARIANCE_FLOOR_ABSOLUTE = 1.0e-10
KNOWN_RETENTION_TARGET = 0.95
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 26069

MAX_JSON_SOURCE_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_FEATURE_CACHE_BYTES = 64 * 1024 * 1024
MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_READINESS_BYTES = 512 * 1024
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_DETAIL_BYTES = 4 * 1024 * 1024
MAX_ISOLATED_FOLD_OUTPUT_BYTES = 3 * 1024 * 1024


OOD_SCORE_POLICY: dict[str, Any] = {
    "score_name": SCORE_NAME,
    "score_orientation": SCORE_ORIENTATION,
    "representation": REPRESENTATION_NAME,
    "representation_dimension": 48,
    "reference_fit": "per_fold_train_known_only_parent_final_states",
    "class_conditional_references": list(L2_CLASSES),
    "covariance": "pooled_diagonal_within_class_variance",
    "variance_floor_multiplier_of_median_positive_variance": VARIANCE_FLOOR_MULTIPLIER,
    "variance_floor_absolute": VARIANCE_FLOOR_ABSOLUTE,
    "unknown_in_model": False,
    "unknown_in_normalization": False,
    "unknown_in_reference_fit": False,
    "unknown_in_threshold_fit": False,
    "unknown_in_checkpoint_or_model_selection": False,
    "threshold_policy": "train_known_empirical_quantile_for_95_percent_retention",
    "threshold_is_product_calibration": False,
    "score_fusion_with_msp": False,
    "cross_fold_state_or_score_pooling": False,
    "semantic_action_mutation": False,
}

IMPLEMENTATION_FILES = (
    Path("src-v3/radar_ln3/known_gate_v26k.py"),
    Path("src-v3/build/known_gate_v26k.py"),
    Path("src-v3/report/known_gate_v26k.py"),
    Path("src-v3/smoke/known_gate_v26k_contract_test.py"),
    Path("src-v3/scripts/v26k_known_gate_l2_train_only_mahalanobis_diagnostic_pipeline.ps1"),
    DOCUMENT,
)

DETAIL_FIELDS = (
    "split",
    "fold_id",
    "parent_sample_id",
    "track_id",
    "capture_id",
    "provenance_component_id",
    "label_name",
    "is_unknown_truth",
    "l1_route",
    "l2_predicted_class",
    "l2_subclass_probability",
    "l2_subclass_margin",
    "closest_reference_class",
    "l2_knownness_score",
    "max_softmax_unknownness",
    "known_target_accepted",
    "known_clutter_accepted",
    "l2_gate_accepted",
    "combined_known_target_accepted",
    "l2_forwarded",
    "accepted_by_train_only_threshold",
    "parent_length",
    "rd_present_at_final",
    "micro_doppler_valid_at_final",
    "dt_valid_at_final",
)


def parse_bool_field(value: Any, *, field: str) -> bool:
    """Parse a serialized boolean without silently coercing malformed values."""

    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"V26k {field} is not a serialized boolean")


def resolve_path(root: Path, value: str | Path) -> Path:
    raw = Path(str(value).strip().replace("\\", "/"))
    return (raw if raw.is_absolute() else root / raw).resolve()


def relative_path(root: Path, value: str | Path) -> str:
    root_path = Path(root).resolve()
    path = resolve_path(root_path, value)
    try:
        return path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"V26k path escapes the project root: {value}") from exc


def read_json(
    path: Path,
    *,
    description: str,
    max_bytes: int = MAX_JSON_SOURCE_BYTES,
) -> dict[str, Any]:
    checked = Path(path)
    if not checked.is_file():
        raise FileNotFoundError(f"V26k {description} is missing: {checked}")
    if int(checked.stat().st_size) > int(max_bytes):
        raise MemoryError(f"V26k {description} exceeds its bound: {checked}")
    payload = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"V26k {description} must be a JSON object: {checked}")
    return payload


def require_bound_file(
    root: Path,
    value: str | Path,
    description: str,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = MAX_JSON_SOURCE_BYTES,
) -> Path:
    path = resolve_path(root, value)
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError(f"V26k {description} escapes the project root: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"V26k {description} is missing: {path}")
    if int(path.stat().st_size) > int(max_bytes):
        raise MemoryError(f"V26k {description} exceeds its bound: {path}")
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"V26k {description} hash differs: {path}")
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable_text(path: Path, text: str, *, max_bytes: int, description: str) -> str:
    if len(text.encode("utf-8")) > int(max_bytes):
        raise MemoryError(f"V26k {description} exceeds its bound: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V26k {description}: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() and temporary.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to replace a different V26k temporary {description}: {temporary}")
    if not temporary.exists():
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    max_bytes: int,
    description: str,
) -> str:
    return write_immutable_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        max_bytes=max_bytes,
        description=description,
    )


def source_binding(
    root: Path,
    value: str | Path,
    *,
    description: str,
    expected_sha256: str | None = None,
    max_bytes: int = MAX_JSON_SOURCE_BYTES,
) -> dict[str, Any]:
    path = require_bound_file(
        root,
        value,
        description,
        expected_sha256=expected_sha256,
        max_bytes=max_bytes,
    )
    return {
        "path": relative_path(root, path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def implementation_sha256(root: Path) -> dict[str, str]:
    return {
        relative_path(root, path): sha256_file(
            require_bound_file(root, path, f"implementation {path}")
        )
        for path in IMPLEMENTATION_FILES
    }


def state_matrix_sha256(states: np.ndarray) -> str:
    value = np.asarray(states, dtype="<f8")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1 or not np.isfinite(value).all():
        raise ValueError("V26k state hash requires a finite non-empty matrix")
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def parent_id_sequence_sha256(parent_ids: Sequence[str]) -> str:
    values = tuple(str(value) for value in parent_ids)
    if not values or len(values) != len(set(values)):
        raise ValueError("V26k parent identifiers must be non-empty and unique")
    return sequence_sha256(values)


def _finite_matrix(states: np.ndarray, *, description: str) -> np.ndarray:
    value = np.asarray(states, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1 or not np.isfinite(value).all():
        raise ValueError(f"V26k {description} must be a finite non-empty matrix")
    return value


def fit_class_conditional_reference(
    states: np.ndarray,
    class_ids: np.ndarray,
) -> dict[str, Any]:
    """Fit three class centres and one pooled diagonal within-class variance."""

    array = _finite_matrix(states, description="reference states")
    labels = np.asarray(class_ids, dtype=np.int64)
    if labels.shape != (array.shape[0],) or not np.isin(labels, np.arange(len(L2_CLASSES))).all():
        raise ValueError("V26k reference states and class ids are not aligned")
    centres: list[np.ndarray] = []
    residual_sum = np.zeros(array.shape[1], dtype=np.float64)
    class_counts: dict[str, int] = {}
    total = 0
    for class_id, name in enumerate(L2_CLASSES):
        selected = array[labels == class_id]
        if selected.shape[0] < 2:
            raise ValueError(f"V26k reference has inadequate train support for {name}")
        centre = selected.mean(axis=0)
        centres.append(centre)
        residual_sum += np.square(selected - centre).sum(axis=0)
        class_counts[name] = int(selected.shape[0])
        total += int(selected.shape[0])
    degrees_of_freedom = total - len(L2_CLASSES)
    if degrees_of_freedom <= 0:
        raise ValueError("V26k pooled diagonal variance has no degrees of freedom")
    pooled = residual_sum / float(degrees_of_freedom)
    positive = pooled[pooled > 0.0]
    variance_floor = max(
        (float(np.median(positive)) if positive.size else 1.0) * VARIANCE_FLOOR_MULTIPLIER,
        VARIANCE_FLOOR_ABSOLUTE,
    )
    pooled = np.maximum(pooled, variance_floor)
    centres_array = np.stack(centres, axis=0)
    if (
        centres_array.shape != (len(L2_CLASSES), array.shape[1])
        or pooled.shape != (array.shape[1],)
        or not np.isfinite(centres_array).all()
        or not np.isfinite(pooled).all()
        or bool((pooled <= 0.0).any())
    ):
        raise AssertionError("V26k fitted reference violates its finite diagonal contract")
    return {
        "centres": centres_array,
        "pooled_variance": pooled,
        "variance_floor": float(variance_floor),
        "class_counts": class_counts,
        "train_parent_count": int(total),
        "pooled_degrees_of_freedom": int(degrees_of_freedom),
        "state_dimension": int(array.shape[1]),
    }


def mahalanobis_scores(
    states: np.ndarray,
    fitted: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimum novelty distance and its closest DBB class."""

    array = _finite_matrix(states, description="scoring states")
    centres = np.asarray(fitted["centres"], dtype=np.float64)
    variance = np.asarray(fitted["pooled_variance"], dtype=np.float64)
    if (
        centres.shape != (len(L2_CLASSES), array.shape[1])
        or variance.shape != (array.shape[1],)
        or not np.isfinite(centres).all()
        or not np.isfinite(variance).all()
        or bool((variance <= 0.0).any())
    ):
        raise ValueError("V26k Mahalanobis reference shape or values are invalid")
    squared = np.square(array[:, None, :] - centres[None, :, :]) / variance[None, None, :]
    distances = np.sqrt(squared.sum(axis=2))
    closest = distances.argmin(axis=1).astype(np.int64, copy=False)
    score = distances[np.arange(array.shape[0]), closest]
    if score.shape != (array.shape[0],) or not np.isfinite(score).all():
        raise AssertionError("V26k Mahalanobis scores are invalid")
    return score, closest


def binary_ranking_metrics(
    known_scores: Sequence[float],
    unknown_scores: Sequence[float],
) -> dict[str, Any] | None:
    """Compute threshold-free ranking metrics, with Unknown as the positive."""

    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)
    if known.ndim != 1 or unknown.ndim != 1 or not len(known) or not len(unknown):
        return None
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ValueError("V26k ranking scores must be finite")
    comparisons = unknown[:, None] - known[None, :]
    auc = float((comparisons > 0.0).mean() + 0.5 * (comparisons == 0.0).mean())
    scores = np.concatenate([unknown, known])
    labels = np.concatenate(
        [np.ones(len(unknown), dtype=np.int64), np.zeros(len(known), dtype=np.int64)]
    )
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_count = int(sorted_labels.sum())
    precision_terms: list[float] = []
    start = 0
    positives_before = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        group_positive = int(group.sum())
        if group_positive:
            precision_at_group = float((positives_before + group_positive) / end)
            precision_terms.extend([precision_at_group] * group_positive)
        positives_before += group_positive
        start = end
    average_precision = float(sum(precision_terms) / positive_count)
    return {
        "auroc_unknown_vs_known": auc,
        "average_precision_unknown": average_precision,
        "known_support": int(len(known)),
        "unknown_support": int(len(unknown)),
        "unknown_prevalence": float(len(unknown) / (len(known) + len(unknown))),
        "threshold_fitting": "not_used_for_ranking",
    }


def train_only_retention_threshold(
    scores: Sequence[float],
    *,
    target_retention: float = KNOWN_RETENTION_TARGET,
) -> dict[str, Any]:
    """Choose an empirical train-only threshold retaining at least the target."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("V26k threshold fitting needs a finite non-empty score vector")
    target = float(target_retention)
    if not 0.0 < target <= 1.0:
        raise ValueError("V26k target retention must be in (0,1]")
    required = max(1, int(np.ceil(target * len(values))))
    ordered = np.sort(values, kind="mergesort")
    threshold = float(ordered[required - 1])
    accepted = values <= threshold + 1.0e-12
    return {
        "target_retention": target,
        "threshold": threshold,
        "train_support": int(len(values)),
        "train_accepted_count": int(accepted.sum()),
        "train_rejected_count": int((~accepted).sum()),
        "train_retention": float(accepted.mean()),
        "threshold_fit_population": "per_fold_train_known_only_parent_final",
        "unknown_threshold_fit_exposure": 0,
    }


def value_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("V26k value summary needs a finite non-empty vector")
    return {
        "support": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def class_summary(rows: Sequence[Mapping[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label_name"])].append(float(row[score_key]))
    result: list[dict[str, Any]] = []
    for label in (*L2_CLASSES, "ClutterTarget", UNKNOWN_BASE_LABEL):
        values = grouped.get(label, [])
        if values:
            result.append({"label_name": label, **value_summary(values)})
    return result


def route_key(split: str, fold_id: int, parent_id: str) -> tuple[str, int, str]:
    split_value = str(split)
    fold_value = int(fold_id)
    parent_value = str(parent_id)
    if split_value not in {"val", "test"} or fold_value not in range(FOLD_COUNT) or not parent_value:
        raise ValueError("V26k route key is malformed")
    return split_value, fold_value, parent_value


def route_key_sequence_sha256(keys: Sequence[tuple[str, int, str]]) -> str:
    values = [f"{split}|{fold}|{parent}" for split, fold, parent in keys]
    return sequence_sha256(values)


def _index_parent_metadata(root: Path, *, split: str) -> dict[str, dict[str, str]]:
    """Read only parent identity metadata from the bound CQ08 index."""

    path = require_bound_file(root, INDEX_PATH, "CQ08 V16 source index", max_bytes=MAX_JSON_SOURCE_BYTES)
    result: dict[str, dict[str, str]] = {}
    event_indices: dict[str, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "schema_version",
            "source_name",
            "split_name",
            "parent_sample_id",
            "label_name",
            "track_id",
            "capture_id",
            "provenance_component_id",
            "causal_event_index",
        }
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("V26k CQ08 index is missing parent identity fields")
        for raw in reader:
            if str(raw.get("split_name")) != str(split):
                continue
            if str(raw.get("schema_version")) != "RADAR-LN3-V16" or str(raw.get("source_name")) != "CQ08":
                raise ValueError("V26k source index is outside the frozen CQ08 V16 contract")
            parent = str(raw.get("parent_sample_id") or "")
            try:
                event_index = int(raw.get("causal_event_index"))
            except (TypeError, ValueError) as exc:
                raise ValueError("V26k source index has an invalid causal event index") from exc
            if not parent or event_index < 0:
                raise ValueError("V26k source index has an invalid parent/event identity")
            identity = {
                "label_name": str(raw.get("label_name") or ""),
                "track_id": str(raw.get("track_id") or ""),
                "capture_id": str(raw.get("capture_id") or ""),
                "provenance_component_id": str(raw.get("provenance_component_id") or ""),
            }
            if any(not value for value in identity.values()):
                raise ValueError("V26k source index has an empty parent identity field")
            existing = result.get(parent)
            if existing is None:
                result[parent] = {**identity, "_event_count": "1", "_last_event": str(event_index)}
            else:
                if any(existing[key] != identity[key] for key in identity):
                    raise ValueError(f"V26k parent identity changes within its sequence: {parent}")
                existing["_event_count"] = str(int(existing["_event_count"]) + 1)
                existing["_last_event"] = str(event_index)
            event_indices[parent].append(event_index)
    for parent, metadata in result.items():
        count = int(metadata.pop("_event_count"))
        metadata.pop("_last_event")
        observed = sorted(event_indices[parent])
        if observed != list(range(count)):
            raise ValueError(f"V26k parent event indices are not contiguous: {parent}")
    return result


def selected_event_ids_for_parents(
    root: Path,
    *,
    split: str,
    parent_ids: Sequence[str],
) -> list[str]:
    """Reconstruct the exact cache order for a requested parent population."""

    requested = {str(value) for value in parent_ids}
    if not requested or len(requested) != len(parent_ids):
        raise ValueError("V26k requested parent ids must be non-empty and unique")
    path = require_bound_file(root, INDEX_PATH, "V26k source index", max_bytes=MAX_JSON_SOURCE_BYTES)
    rows: list[tuple[str, int, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("split_name")) != str(split) or str(raw.get("parent_sample_id")) not in requested:
                continue
            if str(raw.get("schema_version")) != "RADAR-LN3-V16" or str(raw.get("source_name")) != "CQ08":
                raise ValueError("V26k selected event source is outside CQ08 V16")
            sample_id = str(raw.get("sample_id") or "")
            if not sample_id:
                raise ValueError("V26k source index has an empty sample id")
            rows.append((str(raw["parent_sample_id"]), int(raw["causal_event_index"]), sample_id))
    rows.sort(key=lambda value: (value[0], value[1]))
    observed_parents = {parent for parent, _, _ in rows}
    if observed_parents != requested or len({sample for _, _, sample in rows}) != len(rows):
        raise ValueError("V26k selected event ids do not cover the requested parents")
    grouped: dict[str, list[int]] = defaultdict(list)
    for parent, event_index, _ in rows:
        grouped[parent].append(event_index)
    if any(events != list(range(len(events))) for events in grouped.values()):
        raise ValueError("V26k selected parent event indices are not contiguous")
    return [sample for _, _, sample in rows]


def _file_binding_from_field(
    root: Path,
    *,
    path: str | Path,
    expected_sha256: str,
    description: str,
    max_bytes: int,
) -> dict[str, Any]:
    checked = require_bound_file(
        root,
        path,
        description,
        expected_sha256=expected_sha256,
        max_bytes=max_bytes,
    )
    return {
        "path": relative_path(root, checked),
        "sha256": sha256_file(checked),
        "size_bytes": int(checked.stat().st_size),
    }


def validate_v26c_source(root: Path) -> dict[str, Any]:
    """Validate the frozen V26c L2 train-only fold source chain."""

    contract_path = root / V26C_CONTRACT
    readiness_path = root / V26C_READINESS
    contract = read_json(contract_path, description="V26c contract")
    readiness = read_json(readiness_path, description="V26c readiness")
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != V26C_CONTRACT_SCHEMA
        or readiness.get("schema_version") != V26C_READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or int(dict(contract.get("l2") or {}).get("fixed_snapshot_epoch", -1)) != L2_EPOCH
        or int(dict(contract.get("l2") or {}).get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V26k V26c source is stale or outside its Known-only freeze")

    train_metadata = _index_parent_metadata(root, split="train")
    folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        binding = dict(fold.get("l2") or {})
        train_ids = [str(value) for value in binding.get("train_parent_ids") or ()]
        held_out_ids = [str(value) for value in binding.get("held_out_parent_ids") or ()]
        if not train_ids or not held_out_ids or set(train_ids) & set(held_out_ids):
            raise ValueError(f"V26k V26c L2 fold {fold_id} has invalid parent partition")
        if not set(train_ids).issubset(set(fold["train_parent_ids"])):
            raise ValueError(f"V26k V26c L2 train parents escape the fold train partition: {fold_id}")
        if not set(held_out_ids).issubset(set(fold["held_out_parent_ids"])):
            raise ValueError(f"V26k V26c L2 held-out parents escape the fold held-out partition: {fold_id}")
        for parent_id in (*train_ids, *held_out_ids):
            if parent_id not in train_metadata or train_metadata[parent_id]["label_name"] not in L2_CLASSES:
                raise ValueError(f"V26k V26c L2 parent is not a train DBB parent: {parent_id}")
        expected_train_counts = {
            label: sum(train_metadata[parent]["label_name"] == label for parent in train_ids)
            for label in L2_CLASSES
        }
        expected_held_counts = {
            label: sum(train_metadata[parent]["label_name"] == label for parent in held_out_ids)
            for label in L2_CLASSES
        }
        if dict(binding.get("train_label_parent_counts") or {}) != expected_train_counts:
            raise ValueError(f"V26k V26c L2 train label support differs in fold {fold_id}")
        if dict(binding.get("held_out_label_parent_counts") or {}) != expected_held_counts:
            raise ValueError(f"V26k V26c L2 held-out label support differs in fold {fold_id}")
        if (
            int(binding.get("train_parent_count", -1)) != len(train_ids)
            or int(binding.get("held_out_parent_count", -1)) != len(held_out_ids)
            or int(binding.get("fixed_snapshot_epoch", -1)) != L2_EPOCH
        ):
            raise ValueError(f"V26k V26c L2 fold counts or epoch differ: {fold_id}")

        config_path = root / l2_config_path(fold_id)
        data_contract_path = root / l2_data_contract_path(fold_id)
        normalization_path = root / l2_normalization_path(fold_id)
        checkpoint_path = root / l2_checkpoint_path(fold_id)
        field_paths = {
            "config": (config_path, str(binding.get("config_sha256")), MAX_JSON_SOURCE_BYTES),
            "data_contract": (data_contract_path, str(binding.get("data_contract_sha256")), MAX_JSON_SOURCE_BYTES),
            "normalization": (normalization_path, str(binding.get("normalization_sha256")), MAX_JSON_SOURCE_BYTES),
            "checkpoint": (checkpoint_path, "", MAX_CHECKPOINT_BYTES),
        }
        files: dict[str, dict[str, Any]] = {}
        for name, (path, expected_hash, max_bytes) in field_paths.items():
            if name != "checkpoint" and len(expected_hash) != 64:
                raise ValueError(f"V26k V26c L2 fold {fold_id} has no {name} hash")
            files[name] = _file_binding_from_field(
                root,
                path=path,
                expected_sha256=expected_hash if expected_hash else None,
                description=f"V26c L2 fold {fold_id} {name}",
                max_bytes=max_bytes,
            )

        train_event_ids = selected_event_ids_for_parents(
            root, split="train", parent_ids=train_ids
        )
        held_event_ids = selected_event_ids_for_parents(
            root, split="train", parent_ids=held_out_ids
        )
        cache_specs = (
            (
                "train_rd_physical",
                root / l2_train_physical_cache_path(fold_id),
                dict(binding.get("train_rd_physical_cache") or {}),
                train_event_ids,
                4,
            ),
            (
                "train_micro_doppler",
                root / l2_train_micro_cache_path(fold_id),
                dict(binding.get("train_micro_doppler_cache") or {}),
                train_event_ids,
                8,
            ),
            (
                "held_out_rd_physical",
                root / l2_held_out_physical_cache_path(fold_id),
                dict(binding.get("held_out_rd_physical_cache") or {}),
                held_event_ids,
                4,
            ),
            (
                "held_out_micro_doppler",
                root / l2_held_out_micro_cache_path(fold_id),
                dict(binding.get("held_out_micro_doppler_cache") or {}),
                held_event_ids,
                8,
            ),
        )
        cache_bindings: dict[str, dict[str, Any]] = {}
        for name, path, binding_value, event_ids, width in cache_specs:
            expected_hash = str(binding_value.get("sha256") or "")
            if len(expected_hash) != 64 or str(binding_value.get("path")) != relative_path(root, path):
                raise ValueError(f"V26k V26c L2 {name} binding differs in fold {fold_id}")
            checked = require_bound_file(
                root,
                path,
                f"V26c L2 fold {fold_id} {name} cache",
                expected_sha256=expected_hash,
                max_bytes=MAX_FEATURE_CACHE_BYTES,
            )
            actual = validate_feature_cache(
                checked,
                expected_sample_ids=event_ids,
                width=width,
                valid_rank=1,
                description=f"V26c L2 fold {fold_id} {name}",
            )
            cache_bindings[name] = {
                "path": relative_path(root, checked),
                "sha256": actual["sha256"],
                "size_bytes": int(checked.stat().st_size),
                "event_count": int(actual["event_count"]),
                "feature_width": int(width),
                "selected_sample_ids_sha256": sequence_sha256(event_ids),
            }
        folds.append(
            {
                "fold_id": int(fold_id),
                "train_parent_count": len(train_ids),
                "held_out_parent_count": len(held_out_ids),
                "train_parent_ids_sha256": parent_id_sequence_sha256(train_ids),
                "held_out_parent_ids_sha256": parent_id_sequence_sha256(held_out_ids),
                "train_label_parent_counts": expected_train_counts,
                "held_out_label_parent_counts": expected_held_counts,
                "fixed_snapshot_epoch": int(L2_EPOCH),
                "files": files,
                "caches": cache_bindings,
            }
        )
    return {
        "v26c_contract_sha256": sha256_file(contract_path),
        "v26c_readiness_sha256": sha256_file(readiness_path),
        "v26c_contract_schema": V26C_CONTRACT_SCHEMA,
        "v26c_readiness_schema": V26C_READINESS_SCHEMA,
        "fixed_snapshot_epoch": int(L2_EPOCH),
        "folds": folds,
    }


def _read_route_rows(root: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    path = require_bound_file(root, V26F_PARENT_DETAIL_CSV, "V26f parent detail CSV", max_bytes=MAX_DETAIL_BYTES)
    required = {
        "split",
        "fold_id",
        "parent_sample_id",
        "label_name",
        "track_id",
        "capture_id",
        "provenance_component_id",
        "l1_route",
        "known_target_accepted",
        "known_clutter_accepted",
    }
    result: dict[tuple[str, int, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("V26f parent detail CSV lacks L1 route identity fields")
        for raw in reader:
            key = route_key(raw.get("split", ""), int(raw.get("fold_id", -1)), raw.get("parent_sample_id", ""))
            if key in result:
                raise ValueError(f"V26k V26f parent detail repeats a key: {key}")
            label = str(raw.get("label_name") or "")
            route = str(raw.get("l1_route") or "")
            if label not in MIXED_LABELS or route not in {L1_TARGET_ROUTE, L1_CLUTTER_ROUTE}:
                raise ValueError(f"V26k V26f route row is outside the frozen mixed label/route space: {key}")
            result[key] = {
                "label_name": label,
                "track_id": str(raw.get("track_id") or ""),
                "capture_id": str(raw.get("capture_id") or ""),
                "provenance_component_id": str(raw.get("provenance_component_id") or ""),
                "l1_route": route,
                "known_target_accepted": parse_bool_field(
                    raw.get("known_target_accepted"),
                    field=f"V26f route row {key} known_target_accepted",
                ),
                "known_clutter_accepted": parse_bool_field(
                    raw.get("known_clutter_accepted"),
                    field=f"V26f route row {key} known_clutter_accepted",
                ),
            }
    return result


def read_v26f_route_rows(root: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    """Return the already-frozen V26f route rows for report-only grouping.

    The returned route is an observed L1 output from V26f, not a truth-derived
    routing decision.  This small public wrapper keeps the diagnostic report
    from duplicating the CSV parser while preserving the source validation
    performed by :func:`validate_v26f_source`.
    """

    return _read_route_rows(root)


def validate_v26f_source(root: Path) -> dict[str, Any]:
    """Validate the frozen V26f mixed population and route metadata."""

    contract, readiness = validate_v26f_prepared_contract(root)
    contract_path = root / V26F_CONTRACT
    readiness_path = root / V26F_READINESS
    report_path = require_bound_file(root, V26F_REPORT, "V26f report", max_bytes=MAX_REPORT_BYTES)
    detail_path = require_bound_file(root, V26F_PARENT_DETAIL_CSV, "V26f parent detail", max_bytes=MAX_DETAIL_BYTES)
    report = read_json(report_path, description="V26f report", max_bytes=MAX_REPORT_BYTES)
    if (
        contract.get("schema_version") != V26F_CONTRACT_SCHEMA
        or readiness.get("schema_version") != V26F_READINESS_SCHEMA
        or report.get("schema_version") != V26F_REPORT_SCHEMA
        or report.get("status") != "completed_v26f_frozen_mixed_open_set_audit"
        or report.get("contract_sha256") != sha256_file(contract_path)
        or report.get("readiness_sha256") != sha256_file(readiness_path)
        or report.get("parent_detail_csv_sha256") != sha256_file(detail_path)
        or report.get("deployment_eligible") is not False
        or dict(report.get("isolation") or {}).get("unknown_training_exposure") != 0
        or dict(report.get("isolation") or {}).get("unknown_threshold_fit_exposure") != 0
    ):
        raise ValueError("V26k V26f source is stale or outside its frozen audit scope")
    route_rows = _read_route_rows(root)
    expected_keys: set[tuple[str, int, str]] = set()
    for split, counts in EXPECTED_PARENT_COUNTS.items():
        metadata = _index_parent_metadata(root, split=split)
        if {label: sum(value["label_name"] == label for value in metadata.values()) for label in MIXED_LABELS} != counts:
            raise ValueError(f"V26k {split} index support differs from V26f")
        for fold_id in range(FOLD_COUNT):
            for parent_id, identity in metadata.items():
                key = route_key(split, fold_id, parent_id)
                expected_keys.add(key)
                row = route_rows.get(key)
                if row is None:
                    raise ValueError(f"V26k V26f route detail is missing: {key}")
                if any(row[field] != identity[field] for field in ("label_name", "track_id", "capture_id", "provenance_component_id")):
                    raise ValueError(f"V26k V26f route identity differs from the CQ08 index: {key}")
    if set(route_rows) != expected_keys:
        raise ValueError("V26k V26f route detail has extra or missing parent keys")
    cache_bindings = []
    for raw in list(dict(contract.get("data") or {}).get("cache_bindings") or ()):
        row = dict(raw)
        split = str(row.get("split") or "")
        if split not in EXPECTED_PARENT_COUNTS or dict(row.get("parent_counts") or {}) != EXPECTED_PARENT_COUNTS[split]:
            raise ValueError("V26k V26f mixed cache population differs")
        for name in ("l1_course", "l2_rd_physical4", "l2_micro_doppler8"):
            binding = dict(row.get(name) or {})
            path = v26f_require_bound_file(
                root,
                str(binding.get("path") or ""),
                f"V26f {split} {name} cache",
                expected_sha256=str(binding.get("sha256") or ""),
                max_bytes=FEATURE_CACHE_MAX_BYTES,
            )
            cache_bindings.append(
                {
                    "split": split,
                    "name": name,
                    "path": str(binding["path"]),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    keys = sorted(route_rows)
    return {
        "v26f_contract_sha256": sha256_file(contract_path),
        "v26f_readiness_sha256": sha256_file(readiness_path),
        "v26f_report_sha256": sha256_file(report_path),
        "v26f_parent_detail_sha256": sha256_file(detail_path),
        "v26f_contract_schema": V26F_CONTRACT_SCHEMA,
        "v26f_readiness_schema": V26F_READINESS_SCHEMA,
        "v26f_report_schema": V26F_REPORT_SCHEMA,
        "mixed_parent_counts": EXPECTED_PARENT_COUNTS,
        "route_row_count": int(len(route_rows)),
        "route_key_sequence_sha256": route_key_sequence_sha256(keys),
        "mixed_cache_bindings": cache_bindings,
    }


def validate_prepared_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_v26c = validate_v26c_source(root)
    source_v26f = validate_v26f_source(root)
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    contract = read_json(contract_path, description="V26k contract", max_bytes=MAX_CONTRACT_BYTES)
    readiness = read_json(readiness_path, description="V26k readiness", max_bytes=MAX_READINESS_BYTES)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or dict(contract.get("source_v26c") or {}) != source_v26c
        or dict(contract.get("source_v26f") or {}) != source_v26f
        or dict(contract.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
        or dict(contract.get("implementation_sha256") or {}) != implementation_sha256(root)
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or int(contract.get("fixed_l2_epoch", -1)) != L2_EPOCH
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_reference_fit_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or contract.get("new_training") is not False
        or contract.get("semantic_action_mutation") is not False
        or contract.get("product_threshold_mutation") is not False
        or contract.get("deployment_eligible") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V26k prepared contract or readiness is invalid or stale")
    for raw in list(contract.get("source_bindings") or ()):
        item = dict(raw)
        path = require_bound_file(
            root,
            str(item.get("path") or ""),
            f"V26k source binding {item.get('name', '')}",
            expected_sha256=str(item.get("sha256") or ""),
            max_bytes=MAX_JSON_SOURCE_BYTES,
        )
        if int(item.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError("V26k source binding size differs")
    return contract, readiness


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONTRACT",
    "CONTRACT_SCHEMA",
    "DBB_LABELS",
    "DETAIL_CSV",
    "DETAIL_FIELDS",
    "DOCUMENT",
    "EXPECTED_PARENT_COUNTS",
    "FOLD_COUNT",
    "INDEX_PATH",
    "KNOWN_RETENTION_TARGET",
    "L1_TARGET_ROUTE",
    "L2_CLASSES",
    "L2_CHECKPOINT_SCHEMA",
    "L2_EPOCH",
    "MAX_CONTRACT_BYTES",
    "MAX_DETAIL_BYTES",
    "MAX_FEATURE_CACHE_BYTES",
    "MAX_ISOLATED_FOLD_OUTPUT_BYTES",
    "MAX_READINESS_BYTES",
    "MAX_REPORT_BYTES",
    "MIXED_LABELS",
    "OOD_SCORE_POLICY",
    "READINESS",
    "READINESS_SCHEMA",
    "REPORT",
    "REPORT_SCHEMA",
    "REPRESENTATION_NAME",
    "SCORE_NAME",
    "UNKNOWN_BASE_LABEL",
    "binary_ranking_metrics",
    "class_summary",
    "fit_class_conditional_reference",
    "implementation_sha256",
    "mahalanobis_scores",
    "parent_id_sequence_sha256",
    "parse_bool_field",
    "read_json",
    "read_v26f_route_rows",
    "relative_path",
    "require_bound_file",
    "route_key",
    "route_key_sequence_sha256",
    "selected_event_ids_for_parents",
    "sha256_file",
    "source_binding",
    "state_matrix_sha256",
    "train_only_retention_threshold",
    "validate_prepared_contract",
    "validate_v26c_source",
    "validate_v26f_source",
    "value_summary",
    "write_immutable_json",
    "write_immutable_text",
]
