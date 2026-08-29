"""Immutable contract for the V26h train-only L1 representation OOD diagnostic.

V26h deliberately does *not* add an open-set head, retrain a model, fit a
rejection threshold, or alter a V26f/V26g semantic action.  It asks one
narrower question first: after fitting each frozen V26e L1 checkpoint's
KnownTarget/Clutter state references on that fold's train parents only, does
the resulting novelty score rank held-out ``UnknownTarget`` parents above
held-out known parents better than the already-failed max-softmax baseline?

The score is a class-conditional, pooled diagonal Mahalanobis distance over
the classifier's *readout state*.  This is important: ``liquid_final_state``
is the pre-classifier recurrent state while the actual L1 classifier consumes
``classifier[0](liquid_final_state)`` (a LayerNorm).  V26h therefore uses that
LayerNorm output, without gradients, rather than treating a different latent
space as if it were the decision representation.

Every fold owns its own train-only reference distribution.  State vectors,
centres, variances and scores are never pooled across checkpoints.  Unknown
is excluded from model fitting, normalisation, reference fitting, all
threshold work and all selection.  It appears only in the final frozen
mixed-validation/test ranking report.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .known_gate_v26e_oof import (
    CONTRACT as V26E_CONTRACT,
    FOLD_COUNT,
    L1_EPOCH,
    READINESS as V26E_READINESS,
    fold_record,
    l1_checkpoint_path,
    l1_config_path,
    l1_held_out_feature_cache_path,
    l1_train_feature_cache_path,
)
from .known_gate_v26f import (
    CONTRACT as V26F_CONTRACT,
    EXPECTED_PARENT_COUNTS,
    L1_TEST_CACHE,
    L1_VALIDATION_CACHE,
    MAX_REPORT_BYTES as V26F_MAX_REPORT_BYTES,
)

# V26f's contract is deliberately imported under an explicit alias so a
# future V26 module cannot silently change this diagnostic's source chain.
from .known_gate_v26f import (
    PARENT_DETAIL_CSV as V26F_PARENT_DETAIL_CSV,
    READINESS as V26F_READINESS,
    REPORT as V26F_REPORT,
    sha256_file,
    validate_prepared_contract as validate_v26f_prepared_contract,
)


CONTRACT = Path(
    "data/processed/RADAR-LN3-V26h-cq08-l1-train-only-mahalanobis-ood-diagnostic-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic.json"
)
DETAIL_CSV = Path(
    "src-v3/processed/report/RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-parent-details.csv"
)
DOCUMENT = Path("src-v3/docs/radar_ln3_v26h_train_only_mahalanobis_ood_diagnostic.md")

CONTRACT_SCHEMA = "RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-readiness-v1"
REPORT_SCHEMA = "RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-v1"

MAX_REPORT_BYTES = max(8 * 1024 * 1024, int(V26F_MAX_REPORT_BYTES))
MAX_DETAIL_BYTES = 2 * 1024 * 1024
MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_READINESS_BYTES = 512 * 1024
MAX_JSON_SOURCE_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_FEATURE_CACHE_BYTES = 64 * 1024 * 1024
MAX_ISOLATED_FOLD_OUTPUT_BYTES = 2 * 1024 * 1024

L1_CLASSES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
KNOWN_BASE_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
UNKNOWN_BASE_LABEL = "UnknownTarget"
REPRESENTATION_NAME = "classifier_layer_norm_liquid_final_state"
SCORE_NAME = "minimum_class_conditional_pooled_diagonal_mahalanobis_distance"
SCORE_ORIENTATION = "higher_means_less_like_train_known"
VARIANCE_FLOOR_MULTIPLIER = 1.0e-6
VARIANCE_FLOOR_ABSOLUTE = 1.0e-10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 26068

# This is the sole analytical variable.  There is intentionally no threshold,
# no score fusion, no classifier mutation, and no Unknown-derived setting.
OOD_SCORE_POLICY: dict[str, Any] = {
    "score_name": SCORE_NAME,
    "score_orientation": SCORE_ORIENTATION,
    "representation": REPRESENTATION_NAME,
    "representation_dimension": 48,
    "reference_fit": "per_fold_train_known_only_parent_final_states",
    "class_conditional_references": list(L1_CLASSES),
    "covariance": "pooled_diagonal_within_class_variance",
    "variance_floor_multiplier_of_median_positive_variance": VARIANCE_FLOOR_MULTIPLIER,
    "variance_floor_absolute": VARIANCE_FLOOR_ABSOLUTE,
    "unknown_in_reference_fit": False,
    "unknown_in_normalization": False,
    "unknown_in_threshold_fit": False,
    "unknown_in_model_or_checkpoint_selection": False,
    "threshold_fitting": "not_executed",
    "semantic_action_mutation": False,
    "score_fusion_with_msp_or_energy": False,
    "cross_fold_state_pooling": False,
}

IMPLEMENTATION_FILES = (
    Path("src-v3/radar_ln3/known_gate_v26h.py"),
    Path("src-v3/build/known_gate_v26h.py"),
    Path("src-v3/report/known_gate_v26h.py"),
    Path("src-v3/smoke/known_gate_v26h_contract_test.py"),
    Path("src-v3/scripts/v26h_known_gate_train_only_mahalanobis_ood_diagnostic_pipeline.ps1"),
)


def resolve_path(root: Path, value: str | Path) -> Path:
    raw_text = str(value).strip()
    raw = Path(raw_text.replace("\\", "/"))
    return (raw if raw.is_absolute() else root / raw).resolve()


def relative_path(root: Path, value: str | Path) -> str:
    root_path = Path(root).resolve()
    path = resolve_path(root_path, value)
    try:
        return path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"V26h path escapes the project root: {value}") from exc


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
        raise ValueError(f"V26h {description} escapes the project root: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"V26h {description} is missing: {path}")
    if int(path.stat().st_size) > int(max_bytes):
        raise MemoryError(f"V26h {description} exceeds bounded size: {path}")
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"V26h {description} hash differs: {path}")
    return path


def read_json(path: Path, *, description: str, max_bytes: int = MAX_JSON_SOURCE_BYTES) -> dict[str, Any]:
    checked = Path(path)
    if not checked.is_file():
        raise FileNotFoundError(f"V26h {description} is missing: {checked}")
    if int(checked.stat().st_size) > int(max_bytes):
        raise MemoryError(f"V26h {description} exceeds bounded size: {checked}")
    payload = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"V26h {description} must be a JSON object: {checked}")
    return payload


def write_immutable_text(path: Path, text: str, *, max_bytes: int, description: str) -> str:
    if len(text.encode("utf-8")) > int(max_bytes):
        raise MemoryError(f"V26h {description} exceeds bounded size: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V26h {description}: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() and temporary.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to replace a different interrupted V26h {description}: {temporary}"
        )
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


def implementation_sha256(root: Path) -> dict[str, str]:
    return {
        relative_path(root, path): sha256_file(
            require_bound_file(root, path, f"implementation {path}")
        )
        for path in IMPLEMENTATION_FILES
    }


def source_binding(
    root: Path,
    value: str | Path,
    *,
    description: str,
    max_bytes: int = MAX_JSON_SOURCE_BYTES,
) -> dict[str, Any]:
    path = require_bound_file(root, value, description, max_bytes=max_bytes)
    return {
        "path": relative_path(root, path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def state_matrix_sha256(states: np.ndarray) -> str:
    value = np.asarray(states, dtype="<f8")
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("V26h state hash requires a finite two-dimensional matrix")
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def parent_id_sequence_sha256(parent_ids: Sequence[str]) -> str:
    values = tuple(str(value) for value in parent_ids)
    if not values or len(set(values)) != len(values):
        raise ValueError("V26h parent identifiers must be non-empty and unique")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def fit_class_conditional_reference(states: np.ndarray, class_ids: np.ndarray) -> dict[str, Any]:
    """Fit two train-only class centres and a pooled diagonal covariance."""

    array = np.asarray(states, dtype=np.float64)
    labels = np.asarray(class_ids, dtype=np.int64)
    if (
        array.ndim != 2
        or array.shape[0] < len(L1_CLASSES)
        or labels.shape != (array.shape[0],)
        or not np.isfinite(array).all()
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("V26h reference fit needs finite aligned binary-class train states")
    centres: list[np.ndarray] = []
    residual_sum = np.zeros(array.shape[1], dtype=np.float64)
    class_counts: dict[str, int] = {}
    total = 0
    for class_id, name in enumerate(L1_CLASSES):
        selected = array[labels == class_id]
        if selected.shape[0] < 2:
            raise ValueError(f"V26h reference has inadequate train support for {name}")
        centre = selected.mean(axis=0)
        centres.append(centre)
        residual_sum += np.square(selected - centre).sum(axis=0)
        class_counts[name] = int(selected.shape[0])
        total += int(selected.shape[0])
    degrees_of_freedom = total - len(L1_CLASSES)
    if degrees_of_freedom <= 0:
        raise ValueError("V26h pooled diagonal variance has no degrees of freedom")
    pooled = residual_sum / float(degrees_of_freedom)
    positive = pooled[pooled > 0.0]
    variance_floor = max(
        (float(np.median(positive)) if positive.size else 1.0) * VARIANCE_FLOOR_MULTIPLIER,
        VARIANCE_FLOOR_ABSOLUTE,
    )
    pooled = np.maximum(pooled, variance_floor)
    centres_array = np.stack(centres, axis=0)
    if (
        centres_array.shape != (len(L1_CLASSES), array.shape[1])
        or pooled.shape != (array.shape[1],)
        or not np.isfinite(centres_array).all()
        or not np.isfinite(pooled).all()
        or bool((pooled <= 0.0).any())
    ):
        raise AssertionError("V26h fitted reference violates the finite diagonal contract")
    return {
        "centres": centres_array,
        "pooled_variance": pooled,
        "variance_floor": float(variance_floor),
        "class_counts": class_counts,
        "train_parent_count": int(total),
        "pooled_degrees_of_freedom": int(degrees_of_freedom),
        "state_dimension": int(array.shape[1]),
    }


def mahalanobis_scores(states: np.ndarray, fitted: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return novelty score and closest L1 reference class for each state."""

    array = np.asarray(states, dtype=np.float64)
    centres = np.asarray(fitted["centres"], dtype=np.float64)
    variance = np.asarray(fitted["pooled_variance"], dtype=np.float64)
    if (
        array.ndim != 2
        or centres.shape != (len(L1_CLASSES), array.shape[1])
        or variance.shape != (array.shape[1],)
        or not np.isfinite(array).all()
        or not np.isfinite(centres).all()
        or not np.isfinite(variance).all()
        or bool((variance <= 0.0).any())
    ):
        raise ValueError("V26h Mahalanobis scoring received an invalid reference")
    squared = np.square(array[:, None, :] - centres[None, :, :]) / variance[None, None, :]
    distances = np.sqrt(squared.sum(axis=2))
    closest = distances.argmin(axis=1).astype(np.int64, copy=False)
    score = distances[np.arange(array.shape[0]), closest]
    if score.shape != (array.shape[0],) or not np.isfinite(score).all():
        raise AssertionError("V26h Mahalanobis score vector is invalid")
    return score, closest


def binary_ranking_metrics(
    known_scores: Sequence[float], unknown_scores: Sequence[float]
) -> dict[str, Any]:
    """Threshold-free pairwise ranking metrics; high score means novelty."""

    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)
    if known.ndim != 1 or unknown.ndim != 1 or not len(known) or not len(unknown):
        raise ValueError("V26h ranking needs non-empty known and Unknown scores")
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ValueError("V26h ranking scores must be finite")
    comparisons = unknown[:, None] - known[None, :]
    auc = float((comparisons > 0.0).mean() + 0.5 * (comparisons == 0.0).mean())
    # Average precision with positive=Unknown.  The formula handles ties by
    # giving tied examples their common average precision contribution.
    scores = np.concatenate([unknown, known])
    labels = np.concatenate([np.ones(len(unknown), dtype=np.int64), np.zeros(len(known), dtype=np.int64)])
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
        "threshold_fitting": "not_executed",
    }


def validate_v26e_source(root: Path) -> dict[str, Any]:
    """Validate all five V26e fold contracts/checkpoints without inference."""

    contract_path = root / V26E_CONTRACT
    readiness_path = root / V26E_READINESS
    contract = read_json(contract_path, description="V26e contract")
    readiness = read_json(readiness_path, description="V26e readiness")
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version")
        != "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-contract-v1"
        or readiness.get("schema_version")
        != "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-readiness-v1"
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or int(dict(contract.get("l1") or {}).get("fixed_snapshot_epoch", -1)) != L1_EPOCH
    ):
        raise ValueError("V26h V26e source is stale or outside the Known-only freeze")
    # V26e itself has a transitive V26c L2 source binding.  V26h never loads
    # that L2 model, so checking it would make an L1-only representation
    # diagnostic fail merely because an unrelated L2 source is unavailable.
    # The V26f chain below separately validates the frozen mixed L1/L2/L3
    # audit population used for report-only evaluation.
    folds = list(contract.get("folds") or ())
    if len(folds) != FOLD_COUNT:
        raise ValueError("V26h V26e source has incomplete fold coverage")
    checked_folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        binding = dict(fold.get("l1") or {})
        source_paths = {
            "config": root / l1_config_path(fold_id),
            "checkpoint": root / l1_checkpoint_path(fold_id),
            "train_cache": root / l1_train_feature_cache_path(fold_id),
            "held_out_cache": root / l1_held_out_feature_cache_path(fold_id),
        }
        for name, path in source_paths.items():
            bound = require_bound_file(
                root,
                path,
                f"V26e fold {fold_id} {name}",
                max_bytes=MAX_CHECKPOINT_BYTES if name == "checkpoint" else MAX_FEATURE_CACHE_BYTES,
            )
            expected = {
                "config": binding.get("config_sha256"),
                "checkpoint": None,
                "train_cache": dict(binding.get("train_course_cache") or {}).get("sha256"),
                "held_out_cache": dict(binding.get("held_out_course_cache") or {}).get("sha256"),
            }[name]
            if expected is not None and sha256_file(bound) != str(expected):
                raise ValueError(f"V26h V26e fold {fold_id} {name} hash differs")
        if (
            int(binding.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
            or int(binding.get("train_parent_count", -1)) != len(fold["train_parent_ids"])
            or int(binding.get("held_out_parent_count", -1)) != len(fold["held_out_parent_ids"])
            or set(fold["train_parent_ids"]) & set(fold["held_out_parent_ids"])
        ):
            raise ValueError(f"V26h V26e fold {fold_id} parent/checkpoint binding differs")
        checked_folds.append(
            {
                "fold_id": fold_id,
                "train_parent_count": len(fold["train_parent_ids"]),
                "held_out_parent_count": len(fold["held_out_parent_ids"]),
                "config": source_binding(root, source_paths["config"], description=f"V26e fold {fold_id} config"),
                "checkpoint": source_binding(
                    root,
                    source_paths["checkpoint"],
                    description=f"V26e fold {fold_id} checkpoint",
                    max_bytes=MAX_CHECKPOINT_BYTES,
                ),
                "train_cache": source_binding(
                    root,
                    source_paths["train_cache"],
                    description=f"V26e fold {fold_id} train cache",
                    max_bytes=MAX_FEATURE_CACHE_BYTES,
                ),
                "held_out_cache": source_binding(
                    root,
                    source_paths["held_out_cache"],
                    description=f"V26e fold {fold_id} held-out cache",
                    max_bytes=MAX_FEATURE_CACHE_BYTES,
                ),
            }
        )
    return {
        "v26e_contract_sha256": sha256_file(contract_path),
        "v26e_readiness_sha256": sha256_file(readiness_path),
        "v26e_contract_schema": contract.get("schema_version"),
        "v26e_readiness_schema": readiness.get("schema_version"),
        "fixed_epoch": int(L1_EPOCH),
        "folds": checked_folds,
    }


def validate_v26f_source(root: Path) -> dict[str, Any]:
    """Bind V26h's mixed evaluation data to the already-frozen V26f audit."""

    # This validates V26f's own full source-chain bindings, including the
    # frozen L1/L2 checkpoints and mixed cache hashes.  V26h uses only its
    # mixed L1 cache, but preserving the full source validation makes the
    # report's population provenance explicit.
    contract, readiness = validate_v26f_prepared_contract(root)
    report_path = require_bound_file(root, V26F_REPORT, "V26f source report")
    detail_path = require_bound_file(root, V26F_PARENT_DETAIL_CSV, "V26f source parent detail")
    report = read_json(report_path, description="V26f source report")
    if (
        report.get("schema_version") != "RADAR-LN3-V26f-frozen-mixed-open-set-audit-v1"
        or report.get("status") != "completed_v26f_frozen_mixed_open_set_audit"
        or report.get("contract_sha256") != sha256_file(root / V26F_CONTRACT)
        or report.get("readiness_sha256") != sha256_file(root / V26F_READINESS)
        or report.get("parent_detail_csv_sha256") != sha256_file(detail_path)
        or dict(report.get("isolation") or {}).get("unknown_training_exposure") != 0
        or dict(report.get("isolation") or {}).get("unknown_threshold_fit_exposure") != 0
        or report.get("deployment_eligible") is not False
    ):
        raise ValueError("V26h V26f source report is stale or outside frozen audit scope")
    return {
        "v26f_contract_sha256": sha256_file(root / V26F_CONTRACT),
        "v26f_readiness_sha256": sha256_file(root / V26F_READINESS),
        "v26f_report_sha256": sha256_file(report_path),
        "v26f_parent_detail_sha256": sha256_file(detail_path),
        "v26f_contract_schema": contract.get("schema_version"),
        "v26f_readiness_schema": readiness.get("schema_version"),
        "v26f_report_schema": report.get("schema_version"),
        "mixed_parent_counts": EXPECTED_PARENT_COUNTS,
        "mixed_caches": {
            "val": relative_path(root, root / L1_VALIDATION_CACHE),
            "test": relative_path(root, root / L1_TEST_CACHE),
        },
    }


def validate_prepared_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless the V26h contract still matches its frozen sources."""

    source_v26e = validate_v26e_source(root)
    source_v26f = validate_v26f_source(root)
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    contract = read_json(contract_path, description="V26h contract")
    readiness = read_json(readiness_path, description="V26h readiness")
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or dict(contract.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
        or dict(contract.get("source_v26e") or {}) != source_v26e
        or dict(contract.get("source_v26f") or {}) != source_v26f
        or dict(contract.get("implementation_sha256") or {}) != implementation_sha256(root)
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or int(contract.get("unknown_reference_fit_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or contract.get("new_training") is not False
        or contract.get("new_threshold_fit") is not False
        or contract.get("semantic_action_mutation") is not False
        or contract.get("deployment_eligible") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V26h prepared contract/readiness is invalid or stale")
    for raw in list(contract.get("source_bindings") or ()):
        item = dict(raw)
        path = require_bound_file(
            root,
            str(item.get("path", "")),
            f"source binding {item.get('name', '')}",
            expected_sha256=str(item.get("sha256", "")),
            max_bytes=MAX_CHECKPOINT_BYTES,
        )
        if int(item.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError("V26h source binding size differs")
    return contract, readiness


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONTRACT",
    "CONTRACT_SCHEMA",
    "DETAIL_CSV",
    "DOCUMENT",
    "EXPECTED_PARENT_COUNTS",
    "FOLD_COUNT",
    "KNOWN_BASE_LABELS",
    "L1_CLASSES",
    "L1_EPOCH",
    "MAX_CONTRACT_BYTES",
    "MAX_DETAIL_BYTES",
    "MAX_ISOLATED_FOLD_OUTPUT_BYTES",
    "MAX_READINESS_BYTES",
    "MAX_REPORT_BYTES",
    "OOD_SCORE_POLICY",
    "READINESS",
    "READINESS_SCHEMA",
    "REPORT",
    "REPORT_SCHEMA",
    "REPRESENTATION_NAME",
    "SCORE_NAME",
    "UNKNOWN_BASE_LABEL",
    "binary_ranking_metrics",
    "fit_class_conditional_reference",
    "implementation_sha256",
    "mahalanobis_scores",
    "parent_id_sequence_sha256",
    "read_json",
    "relative_path",
    "require_bound_file",
    "resolve_path",
    "sha256_file",
    "source_binding",
    "state_matrix_sha256",
    "validate_prepared_contract",
    "validate_v26e_source",
    "validate_v26f_source",
    "write_immutable_json",
    "write_immutable_text",
]
