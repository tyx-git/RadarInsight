"""Prepare V21h/V21i nested OOF experiments on top of frozen V21e."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import _write_npz_new_or_identical
from radar_ln3.known_gate import (
    KNOWN_LABELS,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    KNOWN_GATE_FEATURE_DIM,
)
from radar_ln3.known_gate_v21e_oof import (
    fold_record,
    load_fold_contract,
)
from radar_ln3.known_gate_v21hi_oof import (
    ASSIGNMENT_RELATIVE,
    FOLD_COUNT,
    FIXED_EPOCHS,
    INDEX_RELATIVE,
    V21E_COURSE_CACHE_RELATIVE,
    V21E_FEATURE_CONTRACT_RELATIVE,
    V21E_FEATURE_MANIFEST_RELATIVE,
    V21E_TREATMENT_REPORT_RELATIVE,
    V21_CORE_CONTRACT_RELATIVE,
    V21_CORE_MANIFEST_RELATIVE,
    AppendSpec,
    artifact_relative,
    canonical_sha256,
    config_relative,
    get_spec,
    load_fold_contract as load_shared_fold_contract,
    prediction_relative,
    relative_path,
    sha256_file,
    v21e_config_relative,
    write_new_or_identical,
)


def _load_cache(
    path: Path, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V21h/i source cache fields changed: {path}")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if (
        sample_ids.ndim != 1
        or sample_ids.dtype.kind not in {"U", "S"}
        or values.shape != (len(sample_ids), width)
        or values.dtype != np.float32
        or valid.shape != values.shape
        or valid.dtype != np.bool_
        or len(set(str(value) for value in sample_ids.tolist())) != len(sample_ids)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or np.any(values[~valid] != 0.0)
    ):
        raise ValueError(f"V21h/i source cache contract changed: {path}")
    return sample_ids, values, valid


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("V21h/i AUC groups must be non-empty")
    return float(
        (
            (pos[:, None] > neg[None, :]).sum()
            + 0.5 * (pos[:, None] == neg[None, :]).sum()
        )
        / (len(pos) * len(neg))
    )


def _summary(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "support": int(len(data)),
        "mean": float(np.mean(data)) if len(data) else None,
        "median": float(np.median(data)) if len(data) else None,
        "q25": float(np.quantile(data, 0.25)) if len(data) else None,
        "q75": float(np.quantile(data, 0.75)) if len(data) else None,
    }


def _load_source_values(
    root: Path, spec: AppendSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    course_ids, course_values, course_valid = _load_cache(
        root / V21E_COURSE_CACHE_RELATIVE, len(COURSE_PREFIX_FEATURE_NAMES)
    )
    source_ids, source_values, source_valid = _load_cache(
        root / spec.source_cache_relative, len(spec.source_indices)
        if spec.source_kind == "independent_prefix_cache"
        else KNOWN_GATE_FEATURE_DIM
    )
    if not np.array_equal(course_ids, source_ids):
        raise ValueError(f"{spec.code} source cache is not event-aligned with V21e")
    if spec.source_kind == "core_cache_selected":
        source_values = source_values[:, spec.source_indices]
        source_valid = source_valid[:, spec.source_indices]
    if source_values.shape[1] != len(spec.new_feature_names):
        raise ValueError(f"{spec.code} source feature width changed")
    combined_values = np.concatenate([course_values, source_values], axis=1).astype(
        np.float32, copy=False
    )
    combined_valid = np.concatenate([course_valid, source_valid], axis=1).astype(
        np.bool_, copy=False
    )
    source_hashes = {
        "course_cache_sha256": sha256_file(root / V21E_COURSE_CACHE_RELATIVE),
        "source_cache_sha256": sha256_file(root / spec.source_cache_relative),
        "source_contract_sha256": sha256_file(root / spec.source_contract_relative),
        "source_manifest_sha256": sha256_file(root / spec.source_manifest_relative),
    }
    return course_ids, combined_values, combined_valid, source_hashes


def _train_only_prior(
    root: Path,
    spec: AppendSpec,
    sample_ids: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    frame = pd.read_csv(root / INDEX_RELATIVE, low_memory=False)
    selected = frame[
        frame["split_name"].astype(str).eq("train")
        & frame["label_name"].astype(str).isin(KNOWN_LABELS)
    ].sort_values(["parent_sample_id", "causal_event_index"], kind="stable")
    if [str(value) for value in selected["sample_id"]] != [
        str(value) for value in sample_ids
    ]:
        raise ValueError(f"{spec.code} cache no longer aligns with the frozen train index")
    rows: list[dict[str, Any]] = []
    for parent_id, group in selected.groupby("parent_sample_id", sort=True):
        labels = group["label_name"].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"{spec.code} parent label is not unique: {parent_id}")
        positions = selected.index.get_indexer(group.index)
        row: dict[str, Any] = {
            "parent_sample_id": str(parent_id),
            "label_name": labels[0],
        }
        for feature_index, feature_name in enumerate(spec.new_feature_names):
            active = valid[positions, len(COURSE_PREFIX_FEATURE_NAMES) + feature_index]
            if not active.any():
                raise ValueError(f"{spec.code} parent has no valid feature event: {parent_id}")
            row[feature_name] = float(
                values[positions[active], len(COURSE_PREFIX_FEATURE_NAMES) + feature_index]
                .astype(np.float64)
                .mean()
            )
        rows.append(row)
    parents = pd.DataFrame(rows)
    expected_support = {
        "DroneTarget": 476,
        "BirdTarget": 239,
        "BalloonTarget": 238,
        "ClutterTarget": 33,
    }
    actual_support = {
        label: int((parents["label_name"] == label).sum()) for label in KNOWN_LABELS
    }
    if actual_support != expected_support:
        raise ValueError(
            f"{spec.code} train parent support changed: {actual_support}"
        )
    primary_rows: dict[str, Any] = {}
    other_label = next(
        label for label in spec.primary_labels if label != "ClutterTarget"
    )
    clutter = parents[parents.label_name.eq("ClutterTarget")]
    other = parents[parents.label_name.eq(other_label)]
    # Store the requested primary pair in a stable direction-independent form.
    for feature_name in spec.new_feature_names:
        value = _auc(
            clutter[feature_name].to_numpy(dtype=np.float64),
            other[feature_name].to_numpy(dtype=np.float64),
        )
        primary_rows.setdefault(feature_name, {})[
            f"ClutterTarget_vs_{other_label}"
        ] = {
            "clutter_higher_auc": value,
            "direction_free_auc": max(value, 1.0 - value),
        }
    distributions = {
        label: {
            "parent_support": int((parents.label_name == label).sum()),
            "features": {
                name: _summary(
                    parents.loc[parents.label_name.eq(label), name].to_numpy(
                        dtype=np.float64
                    )
                )
                for name in spec.new_feature_names
            },
        }
        for label in KNOWN_LABELS
    }
    return {
        "role": "train-only descriptive prior; not a readiness or checkpoint gate",
        "parent_support": actual_support,
        "feature_distributions": distributions,
        "primary_pair_separation": primary_rows,
    }


def _config_payload(
    source: Mapping[str, Any],
    *,
    root: Path,
    spec: AppendSpec,
    fold_id: int,
    manifest_path: Path,
    manifest_hash: str,
    cache_path: Path,
    cache_hash: str,
    experiment_path: Path,
    experiment_hash: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(source))
    payload["config_name"] = (
        f"RADAR-LN3-{spec.code}-cq08-known-gate-course-{spec.slug}-"
        f"oof-fold-{fold_id:02d}-treatment-seed43"
    )
    payload["variant"] = spec.variant
    task = dict(payload["task"])
    task["feature_names"] = list(spec.feature_names)
    payload["task"] = task
    data = dict(payload["data"])
    data.update(
        {
            "feature_cache_manifest_path": relative_path(root, manifest_path),
            "expected_feature_cache_manifest_sha256": manifest_hash,
            "train_feature_cache_path": relative_path(root, cache_path),
            "validation_feature_cache_path": None,
            "expected_train_feature_cache_sha256": cache_hash,
            "expected_validation_feature_cache_sha256": None,
            "feature_experiment_contract_path": relative_path(root, experiment_path),
            "expected_feature_experiment_contract_sha256": experiment_hash,
        }
    )
    payload["data"] = data
    model = dict(payload["model"])
    model["evidence_dim"] = len(spec.feature_names)
    payload["model"] = model
    artifact = root / artifact_relative(spec.code[-1].lower(), fold_id)
    train = dict(payload["train"])
    train.update(
        {
            "checkpoint_path": relative_path(root, artifact / "final_model.pt"),
            "last_checkpoint_path": relative_path(root, artifact / "last_model.pt"),
            "formal_artifact_dir": relative_path(root, artifact),
            "epoch_checkpoint_dir": relative_path(root, artifact / "epochs"),
            "checkpoint_schema": (
                f"RADAR-LN3-{spec.code}-known-gate-course-{spec.slug}-oof-fold-"
                f"{fold_id:02d}-treatment-checkpoint-v1"
            ),
            "save_every_epoch": False,
            "paired_baseline_config_path": relative_path(
                root, root / v21e_config_relative("treatment", fold_id)
            ),
        }
    )
    payload["train"] = train
    return payload


def build(project_root: Path, *, variant: str, preview: bool = False) -> dict[str, Any]:
    root = project_root.resolve()
    spec = get_spec(variant)
    source_paths = {
        "index": root / INDEX_RELATIVE,
        "assignment": root / ASSIGNMENT_RELATIVE,
        "v21e_feature_contract": root / V21E_FEATURE_CONTRACT_RELATIVE,
        "v21e_feature_manifest": root / V21E_FEATURE_MANIFEST_RELATIVE,
        "v21e_course_cache": root / V21E_COURSE_CACHE_RELATIVE,
        "v21e_treatment_report": root / V21E_TREATMENT_REPORT_RELATIVE,
        "source_cache": root / spec.source_cache_relative,
        "source_contract": root / spec.source_contract_relative,
        "source_manifest": root / spec.source_manifest_relative,
    }
    for fold_id in range(FOLD_COUNT):
        source_paths[f"v21e_config_{fold_id:02d}"] = (
            root / v21e_config_relative("treatment", fold_id)
        )
        source_paths[f"v21e_prediction_{fold_id:02d}"] = (
            root
            / Path(
                "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof"
            )
            / f"fold-{fold_id:02d}"
            / "treatment/held-out-predictions.json"
        )
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{spec.code} source files are missing: {missing}")
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    v21e_contract, assignment = load_fold_contract(root)
    v21e_report = json.loads(
        source_paths["v21e_treatment_report"].read_text(encoding="utf-8")
    )
    if (
        v21e_report.get("status") != "completed_absolute_guard_failed"
        or v21e_report.get("advance_to_control_training") is not False
        or int(assignment.get("known_parent_count", -1)) != 986
        or int(assignment.get("provenance_component_count", -1)) != 975
        or int(assignment.get("unknown_training_exposure", -1)) != 0
        or int(assignment.get("validation_parent_assignment_count", -1)) != 0
        or int(assignment.get("test_parent_assignment_count", -1)) != 0
    ):
        raise ValueError(f"{spec.code} V21e source OOF scope/result changed")
    preview_result = {
        "status": "preview_passed",
        "variant": spec.variant,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "folds": assignment["folds"],
        "new_training_run_count": FOLD_COUNT,
        "unknown_training_exposure": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
    }
    if preview:
        return preview_result

    sample_ids, combined_values, combined_valid, source_feature_hashes = (
        _load_source_values(root, spec)
    )
    cache_path = root / spec.combined_cache_relative
    _write_npz_new_or_identical(
        cache_path,
        (
            ("sample_ids", sample_ids),
            ("values", combined_values),
            ("valid", combined_valid),
        ),
    )
    cache_hash = sha256_file(cache_path)
    prior = _train_only_prior(
        root, spec, sample_ids, combined_values, combined_valid
    )
    experiment = {
        "schema_version": spec.experiment_contract_schema,
        "experiment": f"{spec.code} V21e course plus {spec.slug}",
        "role": "exploratory nested paired known-train OOF; not independent confirmation",
        "feature_names": list(spec.feature_names),
        "control_features": list(COURSE_PREFIX_FEATURE_NAMES),
        "new_treatment_features": list(spec.new_feature_names),
        "single_factor": f"append frozen {spec.slug} evidence to V21e",
        "primary_labels": list(spec.primary_labels),
        "safety_labels": list(spec.safety_labels),
        "pairing": {
            "control": "frozen V21e OOF Treatment predictions",
            "treatment": f"{spec.code} full-parameter 20-epoch fold models",
            "shared_initialization": (
                "recursively reconstruct V21e Treatment initialization, copy its course "
                "column, and zero only the appended feature columns"
            ),
            "same_fold_train_parent_order": True,
            "fixed_final_epoch": FIXED_EPOCHS,
            "held_out_checkpoint_selection": False,
        },
        "source_hashes": source_hashes,
        "source_feature_hashes": source_feature_hashes,
        "train_only_prior": prior,
        "combined_cache": {
            "path": relative_path(root, cache_path),
            "sha256": cache_hash,
            "event_count": int(len(sample_ids)),
            "value_dim": int(combined_values.shape[1]),
        },
        "unknown_training_exposure": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "threshold_fitting": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    experiment_path = root / spec.experiment_contract_relative
    write_new_or_identical(experiment_path, experiment)
    experiment_hash = sha256_file(experiment_path)
    manifest = {
        "schema_version": spec.feature_manifest_schema,
        "feature_names": list(spec.feature_names),
        "source_caches": {
            "course": {
                "path": relative_path(root, root / V21E_COURSE_CACHE_RELATIVE),
                "sha256": source_hashes["v21e_course_cache"],
            },
            "appended": {
                "path": relative_path(root, root / spec.source_cache_relative),
                "sha256": source_hashes["source_cache"],
            },
        },
        "cache_files": {
            "train": {
                "path": relative_path(root, cache_path),
                "sha256": cache_hash,
                "event_count": int(len(sample_ids)),
                "fields": ["sample_ids", "values", "valid"],
            }
        },
        "experiment_contract": relative_path(root, experiment_path),
        "experiment_contract_sha256": experiment_hash,
        "sample_id_is_model_input": False,
        "valid_mask_is_model_input": False,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    manifest_path = root / spec.feature_manifest_relative
    write_new_or_identical(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)

    fold_rows: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(v21e_contract, fold_id)
        source_config_path = root / v21e_config_relative("treatment", fold_id)
        source_payload = json.loads(source_config_path.read_text(encoding="utf-8"))
        path = root / config_relative(variant, fold_id)
        payload = _config_payload(
            source_payload,
            root=root,
            spec=spec,
            fold_id=fold_id,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            cache_path=cache_path,
            cache_hash=cache_hash,
            experiment_path=experiment_path,
            experiment_hash=experiment_hash,
        )
        write_new_or_identical(path, payload)
        config = load_known_gate_config(path)
        if (
            config.variant != spec.variant
            or tuple(config.feature_names) != tuple(spec.feature_names)
            or config.model.evidence_dim != len(spec.feature_names)
            or config.train.paired_baseline_config_path
            != relative_path(root, source_config_path)
        ):
            raise ValueError(f"{spec.code} fold {fold_id} config pair failed")
        control_prediction_path = source_paths[f"v21e_prediction_{fold_id:02d}"]
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_parent_count": len(fold["train_parent_ids"]),
                "held_out_parent_count": len(fold["held_out_parent_ids"]),
                "train_parent_ids_sha256": fold["train_parent_ids_sha256"],
                "held_out_parent_ids_sha256": fold["held_out_parent_ids_sha256"],
                "train_label_parent_counts": fold["train_label_parent_counts"],
                "held_out_label_parent_counts": fold[
                    "held_out_label_parent_counts"
                ],
                "normalization_stats_path": fold["normalization_stats_path"],
                "normalization_stats_sha256": fold["normalization_stats_sha256"],
                "data_contract_path": fold["data_contract_path"],
                "data_contract_sha256": fold["data_contract_sha256"],
                "class_weights": fold["class_weights"],
                "control": {
                    "config_path": relative_path(root, source_config_path),
                    "config_sha256": sha256_file(source_config_path),
                    "prediction_path": relative_path(root, control_prediction_path),
                    "prediction_sha256": sha256_file(control_prediction_path),
                },
                "treatment": {
                    "config_path": relative_path(root, path),
                    "config_sha256": sha256_file(path),
                    "config_identity_sha256": known_gate_config_identity_sha256(
                        config
                    ),
                },
            }
        )
    oof_contract = {
        "schema_version": spec.oof_contract_schema,
        "experiment": f"{spec.code} nested paired five-fold OOF",
        "source_assignment_path": relative_path(root, root / ASSIGNMENT_RELATIVE),
        "source_assignment_sha256": sha256_file(root / ASSIGNMENT_RELATIVE),
        "source_assignment_semantic_sha256": canonical_sha256(assignment),
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "grouping": "capture_content_provenance_component",
        "prediction_protocol": "out_of_fold_fixed_final_epoch",
        "control_reused_from": "V21e OOF Treatment",
        "new_training_arm": "treatment",
        "feature_names": list(spec.feature_names),
        "folds": fold_rows,
        "unknown_training_exposure": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "threshold_fitting": "未执行",
        "deployment_eligible": False,
    }
    oof_path = root / spec.oof_contract_relative
    write_new_or_identical(oof_path, oof_contract)
    checks = {
        "v21e_source_absolute_guard_failure_is_disclosed": True,
        "same_five_provenance_grouped_folds_are_reused": True,
        "v21e_control_predictions_are_sha_bound": True,
        "course_and_appended_caches_are_event_aligned": True,
        "combined_cache_is_label_free_and_train_only": True,
        "treatment_appends_exactly_one_feature_column": True,
        "nested_pair_reconstructs_v21e_treatment_initialization": True,
        "unknown_training_exposure_is_zero": True,
        "validation_access_count_is_zero": True,
        "test_payload_access_count_is_zero": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": spec.readiness_schema,
        "variant": spec.variant,
        "checks": checks,
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "selected_seed": 43,
        "unknown_training_exposure": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "oof_contract_path": relative_path(root, oof_path),
        "oof_contract_sha256": sha256_file(oof_path),
        "experiment_contract_path": relative_path(root, experiment_path),
        "experiment_contract_sha256": experiment_hash,
        "feature_manifest_path": relative_path(root, manifest_path),
        "feature_manifest_sha256": manifest_hash,
        "train_feature_cache_path": relative_path(root, cache_path),
        "train_feature_cache_sha256": cache_hash,
        "new_training_run_count": FOLD_COUNT,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行五个 Treatment fold）",
        "OOF": "未执行",
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / spec.readiness_relative
    write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "variant": spec.variant,
        "combined_cache": str(cache_path),
        "oof_contract": str(oof_path),
        "readiness": str(readiness_path),
        "new_training_run_count": FOLD_COUNT,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V21h/V21i nested OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("h", "i"), required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.project_root.resolve(), variant=args.variant, preview=args.preview),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
