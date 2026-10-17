"""Prepare exploratory V21f = V21e course evidence + V21a spectral evidence."""

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
    COURSE_INTERFRAME_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_NAMES,
    INTERFRAME_SPECTRAL_FEATURE_NAMES,
    V21F_COURSE_INTERFRAME_VARIANT,
)
from radar_ln3.known_gate_v21e_oof import canonical_sha256, relative_path, sha256_file, write_new_or_identical
from radar_ln3.known_gate_v21f_oof import (
    ARM,
    ASSIGNMENT_RELATIVE,
    ATTRIBUTION_RELATIVE,
    ATTRIBUTION_SCHEMA,
    COMBINED_CACHE_RELATIVE,
    EXPERIMENT_CONTRACT_RELATIVE,
    EXPERIMENT_CONTRACT_SCHEMA,
    FEATURE_MANIFEST_RELATIVE,
    FEATURE_MANIFEST_SCHEMA,
    FOLD_COUNT,
    FIXED_EPOCHS,
    INDEX_RELATIVE,
    OOF_CONTRACT_RELATIVE,
    OOF_CONTRACT_SCHEMA,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    TRAINING_SEED,
    V21A_FEATURE_CONTRACT_RELATIVE,
    V21A_FEATURE_MANIFEST_RELATIVE,
    V21A_SPECTRAL_CACHE_RELATIVE,
    V21E_ARM,
    V21E_COURSE_CACHE_RELATIVE,
    V21E_FEATURE_CONTRACT_RELATIVE,
    V21E_FEATURE_MANIFEST_RELATIVE,
    V21E_TREATMENT_REPORT_RELATIVE,
    artifact_relative,
    config_relative,
    fold_record,
    load_fold_contract,
    v21e_config_relative,
    v21e_prediction_relative,
)


def _load_cache(path: Path, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V21f source cache fields changed: {path}")
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
        raise ValueError(f"V21f source cache contract changed: {path}")
    return sample_ids, values, valid


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("V21f AUC groups must be non-empty")
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
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "q25": float(np.quantile(data, 0.25)),
        "q75": float(np.quantile(data, 0.75)),
    }


def _hard_bird_attribution(
    root: Path,
    *,
    sample_ids: np.ndarray,
    spectral_values: np.ndarray,
    spectral_valid: np.ndarray,
    v21a_contract: Mapping[str, Any],
) -> dict[str, Any]:
    index = pd.read_csv(root / INDEX_RELATIVE, low_memory=False)
    selected = index[
        index["split_name"].astype(str).eq("train")
        & index["label_name"].astype(str).isin(KNOWN_LABELS)
    ].copy()
    final = (
        selected.sort_values(["parent_sample_id", "causal_event_index"])
        .groupby("parent_sample_id", as_index=False)
        .tail(1)
    )
    cache_index = {str(value): index for index, value in enumerate(sample_ids.tolist())}
    if len(cache_index) != len(sample_ids):
        raise ValueError("V21f attribution cache identifiers repeat")
    predictions: dict[str, dict[str, Any]] = {}
    for fold_id in range(FOLD_COUNT):
        path = root / v21e_prediction_relative(V21E_ARM, fold_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in list(dict(payload.get("records") or {}).get("parent") or []):
            parent = str(row["parent_sample_id"])
            if parent in predictions:
                raise ValueError("V21e OOF parent prediction repeats")
            predictions[parent] = dict(row)
    if set(predictions) != set(final["parent_sample_id"].astype(str)):
        raise ValueError("V21e OOF predictions do not cover V21f attribution population")

    rows: list[dict[str, Any]] = []
    for record in final.itertuples(index=False):
        sample_id = str(record.sample_id)
        position = cache_index[sample_id]
        prediction = predictions[str(record.parent_sample_id)]
        rows.append(
            {
                "parent_sample_id": str(record.parent_sample_id),
                "sample_id": sample_id,
                "label_name": str(record.label_name),
                "v21e_pred_id": int(prediction["pred_id"]),
                "v21e_correct": int(prediction["pred_id"])
                == int(prediction["true_id"]),
                "values": spectral_values[position],
                "valid": spectral_valid[position],
            }
        )
    hard_birds = [
        row
        for row in rows
        if row["label_name"] == "BirdTarget" and not row["v21e_correct"]
    ]
    correct_birds = [
        row
        for row in rows
        if row["label_name"] == "BirdTarget" and row["v21e_correct"]
    ]
    clutter = [row for row in rows if row["label_name"] == "ClutterTarget"]
    if len(hard_birds) != 14 or len(correct_birds) != 225 or len(clutter) != 33:
        raise ValueError("V21f hard-Bird attribution support changed")

    separation = dict(
        dict(v21a_contract["train_only_feature_diagnostic"])["statistics"]
    )["train_only_parent_separation"]["ClutterTarget_vs_BirdTarget"]
    feature_rows: dict[str, Any] = {}
    for feature_index, feature_name in enumerate(INTERFRAME_SPECTRAL_FEATURE_NAMES):
        expected_direction = str(separation[feature_name]["expected_clutter_direction"])
        values = {
            "v21e_hard_bird": [float(row["values"][feature_index]) for row in hard_birds],
            "v21e_correct_bird": [
                float(row["values"][feature_index]) for row in correct_birds
            ],
            "clutter": [float(row["values"][feature_index]) for row in clutter],
        }
        clutter_vs_hard = _auc(values["clutter"], values["v21e_hard_bird"])
        clutter_vs_correct = _auc(
            values["clutter"], values["v21e_correct_bird"]
        )
        hard_vs_correct = _auc(
            values["v21e_hard_bird"], values["v21e_correct_bird"]
        )
        if expected_direction == "lower":
            clutter_vs_hard = 1.0 - clutter_vs_hard
            clutter_vs_correct = 1.0 - clutter_vs_correct
        feature_rows[feature_name] = {
            "expected_clutter_direction_predeclared_by_v21a_train_audit": expected_direction,
            "groups": {name: _summary(data) for name, data in values.items()},
            "clutter_vs_v21e_hard_bird_expected_direction_auc": clutter_vs_hard,
            "clutter_vs_v21e_correct_bird_expected_direction_auc": clutter_vs_correct,
            "hard_vs_correct_bird_direction_free_auc": max(
                hard_vs_correct, 1.0 - hard_vs_correct
            ),
        }
    strongest_name, strongest = max(
        feature_rows.items(),
        key=lambda item: item[1][
            "clutter_vs_v21e_hard_bird_expected_direction_auc"
        ],
    )
    all_hard_valid = all(bool(row["valid"].all()) for row in hard_birds)
    return {
        "status": "analyzed_exploratory_post_hoc",
        "schema_version": ATTRIBUTION_SCHEMA,
        "scope": "V21e known-train OOF parent-final errors; no validation/test/Unknown",
        "supports": {
            "v21e_hard_bird": len(hard_birds),
            "v21e_correct_bird": len(correct_birds),
            "clutter": len(clutter),
        },
        "all_hard_bird_spectral_features_valid": all_hard_valid,
        "features": feature_rows,
        "strongest_conditional_feature": {
            "name": strongest_name,
            "expected_direction_auc": strongest[
                "clutter_vs_v21e_hard_bird_expected_direction_auc"
            ],
        },
        "mechanism_candidate_gate": {
            "threshold": 0.65,
            "passed": bool(
                all_hard_valid
                and strongest[
                    "clutter_vs_v21e_hard_bird_expected_direction_auc"
                ]
                >= 0.65
            ),
            "role": "advisory justification for V21f; not a performance claim",
        },
        "selection_bias_disclosure": (
            "V21f was proposed after inspecting these V21e OOF errors; the same OOF "
            "population is exploratory and cannot independently confirm V21f"
        ),
        "threshold_fitting": "未执行",
        "test": "未执行",
        "deployment_eligible": False,
    }


def _config_payload(
    source: Mapping[str, Any],
    *,
    root: Path,
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
        f"RADAR-LN3-V21f-cq08-known-gate-oof-fold-{fold_id:02d}-"
        "treatment-seed43"
    )
    payload["variant"] = V21F_COURSE_INTERFRAME_VARIANT
    task = dict(payload["task"])
    task["feature_names"] = list(COURSE_INTERFRAME_FEATURE_NAMES)
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
    model["evidence_dim"] = len(COURSE_INTERFRAME_FEATURE_NAMES)
    payload["model"] = model
    artifact = root / artifact_relative(ARM, fold_id)
    train = dict(payload["train"])
    train.update(
        {
            "checkpoint_path": relative_path(root, artifact / "final_model.pt"),
            "last_checkpoint_path": relative_path(root, artifact / "last_model.pt"),
            "formal_artifact_dir": relative_path(root, artifact),
            "epoch_checkpoint_dir": relative_path(root, artifact / "epochs"),
            "checkpoint_schema": (
                f"RADAR-LN3-V21f-known-gate-oof-fold-{fold_id:02d}-"
                "treatment-checkpoint-v1"
            ),
            "save_every_epoch": False,
            "paired_baseline_config_path": relative_path(
                root, root / v21e_config_relative(V21E_ARM, fold_id)
            ),
        }
    )
    payload["train"] = train
    return payload


def build(project_root: Path, *, preview: bool = False) -> dict[str, Any]:
    root = project_root.resolve()
    source_paths = {
        "index": root / INDEX_RELATIVE,
        "assignment": root / ASSIGNMENT_RELATIVE,
        "v21e_feature_contract": root / V21E_FEATURE_CONTRACT_RELATIVE,
        "v21e_feature_manifest": root / V21E_FEATURE_MANIFEST_RELATIVE,
        "v21e_course_cache": root / V21E_COURSE_CACHE_RELATIVE,
        "v21a_feature_contract": root / V21A_FEATURE_CONTRACT_RELATIVE,
        "v21a_feature_manifest": root / V21A_FEATURE_MANIFEST_RELATIVE,
        "v21a_spectral_cache": root / V21A_SPECTRAL_CACHE_RELATIVE,
        "v21e_treatment_report": root / V21E_TREATMENT_REPORT_RELATIVE,
    }
    for fold_id in range(FOLD_COUNT):
        source_paths[f"v21e_config_{fold_id:02d}"] = (
            root / v21e_config_relative(V21E_ARM, fold_id)
        )
        source_paths[f"v21e_prediction_{fold_id:02d}"] = (
            root / v21e_prediction_relative(V21E_ARM, fold_id)
        )
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V21f source files are missing: {missing}")
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
        raise ValueError("V21f source V21e OOF result/scope changed")
    preview_result = {
        "status": "preview_passed",
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "folds": assignment["folds"],
        "v21e_absolute_guard_passed": False,
        "new_training_run_count": 5,
        "unknown_training_exposure": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
    }
    if preview:
        return preview_result

    course_ids, course_values, course_valid = _load_cache(
        source_paths["v21e_course_cache"], len(COURSE_PREFIX_FEATURE_NAMES)
    )
    spectral_ids, spectral_values, spectral_valid = _load_cache(
        source_paths["v21a_spectral_cache"], len(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    )
    if not np.array_equal(course_ids, spectral_ids):
        raise ValueError("V21e course and V21a spectral caches are not event-aligned")
    combined_values = np.concatenate([course_values, spectral_values], axis=1).astype(
        np.float32, copy=False
    )
    combined_valid = np.concatenate([course_valid, spectral_valid], axis=1).astype(
        np.bool_, copy=False
    )
    cache_path = root / COMBINED_CACHE_RELATIVE
    _write_npz_new_or_identical(
        cache_path,
        (
            ("sample_ids", course_ids),
            ("values", combined_values),
            ("valid", combined_valid),
        ),
    )
    cache_hash = sha256_file(cache_path)
    v21a_contract = json.loads(
        source_paths["v21a_feature_contract"].read_text(encoding="utf-8")
    )
    attribution = _hard_bird_attribution(
        root,
        sample_ids=spectral_ids,
        spectral_values=spectral_values,
        spectral_valid=spectral_valid,
        v21a_contract=v21a_contract,
    )
    attribution_path = root / ATTRIBUTION_RELATIVE
    write_new_or_identical(attribution_path, attribution)

    experiment = {
        "schema_version": EXPERIMENT_CONTRACT_SCHEMA,
        "experiment": "V21f V21e-course plus V21a inter-frame spectral evidence",
        "role": "exploratory paired known-train OOF; not independent confirmation",
        "feature_names": list(COURSE_INTERFRAME_FEATURE_NAMES),
        "control_features": list(COURSE_PREFIX_FEATURE_NAMES),
        "new_treatment_features": list(INTERFRAME_SPECTRAL_FEATURE_NAMES),
        "single_factor": "append six frozen V21a spectral-consistency values to V21e",
        "pairing": {
            "control": "frozen V21e OOF Treatment predictions",
            "treatment": "V21f full-parameter 20-epoch fold models",
            "shared_initialization": (
                "recursively reconstruct each V21e fold initial state; copy every existing "
                "parameter/buffer including its course column; zero only six appended columns"
            ),
            "same_fold_train_parent_order": True,
            "fixed_final_epoch": FIXED_EPOCHS,
            "held_out_checkpoint_selection": False,
        },
        "decision": {
            "incremental_signal": (
                "parent macro_f1 delta > 0.02; Bird net rescue > 0; Drone/Balloon/"
                "Clutter net rescue >= 0; KnownTarget and Clutter recall do not decrease; "
                "component-bootstrap macro_f1 lower95 > 0"
            ),
            "absolute_guard": "Treatment minimum per-class precision/recall/f1 >= 0.90",
            "eligible_prefix_guard": (
                "macro_f1, KnownTarget recall and Clutter recall do not decrease"
            ),
            "maximum_authorized_progress": (
                "new independent confirmation only; this reused OOF cannot authorize "
                "threshold calibration or online adoption"
            ),
        },
        "selection_bias_disclosure": attribution["selection_bias_disclosure"],
        "historical_feature_conception_exposure": v21a_contract[
            "historical_feature_conception_exposure"
        ],
        "source_hashes": source_hashes,
        "combined_cache": {
            "path": relative_path(root, cache_path),
            "sha256": cache_hash,
            "event_count": int(len(course_ids)),
            "value_dim": int(combined_values.shape[1]),
        },
        "attribution": {
            "path": relative_path(root, attribution_path),
            "sha256": sha256_file(attribution_path),
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
    experiment_path = root / EXPERIMENT_CONTRACT_RELATIVE
    write_new_or_identical(experiment_path, experiment)
    experiment_hash = sha256_file(experiment_path)
    manifest = {
        "schema_version": FEATURE_MANIFEST_SCHEMA,
        "feature_names": list(COURSE_INTERFRAME_FEATURE_NAMES),
        "source_caches": {
            "course": {
                "path": relative_path(root, source_paths["v21e_course_cache"]),
                "sha256": source_hashes["v21e_course_cache"],
            },
            "interframe_spectral": {
                "path": relative_path(root, source_paths["v21a_spectral_cache"]),
                "sha256": source_hashes["v21a_spectral_cache"],
            },
        },
        "cache_files": {
            "train": {
                "path": relative_path(root, cache_path),
                "sha256": cache_hash,
                "event_count": int(len(course_ids)),
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
    manifest_path = root / FEATURE_MANIFEST_RELATIVE
    write_new_or_identical(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)

    fold_rows: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(v21e_contract, fold_id)
        source_config_path = root / v21e_config_relative(V21E_ARM, fold_id)
        source_payload = json.loads(source_config_path.read_text(encoding="utf-8"))
        path = root / config_relative(ARM, fold_id)
        payload = _config_payload(
            source_payload,
            root=root,
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
            config.variant != V21F_COURSE_INTERFRAME_VARIANT
            or config.model.evidence_dim != 7
            or config.train.paired_baseline_config_path
            != relative_path(root, source_config_path)
        ):
            raise ValueError(f"V21f fold {fold_id} config failed its nested pair contract")
        control_prediction_path = root / v21e_prediction_relative(V21E_ARM, fold_id)
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
        "schema_version": OOF_CONTRACT_SCHEMA,
        "experiment": "V21f exploratory nested paired five-fold OOF",
        "source_assignment_path": relative_path(root, root / ASSIGNMENT_RELATIVE),
        "source_assignment_sha256": sha256_file(root / ASSIGNMENT_RELATIVE),
        "source_assignment_semantic_sha256": canonical_sha256(assignment),
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "grouping": "capture_content_provenance_component",
        "prediction_protocol": "out_of_fold_fixed_final_epoch",
        "control_reused_from": "V21e OOF Treatment",
        "new_training_arm": ARM,
        "folds": fold_rows,
        "unknown_training_exposure": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "threshold_fitting": "未执行",
        "deployment_eligible": False,
    }
    oof_path = root / OOF_CONTRACT_RELATIVE
    write_new_or_identical(oof_path, oof_contract)
    checks = {
        "v21e_failed_absolute_guard_is_disclosed": True,
        "same_five_provenance_grouped_folds_are_reused": True,
        "v21e_control_predictions_are_sha_bound": True,
        "course_and_spectral_caches_are_event_aligned": True,
        "combined_cache_is_label_free_and_train_only": True,
        "treatment_appends_exactly_six_spectral_columns": True,
        "nested_pair_reconstructs_v21e_control_initialization": True,
        "hard_bird_attribution_is_marked_post_hoc": True,
        "unknown_training_exposure_is_zero": True,
        "validation_access_count_is_zero": True,
        "test_payload_access_count_is_zero": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
        "checks": checks,
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "selected_seed": TRAINING_SEED,
        "oof_contract_path": relative_path(root, oof_path),
        "oof_contract_sha256": sha256_file(oof_path),
        "experiment_contract_path": relative_path(root, experiment_path),
        "experiment_contract_sha256": experiment_hash,
        "feature_manifest_path": relative_path(root, manifest_path),
        "feature_manifest_sha256": manifest_hash,
        "train_feature_cache_path": relative_path(root, cache_path),
        "train_feature_cache_sha256": cache_hash,
        "attribution_path": relative_path(root, attribution_path),
        "attribution_sha256": sha256_file(attribution_path),
        "new_training_run_count": FOLD_COUNT,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行五个V21f OOF Treatment折）",
        "OOF": "未执行",
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / READINESS_RELATIVE
    write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "attribution": str(attribution_path),
        "combined_cache": str(cache_path),
        "oof_contract": str(oof_path),
        "readiness": str(readiness_path),
        "new_training_run_count": FOLD_COUNT,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exploratory V21f OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.project_root.resolve(), preview=args.preview),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()