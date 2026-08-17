"""Prepare strict capture/content-grouped paired OOF support for V21e KnownGate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from build.radar_ln3_normalization import build as build_normalization
from radar_ln3.known_gate import (
    KNOWN_LABELS,
    KNOWN_TARGET_LABELS,
    UNKNOWN_LABEL,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_v21e_oof import (
    ARMS,
    ASSIGNMENT_RELATIVE,
    ASSIGNMENT_SCHEMA,
    ASSIGNMENT_SEED,
    BASE_CONTROL_CONFIG_RELATIVE,
    BASE_TREATMENT_CONFIG_RELATIVE,
    CORE_CONTRACT_RELATIVE,
    FEATURE_CONTRACT_RELATIVE,
    FEATURE_MANIFEST_RELATIVE,
    FIXED_EPOCHS,
    FOLD_CONTRACT_RELATIVE,
    FOLD_CONTRACT_SCHEMA,
    FOLD_COUNT,
    INDEX_RELATIVE,
    MULTISEED_REPORT_RELATIVE,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    SEED_SELECTION_RELATIVE,
    TRAINING_SEED,
    TRAIN_FEATURE_CACHE_RELATIVE,
    artifact_relative,
    canonical_sha256,
    config_relative,
    data_contract_relative,
    inverse_frequency_weights,
    normalization_relative,
    relative_path,
    sequence_sha256,
    sha256_file,
    write_new_or_identical,
)


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


def _fold_data_contract(
    base_contract: Mapping[str, Any],
    *,
    root: Path,
    fold_id: int,
    train_ids: list[str],
    held_out_ids: list[str],
    train_counts: Mapping[str, int],
    held_out_counts: Mapping[str, int],
    stats_path: Path,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(base_contract))
    payload["contract_name"] = f"cq08_v21e_known_gate_oof_fold_{fold_id:02d}_v1"
    data_scope = dict(payload.get("data_scope") or {})
    data_scope["oof_fold"] = {
        "fold_id": fold_id,
        "optimizer_parent_count": len(train_ids),
        "held_out_parent_count": len(held_out_ids),
        "optimizer_label_parent_counts": dict(train_counts),
        "held_out_label_parent_counts": dict(held_out_counts),
        "unknown_training_exposure": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
    }
    payload["data_scope"] = data_scope
    normalization = dict(payload.get("normalization_contract") or {})
    normalization.update(
        {
            "path": relative_path(root, stats_path),
            "sha256": sha256_file(stats_path),
            "fit_split": "train",
            "fit_population": "known_fold_train_only",
            "parent_count": len(train_ids),
            "selected_label_parent_counts": dict(stats["selected_label_parent_counts"]),
            "allowed_parent_ids_count": len(train_ids),
            "allowed_parent_ids_sha256": sequence_sha256(train_ids),
            "held_out_used": False,
            "validation_and_test_excluded": True,
        }
    )
    payload["normalization_contract"] = normalization
    training = dict(payload.get("training_contract") or {})
    training.update(
        {
            "role": "paired OOF support only; not deployment training",
            "epochs": FIXED_EPOCHS,
            "early_stopping": False,
            "checkpoint_selection": "fixed final epoch; held-out fold never selects checkpoint",
        }
    )
    payload["training_contract"] = training
    payload["oof_fold_contract"] = {
        "fold_id": fold_id,
        "grouping": "capture_content_provenance_component",
        "train_parent_count": len(train_ids),
        "held_out_parent_count": len(held_out_ids),
        "train_parent_ids_sha256": sequence_sha256(train_ids),
        "held_out_parent_ids_sha256": sequence_sha256(held_out_ids),
        "parent_overlap_count": 0,
        "component_overlap_count": 0,
        "unknown_training_exposure": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "epochs": FIXED_EPOCHS,
        "held_out_checkpoint_selection": False,
        "held_out_prediction_time": "once_after_fixed_final_epoch",
    }
    return payload


def _config_payload(
    base_payload: Mapping[str, Any],
    *,
    root: Path,
    arm: str,
    fold_id: int,
    data_contract_path: Path,
    stats_path: Path,
    stats: Mapping[str, Any],
    class_weights: tuple[float, float],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(base_payload))
    payload["config_name"] = (
        f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_id:02d}-{arm}-seed43"
    )
    data = dict(payload["data"])
    data.update(
        {
            "contract_path": relative_path(root, data_contract_path),
            "normalization_stats_path": relative_path(root, stats_path),
            "expected_contract_sha256": sha256_file(data_contract_path),
            "expected_stats_sha256": sha256_file(stats_path),
            "rd_amplitude_center": float(dict(stats["amplitude"])["center"]),
            "rd_amplitude_scale": float(dict(stats["amplitude"])["scale"]),
        }
    )
    payload["data"] = data
    artifact = root / artifact_relative(arm, fold_id)
    train = dict(payload["train"])
    train.update(
        {
            "seed": TRAINING_SEED,
            "dataloader_seed": TRAINING_SEED,
            "epochs": FIXED_EPOCHS,
            "early_stopping_patience": 0,
            "class_weights": list(class_weights),
            "checkpoint_path": relative_path(root, artifact / "final_model.pt"),
            "last_checkpoint_path": relative_path(root, artifact / "last_model.pt"),
            "formal_artifact_dir": relative_path(root, artifact),
            "epoch_checkpoint_dir": relative_path(root, artifact / "epochs"),
            "checkpoint_schema": (
                f"RADAR-LN3-V21e-known-gate-oof-fold-{fold_id:02d}-{arm}-checkpoint-v1"
            ),
            "save_every_epoch": False,
            "paired_baseline_config_path": (
                relative_path(root, root / config_relative("control", fold_id))
                if arm == "treatment"
                else None
            ),
        }
    )
    payload["train"] = train
    return payload


def build(project_root: Path, *, preview: bool = False) -> dict[str, Any]:
    root = project_root.resolve()
    required = {
        "index": root / INDEX_RELATIVE,
        "core_contract": root / CORE_CONTRACT_RELATIVE,
        "control_config": root / BASE_CONTROL_CONFIG_RELATIVE,
        "treatment_config": root / BASE_TREATMENT_CONFIG_RELATIVE,
        "feature_contract": root / FEATURE_CONTRACT_RELATIVE,
        "feature_manifest": root / FEATURE_MANIFEST_RELATIVE,
        "train_feature_cache": root / TRAIN_FEATURE_CACHE_RELATIVE,
        "multiseed_report": root / MULTISEED_REPORT_RELATIVE,
        "seed_selection": root / SEED_SELECTION_RELATIVE,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V21e OOF sources are missing: {missing}")
    control = load_known_gate_config(required["control_config"])
    treatment = load_known_gate_config(required["treatment_config"])
    selection = json.loads(required["seed_selection"].read_text(encoding="utf-8"))
    multiseed = json.loads(required["multiseed_report"].read_text(encoding="utf-8"))
    selected_checkpoint = dict(selection.get("selected_checkpoint") or {})
    selected_evidence = dict(selection.get("evidence") or {})
    selected_checkpoint_path = root / str(selected_checkpoint.get("checkpoint_path", ""))
    if (
        control.train.seed != TRAINING_SEED
        or treatment.train.seed != TRAINING_SEED
        or int(selection.get("selected_seed", -1)) != TRAINING_SEED
        or multiseed.get("status") != "passed"
        or dict(multiseed.get("decision") or {}).get(
            "advance_to_train_parent_grouped_oof"
        )
        is not True
        or dict(multiseed.get("decision") or {}).get(
            "online_known_gate_adoption_allowed"
        )
        is not False
        or selection.get("retention_decision")
        != "retain_seed43_as_best_observed_V21e_development_checkpoint"
        or not selected_checkpoint_path.is_file()
        or sha256_file(selected_checkpoint_path)
        != str(selected_checkpoint.get("checkpoint_sha256", "")).lower()
        or str(selected_evidence.get("multiseed_report_sha256", "")).lower()
        != sha256_file(required["multiseed_report"])
    ):
        raise ValueError("V21e OOF source decision is not the frozen seed43 candidate")
    if (
        sha256_file(required["index"]) != control.data.expected_index_sha256
        or sha256_file(required["core_contract"]) != control.data.expected_contract_sha256
        or sha256_file(required["feature_contract"])
        != treatment.data.expected_feature_experiment_contract_sha256
        or sha256_file(required["feature_manifest"])
        != treatment.data.expected_feature_cache_manifest_sha256
        or sha256_file(required["train_feature_cache"])
        != treatment.data.expected_train_feature_cache_sha256
    ):
        raise ValueError("V21e OOF frozen source path/hash changed")
    records = _parent_records(required["index"])
    assignment = build_assignments(records)
    preview_result = {
        "status": "preview_passed",
        "schema_version": ASSIGNMENT_SCHEMA,
        "known_parent_count": assignment["known_parent_count"],
        "provenance_component_count": assignment["provenance_component_count"],
        "folds": assignment["folds"],
        "unknown_training_exposure": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
    }
    if preview:
        return preview_result
    assignment_path = root / ASSIGNMENT_RELATIVE
    write_new_or_identical(assignment_path, assignment)
    base_contract = json.loads(required["core_contract"].read_text(encoding="utf-8"))
    base_payloads = {
        "control": json.loads(required["control_config"].read_text(encoding="utf-8")),
        "treatment": json.loads(required["treatment_config"].read_text(encoding="utf-8")),
    }
    fold_records: list[dict[str, Any]] = []
    assignments = list(assignment["assignments"])
    all_ids = sorted(str(row["parent_sample_id"]) for row in assignments)
    for fold_id in range(FOLD_COUNT):
        held_out_rows = [row for row in assignments if int(row["fold_id"]) == fold_id]
        train_rows = [row for row in assignments if int(row["fold_id"]) != fold_id]
        train_ids = sorted(str(row["parent_sample_id"]) for row in train_rows)
        held_out_ids = sorted(str(row["parent_sample_id"]) for row in held_out_rows)
        train_components = {str(row["provenance_component_id"]) for row in train_rows}
        held_out_components = {
            str(row["provenance_component_id"]) for row in held_out_rows
        }
        if (
            set(train_ids) & set(held_out_ids)
            or sorted(train_ids + held_out_ids) != all_ids
            or train_components & held_out_components
        ):
            raise ValueError(f"V21e OOF fold {fold_id} does not isolate provenance")
        train_counts = Counter(str(row["label_name"]) for row in train_rows)
        held_out_counts = Counter(str(row["label_name"]) for row in held_out_rows)
        stats_path = root / normalization_relative(fold_id)
        stats = build_normalization(
            root,
            required["index"],
            stats_path,
            "train",
            allowed_labels=LABELS,
            forbidden_labels=(UNKNOWN_LABEL,),
            allowed_parent_ids=tuple(train_ids),
            schema_version="RADAR-LN3-V21",
            stats_name=f"cq08_v21e_known_gate_oof_fold_{fold_id:02d}_train_v1",
            scope=(
                f"V16 train OOF fold {fold_id:02d}; capture/content-grouped known "
                "fold-train parents only; held-out fold, Unknown, validation and test excluded"
            ),
        )
        if (
            int(stats.get("parent_count", -1)) != len(train_ids)
            or int(stats.get("allowed_parent_ids_count", -1)) != len(train_ids)
            or stats.get("allowed_parent_ids_sha256") != sequence_sha256(train_ids)
            or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
        ):
            raise ValueError(f"V21e OOF fold {fold_id} normalization is not fold-train-only")
        data_contract_path = root / data_contract_relative(fold_id)
        data_contract = _fold_data_contract(
            base_contract,
            root=root,
            fold_id=fold_id,
            train_ids=train_ids,
            held_out_ids=held_out_ids,
            train_counts={label: int(train_counts[label]) for label in LABELS},
            held_out_counts={label: int(held_out_counts[label]) for label in LABELS},
            stats_path=stats_path,
            stats=stats,
        )
        write_new_or_identical(data_contract_path, data_contract)
        target_count = sum(int(train_counts[label]) for label in KNOWN_TARGET_LABELS)
        class_weights = inverse_frequency_weights(
            (target_count, int(train_counts["ClutterTarget"]))
        )
        config_bindings: dict[str, Any] = {}
        for arm in ARMS:
            path = root / config_relative(arm, fold_id)
            payload = _config_payload(
                base_payloads[arm],
                root=root,
                arm=arm,
                fold_id=fold_id,
                data_contract_path=data_contract_path,
                stats_path=stats_path,
                stats=stats,
                class_weights=class_weights,
            )
            write_new_or_identical(path, payload)
            loaded = load_known_gate_config(path)
            config_bindings[arm] = {
                "path": relative_path(root, path),
                "sha256": sha256_file(path),
                "identity_sha256": known_gate_config_identity_sha256(loaded),
            }
        fold_records.append(
            {
                "fold_id": fold_id,
                "train_parent_count": len(train_ids),
                "held_out_parent_count": len(held_out_ids),
                "train_parent_ids": train_ids,
                "held_out_parent_ids": held_out_ids,
                "train_parent_ids_sha256": sequence_sha256(train_ids),
                "held_out_parent_ids_sha256": sequence_sha256(held_out_ids),
                "train_component_count": len(train_components),
                "held_out_component_count": len(held_out_components),
                "component_overlap_count": 0,
                "train_label_parent_counts": {
                    label: int(train_counts[label]) for label in LABELS
                },
                "held_out_label_parent_counts": {
                    label: int(held_out_counts[label]) for label in LABELS
                },
                "normalization_stats_path": relative_path(root, stats_path),
                "normalization_stats_sha256": sha256_file(stats_path),
                "data_contract_path": relative_path(root, data_contract_path),
                "data_contract_sha256": sha256_file(data_contract_path),
                "class_weights": list(class_weights),
                "configs": config_bindings,
                "unknown_training_exposure": 0,
                "validation_parent_count": 0,
                "test_parent_count": 0,
            }
        )
    contract = {
        "schema_version": FOLD_CONTRACT_SCHEMA,
        "experiment": "V21e KnownGate paired five-fold OOF",
        "fit_split": "train",
        "fit_population": "known_only",
        "grouping": "capture_content_provenance_component",
        "prediction_unit": "parent",
        "prediction_protocol": "out_of_fold_fixed_final_epoch",
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "epochs_per_fold": FIXED_EPOCHS,
        "early_stopping": False,
        "held_out_checkpoint_selection": False,
        "held_out_prediction_time": "once_after_fixed_final_epoch",
        "assignment_path": relative_path(root, assignment_path),
        "assignment_sha256": sha256_file(assignment_path),
        "source_bindings": {
            name: {
                "path": relative_path(root, path),
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
        "folds": fold_records,
        "formal_training": "未执行（OOF折模型不是正式部署模型）",
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / FOLD_CONTRACT_RELATIVE
    write_new_or_identical(contract_path, contract)
    checks = {
        "source_multiseed_decision_allows_oof": True,
        "selected_seed_is_43": True,
        "known_train_parent_count_is_986": True,
        "provenance_component_count_is_975": True,
        "five_folds_are_label_balanced": True,
        "no_provenance_component_crosses_folds": True,
        "fold_specific_normalization_excludes_held_out": True,
        "unknown_training_exposure_is_zero": True,
        "validation_parent_assignment_is_zero": True,
        "test_parent_assignment_is_zero": True,
        "fixed_final_epoch_without_held_out_selection": True,
        "control_treatment_configs_are_fold_paired": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
        "checks": checks,
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "assignment_path": relative_path(root, assignment_path),
        "assignment_sha256": sha256_file(assignment_path),
        "fold_contract_path": relative_path(root, contract_path),
        "fold_contract_sha256": sha256_file(contract_path),
        "selected_seed": TRAINING_SEED,
        "treatment_first_futility_gate": {
            "min_per_class_precision_recall_f1": 0.9,
            "if_failed": "stop_before_control_training",
        },
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行五折流水线）",
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
        "assignment": str(assignment_path),
        "fold_contract": str(contract_path),
        "readiness": str(readiness_path),
        "folds": assignment["folds"],
        "training_run_count_if_full_pair": FOLD_COUNT * len(ARMS),
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V21e KnownGate paired OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, preview=args.preview), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
