"""Prepare V24c's strict same-second recurrent-state-skip diagnostic.

V24c changes exactly one dynamical rule relative to the completed V24
KnownGate development baseline.  A causal event whose frozen V16
``tr_dt_reason`` is ``zero_delta_at_second_precision`` is still observed by
the input fusion, but it restores the recurrent state that entered that step.
It does not rebuild features or caches and never uses V24's trained weights as
the treatment initialization source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from build.known_gate_v24 import (
    CORE_CONTRACT,
    CORE_NORMALIZATION,
    EXPECTED_CACHE_EVENT_COUNTS,
    SOURCE_INDEX,
    TRAIN_PARENT_COUNT,
    V21E_TRAIN_CACHE,
    V21E_VALIDATION_CACHE,
    VALIDATION_CLUTTER_PARENT_SUPPORT,
    VALIDATION_PARENT_COUNT,
    _cache_metadata,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V24C_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V24C_MANIFEST_SCHEMA,
    KNOWN_GATE_V24C_READINESS_SCHEMA,
    KNOWN_LABELS,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_SCHEMA,
    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
)


CONTROL_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-baseline-"
    "seed43-config.json"
)
CONTROL_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V24-cq08-known-gate-course-reason-aware-baseline-contract.json"
)
CONTROL_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V24-cq08-known-gate-course-reason-aware-baseline-"
    "feature-cache-manifest.json"
)
CONTROL_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-training-readiness.json"
)
CONTROL_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-development.json"
)
CONTROL_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-"
    "baseline-seed43-formal"
)
EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v24c_strict_same_second_skip_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V24c-cq08-known-gate-course-strict-same-second-skip-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V24c-cq08-known-gate-course-strict-same-second-skip-"
    "feature-cache-manifest.json"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V24c-cq08-liquid-known-gate-course-strict-same-second-skip-"
    "seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V24c-cq08-liquid-known-gate-course-strict-same-second-"
    "skip-seed43-formal"
)

SEED = 43
ZERO_DELTA_REASON = "zero_delta_at_second_precision"
FIRST_EVENT_REASON = "first_event_no_predecessor"
POSITIVE_DELTA_REASON = "positive_delta_at_second_precision"
SPLITS = ("train", "val")


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite a different V24c preparation artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "created"


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_completed_control(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    control_config = json.loads(paths["control_config"].read_text(encoding="utf-8"))
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    readiness = json.loads(paths["control_readiness"].read_text(encoding="utf-8"))
    report = json.loads(paths["control_report"].read_text(encoding="utf-8"))
    contract = json.loads(paths["control_contract"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["control_manifest"].read_text(encoding="utf-8"))
    cache_files = dict(manifest.get("cache_files") or {})
    train_cache = dict(cache_files.get("train") or {})
    validation_cache = dict(cache_files.get("val") or {})
    if (
        control_config.get("schema_version") != "RADAR-LN3-V24"
        or control_config.get("variant")
        != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or str(dict(control_config.get("model") or {}).get("liquid_invalid_dt_update_mode"))
        != "bounded_observation"
        or summary.get("status") != "completed"
        or summary.get("variant") != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or int(summary.get("seed", -1)) != SEED
        or len(list(summary.get("history") or ())) != 20
        or not str(summary.get("initial_model_state_sha256", ""))
        or readiness.get("status") != "passed"
        or report.get("status") != "completed_development_baseline_only"
        or contract.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or dict(contract.get("architecture") or {}).get("zero_dt_mode")
        != "bounded_observation"
        or manifest.get("feature_schema") != COURSE_PREFIX_FEATURE_SCHEMA
        or manifest.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or train_cache.get("path") != V21E_TRAIN_CACHE.as_posix()
        or validation_cache.get("path") != V21E_VALIDATION_CACHE.as_posix()
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V24c requires a completed sealed V24 bounded-observation control")
    return control_config, summary


def _same_second_statistics(index: pd.DataFrame, *, split: str) -> dict[str, Any]:
    selected = index[
        index["split_name"].astype(str).eq(split)
        & index["label_name"].astype(str).isin(KNOWN_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    expected_parent_count = (
        TRAIN_PARENT_COUNT if split == "train" else VALIDATION_PARENT_COUNT
    )
    expected_event_count = EXPECTED_CACHE_EVENT_COUNTS[split]
    if (
        len(selected) != expected_event_count
        or int(selected["parent_sample_id"].nunique()) != expected_parent_count
        or selected["sample_id"].astype(str).duplicated().any()
    ):
        raise ValueError(f"V24c {split} index scope differs from frozen V24 cache")
    required_reasons = {
        FIRST_EVENT_REASON,
        ZERO_DELTA_REASON,
        POSITIVE_DELTA_REASON,
    }
    if not set(selected["tr_dt_reason"].astype(str)).issubset(required_reasons):
        raise ValueError(f"V24c {split} index contains an unsupported dt reason")
    parent_rows: list[dict[str, Any]] = []
    for parent_id, parent in selected.groupby("parent_sample_id", sort=True):
        parent = parent.sort_values("causal_event_index", kind="stable")
        labels = parent["label_name"].astype(str).unique().tolist()
        reasons = parent["tr_dt_reason"].astype(str).tolist()
        if (
            len(labels) != 1
            or reasons[0] != FIRST_EVENT_REASON
            or reasons.count(FIRST_EVENT_REASON) != 1
        ):
            raise ValueError(f"V24c {split} parent reason sequence is invalid: {parent_id}")
        ordinal = 0
        maximum_ordinal = 0
        zero_count = 0
        for reason in reasons:
            if reason in {FIRST_EVENT_REASON, POSITIVE_DELTA_REASON}:
                ordinal = 1
            elif reason == ZERO_DELTA_REASON:
                ordinal += 1
                zero_count += 1
                maximum_ordinal = max(maximum_ordinal, ordinal)
            else:
                raise ValueError(f"V24c unsupported reason in parent {parent_id}")
        parent_rows.append(
            {
                "parent_sample_id": str(parent_id),
                "label_name": labels[0],
                "event_count": int(len(parent)),
                "zero_delta_event_count": int(zero_count),
                "has_same_second_followup": bool(zero_count > 0),
                "maximum_same_second_ordinal": int(maximum_ordinal),
            }
        )
    if len(parent_rows) != expected_parent_count:
        raise AssertionError("V24c parent summary count differs from frozen split")
    parent_frame = pd.DataFrame(parent_rows)
    labels = list(KNOWN_LABELS)
    by_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        parents = parent_frame[parent_frame["label_name"].eq(label)]
        events = selected[selected["label_name"].astype(str).eq(label)]
        if parents.empty:
            raise ValueError(f"V24c {split} lacks known label {label}")
        by_label[label] = {
            "parent_support": int(len(parents)),
            "parents_with_same_second_followup": int(
                parents["has_same_second_followup"].sum()
            ),
            "parents_without_same_second_followup": int(
                (~parents["has_same_second_followup"]).sum()
            ),
            "event_support": int(len(events)),
            "same_second_event_count": int(
                events["tr_dt_reason"].astype(str).eq(ZERO_DELTA_REASON).sum()
            ),
            "same_second_event_fraction": float(
                events["tr_dt_reason"].astype(str).eq(ZERO_DELTA_REASON).mean()
            ),
            "maximum_same_second_ordinal": int(
                parents["maximum_same_second_ordinal"].max()
            ),
        }
    contingency = np.asarray(
        [
            [
                int(
                    (
                        (parent_frame["label_name"] == label)
                        & (~parent_frame["has_same_second_followup"])
                    ).sum()
                ),
                int(
                    (
                        (parent_frame["label_name"] == label)
                        & parent_frame["has_same_second_followup"]
                    ).sum()
                ),
            ]
            for label in labels
        ],
        dtype=np.float64,
    )
    total = float(contingency.sum())
    expected = contingency.sum(axis=1, keepdims=True) * contingency.sum(
        axis=0, keepdims=True
    ) / total
    valid_cells = expected > 0.0
    chi_square = float(
        np.sum(((contingency[valid_cells] - expected[valid_cells]) ** 2) / expected[valid_cells])
    )
    denominator = total * min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = float(np.sqrt(chi_square / denominator)) if denominator > 0.0 else 0.0
    target_present = int(
        parent_frame[
            parent_frame["label_name"].isin(
                ["DroneTarget", "BirdTarget", "BalloonTarget"]
            )
            & parent_frame["has_same_second_followup"]
        ].shape[0]
    )
    any_present = int(parent_frame["has_same_second_followup"].sum())
    no_same_second_target = int(
        parent_frame[
            parent_frame["label_name"].isin(
                ["DroneTarget", "BirdTarget", "BalloonTarget"]
            )
            & (~parent_frame["has_same_second_followup"])
        ].shape[0]
    )
    no_same_second_clutter = int(
        parent_frame[
            parent_frame["label_name"].eq("ClutterTarget")
            & (~parent_frame["has_same_second_followup"])
        ].shape[0]
    )
    return {
        "split": split,
        "descriptive_only_not_a_readiness_performance_gate": True,
        "parent_count": int(len(parent_frame)),
        "event_count": int(len(selected)),
        "by_base_label": by_label,
        "parent_label_by_has_same_second_followup_contingency_rows_no_yes": {
            label: [int(value) for value in row]
            for label, row in zip(labels, contingency.astype(np.int64), strict=True)
        },
        "label_association": {
            "method": "Cramers_V on base_label x parent_has_same_second_followup",
            "cramers_v": cramers_v,
            "chi_square": chi_square,
            "parent_support": int(total),
            "interpretation": "descriptive sampling/measurement association only; not a model input or readiness pass/fail criterion",
        },
        "reason2_present_implies_known_target": {
            "same_second_parent_count": any_present,
            "known_target_parent_count": target_present,
            "precision": float(target_present / any_present) if any_present else None,
        },
        "absence_is_not_sufficient_for_clutter": {
            "known_target_parent_count_without_same_second": no_same_second_target,
            "clutter_parent_count_without_same_second": no_same_second_clutter,
            "holds": bool(no_same_second_target > 0 and no_same_second_clutter > 0),
        },
    }


def _config_payload(
    control_payload: Mapping[str, Any],
    *,
    manifest_hash: str,
    contract_hash: str,
    train_cache_hash: str,
    validation_cache_hash: str,
) -> dict[str, Any]:
    payload = _clone(control_payload)
    artifact = ARTIFACT.as_posix()
    payload["schema_version"] = "RADAR-LN3-V24c"
    payload["config_name"] = (
        "RADAR-LN3-V24c-cq08-liquid-known-gate-course-strict-same-second-skip-seed43"
    )
    payload["variant"] = V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT
    payload["task"]["feature_names"] = list(COURSE_PREFIX_FEATURE_NAMES)
    payload["data"].update(
        {
            "feature_cache_manifest_path": FEATURE_MANIFEST.as_posix(),
            "expected_feature_cache_manifest_sha256": manifest_hash,
            "train_feature_cache_path": V21E_TRAIN_CACHE.as_posix(),
            "validation_feature_cache_path": V21E_VALIDATION_CACHE.as_posix(),
            "expected_train_feature_cache_sha256": train_cache_hash,
            "expected_validation_feature_cache_sha256": validation_cache_hash,
            "feature_experiment_contract_path": EXPERIMENT_CONTRACT.as_posix(),
            "expected_feature_experiment_contract_sha256": contract_hash,
        }
    )
    payload["model"]["evidence_dim"] = len(COURSE_PREFIX_FEATURE_NAMES)
    payload["model"]["liquid_invalid_dt_update_mode"] = "strict_skip"
    payload["train"].update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V24c-v24c_course_strict_same_second_skip-seed43-"
                "known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": CONTROL_CONFIG.as_posix(),
        }
    )
    return payload


def _initialization_audit(
    root: Path,
    config_path: Path,
    control_summary: Mapping[str, Any],
) -> dict[str, Any]:
    treatment_config = load_known_gate_config(config_path)
    treatment, control, pairing = build_known_gate_model_for_config(root, treatment_config)
    if control is None:
        raise AssertionError("V24c is missing its reconstructed V24 paired control")
    treatment_state = _state_dict_sha256(treatment.state_dict())
    control_state = _state_dict_sha256(control.state_dict())
    if (
        treatment_config.variant != V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT
        or treatment_config.model.liquid_invalid_dt_update_mode != "strict_skip"
        or control.config.variant != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or control.config.model.liquid_invalid_dt_update_mode != "bounded_observation"
        or treatment_state != control_state
        or control_state != str(control_summary.get("initial_model_state_sha256", ""))
        or pairing.get("paired_baseline") is not True
        or pairing.get("dynamics_only_pair") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 0
        or pairing.get("new_fusion_columns_exactly_zero") is not True
    ):
        raise AssertionError("V24c paired initialization differs beyond strict same-second dynamics")
    del treatment, control
    return {
        "v24_control_initial_model_state_sha256": control_state,
        "v24c_initial_model_state_sha256": treatment_state,
        "state_dicts_exactly_equal": True,
        "only_dynamics_mode_differs": True,
        "control_mode": "bounded_observation",
        "treatment_mode": "strict_skip",
        "new_fusion_columns": 0,
        "trained_v24_checkpoint_loaded_for_initialization": False,
        "full_parameter_training": True,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paths = {
        "index": root / SOURCE_INDEX,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "train_cache": root / V21E_TRAIN_CACHE,
        "validation_cache": root / V21E_VALIDATION_CACHE,
        "control_config": root / CONTROL_CONFIG,
        "control_contract": root / CONTROL_CONTRACT,
        "control_manifest": root / CONTROL_MANIFEST,
        "control_readiness": root / CONTROL_READINESS,
        "control_report": root / CONTROL_REPORT,
        "control_summary": root / CONTROL_ARTIFACT / "training-summary.json",
        "control_checkpoint": root / CONTROL_ARTIFACT / "best_model.pt",
        "experiment_document": root / EXPERIMENT_DOCUMENT,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V24c frozen source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    control_payload, control_summary = _require_completed_control(paths)
    cache_stats = {
        "train": _cache_metadata(paths["train_cache"], split="train"),
        "val": _cache_metadata(paths["validation_cache"], split="val"),
    }
    index = pd.read_csv(paths["index"], low_memory=False)
    if set(index["schema_version"].astype(str)) != {"RADAR-LN3-V16"}:
        raise ValueError("V24c source index schema differs from the frozen V16 contract")
    same_second = {
        split: _same_second_statistics(index, split=split) for split in SPLITS
    }
    train_clutter = same_second["train"]["by_base_label"]["ClutterTarget"]
    val_clutter = same_second["val"]["by_base_label"]["ClutterTarget"]
    if (
        train_clutter["parents_with_same_second_followup"] != 0
        or val_clutter["parents_with_same_second_followup"] != 0
        or not same_second["train"]["absence_is_not_sufficient_for_clutter"]["holds"]
    ):
        raise ValueError("V24c same-second support audit no longer matches the frozen diagnostic premise")

    contract = {
        "schema_version": KNOWN_GATE_V24C_EXPERIMENT_SCHEMA,
        "experiment_name": "V24c strict same-second recurrent-state-skip diagnostic",
        "internal_variant": V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
        "scope": "development-only KnownGate L1 dynamics diagnostic; no product action claim",
        "fresh_training_not_finetune": True,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_document": _relative(root, paths["experiment_document"]),
        "experiment_document_sha256": hashes["experiment_document"],
        "control": {
            "config": _relative(root, paths["control_config"]),
            "config_sha256": hashes["control_config"],
            "summary": _relative(root, paths["control_summary"]),
            "summary_sha256": hashes["control_summary"],
            "checkpoint": _relative(root, paths["control_checkpoint"]),
            "checkpoint_sha256": hashes["control_checkpoint"],
            "readiness": _relative(root, paths["control_readiness"]),
            "readiness_sha256": hashes["control_readiness"],
            "report": _relative(root, paths["control_report"]),
            "report_sha256": hashes["control_report"],
            "initial_model_state_sha256": str(
                control_summary["initial_model_state_sha256"]
            ),
            "trained_checkpoint_loaded_for_initialization": False,
        },
        "dynamics_variable": {
            "only_changed_model_field": "liquid_invalid_dt_update_mode",
            "control_mode": "bounded_observation",
            "control_zero_delta_update": "alpha=0.25 then recurrent LayerNorm",
            "treatment_mode": "strict_skip",
            "treatment_zero_delta_update": (
                "compute candidate only transiently, then restore exact incoming recurrent state "
                "after recurrent LayerNorm"
            ),
            "zero_delta_event_feature_visibility": (
                "current fusion event is constructed but cannot change the recurrent state "
                "or that event's recurrent readout"
            ),
            "first_event": "unchanged alpha=1 initialization",
            "positive_delta": "unchanged continuous-time update",
            "padding": "unchanged no-update",
            "legacy_v20_skip_is_not_reused": True,
        },
        "data_reuse": {
            "index": _relative(root, paths["index"]),
            "index_sha256": hashes["index"],
            "core_contract": _relative(root, paths["core_contract"]),
            "core_contract_sha256": hashes["core_contract"],
            "normalization": _relative(root, paths["normalization"]),
            "normalization_sha256": hashes["normalization"],
            "known_train_parent_count": TRAIN_PARENT_COUNT,
            "known_validation_parent_count": VALIDATION_PARENT_COUNT,
            "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
            "unknown_training_exposure": 0,
        },
        "core_data_contract_sha256": hashes["core_contract"],
        "course_cache_reuse": {
            "train_cache": _relative(root, paths["train_cache"]),
            "train_cache_sha256": hashes["train_cache"],
            "validation_cache": _relative(root, paths["validation_cache"]),
            "validation_cache_sha256": hashes["validation_cache"],
            "cache_values_rebuilt": False,
            "course_feature_count": 1,
            "raw_echo_payload_access_count": 0,
            "mat_payload_access_count": 0,
            "rd_tensor_payload_access_count": 0,
            "test_payload_access_count": 0,
        },
        "same_second_support_diagnostic": same_second,
        "training": {
            "seed": SEED,
            "dataloader_seed": SEED,
            "epochs": 20,
            "batch_size": 2,
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": 0.1,
            "checkpoint_selection": "parent_min_precision_recall",
            "class_weight_normalization": "batch_weight_sum",
            "all_non_dynamics_training_hyperparameters_frozen_to_v24": True,
        },
        "evaluation": {
            "primary": (
                "diagnose whether same-second recurrent-state accumulation carries useful "
                "or shortcut information under the frozen V24 KnownGate protocol"
            ),
            "required_reports": [
                "parent_final",
                "eligible_prefix",
                "parent_balanced_eligible_prefix",
                "window",
                "clutter_Wilson_intervals_with_support_9",
                "paired_parent_bootstrap",
                "same_second_parent_and_window_strata",
                "track_length_and_capture_strata",
                "per_parent_changes",
            ],
            "validation_used_for_readiness_gate": False,
            "primary_hypothesis_labels": [
                "DroneTarget",
                "BirdTarget",
                "BalloonTarget",
                "ClutterTarget",
            ],
            "safety_labels": [],
        },
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    manifest = {
        "schema_version": KNOWN_GATE_V24C_MANIFEST_SCHEMA,
        "feature_schema": COURSE_PREFIX_FEATURE_SCHEMA,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "source_control_manifest": CONTROL_MANIFEST.as_posix(),
        "source_control_manifest_sha256": hashes["control_manifest"],
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": hashes["core_contract"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": hashes["index"],
        "cache_files": {
            "train": {
                "path": V21E_TRAIN_CACHE.as_posix(),
                "sha256": hashes["train_cache"],
                "fields": ["sample_ids", "values", "valid"],
                **cache_stats["train"],
            },
            "val": {
                "path": V21E_VALIDATION_CACHE.as_posix(),
                "sha256": hashes["validation_cache"],
                "fields": ["sample_ids", "values", "valid"],
                **cache_stats["val"],
            },
        },
        "cache_values_rebuilt": False,
        "sample_id_is_model_input": False,
        "valid_mask_is_model_input": False,
        "validation_label_values_used_for_readiness_gate": False,
        "unknown_excluded": True,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    _write_new_or_identical(root / FEATURE_MANIFEST, manifest)
    manifest_hash = sha256_file(root / FEATURE_MANIFEST)
    config_payload = _config_payload(
        control_payload,
        manifest_hash=manifest_hash,
        contract_hash=contract_hash,
        train_cache_hash=hashes["train_cache"],
        validation_cache_hash=hashes["validation_cache"],
    )
    _write_new_or_identical(root / CONFIG, config_payload)
    initialization = _initialization_audit(root, root / CONFIG, control_summary)
    checks = {
        "completed_v24_control_is_sha_bound_and_not_loaded_as_training_initialization": True,
        "only_liquid_invalid_dt_update_mode_differs_from_v24": True,
        "strict_skip_restores_state_after_recurrent_layer_norm": True,
        "first_positive_and_padding_dt_semantics_remain_frozen": True,
        "v24_course_cache_is_reused_byte_for_byte_without_rebuild": True,
        "v24_control_and_v24c_initial_state_dicts_are_exactly_equal": True,
        "same_second_support_is_descriptive_not_a_readiness_performance_gate": True,
        "same_second_parent_support_is_zero_for_clutter_in_train_and_validation": True,
        "absence_of_same_second_followup_is_not_sufficient_for_clutter": True,
        "unknown_is_excluded_and_test_payload_access_is_zero": True,
        "validation_values_are_not_a_readiness_gate": True,
        "full_20_epoch_training_is_required": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": KNOWN_GATE_V24C_READINESS_SCHEMA,
        "variant": V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
        "checks": checks,
        "config": {
            "path": _relative(root, root / CONFIG),
            "sha256": sha256_file(root / CONFIG),
        },
        "feature_manifest_path": _relative(root, root / FEATURE_MANIFEST),
        "feature_manifest_sha256": manifest_hash,
        "contract_path": _relative(root, root / EXPERIMENT_CONTRACT),
        "contract_sha256": contract_hash,
        "contract_hashes": {
            "index_sha256": hashes["index"],
            "core_contract_sha256": hashes["core_contract"],
            "normalization_sha256": hashes["normalization"],
            "feature_manifest_sha256": manifest_hash,
            "train_feature_cache_sha256": hashes["train_cache"],
            "validation_feature_cache_sha256": hashes["validation_cache"],
            "experiment_contract_sha256": contract_hash,
        },
        "control": contract["control"],
        "initialization": initialization,
        "cache_statistics": cache_stats,
        "same_second_support_diagnostic": same_second,
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "unknown_training_exposure": 0,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": KNOWN_GATE_V24C_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_cache": str(paths["train_cache"]),
        "validation_cache": str(paths["validation_cache"]),
        "same_second_support_diagnostic": same_second,
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V24c strict same-second state-skip KnownGate diagnostic."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
