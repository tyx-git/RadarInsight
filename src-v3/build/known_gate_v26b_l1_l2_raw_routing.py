"""Prepare V26b's frozen L1+L2 raw-routing compatibility audit.

V26b is intentionally not a training experiment.  It binds the frozen V26
KnownGate and V18r DBB-specialist endpoints, produces a label-free mixed-label
validation feature cache for L2, and writes an *unfitted* L3 registry.  The
registry has no thresholds, so V26b can audit raw end-to-end routing but cannot
claim calibrated Unknown rejection or final product readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _CanonicalFrameProvenance,
    _as_bool,
    _write_json_new_or_identical,
    _write_npz_new_or_identical,
)
from build.micro_doppler_v18 import _build_split
from radar_ln3.known_gate import sha256_file
from radar_ln3.l1_l2_raw_routing import MIXED_OPEN_SET_LABELS
from radar_ln3.semantic_decision_registry import (
    SEMANTIC_DECISION_REGISTRY_SCHEMA,
    UNFITTED_CALIBRATION_STATUS,
    validate_semantic_decision_registry,
)
from radar_ln3.task_models import load_task_config


V26_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-"
    "seed43-config.json"
)
V26_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-"
    "readiness.json"
)
V26_SUMMARY = Path(
    "artifacts/RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-"
    "baseline-seed43-formal/training-summary.json"
)
V26_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-"
    "baseline-seed43-formal/best_model.pt"
)
V26_EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-known-only-open-set-baseline-contract.json"
)
V26_AUDIT_VALIDATION_CACHE = Path(
    "data/processed/RADAR-LN3-V26-cq08-known-gate-val-mixed-open-set-course-cache.npz"
)

V18R_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-micro-doppler8-"
    "stable-density-treatment-seed42-config.json"
)
V18R_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18r-stable8-density-weighted-training-readiness.json"
)
V18R_SUMMARY = Path(
    "artifacts/RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-"
    "micro-doppler8-stable-density-treatment-seed42-formal/training-summary.json"
)
V18R_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-"
    "micro-doppler8-stable-density-treatment-seed42-formal/best_model.pt"
)
V18R_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18r-stable8-density-weighted-paired-validation.json"
)
V18_CACHE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V18-cq08-dbb-broadband-multiscale-"
    "micro-doppler8-stable-cache-manifest.json"
)

SOURCE_INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
SOURCE_PROVENANCE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
SOURCE_CANONICAL_FRAME_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)

V26B_CONTRACT = Path(
    "data/processed/RADAR-LN3-V26b-cq08-frozen-l1-l2-raw-routing-contract.json"
)
V26B_CACHE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V26b-cq08-l2-mixed-open-set-feature-cache-manifest.json"
)
V26B_L3_REGISTRY = Path(
    "data/processed/RADAR-LN3-V26b-cq08-semantic-decision-layer-unfitted-registry.json"
)
V26B_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26b-frozen-l1-l2-raw-routing-readiness.json"
)
V26B_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v26b_frozen_l1_l2_raw_routing_audit.md"
)
V26B_VALIDATION_PHYSICAL_CACHE = Path(
    "data/processed/RADAR-LN3-V26b-cq08-l2-val-mixed-open-set-rd-physical4-cache.npz"
)
V26B_VALIDATION_MICRO_CACHE = Path(
    "data/processed/RADAR-LN3-V26b-cq08-l2-val-mixed-open-set-micro-doppler8-cache.npz"
)
V26B_TEST_PHYSICAL_CACHE = Path(
    "data/processed/RADAR-LN3-V26b-cq08-l2-test-mixed-open-set-rd-physical4-cache.npz"
)
V26B_TEST_MICRO_CACHE = Path(
    "data/processed/RADAR-LN3-V26b-cq08-l2-test-mixed-open-set-micro-doppler8-cache.npz"
)

CONTRACT_SCHEMA = "RADAR-LN3-V26b-frozen-l1-l2-raw-routing-contract-v1"
CACHE_MANIFEST_SCHEMA = "RADAR-LN3-V26b-l2-mixed-open-set-feature-cache-v1"
READINESS_SCHEMA = "RADAR-LN3-V26b-frozen-l1-l2-raw-routing-readiness-v1"
EXPECTED_PARENT_COUNTS = {
    "val": {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
        "UnknownTarget": 15,
    },
    "test": {
        "DroneTarget": 109,
        "BirdTarget": 53,
        "BalloonTarget": 58,
        "ClutterTarget": 8,
        "UnknownTarget": 16,
    },
}


def implementation_sha256(project_root: Path) -> dict[str, str]:
    """Return the complete behavior-defining implementation fingerprint."""

    root = project_root.resolve()
    paths = {
        "builder": Path("src-v3/build/known_gate_v26b_l1_l2_raw_routing.py"),
        "v18_extractor": Path("src-v3/build/micro_doppler_v18.py"),
        "l1_l2_dataset": Path("src-v3/radar_ln3/l1_l2_raw_routing.py"),
        "l3_registry": Path("src-v3/radar_ln3/semantic_decision_registry.py"),
        "action_resolver": Path("src-v3/radar_ln3/open_set_architecture.py"),
        "reporter": Path("src-v3/report/known_gate_v26b_l1_l2_raw_routing.py"),
        "smoke": Path("src-v3/smoke/known_gate_v26b_l1_l2_raw_routing_contract_test.py"),
    }
    missing = [name for name, path in paths.items() if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("V26b implementation source is missing: " + ", ".join(missing))
    return {name: sha256_file(root / path) for name, path in paths.items()}


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _required_source_files(root: Path) -> dict[str, Path]:
    return {
        "v26_config": root / V26_CONFIG,
        "v26_readiness": root / V26_READINESS,
        "v26_summary": root / V26_SUMMARY,
        "v26_checkpoint": root / V26_CHECKPOINT,
        "v26_contract": root / V26_EXPERIMENT_CONTRACT,
        "v26_validation_course_cache": root / V26_AUDIT_VALIDATION_CACHE,
        "v18r_config": root / V18R_CONFIG,
        "v18r_readiness": root / V18R_READINESS,
        "v18r_summary": root / V18R_SUMMARY,
        "v18r_checkpoint": root / V18R_CHECKPOINT,
        "v18r_report": root / V18R_REPORT,
        "v18_manifest": root / V18_CACHE_MANIFEST,
        "index": root / SOURCE_INDEX,
        "provenance": root / SOURCE_PROVENANCE,
        "canonical_frame_index": root / SOURCE_CANONICAL_FRAME_INDEX,
        "document": root / V26B_DOCUMENT,
    }


def _validate_sources(root: Path) -> dict[str, Any]:
    files = _required_source_files(root)
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("V26b frozen sources are missing: " + ", ".join(missing))
    hashes = {name: sha256_file(path) for name, path in files.items()}

    v26_config = _load_json(files["v26_config"])
    v26_readiness = _load_json(files["v26_readiness"])
    v26_summary = _load_json(files["v26_summary"])
    v26_contract = _load_json(files["v26_contract"])
    if (
        v26_config.get("variant") != "v26_known_only_open_set_baseline"
        or v26_config.get("task", {}).get("name") != "known_gate"
        or v26_readiness.get("status") != "passed"
        or int(v26_readiness.get("unknown_training_exposure", -1)) != 0
        or int(v26_readiness.get("test_payload_access_count", -1)) != 0
        or v26_summary.get("status") != "completed"
        or int(v26_summary.get("best_epoch", -1)) != 14
        or Path(str(v26_summary.get("best_checkpoint", ""))).resolve()
        != files["v26_checkpoint"].resolve()
        or v26_contract.get("open_set_audit_contract", {}).get("unknown_is_model_class")
        is not False
    ):
        raise ValueError("V26 frozen KnownGate source does not satisfy its strict-open-set contract")

    v18r_config = load_task_config(files["v18r_config"])
    v18r_readiness = _load_json(files["v18r_readiness"])
    v18r_summary = _load_json(files["v18r_summary"])
    v18r_report = _load_json(files["v18r_report"])
    treatment = dict(v18r_report.get("runs", {}).get("treatment") or {})
    if (
        v18r_config.schema_version != "RADAR-LN3-V18r"
        or v18r_config.task.name != "dbb_specialist"
        or tuple(v18r_config.task.class_names)
        != ("DroneTarget", "BirdTarget", "BalloonTarget")
        or v18r_config.model.micro_doppler_dim != 8
        or v18r_config.model.rd_physical_dim != 4
        or v18r_config.model.liquid_invalid_dt_update_mode != "density_weighted"
        or v18r_readiness.get("status") != "passed"
        or v18r_summary.get("status") != "completed"
        or int(v18r_summary.get("best_epoch", -1)) != 15
        or Path(str(v18r_summary.get("best_checkpoint", ""))).resolve()
        != files["v18r_checkpoint"].resolve()
        or Path(str(treatment.get("best_checkpoint", ""))).resolve()
        != files["v18r_checkpoint"].resolve()
        or treatment.get("best_checkpoint_sha256") != hashes["v18r_checkpoint"]
        or treatment.get("config_sha256") != hashes["v18r_config"]
        or treatment.get("independent_best_checkpoint_validation_matches") is not True
    ):
        raise ValueError("V18r frozen DBB source does not match its recorded treatment endpoint")
    return {
        "files": files,
        "hashes": hashes,
        "v26_config": v26_config,
        "v26_readiness": v26_readiness,
        "v18r_config": v18r_config,
    }


def _split_expected_frame(index: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = index[
        index["split_name"].astype(str).eq(split)
        & index["label_name"].astype(str).isin(MIXED_OPEN_SET_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V26b {split} mixed audit source selection is invalid")
    counts = (
        selected.groupby("label_name", sort=False)["parent_sample_id"]
        .nunique()
        .to_dict()
    )
    actual = {label: int(counts.get(label, 0)) for label in MIXED_OPEN_SET_LABELS}
    if actual != EXPECTED_PARENT_COUNTS[split]:
        raise ValueError(
            f"V26b {split} parent support changed: expected={EXPECTED_PARENT_COUNTS[split]}, actual={actual}"
        )
    return selected


def _cache_paths(split: str) -> tuple[Path, Path]:
    if split == "val":
        return V26B_VALIDATION_PHYSICAL_CACHE, V26B_VALIDATION_MICRO_CACHE
    if split == "test":
        return V26B_TEST_PHYSICAL_CACHE, V26B_TEST_MICRO_CACHE
    raise ValueError(f"V26b mixed L2 cache has unsupported split={split}")


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _verify_known_validation_subset(
    root: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    v18r_config: Any,
) -> dict[str, int]:
    """Prove that new mixed validation values equal frozen V18 values exactly."""

    physical_path = root / v18r_config.data.validation_rd_physical_cache_path
    micro_path = root / v18r_config.data.validation_micro_doppler_cache_path
    if not physical_path.is_file() or not micro_path.is_file():
        raise FileNotFoundError("V18r frozen validation feature cache is missing")
    physical = _read_npz(physical_path)
    micro = _read_npz(micro_path)
    required = {"sample_ids", "values", "valid"}
    if set(physical) != required or set(micro) != required:
        raise ValueError("V18r frozen validation feature cache fields are invalid")
    new_ids = [str(value) for value in np.asarray(arrays["sample_ids"]).tolist()]
    new_index = {sample_id: index for index, sample_id in enumerate(new_ids)}
    frozen_ids = [str(value) for value in physical["sample_ids"].tolist()]
    if set(frozen_ids) - set(new_index):
        raise ValueError("V26b mixed validation cache omits a frozen V18 DBB sample")
    positions = np.asarray([new_index[sample_id] for sample_id in frozen_ids], dtype=np.int64)
    if not (
        np.array_equal(arrays["physical_values"][positions], physical["values"])
        and np.array_equal(arrays["summary_values"][positions], micro["values"])
        and np.array_equal(arrays["valid"][positions], physical["valid"])
        and np.array_equal(arrays["valid"][positions], micro["valid"])
    ):
        raise ValueError("V26b mixed validation extraction differs from frozen V18 DBB evidence")
    return {"known_dbb_event_count": len(frozen_ids), "mixed_event_count": len(new_ids)}


def _build_l2_mixed_cache(root: Path, split: str, source: Mapping[str, Any]) -> dict[str, Any]:
    files = dict(source["files"])
    index = pd.read_csv(files["index"], low_memory=False)
    required_columns = {
        "schema_version",
        "sample_id",
        "parent_sample_id",
        "track_id",
        "track_point_index",
        "causal_event_index",
        "split_name",
        "label_name",
        "rd_present",
        "rd_frame_count",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
        "rd_raw_frame_indices_json",
        "cq08_raw_echo_file",
    }
    missing = sorted(required_columns - set(index.columns))
    if missing or index["schema_version"].astype(str).ne("RADAR-LN3-V16").any():
        raise ValueError("V26b source index is incompatible with frozen V18 extraction")
    expected = _split_expected_frame(index, split)
    provenance = _load_json(files["provenance"])
    provenance_by_parent = {
        str(value["parent_sample_id"]): dict(value)
        for value in list(provenance.get("records") or ())
    }
    if len(provenance_by_parent) != int(provenance.get("parent_count", -1)):
        raise ValueError("V26b source provenance is incomplete")
    canonical = _CanonicalFrameProvenance(root, files["canonical_frame_index"])
    arrays, statistics = _build_split(
        root,
        split,
        index,
        provenance_by_parent,
        canonical,
        allowed_labels=MIXED_OPEN_SET_LABELS,
        # The product audit consumes only the frozen 4+8 model values.  Avoid
        # materializing the much larger evidence-profile cache.
        include_evidence=False,
    )
    expected_ids = expected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    if (
        not np.array_equal(arrays["sample_ids"], expected_ids)
        or arrays["physical_values"].shape != (len(expected), 4)
        or arrays["summary_values"].shape != (len(expected), 8)
        or arrays["valid"].shape != (len(expected),)
        or arrays["valid"].dtype != np.bool_
        or not np.array_equal(
            arrays["valid"], expected["rd_present"].map(_as_bool).to_numpy(dtype=np.bool_)
        )
        or np.any(arrays["physical_values"][~arrays["valid"]] != 0.0)
        or np.any(arrays["summary_values"][~arrays["valid"]] != 0.0)
    ):
        raise ValueError("V26b mixed L2 feature cache values/validity are invalid")
    physical_path_relative, micro_path_relative = _cache_paths(split)
    physical_path = root / physical_path_relative
    micro_path = root / micro_path_relative
    _write_npz_new_or_identical(
        physical_path,
        (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["physical_values"]),
            ("valid", arrays["valid"]),
        ),
    )
    _write_npz_new_or_identical(
        micro_path,
        (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["summary_values"]),
            ("valid", arrays["valid"]),
        ),
    )
    result = {
        "split": split,
        "physical_cache": _relative(root, physical_path),
        "physical_cache_sha256": sha256_file(physical_path),
        "micro_doppler_cache": _relative(root, micro_path),
        "micro_doppler_cache_sha256": sha256_file(micro_path),
        "event_count": int(len(expected)),
        "parent_label_counts": EXPECTED_PARENT_COUNTS[split],
        "valid_event_count": int(arrays["valid"].sum()),
        "invalid_event_count": int((~arrays["valid"]).sum()),
        "raw_file_count": int(statistics["stable8_prepare_access_counts"]["raw_file_count"]),
        "rd_tensor_count": int(statistics["stable8_prepare_access_counts"]["rd_tensor_count"]),
        "complete_evidence_profiles_materialized": False,
        "label_free_extraction": True,
    }
    if split == "val":
        result["frozen_v18_known_subset_exact_match"] = _verify_known_validation_subset(
            root,
            arrays,
            v18r_config=source["v18r_config"],
        )
    return result


def _write_cache_manifest(root: Path, validation_cache: Mapping[str, Any]) -> dict[str, Any]:
    """Bind audit cache files independently of either frozen training contract."""

    payload = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "purpose": "label-free frozen-V18r L2 mixed open-set routing audit cache",
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": sha256_file(root / SOURCE_INDEX),
        "validation": dict(validation_cache),
        "test": {
            "physical_cache": V26B_TEST_PHYSICAL_CACHE.as_posix(),
            "micro_doppler_cache": V26B_TEST_MICRO_CACHE.as_posix(),
            "materialization": "after frozen L1/L2 checkpoint loading only",
            "test_payload_access_count_during_prepare": 0,
        },
        "model_input_fields": {
            "rd_physical": ["sample_ids", "values", "valid"],
            "micro_doppler": ["sample_ids", "values", "valid"],
        },
        "label_free_extraction": True,
        "unknown_is_model_class": False,
        "complete_evidence_profiles_materialized": False,
    }
    _write_json_new_or_identical(root / V26B_CACHE_MANIFEST, payload)
    return payload


def _component_binding(root: Path, source: Mapping[str, Any], prefix: str, task_name: str) -> dict[str, str]:
    files = dict(source["files"])
    config = files[f"{prefix}_config"]
    readiness = files[f"{prefix}_readiness"]
    summary = files[f"{prefix}_summary"]
    checkpoint = files[f"{prefix}_checkpoint"]
    return {
        "task_name": task_name,
        "config_path": _relative(root, config),
        "config_sha256": sha256_file(config),
        "checkpoint_path": _relative(root, checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_summary_path": _relative(root, summary),
        "training_summary_sha256": sha256_file(summary),
        "training_readiness_path": _relative(root, readiness),
        "training_readiness_sha256": sha256_file(readiness),
    }


def _write_registry(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": SEMANTIC_DECISION_REGISTRY_SCHEMA,
        "registry_version": "RADAR-LN3-V26b-unfitted-l1-l2-raw-routing-v1",
        "layer_name": "semantic_decision_layer",
        "architecture_schema": "RADAR-LN3-V21-strict-open-set-architecture-v1",
        "component_bindings": {
            "l1_known_gate": _component_binding(root, source, "v26", "known_gate"),
            "l2_dbb_specialist": _component_binding(root, source, "v18r", "dbb_specialist"),
        },
        "training_parameter_authority": {
            "l1": "the hash-bound L1 config/checkpoint/summary above are authoritative for L1 training parameters",
            "l2": "the hash-bound L2 config/checkpoint/summary above are authoritative for L2 training parameters",
            "l3": "stores only score definitions, calibration thresholds, action semantics and immutable component references",
            "copied_training_hyperparameters": False,
        },
        "calibration": {
            "status": UNFITTED_CALIBRATION_STATUS,
            "fit_population": "not_fitted; no OOF or validation/test threshold selection",
            "score_definitions": {
                "l1_known_target_acceptance": (
                    "softmax(L1 KnownGate)[KNOWN_TARGET_EVIDENCE]; raw route probability, not true knownness"
                ),
                "l1_known_clutter_acceptance": (
                    "softmax(L1 KnownGate)[KNOWN_CLUTTER_EVIDENCE]; raw route probability, not true knownness"
                ),
                "l2_subclass_acceptance": (
                    "max(softmax(L2 DBB subclass logits)); subclass confidence, not true DBB knownness"
                ),
            },
            "thresholds": {
                "l1_known_target_acceptance": None,
                "l1_known_clutter_acceptance": None,
                "l2_subclass_acceptance": None,
            },
            "threshold_source": "not_available_until parent-grouped cross-fitted OOF calibration",
        },
        "action_semantics": {
            "resolver": "resolve_semantic_action",
            "minimum_evidence_events": 3,
            "ineligible_action": "PENDING_ABSTAIN",
            "l1_target_l2_reject_action": "UNKNOWN_REJECTED",
            "l1_reject_action": "UNKNOWN_REJECTED",
            "l1_route_conflict_action": "PENDING_ABSTAIN",
        },
        "unknown_is_learned_class": False,
        "pending_is_unknown_rejection_success": False,
        "final_actions_enabled": False,
        "registry_scope": "frozen raw-routing audit only; no calibrated L3 action or deployment claim",
    }
    validate_semantic_decision_registry(payload)
    _write_json_new_or_identical(root / V26B_L3_REGISTRY, payload)
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source = _validate_sources(root)
    validation_cache = _build_l2_mixed_cache(root, "val", source)
    cache_manifest = _write_cache_manifest(root, validation_cache)
    registry = _write_registry(root, source)
    hashes = dict(source["hashes"])
    implementation_hashes = implementation_sha256(root)
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment_name": "V26b frozen L1 KnownGate plus L2 DBB raw-routing compatibility audit",
        "scope": "post-checkpoint frozen raw routing only; no model fitting or calibrated open-set action",
        "source_bindings": {
            name: {"path": _relative(root, path), "sha256": hashes[name]}
            for name, path in source["files"].items()
            if name
            in {
                "v26_config",
                "v26_readiness",
                "v26_summary",
                "v26_checkpoint",
                "v26_contract",
                "v26_validation_course_cache",
                "v18r_config",
                "v18r_readiness",
                "v18r_summary",
                "v18r_checkpoint",
                "v18r_report",
                "v18_manifest",
                "index",
                "provenance",
                "canonical_frame_index",
                "document",
            }
        },
        "implementation_sha256": implementation_hashes,
        "data_contract": {
            "source_index_schema": "RADAR-LN3-V16",
            "mixed_labels": list(MIXED_OPEN_SET_LABELS),
            "validation_l2_cache": validation_cache,
            "cache_manifest": {
                "path": _relative(root, root / V26B_CACHE_MANIFEST),
                "sha256": sha256_file(root / V26B_CACHE_MANIFEST),
            },
            "test_l2_cache": {
                "materialization": "only after both frozen L1/L2 checkpoints are loaded for audit",
                "physical_cache": V26B_TEST_PHYSICAL_CACHE.as_posix(),
                "micro_doppler_cache": V26B_TEST_MICRO_CACHE.as_posix(),
                "test_raw_payload_access_count_during_prepare": 0,
            },
            "extraction": {
                "label_free_current_event_values": True,
                "same_v18_rd_physical4_and_stable8_micro_doppler_extractor": True,
                "validation_known_dbb_subset_bitwise_equal_to_frozen_v18": True,
                "complete_evidence_profiles_materialized": False,
            },
        },
        "l3_registry": {
            "path": _relative(root, root / V26B_L3_REGISTRY),
            "sha256": sha256_file(root / V26B_L3_REGISTRY),
            "calibration_status": registry["calibration"]["status"],
            "final_actions_enabled": registry["final_actions_enabled"],
        },
        "unknown_semantics": {
            "unknown_is_model_class": False,
            "unknown_training_exposure_added": 0,
            "unknown_label_used_for_feature_value": False,
            "unknown_action_threshold_fit": "not_executed",
            "pending_is_unknown_rejection_success": False,
            "unknown_first_rejection_stage_registry": "not_available; must be human-reviewed before stage-specific rejection claims",
        },
        "test_protocol": {
            "test_cache_materialized_during_prepare": False,
            "test_forward_allowed_after_both_checkpoint_hashes_verified": True,
            "test_used_for_training_or_checkpoint_selection": False,
            "test_used_for_l3_threshold_selection": False,
            "test_result_name": "frozen_raw_routing_audit_only",
        },
        "not_executed": [
            "L1 OOF calibration",
            "L2 OOF calibration",
            "L3 threshold fitting",
            "stage-resolved Unknown rejection measurement",
            "stable state evaluation",
            "P95 latency",
            "deployment eligibility",
        ],
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / V26B_CONTRACT, contract)
    checks = {
        "v26_known_gate_checkpoint_is_frozen_and_strict_open_set": True,
        "v18r_dbb_checkpoint_is_frozen_and_hash_bound": True,
        "validation_l2_mixed_cache_uses_label_free_frozen_v18_extraction": True,
        "validation_known_dbb_values_exactly_match_frozen_v18_caches": True,
        "test_l2_cache_is_not_materialized_during_prepare": True,
        "l3_registry_is_unfitted_and_final_actions_are_disabled": True,
        "unknown_is_not_a_model_class_or_training_target": True,
        "pending_is_not_unknown_rejection_success": True,
        "full_product_open_set_claim_is_deferred_until_cross_fitted_l3": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
        "checks": checks,
        "contract_path": _relative(root, root / V26B_CONTRACT),
        "contract_sha256": sha256_file(root / V26B_CONTRACT),
        "l3_registry_path": _relative(root, root / V26B_L3_REGISTRY),
        "l3_registry_sha256": sha256_file(root / V26B_L3_REGISTRY),
        "validation_l2_cache": validation_cache,
        "cache_manifest_path": _relative(root, root / V26B_CACHE_MANIFEST),
        "cache_manifest_sha256": sha256_file(root / V26B_CACHE_MANIFEST),
        "validation_parent_count": sum(EXPECTED_PARENT_COUNTS["val"].values()),
        "validation_unknown_parent_count": EXPECTED_PARENT_COUNTS["val"]["UnknownTarget"],
        "planned_test_parent_count": sum(EXPECTED_PARENT_COUNTS["test"].values()),
        "planned_test_unknown_parent_count": EXPECTED_PARENT_COUNTS["test"]["UnknownTarget"],
        "test_raw_payload_access_count": 0,
        "training": "not_executed",
        "OOF": "not_executed",
        "L3_calibration": "not_executed",
        "test": "not_executed",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / V26B_READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": READINESS_SCHEMA,
        "contract": str(root / V26B_CONTRACT),
        "l3_registry": str(root / V26B_L3_REGISTRY),
        "readiness": str(root / V26B_READINESS),
        "validation_l2_mixed_cache": validation_cache,
        "training": "not_executed",
        "test": "not_executed",
    }


def materialize_test_l2_mixed_cache(project_root: Path) -> dict[str, Any]:
    """Build only after frozen L1/L2 endpoints have been verified by caller."""

    root = project_root.resolve()
    source = _validate_sources(root)
    readiness_path = root / V26B_READINESS
    contract_path = root / V26B_CONTRACT
    manifest_path = root / V26B_CACHE_MANIFEST
    if not readiness_path.is_file() or not contract_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("V26b test cache requires completed V26b preparation")
    readiness = _load_json(readiness_path)
    contract = _load_json(contract_path)
    manifest = _load_json(manifest_path)
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or readiness.get("cache_manifest_sha256") != sha256_file(manifest_path)
        or contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("data_contract", {}).get("cache_manifest", {}).get("sha256")
        != sha256_file(manifest_path)
        or manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA
        or contract.get("test_protocol", {}).get("test_cache_materialized_during_prepare")
        is not False
    ):
        raise ValueError("V26b test cache preparation contract is stale")
    result = _build_l2_mixed_cache(root, "test", source)
    result["cache_manifest_sha256"] = sha256_file(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V26b frozen L1/L2 raw-routing audit artifacts."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


__all__ = [
    "CONTRACT_SCHEMA",
    "EXPECTED_PARENT_COUNTS",
    "READINESS_SCHEMA",
    "V18R_CHECKPOINT",
    "V18R_CONFIG",
    "V18R_READINESS",
    "V18R_SUMMARY",
    "V26B_CACHE_MANIFEST",
    "V26B_CONTRACT",
    "V26B_L3_REGISTRY",
    "V26B_READINESS",
    "V26B_TEST_MICRO_CACHE",
    "V26B_TEST_PHYSICAL_CACHE",
    "V26B_VALIDATION_MICRO_CACHE",
    "V26B_VALIDATION_PHYSICAL_CACHE",
    "V26_AUDIT_VALIDATION_CACHE",
    "V26_CHECKPOINT",
    "V26_CONFIG",
    "V26_READINESS",
    "V26_SUMMARY",
    "build",
    "implementation_sha256",
    "materialize_test_l2_mixed_cache",
]


if __name__ == "__main__":
    main()
