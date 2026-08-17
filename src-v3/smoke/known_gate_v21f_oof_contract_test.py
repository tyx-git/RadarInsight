"""Focused contract smoke for the exploratory V21f KnownGate OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v21e_oof import build_assignments, _parent_records
from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_v21e_oof import canonical_sha256, sha256_file
from radar_ln3.known_gate_v21f_oof import (
    ARM,
    ATTRIBUTION_RELATIVE,
    ATTRIBUTION_SCHEMA,
    COMBINED_CACHE_RELATIVE,
    EXPERIMENT_CONTRACT_RELATIVE,
    EXPERIMENT_CONTRACT_SCHEMA,
    FEATURE_MANIFEST_RELATIVE,
    FEATURE_MANIFEST_SCHEMA,
    FOLD_COUNT,
    INDEX_RELATIVE,
    OOF_CONTRACT_RELATIVE,
    OOF_CONTRACT_SCHEMA,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    V21A_SPECTRAL_CACHE_RELATIVE,
    V21E_COURSE_CACHE_RELATIVE,
    artifact_relative,
    config_relative,
    v21e_config_relative,
    v21e_prediction_relative,
)
from radar_ln3.known_gate_features import (
    COURSE_INTERFRAME_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_NAMES,
    INTERFRAME_SPECTRAL_FEATURE_NAMES,
    V21F_COURSE_INTERFRAME_VARIANT,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs
from train.known_gate_v21e_oof import _state_dict_sha256
from train.known_gate_v21f_oof import _verify_prepared


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cache(path: Path, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "values", "valid"}:
            raise AssertionError(f"unexpected V21f cache fields: {path}")
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
        raise AssertionError(f"V21f cache contract failed: {path}")
    return sample_ids, values, valid


def _check_assignments(root: Path) -> dict[str, Any]:
    records = _parent_records(root / INDEX_RELATIVE)
    first = build_assignments(records)
    second = build_assignments(records)
    if canonical_sha256(first) != canonical_sha256(second):
        raise AssertionError("V21f fold assignment is not deterministic")
    if (
        int(first["fold_count"]) != FOLD_COUNT
        or int(first["known_parent_count"]) != 986
        or int(first["provenance_component_count"]) != 975
        or int(first["unknown_training_exposure"]) != 0
    ):
        raise AssertionError("V21f assignment population changed")
    component_folds: dict[str, set[int]] = {}
    parent_ids: list[str] = []
    for row in first["assignments"]:
        parent = str(row["parent_sample_id"])
        parent_ids.append(parent)
        component_folds.setdefault(str(row["provenance_component_id"]), set()).add(
            int(row["fold_id"])
        )
    if len(parent_ids) != len(set(parent_ids)) or any(
        len(folds) != 1 for folds in component_folds.values()
    ):
        raise AssertionError("V21f provenance component crosses OOF folds")
    return first


def _check_sources(root: Path, assignment: dict[str, Any]) -> dict[str, Any]:
    readiness = _json(root / READINESS_RELATIVE)
    oof_contract_path = root / OOF_CONTRACT_RELATIVE
    experiment_path = root / EXPERIMENT_CONTRACT_RELATIVE
    manifest_path = root / FEATURE_MANIFEST_RELATIVE
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("oof_contract_sha256") != sha256_file(oof_contract_path)
        or readiness.get("experiment_contract_sha256") != sha256_file(experiment_path)
        or readiness.get("feature_manifest_sha256") != sha256_file(manifest_path)
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or readiness.get("validation_used_for_readiness_gate") is not False
    ):
        raise AssertionError("V21f readiness is stale or outside its sealed scope")
    failed = [
        name for name, value in dict(readiness.get("checks") or {}).items() if value is not True
    ]
    if failed:
        raise AssertionError(f"V21f readiness checks failed: {failed}")
    contract = _json(oof_contract_path)
    experiment = _json(experiment_path)
    manifest = _json(manifest_path)
    if (
        contract.get("schema_version") != OOF_CONTRACT_SCHEMA
        or experiment.get("schema_version") != EXPERIMENT_CONTRACT_SCHEMA
        or manifest.get("schema_version") != FEATURE_MANIFEST_SCHEMA
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or int(contract.get("known_parent_count", -1)) != 986
        or int(contract.get("provenance_component_count", -1)) != 975
        or contract.get("control_reused_from") != "V21e OOF Treatment"
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("validation_parent_count", -1)) != 0
        or int(contract.get("test_parent_count", -1)) != 0
        or experiment.get("role")
        != "exploratory paired known-train OOF; not independent confirmation"
        or manifest.get("test_excluded") is not True
        or int(manifest.get("test_payload_access_count", -1)) != 0
    ):
        raise AssertionError("V21f source contracts do not match the frozen protocol")
    course_ids, course_values, course_valid = _load_cache(
        root / V21E_COURSE_CACHE_RELATIVE, len(COURSE_PREFIX_FEATURE_NAMES)
    )
    spectral_ids, spectral_values, spectral_valid = _load_cache(
        root / V21A_SPECTRAL_CACHE_RELATIVE, len(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    )
    combined_ids, combined_values, combined_valid = _load_cache(
        root / COMBINED_CACHE_RELATIVE, len(COURSE_INTERFRAME_FEATURE_NAMES)
    )
    if (
        not np.array_equal(course_ids, spectral_ids)
        or not np.array_equal(course_ids, combined_ids)
        or not np.array_equal(
            combined_values,
            np.concatenate([course_values, spectral_values], axis=1),
        )
        or not np.array_equal(
            combined_valid,
            np.concatenate([course_valid, spectral_valid], axis=1),
        )
    ):
        raise AssertionError("V21f cache alignment or concatenation changed")
    attribution = _json(root / ATTRIBUTION_RELATIVE)
    if (
        attribution.get("schema_version") != ATTRIBUTION_SCHEMA
        or attribution.get("status") != "analyzed_exploratory_post_hoc"
        or attribution.get("supports")
        != {"v21e_hard_bird": 14, "v21e_correct_bird": 225, "clutter": 33}
        or attribution.get("all_hard_bird_spectral_features_valid") is not True
    ):
        raise AssertionError("V21f hard-Bird attribution support changed")
    return {
        "readiness": readiness,
        "contract": contract,
        "experiment": experiment,
        "manifest": manifest,
        "attribution": attribution,
        "course_event_count": int(len(course_ids)),
        "combined_event_count": int(len(combined_ids)),
    }


def _check_pairing(root: Path, assignment: dict[str, Any]) -> dict[str, Any]:
    train_ids = [
        str(row["parent_sample_id"])
        for row in assignment["assignments"]
        if int(row["fold_id"]) != 0
    ]
    config_path = root / config_relative(ARM, 0)
    source_config_path = root / v21e_config_relative("treatment", 0)
    config, _, _, _, _ = _verify_prepared(root, arm=ARM, fold_id=0)
    source_config_loaded = load_known_gate_config(source_config_path)
    treatment, reference, pairing = build_known_gate_model_for_config(root, config)
    if reference is None:
        raise AssertionError("V21f lacks its nested V21e Treatment reference")
    if (
        config.variant != V21F_COURSE_INTERFRAME_VARIANT
        or tuple(config.feature_names) != tuple(COURSE_INTERFRAME_FEATURE_NAMES)
        or tuple(config.feature_names[:1]) != tuple(COURSE_PREFIX_FEATURE_NAMES)
        or config.model.evidence_dim != 7
        or source_config_loaded.model.evidence_dim != 1
        or config.train.paired_baseline_config_path
        != str(v21e_config_relative("treatment", 0)).replace("\\", "/")
        or config.data.validation_feature_cache_path is not None
        or config.data.expected_validation_feature_cache_sha256 is not None
        or pairing.get("new_fusion_columns") != 6
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
    ):
        raise AssertionError("V21f nested pair configuration failed")
    v21e_prediction = _json(
        root / v21e_prediction_relative("treatment", 0)
    )
    reference_hash = _state_dict_sha256(reference.state_dict())
    if reference_hash != v21e_prediction.get("initial_model_state_sha256"):
        raise AssertionError(
            "V21f nested V21e Treatment initial state does not match its OOF artifact"
        )

    selected_ids = train_ids[:]
    if len(selected_ids) < 4:
        raise AssertionError("V21f smoke subset is too small")
    dataset = build_known_gate_dataset(
        root,
        config,
        split_name="train",
        parent_ids=selected_ids,
        allow_feature_cache_superset=True,
    )
    if set(dataset.parent_ids) != set(selected_ids):
        raise AssertionError("V21f Dataset parent subset changed")
    metadata = dict(dataset.feature_cache_metadata or {})
    if (
        metadata.get("subset_from_superset") is not True
        or int(metadata.get("event_count", -1))
        != int(dataset.base.frame["sample_id"].nunique())
        or int(metadata.get("source_cache_event_count", 0))
        <= int(metadata.get("event_count", 0))
    ):
        raise AssertionError("V21f feature-cache subset contract failed")

    loader = build_known_gate_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    treatment.eval()
    reference.eval()
    with torch.inference_mode():
        treatment_logits = treatment(**_model_inputs(batch))["sequence_logits"]
        reference_inputs = _model_inputs(batch)
        reference_inputs["known_gate_features"] = reference_inputs[
            "known_gate_features"
        ][..., :1]
        reference_inputs["known_gate_feature_valid"] = reference_inputs[
            "known_gate_feature_valid"
        ][..., :1]
        reference_logits = reference(**reference_inputs)["sequence_logits"]
    if not torch.equal(treatment_logits, reference_logits):
        raise AssertionError("V21f initial logits differ from nested V21e Treatment")

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    gradient_sum = 0.0
    for candidate in loader:
        treatment.zero_grad(set_to_none=True)
        output = treatment(**_model_inputs(candidate))
        objective = parent_final_task_ce_with_prefix_auxiliary(
            output["sequence_logits"],
            candidate["labels"],
            candidate["tr_padding_mask"],
            class_weights=class_weights,
            prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
            prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
        )
        objective["loss"].backward()
        full_gradient = treatment.fusion[0].weight.grad
        gradient = None if full_gradient is None else full_gradient[:, -6:]
        if gradient is not None and torch.isfinite(gradient).all():
            gradient_sum = max(gradient_sum, float(gradient.abs().sum().item()))
        if gradient_sum > 0.0:
            break
    if not gradient_sum > 0.0:
        raise AssertionError("V21f appended spectral columns receive no gradient")
    new_columns = treatment.state_dict()["fusion.0.weight"][:, reference.fusion_input_dim :]
    if not torch.equal(new_columns, torch.zeros_like(new_columns)):
        raise AssertionError("V21f appended Fusion columns are not zero at initialization")
    return {
        "nested_v21e_initial_state_matches_prediction": True,
        "initial_logits_exactly_equal": True,
        "new_spectral_columns_exactly_zero": True,
        "new_spectral_columns_receive_gradient": True,
        "train_parent_subset_count": len(selected_ids),
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    assignment = _check_assignments(root)
    source = _check_sources(root, assignment)
    pairing = _check_pairing(root, assignment)
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21f-known-gate-course-interframe-oof-contract-smoke-v1",
        "known_parent_count": int(assignment["known_parent_count"]),
        "provenance_component_count": int(assignment["provenance_component_count"]),
        "folds": assignment["folds"],
        "assignment_is_deterministic": True,
        "no_provenance_component_crosses_folds": True,
        "source_cache_alignment_and_concat_exact": True,
        "combined_event_count": source["combined_event_count"],
        "hard_bird_attribution_support": source["attribution"]["supports"],
        **pairing,
        "unknown_training_exposure": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
        "test_payload_access_count": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V21f exploratory OOF contracts.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
