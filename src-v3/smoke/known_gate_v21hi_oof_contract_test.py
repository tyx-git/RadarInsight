"""Focused contract smoke for V21h/V21i nested OOF experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v21e_oof import _parent_records, build_assignments
from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    KNOWN_GATE_FEATURE_DIM,
)
from radar_ln3.known_gate_v21e_oof import canonical_sha256, sha256_file
from radar_ln3.known_gate_v21hi_oof import (
    ASSIGNMENT_RELATIVE,
    FOLD_COUNT,
    INDEX_RELATIVE,
    V21E_COURSE_CACHE_RELATIVE,
    artifact_relative,
    config_relative,
    get_spec,
    prediction_relative,
    v21e_config_relative,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs
from train.known_gate_v21e_oof import _state_dict_sha256
from train.known_gate_v21hi_oof import _verify_prepared


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cache(path: Path, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
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
        raise AssertionError(f"invalid V21h/i cache: {path}")
    return sample_ids, values, valid


def _check_assignment(root: Path) -> dict[str, Any]:
    records = _parent_records(root / INDEX_RELATIVE)
    first = build_assignments(records)
    second = build_assignments(records)
    if canonical_sha256(first) != canonical_sha256(second):
        raise AssertionError("V21h/i assignment is not deterministic")
    components: dict[str, set[int]] = {}
    parents: set[str] = set()
    for row in first["assignments"]:
        parents.add(str(row["parent_sample_id"]))
        components.setdefault(str(row["provenance_component_id"]), set()).add(
            int(row["fold_id"])
        )
    if (
        int(first["fold_count"]) != FOLD_COUNT
        or int(first["known_parent_count"]) != 986
        or int(first["provenance_component_count"]) != 975
        or len(parents) != 986
        or any(len(values) != 1 for values in components.values())
    ):
        raise AssertionError("V21h/i assignment isolation failed")
    return first


def _check_source(root: Path, variant: str, assignment: dict[str, Any]) -> dict[str, Any]:
    spec = get_spec(variant)
    readiness = _json(root / spec.readiness_relative)
    contract_path = root / spec.oof_contract_relative
    experiment_path = root / spec.experiment_contract_relative
    manifest_path = root / spec.feature_manifest_relative
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != spec.readiness_schema
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or readiness.get("experiment_contract_sha256") != sha256_file(experiment_path)
        or readiness.get("feature_manifest_sha256") != sha256_file(manifest_path)
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise AssertionError(f"{spec.code} readiness is stale or outside sealed scope")
    if any(value is not True for value in dict(readiness.get("checks") or {}).values()):
        raise AssertionError(f"{spec.code} readiness contains a failed check")
    contract = _json(contract_path)
    experiment = _json(experiment_path)
    manifest = _json(manifest_path)
    if (
        contract.get("schema_version") != spec.oof_contract_schema
        or experiment.get("schema_version") != spec.experiment_contract_schema
        or manifest.get("schema_version") != spec.feature_manifest_schema
        or contract.get("control_reused_from") != "V21e OOF Treatment"
        or int(contract.get("known_parent_count", -1)) != 986
        or int(contract.get("provenance_component_count", -1)) != 975
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("validation_parent_count", -1)) != 0
        or int(contract.get("test_parent_count", -1)) != 0
        or manifest.get("test_excluded") is not True
    ):
        raise AssertionError(f"{spec.code} source contracts changed")
    course_ids, course_values, course_valid = _load_cache(
        root / V21E_COURSE_CACHE_RELATIVE, len(COURSE_PREFIX_FEATURE_NAMES)
    )
    source_ids, source_values, source_valid = _load_cache(
        root / spec.source_cache_relative,
        len(spec.source_indices)
        if spec.source_kind == "independent_prefix_cache"
        else KNOWN_GATE_FEATURE_DIM,
    )
    combined_ids, combined_values, combined_valid = _load_cache(
        root / spec.combined_cache_relative, len(spec.feature_names)
    )
    if spec.source_kind == "core_cache_selected":
        source_values = source_values[:, spec.source_indices]
        source_valid = source_valid[:, spec.source_indices]
    if (
        not np.array_equal(course_ids, source_ids)
        or not np.array_equal(course_ids, combined_ids)
        or not np.array_equal(
            combined_values,
            np.concatenate([course_values, source_values], axis=1),
        )
        or not np.array_equal(
            combined_valid,
            np.concatenate([course_valid, source_valid], axis=1),
        )
    ):
        raise AssertionError(f"{spec.code} cache alignment/concatenation failed")
    return {
        "readiness": readiness,
        "contract": contract,
        "event_count": int(len(combined_ids)),
    }


def _check_pairing(root: Path, variant: str, assignment: dict[str, Any]) -> dict[str, Any]:
    spec = get_spec(variant)
    config, fold, _, _, _ = _verify_prepared(
        root,
        variant=variant,
        arm="treatment",
        fold_id=0,
    )
    source_config = load_known_gate_config(root / v21e_config_relative("treatment", 0))
    treatment, reference, pairing = build_known_gate_model_for_config(root, config)
    if reference is None:
        raise AssertionError(f"{spec.code} nested V21e Treatment reference is missing")
    v21e_prediction = _json(
        root / Path(
            "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof"
        )
        / "fold-00/treatment/held-out-predictions.json"
    )
    if (
        config.variant != spec.variant
        or tuple(config.feature_names) != tuple(spec.feature_names)
        or tuple(config.feature_names[:1]) != tuple(COURSE_PREFIX_FEATURE_NAMES)
        or config.model.evidence_dim != len(spec.feature_names)
        or source_config.model.evidence_dim != 1
        or config.data.validation_feature_cache_path is not None
        or config.data.expected_validation_feature_cache_sha256 is not None
        or pairing.get("new_fusion_columns") != 1
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
    ):
        raise AssertionError(f"{spec.code} nested pair configuration failed")
    if _state_dict_sha256(reference.state_dict()) != v21e_prediction[
        "initial_model_state_sha256"
    ]:
        raise AssertionError(f"{spec.code} nested V21e initial state hash differs")

    train_ids = [
        str(row["parent_sample_id"])
        for row in assignment["assignments"]
        if int(row["fold_id"]) != 0
    ]
    dataset = build_known_gate_dataset(
        root,
        config,
        split_name="train",
        parent_ids=train_ids,
        allow_feature_cache_superset=True,
    )
    metadata = dict(dataset.feature_cache_metadata or {})
    if (
        set(dataset.parent_ids) != set(train_ids)
        or metadata.get("subset_from_superset") is not True
        or int(metadata.get("event_count", -1))
        != int(dataset.base.frame["sample_id"].nunique())
        or int(metadata.get("source_cache_event_count", 0))
        <= int(metadata.get("event_count", 0))
    ):
        raise AssertionError(f"{spec.code} Dataset/cache subset contract failed")
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
        raise AssertionError(f"{spec.code} initial logits differ from V21e")

    treatment.train()
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
        gradient = None if full_gradient is None else full_gradient[:, -1:]
        if gradient is not None and torch.isfinite(gradient).all():
            gradient_sum = max(gradient_sum, float(gradient.abs().sum().item()))
        if gradient_sum > 0.0:
            break
    new_column = treatment.state_dict()["fusion.0.weight"][:, reference.fusion_input_dim :]
    if not torch.equal(new_column, torch.zeros_like(new_column)):
        raise AssertionError(f"{spec.code} new Fusion column is not zero")
    if not gradient_sum > 0.0:
        raise AssertionError(f"{spec.code} new Fusion column receives no gradient")
    return {
        "nested_v21e_initial_state_matches_prediction": True,
        "initial_logits_exactly_equal": True,
        "new_column_exactly_zero": True,
        "new_column_receives_gradient": True,
        "train_parent_subset_count": len(train_ids),
        "held_out_parent_count": len(fold["held_out_parent_ids"]),
    }


def run(project_root: Path, *, variant: str) -> dict[str, Any]:
    root = project_root.resolve()
    assignment = _check_assignment(root)
    source = _check_source(root, variant, assignment)
    pairing = _check_pairing(root, variant, assignment)
    return {
        "status": "passed",
        "schema_version": f"RADAR-LN3-{get_spec(variant).code}-known-gate-course-oof-contract-smoke-v1",
        "variant": get_spec(variant).variant,
        "known_parent_count": int(assignment["known_parent_count"]),
        "provenance_component_count": int(assignment["provenance_component_count"]),
        "assignment_is_deterministic": True,
        "no_provenance_component_crosses_folds": True,
        "source_cache_alignment_and_concat_exact": True,
        "combined_event_count": source["event_count"],
        **pairing,
        "unknown_training_exposure": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
        "test_payload_access_count": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V21h/V21i nested OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("h", "i"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, variant=args.variant), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
