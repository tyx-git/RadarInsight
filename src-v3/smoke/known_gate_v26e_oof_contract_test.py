"""Focused contract and wiring smoke checks for V26e.

The check validates immutable V26c reuse and one finite V26e L1 forward.  It
does not train, open Unknown data, select a checkpoint, or fit thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    build_known_gate_dataloader,
    build_known_gate_model_for_config,
    build_known_gate_oof_dataset,
)
from radar_ln3.known_gate_v26e_oof import (
    CONTRACT,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    L1_EPOCH,
    L1_PREDICTION_SCHEMA,
    L1_TRAIN_EPOCHS,
    READINESS,
    READINESS_SCHEMA,
    V26C_CONTRACT,
    V26C_READINESS,
    V21E_ASSIGNMENT,
    V21E_FOLD_CONTRACT,
    fold_record,
    l1_config_path,
    l1_held_out_feature_cache_path,
    l1_train_feature_cache_path,
    l2_checkpoint_path,
    l2_prediction_path,
    l2_training_summary_path,
    relative_path,
    require_bound_file,
    selected_event_sample_ids,
    sequence_sha256,
    sha256_file,
    validate_frozen_v26c_sources,
)
from radar_ln3.known_gate_v26c_oof import (
    fold_record as v26c_fold_record,
)
from build.known_gate_v26e_oof import _validate_frozen_l1_data, _validate_frozen_l2
from train.known_gate_v21 import _model_inputs as l1_model_inputs
from train.known_gate_v26e_oof import _runtime_l1_config, _source_protocol


MAX_CACHE_BYTES = 64 * 1024 * 1024
EXPECTED_PARENT_COUNT = 986
KNOWN_LABELS = {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26e smoke expected JSON object: {path}")
    return value


def _cache_ids(root: Path, path: Path, expected: Mapping[str, Any]) -> list[str]:
    require_bound_file(root, path, "V26e cache", max_bytes=MAX_CACHE_BYTES)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise AssertionError(f"V26e cache fields differ: {path}")
        sample_ids = [str(value) for value in archive["sample_ids"].tolist()]
        values = archive["values"]
        valid = archive["valid"]
    if (
        values.ndim != 2
        or values.shape[1] != 1
        or valid.shape != values.shape
        or values.dtype != np.float32
        or valid.ndim != 2
        or valid.dtype != np.bool_
        or len(sample_ids) != values.shape[0]
        or len(sample_ids) != len(set(sample_ids))
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or np.any(values[~valid] != 0.0)
        or int(expected.get("event_count", -1)) != len(sample_ids)
        or str(expected.get("selected_sample_ids_sha256")) != sequence_sha256(sample_ids)
    ):
        raise AssertionError(f"V26e cache shape/content binding differs: {path}")
    return sample_ids


def _assignment(root: Path, contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    assignment_path = root / V21E_ASSIGNMENT
    fold_contract_path = root / V21E_FOLD_CONTRACT
    binding = dict(contract.get("assignment_reuse") or {})
    if (
        binding.get("source") != relative_path(root, assignment_path)
        or binding.get("sha256") != sha256_file(assignment_path)
        or binding.get("source_contract") != relative_path(root, fold_contract_path)
        or binding.get("source_contract_sha256") != sha256_file(fold_contract_path)
    ):
        raise AssertionError("V26e assignment binding is not byte-exact")
    payload = _read(assignment_path)
    rows = list(payload.get("assignments") or ())
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        parent_id = str(row.get("parent_sample_id") or "")
        if not parent_id or parent_id in result:
            raise AssertionError("V26e assignment has an empty or duplicate parent id")
        result[parent_id] = row
    if (
        len(result) != EXPECTED_PARENT_COUNT
        or int(payload.get("known_parent_count", -1)) != EXPECTED_PARENT_COUNT
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or payload.get("grouping") != "capture_content_provenance_component"
        or any(
            str(row.get("label_name")) not in KNOWN_LABELS
            or not str(row.get("provenance_component_id"))
            or int(row.get("fold_id", -1)) not in range(FOLD_COUNT)
            for row in result.values()
        )
    ):
        raise AssertionError("V26e frozen assignment scope is invalid")
    return result


def _first_l1_forward(root: Path, fold: Mapping[str, Any]) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    config, _, _ = _runtime_l1_config(root, fold_id, dict(fold["l1"]))
    dataset = build_known_gate_oof_dataset(
        root,
        config,
        parent_ids=fold["train_parent_ids"],
        feature_cache_path=root / l1_train_feature_cache_path(fold_id),
    )
    batch = next(iter(build_known_gate_dataloader(dataset, batch_size=2, shuffle=False, pin_memory=False)))
    source, _ = _source_protocol(root)
    model, reference, _ = build_known_gate_model_for_config(root, source)
    if reference is not None:
        del reference
    model.config = config
    model.eval()
    with torch.inference_mode():
        output = model(**l1_model_inputs(batch))
    logits = output["sequence_logits"]
    if (
        logits.shape[:2] != batch["tr_features"].shape[:2]
        or logits.shape[-1] != 2
        or not torch.isfinite(logits).all()
    ):
        raise AssertionError("V26e L1 first train batch forward is invalid")
    return {
        "fold_id": fold_id,
        "logits_shape": list(logits.shape),
        "finite": True,
        "class_names": list(KNOWN_GATE_CLASS_NAMES),
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract_path, readiness_path = root / CONTRACT, root / READINESS
    contract, readiness = _read(contract_path), _read(readiness_path)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or int(contract.get("known_train_parent_count", -1)) != EXPECTED_PARENT_COUNT
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or int(contract.get("l1", {}).get("train_epoch_limit", -1)) != L1_TRAIN_EPOCHS
        or int(contract.get("l1", {}).get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or int(contract.get("l2", {}).get("fixed_snapshot_epoch", -1)) != 15
        or contract.get("l2", {}).get("training_executed_in_v26e") is not False
    ):
        raise AssertionError("V26e prepared contract/readiness is invalid")
    assignment = _assignment(root, contract)
    validate_frozen_v26c_sources(root, contract)
    source_contract = _read(root / V26C_CONTRACT)
    all_known_parent_ids = sorted(assignment)
    all_dbb_parent_ids = sorted(
        parent_id
        for parent_id, row in assignment.items()
        if str(row["label_name"]) in {"DroneTarget", "BirdTarget", "BalloonTarget"}
    )
    held_seen: list[str] = []
    folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        train_ids = list(fold["train_parent_ids"])
        held_ids = list(fold["held_out_parent_ids"])
        train_components = {assignment[parent]["provenance_component_id"] for parent in train_ids}
        held_components = {assignment[parent]["provenance_component_id"] for parent in held_ids}
        if train_components & held_components:
            raise AssertionError(f"V26e fold {fold_id} crosses a provenance component")
        l1_binding = dict(fold.get("l1") or {})
        config_path = root / l1_config_path(fold_id)
        if (
            l1_binding.get("config_path") != relative_path(root, config_path)
            or l1_binding.get("config_sha256") != sha256_file(config_path)
        ):
            raise AssertionError(f"V26e fold {fold_id} config binding differs")
        config_payload = _read(config_path)
        declaration = dict(config_payload.get("v26e_oof") or {})
        if (
            config_payload.get("schema_version") != "RADAR-LN3-V26e-OOF-L1"
            or int(declaration.get("train_epoch_limit", -1)) != 32
            or int(declaration.get("fixed_snapshot_epoch", -1)) != 32
            or declaration.get("held_out_checkpoint_selection") is not False
            or config_payload.get("data", {}).get("feature_cache_manifest_path") is not None
            or config_payload.get("data", {}).get("feature_experiment_contract_path") is not None
            or config_payload.get("train", {}).get("paired_baseline_config_path") is not None
        ):
            raise AssertionError(f"V26e fold {fold_id} config declaration differs")
        train_ids_cache = _cache_ids(
            root, root / l1_train_feature_cache_path(fold_id), dict(l1_binding["train_course_cache"])
        )
        held_ids_cache = _cache_ids(
            root, root / l1_held_out_feature_cache_path(fold_id), dict(l1_binding["held_out_course_cache"])
        )
        if set(train_ids_cache) & set(held_ids_cache):
            raise AssertionError(f"V26e fold {fold_id} L1 cache populations overlap")
        if train_ids_cache != selected_event_sample_ids(root, train_ids):
            raise AssertionError(f"V26e fold {fold_id} train cache order differs")
        if held_ids_cache != selected_event_sample_ids(root, held_ids):
            raise AssertionError(f"V26e fold {fold_id} held-out cache order differs")
        _validate_frozen_l1_data(
            root,
            fold,
            all_known_parent_ids=all_known_parent_ids,
        )
        l2_binding = dict(fold.get("l2") or {})
        _validate_frozen_l2(
            root,
            # This validator consumes the sealed V26c fold record.  The
            # V26e record deliberately contains a new L1 binding, while the
            # reused V26c L2 prediction is still bound to V26c's epoch-14 L1.
            # Passing ``fold`` here would therefore compare an immutable V26c
            # artifact against the new V26e L1 path and falsely reject a
            # valid preparation.
            v26c_fold_record(source_contract, fold_id),
            assignment=assignment,
            all_known_parent_ids=all_known_parent_ids,
            all_dbb_parent_ids=all_dbb_parent_ids,
            source_contract=source_contract,
            source_contract_path=root / V26C_CONTRACT,
            source_readiness_path=root / V26C_READINESS,
        )
        for path, expected_hash in (
            (root / l2_checkpoint_path(fold_id), l2_binding.get("checkpoint_sha256")),
            (root / l2_prediction_path(fold_id), l2_binding.get("prediction_sha256")),
            (root / l2_training_summary_path(fold_id), l2_binding.get("summary_sha256")),
        ):
            if not path.is_file() or str(expected_hash) != sha256_file(path):
                raise AssertionError(f"V26e fold {fold_id} V26c L2 reuse binding differs: {path}")
        held_seen.extend(held_ids)
        folds.append(
            {
                "fold_id": fold_id,
                "train_parent_count": len(train_ids),
                "held_out_parent_count": len(held_ids),
                "l1_cache_event_counts": [len(train_ids_cache), len(held_ids_cache)],
                "l2_epoch15_reuse": True,
            }
        )
    if sorted(held_seen) != sorted(assignment) or len(held_seen) != len(set(held_seen)):
        raise AssertionError("V26e held-out coverage does not contain each Known parent once")
    forward = _first_l1_forward(root, fold_record(contract, 0))
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V26e-l1-epoch32-known-only-oof-contract-smoke-v1",
        "contract_path": relative_path(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "fold_count": FOLD_COUNT,
        "known_parent_count": len(assignment),
        "fixed_snapshots": {"l1_epoch": L1_EPOCH, "l2_epoch": 15},
        "fold_checks": folds,
        "first_l1_train_batch_forward": forward,
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "calibration": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V26e fixed epoch-32 L1 OOF isolation.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
