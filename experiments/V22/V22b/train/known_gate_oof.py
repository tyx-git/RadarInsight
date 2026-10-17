"""Train one frozen V21e KnownGate paired OOF arm/fold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from radar_ln3.known_gate import KNOWN_LABELS, known_gate_config_identity_sha256, load_known_gate_config
from radar_ln3.known_gate_oof import FIXED_EPOCHS, READINESS_RELATIVE, READINESS_SCHEMA, config_relative, fold_record, inverse_frequency_weights, load_fold_contract, resolve_path, sha256_file


def _verify_prepared(
    root: Path,
    *,
    arm: str,
    fold_id: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path, str]:
    contract, assignment = load_fold_contract(root)
    fold = fold_record(contract, fold_id)
    readiness_path = root / READINESS_RELATIVE
    if not readiness_path.is_file():
        raise FileNotFoundError(readiness_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("fold_contract_sha256")
        != sha256_file(root / readiness["fold_contract_path"])
        or int(readiness.get("known_parent_count", -1)) != 986
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V21e OOF readiness is invalid or stale")
    binding = dict(dict(fold.get("configs") or {}).get(arm) or {})
    config_path = root / config_relative(arm, fold_id)
    if (
        not config_path.is_file()
        or binding.get("path") != str(config_relative(arm, fold_id)).replace("\\", "/")
        or binding.get("sha256") != sha256_file(config_path)
    ):
        raise ValueError(f"V21e OOF fold {fold_id} {arm} config binding changed")
    config = load_known_gate_config(config_path)
    if (
        known_gate_config_identity_sha256(config) != binding.get("identity_sha256")
        or config.train.epochs != FIXED_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.seed != 43
        or config.train.dataloader_seed != 43
        or config.train.save_every_epoch is not False
    ):
        raise ValueError(f"V21e OOF fold {fold_id} {arm} config violates protocol")
    stats_path = resolve_path(root, config.data.normalization_stats_path)
    data_contract_path = resolve_path(root, config.data.contract_path)
    if (
        not stats_path.is_file()
        or sha256_file(stats_path) != fold["normalization_stats_sha256"]
        or not data_contract_path.is_file()
        or sha256_file(data_contract_path) != fold["data_contract_sha256"]
    ):
        raise ValueError(f"V21e OOF fold {fold_id} stats/data contract changed")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    fold_data_contract = json.loads(data_contract_path.read_text(encoding="utf-8"))
    oof = dict(fold_data_contract.get("oof_fold_contract") or {})
    train_counts = {
        str(key): int(value)
        for key, value in dict(fold["train_label_parent_counts"]).items()
    }
    target_count = sum(train_counts[label] for label in KNOWN_LABELS[:3])
    expected_weights = inverse_frequency_weights(
        (target_count, train_counts["ClutterTarget"])
    )
    if (
        stats.get("schema_version") != "RADAR-LN3-V21"
        or int(stats.get("parent_count", -1)) != len(fold["train_parent_ids"])
        or int(stats.get("allowed_parent_ids_count", -1))
        != len(fold["train_parent_ids"])
        or stats.get("allowed_parent_ids_sha256") != fold["train_parent_ids_sha256"]
        or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
        or int(oof.get("fold_id", -1)) != fold_id
        or oof.get("grouping") != "capture_content_provenance_component"
        or oof.get("train_parent_ids_sha256") != fold["train_parent_ids_sha256"]
        or oof.get("held_out_parent_ids_sha256")
        != fold["held_out_parent_ids_sha256"]
        or oof.get("held_out_checkpoint_selection") is not False
        or int(oof.get("unknown_training_exposure", -1)) != 0
        or int(oof.get("validation_parent_count", -1)) != 0
        or int(oof.get("test_parent_count", -1)) != 0
        or not np.allclose(config.train.class_weights, expected_weights, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"V21e OOF fold {fold_id} contains leakage or drift")
    return config, fold, assignment, config_path, sha256_file(readiness_path)







