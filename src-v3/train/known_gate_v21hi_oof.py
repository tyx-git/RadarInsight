"""Train one V21h/V21i Treatment fold against frozen V21e OOF Control."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from radar_ln3.known_gate import known_gate_config_identity_sha256, load_known_gate_config
from radar_ln3.known_gate_v21e_oof import sha256_file
from radar_ln3.known_gate_v21hi_oof import (
    FOLD_COUNT,
    FIXED_EPOCHS,
    artifact_relative,
    config_relative,
    get_spec,
    prediction_relative,
    v21e_config_relative,
)
from train.known_gate_v21e_oof import (
    _verify_prepared as verify_v21e_prepared,
    train_oof_fold,
)


def _fold_binding(contract: dict[str, Any], fold_id: int) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in list(contract.get("folds") or [])
        if int(row.get("fold_id", -1)) == int(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"V21h/i OOF fold binding is missing or repeated: {fold_id}")
    return matches[0]


def _verify_prepared(
    root: Path,
    *,
    variant: str,
    arm: str,
    fold_id: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path, str]:
    spec = get_spec(variant)
    if arm != "treatment":
        raise ValueError(f"{spec.code} only has a new Treatment arm")
    source_config, fold, assignment, source_config_path, _ = verify_v21e_prepared(
        root,
        arm="treatment",
        fold_id=fold_id,
    )
    readiness_path = root / spec.readiness_relative
    contract_path = root / spec.oof_contract_relative
    if not readiness_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError(f"{spec.code} OOF prepare artifacts are missing")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != spec.readiness_schema
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != spec.oof_contract_schema
        or contract.get("control_reused_from") != "V21e OOF Treatment"
        or int(readiness.get("fold_count", -1)) != FOLD_COUNT
        or int(readiness.get("known_parent_count", -1)) != 986
        or int(readiness.get("provenance_component_count", -1)) != 975
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError(f"{spec.code} OOF readiness/contract is invalid or stale")
    binding = _fold_binding(contract, fold_id)
    config_path = root / config_relative(variant, fold_id)
    treatment_binding = dict(binding.get("treatment") or {})
    control_binding = dict(binding.get("control") or {})
    if (
        treatment_binding.get("config_path")
        != str(config_relative(variant, fold_id)).replace("\\", "/")
        or treatment_binding.get("config_sha256") != sha256_file(config_path)
        or control_binding.get("config_path")
        != str(v21e_config_relative("treatment", fold_id)).replace("\\", "/")
        or control_binding.get("config_sha256") != sha256_file(source_config_path)
        or binding.get("train_parent_ids_sha256")
        != fold["train_parent_ids_sha256"]
        or binding.get("held_out_parent_ids_sha256")
        != fold["held_out_parent_ids_sha256"]
        or binding.get("normalization_stats_sha256")
        != fold["normalization_stats_sha256"]
        or binding.get("data_contract_sha256") != fold["data_contract_sha256"]
    ):
        raise ValueError(f"{spec.code} OOF fold {fold_id} source binding changed")
    config = load_known_gate_config(config_path)
    if (
        known_gate_config_identity_sha256(config)
        != treatment_binding.get("config_identity_sha256")
        or config.variant != spec.variant
        or tuple(config.feature_names) != tuple(spec.feature_names)
        or tuple(config.feature_names[: len(source_config.feature_names)])
        != tuple(source_config.feature_names)
        or config.model.evidence_dim != len(spec.feature_names)
        or source_config.model.evidence_dim != 1
        or config.train.epochs != FIXED_EPOCHS
        or config.train.seed != source_config.train.seed
        or config.train.dataloader_seed != source_config.train.dataloader_seed
        or config.train.save_every_epoch is not False
        or config.train.paired_baseline_config_path
        != str(v21e_config_relative("treatment", fold_id)).replace("\\", "/")
        or not np.allclose(
            config.train.class_weights,
            source_config.train.class_weights,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError(f"{spec.code} OOF fold {fold_id} config violates protocol")
    ignored_data = {
        "feature_cache_manifest_path",
        "expected_feature_cache_manifest_sha256",
        "train_feature_cache_path",
        "validation_feature_cache_path",
        "expected_train_feature_cache_sha256",
        "expected_validation_feature_cache_sha256",
        "feature_experiment_contract_path",
        "expected_feature_experiment_contract_sha256",
    }
    if {
        key: value
        for key, value in asdict(config.data).items()
        if key not in ignored_data
    } != {
        key: value
        for key, value in asdict(source_config.data).items()
        if key not in ignored_data
    }:
        raise ValueError(f"{spec.code} OOF fold {fold_id} changed base data semantics")
    return config, fold, assignment, config_path, sha256_file(readiness_path)


def train_variant_fold(
    project_root: Path,
    *,
    variant: str,
    fold_id: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    spec = get_spec(variant)

    def verify(root: Path, *, arm: str, fold_id: int):
        return _verify_prepared(
            root,
            variant=variant,
            arm=arm,
            fold_id=fold_id,
        )

    return train_oof_fold(
        project_root,
        arm="treatment",
        fold_id=fold_id,
        resume_checkpoint=resume_checkpoint,
        verify_prepared=verify,
        artifact_resolver=lambda _arm, fold: artifact_relative(variant, fold),
        prediction_resolver=lambda _arm, fold: prediction_relative(variant, fold),
        prediction_schema=spec.prediction_schema,
        experiment_name=f"{spec.code} OOF",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V21h/V21i OOF fold.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("h", "i"), required=True)
    parser.add_argument("--fold-id", type=int, choices=range(FOLD_COUNT), required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            train_variant_fold(
                args.project_root.resolve(),
                variant=args.variant,
                fold_id=args.fold_id,
                resume_checkpoint=args.resume_checkpoint,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
