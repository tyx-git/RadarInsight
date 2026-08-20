"""Train one V22 fixed-parent-denominator KnownGate OOF fold."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import (
    KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
    KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_v21e_oof import (
    config_relative as v21e_config_relative,
    fold_record,
    load_fold_contract,
    sha256_file,
)
from radar_ln3.known_gate_v22_oof import (
    CONTRACT_RELATIVE,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    FIXED_EPOCHS,
    PREDICTION_SCHEMA,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    artifact_relative,
    config_relative,
    prediction_relative,
)
from train.known_gate_v21e_oof import train_oof_fold


def _fold_binding(contract: dict[str, Any], fold_id: int) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in list(contract.get("folds") or [])
        if int(row.get("fold_id", -1)) == int(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"V22 OOF fold binding is missing or repeated: {fold_id}")
    return matches[0]


def _verify_prepared(
    root: Path,
    *,
    arm: str,
    fold_id: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path, str]:
    if arm != "treatment":
        raise ValueError("V22 only trains a new Treatment; V21e is the frozen Control")
    source_contract, assignment = load_fold_contract(root)
    fold = fold_record(source_contract, fold_id)
    readiness_path = root / READINESS_RELATIVE
    contract_path = root / CONTRACT_RELATIVE
    if not readiness_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("V22 prepare artifacts are missing")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("control_reused_from") != "V21e OOF Treatment"
        or int(readiness.get("fold_count", -1)) != FOLD_COUNT
        or int(readiness.get("known_parent_count", -1)) != 986
        or int(readiness.get("provenance_component_count", -1)) != 975
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V22 OOF readiness/contract is invalid or stale")
    binding = _fold_binding(contract, fold_id)
    config_path = root / config_relative(fold_id)
    control_path = root / v21e_config_relative("treatment", fold_id)
    treatment_binding = dict(binding.get("treatment") or {})
    control_binding = dict(binding.get("control") or {})
    if (
        treatment_binding.get("config_path")
        != str(config_relative(fold_id)).replace("\\", "/")
        or treatment_binding.get("config_sha256") != sha256_file(config_path)
        or control_binding.get("config_path")
        != str(v21e_config_relative("treatment", fold_id)).replace("\\", "/")
        or control_binding.get("config_sha256") != sha256_file(control_path)
        or binding.get("train_parent_ids_sha256") != fold["train_parent_ids_sha256"]
        or binding.get("held_out_parent_ids_sha256")
        != fold["held_out_parent_ids_sha256"]
        or binding.get("normalization_stats_sha256")
        != fold["normalization_stats_sha256"]
        or binding.get("data_contract_sha256") != fold["data_contract_sha256"]
    ):
        raise ValueError(f"V22 fold {fold_id} source binding changed")
    config = load_known_gate_config(config_path)
    control = load_known_gate_config(control_path)
    ignored_train = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "checkpoint_schema",
        "paired_baseline_config_path",
        "class_weight_normalization",
    }
    if (
        known_gate_config_identity_sha256(config)
        != treatment_binding.get("config_identity_sha256")
        or config.schema_version != "RADAR-LN3-V22"
        or config.variant != control.variant
        or config.feature_names != control.feature_names
        or asdict(config.data) != asdict(control.data)
        or asdict(config.model) != asdict(control.model)
        or {
            key: value
            for key, value in asdict(config.train).items()
            if key not in ignored_train
        }
        != {
            key: value
            for key, value in asdict(control.train).items()
            if key not in ignored_train
        }
        or config.train.class_weight_normalization
        != KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION
        or control.train.class_weight_normalization
        != KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION
        or config.train.epochs != FIXED_EPOCHS
    ):
        raise ValueError(f"V22 fold {fold_id} changes more than loss normalization")
    return config, fold, assignment, config_path, sha256_file(readiness_path)


def train_fold(
    project_root: Path,
    *,
    fold_id: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    return train_oof_fold(
        project_root,
        arm="treatment",
        fold_id=fold_id,
        resume_checkpoint=resume_checkpoint,
        verify_prepared=_verify_prepared,
        artifact_resolver=lambda _arm, fold: artifact_relative(fold),
        prediction_resolver=lambda _arm, fold: prediction_relative(fold),
        prediction_schema=PREDICTION_SCHEMA,
        experiment_name="V22 fixed parent balanced OOF",
        loss_class_weight_normalization=KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V22 KnownGate OOF fold.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(FOLD_COUNT), required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            train_fold(
                args.project_root.resolve(),
                fold_id=args.fold_id,
                resume_checkpoint=args.resume_checkpoint,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
