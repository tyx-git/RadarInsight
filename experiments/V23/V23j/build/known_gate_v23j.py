"""Prepare V23j: V23a official-CfC with a pre-registered prefix-loss ramp."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import known_gate_config_identity_sha256, load_known_gate_config
from radar_ln3.known_gate_v23 import FIXED_EPOCHS, SOURCE_SUMMARY_RELATIVE, sha256_file
from radar_ln3.known_gate_v23_adapters import (
    V23A_VARIANT,
    load_adapter_config,
    source_runtime_report,
    source_v23_parent_order_hashes,
    source_v23_training,
)
from radar_ln3.known_gate_v23e_selection import V23E_SELECTOR_SCHEMA
from radar_ln3.known_gate_v23j import (
    V23J_CHECKPOINT_SCHEMA,
    V23J_CONFIG_SCHEMA,
    V23J_CONTRACT_SCHEMA,
    V23J_PREFIX_RAMP_END_EPOCH,
    V23J_PREFIX_RAMP_END_WEIGHT,
    V23J_PREFIX_RAMP_KIND,
    V23J_PREFIX_RAMP_START_EPOCH,
    V23J_PREFIX_RAMP_START_WEIGHT,
    V23J_READINESS_SCHEMA,
    V23J_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    contract_relative,
    load_config,
    prefix_auxiliary_weight_for_epoch,
    readiness_relative,
    source_v23a_control,
)
from radar_ln3.official_cfc import OfficialCfCCausalEncoder, official_cfc_runtime_binding


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V23j artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _source_summary(root: Path) -> tuple[Path, dict[str, Any]]:
    path = (root / SOURCE_SUMMARY_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = dict(json.loads(path.read_text(encoding="utf-8")))
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != "baseline"
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or summary.get("test_evaluation") != "not_executed"
    ):
        raise ValueError("V23j frozen V21 seed43 source summary is invalid")
    return path, summary


def _ramp_payload() -> dict[str, Any]:
    return {
        "kind": V23J_PREFIX_RAMP_KIND,
        "start_weight": V23J_PREFIX_RAMP_START_WEIGHT,
        "end_weight": V23J_PREFIX_RAMP_END_WEIGHT,
        "start_epoch": V23J_PREFIX_RAMP_START_EPOCH,
        "end_epoch": V23J_PREFIX_RAMP_END_EPOCH,
    }


def _config_payload(
    root: Path, source_config_path: Path, source_config: Any
) -> dict[str, Any]:
    artifact = str(artifact_relative()).replace("\\", "/")
    v23a = source_v23a_control(root)
    adapter_spec = load_adapter_config(root, V23A_VARIANT)[0].adapter
    return {
        "schema_version": V23J_CONFIG_SCHEMA,
        "config_name": config_relative().stem.removesuffix("-config"),
        "variant": V23J_VARIANT,
        "source": {
            "config_path": _relative(root, source_config_path),
            "config_sha256": sha256_file(source_config_path),
            "config_identity_sha256": known_gate_config_identity_sha256(source_config),
            "v23a_control": v23a,
        },
        "adapter": adapter_spec.__dict__,
        "selector": {
            "schema_version": V23E_SELECTOR_SCHEMA,
            "name": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
            "source_v23a_legacy_reference_epoch": int(v23a["legacy_best_epoch"]),
        },
        "prefix_ramp": _ramp_payload(),
        "train": {
            "batch_size": source_config.train.batch_size,
            "learning_rate": source_config.train.learning_rate,
            "seed": 43,
            "dataloader_seed": 43,
            "device": source_config.train.device,
            "epochs": FIXED_EPOCHS,
            "early_stopping_patience": 0,
            "gradient_clip_norm": source_config.train.gradient_clip_norm,
            "checkpoint_selection_metric": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
            "task_min_precision_recall_target": source_config.train.task_min_precision_recall_target,
            "class_weights": list(source_config.train.class_weights),
            "loss_mode": source_config.train.loss_mode,
            "class_weight_normalization": source_config.train.class_weight_normalization,
            "prefix_auxiliary_min_events": source_config.train.prefix_auxiliary_min_events,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": V23J_CHECKPOINT_SCHEMA,
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": source_config.train.cuda_deterministic_algorithms,
            "full_parameter_training": True,
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_config_path = root / "config/RADAR-LN3/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json"
    source_config = load_known_gate_config(source_config_path)
    source_summary_path, _source_summary_payload = _source_summary(root)
    v23_summary_path, v23_summary, v23_checkpoint_path, v23_checkpoint_sha = source_v23_training(root)
    v23_parent_order_hashes = source_v23_parent_order_hashes(v23_summary)
    config_path = root / config_relative()
    _write_new_or_identical(config_path, _config_payload(root, source_config_path, source_config))
    config, loaded_source, loaded_source_path = load_config(root)
    if loaded_source_path != source_config_path or asdict(loaded_source) != asdict(source_config):
        raise AssertionError("V23j source config reload changed its frozen binding")
    runtime_path, runtime, runtime_sha = source_runtime_report(root)
    official_parameter_count = OfficialCfCCausalEncoder(48, 48, backbone_units=48).parameter_count
    ramp_schedule = [
        prefix_auxiliary_weight_for_epoch(epoch, config.prefix_ramp)
        for epoch in range(1, FIXED_EPOCHS + 1)
    ]
    checks = {
        "source_is_v21_seed43_no_evidence_baseline": True,
        "source_v23a_fixed_retention_control_is_completed": True,
        "v23e_parent_final_and_parent_balanced_selector_is_pre_registered": True,
        "official_ncps_runtime_is_exactly_pinned": bool(official_cfc_runtime_binding()),
        "official_dense_cfc_parameter_count_is_14064": official_parameter_count == 14064,
        "only_prefix_auxiliary_weight_schedule_differs_from_v23a": True,
        "prefix_ramp_is_pre_registered_linear_0_1_to_1_0_across_full_20_epochs": (
            ramp_schedule[0] == V23J_PREFIX_RAMP_START_WEIGHT
            and ramp_schedule[-1] == V23J_PREFIX_RAMP_END_WEIGHT
            and ramp_schedule[9] == 0.1 + 0.9 * 9 / 19
        ),
        "v23a_parent_order_has_20_complete_epochs": len(v23_parent_order_hashes) == FIXED_EPOCHS,
        "v21_train_validation_split_and_normalization_are_reused": True,
        "unknown_is_excluded_from_training": True,
        "oof_is_deferred": True,
        "test_payload_access_count_is_zero": True,
    }
    contract = {
        "schema_version": V23J_CONTRACT_SCHEMA,
        "experiment": "V23j V23a fixed-retention official-CfC with only an epoch-linear prefix auxiliary weight ramp from 0.1 to 1.0",
        "source": {
            "v21_config_path": config.source_config_path,
            "v21_config_sha256": config.source_config_sha256,
            "v21_config_identity_sha256": config.source_config_identity_sha256,
            "v21_summary_path": _relative(root, source_summary_path),
            "v21_summary_sha256": sha256_file(source_summary_path),
            "v23_summary_path": _relative(root, v23_summary_path),
            "v23_summary_sha256": sha256_file(v23_summary_path),
            "v23_checkpoint_path": _relative(root, v23_checkpoint_path),
            "v23_checkpoint_sha256": v23_checkpoint_sha,
            "v23a_control": config.source_v23a,
            "v23_parent_order_sha256_by_epoch": list(v23_parent_order_hashes),
        },
        "official_runtime": runtime["runtime"],
        "official_runtime_report": {
            "path": _relative(root, runtime_path),
            "sha256": runtime_sha,
            "schema_version": runtime["schema_version"],
        },
        "adapter": config.adapter.__dict__,
        "selector": config.selector,
        "prefix_ramp": config.prefix_ramp.as_dict(),
        "prefix_ramp_schedule_by_epoch": [
            {"epoch": epoch, "prefix_auxiliary_weight": weight}
            for epoch, weight in enumerate(ramp_schedule, start=1)
        ],
        "single_factor": {
            "changed_component": "epoch-level prefix_auxiliary_weight schedule from V23a fixed 0.1 to pre-registered linear 0.1 at epoch 1 through 1.0 at epoch 20 only",
            "frozen": [
                "official dense 48x1 default lecun_tanh CfC",
                "V23a positive-dt fixed-retention state adapter",
                "raw first/zero/positive dt input",
                "V16 index and V21 train-only normalization",
                "TR/RD front ends, fusion, classifier",
                "batch size, seed, initial state and parent order",
                "optimizer, learning rate, 20 epochs and gradient clipping",
                "class weights and batch_weight_sum normalization",
                "V23e parent-final plus parent-balanced eligible-prefix checkpoint selector",
            ],
            "not_executed": [
                "Unknown training or threshold fitting",
                "OOF",
                "test",
                "L3",
                "full cascade",
                "formal rejection",
                "stable",
                "P95",
            ],
        },
        "formal_protocol": {
            "training_split": "V21 train known-only",
            "development_validation_split": "V21 validation known-only",
            "train_parent_count": 986,
            "validation_parent_count": 232,
            "validation_clutter_parent_support": 9,
            "epochs": FIXED_EPOCHS,
            "prefix_ramp": config.prefix_ramp.as_dict(),
            "checkpoint_selection": config.train.checkpoint_selection_metric,
            "parent_order_pairing": "each treatment epoch must exactly match the frozen V23 epoch parent-order SHA256",
            "full_parameter_training": True,
        },
        "unknown_training_exposure": 0,
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / contract_relative()
    _write_new_or_identical(contract_path, contract)
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": V23J_READINESS_SCHEMA,
        "variant": V23J_VARIANT,
        "checks": checks,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": config_identity_sha256(config),
        "contract_path": _relative(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "source_v23a_control": config.source_v23a,
        "source_v23_parent_order_sha256_by_epoch": list(v23_parent_order_hashes),
        "adapter": config.adapter.__dict__,
        "selector": config.selector,
        "prefix_ramp": config.prefix_ramp.as_dict(),
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_parent_support": 9,
        "selected_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行全新完整20 epoch V23j训练）",
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / readiness_relative()
    _write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "variant": V23J_VARIANT,
        "config": str(config_path),
        "contract": str(contract_path),
        "readiness": str(readiness_path),
        "formal_training": readiness["formal_training"],
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V23j prefix-ramp-only KnownGate experiment.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
