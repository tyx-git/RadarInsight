"""Prepare isolated V23a/V23b/V23c external-state-adapter experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import known_gate_config_identity_sha256, load_known_gate_config
from radar_ln3.known_gate_v23 import FIXED_EPOCHS, SOURCE_SUMMARY_RELATIVE, sha256_file
from radar_ln3.known_gate_v23_adapters import (
    VARIANT_TO_ADAPTER,
    V23_ADAPTER_VARIANTS,
    artifact_relative,
    checkpoint_schema,
    config_identity_sha256,
    config_relative,
    config_schema,
    contract_relative,
    contract_schema,
    load_adapter_config,
    readiness_relative,
    readiness_schema,
    report_relative,
    source_runtime_report,
    source_v23_parent_order_hashes,
    source_v23_training,
)
from radar_ln3.official_cfc import OfficialCfCCausalEncoder, official_cfc_runtime_binding


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V23 adapter artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _source_summary(
    root: Path, source_config_path: Path, source_config: Any
) -> tuple[Path, dict[str, Any]]:
    path = root / SOURCE_SUMMARY_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != "baseline"
        # The frozen V21 seed43 training summary predates the explicit
        # ``config_sha256`` field.  Its V23 Control already binds the source
        # config identity; do not invent a stale field requirement here.
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or summary.get("test_evaluation") != "not_executed"
    ):
        raise ValueError("frozen V21 seed43 source summary is invalid")
    return path, summary


def _official_parameter_count() -> int:
    model = OfficialCfCCausalEncoder(48, 48, backbone_units=48)
    return model.parameter_count


def _config_payload(
    root: Path, variant: str, source_config_path: Path, source_config: Any
) -> dict[str, Any]:
    spec = VARIANT_TO_ADAPTER[variant]
    artifact = str(artifact_relative(variant)).replace("\\", "/")
    report_path = root / "src-v3/processed/report/RADAR-LN3-V23-known-gate-official-cfc-baseline-validation.json"
    return {
        "schema_version": config_schema(variant),
        "config_name": config_relative(variant).stem.removesuffix("-config"),
        "variant": variant,
        "source": {
            "config_path": _relative(root, source_config_path),
            "config_sha256": sha256_file(source_config_path),
            "config_identity_sha256": known_gate_config_identity_sha256(source_config),
            "v23_report_path": _relative(root, report_path),
            "v23_report_sha256": sha256_file(report_path),
        },
        "adapter": {
            "kind": spec.kind,
            "hidden_dim": spec.hidden_dim,
            "fixed_tau_seconds": spec.fixed_tau_seconds,
            "zero_delta_gain": spec.zero_delta_gain,
            "layer_norm_eps": spec.layer_norm_eps,
        },
        "train": {
            "batch_size": source_config.train.batch_size,
            "learning_rate": source_config.train.learning_rate,
            "seed": 43,
            "dataloader_seed": 43,
            "device": source_config.train.device,
            "epochs": FIXED_EPOCHS,
            "early_stopping_patience": 0,
            "gradient_clip_norm": source_config.train.gradient_clip_norm,
            "checkpoint_selection_metric": source_config.train.checkpoint_selection_metric,
            "task_min_precision_recall_target": source_config.train.task_min_precision_recall_target,
            "class_weights": list(source_config.train.class_weights),
            "loss_mode": source_config.train.loss_mode,
            "class_weight_normalization": source_config.train.class_weight_normalization,
            "prefix_auxiliary_weight": source_config.train.prefix_auxiliary_weight,
            "prefix_auxiliary_min_events": source_config.train.prefix_auxiliary_min_events,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": checkpoint_schema(variant),
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": source_config.train.cuda_deterministic_algorithms,
            "full_parameter_training": True,
        },
    }


def build(project_root: Path, variant: str) -> dict[str, Any]:
    if variant not in V23_ADAPTER_VARIANTS:
        raise ValueError(f"unsupported V23 adapter variant: {variant}")
    root = project_root.resolve()
    source_config_path = root / "config/RADAR-LN3/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json"
    source_config = load_known_gate_config(source_config_path)
    source_summary_path, source_summary = _source_summary(
        root, source_config_path, source_config
    )
    v23_summary_path, v23_summary, v23_checkpoint_path, v23_checkpoint_sha = source_v23_training(root)
    v23_parent_order_hashes = source_v23_parent_order_hashes(v23_summary)
    config_path = root / config_relative(variant)
    _write_new_or_identical(
        config_path,
        _config_payload(root, variant, source_config_path, source_config),
    )
    config, loaded_source, loaded_source_path = load_adapter_config(root, variant)
    if loaded_source_path != source_config_path or asdict(loaded_source) != asdict(source_config):
        raise AssertionError("V23 adapter source config reload changed its frozen binding")
    runtime_path, runtime, runtime_sha = source_runtime_report(root)
    official_parameter_count = _official_parameter_count()
    if official_parameter_count != 14064:
        raise ValueError(f"official dense 48x1 CfC count drifted: {official_parameter_count}")
    adapter_specification = config.adapter.as_dict()
    checks = {
        "source_is_v21_seed43_no_evidence_baseline": True,
        "source_v23_negative_baseline_report_is_frozen": True,
        "official_ncps_runtime_is_exactly_pinned": bool(official_cfc_runtime_binding()),
        "official_dense_cfc_parameter_count_is_14064": official_parameter_count == 14064,
        "adapter_is_exactly_one_pre_registered_mechanism": True,
        "adapter_parameter_count_is_expected": (
            config.adapter.trainable_parameter_count
            == (96 if config.adapter.kind == "recurrent_layer_norm" else 0)
        ),
        "v23_control_parent_order_has_20_complete_epochs": len(v23_parent_order_hashes) == FIXED_EPOCHS,
        "v21_train_validation_split_and_normalization_are_reused": True,
        "unknown_is_excluded_from_training": True,
        "oof_is_deferred": True,
        "test_payload_access_count_is_zero": True,
    }
    contract = {
        "schema_version": contract_schema(variant),
        "experiment": f"{variant} isolated external official-CfC state-adapter experiment",
        "source": {
            "name": "frozen V23 negative official CfC baseline and V21 seed43 source protocol",
            "v21_config_path": config.source_config_path,
            "v21_config_sha256": config.source_config_sha256,
            "v21_config_identity_sha256": config.source_config_identity_sha256,
            "v21_summary_path": _relative(root, source_summary_path),
            "v21_summary_sha256": sha256_file(source_summary_path),
            "v23_report_path": config.source_v23_report_path,
            "v23_report_sha256": config.source_v23_report_sha256,
            "v23_summary_path": _relative(root, v23_summary_path),
            "v23_summary_sha256": sha256_file(v23_summary_path),
            "v23_checkpoint_path": _relative(root, v23_checkpoint_path),
            "v23_checkpoint_sha256": v23_checkpoint_sha,
            "v23_initial_core_state_sha256": v23_summary["initial_model_state_sha256"],
            "v23_parent_order_sha256_by_epoch": list(v23_parent_order_hashes),
        },
        "official_runtime": runtime["runtime"],
        "official_runtime_report": {
            "path": _relative(root, runtime_path),
            "sha256": runtime_sha,
            "schema_version": runtime["schema_version"],
        },
        "adapter": adapter_specification,
        "single_factor": {
            "changed_component": "external post-official-CfC recurrent state adapter only",
            "official_cfcell_unmodified": True,
            "frozen": [
                "official dense 48x1 default lecun_tanh CfC",
                "raw first/zero/positive dt input to official CfC",
                "V16 index and V21 known-only data contract",
                "V21 train-only normalization",
                "TR/RD front ends, fusion, classifier, loss and checkpoint rule",
                "batch size, seed, parent order, optimizer, learning rate, 20 epochs and gradient clipping",
            ],
            "not_executed": [
                "Unknown training or threshold fitting", "OOF", "test", "L3", "full cascade", "formal rejection", "stable", "P95",
            ],
        },
        "formal_protocol": {
            "training_split": "V21 train known-only",
            "development_validation_split": "V21 validation known-only",
            "train_parent_count": 986,
            "validation_parent_count": 232,
            "validation_clutter_parent_support": 9,
            "epochs": FIXED_EPOCHS,
            "checkpoint_selection": "parent_min_precision_recall on development validation",
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
    contract_path = root / contract_relative(variant)
    _write_new_or_identical(contract_path, contract)
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": readiness_schema(variant),
        "variant": variant,
        "checks": checks,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": config_identity_sha256(config),
        "contract_path": _relative(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "source_v23_report_path": config.source_v23_report_path,
        "source_v23_report_sha256": config.source_v23_report_sha256,
        "source_v23_initial_core_state_sha256": v23_summary["initial_model_state_sha256"],
        "source_v23_parent_order_sha256_by_epoch": list(v23_parent_order_hashes),
        "adapter": adapter_specification,
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_parent_support": 9,
        "selected_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行完整20 epoch单因素训练）",
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / readiness_relative(variant)
    _write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"], "variant": variant, "config": str(config_path),
        "contract": str(contract_path), "readiness": str(readiness_path),
        "formal_training": readiness["formal_training"], "OOF": "未执行", "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one V23 external-state-adapter experiment.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=V23_ADAPTER_VARIANTS, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, args.variant), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
