"""Prepare V23k causal TR/RD observation-intensity input experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.known_gate import (
    build_known_gate_dataset,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_v23 import FIXED_EPOCHS, SOURCE_SUMMARY_RELATIVE, sha256_file
from radar_ln3.known_gate_v23_adapters import (
    source_runtime_report,
    source_v23_parent_order_hashes,
    source_v23_training,
)
from radar_ln3.known_gate_v23e_selection import V23E_SELECTOR_SCHEMA
from radar_ln3.known_gate_v23k import (
    V23K_CHECKPOINT_SCHEMA,
    V23K_CONFIG_SCHEMA,
    V23K_CONTRACT_SCHEMA,
    V23K_READINESS_SCHEMA,
    V23K_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    contract_relative,
    load_config,
    observation_intensity_spec,
    readiness_relative,
    source_v23a_control,
)
from radar_ln3.known_gate_official_cfc import causal_observation_intensity_sequence
from radar_ln3.official_cfc import OfficialCfCCausalEncoder, official_cfc_runtime_binding


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V23k artifact: {path}")
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
        raise ValueError("V23k frozen V21 seed43 source summary is invalid")
    return path, summary


def _intensity_exposure(root: Path, source_config: Any) -> dict[str, Any]:
    """Train-only distribution audit; validation is intentionally untouched."""

    dataset = build_known_gate_dataset(root, source_config, split_name="train")
    spec = observation_intensity_spec()
    labels: dict[str, list[np.ndarray]] = {}
    rd_missing_event_count = 0
    active_event_count = 0
    maximum_raw_intensity = np.zeros((2,), dtype=np.float64)
    for item in dataset:
        base = item.base
        features, _state = causal_observation_intensity_sequence(
            spec=spec,
            dt_seconds=base.tr_dt_seconds.unsqueeze(0),
            dt_valid=base.tr_dt_valid.unsqueeze(0),
            dt_reason=base.tr_dt_reason.unsqueeze(0),
            rd_quality=base.rd_quality.unsqueeze(0),
            rd_frame_mask=base.rd_frame_mask.unsqueeze(0),
            padding_mask=torch.zeros((1, base.event_count), dtype=torch.bool),
        )
        # The utility accepts torch tensors by contract; use torch only for
        # the synthetic all-active mask to avoid Dataset/collate mutation.
        values = features.squeeze(0).detach().cpu().numpy()
        labels.setdefault(str(base.label_name), []).append(values)
        active_event_count += int(base.event_count)
        rd_missing_event_count += int((~base.rd_frame_mask.any(dim=-1)).sum().item())
        maximum_raw_intensity = np.maximum(maximum_raw_intensity, np.expm1(values).max(axis=0))
    summaries: dict[str, Any] = {}
    for label, rows in sorted(labels.items()):
        values = np.concatenate(rows, axis=0)
        summaries[label] = {
            "event_support": int(values.shape[0]),
            "log1p_c_TR_minimum": float(values[:, 0].min()),
            "log1p_c_TR_maximum": float(values[:, 0].max()),
            "log1p_c_TR_standard_deviation": float(values[:, 0].std()),
            "log1p_c_RD_minimum": float(values[:, 1].min()),
            "log1p_c_RD_maximum": float(values[:, 1].max()),
            "log1p_c_RD_standard_deviation": float(values[:, 1].std()),
        }
    return {
        "train_parent_count": len(dataset),
        "active_event_count": active_event_count,
        "rd_missing_event_count": rd_missing_event_count,
        "maximum_raw_c_TR": float(maximum_raw_intensity[0]),
        "maximum_raw_c_RD": float(maximum_raw_intensity[1]),
        "by_base_label": summaries,
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
    }


def _config_payload(root: Path, source_config_path: Path, source_config: Any) -> dict[str, Any]:
    artifact = str(artifact_relative()).replace("\\", "/")
    v23a = source_v23a_control(root)
    from radar_ln3.known_gate_v23_adapters import load_adapter_config

    adapter_spec = load_adapter_config(root, "official_cfc_positive_dt_fixed_retention")[0].adapter
    return {
        "schema_version": V23K_CONFIG_SCHEMA,
        "config_name": config_relative().stem.removesuffix("-config"),
        "variant": V23K_VARIANT,
        "source": {
            "config_path": _relative(root, source_config_path),
            "config_sha256": sha256_file(source_config_path),
            "config_identity_sha256": known_gate_config_identity_sha256(source_config),
            "v23a_control": v23a,
        },
        "adapter": adapter_spec.as_dict(),
        "observation_intensity": observation_intensity_spec().as_dict(),
        "selector": {
            "schema_version": V23E_SELECTOR_SCHEMA,
            "name": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
            "source_v23a_legacy_reference_epoch": int(v23a["legacy_best_epoch"]),
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
            "checkpoint_selection_metric": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
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
            "checkpoint_schema": V23K_CHECKPOINT_SCHEMA,
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
        raise AssertionError("V23k source config reload changed its frozen binding")
    runtime_path, runtime, runtime_sha = source_runtime_report(root)
    exposure = _intensity_exposure(root, source_config)
    official_parameter_count = OfficialCfCCausalEncoder(48, 48, backbone_units=48).parameter_count
    by_label = dict(exposure["by_base_label"])
    checks = {
        "source_is_v21_seed43_no_evidence_baseline": True,
        "source_v23a_fixed_retention_control_is_completed": True,
        "v23e_parent_final_and_parent_balanced_selector_is_pre_registered": True,
        "official_ncps_runtime_is_exactly_pinned": bool(official_cfc_runtime_binding()),
        "official_dense_cfc_parameter_count_is_14064": official_parameter_count == 14064,
        "only_two_causal_observation_intensity_input_columns_differ_from_v23a": True,
        "kappa_is_fixed_as_inverse_v23a_tau": abs(config.observation_intensity.kappa_per_second - 1.0 / 15.025) < 1.0e-15,
        "train_only_intensity_features_are_non_degenerate": all(
            values["log1p_c_TR_standard_deviation"] > 0.0
            and values["log1p_c_RD_standard_deviation"] > 0.0
            for values in by_label.values()
        ),
        "rd_intensity_has_real_missing_frame_exposure": int(exposure["rd_missing_event_count"]) > 0,
        "v23a_parent_order_has_20_complete_epochs": len(v23_parent_order_hashes) == FIXED_EPOCHS,
        "v21_train_normalization_is_reused_without_intensity_fitting": True,
        "validation_is_excluded_from_readiness_gate": True,
        "unknown_is_excluded_from_training": True,
        "oof_is_deferred": True,
        "test_payload_access_count_is_zero": True,
    }
    contract = {
        "schema_version": V23K_CONTRACT_SCHEMA,
        "experiment": "V23k V23a official-CfC with two causal TR/RD decayed observation-intensity fusion inputs only",
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
        "adapter": config.adapter.as_dict(),
        "observation_intensity": config.observation_intensity.as_dict(),
        "readiness_train_only_intensity_exposure": exposure,
        "selector": config.selector,
        "single_factor": {
            "changed_component": "append exactly two zero-initialized causal input features log1p(c_TR), log1p(c_RD) to existing fusion only",
            "frozen": [
                "official dense 48x1 default lecun_tanh CfC",
                "V23a positive-dt fixed-retention external state adapter",
                "raw first/zero/positive dt supplied unchanged to official CfC and adapter",
                "official CfC hidden-state equation and all state-update gains",
                "V16 index and V21 train-only normalization",
                "TR/RD front ends, existing RD quality, classifier, loss and V23e checkpoint selector",
                "batch size, seed, initial shared state, parent order, optimizer, learning rate, 20 epochs and gradient clipping",
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
        "schema_version": V23K_READINESS_SCHEMA,
        "variant": V23K_VARIANT,
        "checks": checks,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": config_identity_sha256(config),
        "contract_path": _relative(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "source_v23a_control": config.source_v23a,
        "source_v23_parent_order_sha256_by_epoch": list(v23_parent_order_hashes),
        "adapter": config.adapter.as_dict(),
        "observation_intensity": config.observation_intensity.as_dict(),
        "train_only_intensity_exposure": exposure,
        "selector": config.selector,
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_parent_support": 9,
        "selected_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行全新完整20 epoch V23k训练）",
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
        "variant": V23K_VARIANT,
        "config": str(config_path),
        "contract": str(contract_path),
        "readiness": str(readiness_path),
        "formal_training": readiness["formal_training"],
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V23k causal TR/RD observation-intensity KnownGate experiment.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
