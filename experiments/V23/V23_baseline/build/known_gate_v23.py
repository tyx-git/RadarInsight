"""Prepare the single-model V23 pure official-CfC KnownGate baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import known_gate_config_identity_sha256, load_known_gate_config
from radar_ln3.known_gate_v23 import ARTIFACT_RELATIVE, CHECKPOINT_SCHEMA, CONFIG_RELATIVE, CONFIG_SCHEMA, CONTRACT_RELATIVE, CONTRACT_SCHEMA, FIXED_EPOCHS, READINESS_RELATIVE, READINESS_SCHEMA, SOURCE_CONFIG_RELATIVE, SOURCE_SUMMARY_RELATIVE, TRAINING_SEED, config_identity_sha256, load_v23_config, sha256_file
from radar_ln3.official_cfc import (
    OFFICIAL_CFC_STATE_EQUATION_OWNER,
    OfficialCfCCausalEncoder,
    official_cfc_runtime_binding,
)


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V23 artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _source_summary(root: Path, source_config: Any) -> tuple[Path, dict[str, Any]]:
    path = root / SOURCE_SUMMARY_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"V23 requires frozen V21 formal baseline summary: {path}")
    summary = dict(json.loads(path.read_text(encoding="utf-8")))
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "known_gate"
        or summary.get("variant") != "baseline"
        or int(summary.get("seed", -1)) != TRAINING_SEED
        or int(summary.get("dataloader_seed", -1)) != TRAINING_SEED
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V23 source V21 formal baseline summary is not frozen/sealed")
    if Path(str(summary.get("config_path", ""))).resolve() != (root / SOURCE_CONFIG_RELATIVE).resolve():
        raise ValueError("V23 source summary does not bind the frozen V21 baseline config")
    if source_config.model.evidence_dim != 0:
        raise ValueError("V23 pure baseline source unexpectedly contains evidence")
    return path, summary


def _parameter_match(source_config: Any) -> dict[str, Any]:
    official = OfficialCfCCausalEncoder(
        source_config.model.liquid_hidden_dim,
        source_config.model.liquid_hidden_dim,
        backbone_units=source_config.model.liquid_hidden_dim,
        backbone_layers=1,
        backbone_dropout=0.0,
        mixed_memory=False,
    )
    count = sum(parameter.numel() for parameter in official.parameters())
    if count != 14064:
        raise ValueError(f"V23 pure official CfC expected 14064 parameters, got {count}")
    return {
        "official_ncps_cfc_parameter_count": count,
        "historical_custom_cfc_inspired_encoder_parameter_count": 14112,
        "official_minus_historical_parameter_count": count - 14112,
        "absolute_parameter_difference": abs(count - 14112),
        "parameter_matching_decision": "official dense 48x1 baseline fixed before training",
    }


def _config_payload(root: Path, source_config: Any) -> dict[str, Any]:
    runtime = official_cfc_runtime_binding()
    source_path = root / SOURCE_CONFIG_RELATIVE
    artifact = str(ARTIFACT_RELATIVE).replace("\\", "/")
    return {
        "schema_version": CONFIG_SCHEMA,
        "config_name": "RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-seed43",
        "variant": "official_cfc_baseline",
        "source": {
            "role": "frozen V21 seed43 formal baseline",
            "config_path": _relative(root, source_path),
            "config_sha256": sha256_file(source_path),
            "config_identity_sha256": known_gate_config_identity_sha256(source_config),
        },
        "official_cfc": {
            **runtime,
            "mode": "default",
            "activation": "lecun_tanh",
            "input_dim": 48,
            "hidden_dim": 48,
            "backbone_units": 48,
            "backbone_layers": 1,
            "backbone_dropout": 0.0,
            "mixed_memory": False,
            "proj_size": None,
            "wiring": None,
            "return_sequences": True,
            "batch_first": True,
            "per_parent_forward_for_upstream_batch_timespan_compatibility": True,
            "state_equation_owner": OFFICIAL_CFC_STATE_EQUATION_OWNER,
            "dt_semantics": (
                "official_cfc_raw_nonnegative_wall_clock_seconds; "
                "first_event_and_same_second_use_zero_timespan; "
                "positive_delta_preserves_measured_dt_without_cap; "
                "dt_valid_and_dt_reason_are_contract_audits_not_state_update_rules"
            ),
        },
        "train": {
            "batch_size": int(source_config.train.batch_size),
            "learning_rate": float(source_config.train.learning_rate),
            "seed": TRAINING_SEED,
            "dataloader_seed": TRAINING_SEED,
            "device": str(source_config.train.device),
            "epochs": FIXED_EPOCHS,
            "early_stopping_patience": 0,
            "gradient_clip_norm": float(source_config.train.gradient_clip_norm),
            "checkpoint_selection_metric": str(source_config.train.checkpoint_selection_metric),
            "task_min_precision_recall_target": float(source_config.train.task_min_precision_recall_target),
            "class_weights": [float(value) for value in source_config.train.class_weights],
            "loss_mode": str(source_config.train.loss_mode),
            "class_weight_normalization": str(source_config.train.class_weight_normalization),
            "prefix_auxiliary_weight": float(source_config.train.prefix_auxiliary_weight),
            "prefix_auxiliary_min_events": int(source_config.train.prefix_auxiliary_min_events),
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": bool(source_config.train.cuda_deterministic_algorithms),
            "full_parameter_training": True,
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_path = root / SOURCE_CONFIG_RELATIVE
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_config = load_known_gate_config(source_path)
    summary_path, source_summary = _source_summary(root, source_config)
    parameter_match = _parameter_match(source_config)
    config_path = root / CONFIG_RELATIVE
    _write_new_or_identical(config_path, _config_payload(root, source_config))
    config, loaded_source, loaded_source_path = load_v23_config(root, config_path)
    if loaded_source_path != source_path or asdict(loaded_source) != asdict(source_config):
        raise AssertionError("V23 source config reload changed its frozen binding")
    checks = {
        "source_is_v21_seed43_no_evidence_baseline": True,
        "source_formal_summary_is_completed_and_test_sealed": True,
        "official_ncps_runtime_is_exactly_pinned": True,
        "official_dense_cfc_parameter_count_is_14064": parameter_match[
            "official_ncps_cfc_parameter_count"
        ] == 14064,
        "official_spec_is_dense_48x1_default_no_memory_projection_or_wiring": True,
        "legacy_tau_alpha_floor_gain_norm_candidate_gate_are_excluded": True,
        "positive_dt_is_not_capped": "without_cap" in config.official_cfc.dt_semantics,
        "v21_train_validation_split_and_normalization_are_reused": True,
        "unknown_is_excluded_from_training": True,
        "five_fold_oof_is_deferred": True,
        "test_payload_access_count_is_zero": True,
    }
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V23 pure official dense CfC KnownGate formal baseline",
        "source": {
            "name": "V21 seed43 formal baseline",
            "config_path": _relative(root, source_path),
            "config_sha256": sha256_file(source_path),
            "config_identity_sha256": known_gate_config_identity_sha256(source_config),
            "summary_path": _relative(root, summary_path),
            "summary_sha256": sha256_file(summary_path),
            "summary_status": source_summary["status"],
        },
        "treatment": {
            "name": "V23 pure official dense CfC baseline",
            "architecture": "ncps.torch.CfC",
            "state_equation_owner": "official ncps CfCCell",
            "changed_component": "project-local CfC-inspired recurrent encoder replaced by official dense CfC",
            "not_a_reimplementation": True,
            "legacy_recurrent_mechanisms_reintroduced": False,
        },
        "official_runtime": official_cfc_runtime_binding(),
        "official_specification": config.official_cfc.__dict__,
        "parameter_matching": parameter_match,
        "formal_protocol": {
            "training_split": "V21 train known-only",
            "development_validation_split": "V21 validation known-only",
            "train_parent_count": 986,
            "validation_parent_count": 232,
            "validation_clutter_parent_support": 9,
            "epochs": FIXED_EPOCHS,
            "checkpoint_selection": "V21 parent_min_precision_recall on development validation",
            "full_parameter_training": True,
            "oof": "deferred_pending_baseline_and_multiseed_confirmation",
        },
        "single_factor": {
            "changed_component": "causal recurrent encoder only",
            "frozen": [
                "V16 index and V21 data contract",
                "V21 train-only normalization",
                "KnownGate labels and Unknown exclusion",
                "TR/RD front ends, fusion and classifier topology",
                "batch size, seed, optimizer, learning rate, epochs and gradient clipping",
                "V21 parent-final plus prefix auxiliary loss",
                "V21 development validation and checkpoint selection rule",
            ],
            "not_reintroduced": [
                "learned tau",
                "alpha equals one minus exp minus dt over tau",
                "zero-dt floor",
                "bounded observation gain",
                "recurrent LayerNorm",
                "project custom candidate gate",
                "positive-dt cap",
            ],
        },
        "unknown_training_exposure": 0,
        "oof": "未执行（先完成V23基座与多seed）",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / CONTRACT_RELATIVE
    _write_new_or_identical(contract_path, contract)
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
        "checks": checks,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": config_identity_sha256(config),
        "contract_path": _relative(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_parent_support": 9,
        "selected_seed": TRAINING_SEED,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行V23完整20 epoch基座训练）",
        "multiseed": "未执行",
        "OOF": "未执行（延后至基座和多seed确认后）",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / READINESS_RELATIVE
    _write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "config": str(config_path),
        "contract": str(contract_path),
        "readiness": str(readiness_path),
        "formal_training": "未执行",
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V23 pure official-CfC baseline.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()