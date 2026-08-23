"""Prepare V23l's official-CfC full temporal-adapter KnownGate arm."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import build_known_gate_dataset, known_gate_config_identity_sha256, load_known_gate_config
from radar_ln3.known_gate_v23 import FIXED_EPOCHS
from radar_ln3.known_gate_v23e_selection import V23E_SELECTOR_SCHEMA
from radar_ln3.known_gate_v23l import (
    V23L_CHECKPOINT_SCHEMA,
    V23L_CONFIG_SCHEMA,
    V23L_CONTRACT_SCHEMA,
    V23L_READINESS_SCHEMA,
    V23L_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    contract_relative,
    full_temporal_adapter_spec,
    load_config,
    readiness_relative,
    source_v23g_density,
)
from radar_ln3.known_gate_v23g import V23G_TRAINING_SEED, sha256_file
from radar_ln3.official_cfc import OfficialCfCCausalEncoder, official_cfc_runtime_binding


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to overwrite a different V23l prepare artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _same_second_exposure(root: Path, source_config: Any, *, splits: tuple[str, ...]) -> dict[str, Any]:
    """Audit timing exposure from the frozen V21e data only.

    The development values are reported as a descriptive exposure check, never
    used to tune V23l or pass readiness.
    """

    from radar_ln3.data import DT_REASON_ZERO_DELTA

    result: dict[str, Any] = {}
    for split in splits:
        dataset = build_known_gate_dataset(root, source_config, split_name=split)
        labels: dict[str, dict[str, int]] = {}
        total = {
            "parent_count": 0,
            "zero_delta_event_count": 0,
            "density_exposed_parent_count": 0,
            "maximum_causal_same_second_ordinal": 1,
        }
        for item in dataset:
            label = str(item.base.label_name)
            values = labels.setdefault(
                label,
                {
                    "parent_count": 0,
                    "zero_delta_event_count": 0,
                    "density_exposed_parent_count": 0,
                    "maximum_causal_same_second_ordinal": 1,
                },
            )
            ordinal = 0
            maximum = 1
            zero_count = 0
            for reason in item.base.tr_dt_reason.tolist():
                if reason in (1, 3):
                    ordinal = 1
                elif reason == DT_REASON_ZERO_DELTA:
                    ordinal += 1
                    zero_count += 1
                    maximum = max(maximum, ordinal)
                else:
                    raise ValueError("V23l source has unsupported active dt reason")
            for destination in (values, total):
                destination["parent_count"] += 1
                destination["zero_delta_event_count"] += zero_count
                destination["density_exposed_parent_count"] += int(maximum >= 2)
                destination["maximum_causal_same_second_ordinal"] = max(
                    destination["maximum_causal_same_second_ordinal"], maximum
                )
        result[split] = {"labels": labels, "total": total}
    return result


def _config_payload(root: Path, source_path: Path, source_config: Any) -> dict[str, Any]:
    artifact = str(artifact_relative()).replace("\\", "/")
    density = source_v23g_density(root)
    return {
        "schema_version": V23L_CONFIG_SCHEMA,
        "config_name": config_relative().stem.removesuffix("-config"),
        "variant": V23L_VARIANT,
        "source": {
            "config_path": _relative(root, source_path),
            "config_sha256": sha256_file(source_path),
            "config_identity_sha256": known_gate_config_identity_sha256(source_config),
            "v23g_density": density,
        },
        "adapter": full_temporal_adapter_spec().as_dict(),
        "selector": {
            "schema_version": V23E_SELECTOR_SCHEMA,
            "name": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
            "source_v23a_legacy_reference_epoch": 14,
        },
        "train": {
            "batch_size": source_config.train.batch_size,
            "learning_rate": source_config.train.learning_rate,
            "seed": V23G_TRAINING_SEED,
            "dataloader_seed": V23G_TRAINING_SEED,
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
            "checkpoint_schema": V23L_CHECKPOINT_SCHEMA,
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": source_config.train.cuda_deterministic_algorithms,
            "full_parameter_training": True,
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_path = root / "config/RADAR-LN3/RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-seed43-config.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_config = load_known_gate_config(source_path)
    config_path = root / config_relative()
    _write_new_or_identical(config_path, _config_payload(root, source_path, source_config))
    config, loaded_source, loaded_source_path = load_config(root)
    if loaded_source_path != source_path.resolve() or asdict(loaded_source) != asdict(source_config):
        raise AssertionError("V23l source config reload changed its frozen binding")
    density = source_v23g_density(root)
    # Readiness may inspect training data only.  Development timing strata are
    # intentionally deferred to the evaluation report after the 20-epoch run.
    exposure = _same_second_exposure(root, source_config, splits=("train",))
    official_parameter_count = OfficialCfCCausalEncoder(48, 48, backbone_units=48).parameter_count
    checks = {
        "source_is_completed_v21e_course_prefix_protocol": True,
        "self_developed_comparator_is_completed_v23g_density": True,
        "same_course_feature_cache_and_train_normalization_are_reused": True,
        "official_ncps_runtime_is_exactly_pinned": bool(official_cfc_runtime_binding()),
        "official_dense_cfc_parameter_count_is_14064": official_parameter_count == 14064,
        "official_adapter_has_reason_disjoint_fixed_tau_and_density_paths": (
            config.adapter == full_temporal_adapter_spec()
        ),
        "fixed_tau_is_15_025_seconds": config.adapter.fixed_tau_seconds == 15.025,
        "density_gain_is_min_0_25k_0_75": config.adapter.zero_delta_gain == 0.25,
        "same_seed_parent_order_loss_optimizer_and_20_epochs_are_frozen": (
            len(config.source_v23g["parent_order_sha256_by_epoch"]) == FIXED_EPOCHS
        ),
        "train_has_same_second_density_exposure": (
            int(exposure["train"]["total"]["density_exposed_parent_count"]) > 0
        ),
        "v23e_selector_is_frozen": config.selector["schema_version"] == V23E_SELECTOR_SCHEMA,
        "validation_is_excluded_from_readiness_gate": True,
        "unknown_test_oof_l3_cascade_rejection_stable_p95_are_deferred": True,
    }
    contract = {
        "schema_version": V23L_CONTRACT_SCHEMA,
        "experiment": "V23l official CfC full temporal adapter versus completed self-developed V23g density KnownGate",
        "arms": {
            "self_developed_v23g": {
                "architecture": "CfCLiquidCausalEncoder",
                "positive_delta": "learned tau/candidate/gate update",
                "zero_delta": "alpha_k=min(0.25*k,0.75)",
                "source": density,
            },
            "official_v23l": {
                "architecture": "unmodified ncps.torch.CfC dense/default/lecun_tanh/48x1",
                "external_post_state_adapter": config.adapter.as_dict(),
                "positive_delta": "fixed tau=15.025 retention",
                "zero_delta": "alpha_k=min(0.25*k,0.75)",
                "first_event": "official state without external interpolation",
            },
        },
        "shared_protocol": {
            "source_feature_cache": "V21e causal-prefix course circular incoherence; evidence_dim=1",
            "source_data_contract": source_config.data.contract_path,
            "source_normalization": source_config.data.normalization_stats_path,
            "training_split": "V21 known-only train",
            "development_split": "V21 known-only validation",
            "train_parent_count": 986,
            "development_parent_count": 232,
            "development_clutter_parent_support": 9,
            "batch_size": config.train.batch_size,
            "optimizer": "AdamW unchanged from V21e/V23g",
            "learning_rate": config.train.learning_rate,
            "class_weights": list(config.train.class_weights),
            "loss_mode": config.train.loss_mode,
            "prefix_auxiliary_weight": config.train.prefix_auxiliary_weight,
            "prefix_auxiliary_min_events": config.train.prefix_auxiliary_min_events,
            "seed": config.train.seed,
            "epochs": config.train.epochs,
            "checkpoint_selector": config.selector,
            "parent_order": "each V23l epoch exactly equals completed V23g density arm",
        },
        "only_architecture_difference_between_arms": [
            "self-developed recurrent state equation versus official ncps.torch.CfC state equation",
            "official arm represents matching positive-dt fixed retention externally because official CfC has no project learned tau",
        ],
        "explicitly_not_added": [
            "weight decay change", "optimizer change", "learning-rate warmup", "gradient-clip change",
            "OOD objective", "attention", "new loss", "new feature", "data rebuild", "dt cap",
            "recurrent LayerNorm", "per-dimension tau", "custom candidate/gate inside official CfC",
        ],
        "train_same_second_exposure": exposure,
        "interpretation_boundary": {
            "development_same_second_effect": "deferred to post-training evaluation",
            "comparison_is_development_known_gate_only": True,
            "not_final_product_performance": True,
        },
        "unknown_training_exposure": 0,
        "OOF": "未执行",
        "test": "未执行",
        "L3": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / contract_relative()
    _write_new_or_identical(contract_path, contract)
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": V23L_READINESS_SCHEMA,
        "variant": V23L_VARIANT,
        "checks": checks,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": config_identity_sha256(config),
        "contract_path": _relative(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "source_v23g_density": density,
        "adapter": config.adapter.as_dict(),
        "train_same_second_exposure": exposure,
        "selector": config.selector,
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_parent_support": 9,
        "selected_seed": V23G_TRAINING_SEED,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行全新完整20 epoch V23l官方CfC训练）",
        "OOF": "未执行",
        "test": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / readiness_relative()
    _write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "variant": V23L_VARIANT,
        "config": str(config_path),
        "contract": str(contract_path),
        "readiness": str(readiness_path),
        "formal_training": readiness["formal_training"],
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V23l official-CfC full temporal adapter KnownGate experiment.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
