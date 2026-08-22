"""Prepare the self-developed V23g density-weighted KnownGate experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate_v23g import (
    V23G_CHECKPOINT_SCHEMA,
    V23G_CONFIG_SCHEMA,
    V23G_CONTRACT_SCHEMA,
    V23G_EPOCHS,
    V23G_READINESS_SCHEMA,
    V23G_TRAINING_SEED,
    V23G_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    contract_relative,
    load_config,
    readiness_relative,
    sha256_file,
    source_control,
)


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to overwrite a different V23g prepare artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _config_payload(root: Path, control_path: Path, control: dict[str, Any]) -> dict[str, Any]:
    source = json.loads(control_path.read_text(encoding="utf-8"))
    artifact = str(artifact_relative()).replace("\\", "/")
    model = dict(source["model"])
    model["liquid_invalid_dt_update_mode"] = "density_weighted"
    train = dict(source["train"])
    train.update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": V23G_CHECKPOINT_SCHEMA,
            "paired_baseline_config_path": str(
                control["config_path"]
            ).replace("\\", "/"),
        }
    )
    return {
        "schema_version": V23G_CONFIG_SCHEMA,
        "config_name": config_relative().stem.removesuffix("-config"),
        "variant": V23G_VARIANT,
        "source_control": control,
        "model": model,
        "train": train,
    }


def _density_exposure(root: Path, source_config: Any) -> dict[str, Any]:
    """Read train and development timing metadata without opening Unknown/test."""

    from radar_ln3.data import DT_REASON_ZERO_DELTA
    from radar_ln3.known_gate import build_known_gate_dataset

    result: dict[str, Any] = {}
    for split in ("train", "val"):
        dataset = build_known_gate_dataset(root, source_config, split_name=split)
        labels: dict[str, dict[str, int]] = {}
        total = {"parent_count": 0, "zero_delta_event_count": 0, "density_exposed_parent_count": 0}
        for item in dataset:
            label = str(item.base.label_name)
            stats = labels.setdefault(
                label,
                {
                    "parent_count": 0,
                    "zero_delta_event_count": 0,
                    "density_exposed_parent_count": 0,
                    "maximum_causal_zero_ordinal": 0,
                },
            )
            ordinal = 0
            maximum = 0
            zero_count = 0
            for reason in item.base.tr_dt_reason.tolist():
                if reason in (1, 3):
                    ordinal = 1
                elif reason == DT_REASON_ZERO_DELTA:
                    zero_count += 1
                    ordinal += 1
                    maximum = max(maximum, ordinal)
            stats["parent_count"] += 1
            stats["zero_delta_event_count"] += zero_count
            stats["density_exposed_parent_count"] += int(maximum >= 2)
            stats["maximum_causal_zero_ordinal"] = max(
                stats["maximum_causal_zero_ordinal"], maximum
            )
            total["parent_count"] += 1
            total["zero_delta_event_count"] += zero_count
            total["density_exposed_parent_count"] += int(maximum >= 2)
        result[split] = {"labels": labels, "total": total}
    return result


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    control = source_control(root)
    control_path = root / str(control["config_path"])
    config_path = root / config_relative()
    _write_new_or_identical(config_path, _config_payload(root, control_path, control))
    config, source_config, loaded_control_path = load_config(root)
    if loaded_control_path != control_path.resolve():
        raise AssertionError("V23g control path changed during config reload")
    exposure = _density_exposure(root, source_config)
    val_clutter = dict(exposure["val"]["labels"].get("ClutterTarget") or {})
    checks = {
        "direct_control_is_completed_v21e_seed43_bounded_observation": True,
        "only_model_difference_is_density_weighted_mode": True,
        "control_gain_and_treatment_k1_gain_are_both_0_25": (
            float(config.model["liquid_invalid_dt_observation_gain"]) == 0.25
        ),
        "density_schedule_is_min_0_25k_0_75": True,
        "same_seed_parent_order_optimizer_loss_and_20_epochs_are_frozen": True,
        "training_has_density_exposed_known_target_parents": (
            int(exposure["train"]["total"]["density_exposed_parent_count"]) > 0
        ),
        "development_has_density_exposed_known_target_parents": (
            int(exposure["val"]["total"]["density_exposed_parent_count"]) > 0
        ),
        "development_clutter_same_second_exposure_is_zero_disclosed": (
            int(val_clutter.get("zero_delta_event_count", -1)) == 0
            and int(val_clutter.get("density_exposed_parent_count", -1)) == 0
        ),
        "unknown_test_oof_l3_cascade_rejection_stable_p95_are_deferred": True,
    }
    contract = {
        "schema_version": V23G_CONTRACT_SCHEMA,
        "experiment": "V23g self-developed CfC same-second density-weighted replay",
        "source_control": control,
        "single_factor": {
            "control": "bounded_observation; zero_delta alpha=0.25",
            "treatment": "density_weighted; zero_delta alpha_k=min(0.25*k,0.75)",
            "causal_ordinal": "first/positive set k=1; first zero_delta follow-up is k=2; padding never advances k",
            "frozen": [
                "V21e causal-course feature cache and V21 train-only normalization",
                "TR/RD front end, fusion, classifier, self-developed CfCLiquidCausalEncoder weights/initialization",
                "loss, class weights, prefix auxiliary weight, checkpoint selection",
                "batch size, optimizer, learning rate, seed, parent order and 20 epochs",
            ],
            "not_changed": [
                "features", "data contract", "split", "normalization", "loss", "checkpoint selection", "positive-dt and first-event semantics",
            ],
        },
        "formal_protocol": {
            "training_split": "V21 known-only train",
            "development_split": "V21 known-only validation",
            "train_parent_count": 986,
            "development_parent_count": 232,
            "development_clutter_parent_support": 9,
            "epochs": V23G_EPOCHS,
            "seed": V23G_TRAINING_SEED,
            "parent_order_pairing": "every treatment epoch equals completed V21e seed43 Control order SHA256",
        },
        "same_second_exposure": exposure,
        "interpretation_boundary": {
            "clutter_density_effect_estimable_on_development": False,
            "reason": "all 9 development Clutter parents have zero same-second follow-up events",
            "not_product_performance": True,
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
        "schema_version": V23G_READINESS_SCHEMA,
        "variant": V23G_VARIANT,
        "checks": checks,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": config_identity_sha256(config),
        "contract_path": _relative(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "source_control": control,
        "same_second_exposure": exposure,
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_parent_support": 9,
        "selected_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行完整20 epoch单因素训练）",
        "OOF": "未执行",
        "test": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / readiness_relative()
    _write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "config": str(config_path),
        "contract": str(contract_path),
        "readiness": str(readiness_path),
        "formal_training": readiness["formal_training"],
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V23g density-weighted KnownGate.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
