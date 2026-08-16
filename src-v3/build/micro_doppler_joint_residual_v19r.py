from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from build.micro_doppler_residual_v19r import (
    _resolve,
    _sealed,
    _validate_v18r_source,
    _validate_v19_feature_source,
)
from radar_ln3.micro_doppler_joint_residual_v19r import (
    V19R_JOINT_ARTIFACT_RELATIVE_PATH,
    V19R_JOINT_CHECKPOINT_SCHEMA_VERSION,
    V19R_JOINT_CONFIG_NAME,
    V19R_JOINT_CONFIG_RELATIVE_PATH,
    V19R_JOINT_CONTRACT_RELATIVE_PATH,
    V19R_JOINT_PAIRED_PROTOCOL,
    V19R_JOINT_READINESS_RELATIVE_PATH,
    V19R_JOINT_SCHEMA_VERSION,
    V19rJointLateResidualModel,
    sha256_file,
    tensor_state_sha256,
)
from radar_ln3.micro_doppler_residual import V19_TRAINABLE_PARAMETER_COUNT
from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
    build_task_dataloader,
    build_task_dataset,
    build_v18r_stable8_density_models,
    load_task_config,
)


FROZEN_DIAGNOSTIC_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V19r-v18r-micro-doppler-late-residual-validation.json"
)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _write_text_if_same(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V19r joint file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _require_empty_artifact(root: Path) -> None:
    artifact = root / V19R_JOINT_ARTIFACT_RELATIVE_PATH
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(
            f"refusing to prepare V19r joint over a non-empty artifact: {artifact}"
        )


def _build_initial_model(
    root: Path, v18r: Mapping[str, Any], v19: Mapping[str, Any]
) -> tuple[V19rJointLateResidualModel, dict[str, Any], Any]:
    treatment_config = v18r["config"]
    counterpart_path = _resolve(
        root, treatment_config.train.paired_counterpart_config_path or ""
    )
    control_config = load_task_config(counterpart_path)
    _, treatment, initialization = build_v18r_stable8_density_models(
        root, control_config, treatment_config
    )
    model = V19rJointLateResidualModel(
        treatment,
        normalization_median=torch.tensor(
            v19["normalization"]["median"], dtype=torch.float32
        ),
        normalization_scale=torch.tensor(
            v19["normalization"]["scale"], dtype=torch.float32
        ),
    )
    return model, initialization, control_config


def _initial_probe(
    root: Path,
    model: V19rJointLateResidualModel,
    config: Any,
) -> dict[str, Any]:
    dataset = build_task_dataset(
        root,
        config,
        split_name=config.data.train_split_name,
        parent_limit=3,
    )
    batch = next(
        iter(build_task_dataloader(dataset, batch_size=3, shuffle=False))
    )
    tensors = {
        key: value if not isinstance(value, torch.Tensor) else value.cpu()
        for key, value in batch.items()
    }
    model.eval()
    with torch.inference_mode():
        output = model(
            tr_features=tensors["tr_features"],
            tr_missing_mask=tensors["tr_missing_mask"],
            tr_dt_seconds=tensors["tr_dt_seconds"],
            tr_dt_valid=tensors["tr_dt_valid"],
            tr_dt_reason=tensors.get("tr_dt_reason"),
            tr_motion_features=tensors.get("tr_motion_features"),
            tr_motion_valid_mask=tensors.get("tr_motion_valid_mask"),
            rd_physical_features=tensors.get("rd_physical_features"),
            rd_physical_valid=tensors.get("rd_physical_valid"),
            micro_doppler_features=tensors["micro_doppler_features"],
            micro_doppler_valid=tensors["micro_doppler_valid"],
            rd_frames=tensors["rd_frames"],
            rd_frame_mask=tensors["rd_frame_mask"],
            rd_quality=tensors["rd_quality"],
            tr_padding_mask=tensors["tr_padding_mask"],
        )
    if not torch.equal(output["sequence_logits"], output["backbone_sequence_logits"]):
        raise ValueError("V19r joint zero residual does not preserve initial V18r logits")
    if bool((output["residual_logits"] != 0.0).any()):
        raise ValueError("V19r joint initial residual logits are not exactly zero")
    return {
        "parent_sample_ids": list(batch["parent_sample_ids"]),
        "label_names": list(batch["label_names"]),
        "combined_logits_exactly_equal_backbone": True,
        "residual_logits_exactly_zero": True,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    _require_empty_artifact(root)
    v18r = _validate_v18r_source(root)
    v19 = _validate_v19_feature_source(root, v18r)
    model, initialization, control_config = _build_initial_model(root, v18r, v19)
    treatment_config = v18r["config"]
    source_initial_hash = str(
        dict(v18r["summary"].get("paired_initialization") or {}).get(
            "source_v18_initial_treatment_state_sha256", ""
        )
    )
    if (
        source_initial_hash
        != initialization["source_v18_initial_treatment_state_sha256"]
        or tensor_state_sha256(model.backbone.state_dict()) != source_initial_hash
    ):
        raise ValueError("V19r joint initialization does not reproduce V18r from-scratch state")
    if (
        model.residual_trainable_parameter_count != V19_TRAINABLE_PARAMETER_COUNT
        or model.backbone_trainable_parameter_count <= V19_TRAINABLE_PARAMETER_COUNT
        or model.trainable_parameter_count
        != model.backbone_trainable_parameter_count + V19_TRAINABLE_PARAMETER_COUNT
        or any(not parameter.requires_grad for parameter in model.backbone.parameters())
    ):
        raise ValueError("V19r joint does not expose the complete backbone plus 24 residual weights")
    initial_probe = _initial_probe(root, model, treatment_config)
    train_dataset = build_task_dataset(
        root, treatment_config, split_name=treatment_config.data.train_split_name
    )
    validation_dataset = build_task_dataset(
        root, treatment_config, split_name=treatment_config.data.validation_split_name
    )
    if set(train_dataset.parent_ids) & set(validation_dataset.parent_ids):
        raise ValueError("V19r joint train and validation sets share a parent")
    frozen_diagnostic_path = root / FROZEN_DIAGNOSTIC_REPORT
    frozen_diagnostic: dict[str, Any] | None = None
    if frozen_diagnostic_path.is_file():
        frozen_diagnostic = json.loads(frozen_diagnostic_path.read_text(encoding="utf-8"))
        if not _sealed(frozen_diagnostic):
            raise ValueError("V19r frozen diagnostic report is outside sealed scope")

    normalization_path = v19["resolved"]["normalization_path"]
    contract = {
        "schema_version": V19R_JOINT_SCHEMA_VERSION,
        "paired_protocol": V19R_JOINT_PAIRED_PROTOCOL,
        "experiment_scope": "DBB specialist full from-scratch joint training only",
        "single_factor": (
            "V18r stable8+density full model plus a zero-initialized 24-weight "
            "stable8 late residual, with all backbone and residual parameters trained jointly"
        ),
        "control": {
            "name": "completed V18r density+stable8 Treatment",
            "retrained": False,
            "best_epoch": 15,
            "checkpoint": str(v18r["checkpoint_path"]),
            "checkpoint_sha256": sha256_file(v18r["checkpoint_path"]),
            "parent_metrics": dict(
                dict(v18r["summary"]["best_validation_metrics"])["parent_metrics"]
            ),
        },
        "treatment_initialization": {
            "mode": "from_scratch_reconstruction_of_v18r_initial_state",
            "source_initial_backbone_state_sha256": source_initial_hash,
            "residual_zero_initialized": True,
            "initial_combined_logits_exactly_equal_v18r_initial_backbone": True,
            "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
            "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
            "total_trainable_parameter_count": model.trainable_parameter_count,
        },
        "source": {
            "v18r_control_config": str(
                _resolve(root, treatment_config.train.paired_counterpart_config_path or "")
            ),
            "v18r_control_config_sha256": sha256_file(
                _resolve(root, treatment_config.train.paired_counterpart_config_path or "")
            ),
            "v18r_treatment_config": str(v18r["config_path"]),
            "v18r_treatment_config_sha256": sha256_file(v18r["config_path"]),
            "v18r_paired_report": str(v18r["report_path"]),
            "v18r_paired_report_sha256": sha256_file(v18r["report_path"]),
            "v18r_training_summary": str(v18r["summary_path"]),
            "v18r_training_summary_sha256": sha256_file(v18r["summary_path"]),
            "v19_train_only_normalization": str(normalization_path),
            "v19_train_only_normalization_sha256": sha256_file(normalization_path),
            "v19_readiness": str(v19["readiness_path"]),
            "v19_readiness_sha256": sha256_file(v19["readiness_path"]),
            "data_contract_hashes": v18r["contract"]["hashes"],
        },
        "initial_probe": initial_probe,
        "reference_parent_order_sha256_by_epoch": v18r["order_hashes"],
        "frozen_24_weight_diagnostic": (
            None
            if frozen_diagnostic is None
            else {
                "report": str(frozen_diagnostic_path),
                "report_sha256": sha256_file(frozen_diagnostic_path),
                "status": frozen_diagnostic.get("status"),
                "macro_f1_delta": dict(
                    dict(frozen_diagnostic.get("primary_comparison") or {}).get(
                        "v19r_minus_frozen_v18r"
                    )
                    or {}
                ).get("macro_f1"),
                "role": "completed diagnostic only; superseded by full joint training",
            }
        ),
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / V19R_JOINT_CONTRACT_RELATIVE_PATH
    _write_text_if_same(contract_path, _json_text(contract))

    artifact = root / V19R_JOINT_ARTIFACT_RELATIVE_PATH
    config = {
        "schema_version": V19R_JOINT_SCHEMA_VERSION,
        "config_name": V19R_JOINT_CONFIG_NAME,
        "paired_protocol": V19R_JOINT_PAIRED_PROTOCOL,
        "contract_path": str(contract_path),
        "expected_contract_sha256": sha256_file(contract_path),
        "readiness_path": str(root / V19R_JOINT_READINESS_RELATIVE_PATH),
        "v18r_control_config_path": str(
            _resolve(root, treatment_config.train.paired_counterpart_config_path or "")
        ),
        "expected_v18r_control_config_sha256": sha256_file(
            _resolve(root, treatment_config.train.paired_counterpart_config_path or "")
        ),
        "v18r_treatment_config_path": str(v18r["config_path"]),
        "expected_v18r_treatment_config_sha256": sha256_file(v18r["config_path"]),
        "v18r_paired_report_path": str(v18r["report_path"]),
        "expected_v18r_paired_report_sha256": sha256_file(v18r["report_path"]),
        "v18r_training_summary_path": str(v18r["summary_path"]),
        "expected_v18r_training_summary_sha256": sha256_file(v18r["summary_path"]),
        "v18r_best_checkpoint_path": str(v18r["checkpoint_path"]),
        "expected_v18r_best_checkpoint_sha256": sha256_file(v18r["checkpoint_path"]),
        "normalization_path": str(normalization_path),
        "expected_normalization_sha256": sha256_file(normalization_path),
        "source_v19_readiness_path": str(v19["readiness_path"]),
        "expected_source_v19_readiness_sha256": sha256_file(v19["readiness_path"]),
        "model": {
            "backbone": "complete V18r stable8+density LiquidTaskClassifier",
            "backbone_initialization": "reconstructed V18r from-scratch initial state",
            "late_residual": "3x8 bias-free zero-sum bounded stable8 correction",
            "residual_zero_initialized": True,
            "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
            "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
            "total_trainable_parameter_count": model.trainable_parameter_count,
        },
        "train": {
            "seed": 42,
            "dataloader_seed": 42,
            "batch_size": int(treatment_config.train.batch_size),
            "learning_rate": float(treatment_config.train.learning_rate),
            "optimizer": "AdamW_default_weight_decay",
            "epochs": 20,
            "early_stopping_patience": 0,
            "gradient_clip_norm": float(treatment_config.train.gradient_clip_norm),
            "class_weights": list(treatment_config.train.class_weights),
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": float(
                treatment_config.train.prefix_auxiliary_weight
            ),
            "prefix_auxiliary_min_events": int(
                treatment_config.train.prefix_auxiliary_min_events
            ),
            "checkpoint_selection_metric": BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
            "save_every_epoch": True,
            "record_parent_order_hash": True,
            "reference_parent_order_sha256_by_epoch": v18r["order_hashes"],
            "device": str(treatment_config.train.device),
            "checkpoint_schema": V19R_JOINT_CHECKPOINT_SCHEMA_VERSION,
            "artifact_dir": str(artifact),
            "checkpoint_path": str(artifact / "best_model.pt"),
            "last_checkpoint_path": str(artifact / "last_model.pt"),
            "epoch_checkpoint_dir": str(artifact / "epochs"),
        },
        "adoption": {
            "minimum_per_class_precision_recall_f1": 0.95,
            "strict_macro_f1_improvement_over_v18r": 0.02,
            "strict_drone_bird_weak_floor_improvement_over_v18r": 0.02,
            "drone_to_bird_must_not_increase": True,
            "bird_to_drone_must_not_increase": True,
            "minimum_bidirectional_error_reduction": 1,
            "balloon_floor_must_not_decrease": True,
        },
        "scope": {
            "formal_training": "manual_only",
            "full_backbone_training": True,
            "validation_checkpoint_selection": True,
            "validation_threshold_tuning": False,
            "test": "未执行",
            "OOF": "未执行",
            "full_cascade": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "deployment_eligible": False,
        },
    }
    config_path = root / V19R_JOINT_CONFIG_RELATIVE_PATH
    _write_text_if_same(config_path, _json_text(config))

    checks = {
        "v18r_sources_are_hash_bound_and_sealed": True,
        "v19_normalization_is_train_only_and_sealed": True,
        "backbone_reconstructs_v18r_from_scratch_initial_state": True,
        "all_backbone_parameters_are_trainable": all(
            parameter.requires_grad for parameter in model.backbone.parameters()
        ),
        "residual_has_exactly_24_trainable_parameters": (
            model.residual_trainable_parameter_count == V19_TRAINABLE_PARAMETER_COUNT
        ),
        "total_trainable_parameters_include_full_backbone": (
            model.trainable_parameter_count
            == model.backbone_trainable_parameter_count + V19_TRAINABLE_PARAMETER_COUNT
        ),
        "zero_residual_preserves_initial_backbone_logits": True,
        "train_validation_parent_sets_are_disjoint": not bool(
            set(train_dataset.parent_ids) & set(validation_dataset.parent_ids)
        ),
        "test_payload_access_count_is_zero": True,
        "validation_is_excluded_from_readiness_gate": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": (
            "RADAR-LN3-V19r-stable8-density-joint-late-residual-readiness-v1"
        ),
        "paired_protocol": V19R_JOINT_PAIRED_PROTOCOL,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "checks": checks,
        "validation_used_for_readiness_gate": False,
        "train_parent_count": len(train_dataset),
        "validation_parent_count": len(validation_dataset),
        "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
        "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
        "total_trainable_parameter_count": model.trainable_parameter_count,
        "source_initial_backbone_state_sha256": source_initial_hash,
        "test_payload_access_count": 0,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    if readiness["status"] != "passed":
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"V19r joint readiness failed: {failed}")
    readiness_path = root / V19R_JOINT_READINESS_RELATIVE_PATH
    _write_text_if_same(readiness_path, _json_text(readiness))
    return {
        "status": "passed",
        "schema_version": readiness["schema_version"],
        "config": str(config_path),
        "contract": str(contract_path),
        "readiness": str(readiness_path),
        "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
        "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
        "total_trainable_parameter_count": model.trainable_parameter_count,
        "train_parent_count": len(train_dataset),
        "validation_parent_count": len(validation_dataset),
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the corrected V19r full joint late-residual experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
