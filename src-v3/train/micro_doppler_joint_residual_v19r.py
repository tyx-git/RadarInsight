from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from build.micro_doppler_joint_residual_v19r import _build_initial_model
from build.micro_doppler_residual_v19r import (
    _validate_v18r_source,
    _validate_v19_feature_source,
)
from radar_ln3.micro_doppler_joint_residual_v19r import (
    V19R_JOINT_CHECKPOINT_SCHEMA_VERSION,
    V19R_JOINT_CONFIG_NAME,
    V19R_JOINT_CONFIG_RELATIVE_PATH,
    V19R_JOINT_PAIRED_PROTOCOL,
    V19R_JOINT_SCHEMA_VERSION,
    V19rJointLateResidualModel,
    load_v19r_joint_checkpoint,
    require_v19r_joint_parent_order,
    save_v19r_joint_checkpoint,
    sha256_file,
    tensor_state_sha256,
)
from radar_ln3.micro_doppler_residual import V19_TRAINABLE_PARAMETER_COUNT
from radar_ln3.task_models import (
    build_task_dataloader,
    build_task_dataset,
    evaluate_task,
    task_training_objective,
)
from train.micro_doppler_residual_v19r import (
    _atomic_json,
    _capture_training_state,
    _device,
    _performance_guardrails,
    _resolve,
    _restore_training_state,
)
from train.tasks import build_checkpoint_selection, checkpoint_selection_is_better


def _strip_records(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "records"}


def _load_runtime(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != V19R_JOINT_SCHEMA_VERSION
        or config.get("config_name") != V19R_JOINT_CONFIG_NAME
        or config.get("paired_protocol") != V19R_JOINT_PAIRED_PROTOCOL
        or dict(config.get("scope") or {}).get("full_backbone_training") is not True
    ):
        raise ValueError("V19r joint training received a different experiment contract")
    train = dict(config.get("train") or {})
    if (
        int(train.get("seed", -1)) != 42
        or int(train.get("dataloader_seed", -1)) != 42
        or int(train.get("epochs", -1)) != 20
        or int(train.get("early_stopping_patience", -1)) != 0
        or train.get("loss_mode") != "parent_final_task_ce_with_prefix_auxiliary"
        or float(train.get("prefix_auxiliary_weight", -1.0)) != 0.1
        or int(train.get("prefix_auxiliary_min_events", -1)) != 3
        or train.get("checkpoint_schema") != V19R_JOINT_CHECKPOINT_SCHEMA_VERSION
        or train.get("save_every_epoch") is not True
        or train.get("record_parent_order_hash") is not True
    ):
        raise ValueError("V19r joint training protocol differs from the frozen design")
    model_contract = dict(config.get("model") or {})
    if (
        model_contract.get("backbone_initialization")
        != "reconstructed V18r from-scratch initial state"
        or model_contract.get("residual_zero_initialized") is not True
        or int(model_contract.get("residual_trainable_parameter_count", -1))
        != V19_TRAINABLE_PARAMETER_COUNT
        or int(model_contract.get("backbone_trainable_parameter_count", -1))
        <= V19_TRAINABLE_PARAMETER_COUNT
        or int(model_contract.get("total_trainable_parameter_count", -1))
        != int(model_contract.get("backbone_trainable_parameter_count", -1))
        + V19_TRAINABLE_PARAMETER_COUNT
    ):
        raise ValueError("V19r joint config does not authorize complete backbone training")
    config_hash = sha256_file(config_path)
    resolved: dict[str, Path] = {}
    for path_field, hash_field in (
        ("contract_path", "expected_contract_sha256"),
        ("v18r_control_config_path", "expected_v18r_control_config_sha256"),
        ("v18r_treatment_config_path", "expected_v18r_treatment_config_sha256"),
        ("v18r_paired_report_path", "expected_v18r_paired_report_sha256"),
        ("v18r_training_summary_path", "expected_v18r_training_summary_sha256"),
        ("v18r_best_checkpoint_path", "expected_v18r_best_checkpoint_sha256"),
        ("normalization_path", "expected_normalization_sha256"),
        ("source_v19_readiness_path", "expected_source_v19_readiness_sha256"),
    ):
        path = _resolve(root, config[path_field])
        if not path.is_file() or sha256_file(path) != config[hash_field]:
            raise ValueError(f"V19r joint bound source differs: {path_field}={path}")
        resolved[path_field] = path
    readiness_path = _resolve(root, config["readiness_path"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("paired_protocol") != V19R_JOINT_PAIRED_PROTOCOL
        or readiness.get("config_sha256") != config_hash
        or readiness.get("contract_sha256") != config["expected_contract_sha256"]
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or readiness.get("deployment_eligible") is not False
        or int(readiness.get("residual_trainable_parameter_count", -1)) != 24
        or int(readiness.get("backbone_trainable_parameter_count", -1))
        != int(model_contract["backbone_trainable_parameter_count"])
        or int(readiness.get("total_trainable_parameter_count", -1))
        != int(model_contract["total_trainable_parameter_count"])
        or not all(dict(readiness.get("checks") or {}).values())
    ):
        raise ValueError("V19r joint readiness binding is invalid")
    v18r = _validate_v18r_source(root)
    v19 = _validate_v19_feature_source(root, v18r)
    if (
        v18r["config_path"].resolve()
        != resolved["v18r_treatment_config_path"].resolve()
        or v18r["report_path"].resolve()
        != resolved["v18r_paired_report_path"].resolve()
        or v18r["summary_path"].resolve()
        != resolved["v18r_training_summary_path"].resolve()
        or v18r["checkpoint_path"].resolve()
        != resolved["v18r_best_checkpoint_path"].resolve()
        or v19["resolved"]["normalization_path"].resolve()
        != resolved["normalization_path"].resolve()
        or v19["readiness_path"].resolve()
        != resolved["source_v19_readiness_path"].resolve()
    ):
        raise ValueError("V19r joint generated config paths differ from validated sources")
    source_orders = dict(v18r["order_hashes"])
    if source_orders != dict(train.get("reference_parent_order_sha256_by_epoch") or {}):
        raise ValueError("V19r joint parent-order references differ from V18r")
    return {
        "config": config,
        "config_path": config_path,
        "config_sha256": config_hash,
        "readiness": readiness,
        "readiness_path": readiness_path,
        "readiness_sha256": sha256_file(readiness_path),
        "resolved": resolved,
        "v18r": v18r,
        "v19": v19,
        "source_orders": source_orders,
    }


def _provenance(runtime: Mapping[str, Any]) -> dict[str, Any]:
    config = runtime["config"]
    return {
        "contract_sha256": config["expected_contract_sha256"],
        "v18r_control_config_sha256": config[
            "expected_v18r_control_config_sha256"
        ],
        "v18r_treatment_config_sha256": config[
            "expected_v18r_treatment_config_sha256"
        ],
        "v18r_paired_report_sha256": config["expected_v18r_paired_report_sha256"],
        "v18r_training_summary_sha256": config[
            "expected_v18r_training_summary_sha256"
        ],
        "v18r_best_checkpoint_sha256": config[
            "expected_v18r_best_checkpoint_sha256"
        ],
        "normalization_sha256": config["expected_normalization_sha256"],
        "source_v19_readiness_sha256": config[
            "expected_source_v19_readiness_sha256"
        ],
        "readiness_sha256": runtime["readiness_sha256"],
    }


def _forward_model(
    model: V19rJointLateResidualModel, tensors: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    return model(
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


def _write_sample_details(
    path: Path,
    evaluation: Mapping[str, Any],
    *,
    checkpoint: Path,
    epoch: int,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": f"{V19R_JOINT_SCHEMA_VERSION}-sample-detail-v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "epoch": int(epoch),
            "counts": {
                key: len(value)
                for key, value in dict(evaluation.get("records") or {}).items()
            },
            "records": dict(evaluation.get("records") or {}),
            "test": "未执行",
        },
    )


def train(
    project_root: Path,
    config_path: Path,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    runtime = _load_runtime(root, config_path.resolve())
    config = runtime["config"]
    train_config = dict(config["train"])
    treatment_config = runtime["v18r"]["config"]
    artifact = _resolve(root, train_config["artifact_dir"])
    summary_path = artifact / "training-summary.json"
    best_path = _resolve(root, train_config["checkpoint_path"])
    last_path = _resolve(root, train_config["last_checkpoint_path"])
    epoch_dir = _resolve(root, train_config["epoch_checkpoint_dir"])
    if resume_checkpoint is None and artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(
            f"refusing to start V19r joint over a non-empty artifact: {artifact}"
        )
    artifact.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = build_task_dataset(
        root, treatment_config, split_name=treatment_config.data.train_split_name
    )
    validation_dataset = build_task_dataset(
        root, treatment_config, split_name=treatment_config.data.validation_split_name
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(train_config["dataloader_seed"]))
    train_loader = build_task_dataloader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    device = _device(train_config["device"])
    model, initialization, _ = _build_initial_model(
        root, runtime["v18r"], runtime["v19"]
    )
    model.to(device)
    expected_backbone_count = int(
        dict(config["model"])["backbone_trainable_parameter_count"]
    )
    if (
        model.backbone_trainable_parameter_count != expected_backbone_count
        or model.residual_trainable_parameter_count != 24
        or model.trainable_parameter_count
        != int(dict(config["model"])["total_trainable_parameter_count"])
        or any(not parameter.requires_grad for parameter in model.backbone.parameters())
    ):
        raise ValueError("V19r joint runtime does not train the complete model")
    initial_backbone_hash = tensor_state_sha256(model.backbone.state_dict())
    if initial_backbone_hash != runtime["readiness"][
        "source_initial_backbone_state_sha256"
    ]:
        raise ValueError("V19r joint runtime initial backbone hash differs")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_config["learning_rate"])
    )
    class_weights = torch.tensor(
        train_config["class_weights"], dtype=torch.float32, device=device
    )
    provenance = _provenance(runtime)
    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_selection: dict[str, Any] | None = None
    global_step = 0
    start_epoch = 1
    initial_logits_equal = False
    if resume_checkpoint is None:
        probe_batch = next(
            iter(
                build_task_dataloader(
                    train_dataset,
                    batch_size=min(3, len(train_dataset)),
                    shuffle=False,
                )
            )
        )
        probe_tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in probe_batch.items()
        }
        model.eval()
        with torch.inference_mode():
            probe_output = _forward_model(model, probe_tensors)
        initial_logits_equal = torch.equal(
            probe_output["sequence_logits"],
            probe_output["backbone_sequence_logits"],
        ) and not bool((probe_output["residual_logits"] != 0.0).any())
        if not initial_logits_equal:
            raise ValueError("V19r joint zero residual does not preserve initial logits")
    else:
        payload = load_v19r_joint_checkpoint(
            _resolve(root, resume_checkpoint),
            model,
            optimizer=optimizer,
            expected_config_name=V19R_JOINT_CONFIG_NAME,
            expected_config_sha256=runtime["config_sha256"],
            expected_provenance=provenance,
            map_location=device,
        )
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        history, best_epoch, best_selection = _restore_training_state(
            payload["training_state"], generator
        )
        if [int(row["epoch"]) for row in history] != list(range(1, start_epoch)):
            raise ValueError("V19r joint resume history is not contiguous")
        initial_logits_equal = True

    started_at = time.perf_counter()
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, int(train_config["epochs"]) + 1):
            current_epoch = epoch
            model.train()
            epoch_started = time.perf_counter()
            loss_total = 0.0
            main_total = 0.0
            prefix_total = 0.0
            steps = 0
            parent_order: list[str] = []
            backbone_gradient_abs_max = 0.0
            residual_gradient_abs_max = 0.0
            for batch in train_loader:
                parent_order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                output = _forward_model(model, tensors)
                objective = task_training_objective(
                    output, tensors, treatment_config, class_weights=class_weights
                )
                objective["loss"].backward()
                residual_grad = model.residual.linear.weight.grad
                if residual_grad is None or not torch.isfinite(residual_grad).all():
                    raise ValueError("V19r joint residual gradient is missing or non-finite")
                residual_gradient_abs_max = max(
                    residual_gradient_abs_max,
                    float(residual_grad.abs().max().item()),
                )
                backbone_gradients = [
                    parameter.grad
                    for parameter in model.backbone.parameters()
                    if parameter.grad is not None
                ]
                if not backbone_gradients or any(
                    not torch.isfinite(value).all() for value in backbone_gradients
                ):
                    raise ValueError("V19r joint backbone gradient is missing or non-finite")
                backbone_gradient_abs_max = max(
                    backbone_gradient_abs_max,
                    max(float(value.abs().max().item()) for value in backbone_gradients),
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["gradient_clip_norm"])
                )
                optimizer.step()
                global_step += 1
                steps += 1
                loss_total += float(objective["loss"].detach().item())
                main_total += float(objective["main_loss"].detach().item())
                prefix_total += float(
                    objective["prefix_auxiliary_loss"].detach().item()
                )
            order_hash = require_v19r_joint_parent_order(
                parent_order,
                runtime["source_orders"][str(epoch)],
                epoch=epoch,
            )
            validation = evaluate_task(
                model,
                validation_dataset,
                config=treatment_config,
                device=device,
            )
            selection = build_checkpoint_selection(
                validation, train_config["checkpoint_selection_metric"]
            )
            selected_as_best = checkpoint_selection_is_better(
                selection, best_selection
            )
            if selected_as_best:
                best_selection = selection
                best_epoch = epoch
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": loss_total / max(1, steps),
                "train_main_loss": main_total / max(1, steps),
                "train_prefix_auxiliary_loss": prefix_total / max(1, steps),
                "backbone_gradient_absolute_max": backbone_gradient_abs_max,
                "residual_gradient_absolute_max": residual_gradient_abs_max,
                "residual_weight_absolute_max": float(
                    model.residual.linear.weight.detach().abs().max().item()
                ),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "selection_metric": train_config["checkpoint_selection_metric"],
                "checkpoint_selection": selection,
                "selected_as_best": selected_as_best,
                "validation": _strip_records(validation),
                "parent_order_sha256": order_hash,
                "parent_order_count": len(parent_order),
                "parent_order_unique_count": len(set(parent_order)),
            }
            history.append(epoch_record)
            training_state = _capture_training_state(
                generator,
                history=history,
                best_epoch=best_epoch,
                best_selection=best_selection,
            )
            targets = [last_path, epoch_dir / f"epoch-{epoch:03d}.pt"]
            if selected_as_best:
                targets.append(best_path)
            for destination in targets:
                save_v19r_joint_checkpoint(
                    destination,
                    model,
                    optimizer,
                    config_name=V19R_JOINT_CONFIG_NAME,
                    config_sha256=runtime["config_sha256"],
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_record,
                    provenance=provenance,
                    training_state=training_state,
                )
            _atomic_json(
                summary_path,
                {
                    "status": "running",
                    "schema_version": V19R_JOINT_SCHEMA_VERSION,
                    "config_name": V19R_JOINT_CONFIG_NAME,
                    "current_epoch": epoch,
                    "best_epoch": best_epoch,
                    "global_step": global_step,
                    "history": history,
                    "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
                    "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
                    "total_trainable_parameter_count": model.trainable_parameter_count,
                    "test_evaluation": "not_executed",
                    "full_cascade_evaluation": "not_executed",
                    "oof_rejection_calibration": "not_executed",
                    "stable_evaluation": "not_executed",
                    "p95_latency": "not_executed",
                    "deployment_eligible": False,
                },
            )
        if best_epoch <= 0 or not best_path.is_file():
            raise ValueError("V19r joint training did not select a best checkpoint")
        best_payload = load_v19r_joint_checkpoint(
            best_path,
            model,
            expected_config_name=V19R_JOINT_CONFIG_NAME,
            expected_config_sha256=runtime["config_sha256"],
            expected_provenance=provenance,
            map_location=device,
        )
        if int(best_payload["epoch"]) != best_epoch:
            raise ValueError("V19r joint best checkpoint epoch differs")
        best_validation = evaluate_task(
            model,
            validation_dataset,
            config=treatment_config,
            device=device,
        )
        if _strip_records(best_validation) != history[best_epoch - 1]["validation"]:
            raise ValueError("V19r joint independent best evaluation differs")
        guardrails = _performance_guardrails(
            best_validation,
            float(config["adoption"]["minimum_per_class_precision_recall_f1"]),
        )
        sample_detail_path = artifact / "sample_detail_information.json"
        _write_sample_details(
            sample_detail_path,
            best_validation,
            checkpoint=best_path,
            epoch=best_epoch,
        )
        summary = {
            "status": "completed",
            "performance_status": guardrails["performance_status"],
            "schema_version": V19R_JOINT_SCHEMA_VERSION,
            "config_name": V19R_JOINT_CONFIG_NAME,
            "paired_protocol": V19R_JOINT_PAIRED_PROTOCOL,
            "task_name": "dbb_specialist",
            "single_factor": (
                "full from-scratch V18r stable8+density backbone jointly trained with "
                "a zero-initialized 24-weight stable8 late residual"
            ),
            "stop_reason": "epoch_limit",
            "epochs": int(train_config["epochs"]),
            "best_epoch": best_epoch,
            "global_step": global_step,
            "history": history,
            "best_validation_metrics": _strip_records(best_validation),
            "performance_guardrails": guardrails,
            "initial_combined_logits_exactly_equal_v18r_initial_backbone": initial_logits_equal,
            "source_initial_backbone_state_sha256": initial_backbone_hash,
            "best_backbone_state_sha256": tensor_state_sha256(
                model.backbone.state_dict()
            ),
            "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
            "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
            "total_trainable_parameter_count": model.trainable_parameter_count,
            "residual_weight_absolute_max": float(
                model.residual.linear.weight.detach().abs().max().item()
            ),
            "config_path": str(config_path),
            "config_sha256": runtime["config_sha256"],
            "readiness_path": str(runtime["readiness_path"]),
            "readiness_sha256": runtime["readiness_sha256"],
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint": str(last_path),
            "last_checkpoint_sha256": sha256_file(last_path),
            "sample_detail": str(sample_detail_path),
            "sample_detail_sha256": sha256_file(sample_detail_path),
            "elapsed_seconds": time.perf_counter() - started_at,
            "validation_threshold_tuning": "not_executed",
            "test_evaluation": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "oof_rejection_calibration": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "deployment_eligible": False,
        }
        _atomic_json(summary_path, summary)
        return summary
    except Exception as exc:
        _atomic_json(
            summary_path,
            {
                "status": "failed",
                "performance_status": "not_evaluated",
                "schema_version": V19R_JOINT_SCHEMA_VERSION,
                "config_name": V19R_JOINT_CONFIG_NAME,
                "current_epoch": current_epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "history": history,
                "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
                "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
                "total_trainable_parameter_count": model.trainable_parameter_count,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resume_checkpoint": str(last_path) if last_path.is_file() else None,
                "test_evaluation": "not_executed",
                "full_cascade_evaluation": "not_executed",
                "oof_rejection_calibration": "not_executed",
                "stable_evaluation": "not_executed",
                "p95_latency": "not_executed",
                "deployment_eligible": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train corrected V19r with the complete V18r backbone and residual."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--config", type=Path, default=V19R_JOINT_CONFIG_RELATIVE_PATH)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    resume = args.resume_checkpoint
    if resume is not None and not resume.is_absolute():
        resume = root / resume
    result = train(root, config_path, resume_checkpoint=resume)
    print(
        json.dumps(
            {
                "status": result["status"],
                "performance_status": result["performance_status"],
                "best_epoch": result["best_epoch"],
                "backbone_trainable_parameter_count": result[
                    "backbone_trainable_parameter_count"
                ],
                "residual_trainable_parameter_count": result[
                    "residual_trainable_parameter_count"
                ],
                "total_trainable_parameter_count": result[
                    "total_trainable_parameter_count"
                ],
                "best_checkpoint": result["best_checkpoint"],
                "test_evaluation": result["test_evaluation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
