"""Train one V22b full-parameter late-readout OOF fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    evaluate_known_gate,
    load_known_gate_config,
    seed_everything,
)
from radar_ln3.known_gate_v21e_oof import (
    fold_record,
    load_fold_contract,
    resolve_path,
    sequence_sha256,
    sha256_file,
    write_new_or_identical,
)
from radar_ln3.known_gate_v22b import build_v22b_model
from radar_ln3.known_gate_v22b_oof import (
    FIXED_EPOCHS,
    V22B_OOF_CHECKPOINT_SCHEMA,
    V22B_OOF_CONFIG_SCHEMA,
    V22B_OOF_CONTRACT,
    V22B_OOF_CONTRACT_SCHEMA,
    V22B_OOF_PREDICTION_SCHEMA,
    V22B_OOF_READINESS,
    V22B_OOF_READINESS_SCHEMA,
    V22B_OOF_VARIANT,
    artifact_relative,
    config_relative,
    prediction_relative,
)
from radar_ln3.logging_utils import close_training_logger, configure_training_logger, format_duration
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import (
    _capture_training_state,
    _model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _state_dict_sha256,
)
from train.known_gate_v21e_oof import _verify_prepared as verify_v21e_prepared


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _binding(contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    rows = [dict(row) for row in contract["folds"] if int(row["fold_id"]) == fold_id]
    if len(rows) != 1:
        raise ValueError(f"V22b fold binding unavailable: {fold_id}")
    return rows[0]


def _prepared(root: Path, fold_id: int) -> tuple[Any, Any, dict[str, Any], Path, str]:
    source_contract, assignment = load_fold_contract(root)
    contract_path = root / V22B_OOF_CONTRACT
    readiness_path = root / V22B_OOF_READINESS
    contract = _read(contract_path)
    readiness = _read(readiness_path)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != V22B_OOF_CONTRACT_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("schema_version") != V22B_OOF_READINESS_SCHEMA
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("unknown_training_exposure", -1)) != 0
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V22b OOF readiness/contract is invalid")
    fold = fold_record(source_contract, fold_id)
    binding = _binding(contract, fold_id)
    config_path = root / config_relative(fold_id)
    if (
        not config_path.is_file()
        or binding["config_path"] != str(config_relative(fold_id)).replace("\\", "/")
        or binding["config_sha256"] != sha256_file(config_path)
    ):
        raise ValueError(f"V22b fold config binding changed: {fold_id}")
    payload = _read(config_path)
    source_path = root / binding["source_config_path"]
    source_config = load_known_gate_config(source_path)
    verified_source_config, verified_fold, verified_assignment, verified_path, _ = (
        verify_v21e_prepared(root, arm="treatment", fold_id=fold_id)
    )
    if (
        payload.get("schema_version") != V22B_OOF_CONFIG_SCHEMA
        or payload.get("variant") != V22B_OOF_VARIANT
        or int(dict(payload.get("v22b_oof") or {}).get("fold_id", -1)) != fold_id
        or sha256_file(source_path) != binding["source_config_sha256"]
        or source_config.train.epochs != FIXED_EPOCHS
        or source_config.train.save_every_epoch is not False
        or source_config.train.class_weight_normalization != "batch_weight_sum"
        or verified_path.resolve() != source_path.resolve()
        or verified_source_config != source_config
        or verified_fold["train_parent_ids_sha256"]
        != fold["train_parent_ids_sha256"]
        or verified_fold["held_out_parent_ids_sha256"]
        != fold["held_out_parent_ids_sha256"]
        or verified_assignment.get("known_parent_ids_sha256")
        != assignment.get("known_parent_ids_sha256")
    ):
        raise ValueError(f"V22b fold protocol differs: {fold_id}")
    return source_config, fold, assignment, config_path, sha256_file(readiness_path)


def _checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    config_path: Path,
    readiness_sha: str,
    fold_id: int,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_schema": V22B_OOF_CHECKPOINT_SCHEMA,
        "task_name": "known_gate",
        "variant": V22B_OOF_VARIANT,
        "fold_id": fold_id,
        "config_file_sha256": sha256_file(config_path),
        "readiness_sha256": readiness_sha,
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "training_state": dict(training_state),
        "full_parameter_training": True,
        "added_trainable_parameter_count": 24,
        "held_out_checkpoint_selection": False,
        "unknown_is_model_class": False,
    }


def _load_checkpoint(
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    config_path: Path,
    readiness_sha: str,
    fold_id: int,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("checkpoint_schema") != V22B_OOF_CHECKPOINT_SCHEMA
        or payload.get("variant") != V22B_OOF_VARIANT
        or int(payload.get("fold_id", -1)) != fold_id
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha
        or payload.get("full_parameter_training") is not True
        or int(payload.get("added_trainable_parameter_count", -1)) != 24
        or payload.get("held_out_checkpoint_selection") is not False
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V22b resume checkpoint binding differs")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def train_fold(
    project_root: Path,
    *,
    fold_id: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    source_config, fold, assignment, config_path, readiness_sha = _prepared(root, fold_id)
    artifact = root / artifact_relative(fold_id)
    prediction_path = root / prediction_relative(fold_id)
    if prediction_path.exists():
        raise FileExistsError(f"V22b fold prediction already exists: {prediction_path}")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"V22b fold artifact is non-empty: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(
        artifact / "training.log", logger_name=f"radar_ln3.v22b_oof.fold_{fold_id:02d}"
    )
    started = time.perf_counter()
    try:
        if source_config.train.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V22b OOF fold training requires CUDA")
        seed_everything(source_config.train.seed)
        train_dataset = build_known_gate_dataset(
            root,
            source_config,
            split_name="train",
            parent_ids=fold["train_parent_ids"],
            allow_feature_cache_superset=True,
        )
        labels = set(train_dataset.base.frame["label_name"].astype(str))
        if (
            sorted(str(value) for value in train_dataset.parent_ids)
            != fold["train_parent_ids"]
            or labels - set(KNOWN_LABELS)
            or UNKNOWN_LABEL in labels
        ):
            raise ValueError("V22b fold train Dataset violates known-only scope")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(source_config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=source_config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        # Rebuild V21e fold Treatment from its deterministic paired config; do
        # not load the final Control checkpoint, which would change initialization.
        from radar_ln3.known_gate import build_known_gate_model_for_config

        backbone, baseline_reference, source_pairing = build_known_gate_model_for_config(
            root, source_config
        )
        if baseline_reference is None:
            raise AssertionError("V22b source fold lacks its paired baseline reference")
        model = build_v22b_model(backbone)
        initial_model_state_sha = _state_dict_sha256(model.backbone.state_dict())
        baseline_reference_state_sha = _state_dict_sha256(baseline_reference.state_dict())
        probe = next(
            iter(
                build_known_gate_dataloader(
                    train_dataset,
                    batch_size=min(source_config.train.batch_size, len(train_dataset)),
                    shuffle=False,
                )
            )
        )
        model.eval()
        backbone.eval()
        with torch.inference_mode():
            treatment_output = model(**_model_inputs(probe))
            control_output = backbone(**_model_inputs(probe))
        if not torch.equal(treatment_output["sequence_logits"], control_output["sequence_logits"]):
            raise AssertionError("V22b zero-init logits differ from V21e fold Control")
        initial_logits_sha = hashlib.sha256(
            treatment_output["sequence_logits"].detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest()
        residual = treatment_output["rd_temporal_residual"]
        active = ~probe["tr_padding_mask"]
        multiframe = active & (probe["rd_frame_mask"].sum(dim=2) >= 2)
        residual_l2 = torch.linalg.vector_norm(residual, dim=-1)
        if not bool((multiframe & (residual_l2 > 0.0)).any()):
            raise AssertionError("V22b fold probe contains no usable residual")
        model.zero_grad(set_to_none=True)
        probe_loss = torch.nn.functional.cross_entropy(
            model(**_model_inputs(probe))["window_logits"], probe["labels"]
        )
        probe_loss.backward()
        late_grad = model.late_residual_weight.grad
        if late_grad is None or float(late_grad.abs().max().item()) <= 0.0:
            raise AssertionError("V22b fold late residual receives no gradient")
        paired = {
            "source_pairing": source_pairing,
            "initial_logits_exactly_equal": True,
            "initial_logits_sha256": initial_logits_sha,
            "initial_model_state_sha256": initial_model_state_sha,
            "paired_baseline_reference_state_sha256": baseline_reference_state_sha,
            "late_weight_exactly_zero_before_training": bool(
                torch.equal(
                    model.late_residual_weight,
                    torch.zeros_like(model.late_residual_weight),
                )
            ),
            "late_weight_receives_gradient": True,
            "all_model_parameters_trainable": all(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "added_trainable_parameter_count": 24,
            "liquid_state_receives_residual": False,
        }
        if paired["late_weight_exactly_zero_before_training"] is not True:
            raise AssertionError("V22b fold late residual is not zero initialized")
        del baseline_reference, probe, treatment_output, control_output
        model.zero_grad(set_to_none=True)
        seed_everything(source_config.train.seed)
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=source_config.train.learning_rate)
        class_weights = torch.tensor(source_config.train.class_weights, dtype=torch.float32, device=device)
        start_epoch = 1
        global_step = 0
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        last_checkpoint = artifact / "last_model.pt"
        final_checkpoint = artifact / "final_model.pt"
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            try:
                resume_path.relative_to(artifact.resolve())
            except ValueError as error:
                raise ValueError(
                    "V22b resume checkpoint must stay inside its fold artifact"
                ) from error
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                config_path=config_path,
                readiness_sha=readiness_sha,
                fold_id=fold_id,
                device=device,
            )
            run_state = _restore_training_state(payload, generator)
            start_epoch = int(run_state["next_epoch"])
            global_step = int(run_state["global_step"])
            history = list(run_state["history"])
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V22b resume is not an epoch boundary")
            resumed_from = str(resume_path)
        logger.info(
            "run=start | fold=%02d | train=%d | held_out=%d | epochs=20 | held_out_selection=false | unknown=0 | val=0 | test=0",
            fold_id,
            len(train_dataset),
            len(fold["held_out_parent_ids"]),
        )
        steps_per_epoch = len(train_loader)
        control_prediction_path = root / (
            "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof/"
            f"fold-{fold_id:02d}/treatment/held-out-predictions.json"
        )
        control_prediction = _read(control_prediction_path)
        control_history = list(control_prediction.get("fold_training_history") or [])
        control_order_hashes = {
            int(row["epoch"]): str(row["parent_order_sha256"])
            for row in control_history
        }
        control_initial_logits_sha = dict(
            control_prediction.get("paired_initialization") or {}
        ).get("initial_logits_sha256")
        if (
            len(control_history) != FIXED_EPOCHS
            or set(control_order_hashes) != set(range(1, FIXED_EPOCHS + 1))
            or control_initial_logits_sha != initial_logits_sha
        ):
            raise ValueError("V22b frozen Control initialization/order contract differs")
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            model.train()
            epoch_started = time.perf_counter()
            total = final = prefix = grad_max = 0.0
            clip_count = 0
            order: list[str] = []
            for step, batch in enumerate(train_loader, start=1):
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                output = model(**_model_inputs(tensors))
                objective = parent_final_task_ce_with_prefix_auxiliary(
                    output["sequence_logits"],
                    tensors["labels"],
                    tensors["tr_padding_mask"],
                    class_weights=class_weights,
                    class_weight_normalization=source_config.train.class_weight_normalization,
                    prefix_auxiliary_weight=source_config.train.prefix_auxiliary_weight,
                    prefix_auxiliary_min_events=source_config.train.prefix_auxiliary_min_events,
                )
                loss = objective["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), source_config.train.gradient_clip_norm).item())
                if not torch.isfinite(loss).item() or not np.isfinite(grad):
                    raise ValueError("V22b fold produced non-finite loss/gradient")
                clip_count += int(grad > source_config.train.gradient_clip_norm)
                optimizer.step()
                global_step += 1
                total += float(loss.item())
                final += float(objective["final_state_loss"].item())
                prefix += float(objective["prefix_auxiliary_loss"].item())
                grad_max = max(grad_max, grad)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | fold=%02d | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        fold_id, epoch, step, steps_per_epoch, float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()), grad,
                    )
            if len(order) != len(train_dataset) or len(set(order)) != len(order) or sorted(order) != fold["train_parent_ids"]:
                raise ValueError("V22b epoch did not consume each train parent once")
            order_sha = _parent_order_sha256(order)
            if order_sha != control_order_hashes[epoch]:
                raise ValueError(
                    f"V22b fold {fold_id} epoch {epoch} parent order differs from V21e Control"
                )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total / steps_per_epoch,
                "train_final_state_loss": final / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix / steps_per_epoch,
                "gradient_norm_max": grad_max,
                "gradient_clip_count": clip_count,
                "train_steps": steps_per_epoch,
                "parent_order_sha256": order_sha,
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "held_out_forward_count": 0,
                "held_out_checkpoint_selection": False,
                "validation_access_count": 0,
                "test_payload_access_count": 0,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            training_state = _capture_training_state(
                generator,
                run_state={
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "best_epoch": FIXED_EPOCHS,
                    "best_selection": {"protocol": "fixed_final_epoch_no_held_out_selection", "selected_epoch": FIXED_EPOCHS},
                    "history": history,
                },
            )
            checkpoint = _checkpoint(
                model,
                optimizer,
                config_path=config_path,
                readiness_sha=readiness_sha,
                fold_id=fold_id,
                epoch=epoch,
                global_step=global_step,
                history=history,
                training_state=training_state,
            )
            _save_checkpoint(last_checkpoint, checkpoint)
            if epoch == FIXED_EPOCHS:
                _save_checkpoint(final_checkpoint, checkpoint)
            logger.info(
                "epoch | fold=%02d | epoch=%02d/20 | loss=%.6f | order_sha256=%s | elapsed=%s",
                fold_id, epoch, row["train_loss"], row["parent_order_sha256"], format_duration(row["elapsed_seconds"]),
            )
        held_out = build_known_gate_dataset(
            root,
            source_config,
            split_name="train",
            parent_ids=fold["held_out_parent_ids"],
            allow_feature_cache_superset=True,
        )
        if sorted(str(value) for value in held_out.parent_ids) != fold["held_out_parent_ids"]:
            raise ValueError("V22b held-out Dataset differs from fold")
        evaluation = evaluate_known_gate(model, held_out, config=source_config, device=device)
        evaluation["variant"] = V22B_OOF_VARIANT
        final_weight = model.late_residual_weight.detach()
        if float(final_weight.abs().max().item()) <= 0.0:
            raise AssertionError("V22b final late residual remained zero")
        prediction = {
            "status": "passed",
            "schema_version": V22B_OOF_PREDICTION_SCHEMA,
            "experiment_name": "V22b RD temporal late-readout OOF",
            "arm": "treatment",
            "fold_id": fold_id,
            "role": "paired train OOF support only; not a deployment checkpoint",
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "capture_content_provenance_component",
            "prediction_protocol": "out_of_fold_fixed_final_epoch",
            "epochs": FIXED_EPOCHS,
            "held_out_checkpoint_selection": False,
            "held_out_forward_count_before_final_epoch": 0,
            "held_out_prediction_pass_count": 1,
            "unknown_training_exposure": 0,
            "unknown_oof_prediction_count": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "readiness_sha256": readiness_sha,
            "checkpoint_path": str(final_checkpoint),
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "initial_model_state_sha256": initial_model_state_sha,
            "paired_baseline_reference_state_sha256": baseline_reference_state_sha,
            "paired_initialization": paired,
            "late_residual_audit": {
                "added_trainable_parameter_count": 24,
                "final_weight_abs_max": float(final_weight.abs().max().item()),
                "final_weight_l2": float(torch.linalg.vector_norm(final_weight).item()),
                "final_nonzero_weight_count": int(torch.count_nonzero(final_weight).item()),
                "liquid_state_receives_residual": False,
            },
            "train_scope": {
                "parent_count": len(fold["train_parent_ids"]),
                "parent_ids_sha256": sequence_sha256(fold["train_parent_ids"]),
            },
            "held_out_scope": {
                "parent_count": len(fold["held_out_parent_ids"]),
                "parent_ids_sha256": sequence_sha256(fold["held_out_parent_ids"]),
            },
            "fold_training_history": history,
            "evaluation": {key: value for key, value in evaluation.items() if key != "records"},
            "records": evaluation["records"],
            "formal_training": "未执行（OOF折模型不是正式部署模型）",
            "threshold_fitting": "未执行",
            "test": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "deployment_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_new_or_identical(prediction_path, prediction)
        summary = {
            "status": "completed",
            "variant": V22B_OOF_VARIANT,
            "fold_id": fold_id,
            "epoch": FIXED_EPOCHS,
            "global_step": global_step,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "checkpoint_path": str(final_checkpoint),
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "train_parent_count": len(fold["train_parent_ids"]),
            "held_out_parent_count": len(fold["held_out_parent_ids"]),
            "held_out_parent_metrics": evaluation["parent_metrics"],
            "held_out_eligible_prefix_metrics": evaluation["eligible_prefix_metrics"],
            "resume": {"resumed": resumed_from is not None, "checkpoint": resumed_from, "epoch_boundary_only": True},
            "unknown_training_exposure": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "deployment_eligible": False,
        }
        (artifact / "training-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    except Exception as error:
        failure = {
            "status": "failed",
            "variant": V22B_OOF_VARIANT,
            "fold_id": fold_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "test": "未执行",
            "deployment_eligible": False,
        }
        (artifact / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V22b OOF fold.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(5), required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(train_fold(args.project_root.resolve(), fold_id=args.fold_id, resume_checkpoint=args.resume_checkpoint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
