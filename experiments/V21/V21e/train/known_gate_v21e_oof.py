"""Train one frozen V21e KnownGate paired OOF arm/fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    load_known_gate_config,
    seed_everything,
)
from radar_ln3.known_gate_v21e_oof import (
    ARMS,
    FIXED_EPOCHS,
    PREDICTION_SCHEMA,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    artifact_relative,
    config_relative,
    fold_record,
    inverse_frequency_weights,
    load_fold_contract,
    prediction_relative,
    resolve_path,
    sequence_sha256,
    sha256_file,
    write_new_or_identical,
)
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import (
    _capture_training_state,
    _checkpoint_payload,
    _load_checkpoint,
    _model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _save_checkpoint,
    _state_dict_sha256,
)


def _verify_prepared(
    root: Path,
    *,
    arm: str,
    fold_id: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path, str]:
    contract, assignment = load_fold_contract(root)
    fold = fold_record(contract, fold_id)
    readiness_path = root / READINESS_RELATIVE
    if not readiness_path.is_file():
        raise FileNotFoundError(readiness_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("fold_contract_sha256")
        != sha256_file(root / readiness["fold_contract_path"])
        or int(readiness.get("known_parent_count", -1)) != 986
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V21e OOF readiness is invalid or stale")
    binding = dict(dict(fold.get("configs") or {}).get(arm) or {})
    config_path = root / config_relative(arm, fold_id)
    if (
        not config_path.is_file()
        or binding.get("path") != str(config_relative(arm, fold_id)).replace("\\", "/")
        or binding.get("sha256") != sha256_file(config_path)
    ):
        raise ValueError(f"V21e OOF fold {fold_id} {arm} config binding changed")
    config = load_known_gate_config(config_path)
    if (
        known_gate_config_identity_sha256(config) != binding.get("identity_sha256")
        or config.train.epochs != FIXED_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.seed != 43
        or config.train.dataloader_seed != 43
        or config.train.save_every_epoch is not False
    ):
        raise ValueError(f"V21e OOF fold {fold_id} {arm} config violates protocol")
    stats_path = resolve_path(root, config.data.normalization_stats_path)
    data_contract_path = resolve_path(root, config.data.contract_path)
    if (
        not stats_path.is_file()
        or sha256_file(stats_path) != fold["normalization_stats_sha256"]
        or not data_contract_path.is_file()
        or sha256_file(data_contract_path) != fold["data_contract_sha256"]
    ):
        raise ValueError(f"V21e OOF fold {fold_id} stats/data contract changed")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    fold_data_contract = json.loads(data_contract_path.read_text(encoding="utf-8"))
    oof = dict(fold_data_contract.get("oof_fold_contract") or {})
    train_counts = {
        str(key): int(value)
        for key, value in dict(fold["train_label_parent_counts"]).items()
    }
    target_count = sum(train_counts[label] for label in KNOWN_LABELS[:3])
    expected_weights = inverse_frequency_weights(
        (target_count, train_counts["ClutterTarget"])
    )
    if (
        stats.get("schema_version") != "RADAR-LN3-V21"
        or int(stats.get("parent_count", -1)) != len(fold["train_parent_ids"])
        or int(stats.get("allowed_parent_ids_count", -1))
        != len(fold["train_parent_ids"])
        or stats.get("allowed_parent_ids_sha256") != fold["train_parent_ids_sha256"]
        or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
        or int(oof.get("fold_id", -1)) != fold_id
        or oof.get("grouping") != "capture_content_provenance_component"
        or oof.get("train_parent_ids_sha256") != fold["train_parent_ids_sha256"]
        or oof.get("held_out_parent_ids_sha256")
        != fold["held_out_parent_ids_sha256"]
        or oof.get("held_out_checkpoint_selection") is not False
        or int(oof.get("unknown_training_exposure", -1)) != 0
        or int(oof.get("validation_parent_count", -1)) != 0
        or int(oof.get("test_parent_count", -1)) != 0
        or not np.allclose(config.train.class_weights, expected_weights, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"V21e OOF fold {fold_id} contains leakage or drift")
    return config, fold, assignment, config_path, sha256_file(readiness_path)


def _dataset_scope(dataset: Any, expected_ids: list[str], scope: str) -> dict[str, Any]:
    actual = sorted(str(value) for value in dataset.parent_ids)
    labels = set(dataset.base.frame["label_name"].astype(str))
    if (
        actual != sorted(expected_ids)
        or len(actual) != len(set(actual))
        or labels - set(KNOWN_LABELS)
        or UNKNOWN_LABEL in labels
        or set(dataset.base.frame["split_name"].astype(str)) != {"train"}
    ):
        raise ValueError(f"V21e OOF {scope} Dataset violates its train-only scope")
    return {
        "scope": scope,
        "parent_count": len(actual),
        "parent_ids_sha256": sequence_sha256(actual),
        "base_labels": sorted(labels),
        "unknown_parent_count": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
    }


def train_oof_fold(
    project_root: Path,
    *,
    arm: str,
    fold_id: int,
    resume_checkpoint: Path | None = None,
    verify_prepared: Callable[..., tuple[Any, dict[str, Any], dict[str, Any], Path, str]] = _verify_prepared,
    artifact_resolver: Callable[[str, int], Path] = artifact_relative,
    prediction_resolver: Callable[[str, int], Path] = prediction_relative,
    prediction_schema: str = PREDICTION_SCHEMA,
    experiment_name: str = "V21e OOF",
    loss_class_weight_normalization: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    if arm not in ARMS:
        raise ValueError(f"V21e OOF arm must be one of {ARMS}")
    config, fold, assignment, config_path, readiness_hash = verify_prepared(
        root,
        arm=arm,
        fold_id=fold_id,
    )
    normalization_mode = (
        config.train.class_weight_normalization
        if loss_class_weight_normalization is None
        else str(loss_class_weight_normalization)
    )
    if normalization_mode != config.train.class_weight_normalization:
        raise ValueError(
            "OOF loss class-weight normalization differs from the bound config"
        )
    artifact = root / artifact_resolver(arm, fold_id)
    prediction_path = root / prediction_resolver(arm, fold_id)
    if prediction_path.exists():
        raise FileExistsError(f"V21e OOF prediction already exists: {prediction_path}")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            "refusing to overwrite a non-empty V21e OOF fold; use its last checkpoint: "
            f"{artifact}"
        )
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(
        artifact / "training.log",
        logger_name=f"radar_ln3.{experiment_name.lower().replace(' ', '_')}.{arm}.fold_{fold_id:02d}",
    )
    started = time.perf_counter()
    try:
        if config.train.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V21e OOF fold training requires an available CUDA device")
        seed_everything(config.train.seed)
        train_dataset = build_known_gate_dataset(
            root,
            config,
            split_name="train",
            parent_ids=fold["train_parent_ids"],
            allow_feature_cache_superset=config.model.evidence_dim > 0,
        )
        train_scope = _dataset_scope(
            train_dataset,
            fold["train_parent_ids"],
            f"fold_{fold_id:02d}_{arm}_train",
        )
        expected_binary_counts = {
            KNOWN_GATE_CLASS_NAMES[0]: sum(
                int(fold["train_label_parent_counts"][label])
                for label in KNOWN_LABELS[:3]
            ),
            KNOWN_GATE_CLASS_NAMES[1]: int(
                fold["train_label_parent_counts"]["ClutterTarget"]
            ),
        }
        if train_dataset.label_counts() != expected_binary_counts:
            raise ValueError("V21e OOF fold train binary support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model, baseline_reference, pairing = build_known_gate_model_for_config(root, config)
        initial_state_hash = _state_dict_sha256(model.state_dict())
        paired_baseline_state_hash: str | None = None
        if baseline_reference is not None:
            paired_baseline_state_hash = _state_dict_sha256(
                baseline_reference.state_dict()
            )
            probe_loader = build_known_gate_dataloader(
                train_dataset,
                batch_size=min(config.train.batch_size, len(train_dataset)),
                shuffle=False,
            )
            probe = next(iter(probe_loader))
            model.eval()
            baseline_reference.eval()
            with torch.inference_mode():
                treatment_logits = model(**_model_inputs(probe))["sequence_logits"]
                baseline_inputs = _model_inputs(probe)
                baseline_evidence_dim = baseline_reference.config.model.evidence_dim
                if baseline_evidence_dim:
                    baseline_inputs["known_gate_features"] = baseline_inputs[
                        "known_gate_features"
                    ][..., :baseline_evidence_dim]
                    baseline_inputs["known_gate_feature_valid"] = baseline_inputs[
                        "known_gate_feature_valid"
                    ][..., :baseline_evidence_dim]
                else:
                    baseline_inputs["known_gate_features"] = None
                    baseline_inputs["known_gate_feature_valid"] = None
                baseline_logits = baseline_reference(**baseline_inputs)["sequence_logits"]
            if not torch.equal(treatment_logits, baseline_logits):
                raise AssertionError("V21e OOF Treatment initial logits differ from Control")
            pairing["initial_logits_exactly_equal"] = True
            pairing["initial_logits_sha256"] = hashlib.sha256(
                treatment_logits.detach().cpu().contiguous().numpy().tobytes(order="C")
            ).hexdigest()
            pairing["baseline_reference_state_sha256"] = paired_baseline_state_hash
            del baseline_reference, probe_loader, probe, treatment_logits, baseline_logits
        # The Treatment-only equality probe consumes the global torch RNG through
        # its non-shuffled DataLoader. Reset both arms before optimization so any
        # future stochastic layer still sees a paired RNG stream.
        seed_everything(config.train.seed)
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(
            config.train.class_weights,
            dtype=torch.float32,
            device=device,
        )
        start_epoch = 1
        global_step = 0
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            try:
                resume_path.relative_to(artifact.resolve())
            except ValueError as error:
                raise ValueError("OOF resume checkpoint must stay inside its fold artifact") from error
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_hash,
                map_location=device,
            )
            run_state = _restore_training_state(payload, generator)
            start_epoch = int(run_state["next_epoch"])
            global_step = int(run_state["global_step"])
            history = list(run_state.get("history") or [])
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V21e OOF resume state is not an epoch boundary")
            resumed_from = str(resume_path)
        if not 1 <= start_epoch <= FIXED_EPOCHS + 1:
            raise ValueError("V21e OOF resume epoch is outside the fixed budget")
        logger.info(
            "run=start | arm=%s | fold=%02d | train=%d | held_out=%d | epochs=%d | held_out_selection=false | unknown=0 | val=0 | test=0",
            arm,
            fold_id,
            len(train_dataset),
            len(fold["held_out_parent_ids"]),
            FIXED_EPOCHS,
        )
        last_checkpoint = resolve_path(root, config.train.last_checkpoint_path)
        final_checkpoint = resolve_path(root, config.train.checkpoint_path)
        steps_per_epoch = len(train_loader)
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            model.train()
            epoch_started = time.perf_counter()
            total_loss = 0.0
            final_loss = 0.0
            prefix_loss = 0.0
            gradient_max = 0.0
            gradient_clip_count = 0
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
                    class_weight_normalization=normalization_mode,
                    prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                    prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
                )
                loss = objective["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.train.gradient_clip_norm
                    ).item()
                )
                if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
                    raise ValueError("V21e OOF produced non-finite loss or gradient")
                gradient_clip_count += int(grad_norm > config.train.gradient_clip_norm)
                optimizer.step()
                global_step += 1
                total_loss += float(loss.item())
                final_loss += float(objective["final_state_loss"].item())
                prefix_loss += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | arm=%s | fold=%02d | epoch=%02d/%02d | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        arm,
                        fold_id,
                        epoch,
                        FIXED_EPOCHS,
                        step,
                        steps_per_epoch,
                        float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()),
                        grad_norm,
                    )
            if (
                len(order) != len(train_dataset)
                or len(set(order)) != len(order)
                or sorted(order) != fold["train_parent_ids"]
            ):
                raise ValueError("V21e OOF epoch did not consume each fold-train parent once")
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": gradient_clip_count,
                "train_steps": steps_per_epoch,
                "parent_order_sha256": _parent_order_sha256(order),
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "held_out_forward_count": 0,
                "held_out_checkpoint_selection": False,
                "class_weight_normalization": normalization_mode,
                "validation_access_count": 0,
                "test_payload_access_count": 0,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            training_state = _capture_training_state(
                generator,
                run_state={
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "best_epoch": FIXED_EPOCHS,
                    "best_selection": {
                        "protocol": "fixed_final_epoch_no_held_out_selection",
                        "selected_epoch": FIXED_EPOCHS,
                    },
                    "history": history,
                },
            )
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                config=config,
                config_path=config_path,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
                training_state=training_state,
                readiness_sha256=readiness_hash,
            )
            _save_checkpoint(last_checkpoint, checkpoint)
            if epoch == FIXED_EPOCHS:
                _save_checkpoint(final_checkpoint, checkpoint)
            logger.info(
                "epoch | arm=%s | fold=%02d | epoch=%02d/%02d | loss=%.6f | held_out_forward=0 | selection=none | clips=%d | order_sha256=%s | elapsed=%s",
                arm,
                fold_id,
                epoch,
                FIXED_EPOCHS,
                epoch_record["train_loss"],
                gradient_clip_count,
                epoch_record["parent_order_sha256"],
                format_duration(epoch_record["elapsed_seconds"]),
            )
        if not final_checkpoint.is_file() and last_checkpoint.is_file():
            payload = _load_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_hash,
                map_location=device,
            )
            if int(payload.get("epoch", -1)) != FIXED_EPOCHS:
                raise RuntimeError("V21e OOF fold did not reach its fixed final epoch")
            _save_checkpoint(final_checkpoint, payload)
        _load_checkpoint(
            final_checkpoint,
            model,
            None,
            config=config,
            config_path=config_path,
            readiness_sha256=readiness_hash,
            map_location=device,
        )
        held_out_dataset = build_known_gate_dataset(
            root,
            config,
            split_name="train",
            parent_ids=fold["held_out_parent_ids"],
            allow_feature_cache_superset=config.model.evidence_dim > 0,
        )
        held_out_scope = _dataset_scope(
            held_out_dataset,
            fold["held_out_parent_ids"],
            f"fold_{fold_id:02d}_{arm}_held_out",
        )
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise ValueError("V21e OOF train/held-out parent overlap is non-zero")
        evaluation = evaluate_known_gate(
            model,
            held_out_dataset,
            config=config,
            device=device,
        )
        component_by_parent = {
            str(value["parent_sample_id"]): str(value["provenance_component_id"])
            for value in assignment["assignments"]
        }
        for scope, rows in evaluation["records"].items():
            for row in rows:
                row["fold_id"] = fold_id
                row["arm"] = arm
                row["split_name"] = "train_oof_held_out"
                row["provenance_component_id"] = component_by_parent[
                    str(row["parent_sample_id"])
                ]
        predicted_ids = sorted(
            str(row["parent_sample_id"]) for row in evaluation["records"]["parent"]
        )
        if predicted_ids != fold["held_out_parent_ids"]:
            raise ValueError("V21e OOF held-out predictions do not cover the frozen fold")
        prediction = {
            "status": "passed",
            "schema_version": prediction_schema,
            "experiment_name": experiment_name,
            "arm": arm,
            "fold_id": fold_id,
            "role": "paired train OOF support only; not a deployment checkpoint",
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "capture_content_provenance_component",
            "prediction_unit": "parent with causal-prefix audit records",
            "prediction_protocol": "out_of_fold_fixed_final_epoch",
            "loss_aggregation": {
                "class_weight_normalization": normalization_mode,
                "class_weights": [float(value) for value in config.train.class_weights],
                "parent_unit": True,
                "prefixes_averaged_within_parent_before_class_weight": True,
            },
            "epochs": FIXED_EPOCHS,
            "early_stopping": False,
            "held_out_checkpoint_selection": False,
            "held_out_forward_count_before_final_epoch": 0,
            "held_out_prediction_pass_count": 1,
            "unknown_training_exposure": 0,
            "unknown_oof_prediction_count": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_identity_sha256": known_gate_config_identity_sha256(config),
            "readiness_sha256": readiness_hash,
            "checkpoint_path": str(final_checkpoint),
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "final_model_state_sha256": _state_dict_sha256(model.state_dict()),
            "initial_model_state_sha256": initial_state_hash,
            "paired_baseline_reference_state_sha256": paired_baseline_state_hash,
            "paired_initialization": pairing,
            "train_scope": train_scope,
            "held_out_scope": held_out_scope,
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
            "arm": arm,
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
            "resume": {
                "resumed": resumed_from is not None,
                "checkpoint": resumed_from,
                "epoch_boundary_only": True,
            },
            "unknown_training_exposure": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "threshold_fitting": "未执行",
            "deployment_eligible": False,
        }
        (artifact / "training-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "run=complete | arm=%s | fold=%02d | epoch=%02d | parent_macro_f1=%.6f | prediction=%s",
            arm,
            fold_id,
            FIXED_EPOCHS,
            float(evaluation["parent_metrics"]["macro_f1"]),
            prediction_path,
        )
        return summary
    except Exception as error:
        failure = {
            "status": "failed",
            "arm": arm,
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
    parser = argparse.ArgumentParser(description="Train one V21e KnownGate OOF fold.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(5), required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            train_oof_fold(
                args.project_root,
                arm=args.arm,
                fold_id=args.fold_id,
                resume_checkpoint=args.resume_checkpoint,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()