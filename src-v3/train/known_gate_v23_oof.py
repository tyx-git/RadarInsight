"""Train one V23 official-``ncps`` CfC KnownGate OOF treatment fold."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    evaluate_known_gate,
    seed_everything,
)
from radar_ln3.known_gate_official_cfc import OfficialCfCKnownGateClassifier
from radar_ln3.known_gate_v21e_oof import (
    fold_record,
    load_fold_contract,
    resolve_path,
    sequence_sha256,
    sha256_file,
)
from radar_ln3.known_gate_v23 import (
    config_identity_sha256,
    load_v23_config,
)
from radar_ln3.known_gate_v23_oof import (
    ARMS,
    FIXED_EPOCHS,
    FOLD_COUNT,
    artifact_relative,
    config_relative,
    contract_relative,
    contract_schema,
    prediction_relative,
    prediction_schema,
    readiness_relative,
    readiness_schema,
)
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import (
    _capture_training_state,
    _model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _state_dict_sha256,
)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json_new_or_identical(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V23 artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _fold_binding(contract: dict[str, Any], fold_id: int) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in list(contract.get("folds") or [])
        if int(row.get("fold_id", -1)) == int(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"V23 fold binding is unavailable or repeated: {fold_id}")
    return matches[0]


def _verify_prepared(
    root: Path,
    *,
    arm: str,
    fold_id: int,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Path, str, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"V23 arm must be one of {ARMS}")
    source_contract, assignment = load_fold_contract(root)
    fold = fold_record(source_contract, fold_id)
    readiness_path = root / readiness_relative(arm)
    contract_path = root / contract_relative(arm)
    if not readiness_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("V23 prepare artifacts are missing")
    readiness = _read_json(readiness_path)
    contract = _read_json(contract_path)
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != readiness_schema(arm)
        or readiness.get("arm") != arm
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != contract_schema(arm)
        or contract.get("arm") != arm
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or int(contract.get("known_parent_count", -1)) != 986
        or int(contract.get("provenance_component_count", -1)) != 975
        or contract.get("grouping") != "capture_content_provenance_component"
        or contract.get("prediction_protocol") != "out_of_fold_fixed_final_epoch"
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("validation_parent_count", -1)) != 0
        or int(contract.get("test_parent_count", -1)) != 0
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False
        or not all(value is True for value in dict(readiness.get("checks") or {}).values())
    ):
        raise ValueError("V23 readiness/contract is invalid or stale")
    if (
        contract.get("source_fold_contract_sha256") != sha256_file(root / contract["source_fold_contract_path"])
        or contract.get("source_assignment_sha256") != sha256_file(root / contract["source_assignment_path"])
    ):
        raise ValueError("V23 frozen fold/assignment binding changed")
    binding = _fold_binding(contract, fold_id)
    config_path = root / config_relative(arm, fold_id)
    config, source_config, source_config_path = load_v23_config(root, config_path)
    treatment_binding = dict(binding.get("treatment") or {})
    source_binding = dict(binding.get("source") or {})
    source_prediction_path = root / config.source_prediction_path
    if (
        treatment_binding.get("config_path")
        != str(config_relative(arm, fold_id)).replace("\\", "/")
        or treatment_binding.get("config_sha256") != sha256_file(config_path)
        or source_binding.get("config_path") != config.source_config_path.replace("\\", "/")
        or source_binding.get("config_sha256") != config.source_config_sha256
        or source_binding.get("config_identity_sha256") != config.source_config_identity_sha256
        or source_binding.get("prediction_path") != config.source_prediction_path.replace("\\", "/")
        or source_binding.get("prediction_sha256") != config.source_prediction_sha256
        or source_binding.get("resolved_class_weight_normalization")
        != config.train.class_weight_normalization
        or binding.get("train_parent_ids_sha256") != fold["train_parent_ids_sha256"]
        or binding.get("held_out_parent_ids_sha256") != fold["held_out_parent_ids_sha256"]
        or binding.get("normalization_stats_sha256") != fold["normalization_stats_sha256"]
        or binding.get("data_contract_sha256") != fold["data_contract_sha256"]
        or sha256_file(source_prediction_path) != config.source_prediction_sha256
    ):
        raise ValueError(f"V23 {arm} fold {fold_id} source/treatment binding changed")
    stats_path = resolve_path(root, source_config.data.normalization_stats_path)
    data_contract_path = resolve_path(root, source_config.data.contract_path)
    if (
        not stats_path.is_file()
        or sha256_file(stats_path) != fold["normalization_stats_sha256"]
        or not data_contract_path.is_file()
        or sha256_file(data_contract_path) != fold["data_contract_sha256"]
        or source_config.train.class_weight_normalization
        != config.train.class_weight_normalization
        or source_config.train.class_weights != config.train.class_weights
        or source_config.train.loss_mode != config.train.loss_mode
        or source_config.train.prefix_auxiliary_weight != config.train.prefix_auxiliary_weight
        or source_config.train.prefix_auxiliary_min_events
        != config.train.prefix_auxiliary_min_events
    ):
        raise ValueError(f"V23 {arm} fold {fold_id} source data/loss contract changed")
    return (
        config,
        source_config,
        fold,
        assignment,
        config_path,
        sha256_file(readiness_path),
        source_binding,
    )


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
        raise ValueError(f"V23 {scope} Dataset violates train-only known-only scope")
    return {
        "scope": scope,
        "parent_count": len(actual),
        "parent_ids_sha256": sequence_sha256(actual),
        "base_labels": sorted(labels),
        "unknown_parent_count": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
    }


def _checkpoint_payload(
    model: OfficialCfCKnownGateClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    config: Any,
    source_config: Any,
    config_path: Path,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    training_state: dict[str, Any],
    readiness_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": config.train.checkpoint_schema,
        "task_name": "known_gate",
        "architecture": "official_ncps_cfc",
        "arm": config.arm,
        "variant": source_config.variant,
        "config_name": config.config_name,
        "config_identity_sha256": config_identity_sha256(config),
        "config_file_sha256": sha256_file(config_path),
        "source_config_path": config.source_config_path,
        "source_config_sha256": config.source_config_sha256,
        "source_config_identity_sha256": config.source_config_identity_sha256,
        "source_prediction_sha256": config.source_prediction_sha256,
        "readiness_sha256": readiness_sha256,
        "official_cfc": model.backbone_specification(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "training_state": dict(training_state),
        "full_parameter_training": True,
        "unknown_is_model_class": False,
    }


def _load_checkpoint(
    path: Path,
    model: OfficialCfCKnownGateClassifier,
    optimizer: torch.optim.Optimizer | None,
    *,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    map_location: torch.device,
) -> dict[str, Any]:
    payload = dict(torch.load(path, map_location=map_location, weights_only=False))
    official = dict(payload.get("official_cfc") or {})
    expected = model.backbone_specification()
    if (
        payload.get("checkpoint_schema") != config.train.checkpoint_schema
        or payload.get("task_name") != "known_gate"
        or payload.get("architecture") != "official_ncps_cfc"
        or payload.get("arm") != config.arm
        or payload.get("config_identity_sha256") != config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("source_config_sha256") != config.source_config_sha256
        or payload.get("source_config_identity_sha256") != config.source_config_identity_sha256
        or payload.get("source_prediction_sha256") != config.source_prediction_sha256
        or payload.get("readiness_sha256") != readiness_sha256
        or official != expected
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V23 checkpoint schema/config/runtime binding differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _official_spec_matches_config(model: OfficialCfCKnownGateClassifier, config: Any) -> None:
    actual = model.backbone_specification()
    expected = config.official_cfc
    checks = {
        "distribution": expected.distribution,
        "distribution_version": expected.distribution_version,
        "git_commit": expected.git_commit,
        "class_path": expected.class_path,
        "mode": expected.mode,
        "activation": expected.activation,
        "hidden_dim": expected.hidden_dim,
        "backbone_units": expected.backbone_units,
        "backbone_layers": expected.backbone_layers,
        "backbone_dropout": expected.backbone_dropout,
        "mixed_memory": expected.mixed_memory,
        "return_sequences": expected.return_sequences,
        "batch_first": expected.batch_first,
    }
    if any(actual.get(name) != value for name, value in checks.items()):
        raise RuntimeError("V23 constructed official CfC differs from the pinned config")


def train_fold(
    project_root: Path,
    *,
    arm: str,
    fold_id: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    (
        config,
        source_config,
        fold,
        assignment,
        config_path,
        readiness_hash,
        source_binding,
    ) = _verify_prepared(root, arm=arm, fold_id=fold_id)
    artifact = root / artifact_relative(arm, fold_id)
    prediction_path = root / prediction_relative(arm, fold_id)
    if artifact.resolve() != (root / config.train.formal_artifact_dir).resolve():
        raise ValueError("V23 artifact path differs from its sidecar config")
    if prediction_path.exists():
        raise FileExistsError(f"V23 prediction already exists: {prediction_path}")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            "refusing to overwrite a non-empty V23 OOF fold; use its last checkpoint: "
            f"{artifact}"
        )
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(
        artifact / "training.log",
        logger_name=f"radar_ln3.{arm}.official_cfc.fold_{fold_id:02d}",
    )
    started = time.perf_counter()
    try:
        if config.train.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V23 OOF official CfC training requires an available CUDA device")
        seed_everything(config.train.seed)
        train_dataset = build_known_gate_dataset(
            root,
            source_config,
            split_name="train",
            parent_ids=fold["train_parent_ids"],
            allow_feature_cache_superset=True,
        )
        train_scope = _dataset_scope(
            train_dataset,
            fold["train_parent_ids"],
            f"{arm}_fold_{fold_id:02d}_train",
        )
        expected_counts = {
            KNOWN_GATE_CLASS_NAMES[0]: sum(
                int(fold["train_label_parent_counts"][label])
                for label in KNOWN_LABELS[:3]
            ),
            KNOWN_GATE_CLASS_NAMES[1]: int(
                fold["train_label_parent_counts"]["ClutterTarget"]
            ),
        }
        if train_dataset.label_counts() != expected_counts:
            raise ValueError("V23 fold train binary support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model = OfficialCfCKnownGateClassifier(source_config)
        _official_spec_matches_config(model, config)
        initial_state_hash = _state_dict_sha256(model.state_dict())
        model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        source_order_hashes = list(source_binding["history_parent_order_sha256"])
        if len(source_order_hashes) != FIXED_EPOCHS:
            raise ValueError("V23 source prediction lacks the frozen 20-epoch parent-order hashes")
        # The source and official-CfC architectures cannot have equal weights or
        # logits. Reset non-model RNG before optimization to retain every other
        # stochastic condition from the frozen source protocol.
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
        last_checkpoint = root / config.train.last_checkpoint_path
        final_checkpoint = root / config.train.checkpoint_path
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            try:
                resume_path.relative_to(artifact.resolve())
            except ValueError as error:
                raise ValueError("V23 resume checkpoint must remain inside its fold artifact") from error
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
                raise ValueError("V23 resume state is not an epoch boundary")
            resumed_from = str(resume_path)
        if not 1 <= start_epoch <= FIXED_EPOCHS + 1:
            raise ValueError("V23 resume epoch is outside the fixed budget")
        logger.info(
            "run=start | arm=%s | fold=%02d | encoder=official_ncps_cfc | train=%d | held_out=%d | epochs=%d | held_out_selection=false | unknown=0 | val=0 | test=0",
            arm,
            fold_id,
            len(train_dataset),
            len(fold["held_out_parent_ids"]),
            FIXED_EPOCHS,
        )
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
                    class_weight_normalization=config.train.class_weight_normalization,
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
                    raise ValueError("V23 official CfC produced non-finite loss or gradient")
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
            order_hash = _parent_order_sha256(order)
            expected_order_hash = source_order_hashes[epoch - 1]
            if (
                len(order) != len(train_dataset)
                or len(set(order)) != len(order)
                or sorted(order) != fold["train_parent_ids"]
                or order_hash != expected_order_hash
            ):
                raise ValueError("V23 epoch did not preserve the frozen source parent order")
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": gradient_clip_count,
                "train_steps": steps_per_epoch,
                "parent_order_sha256": order_hash,
                "source_parent_order_sha256": expected_order_hash,
                "parent_order_matches_frozen_source": True,
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "held_out_forward_count": 0,
                "held_out_checkpoint_selection": False,
                "class_weight_normalization": config.train.class_weight_normalization,
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
                source_config=source_config,
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
                order_hash,
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
                raise RuntimeError("V23 fold did not reach the fixed final epoch")
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
            source_config,
            split_name="train",
            parent_ids=fold["held_out_parent_ids"],
            allow_feature_cache_superset=True,
        )
        held_out_scope = _dataset_scope(
            held_out_dataset,
            fold["held_out_parent_ids"],
            f"{arm}_fold_{fold_id:02d}_held_out",
        )
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise ValueError("V23 train/held-out parent overlap is non-zero")
        evaluation = evaluate_known_gate(
            model,
            held_out_dataset,
            config=source_config,
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
                row["provenance_component_id"] = component_by_parent[str(row["parent_sample_id"])]
        predicted_ids = sorted(
            str(row["parent_sample_id"]) for row in evaluation["records"]["parent"]
        )
        if predicted_ids != fold["held_out_parent_ids"]:
            raise ValueError("V23 held-out predictions do not cover the frozen fold")
        prediction = {
            "status": "passed",
            "schema_version": prediction_schema(arm),
            "experiment_name": f"{arm.upper()} official ncps CfC architecture-only OOF",
            "arm": arm,
            "fold_id": fold_id,
            "role": "architecture-comparison OOF support only; not a deployment checkpoint",
            "architecture": "official_ncps_cfc",
            "official_cfc": model.backbone_specification(),
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "capture_content_provenance_component",
            "prediction_unit": "parent with causal-prefix audit records",
            "prediction_protocol": "out_of_fold_fixed_final_epoch",
            "loss_aggregation": {
                "class_weight_normalization": config.train.class_weight_normalization,
                "class_weights": [float(value) for value in config.train.class_weights],
                "parent_unit": True,
                "prefixes_averaged_within_parent_before_class_weight": True,
            },
            "source_control": {
                "config_path": config.source_config_path,
                "config_sha256": config.source_config_sha256,
                "config_identity_sha256": config.source_config_identity_sha256,
                "prediction_path": config.source_prediction_path,
                "prediction_sha256": config.source_prediction_sha256,
                "architecture": "project_local_cfc_inspired_encoder",
                "loss_aggregation_provenance": source_binding[
                    "loss_aggregation_provenance"
                ],
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
            "config_identity_sha256": config_identity_sha256(config),
            "readiness_sha256": readiness_hash,
            "checkpoint_path": str(final_checkpoint),
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "final_model_state_sha256": _state_dict_sha256(model.state_dict()),
            "initial_model_state_sha256": initial_state_hash,
            "initialization_relation_to_control": "not equal by design; architecture replacement",
            "model_parameter_count": model_parameter_count,
            "train_scope": train_scope,
            "held_out_scope": held_out_scope,
            "fold_training_history": history,
            "evaluation": {key: value for key, value in evaluation.items() if key != "records"},
            "records": evaluation["records"],
            "formal_training": "未执行（OOF折模型不是正式部署模型）",
            "threshold_fitting": "未执行",
            "test": "未执行",
            "formal_rejection": "未执行",
            "full_cascade": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "deployment_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json_new_or_identical(prediction_path, prediction)
        summary = {
            "status": "completed",
            "arm": arm,
            "architecture": "official_ncps_cfc",
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
        _write_json_new_or_identical(artifact / "training-summary.json", summary)
        logger.info(
            "run=complete | arm=%s | fold=%02d | encoder=official_ncps_cfc | epoch=%02d | parent_macro_f1=%.6f | prediction=%s",
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
            "architecture": "official_ncps_cfc",
            "error_type": type(error).__name__,
            "error": str(error),
            "test": "未执行",
            "deployment_eligible": False,
        }
        _write_json_new_or_identical(artifact / "failure.json", failure)
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V23 official-CfC KnownGate OOF fold.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(FOLD_COUNT), required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            train_fold(
                args.project_root.resolve(),
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
