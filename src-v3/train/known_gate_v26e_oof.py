"""Train one V26e fixed-epoch-32 L1 Known-only OOF fold.

The trainer intentionally has no L2 stage: V26e is an L1 long-horizon
diagnostic and consumes V26c's sealed L2 epoch-15 outputs only during report
construction.  The held-out fold is opened once after epoch 32 and never
participates in training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KnownGateConfig,
    build_known_gate_dataloader,
    build_known_gate_model_for_config,
    build_known_gate_oof_dataset,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    load_known_gate_config,
    seed_everything as seed_known_gate,
)
from radar_ln3.known_gate_v26e_oof import (
    CONTRACT,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    L1_CHECKPOINT_SCHEMA,
    L1_EPOCH,
    L1_PREDICTION_SCHEMA,
    L1_TRAIN_EPOCHS,
    READINESS,
    READINESS_SCHEMA,
    fold_record,
    l1_checkpoint_path,
    l1_config_path,
    l1_data_contract_path,
    l1_held_out_feature_cache_path,
    l1_last_checkpoint_path,
    l1_normalization_path,
    l1_prediction_path,
    l1_train_feature_cache_path,
    l1_training_summary_path,
    require_bound_file,
    relative_path,
    resolve_path,
    sequence_sha256,
    sha256_file,
    write_new_or_identical,
)
from radar_ln3.known_gate_v26e_oof import validate_frozen_v26c_sources

from train.known_gate_v26c_oof import (
    _checkpoint_payload,
    _dataset_scope,
    _load_checkpoint,
    _record_parent_ids,
)
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
)
from train.known_gate_v21 import (
    _capture_training_state,
    _model_inputs as l1_model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _state_dict_sha256,
)


L1_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
L2_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
L1_SOURCE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43-config.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26e expected JSON object: {path}")
    return value


def _require_prepared(root: Path, fold_id: int) -> dict[str, Any]:
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V26e prepare must complete before fold training")
    contract = _read_json(contract_path)
    readiness = _read_json(readiness_path)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or int(contract.get("l1", {}).get("train_epoch_limit", -1)) != L1_TRAIN_EPOCHS
        or int(contract.get("l1", {}).get("fixed_snapshot_epoch", -1)) != L1_EPOCH
    ):
        raise ValueError("V26e readiness/contract is invalid or stale")
    validate_frozen_v26c_sources(root, contract)
    fold = fold_record(contract, fold_id)
    binding = dict(fold.get("l1") or {})
    config_path = root / l1_config_path(fold_id)
    if (
        not config_path.is_file()
        or binding.get("config_path") != relative_path(root, config_path)
        or binding.get("config_sha256") != sha256_file(config_path)
    ):
        raise ValueError(f"V26e L1 fold config binding changed: {fold_id}")
    return {
        "contract": contract,
        "readiness": readiness,
        "fold": fold,
        "binding": binding,
        "contract_sha256": sha256_file(contract_path),
        "readiness_sha256": sha256_file(readiness_path),
    }


def _source_protocol(root: Path) -> tuple[KnownGateConfig, str]:
    path = root / L1_SOURCE_CONFIG
    require_bound_file(root, path, "V26e L1 source config")
    source = load_known_gate_config(path)
    if (
        source.variant != "v26_known_only_open_set_baseline"
        or source.train.seed != 43
        or source.train.dataloader_seed != 43
        or source.model.evidence_dim != 1
    ):
        raise ValueError("V26e source L1 protocol changed")
    return source, sha256_file(path)


def _runtime_l1_config(
    root: Path, fold_id: int, binding: Mapping[str, Any]
) -> tuple[KnownGateConfig, Path, dict[str, Any]]:
    source, _ = _source_protocol(root)
    config_path = root / l1_config_path(fold_id)
    payload = _read_json(config_path)
    declaration = dict(payload.get("v26e_oof") or {})
    payload_data = dict(payload.get("data") or {})
    if any(
        payload_data.get(name) is not None
        for name in (
            "feature_cache_manifest_path",
            "expected_feature_cache_manifest_sha256",
            "feature_experiment_contract_path",
            "expected_feature_experiment_contract_sha256",
        )
    ):
        raise ValueError("V26e L1 config must not claim the obsolete V26 feature manifest")
    if (
        payload.get("schema_version") != "RADAR-LN3-V26e-OOF-L1"
        or payload.get("variant") != source.variant
        or int(declaration.get("fold_id", -1)) != fold_id
        or int(declaration.get("train_epoch_limit", -1)) != L1_TRAIN_EPOCHS
        or int(declaration.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or declaration.get("held_out_checkpoint_selection") is not False
        or int(declaration.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V26e L1 fold config declaration is invalid")
    data = payload_data
    train = dict(payload["train"])
    if train.get("paired_baseline_config_path") is not None:
        raise ValueError("V26e L1 config must not claim a paired baseline")
    source_v26c_path = require_bound_file(
        root,
        str(declaration.get("source_v26c_l1_config_path", "")),
        f"V26e fold {fold_id} V26c source L1 config",
    )
    if sha256_file(source_v26c_path) != str(
        declaration.get("source_v26c_l1_config_sha256", "")
    ):
        raise ValueError("V26e V26c source L1 config hash differs")
    source_v26c_payload = _read_json(source_v26c_path)
    source_v26c_train = dict(source_v26c_payload.get("train") or {})
    source_v26c_data = dict(source_v26c_payload.get("data") or {})
    if tuple(float(value) for value in train.get("class_weights", ())) != tuple(
        float(value) for value in source_v26c_train.get("class_weights", ())
    ):
        raise ValueError("V26e fold class weights differ from the frozen V26c fold config")
    for field in ("rd_amplitude_center", "rd_amplitude_scale"):
        if float(data[field]) != float(source_v26c_data.get(field)):
            raise ValueError(f"V26e fold {field} differs from the frozen V26c fold config")
    model_payload = dict(payload.get("model") or {})
    model_payload.setdefault("rd_temporal_residual_dim", 0)
    if model_payload != asdict(source.model):
        raise ValueError("V26e L1 model declaration differs from the frozen V26 source")
    task_payload = dict(payload.get("task") or {})
    if (
        task_payload.get("name") != source.task_name
        or tuple(task_payload.get("class_names") or ()) != source.class_names
        or dict(task_payload.get("label_to_id") or {}) != dict(source.label_to_id)
        or tuple(task_payload.get("forbidden_training_labels") or ())
        != source.forbidden_training_labels
        or tuple(task_payload.get("feature_names") or ()) != source.feature_names
    ):
        raise ValueError("V26e L1 task declaration differs from the frozen V26 source")
    config = replace(
        source,
        schema_version=str(payload["schema_version"]),
        config_name=str(payload["config_name"]),
        data=replace(
            source.data,
            contract_path=str(data["contract_path"]),
            expected_contract_sha256=str(data["expected_contract_sha256"]),
            normalization_stats_path=str(data["normalization_stats_path"]),
            expected_stats_sha256=str(data["expected_stats_sha256"]),
            rd_amplitude_center=float(data["rd_amplitude_center"]),
            rd_amplitude_scale=float(data["rd_amplitude_scale"]),
            train_feature_cache_path=str(data["train_feature_cache_path"]),
            validation_feature_cache_path=str(data["validation_feature_cache_path"]),
            expected_train_feature_cache_sha256=str(data["expected_train_feature_cache_sha256"]),
            expected_validation_feature_cache_sha256=str(data["expected_validation_feature_cache_sha256"]),
        ),
        train=replace(
            source.train,
            epochs=L1_TRAIN_EPOCHS,
            early_stopping_patience=0,
            class_weights=tuple(float(value) for value in train["class_weights"]),
            checkpoint_path=str(train["checkpoint_path"]),
            last_checkpoint_path=str(train["last_checkpoint_path"]),
            formal_artifact_dir=str(train["formal_artifact_dir"]),
            epoch_checkpoint_dir=str(train["epoch_checkpoint_dir"]),
            checkpoint_schema=L1_CHECKPOINT_SCHEMA,
            save_every_epoch=False,
            paired_baseline_config_path=None,
        ),
    )
    source_data_payload = asdict(source.data)
    runtime_data_payload = asdict(config.data)
    mutable_data_fields = {
        "contract_path",
        "expected_contract_sha256",
        "normalization_stats_path",
        "expected_stats_sha256",
        "train_feature_cache_path",
        "validation_feature_cache_path",
        "expected_train_feature_cache_sha256",
        "expected_validation_feature_cache_sha256",
        "rd_amplitude_center",
        "rd_amplitude_scale",
        "feature_cache_manifest_path",
        "expected_feature_cache_manifest_sha256",
        "feature_experiment_contract_path",
        "expected_feature_experiment_contract_sha256",
    }
    for field in tuple(source_data_payload):
        if field not in mutable_data_fields and runtime_data_payload[field] != source_data_payload[field]:
            raise ValueError(f"V26e L1 data declaration changed frozen field: {field}")
    source_train_payload = asdict(source.train)
    runtime_train_payload = asdict(config.train)
    mutable_train_fields = {
        "epochs",
        "early_stopping_patience",
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "checkpoint_schema",
        "save_every_epoch",
        "paired_baseline_config_path",
    }
    for field in tuple(source_train_payload):
        if field == "class_weights":
            continue
        if field not in mutable_train_fields and runtime_train_payload[field] != source_train_payload[field]:
            raise ValueError(f"V26e L1 train declaration changed frozen field: {field}")
    for field, expected in (
        ("contract_path", data["contract_path"]),
        ("normalization_stats_path", data["normalization_stats_path"]),
        ("train_feature_cache_path", data["train_feature_cache_path"]),
        ("validation_feature_cache_path", data["validation_feature_cache_path"]),
    ):
        value = data[field]
        path = require_bound_file(root, value, f"V26e fold {fold_id} {field}")
        if field == "contract_path":
            expected_hash = data["expected_contract_sha256"]
        elif field == "normalization_stats_path":
            expected_hash = data["expected_stats_sha256"]
        elif field == "train_feature_cache_path":
            expected_hash = data["expected_train_feature_cache_sha256"]
        else:
            expected_hash = data["expected_validation_feature_cache_sha256"]
        if sha256_file(path) != str(expected_hash):
            raise ValueError(f"V26e fold {fold_id} {field} hash differs")
    train_cache = dict(binding.get("train_course_cache") or {})
    held_cache = dict(binding.get("held_out_course_cache") or {})
    if (
        config.train.epochs != L1_TRAIN_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.checkpoint_schema != L1_CHECKPOINT_SCHEMA
        or config.train.paired_baseline_config_path is not None
        or relative_path(root, resolve_path(root, config.data.train_feature_cache_path))
        != train_cache.get("path")
        or relative_path(root, resolve_path(root, config.data.validation_feature_cache_path))
        != held_cache.get("path")
        or config.data.expected_train_feature_cache_sha256 != train_cache.get("sha256")
        or config.data.expected_validation_feature_cache_sha256 != held_cache.get("sha256")
        or relative_path(root, resolve_path(root, config.data.normalization_stats_path))
        != binding.get("normalization_path")
        or config.data.expected_stats_sha256 != binding.get("normalization_sha256")
    ):
        raise ValueError("V26e L1 runtime config/cache binding differs")
    return config, config_path, declaration


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def train_l1(
    project_root: Path,
    *,
    fold_id: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    prepared = _require_prepared(root, fold_id)
    fold = prepared["fold"]
    binding = prepared["binding"]
    config, config_path, declaration = _runtime_l1_config(root, fold_id, binding)
    source, source_sha = _source_protocol(root)
    artifact = root / l1_checkpoint_path(fold_id).parent
    checkpoint = root / l1_checkpoint_path(fold_id)
    last_checkpoint = root / l1_last_checkpoint_path(fold_id)
    prediction_path = root / l1_prediction_path(fold_id)
    summary_path = root / l1_training_summary_path(fold_id)
    if prediction_path.exists():
        raise FileExistsError(f"V26e L1 fold prediction already exists: {prediction_path}")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"V26e L1 fold artifact is non-empty: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(
        artifact / "training.log", logger_name=f"radar_ln3.v26e.l1.fold_{fold_id:02d}"
    )
    started = time.perf_counter()
    try:
        if config.train.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V26e L1 OOF training requires CUDA")
        seed_known_gate(config.train.seed)
        train_dataset = build_known_gate_oof_dataset(
            root,
            config,
            parent_ids=fold["train_parent_ids"],
            feature_cache_path=root / l1_train_feature_cache_path(fold_id),
        )
        train_scope = _dataset_scope(
            train_dataset, fold["train_parent_ids"], L1_LABELS, "l1_train"
        )
        expected_counts = {
            KNOWN_GATE_CLASS_NAMES[0]: sum(
                int(binding["train_label_parent_counts"][label]) for label in L2_LABELS
            ),
            KNOWN_GATE_CLASS_NAMES[1]: int(
                binding["train_label_parent_counts"]["ClutterTarget"]
            ),
        }
        if train_dataset.label_counts() != expected_counts:
            raise ValueError("V26e L1 binary train support differs")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.train.dataloader_seed))
        loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model, reference, pairing = build_known_gate_model_for_config(root, source)
        if reference is not None:
            del reference
        model.config = config
        initial_state_sha = _state_dict_sha256(model.state_dict())
        config_identity = known_gate_config_identity_sha256(config)
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(
            config.train.class_weights, dtype=torch.float32, device=device
        )
        start_epoch = 1
        global_step = 0
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            if resume_path.parent != artifact.resolve():
                raise ValueError("V26e L1 resume checkpoint must be in its fold artifact")
            payload = _load_checkpoint(
                resume_path,
                task="known_gate",
                schema=L1_CHECKPOINT_SCHEMA,
                fold_id=fold_id,
                model=model,
                optimizer=optimizer,
                config_identity=config_identity,
                config_path=config_path,
                contract_sha256=prepared["contract_sha256"],
                readiness_sha256=prepared["readiness_sha256"],
                source_config_sha256=source_sha,
                initial_state_sha256=initial_state_sha,
                device=device,
            )
            state = _restore_training_state(payload, generator)
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            history = list(state.get("history") or ())
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V26e L1 resume is not at an epoch boundary")
            resumed_from = str(resume_path)
        logger.info(
            "run=start | task=l1 | fold=%02d | train=%d | held_out=%d | epochs=%d | held_out_selection=false | unknown=0 | val=0 | test=0",
            fold_id,
            len(train_dataset),
            len(fold["held_out_parent_ids"]),
            L1_TRAIN_EPOCHS,
        )
        steps_per_epoch = len(loader)
        for epoch in range(start_epoch, L1_TRAIN_EPOCHS + 1):
            model.train()
            epoch_started = time.perf_counter()
            total = final = prefix = gradient_max = 0.0
            clip_count = 0
            order: list[str] = []
            for step, batch in enumerate(loader, start=1):
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                # The collate layer exposes ``known_gate_features`` and
                # ``known_gate_feature_valid``.  Keep the authoritative
                # adapter used by the frozen V21--V26c trainer here as well;
                # the old hand-written list used non-existent
                # ``evidence_*`` keys and could never execute a V26e fold.
                output = model(**l1_model_inputs(tensors))
                from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary

                objective = parent_final_task_ce_with_prefix_auxiliary(
                    output["sequence_logits"],
                    tensors["labels"],
                    tensors["tr_padding_mask"],
                    class_weights=class_weights,
                    class_weight_normalization=config.train.class_weight_normalization,
                    prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                    prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
                )
                optimizer.zero_grad(set_to_none=True)
                objective["loss"].backward()
                gradient = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.train.gradient_clip_norm
                    ).item()
                )
                if not torch.isfinite(objective["loss"]).item() or not np.isfinite(gradient):
                    raise ValueError("V26e L1 produced non-finite loss/gradient")
                optimizer.step()
                global_step += 1
                total += float(objective["loss"].item())
                final += float(objective["final_state_loss"].item())
                prefix += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, gradient)
                clip_count += int(gradient > config.train.gradient_clip_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | task=l1 | fold=%02d | epoch=%02d/%02d | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        fold_id,
                        epoch,
                        L1_TRAIN_EPOCHS,
                        step,
                        steps_per_epoch,
                        float(objective["loss"].item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()),
                        gradient,
                    )
            if (
                len(order) != len(train_dataset)
                or len(set(order)) != len(order)
                or sorted(order) != fold["train_parent_ids"]
            ):
                raise ValueError("V26e L1 epoch did not consume each train parent once")
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total / steps_per_epoch,
                "train_final_state_loss": final / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": clip_count,
                "train_steps": steps_per_epoch,
                "parent_order_sha256": _parent_order_sha256(order),
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "held_out_forward_count": 0,
                "held_out_checkpoint_selection": False,
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
                    "history": history,
                },
            )
            payload = _checkpoint_payload(
                task="known_gate",
                schema=L1_CHECKPOINT_SCHEMA,
                fold_id=fold_id,
                model=model,
                optimizer=optimizer,
                config_identity=config_identity,
                config_path=config_path,
                contract_sha256=prepared["contract_sha256"],
                readiness_sha256=prepared["readiness_sha256"],
                source_config_sha256=source_sha,
                initial_state_sha256=initial_state_sha,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
                training_state=training_state,
            )
            _save_checkpoint(last_checkpoint, payload)
            if epoch == L1_EPOCH:
                _save_checkpoint(checkpoint, payload)
            logger.info(
                "epoch | task=l1 | fold=%02d | epoch=%02d/%02d | loss=%.6f | fixed_snapshot=%s | held_out_forward=0 | order_sha256=%s | elapsed=%s",
                fold_id,
                epoch,
                L1_TRAIN_EPOCHS,
                epoch_record["train_loss"],
                epoch == L1_EPOCH,
                epoch_record["parent_order_sha256"],
                format_duration(epoch_record["elapsed_seconds"]),
            )
        if not checkpoint.is_file():
            raise RuntimeError("V26e L1 did not write its fixed epoch-32 checkpoint")
        _load_checkpoint(
            checkpoint,
            task="known_gate",
            schema=L1_CHECKPOINT_SCHEMA,
            fold_id=fold_id,
            model=model,
            optimizer=None,
            config_identity=config_identity,
            config_path=config_path,
            contract_sha256=prepared["contract_sha256"],
            readiness_sha256=prepared["readiness_sha256"],
            source_config_sha256=source_sha,
            initial_state_sha256=initial_state_sha,
            device=device,
        )
        held_out_dataset = build_known_gate_oof_dataset(
            root,
            config,
            parent_ids=fold["held_out_parent_ids"],
            feature_cache_path=root / l1_held_out_feature_cache_path(fold_id),
        )
        held_out_scope = _dataset_scope(
            held_out_dataset, fold["held_out_parent_ids"], L1_LABELS, "l1_held_out"
        )
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise ValueError("V26e L1 train/held-out parent overlap is non-zero")
        evaluation = evaluate_known_gate(model, held_out_dataset, config=config, device=device)
        from train.known_gate_v26c_oof import _add_fold_fields

        _add_fold_fields(evaluation, fold_id=fold_id, task="l1_known_gate")
        if _record_parent_ids(evaluation) != fold["held_out_parent_ids"]:
            raise ValueError("V26e L1 held-out prediction coverage differs from the frozen fold")
        prediction = {
            "status": "passed",
            "schema_version": L1_PREDICTION_SCHEMA,
            "task_name": "known_gate",
            "fold_id": fold_id,
            "role": "fresh V26e L1 OOF support only; not a deployment checkpoint",
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "capture_content_provenance_component",
            "prediction_protocol": "out_of_fold_fixed_epoch_32",
            "fixed_snapshot_epoch": L1_EPOCH,
            "held_out_checkpoint_selection": False,
            "held_out_forward_count_before_fixed_snapshot": 0,
            "held_out_prediction_pass_count": 1,
            "unknown_training_exposure": 0,
            "unknown_oof_prediction_count": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "config_path": relative_path(root, config_path),
            "config_sha256": sha256_file(config_path),
            "config_identity_sha256": config_identity,
            "fold_contract_sha256": prepared["contract_sha256"],
            "readiness_sha256": prepared["readiness_sha256"],
            "source_config_sha256": source_sha,
            "checkpoint_path": relative_path(root, checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "initial_model_state_sha256": initial_state_sha,
            "final_model_state_sha256": _state_dict_sha256(model.state_dict()),
            "source_initialization": pairing,
            "train_scope": train_scope,
            "held_out_scope": held_out_scope,
            "fold_training_history": history,
            "evaluation": {
                key: value for key, value in evaluation.items() if key != "records"
            },
            "records": evaluation["records"],
            "threshold_fitting": "not_executed",
            "test": "not_executed",
            "deployment_eligible": False,
        }
        write_new_or_identical(prediction_path, prediction)
        summary = {
            "status": "completed",
            "task": "l1_known_gate",
            "fold_id": fold_id,
            "fixed_snapshot_epoch": L1_EPOCH,
            "global_step": global_step,
            "prediction_path": relative_path(root, prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "checkpoint_path": relative_path(root, checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "held_out_parent_metrics": evaluation["parent_metrics"],
            "held_out_eligible_prefix_metrics": evaluation["eligible_prefix_metrics"],
            "unknown_training_exposure": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "resume": {
                "resumed": resumed_from is not None,
                "checkpoint": resumed_from,
                "epoch_boundary_only": True,
            },
            "deployment_eligible": False,
        }
        write_new_or_identical(summary_path, summary)
        return summary
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V26e L1 OOF fold.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("l1",), default="l1")
    parser.add_argument("--fold-id", type=int, choices=range(FOLD_COUNT), required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            train_l1(
                args.project_root,
                fold_id=args.fold_id,
                resume_checkpoint=args.resume_checkpoint,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
