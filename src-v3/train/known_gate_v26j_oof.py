"""Train one V26j fixed-epoch-32 causal-RD-GRU L1 OOF fold."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KnownGateConfig,
    build_known_gate_dataloader,
    build_known_gate_oof_dataset,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    load_known_gate_config,
    seed_everything as seed_known_gate,
)
from radar_ln3.known_gate_v26j_oof import (
    CONTRACT,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    L1_CHECKPOINT_SCHEMA,
    L1_CONFIG_SCHEMA,
    L1_EPOCH,
    L1_LABELS,
    L1_PREDICTION_SCHEMA,
    L1_SOURCE_CONFIG,
    L1_TRAIN_EPOCHS,
    L2_LABELS,
    READINESS,
    READINESS_SCHEMA,
    build_v26j_l1_model,
    fold_record,
    l1_checkpoint_path,
    l1_config_path,
    l1_held_out_feature_cache_path,
    l1_last_checkpoint_path,
    l1_prediction_path,
    l1_train_feature_cache_path,
    l1_training_summary_path,
    model_runtime_binding,
    relative_path,
    require_bound_file,
    resolve_path,
    sha256_file,
    validate_frozen_v26c_sources,
    write_new_or_identical,
)
from radar_ln3.logging_utils import close_training_logger, configure_training_logger, format_duration
from train.known_gate_v21 import (
    _capture_training_state,
    _model_inputs as l1_model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _state_dict_sha256,
)
from train.known_gate_v26c_oof import _checkpoint_payload, _dataset_scope, _load_checkpoint, _record_parent_ids
from radar_ln3.known_gate_v26e_oof import l1_config_path as v26e_l1_config_path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26j expected a JSON object: {path}")
    return value


def _require_prepared(root: Path, fold_id: int) -> dict[str, Any]:
    contract_path, readiness_path = root / CONTRACT, root / READINESS
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V26j prepare must complete before fold training")
    contract, readiness = _read_json(contract_path), _read_json(readiness_path)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or contract.get("single_variable", {}).get("field") != "model.rd_frame_aggregation"
        or contract.get("single_variable", {}).get("treatment") != "causal_gru"
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or int(contract.get("l1", {}).get("train_epoch_limit", -1)) != L1_TRAIN_EPOCHS
        or int(contract.get("l1", {}).get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or contract.get("l1", {}).get("rd_frame_aggregation") != "causal_gru"
    ):
        raise ValueError("V26j readiness/contract is invalid or stale")
    validate_frozen_v26c_sources(root, contract)
    fold = fold_record(contract, fold_id)
    binding = dict(fold.get("l1") or {})
    config_path = root / l1_config_path(fold_id)
    if (
        not config_path.is_file()
        or binding.get("config_path") != relative_path(root, config_path)
        or binding.get("config_sha256") != sha256_file(config_path)
        or binding.get("rd_frame_aggregation") != "causal_gru"
    ):
        raise ValueError(f"V26j L1 fold config binding changed: {fold_id}")
    return {"contract": contract, "readiness": readiness, "fold": fold, "binding": binding, "contract_sha256": sha256_file(contract_path), "readiness_sha256": sha256_file(readiness_path)}


def _source_protocol(root: Path) -> tuple[KnownGateConfig, str]:
    path = require_bound_file(root, L1_SOURCE_CONFIG, "V26j masked-mean source config")
    source = load_known_gate_config(path)
    if (
        source.variant != "v26_known_only_open_set_baseline"
        or source.model.rd_frame_aggregation != "masked_mean"
        or source.train.seed != 43
        or source.train.dataloader_seed != 43
        or source.model.evidence_dim != 1
    ):
        raise ValueError("V26j source L1 protocol changed")
    return source, sha256_file(path)


def _runtime_l1_config(root: Path, fold_id: int, binding: Mapping[str, Any]) -> tuple[KnownGateConfig, Path, dict[str, Any]]:
    source, _ = _source_protocol(root)
    config_path = root / l1_config_path(fold_id)
    payload = _read_json(config_path)
    declaration = dict(payload.get("v26j_oof") or {})
    data = dict(payload.get("data") or {})
    model_payload = dict(payload.get("model") or {})
    train = dict(payload.get("train") or {})
    if any(data.get(name) is not None for name in ("feature_cache_manifest_path", "expected_feature_cache_manifest_sha256", "feature_experiment_contract_path", "expected_feature_experiment_contract_sha256")):
        raise ValueError("V26j L1 config must not claim obsolete feature manifest fields")
    if (
        payload.get("schema_version") != L1_CONFIG_SCHEMA
        or payload.get("variant") != "v26j_rd_causal_gru"
        or model_payload.get("rd_frame_aggregation") != "causal_gru"
        or model_payload.get("liquid_state_normalization") != "recurrent"
        or int(declaration.get("fold_id", -1)) != fold_id
        or declaration.get("single_variable") != "rd_frame_aggregation: masked_mean -> causal_gru"
        or int(declaration.get("train_epoch_limit", -1)) != L1_TRAIN_EPOCHS
        or int(declaration.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or declaration.get("held_out_checkpoint_selection") is not False
        or int(declaration.get("unknown_training_exposure", -1)) != 0
        or train.get("paired_baseline_config_path") is not None
    ):
        raise ValueError("V26j L1 fold config declaration is invalid")
    source_v26c_path = require_bound_file(root, str(declaration.get("source_v26c_l1_config_path", "")), f"V26j fold {fold_id} V26c source L1 config")
    if sha256_file(source_v26c_path) != str(declaration.get("source_v26c_l1_config_sha256", "")):
        raise ValueError("V26j V26c source L1 config hash differs")
    direct_v26e_path = require_bound_file(
        root,
        v26e_l1_config_path(fold_id),
        f"V26j fold {fold_id} direct V26e config",
    )
    direct_v26e_payload = _read_json(direct_v26e_path)
    direct_model = dict(direct_v26e_payload.get("model") or {})
    runtime_model = dict(model_payload)
    if (
        direct_v26e_payload.get("schema_version") != "RADAR-LN3-V26e-OOF-L1"
        or direct_v26e_payload.get("variant") != "v26_known_only_open_set_baseline"
        or direct_model.get("rd_frame_aggregation") != "masked_mean"
        or runtime_model.get("rd_frame_aggregation") != "causal_gru"
    ):
        raise ValueError("V26j direct V26e config namespace/aggregation differs")
    direct_model.pop("rd_frame_aggregation", None)
    runtime_model.pop("rd_frame_aggregation", None)
    if direct_model != runtime_model:
        raise ValueError("V26j direct V26e model differs beyond RD aggregation")
    for section, permitted_fields in (
        ("task", frozenset()),
        ("data", frozenset({"contract_path", "expected_contract_sha256"})),
        (
            "train",
            frozenset(
                {
                    "checkpoint_path",
                    "last_checkpoint_path",
                    "formal_artifact_dir",
                    "epoch_checkpoint_dir",
                    "checkpoint_schema",
                }
            ),
        ),
    ):
        direct_section = dict(direct_v26e_payload.get(section) or {})
        runtime_section = dict(payload.get(section) or {})
        for field in set(direct_section) | set(runtime_section):
            if field not in permitted_fields and direct_section.get(field) != runtime_section.get(field):
                raise ValueError(
                    f"V26j direct V26e {section} differs beyond the sealed variable: {field}"
                )
    source_v26c_payload = _read_json(source_v26c_path)
    source_v26c_train = dict(source_v26c_payload.get("train") or {})
    source_v26c_data = dict(source_v26c_payload.get("data") or {})
    if tuple(float(value) for value in train.get("class_weights", ())) != tuple(float(value) for value in source_v26c_train.get("class_weights", ())):
        raise ValueError("V26j class weights differ from frozen V26c")
    for field in ("rd_amplitude_center", "rd_amplitude_scale"):
        if float(data[field]) != float(source_v26c_data.get(field)):
            raise ValueError(f"V26j fold {field} differs from frozen V26c")
    source_model = asdict(source.model)
    runtime_model = dict(model_payload)
    runtime_model.setdefault("rd_temporal_residual_dim", 0)
    source_model["rd_frame_aggregation"] = "causal_gru"
    if runtime_model != source_model:
        raise ValueError("V26j L1 model declaration differs beyond RD aggregation")
    task_payload = dict(payload.get("task") or {})
    if (
        task_payload.get("name") != source.task_name
        or tuple(task_payload.get("class_names") or ()) != source.class_names
        or dict(task_payload.get("label_to_id") or {}) != dict(source.label_to_id)
        or tuple(task_payload.get("forbidden_training_labels") or ()) != source.forbidden_training_labels
        or tuple(task_payload.get("feature_names") or ()) != source.feature_names
    ):
        raise ValueError("V26j task declaration differs from V26e")
    config = load_known_gate_config(config_path)
    source_data = asdict(source.data)
    runtime_data = asdict(config.data)
    mutable_data = {"contract_path", "expected_contract_sha256", "normalization_stats_path", "expected_stats_sha256", "train_feature_cache_path", "validation_feature_cache_path", "expected_train_feature_cache_sha256", "expected_validation_feature_cache_sha256", "rd_amplitude_center", "rd_amplitude_scale", "feature_cache_manifest_path", "expected_feature_cache_manifest_sha256", "feature_experiment_contract_path", "expected_feature_experiment_contract_sha256"}
    if any(runtime_data[key] != source_data[key] for key in source_data if key not in mutable_data):
        raise ValueError("V26j L1 data declaration changed a frozen V26e field")
    source_train = asdict(source.train)
    runtime_train = asdict(config.train)
    mutable_train = {"epochs", "early_stopping_patience", "checkpoint_path", "last_checkpoint_path", "formal_artifact_dir", "epoch_checkpoint_dir", "checkpoint_schema", "save_every_epoch", "paired_baseline_config_path"}
    if any(runtime_train[key] != source_train[key] for key in source_train if key not in mutable_train and key != "class_weights"):
        raise ValueError("V26j L1 train declaration changed a frozen V26e field")
    for field, hash_field in (("contract_path", "expected_contract_sha256"), ("normalization_stats_path", "expected_stats_sha256"), ("train_feature_cache_path", "expected_train_feature_cache_sha256"), ("validation_feature_cache_path", "expected_validation_feature_cache_sha256")):
        path = require_bound_file(root, data[field], f"V26j fold {fold_id} {field}")
        if sha256_file(path) != str(data[hash_field]):
            raise ValueError(f"V26j fold {fold_id} {field} hash differs")
    train_cache, held_cache = dict(binding.get("train_course_cache") or {}), dict(binding.get("held_out_course_cache") or {})
    if (
        config.train.epochs != L1_TRAIN_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.checkpoint_schema != L1_CHECKPOINT_SCHEMA
        or config.model.rd_frame_aggregation != "causal_gru"
        or relative_path(root, resolve_path(root, config.data.train_feature_cache_path)) != train_cache.get("path")
        or relative_path(root, resolve_path(root, config.data.validation_feature_cache_path)) != held_cache.get("path")
        or config.data.expected_train_feature_cache_sha256 != train_cache.get("sha256")
        or config.data.expected_validation_feature_cache_sha256 != held_cache.get("sha256")
        or relative_path(root, resolve_path(root, config.data.normalization_stats_path)) != binding.get("normalization_path")
        or config.data.expected_stats_sha256 != binding.get("normalization_sha256")
    ):
        raise ValueError("V26j runtime config/cache binding differs")
    return config, config_path, declaration


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_v26j_checkpoint(path: Path, *, runtime: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = _load_checkpoint(path, **kwargs)
    if dict(payload.get("rd_runtime") or {}) != dict(runtime):
        raise ValueError("V26j checkpoint RD causal-GRU runtime binding differs")
    return payload


def _verify_epoch_history(
    history: object,
    *,
    fold_id: int,
    parent_count: int,
    steps_per_epoch: int,
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the immutable 32-epoch train-only history without replaying it.

    This is intentionally stricter than the normal resume boundary check.  A
    completed OOF fold must prove that every saved epoch was train-only and
    that the causal-GRU runtime binding did not disappear between epochs.
    """

    if not isinstance(history, list) or len(history) != L1_TRAIN_EPOCHS:
        raise ValueError("V26j completed fold lacks exactly 32 epoch records")
    validated: list[dict[str, Any]] = []
    for expected_epoch, raw in enumerate(history, start=1):
        row = dict(raw) if isinstance(raw, Mapping) else {}
        order_sha256 = str(row.get("parent_order_sha256", ""))
        if (
            int(row.get("epoch", -1)) != expected_epoch
            or int(row.get("global_step", -1)) != expected_epoch * steps_per_epoch
            or int(row.get("train_steps", -1)) != steps_per_epoch
            or int(row.get("parent_order_count", -1)) != parent_count
            or int(row.get("parent_order_unique_count", -1)) != parent_count
            or len(order_sha256) != 64
            or any(character not in "0123456789abcdef" for character in order_sha256)
            or int(row.get("held_out_forward_count", -1)) != 0
            or row.get("held_out_checkpoint_selection") is not False
            or int(row.get("validation_access_count", -1)) != 0
            or int(row.get("test_payload_access_count", -1)) != 0
            or dict(row.get("rd_runtime") or {}) != dict(runtime)
        ):
            raise ValueError(
                f"V26j fold {fold_id} epoch {expected_epoch} history violates the "
                "fixed train-only causal-GRU contract"
            )
        for name in (
            "train_loss",
            "train_final_state_loss",
            "train_prefix_auxiliary_loss",
            "gradient_norm_max",
            "elapsed_seconds",
        ):
            value = float(row.get(name, float("nan")))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"V26j fold {fold_id} epoch {expected_epoch} has invalid {name}"
                )
        if int(row.get("gradient_clip_count", -1)) < 0:
            raise ValueError(
                f"V26j fold {fold_id} epoch {expected_epoch} has invalid clip count"
            )
        validated.append(row)
    return validated


def verify_l1_fold(project_root: Path, *, fold_id: int) -> dict[str, Any]:
    """CPU-only integrity verification for one completed V26j L1 fold.

    The verifier is deliberately read-only and is called by the PowerShell
    pipeline before L3 calibration/audit.  It catches a common false-positive
    completion state: a JSON prediction may exist while the checkpoint was
    produced by the old masked-mean construction or while a resumed run did
    not actually complete all 32 train-only epochs.
    """

    root = project_root.resolve()
    prepared = _require_prepared(root, fold_id)
    fold, binding = prepared["fold"], prepared["binding"]
    config, config_path, _declaration = _runtime_l1_config(root, fold_id, binding)
    _source, source_sha = _source_protocol(root)
    checkpoint = require_bound_file(
        root, l1_checkpoint_path(fold_id), f"V26j fold {fold_id} epoch-32 checkpoint"
    )
    prediction_path = require_bound_file(
        root, l1_prediction_path(fold_id), f"V26j fold {fold_id} held-out prediction"
    )
    summary_path = require_bound_file(
        root, l1_training_summary_path(fold_id), f"V26j fold {fold_id} summary"
    )
    prediction, summary = _read_json(prediction_path), _read_json(summary_path)

    # A CPU map-location is intentional: this verifier must also work after a
    # user releases the GPU or runs calibration on a CPU-only process.
    model, initialization = build_v26j_l1_model(root, config)
    expected_runtime = model_runtime_binding(model)
    initial_state_sha = _state_dict_sha256(model.state_dict())
    config_identity = known_gate_config_identity_sha256(config)
    payload = _load_v26j_checkpoint(
        checkpoint,
        runtime=expected_runtime,
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
        device=torch.device("cpu"),
    )
    runtime_after_load = model_runtime_binding(model)
    parent_count = len(fold["train_parent_ids"])
    steps_per_epoch = int(math.ceil(parent_count / int(config.train.batch_size)))
    checkpoint_history = _verify_epoch_history(
        dict(payload.get("training_state") or {}).get("run_state", {}).get("history"),
        fold_id=fold_id,
        parent_count=parent_count,
        steps_per_epoch=steps_per_epoch,
        runtime=expected_runtime,
    )
    if (
        int(payload.get("epoch", -1)) != L1_EPOCH
        or int(payload.get("global_step", -1)) != L1_EPOCH * steps_per_epoch
        or dict(payload.get("metrics") or {}) != checkpoint_history[-1]
        or dict(payload.get("v26j_initialization") or {}) != dict(initialization)
        or runtime_after_load != expected_runtime
    ):
        raise ValueError(f"V26j fold {fold_id} epoch-32 checkpoint is incomplete or stale")

    record_scopes = dict(prediction.get("records") or {})
    parent_records = list(record_scopes.get("parent") or ())
    expected_held_out = list(fold["held_out_parent_ids"])
    if (
        prediction.get("status") != "passed"
        or prediction.get("schema_version") != L1_PREDICTION_SCHEMA
        or prediction.get("task_name") != "known_gate"
        or int(prediction.get("fold_id", -1)) != fold_id
        or prediction.get("fit_split") != "train"
        or prediction.get("fit_population") != "known_only"
        or prediction.get("prediction_protocol") != "out_of_fold_fixed_epoch_32"
        or int(prediction.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or prediction.get("held_out_checkpoint_selection") is not False
        or int(prediction.get("held_out_forward_count_before_fixed_snapshot", -1)) != 0
        or int(prediction.get("held_out_prediction_pass_count", -1)) != 1
        or int(prediction.get("unknown_training_exposure", -1)) != 0
        or int(prediction.get("unknown_oof_prediction_count", -1)) != 0
        or int(prediction.get("validation_access_count", -1)) != 0
        or int(prediction.get("test_payload_access_count", -1)) != 0
        or prediction.get("config_path") != relative_path(root, config_path)
        or prediction.get("config_sha256") != sha256_file(config_path)
        or prediction.get("config_identity_sha256") != config_identity
        or prediction.get("fold_contract_sha256") != prepared["contract_sha256"]
        or prediction.get("readiness_sha256") != prepared["readiness_sha256"]
        or prediction.get("source_config_sha256") != source_sha
        or prediction.get("checkpoint_path") != relative_path(root, checkpoint)
        or prediction.get("checkpoint_sha256") != sha256_file(checkpoint)
        or prediction.get("initial_model_state_sha256") != initial_state_sha
        or prediction.get("final_model_state_sha256") != _state_dict_sha256(model.state_dict())
        or dict(prediction.get("rd_runtime") or {}) != expected_runtime
        or dict(prediction.get("v26j_initialization") or {}) != dict(initialization)
        or set(record_scopes) != {"window", "eligible_prefix", "parent"}
        or sorted(str(row.get("parent_sample_id", "")) for row in parent_records)
        != expected_held_out
        or len(parent_records) != len(expected_held_out)
        or list(prediction.get("fold_training_history") or ()) != checkpoint_history
    ):
        raise ValueError(f"V26j fold {fold_id} prediction binding differs from epoch-32")
    if (
        summary.get("status") != "completed"
        or summary.get("task") != "l1_known_gate"
        or int(summary.get("fold_id", -1)) != fold_id
        or int(summary.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or int(summary.get("global_step", -1)) != L1_EPOCH * steps_per_epoch
        or dict(summary.get("rd_runtime") or {}) != expected_runtime
        or summary.get("prediction_path") != relative_path(root, prediction_path)
        or summary.get("prediction_sha256") != sha256_file(prediction_path)
        or summary.get("checkpoint_path") != relative_path(root, checkpoint)
        or summary.get("checkpoint_sha256") != sha256_file(checkpoint)
        or int(summary.get("unknown_training_exposure", -1)) != 0
        or int(summary.get("validation_access_count", -1)) != 0
        or int(summary.get("test_payload_access_count", -1)) != 0
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError(f"V26j fold {fold_id} summary binding differs from epoch-32")
    return {
        "status": "passed",
        "role": "read_only_cpu_completed_fold_verification",
        "fold_id": fold_id,
        "checkpoint": relative_path(root, checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction": relative_path(root, prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "training_epoch_count": len(checkpoint_history),
        "steps_per_epoch": steps_per_epoch,
        "global_step": L1_EPOCH * steps_per_epoch,
        "rd_runtime": expected_runtime,
        "unknown_training_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "deployment_eligible": False,
    }


def train_l1(project_root: Path, *, fold_id: int, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    prepared = _require_prepared(root, fold_id)
    fold, binding = prepared["fold"], prepared["binding"]
    config, config_path, _declaration = _runtime_l1_config(root, fold_id, binding)
    _source, source_sha = _source_protocol(root)
    artifact = root / l1_checkpoint_path(fold_id).parent
    checkpoint, last_checkpoint = root / l1_checkpoint_path(fold_id), root / l1_last_checkpoint_path(fold_id)
    prediction_path, summary_path = root / l1_prediction_path(fold_id), root / l1_training_summary_path(fold_id)
    if prediction_path.exists():
        raise FileExistsError(f"V26j L1 fold prediction already exists: {prediction_path}")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"V26j L1 fold artifact is non-empty: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name=f"radar_ln3.v26j.l1.fold_{fold_id:02d}")
    try:
        if config.train.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V26j L1 OOF training requires CUDA")
        seed_known_gate(config.train.seed)
        train_dataset = build_known_gate_oof_dataset(root, config, parent_ids=fold["train_parent_ids"], feature_cache_path=root / l1_train_feature_cache_path(fold_id))
        train_scope = _dataset_scope(train_dataset, fold["train_parent_ids"], L1_LABELS, "v26j_l1_train")
        expected_counts = {KNOWN_GATE_CLASS_NAMES[0]: sum(int(binding["train_label_parent_counts"][label]) for label in L2_LABELS), KNOWN_GATE_CLASS_NAMES[1]: int(binding["train_label_parent_counts"]["ClutterTarget"])}
        if train_dataset.label_counts() != expected_counts:
            raise ValueError("V26j L1 binary train support differs")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.train.dataloader_seed))
        loader = build_known_gate_dataloader(train_dataset, batch_size=config.train.batch_size, shuffle=True, generator=generator)
        model, initialization = build_v26j_l1_model(root, config)
        runtime = model_runtime_binding(model)
        initial_state_sha = _state_dict_sha256(model.state_dict())
        config_identity = known_gate_config_identity_sha256(config)
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        start_epoch, global_step, history, resumed_from = 1, 0, [], None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            if resume_path.parent != artifact.resolve():
                raise ValueError("V26j resume checkpoint must be in its fold artifact")
            payload = _load_v26j_checkpoint(resume_path, runtime=runtime, task="known_gate", schema=L1_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=optimizer, config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, device=device)
            state = _restore_training_state(payload, generator)
            start_epoch, global_step, history = int(state["next_epoch"]), int(state["global_step"]), list(state.get("history") or ())
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V26j resume is not at an epoch boundary")
            resumed_from = str(resume_path)
        logger.info("run=start | task=l1 | fold=%02d | train=%d | held_out=%d | epochs=%d | rd_aggregation=causal_gru | held_out_selection=false | unknown=0 | val=0 | test=0", fold_id, len(train_dataset), len(fold["held_out_parent_ids"]), L1_TRAIN_EPOCHS)
        steps_per_epoch = len(loader)
        for epoch in range(start_epoch, L1_TRAIN_EPOCHS + 1):
            model.train()
            epoch_started, total, final, prefix, gradient_max, clip_count, order = time.perf_counter(), 0.0, 0.0, 0.0, 0.0, 0, []
            for step, batch in enumerate(loader, start=1):
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(**l1_model_inputs(tensors))
                from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
                objective = parent_final_task_ce_with_prefix_auxiliary(output["sequence_logits"], tensors["labels"], tensors["tr_padding_mask"], class_weights=class_weights, class_weight_normalization=config.train.class_weight_normalization, prefix_auxiliary_weight=config.train.prefix_auxiliary_weight, prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events)
                optimizer.zero_grad(set_to_none=True)
                objective["loss"].backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
                if not torch.isfinite(objective["loss"]).item() or not np.isfinite(gradient):
                    raise ValueError("V26j L1 produced a non-finite loss/gradient")
                optimizer.step()
                global_step += 1
                total += float(objective["loss"].item()); final += float(objective["final_state_loss"].item()); prefix += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, gradient); clip_count += int(gradient > config.train.gradient_clip_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info("progress | task=l1 | fold=%02d | epoch=%02d/%02d | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f | rd_aggregation=causal_gru", fold_id, epoch, L1_TRAIN_EPOCHS, step, steps_per_epoch, float(objective["loss"].item()), float(objective["final_state_loss"].item()), float(objective["prefix_auxiliary_loss"].item()), gradient)
            if len(order) != len(train_dataset) or len(set(order)) != len(order) or sorted(order) != fold["train_parent_ids"]:
                raise ValueError("V26j L1 epoch did not consume each train parent once")
            epoch_record = {"epoch": epoch, "global_step": global_step, "train_loss": total / steps_per_epoch, "train_final_state_loss": final / steps_per_epoch, "train_prefix_auxiliary_loss": prefix / steps_per_epoch, "gradient_norm_max": gradient_max, "gradient_clip_count": clip_count, "train_steps": steps_per_epoch, "parent_order_sha256": _parent_order_sha256(order), "parent_order_count": len(order), "parent_order_unique_count": len(set(order)), "held_out_forward_count": 0, "held_out_checkpoint_selection": False, "validation_access_count": 0, "test_payload_access_count": 0, "rd_runtime": runtime, "elapsed_seconds": time.perf_counter() - epoch_started}
            history.append(epoch_record)
            training_state = _capture_training_state(generator, run_state={"next_epoch": epoch + 1, "global_step": global_step, "history": history})
            payload = _checkpoint_payload(task="known_gate", schema=L1_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=optimizer, config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, epoch=epoch, global_step=global_step, metrics=epoch_record, training_state=training_state)
            payload["rd_runtime"] = runtime
            payload["v26j_initialization"] = initialization
            _save_checkpoint(last_checkpoint, payload)
            if epoch == L1_EPOCH:
                _save_checkpoint(checkpoint, payload)
            logger.info("epoch | task=l1 | fold=%02d | epoch=%02d/%02d | loss=%.6f | fixed_snapshot=%s | rd_aggregation=causal_gru | held_out_forward=0 | order_sha256=%s | elapsed=%s", fold_id, epoch, L1_TRAIN_EPOCHS, epoch_record["train_loss"], epoch == L1_EPOCH, epoch_record["parent_order_sha256"], format_duration(epoch_record["elapsed_seconds"]))
        if not checkpoint.is_file():
            raise RuntimeError("V26j L1 did not write its fixed epoch-32 checkpoint")
        _load_v26j_checkpoint(checkpoint, runtime=runtime, task="known_gate", schema=L1_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=None, config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, device=device)
        held_out_dataset = build_known_gate_oof_dataset(root, config, parent_ids=fold["held_out_parent_ids"], feature_cache_path=root / l1_held_out_feature_cache_path(fold_id))
        held_out_scope = _dataset_scope(held_out_dataset, fold["held_out_parent_ids"], L1_LABELS, "v26j_l1_held_out")
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise ValueError("V26j L1 train/held-out parent overlap is non-zero")
        evaluation = evaluate_known_gate(model, held_out_dataset, config=config, device=device)
        from train.known_gate_v26c_oof import _add_fold_fields
        _add_fold_fields(evaluation, fold_id=fold_id, task="l1_known_gate")
        if _record_parent_ids(evaluation) != fold["held_out_parent_ids"]:
            raise ValueError("V26j L1 held-out prediction coverage differs")
        prediction = {"status": "passed", "schema_version": L1_PREDICTION_SCHEMA, "task_name": "known_gate", "fold_id": fold_id, "role": "fresh V26j causal-GRU L1 OOF treatment; not deployment", "fit_split": "train", "fit_population": "known_only", "grouping": "capture_content_provenance_component", "prediction_protocol": "out_of_fold_fixed_epoch_32", "fixed_snapshot_epoch": L1_EPOCH, "held_out_checkpoint_selection": False, "held_out_forward_count_before_fixed_snapshot": 0, "held_out_prediction_pass_count": 1, "unknown_training_exposure": 0, "unknown_oof_prediction_count": 0, "validation_access_count": 0, "test_payload_access_count": 0, "config_path": relative_path(root, config_path), "config_sha256": sha256_file(config_path), "config_identity_sha256": config_identity, "fold_contract_sha256": prepared["contract_sha256"], "readiness_sha256": prepared["readiness_sha256"], "source_config_sha256": source_sha, "checkpoint_path": relative_path(root, checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "initial_model_state_sha256": initial_state_sha, "final_model_state_sha256": _state_dict_sha256(model.state_dict()), "rd_runtime": runtime, "v26j_initialization": initialization, "train_scope": train_scope, "held_out_scope": held_out_scope, "fold_training_history": history, "evaluation": {key: value for key, value in evaluation.items() if key != "records"}, "records": evaluation["records"], "threshold_fitting": "not_executed", "test": "not_executed", "deployment_eligible": False}
        write_new_or_identical(prediction_path, prediction)
        summary = {"status": "completed", "task": "l1_known_gate", "fold_id": fold_id, "fixed_snapshot_epoch": L1_EPOCH, "global_step": global_step, "rd_runtime": runtime, "prediction_path": relative_path(root, prediction_path), "prediction_sha256": sha256_file(prediction_path), "checkpoint_path": relative_path(root, checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "held_out_parent_metrics": evaluation["parent_metrics"], "held_out_eligible_prefix_metrics": evaluation["eligible_prefix_metrics"], "unknown_training_exposure": 0, "validation_access_count": 0, "test_payload_access_count": 0, "resume": {"resumed": resumed_from is not None, "checkpoint": resumed_from, "epoch_boundary_only": True}, "deployment_eligible": False}
        write_new_or_identical(summary_path, summary)
        return summary
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V26j causal-RD-GRU L1 OOF fold.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("l1",), default="l1")
    parser.add_argument("--fold-id", type=int, choices=range(FOLD_COUNT), required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--verify-fold",
        action="store_true",
        help="read-only CPU verification of an already completed fold",
    )
    args = parser.parse_args()
    if args.verify_fold:
        if args.resume_checkpoint is not None:
            raise ValueError("V26j --verify-fold cannot resume a checkpoint")
        result = verify_l1_fold(args.project_root, fold_id=args.fold_id)
    else:
        result = train_l1(
            args.project_root,
            fold_id=args.fold_id,
            resume_checkpoint=args.resume_checkpoint,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
