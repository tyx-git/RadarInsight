"""Train V26c's isolated L1/L2 Known-only provenance-grouped OOF folds.

This module deliberately bypasses the historical full-run config loaders after
they have verified the two source protocols.  Its runtime configs are created
from those frozen source configs via ``dataclasses.replace`` and then bound to
V26c's fold-local config/data-contract/cache hashes.  Therefore a V26c fold
cannot accidentally load a V26/V18r full cache, checkpoint or normalizer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.data import DT_REASON_NAMES
from radar_ln3.known_gate import KNOWN_GATE_CLASS_NAMES, UNKNOWN_LABEL, KnownGateConfig, build_known_gate_dataloader, build_known_gate_model_for_config, build_known_gate_oof_dataset, evaluate_known_gate, known_gate_config_identity_sha256, load_known_gate_config, seed_everything as seed_known_gate
from radar_ln3.known_gate_v26c_oof import CONTRACT, CONTRACT_SCHEMA, L1_CHECKPOINT_SCHEMA, L1_EPOCH, L1_PREDICTION_SCHEMA, L2_CHECKPOINT_SCHEMA, L2_EPOCH, L2_PREDICTION_SCHEMA, READINESS, READINESS_SCHEMA, fold_record, l1_checkpoint_path, l1_config_path, l1_held_out_feature_cache_path, l1_last_checkpoint_path, l1_prediction_path, l1_train_feature_cache_path, l1_training_summary_path, l2_checkpoint_path, l2_config_path, l2_held_out_micro_cache_path, l2_held_out_mixed_micro_cache_path, l2_held_out_mixed_physical_cache_path, l2_held_out_physical_cache_path, l2_last_checkpoint_path, l2_prediction_path, l2_train_micro_cache_path, l2_train_physical_cache_path, l2_training_summary_path, relative_path, resolve_path, sequence_sha256, sha256_file, write_new_or_identical
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
)
from radar_ln3.l1_l2_raw_routing import (
    FrozenL2MixedAuditDataset,
    build_frozen_l2_mixed_audit_dataloader,
    l2_model_inputs as mixed_l2_model_inputs,
    verify_l1_l2_event_alignment,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from radar_ln3.task_models import (
    LiquidTaskConfig,
    build_paired_model_for_config,
    build_task_dataloader,
    build_task_dataset,
    is_product_local_schema,
    load_task_config,
    seed_everything as seed_task,
    summarize_task_records,
    task_config_identity_sha256,
)
from train.known_gate_common import (
    _capture_training_state,
    _model_inputs as l1_model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _state_dict_sha256,
)


L1_SOURCE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43-config.json"
)
L2_SOURCE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-micro-doppler8-"
    "stable-density-treatment-seed42-config.json"
)
L1_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
L2_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
FIXED_EPOCHS = 20


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26c expected JSON object: {path}")
    return value


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _read_config_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload.get("v26c_oof"), dict):
        raise ValueError(f"V26c fold config lacks its OOF declaration: {path}")
    return payload


def _source_protocols(root: Path) -> tuple[KnownGateConfig, LiquidTaskConfig, str, str]:
    l1_path = root / L1_SOURCE_CONFIG
    l2_path = root / L2_SOURCE_CONFIG
    if not l1_path.is_file() or not l2_path.is_file():
        raise FileNotFoundError("V26c source configs are missing")
    l1 = load_known_gate_config(l1_path)
    l2 = load_task_config(l2_path)
    if (
        l1.variant != "v26_known_only_open_set_baseline"
        or l1.train.seed != 43
        or l1.model.evidence_dim != 1
        or l2.schema_version != "RADAR-LN3-V18r"
        or l2.task.name != "dbb_specialist"
        or l2.train.seed != 42
        or l2.model.rd_physical_dim != 4
        or l2.model.micro_doppler_dim != 8
        or l2.model.liquid_invalid_dt_update_mode != "density_weighted"
    ):
        raise ValueError("V26c source L1/L2 protocols changed")
    return l1, l2, sha256_file(l1_path), sha256_file(l2_path)


def _require_prepared(root: Path, fold_id: int) -> dict[str, Any]:
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V26c prepare must complete before fold training")
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
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
    ):
        raise ValueError("V26c readiness/contract is invalid or stale")
    fold = fold_record(contract, fold_id)
    l1_binding = dict(fold.get("l1") or {})
    l2_binding = dict(fold.get("l2") or {})
    for task, binding, config_relative in (
        ("l1", l1_binding, l1_config_path(fold_id)),
        ("l2", l2_binding, l2_config_path(fold_id)),
    ):
        path = root / config_relative
        if (
            not path.is_file()
            or binding.get("config_path") != relative_path(root, path)
            or binding.get("config_sha256") != sha256_file(path)
        ):
            raise ValueError(f"V26c {task} fold config binding changed")
    return {
        "contract": contract,
        "readiness": readiness,
        "fold": fold,
        "l1_binding": l1_binding,
        "l2_binding": l2_binding,
        "contract_sha256": sha256_file(contract_path),
        "readiness_sha256": sha256_file(readiness_path),
    }


def _runtime_l1_config(root: Path, fold_id: int, binding: Mapping[str, Any]) -> tuple[KnownGateConfig, Path, dict[str, Any]]:
    source, _, _, _ = _source_protocols(root)
    config_path = root / l1_config_path(fold_id)
    payload = _read_config_payload(config_path)
    declaration = dict(payload["v26c_oof"])
    if (
        payload.get("schema_version") != "RADAR-LN3-V26c-OOF-L1"
        or payload.get("variant") != source.variant
        or int(declaration.get("fold_id", -1)) != fold_id
        or int(declaration.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or declaration.get("held_out_checkpoint_selection") is not False
        or int(declaration.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V26c L1 fold config declaration is invalid")
    data = dict(payload["data"])
    train = dict(payload["train"])
    config = replace(
        source,
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
    train_cache = dict(binding.get("train_course_cache") or {})
    held_cache = dict(binding.get("held_out_course_cache") or {})
    if (
        config.train.epochs != FIXED_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.checkpoint_schema != L1_CHECKPOINT_SCHEMA
        or config.train.paired_baseline_config_path is not None
        or relative_path(root, resolve_path(root, config.data.train_feature_cache_path)) != train_cache.get("path")
        or relative_path(root, resolve_path(root, config.data.validation_feature_cache_path)) != held_cache.get("path")
        or config.data.expected_train_feature_cache_sha256 != train_cache.get("sha256")
        or config.data.expected_validation_feature_cache_sha256 != held_cache.get("sha256")
    ):
        raise ValueError("V26c L1 runtime config/cache binding differs")
    return config, config_path, declaration


def _runtime_l2_config(root: Path, fold_id: int, binding: Mapping[str, Any]) -> tuple[LiquidTaskConfig, Path, dict[str, Any]]:
    _, source, _, _ = _source_protocols(root)
    config_path = root / l2_config_path(fold_id)
    payload = _read_config_payload(config_path)
    declaration = dict(payload["v26c_oof"])
    if (
        payload.get("schema_version") != "RADAR-LN3-V26c-OOF-L2"
        or int(declaration.get("fold_id", -1)) != fold_id
        or int(declaration.get("fixed_snapshot_epoch", -1)) != L2_EPOCH
        or declaration.get("held_out_checkpoint_selection") is not False
        or int(declaration.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V26c L2 fold config declaration is invalid")
    data = dict(payload["data"])
    train = dict(payload["train"])
    config = replace(
        source,
        schema_version="RADAR-LN3-V18r",
        config_name=str(payload["config_name"]),
        data=replace(
            source.data,
            contract_path=str(data["contract_path"]),
            expected_contract_sha256=str(data["expected_contract_sha256"]),
            normalization_stats_path=str(data["normalization_stats_path"]),
            expected_stats_sha256=str(data["expected_stats_sha256"]),
            rd_amplitude_center=float(data["rd_amplitude_center"]),
            rd_amplitude_scale=float(data["rd_amplitude_scale"]),
            train_rd_physical_cache_path=str(data["train_rd_physical_cache_path"]),
            validation_rd_physical_cache_path=str(data["validation_rd_physical_cache_path"]),
            expected_train_rd_physical_cache_sha256=str(data["expected_train_rd_physical_cache_sha256"]),
            expected_validation_rd_physical_cache_sha256=str(data["expected_validation_rd_physical_cache_sha256"]),
            train_micro_doppler_cache_path=str(data["train_micro_doppler_cache_path"]),
            validation_micro_doppler_cache_path=str(data["validation_micro_doppler_cache_path"]),
            expected_train_micro_doppler_cache_sha256=str(data["expected_train_micro_doppler_cache_sha256"]),
            expected_validation_micro_doppler_cache_sha256=str(data["expected_validation_micro_doppler_cache_sha256"]),
        ),
        train=replace(
            source.train,
            class_weights=tuple(float(value) for value in train["class_weights"]),
            checkpoint_path=str(train["checkpoint_path"]),
            last_checkpoint_path=str(train["last_checkpoint_path"]),
            formal_artifact_dir=str(train["formal_artifact_dir"]),
            epoch_checkpoint_dir=str(train["epoch_checkpoint_dir"]),
            checkpoint_schema=L2_CHECKPOINT_SCHEMA,
            save_every_epoch=False,
            paired_protocol=None,
            paired_arm=None,
            paired_seed=None,
            paired_counterpart_config_path=None,
            allow_parameter_updates=True,
        ),
    )
    train_physical = dict(binding.get("train_rd_physical_cache") or {})
    held_physical = dict(binding.get("held_out_rd_physical_cache") or {})
    train_micro = dict(binding.get("train_micro_doppler_cache") or {})
    held_micro = dict(binding.get("held_out_micro_doppler_cache") or {})
    actual = (
        relative_path(root, resolve_path(root, config.data.train_rd_physical_cache_path)),
        relative_path(root, resolve_path(root, config.data.validation_rd_physical_cache_path)),
        relative_path(root, resolve_path(root, config.data.train_micro_doppler_cache_path)),
        relative_path(root, resolve_path(root, config.data.validation_micro_doppler_cache_path)),
    )
    expected = (
        train_physical.get("path"),
        held_physical.get("path"),
        train_micro.get("path"),
        held_micro.get("path"),
    )
    if (
        config.train.epochs != FIXED_EPOCHS
        or config.train.early_stopping_patience != 0
        or config.train.checkpoint_schema != L2_CHECKPOINT_SCHEMA
        or config.train.paired_protocol is not None
        or config.train.paired_arm is not None
        or actual != expected
        or config.data.expected_train_rd_physical_cache_sha256 != train_physical.get("sha256")
        or config.data.expected_validation_rd_physical_cache_sha256 != held_physical.get("sha256")
        or config.data.expected_train_micro_doppler_cache_sha256 != train_micro.get("sha256")
        or config.data.expected_validation_micro_doppler_cache_sha256 != held_micro.get("sha256")
    ):
        raise ValueError("V26c L2 runtime config/cache binding differs")
    return config, config_path, declaration


def _l2_model_inputs(tensors: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "tr_features",
        "tr_missing_mask",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_dt_reason",
        "rd_frames",
        "rd_frame_mask",
        "rd_quality",
        "tr_padding_mask",
        "rd_physical_features",
        "rd_physical_valid",
        "micro_doppler_features",
        "micro_doppler_valid",
    )
    missing = [key for key in required if key not in tensors]
    if missing:
        raise ValueError(f"V26c L2 batch lacks model inputs: {missing}")
    return {key: tensors[key] for key in required}


def _checkpoint_payload(
    *,
    task: str,
    schema: str,
    fold_id: int,
    model: Any,
    optimizer: torch.optim.Optimizer,
    config_identity: str,
    config_path: Path,
    contract_sha256: str,
    readiness_sha256: str,
    source_config_sha256: str,
    initial_state_sha256: str,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_schema": schema,
        "task_name": task,
        "fold_id": int(fold_id),
        "config_identity_sha256": config_identity,
        "config_file_sha256": sha256_file(config_path),
        "fold_contract_sha256": contract_sha256,
        "readiness_sha256": readiness_sha256,
        "source_config_sha256": source_config_sha256,
        "initial_model_state_sha256": initial_state_sha256,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "training_state": dict(training_state),
        "full_parameter_training": True,
        "held_out_checkpoint_selection": False,
        "unknown_is_model_class": False,
    }


def _load_checkpoint(
    path: Path,
    *,
    task: str,
    schema: str,
    fold_id: int,
    model: Any,
    optimizer: torch.optim.Optimizer | None,
    config_identity: str,
    config_path: Path,
    contract_sha256: str,
    readiness_sha256: str,
    source_config_sha256: str,
    initial_state_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_schema") != schema
        or payload.get("task_name") != task
        or int(payload.get("fold_id", -1)) != int(fold_id)
        or payload.get("config_identity_sha256") != config_identity
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("fold_contract_sha256") != contract_sha256
        or payload.get("readiness_sha256") != readiness_sha256
        or payload.get("source_config_sha256") != source_config_sha256
        or payload.get("initial_model_state_sha256") != initial_state_sha256
        or payload.get("full_parameter_training") is not True
        or payload.get("held_out_checkpoint_selection") is not False
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V26c checkpoint binding differs from the prepared fold")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return dict(payload)


def _dataset_scope(dataset: Any, expected_parent_ids: Sequence[str], allowed_labels: Sequence[str], scope: str) -> dict[str, Any]:
    actual = sorted(str(value) for value in dataset.parent_ids)
    labels = set(dataset.base.frame["label_name"].astype(str))
    expected = sorted(str(value) for value in expected_parent_ids)
    if (
        actual != expected
        or labels != set(allowed_labels)
        or UNKNOWN_LABEL in labels
        or set(dataset.base.frame["split_name"].astype(str)) != {"train"}
    ):
        raise ValueError(f"V26c {scope} Dataset violates its exact Known-only fold scope")
    return {
        "scope": scope,
        "parent_count": len(actual),
        "parent_ids_sha256": sequence_sha256(actual),
        "base_labels": sorted(labels),
        "unknown_parent_count": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
    }


def _record_parent_ids(evaluation: Mapping[str, Any]) -> list[str]:
    rows = list(dict(evaluation.get("records") or {}).get("parent") or ())
    values = sorted(str(row["parent_sample_id"]) for row in rows)
    if len(values) != len(set(values)):
        raise ValueError("V26c prediction repeats a held-out parent")
    return values


L2_LABEL_TO_ID = {label: index for index, label in enumerate(L2_LABELS)}


def _prediction_record_index(prediction: Mapping[str, Any]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    """Index a sealed L1 prediction without invoking the L1 model."""

    raw = dict(prediction.get("records") or {})
    expected_scopes = {"window", "eligible_prefix", "parent"}
    if set(raw) != expected_scopes:
        raise ValueError("V26c sealed L1 prediction has an incomplete record scope")
    indexed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for scope in expected_scopes:
        values: dict[tuple[str, int], dict[str, Any]] = {}
        for source in list(raw.get(scope) or ()):
            row = dict(source)
            key = (str(row["parent_sample_id"]), int(row["event_index"]))
            if key in values:
                raise ValueError(f"V26c sealed L1 prediction repeats {scope} record: {key}")
            values[key] = row
        indexed[scope] = values
    return indexed


def _l2_evaluation_and_joint_scores(
    *,
    root: Path,
    l1_prediction: Mapping[str, Any],
    l1_dataset: Any,
    l2_model: Any,
    l2_dataset: Any,
    config: LiquidTaskConfig,
    device: torch.device,
    fold_id: int,
    l1_prediction_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run exactly one L2 held-out pass and join it to sealed L1 scores.

    L1 is intentionally score-free here: its fixed-epoch OOF JSON is the
    sealed source of L1 probabilities.  The only model forward in this
    function is the L2 forward over the mixed held-out cache, which includes
    Clutter solely to reproduce the physical routing call.  Clutter rows are
    never added to L2 metrics or threshold fitting.
    """

    alignment = verify_l1_l2_event_alignment(l1_dataset.base, l2_dataset.base)
    l1_records = _prediction_record_index(l1_prediction)
    if not all(l1_records.values()):
        raise ValueError("V26c sealed L1 prediction contains an empty record scope")
    l2_loader = build_frozen_l2_mixed_audit_dataloader(l2_dataset, batch_size=1)
    joint: dict[str, list[dict[str, Any]]] = {"window": [], "eligible_prefix": [], "parent": []}
    evaluation_records: dict[str, list[dict[str, Any]]] = {
        "window": [],
        "eligible_prefix": [],
        "parent": [],
    }
    dt_reason_counts = {name: 0 for name in DT_REASON_NAMES}
    dt_invalid = 0
    model_class_names = tuple(str(value) for value in l2_model.class_names)
    if model_class_names != L2_LABELS:
        raise ValueError("V26c L2 model class names differ from the frozen DBB task")
    l2_model.eval()
    with torch.inference_mode():
        for l2_batch in l2_loader:
            if len(l2_batch["parent_sample_ids"]) != 1:
                raise ValueError("V26c L2 held-out pass must use one parent per forward")
            parent_id = str(l2_batch["parent_sample_ids"][0])
            label = str(l2_batch["label_names"][0])
            if label not in L1_LABELS:
                raise ValueError("V26c mixed L2 held-out pass received a non-Known label")
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in l2_batch.items()
            }
            output = l2_model(**mixed_l2_model_inputs(tensors))
            probabilities = torch.softmax(output["sequence_logits"], dim=-1)[0]
            if not torch.isfinite(probabilities).all():
                raise ValueError("V26c L2 held-out prediction contains a non-finite score")
            padding = tensors["tr_padding_mask"][0]
            if label in L2_LABELS:
                dt_invalid += int((output["dt_invalid_mask"][0].bool() & ~padding).sum().item())
                active_reasons = tensors["tr_dt_reason"][0].masked_select(~padding)
                for reason_id, reason_name in enumerate(DT_REASON_NAMES):
                    dt_reason_counts[reason_name] += int((active_reasons == reason_id).sum().item())
            length = int(l2_batch["lengths"][0].item())
            if probabilities.shape[0] < length:
                raise ValueError("V26c L2 sequence is shorter than the sealed L1 sequence")
            for event_index in range(length):
                key = (parent_id, event_index)
                try:
                    l1_row = dict(l1_records["window"][key])
                except KeyError as exc:
                    raise ValueError(f"V26c sealed L1 prediction lacks event {key}") from exc
                if str(l1_row.get("label_name")) != label:
                    raise ValueError(f"V26c sealed L1/L2 label mismatch at {key}")
                l2_row = probabilities[event_index]
                top_id = int(l2_row.argmax().item())
                top_values = torch.topk(l2_row, k=2).values
                row = {
                    "fold_id": int(fold_id),
                    "parent_sample_id": parent_id,
                    "label_name": label,
                    "event_index": event_index,
                    "window_length": event_index + 1,
                    "product_action_eligible": bool(l1_row["product_action_eligible"]),
                    "l1_known_target_probability": float(l1_row["known_target_probability"]),
                    "l1_known_clutter_probability": float(l1_row["known_clutter_probability"]),
                    "l2_top_subclass": model_class_names[top_id],
                    "l2_subclass_probability": float(l2_row[top_id].item()),
                    "l2_subclass_margin": float((top_values[0] - top_values[1]).item()),
                    "l2_inference_only_for_clutter": label == "ClutterTarget",
                    "l2_training_exposure_for_this_parent": 0 if label == "ClutterTarget" else 1,
                }
                joint["window"].append(dict(row))
                if key in l1_records["eligible_prefix"]:
                    joint["eligible_prefix"].append(dict(row))
                if key in l1_records["parent"]:
                    joint["parent"].append(dict(row))
                if label in L2_LABELS:
                    evaluation_row = {
                        "true_id": L2_LABEL_TO_ID[label],
                        "pred_id": top_id,
                        "confidence": float(l2_row[top_id].item()),
                        "parent_sample_id": parent_id,
                        "track_id": str(l2_batch["track_ids"][0]),
                        "source_name": str(l2_batch["source_names"][0]),
                        "label_name": label,
                        "event_index": event_index,
                        "window_length": event_index + 1,
                        "product_action_eligible": bool(l1_row["product_action_eligible"]),
                    }
                    evaluation_records["window"].append(dict(evaluation_row))
                    if key in l1_records["eligible_prefix"]:
                        evaluation_records["eligible_prefix"].append(dict(evaluation_row))
                    if key in l1_records["parent"]:
                        evaluation_records["parent"].append(dict(evaluation_row))

    actual_window = {(str(row["parent_sample_id"]), int(row["event_index"])) for row in joint["window"]}
    if actual_window != set(l1_records["window"]):
        raise ValueError("V26c L2 joint pass coverage differs from sealed L1 window records")
    for scope in ("eligible_prefix", "parent"):
        actual = {(str(row["parent_sample_id"]), int(row["event_index"])) for row in joint[scope]}
        if actual != set(l1_records[scope]):
            raise ValueError(f"V26c L2 joint pass coverage differs from sealed L1 {scope} records")
    expected_dbb_parent_ids = {
        str(parent_id)
        for parent_id, event_index in l1_records["parent"]
        if str(l1_records["parent"][(parent_id, event_index)]["label_name"]) in L2_LABELS
    }
    actual_dbb_parent_ids = {str(row["parent_sample_id"]) for row in evaluation_records["parent"]}
    if actual_dbb_parent_ids != expected_dbb_parent_ids:
        raise ValueError("V26c L2 evaluation parent coverage differs from the DBB held-out fold")
    active_event_count = sum(
        count for name, count in dt_reason_counts.items() if name != "padding"
    )
    evaluation: dict[str, Any] = {
        "task_name": config.task.name,
        "class_names": list(model_class_names),
        "parent_count": len(evaluation_records["parent"]),
        "window_count": len(evaluation_records["window"]),
        "window_metrics": summarize_task_records(
            evaluation_records["window"], class_names=model_class_names, scope="causal_prefix_windows"
        ) if evaluation_records["window"] else None,
        "parent_metrics": summarize_task_records(
            evaluation_records["parent"], class_names=model_class_names, scope="parent_final"
        ) if evaluation_records["parent"] else None,
        "dt_quality": {
            "active_event_count": active_event_count,
            "dt_invalid_event_count": dt_invalid,
            "dt_reason_counts": dt_reason_counts,
            "raw_zero_dt_event_count": (
                dt_reason_counts["first_event_no_predecessor"]
                + dt_reason_counts["zero_delta_at_second_precision"]
            ),
            "zero_delta_is_gap": False,
        },
        "records": evaluation_records,
    }
    if is_product_local_schema(config.schema_version):
        evaluation.update(
            {
                "prefix_action_contract": {
                    "minimum_evidence_events": config.train.prefix_auxiliary_min_events,
                    "events_before_minimum": "diagnostic_only_pending_for_future_product_action",
                    "eligible_prefix_excludes_parent_final": True,
                    "eligible_prefix_count": len(evaluation_records["eligible_prefix"]),
                },
                "eligible_prefix_metrics": summarize_task_records(
                    evaluation_records["eligible_prefix"],
                    class_names=model_class_names,
                    scope="eligible_causal_prefix_auxiliary_excluding_final",
                ) if evaluation_records["eligible_prefix"] else None,
                "full_cascade_evaluation": "未执行",
                "rejection_calibration": "未执行",
                "stable": "未执行",
                "test": "未执行",
            }
        )
    joint_scores = {
        "alignment": alignment,
        "records": joint,
        "l1_scores_reused_from_sealed_prediction": True,
        "l1_forward_count": 0,
        "l1_prediction_path": relative_path(root, l1_prediction_path),
        "l1_prediction_sha256": sha256_file(l1_prediction_path),
        "l2_held_out_forward_count": 1,
        "mixed_l2_inference_only_for_held_out_clutter": True,
    }
    return evaluation, joint_scores


def _load_l1_for_joint_scores(
    root: Path,
    *,
    prepared: Mapping[str, Any],
    fold_id: int,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Load sealed L1 scores and its held-out view; never forward L1."""

    fold = dict(prepared["fold"])
    binding = dict(prepared["l1_binding"])
    config, config_path, _ = _runtime_l1_config(root, fold_id, binding)
    checkpoint = root / l1_checkpoint_path(fold_id)
    summary = root / l1_training_summary_path(fold_id)
    prediction_path = root / l1_prediction_path(fold_id)
    if not checkpoint.is_file() or not summary.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(
            "V26c L2 joint score pass requires the completed fixed L1 fold; "
            f"fold={fold_id}"
        )
    summary_payload = _read_json(summary)
    prediction = _read_json(prediction_path)
    if (
        summary_payload.get("status") != "completed"
        or summary_payload.get("task") != "l1_known_gate"
        or int(summary_payload.get("fold_id", -1)) != fold_id
        or int(summary_payload.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or summary_payload.get("checkpoint_sha256") != sha256_file(checkpoint)
        or prediction.get("schema_version") != L1_PREDICTION_SCHEMA
        or int(prediction.get("fold_id", -1)) != fold_id
        or int(prediction.get("fixed_snapshot_epoch", -1)) != L1_EPOCH
        or prediction.get("held_out_prediction_pass_count") != 1
        or prediction.get("held_out_checkpoint_selection") is not False
        or prediction.get("unknown_training_exposure") != 0
        or prediction.get("validation_access_count") != 0
        or prediction.get("test_payload_access_count") != 0
        or summary_payload.get("prediction_sha256") != sha256_file(prediction_path)
    ):
        raise ValueError("V26c L1 summary/checkpoint binding differs for L2 joint scoring")
    dataset = build_known_gate_oof_dataset(
        root,
        config,
        parent_ids=fold["held_out_parent_ids"],
        feature_cache_path=root / l1_held_out_feature_cache_path(fold_id),
    )
    scope = _dataset_scope(dataset, fold["held_out_parent_ids"], L1_LABELS, "l1_joint_score_held_out")
    if _record_parent_ids(prediction) != fold["held_out_parent_ids"]:
        raise ValueError("V26c sealed L1 prediction coverage differs for L2 joint scoring")
    return prediction, dataset, {
        "config_path": relative_path(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": known_gate_config_identity_sha256(config),
        "checkpoint_path": relative_path(root, checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_summary_path": relative_path(root, summary),
        "training_summary_sha256": sha256_file(summary),
        "prediction_path": relative_path(root, prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "l1_forward_count": 0,
        "held_out_scope": scope,
    }


def _add_fold_fields(evaluation: Mapping[str, Any], *, fold_id: int, task: str) -> None:
    for scope, rows in dict(evaluation.get("records") or {}).items():
        for row in list(rows):
            row["fold_id"] = int(fold_id)
            row["task"] = task
            row["split_name"] = "train_oof_held_out"
            row["prediction_scope"] = str(scope)


def _train_l1(root: Path, fold_id: int, resume_checkpoint: Path | None) -> dict[str, Any]:
    prepared = _require_prepared(root, fold_id)
    fold = prepared["fold"]
    binding = prepared["l1_binding"]
    config, config_path, declaration = _runtime_l1_config(root, fold_id, binding)
    source, _, source_sha, _ = _source_protocols(root)
    artifact = root / l1_checkpoint_path(fold_id).parent
    checkpoint = root / l1_checkpoint_path(fold_id)
    last_checkpoint = root / l1_last_checkpoint_path(fold_id)
    prediction_path = root / l1_prediction_path(fold_id)
    summary_path = root / l1_training_summary_path(fold_id)
    if prediction_path.exists():
        raise FileExistsError(f"V26c L1 fold prediction already exists: {prediction_path}")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"V26c L1 fold artifact is non-empty: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name=f"radar_ln3.v26c.l1.fold_{fold_id:02d}")
    started = time.perf_counter()
    try:
        if config.train.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V26c L1 OOF training requires CUDA")
        seed_known_gate(config.train.seed)
        train_dataset = build_known_gate_oof_dataset(
            root,
            config,
            parent_ids=fold["train_parent_ids"],
            feature_cache_path=root / l1_train_feature_cache_path(fold_id),
        )
        held_out_dataset = build_known_gate_oof_dataset(
            root,
            config,
            parent_ids=fold["held_out_parent_ids"],
            feature_cache_path=root / l1_held_out_feature_cache_path(fold_id),
        )
        train_scope = _dataset_scope(train_dataset, fold["train_parent_ids"], L1_LABELS, "l1_train")
        held_out_scope = _dataset_scope(held_out_dataset, fold["held_out_parent_ids"], L1_LABELS, "l1_held_out")
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise ValueError("V26c L1 train/held-out parent overlap is non-zero")
        expected_counts = {
            KNOWN_GATE_CLASS_NAMES[0]: sum(
                int(binding["train_label_parent_counts"][label]) for label in L2_LABELS
            ),
            KNOWN_GATE_CLASS_NAMES[1]: int(
                binding["train_label_parent_counts"]["ClutterTarget"]
            ),
        }
        if train_dataset.label_counts() != expected_counts:
            raise ValueError("V26c L1 binary train support differs")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        loader = build_known_gate_dataloader(train_dataset, batch_size=config.train.batch_size, shuffle=True, generator=generator)
        # Use the verified full V26 construction for the initial state, then
        # bind it to the fold runtime config. This preserves the V26 course
        # intervention's zero-column initialization without reusing weights.
        model, reference, pairing = build_known_gate_model_for_config(root, source)
        if reference is not None:
            del reference
        model.config = config
        initial_state_sha = _state_dict_sha256(model.state_dict())
        config_identity = known_gate_config_identity_sha256(config)
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        start_epoch = 1
        global_step = 0
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            if resume_path.parent != artifact.resolve():
                raise ValueError("V26c L1 resume checkpoint must be in its fold artifact")
            payload = _load_checkpoint(
                resume_path, task="known_gate", schema=L1_CHECKPOINT_SCHEMA, fold_id=fold_id,
                model=model, optimizer=optimizer, config_identity=config_identity, config_path=config_path,
                contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"],
                source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, device=device,
            )
            state = _restore_training_state(payload, generator)
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            history = list(state.get("history") or ())
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V26c L1 resume is not at an epoch boundary")
            resumed_from = str(resume_path)
        logger.info("run=start | task=l1 | fold=%02d | train=%d | held_out=%d | epochs=20 | held_out_selection=false | unknown=0 | val=0 | test=0", fold_id, len(train_dataset), len(held_out_dataset))
        steps_per_epoch = len(loader)
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            model.train()
            epoch_started = time.perf_counter()
            total = final = prefix = gradient_max = 0.0
            clip_count = 0
            order: list[str] = []
            for step, batch in enumerate(loader, start=1):
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(**l1_model_inputs(tensors))
                objective = parent_final_task_ce_with_prefix_auxiliary(
                    output["sequence_logits"], tensors["labels"], tensors["tr_padding_mask"],
                    class_weights=class_weights, class_weight_normalization=config.train.class_weight_normalization,
                    prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                    prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
                )
                optimizer.zero_grad(set_to_none=True)
                objective["loss"].backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
                if not torch.isfinite(objective["loss"]).item() or not np.isfinite(gradient):
                    raise ValueError("V26c L1 produced non-finite loss/gradient")
                optimizer.step()
                global_step += 1
                total += float(objective["loss"].item())
                final += float(objective["final_state_loss"].item())
                prefix += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, gradient)
                clip_count += int(gradient > config.train.gradient_clip_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info("progress | task=l1 | fold=%02d | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f", fold_id, epoch, step, steps_per_epoch, float(objective["loss"].item()), float(objective["final_state_loss"].item()), float(objective["prefix_auxiliary_loss"].item()), gradient)
            if len(order) != len(train_dataset) or len(set(order)) != len(order) or sorted(order) != fold["train_parent_ids"]:
                raise ValueError("V26c L1 epoch did not consume each train parent once")
            epoch_record = {
                "epoch": epoch, "global_step": global_step, "train_loss": total / steps_per_epoch,
                "train_final_state_loss": final / steps_per_epoch, "train_prefix_auxiliary_loss": prefix / steps_per_epoch,
                "gradient_norm_max": gradient_max, "gradient_clip_count": clip_count, "train_steps": steps_per_epoch,
                "parent_order_sha256": _parent_order_sha256(order), "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)), "held_out_forward_count": 0,
                "held_out_checkpoint_selection": False, "validation_access_count": 0,
                "test_payload_access_count": 0, "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            training_state = _capture_training_state(generator, run_state={"next_epoch": epoch + 1, "global_step": global_step, "history": history})
            payload = _checkpoint_payload(
                task="known_gate", schema=L1_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=optimizer,
                config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"],
                readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha,
                initial_state_sha256=initial_state_sha, epoch=epoch, global_step=global_step,
                metrics=epoch_record, training_state=training_state,
            )
            _save_checkpoint(last_checkpoint, payload)
            if epoch == L1_EPOCH:
                _save_checkpoint(checkpoint, payload)
            logger.info("epoch | task=l1 | fold=%02d | epoch=%02d/20 | loss=%.6f | fixed_snapshot=%s | held_out_forward=0 | order_sha256=%s | elapsed=%s", fold_id, epoch, epoch_record["train_loss"], epoch == L1_EPOCH, epoch_record["parent_order_sha256"], format_duration(epoch_record["elapsed_seconds"]))
        if not checkpoint.is_file():
            raise RuntimeError("V26c L1 did not write its fixed epoch-14 checkpoint")
        _load_checkpoint(checkpoint, task="known_gate", schema=L1_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=None, config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, device=device)
        evaluation = evaluate_known_gate(model, held_out_dataset, config=config, device=device)
        _add_fold_fields(evaluation, fold_id=fold_id, task="l1_known_gate")
        if _record_parent_ids(evaluation) != fold["held_out_parent_ids"]:
            raise ValueError("V26c L1 held-out prediction coverage differs from the frozen fold")
        prediction = {
            "status": "passed", "schema_version": L1_PREDICTION_SCHEMA, "task_name": "known_gate", "fold_id": fold_id,
            "role": "fresh V26c L1 OOF support only; not a deployment checkpoint", "fit_split": "train", "fit_population": "known_only",
            "grouping": "capture_content_provenance_component", "prediction_protocol": "out_of_fold_fixed_epoch_14",
            "fixed_snapshot_epoch": L1_EPOCH, "held_out_checkpoint_selection": False, "held_out_forward_count_before_fixed_snapshot": 0,
            "held_out_prediction_pass_count": 1, "unknown_training_exposure": 0, "unknown_oof_prediction_count": 0,
            "validation_access_count": 0, "test_payload_access_count": 0, "config_path": relative_path(root, config_path),
            "config_sha256": sha256_file(config_path), "config_identity_sha256": config_identity, "fold_contract_sha256": prepared["contract_sha256"],
            "readiness_sha256": prepared["readiness_sha256"], "source_config_sha256": source_sha,
            "checkpoint_path": relative_path(root, checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "initial_model_state_sha256": initial_state_sha, "final_model_state_sha256": _state_dict_sha256(model.state_dict()),
            "source_initialization": pairing, "train_scope": train_scope, "held_out_scope": held_out_scope,
            "fold_training_history": history, "evaluation": {key: value for key, value in evaluation.items() if key != "records"}, "records": evaluation["records"],
            "threshold_fitting": "not_executed", "test": "not_executed", "deployment_eligible": False,
        }
        write_new_or_identical(prediction_path, prediction)
        summary = {"status": "completed", "task": "l1_known_gate", "fold_id": fold_id, "fixed_snapshot_epoch": L1_EPOCH, "global_step": global_step, "prediction_path": relative_path(root, prediction_path), "prediction_sha256": sha256_file(prediction_path), "checkpoint_path": relative_path(root, checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "held_out_parent_metrics": evaluation["parent_metrics"], "held_out_eligible_prefix_metrics": evaluation["eligible_prefix_metrics"], "unknown_training_exposure": 0, "validation_access_count": 0, "test_payload_access_count": 0, "resume": {"resumed": resumed_from is not None, "checkpoint": resumed_from, "epoch_boundary_only": True}, "deployment_eligible": False}
        write_new_or_identical(summary_path, summary)
        return summary
    finally:
        close_training_logger(logger)


def _train_l2(root: Path, fold_id: int, resume_checkpoint: Path | None) -> dict[str, Any]:
    prepared = _require_prepared(root, fold_id)
    fold = prepared["fold"]
    binding = prepared["l2_binding"]
    config, config_path, declaration = _runtime_l2_config(root, fold_id, binding)
    _, source, _, source_sha = _source_protocols(root)
    artifact = root / l2_checkpoint_path(fold_id).parent
    checkpoint = root / l2_checkpoint_path(fold_id)
    last_checkpoint = root / l2_last_checkpoint_path(fold_id)
    prediction_path = root / l2_prediction_path(fold_id)
    summary_path = root / l2_training_summary_path(fold_id)
    if prediction_path.exists():
        raise FileExistsError(f"V26c L2 fold prediction already exists: {prediction_path}")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"V26c L2 fold artifact is non-empty: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name=f"radar_ln3.v26c.l2.fold_{fold_id:02d}")
    started = time.perf_counter()
    try:
        if config.train.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V26c L2 OOF training requires CUDA")
        seed_task(config.train.seed)
        train_dataset = build_task_dataset(root, config, split_name="train", parent_ids=binding["train_parent_ids"], rd_physical_cache_path_override=root / l2_train_physical_cache_path(fold_id), micro_doppler_cache_path_override=root / l2_train_micro_cache_path(fold_id))
        held_out_dataset = build_task_dataset(root, config, split_name="train", parent_ids=binding["held_out_parent_ids"], rd_physical_cache_path_override=root / l2_held_out_physical_cache_path(fold_id), micro_doppler_cache_path_override=root / l2_held_out_micro_cache_path(fold_id))
        train_scope = _dataset_scope(train_dataset, binding["train_parent_ids"], L2_LABELS, "l2_train")
        held_out_scope = _dataset_scope(held_out_dataset, binding["held_out_parent_ids"], L2_LABELS, "l2_held_out")
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise ValueError("V26c L2 train/held-out parent overlap is non-zero")
        held_out_parent_count = len(held_out_dataset)
        # The DBB-only held-out Dataset validates the frozen L2 split/cache
        # binding, but evaluation itself must come from the single mixed pass
        # below.  Release its cache maps before the 20-epoch GPU run so it
        # cannot increase host-memory pressure or be forwarded accidentally.
        del held_out_dataset
        expected_counts = {label: int(binding["train_label_parent_counts"][label]) for label in L2_LABELS}
        if train_dataset.label_counts() != expected_counts:
            raise ValueError("V26c L2 train support differs")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.train.dataloader_seed or config.train.seed))
        loader = build_task_dataloader(train_dataset, batch_size=config.train.batch_size, shuffle=True, generator=generator)
        # Use V18r's verified density treatment construction, then bind the
        # freshly initialized state to the fold-specific runtime config. This
        # reproduces stable8+density initialization while never loading V18r weights.
        model, reference, pairing = build_paired_model_for_config(root, source)
        del reference
        model.config = config
        initial_state_sha = _state_dict_sha256(model.state_dict())
        config_identity = task_config_identity_sha256(config)
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        start_epoch = 1
        global_step = 0
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            if resume_path.parent != artifact.resolve():
                raise ValueError("V26c L2 resume checkpoint must be in its fold artifact")
            payload = _load_checkpoint(resume_path, task="dbb_specialist", schema=L2_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=optimizer, config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, device=device)
            state = _restore_training_state(payload, generator)
            start_epoch, global_step, history = int(state["next_epoch"]), int(state["global_step"]), list(state.get("history") or ())
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V26c L2 resume is not at an epoch boundary")
            resumed_from = str(resume_path)
        logger.info("run=start | task=l2 | fold=%02d | train=%d | held_out=%d | epochs=20 | held_out_selection=false | unknown=0 | val=0 | test=0", fold_id, len(train_dataset), held_out_parent_count)
        steps_per_epoch = len(loader)
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            model.train(); epoch_started = time.perf_counter(); total = final = prefix = gradient_max = 0.0; clip_count = 0; order: list[str] = []
            for step, batch in enumerate(loader, start=1):
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(**_l2_model_inputs(tensors))
                objective = parent_final_task_ce_with_prefix_auxiliary(output["sequence_logits"], tensors["labels"], tensors["tr_padding_mask"], class_weights=class_weights, class_weight_normalization="batch_weight_sum", prefix_auxiliary_weight=config.train.prefix_auxiliary_weight, prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events)
                optimizer.zero_grad(set_to_none=True); objective["loss"].backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
                if not torch.isfinite(objective["loss"]).item() or not np.isfinite(gradient):
                    raise ValueError("V26c L2 produced non-finite loss/gradient")
                optimizer.step(); global_step += 1; total += float(objective["loss"].item()); final += float(objective["final_state_loss"].item()); prefix += float(objective["prefix_auxiliary_loss"].item()); gradient_max = max(gradient_max, gradient); clip_count += int(gradient > config.train.gradient_clip_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info("progress | task=l2 | fold=%02d | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f", fold_id, epoch, step, steps_per_epoch, float(objective["loss"].item()), float(objective["final_state_loss"].item()), float(objective["prefix_auxiliary_loss"].item()), gradient)
            if len(order) != len(train_dataset) or len(set(order)) != len(order) or sorted(order) != binding["train_parent_ids"]:
                raise ValueError("V26c L2 epoch did not consume each train parent once")
            epoch_record = {"epoch": epoch, "global_step": global_step, "train_loss": total / steps_per_epoch, "train_final_state_loss": final / steps_per_epoch, "train_prefix_auxiliary_loss": prefix / steps_per_epoch, "gradient_norm_max": gradient_max, "gradient_clip_count": clip_count, "train_steps": steps_per_epoch, "parent_order_sha256": _parent_order_sha256(order), "parent_order_count": len(order), "parent_order_unique_count": len(set(order)), "held_out_forward_count": 0, "held_out_checkpoint_selection": False, "validation_access_count": 0, "test_payload_access_count": 0, "elapsed_seconds": time.perf_counter() - epoch_started}
            history.append(epoch_record)
            training_state = _capture_training_state(generator, run_state={"next_epoch": epoch + 1, "global_step": global_step, "history": history})
            payload = _checkpoint_payload(task="dbb_specialist", schema=L2_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=optimizer, config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, epoch=epoch, global_step=global_step, metrics=epoch_record, training_state=training_state)
            _save_checkpoint(last_checkpoint, payload)
            if epoch == L2_EPOCH: _save_checkpoint(checkpoint, payload)
            logger.info("epoch | task=l2 | fold=%02d | epoch=%02d/20 | loss=%.6f | fixed_snapshot=%s | held_out_forward=0 | order_sha256=%s | elapsed=%s", fold_id, epoch, epoch_record["train_loss"], epoch == L2_EPOCH, epoch_record["parent_order_sha256"], format_duration(epoch_record["elapsed_seconds"]))
        if not checkpoint.is_file(): raise RuntimeError("V26c L2 did not write its fixed epoch-15 checkpoint")
        _load_checkpoint(checkpoint, task="dbb_specialist", schema=L2_CHECKPOINT_SCHEMA, fold_id=fold_id, model=model, optimizer=None, config_identity=config_identity, config_path=config_path, contract_sha256=prepared["contract_sha256"], readiness_sha256=prepared["readiness_sha256"], source_config_sha256=source_sha, initial_state_sha256=initial_state_sha, device=device)
        # The training loader retains its Dataset and cache mappings.  L2's
        # sole held-out pass has a separate, audited mixed Dataset, so release
        # the no-longer-needed training view before constructing it.
        del loader
        del train_dataset
        l1_prediction, l1_held_out_dataset, l1_binding = _load_l1_for_joint_scores(
            root, prepared=prepared, fold_id=fold_id
        )
        mixed_l2_dataset = FrozenL2MixedAuditDataset(
            root,
            config,
            split_name="train",
            rd_physical_cache_path=root / l2_held_out_mixed_physical_cache_path(fold_id),
            micro_doppler_cache_path=root / l2_held_out_mixed_micro_cache_path(fold_id),
            parent_ids=tuple(fold["held_out_parent_ids"]),
        )
        if sorted(str(value) for value in mixed_l2_dataset.parent_ids) != fold["held_out_parent_ids"]:
            raise ValueError("V26c mixed L2 held-out inference scope differs from L1 fold")
        evaluation, joint_scores = _l2_evaluation_and_joint_scores(
            root=root,
            l1_prediction=l1_prediction,
            l2_model=model,
            l1_dataset=l1_held_out_dataset,
            l2_dataset=mixed_l2_dataset,
            config=config,
            device=device,
            fold_id=fold_id,
            l1_prediction_path=root / l1_prediction_path(fold_id),
        )
        if int(joint_scores.get("l2_held_out_forward_count", -1)) != 1 or joint_scores.get("l1_forward_count") != 0:
            raise AssertionError("V26c L2 held-out protocol did not perform exactly one L2 pass with sealed L1 reuse")
        prediction = {"status": "passed", "schema_version": L2_PREDICTION_SCHEMA, "task_name": "dbb_specialist", "fold_id": fold_id, "role": "fresh V26c L2 OOF support only; not a deployment checkpoint", "fit_split": "train", "fit_population": "dbb_known_only", "grouping": "capture_content_provenance_component", "prediction_protocol": "out_of_fold_fixed_epoch_15", "fixed_snapshot_epoch": L2_EPOCH, "held_out_checkpoint_selection": False, "held_out_forward_count_before_fixed_snapshot": 0, "held_out_prediction_pass_count": 1, "unknown_training_exposure": 0, "unknown_oof_prediction_count": 0, "validation_access_count": 0, "test_payload_access_count": 0, "config_path": relative_path(root, config_path), "config_sha256": sha256_file(config_path), "config_identity_sha256": config_identity, "fold_contract_sha256": prepared["contract_sha256"], "readiness_sha256": prepared["readiness_sha256"], "source_config_sha256": source_sha, "checkpoint_path": relative_path(root, checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "initial_model_state_sha256": initial_state_sha, "final_model_state_sha256": _state_dict_sha256(model.state_dict()), "source_initialization": pairing, "train_scope": train_scope, "held_out_scope": held_out_scope, "fold_training_history": history, "evaluation": {key: value for key, value in evaluation.items() if key != "records"}, "records": evaluation["records"], "joint_l1_l2_scores": joint_scores, "joint_l1_binding": l1_binding, "mixed_l2_inference_only_for_held_out_clutter": True, "held_out_l1_scores_reused_from_sealed_prediction": True, "held_out_l1_forward_count": 0, "held_out_l2_forward_count": 1, "threshold_fitting": "not_executed", "test": "not_executed", "deployment_eligible": False}
        write_new_or_identical(prediction_path, prediction)
        summary = {"status": "completed", "task": "l2_dbb_specialist", "fold_id": fold_id, "fixed_snapshot_epoch": L2_EPOCH, "global_step": global_step, "prediction_path": relative_path(root, prediction_path), "prediction_sha256": sha256_file(prediction_path), "checkpoint_path": relative_path(root, checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "held_out_parent_metrics": evaluation["parent_metrics"], "held_out_eligible_prefix_metrics": evaluation.get("eligible_prefix_metrics"), "unknown_training_exposure": 0, "validation_access_count": 0, "test_payload_access_count": 0, "resume": {"resumed": resumed_from is not None, "checkpoint": resumed_from, "epoch_boundary_only": True}, "deployment_eligible": False}
        write_new_or_identical(summary_path, summary)
        return summary
    finally:
        close_training_logger(logger)


def train(project_root: Path, *, stage: str, fold_id: int, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    if stage == "l1": return _train_l1(root, fold_id, resume_checkpoint)
    if stage == "l2": return _train_l2(root, fold_id, resume_checkpoint)
    raise ValueError("V26c stage must be l1 or l2")



