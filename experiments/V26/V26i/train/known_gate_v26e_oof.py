"""Train one V26e fixed-epoch-32 L1 Known-only OOF fold.

The trainer intentionally has no L2 stage: V26e is an L1 long-horizon
diagnostic and consumes V26c's sealed L2 epoch-15 outputs only during report
construction.  The held-out fold is opened once after epoch 32 and never
participates in training or checkpoint selection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping


from radar_ln3.known_gate import KnownGateConfig, load_known_gate_config
from radar_ln3.known_gate_v26e_oof import L1_CHECKPOINT_SCHEMA, L1_EPOCH, L1_TRAIN_EPOCHS, l1_config_path, require_bound_file, relative_path, resolve_path, sha256_file



L1_SOURCE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43-config.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26e expected JSON object: {path}")
    return value




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







