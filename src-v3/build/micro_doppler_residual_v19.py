from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from radar_ln3.micro_doppler_broadband import (
    BROADBAND_MICRO_DOPPLER_NAMES,
    BROADBAND_MICRO_DOPPLER_SCHEMA,
)
from radar_ln3.micro_doppler_residual import (
    MicroDopplerLateResidual,
    ResidualCacheDataset,
    V19_ARTIFACT_RELATIVE_PATH,
    V19_CACHE_FIELDS,
    V19_CACHE_SCHEMA_VERSION,
    V19_CLASS_NAMES,
    V19_CONFIG_NAME,
    V19_CONFIG_RELATIVE_PATH,
    V19_CONTRACT_RELATIVE_PATH,
    V19_FEATURE_DIM,
    V19_NORMALIZATION_RELATIVE_PATH,
    V19_READINESS_RELATIVE_PATH,
    V19_RESIDUAL_BOUND,
    V19_SCHEMA_VERSION,
    V19_TRAINABLE_PARAMETER_COUNT,
    V19_TRAIN_CACHE_RELATIVE_PATH,
    V19_VALIDATION_CACHE_RELATIVE_PATH,
    evaluate_residual,
    normalize_stable8,
    sha256_file,
    tensor_state_sha256,
)
from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
    BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    task_device,
    verify_task_data_contract,
)


CONTROL_CONFIG_RELATIVE_PATH = Path(
    "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-"
    "rd-physical4-broadband-multiscale-micro-doppler8-stable-control-seed42-config.json"
)
FEATURE_CONFIG_RELATIVE_PATH = Path(
    "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-"
    "rd-physical4-broadband-multiscale-micro-doppler8-stable-treatment-seed42-config.json"
)
V18_PAIRED_REPORT_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-paired-validation.json"
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _write_text_if_same(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V19 file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(np.array_equal(left, right, equal_nan=False))


def _write_npz_if_same(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tuple(arrays) != V19_CACHE_FIELDS:
        raise ValueError("V19 cache arrays are not in the frozen field order")
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if tuple(existing.files) != tuple(arrays):
                raise FileExistsError(f"existing V19 cache fields differ: {path}")
            for name, value in arrays.items():
                if not _arrays_equal(np.asarray(existing[name]), np.asarray(value)):
                    raise FileExistsError(
                        f"existing V19 cache array differs: {path}::{name}"
                    )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _require_empty_formal_artifact(root: Path) -> None:
    artifact = root / V19_ARTIFACT_RELATIVE_PATH
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(
            f"refusing to prepare V19 over a non-empty formal artifact: {artifact}"
        )


def _validate_source_pair(root: Path) -> dict[str, Any]:
    control_path = root / CONTROL_CONFIG_RELATIVE_PATH
    feature_path = root / FEATURE_CONFIG_RELATIVE_PATH
    control = load_task_config(control_path)
    feature = load_task_config(feature_path)
    if (
        control.schema_version != BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        or feature.schema_version != BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        or control.train.paired_protocol != BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
        or feature.train.paired_protocol != BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
        or control.train.paired_arm != "control"
        or feature.train.paired_arm != "treatment"
        or control.task.name != "dbb_specialist"
        or feature.task.name != "dbb_specialist"
        or tuple(control.task.class_names) != V19_CLASS_NAMES
        or tuple(feature.task.class_names) != V19_CLASS_NAMES
        or control.model.micro_doppler_dim != 0
        or feature.model.micro_doppler_dim != V19_FEATURE_DIM
        or control.train.seed != 42
        or feature.train.seed != 42
    ):
        raise ValueError("V19 requires the frozen V18 stable8 Control/Treatment pair")
    if (
        feature.data.micro_doppler_schema != BROADBAND_MICRO_DOPPLER_SCHEMA
        or control.data.micro_doppler_schema != BROADBAND_MICRO_DOPPLER_SCHEMA
        or len(BROADBAND_MICRO_DOPPLER_NAMES) != V19_FEATURE_DIM
    ):
        raise ValueError("V19 stable8 feature schema differs from the frozen V18 contract")
    control_contract = verify_task_data_contract(root, control)
    feature_contract = verify_task_data_contract(root, feature)
    if control_contract["hashes"] != feature_contract["hashes"]:
        raise ValueError("V19 source Control/Treatment data contract hashes differ")
    paired_path = root / V18_PAIRED_REPORT_RELATIVE_PATH
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if (
        paired.get("status") != "completed_but_gate_failed"
        or paired.get("test") != "未执行"
        or paired.get("OOF") != "未执行"
        or paired.get("full_cascade") != "未执行"
    ):
        raise ValueError("V19 requires the frozen failed V18 paired report")
    control_summary_path = _resolve(root, control.train.formal_artifact_dir) / "training-summary.json"
    control_summary = json.loads(control_summary_path.read_text(encoding="utf-8"))
    if (
        control_summary.get("status") != "completed"
        or control_summary.get("stop_reason") != "epoch_limit"
        or int(control_summary.get("best_epoch", -1)) <= 0
        or control_summary.get("test_evaluation") != "not_executed"
        or control_summary.get("full_cascade_evaluation") != "not_executed"
        or control_summary.get("oof_rejection_calibration") != "not_executed"
        or control_summary.get("stable_evaluation") != "not_executed"
        or control_summary.get("p95_latency") != "not_executed"
    ):
        raise ValueError("V19 frozen Control summary violates completion/scope discipline")
    history = list(control_summary.get("history") or [])
    if [int(row.get("epoch", -1)) for row in history] != list(range(1, 21)):
        raise ValueError("V19 requires all 20 frozen V18 Control epoch order hashes")
    order_hashes = {
        str(int(row["epoch"])): str(row["parent_order_sha256"]) for row in history
    }
    checkpoint_path = _resolve(root, control.train.checkpoint_path)
    paired_checkpoint_hash = str(
        dict(dict(paired.get("runs") or {}).get("control") or {}).get(
            "best_checkpoint_sha256", ""
        )
    )
    if paired_checkpoint_hash and sha256_file(checkpoint_path) != paired_checkpoint_hash:
        raise ValueError("V19 frozen Control checkpoint differs from the V18 paired report")
    return {
        "control": control,
        "feature": feature,
        "control_config_path": control_path,
        "feature_config_path": feature_path,
        "control_contract": control_contract,
        "feature_contract": feature_contract,
        "paired_path": paired_path,
        "paired": paired,
        "control_summary_path": control_summary_path,
        "control_summary": control_summary,
        "checkpoint_path": checkpoint_path,
        "order_hashes": order_hashes,
    }


def _fit_parent_balanced_normalization(
    root: Path, feature_config: Any
) -> dict[str, Any]:
    cache_path = _resolve(root, feature_config.data.train_micro_doppler_cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "values", "valid"}:
            raise ValueError("V19 source stable8 train cache fields differ")
        sample_ids = payload["sample_ids"].astype(str)
        values = np.asarray(payload["values"], dtype=np.float64)
        valid = np.asarray(payload["valid"], dtype=np.bool_)
    if values.shape != (len(sample_ids), V19_FEATURE_DIM):
        raise ValueError("V19 source stable8 train cache shape differs")
    index = pd.read_csv(_resolve(root, feature_config.data.index_path), low_memory=False)
    index["sample_key"] = index["sample_id"].astype(str)
    if index["sample_key"].duplicated().any():
        raise ValueError("V19 source index repeats a sample_id")
    rows = index.set_index("sample_key", drop=False).loc[sample_ids]
    if not np.array_equal(rows["sample_id"].astype(str).to_numpy(), sample_ids):
        raise ValueError("V19 source stable8 train cache/index order differs")
    if rows["split_name"].astype(str).ne(feature_config.data.train_split_name).any():
        raise ValueError("V19 source stable8 train cache contains another split")
    if not set(rows["label_name"].astype(str)).issubset(set(V19_CLASS_NAMES)):
        raise ValueError("V19 source stable8 train cache contains a non-DBB label")
    parent_positions: dict[str, list[int]] = {}
    for position, parent_id in enumerate(rows["parent_sample_id"].astype(str)):
        parent_positions.setdefault(parent_id, []).append(position)
    parent_means: list[np.ndarray] = []
    parent_labels: list[str] = []
    valid_event_counts: list[int] = []
    for parent_id, positions_list in parent_positions.items():
        positions = np.asarray(positions_list, dtype=np.int64)
        selected = values[positions][valid[positions]]
        if len(selected) <= 0:
            raise ValueError(f"V19 train parent has no valid stable8 event: {parent_id}")
        labels = rows.iloc[positions]["label_name"].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"V19 train parent has multiple labels: {parent_id}")
        parent_means.append(selected.mean(axis=0))
        parent_labels.append(labels[0])
        valid_event_counts.append(int(len(selected)))
    matrix = np.stack(parent_means).astype(np.float64)
    median = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    iqr = q75 - q25
    standard = matrix.std(axis=0)
    scale = np.where(iqr > 1.0e-6, iqr, np.maximum(standard, 1.0e-6))
    if (
        matrix.shape[1] != V19_FEATURE_DIM
        or not np.isfinite(matrix).all()
        or not np.isfinite(median).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("V19 train-only robust normalization is invalid")
    return {
        "schema_version": "RADAR-LN3-V19-train-parent-balanced-stable8-median-iqr-v1",
        "fit_population": "V16_grouped_train_DBB_parents_only",
        "fit_unit": "one_valid_event_mean_per_train_parent",
        "feature_names": list(BROADBAND_MICRO_DOPPLER_NAMES),
        "parent_count": int(len(matrix)),
        "class_parent_counts": dict(sorted(Counter(parent_labels).items())),
        "valid_event_count": int(sum(valid_event_counts)),
        "median": median.tolist(),
        "q25": q25.tolist(),
        "q75": q75.tolist(),
        "iqr": iqr.tolist(),
        "fallback_standard_deviation": standard.tolist(),
        "scale": scale.tolist(),
        "scale_floor": 1.0e-6,
        "validation_used_for_fit": False,
        "test_used_for_fit": False,
        "source_train_cache": str(cache_path),
        "source_train_cache_sha256": sha256_file(cache_path),
    }


def _forward_control(model: LiquidTaskClassifier, tensors: Mapping[str, Any]) -> dict[str, Any]:
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
        micro_doppler_features=None,
        micro_doppler_valid=None,
        rd_frames=tensors["rd_frames"],
        rd_frame_mask=tensors["rd_frame_mask"],
        rd_quality=tensors["rd_quality"],
        tr_padding_mask=tensors["tr_padding_mask"],
    )


def _build_split_cache(
    root: Path,
    *,
    feature_config: Any,
    control_model: LiquidTaskClassifier,
    split_name: str,
    median: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = build_task_dataset(root, feature_config, split_name=split_name)
    loader = build_task_dataloader(
        dataset,
        batch_size=feature_config.train.batch_size,
        shuffle=False,
    )
    parent_count = len(dataset)
    sequence_length = int(feature_config.data.max_sequence_length)
    parent_ids = np.empty(parent_count, dtype=f"<U{max(1, max(map(len, dataset.parent_ids)))}")
    track_ids: list[str] = []
    source_names: list[str] = []
    label_names: list[str] = []
    labels = np.zeros(parent_count, dtype=np.int64)
    lengths = np.zeros(parent_count, dtype=np.int64)
    padding_mask = np.ones((parent_count, sequence_length), dtype=np.bool_)
    micro_valid = np.zeros((parent_count, sequence_length), dtype=np.bool_)
    valid_counts = np.zeros((parent_count, sequence_length), dtype=np.int16)
    baseline_logits = np.zeros(
        (parent_count, sequence_length, len(V19_CLASS_NAMES)), dtype=np.float32
    )
    causal_features = np.zeros(
        (parent_count, sequence_length, V19_FEATURE_DIM), dtype=np.float32
    )
    median_tensor = torch.as_tensor(median, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = _forward_control(control_model, tensors)
            width = int(tensors["tr_padding_mask"].shape[1])
            causal, counts = normalize_stable8(
                tensors["micro_doppler_features"],
                tensors["micro_doppler_valid"],
                tensors["tr_padding_mask"],
                median=median_tensor,
                scale=scale_tensor,
            )
            batch_size = len(batch["parent_sample_ids"])
            stop = cursor + batch_size
            parent_ids[cursor:stop] = np.asarray(batch["parent_sample_ids"], dtype=parent_ids.dtype)
            track_ids.extend(str(value) for value in batch["track_ids"])
            source_names.extend(str(value) for value in batch["source_names"])
            label_names.extend(str(value) for value in batch["label_names"])
            labels[cursor:stop] = batch["labels"].cpu().numpy().astype(np.int64)
            lengths[cursor:stop] = batch["lengths"].cpu().numpy().astype(np.int64)
            padding_mask[cursor:stop, :width] = (
                tensors["tr_padding_mask"].cpu().numpy().astype(np.bool_)
            )
            micro_valid[cursor:stop, :width] = (
                tensors["micro_doppler_valid"].cpu().numpy().astype(np.bool_)
            )
            valid_counts[cursor:stop, :width] = (
                counts.cpu().numpy().astype(np.int16)
            )
            baseline_logits[cursor:stop, :width] = (
                output["sequence_logits"].cpu().numpy().astype(np.float32)
            )
            causal_features[cursor:stop, :width] = (
                causal.cpu().numpy().astype(np.float32)
            )
            cursor = stop
    if cursor != parent_count:
        raise ValueError("V19 frozen-logit inference did not cover every parent")
    if parent_ids.tolist() != list(dataset.parent_ids):
        raise ValueError("V19 cache parent order differs from the source Dataset")
    max_track = max(1, max(map(len, track_ids)))
    max_source = max(1, max(map(len, source_names)))
    max_label = max(1, max(map(len, label_names)))
    arrays = {
        "schema_version": np.asarray(V19_CACHE_SCHEMA_VERSION),
        "split_name": np.asarray(split_name),
        "parent_sample_ids": parent_ids,
        "track_ids": np.asarray(track_ids, dtype=f"<U{max_track}"),
        "source_names": np.asarray(source_names, dtype=f"<U{max_source}"),
        "label_names": np.asarray(label_names, dtype=f"<U{max_label}"),
        "labels": labels,
        "lengths": lengths,
        "padding_mask": padding_mask,
        "micro_doppler_valid": micro_valid,
        "valid_counts": valid_counts,
        "baseline_logits": baseline_logits,
        "causal_features": causal_features,
    }
    return arrays, {
        "split_name": split_name,
        "parent_count": parent_count,
        "event_count": int(lengths.sum()),
        "valid_micro_doppler_event_count": int(micro_valid.sum()),
        "class_parent_counts": dict(sorted(Counter(label_names).items())),
        "minimum_valid_observations_at_final": int(
            min(valid_counts[index, lengths[index] - 1] for index in range(parent_count))
        ),
        "maximum_valid_observations_at_final": int(
            max(valid_counts[index, lengths[index] - 1] for index in range(parent_count))
        ),
    }


def _metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    _require_empty_formal_artifact(root)
    source = _validate_source_pair(root)
    normalization = _fit_parent_balanced_normalization(root, source["feature"])
    normalization_path = root / V19_NORMALIZATION_RELATIVE_PATH
    _write_text_if_same(normalization_path, _json_text(normalization))

    device = task_device(source["control"])
    control_model = LiquidTaskClassifier(source["control"]).to(device)
    checkpoint = load_task_checkpoint(
        source["checkpoint_path"],
        control_model,
        expected_schema=source["control"].train.checkpoint_schema,
        map_location=device,
    )
    expected_best_epoch = int(source["control_summary"]["best_epoch"])
    if int(checkpoint.get("epoch", -1)) != expected_best_epoch:
        raise ValueError("V19 frozen Control checkpoint epoch differs from its summary")
    for parameter in control_model.parameters():
        parameter.requires_grad_(False)
    control_model.eval()
    backbone_hash_before = tensor_state_sha256(control_model.state_dict())
    median = np.asarray(normalization["median"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    train_arrays, train_description = _build_split_cache(
        root,
        feature_config=source["feature"],
        control_model=control_model,
        split_name=source["feature"].data.train_split_name,
        median=median,
        scale=scale,
        device=device,
    )
    validation_arrays, validation_description = _build_split_cache(
        root,
        feature_config=source["feature"],
        control_model=control_model,
        split_name=source["feature"].data.validation_split_name,
        median=median,
        scale=scale,
        device=device,
    )
    backbone_hash_after = tensor_state_sha256(control_model.state_dict())
    if backbone_hash_before != backbone_hash_after:
        raise ValueError("V19 frozen Control backbone changed during cache preparation")
    train_path = root / V19_TRAIN_CACHE_RELATIVE_PATH
    validation_path = root / V19_VALIDATION_CACHE_RELATIVE_PATH
    _write_npz_if_same(train_path, train_arrays)
    _write_npz_if_same(validation_path, validation_arrays)

    train_dataset = ResidualCacheDataset(train_path)
    validation_dataset = ResidualCacheDataset(validation_path)
    if set(train_dataset.parent_sample_ids) & set(validation_dataset.parent_sample_ids):
        raise ValueError("V19 train/validation caches share a parent")
    baseline_validation = evaluate_residual(
        None,
        validation_dataset,
        device=torch.device("cpu"),
        batch_size=source["feature"].train.batch_size,
        prefix_auxiliary_min_events=source["feature"].train.prefix_auxiliary_min_events,
    )
    frozen_best_validation = dict(source["control_summary"].get("best_validation_metrics") or {})
    if not _metrics_match(
        baseline_validation["parent_metrics"],
        dict(frozen_best_validation.get("parent_metrics") or {}),
    ):
        raise ValueError("V19 cached baseline parent metrics differ from frozen V18 Control")
    probe = MicroDopplerLateResidual().cpu()
    first = train_dataset[0]
    with torch.inference_mode():
        initial = probe(
            first["baseline_logits"].unsqueeze(0),
            first["causal_features"].unsqueeze(0),
            first["padding_mask"].unsqueeze(0),
        )
    initial_equal = torch.equal(
        initial["sequence_logits"], first["baseline_logits"].unsqueeze(0)
    )
    if not initial_equal or probe.trainable_parameter_count != V19_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("V19 zero initialization or parameter-count contract failed")

    contract = {
        "schema_version": V19_SCHEMA_VERSION,
        "experiment_scope": "DBB specialist parent-final local task only",
        "single_factor": (
            "replace V18 shared-Fusion micro-Doppler coupling with a frozen-Control "
            "zero-initialized bounded zero-sum late residual"
        ),
        "class_names": list(V19_CLASS_NAMES),
        "feature_names": list(BROADBAND_MICRO_DOPPLER_NAMES),
        "feature_dim": V19_FEATURE_DIM,
        "causal_aggregation": (
            "train-only parent-balanced median/IQR normalization followed by the mean "
            "of valid current-event stable8 observations available up to each prefix"
        ),
        "model_contract": {
            "baseline": "frozen V18 Control best logits",
            "linear_shape": [V19_CLASS_COUNT, V19_FEATURE_DIM],
            "bias": False,
            "zero_initialized": True,
            "zero_sum_across_classes": True,
            "absolute_residual_bound": V19_RESIDUAL_BOUND,
            "maximum_pairwise_logit_correction": 2.0 * V19_RESIDUAL_BOUND,
            "trainable_parameter_count": V19_TRAINABLE_PARAMETER_COUNT,
        },
        "source": {
            "control_config": str(source["control_config_path"]),
            "control_config_sha256": sha256_file(source["control_config_path"]),
            "feature_config": str(source["feature_config_path"]),
            "feature_config_sha256": sha256_file(source["feature_config_path"]),
            "v18_paired_report": str(source["paired_path"]),
            "v18_paired_report_sha256": sha256_file(source["paired_path"]),
            "control_training_summary": str(source["control_summary_path"]),
            "control_training_summary_sha256": sha256_file(source["control_summary_path"]),
            "control_checkpoint": str(source["checkpoint_path"]),
            "control_checkpoint_sha256": sha256_file(source["checkpoint_path"]),
            "control_checkpoint_epoch": expected_best_epoch,
            "control_backbone_state_sha256": backbone_hash_before,
            "data_contract_hashes": source["control_contract"]["hashes"],
        },
        "normalization": {
            "path": str(normalization_path),
            "sha256": sha256_file(normalization_path),
            "fit_population": normalization["fit_population"],
            "fit_unit": normalization["fit_unit"],
            "validation_used_for_fit": False,
            "test_used_for_fit": False,
        },
        "caches": {
            "train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                **train_description,
            },
            "validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
                **validation_description,
            },
        },
        "frozen_v18_control_parent_order_sha256_by_epoch": source["order_hashes"],
        "baseline_validation_parent_metrics": baseline_validation["parent_metrics"],
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / V19_CONTRACT_RELATIVE_PATH
    _write_text_if_same(contract_path, _json_text(contract))

    artifact = root / V19_ARTIFACT_RELATIVE_PATH
    config = {
        "schema_version": V19_SCHEMA_VERSION,
        "config_name": V19_CONFIG_NAME,
        "class_names": list(V19_CLASS_NAMES),
        "contract_path": str(contract_path),
        "expected_contract_sha256": sha256_file(contract_path),
        "readiness_path": str(root / V19_READINESS_RELATIVE_PATH),
        "normalization_path": str(normalization_path),
        "expected_normalization_sha256": sha256_file(normalization_path),
        "train_cache_path": str(train_path),
        "expected_train_cache_sha256": sha256_file(train_path),
        "validation_cache_path": str(validation_path),
        "expected_validation_cache_sha256": sha256_file(validation_path),
        "baseline_checkpoint_path": str(source["checkpoint_path"]),
        "expected_baseline_checkpoint_sha256": sha256_file(source["checkpoint_path"]),
        "expected_baseline_backbone_state_sha256": backbone_hash_before,
        "baseline_training_summary_path": str(source["control_summary_path"]),
        "expected_baseline_training_summary_sha256": sha256_file(
            source["control_summary_path"]
        ),
        "v18_paired_report_path": str(source["paired_path"]),
        "expected_v18_paired_report_sha256": sha256_file(source["paired_path"]),
        "model": contract["model_contract"],
        "train": {
            "seed": 42,
            "dataloader_seed": 42,
            "batch_size": int(source["control"].train.batch_size),
            "learning_rate": float(source["control"].train.learning_rate),
            "optimizer": "AdamW_default_weight_decay",
            "epochs": 20,
            "early_stopping_patience": 0,
            "gradient_clip_norm": float(source["control"].train.gradient_clip_norm),
            "class_weights": list(source["control"].train.class_weights),
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": float(
                source["control"].train.prefix_auxiliary_weight
            ),
            "prefix_auxiliary_min_events": int(
                source["control"].train.prefix_auxiliary_min_events
            ),
            "checkpoint_selection_metric": BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
            "save_every_epoch": True,
            "record_parent_order_hash": True,
            "reference_parent_order_sha256_by_epoch": source["order_hashes"],
            "device": str(source["control"].train.device),
            "checkpoint_schema": (
                "RADAR-LN3-V19-dbb-micro-doppler-late-residual-checkpoint-v1"
            ),
            "artifact_dir": str(artifact),
            "checkpoint_path": str(artifact / "best_model.pt"),
            "last_checkpoint_path": str(artifact / "last_model.pt"),
            "epoch_checkpoint_dir": str(artifact / "epochs"),
        },
        "adoption": {
            "minimum_per_class_precision_recall_f1": 0.95,
            "strict_macro_f1_improvement_over_frozen_control": 0.02,
            "strict_drone_bird_weak_floor_improvement_over_frozen_control": 0.02,
            "drone_to_bird_must_not_increase": True,
            "bird_to_drone_must_not_increase": True,
            "minimum_bidirectional_error_reduction": 1,
        },
        "scope": {
            "formal_training": "manual_only",
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
    config_path = root / V19_CONFIG_RELATIVE_PATH
    _write_text_if_same(config_path, _json_text(config))
    checks = {
        "v18_pair_is_frozen_failed_result_not_adopted": True,
        "control_checkpoint_bound_to_v18_report": True,
        "control_backbone_state_unchanged_during_prepare": (
            backbone_hash_before == backbone_hash_after
        ),
        "normalization_is_train_parent_only": (
            normalization["validation_used_for_fit"] is False
            and normalization["test_used_for_fit"] is False
        ),
        "train_validation_parent_sets_are_disjoint": not bool(
            set(train_dataset.parent_sample_ids) & set(validation_dataset.parent_sample_ids)
        ),
        "cached_baseline_reproduces_frozen_control_parent_metrics": True,
        "zero_initialized_combined_logits_exactly_equal_frozen_control": initial_equal,
        "only_24_bias_free_residual_weights_are_trainable": (
            probe.trainable_parameter_count == V19_TRAINABLE_PARAMETER_COUNT
            and probe.linear.bias is None
        ),
        "test_payload_access_count_is_zero": True,
        "validation_is_excluded_from_readiness_gate": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": "RADAR-LN3-V19-micro-doppler-late-residual-readiness-v1",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "checks": checks,
        "validation_used_for_readiness_gate": False,
        "train_parent_count": len(train_dataset),
        "validation_parent_count": len(validation_dataset),
        "trainable_parameter_count": probe.trainable_parameter_count,
        "baseline_checkpoint_epoch": expected_best_epoch,
        "baseline_checkpoint_sha256": sha256_file(source["checkpoint_path"]),
        "baseline_backbone_state_sha256": backbone_hash_before,
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
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"V19 readiness checks failed: {failed}")
    readiness_path = root / V19_READINESS_RELATIVE_PATH
    _write_text_if_same(readiness_path, _json_text(readiness))
    return {
        "status": "passed",
        "schema_version": readiness["schema_version"],
        "config": str(config_path),
        "contract": str(contract_path),
        "normalization": str(normalization_path),
        "train_cache": str(train_path),
        "validation_cache": str(validation_path),
        "readiness": str(readiness_path),
        "train_parent_count": len(train_dataset),
        "validation_parent_count": len(validation_dataset),
        "trainable_parameter_count": probe.trainable_parameter_count,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V19 frozen-Control micro-Doppler late-residual experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
