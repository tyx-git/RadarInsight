from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.micro_doppler_broadband import BROADBAND_MICRO_DOPPLER_NAMES
from radar_ln3.micro_doppler_residual import (
    ResidualCacheDataset,
    V19_CONFIG_NAME,
    V19_CONFIG_RELATIVE_PATH,
    V19_READINESS_RELATIVE_PATH,
)
from radar_ln3.micro_doppler_residual_v19r import (
    MicroDopplerLateResidual,
    V19R_ARTIFACT_RELATIVE_PATH,
    V19R_CACHE_FIELDS,
    V19R_CACHE_SCHEMA_VERSION,
    V19R_CHECKPOINT_SCHEMA_VERSION,
    V19R_CONFIG_NAME,
    V19R_CONFIG_RELATIVE_PATH,
    V19R_CONTRACT_RELATIVE_PATH,
    V19R_READINESS_RELATIVE_PATH,
    V19R_SCHEMA_VERSION,
    V19R_TRAIN_CACHE_RELATIVE_PATH,
    V19R_VALIDATION_CACHE_RELATIVE_PATH,
    V19_CLASS_COUNT,
    V19_CLASS_NAMES,
    V19_FEATURE_DIM,
    V19_RESIDUAL_BOUND,
    V19_TRAINABLE_PARAMETER_COUNT,
    V19rResidualCacheDataset,
    evaluate_residual,
    normalize_stable8,
    sha256_file,
    tensor_state_sha256,
)
from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
    V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
    V18R_STABLE8_DENSITY_SCHEMA_VERSION,
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    task_device,
    verify_task_data_contract,
)


V18R_CONFIG_RELATIVE_PATH = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-"
    "micro-doppler8-stable-density-treatment-seed42-config.json"
)
V18R_REPORT_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18r-stable8-density-weighted-paired-validation.json"
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
            raise FileExistsError(f"refusing to overwrite a different V19r file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right, equal_nan=False)
    )


def _write_npz_if_same(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tuple(arrays) != V19R_CACHE_FIELDS:
        raise ValueError("V19r cache arrays are not in the frozen field order")
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if tuple(existing.files) != tuple(arrays):
                raise FileExistsError(f"existing V19r cache fields differ: {path}")
            for name, value in arrays.items():
                if not _arrays_equal(np.asarray(existing[name]), np.asarray(value)):
                    raise FileExistsError(
                        f"existing V19r cache array differs: {path}::{name}"
                    )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _progress(message: str) -> None:
    print(f"[V19r prepare] {message}", flush=True)


def _require_empty_formal_artifact(root: Path) -> None:
    artifact = root / V19R_ARTIFACT_RELATIVE_PATH
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(
            f"refusing to prepare V19r over a non-empty formal artifact: {artifact}"
        )


def _model_state_guard(model: LiquidTaskClassifier) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            rows.append(
                (
                    kind,
                    str(name),
                    id(value),
                    int(value._version),
                    tuple(int(item) for item in value.shape),
                    str(value.dtype),
                    str(value.device),
                )
            )
    return tuple(rows)


def _sealed(value: Mapping[str, Any]) -> bool:
    return (
        value.get("test") == "未执行"
        and value.get("OOF") == "未执行"
        and value.get("full_cascade") == "未执行"
        and value.get("formal_rejection") == "未执行"
        and value.get("stable") == "未执行"
        and value.get("P95") == "未执行"
        and value.get("deployment_eligible") is False
    )


def _validate_v18r_source(root: Path) -> dict[str, Any]:
    config_path = root / V18R_CONFIG_RELATIVE_PATH
    report_path = root / V18R_REPORT_RELATIVE_PATH
    config = load_task_config(config_path)
    if (
        config.schema_version != V18R_STABLE8_DENSITY_SCHEMA_VERSION
        or config.train.paired_protocol != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or config.train.paired_arm != "treatment"
        or config.task.name != "dbb_specialist"
        or tuple(config.task.class_names) != V19_CLASS_NAMES
        or config.model.micro_doppler_dim != V19_FEATURE_DIM
        or config.model.liquid_invalid_dt_update_mode != "density_weighted"
        or config.train.seed != 42
    ):
        raise ValueError("V19r requires the completed V18r density+stable8 Treatment")
    contract = verify_task_data_contract(root, config)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed_but_gate_failed" or not _sealed(report):
        raise ValueError("V19r source paired report is incomplete or outside sealed scope")
    treatment = dict(dict(report.get("runs") or {}).get("treatment") or {})
    summary_path = _resolve(root, config.train.formal_artifact_dir) / "training-summary.json"
    checkpoint_path = _resolve(root, config.train.checkpoint_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("stop_reason") != "epoch_limit"
        or int(summary.get("best_epoch", -1)) != 15
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V19r source V18r summary violates completion/scope discipline")
    history = list(summary.get("history") or [])
    if [int(row.get("epoch", -1)) for row in history] != list(range(1, 21)):
        raise ValueError("V19r requires all 20 V18r parent-order hashes")
    order_hashes = {
        str(int(row["epoch"])): str(row["parent_order_sha256"]) for row in history
    }
    if (
        treatment.get("config_sha256") != sha256_file(config_path)
        or treatment.get("summary_sha256") != sha256_file(summary_path)
        or treatment.get("best_checkpoint_sha256") != sha256_file(checkpoint_path)
        or int(treatment.get("best_epoch", -1)) != 15
    ):
        raise ValueError("V19r V18r source hashes differ from the paired report")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V19r baseline checkpoint is missing: {checkpoint_path}")
    return {
        "config": config,
        "config_path": config_path,
        "contract": contract,
        "report": report,
        "report_path": report_path,
        "summary": summary,
        "summary_path": summary_path,
        "checkpoint_path": checkpoint_path,
        "order_hashes": order_hashes,
    }


def _validate_v19_feature_source(root: Path, v18r: Mapping[str, Any]) -> dict[str, Any]:
    config_path = root / V19_CONFIG_RELATIVE_PATH
    readiness_path = root / V19_READINESS_RELATIVE_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        config.get("config_name") != V19_CONFIG_NAME
        or readiness.get("status") != "passed"
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("validation_used_for_readiness_gate") is not False
        or not _sealed(readiness)
        or not all(dict(readiness.get("checks") or {}).values())
    ):
        raise ValueError("V19r requires the completed, sealed V19 feature-cache readiness")
    resolved: dict[str, Path] = {}
    for path_field, hash_field in (
        ("normalization_path", "expected_normalization_sha256"),
        ("train_cache_path", "expected_train_cache_sha256"),
        ("validation_cache_path", "expected_validation_cache_sha256"),
    ):
        path = _resolve(root, config[path_field])
        if not path.is_file() or sha256_file(path) != config[hash_field]:
            raise ValueError(f"V19r source V19 input differs: {path_field}={path}")
        resolved[path_field] = path
    normalization = json.loads(
        resolved["normalization_path"].read_text(encoding="utf-8")
    )
    if (
        tuple(normalization.get("feature_names") or ())
        != tuple(BROADBAND_MICRO_DOPPLER_NAMES)
        or normalization.get("validation_used_for_fit") is not False
        or normalization.get("test_used_for_fit") is not False
        or normalization.get("source_train_cache_sha256")
        != v18r["config"].data.expected_train_micro_doppler_cache_sha256
    ):
        raise ValueError("V19r source normalization is not the frozen train-only stable8 fit")
    train_cache = ResidualCacheDataset(resolved["train_cache_path"])
    validation_cache = ResidualCacheDataset(resolved["validation_cache_path"])
    if set(train_cache.parent_sample_ids) & set(validation_cache.parent_sample_ids):
        raise ValueError("V19r source V19 caches share a parent")
    return {
        "config": config,
        "config_path": config_path,
        "readiness": readiness,
        "readiness_path": readiness_path,
        "normalization": normalization,
        "resolved": resolved,
        "train_cache": train_cache,
        "validation_cache": validation_cache,
    }


def _forward_v18r(
    model: LiquidTaskClassifier, tensors: Mapping[str, Any]
) -> dict[str, Any]:
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
        micro_doppler_features=tensors.get("micro_doppler_features"),
        micro_doppler_valid=tensors.get("micro_doppler_valid"),
        rd_frames=tensors["rd_frames"],
        rd_frame_mask=tensors["rd_frame_mask"],
        rd_quality=tensors["rd_quality"],
        tr_padding_mask=tensors["tr_padding_mask"],
    )


def _build_cache(
    root: Path,
    *,
    config: Any,
    model: LiquidTaskClassifier,
    split_name: str,
    source: ResidualCacheDataset,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = build_task_dataset(root, config, split_name=split_name)
    if list(dataset.parent_ids) != source.parent_sample_ids.tolist():
        raise ValueError(f"V19r {split_name} parent order differs from V19 cache")
    loader = build_task_dataloader(
        dataset, batch_size=config.train.batch_size, shuffle=False
    )
    arrays = {name: np.asarray(value).copy() for name, value in source.arrays.items()}
    arrays["schema_version"] = np.asarray(V19R_CACHE_SCHEMA_VERSION)
    arrays["baseline_logits"] = np.zeros_like(source.baseline_logits, dtype=np.float32)
    median = torch.tensor(
        normalization["median"], dtype=torch.float32, device=device
    )
    scale = torch.tensor(normalization["scale"], dtype=torch.float32, device=device)
    cursor = 0
    maximum_causal_difference = 0.0
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            batch_size = len(batch["parent_sample_ids"])
            stop = cursor + batch_size
            width = int(tensors["tr_padding_mask"].shape[1])
            if (
                list(map(str, batch["parent_sample_ids"]))
                != source.parent_sample_ids[cursor:stop].tolist()
                or not np.array_equal(
                    batch["labels"].cpu().numpy(), source.labels[cursor:stop]
                )
                or not np.array_equal(
                    batch["lengths"].cpu().numpy(), source.lengths[cursor:stop]
                )
                or not np.array_equal(
                    tensors["tr_padding_mask"].cpu().numpy(),
                    source.padding_mask[cursor:stop, :width],
                )
                or not np.array_equal(
                    tensors["micro_doppler_valid"].cpu().numpy(),
                    source.micro_doppler_valid[cursor:stop, :width],
                )
            ):
                raise ValueError(f"V19r {split_name} source cache alignment differs")
            causal, counts = normalize_stable8(
                tensors["micro_doppler_features"],
                tensors["micro_doppler_valid"],
                tensors["tr_padding_mask"],
                median=median,
                scale=scale,
            )
            expected_causal = torch.from_numpy(
                source.causal_features[cursor:stop, :width]
            ).to(device)
            maximum_causal_difference = max(
                maximum_causal_difference,
                float((causal - expected_causal).abs().max().item()),
            )
            if (
                not torch.allclose(causal, expected_causal, rtol=0.0, atol=2.0e-6)
                or not np.array_equal(
                    counts.cpu().numpy(), source.valid_counts[cursor:stop, :width]
                )
            ):
                raise ValueError(f"V19r {split_name} causal stable8 summary differs")
            output = _forward_v18r(model, tensors)
            arrays["baseline_logits"][cursor:stop, :width] = (
                output["sequence_logits"].cpu().numpy().astype(np.float32)
            )
            cursor = stop
    if cursor != len(source):
        raise ValueError(f"V19r {split_name} inference did not cover every parent")
    final_counts = np.asarray(
        [source.valid_counts[index, source.lengths[index] - 1] for index in range(len(source))]
    )
    return arrays, {
        "split_name": split_name,
        "parent_count": len(source),
        "event_count": int(source.lengths.sum()),
        "valid_micro_doppler_event_count": int(source.micro_doppler_valid.sum()),
        "minimum_valid_observations_at_final": int(final_counts.min()),
        "maximum_valid_observations_at_final": int(final_counts.max()),
        "maximum_absolute_causal_feature_reproduction_difference": maximum_causal_difference,
    }


def _metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    _require_empty_formal_artifact(root)
    _progress("validating frozen V18r baseline and V19 stable8 residual inputs")
    v18r = _validate_v18r_source(root)
    v19 = _validate_v19_feature_source(root, v18r)

    model = LiquidTaskClassifier(v18r["config"]).cpu()
    checkpoint = load_task_checkpoint(
        v18r["checkpoint_path"],
        model,
        expected_schema=v18r["config"].train.checkpoint_schema,
        map_location="cpu",
    )
    if int(checkpoint.get("epoch", -1)) != 15:
        raise ValueError("V19r frozen V18r checkpoint epoch differs")
    del checkpoint
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    backbone_hash_before = tensor_state_sha256(model.state_dict())
    device = task_device(v18r["config"])
    model.to(device)
    guard_before = _model_state_guard(model)

    train_path = root / V19R_TRAIN_CACHE_RELATIVE_PATH
    validation_path = root / V19R_VALIDATION_CACHE_RELATIVE_PATH
    _progress(f"generating frozen V18r baseline logits on {device}")
    train_arrays, train_description = _build_cache(
        root,
        config=v18r["config"],
        model=model,
        split_name=v18r["config"].data.train_split_name,
        source=v19["train_cache"],
        normalization=v19["normalization"],
        device=device,
    )
    validation_arrays, validation_description = _build_cache(
        root,
        config=v18r["config"],
        model=model,
        split_name=v18r["config"].data.validation_split_name,
        source=v19["validation_cache"],
        normalization=v19["normalization"],
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    guard_after = _model_state_guard(model)
    backbone_hash_after = tensor_state_sha256(model.state_dict())
    if guard_before != guard_after or backbone_hash_before != backbone_hash_after:
        raise ValueError("V19r frozen V18r backbone changed during cache preparation")
    _write_npz_if_same(train_path, train_arrays)
    _write_npz_if_same(validation_path, validation_arrays)
    del train_arrays, validation_arrays, model

    train_dataset = V19rResidualCacheDataset(train_path)
    validation_dataset = V19rResidualCacheDataset(validation_path)
    if set(train_dataset.parent_sample_ids) & set(validation_dataset.parent_sample_ids):
        raise ValueError("V19r train/validation caches share a parent")
    _progress("reproducing frozen V18r parent-final validation metrics")
    baseline_validation = evaluate_residual(
        None,
        validation_dataset,
        device=torch.device("cpu"),
        batch_size=v18r["config"].train.batch_size,
        prefix_auxiliary_min_events=v18r["config"].train.prefix_auxiliary_min_events,
    )
    frozen_parent_metrics = dict(
        dict(v18r["summary"].get("best_validation_metrics") or {}).get(
            "parent_metrics"
        )
        or {}
    )
    if not _metrics_match(baseline_validation["parent_metrics"], frozen_parent_metrics):
        raise ValueError("V19r cached baseline metrics differ from frozen V18r best")
    residual_probe = MicroDopplerLateResidual().cpu()
    first = train_dataset[0]
    with torch.inference_mode():
        initial = residual_probe(
            first["baseline_logits"].unsqueeze(0),
            first["causal_features"].unsqueeze(0),
            first["padding_mask"].unsqueeze(0),
        )
    initial_equal = torch.equal(
        initial["sequence_logits"], first["baseline_logits"].unsqueeze(0)
    )
    if (
        not initial_equal
        or residual_probe.trainable_parameter_count != V19_TRAINABLE_PARAMETER_COUNT
        or residual_probe.linear.bias is not None
    ):
        raise ValueError("V19r zero initialization or 24-weight contract failed")

    normalization_path = v19["resolved"]["normalization_path"]
    contract = {
        "schema_version": V19R_SCHEMA_VERSION,
        "experiment_scope": "DBB specialist parent-final local task only",
        "single_factor": (
            "add a zero-initialized bounded zero-sum 24-weight stable8 late residual "
            "to the frozen V18r density+stable8 best logits"
        ),
        "class_names": list(V19_CLASS_NAMES),
        "feature_names": list(BROADBAND_MICRO_DOPPLER_NAMES),
        "feature_dim": V19_FEATURE_DIM,
        "v19_residual_weights_reused": False,
        "model_contract": {
            "baseline": "frozen V18r density+stable8 best logits",
            "linear_shape": [V19_CLASS_COUNT, V19_FEATURE_DIM],
            "bias": False,
            "zero_initialized": True,
            "zero_sum_across_classes": True,
            "absolute_residual_bound": V19_RESIDUAL_BOUND,
            "maximum_pairwise_logit_correction": 2.0 * V19_RESIDUAL_BOUND,
            "trainable_parameter_count": V19_TRAINABLE_PARAMETER_COUNT,
        },
        "source_v18r": {
            "config": str(v18r["config_path"]),
            "config_sha256": sha256_file(v18r["config_path"]),
            "paired_report": str(v18r["report_path"]),
            "paired_report_sha256": sha256_file(v18r["report_path"]),
            "training_summary": str(v18r["summary_path"]),
            "training_summary_sha256": sha256_file(v18r["summary_path"]),
            "checkpoint": str(v18r["checkpoint_path"]),
            "checkpoint_sha256": sha256_file(v18r["checkpoint_path"]),
            "checkpoint_epoch": 15,
            "backbone_state_sha256": backbone_hash_before,
            "data_contract_hashes": v18r["contract"]["hashes"],
        },
        "source_v19_feature_contract": {
            "config": str(v19["config_path"]),
            "config_sha256": sha256_file(v19["config_path"]),
            "readiness": str(v19["readiness_path"]),
            "readiness_sha256": sha256_file(v19["readiness_path"]),
            "normalization": str(normalization_path),
            "normalization_sha256": sha256_file(normalization_path),
            "train_cache_sha256": sha256_file(v19["resolved"]["train_cache_path"]),
            "validation_cache_sha256": sha256_file(
                v19["resolved"]["validation_cache_path"]
            ),
            "reused_fields": (
                "parent metadata, masks, valid counts and train-only normalized causal stable8; "
                "old V19 baseline logits and residual weights are not reused"
            ),
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
        "frozen_v18r_parent_order_sha256_by_epoch": v18r["order_hashes"],
        "baseline_validation_parent_metrics": baseline_validation["parent_metrics"],
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / V19R_CONTRACT_RELATIVE_PATH
    _write_text_if_same(contract_path, _json_text(contract))

    artifact = root / V19R_ARTIFACT_RELATIVE_PATH
    config = {
        "schema_version": V19R_SCHEMA_VERSION,
        "config_name": V19R_CONFIG_NAME,
        "class_names": list(V19_CLASS_NAMES),
        "contract_path": str(contract_path),
        "expected_contract_sha256": sha256_file(contract_path),
        "readiness_path": str(root / V19R_READINESS_RELATIVE_PATH),
        "normalization_path": str(normalization_path),
        "expected_normalization_sha256": sha256_file(normalization_path),
        "train_cache_path": str(train_path),
        "expected_train_cache_sha256": sha256_file(train_path),
        "validation_cache_path": str(validation_path),
        "expected_validation_cache_sha256": sha256_file(validation_path),
        "baseline_checkpoint_path": str(v18r["checkpoint_path"]),
        "expected_baseline_checkpoint_sha256": sha256_file(v18r["checkpoint_path"]),
        "expected_baseline_backbone_state_sha256": backbone_hash_before,
        "baseline_training_summary_path": str(v18r["summary_path"]),
        "expected_baseline_training_summary_sha256": sha256_file(v18r["summary_path"]),
        "v18r_paired_report_path": str(v18r["report_path"]),
        "expected_v18r_paired_report_sha256": sha256_file(v18r["report_path"]),
        "source_v19_config_path": str(v19["config_path"]),
        "expected_source_v19_config_sha256": sha256_file(v19["config_path"]),
        "source_v19_readiness_path": str(v19["readiness_path"]),
        "expected_source_v19_readiness_sha256": sha256_file(v19["readiness_path"]),
        "model": contract["model_contract"],
        "train": {
            "seed": 42,
            "dataloader_seed": 42,
            "batch_size": int(v18r["config"].train.batch_size),
            "learning_rate": float(v18r["config"].train.learning_rate),
            "optimizer": "AdamW_default_weight_decay",
            "epochs": 20,
            "early_stopping_patience": 0,
            "gradient_clip_norm": float(v18r["config"].train.gradient_clip_norm),
            "class_weights": list(v18r["config"].train.class_weights),
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": float(
                v18r["config"].train.prefix_auxiliary_weight
            ),
            "prefix_auxiliary_min_events": int(
                v18r["config"].train.prefix_auxiliary_min_events
            ),
            "checkpoint_selection_metric": BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
            "save_every_epoch": True,
            "record_parent_order_hash": True,
            "reference_parent_order_sha256_by_epoch": v18r["order_hashes"],
            "device": str(v18r["config"].train.device),
            "checkpoint_schema": V19R_CHECKPOINT_SCHEMA_VERSION,
            "artifact_dir": str(artifact),
            "checkpoint_path": str(artifact / "best_model.pt"),
            "last_checkpoint_path": str(artifact / "last_model.pt"),
            "epoch_checkpoint_dir": str(artifact / "epochs"),
        },
        "adoption": {
            "minimum_per_class_precision_recall_f1": 0.95,
            "strict_macro_f1_improvement_over_frozen_v18r": 0.02,
            "strict_drone_bird_weak_floor_improvement_over_frozen_v18r": 0.02,
            "drone_to_bird_must_not_increase": True,
            "bird_to_drone_must_not_increase": True,
            "minimum_bidirectional_error_reduction": 1,
            "balloon_floor_must_not_decrease": True,
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
    config_path = root / V19R_CONFIG_RELATIVE_PATH
    _write_text_if_same(config_path, _json_text(config))
    checks = {
        "v18r_baseline_is_completed_and_hash_bound": True,
        "v18r_checkpoint_epoch_is_15": True,
        "v18r_backbone_unchanged_during_prepare": backbone_hash_before
        == backbone_hash_after,
        "v19_normalization_is_train_parent_only": (
            v19["normalization"]["validation_used_for_fit"] is False
            and v19["normalization"]["test_used_for_fit"] is False
        ),
        "v19_residual_weights_are_not_reused": True,
        "train_validation_parent_sets_are_disjoint": not bool(
            set(train_dataset.parent_sample_ids)
            & set(validation_dataset.parent_sample_ids)
        ),
        "cached_baseline_reproduces_frozen_v18r_parent_metrics": True,
        "zero_initialized_combined_logits_exactly_equal_frozen_v18r": initial_equal,
        "only_24_bias_free_residual_weights_are_trainable": (
            residual_probe.trainable_parameter_count == V19_TRAINABLE_PARAMETER_COUNT
            and residual_probe.linear.bias is None
        ),
        "test_payload_access_count_is_zero": True,
        "validation_is_excluded_from_readiness_gate": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": (
            "RADAR-LN3-V19r-v18r-micro-doppler-late-residual-readiness-v1"
        ),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "checks": checks,
        "validation_used_for_readiness_gate": False,
        "train_parent_count": len(train_dataset),
        "validation_parent_count": len(validation_dataset),
        "trainable_parameter_count": residual_probe.trainable_parameter_count,
        "baseline_checkpoint_epoch": 15,
        "baseline_checkpoint_sha256": sha256_file(v18r["checkpoint_path"]),
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
        raise ValueError(f"V19r readiness checks failed: {failed}")
    readiness_path = root / V19R_READINESS_RELATIVE_PATH
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
        "trainable_parameter_count": residual_probe.trainable_parameter_count,
        "baseline_checkpoint_epoch": 15,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V19r frozen-V18r micro-Doppler late-residual experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
