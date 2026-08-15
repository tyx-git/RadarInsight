from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.data import DT_REASON_ZERO_DELTA
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
    log_low_confidence_records,
    log_metric_set,
)
from radar_ln3.task_models import (
    HIERARCHICAL_SCHEMA_VERSIONS,
    DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION,
    FORMAL_PAIRED_SCHEMA_VERSIONS,
    HIERARCHICAL_CASCADE_SCHEMA_VERSION,
    LiquidTaskClassifier,
    PRODUCT_LOCAL_SCHEMA_VERSION,
    RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION,
    MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
    LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
    V20_DENSITY_PAIRED_PROTOCOL,
    V20_SKIP_PAIRED_PROTOCOL,
    build_paired_model_for_config,
    capture_training_state,
    build_task_dataloader,
    build_task_dataset,
    evaluate_task,
    save_task_checkpoint,
    seed_everything,
    task_device,
    task_training_objective,
    verify_task_data_contract,
    load_task_config,
    load_task_checkpoint,
    is_product_local_schema,
    is_micro_doppler_paired_schema,
    is_liquid_dynamics_paired_schema,
    restore_training_state,
    summarize_task_performance_guardrails,
)


V17_READINESS_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V17-target-anchored-v3-micro-doppler-train-readiness.json"
)
V17_READINESS_SCHEMA = "RADAR-LN3-V17-micro-doppler-training-readiness-v4"
V18_READINESS_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-train-readiness.json"
)
V18_READINESS_SCHEMA = (
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler-"
    "training-readiness-v3-stable8"
)
V20_READINESS_REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V20-zero-delta-skip-training-readiness.json"
)
V20_READINESS_SCHEMA = "RADAR-LN3-V20-zero-delta-skip-training-readiness-v1"
V20_DENSITY_READINESS_REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V20-density-weighted-training-readiness.json"
)
V20_DENSITY_READINESS_SCHEMA = (
    "RADAR-LN3-V20-density-weighted-training-readiness-v1"
)
V20_DENSITY_MULTISEED_READINESS_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V20-density-weighted-multiseed-training-readiness.json"
)
V20_DENSITY_MULTISEED_READINESS_SCHEMA = (
    "RADAR-LN3-V20-density-weighted-multiseed-training-readiness-v1"
)
V20_DENSITY_MULTISEED_NEW_SEEDS = frozenset({43, 44})
V20_READINESS_CONTRACTS = {
    V20_SKIP_PAIRED_PROTOCOL: (V20_READINESS_REPORT, V20_READINESS_SCHEMA),
    V20_DENSITY_PAIRED_PROTOCOL: (
        V20_DENSITY_READINESS_REPORT,
        V20_DENSITY_READINESS_SCHEMA,
    ),
}


def _strip_records(result: dict[str, Any]) -> dict[str, Any]:
    value = dict(result)
    value.pop("records", None)
    return value


def verify_v17_training_readiness(
    project_root: Path,
    config_path: Path,
    contract_hashes: dict[str, Any],
    *,
    schema_version: str = MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
) -> dict[str, Any]:
    if schema_version == BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION:
        relative_report = V18_READINESS_REPORT
        expected_schema = V18_READINESS_SCHEMA
    elif schema_version == MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION:
        relative_report = V17_READINESS_REPORT
        expected_schema = V17_READINESS_SCHEMA
    else:
        raise ValueError("micro-Doppler readiness is restricted to V17/V18 paired schemas")
    report_path = (project_root / relative_report).resolve()
    if not report_path.is_file():
        raise PermissionError(
            "V17 formal training requires the train-only readiness report; run the V17 "
            f"prepare stage first: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_config = config_path.resolve()
    paired_configs = dict(report.get("paired_configs") or {})
    matching_configs = [
        dict(value)
        for value in paired_configs.values()
        if Path(str(dict(value).get("path", ""))).resolve() == expected_config
    ]
    config_bytes_hash = hashlib.sha256(expected_config.read_bytes()).hexdigest()
    checks = dict(report.get("checks") or {})
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != expected_schema
        or len(matching_configs) != 1
        or matching_configs[0].get("sha256") != config_bytes_hash
        or report.get("contract_hashes") != contract_hashes
        or not checks
        or not all(value is True for value in checks.values())
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("test") != "未执行"
        or report.get("formal_training_at_report_generation") != "未执行"
    ):
        raise PermissionError(
            "V17 formal training readiness is missing, stale, failed or outside the frozen "
            f"train-only/test-sealed contract: {report_path}"
        )
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return {
        "path": str(report_path),
        "sha256": digest,
        "schema_version": expected_schema,
        "status": "passed",
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }


def verify_v20_training_readiness(
    project_root: Path,
    config_path: Path,
    contract_hashes: dict[str, Any],
    *,
    paired_protocol: str,
    paired_seed: int,
) -> dict[str, Any]:
    if (
        paired_protocol == V20_DENSITY_PAIRED_PROTOCOL
        and int(paired_seed) in V20_DENSITY_MULTISEED_NEW_SEEDS
    ):
        relative_report = V20_DENSITY_MULTISEED_READINESS_REPORT
        expected_schema = V20_DENSITY_MULTISEED_READINESS_SCHEMA
    else:
        try:
            relative_report, expected_schema = V20_READINESS_CONTRACTS[
                str(paired_protocol)
            ]
        except KeyError as error:
            raise ValueError(
                "V20 training readiness requires a recognized paired protocol"
            ) from error
    report_path = (project_root / relative_report).resolve()
    if not report_path.is_file():
        raise PermissionError(
            "V20 formal training requires the train-only readiness report; run the V20 "
            f"prepare stage first: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_config = config_path.resolve()
    paired_configs = dict(report.get("paired_configs") or {})
    matching_configs = [
        dict(value)
        for value in paired_configs.values()
        if Path(str(dict(value).get("path", ""))).resolve() == expected_config
    ]
    checks = dict(report.get("checks") or {})
    config_hash = hashlib.sha256(expected_config.read_bytes()).hexdigest()
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != expected_schema
        or report.get("paired_protocol") != paired_protocol
        or (
            expected_schema == V20_DENSITY_MULTISEED_READINESS_SCHEMA
            and int(paired_seed) not in set(report.get("new_training_seeds") or [])
        )
        or len(matching_configs) != 1
        or matching_configs[0].get("sha256") != config_hash
        or report.get("contract_hashes") != contract_hashes
        or not checks
        or not all(value is True for value in checks.values())
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("test") != "未执行"
        or report.get("formal_training_at_report_generation") != "未执行"
    ):
        raise PermissionError(
            "V20 formal training readiness is missing, stale, failed or outside the frozen "
            f"train-only/test-sealed contract: {report_path}"
        )
    return {
        "path": str(report_path),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "schema_version": expected_schema,
        "paired_protocol": paired_protocol,
        "status": "passed",
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }


def parent_order_sha256(parent_sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for parent_id in parent_sample_ids:
        encoded = str(parent_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _finite_metric(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"checkpoint selection metric {name!r} is missing")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"checkpoint selection metric {name!r} is not finite")
    return result


def build_checkpoint_selection(validation: dict[str, Any], metric_name: str) -> dict[str, Any]:
    """Build an auditable, deterministic checkpoint ranking contract.

    ``parent_min_precision_recall`` ranks first by the weakest per-class
    precision/recall value and uses parent Macro-F1 only as a tie-breaker.
    A class with no validation support is rejected because its recall cannot
    establish the requested weak-class floor.
    """

    metric_name = str(metric_name)
    if metric_name == BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION:
        metrics = dict(validation.get("parent_metrics") or {})
        per_class = list(metrics.get("per_class") or [])
        class_rows = {str(row.get("class")): row for row in per_class}
        expected = {"DroneTarget", "BirdTarget", "BalloonTarget"}
        if set(class_rows) != expected:
            raise ValueError(
                "broadband micro-Doppler checkpoint selection requires all three DBB classes"
            )
        class_values: list[float] = []
        for class_name in sorted(expected):
            row = class_rows[class_name]
            if int(row.get("support", 0)) <= 0:
                raise ValueError(
                    "broadband micro-Doppler checkpoint selection has a zero-support class"
                )
            for field in ("precision", "recall", "f1"):
                class_values.append(
                    _finite_metric(row.get(field), name=f"{class_name}.{field}")
                )
        matrix = np.asarray(
            metrics.get("confusion_matrix_rows_true_columns_pred"), dtype=np.int64
        )
        if matrix.shape != (3, 3):
            raise ValueError(
                "broadband micro-Doppler checkpoint selection requires a 3x3 parent matrix"
            )
        expected_class_order = ("DroneTarget", "BirdTarget", "BalloonTarget")
        if tuple(str(row.get("class")) for row in per_class) != expected_class_order:
            raise ValueError(
                "broadband micro-Doppler confusion matrix requires class order "
                f"{expected_class_order}"
            )
        bidirectional_errors = int(matrix[0, 1] + matrix[1, 0])
        weak_class_floor = min(class_values)
        macro_f1 = _finite_metric(metrics.get("macro_f1"), name="parent_macro_f1")
        return {
            "metric": metric_name,
            "primary_metric": "parent_min_class_precision_recall_f1",
            "primary_value": weak_class_floor,
            "tie_break_metric": "negative_drone_bird_bidirectional_parent_errors_then_macro_f1",
            "tie_break_value": -float(bidirectional_errors),
            "drone_bird_bidirectional_parent_errors": bidirectional_errors,
            "macro_f1": macro_f1,
            "ranking_key": [weak_class_floor, -float(bidirectional_errors), macro_f1],
        }
    if metric_name == "parent_min_precision_recall":
        metrics = dict(validation.get("parent_metrics") or {})
        per_class = list(metrics.get("per_class") or [])
        if not per_class:
            raise ValueError("parent_min_precision_recall requires per-class parent metrics")
        zero_support = [str(row.get("class")) for row in per_class if int(row.get("support", 0)) <= 0]
        if zero_support:
            raise ValueError(
                "parent_min_precision_recall cannot rank a validation set with zero-support classes: "
                + ",".join(zero_support)
            )
        class_values: list[float] = []
        for row in per_class:
            class_name = str(row.get("class"))
            class_values.append(_finite_metric(row.get("precision"), name=f"{class_name}.precision"))
            class_values.append(_finite_metric(row.get("recall"), name=f"{class_name}.recall"))
        weak_class_floor = min(class_values)
        macro_f1 = _finite_metric(metrics.get("macro_f1"), name="parent_macro_f1")
        return {
            "metric": metric_name,
            "primary_metric": "parent_min_class_precision_recall",
            "primary_value": weak_class_floor,
            "tie_break_metric": "parent_macro_f1",
            "tie_break_value": macro_f1,
            "ranking_key": [weak_class_floor, macro_f1],
        }
    if metric_name.startswith("parent_"):
        scope_name = "parent_metrics"
        value_name = metric_name.removeprefix("parent_")
    elif metric_name.startswith("window_"):
        scope_name = "window_metrics"
        value_name = metric_name.removeprefix("window_")
    else:
        raise ValueError(
            "task checkpoint_selection_metric must start with parent_ or window_; "
            f"received={metric_name!r}"
        )
    metrics = dict(validation.get(scope_name) or {})
    value = _finite_metric(metrics.get(value_name), name=metric_name)
    return {
        "metric": metric_name,
        "primary_metric": metric_name,
        "primary_value": value,
        "tie_break_metric": None,
        "tie_break_value": None,
        "ranking_key": [value],
    }


def unavailable_checkpoint_selection(metric_name: str) -> dict[str, Any]:
    return {
        "metric": str(metric_name),
        "primary_metric": str(metric_name),
        "primary_value": None,
        "tie_break_metric": None,
        "tie_break_value": None,
        "ranking_key": [None],
    }


def checkpoint_selection_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.get("metric") != incumbent.get("metric"):
        raise ValueError("cannot compare checkpoint selections with different metrics")

    def ranking_key(selection: dict[str, Any]) -> tuple[float, ...]:
        values = list(selection.get("ranking_key") or [])
        if not values:
            raise ValueError("checkpoint selection has an empty ranking key")
        return tuple(float("-inf") if value is None else float(value) for value in values)

    return ranking_key(candidate) > ranking_key(incumbent)


def _write_sample_details(
    artifact_dir: Path,
    result: dict[str, Any],
    *,
    checkpoint: Path,
    epoch: int,
    schema_version: str,
) -> Path:
    payload = {
        "schema_version": f"{schema_version}-task-sample-detail-v1",
        "task_name": result.get("task_name"),
        "checkpoint": str(checkpoint),
        "epoch": int(epoch),
        "counts": {key: len(value) for key, value in result.get("records", {}).items()},
        "records": result.get("records", {}),
    }
    path = artifact_dir / "sample_detail_information.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def train_task(
    project_root: Path,
    config_path: Path,
    *,
    max_steps: int | None = None,
    epochs_override: int | None = None,
    skip_validation: bool = False,
    resume_checkpoint: Path | None = None,
    allow_bounded_smoke: bool = False,
) -> dict[str, Any]:
    config = load_task_config(config_path)
    if config.schema_version in HIERARCHICAL_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"{config.schema_version} is a joint three-checkpoint product objective; "
            "use python -m train.cascade with all three matching configs"
        )
    if not config.train.allow_parameter_updates:
        raise PermissionError("task configuration forbids parameter updates")
    if config.schema_version in FORMAL_PAIRED_SCHEMA_VERSIONS:
        if max_steps is not None or epochs_override is not None or skip_validation:
            raise ValueError(
                "formal paired runs forbid max-step/epoch overrides and validation skipping; "
                "use the isolated src-v3/smoke entry for bounded verification"
            )
    if (
        is_product_local_schema(config.schema_version)
        and (max_steps is not None or epochs_override is not None or skip_validation)
        and not allow_bounded_smoke
    ):
        raise ValueError(
            "V16 formal local-task runs forbid max-step/epoch overrides and validation skipping; "
            "use the isolated src-v3/smoke entry for bounded verification"
        )
    seed_everything(config.train.seed)
    contract = verify_task_data_contract(project_root, config)
    v17_readiness: dict[str, Any] | None = None
    readiness_field_name = (
        "v20_training_readiness"
        if is_liquid_dynamics_paired_schema(config.schema_version)
        else (
            "v18_training_readiness"
            if config.schema_version == BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
            else "v17_training_readiness"
        )
    )
    readiness_hash_field_name = readiness_field_name + "_sha256"
    if is_micro_doppler_paired_schema(config.schema_version):
        v17_readiness = verify_v17_training_readiness(
            project_root,
            config_path,
            dict(contract["hashes"]),
            schema_version=config.schema_version,
        )
    elif is_liquid_dynamics_paired_schema(config.schema_version):
        v17_readiness = verify_v20_training_readiness(
            project_root,
            config_path,
            dict(contract["hashes"]),
            paired_protocol=config.train.paired_protocol,
            paired_seed=config.train.seed,
        )
    artifact_dir = project_root / config.train.formal_artifact_dir
    if config.schema_version not in FORMAL_PAIRED_SCHEMA_VERSIONS and resume_checkpoint is not None:
        raise ValueError("explicit epoch-boundary resume is restricted to a formal paired protocol")
    if config.schema_version in FORMAL_PAIRED_SCHEMA_VERSIONS and artifact_dir.exists() and resume_checkpoint is None:
        existing = list(artifact_dir.iterdir())
        if existing:
            raise FileExistsError(
                "refusing to overwrite an existing formal paired artifact directory; "
                f"move/archive it or use a new versioned config: {artifact_dir}"
            )
    if (
        is_product_local_schema(config.schema_version)
        and config.schema_version not in FORMAL_PAIRED_SCHEMA_VERSIONS
        and artifact_dir.exists()
    ):
        existing = list(artifact_dir.iterdir())
        if existing:
            raise FileExistsError(
                "refusing to overwrite an existing V16 product-local artifact directory; "
                f"move/archive it or create a new versioned config: {artifact_dir}"
            )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact_dir / "training.log")
    started_run = time.perf_counter()
    try:
        train_dataset = build_task_dataset(project_root, config, split_name=config.data.train_split_name)
        val_dataset = build_task_dataset(project_root, config, split_name=config.data.validation_split_name)
        loader_generator = torch.Generator(device="cpu")
        loader_seed = config.train.dataloader_seed if config.train.dataloader_seed is not None else config.train.seed
        loader_generator.manual_seed(int(loader_seed))
        train_loader = build_task_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        device = task_device(config)
        model, paired_counterpart, paired_initialization = build_paired_model_for_config(project_root, config)
        if config.schema_version in FORMAL_PAIRED_SCHEMA_VERSIONS:
            if paired_counterpart is None:
                raise RuntimeError("formal paired counterpart model was not built")
            initial_loader = build_task_dataloader(
                train_dataset,
                batch_size=min(config.train.batch_size, len(train_dataset)),
                shuffle=False,
            )
            initial_batch = next(iter(initial_loader))
            initial_tensors = {
                key: value if isinstance(value, torch.Tensor) else value
                for key, value in initial_batch.items()
            }
            shared_inputs = {
                "tr_features": initial_tensors["tr_features"],
                "tr_missing_mask": initial_tensors["tr_missing_mask"],
                "tr_dt_seconds": initial_tensors["tr_dt_seconds"],
                "tr_dt_valid": initial_tensors["tr_dt_valid"],
                "tr_dt_reason": initial_tensors.get("tr_dt_reason"),
                "tr_motion_features": initial_tensors.get("tr_motion_features"),
                "tr_motion_valid_mask": initial_tensors.get("tr_motion_valid_mask"),
                "rd_frames": initial_tensors["rd_frames"],
                "rd_frame_mask": initial_tensors["rd_frame_mask"],
                "rd_quality": initial_tensors["rd_quality"],
                "tr_padding_mask": initial_tensors["tr_padding_mask"],
            }
            if is_micro_doppler_paired_schema(config.schema_version):
                shared_inputs.update(
                    {
                        "rd_physical_features": initial_tensors["rd_physical_features"],
                        "rd_physical_valid": initial_tensors["rd_physical_valid"],
                    }
                )
            active = ~initial_tensors["tr_padding_mask"]
            if config.train.paired_arm == "control":
                control_model, treatment_model = model, paired_counterpart
            else:
                control_model, treatment_model = paired_counterpart, model
            control_model.eval()
            treatment_model.eval()
            with torch.inference_mode():
                control_initial = control_model(**shared_inputs)["sequence_logits"]
                if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
                    physical_probe = torch.zeros(
                        (*active.shape, 4),
                        dtype=initial_tensors["tr_features"].dtype,
                        device=torch.device("cpu"),
                    )
                    physical_probe[active] = 0.5
                    treatment_initial = treatment_model(
                        **shared_inputs,
                        rd_physical_features=physical_probe,
                        rd_physical_valid=active,
                    )["sequence_logits"]
                    initial_probe = "0.5 on every non-padding RD physical event; valid=true; padding zero/invalid"
                elif is_micro_doppler_paired_schema(config.schema_version):
                    treatment_initial = treatment_model(
                        **shared_inputs,
                        micro_doppler_features=initial_tensors["micro_doppler_features"],
                        micro_doppler_valid=initial_tensors["micro_doppler_valid"],
                    )["sequence_logits"]
                    initial_probe = (
                        "same real DBB batch and shared RD-physical4 input; Treatment additionally "
                        f"receives real current-event micro-Doppler{config.model.micro_doppler_dim} "
                        "through exactly-zero Fusion columns"
                    )
                elif is_liquid_dynamics_paired_schema(config.schema_version):
                    treatment_initial = treatment_model(**shared_inputs)["sequence_logits"]
                    initial_probe = (
                        "identical V20 parameters/input; logits must match before the first "
                        "zero-delta Treatment exposure and may differ from that event onward"
                    )
                else:
                    treatment_output = treatment_model(**shared_inputs, return_dbb_auxiliary_logits=True)
                    treatment_initial = treatment_output["sequence_logits"]
                    control_auxiliary = control_model(
                        **shared_inputs,
                        return_dbb_auxiliary_logits=True,
                    )["dbb_auxiliary_sequence_logits"]
                    treatment_auxiliary = treatment_output["dbb_auxiliary_sequence_logits"]
                    if not torch.equal(control_auxiliary, treatment_auxiliary):
                        raise AssertionError("V12 paired auxiliary logits are not exactly equal before training")
                    paired_initialization["initial_auxiliary_logits_exactly_equal"] = True
                    initial_probe = "identical V12 model/input; only the configured training loss weight differs"
            difference = (control_initial - treatment_initial).abs()
            maximum_difference = float(difference.max().item())
            full_logits_equal = bool(torch.equal(control_initial, treatment_initial))
            pre_exposure_logits_equal = full_logits_equal
            pre_exposure_maximum_difference = maximum_difference
            treatment_exposed_event_count = 0
            if is_liquid_dynamics_paired_schema(config.schema_version):
                reasons = initial_tensors.get("tr_dt_reason")
                if reasons is None:
                    raise RuntimeError("V20 paired initialization requires dt reasons")
                exposure_seen = torch.cumsum(
                    (reasons == DT_REASON_ZERO_DELTA).to(dtype=torch.int64), dim=1
                ).gt(0)
                pre_exposure = active & (~exposure_seen)
                treatment_exposed_event_count = int((active & exposure_seen).sum().item())
                if not bool(pre_exposure.any()):
                    raise AssertionError("V20 initialization probe has no pre-exposure event")
                pre_exposure_logits_equal = bool(
                    torch.equal(
                        control_initial[pre_exposure], treatment_initial[pre_exposure]
                    )
                )
                pre_exposure_maximum_difference = float(
                    difference[pre_exposure].max().item()
                )
                if not pre_exposure_logits_equal:
                    raise AssertionError(
                        "V20 paired logits differ before the first zero-delta Treatment "
                        f"exposure: maximum_absolute_difference={pre_exposure_maximum_difference}"
                    )
            elif not full_logits_equal:
                raise AssertionError(
                    "paired CPU reference logits are not exactly equal before training: "
                    f"maximum_absolute_difference={maximum_difference}"
                )
            logits_bytes = control_initial.detach().cpu().contiguous().numpy().tobytes(order="C")
            paired_initialization.update(
                {
                    "initial_cpu_reference_logits_exactly_equal": full_logits_equal,
                    "initial_logits_maximum_absolute_difference": maximum_difference,
                    "initial_pre_treatment_exposure_logits_exactly_equal": (
                        pre_exposure_logits_equal
                    ),
                    "initial_pre_treatment_exposure_maximum_absolute_difference": (
                        pre_exposure_maximum_difference
                    ),
                    "initial_treatment_exposed_event_count": treatment_exposed_event_count,
                    "initial_logits_sha256": hashlib.sha256(logits_bytes).hexdigest(),
                    "initial_logit_probe": initial_probe,
                    "initial_logit_parent_sample_ids": list(initial_batch["parent_sample_ids"]),
                }
            )
            del paired_counterpart, control_model, treatment_model, control_initial, treatment_initial
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        epochs = max(1, int(epochs_override if epochs_override is not None else config.train.epochs))
        global_step = 0
        best_selection: dict[str, Any] | None = None
        best_epoch = -1
        best_validation: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        start_epoch = 1
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint if resume_checkpoint.is_absolute() else project_root / resume_checkpoint
            resume_path = resume_path.resolve()
            try:
                resume_path.relative_to(artifact_dir.resolve())
            except ValueError as exc:
                raise ValueError("paired resume checkpoint must be inside this arm's formal artifact directory") from exc
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)
            resume_payload = load_task_checkpoint(
                resume_path,
                model,
                expected_schema=config.train.checkpoint_schema,
                map_location=device,
                optimizer=optimizer,
            )
            if (
                (
                    is_micro_doppler_paired_schema(config.schema_version)
                    or is_liquid_dynamics_paired_schema(config.schema_version)
                )
                and (
                    v17_readiness is None
                    or dict(resume_payload.get("metrics") or {}).get(
                        readiness_hash_field_name
                    )
                    != v17_readiness["sha256"]
                )
            ):
                raise ValueError(
                    f"{config.schema_version} resume checkpoint is not bound to the current "
                    "passed readiness report"
                )
            run_state = restore_training_state(resume_payload, loader_generator)
            start_epoch = int(run_state["next_epoch"])
            global_step = int(run_state["global_step"])
            best_epoch = int(run_state["best_epoch"])
            best_selection = run_state.get("best_selection")
            history = list(run_state.get("history") or [])
            if start_epoch != int(resume_payload["epoch"]) + 1:
                raise ValueError("paired resume next_epoch is inconsistent with checkpoint epoch")
            if len(history) != int(resume_payload["epoch"]):
                raise ValueError("paired resume history length is inconsistent with checkpoint epoch")
            if not 1 <= start_epoch <= epochs + 1:
                raise ValueError(
                    f"paired resume next_epoch={start_epoch} is outside requested epochs={epochs}; "
                    "epochs+1 is allowed only to finalize evaluation/summary after a complete last epoch"
                )
            resumed_from = str(resume_path)
        steps_per_epoch = len(train_loader)
        remaining_epoch_steps = max(0, epochs - start_epoch + 1) * steps_per_epoch
        planned_steps = remaining_epoch_steps if max_steps is None else min(remaining_epoch_steps, max_steps)
        logger.info(
            "run=start | task=%s | config=%s | device=%s | train_parents=%d | val_parents=%d | batch=%d | epochs=%d | planned_steps=%d | pretraining=not_required",
            config.task.name,
            config.config_name,
            device,
            len(train_dataset),
            len(val_dataset),
            config.train.batch_size,
            epochs,
            planned_steps,
        )
        if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
            logger.info(
                "paired | protocol=%s | arm=%s | seed=%d | loader_seed=%d | shared_exact=%s | new_columns_zero=%s | cuda_bitwise_determinism=false",
                config.train.paired_protocol,
                config.train.paired_arm,
                config.train.seed,
                loader_seed,
                paired_initialization.get("shared_parameters_and_buffers_exactly_equal"),
                paired_initialization.get("treatment_new_fusion_columns_exactly_zero"),
            )
        elif config.schema_version == DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION:
            logger.info(
                "paired | protocol=%s | arm=%s | seed=%d | loader_seed=%d | shared_exact=%s | dbb_aux_weight=%.3f | supervision=final_state_only | cuda_bitwise_determinism=false",
                config.train.paired_protocol,
                config.train.paired_arm,
                config.train.seed,
                loader_seed,
                paired_initialization.get("shared_parameters_and_buffers_exactly_equal"),
                config.train.dbb_auxiliary_loss_weight,
            )
        elif is_micro_doppler_paired_schema(config.schema_version):
            logger.info(
                "paired | protocol=%s | arm=%s | seed=%d | loader_seed=%d | "
                "shared_rd_physical_dim=4 | micro_doppler_dim=%d | shared_exact=%s | "
                "new_columns_zero=%s | full_evidence_tensor_model_input=false | "
                "cuda_bitwise_determinism=false",
                config.train.paired_protocol,
                config.train.paired_arm,
                config.train.seed,
                loader_seed,
                config.model.micro_doppler_dim,
                paired_initialization.get("shared_parameters_and_buffers_exactly_equal"),
                paired_initialization.get("treatment_new_fusion_columns_exactly_zero"),
            )
        elif is_liquid_dynamics_paired_schema(config.schema_version):
            logger.info(
                "paired | protocol=%s | arm=%s | seed=%d | loader_seed=%d | "
                "control_mode=bounded_observation | treatment_mode=%s | shared_exact=%s | "
                "cuda_bitwise_determinism=false",
                config.train.paired_protocol,
                config.train.paired_arm,
                config.train.seed,
                loader_seed,
                "skip"
                if config.train.paired_protocol == V20_SKIP_PAIRED_PROTOCOL
                else "density_weighted",
                paired_initialization.get("shared_parameters_and_buffers_exactly_equal"),
            )
        logger.info(
            "contract | index_sha256=%s | contract_sha256=%s | stats_sha256=%s | classes=%s | loss=%s",
            contract["hashes"]["index_sha256"],
            contract["hashes"]["contract_sha256"],
            contract["hashes"]["stats_sha256"],
            ",".join(config.task.class_names),
            config.train.loss_mode,
        )
        best_checkpoint = project_root / config.train.checkpoint_path
        last_checkpoint = project_root / config.train.last_checkpoint_path
        stop_reason = "epoch_limit"
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_main_loss = 0.0
            epoch_auxiliary_loss = 0.0
            epoch_prefix_auxiliary_loss = 0.0
            epoch_prefix_auxiliary_events = 0
            epoch_prefix_auxiliary_parents = 0
            epoch_auxiliary_dbb_parents = 0
            epoch_auxiliary_cu_parents = 0
            epoch_auxiliary_final_states = 0
            epoch_steps = 0
            epoch_parent_order: list[str] = []
            epoch_started = time.perf_counter()
            for batch in train_loader:
                epoch_parent_order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(
                    tr_features=tensors["tr_features"],
                    tr_missing_mask=tensors["tr_missing_mask"],
                    tr_dt_seconds=tensors["tr_dt_seconds"],
                    tr_dt_valid=tensors["tr_dt_valid"],
                    tr_dt_reason=tensors.get("tr_dt_reason"),
                    tr_motion_features=tensors.get("tr_motion_features"),
                    tr_motion_valid_mask=tensors.get("tr_motion_valid_mask"),
                    rd_physical_features=tensors.get("rd_physical_features"),
                    rd_physical_valid=tensors.get("rd_physical_valid"),
                    micro_doppler_features=(
                        tensors.get("micro_doppler_features")
                        if model.micro_doppler_dim
                        else None
                    ),
                    micro_doppler_valid=(
                        tensors.get("micro_doppler_valid")
                        if model.micro_doppler_dim
                        else None
                    ),
                    rd_frames=tensors["rd_frames"],
                    rd_frame_mask=tensors["rd_frame_mask"],
                    rd_quality=tensors["rd_quality"],
                    tr_padding_mask=tensors["tr_padding_mask"],
                    return_dbb_auxiliary_logits=(
                        config.schema_version == DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION
                    ),
                )
                objective = task_training_objective(output, tensors, config, class_weights=weights)
                loss = objective["loss"]
                main_loss = objective["main_loss"]
                prefix_auxiliary_loss = objective["prefix_auxiliary_loss"]
                prefix_auxiliary_counts = objective["prefix_auxiliary_counts"]
                auxiliary_loss = objective["dbb_auxiliary_loss"]
                auxiliary_counts = objective["dbb_auxiliary_counts"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
                optimizer.step()
                global_step += 1
                epoch_steps += 1
                epoch_loss += float(loss.item())
                epoch_main_loss += float(main_loss.item())
                epoch_prefix_auxiliary_loss += float(prefix_auxiliary_loss.item())
                epoch_prefix_auxiliary_events += int(prefix_auxiliary_counts["eligible_prefix_count"])
                epoch_prefix_auxiliary_parents += int(prefix_auxiliary_counts["eligible_parent_count"])
                epoch_auxiliary_loss += float(auxiliary_loss.item())
                epoch_auxiliary_dbb_parents += auxiliary_counts["selected_dbb_parent_count"]
                epoch_auxiliary_cu_parents += auxiliary_counts["excluded_cu_parent_count"]
                epoch_auxiliary_final_states += auxiliary_counts["selected_final_state_count"]
                if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
                    raise ValueError("task training produced non-finite loss or gradient")
                if epoch_steps == 1 or epoch_steps % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | task=%s | epoch=%02d/%02d | step=%04d/%04d | global=%05d/%05d | total=%.6f | main=%.6f | prefix_aux=%.6f | prefix_w=%.3f | dbb_aux=%.6f | dbb_aux_w=%.3f | grad=%.3f",
                        config.task.name,
                        epoch,
                        epochs,
                        epoch_steps,
                        steps_per_epoch,
                        global_step,
                        planned_steps,
                        float(loss.item()),
                        float(main_loss.item()),
                        float(prefix_auxiliary_loss.item()),
                        config.train.prefix_auxiliary_weight,
                        float(auxiliary_loss.item()),
                        config.train.dbb_auxiliary_loss_weight,
                        grad_norm,
                    )
                if max_steps is not None and global_step >= max_steps:
                    stop_reason = "max_train_steps"
                    break
            if skip_validation:
                validation = {"status": "not_run_by_argument"}
                selection = unavailable_checkpoint_selection(config.train.checkpoint_selection_metric)
            else:
                validation = evaluate_task(model, val_dataset, config=config, device=device)
                selection = build_checkpoint_selection(validation, config.train.checkpoint_selection_metric)
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": epoch_loss / max(1, epoch_steps),
                "train_main_loss": epoch_main_loss / max(1, epoch_steps),
                "train_prefix_auxiliary_loss": epoch_prefix_auxiliary_loss / max(1, epoch_steps),
                "prefix_auxiliary_loss_weight": config.train.prefix_auxiliary_weight,
                "prefix_auxiliary_eligible_event_count": epoch_prefix_auxiliary_events,
                "prefix_auxiliary_eligible_parent_count": epoch_prefix_auxiliary_parents,
                "train_dbb_auxiliary_loss": epoch_auxiliary_loss / max(1, epoch_steps),
                "dbb_auxiliary_loss_weight": config.train.dbb_auxiliary_loss_weight,
                "dbb_auxiliary_selected_parent_count": epoch_auxiliary_dbb_parents,
                "dbb_auxiliary_excluded_cu_parent_count": epoch_auxiliary_cu_parents,
                "dbb_auxiliary_selected_final_state_count": epoch_auxiliary_final_states,
                "train_steps": epoch_steps,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "selection_metric": config.train.checkpoint_selection_metric,
                "selection_metric_value": selection["primary_value"],
                "checkpoint_selection": selection,
                "validation": _strip_records(validation),
                "parent_order_sha256": parent_order_sha256(epoch_parent_order),
                "parent_order_count": len(epoch_parent_order),
                "parent_order_unique_count": len(set(epoch_parent_order)),
                readiness_hash_field_name: (
                    None if v17_readiness is None else v17_readiness["sha256"]
                ),
            }
            if max_steps is None and len(epoch_parent_order) != len(train_dataset):
                raise ValueError(
                    "full task epoch did not consume every selected parent exactly once: "
                    f"consumed={len(epoch_parent_order)}, expected={len(train_dataset)}"
                )
            if max_steps is None and len(set(epoch_parent_order)) != len(epoch_parent_order):
                raise ValueError("full task epoch repeated a parent in the shuffled order")
            history.append(epoch_record)
            selected_as_best = checkpoint_selection_is_better(selection, best_selection)
            if selected_as_best:
                best_selection = selection
                best_epoch = epoch
                best_validation = validation
            logger.info(
                "epoch | task=%s | epoch=%02d/%02d | total=%.6f | main=%.6f | prefix_aux=%.6f | prefix_w=%.3f | dbb_aux=%.6f | dbb_aux_w=%.3f | selection=%s:%s | tie=%s:%s | elapsed=%s",
                config.task.name,
                epoch,
                epochs,
                epoch_record["train_loss"],
                epoch_record["train_main_loss"],
                epoch_record["train_prefix_auxiliary_loss"],
                epoch_record["prefix_auxiliary_loss_weight"],
                epoch_record["train_dbb_auxiliary_loss"],
                epoch_record["dbb_auxiliary_loss_weight"],
                selection["primary_metric"],
                format_metric(selection["primary_value"]),
                selection["tie_break_metric"] or "none",
                format_metric(selection["tie_break_value"]),
                format_duration(epoch_record["elapsed_seconds"]),
            )
            logger.info(
                "order | epoch=%02d/%02d | parents=%d | unique=%d | sha256=%s",
                epoch,
                epochs,
                epoch_record["parent_order_count"],
                epoch_record["parent_order_unique_count"],
                epoch_record["parent_order_sha256"],
            )
            training_state = None
            if config.schema_version in FORMAL_PAIRED_SCHEMA_VERSIONS:
                training_state = capture_training_state(
                    loader_generator,
                    run_state={
                        "next_epoch": epoch + 1,
                        "global_step": global_step,
                        "best_epoch": best_epoch,
                        "best_selection": best_selection,
                        "history": history,
                    },
                )
            save_task_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                config=config,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
                training_state=training_state,
            )
            if config.train.save_every_epoch:
                if config.train.epoch_checkpoint_dir is None:
                    raise ValueError("save_every_epoch requires train.epoch_checkpoint_dir")
                epoch_checkpoint = project_root / config.train.epoch_checkpoint_dir / f"epoch-{epoch:03d}.pt"
                save_task_checkpoint(
                    epoch_checkpoint,
                    model,
                    optimizer,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_record,
                    training_state=training_state,
                )
                logger.info("checkpoint | task=%s | kind=epoch | epoch=%02d | path=%s", config.task.name, epoch, epoch_checkpoint)
            if selected_as_best:
                save_task_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_record,
                    training_state=training_state,
                )
                logger.info(
                    "checkpoint | task=%s | kind=best | path=%s | selection=%s:%s | tie=%s:%s",
                    config.task.name,
                    best_checkpoint,
                    selection["primary_metric"],
                    format_metric(selection["primary_value"]),
                    selection["tie_break_metric"] or "none",
                    format_metric(selection["tie_break_value"]),
                )
            if max_steps is not None and global_step >= max_steps:
                break
            if config.train.early_stopping_patience and best_epoch > 0 and epoch - best_epoch >= config.train.early_stopping_patience:
                stop_reason = "early_stopping"
                break
        if best_validation is None and best_epoch > 0 and best_checkpoint.is_file():
            load_task_checkpoint(
                best_checkpoint,
                model,
                expected_schema=config.train.checkpoint_schema,
                map_location=device,
            )
            best_validation = evaluate_task(model, val_dataset, config=config, device=device)
        sample_details = None
        if best_validation is not None and "records" in best_validation:
            sample_details = _write_sample_details(
                artifact_dir,
                best_validation,
                checkpoint=best_checkpoint,
                epoch=best_epoch,
                schema_version=config.schema_version,
            )
            log_metric_set(logger, "val:parent_final_prefix", best_validation.get("parent_metrics", {}))
            log_metric_set(logger, "val:window_causal_prefix", best_validation.get("window_metrics", {}))
            log_low_confidence_records(logger, best_validation, threshold=0.60)
        else:
            logger.info("metrics | scope=validation | status=not_executed")
        performance_validation = (
            best_validation
            if best_validation is not None and best_validation.get("parent_metrics") is not None
            else None
        )
        performance = summarize_task_performance_guardrails(
            performance_validation,
            target=(
                config.train.task_min_precision_recall_target
                if is_product_local_schema(config.schema_version)
                else None
            ),
        )
        result = {
            "status": "completed" if is_product_local_schema(config.schema_version) else "passed",
            **performance,
            "task_name": config.task.name,
            "config_name": config.config_name,
            "device": str(device),
            "train_parent_count": len(train_dataset),
            "validation_parent_count": len(val_dataset),
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_selection_metric_value": None if best_selection is None else best_selection["primary_value"],
            "best_checkpoint_selection": best_selection,
            "best_validation_metrics": None if best_validation is None else _strip_records(best_validation),
            "stop_reason": stop_reason,
            "history": history,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "sample_detail_information": None if sample_details is None else str(sample_details),
            "formal_pretraining": "not_required_and_not_used",
            "local_task_only": is_product_local_schema(config.schema_version),
            "deployment_eligible": False,
            "deployment_blockers": (
                [
                    "full_cascade_evaluation_not_executed",
                    "oof_rejection_calibration_not_executed",
                    "test_not_executed",
                    "stable_not_executed",
                    "p95_not_executed",
                ]
                if is_product_local_schema(config.schema_version)
                else []
            ),
            "paired_initialization": paired_initialization,
            readiness_field_name: v17_readiness,
            "dataloader_contract": {
                "seed": int(loader_seed),
                "independent_generator": True,
                "epoch_parent_order_hashes": [
                    {
                        "epoch": int(row["epoch"]),
                        "sha256": row["parent_order_sha256"],
                        "count": int(row["parent_order_count"]),
                    }
                    for row in history
                ],
                "cuda_deterministic_algorithms": bool(config.train.cuda_deterministic_algorithms),
                "determinism_claim": (
                    "fixed_shared_initialization_and_sampling_order_not_bitwise_cuda_determinism"
                    if config.schema_version in FORMAL_PAIRED_SCHEMA_VERSIONS
                    else "seeded_existing_task_contract"
                ),
            },
            "epoch_checkpoint_dir": (
                None
                if config.train.epoch_checkpoint_dir is None
                else str(project_root / config.train.epoch_checkpoint_dir)
            ),
            "resume": {
                "resumed": resumed_from is not None,
                "checkpoint": resumed_from,
                "start_epoch": start_epoch,
                "epoch_boundary_only": True,
            },
            "test_evaluation": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "oof_rejection_calibration": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "elapsed_seconds": time.perf_counter() - started_run,
        }
        (artifact_dir / "training-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "run=complete | task=%s | status=%s | performance_status=%s | best_checkpoint=%s | summary=%s",
            config.task.name,
            result["status"],
            result["performance_status"],
            best_checkpoint,
            artifact_dir / "training-summary.json",
        )
        return result
    except Exception as error:
        if is_product_local_schema(config.schema_version):
            failure = {
                "status": "failed",
                "performance_status": "not_evaluated",
                "task_name": config.task.name,
                "config_name": config.config_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "deployment_eligible": False,
                "formal_pretraining": "not_required_and_not_used",
                readiness_field_name: v17_readiness,
                "full_cascade_evaluation": "not_executed",
                "oof_rejection_calibration": "not_executed",
                "test_evaluation": "not_executed",
                "stable_evaluation": "not_executed",
                "p95_latency": "not_executed",
                "elapsed_seconds": time.perf_counter() - started_run,
            }
            (artifact_dir / "training-summary.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.exception(
                "run=failed | task=%s | error_type=%s | summary=%s",
                config.task.name,
                type(error).__name__,
                artifact_dir / "training-summary.json",
            )
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one repaired RADAR-LN3 liquid task; formal training is manual-only.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    result = train_task(
        project_root,
        config_path.resolve(),
        max_steps=args.max_steps,
        epochs_override=args.epochs,
        skip_validation=args.skip_validation,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
