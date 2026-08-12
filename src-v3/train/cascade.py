from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.cascade import (
    EXPECTED_TASKS,
    LiquidSoftCascade,
    build_cascade,
    build_cascade_dataset,
    evaluate_hierarchical_cascade,
    evaluate_strict_open_set_cascade,
    validate_cascade_configs,
    verify_cascade_data_contracts,
)
from radar_ln3.data import build_point_dataloader
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
    log_metric_set,
    progress_interval,
)
from radar_ln3.losses import hierarchical_final_state_nll, strict_open_set_final_state_nll
from radar_ln3.task_models import (
    ANGULAR_KNOWNNESS_SCHEMA_VERSION,
    HIERARCHICAL_SCHEMA_VERSIONS,
    HIERARCHICAL_CASCADE_SCHEMA_VERSION,
    STRICT_OPEN_SET_SCHEMA_VERSION,
    LiquidTaskConfig,
    capture_training_state,
    load_task_config,
    save_task_checkpoint,
    seed_everything,
    task_config_identity_sha256,
    is_strict_open_set_schema,
)


def parent_order_sha256(parent_sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for parent_id in parent_sample_ids:
        encoded = str(parent_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def load_hierarchical_configs(config_paths: Mapping[str, Path]) -> dict[str, LiquidTaskConfig]:
    if set(config_paths) != set(EXPECTED_TASKS):
        raise ValueError(f"hierarchical training requires exactly these configs: {EXPECTED_TASKS}")
    configs = {name: load_task_config(config_paths[name]) for name in EXPECTED_TASKS}
    validate_cascade_configs(configs)
    versions = {config.schema_version for config in configs.values()}
    if len(versions) != 1 or next(iter(versions)) not in HIERARCHICAL_SCHEMA_VERSIONS:
        raise ValueError("joint cascade training accepts one matching V13 or V14 config set")
    return configs


def cascade_forward_batch(model: LiquidSoftCascade, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        tr_features=tensors["tr_features"],
        tr_missing_mask=tensors["tr_missing_mask"],
        tr_dt_seconds=tensors["tr_dt_seconds"],
        tr_dt_valid=tensors["tr_dt_valid"],
        tr_dt_reason=tensors["tr_dt_reason"],
        rd_frames=tensors["rd_frames"],
        rd_frame_mask=tensors["rd_frame_mask"],
        rd_quality=tensors["rd_quality"],
        tr_padding_mask=tensors["tr_padding_mask"],
    )


def cascade_training_objective(
    output: Mapping[str, torch.Tensor],
    tensors: Mapping[str, torch.Tensor],
    configs: Mapping[str, LiquidTaskConfig],
) -> dict[str, torch.Tensor]:
    gate = configs["dbb_cu_gate"]
    dbb = configs["dbb_specialist"]
    cu = configs["cu_specialist"]
    objective = (
        strict_open_set_final_state_nll
        if is_strict_open_set_schema(gate.schema_version)
        else hierarchical_final_state_nll
    )
    keyword_arguments = {
        "prefix_auxiliary_weight": gate.train.prefix_auxiliary_weight,
        "prefix_auxiliary_min_events": gate.train.prefix_auxiliary_min_events,
        "gate_class_weights": output["gate_sequence_logits"].new_tensor(gate.train.class_weights),
        "dbb_class_weights": output["dbb_sequence_logits"].new_tensor(dbb.train.class_weights),
        "cu_class_weights": output["cu_sequence_logits"].new_tensor(cu.train.class_weights),
    }
    if is_strict_open_set_schema(gate.schema_version):
        keyword_arguments.update({
            "dbb_angular_margin": dbb.model.dbb_angular_margin,
            "dbb_cosine_scale": dbb.model.dbb_cosine_scale,
        })
    return objective(
        output["gate_sequence_logits"],
        output["dbb_sequence_logits"],
        output["cu_sequence_logits"],
        tensors["labels"],
        tensors["tr_padding_mask"],
        **keyword_arguments,
    )


def _class_minimum(metrics: Mapping[str, Any], class_names: tuple[str, ...]) -> float:
    rows = {str(row["class"]): row for row in metrics.get("per_class", [])}
    values: list[float] = []
    for name in class_names:
        row = rows.get(name)
        if row is None or int(row.get("support", 0)) <= 0:
            raise ValueError(f"product selection lacks supported class {name}")
        values.extend((float(row["precision"]), float(row["recall"])))
    if not values or any(not np.isfinite(value) for value in values):
        raise ValueError("product selection received a non-finite metric")
    return min(values)


def build_product_checkpoint_selection(
    validation: Mapping[str, Any],
    config: LiquidTaskConfig,
) -> dict[str, Any]:
    inputs = dict(validation.get("checkpoint_selection_inputs") or {})
    if is_strict_open_set_schema(config.schema_version):
        if (
            inputs.get("status") != "available_known_only"
            or int(inputs.get("unknown_parent_count", -1)) != 0
            or inputs.get("unknown_used_for_selection") is not False
        ):
            raise ValueError("V14 checkpoint selection received Unknown exposure or a non-known-only evaluation")
        gate_floor = float(inputs["gate_min_precision_recall"])
        dbb_floor = float(inputs["dbb_min_precision_recall"])
        cu_floor = float(inputs["cu_min_precision_recall"])
        drone_bird_floor = float(inputs["known_parent_drone_bird_min_precision_recall"])
        known_macro_f1 = float(inputs["known_parent_macro_f1"])
        known_metrics = dict(validation["known_closed_set"]["parent_metrics"])
        matrix = np.asarray(
            known_metrics["confusion_matrix_rows_known_true_columns_known_plus_unresolved"],
            dtype=np.int64,
        )
        bird_to_cu_or_unresolved = int(matrix[1, 3:].sum())
        targets = {
            "gate": float(config.train.gate_min_precision_recall_target),
            "dbb": float(config.train.dbb_min_precision_recall_target),
            "cu": float(config.train.cu_min_precision_recall_target),
            "drone_bird": float(config.train.drone_bird_min_precision_recall_target),
        }
        floors = {
            "gate": gate_floor,
            "dbb": dbb_floor,
            "cu": cu_floor,
            "drone_bird": drone_bird_floor,
        }
        passed = {name: floors[name] >= targets[name] for name in floors}
        normalized_completion = min(floors[name] / targets[name] for name in floors)
        ranking_key = [
            float(sum(passed.values())),
            float(normalized_completion),
            float(gate_floor),
            float(dbb_floor),
            float(cu_floor),
            float(drone_bird_floor),
            float(-bird_to_cu_or_unresolved),
            known_macro_f1,
        ]
        return {
            "metric": config.train.checkpoint_selection_metric,
            "primary_metric": "known_only_product_guardrail_pass_count",
            "primary_value": sum(passed.values()),
            "floors": floors,
            "targets": targets,
            "passed": passed,
            "all_product_guardrails_passed": all(passed.values()),
            "minimum_normalized_completion": normalized_completion,
            "bird_to_cu_or_unresolved_count": bird_to_cu_or_unresolved,
            "known_parent_macro_f1": known_macro_f1,
            "unknown_parent_count": 0,
            "unknown_used_for_selection": False,
            "ranking_key": ranking_key,
            "calibrated_rejection_metrics_used": False,
            "unresolved_evidence_used_as_unknown_class": False,
            "reason": "V14 selection uses only known validation parents; Unknown development audit is forbidden from ranking",
        }
    gate_floor = float(inputs["gate_min_precision_recall"])
    dbb_floor = float(inputs["dbb_min_precision_recall"])
    cu_floor = float(inputs["cu_min_precision_recall"])
    diagnostic = dict(validation["diagnostic_hierarchical_argmax"]["parent_metrics"])
    drone_bird_floor = _class_minimum(diagnostic, ("DroneTarget", "BirdTarget"))
    diagnostic_macro_f1 = float(inputs["diagnostic_parent_macro_f1"])
    soft_macro_f1 = float(inputs["soft_parent_macro_f1"])
    matrix = np.asarray(diagnostic["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
    bird_to_cu = int(matrix[1, 3:].sum())
    unknown_to_dbb = int(matrix[4, :3].sum())
    targets = {
        "gate": float(config.train.gate_min_precision_recall_target),
        "dbb": float(config.train.dbb_min_precision_recall_target),
        "cu": float(config.train.cu_min_precision_recall_target),
        "drone_bird": float(config.train.drone_bird_min_precision_recall_target),
    }
    floors = {"gate": gate_floor, "dbb": dbb_floor, "cu": cu_floor, "drone_bird": drone_bird_floor}
    passed = {name: floors[name] >= targets[name] for name in floors}
    normalized_completion = min(floors[name] / targets[name] for name in floors)
    ranking_key = [
        float(sum(passed.values())),
        float(normalized_completion),
        float(gate_floor),
        float(dbb_floor),
        float(cu_floor),
        float(drone_bird_floor),
        float(-unknown_to_dbb),
        float(-bird_to_cu),
        diagnostic_macro_f1,
    ]
    return {
        "metric": config.train.checkpoint_selection_metric,
        "primary_metric": "product_guardrail_pass_count",
        "primary_value": sum(passed.values()),
        "floors": floors,
        "targets": targets,
        "passed": passed,
        "all_product_guardrails_passed": all(passed.values()),
        "minimum_normalized_completion": normalized_completion,
        "unknown_to_dbb_count": unknown_to_dbb,
        "bird_to_cu_count": bird_to_cu,
        "diagnostic_parent_macro_f1": diagnostic_macro_f1,
        "soft_parent_macro_f1": soft_macro_f1,
        "ranking_key": ranking_key,
        "calibrated_rejection_metrics_used": False,
        "soft_posterior_used_for_selection": False,
        "reason": "OOF calibration is not yet available; product selection uses frozen diagnostic hierarchy guardrails only and reports the soft posterior for audit without ranking on it",
    }


def selection_is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    if candidate.get("metric") != incumbent.get("metric"):
        raise ValueError("cannot compare different product checkpoint metrics")
    return tuple(float(value) for value in candidate["ranking_key"]) > tuple(
        float(value) for value in incumbent["ranking_key"]
    )


def _joint_payload(
    model: LiquidSoftCascade,
    optimizer: torch.optim.Optimizer,
    configs: Mapping[str, LiquidTaskConfig],
    loader_generator: torch.Generator,
    *,
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_selection: Mapping[str, Any] | None,
    history: list[dict[str, Any]],
    checkpoint_set_id: str,
) -> dict[str, Any]:
    gate = configs["dbb_cu_gate"]
    product_contract = {
        "loss_mode": gate.train.loss_mode,
        "prefix_auxiliary_weight": gate.train.prefix_auxiliary_weight,
        "prefix_auxiliary_min_events": gate.train.prefix_auxiliary_min_events,
        "checkpoint_selection_metric": gate.train.checkpoint_selection_metric,
        "targets": {
            "gate": gate.train.gate_min_precision_recall_target,
            "dbb": gate.train.dbb_min_precision_recall_target,
            "cu": gate.train.cu_min_precision_recall_target,
            "drone_bird": gate.train.drone_bird_min_precision_recall_target,
        },
        "unknown_action": "UNKNOWN_REJECTED requires later train-parent OOF calibration",
    }
    if is_strict_open_set_schema(gate.schema_version):
        product_contract.update({
            "unknown_training_exposure": 0,
            "training_population": "known_only",
        })
    return {
        "checkpoint_schema": gate.train.joint_checkpoint_schema,
        "schema_version": gate.schema_version,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_set_id": str(checkpoint_set_id),
        "task_configs": {name: configs[name].config_name for name in EXPECTED_TASKS},
        "task_config_identities": {
            name: task_config_identity_sha256(configs[name]) for name in EXPECTED_TASKS
        },
        "task_checkpoint_schemas": {name: configs[name].train.checkpoint_schema for name in EXPECTED_TASKS},
        "data_contract": {
            "index_sha256": gate.data.expected_index_sha256,
            "contract_sha256": gate.data.expected_contract_sha256,
            "stats_sha256": gate.data.expected_stats_sha256,
        },
        "product_contract": product_contract,
        "model_states": {
            "dbb_cu_gate": model.gate.state_dict(),
            "dbb_specialist": model.dbb_specialist.state_dict(),
            "cu_specialist": model.cu_specialist.state_dict(),
        },
        "optimizer_state": optimizer.state_dict(),
        "training_state": capture_training_state(
            loader_generator,
            run_state={
                "next_epoch": int(epoch) + 1,
                "global_step": int(global_step),
                "best_epoch": int(best_epoch),
                "best_selection": None if best_selection is None else dict(best_selection),
                "history": history,
                "checkpoint_set_id": str(checkpoint_set_id),
            },
        ),
    }


def save_joint_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def load_joint_checkpoint(
    path: Path,
    model: LiquidSoftCascade,
    optimizer: torch.optim.Optimizer,
    configs: Mapping[str, LiquidTaskConfig],
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    gate = configs["dbb_cu_gate"]
    if payload.get("checkpoint_schema") != gate.train.joint_checkpoint_schema:
        raise ValueError("joint checkpoint schema mismatch")
    if payload.get("schema_version") != gate.schema_version:
        raise ValueError("joint checkpoint run/configuration schema mismatch")
    checkpoint_set_id = str(payload.get("checkpoint_set_id", ""))
    if len(checkpoint_set_id) != 32 or any(
        character not in "0123456789abcdef" for character in checkpoint_set_id
    ):
        raise ValueError("joint checkpoint lacks a valid checkpoint_set_id")
    expected_hashes = {
        "index_sha256": gate.data.expected_index_sha256,
        "contract_sha256": gate.data.expected_contract_sha256,
        "stats_sha256": gate.data.expected_stats_sha256,
    }
    if payload.get("data_contract") != expected_hashes:
        raise ValueError("joint checkpoint data contract mismatch")
    expected_configs = {name: configs[name].config_name for name in EXPECTED_TASKS}
    expected_config_identities = {
        name: task_config_identity_sha256(configs[name]) for name in EXPECTED_TASKS
    }
    expected_task_schemas = {name: configs[name].train.checkpoint_schema for name in EXPECTED_TASKS}
    if (
        payload.get("task_configs") != expected_configs
        or payload.get("task_config_identities") != expected_config_identities
        or payload.get("task_checkpoint_schemas") != expected_task_schemas
    ):
        raise ValueError("joint checkpoint task configuration identity mismatch")
    expected_product_contract: dict[str, Any] = {
        "loss_mode": gate.train.loss_mode,
        "prefix_auxiliary_weight": gate.train.prefix_auxiliary_weight,
        "prefix_auxiliary_min_events": gate.train.prefix_auxiliary_min_events,
        "checkpoint_selection_metric": gate.train.checkpoint_selection_metric,
        "targets": {
            "gate": gate.train.gate_min_precision_recall_target,
            "dbb": gate.train.dbb_min_precision_recall_target,
            "cu": gate.train.cu_min_precision_recall_target,
            "drone_bird": gate.train.drone_bird_min_precision_recall_target,
        },
        "unknown_action": "UNKNOWN_REJECTED requires later train-parent OOF calibration",
    }
    if is_strict_open_set_schema(gate.schema_version):
        expected_product_contract.update({
            "unknown_training_exposure": 0,
            "training_population": "known_only",
        })
    if payload.get("product_contract") != expected_product_contract:
        raise ValueError("joint checkpoint product objective/action contract mismatch")
    states = dict(payload.get("model_states") or {})
    if set(states) != set(EXPECTED_TASKS):
        raise ValueError("joint checkpoint lacks one or more task states")
    model.gate.load_state_dict(states["dbb_cu_gate"], strict=True)
    model.dbb_specialist.load_state_dict(states["dbb_specialist"], strict=True)
    model.cu_specialist.load_state_dict(states["cu_specialist"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    state = dict(payload["training_state"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu_rng_state"], device="cpu"))
    cuda_states = list(state["torch_cuda_rng_states"])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("joint checkpoint CUDA RNG state count does not match this host")
        torch.cuda.set_rng_state_all([torch.as_tensor(value, device="cpu") for value in cuda_states])
    loader_generator.set_state(torch.as_tensor(state["dataloader_generator_state"], device="cpu"))
    run_state = dict(state["run_state"])
    if run_state.get("checkpoint_set_id") != checkpoint_set_id:
        raise ValueError("joint checkpoint set identity differs from its run state")
    return run_state


def save_task_checkpoint_set(
    project_root: Path,
    model: LiquidSoftCascade,
    configs: Mapping[str, LiquidTaskConfig],
    *,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    kind: str,
    checkpoint_set_id: str,
) -> dict[str, str]:
    models = {
        "dbb_cu_gate": model.gate,
        "dbb_specialist": model.dbb_specialist,
        "cu_specialist": model.cu_specialist,
    }
    paths: dict[str, str] = {}
    for name in EXPECTED_TASKS:
        config = configs[name]
        if kind == "best":
            path = resolve_path(project_root, config.train.checkpoint_path)
        elif kind == "last":
            path = resolve_path(project_root, config.train.last_checkpoint_path)
        elif kind == "epoch":
            if config.train.epoch_checkpoint_dir is None:
                raise ValueError("hierarchical epoch checkpoint directory is missing")
            path = resolve_path(project_root, config.train.epoch_checkpoint_dir) / f"epoch-{epoch:03d}.pt"
        else:
            raise ValueError(f"unsupported task checkpoint kind: {kind}")
        checkpoint_metrics = dict(metrics)
        existing_set_id = checkpoint_metrics.get("checkpoint_set_id")
        if existing_set_id not in {None, checkpoint_set_id}:
            raise ValueError("task checkpoint metrics carry a different checkpoint_set_id")
        checkpoint_metrics["checkpoint_set_id"] = checkpoint_set_id
        save_task_checkpoint(
            path,
            models[name],
            None,
            config=config,
            epoch=epoch,
            global_step=global_step,
            metrics=checkpoint_metrics,
        )
        paths[name] = str(path)
    return paths


def _strip_records(result: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(result)
    value.pop("records", None)
    return value


def train_hierarchical_cascade(
    project_root: Path,
    *,
    config_paths: Mapping[str, Path],
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    configs = load_hierarchical_configs(config_paths)
    gate = configs["dbb_cu_gate"]
    strict_open_set = is_strict_open_set_schema(gate.schema_version)
    schema_label = gate.schema_version.replace("RADAR-LN3-", "")
    if gate.train.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"{gate.schema_version} formal training requires an available CUDA device; CPU fallback is forbidden")
    if not all(config.train.allow_parameter_updates for config in configs.values()):
        raise PermissionError(f"one or more {gate.schema_version} configs forbid parameter updates")
    contracts = verify_cascade_data_contracts(project_root, configs)
    artifact_dir = resolve_path(project_root, gate.train.joint_artifact_dir or "")
    if artifact_dir.exists() and resume_checkpoint is None and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {gate.schema_version} joint artifact: {artifact_dir}")
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        try:
            resume_checkpoint.relative_to(artifact_dir)
        except ValueError as exc:
            raise ValueError(f"{gate.schema_version} resume checkpoint must be inside the joint artifact") from exc
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(
        artifact_dir / "training.log",
        logger_name=f"radar_ln3.training.{schema_label.lower()}_cascade",
    )
    started = time.perf_counter()
    try:
        seed_everything(gate.train.seed)
        device = torch.device("cuda")
        dataset_scope = "known_only" if strict_open_set else None
        train_dataset = build_cascade_dataset(
            project_root,
            gate,
            split_name=gate.data.train_split_name,
            label_scope=dataset_scope,
        )
        val_dataset = build_cascade_dataset(
            project_root,
            gate,
            split_name=gate.data.validation_split_name,
            label_scope=dataset_scope,
        )
        if strict_open_set:
            train_labels = set(train_dataset.frame["label_name"].astype(str))
            val_labels = set(val_dataset.frame["label_name"].astype(str))
            if "UnknownTarget" in train_labels or "UnknownTarget" in val_labels:
                raise ValueError("V14 optimizer/checkpoint Dataset contains UnknownTarget")
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(int(gate.train.dataloader_seed))
        train_loader = build_point_dataloader(
            train_dataset,
            batch_size=gate.train.batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        model = build_cascade(configs, device=device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=gate.train.learning_rate)
        global_step = 0
        start_epoch = 1
        best_epoch = -1
        best_selection: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        if resume_checkpoint is not None:
            state = load_joint_checkpoint(resume_checkpoint, model, optimizer, configs, loader_generator)
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            best_epoch = int(state["best_epoch"])
            best_selection = state.get("best_selection")
            history = list(state.get("history") or [])
            if start_epoch != int(torch.load(resume_checkpoint, map_location="cpu", weights_only=False)["epoch"]) + 1:
                raise ValueError(f"{gate.schema_version} joint resume next_epoch is inconsistent")
        if not 1 <= start_epoch <= gate.train.epochs:
            raise ValueError(f"{gate.schema_version} resume epoch is outside the fixed epoch budget")
        logger.info(
            "run=start | schema=%s | mode=joint_three_checkpoint | population=%s | device=cuda | train_parents=%d | val_parents=%d | batch=%d | epochs=%d | loss=%s | prefix_aux_weight=%.3f | prefix_min_events=%d | pretraining=not_required",
            schema_label,
            "known_only" if strict_open_set else "all_five_labeled_classes",
            len(train_dataset),
            len(val_dataset),
            gate.train.batch_size,
            gate.train.epochs,
            gate.train.loss_mode,
            gate.train.prefix_auxiliary_weight,
            gate.train.prefix_auxiliary_min_events,
        )
        for epoch in range(start_epoch, gate.train.epochs + 1):
            model.train()
            epoch_values = {
                "loss": 0.0,
                "final_hierarchical_nll": 0.0,
                "final_gate_nll": 0.0,
                "final_specialist_nll": 0.0,
                "prefix_auxiliary_nll": 0.0,
            }
            epoch_steps = 0
            order: list[str] = []
            epoch_started = time.perf_counter()
            for batch in train_loader:
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = cascade_forward_batch(model, tensors)
                objective = cascade_training_objective(output, tensors, configs)
                optimizer.zero_grad(set_to_none=True)
                objective["loss"].backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gate.train.gradient_clip_norm).item())
                optimizer.step()
                if not np.isfinite(grad_norm) or not torch.isfinite(objective["loss"]).item():
                    raise ValueError(f"{gate.schema_version} joint training produced a non-finite loss or gradient")
                global_step += 1
                epoch_steps += 1
                for name in epoch_values:
                    epoch_values[name] += float(objective[name].item())
                if epoch_steps == 1 or epoch_steps % progress_interval(len(train_loader)) == 0:
                    logger.info(
                        "progress | epoch=%02d/%02d | step=%04d/%04d | global=%05d | total=%.6f | final=%.6f | gate=%.6f | specialist=%.6f | prefix=%.6f | prefix_w=%.3f | grad=%.3f",
                        epoch,
                        gate.train.epochs,
                        epoch_steps,
                        len(train_loader),
                        global_step,
                        float(objective["loss"].item()),
                        float(objective["final_hierarchical_nll"].item()),
                        float(objective["final_gate_nll"].item()),
                        float(objective["final_specialist_nll"].item()),
                        float(objective["prefix_auxiliary_nll"].item()),
                        gate.train.prefix_auxiliary_weight,
                        grad_norm,
                    )
            if len(order) != len(train_dataset) or len(set(order)) != len(order):
                raise ValueError(f"{gate.schema_version} epoch did not consume each selected parent exactly once")
            model.eval()
            validation = (
                evaluate_strict_open_set_cascade(
                    model,
                    val_dataset,
                    device=device,
                    batch_size=gate.train.batch_size,
                    split_name=gate.data.validation_split_name,
                    evaluation_scope="known_only_checkpoint_selection",
                )
                if strict_open_set
                else evaluate_hierarchical_cascade(
                    model,
                    val_dataset,
                    device=device,
                    batch_size=gate.train.batch_size,
                    split_name=gate.data.validation_split_name,
                )
            )
            selection = build_product_checkpoint_selection(validation, gate)
            selected_as_best = selection_is_better(selection, best_selection)
            if selected_as_best:
                best_selection = selection
                best_epoch = epoch
            checkpoint_set_id = uuid.uuid4().hex
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_set_id": checkpoint_set_id,
                **{name: value / max(1, epoch_steps) for name, value in epoch_values.items()},
                "train_steps": epoch_steps,
                "parent_order_sha256": parent_order_sha256(order),
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "checkpoint_selection": selection,
                "validation": _strip_records(validation),
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            save_task_checkpoint_set(
                project_root,
                model,
                configs,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
                kind="last",
                checkpoint_set_id=checkpoint_set_id,
            )
            save_task_checkpoint_set(
                project_root,
                model,
                configs,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
                kind="epoch",
                checkpoint_set_id=checkpoint_set_id,
            )
            payload = _joint_payload(
                model,
                optimizer,
                configs,
                loader_generator,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_selection=best_selection,
                history=history,
                checkpoint_set_id=checkpoint_set_id,
            )
            save_joint_checkpoint(resolve_path(project_root, gate.train.joint_last_checkpoint_path or ""), payload)
            if selected_as_best:
                best_paths = save_task_checkpoint_set(
                    project_root,
                    model,
                    configs,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_record,
                    kind="best",
                    checkpoint_set_id=checkpoint_set_id,
                )
                save_joint_checkpoint(resolve_path(project_root, gate.train.joint_best_checkpoint_path or ""), payload)
                if strict_open_set:
                    logger.info(
                        "checkpoint | kind=joint_best | epoch=%02d | guardrails=%d/4 | min_completion=%s | gate=%s | dbb=%s | clutter_accept=%s | drone_bird=%s | bird_to_cu_or_unresolved=%d | unknown_used_for_selection=false | paths=%s",
                        epoch,
                        selection["primary_value"],
                        format_metric(selection["minimum_normalized_completion"]),
                        format_metric(selection["floors"]["gate"]),
                        format_metric(selection["floors"]["dbb"]),
                        format_metric(selection["floors"]["cu"]),
                        format_metric(selection["floors"]["drone_bird"]),
                        selection["bird_to_cu_or_unresolved_count"],
                        best_paths,
                    )
                else:
                    logger.info(
                        "checkpoint | kind=joint_best | epoch=%02d | guardrails=%d/4 | min_completion=%s | gate=%s | dbb=%s | cu=%s | drone_bird=%s | unknown_to_dbb=%d | bird_to_cu=%d | paths=%s",
                        epoch,
                        selection["primary_value"],
                        format_metric(selection["minimum_normalized_completion"]),
                        format_metric(selection["floors"]["gate"]),
                        format_metric(selection["floors"]["dbb"]),
                        format_metric(selection["floors"]["cu"]),
                        format_metric(selection["floors"]["drone_bird"]),
                        selection["unknown_to_dbb_count"],
                        selection["bird_to_cu_count"],
                        best_paths,
                    )
            selection_macro_f1 = (
                selection["known_parent_macro_f1"]
                if strict_open_set
                else selection["diagnostic_parent_macro_f1"]
            )
            logger.info(
                "epoch | epoch=%02d/%02d | total=%.6f | final=%.6f | prefix=%.6f | guardrails=%d/4 | min_completion=%s | macro_f1=%s | elapsed=%s | order_sha256=%s",
                epoch,
                gate.train.epochs,
                epoch_record["loss"],
                epoch_record["final_hierarchical_nll"],
                epoch_record["prefix_auxiliary_nll"],
                selection["primary_value"],
                format_metric(selection["minimum_normalized_completion"]),
                format_metric(selection_macro_f1),
                format_duration(epoch_record["elapsed_seconds"]),
                epoch_record["parent_order_sha256"],
            )
        best_joint_path = resolve_path(project_root, gate.train.joint_best_checkpoint_path or "")
        if not best_joint_path.is_file():
            raise RuntimeError(f"{gate.schema_version} training completed without a joint best checkpoint")
        final_state = load_joint_checkpoint(
            best_joint_path,
            model,
            optimizer,
            configs,
            loader_generator,
        )
        model.eval()
        best_validation = (
            evaluate_strict_open_set_cascade(
                model,
                val_dataset,
                device=device,
                batch_size=gate.train.batch_size,
                split_name=gate.data.validation_split_name,
                evaluation_scope="known_only_checkpoint_selection",
            )
            if strict_open_set
            else evaluate_hierarchical_cascade(
                model,
                val_dataset,
                device=device,
                batch_size=gate.train.batch_size,
                split_name=gate.data.validation_split_name,
            )
        )
        development_open_set_audit: dict[str, Any] | str = "not_applicable_to_V13"
        development_unknown_access_count = 0
        if strict_open_set and gate.schema_version != ANGULAR_KNOWNNESS_SCHEMA_VERSION:
            development_dataset = build_cascade_dataset(
                project_root,
                gate,
                split_name=gate.data.validation_split_name,
                label_scope="development_open_set_audit",
            )
            development_open_set_audit = evaluate_strict_open_set_cascade(
                model,
                development_dataset,
                device=device,
                batch_size=gate.train.batch_size,
                split_name=gate.data.validation_split_name,
                evaluation_scope="development_open_set_audit",
            )
            if development_open_set_audit["checkpoint_selection_inputs"]["status"] != "forbidden_for_development_open_set_audit":
                raise RuntimeError("V14 development Unknown audit exposed checkpoint selection inputs")
            development_unknown_access_count = int(
                development_open_set_audit["unknown_audit_parent_count"]
            )
        elif gate.schema_version == ANGULAR_KNOWNNESS_SCHEMA_VERSION:
            development_open_set_audit = (
                "未执行：V15 development Unknown 仅可在两臂正式checkpoint、嵌套OOF和"
                "对应校准均冻结后，通过独立一次审计入口访问"
            )
        sample_path = artifact_dir / "sample_detail_information.json"
        sample_path.write_text(
            json.dumps(
                {
                    "schema_version": f"{gate.schema_version}-hierarchical-sample-detail-v1",
                    "best_epoch": best_epoch,
                    "known_only_checkpoint_validation_records": best_validation.get("records", {}),
                    "development_open_set_audit_records": (
                        development_open_set_audit.get("records", {})
                        if isinstance(development_open_set_audit, dict)
                        else {}
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if strict_open_set:
            log_metric_set(
                logger,
                "val:known_only_parent_final_not_calibrated_rejection",
                best_validation["known_closed_set"]["parent_metrics"],
            )
        else:
            log_metric_set(
                logger,
                "val:diagnostic_hierarchical_parent_final_not_calibrated_rejection",
                best_validation["diagnostic_hierarchical_argmax"]["parent_metrics"],
            )
        for name, metrics in best_validation["component_metrics"].items():
            log_metric_set(logger, f"val:{name}", metrics)
        result = {
            "status": "passed",
            "schema_version": gate.schema_version,
            "device": str(device),
            "configs": {name: str(config_paths[name]) for name in EXPECTED_TASKS},
            "data_contracts": contracts,
            "train_parent_count": len(train_dataset),
            "validation_parent_count": len(val_dataset),
            "training_population": "known_only" if strict_open_set else "all_five_labeled_classes",
            "unknown_training_exposure": 0 if strict_open_set else "V13_supervised_unknown",
            "development_unknown_access_count_during_formal_training": development_unknown_access_count,
            "epochs": gate.train.epochs,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_checkpoint_set_id": final_state.get("checkpoint_set_id"),
            "best_checkpoint_selection": best_selection,
            "best_checkpoints": {name: str(resolve_path(project_root, configs[name].train.checkpoint_path)) for name in EXPECTED_TASKS},
            "best_joint_checkpoint": str(best_joint_path),
            "last_joint_checkpoint": str(resolve_path(project_root, gate.train.joint_last_checkpoint_path or "")),
            "sample_detail_information": str(sample_path),
            "best_validation": _strip_records(best_validation),
            "development_open_set_audit": (
                _strip_records(development_open_set_audit)
                if isinstance(development_open_set_audit, dict)
                else development_open_set_audit
            ),
            "history": history,
            "formal_pretraining": "not_required_and_not_used",
            "oof_calibration": "未执行",
            "calibrated_rejection": "未执行",
            "stable_action_evaluation": "未执行",
            "test_evaluation": "未执行",
            "p95_latency": "未执行",
            "resume": {
                "resumed": resume_checkpoint is not None,
                "checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
                "next_epoch_at_loaded_best": final_state.get("next_epoch"),
                "epoch_boundary_only": True,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary = artifact_dir / "training-summary.json"
        summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "run=complete | schema=%s | best_epoch=%d | joint_best=%s | summary=%s",
            schema_label,
            best_epoch,
            best_joint_path,
            summary,
        )
        return result
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a V13/V14/V15 three-checkpoint liquid hierarchical cascade jointly.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--dbb-config", type=Path, required=True)
    parser.add_argument("--cu-config", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()

    def absolute(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    result = train_hierarchical_cascade(
        root,
        config_paths={
            "dbb_cu_gate": absolute(args.gate_config),
            "dbb_specialist": absolute(args.dbb_config),
            "cu_specialist": absolute(args.cu_config),
        },
        resume_checkpoint=None if args.resume_checkpoint is None else absolute(args.resume_checkpoint),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
