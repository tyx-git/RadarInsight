"""Run V25a's full V24-equivalent gradient-telemetry audit training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v25a import CONFIG
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_SELECTION_METRIC,
    V25A_GRADIENT_TELEMETRY_VARIANT,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    known_gate_device,
    load_known_gate_config,
    seed_everything,
)
from radar_ln3.known_gate_gradient_telemetry import V25AGradientTelemetryRecorder
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_common import (
    _capture_training_state,
    _load_checkpoint,
    _model_inputs,
    _parent_order_sha256,
    _performance,
    _save_checkpoint,
    _selection,
    _state_dict_sha256,
    _verify_contract,
    _verify_readiness,
)


V25A_EPOCH_COUNT = 20
V25A_BATCH_SIZE = 2
V25A_LEARNING_RATE = 1.0e-4
V25A_GRADIENT_CLIP_NORM = 100.0
V25A_ADAMW_DEFAULT_WEIGHT_DECAY = 0.01
V25A_CLASS_WEIGHT_NORMALIZATION = "batch_weight_sum"
V25A_FROZEN_MODEL_FIELDS = {
    "tr_value_dim": 14,
    "tr_missing_dim": 11,
    "rd_quality_dim": 7,
    "hidden_dim": 48,
    "rd_embedding_dim": 24,
    "liquid_hidden_dim": 48,
    "rd_height": 21,
    "rd_width": 384,
    "liquid_tau_min_seconds": 0.05,
    "liquid_tau_max_seconds": 30.0,
    "liquid_zero_dt_event_floor_seconds": 0.001,
    "liquid_dt_max_seconds": 120.0,
    "liquid_invalid_dt_update_mode": "bounded_observation",
    "liquid_invalid_dt_observation_gain": 0.25,
    "liquid_state_normalization": "recurrent",
    "rd_frame_aggregation": "masked_mean",
    "dt_reason_mode": "v3_reason_aware_dynamics_only",
    "evidence_dim": 1,
    "rd_temporal_residual_dim": 0,
}


def _require_v24_equivalent_config(config: Any) -> None:
    """Fail closed if V25a drifts from the V24 training semantics."""

    if (
        int(config.train.epochs) != V25A_EPOCH_COUNT
        or int(config.train.batch_size) != V25A_BATCH_SIZE
        or abs(float(config.train.learning_rate) - V25A_LEARNING_RATE) > 1.0e-15
        or abs(float(config.train.gradient_clip_norm) - V25A_GRADIENT_CLIP_NORM)
        > 1.0e-12
        or str(config.train.class_weight_normalization)
        != V25A_CLASS_WEIGHT_NORMALIZATION
        or tuple(float(value) for value in config.train.class_weights)
        != (0.5173137460650578, 14.93939393939394)
        or float(config.train.prefix_auxiliary_weight) != 0.1
        or int(config.train.prefix_auxiliary_min_events) != 3
        or {
            name: getattr(config.model, name)
            for name in V25A_FROZEN_MODEL_FIELDS
        }
        != V25A_FROZEN_MODEL_FIELDS
        or config.train.loss_mode != "parent_final_task_ce_with_prefix_auxiliary"
        or config.train.checkpoint_selection_metric != "parent_min_precision_recall"
        or abs(float(config.train.task_min_precision_recall_target) - 0.9) > 1.0e-12
        or int(config.train.seed) != 43
        or int(config.train.dataloader_seed) != 43
        or int(config.train.early_stopping_patience) != 0
        or config.train.cuda_deterministic_algorithms
        or not config.train.record_parent_order_hash
        or not config.train.save_every_epoch
        or not config.train.full_parameter_training
    ):
        raise ValueError("V25a config drifted from the frozen V24 training semantics")


def _assert_v24_history_order(
    history: Sequence[Mapping[str, Any]],
    expected_orders: Mapping[int, str],
) -> None:
    for expected_epoch, row in enumerate(history, start=1):
        if (
            int(row.get("epoch", -1)) != expected_epoch
            or int(row.get("parent_order_count", -1)) != 986
            or int(row.get("parent_order_unique_count", -1)) != 986
            or str(row.get("parent_order_sha256", ""))
            != str(expected_orders.get(expected_epoch, ""))
        ):
            raise ValueError("V25a resumed history does not reproduce frozen V24 order")


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    config: Any,
    config_path: Path,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
    readiness_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": config.train.checkpoint_schema,
        "task_name": "known_gate",
        "variant": config.variant,
        "config_name": config.config_name,
        "config_identity_sha256": known_gate_config_identity_sha256(config),
        "config_file_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "readiness_sha256": readiness_sha256,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "training_state": dict(training_state),
        "full_parameter_training": True,
        "unknown_is_model_class": False,
    }


def _restore_training_state(
    payload: Mapping[str, Any], generator: torch.Generator
) -> dict[str, Any]:
    state = dict(payload.get("training_state") or {})
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_generator_state",
        "run_state",
    }
    if set(state) != required:
        raise ValueError("V25a checkpoint lacks complete epoch-boundary recovery state")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu_rng_state"], device="cpu"))
    cuda_states = list(state["torch_cuda_rng_states"] or ())
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("V25a checkpoint CUDA RNG state count differs from this host")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value, device="cpu") for value in cuda_states]
        )
    generator.set_state(torch.as_tensor(state["dataloader_generator_state"], device="cpu"))
    run_state = state["run_state"]
    if not isinstance(run_state, Mapping):
        raise ValueError("V25a checkpoint training_state lacks run_state")
    return dict(run_state)


def _reconcile_resumed_epoch_checkpoint_artifacts(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    history: Sequence[Mapping[str, Any]],
    best_epoch: int,
    resume_payload: Mapping[str, Any],
    last_checkpoint: Path,
    best_checkpoint: Path,
    epoch_checkpoint_dir: Path,
) -> dict[str, Any]:
    """Normalize recoverable epoch-boundary checkpoint files after resume.

    The trainer writes ``last_model.pt`` first, then the epoch checkpoint and
    (when selected) the best checkpoint, before promoting the telemetry file.
    A stop within that small write-only window leaves a complete in-memory
    epoch state in ``last_model.pt`` but can leave one or both companion
    checkpoint paths absent.  Rewriting the latest boundary from the loaded
    state is faithful; recreating an older missing epoch is not, so it fails
    closed.
    """

    checkpoint_epoch = int(resume_payload.get("epoch", -1))
    if (
        checkpoint_epoch < 1
        or len(history) != checkpoint_epoch
        or checkpoint_epoch != int(history[-1].get("epoch", -1))
        or checkpoint_epoch != int(resume_payload.get("metrics", {}).get("epoch", -1))
        or int(resume_payload.get("global_step", -1))
        != int(history[-1].get("global_step", -1))
        or not 1 <= int(best_epoch) <= checkpoint_epoch
    ):
        raise ValueError("V25a resumed checkpoint/history/best-epoch boundary is invalid")
    for row in history[:-1]:
        epoch = int(row.get("epoch", -1))
        path = epoch_checkpoint_dir / f"epoch-{epoch:03d}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                "V25a cannot reconstruct a missing older epoch checkpoint: "
                f"{path}"
            )

    selected = dict(history[int(best_epoch) - 1].get("checkpoint_selection") or {})
    if not selected:
        raise ValueError("V25a resumed history lacks the selected checkpoint rule")
    training_state = _capture_training_state(
        generator,
        run_state={
            "next_epoch": checkpoint_epoch + 1,
            "global_step": int(history[-1]["global_step"]),
            "best_epoch": int(best_epoch),
            "best_selection": selected,
            "history": [dict(row) for row in history],
        },
    )
    repaired_payload = _checkpoint_payload(
        model,
        optimizer,
        config=config,
        config_path=config_path,
        epoch=checkpoint_epoch,
        global_step=int(history[-1]["global_step"]),
        metrics=history[-1],
        training_state=training_state,
        readiness_sha256=readiness_sha256,
    )
    epoch_path = epoch_checkpoint_dir / f"epoch-{checkpoint_epoch:03d}.pt"
    _save_checkpoint(last_checkpoint, repaired_payload)
    _save_checkpoint(epoch_path, repaired_payload)
    rewritten_paths = [str(last_checkpoint), str(epoch_path)]
    if int(best_epoch) == checkpoint_epoch:
        _save_checkpoint(best_checkpoint, repaired_payload)
        rewritten_paths.append(str(best_checkpoint))
    elif not best_checkpoint.is_file():
        raise FileNotFoundError(
            "V25a resumed run is missing its earlier selected best checkpoint: "
            f"{best_checkpoint}"
        )
    return {
        "performed": True,
        "checkpoint_epoch": checkpoint_epoch,
        "best_epoch": int(best_epoch),
        "rewritten_paths": rewritten_paths,
        "older_epoch_checkpoints_preserved": checkpoint_epoch - 1,
        "reason": "normalize_checkpoint_bound_telemetry_and_repair_latest_epoch_boundary_files",
    }


def _source_v24_orders(readiness: Mapping[str, Any]) -> dict[int, str]:
    values = dict(readiness.get("source_v24_epoch_parent_order_hashes") or {})
    result = {int(epoch): str(value) for epoch, value in values.items()}
    if set(result) != set(range(1, 21)) or any(not value for value in result.values()):
        raise ValueError("V25a readiness lacks the complete frozen V24 parent order")
    return result


def _completed_summary(
    *,
    root: Path,
    config_path: Path,
    config: Any,
    artifact_dir: Path,
    readiness: Mapping[str, Any],
    contract: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: Any,
    validation_dataset: Any,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_selection: Mapping[str, Any],
    initial_state_sha256: str,
    paired_initialization: Mapping[str, Any],
    resumed_from: str | None,
    resume_checkpoint_recovery: Mapping[str, Any],
    telemetry: V25AGradientTelemetryRecorder,
    started: float,
    device: torch.device,
) -> dict[str, Any]:
    if not history or len(history) != 20 or best_epoch <= 0:
        raise ValueError("V25a cannot finalize an incomplete 20-epoch training run")
    best_checkpoint = root / config.train.checkpoint_path
    last_checkpoint = root / config.train.last_checkpoint_path
    epoch_checkpoint_dir = root / config.train.epoch_checkpoint_dir
    expected_epoch_checkpoints = [
        epoch_checkpoint_dir / f"epoch-{epoch:03d}.pt" for epoch in range(1, 21)
    ]
    if (
        not best_checkpoint.is_file()
        or not last_checkpoint.is_file()
        or any(not path.is_file() for path in expected_epoch_checkpoints)
    ):
        raise ValueError("V25a required checkpoints are missing")
    _load_checkpoint(
        best_checkpoint,
        model,
        None,
        config=config,
        config_path=config_path,
        readiness_sha256=str(readiness["sha256"]),
        map_location=device,
    )
    best_validation = evaluate_known_gate(
        model, validation_dataset, config=config, device=device
    )
    sample_detail = {
        "schema_version": "RADAR-LN3-V25a-known-gate-sample-detail-v1",
        "task_name": "known_gate",
        "variant": config.variant,
        "seed": config.train.seed,
        "dataloader_seed": config.train.dataloader_seed,
        "checkpoint": str(best_checkpoint),
        "epoch": best_epoch,
        "counts": {key: len(value) for key, value in best_validation["records"].items()},
        "records": best_validation["records"],
    }
    sample_detail_path = artifact_dir / "sample_detail_information.json"
    sample_detail_path.write_text(
        json.dumps(sample_detail, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    telemetry_summary = telemetry.finalize(completed_history=history)
    _assert_v24_history_order(
        history,
        _source_v24_orders(readiness),
    )
    performance = _performance(best_validation, config.train.task_min_precision_recall_target)
    final_state_sha256 = _state_dict_sha256(model.state_dict())
    return {
        "status": "completed",
        **performance,
        "task_name": "known_gate",
        "variant": config.variant,
        "seed": config.train.seed,
        "dataloader_seed": config.train.dataloader_seed,
        "config_name": config.config_name,
        "config_path": str(config_path),
        "device": str(device),
        "train_parent_count": len(train_dataset),
        "validation_parent_count": len(validation_dataset),
        "validation_clutter_support": 9,
        "global_step": int(history[-1]["global_step"]),
        "best_epoch": best_epoch,
        "best_checkpoint_selection": dict(best_selection),
        "best_validation_metrics": {
            key: value for key, value in best_validation.items() if key != "records"
        },
        "history": history,
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "sample_detail_information": str(sample_detail_path),
        "gradient_telemetry": telemetry_summary,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "full_parameter_training": True,
        "all_model_parameters_trainable": all(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "training_semantics_audit": {
            "optimizer": "torch.optim.AdamW(model.parameters(), lr=1e-4) defaults",
            "optimizer_weight_decay": V25A_ADAMW_DEFAULT_WEIGHT_DECAY,
            "global_gradient_clip_norm": V25A_GRADIENT_CLIP_NORM,
            "class_weight_normalization": V25A_CLASS_WEIGHT_NORMALIZATION,
            "telemetry_only_change": True,
            "telemetry_capture_timing": "after_loss_backward_before_existing_global_norm_clip",
            "telemetry_changes_gradients": False,
            "telemetry_changes_optimizer": False,
            "telemetry_changes_loss": False,
            "telemetry_changes_data_order": False,
        },
        "initial_model_state_sha256": initial_state_sha256,
        "best_model_state_sha256": final_state_sha256,
        "model_state_changed_from_initialization": initial_state_sha256 != final_state_sha256,
        "paired_initialization": dict(paired_initialization),
        "training_readiness": dict(readiness),
        "data_contract": dict(contract),
        "dataloader_contract": {
            "seed": config.train.dataloader_seed,
            "independent_generator": True,
            "source_v24_epoch_parent_order_hashes_exactly_reproduced": True,
            "epoch_parent_order_hashes": [
                {
                    "epoch": int(row["epoch"]),
                    "sha256": str(row["parent_order_sha256"]),
                    "count": int(row["parent_order_count"]),
                }
                for row in history
            ],
            "cuda_deterministic_algorithms": False,
            "determinism_claim": "fixed_initialization_and_parent_order_not_bitwise_cuda_determinism",
        },
        "resume": {
            "resumed": resumed_from is not None,
            "checkpoint": resumed_from,
            "epoch_boundary_only": True,
            "checkpoint_artifact_recovery": dict(resume_checkpoint_recovery),
        },
        "unknown_is_model_class": False,
        "local_argmax_is_product_action": False,
        "test_evaluation": "not_executed",
        "oof_threshold_calibration": "not_executed",
        "full_cascade_evaluation": "not_executed",
        "formal_rejection": "not_executed",
        "stable_evaluation": "not_executed",
        "p95_latency": "not_executed",
        "deployment_eligible": False,
        "deployment_blockers": [
            "gradient_telemetry_is_diagnostic_not_an_adopted_model_change",
            "multi_seed_confirmation_not_executed",
            "parent_grouped_oof_thresholds_not_fitted",
            "full_cascade_not_executed",
            "test_not_executed",
            "formal_rejection_not_executed",
            "stable_not_executed",
            "p95_not_executed",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }


def train(project_root: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = (root / CONFIG).resolve()
    config = load_known_gate_config(config_path)
    if config.variant != V25A_GRADIENT_TELEMETRY_VARIANT:
        raise ValueError("V25a trainer requires its telemetry-only config")
    _require_v24_equivalent_config(config)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    artifact_dir = (root / config.train.formal_artifact_dir).resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            "refusing to overwrite a non-empty V25a formal artifact; use "
            f"--resume-checkpoint at an epoch boundary or archive it: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact_dir / "training.log")
    started = time.perf_counter()
    try:
        seed_everything(config.train.seed)
        train_dataset = build_known_gate_dataset(root, config, split_name="train")
        validation_dataset = build_known_gate_dataset(root, config, split_name="val")
        if train_dataset.label_counts() != {
            KNOWN_GATE_CLASS_NAMES[0]: 953,
            KNOWN_GATE_CLASS_NAMES[1]: 33,
        } or validation_dataset.label_counts() != {
            KNOWN_GATE_CLASS_NAMES[0]: 223,
            KNOWN_GATE_CLASS_NAMES[1]: 9,
        }:
            raise ValueError("V25a KnownGate train/validation parent support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model, baseline_reference, paired_initialization = build_known_gate_model_for_config(root, config)
        if baseline_reference is None:
            raise AssertionError("V25a fresh paired reference is missing")
        initial_state_sha256 = _state_dict_sha256(model.state_dict())
        source_initial = str(readiness["source_v24_initial_model_state_sha256"])
        paired_v21_initial = str(
            readiness["paired_v21_seed43_baseline_initial_model_state_sha256"]
        )
        baseline_initial = _state_dict_sha256(baseline_reference.state_dict())
        if (
            initial_state_sha256 != source_initial
            or baseline_initial != paired_v21_initial
        ):
            raise AssertionError(
                "V25a initial state does not exactly reproduce the sealed V24/V21 pair"
            )
        paired_initialization.update(
            {
                "sealed_v24_initial_model_state_reproduced": True,
                "sealed_v24_initial_model_state_sha256": initial_state_sha256,
                "paired_v21_seed43_baseline_initial_model_state_reproduced": True,
                "paired_v21_seed43_baseline_initial_model_state_sha256": baseline_initial,
                "gradient_telemetry_only": True,
                "gradient_telemetry_changes_training_semantics": False,
            }
        )
        # Match V24's pre-training paired-initialization audit exactly.  This
        # runs before the model is moved to CUDA and consumes no training
        # update, but retaining it avoids silently changing the source
        # baseline's initialization/audit execution path.
        audit_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=min(config.train.batch_size, len(train_dataset)),
            shuffle=False,
        )
        probe = next(iter(audit_loader))
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
            raise AssertionError("V25a initial logits differ from the paired V24 reference")
        paired_initialization.update(
            {
                "initial_logits_exactly_equal": True,
                "initial_logits_sha256": hashlib.sha256(
                    treatment_logits.detach().cpu().contiguous().numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
        del audit_loader, probe, treatment_logits, baseline_logits
        del baseline_reference
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V25a must train every model parameter")
        device = known_gate_device(config)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        if any(
            abs(float(group.get("weight_decay", float("nan"))) - V25A_ADAMW_DEFAULT_WEIGHT_DECAY)
            > 1.0e-15
            for group in optimizer.param_groups
        ):
            raise AssertionError("V25a AdamW default weight decay differs from V24")
        class_weights = torch.tensor(
            config.train.class_weights, dtype=torch.float32, device=device
        )
        expected_orders = _source_v24_orders(readiness)
        telemetry = V25AGradientTelemetryRecorder(
            artifact_dir=artifact_dir,
            config_path=config_path,
            config_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            config_identity_sha256=known_gate_config_identity_sha256(config),
            source_summary_path=Path(str(readiness["source_v24_summary_path"])),
            source_summary_sha256=str(readiness["source_v24_summary_sha256"]),
            expected_parent_order_hashes=expected_orders,
        )
        best_checkpoint = root / config.train.checkpoint_path
        last_checkpoint = root / config.train.last_checkpoint_path
        epoch_checkpoint_dir = root / config.train.epoch_checkpoint_dir
        start_epoch = 1
        global_step = 0
        best_epoch = 0
        best_selection: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        resume_payload: Mapping[str, Any] | None = None
        resume_checkpoint_recovery: dict[str, Any] = {
            "performed": False,
            "reason": "fresh_training_run",
        }
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            if resume_path != last_checkpoint.resolve():
                raise PermissionError(
                    "V25a only resumes from its canonical last_model.pt so an older "
                    "epoch checkpoint cannot overwrite later checkpoint-bound telemetry"
                )
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=str(readiness["sha256"]),
                map_location=device,
            )
            run_state = _restore_training_state(payload, generator)
            start_epoch = int(run_state["next_epoch"])
            global_step = int(run_state["global_step"])
            best_epoch = int(run_state["best_epoch"])
            best_selection = run_state.get("best_selection")
            history = [dict(row) for row in list(run_state.get("history") or ())]
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V25a resume state is not an epoch boundary")
            if not 1 <= start_epoch <= config.train.epochs + 1:
                raise ValueError("V25a resume next_epoch is outside the fixed 20-epoch run")
            _assert_v24_history_order(history, expected_orders)
            resumed_from = str(resume_path)
            resume_payload = payload
        steps_per_epoch = len(train_loader)
        telemetry.start(
            model=model,
            steps_per_epoch=steps_per_epoch,
            initial_model_state_sha256=initial_state_sha256,
            completed_history=history,
            resumed=resumed_from is not None,
        )
        # ``start`` can promote the one checkpoint-bound pending telemetry file
        # left by an interruption after checkpoint persistence.  Normalize all
        # resumed history rows to final immutable descriptors before a later
        # checkpoint copies that history forward.
        for history_row in history:
            history_row["gradient_telemetry"] = telemetry.committed_epoch_descriptor(
                epoch=int(history_row["epoch"]),
                expected_global_step_end=int(history_row["global_step"]),
                expected_parent_order_sha256=str(history_row["parent_order_sha256"]),
            )
        _assert_v24_history_order(history, expected_orders)
        if resume_payload is not None:
            resume_checkpoint_recovery = _reconcile_resumed_epoch_checkpoint_artifacts(
                model=model,
                optimizer=optimizer,
                generator=generator,
                config=config,
                config_path=config_path,
                readiness_sha256=str(readiness["sha256"]),
                history=history,
                best_epoch=best_epoch,
                resume_payload=resume_payload,
                last_checkpoint=last_checkpoint,
                best_checkpoint=best_checkpoint,
                epoch_checkpoint_dir=epoch_checkpoint_dir,
            )
        logger.info(
            "run=start | task=known_gate | variant=%s | device=%s | train_parents=%d | "
            "val_parents=%d | epochs=20 | planned_steps=%d | telemetry=preclip_read_only | "
            "unknown_training_exposure=0",
            config.variant,
            device,
            len(train_dataset),
            len(validation_dataset),
            max(0, config.train.epochs - start_epoch + 1) * steps_per_epoch,
        )
        for epoch in range(start_epoch, config.train.epochs + 1):
            model.train()
            total_loss = 0.0
            final_loss = 0.0
            prefix_loss = 0.0
            gradient_max = 0.0
            gradient_clip_count = 0
            epoch_order: list[str] = []
            epoch_started = time.perf_counter()
            for step, batch in enumerate(train_loader, start=1):
                epoch_order.extend(str(value) for value in batch["parent_sample_ids"])
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
                    prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                    prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
                )
                loss = objective["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                record = telemetry.capture_after_backward(
                    batch=batch,
                    tensors=tensors,
                    objective=objective,
                    epoch=epoch,
                    step=step,
                    global_step=global_step + 1,
                )
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.train.gradient_clip_norm
                    ).item()
                )
                if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
                    raise ValueError("V25a training produced non-finite loss or gradient")
                telemetry.record_after_clip_before_optimizer_step(
                    record,
                    pre_clip_global_norm=grad_norm,
                    clip_threshold=config.train.gradient_clip_norm,
                )
                optimizer.step()
                global_step += 1
                total_loss += float(loss.item())
                final_loss += float(objective["final_state_loss"].item())
                prefix_loss += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, grad_norm)
                gradient_clip_count += int(grad_norm > config.train.gradient_clip_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | variant=%s | epoch=%02d/20 | step=%04d/%04d | "
                        "total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        config.variant,
                        epoch,
                        step,
                        steps_per_epoch,
                        float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()),
                        grad_norm,
                    )
            if (
                len(epoch_order) != len(train_dataset)
                or len(set(epoch_order)) != len(epoch_order)
            ):
                raise ValueError("V25a full epoch did not consume each train parent exactly once")
            parent_order_sha256 = _parent_order_sha256(epoch_order)
            if parent_order_sha256 != expected_orders[epoch]:
                raise AssertionError(
                    "V25a parent order differs from frozen V24 at epoch "
                    f"{epoch}: observed={parent_order_sha256}, expected={expected_orders[epoch]}"
                )
            validation = evaluate_known_gate(
                model, validation_dataset, config=config, device=device
            )
            selection = _selection(validation)
            selected_as_best = (
                best_selection is None
                or tuple(float(value) for value in selection["ranking_key"])
                > tuple(float(value) for value in best_selection["ranking_key"])
            )
            if selected_as_best:
                best_epoch = epoch
                best_selection = selection
            telemetry_staged = telemetry.stage_epoch(
                epoch=epoch,
                global_step=global_step,
                parent_order_sha256=parent_order_sha256,
            )
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": gradient_clip_count,
                "train_steps": steps_per_epoch,
                "selection_metric": KNOWN_GATE_SELECTION_METRIC,
                "checkpoint_selection": selection,
                "validation": {
                    key: value for key, value in validation.items() if key != "records"
                },
                "parent_order_sha256": parent_order_sha256,
                "parent_order_count": len(epoch_order),
                "parent_order_unique_count": len(set(epoch_order)),
                "gradient_telemetry": telemetry_staged,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            training_state = _capture_training_state(
                generator,
                run_state={
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_selection": best_selection,
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
                readiness_sha256=str(readiness["sha256"]),
            )
            _save_checkpoint(last_checkpoint, checkpoint)
            epoch_path = epoch_checkpoint_dir / f"epoch-{epoch:03d}.pt"
            _save_checkpoint(epoch_path, checkpoint)
            if selected_as_best:
                _save_checkpoint(best_checkpoint, checkpoint)
            telemetry_committed = telemetry.commit_epoch(
                epoch=epoch,
                staged=telemetry_staged,
                expected_global_step_end=global_step,
                expected_parent_order_sha256=parent_order_sha256,
            )
            epoch_record["gradient_telemetry"] = telemetry_committed
            # Replace the epoch-boundary checkpoints with their immutable final
            # telemetry path/hash.  The weights and optimizer state are identical.
            training_state = _capture_training_state(
                generator,
                run_state={
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_selection": best_selection,
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
                readiness_sha256=str(readiness["sha256"]),
            )
            _save_checkpoint(last_checkpoint, checkpoint)
            _save_checkpoint(epoch_path, checkpoint)
            if selected_as_best:
                _save_checkpoint(best_checkpoint, checkpoint)
            logger.info(
                "epoch | variant=%s | epoch=%02d/20 | loss=%.6f | selection=%s:%s | "
                "tie=%s:%s | best=%s | clips=%d | order_sha256=%s | elapsed=%s",
                config.variant,
                epoch,
                epoch_record["train_loss"],
                selection["primary_metric"],
                format_metric(selection["primary_value"]),
                selection["tie_break_metric"],
                format_metric(selection["tie_break_value"]),
                selected_as_best,
                gradient_clip_count,
                parent_order_sha256,
                format_duration(epoch_record["elapsed_seconds"]),
            )
        if best_selection is None:
            raise ValueError("V25a training did not select a best checkpoint")
        result = _completed_summary(
            root=root,
            config_path=config_path,
            config=config,
            artifact_dir=artifact_dir,
            readiness=readiness,
            contract=contract,
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            history=history,
            best_epoch=best_epoch,
            best_selection=best_selection,
            initial_state_sha256=initial_state_sha256,
            paired_initialization=paired_initialization,
            resumed_from=resumed_from,
            resume_checkpoint_recovery=resume_checkpoint_recovery,
            telemetry=telemetry,
            started=started,
            device=device,
        )
        (artifact_dir / "training-summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info(
            "run=complete | task=known_gate | variant=%s | status=completed | "
            "performance_status=%s | best_epoch=%02d | summary=%s",
            config.variant,
            result["performance_status"],
            best_epoch,
            artifact_dir / "training-summary.json",
        )
        return result
    except Exception as error:
        failure = {
            "status": "failed",
            "performance_status": "not_evaluated",
            "task_name": "known_gate",
            "variant": config.variant,
            "config_name": config.config_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "unknown_is_model_class": False,
            "test_evaluation": "not_executed",
            "oof_threshold_calibration": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "formal_rejection": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "deployment_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (artifact_dir / "training-summary.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.exception("run=failed | task=known_gate | variant=%s", config.variant)
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full 20-epoch V25a pre-clip gradient telemetry audit."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    resume = args.resume_checkpoint
    if resume is not None and not resume.is_absolute():
        resume = root / resume
    print(json.dumps(train(root, resume_checkpoint=resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()