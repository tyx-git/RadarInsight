"""Train V23j with only a pre-registered epoch-linear prefix-loss ramp."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    evaluate_known_gate,
    known_gate_device,
    seed_everything,
)
from radar_ln3.known_gate_official_cfc import (
    build_official_cfc_model_from_frozen_source,
)
from radar_ln3.known_gate_v23 import FIXED_EPOCHS, sha256_file
from radar_ln3.known_gate_v23_adapters import (
    source_runtime_report,
    source_v23_parent_order_hashes,
    source_v23_training,
)
from radar_ln3.known_gate_v23e_selection import (
    assess_checkpoint,
    frozen_v23a_legacy_reference,
    is_better_candidate,
    selector_performance,
)
from radar_ln3.known_gate_v23j import (
    V23J_CHECKPOINT_SCHEMA,
    V23J_VARIANT,
    artifact_relative,
    checkpoint_prefix_ramp_metadata,
    config_identity_sha256,
    config_relative,
    contract_relative,
    load_config,
    prefix_auxiliary_weight_for_epoch,
    readiness_relative,
    source_v23a_training,
    validate_checkpoint_prefix_ramp_metadata,
)
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_common import (
    _capture_training_state,
    _model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _state_dict_sha256,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _verify_prepared(
    root: Path,
) -> tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]:
    config_path = root / config_relative()
    contract_path = root / contract_relative()
    readiness_path = root / readiness_relative()
    if not all(path.is_file() for path in (config_path, contract_path, readiness_path)):
        raise FileNotFoundError("V23j prepare artifacts are missing")
    config, source_config, _source_path = load_config(root)
    readiness = dict(json.loads(readiness_path.read_text(encoding="utf-8")))
    contract = dict(json.loads(contract_path.read_text(encoding="utf-8")))
    failed_checks = [
        name
        for name, value in dict(readiness.get("checks") or {}).items()
        if value is not True
    ]
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-readiness-v1"
        or readiness.get("variant") != V23J_VARIANT
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or int(readiness.get("train_parent_count", -1)) != 986
        or int(readiness.get("validation_parent_count", -1)) != 232
        or int(readiness.get("validation_clutter_parent_support", -1)) != 9
        or int(readiness.get("selected_seed", -1)) != 43
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False
        or dict(readiness.get("prefix_ramp") or {})
        != config.prefix_ramp.as_dict()
        or failed_checks
        or contract.get("schema_version")
        != "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-contract-v1"
        or dict(contract.get("prefix_ramp") or {}) != config.prefix_ramp.as_dict()
        or contract.get("unknown_training_exposure") != 0
        or contract.get("test") != "未执行"
        or contract.get("deployment_eligible") is not False
        or config.train.checkpoint_schema != V23J_CHECKPOINT_SCHEMA
        or config.train.epochs != FIXED_EPOCHS
    ):
        raise ValueError("V23j readiness/contract is stale or outside sealed scope")
    return config, source_config, readiness, contract_path, contract


def _initialization_audit(
    model: Any, source_v23a_summary: Mapping[str, Any]
) -> dict[str, Any]:
    state_hash = _state_dict_sha256(model.state_dict())
    source_hash = str(source_v23a_summary["initial_model_state_sha256"])
    if state_hash != source_hash:
        raise RuntimeError("V23j initial model state does not exactly replay frozen V23a/V23")
    return {
        "source_v23a_initial_model_state_sha256": source_hash,
        "treatment_initial_model_state_sha256": state_hash,
        "initial_state_exactly_matches_v23_and_v23a": True,
        "only_training_objective_prefix_weight_schedule_differs_from_v23a": True,
    }


def _checkpoint_payload(
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    runtime_path: Path,
    runtime_sha256: str,
    initialization_audit: Mapping[str, Any],
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_schema": V23J_CHECKPOINT_SCHEMA,
        "task_name": "known_gate",
        "variant": V23J_VARIANT,
        "architecture": "official_ncps_cfc_external_state_adapter",
        "config_name": config.config_name,
        "config_identity_sha256": config_identity_sha256(config),
        "config_file_sha256": sha256_file(config_path),
        "readiness_sha256": readiness_sha256,
        "runtime_compatibility": {"path": str(runtime_path), "sha256": runtime_sha256},
        "official_cfc": model.backbone_specification(),
        "external_state_adapter": model.external_state_adapter_specification(),
        "initialization_audit": dict(initialization_audit),
        "selector": config.selector,
        **checkpoint_prefix_ramp_metadata(epoch=epoch, ramp=config.prefix_ramp),
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
    model: Any,
    optimizer: torch.optim.Optimizer | None,
    *,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    runtime_path: Path,
    runtime_sha256: str,
    initialization_audit: Mapping[str, Any],
    map_location: torch.device,
) -> dict[str, Any]:
    payload = dict(torch.load(path, map_location=map_location, weights_only=False))
    if (
        payload.get("checkpoint_schema") != V23J_CHECKPOINT_SCHEMA
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != V23J_VARIANT
        or payload.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or payload.get("config_identity_sha256") != config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or dict(payload.get("runtime_compatibility") or {})
        != {"path": str(runtime_path), "sha256": runtime_sha256}
        or payload.get("official_cfc") != model.backbone_specification()
        or payload.get("external_state_adapter")
        != model.external_state_adapter_specification()
        or dict(payload.get("initialization_audit") or {})
        != dict(initialization_audit)
        or dict(payload.get("selector") or {}) != config.selector
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V23j checkpoint schema/config/runtime/readiness binding differs")
    epoch = int(payload.get("epoch", -1))
    validate_checkpoint_prefix_ramp_metadata(
        payload, epoch=epoch, ramp=config.prefix_ramp
    )
    metrics = dict(payload.get("metrics") or {})
    expected_weight = prefix_auxiliary_weight_for_epoch(epoch, config.prefix_ramp)
    if float(metrics.get("prefix_auxiliary_weight", -1.0)) != expected_weight:
        raise ValueError("V23j checkpoint metrics lack the correct epoch prefix weight")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _assert_parent_order_history(
    history: list[Mapping[str, Any]], source_hashes: tuple[str, ...], config: Any
) -> None:
    if len(history) > len(source_hashes):
        raise ValueError("V23j history is longer than frozen V23 order contract")
    for expected_epoch, row in enumerate(history, start=1):
        record = dict(row)
        expected_weight = prefix_auxiliary_weight_for_epoch(
            expected_epoch, config.prefix_ramp
        )
        if (
            int(record.get("epoch", -1)) != expected_epoch
            or record.get("parent_order_sha256") != source_hashes[expected_epoch - 1]
            or record.get("v23_control_parent_order_sha256")
            != source_hashes[expected_epoch - 1]
            or record.get("parent_order_exactly_matches_v23_control") is not True
            or int(record.get("parent_order_count", -1)) != 986
            or int(record.get("parent_order_unique_count", -1)) != 986
            or float(record.get("prefix_auxiliary_weight", -1.0)) != expected_weight
        ):
            raise ValueError("V23j parent order or prefix-ramp history differs from frozen contract")


def _epoch_one_pairing_audit(
    history: list[Mapping[str, Any]],
    source_v23a_summary: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Verify the V23a-identical first epoch before reporting a V23j run.

    The schedule's first coefficient is exactly V23a's 0.1, and both runs use
    the frozen initial state and identical parent order.  This audit makes a
    drift in the epoch-one implementation fail closed rather than relying only
    on the synthetic smoke batch.
    """

    if not history:
        return {
            "available": False,
            # The official-CfC path intentionally runs with CUDA deterministic
            # algorithms disabled.  Epoch-one numerical equality is therefore
            # diagnostic only; the reproducibility contract is the frozen
            # initialization, objective, and parent order.
            "numerical_epoch_one_loss_replay_required": False,
        }
    candidate = dict(history[0])
    source_history = list(source_v23a_summary.get("history") or ())
    if not source_history:
        raise ValueError("V23j V23a source history is missing")
    source = dict(source_history[0])
    expected_weight = prefix_auxiliary_weight_for_epoch(1, config.prefix_ramp)
    required_equal = (
        int(candidate.get("epoch", -1)) == 1,
        int(source.get("epoch", -1)) == 1,
        float(candidate.get("prefix_auxiliary_weight", -1.0)) == expected_weight,
        candidate.get("parent_order_sha256") == source.get("parent_order_sha256"),
        candidate.get("v23_control_parent_order_sha256")
        == source.get("v23_control_parent_order_sha256"),
        int(candidate.get("parent_order_count", -1))
        == int(source.get("parent_order_count", -2))
        == 986,
        int(candidate.get("parent_order_unique_count", -1))
        == int(source.get("parent_order_unique_count", -2))
        == 986,
    )
    if not all(required_equal):
        raise ValueError("V23j epoch-one replay differs from frozen V23a/V23")
    numerical_loss_replay = (
        float(candidate.get("train_loss", float("nan")))
        == float(source.get("train_loss", float("nan")))
        and float(candidate.get("train_final_state_loss", float("nan")))
        == float(source.get("train_final_state_loss", float("nan")))
        and float(candidate.get("train_prefix_auxiliary_loss", float("nan")))
        == float(source.get("train_prefix_auxiliary_loss", float("nan")))
    )
    return {
        "available": True,
        "numerical_epoch_one_loss_replay_required": False,
        "epoch": 1,
        "prefix_auxiliary_weight": expected_weight,
        "parent_order_sha256": candidate["parent_order_sha256"],
        "train_loss": candidate["train_loss"],
        "train_final_state_loss": candidate["train_final_state_loss"],
        "train_prefix_auxiliary_loss": candidate["train_prefix_auxiliary_loss"],
        "structural_pairing_exactly_matches_frozen_v23a_epoch_one": True,
        "numerical_loss_replay_matches_frozen_v23a_epoch_one": numerical_loss_replay,
        "cuda_deterministic_algorithms": bool(
            config.train.cuda_deterministic_algorithms
        ),
    }


def train(project_root: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    config, source_config, readiness, _contract_path, _contract = _verify_prepared(root)
    runtime_path, runtime, runtime_sha = source_runtime_report(root)
    (
        _source_v23_summary_path,
        source_v23_summary,
        _source_checkpoint_path,
        _source_checkpoint_sha,
    ) = source_v23_training(root)
    (
        _source_v23a_config_path,
        _source_v23a_config,
        _source_v23a_summary_path,
        source_v23a_summary,
        _source_v23a_checkpoint_path,
    ) = source_v23a_training(root)
    source_parent_order_hashes = source_v23_parent_order_hashes(source_v23_summary)
    config_path = root / config_relative()
    artifact = root / artifact_relative()
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"refusing to overwrite non-empty V23j artifact: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name="radar_ln3.v23j")
    started = time.perf_counter()
    try:
        seed_everything(config.train.seed)
        train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
        validation_dataset = build_known_gate_dataset(root, source_config, split_name="val")
        if (
            train_dataset.label_counts()
            != {KNOWN_GATE_CLASS_NAMES[0]: 953, KNOWN_GATE_CLASS_NAMES[1]: 33}
            or validation_dataset.label_counts()
            != {KNOWN_GATE_CLASS_NAMES[0]: 223, KNOWN_GATE_CLASS_NAMES[1]: 9}
        ):
            raise ValueError("V23j inherited V21 support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model, source_copy_audit = build_official_cfc_model_from_frozen_source(
            root, source_config, state_adapter_spec=config.adapter
        )
        if model.external_state_adapter_specification() != config.adapter.as_dict():
            raise RuntimeError("V23j constructed an adapter different from frozen V23a")
        if sum(parameter.numel() for parameter in model.parameters()) != 27434:
            raise RuntimeError("V23j parameter count differs from V23a")
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V23j requires all model parameters trainable")
        initialization_audit = _initialization_audit(model, source_v23a_summary)
        device = known_gate_device(source_config)
        if config.train.device == "cuda" and device.type != "cuda":
            raise RuntimeError("V23j formal training requires CUDA")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(
            config.train.class_weights, dtype=torch.float32, device=device
        )
        readiness_sha = sha256_file(root / readiness_relative())
        legacy_reference = frozen_v23a_legacy_reference(root)
        best_checkpoint, last_checkpoint = artifact / "best_model.pt", artifact / "last_model.pt"
        start_epoch = global_step = best_epoch = 0
        start_epoch = 1
        best_selector: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            if resume_path != last_checkpoint.resolve():
                raise ValueError(
                    "V23j resume must use its current last_model.pt so the restored "
                    "best-selector state cannot disagree with an older epoch checkpoint"
                )
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_sha,
                runtime_path=runtime_path,
                runtime_sha256=runtime_sha,
                initialization_audit=initialization_audit,
                map_location=device,
            )
            state = _restore_training_state(payload, generator)
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            best_epoch = int(state["best_epoch"])
            best_selector = (
                dict(state["best_selector"]) if state.get("best_selector") else None
            )
            history = list(state.get("history") or [])
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V23j resume is not at an epoch boundary")
            _assert_parent_order_history(history, source_parent_order_hashes, config)
            resumed_from = str(resume_path)
        if not 1 <= start_epoch <= FIXED_EPOCHS + 1:
            raise ValueError("V23j resume epoch is outside fixed 20 epochs")
        if history:
            _epoch_one_pairing_audit(history, source_v23a_summary, config)
        logger.info(
            "run=start | task=known_gate | variant=%s | encoder=official_ncps_cfc_external_state_adapter | prefix_ramp=linear_0.1_to_1.0 | device=%s | train=%d | val=%d | epochs=20 | unknown=0 | test=0 | oof=deferred",
            V23J_VARIANT,
            device,
            len(train_dataset),
            len(validation_dataset),
        )
        steps_per_epoch = len(train_loader)
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            model.train()
            epoch_started = time.perf_counter()
            prefix_weight = prefix_auxiliary_weight_for_epoch(epoch, config.prefix_ramp)
            total_loss = final_loss = prefix_loss = 0.0
            gradient_max = 0.0
            clips = 0
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
                    prefix_auxiliary_weight=prefix_weight,
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
                if not bool(torch.isfinite(loss)) or not np.isfinite(grad_norm):
                    raise ValueError("V23j produced non-finite loss or gradient")
                clips += int(grad_norm > config.train.gradient_clip_norm)
                optimizer.step()
                global_step += 1
                total_loss += float(loss.item())
                final_loss += float(objective["final_state_loss"].item())
                prefix_loss += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | variant=v23j_prefix_ramp | epoch=%02d/20 | prefix_weight=%.12f | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        epoch,
                        prefix_weight,
                        step,
                        steps_per_epoch,
                        float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()),
                        grad_norm,
                    )
            if len(order) != len(train_dataset) or len(set(order)) != len(order):
                raise ValueError("V23j epoch did not consume every train parent exactly once")
            order_sha = _parent_order_sha256(order)
            expected_order_sha = source_parent_order_hashes[epoch - 1]
            if order_sha != expected_order_sha:
                raise ValueError(
                    f"V23j DataLoader parent order differs from frozen V23 at epoch {epoch}: "
                    f"actual={order_sha}, expected={expected_order_sha}"
                )
            validation = evaluate_known_gate(
                model, validation_dataset, config=source_config, device=device
            )
            selector = assess_checkpoint(
                validation, reference=legacy_reference, epoch=epoch
            )
            selected = is_better_candidate(selector, best_selector)
            if selected:
                best_epoch, best_selector = epoch, selector
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "prefix_auxiliary_weight": prefix_weight,
                "prefix_ramp": config.prefix_ramp.as_dict(),
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": clips,
                "selection_metric": config.train.checkpoint_selection_metric,
                "checkpoint_selection": selector,
                "validation": {
                    key: value for key, value in validation.items() if key != "records"
                },
                "parent_order_sha256": order_sha,
                "v23_control_parent_order_sha256": expected_order_sha,
                "parent_order_exactly_matches_v23_control": True,
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(record)
            training_state = _capture_training_state(
                generator,
                run_state={
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_selector": best_selector,
                    "history": history,
                },
            )
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_sha,
                runtime_path=runtime_path,
                runtime_sha256=runtime_sha,
                initialization_audit=initialization_audit,
                epoch=epoch,
                global_step=global_step,
                metrics=record,
                training_state=training_state,
            )
            _save_checkpoint(last_checkpoint, checkpoint)
            _save_checkpoint(artifact / "epochs" / f"epoch-{epoch:03d}.pt", checkpoint)
            if selected:
                _save_checkpoint(best_checkpoint, checkpoint)
            logger.info(
                "epoch | variant=v23j_prefix_ramp | epoch=%02d/20 | prefix_weight=%.12f | loss=%.6f | feasible=%s | prefix_clutter_recall=%s | prefix_clutter_f1=%s | best=%s | clips=%d | order_sha256=%s | elapsed=%s",
                epoch,
                prefix_weight,
                record["train_loss"],
                selector["feasible"],
                format_metric(
                    selector["metrics"]["eligible_prefix_parent_balanced"]["per_class"][1][
                        "recall"
                    ]
                ),
                format_metric(
                    selector["metrics"]["eligible_prefix_parent_balanced"]["per_class"][1][
                        "f1"
                    ]
                ),
                selected,
                clips,
                order_sha,
                format_duration(record["elapsed_seconds"]),
            )
        _assert_parent_order_history(history, source_parent_order_hashes, config)
        epoch_one_pairing = _epoch_one_pairing_audit(
            history, source_v23a_summary, config
        )
        common = {
            "task_name": "known_gate",
            "variant": V23J_VARIANT,
            "architecture": "official_ncps_cfc_external_state_adapter",
            "selector": config.selector,
            "prefix_ramp": config.prefix_ramp.as_dict(),
            "prefix_ramp_schedule_by_epoch": [
                {
                    "epoch": epoch,
                    "prefix_auxiliary_weight": prefix_auxiliary_weight_for_epoch(
                        epoch, config.prefix_ramp
                    ),
                }
                for epoch in range(1, FIXED_EPOCHS + 1)
            ],
            "seed": 43,
            "dataloader_seed": 43,
            "config_name": config.config_name,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_identity_sha256": config_identity_sha256(config),
            "source_control": config.source_v23a,
            "train_parent_count": len(train_dataset),
            "validation_parent_count": len(validation_dataset),
            "validation_clutter_support": 9,
            "global_step": global_step,
            "history": history,
            "last_checkpoint": str(last_checkpoint),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "full_parameter_training": True,
            "all_model_parameters_trainable": True,
            "initial_model_state_sha256": initialization_audit[
                "treatment_initial_model_state_sha256"
            ],
            "initialization_audit": initialization_audit,
            "epoch_one_v23a_pairing_audit": epoch_one_pairing,
            "source_copy_initialization_audit": source_copy_audit,
            "training_readiness": {
                "path": str(root / readiness_relative()),
                "sha256": readiness_sha,
                "schema_version": readiness["schema_version"],
                "status": "passed",
                "validation_used_for_readiness_gate": False,
                "test": "未执行",
            },
            "official_runtime_compatibility": {
                "path": str(runtime_path),
                "sha256": runtime_sha,
                "schema_version": runtime["schema_version"],
                "checks": dict(runtime["checks"]),
            },
            "data_contract": {
                "source_data_contract_path": source_config.data.contract_path,
                "source_data_contract_sha256": source_config.data.expected_contract_sha256,
                "normalization_path": source_config.data.normalization_stats_path,
                "normalization_sha256": source_config.data.expected_stats_sha256,
                "index_path": source_config.data.index_path,
                "index_sha256": source_config.data.expected_index_sha256,
            },
            "dataloader_contract": {
                "seed": 43,
                "independent_generator": True,
                "epoch_parent_order_hashes": [
                    {
                        "epoch": int(row["epoch"]),
                        "sha256": row["parent_order_sha256"],
                        "v23_control_sha256": row["v23_control_parent_order_sha256"],
                        "exactly_matches_v23_control": row[
                            "parent_order_exactly_matches_v23_control"
                        ],
                        "count": int(row["parent_order_count"]),
                    }
                    for row in history
                ],
                "all_epoch_parent_orders_exactly_match_v23_control": True,
                "cuda_deterministic_algorithms": False,
            },
            "resume": {
                "resumed": resumed_from is not None,
                "checkpoint": resumed_from,
                "start_epoch": start_epoch,
                "epoch_boundary_only": True,
            },
            "unknown_is_model_class": False,
            "local_argmax_is_product_action": False,
            "validation_interpretation": "development-only KnownGate local evidence, not final product action",
            "multiseed": "未执行",
            "OOF": "未执行",
            "test_evaluation": "not_executed",
            "oof_threshold_calibration": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "formal_rejection": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "deployment_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if best_checkpoint.is_file() and best_epoch > 0 and best_selector is not None:
            _load_checkpoint(
                best_checkpoint,
                model,
                None,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_sha,
                runtime_path=runtime_path,
                runtime_sha256=runtime_sha,
                initialization_audit=initialization_audit,
                map_location=device,
            )
            best_validation = evaluate_known_gate(
                model, validation_dataset, config=source_config, device=device
            )
            final_state_sha = _state_dict_sha256(model.state_dict())
            detail_path = artifact / "sample_detail_information.json"
            _write_json(
                detail_path,
                {
                    "schema_version": V23J_CHECKPOINT_SCHEMA + "-sample-detail-v1",
                    "task_name": "known_gate",
                    "variant": V23J_VARIANT,
                    "architecture": "official_ncps_cfc_external_state_adapter",
                    "adapter": model.external_state_adapter_specification(),
                    "selector": config.selector,
                    "prefix_ramp": config.prefix_ramp.as_dict(),
                    "prefix_auxiliary_weight_for_epoch": prefix_auxiliary_weight_for_epoch(
                        best_epoch, config.prefix_ramp
                    ),
                    "seed": 43,
                    "checkpoint": str(best_checkpoint),
                    "epoch": best_epoch,
                    "counts": {
                        key: len(value) for key, value in best_validation["records"].items()
                    },
                    "records": best_validation["records"],
                    "scope": "development validation KnownGate only; V23e selector feasible checkpoint; not OOF or test",
                },
            )
            performance = selector_performance(best_selector)
            result = {
                "status": "completed",
                **performance,
                **common,
                "official_cfc": model.backbone_specification(),
                "external_state_adapter": model.external_state_adapter_specification(),
                "best_epoch": best_epoch,
                "best_checkpoint_selection": best_selector,
                "best_validation_metrics": {
                    key: value for key, value in best_validation.items() if key != "records"
                },
                "best_checkpoint": str(best_checkpoint),
                "sample_detail_information": str(detail_path),
                "best_model_state_sha256": final_state_sha,
                "model_state_changed_from_initialization": initialization_audit[
                    "treatment_initial_model_state_sha256"
                ]
                != final_state_sha,
            }
        else:
            final_state_sha = _state_dict_sha256(model.state_dict())
            result = {
                "status": "completed_no_feasible_checkpoint",
                "performance_status": "failed",
                **common,
                "best_epoch": None,
                "best_checkpoint_selection": None,
                "best_checkpoint": None,
                "sample_detail_information": None,
                "last_model_state_sha256": final_state_sha,
                "model_state_changed_from_initialization": initialization_audit[
                    "treatment_initial_model_state_sha256"
                ]
                != final_state_sha,
                "conclusion": "No V23j epoch met the frozen V23e feasibility constraints; no best_model.pt was emitted.",
            }
        _write_json(artifact / "training-summary.json", result)
        logger.info(
            "run=complete | task=known_gate | variant=%s | status=%s | best_epoch=%s",
            V23J_VARIANT,
            result["status"],
            result["best_epoch"],
        )
        return result
    except Exception as error:
        _write_json(
            artifact / "training-summary.json",
            {
                "status": "failed",
                "performance_status": "not_evaluated",
                "task_name": "known_gate",
                "variant": V23J_VARIANT,
                "architecture": "official_ncps_cfc_external_state_adapter",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "OOF": "未执行",
                "test_evaluation": "not_executed",
                "deployment_eligible": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        logger.exception("run=failed | task=known_gate | variant=%s", V23J_VARIANT)
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V23j prefix-ramp official-CfC experiment.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    resume = (
        args.resume_checkpoint
        if args.resume_checkpoint is None or args.resume_checkpoint.is_absolute()
        else root / args.resume_checkpoint
    )
    print(json.dumps(train(root, resume_checkpoint=resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
