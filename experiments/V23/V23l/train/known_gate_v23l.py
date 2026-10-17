"""Train V23l: official CfC with fixed positive-dt retention plus density."""

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
from radar_ln3.known_gate_official_cfc import build_official_cfc_model_from_frozen_source
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23e_selection import (
    assess_checkpoint,
    frozen_v23a_legacy_reference,
    is_better_candidate,
    selector_performance,
)
from radar_ln3.known_gate_v23l import (
    V23L_CHECKPOINT_SCHEMA,
    V23L_READINESS_SCHEMA,
    V23L_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    contract_relative,
    load_config,
    readiness_relative,
)
from radar_ln3.logging_utils import close_training_logger, configure_training_logger, format_duration, format_metric
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_common import (
    _capture_training_state,
    _model_inputs,
    _parent_order_sha256,
    _restore_training_state,
    _state_dict_sha256,
    _verify_contract,
    _verify_readiness,
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


def _verify_prepared(root: Path) -> tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]:
    config_path = root / config_relative()
    contract_path = root / contract_relative()
    readiness_path = root / readiness_relative()
    if not all(path.is_file() for path in (config_path, contract_path, readiness_path)):
        raise FileNotFoundError("V23l prepare artifacts are missing")
    config, source_config, source_path = load_config(root)
    readiness = dict(json.loads(readiness_path.read_text(encoding="utf-8")))
    contract = dict(json.loads(contract_path.read_text(encoding="utf-8")))
    failed = [name for name, value in dict(readiness.get("checks") or {}).items() if value is not True]
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != V23L_READINESS_SCHEMA
        or readiness.get("variant") != V23L_VARIANT
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or readiness.get("source_v23g_density") != config.source_v23g
        or readiness.get("adapter") != config.adapter.as_dict()
        or readiness.get("selector") != config.selector
        or int(readiness.get("train_parent_count", -1)) != 986
        or int(readiness.get("validation_parent_count", -1)) != 232
        or int(readiness.get("validation_clutter_parent_support", -1)) != 9
        or int(readiness.get("selected_seed", -1)) != 43
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False
        or failed
        or contract.get("schema_version")
        != "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-contract-v1"
        or dict(contract.get("arms") or {}).get("official_v23l", {}).get(
            "external_post_state_adapter"
        )
        != config.adapter.as_dict()
        or dict(contract.get("arms") or {}).get("self_developed_v23g", {}).get(
            "source"
        )
        != config.source_v23g
        or contract.get("unknown_training_exposure") != 0
        or contract.get("test") != "未执行"
        or contract.get("deployment_eligible") is not False
        or config.train.checkpoint_schema != V23L_CHECKPOINT_SCHEMA
        or config.train.epochs != 20
    ):
        raise ValueError("V23l readiness/contract is stale or outside sealed scope")
    # Re-run V21e's sealed input-contract checks before the official backbone
    # sees any payload.  This checks the exact V16 index, train-only
    # normalization and causal-course cache without opening Unknown or test.
    source_contract = _verify_contract(root, source_config)
    _verify_readiness(root, source_path, source_config, source_contract)
    return config, source_config, readiness, contract_path, contract


def _source_parent_orders(config: Any) -> tuple[str, ...]:
    values = tuple(str(value) for value in config.source_v23g["parent_order_sha256_by_epoch"])
    if len(values) != 20 or any(len(value) != 64 for value in values):
        raise ValueError("V23l source V23g parent-order sequence is invalid")
    return values


def _initialization_audit(
    root: Path, model: Any, source_config: Any
) -> dict[str, Any]:
    """Verify that V23l carries V21e's complete non-recurrent course model."""

    from radar_ln3.known_gate import build_known_gate_model_for_config

    source_model, _reference, source_pairing = build_known_gate_model_for_config(
        root, source_config
    )
    source_state = source_model.state_dict()
    model_state = model.state_dict()
    shared_names = [
        name
        for name in source_state
        if not name.startswith("liquid.") and not name.startswith("state_adapter.")
    ]
    if any(name not in model_state or not torch.equal(source_state[name], model_state[name]) for name in shared_names):
        raise RuntimeError("V23l non-recurrent V21e course initialization is not tensor-exact")
    source_hash = _state_dict_sha256(source_state)
    return {
        "strategy": "copy complete frozen V21e course-prefix non-recurrent state then seed official CfC",
        "source_v21e_initial_model_state_sha256": source_hash,
        "source_v21e_pairing": dict(source_pairing),
        "shared_non_recurrent_tensor_count": len(shared_names),
        "shared_non_recurrent_state_exactly_matches_v21e": True,
        "official_recurrent_state_is_distinct_by_architecture": True,
        "adapter_trainable_parameter_count": config_adapter_parameter_count(model),
        "initial_full_model_state_sha256": _state_dict_sha256(model_state),
    }


def config_adapter_parameter_count(model: Any) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("state_adapter.")
    )


def _checkpoint_payload(
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    initialization_audit: Mapping[str, Any],
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_schema": V23L_CHECKPOINT_SCHEMA,
        "task_name": "known_gate",
        "variant": V23L_VARIANT,
        "architecture": "official_ncps_cfc_external_fixed_retention_density_adapter_course_prefix",
        "config_name": config.config_name,
        "config_identity_sha256": config_identity_sha256(config),
        "config_file_sha256": sha256_file(config_path),
        "readiness_sha256": readiness_sha256,
        "official_cfc": model.backbone_specification(),
        "external_state_adapter": model.external_state_adapter_specification(),
        "source_v23g_density": config.source_v23g,
        "selector": config.selector,
        "initialization_audit": dict(initialization_audit),
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
    initialization_audit: Mapping[str, Any],
    map_location: torch.device,
) -> dict[str, Any]:
    payload = dict(torch.load(path, map_location=map_location, weights_only=False))
    if (
        payload.get("checkpoint_schema") != V23L_CHECKPOINT_SCHEMA
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != V23L_VARIANT
        or payload.get("architecture")
        != "official_ncps_cfc_external_fixed_retention_density_adapter_course_prefix"
        or payload.get("config_identity_sha256") != config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or payload.get("official_cfc") != model.backbone_specification()
        or payload.get("external_state_adapter")
        != model.external_state_adapter_specification()
        or payload.get("source_v23g_density") != config.source_v23g
        or payload.get("selector") != config.selector
        or dict(payload.get("initialization_audit") or {}) != dict(initialization_audit)
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V23l checkpoint schema/config/readiness binding differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _assert_parent_order_history(history: list[Mapping[str, Any]], source_orders: tuple[str, ...]) -> None:
    if len(history) > len(source_orders):
        raise ValueError("V23l history is longer than frozen V23g order contract")
    for epoch, record in enumerate(history, start=1):
        value = dict(record)
        if (
            int(value.get("epoch", -1)) != epoch
            or value.get("parent_order_sha256") != source_orders[epoch - 1]
            or value.get("v23g_density_parent_order_sha256") != source_orders[epoch - 1]
            or value.get("parent_order_exactly_matches_v23g_density") is not True
            or int(value.get("parent_order_count", -1)) != 986
            or int(value.get("parent_order_unique_count", -1)) != 986
        ):
            raise ValueError("V23l parent order differs from completed V23g density arm")


def _write_detail(
    path: Path,
    *,
    schema_suffix: str,
    model: Any,
    config: Any,
    checkpoint: Path,
    epoch: int,
    validation: Mapping[str, Any],
    scope: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": V23L_CHECKPOINT_SCHEMA + schema_suffix,
            "task_name": "known_gate",
            "variant": V23L_VARIANT,
            "architecture": "official_ncps_cfc_external_fixed_retention_density_adapter_course_prefix",
            "adapter": model.external_state_adapter_specification(),
            "source_v23g_density": config.source_v23g,
            "selector": config.selector,
            "seed": 43,
            "checkpoint": str(checkpoint),
            "epoch": int(epoch),
            "counts": {key: len(value) for key, value in validation["records"].items()},
            "records": validation["records"],
            "scope": scope,
        },
    )


def train(project_root: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    config, source_config, readiness, _contract_path, _contract = _verify_prepared(root)
    source_orders = _source_parent_orders(config)
    config_path = root / config_relative()
    artifact = root / artifact_relative()
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"refusing to overwrite non-empty V23l artifact: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name="radar_ln3.v23l")
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
            raise ValueError("V23l inherited V21e support changed")
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
        if (
            model.external_state_adapter_specification() != config.adapter.as_dict()
            or model.evidence_dim != 1
            or model.observation_intensity_specification() is not None
        ):
            raise RuntimeError("V23l constructed a mechanism different from its frozen contract")
        # V23a's bare official model has 27,434 parameters.  V23l preserves
        # V21e's one causal-course fusion input, which adds exactly one column
        # of 48 fusion weights and therefore has 27,482 parameters.
        if sum(parameter.numel() for parameter in model.parameters()) != 27482:
            raise RuntimeError("V23l parameter count differs from V23a plus V21e course input")
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V23l requires all model parameters trainable")
        initialization_audit = _initialization_audit(root, model, source_config)
        device = known_gate_device(source_config)
        if config.train.device == "cuda" and device.type != "cuda":
            raise RuntimeError("V23l formal training requires CUDA")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        readiness_sha = sha256_file(root / readiness_relative())
        legacy_reference = frozen_v23a_legacy_reference(root)
        best_checkpoint = artifact / "best_model.pt"
        last_checkpoint = artifact / "last_model.pt"
        start_epoch = 1
        global_step = 0
        best_epoch = 0
        best_selector: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            try:
                resume_path.relative_to(artifact)
            except ValueError as error:
                raise ValueError("V23l resume checkpoint must be inside its artifact") from error
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_sha,
                initialization_audit=initialization_audit,
                map_location=device,
            )
            state = _restore_training_state(payload, generator)
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            best_epoch = int(state["best_epoch"])
            best_selector = dict(state["best_selector"]) if state.get("best_selector") else None
            history = list(state.get("history") or [])
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V23l resume is not at an epoch boundary")
            _assert_parent_order_history(history, source_orders)
            resumed_from = str(resume_path)
        if not 1 <= start_epoch <= 21:
            raise ValueError("V23l resume epoch is outside fixed 20 epochs")
        logger.info(
            "run=start | task=known_gate | variant=%s | encoder=official_ncps_cfc | "
            "adapter=fixed_positive_tau_plus_zero_density | course_feature=1 | device=%s | "
            "train=%d | val=%d | epochs=20 | unknown=0 | test=0 | oof=deferred",
            V23L_VARIANT,
            device,
            len(train_dataset),
            len(validation_dataset),
        )
        steps_per_epoch = len(train_loader)
        for epoch in range(start_epoch, 21):
            model.train()
            epoch_started = time.perf_counter()
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
                    prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
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
                    raise ValueError("V23l produced non-finite loss or gradient")
                clips += int(grad_norm > config.train.gradient_clip_norm)
                optimizer.step()
                global_step += 1
                total_loss += float(loss.item())
                final_loss += float(objective["final_state_loss"].item())
                prefix_loss += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | variant=v23l_official_full_temporal | epoch=%02d/20 | "
                        "step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        epoch,
                        step,
                        steps_per_epoch,
                        float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()),
                        grad_norm,
                    )
            if len(order) != len(train_dataset) or len(set(order)) != len(order):
                raise ValueError("V23l epoch did not consume every train parent exactly once")
            order_sha = _parent_order_sha256(order)
            expected_order = source_orders[epoch - 1]
            if order_sha != expected_order:
                raise ValueError(
                    f"V23l DataLoader parent order differs from V23g at epoch {epoch}: "
                    f"actual={order_sha}, expected={expected_order}"
                )
            validation = evaluate_known_gate(
                model, validation_dataset, config=source_config, device=device
            )
            selector = assess_checkpoint(validation, reference=legacy_reference, epoch=epoch)
            selected = is_better_candidate(selector, best_selector)
            if selected:
                best_epoch, best_selector = epoch, selector
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": clips,
                "selection_metric": config.train.checkpoint_selection_metric,
                "checkpoint_selection": selector,
                "validation": {key: value for key, value in validation.items() if key != "records"},
                "parent_order_sha256": order_sha,
                "v23g_density_parent_order_sha256": expected_order,
                "parent_order_exactly_matches_v23g_density": True,
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
                "epoch | variant=v23l_official_full_temporal | epoch=%02d/20 | loss=%.6f | "
                "feasible=%s | prefix_clutter_recall=%s | prefix_clutter_f1=%s | best=%s | "
                "clips=%d | order_sha256=%s | elapsed=%s",
                epoch,
                record["train_loss"],
                selector["feasible"],
                format_metric(
                    selector["metrics"]["eligible_prefix_parent_balanced"]["per_class"][1]["recall"]
                ),
                format_metric(
                    selector["metrics"]["eligible_prefix_parent_balanced"]["per_class"][1]["f1"]
                ),
                selected,
                clips,
                order_sha,
                format_duration(record["elapsed_seconds"]),
            )
        _assert_parent_order_history(history, source_orders)
        epoch20_validation = evaluate_known_gate(
            model, validation_dataset, config=source_config, device=device
        )
        epoch20_detail = artifact / "epoch-020-sample-detail-information.json"
        _write_detail(
            epoch20_detail,
            schema_suffix="-epoch20-sample-detail-v1",
            model=model,
            config=config,
            checkpoint=last_checkpoint,
            epoch=20,
            validation=epoch20_validation,
            scope=(
                "fixed same-epoch V23l development diagnostic only; never a checkpoint-"
                "selection override, OOF or test"
            ),
        )
        common = {
            "task_name": "known_gate",
            "variant": V23L_VARIANT,
            "architecture": "official_ncps_cfc_external_fixed_retention_density_adapter_course_prefix",
            "official_cfc": model.backbone_specification(),
            "external_state_adapter": model.external_state_adapter_specification(),
            "source_v23g_density": config.source_v23g,
            "selector": config.selector,
            "seed": 43,
            "dataloader_seed": 43,
            "config_name": config.config_name,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_identity_sha256": config_identity_sha256(config),
            "train_parent_count": len(train_dataset),
            "validation_parent_count": len(validation_dataset),
            "validation_clutter_support": 9,
            "global_step": global_step,
            "history": history,
            "last_checkpoint": str(last_checkpoint),
            "epoch20_sample_detail_information": str(epoch20_detail),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "full_parameter_training": True,
            "all_model_parameters_trainable": True,
            "model_state_changed_from_initialization": (
                initialization_audit["initial_full_model_state_sha256"]
                != _state_dict_sha256(model.state_dict())
            ),
            "initialization_audit": initialization_audit,
            "source_copy_initialization_audit": source_copy_audit,
            "training_readiness": {
                "path": str(root / readiness_relative()),
                "sha256": readiness_sha,
                "schema_version": readiness["schema_version"],
                "status": "passed",
                "validation_used_for_readiness_gate": False,
                "test": "未执行",
            },
            "dataloader_contract": {
                "seed": 43,
                "independent_generator": True,
                "epoch_parent_order_hashes": [
                    {
                        "epoch": int(row["epoch"]),
                        "sha256": row["parent_order_sha256"],
                        "v23g_density_sha256": row["v23g_density_parent_order_sha256"],
                        "exactly_matches_v23g_density": row[
                            "parent_order_exactly_matches_v23g_density"
                        ],
                        "count": int(row["parent_order_count"]),
                    }
                    for row in history
                ],
                "all_epoch_parent_orders_exactly_match_v23g_density": True,
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
                initialization_audit=initialization_audit,
                map_location=device,
            )
            best_validation = evaluate_known_gate(
                model, validation_dataset, config=source_config, device=device
            )
            detail_path = artifact / "sample_detail_information.json"
            _write_detail(
                detail_path,
                schema_suffix="-sample-detail-v1",
                model=model,
                config=config,
                checkpoint=best_checkpoint,
                epoch=best_epoch,
                validation=best_validation,
                scope="development validation KnownGate only; V23e feasible checkpoint; not OOF or test",
            )
            result = {
                "status": "completed",
                **selector_performance(best_selector),
                **common,
                "best_epoch": best_epoch,
                "best_checkpoint_selection": best_selector,
                "best_validation_metrics": {
                    key: value for key, value in best_validation.items() if key != "records"
                },
                "best_checkpoint": str(best_checkpoint),
                "sample_detail_information": str(detail_path),
                "best_model_state_sha256": _state_dict_sha256(model.state_dict()),
            }
        else:
            result = {
                "status": "completed_no_feasible_checkpoint",
                "performance_status": "failed",
                **common,
                "best_epoch": None,
                "best_checkpoint_selection": None,
                "best_checkpoint": None,
                "sample_detail_information": None,
                "conclusion": (
                    "No V23l epoch met the frozen V23e feasibility constraints; no "
                    "best_model.pt was emitted; epoch-20 remains diagnostic only."
                ),
            }
        _write_json(artifact / "training-summary.json", result)
        logger.info(
            "run=complete | task=known_gate | variant=%s | status=%s | best_epoch=%s",
            V23L_VARIANT,
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
                "variant": V23L_VARIANT,
                "architecture": "official_ncps_cfc_external_fixed_retention_density_adapter_course_prefix",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "OOF": "未执行",
                "test_evaluation": "not_executed",
                "deployment_eligible": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        logger.exception("run=failed | task=known_gate | variant=%s", V23L_VARIANT)
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V23l official-CfC full temporal adapter KnownGate experiment.")
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
