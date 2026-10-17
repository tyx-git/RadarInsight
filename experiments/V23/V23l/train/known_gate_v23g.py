"""Train V23g: V21e self-developed KnownGate with density-weighted updates only."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KnownGateClassifier,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    evaluate_known_gate,
    known_gate_device,
    seed_everything,
)
from radar_ln3.known_gate_v23g import (
    V23G_CHECKPOINT_SCHEMA,
    V23G_EPOCHS,
    V23G_READINESS_SCHEMA,
    V23G_TRAINING_SEED,
    V23G_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    contract_relative,
    load_config,
    readiness_relative,
    sha256_file,
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
    _is_better,
    _model_inputs,
    _parent_order_sha256,
    _performance,
    _restore_training_state,
    _selection,
    _state_dict_sha256,
    _verify_contract,
    _verify_readiness,
)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


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


def _density_config(source_config: Any, wrapper: Any) -> Any:
    """Derive the executable KnownGate config without using a new data schema."""

    expected_model = dict(wrapper.model)
    expected_train = dict(wrapper.train)
    model = replace(
        source_config.model,
        liquid_invalid_dt_update_mode="density_weighted",
    )
    train = replace(
        source_config.train,
        checkpoint_path=str(expected_train["checkpoint_path"]),
        last_checkpoint_path=str(expected_train["last_checkpoint_path"]),
        formal_artifact_dir=str(expected_train["formal_artifact_dir"]),
        epoch_checkpoint_dir=str(expected_train["epoch_checkpoint_dir"]),
        checkpoint_schema=str(expected_train["checkpoint_schema"]),
        paired_baseline_config_path=str(expected_train["paired_baseline_config_path"]),
    )
    actual_model = dict(model.__dict__)
    if actual_model.get("rd_temporal_residual_dim") == 0 and (
        "rd_temporal_residual_dim" not in expected_model
    ):
        actual_model.pop("rd_temporal_residual_dim")
    if actual_model != expected_model:
        raise ValueError("V23g executable model config differs from prepared wrapper")
    actual_train = dict(train.__dict__)
    # The sealed V21e file predates explicit serialization of the historical
    # default.  Runtime parsing makes it explicit, so normalize only that
    # representation difference before checking the frozen protocol.
    if expected_train.get("class_weight_normalization") is None:
        expected_train["class_weight_normalization"] = "batch_weight_sum"
    expected_train["class_weights"] = tuple(expected_train["class_weights"])
    if actual_train != expected_train:
        raise ValueError("V23g executable train config differs from prepared wrapper")
    return replace(
        source_config,
        schema_version="RADAR-LN3-V23g-runtime",
        config_name=str(wrapper.config_name),
        variant=V23G_VARIANT,
        model=model,
        train=train,
    )


def _control_parent_orders(control: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(str(value) for value in control["parent_order_sha256_by_epoch"])
    if len(values) != V23G_EPOCHS or any(len(value) != 64 for value in values):
        raise ValueError("V23g direct control has an invalid parent-order sequence")
    return values


def _build_density_model(root: Path, source_config: Any, treatment_config: Any) -> tuple[KnownGateClassifier, dict[str, Any]]:
    """Reproduce V21e's paired initialization, then switch only its update mode."""

    model, _baseline, source_pairing = build_known_gate_model_for_config(root, source_config)
    model.config = treatment_config
    model.liquid.invalid_dt_update_mode = "density_weighted"
    if model.liquid.invalid_dt_observation_gain != 0.25:
        raise ValueError("V23g requires the inherited fixed 0.25 observation gain")
    if model.config.model.liquid_invalid_dt_update_mode != "density_weighted":
        raise RuntimeError("V23g model configuration did not switch to density_weighted")
    return model, dict(source_pairing)


def _first_density_exposure_audit(
    root: Path,
    source_config: Any,
    treatment_config: Any,
    control_initial_sha256: str,
) -> dict[str, Any]:
    """Audit exact pre-zero replay and real first-zero density divergence."""

    control, _control_baseline, _control_pairing = build_known_gate_model_for_config(
        root, source_config
    )
    treatment, treatment_pairing = _build_density_model(
        root, source_config, treatment_config
    )
    control_hash = _state_dict_sha256(control.state_dict())
    treatment_hash = _state_dict_sha256(treatment.state_dict())
    if control_hash != control_initial_sha256 or treatment_hash != control_initial_sha256:
        raise RuntimeError("V23g cannot reproduce frozen V21e initial parameter state")
    dataset = build_known_gate_dataset(root, source_config, split_name="train")
    selected = None
    zero_index = None
    for item in dataset:
        for event_index, reason in enumerate(item.base.tr_dt_reason.tolist()):
            if reason == 2:
                selected = item
                zero_index = event_index
                break
        if selected is not None:
            break
    if selected is None or zero_index is None:
        raise RuntimeError("V23g train population has no same-second follow-up")
    batch = next(iter(build_known_gate_dataloader([selected], batch_size=1, shuffle=False)))
    control.eval()
    treatment.eval()
    with torch.inference_mode():
        control_output = control(**_model_inputs(batch))
        treatment_output = treatment(**_model_inputs(batch))
    pre_zero = float(
        (
            control_output["liquid_sequence"][:, :zero_index]
            - treatment_output["liquid_sequence"][:, :zero_index]
        )
        .abs()
        .max()
        .item()
        if zero_index > 0
        else 0.0
    )
    first_zero_difference = float(
        (
            control_output["liquid_sequence"][:, zero_index]
            - treatment_output["liquid_sequence"][:, zero_index]
        )
        .abs()
        .max()
        .item()
    )
    if pre_zero > 1.0e-6:
        raise RuntimeError("V23g differs from fixed-gain control before same-second exposure")
    if first_zero_difference <= 1.0e-8:
        raise RuntimeError("V23g density mode did not affect a real same-second event")
    return {
        "source_v21e_initial_model_state_sha256": control_initial_sha256,
        "reproduced_control_initial_model_state_sha256": control_hash,
        "treatment_initial_model_state_sha256": treatment_hash,
        "full_initial_state_exactly_matches_v21e": True,
        "source_v21e_pairing": treatment_pairing,
        "pre_first_zero_delta_replays_fixed_gain_control": True,
        "first_zero_delta_differs_from_fixed_gain_control": True,
        "max_state_error_before_first_zero_delta": pre_zero,
        "first_zero_delta_state_difference": first_zero_difference,
        "tolerance": {"atol": 1.0e-6, "rtol": 1.0e-6},
        "train_parent_id": str(batch["parent_sample_ids"][0]),
        "first_zero_delta_event_index": int(zero_index),
    }


def _verify_prepared(root: Path) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    wrapper, source_config, source_path = load_config(root)
    config_path = root / config_relative()
    contract_path = root / contract_relative()
    readiness_path = root / readiness_relative()
    if not all(path.is_file() for path in (config_path, contract_path, readiness_path)):
        raise FileNotFoundError("V23g preparation artifacts are missing")
    readiness = _read_json(readiness_path)
    contract = _read_json(contract_path)
    failed = [name for name, value in dict(readiness.get("checks") or {}).items() if value is not True]
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != V23G_READINESS_SCHEMA
        or readiness.get("variant") != V23G_VARIANT
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("config_identity_sha256") != config_identity_sha256(wrapper)
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or int(readiness.get("train_parent_count", -1)) != 986
        or int(readiness.get("validation_parent_count", -1)) != 232
        or int(readiness.get("validation_clutter_parent_support", -1)) != 9
        or int(readiness.get("selected_seed", -1)) != 43
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False
        or failed
        or contract.get("schema_version") != "RADAR-LN3-V23g-known-gate-density-weighted-contract-v1"
        or contract.get("source_control") != wrapper.source_control
        or contract.get("unknown_training_exposure") != 0
        or contract.get("test") != "未执行"
        or contract.get("deployment_eligible") is not False
    ):
        raise ValueError("V23g readiness/contract is stale or outside sealed scope")
    # Re-run the already sealed V21e input contract checks.  It validates the
    # exact V16 index, known-only normalization and course cache, without
    # opening Unknown/test payloads.
    source_contract = _verify_contract(root, source_config)
    _verify_readiness(root, source_path, source_config, source_contract)
    return wrapper, source_config, _density_config(source_config, wrapper), readiness, contract


def _checkpoint_payload(
    model: KnownGateClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    wrapper: Any,
    config_path: Path,
    readiness_sha256: str,
    initialization_audit: Mapping[str, Any],
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_schema": V23G_CHECKPOINT_SCHEMA,
        "task_name": "known_gate",
        "variant": V23G_VARIANT,
        "architecture": "self_developed_cfc_liquid_causal_encoder_density_weighted",
        "config_name": wrapper.config_name,
        "config_identity_sha256": config_identity_sha256(wrapper),
        "config_file_sha256": sha256_file(config_path),
        "readiness_sha256": readiness_sha256,
        "source_control": wrapper.source_control,
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
    model: KnownGateClassifier,
    optimizer: torch.optim.Optimizer | None,
    *,
    wrapper: Any,
    config_path: Path,
    readiness_sha256: str,
    initialization_audit: Mapping[str, Any],
    map_location: torch.device,
) -> dict[str, Any]:
    payload = dict(torch.load(path, map_location=map_location, weights_only=False))
    if (
        payload.get("checkpoint_schema") != V23G_CHECKPOINT_SCHEMA
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != V23G_VARIANT
        or payload.get("architecture")
        != "self_developed_cfc_liquid_causal_encoder_density_weighted"
        or payload.get("config_identity_sha256") != config_identity_sha256(wrapper)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or dict(payload.get("source_control") or {}) != dict(wrapper.source_control)
        or dict(payload.get("initialization_audit") or {}) != dict(initialization_audit)
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V23g checkpoint schema/config/readiness binding differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def train(project_root: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    wrapper, source_config, treatment_config, readiness, _contract = _verify_prepared(root)
    config_path = root / config_relative()
    artifact = root / artifact_relative()
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            f"refusing to overwrite non-empty V23g artifact: {artifact}"
        )
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name="radar_ln3.v23g")
    started = time.perf_counter()
    try:
        seed_everything(V23G_TRAINING_SEED)
        train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
        validation_dataset = build_known_gate_dataset(root, source_config, split_name="val")
        if (
            train_dataset.label_counts()
            != {KNOWN_GATE_CLASS_NAMES[0]: 953, KNOWN_GATE_CLASS_NAMES[1]: 33}
            or validation_dataset.label_counts()
            != {KNOWN_GATE_CLASS_NAMES[0]: 223, KNOWN_GATE_CLASS_NAMES[1]: 9}
        ):
            raise ValueError("V23g inherited V21e parent support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(V23G_TRAINING_SEED)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=treatment_config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        initialization_audit = _first_density_exposure_audit(
            root,
            source_config,
            treatment_config,
            str(wrapper.source_control["initial_model_state_sha256"]),
        )
        model, _source_pairing = _build_density_model(
            root, source_config, treatment_config
        )
        if _state_dict_sha256(model.state_dict()) != str(
            wrapper.source_control["initial_model_state_sha256"]
        ):
            raise RuntimeError("V23g training model initial state is not frozen V21e state")
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V23g requires every inherited parameter to train")
        device = known_gate_device(source_config)
        if treatment_config.train.device == "cuda" and device.type != "cuda":
            raise RuntimeError("V23g formal training requires CUDA")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=treatment_config.train.learning_rate)
        class_weights = torch.tensor(
            treatment_config.train.class_weights, dtype=torch.float32, device=device
        )
        readiness_sha256 = sha256_file(root / readiness_relative())
        best_checkpoint = artifact / "best_model.pt"
        last_checkpoint = artifact / "last_model.pt"
        parent_orders = _control_parent_orders(wrapper.source_control)
        start_epoch = 1
        global_step = 0
        best_epoch = 0
        best_selection: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            try:
                resume_path.relative_to(artifact)
            except ValueError as error:
                raise ValueError("V23g resume checkpoint must be inside its artifact") from error
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                wrapper=wrapper,
                config_path=config_path,
                readiness_sha256=readiness_sha256,
                initialization_audit=initialization_audit,
                map_location=device,
            )
            state = _restore_training_state(payload, generator)
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            best_epoch = int(state["best_epoch"])
            best_selection = (
                dict(state["best_selection"])
                if state.get("best_selection") is not None
                else None
            )
            history = list(state.get("history") or ())
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V23g resume is not at an epoch boundary")
            resumed_from = str(resume_path)
        if not 1 <= start_epoch <= V23G_EPOCHS + 1:
            raise ValueError("V23g resume next_epoch is outside the 20-epoch run")
        logger.info(
            "run=start | task=known_gate | variant=%s | encoder=self_developed_cfc_density_weighted | device=%s | train=%d | val=%d | epochs=20 | unknown=0 | test=0 | oof=deferred",
            V23G_VARIANT,
            device,
            len(train_dataset),
            len(validation_dataset),
        )
        steps_per_epoch = len(train_loader)
        for epoch in range(start_epoch, V23G_EPOCHS + 1):
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
                    class_weight_normalization=treatment_config.train.class_weight_normalization,
                    prefix_auxiliary_weight=treatment_config.train.prefix_auxiliary_weight,
                    prefix_auxiliary_min_events=treatment_config.train.prefix_auxiliary_min_events,
                )
                loss = objective["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), treatment_config.train.gradient_clip_norm
                    ).item()
                )
                if not bool(torch.isfinite(loss)) or not np.isfinite(grad_norm):
                    raise ValueError("V23g produced non-finite loss or gradient")
                clips += int(grad_norm > treatment_config.train.gradient_clip_norm)
                optimizer.step()
                global_step += 1
                total_loss += float(loss.item())
                final_loss += float(objective["final_state_loss"].item())
                prefix_loss += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | variant=%s | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        V23G_VARIANT,
                        epoch,
                        step,
                        steps_per_epoch,
                        float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()),
                        grad_norm,
                    )
            if len(order) != len(train_dataset) or len(set(order)) != len(order):
                raise ValueError("V23g epoch did not consume every train parent exactly once")
            order_sha256 = _parent_order_sha256(order)
            expected_order = parent_orders[epoch - 1]
            if order_sha256 != expected_order:
                raise ValueError(
                    "V23g parent order differs from completed V21e control "
                    f"at epoch {epoch}: actual={order_sha256}, expected={expected_order}"
                )
            validation = evaluate_known_gate(
                model, validation_dataset, config=treatment_config, device=device
            )
            selection = _selection(validation)
            selected = _is_better(selection, best_selection)
            if selected:
                best_epoch, best_selection = epoch, selection
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": clips,
                "selection_metric": treatment_config.train.checkpoint_selection_metric,
                "checkpoint_selection": selection,
                "validation": {key: value for key, value in validation.items() if key != "records"},
                "parent_order_sha256": order_sha256,
                "v21e_control_parent_order_sha256": expected_order,
                "parent_order_exactly_matches_v21e_control": True,
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
                    "best_selection": best_selection,
                    "history": history,
                },
            )
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                wrapper=wrapper,
                config_path=config_path,
                readiness_sha256=readiness_sha256,
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
                "epoch | variant=%s | epoch=%02d/20 | loss=%.6f | selection=%s:%s | tie=%s:%s | best=%s | clips=%d | order_sha256=%s | elapsed=%s",
                V23G_VARIANT,
                epoch,
                record["train_loss"],
                selection["primary_metric"],
                format_metric(selection["primary_value"]),
                selection["tie_break_metric"],
                format_metric(selection["tie_break_value"]),
                selected,
                clips,
                order_sha256,
                format_duration(record["elapsed_seconds"]),
            )
        if not best_checkpoint.is_file() or best_epoch <= 0 or best_selection is None:
            raise ValueError("V23g did not produce a best checkpoint")
        _load_checkpoint(
            best_checkpoint,
            model,
            None,
            wrapper=wrapper,
            config_path=config_path,
            readiness_sha256=readiness_sha256,
            initialization_audit=initialization_audit,
            map_location=device,
        )
        best_validation = evaluate_known_gate(
            model, validation_dataset, config=treatment_config, device=device
        )
        detail_path = artifact / "sample_detail_information.json"
        _write_json(
            detail_path,
            {
                "schema_version": V23G_CHECKPOINT_SCHEMA + "-sample-detail-v1",
                "task_name": "known_gate",
                "variant": V23G_VARIANT,
                "architecture": "self_developed_cfc_liquid_causal_encoder_density_weighted",
                "seed": 43,
                "checkpoint": str(best_checkpoint),
                "epoch": best_epoch,
                "counts": {key: len(value) for key, value in best_validation["records"].items()},
                "records": best_validation["records"],
                "scope": "development validation KnownGate only; not OOF or test",
            },
        )
        performance = _performance(
            best_validation, treatment_config.train.task_min_precision_recall_target
        )
        result = {
            "status": "completed",
            **performance,
            "task_name": "known_gate",
            "variant": V23G_VARIANT,
            "architecture": "self_developed_cfc_liquid_causal_encoder_density_weighted",
            "dynamics": {
                "control_mode": "bounded_observation",
                "treatment_mode": "density_weighted",
                "zero_delta_formula": "alpha_k=min(0.25*k,0.75); k includes causal first/positive observation and increments on each zero_delta",
            },
            "seed": 43,
            "dataloader_seed": 43,
            "config_name": wrapper.config_name,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_identity_sha256": config_identity_sha256(wrapper),
            "source_control": wrapper.source_control,
            "train_parent_count": len(train_dataset),
            "validation_parent_count": len(validation_dataset),
            "validation_clutter_support": 9,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_checkpoint_selection": best_selection,
            "best_validation_metrics": {key: value for key, value in best_validation.items() if key != "records"},
            "history": history,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "sample_detail_information": str(detail_path),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "full_parameter_training": True,
            "all_model_parameters_trainable": True,
            "initial_model_state_sha256": initialization_audit[
                "treatment_initial_model_state_sha256"
            ],
            "best_model_state_sha256": _state_dict_sha256(model.state_dict()),
            "model_state_changed_from_initialization": (
                initialization_audit["treatment_initial_model_state_sha256"]
                != _state_dict_sha256(model.state_dict())
            ),
            "initialization_audit": initialization_audit,
            "training_readiness": {
                "path": str(root / readiness_relative()),
                "sha256": readiness_sha256,
                "schema_version": V23G_READINESS_SCHEMA,
                "status": "passed",
                "validation_used_for_readiness_gate": False,
                "test": "未执行",
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
                        "v21e_control_sha256": row["v21e_control_parent_order_sha256"],
                        "exactly_matches_v21e_control": row[
                            "parent_order_exactly_matches_v21e_control"
                        ],
                        "count": int(row["parent_order_count"]),
                    }
                    for row in history
                ],
                "all_epoch_parent_orders_exactly_match_v21e_control": True,
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
        _write_json(artifact / "training-summary.json", result)
        logger.info(
            "run=complete | task=known_gate | variant=%s | status=completed | performance_status=%s | best_epoch=%02d",
            V23G_VARIANT,
            result["performance_status"],
            best_epoch,
        )
        return result
    except Exception as error:
        _write_json(
            artifact / "training-summary.json",
            {
                "status": "failed",
                "performance_status": "not_evaluated",
                "task_name": "known_gate",
                "variant": V23G_VARIANT,
                "architecture": "self_developed_cfc_liquid_causal_encoder_density_weighted",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "OOF": "未执行",
                "test_evaluation": "not_executed",
                "deployment_eligible": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        logger.exception("run=failed | task=known_gate | variant=%s", V23G_VARIANT)
        raise
    finally:
        close_training_logger(logger)



