"""Train one fully isolated V23a/V23b/V23c official-CfC adapter experiment."""

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
    OfficialCfCKnownGateClassifier,
    build_official_cfc_model_from_frozen_source,
)
from radar_ln3.known_gate_v23 import FIXED_EPOCHS, sha256_file
from radar_ln3.known_gate_v23_adapters import (
    V23_ADAPTER_VARIANTS,
    artifact_relative,
    checkpoint_schema,
    config_identity_sha256,
    config_relative,
    contract_relative,
    contract_schema,
    load_adapter_config,
    readiness_relative,
    readiness_schema,
    source_v23_parent_order_hashes,
    source_runtime_report,
    source_v23_training,
    state_dict_sha256 as adapter_state_dict_sha256,
)
from radar_ln3.logging_utils import close_training_logger, configure_training_logger, format_duration, format_metric
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import (
    _capture_training_state,
    _is_better,
    _model_inputs,
    _parent_order_sha256,
    _performance,
    _restore_training_state,
    _selection,
    _state_dict_sha256,
)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _initialization_pairing_audit(
    model: OfficialCfCKnownGateClassifier,
    *,
    source_v23_summary: Mapping[str, Any],
    adapter_parameter_count: int,
) -> dict[str, Any]:
    """Bind each treatment's initial state to the frozen V23 Control.

    V23a/V23c add no state tensors, so their complete state must hash exactly
    like V23's saved initialization.  V23b adds only its two affine
    LayerNorm tensors; after excluding that namespace, its entire remaining
    core must still be byte-identical to the same V23 initialization.
    """

    state = model.state_dict()
    source_hash = str(source_v23_summary["initial_model_state_sha256"])
    full_hash = _state_dict_sha256(state)
    adapter_names = tuple(sorted(name for name in state if name.startswith("state_adapter.")))
    actual_adapter_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("state_adapter.")
    )
    if actual_adapter_parameter_count != adapter_parameter_count:
        raise RuntimeError("V23 adapter trainable parameter count differs from its frozen spec")

    if adapter_parameter_count == 0:
        core_hash = adapter_state_dict_sha256(state)
        full_matches_v23 = full_hash == source_hash
        core_matches_v23 = core_hash == source_hash
        if adapter_names or not full_matches_v23 or not core_matches_v23:
            raise RuntimeError("V23 zero-parameter adapter initial state differs from frozen V23")
    else:
        core_hash = adapter_state_dict_sha256(state, exclude_prefixes=("state_adapter.",))
        full_matches_v23 = False
        core_matches_v23 = core_hash == source_hash
        if adapter_names != (
            "state_adapter.recurrent_norm.bias",
            "state_adapter.recurrent_norm.weight",
        ) or not core_matches_v23:
            raise RuntimeError("V23 LayerNorm adapter core initial state differs from frozen V23")
    return {
        "source_v23_initial_model_state_sha256": source_hash,
        "treatment_full_initial_model_state_sha256": full_hash,
        "treatment_core_initial_model_state_sha256": core_hash,
        "adapter_state_names": list(adapter_names),
        "adapter_trainable_parameter_count": actual_adapter_parameter_count,
        "full_initial_state_exactly_matches_v23": full_matches_v23,
        "core_initial_state_exactly_matches_v23": core_matches_v23,
    }


def _assert_v23_parent_order_history(
    history: list[Mapping[str, Any]],
    source_hashes: tuple[str, ...],
) -> None:
    """Reject a resumed/current run whose parent order deviates from V23."""

    if len(history) > len(source_hashes):
        raise ValueError("V23 adapter history is longer than the frozen 20-epoch Control")
    for expected_epoch, row in enumerate(history, start=1):
        record = dict(row)
        if (
            int(record.get("epoch", -1)) != expected_epoch
            or record.get("parent_order_sha256") != source_hashes[expected_epoch - 1]
            or int(record.get("parent_order_count", -1)) != 986
            or int(record.get("parent_order_unique_count", -1)) != 986
            or record.get("v23_control_parent_order_sha256") != source_hashes[expected_epoch - 1]
            or record.get("parent_order_exactly_matches_v23_control") is not True
        ):
            raise ValueError("V23 adapter parent order differs from its frozen Control")


def _verify_prepared(root: Path, variant: str) -> tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]:
    config_path = root / config_relative(variant)
    contract_path = root / contract_relative(variant)
    readiness_path = root / readiness_relative(variant)
    if not all(path.is_file() for path in (config_path, contract_path, readiness_path)):
        raise FileNotFoundError("V23 adapter prepare artifacts are missing")
    config, source_config, _source_path = load_adapter_config(root, variant)
    readiness = _read_json(readiness_path)
    contract = _read_json(contract_path)
    failed_checks = [name for name, value in dict(readiness.get("checks") or {}).items() if value is not True]
    if (
        readiness.get("status") != "passed" or readiness.get("schema_version") != readiness_schema(variant)
        or readiness.get("variant") != variant or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or int(readiness.get("train_parent_count", -1)) != 986
        or int(readiness.get("validation_parent_count", -1)) != 232
        or int(readiness.get("validation_clutter_parent_support", -1)) != 9
        or int(readiness.get("selected_seed", -1)) != 43
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False or failed_checks
        or contract.get("schema_version") != contract_schema(variant)
        or contract.get("unknown_training_exposure") != 0
        or contract.get("test") != "未执行" or contract.get("deployment_eligible") is not False
        or config.train.checkpoint_schema != checkpoint_schema(variant)
        or config.train.epochs != FIXED_EPOCHS
    ):
        raise ValueError("V23 adapter readiness/contract is stale or outside sealed scope")
    return config, source_config, readiness, contract_path, contract


def _checkpoint_payload(
    model: OfficialCfCKnownGateClassifier, optimizer: torch.optim.Optimizer, *, variant: str,
    config: Any, config_path: Path, readiness_sha256: str, runtime_path: Path, runtime_sha256: str,
    initialization_audit: Mapping[str, Any], pairing_audit: Mapping[str, Any],
    epoch: int, global_step: int, metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_schema": checkpoint_schema(variant), "task_name": "known_gate", "variant": variant,
        "architecture": "official_ncps_cfc_external_state_adapter", "config_name": config.config_name,
        "config_identity_sha256": config_identity_sha256(config), "config_file_sha256": sha256_file(config_path),
        "source_config_path": config.source_config_path, "source_config_sha256": config.source_config_sha256,
        "source_config_identity_sha256": config.source_config_identity_sha256,
        "source_v23_report_path": config.source_v23_report_path, "source_v23_report_sha256": config.source_v23_report_sha256,
        "readiness_sha256": readiness_sha256,
        "runtime_compatibility": {"path": str(runtime_path), "sha256": runtime_sha256},
        "official_cfc": model.backbone_specification(), "external_state_adapter": model.external_state_adapter_specification(),
        "initialization_audit": dict(initialization_audit),
        "v23_initialization_pairing": dict(pairing_audit),
        "epoch": int(epoch), "global_step": int(global_step),
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "metrics": dict(metrics),
        "training_state": dict(training_state), "full_parameter_training": True, "unknown_is_model_class": False,
    }


def _load_checkpoint(
    path: Path, model: OfficialCfCKnownGateClassifier, optimizer: torch.optim.Optimizer | None, *, variant: str,
    config: Any, config_path: Path, readiness_sha256: str, runtime_path: Path, runtime_sha256: str,
    initialization_audit: Mapping[str, Any], pairing_audit: Mapping[str, Any], map_location: torch.device,
) -> dict[str, Any]:
    payload = dict(torch.load(path, map_location=map_location, weights_only=False))
    if (
        payload.get("checkpoint_schema") != checkpoint_schema(variant) or payload.get("variant") != variant
        or payload.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or payload.get("config_identity_sha256") != config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or dict(payload.get("runtime_compatibility") or {}) != {"path": str(runtime_path), "sha256": runtime_sha256}
        or payload.get("official_cfc") != model.backbone_specification()
        or payload.get("external_state_adapter") != model.external_state_adapter_specification()
        or dict(payload.get("initialization_audit") or {}) != dict(initialization_audit)
        or dict(payload.get("v23_initialization_pairing") or {}) != dict(pairing_audit)
        or payload.get("full_parameter_training") is not True or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V23 adapter checkpoint schema/config/runtime/readiness binding differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def train(project_root: Path, *, variant: str, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    if variant not in V23_ADAPTER_VARIANTS:
        raise ValueError(f"unsupported V23 adapter variant: {variant}")
    root = project_root.resolve()
    config, source_config, readiness, _contract_path, _contract = _verify_prepared(root, variant)
    runtime_path, runtime, runtime_sha256 = source_runtime_report(root)
    _source_summary_path, source_v23_summary, _source_checkpoint_path, _source_checkpoint_sha256 = (
        source_v23_training(root)
    )
    source_parent_order_hashes = source_v23_parent_order_hashes(source_v23_summary)
    config_path = root / config_relative(variant)
    artifact = root / artifact_relative(variant)
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"refusing to overwrite non-empty V23 adapter artifact: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name=f"radar_ln3.{variant}")
    started = time.perf_counter()
    try:
        seed_everything(config.train.seed)
        train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
        validation_dataset = build_known_gate_dataset(root, source_config, split_name="val")
        if train_dataset.label_counts() != {KNOWN_GATE_CLASS_NAMES[0]: 953, KNOWN_GATE_CLASS_NAMES[1]: 33} or validation_dataset.label_counts() != {KNOWN_GATE_CLASS_NAMES[0]: 223, KNOWN_GATE_CLASS_NAMES[1]: 9}:
            raise ValueError("V23 adapter inherited V21 support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(train_dataset, batch_size=config.train.batch_size, shuffle=True, generator=generator)
        model, initialization_audit = build_official_cfc_model_from_frozen_source(root, source_config, state_adapter_spec=config.adapter)
        if model.external_state_adapter_specification() != config.adapter.as_dict():
            raise RuntimeError("V23 adapter constructed a different external mechanism")
        expected_parameters = 27434 + config.adapter.trainable_parameter_count
        if sum(parameter.numel() for parameter in model.parameters()) != expected_parameters:
            raise RuntimeError("V23 adapter parameter count differs from pre-registered mechanism")
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V23 adapter requires every parameter trainable")
        initial_state_sha256 = _state_dict_sha256(model.state_dict())
        pairing_audit = _initialization_pairing_audit(
            model,
            source_v23_summary=source_v23_summary,
            adapter_parameter_count=config.adapter.trainable_parameter_count,
        )
        device = known_gate_device(source_config)
        if config.train.device == "cuda" and device.type != "cuda":
            raise RuntimeError("V23 adapter formal training requires CUDA")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        readiness_sha256 = sha256_file(root / readiness_relative(variant))
        best_checkpoint, last_checkpoint = artifact / "best_model.pt", artifact / "last_model.pt"
        start_epoch = 1
        global_step = best_epoch = 0
        best_selection: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            try:
                resume_path.relative_to(artifact)
            except ValueError as error:
                raise ValueError("V23 adapter resume checkpoint must be inside its artifact") from error
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                variant=variant,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_sha256,
                runtime_path=runtime_path,
                runtime_sha256=runtime_sha256,
                initialization_audit=initialization_audit,
                pairing_audit=pairing_audit,
                map_location=device,
            )
            state = _restore_training_state(payload, generator)
            start_epoch, global_step, best_epoch = int(state["next_epoch"]), int(state["global_step"]), int(state["best_epoch"])
            best_selection = dict(state["best_selection"]) if state.get("best_selection") else None
            history = list(state.get("history") or [])
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V23 adapter resume is not at an epoch boundary")
            _assert_v23_parent_order_history(history, source_parent_order_hashes)
            resumed_from = str(resume_path)
        if not 1 <= start_epoch <= FIXED_EPOCHS + 1:
            raise ValueError("V23 adapter resume epoch is outside fixed 20 epochs")
        logger.info("run=start | task=known_gate | variant=%s | encoder=official_ncps_cfc_external_state_adapter | adapter=%s | device=%s | train=%d | val=%d | epochs=20 | unknown=0 | test=0 | oof=deferred", variant, config.adapter.kind, device, len(train_dataset), len(validation_dataset))
        steps_per_epoch = len(train_loader)
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            model.train(); epoch_started = time.perf_counter(); total_loss = final_loss = prefix_loss = 0.0; gradient_max = 0.0; clips = 0; order: list[str] = []
            for step, batch in enumerate(train_loader, start=1):
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(**_model_inputs(tensors))
                objective = parent_final_task_ce_with_prefix_auxiliary(output["sequence_logits"], tensors["labels"], tensors["tr_padding_mask"], class_weights=class_weights, class_weight_normalization=config.train.class_weight_normalization, prefix_auxiliary_weight=config.train.prefix_auxiliary_weight, prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events)
                loss = objective["loss"]
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
                if not bool(torch.isfinite(loss)) or not np.isfinite(grad_norm):
                    raise ValueError("V23 adapter produced non-finite loss or gradient")
                clips += int(grad_norm > config.train.gradient_clip_norm); optimizer.step(); global_step += 1
                total_loss += float(loss.item()); final_loss += float(objective["final_state_loss"].item()); prefix_loss += float(objective["prefix_auxiliary_loss"].item()); gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info("progress | variant=%s | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f", variant, epoch, step, steps_per_epoch, float(loss.item()), float(objective["final_state_loss"].item()), float(objective["prefix_auxiliary_loss"].item()), grad_norm)
            if len(order) != len(train_dataset) or len(set(order)) != len(order):
                raise ValueError("V23 adapter epoch did not consume every train parent exactly once")
            parent_order_sha256 = _parent_order_sha256(order)
            expected_order_sha256 = source_parent_order_hashes[epoch - 1]
            if parent_order_sha256 != expected_order_sha256:
                raise ValueError(
                    "V23 adapter DataLoader parent order differs from frozen V23 Control "
                    f"at epoch {epoch}: actual={parent_order_sha256}, expected={expected_order_sha256}"
                )
            validation = evaluate_known_gate(model, validation_dataset, config=source_config, device=device)
            selection = _selection(validation); selected = _is_better(selection, best_selection)
            if selected: best_epoch, best_selection = epoch, selection
            record = {"epoch": epoch, "global_step": global_step, "train_loss": total_loss / steps_per_epoch, "train_final_state_loss": final_loss / steps_per_epoch, "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch, "gradient_norm_max": gradient_max, "gradient_clip_count": clips, "selection_metric": config.train.checkpoint_selection_metric, "checkpoint_selection": selection, "validation": {key: value for key, value in validation.items() if key != "records"}, "parent_order_sha256": parent_order_sha256, "v23_control_parent_order_sha256": expected_order_sha256, "parent_order_exactly_matches_v23_control": True, "parent_order_count": len(order), "parent_order_unique_count": len(set(order)), "elapsed_seconds": time.perf_counter() - epoch_started}
            history.append(record)
            training_state = _capture_training_state(generator, run_state={"next_epoch": epoch + 1, "global_step": global_step, "best_epoch": best_epoch, "best_selection": best_selection, "history": history})
            checkpoint = _checkpoint_payload(model, optimizer, variant=variant, config=config, config_path=config_path, readiness_sha256=readiness_sha256, runtime_path=runtime_path, runtime_sha256=runtime_sha256, initialization_audit=initialization_audit, pairing_audit=pairing_audit, epoch=epoch, global_step=global_step, metrics=record, training_state=training_state)
            _save_checkpoint(last_checkpoint, checkpoint); _save_checkpoint(artifact / "epochs" / f"epoch-{epoch:03d}.pt", checkpoint)
            if selected: _save_checkpoint(best_checkpoint, checkpoint)
            logger.info("epoch | variant=%s | epoch=%02d/20 | loss=%.6f | selection=%s:%s | tie=%s:%s | best=%s | clips=%d | order_sha256=%s | elapsed=%s", variant, epoch, record["train_loss"], selection["primary_metric"], format_metric(selection["primary_value"]), selection["tie_break_metric"], format_metric(selection["tie_break_value"]), selected, clips, record["parent_order_sha256"], format_duration(record["elapsed_seconds"]))
        if not best_checkpoint.is_file() or best_epoch <= 0 or best_selection is None:
            raise ValueError("V23 adapter training did not produce best checkpoint")
        _assert_v23_parent_order_history(history, source_parent_order_hashes)
        _load_checkpoint(best_checkpoint, model, None, variant=variant, config=config, config_path=config_path, readiness_sha256=readiness_sha256, runtime_path=runtime_path, runtime_sha256=runtime_sha256, initialization_audit=initialization_audit, pairing_audit=pairing_audit, map_location=device)
        best_validation = evaluate_known_gate(model, validation_dataset, config=source_config, device=device)
        detail_path = artifact / "sample_detail_information.json"
        _write_json(detail_path, {"schema_version": f"{checkpoint_schema(variant)}-sample-detail-v1", "task_name": "known_gate", "variant": variant, "architecture": "official_ncps_cfc_external_state_adapter", "adapter": model.external_state_adapter_specification(), "seed": 43, "checkpoint": str(best_checkpoint), "epoch": best_epoch, "counts": {key: len(value) for key, value in best_validation["records"].items()}, "records": best_validation["records"], "scope": "development validation KnownGate only; not OOF or test"})
        performance = _performance(best_validation, config.train.task_min_precision_recall_target)
        final_state_sha256 = _state_dict_sha256(model.state_dict())
        result = {"status": "completed", **performance, "task_name": "known_gate", "variant": variant, "architecture": "official_ncps_cfc_external_state_adapter", "official_cfc": model.backbone_specification(), "external_state_adapter": model.external_state_adapter_specification(), "seed": 43, "dataloader_seed": 43, "config_name": config.config_name, "config_path": str(config_path), "config_sha256": sha256_file(config_path), "config_identity_sha256": config_identity_sha256(config), "source_control": {"v23_report_path": config.source_v23_report_path, "v23_report_sha256": config.source_v23_report_sha256, "v21_config_path": config.source_config_path, "v21_config_sha256": config.source_config_sha256}, "train_parent_count": len(train_dataset), "validation_parent_count": len(validation_dataset), "validation_clutter_support": 9, "global_step": global_step, "best_epoch": best_epoch, "best_checkpoint_selection": best_selection, "best_validation_metrics": {key: value for key, value in best_validation.items() if key != "records"}, "history": history, "best_checkpoint": str(best_checkpoint), "last_checkpoint": str(last_checkpoint), "sample_detail_information": str(detail_path), "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()), "full_parameter_training": True, "all_model_parameters_trainable": True, "initial_model_state_sha256": initial_state_sha256, "best_model_state_sha256": final_state_sha256, "model_state_changed_from_initialization": initial_state_sha256 != final_state_sha256, "initialization_audit": initialization_audit, "v23_initialization_pairing": pairing_audit, "training_readiness": {"path": str(root / readiness_relative(variant)), "sha256": readiness_sha256, "schema_version": readiness_schema(variant), "status": "passed", "validation_used_for_readiness_gate": False, "test": "未执行"}, "official_runtime_compatibility": {"path": str(runtime_path), "sha256": runtime_sha256, "schema_version": runtime["schema_version"], "checks": dict(runtime["checks"])}, "data_contract": {"source_data_contract_path": source_config.data.contract_path, "source_data_contract_sha256": source_config.data.expected_contract_sha256, "normalization_path": source_config.data.normalization_stats_path, "normalization_sha256": source_config.data.expected_stats_sha256, "index_path": source_config.data.index_path, "index_sha256": source_config.data.expected_index_sha256}, "dataloader_contract": {"seed": 43, "independent_generator": True, "epoch_parent_order_hashes": [{"epoch": int(row["epoch"]), "sha256": row["parent_order_sha256"], "v23_control_sha256": row["v23_control_parent_order_sha256"], "exactly_matches_v23_control": row["parent_order_exactly_matches_v23_control"], "count": int(row["parent_order_count"])} for row in history], "all_epoch_parent_orders_exactly_match_v23_control": True, "cuda_deterministic_algorithms": False}, "resume": {"resumed": resumed_from is not None, "checkpoint": resumed_from, "start_epoch": start_epoch, "epoch_boundary_only": True}, "unknown_is_model_class": False, "local_argmax_is_product_action": False, "validation_interpretation": "development-only KnownGate local evidence, not final product action", "multiseed": "未执行", "OOF": "未执行", "test_evaluation": "not_executed", "oof_threshold_calibration": "not_executed", "full_cascade_evaluation": "not_executed", "formal_rejection": "not_executed", "stable_evaluation": "not_executed", "p95_latency": "not_executed", "deployment_eligible": False, "elapsed_seconds": time.perf_counter() - started}
        _write_json(artifact / "training-summary.json", result)
        logger.info("run=complete | task=known_gate | variant=%s | status=completed | performance_status=%s | best_epoch=%02d", variant, result["performance_status"], best_epoch)
        return result
    except Exception as error:
        _write_json(artifact / "training-summary.json", {"status": "failed", "performance_status": "not_evaluated", "task_name": "known_gate", "variant": variant, "architecture": "official_ncps_cfc_external_state_adapter", "error_type": type(error).__name__, "error_message": str(error), "OOF": "未执行", "test_evaluation": "not_executed", "deployment_eligible": False, "elapsed_seconds": time.perf_counter() - started})
        logger.exception("run=failed | task=known_gate | variant=%s", variant)
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V23 external official-CfC adapter experiment.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=V23_ADAPTER_VARIANTS, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    resume = args.resume_checkpoint if args.resume_checkpoint is None or args.resume_checkpoint.is_absolute() else root / args.resume_checkpoint
    print(json.dumps(train(root, variant=args.variant, resume_checkpoint=resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
