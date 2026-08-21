"""Train the single full V23 pure official-CfC KnownGate baseline."""

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
from radar_ln3.known_gate_v23 import (
    ARTIFACT_RELATIVE,
    CHECKPOINT_SCHEMA,
    CONFIG_RELATIVE,
    CONTRACT_RELATIVE,
    CONTRACT_SCHEMA,
    FIXED_EPOCHS,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    TRAINING_SEED,
    config_identity_sha256,
    load_v23_config,
    sha256_file,
)
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
)
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


def _verify_prepared(root: Path) -> tuple[Any, Any, Path, dict[str, Any], Path, dict[str, Any]]:
    config_path = (root / CONFIG_RELATIVE).resolve()
    contract_path = (root / CONTRACT_RELATIVE).resolve()
    readiness_path = (root / READINESS_RELATIVE).resolve()
    if not config_path.is_file() or not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V23 baseline prepare artifacts are missing")
    config, source_config, source_path = load_v23_config(root, config_path)
    contract = _read_json(contract_path)
    readiness = _read_json(readiness_path)
    failed_checks = [name for name, value in dict(readiness.get("checks") or {}).items() if value is not True]
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or readiness.get("train_parent_count") != 986
        or readiness.get("validation_parent_count") != 232
        or readiness.get("validation_clutter_parent_support") != 9
        or readiness.get("selected_seed") != TRAINING_SEED
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test_payload_access_count") != 0
        or readiness.get("deployment_eligible") is not False
        or failed_checks
        or contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("unknown_training_exposure") != 0
        or contract.get("oof") != "未执行（先完成V23基座与多seed）"
        or contract.get("test") != "未执行"
        or contract.get("deployment_eligible") is not False
        or config.train.checkpoint_schema != CHECKPOINT_SCHEMA
        or config.train.epochs != FIXED_EPOCHS
    ):
        raise ValueError("V23 baseline readiness/contract is stale or outside the sealed scope")
    return config, source_config, source_path, readiness, contract_path, contract


def _checkpoint_payload(
    model: OfficialCfCKnownGateClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    config: Any,
    source_config: Any,
    config_path: Path,
    readiness_sha256: str,
    initialization_audit: Mapping[str, Any],
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "task_name": "known_gate",
        "variant": "official_cfc_baseline",
        "architecture": "official_ncps_cfc",
        "config_name": config.config_name,
        "config_identity_sha256": config_identity_sha256(config),
        "config_file_sha256": sha256_file(config_path),
        "source_config_path": config.source_config_path,
        "source_config_sha256": config.source_config_sha256,
        "source_config_identity_sha256": config.source_config_identity_sha256,
        "source_variant": source_config.variant,
        "readiness_sha256": readiness_sha256,
        "official_cfc": model.backbone_specification(),
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
    model: OfficialCfCKnownGateClassifier,
    optimizer: torch.optim.Optimizer | None,
    *,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    initialization_audit: Mapping[str, Any],
    map_location: torch.device,
) -> dict[str, Any]:
    payload = dict(torch.load(path, map_location=map_location, weights_only=False))
    expected_spec = model.backbone_specification()
    if (
        payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != "official_cfc_baseline"
        or payload.get("architecture") != "official_ncps_cfc"
        or payload.get("config_identity_sha256") != config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("source_config_sha256") != config.source_config_sha256
        or payload.get("source_config_identity_sha256") != config.source_config_identity_sha256
        or payload.get("readiness_sha256") != readiness_sha256
        or payload.get("official_cfc") != expected_spec
        or dict(payload.get("initialization_audit") or {}) != dict(initialization_audit)
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V23 checkpoint schema/config/runtime/readiness binding differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _official_spec_matches_config(model: OfficialCfCKnownGateClassifier, config: Any) -> None:
    actual = model.backbone_specification()
    expected = config.official_cfc
    required = {
        "distribution": expected.distribution,
        "distribution_version": expected.distribution_version,
        "direct_url": expected.direct_url,
        "git_commit": expected.git_commit,
        "class_path": expected.class_path,
        "class_module": expected.class_module,
        "mode": expected.mode,
        "activation": expected.activation,
        "input_dim": expected.input_dim,
        "hidden_dim": expected.hidden_dim,
        "backbone_units": expected.backbone_units,
        "backbone_layers": expected.backbone_layers,
        "backbone_dropout": expected.backbone_dropout,
        "mixed_memory": expected.mixed_memory,
        "proj_size": expected.proj_size,
        "wiring": expected.wiring,
        "return_sequences": expected.return_sequences,
        "batch_first": expected.batch_first,
        "state_equation_owner": expected.state_equation_owner,
        "dt_semantics": expected.dt_semantics,
    }
    if any(actual.get(name) != value for name, value in required.items()):
        raise RuntimeError("V23 constructed official CfC differs from its frozen config")
    legacy = dict(actual.get("project_legacy_recurrent_mechanisms") or {})
    if not legacy or any(legacy.values()) or actual.get("parameter_count") != 14064:
        raise RuntimeError("V23 official CfC reintroduced a legacy recurrent mechanism")


def train(project_root: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    config, source_config, _source_path, readiness, _contract_path, _contract = _verify_prepared(root)
    config_path = (root / CONFIG_RELATIVE).resolve()
    artifact = (root / ARTIFACT_RELATIVE).resolve()
    if artifact != (root / config.train.formal_artifact_dir).resolve():
        raise ValueError("V23 formal artifact path differs from its config")
    if artifact.exists() and any(artifact.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            "refusing to overwrite a non-empty V23 formal artifact; resume at an epoch boundary or archive it: "
            f"{artifact}"
        )
    artifact.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact / "training.log", logger_name="radar_ln3.v23.official_cfc")
    started = time.perf_counter()
    try:
        seed_everything(config.train.seed)
        train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
        validation_dataset = build_known_gate_dataset(root, source_config, split_name="val")
        if train_dataset.label_counts() != {
            KNOWN_GATE_CLASS_NAMES[0]: 953,
            KNOWN_GATE_CLASS_NAMES[1]: 33,
        } or validation_dataset.label_counts() != {
            KNOWN_GATE_CLASS_NAMES[0]: 223,
            KNOWN_GATE_CLASS_NAMES[1]: 9,
        }:
            raise ValueError("V23 inherited V21 train/validation support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model, initialization_audit = build_official_cfc_model_from_frozen_source(root, source_config)
        _official_spec_matches_config(model, config)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V23 must train every official-CfC model parameter")
        initial_state_sha256 = _state_dict_sha256(model.state_dict())
        device = known_gate_device(source_config)
        if config.train.device == "cuda" and device.type != "cuda":
            raise RuntimeError("V23 formal official CfC training requires an available CUDA device")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        readiness_sha256 = sha256_file(root / READINESS_RELATIVE)
        best_checkpoint = artifact / "best_model.pt"
        last_checkpoint = artifact / "last_model.pt"
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
                raise ValueError("V23 resume checkpoint must be inside its formal artifact") from error
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness_sha256,
                initialization_audit=initialization_audit,
                map_location=device,
            )
            run_state = _restore_training_state(payload, generator)
            start_epoch = int(run_state["next_epoch"])
            global_step = int(run_state["global_step"])
            best_epoch = int(run_state["best_epoch"])
            best_selection = dict(run_state["best_selection"]) if run_state.get("best_selection") else None
            history = list(run_state.get("history") or [])
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V23 resume state is not an epoch boundary")
            resumed_from = str(resume_path)
        if not 1 <= start_epoch <= FIXED_EPOCHS + 1:
            raise ValueError("V23 resume epoch is outside the full 20-epoch budget")
        logger.info(
            "run=start | task=known_gate | variant=official_cfc_baseline | encoder=official_ncps_cfc | device=%s | train=%d | val=%d | epochs=20 | unknown=0 | test=0 | oof=deferred",
            device,
            len(train_dataset),
            len(validation_dataset),
        )
        steps_per_epoch = len(train_loader)
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            model.train()
            epoch_started = time.perf_counter()
            total_loss = final_loss = prefix_loss = 0.0
            gradient_max = 0.0
            gradient_clip_count = 0
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
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
                if not bool(torch.isfinite(loss)) or not np.isfinite(grad_norm):
                    raise ValueError("V23 official CfC produced non-finite loss or gradient")
                gradient_clip_count += int(grad_norm > config.train.gradient_clip_norm)
                optimizer.step()
                global_step += 1
                total_loss += float(loss.item())
                final_loss += float(objective["final_state_loss"].item())
                prefix_loss += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | variant=official_cfc_baseline | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        epoch, step, steps_per_epoch, float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()), grad_norm,
                    )
            if len(order) != len(train_dataset) or len(set(order)) != len(order):
                raise ValueError("V23 full epoch did not consume each train parent exactly once")
            validation = evaluate_known_gate(model, validation_dataset, config=source_config, device=device)
            selection = _selection(validation)
            selected_as_best = _is_better(selection, best_selection)
            if selected_as_best:
                best_epoch = epoch
                best_selection = selection
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "gradient_clip_count": gradient_clip_count,
                "selection_metric": config.train.checkpoint_selection_metric,
                "checkpoint_selection": selection,
                "validation": {key: value for key, value in validation.items() if key != "records"},
                "parent_order_sha256": _parent_order_sha256(order),
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            state = _capture_training_state(
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
                source_config=source_config,
                config_path=config_path,
                readiness_sha256=readiness_sha256,
                initialization_audit=initialization_audit,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
                training_state=state,
            )
            _save_checkpoint(last_checkpoint, checkpoint)
            _save_checkpoint(artifact / "epochs" / f"epoch-{epoch:03d}.pt", checkpoint)
            if selected_as_best:
                _save_checkpoint(best_checkpoint, checkpoint)
            logger.info(
                "epoch | variant=official_cfc_baseline | epoch=%02d/20 | loss=%.6f | selection=%s:%s | tie=%s:%s | best=%s | clips=%d | order_sha256=%s | elapsed=%s",
                epoch, epoch_record["train_loss"], selection["primary_metric"],
                format_metric(selection["primary_value"]), selection["tie_break_metric"],
                format_metric(selection["tie_break_value"]), selected_as_best,
                gradient_clip_count, epoch_record["parent_order_sha256"],
                format_duration(epoch_record["elapsed_seconds"]),
            )
        if not best_checkpoint.is_file() or best_epoch <= 0 or best_selection is None:
            raise ValueError("V23 training did not produce a development-selected checkpoint")
        _load_checkpoint(
            best_checkpoint, model, None, config=config, config_path=config_path,
            readiness_sha256=readiness_sha256, initialization_audit=initialization_audit,
            map_location=device,
        )
        best_validation = evaluate_known_gate(model, validation_dataset, config=source_config, device=device)
        sample_detail_path = artifact / "sample_detail_information.json"
        _write_json(sample_detail_path, {
            "schema_version": "RADAR-LN3-V23-known-gate-official-cfc-baseline-sample-detail-v1",
            "task_name": "known_gate",
            "variant": "official_cfc_baseline",
            "architecture": "official_ncps_cfc",
            "seed": TRAINING_SEED,
            "checkpoint": str(best_checkpoint),
            "epoch": best_epoch,
            "counts": {key: len(value) for key, value in best_validation["records"].items()},
            "records": best_validation["records"],
            "scope": "development validation KnownGate only; not OOF or test",
        })
        performance = _performance(best_validation, config.train.task_min_precision_recall_target)
        final_state_sha256 = _state_dict_sha256(model.state_dict())
        result = {
            "status": "completed",
            **performance,
            "task_name": "known_gate",
            "variant": "official_cfc_baseline",
            "architecture": "official_ncps_cfc",
            "official_cfc": model.backbone_specification(),
            "seed": TRAINING_SEED,
            "dataloader_seed": TRAINING_SEED,
            "config_name": config.config_name,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_identity_sha256": config_identity_sha256(config),
            "source_control": {
                "config_path": config.source_config_path,
                "config_sha256": config.source_config_sha256,
                "config_identity_sha256": config.source_config_identity_sha256,
                "architecture": "project_local_cfc_inspired_encoder",
            },
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
            "sample_detail_information": str(sample_detail_path),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "full_parameter_training": True,
            "all_model_parameters_trainable": True,
            "initial_model_state_sha256": initial_state_sha256,
            "best_model_state_sha256": final_state_sha256,
            "model_state_changed_from_initialization": initial_state_sha256 != final_state_sha256,
            "initialization_audit": initialization_audit,
            "training_readiness": {
                "path": str(root / READINESS_RELATIVE),
                "sha256": readiness_sha256,
                "schema_version": READINESS_SCHEMA,
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
                "seed": TRAINING_SEED,
                "independent_generator": True,
                "epoch_parent_order_hashes": [
                    {"epoch": int(row["epoch"]), "sha256": row["parent_order_sha256"], "count": int(row["parent_order_count"])}
                    for row in history
                ],
                "cuda_deterministic_algorithms": False,
                "determinism_claim": "fixed_initialization_and_parent_order_not_bitwise_cuda_determinism",
            },
            "resume": {
                "resumed": resumed_from is not None,
                "checkpoint": resumed_from,
                "start_epoch": start_epoch,
                "epoch_boundary_only": True,
            },
            "unknown_is_model_class": False,
            "local_argmax_is_product_action": False,
            "validation_interpretation": "development-only KnownGate local evidence, not final CLUTTER_ACCEPTED/UNKNOWN_REJECTED product action",
            "multiseed": "未执行",
            "OOF": "未执行（延后至基座和多seed确认后）",
            "test_evaluation": "not_executed",
            "oof_threshold_calibration": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "formal_rejection": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "deployment_eligible": False,
            "deployment_blockers": [
                "multiseed_confirmation_not_executed",
                "parent_grouped_oof_not_executed",
                "strict_open_set_semantic_decision_layer_not_executed",
                "full_cascade_not_executed",
                "test_not_executed",
                "formal_rejection_not_executed",
                "stable_not_executed",
                "p95_not_executed",
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(artifact / "training-summary.json", result)
        logger.info(
            "run=complete | task=known_gate | variant=official_cfc_baseline | status=completed | performance_status=%s | best_epoch=%02d | summary=%s",
            result["performance_status"], best_epoch, artifact / "training-summary.json",
        )
        return result
    except Exception as error:
        _write_json(artifact / "training-summary.json", {
            "status": "failed",
            "performance_status": "not_evaluated",
            "task_name": "known_gate",
            "variant": "official_cfc_baseline",
            "architecture": "official_ncps_cfc",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "OOF": "未执行",
            "test_evaluation": "not_executed",
            "deployment_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
        })
        logger.exception("run=failed | task=known_gate | variant=official_cfc_baseline")
        raise
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23 pure official-CfC full baseline training.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    resume = args.resume_checkpoint
    if resume is not None and not resume.is_absolute():
        resume = root / resume
    print(json.dumps(train(root, resume_checkpoint=resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
