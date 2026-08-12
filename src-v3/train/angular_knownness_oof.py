from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.angular_knownness_oof import (
    NESTED_OOF_MAX_EPOCHS,
    NESTED_OOF_PREDICTION_SCHEMA,
    claim_outer_held_out_forward_once,
    complete_outer_held_out_forward_claim,
    derive_nested_stage_configs,
    load_nested_oof_contract,
    model_state_sha256,
    nested_artifact_root,
    nested_config_identities,
    nested_fold_record,
    nested_prediction_path,
    predict_outer_held_out_once,
    validate_completed_outer_held_out_forward_claim,
    verify_nested_dataset,
)
from radar_ln3.cascade import (
    EXPECTED_TASKS,
    build_cascade,
    build_cascade_dataset,
    evaluate_strict_open_set_cascade,
    verify_cascade_data_contracts,
)
from radar_ln3.data import build_point_dataloader
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
    progress_interval,
)
from radar_ln3.strict_open_set_oof import sequence_sha256, sha256_file
from radar_ln3.task_models import (
    ANGULAR_KNOWNNESS_SCHEMA_VERSION,
    seed_everything,
)
from train.cascade import (
    _joint_payload,
    _strip_records,
    build_product_checkpoint_selection,
    cascade_forward_batch,
    cascade_training_objective,
    load_hierarchical_configs,
    load_joint_checkpoint,
    parent_order_sha256,
    resolve_path,
    save_joint_checkpoint,
    save_task_checkpoint_set,
    selection_is_better,
)


TASK_SLUGS = {
    "dbb_cu_gate": "dbb-cu-gate",
    "dbb_specialist": "dbb-specialist",
    "cu_specialist": "cu-specialist",
}


def _config_path(project_root: Path, arm: str, task_name: str) -> Path:
    if arm not in {"control", "treatment"} or task_name not in TASK_SLUGS:
        raise ValueError("V15 nested OOF requires a valid arm and task name")
    return (
        project_root
        / "config"
        / "RADAR-LN3"
        / (
            "RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-"
            f"{arm}-{TASK_SLUGS[task_name]}-config.json"
        )
    )


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V15 nested OOF result: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _without_contract_payloads(contracts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "paths": dict(value["paths"]),
            "hashes": dict(value["hashes"]),
        }
        for name, value in contracts.items()
    }


def _load_completed_stage(path: Path, *, arm: str, outer_fold_id: int, stage: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "passed"
        or payload.get("schema_version") != ANGULAR_KNOWNNESS_SCHEMA_VERSION
        or payload.get("arm") != arm
        or int(payload.get("outer_fold_id", -1)) != outer_fold_id
        or payload.get("stage") != stage
    ):
        raise ValueError(f"V15 nested OOF stage summary is not reusable: {path}")
    return {
        **payload,
        "summary_path": str(path),
        "summary_sha256": sha256_file(path),
    }


def _check_resume_path(path: Path | None, stage_root: Path) -> None:
    if path is None:
        return
    target = path.resolve()
    try:
        target.relative_to(stage_root.resolve())
    except ValueError as exc:
        raise ValueError("V15 nested OOF resume checkpoint must be inside its stage artifact") from exc
    if not target.is_file():
        raise FileNotFoundError(target)


def _stage_resume_path(stage_root: Path) -> Path:
    return stage_root / "joint_last_model.pt"


def _run_stage(
    project_root: Path,
    *,
    arm: str,
    outer_fold_id: int,
    stage: str,
    configs: Mapping[str, Any],
    train_dataset: Any,
    selection_dataset: Any | None,
    train_scope: Mapping[str, Any],
    selection_scope: Mapping[str, Any] | None,
    epoch_limit: int,
    inner_selection: Mapping[str, Any] | None,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    if stage not in {"inner_selection", "outer_retrain"}:
        raise ValueError("V15 nested OOF training stage is invalid")
    gate = configs["dbb_cu_gate"]
    if gate.train.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V15 nested OOF training requires an available CUDA device")
    if epoch_limit < 1 or epoch_limit > NESTED_OOF_MAX_EPOCHS:
        raise ValueError("V15 nested OOF epoch limit is outside [1,20]")
    if stage == "inner_selection" and selection_dataset is None:
        raise ValueError("V15 inner selection requires a known-only selection Dataset")
    if stage == "outer_retrain" and inner_selection is None:
        raise ValueError("V15 outer retrain requires a frozen inner-selection result")
    stage_root = resolve_path(project_root, gate.train.joint_artifact_dir or "")
    summary_path = stage_root / "stage-summary.json"
    completed = _load_completed_stage(
        summary_path,
        arm=arm,
        outer_fold_id=outer_fold_id,
        stage=stage,
    )
    if completed is not None:
        return {**completed, "write_status": "reused_completed"}
    _check_resume_path(resume_checkpoint, stage_root)
    if stage_root.exists() and any(stage_root.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            "V15 nested OOF stage artifact is non-empty; provide its exact joint_last_model.pt "
            f"to resume: {stage_root}"
        )
    stage_root.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(
        stage_root / "training.log",
        logger_name=(
            f"radar_ln3.training.v15_{arm}_nested_outer_{outer_fold_id:02d}_{stage}"
        ),
    )
    started = time.perf_counter()
    try:
        seed_everything(gate.train.seed)
        device = torch.device("cuda")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(gate.train.dataloader_seed))
        loader = build_point_dataloader(
            train_dataset,
            batch_size=gate.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model = build_cascade(configs, device=device).train()
        initialization_sha256 = model_state_sha256(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=gate.train.learning_rate)
        global_step = 0
        start_epoch = 1
        best_epoch = -1
        best_selection: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        checkpoint_set_id = ""
        if resume_checkpoint is not None:
            state = load_joint_checkpoint(
                resume_checkpoint.resolve(), model, optimizer, configs, generator
            )
            if (
                state.get("nested_oof_stage") != stage
                or state.get("nested_oof_arm") != arm
                or int(state.get("nested_oof_outer_fold_id", -1)) != outer_fold_id
                or state.get("initialization_state_sha256") != initialization_sha256
            ):
                raise ValueError("V15 nested OOF resume checkpoint belongs to a different stage/protocol")
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            best_epoch = int(state["best_epoch"])
            saved_selection = state.get("best_selection")
            best_selection = None if saved_selection is None else dict(saved_selection)
            history = list(state.get("history") or [])
            checkpoint_set_id = str(state.get("checkpoint_set_id", ""))
        if not 1 <= start_epoch <= epoch_limit + 1:
            raise ValueError("V15 nested OOF resume point is outside the stage epoch budget")
        logger.info(
            "run=start | arm=%s | outer_fold=%02d | stage=%s | train_parents=%d | selection_parents=%d | held_out_forward=0 | unknown=0 | validation=0 | test=0 | epochs=%d | init_sha256=%s",
            arm,
            outer_fold_id,
            stage,
            len(train_dataset),
            0 if selection_dataset is None else len(selection_dataset),
            epoch_limit,
            initialization_sha256,
        )
        for epoch in range(start_epoch, epoch_limit + 1):
            model.train()
            epoch_started = time.perf_counter()
            values = {
                "loss": 0.0,
                "final_hierarchical_nll": 0.0,
                "final_gate_nll": 0.0,
                "final_specialist_nll": 0.0,
                "prefix_auxiliary_nll": 0.0,
            }
            order: list[str] = []
            steps = 0
            for batch in loader:
                order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                output = cascade_forward_batch(model, tensors)
                objective = cascade_training_objective(output, tensors, configs)
                optimizer.zero_grad(set_to_none=True)
                objective["loss"].backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gate.train.gradient_clip_norm
                    ).item()
                )
                optimizer.step()
                if not np.isfinite(grad_norm) or not torch.isfinite(objective["loss"]).item():
                    raise ValueError("V15 nested OOF stage produced a non-finite loss or gradient")
                global_step += 1
                steps += 1
                for name in values:
                    values[name] += float(objective[name].item())
                if steps == 1 or steps % progress_interval(len(loader)) == 0:
                    logger.info(
                        "progress | arm=%s | outer_fold=%02d | stage=%s | epoch=%02d/%02d | step=%04d/%04d | global=%05d | loss=%.6f | grad=%.3f",
                        arm,
                        outer_fold_id,
                        stage,
                        epoch,
                        epoch_limit,
                        steps,
                        len(loader),
                        global_step,
                        float(objective["loss"].item()),
                        grad_norm,
                    )
            expected_ids = sorted(str(value) for value in train_dataset.parent_ids)
            if (
                len(order) != len(train_dataset)
                or len(set(order)) != len(order)
                or sorted(order) != expected_ids
            ):
                raise ValueError("V15 nested OOF epoch did not consume each selected parent exactly once")
            selection: dict[str, Any]
            validation: dict[str, Any] | None = None
            selected_as_best = False
            if stage == "inner_selection":
                model.eval()
                validation = evaluate_strict_open_set_cascade(
                    model,
                    selection_dataset,
                    device=device,
                    batch_size=gate.train.batch_size,
                    split_name=gate.data.train_split_name,
                    evaluation_scope="known_only_checkpoint_selection",
                )
                selection = build_product_checkpoint_selection(validation, gate)
                selected_as_best = selection_is_better(selection, best_selection)
                if selected_as_best:
                    best_epoch = epoch
                    best_selection = selection
            else:
                frozen_selection = dict(inner_selection or {})
                selection = {
                    "metric": gate.train.checkpoint_selection_metric,
                    "protocol": "outer_train_retrain_to_inner_selected_epoch",
                    "selected_epoch": epoch_limit,
                    "inner_selection_epoch": int(frozen_selection["selected_epoch"]),
                    "inner_selection_sha256": str(frozen_selection["summary_sha256"]),
                    "ranking_key": list(frozen_selection["selection"]["ranking_key"]),
                    "unknown_parent_count": 0,
                    "unknown_used_for_selection": False,
                    "outer_held_out_used_for_selection": False,
                    "validation_used_for_selection": False,
                    "test_used_for_selection": False,
                }
                best_epoch = epoch_limit
                best_selection = selection
                selected_as_best = epoch == epoch_limit
            checkpoint_set_id = uuid.uuid4().hex
            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_set_id": checkpoint_set_id,
                **{name: value / max(1, steps) for name, value in values.items()},
                "train_steps": steps,
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "parent_order_sha256": parent_order_sha256(order),
                "train_parent_set_sha256": sequence_sha256(sorted(order)),
                "held_out_forward_count": 0,
                "outer_held_out_checkpoint_selection": False,
                "unknown_training_exposure": 0,
                "validation_access_count": 0,
                "test_payload_access_count": 0,
                "checkpoint_selection": selection,
                "selection": None if validation is None else _strip_records(validation),
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
            checkpoint_payload = _joint_payload(
                model,
                optimizer,
                configs,
                generator,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_selection=best_selection,
                history=history,
                checkpoint_set_id=checkpoint_set_id,
            )
            checkpoint_payload["training_state"]["run_state"].update(
                {
                    "nested_oof_stage": stage,
                    "nested_oof_arm": arm,
                    "nested_oof_outer_fold_id": outer_fold_id,
                    "initialization_state_sha256": initialization_sha256,
                }
            )
            save_joint_checkpoint(_stage_resume_path(stage_root), checkpoint_payload)
            if selected_as_best:
                save_task_checkpoint_set(
                    project_root,
                    model,
                    configs,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_record,
                    kind="best",
                    checkpoint_set_id=checkpoint_set_id,
                )
                save_joint_checkpoint(
                    resolve_path(project_root, gate.train.joint_best_checkpoint_path or ""),
                    checkpoint_payload,
                )
            logger.info(
                "epoch | arm=%s | outer_fold=%02d | stage=%s | epoch=%02d/%02d | loss=%.6f | selected=%s | held_out_forward=0 | elapsed=%s | order_sha256=%s",
                arm,
                outer_fold_id,
                stage,
                epoch,
                epoch_limit,
                epoch_record["loss"],
                selected_as_best,
                format_duration(epoch_record["elapsed_seconds"]),
                epoch_record["parent_order_sha256"],
            )
        if best_epoch < 1 or best_selection is None or not history:
            raise RuntimeError("V15 nested OOF stage completed without a selected epoch")
        result = {
            "status": "passed",
            "schema_version": ANGULAR_KNOWNNESS_SCHEMA_VERSION,
            "arm": arm,
            "outer_fold_id": outer_fold_id,
            "stage": stage,
            "epoch_limit": epoch_limit,
            "selected_epoch": best_epoch,
            "selection": best_selection,
            "initialization_state_sha256": initialization_sha256,
            "data_contracts": _without_contract_payloads(
                verify_cascade_data_contracts(project_root, configs)
            ),
            "train_scope": dict(train_scope),
            "selection_scope": None if selection_scope is None else dict(selection_scope),
            "unknown_training_exposure": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "outer_held_out_forward_count": 0,
            "history": history,
            "checkpoint_paths": {
                name: str(resolve_path(project_root, configs[name].train.checkpoint_path))
                for name in EXPECTED_TASKS
            },
            "joint_best_checkpoint": str(
                resolve_path(project_root, gate.train.joint_best_checkpoint_path or "")
            ),
            "joint_last_checkpoint": str(_stage_resume_path(stage_root)),
            "formal_training": "未执行（此阶段仅为嵌套OOF支持训练）",
            "threshold_fitting": "未执行",
            "development_unknown_audit": "未执行",
            "test": "未执行",
            "p95": "未执行",
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["write_status"] = _write_new_or_identical(summary_path, result)
        result["summary_path"] = str(summary_path)
        result["summary_sha256"] = sha256_file(summary_path)
        logger.info(
            "run=complete | arm=%s | outer_fold=%02d | stage=%s | selected_epoch=%02d | held_out_forward=0 | summary=%s",
            arm,
            outer_fold_id,
            stage,
            best_epoch,
            summary_path,
        )
        return result
    finally:
        close_training_logger(logger)


def train_nested_oof_fold(
    project_root: Path,
    *,
    arm: str,
    config_paths: Mapping[str, Path],
    outer_fold_id: int,
    inner_resume_checkpoint: Path | None = None,
    outer_resume_checkpoint: Path | None = None,
    prediction_batch_size: int | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    if arm not in {"control", "treatment"}:
        raise ValueError("V15 nested OOF arm must be control or treatment")
    base_configs = load_hierarchical_configs(config_paths)
    if any(
        config.schema_version != ANGULAR_KNOWNNESS_SCHEMA_VERSION
        for config in base_configs.values()
    ):
        raise ValueError("V15 nested OOF requires V15 configurations")
    if {config.train.paired_arm for config in base_configs.values()} != {arm}:
        raise ValueError("V15 nested OOF CLI arm differs from configuration paired arm")
    if base_configs["dbb_cu_gate"].train.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V15 nested OOF requires an available CUDA device")
    verify_cascade_data_contracts(root, base_configs)
    contract = load_nested_oof_contract(root)
    fold = nested_fold_record(contract, outer_fold_id)
    inner_configs = derive_nested_stage_configs(root, base_configs, fold, stage="inner_selection")
    outer_configs = derive_nested_stage_configs(root, base_configs, fold, stage="outer_retrain")
    inner_contracts = verify_cascade_data_contracts(root, inner_configs)
    outer_contracts = verify_cascade_data_contracts(root, outer_configs)
    inner_train_ids = list(fold["inner_train"]["train_parent_ids"])
    inner_selection_ids = list(fold["inner_selection_parent_ids"])
    outer_train_ids = list(fold["outer_retrain"]["train_parent_ids"])
    outer_held_out_ids = list(fold["outer_held_out_parent_ids"])
    inner_train_dataset = build_cascade_dataset(
        root,
        inner_configs["dbb_cu_gate"],
        split_name="train",
        label_scope="known_only",
        parent_ids=inner_train_ids,
    )
    inner_selection_dataset = build_cascade_dataset(
        root,
        inner_configs["dbb_cu_gate"],
        split_name="train",
        label_scope="known_only",
        parent_ids=inner_selection_ids,
    )
    outer_train_dataset = build_cascade_dataset(
        root,
        outer_configs["dbb_cu_gate"],
        split_name="train",
        label_scope="known_only",
        parent_ids=outer_train_ids,
    )
    outer_held_out_dataset = build_cascade_dataset(
        root,
        outer_configs["dbb_cu_gate"],
        split_name="train",
        label_scope="known_only",
        parent_ids=outer_held_out_ids,
    )
    inner_train_scope = verify_nested_dataset(
        inner_train_dataset,
        expected_parent_ids=inner_train_ids,
        scope=f"outer_{outer_fold_id:02d}_inner_train",
    )
    inner_selection_scope = verify_nested_dataset(
        inner_selection_dataset,
        expected_parent_ids=inner_selection_ids,
        scope=f"outer_{outer_fold_id:02d}_inner_selection",
    )
    outer_train_scope = verify_nested_dataset(
        outer_train_dataset,
        expected_parent_ids=outer_train_ids,
        scope=f"outer_{outer_fold_id:02d}_outer_train",
    )
    outer_held_out_scope = verify_nested_dataset(
        outer_held_out_dataset,
        expected_parent_ids=outer_held_out_ids,
        scope=f"outer_{outer_fold_id:02d}_held_out",
    )
    if (
        set(inner_train_dataset.parent_ids) & set(inner_selection_dataset.parent_ids)
        or set(inner_train_dataset.parent_ids) & set(outer_held_out_dataset.parent_ids)
        or set(inner_selection_dataset.parent_ids) & set(outer_held_out_dataset.parent_ids)
        or set(outer_train_dataset.parent_ids) & set(outer_held_out_dataset.parent_ids)
    ):
        raise ValueError("V15 nested OOF parent partitions overlap")
    inner_result = _run_stage(
        root,
        arm=arm,
        outer_fold_id=outer_fold_id,
        stage="inner_selection",
        configs=inner_configs,
        train_dataset=inner_train_dataset,
        selection_dataset=inner_selection_dataset,
        train_scope=inner_train_scope,
        selection_scope=inner_selection_scope,
        epoch_limit=NESTED_OOF_MAX_EPOCHS,
        inner_selection=None,
        resume_checkpoint=inner_resume_checkpoint,
    )
    if int(inner_result["selected_epoch"]) not in range(1, NESTED_OOF_MAX_EPOCHS + 1):
        raise ValueError("V15 nested OOF inner selection chose an invalid epoch")
    inner_result = {
        **inner_result,
        "summary_sha256": str(inner_result["summary_sha256"]),
    }
    outer_result = _run_stage(
        root,
        arm=arm,
        outer_fold_id=outer_fold_id,
        stage="outer_retrain",
        configs=outer_configs,
        train_dataset=outer_train_dataset,
        selection_dataset=None,
        train_scope=outer_train_scope,
        selection_scope=None,
        epoch_limit=int(inner_result["selected_epoch"]),
        inner_selection=inner_result,
        resume_checkpoint=outer_resume_checkpoint,
    )
    if outer_result["initialization_state_sha256"] != inner_result["initialization_state_sha256"]:
        raise ValueError("V15 nested OOF outer retrain did not start from the frozen inner initialization")
    prediction_path = nested_prediction_path(root, arm, outer_fold_id)
    outer_identities = nested_config_identities(outer_configs)
    dbb_score_name = outer_configs["dbb_specialist"].model.dbb_knownness_score_name
    if prediction_path.exists():
        existing = json.loads(prediction_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") != NESTED_OOF_PREDICTION_SCHEMA
            or existing.get("arm") != arm
            or int(existing.get("outer_fold_id", -1)) != outer_fold_id
        ):
            raise FileExistsError(f"V15 nested OOF prediction path is not reusable: {prediction_path}")
        validate_completed_outer_held_out_forward_claim(
            root,
            contract=contract,
            arm=arm,
            fold=fold,
            outer_config_identities=outer_identities,
            dbb_knownness_score_name=dbb_score_name,
            prediction_path=prediction_path,
        )
        return {
            "status": "reused_completed",
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "inner_selection": inner_result,
            "outer_retrain": outer_result,
        }
    device = torch.device("cuda")
    seed_everything(outer_configs["dbb_cu_gate"].train.seed)
    final_model = build_cascade(outer_configs, device=device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=outer_configs["dbb_cu_gate"].train.learning_rate
    )
    final_generator = torch.Generator(device="cpu")
    final_generator.manual_seed(int(outer_configs["dbb_cu_gate"].train.dataloader_seed))
    outer_checkpoint = resolve_path(
        root, outer_configs["dbb_cu_gate"].train.joint_best_checkpoint_path or ""
    )
    final_state = load_joint_checkpoint(
        outer_checkpoint,
        final_model,
        final_optimizer,
        outer_configs,
        final_generator,
    )
    if (
        int(final_state.get("best_epoch", -1)) != int(inner_result["selected_epoch"])
        or final_state.get("nested_oof_stage") != "outer_retrain"
        or int(final_state.get("nested_oof_outer_fold_id", -1)) != outer_fold_id
    ):
        raise ValueError("V15 nested OOF outer checkpoint does not match frozen selected epoch")
    final_model.eval()
    forward_claim_path = claim_outer_held_out_forward_once(
        root,
        contract=contract,
        arm=arm,
        fold=fold,
        outer_config_identities=outer_identities,
        dbb_knownness_score_name=dbb_score_name,
    )
    records = predict_outer_held_out_once(
        final_model,
        outer_held_out_dataset,
        device=device,
        batch_size=prediction_batch_size or outer_configs["dbb_cu_gate"].train.batch_size,
        outer_fold_id=outer_fold_id,
    )
    predicted_ids = sorted(str(record["parent_sample_id"]) for record in records)
    if predicted_ids != sorted(outer_held_out_ids):
        raise ValueError("V15 nested OOF outer held-out prediction does not cover exactly its fold")
    payload = {
        "status": "passed",
        "schema_version": NESTED_OOF_PREDICTION_SCHEMA,
        "model_schema_version": ANGULAR_KNOWNNESS_SCHEMA_VERSION,
        "arm": arm,
        "outer_fold_id": outer_fold_id,
        "fit_split": "train",
        "fit_population": "known_only",
        "grouping": "parent",
        "prediction_protocol": "nested_out_of_fold",
        "epoch_protocol": "inner_parent_selection_then_outer_train_retrain",
        "inner_selected_epoch": int(inner_result["selected_epoch"]),
        "outer_retrain_epoch": int(outer_result["selected_epoch"]),
        "inner_selection_summary": str(inner_result["summary_path"]),
        "inner_selection_summary_sha256": str(inner_result["summary_sha256"]),
        "outer_retrain_summary": str(outer_result["summary_path"]),
        "outer_retrain_summary_sha256": str(outer_result["summary_sha256"]),
        "inner_initialization_state_sha256": str(inner_result["initialization_state_sha256"]),
        "outer_initialization_state_sha256": str(outer_result["initialization_state_sha256"]),
        "outer_retrain_started_from_frozen_initialization": True,
        "outer_held_out_used_for_normalization": False,
        "outer_held_out_used_for_epoch_selection": False,
        "outer_held_out_forward_count_before_final": 0,
        "outer_held_out_prediction_pass_count": 1,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "nested_oof_contract": str(contract["_contract_path"]),
        "nested_oof_contract_sha256": str(contract["_contract_sha256"]),
        "base_config_identities": nested_config_identities(base_configs),
        "inner_config_identities": nested_config_identities(inner_configs),
        "outer_config_identities": outer_identities,
        "inner_data_contracts": _without_contract_payloads(inner_contracts),
        "outer_data_contracts": _without_contract_payloads(outer_contracts),
        "inner_train_scope": inner_train_scope,
        "inner_selection_scope": inner_selection_scope,
        "outer_train_scope": outer_train_scope,
        "outer_held_out_scope": outer_held_out_scope,
        "checkpoint_paths": outer_result["checkpoint_paths"],
        "checkpoint_sha256": {
            name: sha256_file(Path(path))
            for name, path in dict(outer_result["checkpoint_paths"]).items()
        },
        "checkpoint_set_id": str(final_state["checkpoint_set_id"]),
        "dbb_knownness_score_name": dbb_score_name,
        "outer_held_out_forward_claim_path": str(forward_claim_path),
        "oof_prediction_parent_count": len(records),
        "oof_prediction_parent_ids_sha256": sequence_sha256(predicted_ids),
        "records": records,
        "formal_training": "未执行（这些是嵌套OOF校准支持模型）",
        "threshold_fitting": "未执行",
        "development_unknown_audit": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    payload["write_status"] = _write_new_or_identical(prediction_path, payload)
    completed_claim = complete_outer_held_out_forward_claim(
        forward_claim_path,
        prediction_path=prediction_path,
    )
    if completed_claim.get("prediction_sha256") != sha256_file(prediction_path):
        raise RuntimeError("V15 nested OOF completed forward claim hash differs from prediction")
    summary = dict(payload)
    summary.pop("records", None)
    summary_path = prediction_path.parent / "outer-fold-summary.json"
    summary["prediction_path"] = str(prediction_path)
    summary["prediction_sha256"] = sha256_file(prediction_path)
    _write_new_or_identical(summary_path, summary)
    return {
        "status": "passed",
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "inner_selection": inner_result,
        "outer_retrain": outer_result,
        "outer_held_out_parent_count": len(records),
        "unknown_training_exposure": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "formal_training": "未执行",
        "threshold_fitting": "未执行",
        "development_unknown_audit": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one V15 Control/Treatment nested known-only OOF outer fold: inner epoch "
            "selection, frozen-initialization outer retrain, then exactly one held-out forward pass."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--outer-fold-id", type=int, required=True)
    parser.add_argument("--gate-config", type=Path, default=None)
    parser.add_argument("--dbb-config", type=Path, default=None)
    parser.add_argument("--cu-config", type=Path, default=None)
    parser.add_argument("--inner-resume-checkpoint", type=Path, default=None)
    parser.add_argument("--outer-resume-checkpoint", type=Path, default=None)
    parser.add_argument("--prediction-batch-size", type=int, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()

    def absolute(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    supplied = {
        "dbb_cu_gate": args.gate_config,
        "dbb_specialist": args.dbb_config,
        "cu_specialist": args.cu_config,
    }
    config_paths = {
        name: absolute(path) if path is not None else _config_path(root, args.arm, name)
        for name, path in supplied.items()
    }
    result = train_nested_oof_fold(
        root,
        arm=args.arm,
        config_paths=config_paths,
        outer_fold_id=args.outer_fold_id,
        inner_resume_checkpoint=(
            None if args.inner_resume_checkpoint is None else absolute(args.inner_resume_checkpoint)
        ),
        outer_resume_checkpoint=(
            None if args.outer_resume_checkpoint is None else absolute(args.outer_resume_checkpoint)
        ),
        prediction_batch_size=args.prediction_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
