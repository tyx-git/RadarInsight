from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.cascade import (
    EXPECTED_TASKS,
    build_cascade,
    build_cascade_dataset,
    verify_cascade_data_contracts,
)
from radar_ln3.data import build_point_dataloader
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    progress_interval,
)
from radar_ln3.strict_open_set_oof import (
    OOF_FIXED_EPOCHS,
    OOF_PREDICTION_SCHEMA,
    derive_oof_fold_configs,
    fold_config_identities,
    fold_prediction_path,
    fold_record,
    load_oof_fold_contract,
    predict_oof_parent_final,
    sequence_sha256,
    sha256_file,
    verify_oof_dataset,
)
from radar_ln3.task_models import seed_everything
from train.cascade import (
    _joint_payload,
    cascade_forward_batch,
    cascade_training_objective,
    load_hierarchical_configs,
    load_joint_checkpoint,
    parent_order_sha256,
    resolve_path,
    save_joint_checkpoint,
    save_task_checkpoint_set,
)


DEFAULT_CONFIGS = {
    "dbb_cu_gate": Path(
        "config/RADAR-LN3/RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-cu-gate-config.json"
    ),
    "dbb_specialist": Path(
        "config/RADAR-LN3/RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-specialist-config.json"
    ),
    "cu_specialist": Path(
        "config/RADAR-LN3/RADAR-LN3-V14-cq08-liquid-strict-open-set-cu-specialist-config.json"
    ),
}


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V14 OOF result: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def train_oof_fold(
    project_root: Path,
    *,
    config_paths: dict[str, Path],
    fold_id: int,
    resume_checkpoint: Path | None = None,
    prediction_batch_size: int | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    base_configs = load_hierarchical_configs(config_paths)
    verify_cascade_data_contracts(root, base_configs)
    fold_contract, assignment = load_oof_fold_contract(root)
    fold = fold_record(fold_contract, int(fold_id))
    configs = derive_oof_fold_configs(root, base_configs, fold)
    fold_contracts = verify_cascade_data_contracts(root, configs)
    gate = configs["dbb_cu_gate"]
    if gate.train.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V14 parent-grouped OOF fold training requires an available CUDA device")
    if gate.train.epochs != OOF_FIXED_EPOCHS or gate.train.early_stopping_patience != 0:
        raise ValueError("V14 OOF fold must use fixed 20 epochs with early stopping disabled")
    fold_root = resolve_path(root, gate.train.joint_artifact_dir or "")
    prediction_path = fold_prediction_path(root, int(fold_id))
    if prediction_path.exists():
        raise FileExistsError(f"V14 OOF fold {fold_id} is already complete: {prediction_path}")
    if fold_root.exists() and resume_checkpoint is None and any(fold_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty V14 OOF fold artifact: {fold_root}")
    fold_root.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        try:
            resume_checkpoint.relative_to(fold_root)
        except ValueError as exc:
            raise ValueError("V14 OOF resume checkpoint must be inside the selected fold artifact") from exc
    logger = configure_training_logger(
        fold_root / "training.log",
        logger_name=f"radar_ln3.training.v14_oof_fold_{int(fold_id):02d}",
    )
    started = time.perf_counter()
    try:
        seed_everything(gate.train.seed)
        device = torch.device("cuda")
        train_dataset = build_cascade_dataset(
            root,
            gate,
            split_name=gate.data.train_split_name,
            label_scope="known_only",
            parent_ids=fold["train_parent_ids"],
        )
        train_scope = verify_oof_dataset(
            train_dataset,
            expected_parent_ids=fold["train_parent_ids"],
            scope=f"fold_{int(fold_id):02d}_train",
        )
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
        history: list[dict[str, Any]] = []
        last_checkpoint_set_id = ""
        if resume_checkpoint is not None:
            state = load_joint_checkpoint(
                resume_checkpoint,
                model,
                optimizer,
                configs,
                loader_generator,
            )
            start_epoch = int(state["next_epoch"])
            global_step = int(state["global_step"])
            history = list(state.get("history") or [])
            last_checkpoint_set_id = str(state.get("checkpoint_set_id", ""))
        if not 1 <= start_epoch <= OOF_FIXED_EPOCHS + 1:
            raise ValueError("V14 OOF resume point is outside the fixed epoch budget")
        logger.info(
            "run=start | fold=%02d | population=known_train_fold_only | train=%d | held_out=%d | unknown=0 | val=0 | test=0 | epochs=%d | held_out_checkpoint_selection=false",
            int(fold_id),
            len(train_dataset),
            len(fold["held_out_parent_ids"]),
            OOF_FIXED_EPOCHS,
        )
        for epoch in range(start_epoch, OOF_FIXED_EPOCHS + 1):
            model.train()
            epoch_started = time.perf_counter()
            epoch_values = {
                "loss": 0.0,
                "final_hierarchical_nll": 0.0,
                "final_gate_nll": 0.0,
                "final_specialist_nll": 0.0,
                "prefix_auxiliary_nll": 0.0,
            }
            order: list[str] = []
            steps = 0
            for batch in train_loader:
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
                        model.parameters(),
                        gate.train.gradient_clip_norm,
                    ).item()
                )
                optimizer.step()
                if not np.isfinite(grad_norm) or not torch.isfinite(objective["loss"]).item():
                    raise ValueError("V14 OOF fold produced a non-finite loss or gradient")
                global_step += 1
                steps += 1
                for name in epoch_values:
                    epoch_values[name] += float(objective[name].item())
                if steps == 1 or steps % progress_interval(len(train_loader)) == 0:
                    logger.info(
                        "progress | fold=%02d | epoch=%02d/%02d | step=%04d/%04d | global=%05d | loss=%.6f | grad=%.3f",
                        int(fold_id),
                        epoch,
                        OOF_FIXED_EPOCHS,
                        steps,
                        len(train_loader),
                        global_step,
                        float(objective["loss"].item()),
                        grad_norm,
                    )
            if (
                len(order) != len(train_dataset)
                or len(set(order)) != len(order)
                or sorted(order) != sorted(fold["train_parent_ids"])
            ):
                raise ValueError("V14 OOF epoch did not consume each fold-train parent exactly once")
            last_checkpoint_set_id = uuid.uuid4().hex
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_set_id": last_checkpoint_set_id,
                **{name: value / max(1, steps) for name, value in epoch_values.items()},
                "train_steps": steps,
                "parent_order_count": len(order),
                "parent_order_unique_count": len(set(order)),
                "parent_order_sha256": parent_order_sha256(order),
                "train_parent_set_sha256": sequence_sha256(sorted(order)),
                "held_out_forward_count": 0,
                "held_out_checkpoint_selection": False,
                "validation_access_count": 0,
                "test_payload_access_count": 0,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            resume_payload = _joint_payload(
                model,
                optimizer,
                configs,
                loader_generator,
                epoch=epoch,
                global_step=global_step,
                best_epoch=0,
                best_selection=None,
                history=history,
                checkpoint_set_id=last_checkpoint_set_id,
            )
            save_joint_checkpoint(
                resolve_path(root, gate.train.joint_last_checkpoint_path or ""),
                resume_payload,
            )
            logger.info(
                "epoch | fold=%02d | epoch=%02d/%02d | loss=%.6f | held_out_forward=0 | selection=none | elapsed=%s | order_sha256=%s",
                int(fold_id),
                epoch,
                OOF_FIXED_EPOCHS,
                epoch_record["loss"],
                format_duration(epoch_record["elapsed_seconds"]),
                epoch_record["parent_order_sha256"],
            )
        if not history or int(history[-1]["epoch"]) != OOF_FIXED_EPOCHS:
            raise RuntimeError("V14 OOF fold did not reach its fixed final epoch")
        if len(last_checkpoint_set_id) != 32:
            raise RuntimeError("V14 OOF fold lacks its final checkpoint set identity")
        fixed_selection = {
            "metric": gate.train.checkpoint_selection_metric,
            "protocol": "fixed_final_epoch_no_held_out_selection",
            "ranking_key": [0.0] * 8,
            "selected_epoch": OOF_FIXED_EPOCHS,
            "held_out_used_for_selection": False,
            "unknown_parent_count": 0,
            "unknown_used_for_selection": False,
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
        }
        final_metrics = {
            **history[-1],
            "checkpoint_selection": fixed_selection,
            "oof_fold_id": int(fold_id),
            "oof_role": "calibration_support_only_not_deployment",
        }
        checkpoint_paths = save_task_checkpoint_set(
            root,
            model,
            configs,
            epoch=OOF_FIXED_EPOCHS,
            global_step=global_step,
            metrics=final_metrics,
            kind="best",
            checkpoint_set_id=last_checkpoint_set_id,
        )
        final_payload = _joint_payload(
            model,
            optimizer,
            configs,
            loader_generator,
            epoch=OOF_FIXED_EPOCHS,
            global_step=global_step,
            best_epoch=OOF_FIXED_EPOCHS,
            best_selection=fixed_selection,
            history=history,
            checkpoint_set_id=last_checkpoint_set_id,
        )
        save_joint_checkpoint(
            resolve_path(root, gate.train.joint_best_checkpoint_path or ""),
            final_payload,
        )
        held_out_dataset = build_cascade_dataset(
            root,
            gate,
            split_name=gate.data.train_split_name,
            label_scope="known_only",
            parent_ids=fold["held_out_parent_ids"],
        )
        held_out_scope = verify_oof_dataset(
            held_out_dataset,
            expected_parent_ids=fold["held_out_parent_ids"],
            scope=f"fold_{int(fold_id):02d}_held_out",
        )
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise ValueError("V14 OOF held-out parents leaked into the optimizer Dataset")
        model.eval()
        records = predict_oof_parent_final(
            model,
            held_out_dataset,
            device=device,
            batch_size=prediction_batch_size or gate.train.batch_size,
            fold_id=int(fold_id),
        )
        predicted_ids = sorted(str(value["parent_sample_id"]) for value in records)
        if predicted_ids != fold["held_out_parent_ids"]:
            raise ValueError("V14 OOF final prediction does not match the frozen held-out fold")
        payload = {
            "status": "passed",
            "schema_version": OOF_PREDICTION_SCHEMA,
            "fold_id": int(fold_id),
            "role": "known-only train parent-grouped OOF calibration support; not a deployment checkpoint",
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "parent",
            "prediction_protocol": "out_of_fold_fixed_final_epoch",
            "epochs": OOF_FIXED_EPOCHS,
            "early_stopping": False,
            "held_out_checkpoint_selection": False,
            "held_out_forward_count_before_final_epoch": 0,
            "held_out_prediction_pass_count": 1,
            "unknown_training_exposure": 0,
            "unknown_oof_prediction_count": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
            "assignment_manifest": str(root / "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json"),
            "assignment_manifest_sha256": sha256_file(
                root / "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json"
            ),
            "fold_contract": str(root / "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json"),
            "fold_contract_sha256": sha256_file(
                root / "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json"
            ),
            "assignment_seed": int(assignment["assignment_seed"]),
            "normalization_stats": str(resolve_path(root, gate.data.normalization_stats_path)),
            "normalization_stats_sha256": gate.data.expected_stats_sha256,
            "train_scope": train_scope,
            "held_out_scope": held_out_scope,
            "derived_config_identity_sha256": fold_config_identities(configs),
            "fold_data_contracts": fold_contracts,
            "checkpoint_paths": checkpoint_paths,
            "checkpoint_sha256": {
                name: sha256_file(checkpoint_paths[name]) for name in EXPECTED_TASKS
            },
            "checkpoint_set_id": last_checkpoint_set_id,
            "epoch": OOF_FIXED_EPOCHS,
            "global_step": global_step,
            "fold_training_history": history,
            "oof_prediction_parent_count": len(records),
            "oof_prediction_parent_ids_sha256": sequence_sha256(predicted_ids),
            "records": records,
            "formal_training": "未执行（这些是OOF校准支持折模型，不是正式部署模型）",
            "threshold_fitting": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "test": "未执行",
            "p95": "未执行",
            "elapsed_seconds": time.perf_counter() - started,
        }
        payload["write_status"] = _write_new_or_identical(prediction_path, payload)
        summary = dict(payload)
        summary.pop("records", None)
        summary_path = fold_root / "training-summary.json"
        summary["prediction_path"] = str(prediction_path)
        summary["prediction_sha256"] = sha256_file(prediction_path)
        _write_new_or_identical(summary_path, summary)
        logger.info(
            "run=complete | fold=%02d | epoch=%02d | held_out_predictions=%d | prediction=%s | threshold_fitting=not_executed",
            int(fold_id),
            OOF_FIXED_EPOCHS,
            len(records),
            prediction_path,
        )
        return summary
    finally:
        close_training_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one V14 known-only parent-grouped OOF fold for exactly 20 epochs, "
            "then predict its held-out train parents once."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_CONFIGS["dbb_cu_gate"])
    parser.add_argument("--dbb-config", type=Path, default=DEFAULT_CONFIGS["dbb_specialist"])
    parser.add_argument("--cu-config", type=Path, default=DEFAULT_CONFIGS["cu_specialist"])
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--prediction-batch-size", type=int, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()

    def absolute(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    result = train_oof_fold(
        root,
        config_paths={
            "dbb_cu_gate": absolute(args.gate_config),
            "dbb_specialist": absolute(args.dbb_config),
            "cu_specialist": absolute(args.cu_config),
        },
        fold_id=args.fold_id,
        resume_checkpoint=(
            None if args.resume_checkpoint is None else absolute(args.resume_checkpoint)
        ),
        prediction_batch_size=args.prediction_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
