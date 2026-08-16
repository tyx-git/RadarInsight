from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.micro_doppler_residual_v19r import (
    MicroDopplerLateResidual,
    V19R_CHECKPOINT_SCHEMA_VERSION,
    V19R_CONFIG_NAME,
    V19R_CONFIG_RELATIVE_PATH,
    V19R_SCHEMA_VERSION,
    V19_CLASS_NAMES,
    V19_RESIDUAL_BOUND,
    V19_TRAINABLE_PARAMETER_COUNT,
    V19rResidualCacheDataset,
    build_residual_dataloader,
    evaluate_residual,
    load_v19r_residual_checkpoint,
    require_v19r_parent_order,
    residual_objective,
    save_v19r_residual_checkpoint,
    sha256_file,
    strip_evaluation_records,
)
from radar_ln3.task_models import seed_everything
from train.tasks import build_checkpoint_selection, checkpoint_selection_is_better


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_runtime(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != V19R_SCHEMA_VERSION
        or config.get("config_name") != V19R_CONFIG_NAME
        or tuple(config.get("class_names") or ()) != V19_CLASS_NAMES
    ):
        raise ValueError("V19r training received a different config contract")
    expected_model = {
        "baseline": "frozen V18r density+stable8 best logits",
        "linear_shape": [3, 8],
        "bias": False,
        "zero_initialized": True,
        "zero_sum_across_classes": True,
        "absolute_residual_bound": V19_RESIDUAL_BOUND,
        "maximum_pairwise_logit_correction": 2.0 * V19_RESIDUAL_BOUND,
        "trainable_parameter_count": V19_TRAINABLE_PARAMETER_COUNT,
    }
    if dict(config.get("model") or {}) != expected_model:
        raise ValueError("V19r model contract differs from the frozen 24-weight residual")
    train = dict(config.get("train") or {})
    if (
        int(train.get("seed", -1)) != 42
        or int(train.get("dataloader_seed", -1)) != 42
        or int(train.get("epochs", -1)) != 20
        or int(train.get("early_stopping_patience", -1)) != 0
        or train.get("loss_mode") != "parent_final_task_ce_with_prefix_auxiliary"
        or float(train.get("prefix_auxiliary_weight", -1.0)) != 0.1
        or int(train.get("prefix_auxiliary_min_events", -1)) != 3
        or train.get("checkpoint_schema") != V19R_CHECKPOINT_SCHEMA_VERSION
        or train.get("save_every_epoch") is not True
        or train.get("record_parent_order_hash") is not True
    ):
        raise ValueError("V19r training protocol differs from the frozen design")
    config_hash = sha256_file(config_path)
    path_hash_fields = (
        ("contract_path", "expected_contract_sha256"),
        ("normalization_path", "expected_normalization_sha256"),
        ("train_cache_path", "expected_train_cache_sha256"),
        ("validation_cache_path", "expected_validation_cache_sha256"),
        ("baseline_checkpoint_path", "expected_baseline_checkpoint_sha256"),
        (
            "baseline_training_summary_path",
            "expected_baseline_training_summary_sha256",
        ),
        ("v18r_paired_report_path", "expected_v18r_paired_report_sha256"),
        ("source_v19_config_path", "expected_source_v19_config_sha256"),
        ("source_v19_readiness_path", "expected_source_v19_readiness_sha256"),
    )
    resolved: dict[str, Path] = {}
    for path_field, hash_field in path_hash_fields:
        path = _resolve(root, config[path_field])
        if not path.is_file() or sha256_file(path) != config[hash_field]:
            raise ValueError(f"V19r bound input differs: {path_field}={path}")
        resolved[path_field] = path
    readiness_path = _resolve(root, config["readiness_path"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("config_sha256") != config_hash
        or readiness.get("contract_sha256") != config["expected_contract_sha256"]
        or readiness.get("baseline_checkpoint_sha256")
        != config["expected_baseline_checkpoint_sha256"]
        or readiness.get("baseline_backbone_state_sha256")
        != config["expected_baseline_backbone_state_sha256"]
        or int(readiness.get("baseline_checkpoint_epoch", -1)) != 15
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or readiness.get("deployment_eligible") is not False
        or not all(dict(readiness.get("checks") or {}).values())
    ):
        raise ValueError("V19r readiness binding is invalid")
    source_summary = json.loads(
        resolved["baseline_training_summary_path"].read_text(encoding="utf-8")
    )
    source_orders = {
        str(int(row["epoch"])): str(row["parent_order_sha256"])
        for row in source_summary.get("history") or []
    }
    if (
        source_summary.get("status") != "completed"
        or int(source_summary.get("best_epoch", -1)) != 15
        or source_orders
        != dict(train.get("reference_parent_order_sha256_by_epoch") or {})
    ):
        raise ValueError("V19r config parent-order references differ from V18r")
    paired = json.loads(
        resolved["v18r_paired_report_path"].read_text(encoding="utf-8")
    )
    treatment = dict(dict(paired.get("runs") or {}).get("treatment") or {})
    if (
        paired.get("status") != "completed_but_gate_failed"
        or paired.get("test") != "未执行"
        or treatment.get("best_checkpoint_sha256")
        != config["expected_baseline_checkpoint_sha256"]
        or int(treatment.get("best_epoch", -1)) != 15
    ):
        raise ValueError("V19r source V18r result is not the frozen completed experiment")
    v19_readiness = json.loads(
        resolved["source_v19_readiness_path"].read_text(encoding="utf-8")
    )
    if (
        v19_readiness.get("status") != "passed"
        or v19_readiness.get("config_sha256")
        != config["expected_source_v19_config_sha256"]
        or v19_readiness.get("test") != "未执行"
        or v19_readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V19r source V19 feature contract is stale or unsealed")
    return {
        "config": config,
        "config_path": config_path,
        "config_sha256": config_hash,
        "readiness_path": readiness_path,
        "readiness_sha256": sha256_file(readiness_path),
        "resolved": resolved,
        "source_summary": source_summary,
        "source_orders": source_orders,
    }


def _device(name: str) -> torch.device:
    value = str(name).strip().lower()
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V19r formal training requires CUDA from the frozen config")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported V19r device: {name}")


def _provenance(runtime: Mapping[str, Any]) -> dict[str, Any]:
    config = runtime["config"]
    return {
        "contract_sha256": config["expected_contract_sha256"],
        "normalization_sha256": config["expected_normalization_sha256"],
        "train_cache_sha256": config["expected_train_cache_sha256"],
        "validation_cache_sha256": config["expected_validation_cache_sha256"],
        "baseline_checkpoint_sha256": config["expected_baseline_checkpoint_sha256"],
        "baseline_backbone_state_sha256": config[
            "expected_baseline_backbone_state_sha256"
        ],
        "baseline_training_summary_sha256": config[
            "expected_baseline_training_summary_sha256"
        ],
        "v18r_paired_report_sha256": config["expected_v18r_paired_report_sha256"],
        "source_v19_config_sha256": config["expected_source_v19_config_sha256"],
        "source_v19_readiness_sha256": config[
            "expected_source_v19_readiness_sha256"
        ],
        "readiness_sha256": runtime["readiness_sha256"],
    }


def _capture_training_state(
    generator: torch.Generator,
    *,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "dataloader_generator_state": generator.get_state(),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_selection": None if best_selection is None else dict(best_selection),
    }
    if torch.cuda.is_available():
        state["cuda_random_states"] = torch.cuda.get_rng_state_all()
    return state


def _restore_training_state(
    state: Mapping[str, Any], generator: torch.Generator
) -> tuple[list[dict[str, Any]], int, dict[str, Any] | None]:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    generator.set_state(state["dataloader_generator_state"])
    if torch.cuda.is_available() and "cuda_random_states" in state:
        torch.cuda.set_rng_state_all(state["cuda_random_states"])
    history = [dict(row) for row in state.get("history") or []]
    best = state.get("best_selection")
    return history, int(state.get("best_epoch", -1)), None if best is None else dict(best)


def _initial_equality(
    model: MicroDopplerLateResidual,
    datasets: list[V19rResidualCacheDataset],
    *,
    device: torch.device,
    batch_size: int,
) -> bool:
    model.eval()
    with torch.inference_mode():
        for dataset in datasets:
            loader = build_residual_dataloader(
                dataset, batch_size=batch_size, shuffle=False
            )
            for batch in loader:
                baseline = batch["baseline_logits"].to(device)
                output = model(
                    baseline,
                    batch["causal_features"].to(device),
                    batch["padding_mask"].to(device),
                )
                if not torch.equal(output["sequence_logits"], baseline):
                    return False
                if bool((output["residual_logits"] != 0.0).any()):
                    return False
    return True


def _performance_guardrails(evaluation: Mapping[str, Any], target: float) -> dict[str, Any]:
    rows = list(dict(evaluation.get("parent_metrics") or {}).get("per_class") or [])
    guardrails: list[dict[str, Any]] = []
    values: list[float] = []
    for row in rows:
        metrics = {name: float(row[name]) for name in ("precision", "recall", "f1")}
        values.extend(metrics.values())
        guardrails.append(
            {
                "class": str(row["class"]),
                "support": int(row["support"]),
                **metrics,
                "target": float(target),
                "passed": all(value >= target for value in metrics.values()),
            }
        )
    passed = len(guardrails) == len(V19_CLASS_NAMES) and all(
        row["passed"] for row in guardrails
    )
    return {
        "performance_status": "passed" if passed else "failed",
        "target_min_per_class_precision_recall_f1": float(target),
        "observed_min_per_class_precision_recall_f1": min(values) if values else None,
        "guardrails": guardrails,
    }


def train(
    project_root: Path,
    config_path: Path,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    runtime = _load_runtime(root, config_path.resolve())
    config = runtime["config"]
    train_config = dict(config["train"])
    artifact = _resolve(root, train_config["artifact_dir"])
    summary_path = artifact / "training-summary.json"
    best_path = _resolve(root, train_config["checkpoint_path"])
    last_path = _resolve(root, train_config["last_checkpoint_path"])
    epoch_dir = _resolve(root, train_config["epoch_checkpoint_dir"])
    if resume_checkpoint is None and artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(
            f"refusing to start V19r over a non-empty formal artifact: {artifact}"
        )
    artifact.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    device = _device(train_config["device"])
    seed_everything(int(train_config["seed"]))
    train_dataset = V19rResidualCacheDataset(runtime["resolved"]["train_cache_path"])
    validation_dataset = V19rResidualCacheDataset(
        runtime["resolved"]["validation_cache_path"]
    )
    if train_dataset.split_name != "train" or validation_dataset.split_name != "val":
        raise ValueError("V19r train/validation cache split names differ")
    model = MicroDopplerLateResidual().to(device)
    if model.trainable_parameter_count != V19_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("V19r model does not expose exactly 24 trainable weights")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_config["learning_rate"])
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(train_config["dataloader_seed"]))
    provenance = _provenance(runtime)
    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_selection: dict[str, Any] | None = None
    global_step = 0
    start_epoch = 1
    initial_equal = False
    if resume_checkpoint is None:
        initial_equal = _initial_equality(
            model,
            [train_dataset, validation_dataset],
            device=device,
            batch_size=int(train_config["batch_size"]),
        )
        if not initial_equal:
            raise ValueError("V19r zero-init logits do not exactly equal frozen V18r")
    else:
        resume_path = _resolve(root, resume_checkpoint)
        payload = load_v19r_residual_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            expected_config_name=V19R_CONFIG_NAME,
            expected_config_sha256=runtime["config_sha256"],
            expected_provenance=provenance,
            map_location=device,
        )
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        history, best_epoch, best_selection = _restore_training_state(
            payload["training_state"], generator
        )
        if [int(row["epoch"]) for row in history] != list(range(1, start_epoch)):
            raise ValueError("V19r resume history is not contiguous")
        initial_equal = True
    loader = build_residual_dataloader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    class_weights = torch.tensor(
        train_config["class_weights"], dtype=torch.float32, device=device
    )
    started_at = time.perf_counter()
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, int(train_config["epochs"]) + 1):
            current_epoch = epoch
            model.train()
            epoch_started = time.perf_counter()
            loss_total = 0.0
            main_total = 0.0
            prefix_total = 0.0
            steps = 0
            parent_order: list[str] = []
            gradient_abs_max = 0.0
            for batch in loader:
                parent_order.extend(str(value) for value in batch["parent_sample_id"])
                tensors = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    tensors["baseline_logits"],
                    tensors["causal_features"],
                    tensors["padding_mask"],
                )
                objective = residual_objective(
                    output["sequence_logits"],
                    tensors,
                    class_weights=class_weights,
                    prefix_auxiliary_weight=float(
                        train_config["prefix_auxiliary_weight"]
                    ),
                    prefix_auxiliary_min_events=int(
                        train_config["prefix_auxiliary_min_events"]
                    ),
                )
                objective["loss"].backward()
                if model.linear.weight.grad is None or not torch.isfinite(
                    model.linear.weight.grad
                ).all():
                    raise ValueError("V19r residual gradient is missing or non-finite")
                gradient_abs_max = max(
                    gradient_abs_max,
                    float(model.linear.weight.grad.abs().max().item()),
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["gradient_clip_norm"])
                )
                optimizer.step()
                global_step += 1
                steps += 1
                loss_total += float(objective["loss"].detach().item())
                main_total += float(objective["final_state_loss"].detach().item())
                prefix_total += float(
                    objective["prefix_auxiliary_loss"].detach().item()
                )
            order_hash = require_v19r_parent_order(
                parent_order,
                runtime["source_orders"][str(epoch)],
                epoch=epoch,
            )
            validation = evaluate_residual(
                model,
                validation_dataset,
                device=device,
                batch_size=int(train_config["batch_size"]),
                prefix_auxiliary_min_events=int(
                    train_config["prefix_auxiliary_min_events"]
                ),
            )
            selection = build_checkpoint_selection(
                validation, train_config["checkpoint_selection_metric"]
            )
            selected_as_best = checkpoint_selection_is_better(
                selection, best_selection
            )
            if selected_as_best:
                best_selection = selection
                best_epoch = epoch
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": loss_total / max(1, steps),
                "train_main_loss": main_total / max(1, steps),
                "train_prefix_auxiliary_loss": prefix_total / max(1, steps),
                "gradient_absolute_max": gradient_abs_max,
                "residual_weight_absolute_max": float(
                    model.linear.weight.detach().abs().max().item()
                ),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "selection_metric": train_config["checkpoint_selection_metric"],
                "checkpoint_selection": selection,
                "selected_as_best": selected_as_best,
                "validation": strip_evaluation_records(validation),
                "parent_order_sha256": order_hash,
                "parent_order_count": len(parent_order),
                "parent_order_unique_count": len(set(parent_order)),
            }
            history.append(epoch_record)
            training_state = _capture_training_state(
                generator,
                history=history,
                best_epoch=best_epoch,
                best_selection=best_selection,
            )
            checkpoint_targets = [
                last_path,
                epoch_dir / f"epoch-{epoch:03d}.pt",
            ]
            if selected_as_best:
                checkpoint_targets.append(best_path)
            for destination in checkpoint_targets:
                save_v19r_residual_checkpoint(
                    destination,
                    model,
                    optimizer,
                    config_name=V19R_CONFIG_NAME,
                    config_sha256=runtime["config_sha256"],
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_record,
                    provenance=provenance,
                    training_state=training_state,
                )
            _atomic_json(
                summary_path,
                {
                    "status": "running",
                    "schema_version": V19R_SCHEMA_VERSION,
                    "config_name": V19R_CONFIG_NAME,
                    "current_epoch": epoch,
                    "best_epoch": best_epoch,
                    "global_step": global_step,
                    "history": history,
                    "test_evaluation": "not_executed",
                    "full_cascade_evaluation": "not_executed",
                    "oof_rejection_calibration": "not_executed",
                    "stable_evaluation": "not_executed",
                    "p95_latency": "not_executed",
                    "deployment_eligible": False,
                },
            )
        if best_epoch <= 0 or not best_path.is_file():
            raise ValueError("V19r training did not select a best checkpoint")
        best_payload = load_v19r_residual_checkpoint(
            best_path,
            model,
            expected_config_name=V19R_CONFIG_NAME,
            expected_config_sha256=runtime["config_sha256"],
            expected_provenance=provenance,
            map_location=device,
        )
        if int(best_payload["epoch"]) != best_epoch:
            raise ValueError("V19r best checkpoint epoch differs from training history")
        best_validation = evaluate_residual(
            model,
            validation_dataset,
            device=device,
            batch_size=int(train_config["batch_size"]),
            prefix_auxiliary_min_events=int(
                train_config["prefix_auxiliary_min_events"]
            ),
        )
        if strip_evaluation_records(best_validation) != history[best_epoch - 1][
            "validation"
        ]:
            raise ValueError("V19r independent best-checkpoint evaluation differs")
        guardrails = _performance_guardrails(
            best_validation,
            float(config["adoption"]["minimum_per_class_precision_recall_f1"]),
        )
        sample_detail_path = artifact / "sample_detail_information.json"
        _atomic_json(
            sample_detail_path,
            {
                "schema_version": f"{V19R_SCHEMA_VERSION}-sample-detail-v1",
                "checkpoint": str(best_path),
                "checkpoint_sha256": sha256_file(best_path),
                "epoch": best_epoch,
                "counts": {
                    key: len(value) for key, value in best_validation["records"].items()
                },
                "records": best_validation["records"],
                "test": "未执行",
            },
        )
        summary = {
            "status": "completed",
            "performance_status": guardrails["performance_status"],
            "schema_version": V19R_SCHEMA_VERSION,
            "config_name": V19R_CONFIG_NAME,
            "task_name": "dbb_specialist",
            "single_factor": (
                "frozen V18r density+stable8 logits plus a zero-initialized "
                "24-weight bounded zero-sum stable8 late residual"
            ),
            "stop_reason": "epoch_limit",
            "epochs": int(train_config["epochs"]),
            "best_epoch": best_epoch,
            "global_step": global_step,
            "history": history,
            "best_validation_metrics": strip_evaluation_records(best_validation),
            "performance_guardrails": guardrails,
            "initial_combined_logits_exactly_equal_frozen_v18r": initial_equal,
            "trainable_parameter_count": model.trainable_parameter_count,
            "residual_weight_absolute_max": float(
                model.linear.weight.detach().abs().max().item()
            ),
            "baseline_checkpoint_path": config["baseline_checkpoint_path"],
            "baseline_checkpoint_sha256": config[
                "expected_baseline_checkpoint_sha256"
            ],
            "baseline_backbone_state_sha256": config[
                "expected_baseline_backbone_state_sha256"
            ],
            "config_path": str(config_path),
            "config_sha256": runtime["config_sha256"],
            "readiness_path": str(runtime["readiness_path"]),
            "readiness_sha256": runtime["readiness_sha256"],
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint": str(last_path),
            "last_checkpoint_sha256": sha256_file(last_path),
            "sample_detail": str(sample_detail_path),
            "sample_detail_sha256": sha256_file(sample_detail_path),
            "elapsed_seconds": time.perf_counter() - started_at,
            "validation_threshold_tuning": "not_executed",
            "test_evaluation": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "oof_rejection_calibration": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "deployment_eligible": False,
        }
        _atomic_json(summary_path, summary)
        return summary
    except Exception as exc:
        _atomic_json(
            summary_path,
            {
                "status": "failed",
                "performance_status": "not_evaluated",
                "schema_version": V19R_SCHEMA_VERSION,
                "config_name": V19R_CONFIG_NAME,
                "current_epoch": current_epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "history": history,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resume_checkpoint": str(last_path) if last_path.is_file() else None,
                "test_evaluation": "not_executed",
                "full_cascade_evaluation": "not_executed",
                "oof_rejection_calibration": "not_executed",
                "stable_evaluation": "not_executed",
                "p95_latency": "not_executed",
                "deployment_eligible": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the V19r frozen-V18r micro-Doppler late residual."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--config", type=Path, default=V19R_CONFIG_RELATIVE_PATH)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    resume = args.resume_checkpoint
    if resume is not None and not resume.is_absolute():
        resume = root / resume
    result = train(root, config_path, resume_checkpoint=resume)
    print(
        json.dumps(
            {
                "status": result["status"],
                "performance_status": result["performance_status"],
                "best_epoch": result["best_epoch"],
                "best_checkpoint": result["best_checkpoint"],
                "test_evaluation": result["test_evaluation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
