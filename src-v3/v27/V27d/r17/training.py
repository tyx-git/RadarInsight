"""可选的 V27d-r17 TR-only L2 降级重训。

L1 永远冻结；这里只训练 DBB 三分类 L2。Unknown、test、RD、L3 和任何
Unknown/Pending 伪标签都不进入训练或 checkpoint 选择。
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from v27.v27a.checkpoint import tensor_state_hash
from v27.v27_baseline.metrics import summarize_records

from .data import R17Dataset, build_r17_loader, r17_l2_inputs
from .frozen import FrozenR17Models, load_frozen_models
from .protocol import (
    R17_CHECKPOINT_SCHEMA,
    R17_L2_CLASS_NAMES,
    R17Config,
    artifact_path,
    canonical_sha256,
    load_manifest,
    seal_payload,
    sha256_file,
    write_json_if_identical,
    write_torch_if_identical,
)


def _seed(seed: int) -> None:
    if int(seed) < 0:
        raise ValueError("r17 training seed must be non-negative")
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(config: R17Config, requested: str | None = None) -> torch.device:
    value = str(requested or config.training.get("device", "cpu")).lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("r17 training requested CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r17 training device: {value}")


def _class_weights(dataset: R17Dataset) -> tuple[float, ...]:
    counts = {name: 0 for name in R17_L2_CLASS_NAMES}
    for item in dataset:
        if item.label_name not in counts:
            raise ValueError("r17 L2 dataset contains a non-DBB label")
        counts[item.label_name] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"r17 L2 training support is incomplete: {counts}")
    total = float(sum(counts.values()))
    return tuple(total / (len(counts) * counts[name]) for name in R17_L2_CLASS_NAMES)


def _evaluate_model(
    model: torch.nn.Module,
    dataset: R17Dataset,
    *,
    config: R17Config,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    loader = build_r17_loader(dataset, batch_size=int(config.training["batch_size"]), shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            output = model(**r17_l2_inputs(batch, device))
            logits = output["window_logits"].detach().cpu()
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                final = logits[row]
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "split_name": str(dataset.split_name),
                        "label_name": str(batch["label_names"][row]),
                        "true_id": int(batch["l2_labels"][row]),
                        "pred_id": int(final.argmax().item()),
                        "probabilities": [float(v) for v in torch.softmax(final, -1).tolist()],
                        "logits": [float(v) for v in final.tolist()],
                        "event_count": int(batch["lengths"][row]),
                        "raw_event_count": int(batch["raw_event_counts"][row]),
                        "l2_range_clipped_event_count": int(batch["l2_range_clipped_count"][row]),
                        "l2_speed_clipped_event_count": int(batch["l2_speed_clipped_count"][row]),
                    }
                )
    return summarize_records(records, class_names=R17_L2_CLASS_NAMES, scope="parent_final"), records


def _checkpoint_payload(
    root: Path,
    config: R17Config,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    model: torch.nn.Module,
    *,
    seed: int,
    epoch: int,
    global_step: int,
    class_weights: tuple[float, ...],
    metrics: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    return {
        "checkpoint_schema": R17_CHECKPOINT_SCHEMA,
        "experiment_id": "V27d-r17",
        "task": "l2",
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_path": config.path.relative_to(root).as_posix(),
        "config_sha256": config.sha256,
        "config_identity_sha256": config.identity_sha256,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "source_frozen_l2_checkpoint_sha256": config.source["r23_checkpoint_sha256"],
        "source_frozen_l2_epoch": 1,
        "class_names": list(R17_L2_CLASS_NAMES),
        "class_weights": list(class_weights),
        "loss_name": config.training["loss_name"],
        "loss_reduction": config.training["loss_reduction"],
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "l3_calibration": "not_executed",
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "model_state": state,
        "model_state_sha256": tensor_state_hash(state),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "runtime_input_contract": dict(manifest["runtime_contract"]),
    }


def train_l2(
    project_root: str | Path,
    config: R17Config,
    *,
    seed: int,
    device: str | None = None,
) -> dict[str, Any]:
    if int(seed) not in [int(value) for value in config.training["seeds"]]:
        raise ValueError("r17 training seed is not registered")
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    _seed(int(seed))
    runtime_device = _device(config, device)
    frozen = load_frozen_models(root, config)
    model = copy.deepcopy(frozen.l2_model).to(runtime_device)
    for parameter in model.trajectory_heads.parameters():
        parameter.requires_grad_(False)
    model.trajectory_heads.eval()
    train_set = R17Dataset(root, config, split_name="train", task="l2")
    validation_set = R17Dataset(root, config, split_name="validation", task="l2")
    class_weights = _class_weights(train_set)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=runtime_device)
    loader = build_r17_loader(
        train_set,
        batch_size=int(config.training["batch_size"]),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.training["l2_learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch: int | None = None
    best_payload: dict[str, Any] | None = None
    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, int(config.training["epochs"]) + 1):
        model.train()
        model.trajectory_heads.eval()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**r17_l2_inputs(batch, runtime_device))
            loss = parent_final_fixed_parent_count_ce(
                output["sequence_logits"],
                batch["l2_labels"].to(runtime_device),
                batch["tr_padding_mask"].to(runtime_device),
                class_weights=weights,
            )["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("r17 L2 loss is non-finite")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(config.training["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                ).item()
            )
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, norm)
            clipped_steps += int(norm > float(config.training["gradient_clip_norm"]))
        validation_metrics, validation_records = _evaluate_model(
            model, validation_set, config=config, device=runtime_device
        )
        recalls = [float(row["recall"]) for row in validation_metrics["per_class"]]
        key = (float(validation_metrics["macro_f1"]), min(recalls), -float(epoch))
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "gradient_norm_max": gradient_max,
            "gradient_clipped_steps": clipped_steps,
            "validation_parent_metrics": validation_metrics,
            "validation_parent_records": validation_records,
            "selection_key": list(key),
        }
        history.append(record)
        metrics = {"validation_parent_metrics": validation_metrics, "selection_key": list(key)}
        epoch_payload = _checkpoint_payload(
            root, config, manifest, manifest_path, manifest_sha256, model,
            seed=int(seed), epoch=epoch, global_step=global_step,
            class_weights=class_weights, metrics=metrics, optimizer=optimizer,
        )
        epoch_path = artifact_path(
            root, config.artifacts["epoch_checkpoint_template"], task="l2", seed=int(seed), epoch=epoch
        )
        write_torch_if_identical(epoch_path, epoch_payload)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            # Keep the selected payload in memory and write the terminal best
            # path once after the loop.  The immutable artifact writer must not
            # be asked to replace an earlier, now-inferior best checkpoint.
            best_payload = epoch_payload
    if best_payload is None or best_epoch is None:
        raise RuntimeError("r17 L2 training produced no selectable checkpoint")
    best_path = artifact_path(
        root, config.artifacts["checkpoint_template"], task="l2", seed=int(seed)
    )
    write_torch_if_identical(best_path, best_payload)
    summary = {
        "schema_version": "RADAR-LN3-V27d-r17-tr-only-training-summary-v1",
        "experiment_id": "V27d-r17",
        "status": "completed_l2_tr_only_degraded_training",
        "task": "l2",
        "seed": int(seed),
        "device": str(runtime_device),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "class_weights": list(class_weights),
        "best_epoch": best_epoch,
        "epochs": len(history),
        "history": history,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = artifact_path(root, config.artifacts["training_summary_template"], task="l2", seed=int(seed))
    write_json_if_identical(summary_path, seal_payload(summary))
    return {**summary, "summary_path": str(summary_path)}


__all__ = ["train_l2"]
