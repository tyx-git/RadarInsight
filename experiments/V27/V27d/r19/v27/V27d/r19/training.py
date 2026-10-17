"""Paired r19 L2 training for the SNR-channel ablation."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from v27.V27d.r17.frozen import FrozenR17Models, load_frozen_models
from v27.v27a.checkpoint import tensor_state_hash
from v27.v27_baseline.metrics import summarize_records

from .data import R19Dataset, build_r19_loader, r17_l2_inputs
from .protocol import (
    R19_ARMS,
    R19_CHECKPOINT_SCHEMA,
    R19_EXPERIMENT_ID,
    R19_L2_CLASSES,
    R19_TRAINING_SUMMARY_SCHEMA,
    R19Config,
    artifact_path,
    canonical_sha256,
    load_manifest,
    project_path,
    seal,
    sha256_file,
    write_json_if_identical,
    write_torch_if_identical,
)


def _device(config: R19Config, requested: str | None) -> torch.device:
    value = str(requested or config.training.get("device", "cpu")).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("r19 requested CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r19 device: {value}")


def _seed(seed: int) -> None:
    value = int(seed)
    if value < 0:
        raise ValueError("r19 seed must be non-negative")
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _class_weights(dataset: R19Dataset) -> tuple[float, ...]:
    counts = {name: 0 for name in R19_L2_CLASSES}
    for item in dataset:
        if item.label_name not in counts:
            raise ValueError(f"r19 non-DBB training label: {item.label_name}")
        counts[item.label_name] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"r19 training support is incomplete: {counts}")
    total = float(sum(counts.values()))
    return tuple(total / (len(counts) * counts[name]) for name in R19_L2_CLASSES)


def _evaluate(model: torch.nn.Module, dataset: R19Dataset, *, device: torch.device, batch_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r19_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r17_l2_inputs(batch, device))
            logits = output["window_logits"].detach().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("r19 validation logits are non-finite")
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                final = logits[row]
                records.append({
                    "parent_sample_id": str(parent_id),
                    "source_suffix": str(parent_id).split("|")[2],
                    "label_name": str(batch["label_names"][row]),
                    "true_id": int(batch["l2_labels"][row]),
                    "pred_id": int(final.argmax().item()),
                    "probabilities": [float(value) for value in torch.softmax(final, dim=-1).tolist()],
                    "event_count": int(batch["lengths"][row]),
                })
    if [row["parent_sample_id"] for row in records] != dataset.parent_ids:
        raise ValueError("r19 evaluation changed parent order")
    return summarize_records(records, class_names=R19_L2_CLASSES, scope="parent_final"), records


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _checkpoint_payload(*, root: Path, config: R19Config, manifest_path: Path, manifest_sha256: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, arm: str, seed: int, epoch: int, global_step: int, class_weights: tuple[float, ...], validation_metrics: Mapping[str, Any], selection_key: tuple[float, float, float]) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    return {
        "checkpoint_schema": R19_CHECKPOINT_SCHEMA,
        "experiment_id": R19_EXPERIMENT_ID,
        "arm": arm,
        "task": "l2",
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_path": config.path.relative_to(root).as_posix(),
        "config_sha256": config.sha256,
        "config_identity_sha256": config.identity_sha256,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "source_r18_manifest_sha256": config.source["r18_manifest_sha256"],
        "source_frozen_l2_checkpoint_sha256": config.r17.source["r23_checkpoint_sha256"],
        "snr_arm": arm,
        "radial_speed_retained": True,
        "class_names": list(R19_L2_CLASSES),
        "class_weights": list(class_weights),
        "loss_name": config.training["loss_name"],
        "loss_reduction": config.training["loss_reduction"],
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "model_contract": contract,
        "model_contract_sha256": canonical_sha256(contract),
        "model_state": state,
        "model_state_sha256": tensor_state_hash(state),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "validation_parent_metrics": dict(validation_metrics),
        "selection_key": list(selection_key),
    }


def load_checkpoint(
    project_root: str | Path,
    config: R19Config,
    *,
    arm: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
    frozen: FrozenR17Models | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load an r19 arm only when every provenance binding matches."""

    if arm not in R19_ARMS:
        raise ValueError("r19 checkpoint arm is invalid")
    root = Path(project_root).resolve()
    _, manifest_path, manifest_sha256 = load_manifest(
        root, config, allow_code_drift=True
    )
    path = (
        artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
        if checkpoint_path is None
        else project_path(root, checkpoint_path, must_exist=True)
    )
    try:
        path.relative_to((root / "src-v3" / "v27" / "V27d" / "r19").resolve())
    except ValueError as exc:
        raise ValueError("r19 checkpoint escaped experiment root") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("r19 checkpoint is not an object")
    if (
        payload.get("checkpoint_schema") != R19_CHECKPOINT_SCHEMA
        or payload.get("experiment_id") != R19_EXPERIMENT_ID
        or payload.get("arm") != arm
        or payload.get("task") != "l2"
        or int(payload.get("seed", -1)) != int(seed)
        or int(payload.get("epoch", 0)) <= 0
        or int(payload.get("epoch", 0)) > int(config.training["epochs"])
        or payload.get("config_sha256") != config.sha256
        or payload.get("config_identity_sha256") != config.identity_sha256
        or payload.get("manifest_path") != manifest_path.relative_to(root).as_posix()
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("source_r18_manifest_sha256") != config.source["r18_manifest_sha256"]
        or payload.get("source_frozen_l2_checkpoint_sha256") != config.r17.source["r23_checkpoint_sha256"]
        or payload.get("snr_arm") != arm
        or payload.get("radial_speed_retained") is not True
        or payload.get("class_names") != list(R19_L2_CLASSES)
        or payload.get("loss_name") != config.training["loss_name"]
        or payload.get("loss_reduction") != config.training["loss_reduction"]
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or int(payload.get("test_payload_access_count", -1)) != 0
        or int(payload.get("rd_payload_access_count", -1)) != 0
    ):
        raise ValueError("r19 checkpoint provenance differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("r19 checkpoint model state hash differs")
    source = frozen or load_frozen_models(root, config.r17)
    model = copy.deepcopy(source.l2_model).cpu()
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("r19 checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    return model.eval(), dict(payload)


def train_arm(project_root: str | Path, config: R19Config, *, arm: str, seed: int, device: str | None = None) -> dict[str, Any]:
    if arm not in R19_ARMS:
        raise ValueError("r19 arm is invalid")
    if int(seed) not in [int(value) for value in config.training["seeds"]]:
        raise ValueError("r19 seed is not registered")
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    _seed(int(seed))
    runtime_device = _device(config, device)
    frozen = load_frozen_models(root, config.r17)
    train_set = R19Dataset(root, config, split_name="train", arm=arm)
    validation_set = R19Dataset(root, config, split_name="validation", arm=arm)
    expected_train = [parent_id for parent_id, split in manifest["parent_assignments"].items() if split == "train"]
    expected_validation = [parent_id for parent_id, split in manifest["parent_assignments"].items() if split == "validation"]
    if train_set.parent_ids != expected_train or validation_set.parent_ids != expected_validation:
        raise ValueError("r19 dataset order differs from the sealed parent split")
    class_weights = _class_weights(train_set)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=runtime_device)
    batch_size = int(config.training["batch_size"])
    loader = build_r19_loader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator(device="cpu").manual_seed(int(seed)))
    model = copy.deepcopy(frozen.l2_model).to(runtime_device)
    for parameter in model.trajectory_heads.parameters():
        parameter.requires_grad_(False)
    model.trajectory_heads.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config.training["l2_learning_rate"]), weight_decay=float(config.training["weight_decay"]))
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_payload: dict[str, Any] | None = None
    best_epoch: int | None = None
    global_step = 0
    started = time.perf_counter()
    epochs = int(config.training["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        model.trajectory_heads.eval()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**r17_l2_inputs(batch, runtime_device))
            loss = parent_final_fixed_parent_count_ce(output["sequence_logits"], batch["l2_labels"].to(runtime_device), batch["tr_padding_mask"].to(runtime_device), class_weights=weights)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("r19 loss is non-finite")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(config.training["gradient_clip_norm"]), error_if_nonfinite=True).item())
            if not np.isfinite(norm):
                raise FloatingPointError("r19 gradient norm is non-finite")
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, norm)
            clipped_steps += int(norm > float(config.training["gradient_clip_norm"]))
        validation_metrics, _ = _evaluate(model, validation_set, device=runtime_device, batch_size=batch_size)
        recalls = [float(row["recall"]) for row in validation_metrics["per_class"]]
        key = (float(validation_metrics["macro_f1"]), min(recalls), -float(epoch))
        payload = _checkpoint_payload(root=root, config=config, manifest_path=manifest_path, manifest_sha256=manifest_sha256, model=model, optimizer=optimizer, arm=arm, seed=int(seed), epoch=epoch, global_step=global_step, class_weights=class_weights, validation_metrics=validation_metrics, selection_key=key)
        epoch_path = artifact_path(root, config.artifacts["epoch_checkpoint_template"], arm=arm, seed=int(seed), epoch=epoch)
        write_torch_if_identical(epoch_path, payload)
        history.append({"epoch": epoch, "global_step": global_step, "train_loss": float(np.mean(losses)), "gradient_norm_max": gradient_max, "gradient_clipped_steps": clipped_steps, "validation_parent_metrics": validation_metrics, "selection_key": list(key)})
        print(f"[V27d-r19] arm={arm} epoch={epoch}/{epochs} loss={np.mean(losses):.6f} val_macro_f1={validation_metrics['macro_f1']:.6f}", flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_payload = key, epoch, payload
    if best_payload is None or best_epoch is None or best_key is None:
        raise RuntimeError("r19 produced no selectable checkpoint")
    best_path = artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
    write_torch_if_identical(best_path, best_payload)
    summary = {"schema_version": R19_TRAINING_SUMMARY_SCHEMA, "experiment_id": R19_EXPERIMENT_ID, "status": "completed_snr_ablation_training", "arm": arm, "seed": int(seed), "best_epoch": best_epoch, "epochs": len(history), "manifest_path": manifest_path.relative_to(root).as_posix(), "manifest_sha256": manifest_sha256, "checkpoint_path": best_path.relative_to(root).as_posix(), "checkpoint_sha256": sha256_file(best_path), "checkpoint_model_state_sha256": best_payload["model_state_sha256"], "class_weights": list(class_weights), "radial_speed_retained": True, "history": history, "unknown_training_exposure": 0, "test_payload_access_count": 0, "rd_payload_access_count": 0, "elapsed_seconds": time.perf_counter() - started}
    summary_path = artifact_path(root, config.artifacts["training_summary_template"], arm=arm, seed=int(seed))
    write_json_if_identical(summary_path, seal(summary))
    return {**summary, "summary_path": str(summary_path)}


__all__ = ["load_checkpoint", "train_arm"]
