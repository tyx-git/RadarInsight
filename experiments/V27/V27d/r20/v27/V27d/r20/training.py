"""Paired r20 training for the structured radial-speed decomposition."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from v27.V27d.r19.training import load_checkpoint as load_r19_checkpoint
from v27.v27a.checkpoint import tensor_state_hash
from v27.v27_baseline.metrics import summarize_records

from .data import R20Dataset, build_r20_loader, r20_model_input_kwargs
from .model import R20L2Model
from .protocol import (
    R20_ARMS,
    R20_CHECKPOINT_SCHEMA,
    R20_EXPERIMENT_ID,
    R20_L2_CLASSES,
    R20_TRAINING_SUMMARY_SCHEMA,
    R20Config,
    artifact_path,
    canonical_sha256,
    load_manifest,
    project_path,
    seal,
    sha256_file,
    write_json_if_identical,
    write_torch_if_identical,
)


def _device(config: R20Config, requested: str | None) -> torch.device:
    value = str(requested or config.training.get("device", "cpu")).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("r20 requested CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r20 device: {value}")


def _seed(seed: int) -> None:
    value = int(seed)
    if value < 0:
        raise ValueError("r20 seed must be non-negative")
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _class_weights(dataset: R20Dataset) -> tuple[float, ...]:
    counts = {name: 0 for name in R20_L2_CLASSES}
    for item in dataset:
        label = item.sequence.label_name
        if label not in counts:
            raise ValueError(f"r20 training received non-DBB label: {label}")
        counts[label] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"r20 training support is incomplete: {counts}")
    total = float(sum(counts.values()))
    return tuple(total / (len(counts) * counts[name]) for name in R20_L2_CLASSES)


def _freeze_unused_heads(model: R20L2Model) -> None:
    heads = getattr(model.base_model, "trajectory_heads", None)
    if heads is None:
        raise ValueError("r20 base model has no trajectory heads to freeze")
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in heads.parameters()):
        raise AssertionError("r20 trajectory heads were not frozen")


def _build_from_r19(root: Path, config: R20Config, *, device: torch.device) -> tuple[R20L2Model, dict[str, Any]]:
    source_path = project_path(root, config.source["r19_checkpoint_path"], must_exist=True)
    model, payload = load_r19_checkpoint(
        root,
        config.r19,
        arm="snr_missing_only",
        seed=int(config.source["r19_checkpoint_seed"]),
        checkpoint_path=source_path,
    )
    if sha256_file(source_path) != config.source["r19_checkpoint_sha256"]:
        raise ValueError("r20 direct r19 checkpoint hash differs")
    if payload.get("model_state_sha256") != config.source["r19_checkpoint_model_state_sha256"]:
        raise ValueError("r20 direct r19 checkpoint state hash differs")
    if (
        payload.get("arm") != config.source["r19_checkpoint_arm"]
        or int(payload.get("seed", -1)) != int(config.source["r19_checkpoint_seed"])
        or int(payload.get("epoch", -1)) != int(config.source["r19_checkpoint_epoch"])
    ):
        raise ValueError("r20 direct r19 checkpoint identity differs")
    wrapped = R20L2Model(config, base_model=copy.deepcopy(model).cpu(), arm="zero_control")
    _freeze_unused_heads(wrapped)
    return wrapped.to(device), dict(payload)


def _evaluate(
    model: R20L2Model,
    dataset: R20Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r20_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r20_model_input_kwargs(batch, device))
            logits = output["window_logits"].detach().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("r20 evaluation logits are non-finite")
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                final = logits[row]
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "source_suffix": str(parent_id).split("|")[2],
                        "label_name": str(batch["label_names"][row]),
                        "true_id": int(batch["l2_labels"][row]),
                        "pred_id": int(final.argmax().item()),
                        "probabilities": [
                            float(value)
                            for value in torch.softmax(final, dim=-1).tolist()
                        ],
                        "event_count": int(batch["lengths"][row]),
                        "radial_valid_event_counts": [
                            int(value)
                            for value in batch["radial_sidecar_valid"][row]
                            .sum(dim=0)
                            .tolist()
                        ],
                    }
                )
    if [row["parent_sample_id"] for row in records] != dataset.parent_ids:
        raise ValueError("r20 evaluation changed parent order")
    return (
        summarize_records(records, class_names=R20_L2_CLASSES, scope="parent_final"),
        records,
    )


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


def _checkpoint_payload(
    *,
    root: Path,
    config: R20Config,
    manifest_path: Path,
    manifest_sha256: str,
    model: R20L2Model,
    optimizer: torch.optim.Optimizer,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    class_weights: tuple[float, ...],
    validation_metrics: Mapping[str, Any],
    selection_key: tuple[float, float, float],
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    return {
        "checkpoint_schema": R20_CHECKPOINT_SCHEMA,
        "experiment_id": R20_EXPERIMENT_ID,
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
        "source_r19_checkpoint_path": config.source["r19_checkpoint_path"],
        "source_r19_checkpoint_sha256": config.source["r19_checkpoint_sha256"],
        "source_r19_checkpoint_model_state_sha256": config.source["r19_checkpoint_model_state_sha256"],
        "source_r19_checkpoint_arm": config.source["r19_checkpoint_arm"],
        "source_r19_checkpoint_seed": int(config.source["r19_checkpoint_seed"]),
        "source_r19_checkpoint_epoch": int(config.source["r19_checkpoint_epoch"]),
        "raw_radial_scalar_removed": True,
        "radial_sidecar_value_names": list(config.data["radial_value_names"]),
        "radial_sidecar_valid_names": list(config.data["radial_valid_names"]),
        "radial_sidecar_positive_dt_boundary": True,
        "class_names": list(R20_L2_CLASSES),
        "class_weights": list(class_weights),
        "loss_name": config.training["loss_name"],
        "loss_reduction": config.training["loss_reduction"],
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "l3": "not_executed",
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
    config: R20Config,
    *,
    arm: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> tuple[R20L2Model, dict[str, Any]]:
    if arm not in R20_ARMS:
        raise ValueError("r20 checkpoint arm is invalid")
    if int(seed) not in [int(value) for value in config.training["seeds"]]:
        raise ValueError("r20 checkpoint seed is not registered")
    root = Path(project_root).resolve()
    _, manifest_path, manifest_sha256 = load_manifest(root, config, allow_code_drift=True)
    path = (
        artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
        if checkpoint_path is None
        else project_path(root, checkpoint_path, must_exist=True)
    )
    try:
        path.relative_to((root / "src-v3" / "v27" / "V27d" / "r20").resolve())
    except ValueError as exc:
        raise ValueError("r20 checkpoint escaped the experiment root") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("r20 checkpoint is not an object")
    if (
        payload.get("checkpoint_schema") != R20_CHECKPOINT_SCHEMA
        or payload.get("experiment_id") != R20_EXPERIMENT_ID
        or payload.get("arm") != arm
        or payload.get("task") != "l2"
        or int(payload.get("seed", -1)) != int(seed)
        or int(payload.get("epoch", 0)) <= 0
        or int(payload.get("epoch", 0)) > int(config.training["epochs"])
        or payload.get("config_sha256") != config.sha256
        or payload.get("config_identity_sha256") != config.identity_sha256
        or payload.get("manifest_path") != manifest_path.relative_to(root).as_posix()
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("source_r19_checkpoint_sha256") != config.source["r19_checkpoint_sha256"]
        or payload.get("source_r19_checkpoint_model_state_sha256") != config.source["r19_checkpoint_model_state_sha256"]
        or payload.get("raw_radial_scalar_removed") is not True
        or payload.get("radial_sidecar_value_names") != list(config.data["radial_value_names"])
        or payload.get("radial_sidecar_valid_names") != list(config.data["radial_valid_names"])
        or payload.get("class_names") != list(R20_L2_CLASSES)
        or payload.get("loss_name") != config.training["loss_name"]
        or payload.get("loss_reduction") != config.training["loss_reduction"]
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or int(payload.get("test_payload_access_count", -1)) != 0
        or int(payload.get("rd_payload_access_count", -1)) != 0
    ):
        raise ValueError("r20 checkpoint provenance differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("r20 checkpoint model state hash differs")
    model, _ = _build_from_r19(root, config, device=torch.device("cpu"))
    model.arm = arm
    if payload.get("model_contract") != model.model_contract() or payload.get("model_contract_sha256") != canonical_sha256(model.model_contract()):
        raise ValueError("r20 checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    _freeze_unused_heads(model)
    return model.eval(), dict(payload)


def train_arm(
    project_root: str | Path,
    config: R20Config,
    *,
    arm: str,
    seed: int,
    device: str | None = None,
) -> dict[str, Any]:
    if arm not in R20_ARMS:
        raise ValueError("r20 arm is invalid")
    if int(seed) not in [int(value) for value in config.training["seeds"]]:
        raise ValueError("r20 seed is not registered")
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    _seed(int(seed))
    runtime_device = _device(config, device)
    model, source_payload = _build_from_r19(root, config, device=runtime_device)
    model.arm = arm
    train_set = R20Dataset(root, config, split_name="train", arm=arm)
    validation_set = R20Dataset(root, config, split_name="validation", arm=arm)
    expected_train = [parent_id for parent_id, split in manifest["parent_assignments"].items() if split == "train"]
    expected_validation = [parent_id for parent_id, split in manifest["parent_assignments"].items() if split == "validation"]
    if train_set.parent_ids != expected_train or validation_set.parent_ids != expected_validation:
        raise ValueError("r20 dataset order differs from the sealed parent split")
    class_weights = _class_weights(train_set)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=runtime_device)
    batch_size = int(config.training["batch_size"])
    loader = build_r20_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    _freeze_unused_heads(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("r20 arm has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.training["l2_learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_payload: dict[str, Any] | None = None
    best_epoch: int | None = None
    global_step = 0
    started = time.perf_counter()
    epochs = int(config.training["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        model.base_model.trajectory_heads.eval()
        losses: list[float] = []
        gradient_max = 0.0
        clipped_steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**r20_model_input_kwargs(batch, runtime_device))
            loss = parent_final_fixed_parent_count_ce(
                output["sequence_logits"],
                batch["l2_labels"].to(runtime_device),
                batch["tr_padding_mask"].to(runtime_device),
                class_weights=weights,
            )["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("r20 loss is non-finite")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    float(config.training["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                ).item()
            )
            if not np.isfinite(norm):
                raise FloatingPointError("r20 gradient norm is non-finite")
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, norm)
            clipped_steps += int(norm > float(config.training["gradient_clip_norm"]))
        validation_metrics, _ = _evaluate(
            model,
            validation_set,
            device=runtime_device,
            batch_size=batch_size,
        )
        recalls = [float(row["recall"]) for row in validation_metrics["per_class"]]
        key = (float(validation_metrics["macro_f1"]), min(recalls), -float(epoch))
        payload = _checkpoint_payload(
            root=root,
            config=config,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            model=model,
            optimizer=optimizer,
            arm=arm,
            seed=int(seed),
            epoch=epoch,
            global_step=global_step,
            class_weights=class_weights,
            validation_metrics=validation_metrics,
            selection_key=key,
        )
        epoch_path = artifact_path(
            root,
            config.artifacts["epoch_checkpoint_template"],
            arm=arm,
            seed=int(seed),
            epoch=epoch,
        )
        write_torch_if_identical(epoch_path, payload)
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(np.mean(losses)),
                "gradient_norm_max": gradient_max,
                "gradient_clipped_steps": clipped_steps,
                "validation_parent_metrics": validation_metrics,
                "selection_key": list(key),
            }
        )
        print(
            f"[V27d-r20] arm={arm} epoch={epoch}/{epochs} "
            f"loss={np.mean(losses):.6f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.6f}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key, best_epoch, best_payload = key, epoch, payload
    if best_payload is None or best_epoch is None or best_key is None:
        raise RuntimeError("r20 produced no selectable checkpoint")
    best_path = artifact_path(
        root,
        config.artifacts["checkpoint_template"],
        arm=arm,
        seed=int(seed),
    )
    write_torch_if_identical(best_path, best_payload)
    summary = {
        "schema_version": R20_TRAINING_SUMMARY_SCHEMA,
        "experiment_id": R20_EXPERIMENT_ID,
        "status": "completed_radial_speed_decomposition_training",
        "arm": arm,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "epochs": len(history),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": best_path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_model_state_sha256": best_payload["model_state_sha256"],
        "source_r19_checkpoint_sha256": source_payload.get("checkpoint_sha256"),
        "class_weights": list(class_weights),
        "raw_radial_scalar_removed": True,
        "radial_sidecar_value_names": list(config.data["radial_value_names"]),
        "radial_sidecar_valid_names": list(config.data["radial_valid_names"]),
        "history": history,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "l3": "not_executed",
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = artifact_path(
        root,
        config.artifacts["training_summary_template"],
        arm=arm,
        seed=int(seed),
    )
    write_json_if_identical(summary_path, seal(summary))
    return {**summary, "summary_path": str(summary_path)}


__all__ = ["load_checkpoint", "train_arm"]
