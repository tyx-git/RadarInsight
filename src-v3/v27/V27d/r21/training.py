"""r21 paired training with only the radial adapter trainable."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce
from v27.V27d.r20.training import load_checkpoint as load_r20_checkpoint
from v27.v27a.checkpoint import tensor_state_hash
from v27.v27_baseline.metrics import summarize_records

from .data import R21Dataset, build_r21_loader, r21_model_input_kwargs
from .model import R21L2Model, assert_base_state_unchanged, freeze_base_train_adapter
from .protocol import (
    R21_ARMS,
    R21_CHECKPOINT_SCHEMA,
    R21_EXPERIMENT_ID,
    R21_L2_CLASSES,
    R21_TRAINING_SUMMARY_SCHEMA,
    R21Config,
    artifact_path,
    canonical_sha256,
    load_manifest,
    project_path,
    seal,
    sha256_file,
    write_json_if_identical,
    write_torch_if_identical,
)


def _device(config: R21Config, requested: str | None) -> torch.device:
    value = str(requested or config.training.get("device", "cpu")).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("r21 requested CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r21 device: {value}")


def _seed(seed: int) -> None:
    value = int(seed)
    if value < 0:
        raise ValueError("r21 seed must be non-negative")
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _class_weights(dataset: R21Dataset) -> tuple[float, ...]:
    counts = {name: 0 for name in R21_L2_CLASSES}
    for item in dataset:
        label = item.sequence.label_name
        if label not in counts:
            raise ValueError(f"r21 training received non-DBB label: {label}")
        counts[label] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"r21 training support is incomplete: {counts}")
    total = float(sum(counts.values()))
    return tuple(total / (len(counts) * counts[name]) for name in R21_L2_CLASSES)


def _base_state_hash(model: R21L2Model) -> str:
    return tensor_state_hash(model.base_state())


def _build_from_r20(root: Path, config: R21Config, *, device: torch.device, arm: str) -> tuple[R21L2Model, dict[str, Any]]:
    source_path = project_path(root, config.source["r20_source_checkpoint_path"], must_exist=True)
    model, payload = load_r20_checkpoint(
        root,
        config.r20,
        arm="zero_control",
        seed=int(config.source["r20_source_checkpoint_seed"]),
        checkpoint_path=source_path,
    )
    if sha256_file(source_path) != config.source["r20_source_checkpoint_sha256"]:
        raise ValueError("r21 direct r20 source checkpoint hash differs")
    if payload.get("model_state_sha256") != config.source["r20_source_checkpoint_model_state_sha256"]:
        raise ValueError("r21 direct r20 source state hash differs")
    if int(payload.get("epoch", -1)) != int(config.source["r20_source_checkpoint_epoch"]):
        raise ValueError("r21 direct r20 source epoch differs")
    base = copy.deepcopy(model).cpu()
    wrapped = R21L2Model(config, base_model=base, arm=arm).to(device)
    freeze_base_train_adapter(wrapped)
    return wrapped, dict(payload)


def _evaluate(
    model: R21L2Model,
    dataset: R21Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r21_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r21_model_input_kwargs(batch, device))
            logits = output["window_logits"].detach().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("r21 evaluation logits are non-finite")
            for index, parent_id in enumerate(batch["parent_sample_ids"]):
                final = logits[index]
                rows.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "source_suffix": str(parent_id).split("|")[2],
                        "label_name": str(batch["label_names"][index]),
                        "true_id": int(batch["l2_labels"][index]),
                        "pred_id": int(final.argmax().item()),
                        "probabilities": [float(value) for value in torch.softmax(final, dim=-1).tolist()],
                        "event_count": int(batch["lengths"][index]),
                    }
                )
    if [row["parent_sample_id"] for row in rows] != dataset.parent_ids:
        raise ValueError("r21 evaluation changed parent order")
    return summarize_records(rows, class_names=R21_L2_CLASSES, scope="parent_final"), rows


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
    config: R21Config,
    manifest_path: Path,
    manifest_sha256: str,
    model: R21L2Model,
    optimizer: torch.optim.Optimizer,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    class_weights: tuple[float, ...],
    validation_metrics: Mapping[str, Any],
    selection_key: tuple[float, float, float],
    frozen_base_state_sha256: str,
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    contract = model.model_contract()
    trainable_state = {
        name: value for name, value in state.items() if name.startswith("base_model.radial_adapter.")
    }
    return {
        "checkpoint_schema": R21_CHECKPOINT_SCHEMA,
        "experiment_id": R21_EXPERIMENT_ID,
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
        "source_r20_checkpoint_path": config.source["r20_source_checkpoint_path"],
        "source_r20_checkpoint_sha256": config.source["r20_source_checkpoint_sha256"],
        "source_r20_checkpoint_model_state_sha256": config.source["r20_source_checkpoint_model_state_sha256"],
        "source_r20_checkpoint_arm": config.source["r20_source_checkpoint_arm"],
        "source_r20_checkpoint_seed": int(config.source["r20_source_checkpoint_seed"]),
        "source_r20_checkpoint_epoch": int(config.source["r20_source_checkpoint_epoch"]),
        "frozen_base_state_sha256": frozen_base_state_sha256,
        "trainable_scope": "radial_adapter_only",
        "trainable_state_sha256": tensor_state_hash(trainable_state),
        "raw_radial_scalar_removed": True,
        "radial_sidecar_value_names": list(config.data["radial_value_names"]),
        "radial_sidecar_valid_names": list(config.data["radial_valid_names"]),
        "class_names": list(R21_L2_CLASSES),
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
    config: R21Config,
    *,
    arm: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> tuple[R21L2Model, dict[str, Any]]:
    if arm not in R21_ARMS:
        raise ValueError("r21 checkpoint arm is invalid")
    root = Path(project_root).resolve()
    _, manifest_path, manifest_sha256 = load_manifest(root, config, allow_code_drift=True)
    path = (
        artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
        if checkpoint_path is None
        else project_path(root, checkpoint_path, must_exist=True)
    )
    try:
        path.relative_to((root / "src-v3" / "v27" / "V27d" / "r21").resolve())
    except ValueError as exc:
        raise ValueError("r21 checkpoint escaped the experiment root") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("r21 checkpoint is not an object")
    required = (
        payload.get("checkpoint_schema") == R21_CHECKPOINT_SCHEMA,
        payload.get("experiment_id") == R21_EXPERIMENT_ID,
        payload.get("arm") == arm,
        payload.get("task") == "l2",
        int(payload.get("seed", -1)) == int(seed),
        int(payload.get("epoch", 0)) > 0,
        int(payload.get("epoch", 0)) <= int(config.training["epochs"]),
        payload.get("config_sha256") == config.sha256,
        payload.get("config_identity_sha256") == config.identity_sha256,
        payload.get("manifest_path") == manifest_path.relative_to(root).as_posix(),
        payload.get("manifest_sha256") == manifest_sha256,
        payload.get("source_r20_checkpoint_sha256") == config.source["r20_source_checkpoint_sha256"],
        payload.get("source_r20_checkpoint_model_state_sha256") == config.source["r20_source_checkpoint_model_state_sha256"],
        payload.get("trainable_scope") == "radial_adapter_only",
        payload.get("frozen_base_state_sha256") is not None,
        payload.get("class_names") == list(R21_L2_CLASSES),
        payload.get("loss_name") == config.training["loss_name"],
        payload.get("loss_reduction") == config.training["loss_reduction"],
        int(payload.get("unknown_training_exposure", -1)) == 0,
        int(payload.get("test_payload_access_count", -1)) == 0,
        int(payload.get("rd_payload_access_count", -1)) == 0,
    )
    if not all(required):
        raise ValueError("r21 checkpoint provenance differs")
    model, _ = _build_from_r20(root, config, device=torch.device("cpu"), arm=arm)
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("r21 model state hash differs")
    trainable_state = {
        name: value
        for name, value in state.items()
        if str(name).startswith("base_model.radial_adapter.")
    }
    if payload.get("trainable_state_sha256") != tensor_state_hash(trainable_state):
        raise ValueError("r21 trainable adapter state hash differs")
    model.load_state_dict(dict(state), strict=True)
    freeze_base_train_adapter(model)
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("r21 checkpoint model contract differs")
    if _base_state_hash(model) != payload.get("frozen_base_state_sha256"):
        raise ValueError("r21 frozen base state hash differs after loading")
    return model.eval(), dict(payload)


def train_arm(
    project_root: str | Path,
    config: R21Config,
    *,
    arm: str,
    seed: int,
    device: str | None = None,
) -> dict[str, Any]:
    if arm not in R21_ARMS:
        raise ValueError("r21 arm is invalid")
    if int(seed) not in [int(value) for value in config.training["seeds"]]:
        raise ValueError("r21 seed is not registered")
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    _seed(int(seed))
    runtime_device = _device(config, device)
    model, source_payload = _build_from_r20(root, config, device=runtime_device, arm=arm)
    frozen_base_state = model.base_state()
    frozen_base_hash = _base_state_hash(model)
    train_set = R21Dataset(root, config, split_name="train", arm=arm)
    validation_set = R21Dataset(root, config, split_name="validation", arm=arm)
    expected_train = [parent_id for parent_id, split in manifest["parent_assignments"].items() if split == "train"]
    expected_validation = [parent_id for parent_id, split in manifest["parent_assignments"].items() if split == "validation"]
    if train_set.parent_ids != expected_train or validation_set.parent_ids != expected_validation:
        raise ValueError("r21 dataset order differs from the sealed parent split")
    class_weights = _class_weights(train_set)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=runtime_device)
    loader = build_r21_loader(
        train_set,
        batch_size=int(config.training["batch_size"]),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    freeze_base_train_adapter(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
    for epoch in range(1, int(config.training["epochs"]) + 1):
        model.train()
        model.base_model.base_model.eval()
        model.base_model.radial_adapter.train()
        losses: list[float] = []
        gradient_max = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**r21_model_input_kwargs(batch, runtime_device))
            loss = parent_final_fixed_parent_count_ce(
                output["sequence_logits"],
                batch["l2_labels"].to(runtime_device),
                batch["tr_padding_mask"].to(runtime_device),
                class_weights=weights,
            )["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("r21 loss is non-finite")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    float(config.training["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                ).item()
            )
            if not np.isfinite(norm):
                raise FloatingPointError("r21 gradient norm is non-finite")
            optimizer.step()
            assert_base_state_unchanged(model, frozen_base_state)
            global_step += 1
            losses.append(float(loss.detach().item()))
            gradient_max = max(gradient_max, norm)
        validation_metrics, _ = _evaluate(
            model,
            validation_set,
            device=runtime_device,
            batch_size=int(config.training["batch_size"]),
        )
        assert_base_state_unchanged(model, frozen_base_state)
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
            frozen_base_state_sha256=frozen_base_hash,
        )
        epoch_path = artifact_path(root, config.artifacts["epoch_checkpoint_template"], arm=arm, seed=int(seed), epoch=epoch)
        write_torch_if_identical(epoch_path, payload)
        history.append({
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "gradient_norm_max": gradient_max,
            "validation_parent_metrics": validation_metrics,
            "selection_key": list(key),
            "frozen_base_state_sha256": frozen_base_hash,
        })
        print(f"[V27d-r21] arm={arm} epoch={epoch}/{config.training['epochs']} loss={np.mean(losses):.6f} val_macro_f1={validation_metrics['macro_f1']:.6f}", flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_payload = key, epoch, payload
    if best_payload is None or best_epoch is None:
        raise RuntimeError("r21 produced no selectable checkpoint")
    best_path = artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
    write_torch_if_identical(best_path, best_payload)
    summary = {
        "schema_version": R21_TRAINING_SUMMARY_SCHEMA,
        "experiment_id": R21_EXPERIMENT_ID,
        "status": "completed_radial_adapter_only_training",
        "arm": arm,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "epochs": len(history),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": best_path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_model_state_sha256": best_payload["model_state_sha256"],
        "source_r20_checkpoint_sha256": config.source["r20_source_checkpoint_sha256"],
        "frozen_base_state_sha256": frozen_base_hash,
        "trainable_scope": "radial_adapter_only",
        "class_weights": list(class_weights),
        "history": history,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "l3": "not_executed",
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = artifact_path(root, config.artifacts["training_summary_template"], arm=arm, seed=int(seed))
    write_json_if_identical(summary_path, seal(summary))
    return {**summary, "summary_path": str(summary_path)}


__all__ = ["load_checkpoint", "train_arm"]
