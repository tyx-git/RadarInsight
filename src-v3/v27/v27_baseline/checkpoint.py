"""Task-bound, fail-closed V27 checkpoint serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import atomic_torch_save, canonical_sha256, resolve_project_path, sha256_file

from .model import V27LiquidV2TaskModel
from .protocol import V27_CHECKPOINT_SCHEMA, V27Config, load_and_verify_v27_manifest, task_class_weights, task_class_names, verify_v27_data_contract


def _tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27 state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((name, str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def _metadata(*, root: Path, config: V27Config, manifest: Mapping[str, Any], data_binding: Mapping[str, Any], model: V27LiquidV2TaskModel, epoch: int, global_step: int, metrics: Mapping[str, Any], kind: str, seed: int) -> dict[str, Any]:
    task = model.task
    return {
        "checkpoint_schema": V27_CHECKPOINT_SCHEMA, "experiment_id": "V27", "task": task, "checkpoint_kind": str(kind), "seed": int(seed),
        "config_path": config.config_path.relative_to(root).as_posix(), "config_sha256": config.config_sha256, "config_identity_sha256": config.config_identity_sha256,
        "manifest_path": resolve_project_path(root, config.artifacts.manifest_path).relative_to(root).as_posix(), "manifest_sha256": sha256_file(resolve_project_path(root, config.artifacts.manifest_path)),
        "data_registry_sha256": data_binding["data_registry_sha256"], "parent_split_sha256": data_binding["parent_split_sha256"], "feature_config_sha256": data_binding["feature_config_sha256"],
        "state_equation_sha256": manifest["state_equation_sha256"], "time_policy_sha256": manifest["time_policy_hash"],
        "loss_name": manifest["tasks"][task]["loss"]["name"], "loss_reduction": manifest["tasks"][task]["loss"]["reduction"], "class_weights": list(task_class_weights(config, task)), "class_names": list(task_class_names(task)),
        "knownness_contract": dict(manifest["tasks"][task]["knownness"]), "quality_contract": dict(manifest["tasks"][task]["quality"]), "l3_schema_version": manifest["l3"]["schema_version"],
        "unknown_is_model_class": False, "unknown_training_exposure": 0, "model_contract": model.model_contract(), "model_contract_sha256": canonical_sha256(model.model_contract()), "epoch": int(epoch), "global_step": int(global_step), "metrics": dict(metrics),
    }


def save_v27_checkpoint(path: str | Path, *, project_root: str | Path, config: V27Config, model: V27LiquidV2TaskModel, optimizer: torch.optim.Optimizer | None, manifest: Mapping[str, Any], data_binding: Mapping[str, Any], epoch: int, global_step: int, metrics: Mapping[str, Any], kind: str, seed: int | None = None, training_state: Mapping[str, Any] | None = None) -> Path:
    root = Path(project_root).resolve()
    actual_seed = int(config.training.seeds[0] if seed is None else seed)
    if actual_seed not in config.training.seeds:
        raise ValueError("V27 checkpoint seed is not registered")
    target = resolve_project_path(root, path)
    payload: dict[str, Any] = _metadata(root=root, config=config, manifest=manifest, data_binding=data_binding, model=model, epoch=epoch, global_step=global_step, metrics=metrics, kind=kind, seed=actual_seed)
    payload["model_state"] = model.state_dict()
    payload["model_state_sha256"] = _tensor_state_hash(payload["model_state"])
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    atomic_torch_save(target, payload)
    return target


def load_v27_checkpoint(path: str | Path, *, project_root: str | Path, config: V27Config, model: V27LiquidV2TaskModel, optimizer: torch.optim.Optimizer | None = None, manifest: Mapping[str, Any] | None = None, data_binding: Mapping[str, Any] | None = None, map_location: str | torch.device = "cpu", seed: int | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    payload = torch.load(resolve_project_path(root, path, must_exist=True), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != V27_CHECKPOINT_SCHEMA:
        raise ValueError("V27 rejects non-V27/historical checkpoints")
    if payload.get("task") != model.task:
        raise ValueError("V27 rejects cross-task L1/L2 checkpoint loading")
    checkpoint_seed = payload.get("seed")
    if isinstance(checkpoint_seed, bool) or not isinstance(checkpoint_seed, int) or checkpoint_seed not in config.training.seeds or (seed is not None and int(seed) != checkpoint_seed):
        raise ValueError("V27 checkpoint seed differs")
    if payload.get("checkpoint_kind") not in {"epoch", "best", "last", "smoke"} or int(payload.get("epoch", 0)) <= 0 or int(payload.get("global_step", -1)) < 0:
        raise ValueError("V27 checkpoint epoch/kind differs")
    verified_data = verify_v27_data_contract(root, config)
    if data_binding is not None and dict(data_binding) != verified_data:
        raise ValueError("V27 supplied data registry differs")
    current_manifest, manifest_path, manifest_hash = load_and_verify_v27_manifest(root, config, verified_data)
    if manifest is not None and dict(manifest) != current_manifest:
        raise ValueError("V27 supplied manifest differs")
    if payload.get("manifest_path") != manifest_path.relative_to(root).as_posix() or payload.get("manifest_sha256") != manifest_hash:
        raise ValueError("V27 checkpoint manifest binding differs")
    expected = _metadata(root=root, config=config, manifest=current_manifest, data_binding=verified_data, model=model, epoch=int(payload["epoch"]), global_step=int(payload["global_step"]), metrics=dict(payload.get("metrics") or {}), kind=str(payload["checkpoint_kind"]), seed=checkpoint_seed)
    for key, value in expected.items():
        if key not in {"metrics", "model_contract"} and payload.get(key) != value:
            raise ValueError(f"V27 checkpoint metadata differs: {key}")
    if payload.get("model_contract") != expected["model_contract"]:
        raise ValueError("V27 checkpoint model contract differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != _tensor_state_hash(state):
        raise ValueError("V27 checkpoint state hash differs")
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        saved = payload.get("optimizer_state")
        if not isinstance(saved, Mapping):
            raise ValueError("V27 checkpoint lacks requested optimizer state")
        optimizer.load_state_dict(saved)
    return dict(payload)


__all__ = ["load_v27_checkpoint", "save_v27_checkpoint"]
