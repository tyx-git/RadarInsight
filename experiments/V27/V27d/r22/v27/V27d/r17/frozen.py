"""加载 r17 指定的冻结 L1/L2 checkpoint。

这里不依赖 r15 的过期实验 manifest；checkpoint 本身、模型 contract 和
state hash 必须同时通过校验。r23 的 stable8/RD 分支不进入 Excel TR-only
运行时，只取 checkpoint 中的 trainable L2 backbone。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from v27.V27d.r15.model import build_r15_model, model_state_hash as r15_state_hash
from v27.V27d.r15.protocol import load_config as load_r15_config
from v27.v27a.checkpoint import tensor_state_hash
from v27.v27c.r23.protocol import load_r23_context
from v27.v27c.r23.training import build_initialized_model

from .protocol import R17Config, project_path, sha256_file


def _state_mapping(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("r17 checkpoint model_state is missing")
    return state


def _require_digest(payload: Mapping[str, Any], key: str, expected: str, context: str) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise ValueError(f"r17 {context} {key} differs")


@dataclass(frozen=True)
class FrozenR17Models:
    project_root: Path
    config: R17Config
    l1_model: torch.nn.Module
    l2_integrated_model: torch.nn.Module
    l2_model: torch.nn.Module
    l1_checkpoint: dict[str, Any]
    l2_checkpoint: dict[str, Any]
    l3_registry: dict[str, Any]


def _load_r15(root: Path, config: R17Config) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = project_path(root, config.source["r15_checkpoint_path"], must_exist=True)
    if sha256_file(path) != config.source["r15_checkpoint_sha256"]:
        raise ValueError("r17 r15 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("r17 r15 checkpoint is not a mapping")
    if (
        payload.get("experiment_id") != "V27d-r15"
        or payload.get("arm") != "v10_observation_increment"
        or int(payload.get("seed", -1)) != 42
        or int(payload.get("epoch", -1)) != 11
    ):
        raise ValueError("r17 r15 checkpoint identity differs")
    _require_digest(payload, "model_contract_sha256", config.source["r15_model_contract_sha256"], "r15")
    _require_digest(payload, "model_state_sha256", config.source["r15_model_state_sha256"], "r15")
    r15_config = load_r15_config(root, config.source["r15_config_path"])
    context = SimpleNamespace(project_root=root, config=r15_config)
    model = build_r15_model(context, arm="v10_observation_increment", seed=42, device=torch.device("cpu"))
    if payload.get("model_contract") != model.model_contract():
        raise ValueError("r17 r15 model contract differs")
    model.load_state_dict(dict(_state_mapping(payload)), strict=True)
    if r15_state_hash(model) != config.source["r15_model_state_sha256"]:
        raise ValueError("r17 r15 loaded state hash differs")
    model.eval()
    return model, dict(payload)


def _load_r23(root: Path, config: R17Config) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    path = project_path(root, config.source["r23_checkpoint_path"], must_exist=True)
    if sha256_file(path) != config.source["r23_checkpoint_sha256"]:
        raise ValueError("r17 r23 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("r17 r23 checkpoint is not a mapping")
    if (
        payload.get("experiment_id") != "V27c-r23"
        or payload.get("arm") != "robust_control"
        or int(payload.get("seed", -1)) != 42
        or int(payload.get("epoch", -1)) != 1
    ):
        raise ValueError("r17 r23 checkpoint identity differs")
    _require_digest(payload, "model_contract_sha256", config.source["r23_model_contract_sha256"], "r23")
    _require_digest(payload, "model_state_sha256", config.source["r23_model_state_sha256"], "r23")
    r23_context = load_r23_context(root, config.source["r23_config_path"])
    integrated = build_initialized_model(r23_context, arm="robust_control", seed=42, device=torch.device("cpu"))
    if payload.get("model_contract") != integrated.model_contract():
        raise ValueError("r17 r23 model contract differs")
    state = _state_mapping(payload)
    if tensor_state_hash(state) != config.source["r23_model_state_sha256"]:
        raise ValueError("r17 r23 checkpoint state hash cannot be recomputed")
    integrated.load_state_dict(dict(state), strict=True)
    loaded_state = {name: value.detach().cpu() for name, value in integrated.state_dict().items()}
    if tensor_state_hash(loaded_state) != config.source["r23_model_state_sha256"]:
        raise ValueError("r17 r23 loaded state hash differs")
    integrated.eval()
    base = integrated.trainable_base_model
    base.eval()
    return integrated, base, dict(payload)


def load_frozen_models(project_root: str | Path, config: R17Config) -> FrozenR17Models:
    root = Path(project_root).resolve()
    # Recompute the source configuration/registry hashes before loading any
    # checkpoint.  A declared digest in r17's config is provenance, not proof.
    for path_key, hash_key in (
        ("r15_config_path", "r15_config_sha256"),
        ("r23_config_path", "r23_config_sha256"),
        ("l3_example_registry_path", "l3_example_registry_sha256"),
    ):
        path = project_path(root, config.source[path_key], must_exist=True)
        if sha256_file(path) != config.source[hash_key]:
            raise ValueError(f"r17 source file hash differs: {path_key}")
    l1_model, l1_checkpoint = _load_r15(root, config)
    l2_integrated, l2_model, l2_checkpoint = _load_r23(root, config)
    l3_path = project_path(root, config.source["l3_example_registry_path"], must_exist=True)
    if sha256_file(l3_path) != config.source["l3_example_registry_sha256"]:
        raise ValueError("r17 L3 example registry file hash differs")
    l3_registry = __import__("json").loads(l3_path.read_text(encoding="utf-8"))
    if not isinstance(l3_registry, dict) or l3_registry.get("status") != "unavailable":
        raise ValueError("r17 L3 example registry must remain unavailable")
    return FrozenR17Models(
        project_root=root,
        config=config,
        l1_model=l1_model,
        l2_integrated_model=l2_integrated,
        l2_model=l2_model,
        l1_checkpoint=l1_checkpoint,
        l2_checkpoint=l2_checkpoint,
        l3_registry=l3_registry,
    )






__all__ = ["FrozenR17Models", "load_frozen_models"]