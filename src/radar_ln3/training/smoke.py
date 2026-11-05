"""历史 checkpoint 到正式 L1/L2 模型的最小迁移 smoke。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radar_ln3.checkpoint import Checkpoint
from radar_ln3.data.paths import ensure_path_inside_root
from radar_ln3.model import build_l1_model, build_l2_model, load_model_config

from .l1 import _device as _l1_device
from .l2 import _device as _l2_device


def _load_historical_payload(project_root: str | Path, path: str | Path) -> dict[str, Any]:
    """读取项目内显式指定的历史 checkpoint，不把它当作正式配置来源。"""

    root = Path(project_root).resolve()
    target = Path(path)
    target = (root / target).resolve() if not target.is_absolute() else target.resolve()
    ensure_path_inside_root(target, root)
    if target.suffix.lower() != ".pt" or not target.is_file():
        raise FileNotFoundError(target)
    try:
        payload = Checkpoint.load_payload(target)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"历史 checkpoint 无法读取：{target}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("历史 checkpoint 根节点必须是对象")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("历史 checkpoint 缺少 model_state")
    return dict(payload)


def load_l1_smoke_model(
    project_root: str | Path,
    checkpoint_path: str | Path,
    *,
    arm: str = "observation_increment",
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """加载历史 L1 checkpoint，验证其能映射到正式 L1 模型。"""

    root = Path(project_root).resolve()
    config = load_model_config(root)
    payload = _load_historical_payload(root, checkpoint_path)
    state = dict(payload["model_state"])
    base_state = {
        str(name)[len("base.") :]: value
        for name, value in state.items()
        if str(name).startswith("base.")
    }
    adapter_state = {
        str(name)[len("observation_adapter.") :]: value
        for name, value in state.items()
        if str(name).startswith("observation_adapter.")
    }
    if not base_state or not adapter_state:
        raise ValueError("历史 L1 checkpoint 的 state 前缀不完整")
    model = build_l1_model(
        config,
        arm=arm,
        seed=config.training.seed,
        base_state=base_state,
        device=_l1_device(device),
    )
    incompatible = model.observation_adapter.load_state_dict(adapter_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("历史 L1 observation adapter state 映射不完整")
    return model, payload


def load_l2_smoke_model(
    project_root: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """加载历史 L2 checkpoint，验证其能映射到正式 L2 模型。"""

    root = Path(project_root).resolve()
    config = load_model_config(root)
    payload = _load_historical_payload(root, checkpoint_path)
    state = dict(payload["model_state"])
    base_state = Checkpoint.load(root, config.artifacts.l2_base_weight_path).state
    stable_state = Checkpoint.load(root, config.artifacts.l2_stable_weight_path).state
    model = build_l2_model(
        config,
        arm=str(payload.get("arm", "robust_control")),
        base_state=base_state,
        stable_state=stable_state,
        seed=config.training.seed,
    ).to(_l2_device(device))
    expected = set(model.state_dict())
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(f"历史 L2 checkpoint state 映射不完整：missing={missing[:3]}, extra={extra[:3]}")
    model.load_state_dict(state, strict=True)
    return model, payload


__all__ = ["load_l1_smoke_model", "load_l2_smoke_model"]
