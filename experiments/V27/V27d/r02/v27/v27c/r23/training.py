"""Fixed-endpoint paired L2 retraining for V27c-r23."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from v27.v27_baseline.base_binding import classifier_state_hash

from .model import R23IntegratedL2Model, build_r23_model
from .protocol import R23Context












def _load_initial_states(context: R23Context) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    l2_path = context.project_root / context.config.base.l2_checkpoint_path
    stable_path = context.project_root / context.config.base.v27a_main_checkpoint_path
    l2 = torch.load(l2_path, map_location="cpu", weights_only=False)
    stable = torch.load(stable_path, map_location="cpu", weights_only=False)
    if not isinstance(l2, Mapping) or not isinstance(stable, Mapping):
        raise ValueError("V27c-r23 initial checkpoints are not mappings")
    l2_state = l2.get("model_state")
    stable_state = stable.get("model_state")
    if not isinstance(l2_state, Mapping) or not isinstance(stable_state, Mapping):
        raise ValueError("V27c-r23 initial checkpoints lack model states")
    if (
        l2.get("model_state_sha256") != context.config.base.expected_l2_state_sha256
        or stable.get("model_state_sha256") != context.config.base.expected_v27a_main_state_sha256
        or classifier_state_hash(l2_state)
        != context.config.base.expected_l2_classifier_state_sha256
    ):
        raise ValueError("V27c-r23 initial checkpoint state bindings differ")
    return dict(l2_state), dict(stable_state)


def build_initialized_model(
    context: R23Context, *, arm: str, seed: int, device: torch.device
) -> R23IntegratedL2Model:
    l2_state, stable_state = _load_initial_states(context)
    return build_r23_model(
        context.config,
        arm=arm,
        v27_config=context.v27_config,
        v27_l2_state=l2_state,
        v27a_config=context.v27a_config,
        v27a_main_state=stable_state,
        seed=seed,
    ).to(device)














__all__ = ["build_initialized_model"]