"""Fixed-endpoint paired L2 retraining for V27c-r22."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radar_ln3.runtime import canonical_sha256
from v27.v27_baseline.base_binding import classifier_state_hash

from .model import R22IntegratedL2Model, build_r22_model
from .protocol import R22_ARMS, R22_CHECKPOINT_SCHEMA_VERSION, R22_L2_CLASSES, R22Context, r22_artifact_path






def _path(context: R22Context, field: str, *, arm: str, seed: int, epoch: int | None = None) -> Path:
    return r22_artifact_path(
        context.project_root,
        getattr(context.config.artifacts, field),
        arm=arm,
        seed=seed,
        epoch=epoch,
    )


def _load_initial_states(context: R22Context) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    l2_path = context.project_root / context.config.base.l2_checkpoint_path
    stable_path = context.project_root / context.config.base.v27a_main_checkpoint_path
    l2 = torch.load(l2_path, map_location="cpu", weights_only=False)
    stable = torch.load(stable_path, map_location="cpu", weights_only=False)
    if not isinstance(l2, Mapping) or not isinstance(stable, Mapping):
        raise ValueError("V27c-r22 initial checkpoints are not mappings")
    l2_state = l2.get("model_state")
    stable_state = stable.get("model_state")
    if not isinstance(l2_state, Mapping) or not isinstance(stable_state, Mapping):
        raise ValueError("V27c-r22 initial checkpoints lack model states")
    if (
        l2.get("model_state_sha256") != context.config.base.expected_l2_state_sha256
        or stable.get("model_state_sha256") != context.config.base.expected_v27a_main_state_sha256
        or classifier_state_hash(l2_state)
        != context.config.base.expected_l2_classifier_state_sha256
    ):
        raise ValueError("V27c-r22 initial checkpoint state bindings differ")
    return dict(l2_state), dict(stable_state)






def _state_hash(state: Mapping[str, Any]) -> str:
    from v27.v27a.checkpoint import tensor_state_hash

    return tensor_state_hash(state)


def load_r22_checkpoint(
    context: R22Context,
    *,
    arm: str,
    seed: int,
    device: torch.device | None = None,
) -> tuple[R22IntegratedL2Model, dict[str, Any]]:
    if arm not in R22_ARMS or seed not in context.config.training.seeds:
        raise ValueError("V27c-r22 checkpoint arm or seed is invalid")
    path = _path(
        context,
        "checkpoint_template",
        arm=arm,
        seed=seed,
        epoch=context.config.training.primary_epoch,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r22 checkpoint is not an object")
    checked = dict(payload)
    expected_scope = {
        "unknown_training_exposure": 0,
        "clutter_training_exposure": 0,
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "calibration": "not_in_scope",
        "l3": "not_in_scope",
        "test_payload_access_count": 0,
    }
    if (
        checked.get("checkpoint_schema") != R22_CHECKPOINT_SCHEMA_VERSION
        or checked.get("experiment_id") != context.config.experiment_id
        or checked.get("arm") != arm
        or int(checked.get("seed", -1)) != int(seed)
        or int(checked.get("epoch", -1)) != context.config.training.primary_epoch
        or checked.get("manifest_sha256") != context.manifest_sha256
        or checked.get("input_binding_sha256") != canonical_sha256(context.input_binding)
        or checked.get("l1_checkpoint_deserialized") is not False
        or checked.get("l1_forward_calls") != 0
        or checked.get("l1_optimizer_parameter_count") != 0
        or checked.get("l2_initial_checkpoint_sha256")
        != context.config.base.expected_l2_checkpoint_sha256
        or checked.get("l2_initial_classifier_state_sha256")
        != context.config.base.expected_l2_classifier_state_sha256
        or checked.get("v27a_main_initial_checkpoint_sha256")
        != context.config.base.expected_v27a_main_checkpoint_sha256
        or checked.get("class_names") != list(R22_L2_CLASSES)
        or checked.get("class_weights") != list(context.config.base.expected_l2_class_weights)
        or checked.get("optimizer") != context.config.training.optimizer
        or checked.get("loss_name") != context.config.training.loss_name
        or checked.get("loss_reduction") != context.config.training.loss_reduction
        or checked.get("scope") != expected_scope
    ):
        raise ValueError("V27c-r22 checkpoint identity differs")
    l2_state, stable_state = _load_initial_states(context)
    model = build_r22_model(
        context.config,
        arm=arm,
        v27_config=context.v27_config,
        v27_l2_state=l2_state,
        v27a_config=context.v27a_config,
        v27a_main_state=stable_state,
        seed=seed,
    )
    state = checked.get("model_state")
    if not isinstance(state, Mapping) or checked.get("model_state_sha256") != _state_hash(state):
        raise ValueError("V27c-r22 checkpoint model-state hash differs")
    if (
        checked.get("model_contract") != model.model_contract()
        or checked.get("model_contract_sha256") != canonical_sha256(model.model_contract())
    ):
        raise ValueError("V27c-r22 checkpoint model contract differs")
    model.load_state_dict(dict(state), strict=True)
    return model.to(torch.device("cpu") if device is None else device), checked








__all__ = ["load_r22_checkpoint"]