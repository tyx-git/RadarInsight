"""Immutable bindings for the V27 classification base checkpoints.

The classification checkpoints are the experimental boundary for the V27
baseline.  Later knownness, quality and calibration stages are allowed to
change only ``trajectory_heads``; this module records the exact base files
and the classifier-only state hash so that such changes can be checked
without rewriting the historical classification checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file

from .protocol import V27_L1_TASK_NAME, V27_L2_TASK_NAME


FROZEN_BASE_BINDING_SCHEMA = "RADAR-LN3-V27-frozen-classification-base-binding-v1"
DEFAULT_FROZEN_BASE_BINDING_PATH = Path(
    "src-v3/v27/v27_baseline/runs/V27-frozen-classification-bases-seed42.json"
)
_HEAD_PREFIX = "trajectory_heads."
# The classifier checkpoint is immutable.  These are the *only* keys that a
# later, independently fitted open-set stage may replace.  Do not use a broad
# ``startswith`` exemption here: a future accidental module under
# ``trajectory_heads`` must not silently become mutable classifier state.
_MUTABLE_TRAJECTORY_HEAD_SUFFIXES = frozenset(
    {
        "trajectory_encoder.weight_ih_l0",
        "trajectory_encoder.weight_hh_l0",
        "trajectory_encoder.bias_ih_l0",
        "trajectory_encoder.bias_hh_l0",
        "knownness_projection.0.weight",
        "knownness_projection.0.bias",
        "knownness_projection.1.weight",
        "knownness_projection.1.bias",
        "knownness_head.weight",
        "knownness_head.bias",
        "quality_head.0.weight",
        "quality_head.0.bias",
        "quality_head.2.weight",
        "quality_head.2.bias",
    }
)


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    """Hash state tensors deterministically, independent of torch serialization."""

    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27 base state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend(
            (
                str(name),
                str(tensor.dtype),
                repr(tuple(tensor.shape)),
                tensor.numpy().tobytes().hex(),
            )
        )
    return canonical_sha256(parts)


def mutable_trajectory_head_keys(state: Mapping[str, Any]) -> frozenset[str]:
    """Return the exact permitted mutable state keys, failing on drift."""

    declared = {
        name for name in state if isinstance(name, str) and name.startswith(_HEAD_PREFIX)
    }
    expected = {
        f"{_HEAD_PREFIX}{suffix}" for suffix in _MUTABLE_TRAJECTORY_HEAD_SUFFIXES
    }
    if declared != expected:
        missing = sorted(expected - declared)
        unexpected = sorted(declared - expected)
        raise ValueError(
            "V27 trajectory-head state boundary differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return frozenset(expected)


def classifier_state_hash(state: Mapping[str, Any]) -> str:
    """Hash every parameter/buffer outside the explicitly mutable heads."""

    mutable = mutable_trajectory_head_keys(state)
    classifier_state = {name: value for name, value in state.items() if name not in mutable}
    if not classifier_state:
        raise ValueError("V27 checkpoint has no classifier/base state")
    return tensor_state_hash(classifier_state)




def _load_checkpoint(root: Path, path: str | Path, *, task: str) -> tuple[Path, dict[str, Any]]:
    target = resolve_project_path(root, path, must_exist=True)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"V27 frozen base checkpoint is not an object: {target}")
    payload = dict(payload)
    if payload.get("task") != task:
        raise ValueError(f"V27 frozen base task differs: expected={task}, actual={payload.get('task')}")
    if payload.get("checkpoint_kind") != "epoch":
        raise ValueError("V27 frozen classification bases must be epoch checkpoints")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"V27 frozen base checkpoint has no model_state: {target}")
    declared_state_hash = payload.get("model_state_sha256")
    if declared_state_hash != tensor_state_hash(state):
        raise ValueError(f"V27 frozen base model state hash differs: {target}")
    if payload.get("unknown_is_model_class") is not False or int(payload.get("unknown_training_exposure", -1)) != 0:
        raise ValueError("V27 frozen classification base violates the Unknown contract")
    return target, payload






def verify_frozen_base_binding(
    binding: Mapping[str, Any],
    *,
    project_root: str | Path,
    require_files: bool = True,
) -> dict[str, Any]:
    """Verify binding integrity and the current bytes of both base files."""

    root = Path(project_root).resolve()
    payload = dict(binding)
    declared = payload.pop("binding_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("V27 frozen base binding hash cannot be recomputed")
    if payload.get("schema_version") != FROZEN_BASE_BINDING_SCHEMA or payload.get("status") != "frozen":
        raise ValueError("V27 frozen base binding schema/status differs")
    if payload.get("classification_only") is not True or payload.get("classifier_state_must_remain_identical") is not True:
        raise ValueError("V27 frozen base binding safety flags differ")
    shared = payload.get("shared_binding")
    if not isinstance(shared, Mapping):
        raise ValueError("V27 frozen base binding lacks shared provenance")
    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        record = payload.get(task_to_binding_key(task))
        if not isinstance(record, Mapping):
            raise ValueError(f"V27 frozen base binding lacks {task}")
        path = resolve_project_path(root, str(record.get("checkpoint_path", "")), must_exist=require_files)
        if require_files:
            if sha256_file(path) != str(record.get("checkpoint_sha256", "")):
                raise ValueError(f"V27 frozen {task} checkpoint file hash differs")
            _, checkpoint = _load_checkpoint(root, path, task=task)
            if int(checkpoint.get("epoch", -1)) != int(record.get("epoch", -2)):
                raise ValueError(f"V27 frozen {task} checkpoint epoch differs")
            if checkpoint.get("model_state_sha256") != record.get("model_state_sha256"):
                raise ValueError(f"V27 frozen {task} model state hash differs")
            if checkpoint.get("config_sha256") != record.get("config_sha256"):
                raise ValueError(f"V27 frozen {task} config hash differs")
            if checkpoint.get("manifest_sha256") != record.get("manifest_sha256"):
                raise ValueError(f"V27 frozen {task} manifest hash differs")
            state = checkpoint.get("model_state")
            if not isinstance(state, Mapping):
                raise ValueError(f"V27 frozen {task} checkpoint has no model state")
            if classifier_state_hash(state) != record.get("classifier_state_sha256"):
                raise ValueError(f"V27 frozen {task} classifier state hash differs")
        for field, value in shared.items():
            if record.get(field) != value:
                raise ValueError(f"V27 frozen {task} differs from shared binding at {field}")
        if int(record.get("epoch", -1)) != (14 if task == V27_L1_TASK_NAME else 11):
            raise ValueError(f"V27 frozen {task} epoch differs")
        if int(record.get("seed", -1)) != int(payload.get("seed", -2)):
            raise ValueError(f"V27 frozen {task} seed differs")
    result = {**payload, "binding_sha256": declared}
    return result


def task_to_binding_key(task: str) -> str:
    if task == V27_L1_TASK_NAME:
        return "l1"
    if task == V27_L2_TASK_NAME:
        return "l2"
    raise ValueError(f"unsupported V27 base task: {task}")




def load_frozen_base_binding(project_root: str | Path, path: str | Path = DEFAULT_FROZEN_BASE_BINDING_PATH) -> dict[str, Any]:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, path, must_exist=True)
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("V27 frozen base binding must be a JSON object")
    return verify_frozen_base_binding(raw, project_root=root, require_files=True)




__all__ = ["DEFAULT_FROZEN_BASE_BINDING_PATH", "FROZEN_BASE_BINDING_SCHEMA", "classifier_state_hash", "load_frozen_base_binding", "mutable_trajectory_head_keys", "task_to_binding_key", "tensor_state_hash", "verify_frozen_base_binding"]