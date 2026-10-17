"""Independent knownness/quality fitting and dynamic V27 calibration.

The classification checkpoints are frozen before this module is entered.  The
module consequently has two deliberately separate responsibilities:

* materialize the causal trajectory features emitted by a frozen L1/L2 base;
* fit only ``trajectory_heads`` and then fit a dynamic, fail-closed L3
  registry on the independent calibration role.

No function in this module opens the test split.  Unknown labels are used only
after the frozen classification boundary and never become classifier targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file

from .base_binding import tensor_state_hash
from .open_set_data import load_open_set_pool_artifacts
from .protocol import V27_L1_TASK_NAME, V27_L2_TASK_NAME, v27_artifact_path
from .runtime import read_v27_sealed_json


REPRESENTATION_CACHE_SCHEMA = "RADAR-LN3-V27-frozen-trajectory-representation-v1"
REPRESENTATION_MANIFEST_SCHEMA = "RADAR-LN3-V27-representation-cache-manifest-v1"
OPEN_SET_ROLES = ("representation_fit", "representation_development", "calibration")
CANONICAL_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
LABEL_TO_CODE = {name: index for index, name in enumerate(CANONICAL_LABELS)}






def _require_task(task: str) -> str:
    value = str(task)
    if value not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
        raise ValueError(f"unsupported V27 open-set task: {task}")
    return value


def _require_role(role: str) -> str:
    value = str(role)
    if value not in OPEN_SET_ROLES:
        raise ValueError(f"unsupported V27 open-set role: {role}")
    return value


def _cache_path(context: Any, *, task: str, role: str) -> Path:
    _require_task(task)
    _require_role(role)
    return v27_artifact_path(
        context.project_root,
        context.config.artifacts.representation_cache_template,
        int(context.config.training.seeds[0]),
        task=task,
        role=role,
    )
















def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash(values)


def _representation_cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27 representation cache lacks tensors")
    tensor_hash = _tensor_digest(
        {str(name): value for name, value in tensors.items()}
    )
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256"}
    }
    return canonical_sha256({"metadata": metadata, "tensor_sha256": tensor_hash})












def _validate_representation_payload(
    payload: Mapping[str, Any],
    *,
    context: Any,
    task: str,
    role: str,
    pool_registry: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(payload)
    declared = raw.pop("cache_sha256", None)
    if not isinstance(declared, str) or declared != _representation_cache_digest(payload):
        raise ValueError("V27 representation cache hash cannot be recomputed")
    if raw.get("schema_version") != REPRESENTATION_CACHE_SCHEMA:
        raise ValueError("V27 representation cache schema differs")
    if raw.get("task") != _require_task(task) or raw.get("role") != _require_role(role):
        raise ValueError("V27 representation cache task/role differs")
    if raw.get("system_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27 representation cache manifest binding differs")
    if raw.get("pool_registry_sha256") != pool_registry["registry_sha256"]:
        raise ValueError("V27 representation cache pool binding differs")
    if raw.get("open_set_feature_cache_manifest_sha256") != feature_manifest[
        "cache_manifest_sha256"
    ]:
        raise ValueError("V27 representation cache feature binding differs")
    expected_role = pool_registry["roles"][_require_role(role)]
    parent_ids = [str(value) for value in raw.get("parent_ids") or ()]
    if parent_ids != [str(value) for value in expected_role["parent_ids"]]:
        raise ValueError("V27 representation cache parent order differs")
    if raw.get("parent_ids_sha256") != canonical_sha256(parent_ids):
        raise ValueError("V27 representation cache parent hash differs")
    labels = [str(value) for value in raw.get("label_names") or ()]
    if len(labels) != len(parent_ids) or any(value not in LABEL_TO_CODE for value in labels):
        raise ValueError("V27 representation cache labels differ")
    if raw.get("label_codes") != [LABEL_TO_CODE[value] for value in labels]:
        raise ValueError("V27 representation cache label codes differ")
    tensors = raw.pop("tensors", None)
    if not isinstance(tensors, Mapping):
        raise ValueError("V27 representation cache tensors are missing")
    tensors = {str(name): value for name, value in tensors.items()}
    required = {
        "summary_features",
        "event_trajectory",
        "task_facts_sequence",
        "event_facts_sequence",
        "padding_mask",
        "lengths",
        "sequence_logits",
    }
    if set(tensors) != required:
        raise ValueError("V27 representation cache tensor fields differ")
    count = len(parent_ids)
    max_length = int(context.config.data.max_sequence_length)
    hidden = int(context.config.model.liquid_hidden_dim)
    task_facts = 8 if task == V27_L1_TASK_NAME else 7
    classes = 2 if task == V27_L1_TASK_NAME else 3
    expected_shapes = {
        "summary_features": (count, max_length, hidden * 14),
        "event_trajectory": (count, max_length, hidden * 7 + 19),
        "task_facts_sequence": (count, max_length, task_facts),
        "event_facts_sequence": (count, max_length, 19),
        "padding_mask": (count, max_length),
        "lengths": (count,),
        "sequence_logits": (count, max_length, classes),
    }
    for name, shape in expected_shapes.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(
                f"V27 representation cache tensor shape differs: {name} "
                f"expected={shape}, actual={getattr(value, 'shape', None)}"
            )
        if name == "padding_mask":
            if value.dtype != torch.bool:
                raise ValueError("V27 representation cache padding mask must be bool")
        elif name == "lengths":
            if value.dtype != torch.long:
                raise ValueError("V27 representation cache lengths must be int64")
        elif not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"V27 representation cache tensor is non-finite: {name}")
    lengths = tensors["lengths"]
    if bool((lengths < 1).any()) or bool(
        (lengths > max_length).any()
    ):
        raise ValueError("V27 representation cache lengths are out of bounds")
    expected_padding = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(tensors["padding_mask"].cpu(), expected_padding):
        raise ValueError("V27 representation cache padding differs from lengths")
    padding = tensors["padding_mask"]
    for name in (
        "summary_features",
        "event_trajectory",
        "task_facts_sequence",
        "event_facts_sequence",
    ):
        if bool(tensors[name].masked_select(padding.unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27 padded representation values are non-zero: {name}")
    if bool(tensors["sequence_logits"].masked_select(padding.unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27 padded representation logits are non-zero")
    return {
        **raw,
        "cache_sha256": declared,
        "tensors": tensors,
    }






def load_representation_cache_manifest(context: Any) -> dict[str, Any]:
    pool_registry, feature_manifest = load_open_set_pool_artifacts(context)
    path = resolve_project_path(
        context.project_root,
        context.config.artifacts.representation_cache_manifest_path,
        must_exist=True,
    )
    raw = read_v27_sealed_json(path)
    payload = dict(raw)
    sealed = payload.pop("payload_sha256", None)
    declared = payload.pop("representation_cache_manifest_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("V27 representation manifest hash cannot be recomputed")
    if sealed != canonical_sha256(
        {**payload, "representation_cache_manifest_sha256": declared}
    ):
        raise ValueError("V27 representation manifest payload hash differs")
    if (
        payload.get("schema_version") != REPRESENTATION_MANIFEST_SCHEMA
        or payload.get("status") != "prepared"
        or payload.get("system_manifest_sha256") != context.manifest_sha256
        or payload.get("pool_registry_sha256") != pool_registry["registry_sha256"]
        or payload.get("open_set_feature_cache_manifest_sha256")
        != feature_manifest["cache_manifest_sha256"]
    ):
        raise ValueError("V27 representation manifest binding differs")
    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != {
        f"{task}:{role}" for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME) for role in OPEN_SET_ROLES
    }:
        raise ValueError("V27 representation manifest task set differs")
    result = {
        **payload,
        "representation_cache_manifest_sha256": declared,
        "payload_sha256": sealed,
    }
    for key, entry in tasks.items():
        task, role = key.split(":", 1)
        path_value = resolve_project_path(
            context.project_root, str(entry.get("path", "")), must_exist=True
        )
        if sha256_file(path_value) != entry.get("sha256"):
            raise ValueError(f"V27 representation cache file hash differs: {key}")
        checked = load_representation_cache(context, task=task, role=role)
        if checked["cache_sha256"] != entry.get("cache_sha256"):
            raise ValueError(f"V27 representation cache digest differs: {key}")
    return result


def load_representation_cache(context: Any, *, task: str, role: str) -> dict[str, Any]:
    task = _require_task(task)
    role = _require_role(role)
    pool_registry, feature_manifest = load_open_set_pool_artifacts(context)
    path = _cache_path(context, task=task, role=role)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27 representation cache is not an object")
    return _validate_representation_payload(
        payload,
        context=context,
        task=task,
        role=role,
        pool_registry=pool_registry,
        feature_manifest=feature_manifest,
    )


















































































__all__ = ["LABEL_TO_CODE", "OPEN_SET_ROLES", "REPRESENTATION_CACHE_SCHEMA", "REPRESENTATION_MANIFEST_SCHEMA", "load_representation_cache", "load_representation_cache_manifest"]