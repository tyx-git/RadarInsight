"""Open-set pool and immutable historical-cache validation for V27d."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file

from .protocol import (
    V27DContext,
    V27D_KNOWN_LABELS,
    V27D_LABELS,
    V27D_ROLES,
    project_path,
)


POOL_SCHEMA = "RADAR-LN3-V27-open-set-pool-registry-v1"
FEATURE_MANIFEST_SCHEMA = "RADAR-LN3-V27-open-set-feature-cache-manifest-v1"
REPRESENTATION_MANIFEST_SCHEMA = "RADAR-LN3-V27-representation-cache-manifest-v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d source JSON is not an object: {path}")
    return dict(value)


def _verify_sealed_pool(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    sealed = value.pop("payload_sha256", None)
    if not isinstance(sealed, str) or sealed != canonical_sha256(value):
        raise ValueError("V27d open-set pool sealed hash differs")
    registry_hash = value.pop("registry_sha256", None)
    if not isinstance(registry_hash, str) or registry_hash != canonical_sha256(value):
        raise ValueError("V27d open-set pool registry hash differs")
    value["registry_sha256"] = registry_hash
    value["payload_sha256"] = sealed
    return value


def _verify_source_manifest(raw: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    value = dict(raw)
    if value.get("schema_version") != schema or value.get("status") != "prepared":
        raise ValueError("V27d source manifest schema/status differs")
    sealed = value.get("payload_sha256")
    if not isinstance(sealed, str):
        raise ValueError("V27d source manifest is not sealed")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if sealed != canonical_sha256(body):
        raise ValueError("V27d source manifest sealed hash differs")
    return value


def _historical_alias(root: Path, raw_path: str | Path) -> tuple[Path, bool]:
    """Resolve only the known V27 directory migration alias.

    The frozen binding predates the rename to ``v27_baseline``.  No general
    fallback is allowed: accepting arbitrary alternate paths would make a
    checkpoint's declared provenance meaningless.
    """

    text = str(raw_path).replace("\\", "/").strip()
    direct = project_path(root, text)
    if direct.exists():
        return direct, False
    prefix = "src-v3/v27/"
    baseline_prefix = "src-v3/v27/v27_baseline/"
    if text.startswith(prefix) and not text.startswith(baseline_prefix):
        suffix = text[len(prefix) :]
        candidate = project_path(root, f"src-v3/v27/v27_baseline/{suffix}")
        if candidate.exists():
            return candidate, True
    raise FileNotFoundError(direct)


def _tensor_state_hash(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d checkpoint state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend(
            (str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex())
        )
    return canonical_sha256(parts)


def load_frozen_binding(context: V27DContext) -> dict[str, Any]:
    root = context.project_root
    path = project_path(root, context.config.source.frozen_binding_path, must_exist=True)
    if sha256_file(path) != context.config.source.expected_hashes["frozen_binding"]:
        raise ValueError("V27d frozen binding file hash differs")
    raw = _read_object(path)
    declared = raw.pop("binding_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(raw):
        raise ValueError("V27d frozen binding payload hash differs")
    if raw.get("schema_version") != "RADAR-LN3-V27-frozen-classification-base-binding-v1":
        raise ValueError("V27d frozen binding schema differs")
    if raw.get("status") != "frozen" or raw.get("classification_only") is not True:
        raise ValueError("V27d frozen binding is not classification-only")
    if raw.get("classifier_state_must_remain_identical") is not True:
        raise ValueError("V27d frozen binding does not freeze classifier state")
    if int(raw.get("seed", -1)) != 42:
        raise ValueError("V27d frozen binding seed differs")

    expected_epochs = {"l1": 14, "l2": 11}
    resolved: dict[str, str] = {}
    for key, epoch in expected_epochs.items():
        record = raw.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"V27d frozen binding lacks {key}")
        if int(record.get("epoch", -1)) != epoch or int(record.get("seed", -1)) != 42:
            raise ValueError(f"V27d frozen {key} epoch/seed differs")
        task = "known_gate" if key == "l1" else "dbb_specialist"
        if record.get("task") != task:
            raise ValueError(f"V27d frozen {key} task differs")
        checkpoint, aliased = _historical_alias(root, str(record.get("checkpoint_path", "")))
        if sha256_file(checkpoint) != str(record.get("checkpoint_sha256", "")):
            raise ValueError(f"V27d frozen {key} checkpoint hash differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("task") != task:
            raise ValueError(f"V27d frozen {key} checkpoint payload differs")
        state = payload.get("model_state")
        if not isinstance(state, Mapping) or payload.get("model_state_sha256") != _tensor_state_hash(state):
            raise ValueError(f"V27d frozen {key} model state hash differs")
        if payload.get("model_state_sha256") != record.get("model_state_sha256"):
            raise ValueError(f"V27d frozen {key} declared state hash differs")
        if payload.get("unknown_is_model_class") is not False or int(payload.get("unknown_training_exposure", -1)) != 0:
            raise ValueError(f"V27d frozen {key} violates Unknown class contract")
        resolved[key] = relative_project_path(root, checkpoint)
        raw[key] = {
            **dict(record),
            "resolved_checkpoint_path": resolved[key],
            "historical_path_alias_used": aliased,
        }
    raw["binding_sha256"] = declared
    raw["resolved_binding_path"] = relative_project_path(root, path)
    return raw


def _role_rebuild(frame: pd.DataFrame, *, role: str, split_name: str, parent_ids: list[str]) -> dict[str, Any]:
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise ValueError(f"V27d {role} parent ids are not unique")
    selected = frame[frame["parent_sample_id"].astype(str).isin(parent_ids)].copy()
    metadata_columns = ["provenance_component_id", "capture_id", "split_name", "label_name"]
    conflicts = selected.groupby("parent_sample_id", sort=False)[metadata_columns].nunique(dropna=False)
    if bool((conflicts > 1).any().any()):
        raise ValueError(f"V27d {role} parent metadata is inconsistent")
    parent = selected[
        ["parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name"]
    ].drop_duplicates("parent_sample_id")
    if len(parent) != len(parent_ids):
        raise ValueError(f"V27d {role} parent coverage differs")
    if set(parent["split_name"].astype(str)) != {split_name}:
        raise ValueError(f"V27d {role} crosses a split boundary")
    positions = {parent_id: index for index, parent_id in enumerate(parent_ids)}
    parent["_order"] = parent["parent_sample_id"].astype(str).map(positions)
    parent = parent.sort_values("_order", kind="stable")
    selected["_order"] = selected["parent_sample_id"].astype(str).map(positions)
    selected = selected.sort_values(["_order", "causal_event_index"], kind="stable")
    sample_ids = selected["sample_id"].astype(str).tolist()
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"V27d {role} repeats an event id")
    labels = parent["label_name"].astype(str).tolist()
    label_counts = {label: labels.count(label) for label in V27D_LABELS}
    if any(count <= 0 for count in label_counts.values()):
        raise ValueError(f"V27d {role} lacks a mixed-label support group: {label_counts}")
    return {
        "role": role,
        "split_name": split_name,
        "parent_ids": parent_ids,
        "parent_count": len(parent_ids),
        "parent_label_counts": label_counts,
        "sample_ids": sample_ids,
        "sample_ids_sha256": canonical_sha256(sample_ids),
        "event_count": len(sample_ids),
        "provenance_component_ids": parent["provenance_component_id"].astype(str).tolist(),
        "capture_ids": parent["capture_id"].astype(str).tolist(),
    }


def load_pools(context: V27DContext) -> dict[str, Any]:
    root = context.project_root
    source = context.config.source
    path = project_path(root, source.baseline_pool_registry_path, must_exist=True)
    registry = _verify_sealed_pool(_read_object(path))
    if registry.get("schema_version") != POOL_SCHEMA or registry.get("experiment_id") != "V27":
        raise ValueError("V27d pool registry identity differs")
    index = pd.read_csv(project_path(root, source.index_path, must_exist=True), low_memory=False)
    required = {
        "sample_id", "parent_sample_id", "provenance_component_id", "capture_id",
        "split_name", "label_name", "causal_event_index",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"V27d pool index lacks columns: {missing}")
    index["sample_id"] = index["sample_id"].astype(str)
    index["parent_sample_id"] = index["parent_sample_id"].astype(str)
    if index["sample_id"].duplicated().any() or index["parent_sample_id"].eq("").any():
        raise ValueError("V27d pool index has duplicate/empty identities")
    parent_meta = index[
        ["parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name"]
    ].drop_duplicates("parent_sample_id")
    if len(parent_meta) != index["parent_sample_id"].nunique():
        raise ValueError("V27d pool index parent metadata is inconsistent")
    roles = registry.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(V27D_ROLES):
        raise ValueError("V27d pool registry role set differs")
    checked_roles: dict[str, Any] = {}
    parent_sets: dict[str, set[str]] = {}
    component_sets: dict[str, set[str]] = {}
    for role in V27D_ROLES:
        record = dict(roles[role])
        expected_split = "train" if role != "calibration" else "val"
        if record.get("role") != role or record.get("split_name") != expected_split:
            raise ValueError(f"V27d pool role contract differs: {role}")
        ids = [str(value) for value in record.get("parent_ids") or ()]
        rebuilt = _role_rebuild(index, role=role, split_name=expected_split, parent_ids=ids)
        for field in (
            "role", "split_name", "parent_ids", "parent_count", "parent_label_counts",
            "sample_ids", "sample_ids_sha256", "event_count", "provenance_component_ids",
            "capture_ids",
        ):
            if record.get(field) != rebuilt.get(field):
                raise ValueError(f"V27d pool role differs from current index: {role}/{field}")
        parent_sets[role] = set(ids)
        component_sets[role] = set(rebuilt["provenance_component_ids"])
        checked_roles[role] = rebuilt
    for index_name, left in enumerate(V27D_ROLES):
        for right in V27D_ROLES[index_name + 1 :]:
            if parent_sets[left] & parent_sets[right] or component_sets[left] & component_sets[right]:
                raise ValueError(f"V27d pool roles overlap: {left}/{right}")
    result = {**registry, "roles": checked_roles, "index_frame": index}
    return result


def load_source_manifests(context: V27DContext, pools: Mapping[str, Any]) -> dict[str, Any]:
    root = context.project_root
    source = context.config.source
    feature_path = project_path(root, source.baseline_feature_manifest_path, must_exist=True)
    representation_path = project_path(root, source.baseline_representation_manifest_path, must_exist=True)
    feature = _verify_source_manifest(_read_object(feature_path), schema=FEATURE_MANIFEST_SCHEMA)
    representation = _verify_source_manifest(_read_object(representation_path), schema=REPRESENTATION_MANIFEST_SCHEMA)
    if feature.get("pool_registry_sha256") != pools["registry_sha256"]:
        raise ValueError("V27d feature manifest pool binding differs")
    if representation.get("pool_registry_sha256") != pools["registry_sha256"]:
        raise ValueError("V27d representation manifest pool binding differs")
    tasks = representation.get("tasks")
    expected_tasks = {f"{task}:{role}" for task in ("known_gate", "dbb_specialist") for role in V27D_ROLES}
    if not isinstance(tasks, Mapping) or set(tasks) != expected_tasks:
        raise ValueError("V27d representation task/role set differs")
    return {"feature": feature, "representation": representation}


__all__ = [
    "FEATURE_MANIFEST_SCHEMA",
    "POOL_SCHEMA",
    "REPRESENTATION_MANIFEST_SCHEMA",
    "load_frozen_binding",
    "load_pools",
    "load_source_manifests",
]
