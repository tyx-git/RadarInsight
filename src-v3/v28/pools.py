"""Parent-grouped V28-v2 role registry.

The registry is created only after both classifier endpoints are frozen.  It
never supplies labels to the classifier; it only defines the later
representation/head/calibration populations and their provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file

from .protocol import (
    V28Context,
    V28_L1_TASK,
    V28_L2_TASK,
    V28_POOL_SCHEMA,
    V28_ROLES,
    artifact_path,
    read_sealed_json,
    write_sealed_json,
)


def _partition(value: str) -> int:
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "big") % 2


def _load_parent_frame(context: V28Context) -> pd.DataFrame:
    path = context.project_root / context.config.data["index_path"]
    frame = pd.read_csv(path, low_memory=False)
    required = {"parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name", "sample_id", "causal_event_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V28 role index is missing columns: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("V28 role index repeats sample IDs")
    parents = frame[["parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name"]].drop_duplicates("parent_sample_id")
    if len(parents) != frame["parent_sample_id"].nunique():
        raise ValueError("V28 role parent metadata is inconsistent")
    return frame


def _role_record(frame: pd.DataFrame, *, role: str, split_name: str, parent_ids: list[str]) -> dict[str, Any]:
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise ValueError(f"V28 {role} parent IDs are empty or duplicated")
    rows = frame[frame["parent_sample_id"].astype(str).isin(parent_ids)].copy()
    parent = rows[["parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name"]].drop_duplicates("parent_sample_id")
    if len(parent) != len(parent_ids) or set(parent["split_name"].astype(str)) != {split_name}:
        raise ValueError(f"V28 {role} parent split/coverage differs")
    positions = {value: index for index, value in enumerate(parent_ids)}
    parent = parent.assign(_order=parent["parent_sample_id"].astype(str).map(positions)).sort_values("_order", kind="stable")
    ordered_rows = rows.assign(_order=rows["parent_sample_id"].astype(str).map(positions)).sort_values(["_order", "causal_event_index"], kind="stable")
    label_counts = parent["label_name"].astype(str).value_counts().to_dict()
    return {
        "role": role,
        "split_name": split_name,
        "parent_ids": parent_ids,
        "parent_ids_sha256": canonical_sha256(parent_ids),
        "parent_count": len(parent_ids),
        "parent_label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "event_count": int(len(ordered_rows)),
        "sample_ids": ordered_rows["sample_id"].astype(str).tolist(),
        "sample_ids_sha256": canonical_sha256(ordered_rows["sample_id"].astype(str).tolist()),
        "provenance_component_ids": parent["provenance_component_id"].astype(str).tolist(),
        "provenance_component_ids_sha256": canonical_sha256(parent["provenance_component_id"].astype(str).tolist()),
        "capture_ids": parent["capture_id"].astype(str).tolist(),
    }


def build_pool_registry(context: V28Context, *, endpoint_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Create disjoint train-role and validation calibration populations."""

    frame = _load_parent_frame(context)
    parents = frame[["parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name"]].drop_duplicates("parent_sample_id")
    train = parents[parents["split_name"].astype(str).eq(context.config.data["train_split_name"])].copy()
    validation = parents[parents["split_name"].astype(str).eq(context.config.data["validation_split_name"])].copy()
    if train.empty or validation.empty:
        raise ValueError("V28 role source split is empty")
    train["_bucket"] = train["provenance_component_id"].astype(str).map(_partition)
    fit = train[train["_bucket"].eq(0)].sort_values("parent_sample_id")["parent_sample_id"].astype(str).tolist()
    development = train[train["_bucket"].eq(1)].sort_values("parent_sample_id")["parent_sample_id"].astype(str).tolist()
    calibration = validation.sort_values("parent_sample_id")["parent_sample_id"].astype(str).tolist()
    if set(fit) & set(development) or set(fit) & set(calibration) or set(development) & set(calibration):
        raise ValueError("V28 role registry has overlapping parents")
    role_ids = {"representation_fit": fit, "representation_development": development, "calibration": calibration}
    roles: dict[str, dict[str, Any]] = {}
    for task, excluded in ((V28_L1_TASK, frozenset()), (V28_L2_TASK, frozenset({"ClutterTarget"}))):
        roles[task] = {}
        for role, ids in role_ids.items():
            selected = [parent_id for parent_id in ids if str(parents.loc[parents["parent_sample_id"].astype(str).eq(parent_id), "label_name"].iloc[0]) not in excluded]
            entry = _role_record(frame, role=role, split_name=context.config.data["validation_split_name"] if role == "calibration" else context.config.data["train_split_name"], parent_ids=selected)
            if entry["parent_count"] <= 0 or not entry["parent_label_counts"]:
                raise ValueError(f"V28 {task}/{role} role is empty")
            roles[task][role] = entry
    payload = {
        "schema_version": V28_POOL_SCHEMA,
        "experiment_id": "V28",
        "status": "prepared",
        "system_manifest_sha256": context.manifest_sha256,
        "index_path": relative_project_path(context.project_root, context.project_root / context.config.data["index_path"]),
        "index_sha256": sha256_file(context.project_root / context.config.data["index_path"]),
        "endpoint_binding": dict(endpoint_binding),
        "partition": {"unit": "provenance_component_id", "method": "sha256_first_64_bits_mod_2", "fit_bucket": 0, "development_bucket": 1},
        "roles": roles,
        "policy": {
            "classification_training": "known-only; registry not consumed",
            "representation_fit": "frozen classifier outputs; Unknown allowed after endpoint freeze",
            "representation_development": "frozen classifier outputs; head selection only",
            "calibration": "validation role; threshold fitting only",
            "l1_unknown": "allowed after freeze",
            "l2_unknown": "allowed after freeze",
            "l2_clutter": "excluded from classifier and auxiliary head; endpoint pollution audit only",
            "unknown_normalization_exposure": 0,
        },
        "post_selection_audit": {"status": "absent", "parent_count": 0, "reason": "no independent source pool exposed"},
        "test_payload_access_count": 0,
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


def registry_path(context: V28Context) -> Path:
    return artifact_path(context.project_root, str(context.config.artifacts["pool_registry_path"]))


def write_pool_registry(context: V28Context, payload: Mapping[str, Any]) -> Path:
    path = registry_path(context)
    write_sealed_json(path, payload)
    return path


def verify_pool_registry(context: V28Context, payload: Mapping[str, Any], *, endpoint_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = dict(payload)
    envelope = value.pop("payload_sha256", None)
    if envelope != canonical_sha256(value):
        raise ValueError("V28 pool registry envelope hash differs")
    declared = value.pop("registry_sha256", None)
    if declared != canonical_sha256(value):
        raise ValueError("V28 pool registry hash differs")
    if value.get("schema_version") != V28_POOL_SCHEMA or value.get("status") != "prepared" or value.get("system_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V28 pool registry identity differs")
    if endpoint_binding is not None and dict(value.get("endpoint_binding") or {}) != dict(endpoint_binding):
        raise ValueError("V28 pool registry endpoint binding differs")
    roles = value.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {V28_L1_TASK, V28_L2_TASK}:
        raise ValueError("V28 pool registry task-role sections differ")
    frame = _load_parent_frame(context)
    all_ids: set[str] = set()
    for task in (V28_L1_TASK, V28_L2_TASK):
        task_roles = roles[task]
        if set(task_roles) != set(V28_ROLES):
            raise ValueError(f"V28 pool registry roles differ for {task}")
        for role in V28_ROLES:
            entry = task_roles[role]
            expected = _role_record(frame, role=role, split_name=str(entry["split_name"]), parent_ids=[str(v) for v in entry["parent_ids"]])
            for key in ("parent_count", "parent_ids_sha256", "event_count", "sample_ids_sha256", "parent_label_counts"):
                if entry.get(key) != expected.get(key):
                    raise ValueError(f"V28 pool registry role differs: {task}/{role}/{key}")
            ids = set(expected["parent_ids"])
            if all_ids & ids and task == V28_L1_TASK:
                raise ValueError("V28 pool registry roles overlap")
            if task == V28_L1_TASK:
                all_ids |= ids
    return {**value, "registry_sha256": declared, "payload_sha256": envelope}


def load_pool_registry(context: V28Context, *, endpoint_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = registry_path(context)
    checked = verify_pool_registry(context, read_sealed_json(path), endpoint_binding=endpoint_binding)
    return checked


__all__ = ["build_pool_registry", "load_pool_registry", "registry_path", "verify_pool_registry", "write_pool_registry"]
