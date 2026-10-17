"""Open-set pool and immutable historical-cache validation for V27d."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from radar_ln3.runtime import canonical_sha256

from .protocol import V27DContext, V27D_LABELS, V27D_ROLES, project_path


POOL_SCHEMA = "RADAR-LN3-V27-open-set-pool-registry-v1"


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




__all__ = ["POOL_SCHEMA", "load_pools"]