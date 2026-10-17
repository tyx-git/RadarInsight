"""Parent-grouped V28-v2 role registry.

The registry is created only after both classifier endpoints are frozen.  It
never supplies labels to the classifier; it only defines the later
representation/head/calibration populations and their provenance.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from radar_ln3.runtime import canonical_sha256

from .protocol import V28Context, V28_AUDIT_ROLE, V28_L1_TASK, V28_L2_TASK, V28_POOL_SCHEMA, V28_ROLES




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
            allowed_labels = (
                {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}
                if task == V28_L1_TASK
                else {"DroneTarget", "BirdTarget", "BalloonTarget", "UnknownTarget"}
            )
            if set(expected["parent_label_counts"]) - allowed_labels:
                raise ValueError(f"V28 pool registry labels differ: {task}/{role}")
            ids = set(expected["parent_ids"])
            if all_ids & ids and task == V28_L1_TASK:
                raise ValueError("V28 pool registry roles overlap")
            if task == V28_L1_TASK:
                all_ids |= ids
    audit = value.get("post_selection_audit")
    if not isinstance(audit, Mapping) or audit.get("status") != "registered" or audit.get("role") != V28_AUDIT_ROLE:
        raise ValueError("V28 post-selection audit pool is missing or not registered")
    if int(audit.get("payload_access_count", -1)) != 0 or audit.get("threshold_fitting_allowed") is not False:
        raise ValueError("V28 post-selection audit pool violated the fitting boundary")
    expected_audit = _role_record(frame, role=V28_AUDIT_ROLE, split_name=str(audit.get("split_name")), parent_ids=[str(item) for item in audit.get("parent_ids", [])])
    for key in ("parent_count", "parent_ids_sha256", "event_count", "sample_ids_sha256", "parent_label_counts"):
        if audit.get(key) != expected_audit.get(key):
            raise ValueError(f"V28 post-selection audit pool differs: {key}")
    if set(expected_audit["parent_ids"]) & all_ids:
        raise ValueError("V28 post-selection audit pool overlaps a fitting role")
    return {**value, "registry_sha256": declared, "payload_sha256": envelope}




__all__ = ["verify_pool_registry"]