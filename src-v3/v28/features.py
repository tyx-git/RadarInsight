"""V28-owned classifier and frozen-role feature-cache preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file
from build.micro_doppler_v18 import _CanonicalFrameProvenance, _build_split

from .pools import build_pool_registry, write_pool_registry
from .protocol import V28Context, V28_L2_TASK, V28_ROLES, load_config, prepare_manifest, read_sealed_json, write_sealed_json


def _load_cache(path: Path, width: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V28 feature source fields differ: {path}")
        ids = [str(value) for value in archive["sample_ids"].tolist()]
        values = np.asarray(archive["values"])
        valid = np.asarray(archive["valid"])
    if len(ids) != len(set(ids)) or values.shape != (len(ids), width) or values.dtype != np.float32 or valid.shape != (len(ids),) or valid.dtype != np.bool_ or not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0) or np.any(values[~valid] != 0.0):
        raise ValueError(f"V28 feature source values differ: {path}")
    return {sample_id: values[index] for index, sample_id in enumerate(ids)}, {sample_id: bool(valid[index]) for index, sample_id in enumerate(ids)}


def _write_npz(path: Path, ids: list[str], values: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    np.savez_compressed(temporary, sample_ids=np.asarray(ids, dtype=np.str_), values=np.asarray(values, dtype=np.float32), valid=np.asarray(valid, dtype=np.bool_))
    generated = Path(str(temporary) + ".npz")
    if generated != path:
        generated.replace(path)
    if temporary.exists():
        temporary.unlink()
    return {"path": relative_project_path(path.parents[3], path) if len(path.parents) >= 4 else path.as_posix(), "sha256": sha256_file(path), "sample_ids_sha256": canonical_sha256(ids), "event_count": len(ids), "valid_event_count": int(np.asarray(valid).sum())}


def _subset(ids: list[str], values: Mapping[str, np.ndarray], valid: Mapping[str, bool], width: int) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros((len(ids), width), dtype=np.float32)
    mask = np.zeros((len(ids),), dtype=np.bool_)
    for index, sample_id in enumerate(ids):
        if sample_id not in values or sample_id not in valid:
            raise ValueError(f"V28 source cache lacks sample: {sample_id}")
        result[index] = values[sample_id]
        mask[index] = bool(valid[sample_id])
    return result, mask


def _build_v28_source_maps(root: Path, config: Any, *, include_test: bool = False) -> dict[str, dict[str, Any]]:
    """Extract V28-owned DBB/Unknown features from the current raw contract."""

    index_path = root / config.data["index_path"]
    frame = pd.read_csv(index_path, low_memory=False)
    provenance_path = root / "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
    canonical_path = root / "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
    if not provenance_path.is_file() or not canonical_path.is_file():
        raise FileNotFoundError("V28 source extraction requires V16 provenance and canonical frame index")
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance = {str(item["parent_sample_id"]): dict(item) for item in provenance_payload.get("records", [])}
    if len(provenance) != int(provenance_payload.get("parent_count", -1)):
        raise ValueError("V28 V16 provenance parent count differs")
    canonical = _CanonicalFrameProvenance(root, canonical_path)
    source_manifest: dict[str, Any] = {
        "schema_version": "RADAR-LN3-V28-source-feature-manifest-v2",
        "experiment_id": "V28",
        "index_sha256": sha256_file(index_path),
        "provenance_sha256": sha256_file(provenance_path),
        "canonical_frame_index_sha256": sha256_file(canonical_path),
        "splits": {},
        "test_payload_access_count": 0,
    }
    maps: dict[str, dict[str, Any]] = {}
    allowed = ("DroneTarget", "BirdTarget", "BalloonTarget", "UnknownTarget")
    split_names = [config.data["train_split_name"], config.data["validation_split_name"]]
    if include_test:
        split_names.append(config.data["test_split_name"])
    for split in split_names:
        physical_path = root / "src-v3/v28/calibration" / f"V28-source-{split}-rd-physical4-v2-cache.npz"
        micro_path = root / "src-v3/v28/calibration" / f"V28-source-{split}-broadband-micro-doppler8-v2-cache.npz"
        expected_ids = _source_event_ids(
            frame,
            split,
            include_clutter=split == str(config.data["test_split_name"]),
        )
        reusable = False
        if physical_path.is_file() and micro_path.is_file():
            physical_values, physical_valid = _load_cache(physical_path, 4)
            micro_values, micro_valid = _load_cache(micro_path, 8)
            reusable = list(physical_values) == expected_ids and list(micro_values) == expected_ids
        if reusable:
            source_manifest["splits"][split] = {"physical": relative_project_path(root, physical_path), "physical_sha256": sha256_file(physical_path), "micro": relative_project_path(root, micro_path), "micro_sha256": sha256_file(micro_path), "reused": True}
        else:
            split_allowed = (*allowed, "ClutterTarget") if split == config.data["test_split_name"] else allowed
            arrays, stats = _build_split(root, split, frame, provenance, canonical, allowed_labels=split_allowed, include_evidence=False)
            ids = [str(value) for value in arrays["sample_ids"].tolist()]
            _write_npz(physical_path, ids, arrays["physical_values"], arrays["valid"])
            _write_npz(micro_path, ids, arrays["summary_values"], arrays["valid"])
            physical_values, physical_valid = _load_cache(physical_path, 4)
            micro_values, micro_valid = _load_cache(micro_path, 8)
            source_manifest["splits"][split] = {"physical": relative_project_path(root, physical_path), "physical_sha256": sha256_file(physical_path), "micro": relative_project_path(root, micro_path), "micro_sha256": sha256_file(micro_path), "reused": False, "extraction_statistics": stats}
    source_manifest["source_manifest_sha256"] = canonical_sha256(source_manifest)
    source_manifest_path = root / "src-v3/v28/calibration/V28-source-feature-manifest-v2.json"
    write_sealed_json(source_manifest_path, source_manifest)
    for split in split_names:
        physical_map, physical_valid = _load_cache(root / source_manifest["splits"][split]["physical"], 4)
        micro_map, micro_valid = _load_cache(root / source_manifest["splits"][split]["micro"], 8)
        maps[split] = {"physical": physical_map, "physical_valid": physical_valid, "micro": micro_map, "micro_valid": micro_valid}
    return maps


def _classifier_parent_ids(frame: pd.DataFrame, split_name: str, *, task: str, bucket: int | None) -> list[str]:
    allowed = {"DroneTarget", "BirdTarget", "BalloonTarget"} if task == V28_L2_TASK else {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"}
    parents = frame.drop_duplicates("parent_sample_id")
    selected = parents[parents["split_name"].astype(str).eq(split_name) & parents["label_name"].astype(str).isin(allowed)].copy()
    if bucket is not None:
        selected = selected[selected["provenance_component_id"].astype(str).map(lambda value: int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % 2 == bucket)]
    return frame[frame["parent_sample_id"].astype(str).isin(selected["parent_sample_id"].astype(str))].sort_values(["parent_sample_id", "causal_event_index"], kind="stable")["sample_id"].astype(str).tolist()


def _all_event_ids(frame: pd.DataFrame, split_name: str) -> list[str]:
    selected = frame[frame["split_name"].astype(str).eq(str(split_name))].copy()
    if selected.empty:
        raise ValueError(f"V28 replay split is empty: {split_name}")
    return selected.sort_values(["parent_sample_id", "causal_event_index"], kind="stable")["sample_id"].astype(str).tolist()


def _source_event_ids(frame: pd.DataFrame, split_name: str, *, include_clutter: bool) -> list[str]:
    labels = {"DroneTarget", "BirdTarget", "BalloonTarget", "UnknownTarget"}
    if include_clutter:
        labels.add("ClutterTarget")
    selected = frame[
        frame["split_name"].astype(str).eq(str(split_name))
        & frame["label_name"].astype(str).isin(labels)
    ]
    if selected.empty:
        raise ValueError(f"V28 source split has no registered events: {split_name}")
    return selected.sort_values(["parent_sample_id", "causal_event_index"], kind="stable")["sample_id"].astype(str).tolist()


def prepare_classifier_features(project_root: str | Path) -> dict[str, Any]:
    """Prepare exact known-only train/development caches without Unknown."""

    root = Path(project_root).resolve()
    config = load_config(root, allow_missing_derived=True)
    frame = pd.read_csv(root / config.data["index_path"], low_memory=False)
    # Test data is intentionally not materialized during classifier feature
    # preparation.  It is created only after endpoints are frozen.
    source_maps = _build_v28_source_maps(root, config, include_test=False)
    records: dict[str, Any] = {}
    for task in ("l1_known_gate", V28_L2_TASK):
        for cache_name, split_name, bucket in (
            ("classifier-train", config.data["train_split_name"], 0),
            ("classifier-development", config.data["train_split_name"], 1),
        ):
            ids = _classifier_parent_ids(frame, split_name, task=task, bucket=bucket)
            if task == "l1_known_gate":
                # L1 does not consume physical/micro features; registry records
                # only an explicit zero-width-free known-only population.
                records[f"{task}:{cache_name}"] = {"parent_ids_sha256": canonical_sha256(sorted(set(frame[frame["sample_id"].astype(str).isin(ids)]["parent_sample_id"].astype(str)))), "event_count": len(ids), "unknown_exposure": 0}
                continue
            physical_map = source_maps[split_name]["physical"]
            physical_valid_map = source_maps[split_name]["physical_valid"]
            micro_map = source_maps[split_name]["micro"]
            micro_valid_map = source_maps[split_name]["micro_valid"]
            physical, physical_valid = _subset(ids, physical_map, physical_valid_map, 4)
            micro, micro_valid = _subset(ids, micro_map, micro_valid_map, 8)
            for name, values, valid in (("rd-physical4", physical, physical_valid), ("broadband-micro-doppler8", micro, micro_valid)):
                path = root / "src-v3/v28/calibration" / f"V28-{cache_name}-{name}-v2-cache.npz"
                records[f"{task}:{cache_name}:{name}"] = _write_npz(path, ids, values, valid)
    manifest = {"schema_version": "RADAR-LN3-V28-classifier-feature-manifest-v2", "experiment_id": "V28", "status": "prepared", "config_sha256": config.sha256, "index_sha256": sha256_file(root / config.data["index_path"]), "caches": records, "unknown_exposure": 0, "test_payload_access_count": 0}
    manifest["feature_manifest_sha256"] = canonical_sha256(manifest)
    path = root / "src-v3/v28/calibration/V28-classifier-feature-manifest-v2.json"
    write_sealed_json(path, manifest)
    return {"path": relative_project_path(root, path), "sha256": sha256_file(path), "feature_manifest_sha256": manifest["feature_manifest_sha256"], "caches": records}


def prepare_open_set_features(project_root: str | Path, *, endpoint_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare frozen role registry and V28-owned L2 open-set feature caches."""

    root = Path(project_root).resolve()
    config = load_config(root, allow_missing_derived=True)
    manifest_path = root / config.artifacts["manifest_path"]
    if not manifest_path.is_file():
        prepare_manifest(root)
    manifest = read_sealed_json(manifest_path)
    context = V28Context(root, config, manifest, manifest_path, sha256_file(manifest_path))
    pool = build_pool_registry(context, endpoint_binding=endpoint_binding)
    write_pool_registry(context, pool)
    frame = pd.read_csv(root / config.data["index_path"], low_memory=False)
    labels = {str(row.sample_id): str(row.label_name) for row in frame.itertuples(index=False)}
    source_maps = _build_v28_source_maps(root, config, include_test=True)
    train_physical, train_physical_valid = source_maps[config.data["train_split_name"]]["physical"], source_maps[config.data["train_split_name"]]["physical_valid"]
    train_micro, train_micro_valid = source_maps[config.data["train_split_name"]]["micro"], source_maps[config.data["train_split_name"]]["micro_valid"]
    val_physical, val_physical_valid = source_maps[config.data["validation_split_name"]]["physical"], source_maps[config.data["validation_split_name"]]["physical_valid"]
    val_micro, val_micro_valid = source_maps[config.data["validation_split_name"]]["micro"], source_maps[config.data["validation_split_name"]]["micro_valid"]
    unknown_physical, unknown_physical_valid = train_physical, train_physical_valid
    unknown_micro, unknown_micro_valid = train_micro, train_micro_valid
    records: dict[str, Any] = {}
    for role in V28_ROLES:
        ids = pool["roles"][V28_L2_TASK][role]["sample_ids"]
        physical_map, physical_valid_map, micro_map, micro_valid_map = (val_physical, val_physical_valid, val_micro, val_micro_valid) if role == "calibration" else (train_physical, train_physical_valid, train_micro, train_micro_valid)
        physical = np.zeros((len(ids), 4), dtype=np.float32); micro = np.zeros((len(ids), 8), dtype=np.float32); physical_valid = np.zeros((len(ids),), dtype=np.bool_); micro_valid = np.zeros((len(ids),), dtype=np.bool_)
        for index, sample_id in enumerate(ids):
            label = labels[sample_id]
            maps = (unknown_physical, unknown_physical_valid, unknown_micro, unknown_micro_valid) if label == "UnknownTarget" and role != "calibration" else (physical_map, physical_valid_map, micro_map, micro_valid_map)
            if label == "ClutterTarget":
                raise ValueError("V28 L2 role contains Clutter")
            physical[index], physical_valid[index] = maps[0][sample_id], bool(maps[1][sample_id])
            micro[index], micro_valid[index] = maps[2][sample_id], bool(maps[3][sample_id])
        role_sources = config.data["l2"]["open_set_caches"][role]
        for name, values, valid, source_key in (
            ("rd-physical4", physical, physical_valid, "rd_physical_cache"),
            ("broadband-micro-doppler8", micro, micro_valid, "micro_doppler_cache"),
        ):
            path = root / str(role_sources[source_key])
            records[f"{role}:{name}"] = _write_npz(path, ids, values, valid)
    # Test features are generated only after both classifier endpoints are
    # supplied.  They are for independent replay/audit, never classifier or
    # auxiliary-head fitting.
    test_ids = _all_event_ids(frame, config.data["test_split_name"])
    test_physical, test_physical_valid = _subset(
        test_ids,
        source_maps[config.data["test_split_name"]]["physical"],
        source_maps[config.data["test_split_name"]]["physical_valid"],
        4,
    )
    test_micro, test_micro_valid = _subset(
        test_ids,
        source_maps[config.data["test_split_name"]]["micro"],
        source_maps[config.data["test_split_name"]]["micro_valid"],
        8,
    )
    for name, values, valid in (("rd-physical4", test_physical, test_physical_valid), ("broadband-micro-doppler8", test_micro, test_micro_valid)):
        path = root / "src-v3/v28/calibration" / f"V28-test-{name}-v2-cache.npz"
        records[f"post_selection_replay:{name}"] = _write_npz(path, test_ids, values, valid)
    payload = {"schema_version": "RADAR-LN3-V28-feature-manifest-v2", "experiment_id": "V28", "status": "prepared", "system_manifest_sha256": context.manifest_sha256, "pool_registry_sha256": pool["registry_sha256"], "endpoint_binding": dict(endpoint_binding), "caches": records, "test_payload_access_count": 0}
    payload["feature_manifest_sha256"] = canonical_sha256(payload)
    path = root / "src-v3/v28/calibration/V28-feature-manifest-v2.json"
    write_sealed_json(path, payload)
    return {"path": relative_project_path(root, path), "sha256": sha256_file(path), "feature_manifest_sha256": payload["feature_manifest_sha256"], "caches": records}


__all__ = ["prepare_classifier_features", "prepare_open_set_features"]
