"""Immutable protocol for the V27d-r18 source-stratified radial-speed audit."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from v27.V27d.r17.protocol import R17Config, canonical_sha256, load_config as load_r17_config, project_path, sha256_file
from v27.V27d.r17.data import load_r17_data_manifest

R18_EXPERIMENT_ID = "V27d-r18"
R18_SCHEMA_VERSION = "RADAR-LN3-V27d-r18-source-stratified-radial-speed-v3"
R18_CONFIG_SCHEMA = "RADAR-LN3-V27d-r18-config-v3"
R18_ROOT = Path("src-v3") / "v27" / "V27d" / "r18"
R18_L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"r18 {name} must be non-empty text")
    return value.strip()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"r18 {name} must be an object")
    return dict(value)


class R18Config:
    def __init__(self, payload: dict[str, Any], path: Path, r17: R17Config) -> None:
        self.payload = payload
        self.path = path
        self.r17 = r17
        self.sha256 = sha256_file(path)
        self.identity_sha256 = canonical_sha256(payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    @property
    def source(self) -> Mapping[str, Any]:
        return self.payload["source"]

    @property
    def data(self) -> Mapping[str, Any]:
        return self.payload["data"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> R18Config:
    root = Path(project_root).resolve()
    path = project_path(
        root,
        config_path
        or R18_ROOT / "config" / "V27d-r18-source-stratified-radial-speed.json",
        must_exist=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("r18 configuration must be an object")
    value = dict(payload)
    if value.get("schema_version") != R18_CONFIG_SCHEMA or value.get("experiment_id") != R18_EXPERIMENT_ID:
        raise ValueError("r18 configuration identity differs")
    source = _mapping(value.get("source"), "source")
    r17_config_path = _text(source.get("r17_config_path"), "source.r17_config_path")
    r17 = load_r17_config(root, r17_config_path)
    if _text(source.get("dataset"), "source.dataset") != "OriData/excel/TR-only-processed-r17":
        raise ValueError("r18 dataset source differs")
    value["source"] = source
    data = _mapping(value.get("data"), "data")
    if _text(data.get("split_strategy"), "data.split_strategy") != "source_suffix+label_parent_stratified_deterministic":
        raise ValueError("r18 split strategy differs")
    if isinstance(data.get("split_seed"), bool) or not isinstance(data.get("split_seed"), int) or int(data["split_seed"]) < 0:
        raise ValueError("r18 split seed is invalid")
    if float(data.get("validation_fraction", 0.0)) <= 0.0 or float(data.get("validation_fraction", 0.0)) >= 1.0:
        raise ValueError("r18 validation fraction is invalid")
    if int(data.get("minimum_group_size_for_validation", 0)) < 2:
        raise ValueError("r18 minimum group size is invalid")
    value["data"] = data
    training = _mapping(value.get("training"), "training")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("r18 training seeds are invalid")
    for key in ("epochs", "batch_size", "gradient_clip_norm"):
        if float(training.get(key, 0)) <= 0.0:
            raise ValueError(f"r18 training.{key} is invalid")
    if _text(training.get("loss_name"), "training.loss_name") != r17.training["loss_name"]:
        raise ValueError("r18 loss differs from r17")
    if _text(training.get("loss_reduction"), "training.loss_reduction") != r17.training["loss_reduction"]:
        raise ValueError("r18 loss reduction differs from r17")
    value["training"] = training
    artifacts = _mapping(value.get("artifacts"), "artifacts")
    for key in ("manifest_path", "diagnostic_report_path", "comparison_report_path", "smoke_path", "checkpoint_template", "epoch_checkpoint_template", "training_summary_template"):
        _text(artifacts.get(key), f"artifacts.{key}")
    value["artifacts"] = artifacts
    return R18Config(value, path, r17)








def _parent_rows(root: Path, config: R18Config) -> list[dict[str, Any]]:
    manifest = load_r17_data_manifest(root, config.r17)
    parents_path = project_path(root, manifest["parents_path"], must_exist=True)
    return [json.loads(line) for line in parents_path.read_text(encoding="utf-8").splitlines() if line.strip()]






def build_source_stratified_split(root: str | Path, config: R18Config) -> dict[str, Any]:
    project_root = Path(root).resolve()
    rows = _parent_rows(project_root, config)
    fraction = float(config.data["validation_fraction"])
    minimum = int(config.data["minimum_group_size_for_validation"])
    assignments: dict[str, str] = {}
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        label = str(row["label_name"])
        if label in R18_L2_CLASSES:
            groups[(str(row["source_suffix"]), label)].append(str(row["parent_sample_id"]))
    for (source, label), parent_ids in sorted(groups.items()):
        split_seed = int(config.data["split_seed"])
        ordered = sorted(parent_ids, key=lambda pid: hashlib.sha256(f"{source}|{label}|{split_seed}|{pid}".encode()).hexdigest())
        validation_count = 0
        if len(ordered) >= minimum:
            validation_count = min(max(1, int(round(len(ordered) * fraction))), len(ordered) - 1)
        for parent_id in ordered[:validation_count]:
            assignments[parent_id] = "validation"
        for parent_id in ordered[validation_count:]:
            assignments[parent_id] = "train"
    if not assignments:
        raise ValueError("r18 source-stratified split is empty")
    split_rows = [row for row in rows if str(row["parent_sample_id"]) in assignments]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in split_rows:
        parent_id = str(row["parent_sample_id"])
        counts[assignments[parent_id]][str(row["label_name"])] += 1
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in split_rows:
        source_counts[str(row["source_suffix"])][assignments[str(row["parent_sample_id"])] + "|" + str(row["label_name"])] += 1
    return {
        "strategy": "source_suffix+label_parent_stratified_deterministic",
        "seed": int(config.data["split_seed"]),
        "validation_fraction": fraction,
        "minimum_group_size_for_validation": minimum,
        "parent_count": len(assignments),
        "split_counts": {split: dict(sorted(value.items())) for split, value in sorted(counts.items())},
        "source_counts": {source: dict(sorted(value.items())) for source, value in sorted(source_counts.items())},
        "assignments": dict(sorted(assignments.items())),
        "parent_ids_sha256": canonical_sha256(assignments),
    }








__all__ = ["R18Config", "R18_EXPERIMENT_ID", "R18_L2_CLASSES", "R18_ROOT", "R18_SCHEMA_VERSION", "build_source_stratified_split", "load_config"]