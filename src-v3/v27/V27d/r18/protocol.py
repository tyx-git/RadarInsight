"""Immutable protocol for the V27d-r18 source-stratified radial-speed audit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from v27.V27d.r17.protocol import (
    R17Config,
    canonical_sha256,
    load_config as load_r17_config,
    load_manifest as load_r17_manifest,
    project_path,
    relative_path,
    sha256_file,
    write_torch_if_identical as _write_r17_torch_if_identical,
)
from v27.V27d.r17.data import load_r17_data_manifest

R18_EXPERIMENT_ID = "V27d-r18"
R18_SCHEMA_VERSION = "RADAR-LN3-V27d-r18-source-stratified-radial-speed-v3"
R18_CONFIG_SCHEMA = "RADAR-LN3-V27d-r18-config-v3"
R18_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r18-manifest-v3"
R18_REPORT_SCHEMA = "RADAR-LN3-V27d-r18-radial-speed-diagnostic-v3"
R18_SMOKE_SCHEMA = "RADAR-LN3-V27d-r18-contract-smoke-v3"
R18_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r18-l2-checkpoint-v3"
R18_TRAINING_SUMMARY_SCHEMA = "RADAR-LN3-V27d-r18-training-summary-v4"
R18_COMPARISON_REPORT_SCHEMA = "RADAR-LN3-V27d-r18-trained-comparison-v2"
R18_ROOT = Path("src-v3") / "v27" / "V27d" / "r18"
R18_DATA_ROOT = Path("data") / "processed" / "v27d_r17"
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


def artifact_path(project_root: str | Path, template: str, *, arm: str | None = None, seed: int | None = None, epoch: int | None = None) -> Path:
    root = Path(project_root).resolve()
    if "{arm}" in template and arm not in {"full", "radial_speed_ablation"}:
        raise ValueError("r18 artifact arm is invalid")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("r18 artifact seed is invalid")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("r18 artifact epoch is invalid")
    rendered = template.format(arm=arm, seed=seed, epoch=epoch)
    target = project_path(root, rendered)
    try:
        target.relative_to((root / R18_ROOT).resolve())
    except ValueError as exc:
        raise ValueError("r18 artifact escaped experiment root") from exc
    return target


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "payload_sha256" in value:
        raise ValueError("r18 payload is already sealed")
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"r18 refuses to overwrite a different artifact: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parent_rows(root: Path, config: R18Config) -> list[dict[str, Any]]:
    manifest = load_r17_data_manifest(root, config.r17)
    parents_path = project_path(root, manifest["parents_path"], must_exist=True)
    return [json.loads(line) for line in parents_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _code_hashes(root: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in (root / R18_ROOT).glob("*.py")
        if "__pycache__" not in path.parts
    )
    return {relative_path(root, path): sha256_file(path) for path in paths}


def write_torch_if_identical(path: str | Path, payload: Any) -> None:
    """Atomically create an r18 torch artifact and reject divergent rewrites."""

    _write_r17_torch_if_identical(path, payload)


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


def build_manifest(root: str | Path, config: R18Config) -> dict[str, Any]:
    project_root = Path(root).resolve()
    split = build_source_stratified_split(project_root, config)
    r17_data_manifest = project_path(
        project_root, config.r17.artifacts["data_manifest_path"], must_exist=True
    )
    r17_manifest = project_path(
        project_root, config.r17.artifacts["manifest_path"], must_exist=True
    )
    frozen_l2 = project_path(
        project_root, config.r17.source["r23_checkpoint_path"], must_exist=True
    )
    load_r17_manifest(
        project_root, config.r17, allow_evaluation_code_drift=True
    )
    return {
        "schema_version": R18_MANIFEST_SCHEMA,
        "experiment_id": R18_EXPERIMENT_ID,
        "variant": config["variant"],
        "config": {"path": relative_path(project_root, config.path), "sha256": config.sha256, "identity_sha256": config.identity_sha256},
        "source": {
            "dataset": config.source["dataset"],
            "r17_config_path": config.source["r17_config_path"],
            "r17_config_sha256": config.r17.sha256,
            "r17_data_manifest_path": relative_path(project_root, r17_data_manifest),
            "r17_data_manifest_sha256": sha256_file(r17_data_manifest),
            "r17_manifest_path": relative_path(project_root, r17_manifest),
            "r17_manifest_sha256": sha256_file(r17_manifest),
            "frozen_l2_checkpoint_path": relative_path(project_root, frozen_l2),
            "frozen_l2_checkpoint_sha256": sha256_file(frozen_l2),
        },
        "split": {key: value for key, value in split.items() if key != "assignments"},
        "parent_assignments": split["assignments"],
        "model": {"frozen_l2": "V27c-r23 robust_control seed42 epoch1", "class_names": list(R18_L2_CLASSES), "radial_speed_ablation": "remove radial_speed_mps before TR and causal observation-increment encoding"},
        "training_boundary": {"unknown_training_exposure": 0, "test_payload_access_count": 0, "rd_payload_access_count": 0, "l3": "not_executed"},
        "code_hashes": _code_hashes(project_root),
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[R18Config, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    path = artifact_path(root, config.artifacts["manifest_path"])
    write_json_if_identical(path, seal(build_manifest(root, config)))
    return config, path, sha256_file(path)


def load_manifest(project_root: str | Path, config: R18Config) -> tuple[dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    path = artifact_path(root, config.artifacts["manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("r18 manifest is not an object")
    declared = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if declared != canonical_sha256(body) or body != build_manifest(root, config):
        raise ValueError("r18 manifest differs from config, data or split")
    return dict(payload), path, sha256_file(path)


__all__ = ["R18_CHECKPOINT_SCHEMA", "R18_COMPARISON_REPORT_SCHEMA", "R18Config", "R18_EXPERIMENT_ID", "R18_L2_CLASSES", "R18_MANIFEST_SCHEMA", "R18_REPORT_SCHEMA", "R18_ROOT", "R18_SCHEMA_VERSION", "R18_SMOKE_SCHEMA", "R18_TRAINING_SUMMARY_SCHEMA", "artifact_path", "build_manifest", "build_source_stratified_split", "canonical_sha256", "load_config", "load_manifest", "prepare_manifest", "project_path", "relative_path", "seal", "sha256_file", "write_json_if_identical", "write_torch_if_identical"]
