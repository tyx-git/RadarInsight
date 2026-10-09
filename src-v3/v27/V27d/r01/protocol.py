"""V27d-r01 独立协议：只读引用 r20，所有新产物落在 r01。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

R01_EXPERIMENT_ID = "V27d-r01"
R01_SCHEMA_VERSION = "RADAR-LN3-V27d-r01-denoising-replication-v1"
R01_CONFIG_SCHEMA = "RADAR-LN3-V27d-r01-config-v1"
R01_LAYOUT_CANDIDATES = ((50, "q_plus_1"), (62, "q"), (74, "q"))
R01_TARGET_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
R01_PROTECTED_LABELS = ("DroneTarget", "BirdTarget")
R01_X5A_FIELD_NAMES = tuple(f"u16_{index:02d}" for index in range(12))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = root / (config_path or Path("src-v3/v27/V27d/r01/config/V27d-r01-denoising-replication.json"))
    if not path.is_absolute():
        path = root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("r01 config must be an object")
    result = dict(payload)
    if result.get("schema_version") != R01_CONFIG_SCHEMA:
        raise ValueError("r01 config schema differs")
    if result.get("experiment_id") != R01_EXPERIMENT_ID:
        raise ValueError("r01 experiment identity differs")
    if result.get("data", {}).get("pointwise_input") is not True:
        raise ValueError("r01 must use pointwise input")
    if result.get("training", {}).get("seed") != 42:
        raise ValueError("r01 seed must be 42")
    result["_path"] = str(path)
    result["_sha256"] = sha256_file(path)
    result["_identity_sha256"] = canonical_sha256(payload)
    return result


def project_path(project_root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = Path(project_root).resolve()
    path = Path(value)
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("r01 path escaped project root") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def r01_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return root / "src-v3" / "v27" / "V27d" / "r01"


def artifact_path(project_root: str | Path, relative_path: str | Path) -> Path:
    root = Path(project_root).resolve()
    target = project_path(root, relative_path)
    try:
        target.relative_to(r01_root(root).resolve())
    except ValueError as exc:
        raise ValueError("r01 artifact must remain inside r01") from exc
    return target


def hash_tree(root: str | Path) -> dict[str, str]:
    base = r01_root(root)
    return {
        path.relative_to(Path(root).resolve()).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def validate_r20_reference(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    parent_manifest = project_path(root, config["source"]["parent_manifest_path"], must_exist=True)
    return {
        "path": parent_manifest.relative_to(root).as_posix(),
        "sha256": sha256_file(parent_manifest),
        "exists": True,
        "read_only_reference": True,
    }


__all__ = [
    "R01_CONFIG_SCHEMA",
    "R01_EXPERIMENT_ID",
    "R01_LAYOUT_CANDIDATES",
    "R01_PROTECTED_LABELS",
    "R01_SCHEMA_VERSION",
    "R01_TARGET_LABELS",
    "R01_X5A_FIELD_NAMES",
    "artifact_path",
    "canonical_sha256",
    "hash_tree",
    "load_config",
    "project_path",
    "r01_root",
    "sha256_file",
    "validate_r20_reference",
]
