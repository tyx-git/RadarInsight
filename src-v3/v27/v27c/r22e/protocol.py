"""Immutable provenance contract for the V27c-r22 evaluation correction.

``r22e`` is deliberately a derived evaluator, not a new model experiment.  It
never constructs an optimizer, changes an r22 checkpoint, or writes below the
frozen r22 directory.  Its only purpose is to evaluate the original r22
checkpoints after an installed NumPy 2 compatibility issue stopped r22's
original evaluator before it could emit a report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)
from v27.v27c.r22.protocol import (
    R22_MANIFEST_SCHEMA_VERSION,
    R22Context,
    load_r22_context,
    read_sealed_json as read_r22_sealed_json,
)


R22E_EXPERIMENT_ID = "V27c-r22e"
R22E_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27c-r22e-evaluation-manifest-v1"
R22E_REPORT_SCHEMA_VERSION = "RADAR-LN3-V27c-r22e-paired-validation-report-v1"
R22E_SUMMARY_SCHEMA_VERSION = "RADAR-LN3-V27c-r22e-three-seed-summary-v1"

_R22E_ROOT = Path("src-v3") / "v27" / "v27c" / "r22e"
_R22E_MANIFEST = _R22E_ROOT / "manifests" / "V27c-r22e-evaluation-manifest.json"
_R22E_REPORT_TEMPLATE = _R22E_ROOT / "reports" / "V27c-r22e-seed{seed}-paired-validation.json"
_R22E_SUMMARY = _R22E_ROOT / "reports" / "V27c-r22e-three-seed-summary.json"


@dataclass(frozen=True)
class R22EContext:
    """One validated r22e execution context bound to the frozen r22 round."""

    project_root: Path
    upstream: R22Context
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _artifact_path(project_root: str | Path, value: Path) -> Path:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, value)
    scope = (root / _R22E_ROOT).resolve()
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27c-r22e artifacts must remain inside its own directory") from exc
    return target


def r22e_manifest_path(project_root: str | Path) -> Path:
    return _artifact_path(project_root, _R22E_MANIFEST)


def r22e_report_path(project_root: str | Path, *, seed: int) -> Path:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27c-r22e report seed must be a non-negative integer")
    return _artifact_path(project_root, Path(str(_R22E_REPORT_TEMPLATE).format(seed=int(seed))))


def r22e_summary_path(project_root: str | Path) -> Path:
    return _artifact_path(project_root, _R22E_SUMMARY)


def _source_paths() -> tuple[Path, ...]:
    return (
        _R22E_ROOT / "__init__.py",
        _R22E_ROOT / "protocol.py",
        _R22E_ROOT / "evaluation.py",
        _R22E_ROOT / "cli.py",
    )


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27c-r22e payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json_if_identical(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create an immutable r22e artifact or reuse the same semantic payload."""

    text = json_text(_seal(payload), indent=2)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V27c-r22e artifact: {path}")
        return path
    atomic_write_text(path, text)
    return path


def read_sealed_json(path: Path, *, schema: str) -> dict[str, Any]:
    """Read and validate one r22e immutable JSON artifact."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"V27c-r22e artifact is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r22e artifact root must be an object")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("schema_version") != schema or payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("V27c-r22e artifact schema or payload hash differs")
    return dict(payload)


def build_r22e_manifest(project_root: str | Path, upstream: R22Context) -> dict[str, Any]:
    """Capture this evaluator and the frozen r22 source manifest it consumes."""

    root = Path(project_root).resolve()
    source_hashes: dict[str, str] = {}
    for path in _source_paths():
        target = root / path
        if not target.is_file():
            raise FileNotFoundError(f"V27c-r22e source is missing: {target}")
        source_hashes[path.as_posix()] = sha256_file(target)
    upstream_manifest = upstream.manifest
    if upstream_manifest.get("schema_version") != R22_MANIFEST_SCHEMA_VERSION:
        raise ValueError("V27c-r22e upstream r22 manifest schema differs")
    return {
        "schema_version": R22E_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R22E_EXPERIMENT_ID,
        "kind": "post_training_evaluation_correction_only",
        "upstream": {
            "experiment_id": upstream.config.experiment_id,
            "manifest_path": relative_project_path(root, upstream.manifest_path),
            "manifest_file_sha256": upstream.manifest_sha256,
            "manifest_payload_sha256": upstream_manifest["payload_sha256"],
            "manifest_schema_version": upstream_manifest["schema_version"],
            "source_hashes": upstream_manifest["code_hashes"],
        },
        "correction": {
            "failure": "numpy.ndarray.amax instance method is unavailable in the installed NumPy 2 runtime",
            "semantic_replacement": "np.abs(values).amax(axis=-1) -> np.max(np.abs(values), axis=-1)",
            "training_or_checkpoint_change": False,
            "reporting_improvement": "record source motion coverage separately from effective motion actually supplied to the arm",
        },
        "scope": {
            "known_only_validation": True,
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
        "evaluator_code_hashes": source_hashes,
    }


def prepare_r22e_manifest(project_root: str | Path) -> tuple[R22EContext, Path]:
    """Write the evaluator manifest before emitting any derived report."""

    root = Path(project_root).resolve()
    upstream = load_r22_context(root)
    path = r22e_manifest_path(root)
    payload = build_r22e_manifest(root, upstream)
    write_sealed_json_if_identical(path, payload)
    manifest = read_sealed_json(path, schema=R22E_MANIFEST_SCHEMA_VERSION)
    return (
        R22EContext(
            project_root=root,
            upstream=upstream,
            manifest=manifest,
            manifest_path=path,
            manifest_sha256=sha256_file(path),
        ),
        path,
    )


def load_r22e_context(project_root: str | Path) -> R22EContext:
    """Load only when both frozen r22 and r22e provenance are current."""

    root = Path(project_root).resolve()
    upstream = load_r22_context(root)
    path = r22e_manifest_path(root)
    manifest = read_sealed_json(path, schema=R22E_MANIFEST_SCHEMA_VERSION)
    expected = build_r22e_manifest(root, upstream)
    body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if body != expected:
        raise ValueError("V27c-r22e manifest differs from the current evaluator or frozen r22")
    return R22EContext(
        project_root=root,
        upstream=upstream,
        manifest=manifest,
        manifest_path=path,
        manifest_sha256=sha256_file(path),
    )


__all__ = [
    "R22EContext",
    "R22E_EXPERIMENT_ID",
    "R22E_MANIFEST_SCHEMA_VERSION",
    "R22E_REPORT_SCHEMA_VERSION",
    "R22E_SUMMARY_SCHEMA_VERSION",
    "build_r22e_manifest",
    "load_r22e_context",
    "prepare_r22e_manifest",
    "r22e_manifest_path",
    "r22e_report_path",
    "r22e_summary_path",
    "read_sealed_json",
    "write_sealed_json_if_identical",
]

