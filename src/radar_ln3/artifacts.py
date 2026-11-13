"""Artifact root preparation, L3 output layout, and explicit overwrite handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import zipfile


DEFAULT_ARTIFACT_ROOT = Path("artifacts/radar_ln3")
L3_FEATURE_CACHE_ROOT = Path("data/cq08/cache/features/l3")


@dataclass(frozen=True)
class L3Layout:
    """L3 脚本共享的 Root 派生布局；字段一律是项目内相对路径。"""

    root: Path
    heads: Path
    l1_head: Path
    l2_head: Path
    l1_checkpoint: Path
    l2_checkpoint: Path
    representations: Path
    exploratory_calibration: Path
    calibration: Path
    audit_manifest: Path
    audit_report: Path
    audit_evidence: Path
    head_fit_feature_cache: Path
    validation_feature_cache: Path
    audit_feature_cache: Path


def artifact_root_relative(
    project_root: str | Path,
    value: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """把 Root 规范成项目内相对路径；越界直接拒绝。"""

    root = Path(project_root).resolve()
    candidate = Path(value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Root 必须位于项目目录内：{value}") from exc


def l3_layout(
    project_root: str | Path,
    root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> L3Layout:
    """返回 L3 正式产物与 L3 角色 feature cache 的固定相对布局。"""

    base = artifact_root_relative(project_root, root)
    l3 = base / "l3"
    cache = L3_FEATURE_CACHE_ROOT
    return L3Layout(
        root=base,
        heads=l3 / "heads",
        l1_head=l3 / "heads" / "l1-knownness.pt",
        l2_head=l3 / "heads" / "l2-knownness.pt",
        # endpoint checkpoint 属于 Root，而不是固定正式目录：非默认 Root 的
        # L3 链路必须绑定该 Root 下的 L1/L2 端点。
        l1_checkpoint=base / "l1" / "l1.pt",
        l2_checkpoint=base / "l2" / "l2.pt",
        representations=l3 / "representations",
        exploratory_calibration=l3 / "radar_ln3_calibration.exploratory.json",
        calibration=l3 / "radar_ln3_calibration.json",
        audit_manifest=l3 / "radar_ln3_audit_manifest.json",
        audit_report=l3 / "radar_ln3_audit_report.json",
        audit_evidence=l3 / "audit-evidence.jsonl",
        head_fit_feature_cache=cache / "head_fit" / "feature-cache-manifest.json",
        validation_feature_cache=cache / "validation" / "feature-cache-manifest.json",
        audit_feature_cache=cache / "audit" / "feature-cache-manifest.json",
    )


def is_default_artifact_root(root: str | Path) -> bool:
    """判断布局 Root 是否就是正式 Root；非正式 Root 不登记产品 manifest。"""

    return os.path.normcase(str(Path(root))) == os.path.normcase(
        str(DEFAULT_ARTIFACT_ROOT)
    )


def prepare_artifact_directory(path: str | Path, *, overwrite: bool) -> Path:
    """Create an empty artifact directory, archiving prior contents only on request."""

    target = Path(path).resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(target)
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"artifact root 目录非空：{target}；使用 --overwrite / -Overwrite 明确覆盖"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = target.with_name(f"{target.name}-backup-{stamp}.zip")
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            for item in target.rglob("*"):
                if item.is_file():
                    bundle.write(item, item.relative_to(target.parent))
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def prepare_artifact_stage(
    root: str | Path, stage: str, *, overwrite: bool
) -> Path:
    """Prepare just one stage directory, preserving the other cascade stages."""

    if stage not in {"l1", "l2", "l3"}:
        raise ValueError(f"unsupported artifact stage: {stage}")
    return prepare_artifact_directory(Path(root) / stage, overwrite=overwrite)


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "L3Layout",
    "artifact_root_relative",
    "is_default_artifact_root",
    "l3_layout",
    "prepare_artifact_directory",
    "prepare_artifact_stage",
]
