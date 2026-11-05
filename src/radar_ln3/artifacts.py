"""Artifact root preparation and explicit overwrite handling."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import zipfile


DEFAULT_ARTIFACT_ROOT = Path("artifacts/radar_ln3")


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


__all__ = ["DEFAULT_ARTIFACT_ROOT", "prepare_artifact_directory", "prepare_artifact_stage"]
