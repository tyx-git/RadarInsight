from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np





























def _normalize_windows_acl(path: Path) -> None:
    if os.name != "nt":
        return
    completed = subprocess.run(
        ["icacls", str(path), "/reset"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PermissionError(f"failed to adopt parent ACL for V17 cache {path}: {detail}")


def _same_npz(path: Path, arrays: Sequence[tuple[str, np.ndarray]]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {name for name, _ in arrays}:
                return False
            return all(np.array_equal(np.asarray(payload[name]), array) for name, array in arrays)
    except (OSError, ValueError):
        return False


def _write_npz_new_or_identical(
    path: Path,
    arrays: Sequence[tuple[str, np.ndarray]],
) -> str:
    arrays = tuple((str(name), np.asarray(value)) for name, value in arrays)
    if path.exists():
        if not _same_npz(path, arrays):
            raise FileExistsError(f"refusing to overwrite a different versioned cache: {path}")
        _normalize_windows_acl(path)
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as target:
            created = True
            with zipfile.ZipFile(
                target, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:
                for name, array in arrays:
                    import io

                    stream = io.BytesIO()
                    np.lib.format.write_array(stream, array, allow_pickle=False)
                    info = zipfile.ZipInfo(
                        f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(
                        info,
                        stream.getvalue(),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9,
                    )
            target.flush()
            os.fsync(target.fileno())
        _normalize_windows_acl(path)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    if not _same_npz(path, arrays):
        path.unlink(missing_ok=True)
        raise ValueError(f"new V17 cache failed its immediate round-trip check: {path}")
    return "created"



















