from __future__ import annotations

"""Deterministic content fingerprints for track-like training samples.

Fingerprints intentionally cover only numerical model inputs, not sample IDs,
labels, source paths, or other provenance.  They are used to keep semantic
copies in one split component and must never be used as model features.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import TRACK_FEATURE_COLUMNS


TRACK_CONTENT_FINGERPRINT_ALGORITHM = "sha256-track-model-inputs-v1"
FILE_CONTENT_FINGERPRINT_ALGORITHM = "sha256-file-bytes-v1"


def _canonical_float64(values: np.ndarray) -> np.ndarray:
    """Return a contiguous little-endian array with one canonical NaN payload."""
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=np.nan, posinf=np.inf, neginf=-np.inf)
    return np.ascontiguousarray(array.astype("<f8", copy=False))


def track_content_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered numerical track inputs while excluding identity and labels."""
    missing = sorted(set(TRACK_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"track content fingerprint missing columns: {missing}")
    numeric = frame[TRACK_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    array = _canonical_float64(numeric.to_numpy(dtype=np.float64, copy=True))
    digest = hashlib.sha256()
    digest.update(TRACK_CONTENT_FINGERPRINT_ALGORITHM.encode("utf-8"))
    digest.update(len(TRACK_FEATURE_COLUMNS).to_bytes(4, "little"))
    for column in TRACK_FEATURE_COLUMNS:
        encoded = column.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    digest.update(array.shape[0].to_bytes(8, "little"))
    digest.update(array.shape[1].to_bytes(4, "little"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_content_fingerprint(path: str | Path) -> str:
    """Hash a file's raw bytes in bounded chunks."""
    resolved = Path(path)
    digest = hashlib.sha256()
    digest.update(FILE_CONTENT_FINGERPRINT_ALGORITHM.encode("utf-8"))
    with resolved.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FILE_CONTENT_FINGERPRINT_ALGORITHM",
    "TRACK_CONTENT_FINGERPRINT_ALGORITHM",
    "file_content_fingerprint",
    "track_content_fingerprint",
]
