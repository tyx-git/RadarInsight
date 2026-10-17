"""Read-only V18 evidence-cache audit and V27a normalization fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


from .protocol import V27A_NORMALIZATION_SCHEMA_VERSION
























def load_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from .protocol import read_sealed_json

    payload = read_sealed_json(path, schema=V27A_NORMALIZATION_SCHEMA_VERSION)
    median = np.asarray(payload["median"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if median.shape != (8,) or scale.shape != (8,):
        raise ValueError("V27a normalization vectors must have shape [8]")
    if not np.isfinite(median).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("V27a normalization vectors must be finite and positive")
    return median, scale, payload




__all__ = ["load_normalization"]