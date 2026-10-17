"""Prepare V24's independently trained course-reason-aware KnownGate baseline.

V24 consumes the frozen V21e causal-course cache but deliberately does not
load, fine-tune, or otherwise derive weights from a V21e trained checkpoint.
Its fresh random state is constructed through the V21 seed43 baseline config,
then the one course evidence fusion column is zero initialized.  This makes it
the clean paired control for V24a--V24c.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np



SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
CORE_CONTRACT = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-data-contract.json"
)
CORE_NORMALIZATION = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-only-train-normalization-stats.json"
)
V21E_TRAIN_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-train-causal-course-consistency-cache.npz"
)
V21E_VALIDATION_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-val-causal-course-consistency-cache.npz"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-baseline-"
    "seed43-config.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-"
    "baseline-seed43-formal"
)

TRAIN_PARENT_COUNT = 986
VALIDATION_PARENT_COUNT = 232
VALIDATION_CLUTTER_PARENT_SUPPORT = 9
EXPECTED_CACHE_EVENT_COUNTS = {"train": 18923, "val": 4452}










def _cache_metadata(path: Path, *, split: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V24 {split} course cache fields differ")
        sample_ids = np.asarray(cache["sample_ids"])
        values = np.asarray(cache["values"])
        valid = np.asarray(cache["valid"])
    expected_events = EXPECTED_CACHE_EVENT_COUNTS[split]
    if (
        sample_ids.shape != (expected_events,)
        or sample_ids.dtype.kind not in {"U", "S"}
        or len({str(value) for value in sample_ids.tolist()}) != expected_events
        or values.shape != (expected_events, 1)
        or values.dtype != np.float32
        or valid.shape != values.shape
        or valid.dtype != np.bool_
        or not np.isfinite(values).all()
        or bool((values < 0.0).any())
        or bool((values > 1.0).any())
        or bool((values[~valid] != 0.0).any())
    ):
        raise ValueError(f"V24 {split} frozen course cache is malformed")
    return {
        "event_count": expected_events,
        "feature_width": 1,
        "valid_value_count": int(valid.sum()),
        "invalid_value_count": int((~valid).sum()),
        "valid_value_fraction": float(valid.mean()),
    }











