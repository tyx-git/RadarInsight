"""Prepare V24's independently trained course-reason-aware KnownGate baseline.

V24 consumes the frozen V21e causal-course cache but deliberately does not
load, fine-tune, or otherwise derive weights from a V21e trained checkpoint.
Its fresh random state is constructed through the V21 seed43 baseline config,
then the one course evidence fusion column is zero initialized.  This makes it
the clean paired control for V24a--V24c.
"""

from __future__ import annotations

from pathlib import Path




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





















