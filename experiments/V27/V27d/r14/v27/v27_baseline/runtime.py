"""Shared execution helpers for the sealed V27 baseline.

This module is intentionally small and policy-free beyond the V27 protocol:
it is the single place where an entry point resolves the verified context,
chooses the configured device, seeds the three RNG families, and writes a
mutable JSON artifact.  It never opens test or Unknown payloads.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch






def seed_v27(seed: int, *, deterministic_cuda: bool = False) -> None:
    """Seed Python, NumPy and Torch without changing the configured policy."""

    value = int(seed)
    if value < 0:
        raise ValueError("V27 seed must be non-negative")
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if deterministic_cuda:
        # The flag is explicit in the manifest.  Failures are preferable to a
        # silently nondeterministic baseline when a future operator enables it.
        torch.use_deterministic_algorithms(True, warn_only=False)






















def environment_snapshot() -> dict[str, Any]:
    """Return only deterministic runtime facts useful in a training summary."""

    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


__all__ = ["environment_snapshot", "seed_v27"]
