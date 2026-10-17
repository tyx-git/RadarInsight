from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .runtime import preflight_npz_archive


RD_PHYSICAL_NAMES = (
    "velocity_bandwidth_mps_over_85",
    "zero_velocity_energy_fraction_abs_le_2_mps",
    "velocity_spectral_entropy_over_log_384",
    "top10_velocity_energy_fraction",
)
RD_PHYSICAL_VALUE_DIM = len(RD_PHYSICAL_NAMES)












def load_rd_physical_cache(
    path: str | Path,
    *,
    expected_sample_ids: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = Path(path)
    preflight_npz_archive(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {"sample_ids", "values", "valid"}
        if set(payload.files) != required:
            raise ValueError(f"RD physical cache fields must be exactly {sorted(required)}: {cache_path}")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("RD physical cache sample_ids must be a one-dimensional string array")
    identifiers = [str(value) for value in sample_ids.tolist()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("RD physical cache contains duplicate sample_ids")
    if values.shape != (len(identifiers), RD_PHYSICAL_VALUE_DIM) or values.dtype != np.float32:
        raise ValueError("RD physical cache values must be float32 with shape [N,4]")
    if valid.shape != (len(identifiers),) or valid.dtype != np.bool_:
        raise ValueError("RD physical cache valid must be bool with shape [N]")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("RD physical cache values must be finite and bounded to [0,1]")
    if np.any(values[~valid] != 0.0):
        raise ValueError("invalid RD physical cache rows must be exactly zero")
    if expected_sample_ids is not None:
        expected = [str(value) for value in expected_sample_ids]
        if len(expected) != len(set(expected)):
            raise ValueError("expected RD physical sample_ids contain duplicates")
        missing = sorted(set(expected) - set(identifiers))
        extra = sorted(set(identifiers) - set(expected))
        if missing or extra or len(expected) != len(identifiers):
            raise ValueError(
                "RD physical cache sample set does not exactly match the Dataset split: "
                f"missing={missing[:5]}, extra={extra[:5]}, expected={len(expected)}, actual={len(identifiers)}"
            )
    mapping = {sample_id: values[index] for index, sample_id in enumerate(identifiers)}
    valid_mapping = {sample_id: bool(valid[index]) for index, sample_id in enumerate(identifiers)}
    return mapping, {
        "path": str(cache_path),
        "event_count": len(identifiers),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_by_sample_id": valid_mapping,
    }


__all__ = ["RD_PHYSICAL_NAMES", "RD_PHYSICAL_VALUE_DIM", "load_rd_physical_cache"]