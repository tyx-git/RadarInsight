from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .runtime import preflight_npz_archive


MICRO_DOPPLER_NAMES = (
    "peak_band_energy_time_cv_clipped_0_2_over_2",
    "minus_10db_velocity_spread_ratio",
    "modulation_period_seconds_clipped_0_1s_over_0_1s",
)
MICRO_DOPPLER_VALUE_DIM = len(MICRO_DOPPLER_NAMES)



































def load_micro_doppler_cache(
    path: str | Path,
    *,
    expected_sample_ids: Iterable[str] | None = None,
    expected_value_dim: int = MICRO_DOPPLER_VALUE_DIM,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = Path(path)
    preflight_npz_archive(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {"sample_ids", "values", "valid"}
        if set(payload.files) != required:
            raise ValueError(f"micro-Doppler cache fields must be exactly {sorted(required)}")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("micro-Doppler cache sample_ids must be a one-dimensional string array")
    identifiers = [str(value) for value in sample_ids.tolist()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("micro-Doppler cache contains duplicate sample_ids")
    expected_dim = int(expected_value_dim)
    if expected_dim <= 0:
        raise ValueError("micro-Doppler cache expected value dimension must be positive")
    if values.shape != (len(identifiers), expected_dim) or values.dtype != np.float32:
        raise ValueError(
            "micro-Doppler cache values must be float32 with the configured shape: "
            f"expected={[len(identifiers), expected_dim]}, actual={list(values.shape)}"
        )
    if valid.shape != (len(identifiers),) or valid.dtype != np.bool_:
        raise ValueError("micro-Doppler cache valid must be bool with shape [N]")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("micro-Doppler cache values must be finite and bounded to [0,1]")
    if np.any(values[~valid] != 0.0):
        raise ValueError("invalid micro-Doppler cache rows must be exactly zero")
    if expected_sample_ids is not None:
        expected = [str(value) for value in expected_sample_ids]
        if len(expected) != len(set(expected)):
            raise ValueError("expected micro-Doppler sample_ids contain duplicates")
        missing = sorted(set(expected) - set(identifiers))
        extra = sorted(set(identifiers) - set(expected))
        if missing or extra or len(expected) != len(identifiers):
            raise ValueError(
                "micro-Doppler cache sample set does not exactly match the Dataset split: "
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




__all__ = ["MICRO_DOPPLER_NAMES", "MICRO_DOPPLER_VALUE_DIM", "load_micro_doppler_cache"]