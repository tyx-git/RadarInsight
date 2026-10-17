"""r20 views over the sealed r19 TR-only population.

The ordinary TR encoder no longer receives the radial-speed value in either
arm.  The treatment arm receives a separate causal sidecar with four radial
descriptors and four validity flags; the control arm receives an exact zero
sidecar of the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from radar_ln3.features import encode_tr_feature_values
from v27.V27d.r17.data import R17Sequence, _event_values, _l2_robust_values, collate_r17
from v27.V27d.r19.data import R19Dataset

from .protocol import (
    R20_ARMS,
    R20_RADIAL_VALUE_DIM,
    R20Config,
)


@dataclass(frozen=True)
class R20Sequence:
    """One r20 parent and its arm-specific radial sidecar."""

    sequence: R17Sequence
    radial_values: torch.Tensor
    radial_valid: torch.Tensor


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _radial_sidecar(
    item: R17Sequence,
    config: R20Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal radial descriptors from current and previous TR events.

    Index meanings are fixed by the r20 protocol:

    0. ``abs(radial_speed_mps) / radial_abs_scale_mps``;
    1. signed radial direction in ``{-1, 0, +1}``;
    2. ``(radial_t - radial_(t-1)) / dt / radial_rate_scale``;
    3. ``sign(mean_radial) * sign(range_delta)``.

    The last two fields are valid only for a strictly positive, valid dt.  A
    missing radial/range value or an ambiguous predecessor makes the affected
    field invalid and leaves its value exactly zero.
    """

    values: list[dict[str, Any]] = []
    for row in item.event_rows[: item.event_count]:
        current = _event_values(row)
        current, _, _ = _l2_robust_values(current, config.r17)
        values.append(current)

    data = config.data
    absolute_scale = float(data["radial_abs_scale_mps"])
    rate_scale = float(data["radial_rate_scale_mps_per_s"])
    clip = float(data["sidecar_clip_absolute"])
    output = np.zeros((len(values), R20_RADIAL_VALUE_DIM), dtype=np.float32)
    valid = np.zeros((len(values), R20_RADIAL_VALUE_DIM), dtype=bool)

    previous_radial: float | None = None
    previous_range: float | None = None
    for index, (current, dt, dt_valid) in enumerate(
        zip(
            values,
            item.tr_dt_seconds.tolist(),
            item.tr_dt_valid.tolist(),
            strict=True,
        )
    ):
        radial = float(current["radial_speed_mps"]) if _finite(current.get("radial_speed_mps")) else None
        range_m = float(current["range_m"]) if _finite(current.get("range_m")) else None

        if radial is not None:
            output[index, 0] = np.float32(np.clip(abs(radial) / absolute_scale, 0.0, clip))
            output[index, 1] = np.float32(np.sign(radial))
            valid[index, 0:2] = True

        positive_dt = bool(dt_valid) and float(dt) > 0.0
        if positive_dt and radial is not None and previous_radial is not None:
            output[index, 2] = np.float32(
                np.clip((radial - previous_radial) / float(dt) / rate_scale, -clip, clip)
            )
            valid[index, 2] = True

        if (
            positive_dt
            and radial is not None
            and previous_radial is not None
            and range_m is not None
            and previous_range is not None
        ):
            mean_radial = 0.5 * (radial + previous_radial)
            range_delta = range_m - previous_range
            output[index, 3] = np.float32(np.sign(mean_radial) * np.sign(range_delta))
            valid[index, 3] = True

        previous_radial = radial
        previous_range = range_m

    output[~valid] = 0.0
    if not np.isfinite(output).all():
        raise ValueError("r20 radial sidecar produced a non-finite value")
    return output, valid


def _remove_raw_radial_scalar(item: R17Sequence, config: R20Config) -> R17Sequence:
    """Remove radial value from the ordinary TR encoder while preserving schema."""

    values: list[np.ndarray] = []
    missing: list[np.ndarray] = []
    for row in item.event_rows[: item.event_count]:
        current = _event_values(row)
        current, _, _ = _l2_robust_values(current, config.r17)
        current["radial_speed_mps"] = None
        encoded, encoded_missing = encode_tr_feature_values(current)
        values.append(encoded)
        missing.append(encoded_missing)
    transformed = replace(
        item,
        tr_features=torch.from_numpy(np.stack(values).astype(np.float32, copy=False)),
        tr_missing_mask=torch.from_numpy(np.stack(missing).astype(np.float32, copy=False)),
    )
    if transformed.tr_features.shape[-1] <= 1 or transformed.tr_missing_mask.shape[-1] <= 1:
        raise ValueError("r20 TR schema does not contain the registered radial channels")
    if bool(transformed.tr_features[..., 1].ne(0).any()):
        raise ValueError("r20 raw radial value was not removed")
    if not bool(transformed.tr_missing_mask[..., 1].eq(1).all()):
        raise ValueError("r20 raw radial missingness was not sealed")
    return transformed


class R20Dataset(Dataset[R20Sequence]):
    """Paired known-only r20 data view over r19's exact parent split."""

    def __init__(
        self,
        project_root: str | Path,
        config: R20Config,
        *,
        split_name: str,
        arm: str,
    ) -> None:
        if split_name not in {"train", "validation"}:
            raise ValueError("r20 split must be train or validation")
        if arm not in R20_ARMS:
            raise ValueError("r20 arm is invalid")
        root = Path(project_root).resolve()
        base = R19Dataset(
            root,
            config.r19,
            split_name=split_name,
            arm="snr_missing_only",
        )
        self.project_root = root
        self.config = config
        self.split_name = split_name
        self.arm = arm
        self.parent_ids = list(base.parent_ids)
        self._items: list[R20Sequence] = []
        for position in range(len(base)):
            original = base[position]
            sequence = _remove_raw_radial_scalar(original, config)
            sidecar, sidecar_valid = _radial_sidecar(original, config)
            if arm == "zero_control":
                sidecar.fill(0.0)
                sidecar_valid.fill(False)
            self._items.append(
                R20Sequence(
                    sequence=sequence,
                    radial_values=torch.from_numpy(sidecar.astype(np.float32, copy=False)),
                    radial_valid=torch.from_numpy(sidecar_valid.astype(bool, copy=False)),
                )
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> R20Sequence:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("r20 dataset index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError("r20 dataset index is out of range")
        return self._items[index]


def collate_r20(items: list[R20Sequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("r20 cannot collate an empty batch")
    base = collate_r17([item.sequence for item in items])
    width = int(base["tr_features"].shape[1])
    batch_size = len(items)
    radial = torch.zeros((batch_size, width, R20_RADIAL_VALUE_DIM), dtype=torch.float32)
    radial_valid = torch.zeros((batch_size, width, R20_RADIAL_VALUE_DIM), dtype=torch.bool)
    for row, item in enumerate(items):
        length = item.sequence.event_count
        if item.radial_values.shape != (length, R20_RADIAL_VALUE_DIM):
            raise ValueError("r20 radial sidecar value shape differs from parent length")
        if item.radial_valid.shape != item.radial_values.shape or item.radial_valid.dtype != torch.bool:
            raise ValueError("r20 radial sidecar validity shape or dtype differs")
        if bool(item.radial_values.masked_select(~item.radial_valid).ne(0).any()):
            raise ValueError("r20 invalid radial sidecar values must be zero")
        radial[row, :length] = item.radial_values
        radial_valid[row, :length] = item.radial_valid
    padding = base["tr_padding_mask"]
    if bool(radial.masked_select(padding.unsqueeze(-1)).ne(0).any()):
        raise ValueError("r20 padded radial sidecar values must be zero")
    if bool(radial_valid.masked_select(padding.unsqueeze(-1)).any()):
        raise ValueError("r20 padded radial sidecar validity must be false")
    base["radial_sidecar_values"] = radial
    base["radial_sidecar_valid"] = radial_valid
    return base






__all__ = ["R20Dataset", "R20Sequence", "collate_r20"]