"""Read-only mixed-label L2 Dataset helpers for V26b routing audits.

The V18r DBB specialist was trained only on Drone/Bird/Balloon.  This module
does not change that fact: it merely lets a frozen checkpoint run on a mixed
validation population after a separately versioned, label-free current-event
feature cache has been built.  Its outputs are raw routes, never a calibrated
open-set decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from .data import CQ08PointSequenceDataset, PointSequence, build_point_dataloader
from .known_gate import KNOWN_LABELS, UNKNOWN_LABEL
from .micro_doppler_broadband import BROADBAND_MICRO_DOPPLER_VALUE_DIM
from .rd_physical import RD_PHYSICAL_VALUE_DIM
from .task_models import LiquidTaskConfig, resolve_project_path


MIXED_OPEN_SET_LABELS = (*KNOWN_LABELS, UNKNOWN_LABEL)


class FrozenL2MixedAuditDataset(Dataset[PointSequence]):
    """Mixed-label view using L2's own frozen normalization and feature cache."""

    def __init__(
        self,
        project_root: Path,
        config: LiquidTaskConfig,
        *,
        split_name: str,
        rd_physical_cache_path: Path,
        micro_doppler_cache_path: Path,
        parent_ids: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config = config
        if config.task.name != "dbb_specialist":
            raise ValueError("mixed L2 audit requires a DBB-specialist configuration")
        if config.model.rd_physical_dim != RD_PHYSICAL_VALUE_DIM:
            raise ValueError("mixed L2 audit requires the frozen four RD physical inputs")
        if config.model.micro_doppler_dim != BROADBAND_MICRO_DOPPLER_VALUE_DIM:
            raise ValueError("mixed L2 audit requires the frozen stable8 micro-Doppler inputs")
        self.base = CQ08PointSequenceDataset(
            self.project_root,
            index_path=resolve_project_path(self.project_root, config.data.index_path),
            split_name=split_name,
            max_sequence_length=config.data.max_sequence_length,
            max_rd_frames_per_point=config.data.max_rd_frames_per_point,
            rd_amplitude_center=config.data.rd_amplitude_center,
            rd_amplitude_scale=config.data.rd_amplitude_scale,
            exclude_anomalous_parents=config.data.exclude_anomalous_parents,
            expected_schema_version=config.data.index_schema_version,
            rd_physical_cache_path=rd_physical_cache_path,
            micro_doppler_cache_path=micro_doppler_cache_path,
            micro_doppler_value_dim=BROADBAND_MICRO_DOPPLER_VALUE_DIM,
            allowed_labels=MIXED_OPEN_SET_LABELS,
            forbidden_labels=(),
            allowed_parent_ids=parent_ids,
            # Keep one normalized RD archive resident at a time.  This bounds
            # the audit process independently of the number of parents.
            tensor_cache_size=1,
        )
        self.parent_ids = list(self.base.parent_ids)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> PointSequence:
        return self.base[index]

    def label_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in MIXED_OPEN_SET_LABELS}
        for parent_id in self.parent_ids:
            label = str(self.base._groups[str(parent_id)].iloc[0]["label_name"])
            counts[label] += 1
        return counts


def build_frozen_l2_mixed_audit_dataloader(
    dataset: FrozenL2MixedAuditDataset,
    *,
    batch_size: int,
) -> DataLoader:
    return build_point_dataloader(dataset.base, batch_size=max(1, int(batch_size)), shuffle=False)


def verify_l1_l2_event_alignment(l1_base: CQ08PointSequenceDataset, l2_base: CQ08PointSequenceDataset) -> dict[str, int]:
    """Require exact parent/event keys before combining independent forwards."""

    l1_parents = [str(value) for value in l1_base.parent_ids]
    l2_parents = [str(value) for value in l2_base.parent_ids]
    if l1_parents != l2_parents:
        raise ValueError("L1/L2 mixed audit parent order or set differs")
    event_count = 0
    for parent_id in l1_parents:
        left = l1_base._groups[parent_id]
        right = l2_base._groups[parent_id]
        fields = ("sample_id", "causal_event_index", "label_name")
        for field_index, field in enumerate(fields):
            left_values = left[field].astype(str).tolist()
            right_values = right[field].astype(str).tolist()
            if left_values != right_values:
                raise ValueError(
                    f"L1/L2 mixed audit event alignment differs for parent={parent_id}; field={field}"
                )
            # All three fields must have equal sequence length after the exact
            # equality check above.  Count it once from sample_id rather than
            # relying on a dataframe implementation detail.
            if field_index == 0:
                event_count += len(left_values)
    return {"parent_count": len(l1_parents), "event_count": event_count}


def l2_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Return only the frozen DBB model inputs from a mixed audit batch."""

    required = (
        "tr_features",
        "tr_missing_mask",
        "tr_dt_seconds",
        "rd_frames",
        "rd_frame_mask",
        "rd_quality",
        "tr_padding_mask",
        "tr_dt_valid",
        "tr_dt_reason",
        "rd_physical_features",
        "rd_physical_valid",
        "micro_doppler_features",
        "micro_doppler_valid",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"mixed L2 audit batch lacks required inputs: {missing}")
    return {name: batch[name] for name in required}


__all__ = ["FrozenL2MixedAuditDataset", "MIXED_OPEN_SET_LABELS", "build_frozen_l2_mixed_audit_dataloader", "l2_model_inputs", "verify_l1_l2_event_alignment"]