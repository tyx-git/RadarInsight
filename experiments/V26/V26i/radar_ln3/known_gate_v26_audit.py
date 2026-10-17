"""Audit-only mixed Known/Unknown datasets for V26.

This module intentionally does not expose a training collate path.  It maps
UnknownTarget to ``-1`` only in audit batches, never to a model class, and the
report filters that truth value before computing KnownGate metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CQ08PointSequenceDataset, collate_point_sequences
from .known_gate import (
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    KnownGateConfig,
    KnownGateSequence,
    _resolve,
)
from .known_gate_features import (
    V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
    V26J_RD_CAUSAL_GRU_VARIANT,
    load_known_gate_feature_cache,
)


class V26OpenSetAuditDataset(Dataset[KnownGateSequence]):
    """Parent-level mixed split dataset used only after checkpoint freezing."""

    def __init__(self, project_root: Path, config: KnownGateConfig, *, split_name: str, cache_path: Path) -> None:
        self.project_root = project_root.resolve()
        self.config = config
        self.base = CQ08PointSequenceDataset(
            self.project_root,
            index_path=_resolve(self.project_root, config.data.index_path),
            split_name=split_name,
            max_sequence_length=config.data.max_sequence_length,
            max_rd_frames_per_point=config.data.max_rd_frames_per_point,
            rd_amplitude_center=config.data.rd_amplitude_center,
            rd_amplitude_scale=config.data.rd_amplitude_scale,
            exclude_anomalous_parents=False,
            expected_schema_version=config.data.index_schema_version,
            allowed_labels=(*KNOWN_LABELS, UNKNOWN_LABEL),
            forbidden_labels=(),
            # The audit is deliberately parent-serial at the Dataset layer;
            # keeping one RD archive resident bounds peak memory while the
            # mixed validation/test pass is materialized after checkpoint
            # freezing.
            tensor_cache_size=1,
        )
        # V26j changes the L1 RD aggregation only.  It deliberately reuses the
        # same one-column causal-course cache, so the audit surface remains
        # limited to the two strict-open-set KnownGate variants rather than
        # becoming a generic mixed-data loader for arbitrary experiments.
        if (
            config.model.evidence_dim != 1
            or config.variant
            not in {
                V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
                V26J_RD_CAUSAL_GRU_VARIANT,
            }
        ):
            raise ValueError(
                "V26 mixed audit requires V26/V26j one-column causal-course evidence"
            )
        values, valid, metadata = load_known_gate_feature_cache(
            cache_path,
            expected_sample_ids=self.base.frame["sample_id"].astype(str).tolist(),
            variant=config.variant,
        )
        self._values = values
        self._valid = valid
        self.cache_metadata = metadata
        self.parent_ids = list(self.base.parent_ids)
        self._base_indices = {
            str(parent_id): index for index, parent_id in enumerate(self.base.parent_ids)
        }

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> KnownGateSequence:
        parent_id = self.parent_ids[int(index)]
        base_item = self.base[self._base_indices[parent_id]]
        sample_ids = self.base._groups[parent_id]["sample_id"].astype(str).tolist()
        values = torch.from_numpy(
            np.stack([self._values[sample_id] for sample_id in sample_ids], axis=0).astype(np.float32)
        )
        valid = torch.from_numpy(
            np.stack([self._valid[sample_id] for sample_id in sample_ids], axis=0).astype(np.bool_)
        )
        if values.shape != (base_item.event_count, 1) or valid.shape != values.shape:
            raise ValueError("V26 audit feature cache shape differs from its causal sequence")
        return KnownGateSequence(base=base_item, evidence_features=values, evidence_valid=valid)

    def label_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in (*KNOWN_LABELS, UNKNOWN_LABEL)}
        for parent_id in self.parent_ids:
            label = str(self.base._groups[parent_id].iloc[0]["label_name"])
            counts[label] += 1
        return counts


def collate_v26_open_set_audit(items: list[KnownGateSequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty V26 audit batch")
    batch = collate_point_sequences([item.base for item in items])
    max_length = int(batch["tr_features"].shape[1])
    evidence = torch.zeros((len(items), max_length, 1), dtype=torch.float32)
    valid = torch.zeros((len(items), max_length, 1), dtype=torch.bool)
    labels: list[int] = []
    task_label_names: list[str] = []
    for batch_index, item in enumerate(items):
        length = int(item.base.event_count)
        if item.evidence_features.shape != (length, 1) or item.evidence_valid.shape != (length, 1):
            raise ValueError("V26 audit batch contains incompatible evidence shapes")
        evidence[batch_index, :length] = item.evidence_features
        valid[batch_index, :length] = item.evidence_valid
        label_name = str(item.base.label_name)
        labels.append(-1 if label_name == UNKNOWN_LABEL else KNOWN_GATE_LABEL_TO_ID[label_name])
        task_label_names.append(
            "UNKNOWN_OPEN_SET_TRUTH" if label_name == UNKNOWN_LABEL else KNOWN_GATE_CLASS_NAMES[labels[-1]]
        )
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    batch["task_label_names"] = task_label_names
    batch["base_label_names"] = [item.base.label_name for item in items]
    if bool(valid.masked_select(batch["tr_padding_mask"].unsqueeze(-1)).any()):
        raise ValueError("padded V26 audit events must not contain valid evidence")
    batch["known_gate_features"] = evidence
    batch["known_gate_feature_valid"] = valid
    return batch




__all__ = ["V26OpenSetAuditDataset", "collate_v26_open_set_audit"]