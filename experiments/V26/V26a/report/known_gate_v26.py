"""Evaluate V26 known-only metrics and the post-checkpoint open-set audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v26 import EXPECTED_PARENT_COUNTS
from radar_ln3.known_gate import KNOWN_GATE_LABEL_TO_ID, UNKNOWN_LABEL, sha256_file, summarize_known_gate_records, summarize_parent_balanced_eligible_prefix_records
from radar_ln3.known_gate_v26_audit import (
    V26OpenSetAuditDataset,
    build_v26_open_set_audit_dataloader,
)
from train.known_gate_common import _model_inputs






def _metric_rows(rows: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any] | None:
    return summarize_known_gate_records(rows, scope=scope) if rows else None


def _known_id(label_name: str) -> int:
    if label_name not in KNOWN_GATE_LABEL_TO_ID:
        raise ValueError(f"V26 audit attempted to score a non-known label: {label_name}")
    return int(KNOWN_GATE_LABEL_TO_ID[label_name])


def _record(
    *,
    parent_id: str,
    label_name: str,
    track_id: str,
    source_name: str,
    capture_id: str,
    provenance_component_id: str,
    view_name: str,
    event_index: int,
    parent_length: int,
    probability: torch.Tensor,
    rd_frame_count: int,
    minimum_evidence_events: int,
) -> dict[str, Any]:
    pred_id = int(probability.argmax().item())
    return {
        "parent_sample_id": parent_id,
        "label_name": label_name,
        "true_id": _known_id(label_name) if label_name in KNOWN_GATE_LABEL_TO_ID else -1,
        "pred_id": pred_id,
        "known_target_probability": float(probability[0].item()),
        "known_clutter_probability": float(probability[1].item()),
        "confidence": float(probability[pred_id].item()),
        "track_id": track_id,
        "source_name": source_name,
        "capture_id": capture_id,
        "provenance_component_id": provenance_component_id,
        "view_name": view_name,
        "event_index": int(event_index),
        "window_length": int(event_index + 1),
        "parent_length": int(parent_length),
        "product_action_eligible": bool(
            event_index + 1 >= minimum_evidence_events
        ),
        "rd_frame_count": int(rd_frame_count),
    }


def _run_split(
    model: torch.nn.Module,
    config: Any,
    root: Path,
    *,
    split: str,
    cache_path: Path,
    device: torch.device,
    minimum_evidence_events: int,
) -> dict[str, Any]:
    dataset = V26OpenSetAuditDataset(
        root,
        config,
        split_name=split,
        cache_path=cache_path,
    )
    expected_support = EXPECTED_PARENT_COUNTS[split]
    if dataset.label_counts() != expected_support:
        raise ValueError(
            f"V26 {split} mixed audit parent support changed: "
            f"expected={expected_support}, actual={dataset.label_counts()}"
        )
    loader = build_v26_open_set_audit_dataloader(
        dataset,
        batch_size=config.train.batch_size,
    )
    parent_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    unknown_parent_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        model.eval()
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(**_model_inputs(tensors))
            probabilities = torch.softmax(output["sequence_logits"], dim=-1)
            for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][batch_index].item())
                label_name = str(batch["base_label_names"][batch_index])
                group = dataset.base._groups[str(parent_id)]
                first = group.iloc[0]
                metadata = {
                    "track_id": str(first["track_id"]),
                    "source_name": str(first["source_name"]),
                    "capture_id": str(first.get("capture_id", "")),
                    "provenance_component_id": str(first.get("provenance_component_id", "")),
                    "view_name": str(first.get("view_name", "")),
                }
                parent_final: dict[str, Any] | None = None
                for event_index in range(length):
                    row = _record(
                        parent_id=str(parent_id),
                        label_name=label_name,
                        event_index=event_index,
                        parent_length=length,
                        probability=probabilities[batch_index, event_index],
                        rd_frame_count=int(
                            batch["rd_frame_mask"][batch_index, event_index].sum().item()
                        ),
                        minimum_evidence_events=minimum_evidence_events,
                        **metadata,
                    )
                    window_rows.append(row)
                    if row["product_action_eligible"] and event_index + 1 < length:
                        eligible_rows.append(dict(row))
                    parent_final = row
                if parent_final is None:
                    raise ValueError(f"V26 audit found an empty parent: {parent_id}")
                parent_rows.append(parent_final)
                if label_name == UNKNOWN_LABEL:
                    unknown_parent_rows.append(parent_final)
    known_parent = [row for row in parent_rows if row["true_id"] >= 0]
    known_window = [row for row in window_rows if row["true_id"] >= 0]
    known_eligible = [row for row in eligible_rows if row["true_id"] >= 0]
    unknown_routes = {
        "Unknown_to_KnownTarget": sum(row["pred_id"] == 0 for row in unknown_parent_rows),
        "Unknown_to_KnownClutter": sum(row["pred_id"] == 1 for row in unknown_parent_rows),
    }
    unknown_false_accept_parent_count = int(sum(unknown_routes.values()))
    unknown_parent_support = len(unknown_parent_rows)
    confidence = np.asarray(
        [float(row["confidence"]) for row in unknown_parent_rows], dtype=np.float64
    )
    return {
        "split": split,
        "parent_count": len(parent_rows),
        "known_parent_count": len(known_parent),
        "unknown_parent_count": len(unknown_parent_rows),
        "label_counts": dataset.label_counts(),
        "known_metrics": {
            "parent_final": _metric_rows(known_parent, scope="known_parent_final"),
            "eligible_prefix": _metric_rows(known_eligible, scope="known_eligible_prefix"),
            "parent_balanced_eligible_prefix": (
                summarize_parent_balanced_eligible_prefix_records(
                    known_eligible,
                    scope="known_parent_balanced_eligible_prefix",
                )
                if known_eligible
                else None
            ),
            "window": _metric_rows(known_window, scope="known_causal_windows"),
        },
        "open_set_diagnostics": {
            "parent_final_routes": unknown_routes,
            "unknown_false_accept_rate": (
                float(unknown_false_accept_parent_count / unknown_parent_support)
                if unknown_parent_support
                else None
            ),
            "unknown_false_accept_parent_count": unknown_false_accept_parent_count,
            "unknown_rejection_available": False,
            "unknown_rejected_parent_count": 0,
            "max_known_confidence": {
                "support": int(confidence.size),
                "mean": float(confidence.mean()) if confidence.size else None,
                "median": float(np.median(confidence)) if confidence.size else None,
                "minimum": float(confidence.min()) if confidence.size else None,
                "maximum": float(confidence.max()) if confidence.size else None,
            },
            "pending_abstain_available": False,
            "events_before_minimum_evidence": sum(
                not bool(row["product_action_eligible"]) for row in window_rows
            ),
        },
        "clutter_support": {
            "parent_final": sum(
                row["label_name"] == "ClutterTarget" for row in known_parent
            ),
            "eligible_prefix_parent_support": sum(
                row["label_name"] == "ClutterTarget" for row in known_eligible
            ),
        },
        "records": {
            "parent": parent_rows,
            "eligible_prefix": eligible_rows,
            "window": window_rows,
            "unknown_parent_final": unknown_parent_rows,
        },
        "audit_cache_sha256": sha256_file(cache_path),
        "audit_cache_metadata": dataset.cache_metadata,
    }









