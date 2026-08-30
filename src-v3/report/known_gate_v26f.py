"""Run the frozen V26f mixed Known/Unknown L1 -> L2 -> L3 audit.

V26f is deliberately inference-only.  It reuses the sealed V26e L1 epoch-32
fold checkpoints, the sealed V26c L2 epoch-15 fold checkpoints, and the
cross-fitted Known-only thresholds recorded by V26e.  No model, threshold,
cache, or normalization artifact is created by this module.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    build_known_gate_model_for_config,
    known_gate_config_identity_sha256,
    known_gate_device,
)
from radar_ln3.known_gate_v26_audit import (
    V26OpenSetAuditDataset,
    build_v26_open_set_audit_dataloader,
)
from radar_ln3.known_gate_v26f import (
    ACTION_NAMES,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONTRACT,
    CONTRACT_SCHEMA,
    DBB_LABELS,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    L1_EPOCH,
    L1_LABELS,
    L1_TEST_CACHE,
    L1_VALIDATION_CACHE,
    L2_EPOCH,
    L2_TEST_MICRO_CACHE,
    L2_TEST_PHYSICAL_CACHE,
    L2_VALIDATION_MICRO_CACHE,
    L2_VALIDATION_PHYSICAL_CACHE,
    MAX_DETAIL_BYTES,
    MAX_REPORT_BYTES,
    MIXED_LABELS,
    PARENT_DETAIL_CSV,
    READINESS,
    REPORT,
    REPORT_SCHEMA,
    V26E_CONTRACT,
    V26E_READINESS,
    V26E_REPORT,
    fold_contract_entry,
    relative_path,
    read_json,
    sha256_file,
    validate_prepared_contract,
    write_immutable_text,
)
from radar_ln3.l1_l2_raw_routing import (
    FrozenL2MixedAuditDataset,
    build_frozen_l2_mixed_audit_dataloader,
    l2_model_inputs,
    verify_l1_l2_event_alignment,
)
from radar_ln3.open_set_architecture import ProductAction, resolve_semantic_action
from radar_ln3.task_models import build_paired_model_for_config
from train.known_gate_v21 import _model_inputs as l1_model_inputs
from train.known_gate_v26c_oof import (
    _load_checkpoint as load_fold_checkpoint,
    _runtime_l2_config,
    _source_protocols,
)
from train.known_gate_v26e_oof import (
    _runtime_l1_config,
    _source_protocol,
)
from train.known_gate_v21 import _state_dict_sha256 as known_gate_state_sha256
from train.known_gate_v26c_oof import _state_dict_sha256 as task_state_sha256


ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
KNOWN_ACTION_INDEX = {name: index for index, name in enumerate((*DBB_LABELS, "ClutterTarget"))}
MIXED_LABEL_INDEX = {name: index for index, name in enumerate(MIXED_LABELS)}
KNOWN_LABEL_SET = frozenset(L1_LABELS)
UNKNOWN_LABEL = "UnknownTarget"
MINIMUM_EVIDENCE_EVENTS = 3
EPS = 1.0e-12


def _finite_probability(value: Any, *, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"V26f {field} is outside [0,1]")
    return result


def _read_thresholds(root: Path) -> dict[int, dict[str, float]]:
    report = read_json(root / V26E_REPORT, description="V26e cross-fitted threshold report")
    if (
        report.get("schema_version")
        != "RADAR-LN3-V26e-l1-epoch32-known-only-cross-fitted-l3-report-v1"
        or report.get("status")
        != "completed_v26e_fixed_epoch32_known_only_l3_diagnostic"
        or report.get("mixed_unknown_audit") != "not_executed"
        or report.get("deployment_eligible") is not False
    ):
        raise ValueError("V26f V26e threshold source is not a sealed Known-only diagnostic")
    rows = list(report.get("cross_fit") or ())
    if len(rows) != FOLD_COUNT:
        raise ValueError("V26f V26e threshold source does not contain five folds")
    result: dict[int, dict[str, float]] = {}
    required = {
        "l1_known_target_acceptance",
        "l1_known_clutter_acceptance",
        "l2_subclass_acceptance",
    }
    for raw in rows:
        row = dict(raw)
        fold_id = int(row.get("fold_id", -1))
        if fold_id in result or fold_id not in range(FOLD_COUNT):
            raise ValueError("V26f V26e threshold fold ids are repeated or invalid")
        thresholds = dict(row.get("thresholds") or {})
        if set(thresholds) != required:
            raise ValueError(f"V26f threshold fields differ for fold {fold_id}")
        values = {
            name: _finite_probability(thresholds[name], field=f"fold {fold_id} {name}")
            for name in required
        }
        if values["l1_known_target_acceptance"] + values["l1_known_clutter_acceptance"] <= 1.0:
            raise ValueError(f"V26f fold {fold_id} L1 thresholds permit dual acceptance")
        result[fold_id] = values
    if set(result) != set(range(FOLD_COUNT)):
        raise ValueError("V26f threshold source has incomplete fold coverage")
    return result


def _load_fold_models(
    root: Path,
    *,
    fold_id: int,
    prepared_contract: Mapping[str, Any],
) -> tuple[Any, Any, torch.device, dict[str, Any]]:
    """Construct fresh runtime models and load only the bound frozen snapshots."""

    v26e_contract = read_json(root / V26E_CONTRACT, description="V26e contract")
    v26c_contract = read_json(
        root / "data/processed/RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration-contract.json",
        description="V26c contract",
    )
    e_fold = next(
        row for row in list(v26e_contract.get("folds") or ()) if int(row["fold_id"]) == fold_id
    )
    c_fold = next(
        row for row in list(v26c_contract.get("folds") or ()) if int(row["fold_id"]) == fold_id
    )

    source_l1, source_l1_sha = _source_protocol(root)
    l1_config, l1_config_path, _ = _runtime_l1_config(
        root, fold_id, dict(e_fold.get("l1") or {})
    )
    l1_model, l1_reference, l1_pairing = build_known_gate_model_for_config(root, source_l1)
    del l1_reference, l1_pairing
    l1_model.config = l1_config
    l1_initial_sha = known_gate_state_sha256(l1_model.state_dict())
    l1_device = known_gate_device(l1_config)
    l1_model.to(l1_device)
    l1_checkpoint = root / str(dict(e_fold["l1"]).get("checkpoint_path") or "")
    # The canonical path is also stored in the V26f fold binding; use that
    # bound path rather than deriving an artifact name from the fold number.
    l1_checkpoint = root / next(
        item["path"]
        for item in list(fold_contract_entry(prepared_contract, fold_id)["source_files"])
        if item["name"] == "l1_checkpoint"
    )
    load_fold_checkpoint(
        l1_checkpoint,
        task="known_gate",
        schema=l1_config.train.checkpoint_schema,
        fold_id=fold_id,
        model=l1_model,
        optimizer=None,
        config_identity=known_gate_config_identity_sha256(l1_config),
        config_path=l1_config_path,
        contract_sha256=sha256_file(root / V26E_CONTRACT),
        readiness_sha256=sha256_file(root / V26E_READINESS),
        source_config_sha256=source_l1_sha,
        initial_state_sha256=l1_initial_sha,
        device=l1_device,
    )

    _source_l1_again, source_l2, _source_l1_sha_again, source_l2_sha = _source_protocols(root)
    l2_config, l2_config_path, _ = _runtime_l2_config(
        root, fold_id, dict(c_fold.get("l2") or {})
    )
    l2_model, l2_reference, l2_pairing = build_paired_model_for_config(root, source_l2)
    del l2_reference, l2_pairing
    l2_model.config = l2_config
    l2_initial_sha = task_state_sha256(l2_model.state_dict())
    l2_device = l1_device
    l2_model.to(l2_device)
    l2_checkpoint = root / next(
        item["path"]
        for item in list(fold_contract_entry(prepared_contract, fold_id)["source_files"])
        if item["name"] == "l2_checkpoint"
    )
    load_fold_checkpoint(
        l2_checkpoint,
        task="dbb_specialist",
        schema=l2_config.train.checkpoint_schema,
        fold_id=fold_id,
        model=l2_model,
        optimizer=None,
        config_identity=__import__("radar_ln3.task_models", fromlist=["task_config_identity_sha256"]).task_config_identity_sha256(l2_config),
        config_path=l2_config_path,
        contract_sha256=sha256_file(
            root / "data/processed/RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration-contract.json"
        ),
        readiness_sha256=sha256_file(
            root / "src-v3/processed/report/RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness.json"
        ),
        source_config_sha256=source_l2_sha,
        initial_state_sha256=l2_initial_sha,
        device=l2_device,
    )
    l1_model.eval()
    l2_model.eval()
    if l1_device != l2_device:
        raise ValueError("V26f L1/L2 devices differ")
    return l1_model, l2_model, l1_device, {
        "l1_config_path": relative_path(root, l1_config_path),
        "l1_config_sha256": sha256_file(l1_config_path),
        "l1_checkpoint_path": relative_path(root, l1_checkpoint),
        "l1_checkpoint_sha256": sha256_file(l1_checkpoint),
        "l2_config_path": relative_path(root, l2_config_path),
        "l2_config_sha256": sha256_file(l2_config_path),
        "l2_checkpoint_path": relative_path(root, l2_checkpoint),
        "l2_checkpoint_sha256": sha256_file(l2_checkpoint),
        "l1_fixed_epoch": L1_EPOCH,
        "l2_fixed_epoch": L2_EPOCH,
    }


def _metadata(dataset: Any, parent_id: str) -> dict[str, str]:
    group = dataset.base._groups[str(parent_id)]
    first = group.iloc[0]
    return {
        "track_id": str(first.get("track_id", "")),
        "capture_id": str(first.get("capture_id", "")),
        "provenance_component_id": str(first.get("provenance_component_id", "")),
    }


def _event_rows(
    *,
    split: str,
    fold_id: int,
    l1_dataset: Any,
    l1_batch: Mapping[str, Any],
    l1_output: Mapping[str, torch.Tensor],
    l2_batch: Mapping[str, Any],
    l2_output: Mapping[str, torch.Tensor],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    parent_id = str(l1_batch["parent_sample_ids"][0])
    length = int(l1_batch["lengths"][0].item())
    if length <= 0:
        raise ValueError(f"V26f encountered an empty parent: {parent_id}")
    l1_probabilities = torch.softmax(l1_output["sequence_logits"], dim=-1)[0]
    l2_probabilities = torch.softmax(l2_output["sequence_logits"], dim=-1)[0]
    if l1_probabilities.shape[0] < length or l2_probabilities.shape[0] < length:
        raise ValueError(f"V26f model output is shorter than parent sequence: {parent_id}")
    if not torch.isfinite(l1_probabilities[:length]).all() or not torch.isfinite(l2_probabilities[:length]).all():
        raise ValueError(f"V26f fold {fold_id} emitted a non-finite probability")
    label = str(l1_batch["base_label_names"][0])
    if label != str(l2_batch["label_names"][0]):
        raise ValueError(f"V26f L1/L2 label mismatch for parent {parent_id}")
    metadata = _metadata(l1_dataset, parent_id)
    rows: list[dict[str, Any]] = []
    for event_index in range(length):
        l1_row = l1_probabilities[event_index]
        l2_row = l2_probabilities[event_index]
        l1_id = int(l1_row.argmax().item())
        l2_id = int(l2_row.argmax().item())
        top_values = torch.topk(l2_row, k=2).values
        target_probability = _finite_probability(l1_row[0].item(), field="L1 target probability")
        clutter_probability = _finite_probability(l1_row[1].item(), field="L1 clutter probability")
        subclass_probability = _finite_probability(l2_row[l2_id].item(), field="L2 subclass probability")
        known_target_accepted = target_probability + EPS >= float(thresholds["l1_known_target_acceptance"])
        known_clutter_accepted = clutter_probability + EPS >= float(thresholds["l1_known_clutter_acceptance"])
        # This is intentionally based only on the model's L1 output.  A true
        # Clutter/Unknown label may still be routed through L2 and must not be
        # hidden by label-aware control flow.
        use_l2 = bool(known_target_accepted)
        dbb_subclass = ProductAction(DBB_LABELS[l2_id]) if use_l2 else None
        dbb_subclass_accepted = bool(use_l2 and subclass_probability + EPS >= float(thresholds["l2_subclass_acceptance"]))
        decision = resolve_semantic_action(
            observation_eligible=event_index + 1 >= MINIMUM_EVIDENCE_EVENTS,
            known_target_accepted=known_target_accepted,
            known_clutter_accepted=known_clutter_accepted,
            dbb_subclass=dbb_subclass,
            dbb_subclass_accepted=dbb_subclass_accepted,
        )
        rows.append(
            {
                "split": split,
                "fold_id": int(fold_id),
                "parent_sample_id": parent_id,
                "label_name": label,
                "event_index": int(event_index),
                "window_length": int(event_index + 1),
                "parent_length": int(length),
                "product_action_eligible": bool(event_index + 1 >= MINIMUM_EVIDENCE_EVENTS),
                "l1_route": "KNOWN_TARGET_EVIDENCE" if l1_id == 0 else "KNOWN_CLUTTER_EVIDENCE",
                "l1_target_probability": target_probability,
                "l1_clutter_probability": clutter_probability,
                "l1_confidence": _finite_probability(l1_row[l1_id].item(), field="L1 confidence"),
                "known_target_accepted": bool(known_target_accepted),
                "known_clutter_accepted": bool(known_clutter_accepted),
                "l2_top_subclass": DBB_LABELS[l2_id],
                "l2_subclass_probability": subclass_probability,
                "l2_subclass_margin": float((top_values[0] - top_values[1]).item()),
                "l2_evidence_used_for_action": bool(use_l2),
                "l2_subclass_accepted": bool(dbb_subclass_accepted),
                "semantic_action": decision.action.value,
                "semantic_reason": decision.reason,
                "l2_forwarded": True,
                **metadata,
            }
        )
    return rows


def _run_split(
    root: Path,
    *,
    split: str,
    fold_id: int,
    l1_model: Any,
    l2_model: Any,
    device: torch.device,
    l1_config: Any,
    l2_config: Any,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    split_key = "val" if split == "val" else "test"
    l1_cache = root / (L1_VALIDATION_CACHE if split_key == "val" else L1_TEST_CACHE)
    l2_physical = root / (
        L2_VALIDATION_PHYSICAL_CACHE if split_key == "val" else L2_TEST_PHYSICAL_CACHE
    )
    l2_micro = root / (
        L2_VALIDATION_MICRO_CACHE if split_key == "val" else L2_TEST_MICRO_CACHE
    )
    l1_dataset = V26OpenSetAuditDataset(
        root, l1_config, split_name=split, cache_path=l1_cache
    )
    l2_dataset = FrozenL2MixedAuditDataset(
        root,
        l2_config,
        split_name=split,
        rd_physical_cache_path=l2_physical,
        micro_doppler_cache_path=l2_micro,
    )
    expected = EXPECTED_PARENT_COUNTS[split_key]
    if l1_dataset.label_counts() != expected or l2_dataset.label_counts() != expected:
        raise ValueError(f"V26f {split} mixed parent support differs from contract")
    alignment = verify_l1_l2_event_alignment(l1_dataset.base, l2_dataset.base)
    l1_loader = build_v26_open_set_audit_dataloader(l1_dataset, batch_size=1)
    l2_loader = build_frozen_l2_mixed_audit_dataloader(l2_dataset, batch_size=1)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for l1_batch, l2_batch in zip(l1_loader, l2_loader, strict=True):
            if str(l1_batch["parent_sample_ids"][0]) != str(l2_batch["parent_sample_ids"][0]):
                raise ValueError("V26f paired L1/L2 parent order differs")
            l1_tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in l1_batch.items()
            }
            l2_tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in l2_batch.items()
            }
            l1_output = l1_model(**l1_model_inputs(l1_tensors))
            l2_output = l2_model(**l2_model_inputs(l2_tensors))
            rows.extend(
                _event_rows(
                    split=split,
                    fold_id=fold_id,
                    l1_dataset=l1_dataset,
                    l1_batch=l1_batch,
                    l1_output=l1_output,
                    l2_batch=l2_batch,
                    l2_output=l2_output,
                    thresholds=thresholds,
                )
            )
    expected_parent_ids = set(str(value) for value in l1_dataset.parent_ids)
    actual_parent_ids = {str(row["parent_sample_id"]) for row in rows}
    if actual_parent_ids != expected_parent_ids:
        raise ValueError(f"V26f {split} forward coverage differs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_sample_id"])].append(row)
    parent_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    for parent_id, group in grouped.items():
        group.sort(key=lambda value: int(value["event_index"]))
        if [int(row["event_index"]) for row in group] != list(range(len(group))):
            raise ValueError(f"V26f event order is not contiguous: {parent_id}")
        parent_rows.append(dict(group[-1]))
        eligible_rows.extend(dict(row) for row in group if bool(row["product_action_eligible"]) and row is not group[-1])
    parent_rows.sort(key=lambda value: str(value["parent_sample_id"]))
    eligible_rows.sort(key=lambda value: (str(value["parent_sample_id"]), int(value["event_index"])))
    return {
        "split": split,
        "alignment": alignment,
        "event_rows": rows,
        "eligible_prefix_rows": eligible_rows,
        "parent_rows": parent_rows,
        "parent_counts": expected,
        "event_count": len(rows),
    }


def _weighted_rows(rows: Sequence[Mapping[str, Any]], *, parent_balanced: bool) -> list[tuple[Mapping[str, Any], float]]:
    if not rows:
        return []
    if not parent_balanced:
        return [(row, 1.0) for row in rows]
    counts = Counter(str(row["parent_sample_id"]) for row in rows)
    if any(value <= 0 for value in counts.values()):
        raise ValueError("V26f parent-balanced rows contain an invalid parent count")
    return [(row, 1.0 / float(counts[str(row["parent_sample_id"])])) for row in rows]


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    parent_balanced: bool = False,
) -> dict[str, Any] | None:
    if not rows:
        return None
    weighted = _weighted_rows(rows, parent_balanced=parent_balanced)
    matrix = np.zeros((len(MIXED_LABELS), len(ACTION_NAMES)), dtype=np.float64)
    for row, weight in weighted:
        label = str(row["label_name"])
        action = str(row["semantic_action"])
        if label not in MIXED_LABEL_INDEX or action not in ACTION_INDEX:
            raise ValueError("V26f metric row is outside the mixed action space")
        matrix[MIXED_LABEL_INDEX[label], ACTION_INDEX[action]] += float(weight)
    expected_action = {
        **{label: label for label in DBB_LABELS},
        "ClutterTarget": "ClutterTarget",
        "UnknownTarget": "UNKNOWN_REJECTED",
    }
    per_class: list[dict[str, Any]] = []
    correct_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in MIXED_LABELS:
        label_index = MIXED_LABEL_INDEX[label]
        action_index = ACTION_INDEX[expected_action[label]]
        tp = float(matrix[label_index, action_index])
        support = float(matrix[label_index].sum())
        predicted = float(matrix[:, action_index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        correct_values.append(tp)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        action_counts = {
            action: float(matrix[label_index, ACTION_INDEX[action]])
            for action in ACTION_NAMES
            if matrix[label_index, ACTION_INDEX[action]] > 0.0
        }
        per_class.append(
            {
                "class": label,
                "expected_action": expected_action[label],
                "support": support if parent_balanced else int(support),
                "accepted_as_expected_action": tp if parent_balanced else int(tp),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "action_counts": action_counts,
            }
        )
    total = float(matrix.sum())
    correct = float(sum(correct_values))
    known_rows = [row for row in rows if str(row["label_name"]) != UNKNOWN_LABEL]
    unknown_rows = [row for row in rows if str(row["label_name"]) == UNKNOWN_LABEL]
    known_target_rows = [row for row in rows if str(row["label_name"]) in DBB_LABELS]
    clutter_rows = [row for row in rows if str(row["label_name"]) == "ClutterTarget"]
    known_target_correct = sum(
        float(weight)
        for row, weight in weighted
        if str(row["label_name"]) in DBB_LABELS and str(row["semantic_action"]) == str(row["label_name"])
    )
    clutter_correct = sum(
        float(weight)
        for row, weight in weighted
        if str(row["label_name"]) == "ClutterTarget" and str(row["semantic_action"]) == "ClutterTarget"
    )
    unknown_rejected = sum(
        float(weight) for row, weight in weighted if str(row["label_name"]) == UNKNOWN_LABEL and str(row["semantic_action"]) == "UNKNOWN_REJECTED"
    )
    unknown_false_accept = sum(
        float(weight)
        for row, weight in weighted
        if str(row["label_name"]) == UNKNOWN_LABEL and str(row["semantic_action"]) in KNOWN_ACTION_INDEX
    )
    unknown_pending = sum(
        float(weight) for row, weight in weighted if str(row["label_name"]) == UNKNOWN_LABEL and str(row["semantic_action"]) == "PENDING_ABSTAIN"
    )
    known_target_mass = sum(weight for row, weight in weighted if str(row["label_name"]) in DBB_LABELS)
    clutter_mass = sum(weight for row, weight in weighted if str(row["label_name"]) == "ClutterTarget")
    unknown_mass = sum(weight for row, weight in weighted if str(row["label_name"]) == UNKNOWN_LABEL)
    known_action_correct = sum(
        float(weight)
        for row, weight in weighted
        if str(row["label_name"]) != UNKNOWN_LABEL
        and str(row["semantic_action"]) == str(row["label_name"])
    )
    known_action_total = sum(weight for row, weight in weighted if str(row["label_name"]) != UNKNOWN_LABEL)
    accepted_mass = sum(
        float(weight) for row, weight in weighted if str(row["semantic_action"]) in KNOWN_ACTION_INDEX
    )
    accepted_correct = sum(
        float(weight)
        for row, weight in weighted
        if str(row["semantic_action"]) in KNOWN_ACTION_INDEX
        and str(row["label_name"]) == str(row["semantic_action"])
    )
    return {
        "scope": scope,
        "aggregation": "parent_balanced" if parent_balanced else "event_or_parent",
        "event_support": len(rows),
        "parent_support": len({str(row["parent_sample_id"]) for row in rows}),
        "weighted_support": total,
        "action_names": list(ACTION_NAMES),
        "label_names": list(MIXED_LABELS),
        "confusion_rows_true_label_columns_action": matrix.tolist(),
        "per_class": per_class,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "known_target_recall": known_target_correct / known_target_mass if known_target_mass else None,
        "clutter_precision": (
            clutter_correct / float(matrix[:, ACTION_INDEX["ClutterTarget"]].sum())
            if matrix[:, ACTION_INDEX["ClutterTarget"]].sum()
            else None
        ),
        "clutter_recall": clutter_correct / clutter_mass if clutter_mass else None,
        "clutter_f1": next(
            float(item["f1"]) for item in per_class if item["class"] == "ClutterTarget"
        ),
        "unknown_rejection_rate": unknown_rejected / unknown_mass if unknown_mass else None,
        "unknown_false_accept_rate": unknown_false_accept / unknown_mass if unknown_mass else None,
        "unknown_pending_rate": unknown_pending / unknown_mass if unknown_mass else None,
        "known_to_unknown_rejection_rate": (
            sum(
                float(weight)
                for row, weight in weighted
                if str(row["label_name"]) != UNKNOWN_LABEL
                and str(row["semantic_action"]) == "UNKNOWN_REJECTED"
            )
            / known_action_total
            if known_action_total
            else None
        ),
        "known_pending_rate": (
            sum(
                float(weight)
                for row, weight in weighted
                if str(row["label_name"]) != UNKNOWN_LABEL
                and str(row["semantic_action"]) == "PENDING_ABSTAIN"
            )
            / known_action_total
            if known_action_total
            else None
        ),
        "known_action_accuracy": known_action_correct / known_action_total if known_action_total else None,
        "accepted_action_accuracy": accepted_correct / accepted_mass if accepted_mass else None,
        "accepted_action_coverage": accepted_mass / total if total else None,
    }


def _l1_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if str(row["label_name"]) != UNKNOWN_LABEL]
    targets = [row for row in known if str(row["label_name"]) in DBB_LABELS]
    clutter = [row for row in known if str(row["label_name"]) == "ClutterTarget"]
    unknown = [row for row in rows if str(row["label_name"]) == UNKNOWN_LABEL]
    target_route = lambda row: bool(row["known_target_accepted"])
    clutter_route = lambda row: bool(row["known_clutter_accepted"])
    return {
        "support": len(rows),
        "known_support": len(known),
        "known_target_support": len(targets),
        "clutter_support": len(clutter),
        "unknown_support": len(unknown),
        "known_target_acceptance_recall": sum(target_route(row) for row in targets) / len(targets) if targets else None,
        "clutter_acceptance_recall": sum(clutter_route(row) for row in clutter) / len(clutter) if clutter else None,
        "known_to_clutter_acceptance_rate": sum(clutter_route(row) for row in targets) / len(targets) if targets else None,
        "clutter_to_target_acceptance_rate": sum(target_route(row) for row in clutter) / len(clutter) if clutter else None,
        "unknown_target_acceptance_rate": sum(target_route(row) for row in unknown) / len(unknown) if unknown else None,
        "unknown_clutter_acceptance_rate": sum(clutter_route(row) for row in unknown) / len(unknown) if unknown else None,
        "route_counts": dict(Counter(str(row["l1_route"]) for row in rows)),
    }


def _l2_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dbb = [row for row in rows if str(row["label_name"]) in DBB_LABELS]
    used = [row for row in rows if bool(row["l2_evidence_used_for_action"])]
    accepted = [row for row in used if bool(row["l2_subclass_accepted"])]
    correct = [row for row in dbb if str(row["l2_top_subclass"]) == str(row["label_name"])]
    correct_accepted = [row for row in correct if bool(row["l2_subclass_accepted"])]
    matrix = {label: {other: 0 for other in DBB_LABELS} for label in DBB_LABELS}
    for row in dbb:
        matrix[str(row["label_name"])][str(row["l2_top_subclass"])] += 1
    return {
        "all_forward_count": len(rows),
        "model_route_used_count": len(used),
        "model_route_used_rate": len(used) / len(rows) if rows else None,
        "model_route_accepted_count": len(accepted),
        "dbb_support": len(dbb),
        "dbb_top_subclass_accuracy": len(correct) / len(dbb) if dbb else None,
        "dbb_correct_subclass_accepted_recall": len(correct_accepted) / len(dbb) if dbb else None,
        "dbb_subclass_acceptance_rate": len(accepted) / len(used) if used else None,
        "confusion_rows_true_label_columns_top_subclass": matrix,
    }


def _capture_strata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        capture = str(row.get("capture_id", ""))
        value = groups.setdefault(
            capture,
            {"capture_id": capture, "parent_support": 0, "label_counts": Counter(), "action_counts": Counter()},
        )
        value["parent_support"] += 1
        value["label_counts"][str(row["label_name"])] += 1
        value["action_counts"][str(row["semantic_action"])] += 1
    return [
        {
            "capture_id": key,
            "parent_support": int(value["parent_support"]),
            "label_counts": dict(sorted(value["label_counts"].items())),
            "action_counts": dict(sorted(value["action_counts"].items())),
        }
        for key, value in sorted(groups.items())
    ]


def _bootstrap(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    if not rows:
        return {"replicates": 0, "metrics": {}}
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label_name"])].append(row)
    if set(by_label) != set(MIXED_LABELS):
        raise ValueError("V26f parent bootstrap lacks one or more mixed-label strata")
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = defaultdict(list)
    labels = tuple(sorted(by_label))
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[Mapping[str, Any]] = []
        for label in labels:
            group = by_label[label]
            indices = rng.integers(0, len(group), size=len(group))
            sampled.extend(group[int(index)] for index in indices)
        metric = _metrics(sampled, scope="parent_bootstrap")
        assert metric is not None
        for name in (
            "accuracy",
            "macro_f1",
            "known_target_recall",
            "clutter_precision",
            "clutter_recall",
            "clutter_f1",
            "unknown_rejection_rate",
            "unknown_false_accept_rate",
            "unknown_pending_rate",
        ):
            value = metric.get(name)
            if value is not None and np.isfinite(float(value)):
                values[name].append(float(value))
    return {
        "method": "stratified parent bootstrap by mixed truth label; frozen actions and thresholds",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "confidence_level": 0.95,
        "stratum_support": {label: len(by_label[label]) for label in labels},
        "metrics": {
            name: {
                "median": float(np.median(data)),
                "lower_95": float(np.quantile(data, 0.025)),
                "upper_95": float(np.quantile(data, 0.975)),
            }
            for name, data in sorted(values.items())
        },
    }


def _fold_summary_metrics(folds: Sequence[Mapping[str, Any]], *, split: str, scope: str) -> dict[str, Any]:
    entries = []
    for fold in folds:
        metric = dict(dict(fold[split]).get("metrics", {}).get(scope) or {})
        if metric:
            entries.append((int(fold["fold_id"]), metric))
    scalar_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "known_target_recall",
        "clutter_precision",
        "clutter_recall",
        "clutter_f1",
        "unknown_rejection_rate",
        "unknown_false_accept_rate",
        "unknown_pending_rate",
    )
    summary: dict[str, Any] = {"scope": scope, "fold_count": len(entries), "fold_values": {}}
    for name in scalar_names:
        values = [float(metric[name]) for _, metric in entries if metric.get(name) is not None]
        if not values:
            continue
        summary["fold_values"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _parent_detail_csv(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "split",
        "fold_id",
        "parent_sample_id",
        "label_name",
        "track_id",
        "capture_id",
        "provenance_component_id",
        "parent_length",
        "l1_route",
        "l1_target_probability",
        "l1_clutter_probability",
        "known_target_accepted",
        "known_clutter_accepted",
        "l2_evidence_used_for_action",
        "l2_top_subclass",
        "l2_subclass_probability",
        "l2_subclass_accepted",
        "semantic_action",
        "semantic_reason",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda value: (str(value["split"]), int(value["fold_id"]), str(value["parent_sample_id"]))):
        writer.writerow({field: row.get(field) for field in fields})
    text = stream.getvalue()
    return write_immutable_text(
        root / PARENT_DETAIL_CSV,
        text,
        max_bytes=MAX_DETAIL_BYTES,
        description="V26f parent detail CSV",
    )


def _compact_split(split_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "alignment": dict(split_result["alignment"]),
        "event_count": int(split_result["event_count"]),
        "parent_count": len(split_result["parent_rows"]),
        "eligible_prefix_event_count": len(split_result["eligible_prefix_rows"]),
        "parent_counts": dict(split_result["parent_counts"]),
        "metrics": {
            "window_mixed": _metrics(split_result["event_rows"], scope="window"),
            "window_known_only": _metrics(
                [row for row in split_result["event_rows"] if str(row["label_name"]) != UNKNOWN_LABEL],
                scope="window_known_only",
            ),
            "eligible_prefix_mixed": _metrics(
                split_result["eligible_prefix_rows"], scope="eligible_prefix"
            ),
            "eligible_prefix_parent_balanced_mixed": _metrics(
                split_result["eligible_prefix_rows"],
                scope="eligible_prefix_parent_balanced",
                parent_balanced=True,
            ),
            "parent_final_mixed": _metrics(split_result["parent_rows"], scope="parent_final"),
            "parent_final_known_only": _metrics(
                [row for row in split_result["parent_rows"] if str(row["label_name"]) != UNKNOWN_LABEL],
                scope="parent_final_known_only",
            ),
        },
        "l1_diagnostics": {
            "window": _l1_diagnostics(split_result["event_rows"]),
            "parent_final": _l1_diagnostics(split_result["parent_rows"]),
        },
        "l2_diagnostics": {
            "window": _l2_diagnostics(split_result["event_rows"]),
            "parent_final": _l2_diagnostics(split_result["parent_rows"]),
        },
        "capture_strata_parent_final": _capture_strata(split_result["parent_rows"]),
        "parent_bootstrap": _bootstrap(
            split_result["parent_rows"],
            seed=BOOTSTRAP_SEED + (0 if str(split_result["split"]) == "val" else 1),
        ),
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prepared_contract, prepared_readiness = validate_prepared_contract(root)
    thresholds_by_fold = _read_thresholds(root)
    all_fold_reports: list[dict[str, Any]] = []
    parent_details: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        l1_model, l2_model, device, model_binding = _load_fold_models(
            root, fold_id=fold_id, prepared_contract=prepared_contract
        )
        v26e_contract = read_json(root / V26E_CONTRACT, description="V26e contract")
        v26e_fold = next(row for row in v26e_contract["folds"] if int(row["fold_id"]) == fold_id)
        l1_config, _, _ = _runtime_l1_config(root, fold_id, dict(v26e_fold["l1"]))
        v26c_contract = read_json(
            root / "data/processed/RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration-contract.json",
            description="V26c contract",
        )
        v26c_fold = next(row for row in v26c_contract["folds"] if int(row["fold_id"]) == fold_id)
        l2_config, _, _ = _runtime_l2_config(root, fold_id, dict(v26c_fold["l2"]))
        split_results: dict[str, dict[str, Any]] = {}
        for split in ("val", "test"):
            result = _run_split(
                root,
                split=split,
                fold_id=fold_id,
                l1_model=l1_model,
                l2_model=l2_model,
                device=device,
                l1_config=l1_config,
                l2_config=l2_config,
                thresholds=thresholds_by_fold[fold_id],
            )
            split_results[split] = _compact_split(result)
            parent_details.extend(result["parent_rows"])
        all_fold_reports.append(
            {
                "fold_id": fold_id,
                "thresholds": dict(thresholds_by_fold[fold_id]),
                "model_bindings": model_binding,
                "device": str(device),
                "validation": split_results["val"],
                "test": split_results["test"],
            }
        )
        del l1_model, l2_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail_status = _parent_detail_csv(root, parent_details)
    aggregate = {
        "validation": {
            "parent_final_mixed": _fold_summary_metrics(all_fold_reports, split="validation", scope="parent_final_mixed"),
            "eligible_prefix_parent_balanced_mixed": _fold_summary_metrics(
                all_fold_reports, split="validation", scope="eligible_prefix_parent_balanced_mixed"
            ),
        },
        "test": {
            "parent_final_mixed": _fold_summary_metrics(all_fold_reports, split="test", scope="parent_final_mixed"),
            "eligible_prefix_parent_balanced_mixed": _fold_summary_metrics(
                all_fold_reports, split="test", scope="eligible_prefix_parent_balanced_mixed"
            ),
        },
        "interpretation": "fold-level diagnostic mean/std/range; five fold endpoints are not pooled into a deployable model",
    }
    report = {
        "status": "completed_v26f_frozen_mixed_open_set_audit",
        "schema_version": REPORT_SCHEMA,
        "experiment": "V26f frozen mixed Known/Unknown L1-L2-semantic audit",
        "scope": "CQ08 mixed validation/test inference only after V26e L1 epoch-32 and V26c L2 epoch-15 freeze",
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "readiness_path": relative_path(root, root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "threshold_source": {
            "path": relative_path(root, root / V26E_REPORT),
            "sha256": sha256_file(root / V26E_REPORT),
            "thresholds_by_fold": {str(key): value for key, value in thresholds_by_fold.items()},
            "fit_population": "V26e Known-only outer-cross-fitted parent-final records",
            "refit_in_v26f": False,
        },
        "data": {
            "source": "CQ08",
            "index": relative_path(root, root / "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"),
            "mixed_labels": list(MIXED_LABELS),
            "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
            "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        },
        "fold_count": FOLD_COUNT,
        "folds": all_fold_reports,
        "aggregate_fold_diagnostics": aggregate,
        "parent_detail_csv": relative_path(root, root / PARENT_DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_CSV),
        "parent_detail_csv_write": detail_status,
        "routing_semantics": {
            "l1_then_l2_then_semantic_decision": True,
            "l2_forwarded_for_every_mixed_parent": True,
            "l2_action_use_depends_on_model_l1_target_route_only": True,
            "true_label_never_controls_routing": True,
            "unknown_is_not_a_learned_class": True,
            "unknown_rejection_action": "UNKNOWN_REJECTED",
            "pending_action": "PENDING_ABSTAIN",
            "pending_is_not_unknown_rejection_success": True,
            "minimum_evidence_events": MINIMUM_EVIDENCE_EVENTS,
        },
        "isolation": {
            "unknown_training_exposure": 0,
            "unknown_normalization_exposure": 0,
            "unknown_checkpoint_selection_exposure": 0,
            "unknown_threshold_fit_exposure": 0,
            "validation_used_for_model_or_threshold_selection": False,
            "test_used_for_model_or_threshold_selection": False,
            "new_training": False,
            "new_threshold_fit": False,
        },
        "not_executed": {
            "training": True,
            "oof_retraining": True,
            "threshold_fitting": True,
            "deployment_registry": True,
            "full_production_stable": True,
            "p95": True,
        },
        "source_readiness_sha256": sha256_file(root / READINESS),
        "deployment_eligible": False,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    write_immutable_text(root / REPORT, text, max_bytes=MAX_REPORT_BYTES, description="V26f report")
    return {
        "status": report["status"],
        "report": str(root / REPORT),
        "report_sha256": sha256_file(root / REPORT),
        "parent_detail_csv": str(root / PARENT_DETAIL_CSV),
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "fold_count": FOLD_COUNT,
        "unknown_training_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V26f frozen mixed Known/Unknown audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
