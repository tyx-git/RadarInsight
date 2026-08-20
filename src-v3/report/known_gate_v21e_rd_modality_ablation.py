"""Read-only RD modality ablation for the frozen V21e KnownGate OOF models.

This diagnostic never trains, updates parameters, selects a checkpoint, or
touches validation/test/Unknown payloads.  It first replays the official OOF
predictions, then evaluates two inference-only counterfactuals:

* ``rd_content_zero_quality_preserved`` removes the RD embedding while retaining
  the original causal RD quality vector.  This is intentionally off-manifold
  and is used only to isolate RD tensor-content reliance.
* ``rd_contract_missing`` removes RD content and replaces quality with the
  production missing-RD vector for every active event.

Neither counterfactual estimates the performance of a model retrained without
RD.  The report describes conditional reliance of the frozen model only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.data import RD_QUALITY_NAMES
from radar_ln3.known_gate import (
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_LABELS,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    summarize_known_gate_records,
)
from radar_ln3.known_gate_v21e_oof import (
    ASSIGNMENT_RELATIVE,
    FOLD_CONTRACT_RELATIVE,
    FOLD_COUNT,
    READINESS_RELATIVE,
    config_relative,
    load_fold_contract,
    prediction_relative,
    relative_path,
    resolve_path,
    sequence_sha256,
    sha256_file,
    write_new_or_identical,
)
from report.known_gate_v21e_oof import (
    _base_label_outcomes,
    _component_bootstrap,
    _delta,
    _load_arm,
    _metric_vector,
    _ordered_pair,
)
from train.known_gate_v21 import _load_checkpoint, _state_dict_sha256
from train.known_gate_v21e_oof import _verify_prepared


SCHEMA_VERSION = "RADAR-LN3-V21e-known-gate-rd-modality-ablation-v2"
REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-rd-modality-read-only-ablation.json"
)
PARENT_ATTRIBUTION_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-rd-modality-read-only-parent-attribution.csv"
)

ORIGINAL = "original_replay"
CONTENT_ZERO = "rd_content_zero_quality_preserved"
CONTRACT_MISSING = "rd_contract_missing"
ARMS = (ORIGINAL, CONTENT_ZERO, CONTRACT_MISSING)
SCOPES = ("parent", "window", "eligible_prefix")
REPLAY_PROBABILITY_ATOL = 1.0e-6

_RD_PRESENT_INDEX = RD_QUALITY_NAMES.index("rd_present")
_RD_MISSING_INDEX = RD_QUALITY_NAMES.index("rd_missing")


def _model_inputs(tensors: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tr_features": tensors["tr_features"],
        "tr_missing_mask": tensors["tr_missing_mask"],
        "tr_dt_seconds": tensors["tr_dt_seconds"],
        "tr_dt_valid": tensors["tr_dt_valid"],
        "tr_dt_reason": tensors["tr_dt_reason"],
        "rd_frames": tensors["rd_frames"],
        "rd_frame_mask": tensors["rd_frame_mask"],
        "rd_quality": tensors["rd_quality"],
        "tr_padding_mask": tensors["tr_padding_mask"],
        "known_gate_features": tensors.get("known_gate_features"),
        "known_gate_feature_valid": tensors.get("known_gate_feature_valid"),
    }


def ablated_model_inputs(
    tensors: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    """Return model inputs for one frozen inference arm without mutating input."""

    if arm not in ARMS:
        raise ValueError(f"unsupported V21e RD ablation arm: {arm}")
    result = _model_inputs(tensors)
    if arm == ORIGINAL:
        return result

    result["rd_frames"] = torch.zeros_like(tensors["rd_frames"])
    result["rd_frame_mask"] = torch.zeros_like(tensors["rd_frame_mask"])
    if arm == CONTENT_ZERO:
        # Deliberately retain the original quality metadata.  This is an
        # attribution-only counterfactual and not a physically valid event.
        result["rd_quality"] = tensors["rd_quality"]
        return result

    active = ~tensors["tr_padding_mask"].to(dtype=torch.bool)
    quality = torch.zeros_like(tensors["rd_quality"])
    quality[..., _RD_MISSING_INDEX] = active.to(dtype=quality.dtype)
    result["rd_quality"] = quality
    return result


def _append_records(
    destination: dict[str, list[dict[str, Any]]],
    *,
    arm: str,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    config: Any,
    fold_id: int,
    component_by_parent: Mapping[str, str],
) -> None:
    logits = output["sequence_logits"].detach().cpu()
    probabilities = torch.softmax(logits, dim=-1)
    for batch_index, parent_id_value in enumerate(batch["parent_sample_ids"]):
        parent_id = str(parent_id_value)
        length = int(batch["lengths"][batch_index].item())
        true_id = int(batch["labels"][batch_index].item())
        final: dict[str, Any] | None = None
        for event_index in range(length):
            event_probability = probabilities[batch_index, event_index]
            event_logits = logits[batch_index, event_index]
            pred_id = int(event_probability.argmax().item())
            original_frame_count = int(
                batch["rd_frame_mask"][batch_index, event_index].sum().item()
            )
            original_quality = batch["rd_quality"][batch_index, event_index]
            final = {
                "true_id": true_id,
                "pred_id": pred_id,
                "confidence": float(event_probability[pred_id].item()),
                "known_target_probability": float(event_probability[0].item()),
                "known_clutter_probability": float(event_probability[1].item()),
                "known_target_logit_margin": float(
                    (event_logits[0] - event_logits[1]).item()
                ),
                "parent_sample_id": parent_id,
                "track_id": str(batch["track_ids"][batch_index]),
                "source_name": str(batch["source_names"][batch_index]),
                "label_name": str(batch["base_label_names"][batch_index]),
                "event_index": event_index,
                "window_length": event_index + 1,
                "product_action_eligible": bool(
                    event_index + 1 >= config.train.prefix_auxiliary_min_events
                ),
                "rd_frame_count": original_frame_count,
                "rd_present": original_frame_count > 0,
                "rd_multiple_frames": original_frame_count > 1,
                "parent_length": length,
                "rd_quality_missing": bool(
                    float(original_quality[_RD_MISSING_INDEX].item()) > 0.5
                ),
                "rd_quality_frame_count_scaled": float(
                    original_quality[RD_QUALITY_NAMES.index("rd_frame_count_scaled")].item()
                ),
                "fold_id": int(fold_id),
                "arm": arm,
                "split_name": "train_oof_held_out",
                "provenance_component_id": str(component_by_parent[parent_id]),
                "effective_rd_content": bool(arm == ORIGINAL and original_frame_count > 0),
                "effective_rd_quality_mode": (
                    "original"
                    if arm in {ORIGINAL, CONTENT_ZERO}
                    else "contract_missing"
                ),
            }
            destination["window"].append(dict(final))
            if (
                event_index + 1 >= config.train.prefix_auxiliary_min_events
                and event_index + 1 < length
            ):
                destination["eligible_prefix"].append(dict(final))
        if final is None:
            raise ValueError(f"RD ablation encountered empty parent: {parent_id}")
        destination["parent"].append(final)


def _compare_replay(
    official: Mapping[str, Sequence[Mapping[str, Any]]],
    replay: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in SCOPES:
        before, after = _ordered_pair(official[scope], replay[scope], scope=scope)
        hard_mismatch = 0
        target_errors: list[float] = []
        clutter_errors: list[float] = []
        for frozen, reproduced in zip(before, after, strict=True):
            if int(frozen["pred_id"]) != int(reproduced["pred_id"]):
                hard_mismatch += 1
            target_errors.append(
                abs(
                    float(frozen["known_target_probability"])
                    - float(reproduced["known_target_probability"])
                )
            )
            clutter_errors.append(
                abs(
                    float(frozen["known_clutter_probability"])
                    - float(reproduced["known_clutter_probability"])
                )
            )
        maximum = max(target_errors + clutter_errors, default=0.0)
        result[scope] = {
            "support": len(before),
            "hard_prediction_mismatch_count": hard_mismatch,
            "maximum_probability_absolute_error": maximum,
            "passed": bool(
                hard_mismatch == 0 and maximum <= REPLAY_PROBABILITY_ATOL
            ),
        }
    if not all(row["passed"] for row in result.values()):
        raise AssertionError(
            "frozen V21e OOF replay differs from official predictions; "
            "RD ablation is invalid"
        )
    return result


def _register_source(
    root: Path, sources: dict[str, str], path: Path
) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    key = relative_path(root, resolved)
    value = sha256_file(resolved)
    if key in sources and sources[key] != value:
        raise ValueError(f"source changed while registering: {key}")
    sources[key] = value


def _tensor_block_audit(value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().float().cpu()
    return {
        "frobenius_norm": float(torch.linalg.vector_norm(tensor).item()),
        "mean_absolute_weight": float(tensor.abs().mean().item()),
        "maximum_absolute_weight": float(tensor.abs().max().item()),
    }


def _fusion_input_weight_audit(model: Any, config: Any) -> dict[str, Any]:
    linear = model.fusion[0]
    weight = linear.weight.detach()
    hidden = int(config.model.hidden_dim)
    rd_embedding = int(config.model.rd_embedding_dim)
    rd_quality = int(config.model.rd_quality_dim)
    evidence = int(config.model.evidence_dim)
    expected = hidden + rd_embedding + rd_quality + evidence
    if tuple(weight.shape) != (hidden, expected):
        raise ValueError(
            "V21e fusion weight shape differs from the declared input blocks"
        )
    tr_end = hidden
    rd_end = tr_end + rd_embedding
    quality_end = rd_end + rd_quality
    blocks = {
        "tr_embedding": weight[:, :tr_end],
        "rd_embedding": weight[:, tr_end:rd_end],
        "rd_quality": weight[:, rd_end:quality_end],
        "course_evidence": weight[:, quality_end:],
    }
    return {
        "fusion_weight_shape": list(weight.shape),
        "input_order": [
            "tr_embedding",
            "rd_embedding",
            "rd_quality",
            "course_evidence",
        ],
        "block_dimensions": {
            name: int(value.shape[1]) for name, value in blocks.items()
        },
        "block_weight_magnitudes": {
            name: _tensor_block_audit(value) for name, value in blocks.items()
        },
        "rd_quality_column_weight_magnitudes": {
            name: _tensor_block_audit(weight[:, rd_end + index])
            for index, name in enumerate(RD_QUALITY_NAMES)
        },
        "interpretation": (
            "Weight magnitudes confirm that the input blocks are connected; "
            "they do not identify class direction after LayerNorm, SiLU and "
            "the liquid recurrent dynamics."
        ),
    }


def _run_fold(
    root: Path,
    *,
    fold_id: int,
    official_payload: Mapping[str, Any],
    component_by_parent: Mapping[str, str],
    sources: dict[str, str],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    config, fold, _assignment, config_path, readiness_hash = _verify_prepared(
        root, arm="treatment", fold_id=fold_id
    )
    if config.train.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "V21e RD ablation requires CUDA to replay the frozen CUDA OOF output"
        )
    checkpoint_path = resolve_path(root, config.train.checkpoint_path)
    prediction_path = root / prediction_relative("treatment", fold_id)
    if (
        Path(str(official_payload["checkpoint_path"])).resolve()
        != checkpoint_path.resolve()
        or Path(str(official_payload["config_path"])).resolve()
        != config_path.resolve()
        or sha256_file(checkpoint_path)
        != official_payload["checkpoint_sha256"]
        or str(official_payload["config_sha256"]) != sha256_file(config_path)
    ):
        raise ValueError(f"V21e fold {fold_id} source binding changed")

    for path in (
        root / FOLD_CONTRACT_RELATIVE,
        root / ASSIGNMENT_RELATIVE,
        root / READINESS_RELATIVE,
        config_path,
        checkpoint_path,
        prediction_path,
        resolve_path(root, config.data.index_path),
        resolve_path(root, config.data.normalization_stats_path),
        resolve_path(root, config.data.contract_path),
    ):
        _register_source(root, sources, path)
    if config.data.train_feature_cache_path is not None:
        _register_source(
            root,
            sources,
            resolve_path(root, config.data.train_feature_cache_path),
        )
    if config.train.paired_baseline_config_path is not None:
        _register_source(
            root,
            sources,
            resolve_path(root, config.train.paired_baseline_config_path),
        )

    model, baseline_reference, _pairing = build_known_gate_model_for_config(
        root, config
    )
    del baseline_reference
    device = torch.device("cuda")
    model.to(device)
    _load_checkpoint(
        checkpoint_path,
        model,
        None,
        config=config,
        config_path=config_path,
        readiness_sha256=readiness_hash,
        map_location=device,
    )
    state_before = _state_dict_sha256(model.state_dict())
    if state_before != official_payload["final_model_state_sha256"]:
        raise ValueError(f"V21e fold {fold_id} model state differs from prediction")

    dataset = build_known_gate_dataset(
        root,
        config,
        split_name="train",
        parent_ids=fold["held_out_parent_ids"],
        allow_feature_cache_superset=True,
    )
    if (
        sorted(dataset.parent_ids) != fold["held_out_parent_ids"]
        or sequence_sha256(sorted(dataset.parent_ids))
        != fold["held_out_parent_ids_sha256"]
    ):
        raise ValueError(f"V21e fold {fold_id} held-out Dataset drifted")
    rd_tensor_paths = sorted(
        {
            str(value).strip()
            for value in dataset.base.frame["rd_tensor_path"].tolist()
            if value is not None
            and not (
                isinstance(value, (float, np.floating))
                and math.isnan(float(value))
            )
            and str(value).strip().lower() not in {"", "nan", "none"}
        }
    )
    for value in rd_tensor_paths:
        _register_source(root, sources, resolve_path(root, value))
    fusion_weight_audit = _fusion_input_weight_audit(model, config)
    loader = build_known_gate_dataloader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
    )
    records = {
        arm: {scope: [] for scope in SCOPES}
        for arm in ARMS
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            for arm in ARMS:
                output = model(**ablated_model_inputs(tensors, arm))
                _append_records(
                    records[arm],
                    arm=arm,
                    output=output,
                    batch=batch,
                    config=config,
                    fold_id=fold_id,
                    component_by_parent=component_by_parent,
                )
    state_after = _state_dict_sha256(model.state_dict())
    if state_after != state_before:
        raise AssertionError(f"V21e fold {fold_id} state mutated during ablation")
    replay = _compare_replay(
        dict(official_payload["records"]), records[ORIGINAL]
    )
    audit = {
        "fold_id": fold_id,
        "held_out_parent_count": len(dataset),
        "checkpoint_path": relative_path(root, checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": relative_path(root, config_path),
        "config_sha256": sha256_file(config_path),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "held_out_rd_tensor_file_count": len(rd_tensor_paths),
        "fusion_input_weight_audit": fusion_weight_audit,
        "official_replay": replay,
        "unknown_access_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
    }
    del loader, dataset, model
    torch.cuda.empty_cache()
    return records, audit


def _negate(values: Mapping[str, float]) -> dict[str, float]:
    return {key: -float(value) for key, value in values.items()}


def _exact_mcnemar(
    original: Sequence[Mapping[str, Any]],
    ablated: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    before, after = _ordered_pair(original, ablated, scope=scope)
    harmed = rescued = 0
    for left, right in zip(before, after, strict=True):
        left_correct = int(left["pred_id"]) == int(left["true_id"])
        right_correct = int(right["pred_id"]) == int(right["true_id"])
        harmed += int(left_correct and not right_correct)
        rescued += int((not left_correct) and right_correct)
    discordant = harmed + rescued
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(harmed, rescued) + 1)
        ) / (2.0**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "method": "two-sided exact McNemar/binomial test on discordant paired predictions",
        "harmed_by_ablation": harmed,
        "rescued_by_ablation": rescued,
        "discordant_count": discordant,
        "p_value_two_sided": float(p_value),
    }


def _effect_summary(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("stratified RD effect rows differ in length")
    changed = harmed = rescued = 0
    shifts: list[float] = []
    labels: Counter[str] = Counter()
    before_correct = after_correct = 0
    for left, right in zip(before, after, strict=True):
        if str(left["parent_sample_id"]) != str(right["parent_sample_id"]):
            raise ValueError("stratified RD effect parent order differs")
        left_ok = int(left["pred_id"]) == int(left["true_id"])
        right_ok = int(right["pred_id"]) == int(right["true_id"])
        before_correct += int(left_ok)
        after_correct += int(right_ok)
        changed += int(int(left["pred_id"]) != int(right["pred_id"]))
        harmed += int(left_ok and not right_ok)
        rescued += int((not left_ok) and right_ok)
        labels[str(left["label_name"])] += 1
        shifts.append(
            float(right["known_target_logit_margin"])
            - float(left["known_target_logit_margin"])
        )
    support = len(before)
    data = np.asarray(shifts, dtype=np.float64)
    return {
        "support": support,
        "base_label_support": dict(sorted(labels.items())),
        "original_accuracy": float(before_correct / support) if support else None,
        "ablated_accuracy": float(after_correct / support) if support else None,
        "accuracy_delta": float((after_correct - before_correct) / support)
        if support
        else None,
        "changed": changed,
        "harmed_by_ablation": harmed,
        "rescued_by_ablation": rescued,
        "net_rescue_under_ablation": rescued - harmed,
        "known_target_margin_shift_mean": float(data.mean()) if data.size else None,
        "known_target_margin_shift_median": float(np.median(data))
        if data.size
        else None,
    }


def _frame_bucket(value: int) -> str:
    count = int(value)
    if count <= 0:
        return "K=0"
    if count == 1:
        return "K=1"
    if count <= 4:
        return "K=2-4"
    if count == 5:
        return "K=5"
    return "K>=6"


def _availability_bucket(value: float) -> str:
    fraction = float(value)
    if fraction <= 0.0:
        return "none"
    if fraction <= 0.5:
        return "partial_le_0.5"
    if fraction < 1.0:
        return "partial_gt_0.5"
    return "all_events"


def _parent_length_bucket(value: int) -> str:
    return "length_1-16" if int(value) <= 16 else "length_17+"


def _parent_event_metadata(
    original_windows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in original_windows:
        grouped[str(row["parent_sample_id"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for parent_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["event_index"]))
        frame_counts = np.asarray(
            [int(row["rd_frame_count"]) for row in rows], dtype=np.float64
        )
        present = frame_counts > 0
        result[parent_id] = {
            "event_count": len(rows),
            "rd_present_event_count": int(present.sum()),
            "rd_availability_fraction": float(present.mean()),
            "rd_frame_count_mean": float(frame_counts.mean()),
            "rd_frame_count_max": int(frame_counts.max(initial=0)),
            "final_rd_frame_count": int(frame_counts[-1]),
        }
    return result


def _rd_missing_support_audit(
    original_windows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in original_windows:
        by_label[str(row["label_name"])].append(row)
        by_parent[str(row["parent_sample_id"])].append(row)

    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        events = by_label[label]
        parent_ids = sorted(
            {str(row["parent_sample_id"]) for row in events}
        )
        missing_events = sum(not bool(row["rd_present"]) for row in events)
        parents_with_missing = sum(
            any(not bool(row["rd_present"]) for row in by_parent[parent_id])
            for parent_id in parent_ids
        )
        parents_with_no_rd = sum(
            all(not bool(row["rd_present"]) for row in by_parent[parent_id])
            for parent_id in parent_ids
        )
        result[label] = {
            "event_count": len(events),
            "rd_missing_event_count": int(missing_events),
            "rd_missing_event_fraction": float(missing_events / len(events))
            if events
            else None,
            "parent_count": len(parent_ids),
            "parent_with_any_missing_rd_count": int(parents_with_missing),
            "parent_with_any_missing_rd_fraction": float(
                parents_with_missing / len(parent_ids)
            )
            if parent_ids
            else None,
            "parent_with_no_rd_event_count": int(parents_with_no_rd),
        }

    target_events = sum(result[label]["event_count"] for label in KNOWN_LABELS[:3])
    target_missing_events = sum(
        result[label]["rd_missing_event_count"] for label in KNOWN_LABELS[:3]
    )
    target_parents = sum(result[label]["parent_count"] for label in KNOWN_LABELS[:3])
    target_parents_with_missing = sum(
        result[label]["parent_with_any_missing_rd_count"]
        for label in KNOWN_LABELS[:3]
    )
    clutter = result["ClutterTarget"]
    return {
        "by_base_label": result,
        "known_target_aggregate": {
            "event_count": target_events,
            "rd_missing_event_count": target_missing_events,
            "rd_missing_event_fraction": float(
                target_missing_events / target_events
            ),
            "parent_count": target_parents,
            "parent_with_any_missing_rd_count": target_parents_with_missing,
            "parent_with_any_missing_rd_fraction": float(
                target_parents_with_missing / target_parents
            ),
        },
        "support_gap": {
            "target_missing_events_exist": target_missing_events > 0,
            "clutter_missing_event_count": int(
                clutter["rd_missing_event_count"]
            ),
            "clutter_missing_events_absent": bool(
                clutter["rd_missing_event_count"] == 0
            ),
            "all_event_missing_parent_count": int(
                sum(
                    row["parent_with_no_rd_event_count"]
                    for row in result.values()
                )
            ),
            "all_events_missing_counterfactual_is_out_of_parent_support": bool(
                all(
                    row["parent_with_no_rd_event_count"] == 0
                    for row in result.values()
                )
            ),
        },
        "interpretation": (
            "RD-missing events occur in known targets but not in the 33 "
            "Clutter parents.  Therefore the all-events-missing arm probes a "
            "known class-support gap and cannot estimate in-distribution "
            "missing-RD performance."
        ),
    }


def _stratified_effects(
    original: Sequence[Mapping[str, Any]],
    ablated: Sequence[Mapping[str, Any]],
    parent_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(original, ablated, scope="parent")
    dimensions: dict[str, dict[str, list[int]]] = {
        key: defaultdict(list)
        for key in (
            "base_label",
            "fold",
            "final_rd_frame_count",
            "rd_availability_fraction",
            "parent_length",
            "final_rd_missing",
        )
    }
    for index, row in enumerate(before):
        parent_id = str(row["parent_sample_id"])
        metadata = parent_metadata[parent_id]
        values = {
            "base_label": str(row["label_name"]),
            "fold": f"fold_{int(row['fold_id']):02d}",
            "final_rd_frame_count": _frame_bucket(
                int(metadata["final_rd_frame_count"])
            ),
            "rd_availability_fraction": _availability_bucket(
                float(metadata["rd_availability_fraction"])
            ),
            "parent_length": _parent_length_bucket(int(row["parent_length"])),
            "final_rd_missing": "missing"
            if bool(row["rd_quality_missing"])
            else "present",
        }
        for dimension, value in values.items():
            dimensions[dimension][value].append(index)
    return {
        dimension: {
            value: _effect_summary(
                [before[index] for index in indices],
                [after[index] for index in indices],
            )
            for value, indices in sorted(groups.items())
        }
        for dimension, groups in dimensions.items()
    }


def _changed_parent_records(
    original: Sequence[Mapping[str, Any]],
    ablated: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    before, after = _ordered_pair(original, ablated, scope="parent")
    result: list[dict[str, Any]] = []
    for left, right in zip(before, after, strict=True):
        if int(left["pred_id"]) == int(right["pred_id"]):
            continue
        left_ok = int(left["pred_id"]) == int(left["true_id"])
        right_ok = int(right["pred_id"]) == int(right["true_id"])
        result.append(
            {
                "parent_sample_id": str(left["parent_sample_id"]),
                "track_id": str(left["track_id"]),
                "fold_id": int(left["fold_id"]),
                "provenance_component_id": str(left["provenance_component_id"]),
                "label_name": str(left["label_name"]),
                "original_pred_id": int(left["pred_id"]),
                "ablated_pred_id": int(right["pred_id"]),
                "original_correct": left_ok,
                "ablated_correct": right_ok,
                "transition": "rescued_by_ablation"
                if (not left_ok and right_ok)
                else "harmed_by_ablation"
                if (left_ok and not right_ok)
                else "changed_without_correctness_change",
                "original_known_target_probability": float(
                    left["known_target_probability"]
                ),
                "ablated_known_target_probability": float(
                    right["known_target_probability"]
                ),
                "known_target_probability_shift": float(
                    right["known_target_probability"]
                    - left["known_target_probability"]
                ),
                "original_known_target_logit_margin": float(
                    left["known_target_logit_margin"]
                ),
                "ablated_known_target_logit_margin": float(
                    right["known_target_logit_margin"]
                ),
                "known_target_logit_margin_shift": float(
                    right["known_target_logit_margin"]
                    - left["known_target_logit_margin"]
                ),
                "final_rd_frame_count": int(left["rd_frame_count"]),
                "parent_length": int(left["parent_length"]),
            }
        )
    return result


def _top_margin_changes(
    original: Sequence[Mapping[str, Any]],
    ablated: Sequence[Mapping[str, Any]],
    *,
    limit: int = 25,
) -> list[dict[str, Any]]:
    before, after = _ordered_pair(original, ablated, scope="parent")
    rows = []
    for left, right in zip(before, after, strict=True):
        shift = float(right["known_target_logit_margin"]) - float(
            left["known_target_logit_margin"]
        )
        rows.append(
            {
                "parent_sample_id": str(left["parent_sample_id"]),
                "track_id": str(left["track_id"]),
                "fold_id": int(left["fold_id"]),
                "label_name": str(left["label_name"]),
                "original_pred_id": int(left["pred_id"]),
                "ablated_pred_id": int(right["pred_id"]),
                "original_known_target_logit_margin": float(
                    left["known_target_logit_margin"]
                ),
                "ablated_known_target_logit_margin": float(
                    right["known_target_logit_margin"]
                ),
                "known_target_logit_margin_shift": shift,
                "absolute_margin_shift": abs(shift),
                "final_rd_frame_count": int(left["rd_frame_count"]),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["absolute_margin_shift"]),
            str(row["parent_sample_id"]),
        )
    )
    return rows[: int(limit)]


def _parent_attribution_rows(
    original: Sequence[Mapping[str, Any]],
    content_zero: Sequence[Mapping[str, Any]],
    contract_missing: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    before, content = _ordered_pair(original, content_zero, scope="parent")
    before_again, missing = _ordered_pair(
        original, contract_missing, scope="parent"
    )
    if [row["parent_sample_id"] for row in before] != [
        row["parent_sample_id"] for row in before_again
    ]:
        raise ValueError("RD attribution parent sets differ")
    rows: list[dict[str, Any]] = []
    for base, no_content, no_modality in zip(
        before, content, missing, strict=True
    ):
        parent_id = str(base["parent_sample_id"])
        meta = metadata[parent_id]
        expected = int(base["true_id"])
        rows.append(
            {
                "parent_sample_id": parent_id,
                "track_id": str(base["track_id"]),
                "fold_id": int(base["fold_id"]),
                "provenance_component_id": str(base["provenance_component_id"]),
                "label_name": str(base["label_name"]),
                "true_id": expected,
                "parent_length": int(base["parent_length"]),
                **meta,
                "original_pred_id": int(base["pred_id"]),
                "original_correct": int(base["pred_id"]) == expected,
                "original_known_target_probability": float(
                    base["known_target_probability"]
                ),
                "original_known_target_logit_margin": float(
                    base["known_target_logit_margin"]
                ),
                "content_zero_pred_id": int(no_content["pred_id"]),
                "content_zero_correct": int(no_content["pred_id"]) == expected,
                "content_zero_known_target_probability": float(
                    no_content["known_target_probability"]
                ),
                "content_zero_known_target_logit_margin": float(
                    no_content["known_target_logit_margin"]
                ),
                "content_zero_margin_shift": float(
                    no_content["known_target_logit_margin"]
                    - base["known_target_logit_margin"]
                ),
                "contract_missing_pred_id": int(no_modality["pred_id"]),
                "contract_missing_correct": int(no_modality["pred_id"]) == expected,
                "contract_missing_known_target_probability": float(
                    no_modality["known_target_probability"]
                ),
                "contract_missing_known_target_logit_margin": float(
                    no_modality["known_target_logit_margin"]
                ),
                "contract_missing_margin_shift": float(
                    no_modality["known_target_logit_margin"]
                    - base["known_target_logit_margin"]
                ),
            }
        )
    return rows


def _write_csv_new_or_identical(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> str:
    if not rows:
        raise ValueError("RD parent attribution cannot be empty")
    stream = io.StringIO(newline="")
    fieldnames = list(rows[0])
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if list(row) != fieldnames:
            raise ValueError("RD parent attribution fields are inconsistent")
        writer.writerow(dict(row))
    text = stream.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to overwrite different RD attribution: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return "created"


def _fold_metrics(
    records_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        metrics: dict[str, Any] = {}
        for arm in ARMS:
            selected = [
                row
                for row in records_by_arm[arm]["parent"]
                if int(row["fold_id"]) == fold_id
            ]
            metrics[arm] = summarize_known_gate_records(
                selected, scope=f"v21e_rd_ablation_{arm}_fold_{fold_id:02d}"
            )
        rows.append(
            {
                "fold_id": fold_id,
                "held_out_parent_count": int(metrics[ORIGINAL]["support"]),
                "metrics": metrics,
                "content_zero_minus_original": _delta(
                    metrics[ORIGINAL], metrics[CONTENT_ZERO]
                ),
                "contract_missing_minus_original": _delta(
                    metrics[ORIGINAL], metrics[CONTRACT_MISSING]
                ),
            }
        )
    return rows


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    _contract, assignment = load_fold_contract(root)
    official = _load_arm(root, "treatment")
    component_by_parent = {
        str(row["parent_sample_id"]): str(row["provenance_component_id"])
        for row in assignment["assignments"]
    }
    if len(component_by_parent) != 986:
        raise ValueError("V21e RD ablation assignment does not cover 986 parents")

    sources: dict[str, str] = {}
    records_by_arm = {
        arm: {scope: [] for scope in SCOPES}
        for arm in ARMS
    }
    fold_audits: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold_records, audit = _run_fold(
            root,
            fold_id=fold_id,
            official_payload=official["predictions"][fold_id],
            component_by_parent=component_by_parent,
            sources=sources,
        )
        for arm in ARMS:
            for scope in SCOPES:
                records_by_arm[arm][scope].extend(fold_records[arm][scope])
        fold_audits.append(audit)

    official_parent_ids = sorted(
        str(row["parent_sample_id"])
        for row in official["records_by_scope"]["parent"]
    )
    replay_parent_ids = sorted(
        str(row["parent_sample_id"])
        for row in records_by_arm[ORIGINAL]["parent"]
    )
    if (
        official_parent_ids != replay_parent_ids
        or len(replay_parent_ids) != 986
        or len(set(replay_parent_ids)) != 986
        or sequence_sha256(replay_parent_ids)
        != assignment["known_parent_ids_sha256"]
    ):
        raise ValueError("V21e RD ablation does not cover every OOF parent once")

    metrics_by_arm = {
        ORIGINAL: official["metrics_by_scope"],
        CONTENT_ZERO: {
            scope: summarize_known_gate_records(
                records_by_arm[CONTENT_ZERO][scope],
                scope=f"oof_{CONTENT_ZERO}_{scope}",
            )
            for scope in SCOPES
        },
        CONTRACT_MISSING: {
            scope: summarize_known_gate_records(
                records_by_arm[CONTRACT_MISSING][scope],
                scope=f"oof_{CONTRACT_MISSING}_{scope}",
            )
            for scope in SCOPES
        },
    }
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    reliance: dict[str, dict[str, dict[str, float]]] = {}
    for arm in (CONTENT_ZERO, CONTRACT_MISSING):
        deltas[arm] = {
            scope: _delta(
                metrics_by_arm[ORIGINAL][scope], metrics_by_arm[arm][scope]
            )
            for scope in SCOPES
        }
        reliance[arm] = {
            scope: _negate(deltas[arm][scope]) for scope in SCOPES
        }

    parent_metadata = _parent_event_metadata(
        records_by_arm[ORIGINAL]["window"]
    )
    rd_missing_support = _rd_missing_support_audit(
        records_by_arm[ORIGINAL]["window"]
    )
    analyses: dict[str, Any] = {}
    for arm in (CONTENT_ZERO, CONTRACT_MISSING):
        analyses[arm] = {
            "parent_base_label_outcomes": _base_label_outcomes(
                records_by_arm[ORIGINAL]["parent"],
                records_by_arm[arm]["parent"],
                scope="parent",
            ),
            "eligible_prefix_base_label_outcomes": _base_label_outcomes(
                records_by_arm[ORIGINAL]["eligible_prefix"],
                records_by_arm[arm]["eligible_prefix"],
                scope="eligible_prefix",
            ),
            "paired_component_bootstrap": _component_bootstrap(
                assignment,
                records_by_arm[ORIGINAL]["parent"],
                records_by_arm[arm]["parent"],
            ),
            "exact_mcnemar_parent": _exact_mcnemar(
                records_by_arm[ORIGINAL]["parent"],
                records_by_arm[arm]["parent"],
                scope="parent",
            ),
            "stratified_parent_effects": _stratified_effects(
                records_by_arm[ORIGINAL]["parent"],
                records_by_arm[arm]["parent"],
                parent_metadata,
            ),
            "changed_parent_records": _changed_parent_records(
                records_by_arm[ORIGINAL]["parent"],
                records_by_arm[arm]["parent"],
            ),
            "largest_absolute_parent_margin_changes": _top_margin_changes(
                records_by_arm[ORIGINAL]["parent"],
                records_by_arm[arm]["parent"],
            ),
        }

    parent_rows = _parent_attribution_rows(
        records_by_arm[ORIGINAL]["parent"],
        records_by_arm[CONTENT_ZERO]["parent"],
        records_by_arm[CONTRACT_MISSING]["parent"],
        parent_metadata,
    )
    csv_path = root / PARENT_ATTRIBUTION_RELATIVE
    csv_status = _write_csv_new_or_identical(csv_path, parent_rows)

    source_after = {
        path: sha256_file(root / Path(path)) for path in sorted(sources)
    }
    if source_after != dict(sorted(sources.items())):
        raise AssertionError("a frozen V21e source changed during RD ablation")

    content_bootstrap = analyses[CONTENT_ZERO]["paired_component_bootstrap"]
    missing_bootstrap = analyses[CONTRACT_MISSING]["paired_component_bootstrap"]
    content_parent_delta = deltas[CONTENT_ZERO]["parent"]
    missing_parent_delta = deltas[CONTRACT_MISSING]["parent"]
    content_upper = float(
        content_bootstrap["treatment_minus_control"]["macro_f1"]["upper_95"]
    )
    missing_upper = float(
        missing_bootstrap["treatment_minus_control"]["macro_f1"]["upper_95"]
    )
    report = {
        "status": "completed_read_only_diagnostic",
        "schema_version": SCHEMA_VERSION,
        "experiment": "V21e frozen OOF KnownGate RD modality read-only ablation",
        "scope": "known train capture/content-grouped OOF only; Unknown/validation/test excluded",
        "arms": {
            ORIGINAL: "exact replay of frozen V21e OOF Treatment",
            CONTENT_ZERO: (
                "RD embedding forced to exact zero while original causal RD quality "
                "metadata is retained; attribution-only off-manifold probe"
            ),
            CONTRACT_MISSING: (
                "RD embedding forced to exact zero and every active event receives "
                "the production missing-RD quality vector [0,0,0,1,0,0,0]"
            ),
        },
        "official_replay": {
            "all_folds_passed": all(
                all(row["passed"] for row in audit["official_replay"].values())
                for audit in fold_audits
            ),
            "probability_absolute_tolerance": REPLAY_PROBABILITY_ATOL,
            "fold_audits": fold_audits,
        },
        "metrics_by_arm_and_scope": metrics_by_arm,
        "ablated_minus_original": deltas,
        "original_minus_ablated_conditional_reliance": reliance,
        "arm_analyses": analyses,
        "fold_summaries": _fold_metrics(records_by_arm),
        "rd_missing_support_audit": rd_missing_support,
        "parent_attribution": {
            "path": relative_path(root, csv_path),
            "sha256": sha256_file(csv_path),
            "write_contract": "create_new_or_reuse_byte_identical",
            "row_count": len(parent_rows),
        },
        "decision": {
            "rd_content_removal_parent_macro_f1_delta": float(
                content_parent_delta["macro_f1"]
            ),
            "rd_content_removal_macro_f1_bootstrap_upper_95_below_zero": bool(
                content_upper < 0.0
            ),
            "rd_content_removal_practical_drop_at_least_0_02": bool(
                content_parent_delta["macro_f1"] <= -0.02
            ),
            "contract_rd_removal_parent_macro_f1_delta": float(
                missing_parent_delta["macro_f1"]
            ),
            "contract_rd_removal_macro_f1_bootstrap_upper_95_below_zero": bool(
                missing_upper < 0.0
            ),
            "contract_rd_removal_practical_drop_at_least_0_02": bool(
                missing_parent_delta["macro_f1"] <= -0.02
            ),
            "rd_content_conditional_reliance_observed": bool(
                content_parent_delta["macro_f1"] <= -0.02
                and content_upper < 0.0
            ),
            "rd_quality_contract_has_class_support_gap": bool(
                rd_missing_support["support_gap"][
                    "target_missing_events_exist"
                ]
                and rd_missing_support["support_gap"][
                    "clutter_missing_events_absent"
                ]
            ),
            "all_events_missing_counterfactual_is_out_of_parent_support": bool(
                rd_missing_support["support_gap"][
                    "all_events_missing_counterfactual_is_out_of_parent_support"
                ]
            ),
            "diagnostic_only_not_an_adoption_gate": True,
            "advance_to_test_or_deployment": False,
        },
        "interpretation_contract": {
            "frozen_model_conditional_reliance_not_retrained_no_rd_performance": True,
            "content_zero_quality_preserved_is_off_manifold": True,
            "contract_missing_is_legal_input_but_all_events_missing_is_distribution_shift": True,
            "nonlinear_arm_differences_are_not_an_additive_feature_decomposition": True,
            "fold_models_are_not_deployment_checkpoints": True,
            "unknown_is_not_a_training_class": True,
        },
        "source_integrity": {
            "source_file_count": len(sources),
            "pre_and_post_sha256_equal": True,
            "source_sha256": dict(sorted(sources.items())),
            "model_states_unchanged": all(
                audit["model_state_unchanged"] for audit in fold_audits
            ),
            "training_or_optimizer_step_count": 0,
            "checkpoint_selection_count": 0,
            "threshold_fit_count": 0,
            "unknown_access_count": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
        },
        "multiple_comparison_note": (
            "Two predeclared ablation arms and multiple descriptive strata are "
            "reported; strata are exploratory and are not separate adoption tests."
        ),
        "test": "未执行",
        "threshold_fitting": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / REPORT_RELATIVE
    status = write_new_or_identical(output, report)
    report["report_path"] = str(output)
    report["report_write_status"] = status
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen V21e OOF read-only RD modality ablation."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.project_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
