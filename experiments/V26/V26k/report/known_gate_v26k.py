"""Run the V26k frozen L2 strict-open-set knownness diagnostic.

This module is inference-only.  It reloads the frozen V26c DBB specialist for
one fold at a time, fits a class-conditional reference from that fold's
train-only Drone/Bird/Balloon parents, and then scores the already frozen V26f
mixed validation/test population.  UnknownTarget is not available to the
model, normalization, reference fit, threshold choice, or checkpoint choice;
it is joined only after the score and the train-only diagnostic operating
point have been frozen.

The expensive work runs in one short-lived CPU subprocess per fold.  This is
an execution boundary for Windows/PyTorch allocator safety and does not alter
the model, input, ordering, or numerical score protocol.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.data import collate_point_sequences
from radar_ln3.known_gate_v26f import CONTRACT as V26F_CONTRACT, FOLD_COUNT as V26F_FOLD_COUNT, FEATURE_CACHE_MAX_BYTES, fold_contract_entry, require_bound_file as v26f_require_bound_file, validate_prepared_contract as validate_v26f_prepared_contract
from radar_ln3.known_gate_v26k import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONTRACT, DBB_LABELS, DETAIL_CSV, DETAIL_FIELDS, EXPECTED_PARENT_COUNTS as V26K_EXPECTED_PARENT_COUNTS, FOLD_COUNT, L1_TARGET_ROUTE, L2_CLASSES, L2_EPOCH, MAX_DETAIL_BYTES, MAX_ISOLATED_FOLD_OUTPUT_BYTES, MAX_REPORT_BYTES, OOD_SCORE_POLICY, READINESS, REPORT, REPORT_SCHEMA, UNKNOWN_BASE_LABEL, binary_ranking_metrics, class_summary, fit_class_conditional_reference, implementation_sha256, mahalanobis_scores, parent_id_sequence_sha256, read_v26f_route_rows, read_json, relative_path, sha256_file, state_matrix_sha256, train_only_retention_threshold, validate_prepared_contract, write_immutable_text
from radar_ln3.l1_l2_raw_routing import (
    FrozenL2MixedAuditDataset,
    l2_model_inputs,
)
from radar_ln3.task_models import (
    build_paired_model_for_config,
    build_task_dataset,
    collate_task_sequences,
    task_config_identity_sha256,
)
from train.known_gate_common import _state_dict_sha256
from train.known_gate_v26c_oof import (
    CONTRACT as V26C_CONTRACT,
    L2_CHECKPOINT_SCHEMA,
    READINESS as V26C_READINESS,
    _load_checkpoint,
    _runtime_l2_config,
    _source_protocols,
    fold_record,
    l2_train_micro_cache_path,
    l2_train_physical_cache_path,
)


V26C_CONTRACT_PATH = Path(str(V26C_CONTRACT))
V26C_READINESS_PATH = Path(str(V26C_READINESS))
V26F_CONTRACT_PATH = Path(str(V26F_CONTRACT))

EPS = 1.0e-12
L2_LABEL_TO_ID = {name: index for index, name in enumerate(L2_CLASSES)}
MAX_JSON_BYTES = 64 * 1024 * 1024


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if float(denominator) else None




def _clear_dataset_cache(dataset: Any) -> None:
    """Release decoded RD archives after each bounded parent boundary."""

    backing = getattr(dataset, "base", dataset)
    cache = getattr(backing, "_tensor_cache", None)
    items = getattr(cache, "_items", None)
    if items is not None and hasattr(items, "clear"):
        items.clear()


def _classifier_embedding(
    model: Any,
    output: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact 48-D input to the frozen L2 final Linear layer.

    V26c is required to be the ordinary ``LayerNorm -> Linear`` readout.  A
    future angular/prototype configuration is rejected rather than silently
    treating a different vector as the requested embedding.
    """

    state = output.get("liquid_final_state")
    classifier = getattr(model, "classifier", None)
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise ValueError("V26k L2 output lacks a [batch, hidden] final state")
    if not isinstance(classifier, torch.nn.Sequential) or len(classifier) != 2:
        raise ValueError("V26k requires the frozen LayerNorm -> Linear L2 classifier")
    normalizer, linear = classifier[0], classifier[1]
    if not isinstance(normalizer, torch.nn.LayerNorm) or not isinstance(linear, torch.nn.Linear):
        raise ValueError("V26k L2 classifier is not LayerNorm -> Linear")
    if linear.in_features != 48 or state.shape[-1] != 48:
        raise ValueError("V26k L2 embedding dimension is not 48")
    embedding = normalizer(state)
    reconstructed = linear(embedding)
    declared = output.get("window_logits")
    if (
        not isinstance(declared, torch.Tensor)
        or declared.shape != reconstructed.shape
        or not torch.allclose(reconstructed, declared, rtol=1.0e-5, atol=1.0e-6)
    ):
        raise ValueError("V26k L2 embedding does not reconstruct the frozen final logits")
    if not torch.isfinite(embedding).all() or not torch.isfinite(declared).all():
        raise ValueError("V26k L2 embedding/logits are non-finite")
    return embedding, declared


def _forward_parent_dataset(
    dataset: Any,
    *,
    collate: Callable[[list[Any]], Mapping[str, Any]],
    model: Any,
    device: torch.device,
    expected_parent_ids: Sequence[str],
    scope: str,
) -> list[dict[str, Any]]:
    """Forward one parent at a time and retain only compact final-state rows."""

    expected = tuple(str(value) for value in expected_parent_ids)
    observed = tuple(str(value) for value in dataset.parent_ids)
    if observed != expected:
        raise ValueError(f"V26k {scope} parent order differs from its frozen binding")
    rows: list[dict[str, Any]] = []
    allowed = set(L2_CLASSES) if scope == "train_reference" else {*DBB_LABELS, "ClutterTarget", UNKNOWN_BASE_LABEL}
    for index, requested_parent_id in enumerate(expected):
        item = dataset[index]
        batch = collate([item])
        tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            output = model(**l2_model_inputs(tensors))
            embedding, logits = _classifier_embedding(model, output)
            probabilities = torch.softmax(logits, dim=-1)
        parent_id = str(batch["parent_sample_ids"][0])
        if parent_id != requested_parent_id:
            raise ValueError(f"V26k {scope} parent identity changed during forward")
        label = str(batch["label_names"][0])
        if label not in allowed:
            raise ValueError(f"V26k {scope} contains an unsupported label: {label}")
        length = int(batch["lengths"][0].item())
        if length < 1 or embedding.shape != (1, 48) or probabilities.shape != (1, 3):
            raise ValueError(f"V26k {scope} produced an invalid parent output: {parent_id}")
        probability_row = probabilities[0].detach().to("cpu", dtype=torch.float64).numpy().copy()
        embedding_row = embedding[0].detach().to("cpu", dtype=torch.float64).numpy().copy()
        if not np.isfinite(probability_row).all() or not np.isfinite(embedding_row).all():
            raise ValueError(f"V26k {scope} emitted a non-finite parent output: {parent_id}")
        top_id = int(np.argmax(probability_row))
        ordered = np.sort(probability_row)[::-1]
        base = dataset.base
        first = base._groups[parent_id].iloc[0]
        rows.append(
            {
                "parent_sample_id": parent_id,
                "label_name": label,
                "embedding": embedding_row,
                "l2_predicted_class": L2_CLASSES[top_id],
                "l2_subclass_probability": float(probability_row[top_id]),
                "l2_subclass_margin": float(ordered[0] - ordered[1]),
                "max_softmax_unknownness": float(1.0 - probability_row[top_id]),
                "parent_length": int(length),
                "track_id": str(first.get("track_id", "")),
                "capture_id": str(first.get("capture_id", "")),
                "provenance_component_id": str(first.get("provenance_component_id", "")),
                "rd_present_at_final": bool(batch["rd_frame_mask"][0, length - 1].any().item()),
                "micro_doppler_valid_at_final": bool(
                    batch["micro_doppler_valid"][0, length - 1].item()
                ),
                "dt_valid_at_final": bool(batch["tr_dt_valid"][0, length - 1].item()),
            }
        )
        del tensors, output, embedding, logits, probabilities, batch, item
        _clear_dataset_cache(dataset)
        if (index + 1) % 16 == 0:
            gc.collect()
    if tuple(str(row["parent_sample_id"]) for row in rows) != expected:
        raise ValueError(f"V26k {scope} parent coverage differs after forward")
    return rows


def _fit_reference(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    if not rows:
        raise ValueError("V26k cannot fit an empty L2 train reference")
    states = np.stack(
        [np.asarray(row["embedding"], dtype=np.float64) for row in rows], axis=0
    )
    class_ids = np.asarray(
        [L2_LABEL_TO_ID[str(row["label_name"])] for row in rows], dtype=np.int64
    )
    fitted = fit_class_conditional_reference(states, class_ids)
    train_scores, _ = mahalanobis_scores(states, fitted)
    threshold = train_only_retention_threshold(train_scores)
    reference = {
        "fit_population": "per_fold_train_known_only_dbb_parent_final_embeddings",
        "parent_count": int(len(rows)),
        "parent_ids_sha256": parent_id_sequence_sha256(
            [str(row["parent_sample_id"]) for row in rows]
        ),
        "state_matrix_sha256": state_matrix_sha256(states),
        "state_dimension": int(states.shape[1]),
        "class_counts": dict(fitted["class_counts"]),
        "pooled_degrees_of_freedom": int(fitted["pooled_degrees_of_freedom"]),
        "variance_floor": float(fitted["variance_floor"]),
        "train_score_summary": {
            "support": int(len(train_scores)),
            "mean": float(train_scores.mean()),
            "std": float(train_scores.std(ddof=0)),
            "median": float(np.median(train_scores)),
            "p95": float(np.quantile(train_scores, 0.95)),
            "maximum": float(train_scores.max()),
        },
        "unknown_reference_fit_exposure": 0,
    }
    return fitted, {"reference": reference, "threshold": threshold}, train_scores


def _score_mixed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fitted: Mapping[str, Any],
    threshold: Mapping[str, Any],
    route_rows: Mapping[tuple[str, int, str], Mapping[str, Any]],
    split: str,
    fold_id: int,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("V26k cannot score an empty mixed population")
    states = np.stack(
        [np.asarray(row["embedding"], dtype=np.float64) for row in rows], axis=0
    )
    scores, closest = mahalanobis_scores(states, fitted)
    limit = float(threshold["threshold"])
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        parent_id = str(raw["parent_sample_id"])
        key = (str(split), int(fold_id), parent_id)
        route = route_rows.get(key)
        if route is None:
            raise ValueError(f"V26k missing frozen V26f route row: {key}")
        if str(route["label_name"]) != str(raw["label_name"]):
            raise ValueError(f"V26k V26f route label differs from mixed input: {key}")
        accepted = bool(float(scores[index]) <= limit + EPS)
        l1_target_accepted = bool(route["known_target_accepted"])
        l1_clutter_accepted = bool(route["known_clutter_accepted"])
        l1_route = str(route["l1_route"])
        combined = bool(
            l1_route == L1_TARGET_ROUTE and l1_target_accepted and accepted
        )
        result.append(
            {
                "split": str(split),
                "fold_id": int(fold_id),
                "parent_sample_id": parent_id,
                "track_id": str(raw["track_id"]),
                "capture_id": str(raw["capture_id"]),
                "provenance_component_id": str(raw["provenance_component_id"]),
                "label_name": str(raw["label_name"]),
                "is_unknown_truth": bool(str(raw["label_name"]) == UNKNOWN_BASE_LABEL),
                "l1_route": l1_route,
                "l2_predicted_class": str(raw["l2_predicted_class"]),
                "l2_subclass_probability": float(raw["l2_subclass_probability"]),
                "l2_subclass_margin": float(raw["l2_subclass_margin"]),
                "closest_reference_class": L2_CLASSES[int(closest[index])],
                "l2_knownness_score": float(scores[index]),
                "max_softmax_unknownness": float(raw["max_softmax_unknownness"]),
                "known_target_accepted": l1_target_accepted,
                "known_clutter_accepted": l1_clutter_accepted,
                "l2_gate_accepted": accepted,
                "combined_known_target_accepted": combined,
                "l2_forwarded": True,
                "accepted_by_train_only_threshold": accepted,
                "parent_length": int(raw["parent_length"]),
                "rd_present_at_final": bool(raw["rd_present_at_final"]),
                "micro_doppler_valid_at_final": bool(raw["micro_doppler_valid_at_final"]),
                "dt_valid_at_final": bool(raw["dt_valid_at_final"]),
            }
        )
    return result


def _ranking(
    known_rows: Sequence[Mapping[str, Any]],
    unknown_rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, Any] | None:
    return binary_ranking_metrics(
        [float(row[key]) for row in known_rows],
        [float(row[key]) for row in unknown_rows],
    )


def _length_stratum(length: int) -> str:
    value = int(length)
    if value <= 5:
        return "01-05"
    if value <= 10:
        return "06-10"
    if value <= 20:
        return "11-20"
    return "21-29"


def _capture_strata(rows: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["capture_id"])].append(row)
    result: list[dict[str, Any]] = []
    for capture_id, members in sorted(grouped.items()):
        known = [row for row in members if str(row["label_name"]) in DBB_LABELS]
        unknown = [row for row in members if str(row["label_name"]) == UNKNOWN_BASE_LABEL]
        target_known = [row for row in known if str(row["l1_route"]) == L1_TARGET_ROUTE]
        target_unknown = [
            row for row in unknown if str(row["l1_route"]) == L1_TARGET_ROUTE
        ]
        result.append(
            {
                "capture_id": capture_id,
                "parent_support": int(len(members)),
                "known_parent_support": int(len(known)),
                "unknown_parent_support": int(len(unknown)),
                "l1_target_route_known_support": int(len(target_known)),
                "l1_target_route_unknown_support": int(len(target_unknown)),
                "known_score_median": (
                    float(np.median([float(row["l2_knownness_score"]) for row in known]))
                    if known
                    else None
                ),
                "unknown_score_median": (
                    float(np.median([float(row["l2_knownness_score"]) for row in unknown]))
                    if unknown
                    else None
                ),
                "target_route_unknown_vs_known_auroc": (
                    _ranking(target_known, target_unknown, key="l2_knownness_score")[
                        "auroc_unknown_vs_known"
                    ]
                    if target_known and target_unknown
                    else None
                ),
                "target_route_unknown_rejection_rate": (
                    _safe_divide(
                        sum(
                            int(
                                not bool(row["l2_gate_accepted"])
                                and bool(row["known_target_accepted"])
                            )
                            for row in target_unknown
                        ),
                        len(target_unknown),
                    )
                    if target_unknown
                    else None
                ),
                "threshold": float(threshold),
            }
        )
    return result


def _length_strata(rows: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_length_stratum(int(row["parent_length"]))].append(row)
    result: list[dict[str, Any]] = []
    for name, members in sorted(grouped.items()):
        known = [row for row in members if str(row["label_name"]) in DBB_LABELS]
        unknown = [row for row in members if str(row["label_name"]) == UNKNOWN_BASE_LABEL]
        target_known = [row for row in known if str(row["l1_route"]) == L1_TARGET_ROUTE]
        target_unknown = [
            row for row in unknown if str(row["l1_route"]) == L1_TARGET_ROUTE
        ]
        result.append(
            {
                "parent_length_stratum": name,
                "parent_support": int(len(members)),
                "known_support": int(len(known)),
                "unknown_support": int(len(unknown)),
                "target_route_known_support": int(len(target_known)),
                "target_route_unknown_support": int(len(target_unknown)),
                "target_route_auroc": (
                    _ranking(target_known, target_unknown, key="l2_knownness_score")[
                        "auroc_unknown_vs_known"
                    ]
                    if target_known and target_unknown
                    else None
                ),
                "dbb_threshold_retention": _safe_divide(
                    sum(bool(row["l2_gate_accepted"]) for row in known), len(known)
                ),
                "unknown_threshold_rejection": _safe_divide(
                    sum(not bool(row["l2_gate_accepted"]) for row in unknown), len(unknown)
                ),
                "threshold": float(threshold),
            }
        )
    return result


def _operating_point(rows: Sequence[Mapping[str, Any]], threshold: Mapping[str, Any]) -> dict[str, Any]:
    limit = float(threshold["threshold"])
    dbb = [row for row in rows if str(row["label_name"]) in DBB_LABELS]
    clutter = [row for row in rows if str(row["label_name"]) == "ClutterTarget"]
    unknown = [row for row in rows if str(row["label_name"]) == UNKNOWN_BASE_LABEL]
    target_dbb = [row for row in dbb if str(row["l1_route"]) == L1_TARGET_ROUTE]
    target_unknown = [
        row for row in unknown if str(row["l1_route"]) == L1_TARGET_ROUTE
    ]
    target_unknown_l1_accepted = [
        row for row in target_unknown if bool(row["known_target_accepted"])
    ]
    by_class: dict[str, Any] = {}
    for label in DBB_LABELS:
        members = [row for row in dbb if str(row["label_name"]) == label]
        by_class[label] = {
            "support": int(len(members)),
            "threshold_retention": _safe_divide(
                sum(bool(row["l2_gate_accepted"]) for row in members), len(members)
            ),
            "subclass_accuracy": _safe_divide(
                sum(str(row["l2_predicted_class"]) == label for row in members), len(members)
            ),
            "combined_known_target_acceptance_recall": _safe_divide(
                sum(bool(row["combined_known_target_accepted"]) for row in members), len(members)
            ),
        }
    l1_recall = _safe_divide(
        sum(bool(row["known_target_accepted"]) for row in dbb), len(dbb)
    )
    combined_recall = _safe_divide(
        sum(bool(row["combined_known_target_accepted"]) for row in dbb), len(dbb)
    )
    return {
        "train_only_threshold": limit,
        "threshold_policy": threshold.get("threshold_fit_population"),
        "threshold_train_support": int(threshold["train_support"]),
        "threshold_train_retention": float(threshold["train_retention"]),
        "dbb_support_all_routes": int(len(dbb)),
        "dbb_threshold_retention_all_routes": _safe_divide(
            sum(bool(row["l2_gate_accepted"]) for row in dbb), len(dbb)
        ),
        "dbb_l1_target_route_support": int(len(target_dbb)),
        "dbb_threshold_retention_l1_target_route": _safe_divide(
            sum(bool(row["l2_gate_accepted"]) for row in target_dbb), len(target_dbb)
        ),
        "dbb_l2_subclass_accuracy_all_routes": _safe_divide(
            sum(str(row["l2_predicted_class"]) == str(row["label_name"]) for row in dbb), len(dbb)
        ),
        "l1_known_target_recall_before_l2_gate": l1_recall,
        "known_target_recall_after_l1_l2_gate": combined_recall,
        "known_target_recall_after_l2_on_l1_target_route": _safe_divide(
            sum(bool(row["known_target_accepted"]) and bool(row["l2_gate_accepted"]) for row in target_dbb),
            len(target_dbb),
        ),
        "per_dbb_class": by_class,
        "unknown_support_all_routes": int(len(unknown)),
        "unknown_rejection_rate_all_routes_by_l2_score": _safe_divide(
            sum(not bool(row["l2_gate_accepted"]) for row in unknown), len(unknown)
        ),
        "unknown_false_accept_rate_all_routes_by_l2_score": _safe_divide(
            sum(bool(row["l2_gate_accepted"]) for row in unknown), len(unknown)
        ),
        "unknown_l1_target_route_support": int(len(target_unknown)),
        "unknown_l1_target_route_rejection_rate_by_l2_score": _safe_divide(
            sum(not bool(row["l2_gate_accepted"]) for row in target_unknown), len(target_unknown)
        ),
        "unknown_l1_target_route_false_accept_rate_by_l2_score": _safe_divide(
            sum(bool(row["l2_gate_accepted"]) for row in target_unknown), len(target_unknown)
        ),
        "unknown_l1_target_accepted_support": int(len(target_unknown_l1_accepted)),
        "unknown_rejection_rate_given_l1_target_acceptance": _safe_divide(
            sum(not bool(row["l2_gate_accepted"]) for row in target_unknown_l1_accepted),
            len(target_unknown_l1_accepted),
        ),
        "unknown_effective_false_accept_rate_given_l1_target_acceptance": _safe_divide(
            sum(bool(row["l2_gate_accepted"]) for row in target_unknown_l1_accepted),
            len(target_unknown_l1_accepted),
        ),
        "clutter_support": int(len(clutter)),
        "clutter_threshold_acceptance_rate": _safe_divide(
            sum(bool(row["l2_gate_accepted"]) for row in clutter), len(clutter)
        ),
        "all_mixed_l2_forwarded": bool(rows) and all(bool(row["l2_forwarded"]) for row in rows),
        "unknown_is_not_a_l2_model_class": True,
    }


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: Mapping[str, Any],
    seed: int,
) -> dict[str, Any] | None:
    """Parent bootstrap for the primary L1-target-route ranking and gate rates."""

    known = [
        row
        for row in rows
        if str(row["label_name"]) in DBB_LABELS and str(row["l1_route"]) == L1_TARGET_ROUTE
    ]
    unknown = [
        row
        for row in rows
        if str(row["label_name"]) == UNKNOWN_BASE_LABEL
        and str(row["l1_route"]) == L1_TARGET_ROUTE
    ]
    dbb = [row for row in rows if str(row["label_name"]) in DBB_LABELS]
    if not known or not unknown or not dbb:
        return None
    rng = np.random.default_rng(int(seed))
    known_indices = np.arange(len(known), dtype=np.int64)
    unknown_indices = np.arange(len(unknown), dtype=np.int64)
    dbb_indices = np.arange(len(dbb), dtype=np.int64)
    values: dict[str, list[float]] = {
        "auroc_unknown_vs_known": [],
        "average_precision_unknown": [],
        "dbb_l2_retention": [],
        "dbb_combined_known_target_recall": [],
        "unknown_rejection_given_l1_target_acceptance": [],
    }
    for _ in range(int(BOOTSTRAP_REPLICATES)):
        sampled_known = rng.choice(known_indices, size=len(known_indices), replace=True)
        sampled_unknown = rng.choice(unknown_indices, size=len(unknown_indices), replace=True)
        sampled_dbb = rng.choice(dbb_indices, size=len(dbb_indices), replace=True)
        ranking = binary_ranking_metrics(
            [float(known[index]["l2_knownness_score"]) for index in sampled_known],
            [float(unknown[index]["l2_knownness_score"]) for index in sampled_unknown],
        )
        values["auroc_unknown_vs_known"].append(float(ranking["auroc_unknown_vs_known"]))
        values["average_precision_unknown"].append(float(ranking["average_precision_unknown"]))
        sampled_known_rows = [known[index] for index in sampled_known]
        sampled_unknown_rows = [unknown[index] for index in sampled_unknown]
        sampled_dbb_rows = [dbb[index] for index in sampled_dbb]
        values["dbb_l2_retention"].append(
            float(np.mean([bool(row["l2_gate_accepted"]) for row in sampled_known_rows]))
        )
        values["dbb_combined_known_target_recall"].append(
            float(np.mean([bool(row["combined_known_target_accepted"]) for row in sampled_dbb_rows]))
        )
        accepted_unknown = [
            row
            for row in sampled_unknown_rows
            if bool(row["known_target_accepted"])
        ]
        values["unknown_rejection_given_l1_target_acceptance"].append(
            float(
                np.mean([not bool(row["l2_gate_accepted"]) for row in accepted_unknown])
            )
            if accepted_unknown
            else float("nan")
        )
    result_metrics: dict[str, Any] = {}
    for key, raw in values.items():
        array = np.asarray(raw, dtype=np.float64)
        finite = array[np.isfinite(array)]
        result_metrics[key] = (
            {
                "median": float(np.median(finite)),
                "lower_95": float(np.quantile(finite, 0.025)),
                "upper_95": float(np.quantile(finite, 0.975)),
                "finite_replicates": int(len(finite)),
            }
            if len(finite)
            else None
        )
    return {
        "method": "parent bootstrap; primary L1-target-route known and Unknown rows resampled independently",
        "replicates": int(BOOTSTRAP_REPLICATES),
        "seed": int(seed),
        "confidence_level": 0.95,
        "known_parent_support": int(len(known)),
        "unknown_parent_support": int(len(unknown)),
        "dbb_parent_support": int(len(dbb)),
        "threshold": float(threshold["threshold"]),
        "metrics": result_metrics,
    }


def _fold_metrics(rows: Sequence[Mapping[str, Any]], threshold: Mapping[str, Any]) -> dict[str, Any]:
    known_target_route = [
        row
        for row in rows
        if str(row["l1_route"]) == L1_TARGET_ROUTE
        and str(row["label_name"]) in DBB_LABELS
    ]
    unknown_target_route = [
        row
        for row in rows
        if str(row["l1_route"]) == L1_TARGET_ROUTE
        and str(row["label_name"]) == UNKNOWN_BASE_LABEL
    ]
    all_known = [row for row in rows if str(row["label_name"]) in DBB_LABELS]
    all_unknown = [row for row in rows if str(row["label_name"]) == UNKNOWN_BASE_LABEL]
    clutter = [row for row in rows if str(row["label_name"]) == "ClutterTarget"]
    return {
        "support": int(len(rows)),
        "label_counts": dict(Counter(str(row["label_name"]) for row in rows)),
        "route_counts": dict(Counter(str(row["l1_route"]) for row in rows)),
        "primary_l1_target_route": {
            "known_support": int(len(known_target_route)),
            "unknown_support": int(len(unknown_target_route)),
            "mahalanobis": _ranking(
                known_target_route, unknown_target_route, key="l2_knownness_score"
            ),
            "msp_comparator": _ranking(
                known_target_route, unknown_target_route, key="max_softmax_unknownness"
            ),
        },
        "secondary_all_routes": {
            "known_support": int(len(all_known)),
            "unknown_support": int(len(all_unknown)),
            "mahalanobis": _ranking(all_known, all_unknown, key="l2_knownness_score"),
            "msp_comparator": _ranking(
                all_known, all_unknown, key="max_softmax_unknownness"
            ),
        },
        "secondary_unknown_vs_clutter": {
            "clutter_support": int(len(clutter)),
            "unknown_support": int(len(all_unknown)),
            "mahalanobis": _ranking(clutter, all_unknown, key="l2_knownness_score"),
            "msp_comparator": _ranking(
                clutter, all_unknown, key="max_softmax_unknownness"
            ),
        },
        "operating_point": _operating_point(rows, threshold),
        "per_label_mahalanobis": class_summary(rows, score_key="l2_knownness_score"),
        "per_label_msp_unknownness": class_summary(
            rows, score_key="max_softmax_unknownness"
        ),
        "capture_strata": _capture_strata(rows, float(threshold["threshold"])),
        "parent_length_strata": _length_strata(rows, float(threshold["threshold"])),
        "parent_bootstrap": _bootstrap(
            rows,
            threshold=threshold,
            seed=int(BOOTSTRAP_SEED),
        ),
    }


def _aggregate_values(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("V26k aggregate values are invalid")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "fold_count": int(len(array)),
    }


def _aggregate_fold_metrics(
    folds: Sequence[Mapping[str, Any]], *, split: str
) -> dict[str, Any]:
    paths = (
        ("primary_l1_target_route", "mahalanobis", "auroc_unknown_vs_known"),
        ("primary_l1_target_route", "mahalanobis", "average_precision_unknown"),
        ("primary_l1_target_route", "msp_comparator", "auroc_unknown_vs_known"),
        ("primary_l1_target_route", "msp_comparator", "average_precision_unknown"),
        (
            "operating_point",
            "dbb_threshold_retention_l1_target_route",
            None,
        ),
        ("operating_point", "known_target_recall_after_l1_l2_gate", None),
        (
            "operating_point",
            "unknown_effective_false_accept_rate_given_l1_target_acceptance",
            None,
        ),
    )
    result: dict[str, Any] = {"split": str(split), "fold_count": int(len(folds)), "metrics": {}}
    for first, second, third in paths:
        values: list[float] = []
        for fold in folds:
            value: Any = fold[split]["metrics"][first]
            if third is None:
                value = value[second]
            else:
                value = value[second][third]
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        result["metrics"][
            ".".join(value for value in (first, second, third) if value is not None)
        ] = _aggregate_values(values) if values else None
    return result


def _compact_split(split_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_count": int(split_result["parent_count"]),
        "parent_ids_sha256": str(split_result["parent_ids_sha256"]),
        "label_counts": dict(split_result["metrics"]["label_counts"]),
        "route_counts": dict(split_result["metrics"]["route_counts"]),
        "metrics": dict(split_result["metrics"]),
    }


def _mixed_cache_bindings(
    root: Path,
    contract: Mapping[str, Any],
    *,
    split: str,
) -> tuple[Path, Path]:
    rows = [
        dict(row)
        for row in list(dict(contract.get("data") or {}).get("cache_bindings") or ())
        if str(dict(row).get("split")) == str(split)
    ]
    if len(rows) != 1:
        raise ValueError(f"V26k V26f mixed cache binding is missing or repeated: {split}")
    row = rows[0]
    physical = dict(row.get("l2_rd_physical4") or {})
    micro = dict(row.get("l2_micro_doppler8") or {})
    physical_path = v26f_require_bound_file(
        root,
        str(physical.get("path") or ""),
        f"V26f {split} L2 RD cache",
        expected_sha256=str(physical.get("sha256") or ""),
        max_bytes=FEATURE_CACHE_MAX_BYTES,
    )
    micro_path = v26f_require_bound_file(
        root,
        str(micro.get("path") or ""),
        f"V26f {split} L2 micro-Doppler cache",
        expected_sha256=str(micro.get("sha256") or ""),
        max_bytes=FEATURE_CACHE_MAX_BYTES,
    )
    return physical_path, micro_path


def _load_l2_runtime(
    root: Path,
    *,
    fold_id: int,
    prepared_v26k: Mapping[str, Any],
    prepared_v26f: Mapping[str, Any],
) -> tuple[Any, Any, torch.device, dict[str, Any], dict[str, Any]]:
    v26c_contract = read_json(root / V26C_CONTRACT_PATH, description="V26c contract")
    v26c_fold = fold_record(v26c_contract, fold_id)
    l2_binding = dict(v26c_fold["l2"])
    config, config_path, _declaration = _runtime_l2_config(root, fold_id, l2_binding)
    _, source_l2, _, source_l2_sha = _source_protocols(root)
    model, reference, pairing = build_paired_model_for_config(root, source_l2)
    del reference
    model.config = config
    initial_state_sha = _state_dict_sha256(model.state_dict())
    device = torch.device("cpu")
    model.to(device)

    v26f_fold = fold_contract_entry(prepared_v26f, fold_id)
    source_files = {
        str(dict(item).get("name")): dict(item)
        for item in list(v26f_fold.get("source_files") or ())
    }
    required = {"l2_config", "l2_checkpoint"}
    if not required.issubset(source_files):
        raise ValueError(f"V26k V26f fold {fold_id} lacks L2 source bindings")
    checkpoint_binding = source_files["l2_checkpoint"]
    checkpoint = v26f_require_bound_file(
        root,
        str(checkpoint_binding.get("path") or ""),
        f"V26f fold {fold_id} L2 checkpoint",
        expected_sha256=str(checkpoint_binding.get("sha256") or ""),
        max_bytes=MAX_JSON_BYTES,
    )
    config_binding = source_files["l2_config"]
    if str(config_binding.get("path")) != relative_path(root, config_path):
        raise ValueError(f"V26k V26f/V26c L2 config path differs in fold {fold_id}")
    if str(config_binding.get("sha256")) != sha256_file(config_path):
        raise ValueError(f"V26k V26f/V26c L2 config hash differs in fold {fold_id}")
    if config.train.checkpoint_schema != L2_CHECKPOINT_SCHEMA:
        raise ValueError(f"V26k L2 checkpoint schema differs in fold {fold_id}")
    _load_checkpoint(
        checkpoint,
        task="dbb_specialist",
        schema=L2_CHECKPOINT_SCHEMA,
        fold_id=fold_id,
        model=model,
        optimizer=None,
        config_identity=task_config_identity_sha256(config),
        config_path=config_path,
        contract_sha256=sha256_file(root / V26C_CONTRACT_PATH),
        readiness_sha256=sha256_file(root / V26C_READINESS_PATH),
        source_config_sha256=source_l2_sha,
        initial_state_sha256=initial_state_sha,
        device=device,
    )
    model.eval()
    binding = {
        "config_path": relative_path(root, config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": relative_path(root, checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": _state_dict_sha256(model.state_dict()),
        "fixed_epoch": int(L2_EPOCH),
        "device": str(device),
        "source_config_sha256": source_l2_sha,
        "pairing_metadata": pairing,
    }
    return model, config, device, binding, v26c_fold


def _run_fold_worker(
    root: Path,
    *,
    fold_id: int,
) -> dict[str, Any]:
    """Worker entry point; validate every source before any mixed scoring."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    prepared_v26k, _ = validate_prepared_contract(root)
    prepared_v26f, _ = validate_v26f_prepared_contract(root)
    model, config, device, model_binding, v26c_fold = _load_l2_runtime(
        root,
        fold_id=fold_id,
        prepared_v26k=prepared_v26k,
        prepared_v26f=prepared_v26f,
    )
    route_rows = read_v26f_route_rows(root)
    train_ids = tuple(str(value) for value in dict(v26c_fold["l2"])["train_parent_ids"])
    train_dataset = build_task_dataset(
        root,
        config,
        split_name="train",
        parent_ids=train_ids,
        rd_physical_cache_path_override=root / l2_train_physical_cache_path(fold_id),
        micro_doppler_cache_path_override=root / l2_train_micro_cache_path(fold_id),
    )
    if tuple(str(value) for value in train_dataset.parent_ids) != train_ids:
        raise ValueError(f"V26k L2 train parent order differs in fold {fold_id}")
    expected_train_counts = dict(v26c_fold["l2"]["train_label_parent_counts"])
    if train_dataset.label_counts() != expected_train_counts:
        raise ValueError(f"V26k L2 train label counts differ in fold {fold_id}")
    train_rows = _forward_parent_dataset(
        train_dataset,
        collate=collate_task_sequences(train_dataset.spec),
        model=model,
        device=device,
        expected_parent_ids=train_ids,
        scope="train_reference",
    )
    fitted, reference_and_threshold, _train_scores = _fit_reference(train_rows)
    reference = dict(reference_and_threshold["reference"])
    threshold = dict(reference_and_threshold["threshold"])
    del train_dataset, train_rows
    gc.collect()

    split_results: dict[str, dict[str, Any]] = {}
    v26f_contract = read_json(root / V26F_CONTRACT_PATH, description="V26f contract")
    for split in ("val", "test"):
        physical, micro = _mixed_cache_bindings(root, v26f_contract, split=split)
        mixed_dataset = FrozenL2MixedAuditDataset(
            root,
            config,
            split_name=split,
            rd_physical_cache_path=physical,
            micro_doppler_cache_path=micro,
        )
        expected_counts = dict(V26K_EXPECTED_PARENT_COUNTS[split])
        if mixed_dataset.label_counts() != expected_counts:
            raise ValueError(f"V26k {split} mixed labels differ in fold {fold_id}")
        expected_ids = tuple(str(value) for value in mixed_dataset.parent_ids)
        mixed_rows = _forward_parent_dataset(
            mixed_dataset,
            collate=collate_point_sequences,
            model=model,
            device=device,
            expected_parent_ids=expected_ids,
            scope=split,
        )
        scored = _score_mixed_rows(
            mixed_rows,
            fitted=fitted,
            threshold=threshold,
            route_rows=route_rows,
            split=split,
            fold_id=fold_id,
        )
        split_results[split] = {
            "parent_count": int(len(scored)),
            "parent_ids_sha256": parent_id_sequence_sha256(
                [str(row["parent_sample_id"]) for row in scored]
            ),
            "metrics": _fold_metrics(scored, threshold),
            "rows": scored,
        }
        del mixed_dataset, mixed_rows, scored
        gc.collect()
    result = {
        "status": "passed",
        "fold_id": int(fold_id),
        "execution_device": str(device),
        "model_binding": model_binding,
        "reference": reference,
        "threshold": threshold,
        "validation": split_results["val"],
        "test": split_results["test"],
    }
    del model, config, fitted, split_results
    gc.collect()
    return result


def _isolated_fold_command(root: Path, *, fold_id: int) -> list[str]:
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "report.known_gate_v26k",
        "--project-root",
        str(root),
        "--isolated-fold",
        "--fold-id",
        str(int(fold_id)),
    ]


def _run_isolated_fold(root: Path, *, fold_id: int) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        _isolated_fold_command(root, fold_id=fold_id),
        cwd=str(root / "src-v3"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=1800,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "V26k isolated fold worker failed; "
            f"fold={fold_id}; returncode={completed.returncode}; "
            f"stderr={completed.stderr[-6000:]}"
        )
    output = completed.stdout.strip()
    if not output or len(output.encode("utf-8")) > MAX_ISOLATED_FOLD_OUTPUT_BYTES:
        raise MemoryError("V26k isolated fold output is missing or exceeds its bound")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("V26k isolated fold worker emitted non-JSON output") from exc
    if not isinstance(payload, dict):
        raise ValueError("V26k isolated fold worker output must be an object")
    if (
        payload.get("status") != "passed"
        or int(payload.get("fold_id", -1)) != int(fold_id)
        or set(payload.get("validation", {})) != {"parent_count", "parent_ids_sha256", "metrics", "rows"}
        or set(payload.get("test", {})) != {"parent_count", "parent_ids_sha256", "metrics", "rows"}
    ):
        raise ValueError("V26k isolated fold worker binding differs")
    return payload


def _write_parent_details(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=DETAIL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(
        rows,
        key=lambda value: (
            str(value["split"]),
            int(value["fold_id"]),
            str(value["parent_sample_id"]),
        ),
    ):
        writer.writerow({field: row.get(field) for field in DETAIL_FIELDS})
    return write_immutable_text(
        root / DETAIL_CSV,
        stream.getvalue(),
        max_bytes=MAX_DETAIL_BYTES,
        description="V26k parent detail CSV",
    )


def _assert_population(rows: Sequence[Mapping[str, Any]]) -> None:
    for split, expected in V26K_EXPECTED_PARENT_COUNTS.items():
        for fold_id in range(FOLD_COUNT):
            members = [
                row
                for row in rows
                if str(row["split"]) == split and int(row["fold_id"]) == fold_id
            ]
            observed = Counter(str(row["label_name"]) for row in members)
            if dict(observed) != dict(expected):
                raise ValueError(
                    f"V26k {split} fold {fold_id} population differs: {dict(observed)}"
                )
            ids = [str(row["parent_sample_id"]) for row in members]
            if len(ids) != len(set(ids)):
                raise ValueError(f"V26k {split} fold {fold_id} repeats a parent")
            if any(not bool(row["l2_forwarded"]) for row in members):
                raise ValueError(f"V26k {split} fold {fold_id} did not forward all L2 rows")


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prepared_contract, _readiness = validate_prepared_contract(root)
    del _readiness
    folds: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        worker = _run_isolated_fold(root, fold_id=fold_id)
        folds.append(
            {
                "fold_id": int(fold_id),
                "execution_device": worker["execution_device"],
                "model_binding": worker["model_binding"],
                "reference": worker["reference"],
                "threshold": worker["threshold"],
                "validation": _compact_split(worker["validation"]),
                "test": _compact_split(worker["test"]),
            }
        )
        all_details.extend(list(worker["validation"]["rows"]))
        all_details.extend(list(worker["test"]["rows"]))
    _assert_population(all_details)
    detail_status = _write_parent_details(root, all_details)
    report = {
        "status": "completed_v26k_l2_train_only_mahalanobis_knownness_diagnostic",
        "schema_version": REPORT_SCHEMA,
        "experiment": "V26k frozen L2 strict-open-set knownness diagnostic",
        "scope": (
            "CQ08 V26c L2 epoch-15 DBB specialist train-only reference fitting plus "
            "V26f frozen mixed validation/test scoring; no L1/L2/L3 model or action mutation"
        ),
        "direct_source_experiment": {
            "l2_checkpoint_and_train_reference": "V26c",
            "mixed_population_and_frozen_l1_route": "V26f",
        },
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "readiness_path": relative_path(root, root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "implementation_sha256": implementation_sha256(root),
        "data": {
            "source": "CQ08",
            "index": relative_path(root, root / "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"),
            "validation_parent_counts": V26K_EXPECTED_PARENT_COUNTS["val"],
            "test_parent_counts": V26K_EXPECTED_PARENT_COUNTS["test"],
            "primary_ranking_subset": (
                "DBB and Unknown parents whose frozen V26f observed L1 route is "
                "KNOWN_TARGET_EVIDENCE"
            ),
            "clutter_handling": "reported separately; never included in the DBB reference fit",
        },
        "representation": {
            "name": "classifier_layer_norm_liquid_final_state",
            "dimension": 48,
            "verified_path": "L2 liquid_final_state -> classifier[0] LayerNorm -> classifier[1] Linear",
            "raw_liquid_final_state_used_directly": False,
            "logit_reconstruction_asserted": True,
            "angular_prototype_embedding": False,
        },
        "ood_score_policy": OOD_SCORE_POLICY,
        "unknown_protocol": {
            "unknown_training_exposure": 0,
            "unknown_normalization_exposure": 0,
            "unknown_reference_fit_exposure": 0,
            "unknown_threshold_fit_exposure": 0,
            "unknown_checkpoint_selection_exposure": 0,
            "unknown_truth_used_only_after_score_and_train_threshold_freeze": True,
        },
        "fold_count": int(FOLD_COUNT),
        "fixed_l2_epoch": int(L2_EPOCH),
        "folds": folds,
        "aggregate_fold_diagnostics": {
            "validation": _aggregate_fold_metrics(folds, split="validation"),
            "test": _aggregate_fold_metrics(folds, split="test"),
            "interpretation": (
                "means/std/range are across fold-specific scores; no state, score, "
                "threshold or action is pooled for deployment"
            ),
        },
        "parent_detail_csv": relative_path(root, root / DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / DETAIL_CSV),
        "parent_detail_csv_write": detail_status,
        "l1_route_source": {
            "source": "V26f frozen parent-final route rows",
            "l1_forward_in_v26k": 0,
            "route_truth_controls_l2": False,
        },
        "l2_semantics": {
            "l2_forwarded_for_every_mixed_parent": True,
            "l2_score_is_diagnostic_knownness_not_product_action": True,
            "strict_open_set_gate_connected_to_l3": False,
            "pending_abstain": "not_executed",
            "semantic_action_replay": "not_executed",
        },
        "not_executed": {
            "training": True,
            "oof_retraining": True,
            "new_normalization": True,
            "checkpoint_selection": True,
            "product_threshold_calibration": True,
            "l3_registry_mutation": True,
            "semantic_action_replay": True,
            "full_cascade": True,
            "deployment": True,
        },
        "deployment_eligible": False,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    write_immutable_text(
        root / REPORT,
        text,
        max_bytes=MAX_REPORT_BYTES,
        description="V26k report",
    )
    return {
        "status": report["status"],
        "report": str(root / REPORT),
        "report_sha256": sha256_file(root / REPORT),
        "parent_detail_csv": str(root / DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / DETAIL_CSV),
        "fold_count": int(FOLD_COUNT),
        "training": "not_executed",
        "threshold_fitting": "train-only diagnostic operating point; not product calibration",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V26k frozen L2 knownness diagnostic.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--isolated-fold", action="store_true")
    parser.add_argument("--fold-id", type=int, choices=range(V26F_FOLD_COUNT))
    args = parser.parse_args()
    if args.isolated_fold:
        if args.fold_id is None:
            parser.error("--isolated-fold requires --fold-id")
        print(
            json.dumps(
                _run_fold_worker(args.project_root.resolve(), fold_id=int(args.fold_id)),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return
    if args.fold_id is not None:
        parser.error("--fold-id requires --isolated-fold")
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()