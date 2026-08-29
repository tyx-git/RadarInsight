"""Run V26i's fixed-bimodal-Clutter L1 novelty geometry diagnostic.

This is a V26h-derived, no-training report.  Every fold reloads its exact
frozen V26e L1 endpoint, fits a fixed two-component Clutter reference using
only its train Known parents, then scores the same V26f mixed parent set.  No
threshold, semantic action, cache, checkpoint, or model parameter is written.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.known_gate import build_known_gate_oof_dataset
from radar_ln3.known_gate_v26_audit import V26OpenSetAuditDataset
from radar_ln3.known_gate_v26e_oof import (
    fold_record,
    l1_train_feature_cache_path,
)
from report.known_gate_v26h import (
    L1_CLASSES,
    _clear_dataset_cache,
    _fold_l1_runtime,
    _forward_dataset,
)
from radar_ln3.known_gate_v26h import (
    fit_class_conditional_reference as _v26h_fit_class_conditional_reference,
)
from radar_ln3.known_gate_v26h import (
    L1_VALIDATION_CACHE,
    L1_TEST_CACHE,
    V26E_CONTRACT,
)
from radar_ln3.known_gate_v26i import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BimodalClutterReferenceInfeasibleError,
    CONTRACT,
    DETAIL_CSV,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    KNOWN_BASE_LABELS,
    KNOWN_TARGET_RECALL_MINIMUM,
    MAX_DETAIL_BYTES,
    MAX_ISOLATED_FOLD_OUTPUT_BYTES,
    MAX_REPORT_BYTES,
    OOD_SCORE_POLICY,
    READINESS,
    REPORT,
    REPORT_SCHEMA,
    UNKNOWN_BASE_LABEL,
    V26H_CONTRACT,
    V26H_DETAIL_CSV,
    bimodal_clutter_mahalanobis_scores,
    binary_ranking_metrics,
    fit_bimodal_clutter_reference,
    implementation_sha256,
    parent_id_sequence_sha256,
    read_json,
    relative_path,
    sha256_file,
    state_matrix_sha256,
    validate_prepared_contract,
    write_immutable_json,
    write_immutable_text,
)


V26E_CONTRACT_PATH = Path(str(V26E_CONTRACT))
DETAIL_FIELDS = (
    "split",
    "fold_id",
    "parent_sample_id",
    "track_id",
    "capture_id",
    "provenance_component_id",
    "label_name",
    "is_unknown_truth",
    "closest_reference_class",
    "closest_clutter_component",
    "bimodal_clutter_mahalanobis_novelty_score",
    "v26h_single_clutter_mahalanobis_novelty_score",
    "novelty_delta_v26i_minus_v26h",
    "max_softmax_unknownness",
    "known_target_probability",
    "known_clutter_probability",
    "l1_predicted_class",
    "parent_length",
    "rd_present_at_final",
    "course_valid_at_final",
)


def _reference_fit(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("V26i cannot fit an empty train reference")
    states = np.stack([np.asarray(row["state"], dtype=np.float64) for row in rows], axis=0)
    class_ids = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    parent_ids = tuple(str(row["parent_sample_id"]) for row in rows)
    # Freeze V26h's target centre and pooled variance exactly.  Re-estimating
    # either would alter the score scale for every class and turn this into a
    # multi-variable comparison.  Only the one Clutter centre becomes two.
    control = _v26h_fit_class_conditional_reference(states, class_ids)
    control_metadata = {
        "fit_population": "per_fold_train_known_only_parent_final",
        "parent_count": int(len(rows)),
        "parent_ids_sha256": parent_id_sequence_sha256(parent_ids),
        "state_matrix_sha256": state_matrix_sha256(states),
        "state_dimension": int(states.shape[1]),
        "class_counts": dict(control["class_counts"]),
        "pooled_degrees_of_freedom": int(control["pooled_degrees_of_freedom"]),
        "variance_floor": float(control["variance_floor"]),
        "unknown_reference_fit_exposure": 0,
    }
    fitted = fit_bimodal_clutter_reference(
        states,
        class_ids,
        parent_ids,
        control_known_target_centre=np.asarray(control["centres"][0], dtype=np.float64),
        control_pooled_variance=np.asarray(control["pooled_variance"], dtype=np.float64),
    )
    metadata = {
        "fit_population": "per_fold_train_known_only_parent_final",
        "parent_count": int(len(rows)),
        "parent_ids_sha256": parent_id_sequence_sha256(parent_ids),
        "state_matrix_sha256": state_matrix_sha256(states),
        "state_dimension": int(states.shape[1]),
        "class_counts": dict(fitted["class_counts"]),
        "clutter_component_counts": list(fitted["clutter_component_counts"]),
        "pooled_degrees_of_freedom": fitted["pooled_degrees_of_freedom"],
        "variance_floor": fitted["variance_floor"],
        "control_reference": {
            "source": (
                "V26h exact one-centre reference reconstructed from same fold train states; "
                "mixed scores are additionally checked against the immutable V26h detail CSV"
            ),
            "known_target_centre_sha256": state_matrix_sha256(
                np.asarray(control["centres"][0], dtype=np.float64).reshape(1, -1)
            ),
            "pooled_variance_sha256": state_matrix_sha256(
                np.asarray(control["pooled_variance"], dtype=np.float64).reshape(1, -1)
            ),
            "control_pooled_degrees_of_freedom": int(control["pooled_degrees_of_freedom"]),
            "control_variance_floor": float(control["variance_floor"]),
            "control_metadata": control_metadata,
        },
        "clustering": dict(fitted["clustering"]),
        "unknown_reference_fit_exposure": 0,
    }
    return fitted, metadata


def _v26h_detail_lookup(root: Path, *, split: str, fold_id: int) -> dict[str, Mapping[str, str]]:
    """Load immutable paired V26h scores for an exact score-replay assertion."""

    path = root / V26H_DETAIL_CSV
    result: dict[str, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row["split"]) != split or int(row["fold_id"]) != int(fold_id):
                continue
            parent_id = str(row["parent_sample_id"])
            if parent_id in result:
                raise ValueError(f"V26i V26h detail repeats {split} fold {fold_id} parent")
            result[parent_id] = row
    expected_count = int(sum(EXPECTED_PARENT_COUNTS[split].values()))
    if len(result) != expected_count:
        raise ValueError(f"V26i V26h detail population differs for {split} fold {fold_id}")
    return result


def _scored_rows(
    rows: Sequence[Mapping[str, Any]],
    fitted: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    v26h_rows: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("V26i cannot score an empty mixed population")
    states = np.stack([np.asarray(row["state"], dtype=np.float64) for row in rows], axis=0)
    scores, closest_classes, closest_components = bimodal_clutter_mahalanobis_scores(states, fitted)
    target = np.asarray(control["centres"][0], dtype=np.float64)
    clutter = np.asarray(control["centres"][1], dtype=np.float64)
    variance = np.asarray(control["pooled_variance"], dtype=np.float64)
    control_distances = np.sqrt(
        (np.square(states[:, None, :] - np.stack((target, clutter), axis=0)[None, :, :]) / variance[None, None, :]).sum(axis=2)
    )
    control_scores = control_distances.min(axis=1)
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        parent_id = str(raw["parent_sample_id"])
        baseline = dict(v26h_rows.get(parent_id) or {})
        if not baseline:
            raise ValueError(f"V26i missing paired V26h score for {parent_id}")
        old_score = float(baseline["mahalanobis_novelty_score"])
        recomputed_control_score = float(control_scores[index])
        if not np.isclose(recomputed_control_score, old_score, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(
                "V26i V26h paired control score reconstruction differs: "
                f"parent={parent_id}; recomputed={recomputed_control_score}; recorded={old_score}"
            )
        row = {key: value for key, value in raw.items() if key != "state"}
        row.update(
            {
                "closest_reference_class": L1_CLASSES[int(closest_classes[index])],
                "closest_clutter_component": int(closest_components[index]),
                "bimodal_clutter_mahalanobis_novelty_score": float(scores[index]),
                "v26h_single_clutter_mahalanobis_novelty_score": old_score,
                "novelty_delta_v26i_minus_v26h": float(scores[index] - old_score),
                "max_softmax_unknownness": float(
                    1.0
                    - max(
                        float(raw["known_target_probability"]),
                        float(raw["known_clutter_probability"]),
                    )
                ),
                "is_unknown_truth": bool(raw["label_name"] == UNKNOWN_BASE_LABEL),
            }
        )
        result.append(row)
    if len(result) != len(v26h_rows) or set(str(row["parent_sample_id"]) for row in result) != set(v26h_rows):
        raise ValueError("V26i scored parents do not exactly match paired V26h detail")
    return result


def _score_values(rows: Sequence[Mapping[str, Any]], *, label: str | None = None) -> list[float]:
    return [
        float(row["bimodal_clutter_mahalanobis_novelty_score"])
        for row in rows
        if label is None or str(row["label_name"]) == label
    ]


def _per_label_summary(rows: Sequence[Mapping[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label_name"])].append(float(row[score_key]))
    result: list[dict[str, Any]] = []
    for label in (*KNOWN_BASE_LABELS, UNKNOWN_BASE_LABEL):
        values = np.asarray(grouped.get(label, ()), dtype=np.float64)
        if not len(values):
            continue
        result.append(
            {
                "label_name": label,
                "support": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(values.max()),
            }
        )
    return result


def _auc_unknown_vs(rows: Sequence[Mapping[str, Any]], *, known_labels: Sequence[str]) -> dict[str, Any] | None:
    unknown = _score_values(rows, label=UNKNOWN_BASE_LABEL)
    known = [
        float(row["bimodal_clutter_mahalanobis_novelty_score"])
        for row in rows
        if str(row["label_name"]) in set(known_labels)
    ]
    return binary_ranking_metrics(known, unknown) if known and unknown else None


def _route_conditioned_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ("closest_reference_class", "l1_predicted_class"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[column])].append(row)
        output: dict[str, Any] = {}
        for key, members in sorted(grouped.items()):
            unknown = _score_values(members, label=UNKNOWN_BASE_LABEL)
            clutter = _score_values(members, label="ClutterTarget")
            output[key] = {
                "unknown_parent_support": int(len(unknown)),
                "clutter_parent_support": int(len(clutter)),
                "unknown_score_median": float(np.median(unknown)) if unknown else None,
                "clutter_score_median": float(np.median(clutter)) if clutter else None,
                "unknown_vs_clutter": (
                    binary_ranking_metrics(clutter, unknown) if unknown and clutter else None
                ),
            }
        result[column] = output
    return result


def _bootstrap_pairwise(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    groups = {
        label: [row for row in rows if str(row["label_name"]) == label]
        for label in (*KNOWN_BASE_LABELS, UNKNOWN_BASE_LABEL)
    }
    if not groups[UNKNOWN_BASE_LABEL] or not groups["ClutterTarget"]:
        raise ValueError("V26i pairwise bootstrap needs Unknown and Clutter parents")
    rng = np.random.default_rng(seed)
    labels = tuple(groups)
    values: dict[str, list[float]] = {
        "unknown_vs_known_auroc": [],
        "unknown_vs_clutter_auroc": [],
        "unknown_vs_drone_auroc": [],
        "unknown_vs_balloon_auroc": [],
        "clutter_novelty_delta_v26i_minus_v26h": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: dict[str, list[Mapping[str, Any]]] = {}
        for label in labels:
            source = groups[label]
            sampled[label] = [source[int(index)] for index in rng.choice(len(source), size=len(source), replace=True)]
        unknown = _score_values(sampled[UNKNOWN_BASE_LABEL])
        known = [
            float(row["bimodal_clutter_mahalanobis_novelty_score"])
            for label in KNOWN_BASE_LABELS
            for row in sampled[label]
        ]
        values["unknown_vs_known_auroc"].append(
            binary_ranking_metrics(known, unknown)["auroc_unknown_vs_known"]
        )
        for name, label in (
            ("unknown_vs_clutter_auroc", "ClutterTarget"),
            ("unknown_vs_drone_auroc", "DroneTarget"),
            ("unknown_vs_balloon_auroc", "BalloonTarget"),
        ):
            values[name].append(
                binary_ranking_metrics(_score_values(sampled[label]), unknown)["auroc_unknown_vs_known"]
            )
        deltas = [float(row["novelty_delta_v26i_minus_v26h"]) for row in sampled["ClutterTarget"]]
        values["clutter_novelty_delta_v26i_minus_v26h"].append(float(np.mean(deltas)))
    return {
        "method": "stratified parent bootstrap; all label strata resampled independently",
        "replicates": int(BOOTSTRAP_REPLICATES),
        "seed": int(seed),
        "confidence_level": 0.95,
        "stratum_support": {label: int(len(rows)) for label, rows in groups.items()},
        "metrics": {
            key: {
                "median": float(np.median(value)),
                "lower_95": float(np.quantile(value, 0.025)),
                "upper_95": float(np.quantile(value, 0.975)),
            }
            for key, value in values.items()
        },
    }


def _fold_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known = [
        float(row["bimodal_clutter_mahalanobis_novelty_score"])
        for row in rows
        if not bool(row["is_unknown_truth"])
    ]
    unknown = _score_values(rows, label=UNKNOWN_BASE_LABEL)
    old_known = [
        float(row["v26h_single_clutter_mahalanobis_novelty_score"])
        for row in rows
        if not bool(row["is_unknown_truth"])
    ]
    old_unknown = [
        float(row["v26h_single_clutter_mahalanobis_novelty_score"])
        for row in rows
        if bool(row["is_unknown_truth"])
    ]
    clutter_deltas = [
        float(row["novelty_delta_v26i_minus_v26h"])
        for row in rows
        if str(row["label_name"]) == "ClutterTarget"
    ]
    return {
        "v26i_bimodal": binary_ranking_metrics(known, unknown),
        "v26h_paired_single_clutter_comparator": binary_ranking_metrics(old_known, old_unknown),
        "per_label_v26i_bimodal": _per_label_summary(
            rows, score_key="bimodal_clutter_mahalanobis_novelty_score"
        ),
        "per_label_v26h_single_clutter": _per_label_summary(
            rows, score_key="v26h_single_clutter_mahalanobis_novelty_score"
        ),
        "pairwise_unknown_ranking": {
            "unknown_vs_drone": _auc_unknown_vs(rows, known_labels=("DroneTarget",)),
            "unknown_vs_bird": _auc_unknown_vs(rows, known_labels=("BirdTarget",)),
            "unknown_vs_balloon": _auc_unknown_vs(rows, known_labels=("BalloonTarget",)),
            "unknown_vs_clutter": _auc_unknown_vs(rows, known_labels=("ClutterTarget",)),
        },
        "clutter_novelty_delta_v26i_minus_v26h": {
            "support": int(len(clutter_deltas)),
            "mean": float(np.mean(clutter_deltas)),
            "median": float(np.median(clutter_deltas)),
            "all_clutter_scores_not_higher": bool(all(value <= 1.0e-10 for value in clutter_deltas)),
        },
        "route_conditioned": _route_conditioned_metrics(rows),
        "bootstrap": _bootstrap_pairwise(rows, seed=BOOTSTRAP_SEED + int(rows[0]["fold_id"])),
    }


def _aggregate_fold_metrics(folds: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    measures = (
        ("v26i_bimodal", "auroc_unknown_vs_known"),
        ("v26i_bimodal", "average_precision_unknown"),
        ("v26h_paired_single_clutter_comparator", "auroc_unknown_vs_known"),
        ("v26h_paired_single_clutter_comparator", "average_precision_unknown"),
        ("pairwise_unknown_ranking", "unknown_vs_clutter.auroc_unknown_vs_known"),
        ("pairwise_unknown_ranking", "unknown_vs_drone.auroc_unknown_vs_known"),
        ("pairwise_unknown_ranking", "unknown_vs_balloon.auroc_unknown_vs_known"),
        ("clutter_novelty_delta_v26i_minus_v26h", "mean"),
    )
    summary: dict[str, Any] = {"split": split, "fold_count": len(folds), "fold_values": {}}
    for scope, dotted_key in measures:
        path = dotted_key.split(".")
        collected: list[float] = []
        for fold in folds:
            value: Any = fold[split]["metrics"][scope]
            for key in path:
                value = value[key]
            collected.append(float(value))
        array = np.asarray(collected, dtype=np.float64)
        summary["fold_values"][f"{scope}.{dotted_key}"] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "direction_nonnegative_vs_v26h": (
                bool((array >= -1.0e-12).all())
                if scope == "clutter_novelty_delta_v26i_minus_v26h"
                else None
            ),
        }
    return summary


def _parent_detail_csv(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=DETAIL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(
        rows,
        key=lambda value: (str(value["split"]), int(value["fold_id"]), str(value["parent_sample_id"])),
    ):
        writer.writerow({field: row.get(field) for field in DETAIL_FIELDS})
    return write_immutable_text(
        root / DETAIL_CSV,
        stream.getvalue(),
        max_bytes=MAX_DETAIL_BYTES,
        description="parent detail CSV",
    )


def _assert_mixed_population(rows: Sequence[Mapping[str, Any]]) -> None:
    for split, expected_labels in EXPECTED_PARENT_COUNTS.items():
        for fold_id in range(FOLD_COUNT):
            fold_rows = [
                row for row in rows if str(row["split"]) == split and int(row["fold_id"]) == fold_id
            ]
            observed: dict[str, int] = defaultdict(int)
            parent_ids: list[str] = []
            for row in fold_rows:
                observed[str(row["label_name"])] += 1
                parent_ids.append(str(row["parent_sample_id"]))
            if dict(observed) != dict(expected_labels):
                raise ValueError(f"V26i {split} fold {fold_id} mixed parent support differs from V26h")
            if len(parent_ids) != len(set(parent_ids)):
                raise ValueError(f"V26i {split} fold {fold_id} repeats a mixed parent")


def _isolated_fold_command(root: Path, *, fold_id: int) -> list[str]:
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "report.known_gate_v26i",
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
            "V26i isolated fold worker failed; "
            f"fold={fold_id}; returncode={completed.returncode}; stderr={completed.stderr[-4000:]}"
        )
    output = completed.stdout.strip()
    if not output or len(output.encode("utf-8")) > MAX_ISOLATED_FOLD_OUTPUT_BYTES:
        raise MemoryError("V26i isolated fold output is missing or exceeds its bound")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("V26i isolated fold worker emitted non-JSON output") from exc
    if not isinstance(payload, dict) or int(payload.get("fold_id", -1)) != int(fold_id):
        raise ValueError("V26i isolated fold worker binding differs")
    if payload.get("status") == "infeasible_fixed_k2_reference":
        if (
            set(payload) != {"status", "fold_id", "reason", "metadata"}
            or not isinstance(payload.get("reason"), str)
            or not isinstance(payload.get("metadata"), dict)
        ):
            raise ValueError("V26i isolated infeasible-fold payload differs")
        return payload
    if (
        payload.get("status") != "passed"
        or set(payload.get("validation", {})) != {"parent_count", "parent_ids_sha256", "metrics", "rows"}
        or set(payload.get("test", {})) != {"parent_count", "parent_ids_sha256", "metrics", "rows"}
    ):
        raise ValueError("V26i isolated fold worker binding differs")
    return payload


def _run_isolated_fold_worker(root: Path, *, fold_id: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    prepared_contract, _readiness = validate_prepared_contract(root)
    v26h_contract = read_json(root / V26H_CONTRACT, description="V26h contract")
    source_v26e = dict(v26h_contract.get("source_v26e") or {})
    model, config, device, binding = _fold_l1_runtime(
        root,
        fold_id=fold_id,
        prepared_contract={"source_v26e": source_v26e},
        force_cpu=True,
    )
    # The V26h runtime's only need for ``prepared_contract`` is its V26e bound
    # checkpoint list.  V26i obtains that exact list from V26h's immutable
    # contract; no model source is recomputed or relaxed.
    if device.type != "cpu":
        raise AssertionError("V26i isolated fold worker escaped its CPU boundary")
    v26e_contract = read_json(root / V26E_CONTRACT_PATH, description="V26e contract")
    v26e_fold = fold_record(v26e_contract, fold_id)
    train_ids = tuple(str(value) for value in v26e_fold["train_parent_ids"])
    train_dataset = build_known_gate_oof_dataset(
        root,
        config,
        parent_ids=train_ids,
        feature_cache_path=root / l1_train_feature_cache_path(fold_id),
    )
    train_rows = _forward_dataset(
        train_dataset,
        model=model,
        device=device,
        fold_id=fold_id,
        split="train_reference",
        expected_parent_ids=train_ids,
    )
    try:
        fitted, reference_metadata = _reference_fit(train_rows)
    except BimodalClutterReferenceInfeasibleError as exc:
        # This is an expected, pre-registered terminal result, not a worker
        # failure.  The parent process cannot catch a typed exception raised
        # in this subprocess, so preserve the exact reason and metadata in a
        # compact JSON payload.  Do not repair the protocol by reducing K,
        # borrowing held-out parents, or fitting on Unknown data.
        _clear_dataset_cache(train_dataset)
        del train_dataset, train_rows, model, config, prepared_contract, v26h_contract
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "infeasible_fixed_k2_reference",
            "fold_id": int(fold_id),
            "reason": str(exc),
            "metadata": dict(exc.metadata),
        }
    control_states = np.stack([np.asarray(row["state"], dtype=np.float64) for row in train_rows], axis=0)
    control_class_ids = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    control = _v26h_fit_class_conditional_reference(control_states, control_class_ids)
    control_metadata = {
        "fit_population": "per_fold_train_known_only_parent_final",
        "parent_count": int(len(train_rows)),
        "parent_ids_sha256": parent_id_sequence_sha256(
            [str(row["parent_sample_id"]) for row in train_rows]
        ),
        "state_matrix_sha256": state_matrix_sha256(control_states),
        "state_dimension": int(control_states.shape[1]),
        "class_counts": dict(control["class_counts"]),
        "pooled_degrees_of_freedom": int(control["pooled_degrees_of_freedom"]),
        "variance_floor": float(control["variance_floor"]),
        "unknown_reference_fit_exposure": 0,
    }
    if (
        state_matrix_sha256(
            np.asarray(control["centres"][0], dtype=np.float64).reshape(1, -1)
        )
        != reference_metadata["control_reference"]["known_target_centre_sha256"]
        or state_matrix_sha256(
            np.asarray(control["pooled_variance"], dtype=np.float64).reshape(1, -1)
        )
        != reference_metadata["control_reference"]["pooled_variance_sha256"]
    ):
        raise AssertionError("V26i frozen V26h control reference reconstruction differs")
    _clear_dataset_cache(train_dataset)
    del train_dataset, train_rows, control_states, control_class_ids
    gc.collect()

    split_results: dict[str, dict[str, Any]] = {}
    for split, cache_path in (("val", L1_VALIDATION_CACHE), ("test", L1_TEST_CACHE)):
        mixed_dataset = V26OpenSetAuditDataset(root, config, split_name=split, cache_path=root / cache_path)
        expected_ids = tuple(str(value) for value in mixed_dataset.parent_ids)
        if mixed_dataset.label_counts() != dict(EXPECTED_PARENT_COUNTS[split]):
            raise ValueError(f"V26i {split} mixed labels differ from frozen V26h counts")
        mixed_rows = _forward_dataset(
            mixed_dataset,
            model=model,
            device=device,
            fold_id=fold_id,
            split=split,
            expected_parent_ids=expected_ids,
        )
        paired_v26h_rows = _v26h_detail_lookup(root, split=split, fold_id=fold_id)
        scored = _scored_rows(
            mixed_rows,
            fitted,
            control,
            v26h_rows=paired_v26h_rows,
        )
        split_results[split] = {
            "parent_count": int(len(scored)),
            "parent_ids_sha256": parent_id_sequence_sha256([str(row["parent_sample_id"]) for row in scored]),
            "metrics": _fold_metrics(scored),
            "rows": scored,
        }
        _clear_dataset_cache(mixed_dataset)
        del mixed_dataset, mixed_rows, paired_v26h_rows, scored
        gc.collect()
    result = {
        "status": "passed",
        "fold_id": int(fold_id),
        "execution_device": str(device),
        "model_binding": binding,
        "reference": reference_metadata,
        "validation": split_results["val"],
        "test": split_results["test"],
    }
    del model, config, fitted, control, control_metadata, split_results, prepared_contract, v26h_contract
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prepared_contract, readiness = validate_prepared_contract(root)
    del prepared_contract, readiness
    folds: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    infeasible_folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = _run_isolated_fold(root, fold_id=fold_id)
        if fold["status"] == "infeasible_fixed_k2_reference":
            infeasible_folds.append(
                {
                    "fold_id": int(fold_id),
                    "reason": str(fold["reason"]),
                    "metadata": dict(fold["metadata"]),
                }
            )
            # Fixed K=2 was a pre-registered all-fold requirement.  A single
            # infeasible fold already makes the treatment non-evaluable; stop
            # rather than consuming further CPU or producing a misleading
            # partial aggregate.
            break
        folds.append(fold)
        details.extend(list(fold["validation"]["rows"]))
        details.extend(list(fold["test"]["rows"]))
    if infeasible_folds:
        report = {
            "status": "completed_v26i_infeasible_fixed_k2_reference",
            "schema_version": REPORT_SCHEMA,
            "experiment": "V26i frozen L1 train-only fixed-bimodal-Clutter Mahalanobis geometry diagnostic",
            "contract_path": relative_path(root, root / CONTRACT),
            "contract_sha256": sha256_file(root / CONTRACT),
            "readiness_path": relative_path(root, root / READINESS),
            "readiness_sha256": sha256_file(root / READINESS),
            "ood_score_policy": OOD_SCORE_POLICY,
            "infeasible_folds": infeasible_folds,
            "threshold_fitting": "not_executed",
            "semantic_action_replay": "not_executed",
            "deployment_eligible": False,
        }
        write_immutable_json(root / REPORT, report, max_bytes=MAX_REPORT_BYTES, description="report")
        return {"status": report["status"], "report": str(root / REPORT), "deployment_eligible": False}
    if len(folds) != FOLD_COUNT:
        raise AssertionError("V26i completed without full fold coverage")
    _assert_mixed_population(details)
    detail_status = _parent_detail_csv(root, details)
    aggregate = {
        "validation": _aggregate_fold_metrics(folds, split="validation"),
        "test": _aggregate_fold_metrics(folds, split="test"),
        "interpretation": (
            "fold-level threshold-free diagnostics; no state, centre, variance, threshold "
            "or semantic action is pooled for deployment"
        ),
    }
    report = {
        "status": "completed_v26i_train_only_bimodal_clutter_mahalanobis_ranking_diagnostic",
        "schema_version": REPORT_SCHEMA,
        "experiment": "V26i frozen L1 train-only fixed-bimodal-Clutter Mahalanobis geometry diagnostic",
        "scope": "CQ08 V26h-derived train-only reference fitting plus frozen mixed validation/test ranking; no action replay or calibration",
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "readiness_path": relative_path(root, root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "ood_score_policy": OOD_SCORE_POLICY,
        "representation": {
            "name": "classifier_layer_norm_liquid_final_state",
            "verified_classifier_path": "KnownGateClassifier.classifier[0] LayerNorm before classifier[1] Linear",
            "raw_liquid_final_state_used_directly": False,
            "gradient_tracking": False,
        },
        "implementation_sha256": implementation_sha256(root),
        "source_protocol": {
            "direct_source": "V26h completed immutable source chain",
            "v26e_l1": "five fixed epoch-32 Known-only OOF checkpoints; each fold reference fits only own train parents",
            "v26f_mixed_data": "frozen mixed val/test caches after model/checkpoint freeze",
            "unknown_training_exposure": 0,
            "unknown_normalization_exposure": 0,
            "unknown_reference_fit_exposure": 0,
            "unknown_checkpoint_selection_exposure": 0,
            "unknown_threshold_fit_exposure": 0,
            "unknown_truth_use": "final threshold-free ranking report only",
        },
        "future_action_gate": {
            "known_target_recall_minimum": KNOWN_TARGET_RECALL_MINIMUM,
            "status": "not_evaluable_v26i_does_not_change_or_replay_actions",
        },
        "folds": folds,
        "aggregate_fold_diagnostics": aggregate,
        "parent_detail_csv": relative_path(root, root / DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / DETAIL_CSV),
        "parent_detail_csv_write": detail_status,
        "comparison_boundary": {
            "direct_control": "V26h paired single-centre KnownClutter score on exact same checkpoint, parent and split",
            "only_variable": "KnownClutter reference geometry one centre to fixed two centres",
            "threshold_selection": "not_executed",
            "semantic_action_change": "not_executed",
            "known_target_recall_gate": "reserved for a future threshold/action experiment; not claimable here",
        },
        "not_executed": {
            "training": True,
            "oof_retraining": True,
            "normalization": True,
            "checkpoint_selection": True,
            "threshold_fitting": True,
            "l3_registry_mutation": True,
            "semantic_action_replay": True,
            "deployment": True,
        },
        "deployment_eligible": False,
    }
    write_immutable_json(root / REPORT, report, max_bytes=MAX_REPORT_BYTES, description="report")
    return {
        "status": report["status"],
        "report": str(root / REPORT),
        "report_sha256": sha256_file(root / REPORT),
        "parent_detail_csv": str(root / DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / DETAIL_CSV),
        "fold_count": int(FOLD_COUNT),
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V26i fixed-bimodal-Clutter Mahalanobis diagnostic.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--isolated-fold", action="store_true")
    parser.add_argument("--fold-id", type=int, choices=range(FOLD_COUNT))
    args = parser.parse_args()
    if args.isolated_fold:
        if args.fold_id is None:
            parser.error("--isolated-fold requires --fold-id")
        print(json.dumps(_run_isolated_fold_worker(args.project_root.resolve(), fold_id=int(args.fold_id)), ensure_ascii=False, separators=(",", ":")))
        return
    if args.fold_id is not None:
        parser.error("--fold-id requires --isolated-fold")
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
