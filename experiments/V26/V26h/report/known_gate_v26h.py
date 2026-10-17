"""Run the V26h frozen train-only L1 representation OOD diagnostic.

V26h is deliberately a diagnostic, not a product change.  It reloads each
frozen V26e L1 epoch-32 checkpoint, fits a class-conditional reference from
that fold's Known train parents only, and evaluates a threshold-free novelty
ranking on the already frozen V26f mixed validation/test populations.

The expensive RD forward is run in one short-lived CPU subprocess per fold.
That boundary is intentional on Windows: the project RD tensors are large and
PyTorch/CUDA may retain allocator pages across many variable-length parents.
The worker returns only final 48-D states (for train fitting) and compact
scored mixed rows; it never writes or mutates a model, cache, threshold, or
semantic action.
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

from radar_ln3.known_gate import (
    KnownGateConfig,
    build_known_gate_model_for_config,
    build_known_gate_oof_dataset,
    collate_known_gate_sequences,
    known_gate_config_identity_sha256,
    known_gate_device,
)
from radar_ln3.known_gate_v26_audit import (
    V26OpenSetAuditDataset,
    collate_v26_open_set_audit,
)
from radar_ln3.known_gate_v26e_oof import (
    CONTRACT as V26E_CONTRACT,
    READINESS as V26E_READINESS,
    fold_record,
    l1_train_feature_cache_path,
)
from radar_ln3.known_gate_v26h import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONTRACT,
    DETAIL_CSV,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    KNOWN_BASE_LABELS,
    L1_CLASSES,
    L1_EPOCH,
    L1_TEST_CACHE,
    L1_VALIDATION_CACHE,
    MAX_DETAIL_BYTES,
    MAX_ISOLATED_FOLD_OUTPUT_BYTES,
    MAX_REPORT_BYTES,
    OOD_SCORE_POLICY,
    READINESS,
    REPORT,
    REPORT_SCHEMA,
    REPRESENTATION_NAME,
    UNKNOWN_BASE_LABEL,
    binary_ranking_metrics,
    fit_class_conditional_reference,
    implementation_sha256,
    mahalanobis_scores,
    parent_id_sequence_sha256,
    read_json,
    relative_path,
    sha256_file,
    state_matrix_sha256,
    validate_prepared_contract,
    write_immutable_json,
    write_immutable_text,
)
from train.known_gate_common import _model_inputs as l1_model_inputs
from train.known_gate_common import _state_dict_sha256
from train.known_gate_v26c_oof import _load_checkpoint
from train.known_gate_v26e_oof import _runtime_l1_config, _source_protocol


V26E_CONTRACT_PATH = Path(str(V26E_CONTRACT))
V26E_READINESS_PATH = Path(str(V26E_READINESS))

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
    "mahalanobis_novelty_score",
    "max_softmax_unknownness",
    "known_target_probability",
    "known_clutter_probability",
    "l1_predicted_class",
    "parent_length",
    "rd_present_at_final",
    "course_valid_at_final",
)


def _clear_dataset_cache(dataset: Any) -> None:
    """Release decoded RD arrays after a parent/fold forward boundary."""

    backing = getattr(dataset, "_dataset", dataset)
    base = getattr(backing, "base", None)
    cache = getattr(base, "_tensor_cache", None)
    items = getattr(cache, "_items", None)
    if items is not None and hasattr(items, "clear"):
        items.clear()


def _fold_l1_runtime(
    root: Path,
    *,
    fold_id: int,
    prepared_contract: Mapping[str, Any],
    force_cpu: bool = False,
) -> tuple[Any, KnownGateConfig, torch.device, dict[str, Any]]:
    """Construct the exact V26e fold model and load its bound snapshot."""

    v26e_contract = read_json(root / V26E_CONTRACT_PATH, description="V26e contract")
    v26e_fold = fold_record(v26e_contract, fold_id)
    l1_config, l1_config_path, _declaration = _runtime_l1_config(
        root, fold_id, dict(v26e_fold["l1"])
    )
    source, source_sha = _source_protocol(root)
    model, reference, pairing = build_known_gate_model_for_config(root, source)
    del reference, pairing
    model.config = l1_config
    initial_state_sha = _state_dict_sha256(model.state_dict())
    # CUDA_VISIBLE_DEVICES can leave ``torch.cuda.is_available()`` true in a
    # Windows subprocess while exposing zero logical devices.  In that state
    # ``known_gate_device`` would incorrectly return cuda:0 and torch.load
    # would fail before the diagnostic starts.  The isolated worker is
    # intentionally CPU-only, so make that execution boundary explicit rather
    # than relying on process-global CUDA discovery.
    device = torch.device("cpu") if force_cpu else known_gate_device(l1_config)
    if force_cpu and device.type != "cpu":
        raise AssertionError("V26h isolated fold runtime must use CPU")
    model.to(device)

    source_fold = next(
        value
        for value in list(dict(prepared_contract["source_v26e"]).get("folds") or ())
        if int(value["fold_id"]) == int(fold_id)
    )
    checkpoint_binding = dict(source_fold["checkpoint"])
    checkpoint = root / str(checkpoint_binding["path"])
    if (
        relative_path(root, checkpoint) != checkpoint_binding["path"]
        or sha256_file(checkpoint) != checkpoint_binding["sha256"]
    ):
        raise ValueError(f"V26h fold {fold_id} checkpoint binding differs")
    _load_checkpoint(
        checkpoint,
        task="known_gate",
        schema=l1_config.train.checkpoint_schema,
        fold_id=fold_id,
        model=model,
        optimizer=None,
        config_identity=known_gate_config_identity_sha256(l1_config),
        config_path=l1_config_path,
        contract_sha256=sha256_file(root / V26E_CONTRACT_PATH),
        readiness_sha256=sha256_file(root / V26E_READINESS_PATH),
        source_config_sha256=source_sha,
        initial_state_sha256=initial_state_sha,
        device=device,
    )
    model.eval()
    return model, l1_config, device, {
        "config_path": relative_path(root, l1_config_path),
        "config_sha256": sha256_file(l1_config_path),
        "checkpoint_path": relative_path(root, checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": _state_dict_sha256(model.state_dict()),
        "fixed_epoch": int(L1_EPOCH),
        "device": str(device),
    }


def _classifier_readout_state(
    model: Any,
    output: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return the exact state consumed by the final L1 Linear layer."""

    state = output.get("liquid_final_state")
    classifier = getattr(model, "classifier", None)
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise ValueError("V26h model output lacks a [batch,hidden] final state")
    if not isinstance(classifier, torch.nn.Sequential) or len(classifier) != 2:
        raise ValueError("V26h expects the frozen LayerNorm -> Linear classifier")
    normalizer, linear = classifier[0], classifier[1]
    if not isinstance(normalizer, torch.nn.LayerNorm) or not isinstance(linear, torch.nn.Linear):
        raise ValueError("V26h classifier structure differs from the frozen L1 contract")
    value = normalizer(state)
    if value.shape[-1] != linear.in_features or not torch.isfinite(value).all():
        raise ValueError("V26h classifier readout state is invalid")
    reconstructed = linear(value)
    declared = output.get("window_logits")
    if (
        not isinstance(declared, torch.Tensor)
        or declared.shape != reconstructed.shape
        or not torch.allclose(reconstructed, declared, rtol=1.0e-5, atol=1.0e-6)
    ):
        raise ValueError("V26h readout state does not reconstruct frozen final logits")
    return value


def _forward_dataset(
    dataset: Any,
    *,
    model: Any,
    device: torch.device,
    fold_id: int,
    split: str,
    expected_parent_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Forward parent-final states serially and retain only compact rows."""

    if split not in {"train_reference", "val", "test"}:
        raise ValueError("V26h received an unsupported split")
    expected = tuple(str(value) for value in expected_parent_ids)
    observed = tuple(str(value) for value in dataset.parent_ids)
    if observed != expected:
        raise ValueError(f"V26h {split} parent order differs from its frozen contract")

    rows: list[dict[str, Any]] = []
    for parent_index, requested_parent_id in enumerate(expected, start=1):
        item = dataset[parent_index - 1]
        batch = (
            collate_known_gate_sequences([item])
            if split == "train_reference"
            else collate_v26_open_set_audit([item])
        )
        tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            output = model(**l1_model_inputs(tensors))
            state = _classifier_readout_state(model, output)
            logits = output["window_logits"]
            probabilities = torch.softmax(logits, dim=-1)
        parent_id = str(batch["parent_sample_ids"][0])
        if parent_id != requested_parent_id:
            raise ValueError(f"V26h {split} parent identity changed during forward")
        label = str(batch["base_label_names"][0])
        allowed = KNOWN_BASE_LABELS if split == "train_reference" else (*KNOWN_BASE_LABELS, UNKNOWN_BASE_LABEL)
        if label not in allowed:
            raise ValueError(f"V26h {split} contains an unsupported truth label: {label}")
        length = int(batch["lengths"][0].item())
        if length < 1 or state.shape != (1, 48) or probabilities.shape != (1, 2):
            raise ValueError(f"V26h {split} produced an invalid parent output: {parent_id}")
        state_row = state[0].detach().to("cpu", dtype=torch.float64).numpy().copy()
        probability_row = probabilities[0].detach().to("cpu", dtype=torch.float64).numpy().copy()
        if not np.isfinite(state_row).all() or not np.isfinite(probability_row).all():
            raise ValueError(f"V26h {split} emitted a non-finite parent output: {parent_id}")
        first = dataset.base._groups[parent_id].iloc[0]
        rows.append(
            {
                "parent_sample_id": parent_id,
                "label_name": label,
                "class_id": int(batch["labels"][0].item()),
                "state": state_row,
                "known_target_probability": float(probability_row[0]),
                "known_clutter_probability": float(probability_row[1]),
                "l1_predicted_class": L1_CLASSES[int(probability_row.argmax())],
                "parent_length": length,
                "track_id": str(first.get("track_id", "")),
                "capture_id": str(first.get("capture_id", "")),
                "provenance_component_id": str(first.get("provenance_component_id", "")),
                "rd_present_at_final": bool(batch["rd_frame_mask"][0, length - 1].any().item()),
                "course_valid_at_final": bool(
                    batch["known_gate_feature_valid"][0, length - 1, 0].item()
                ),
                "fold_id": int(fold_id),
                "split": split,
            }
        )
        del tensors, output, state, logits, probabilities, batch, item
        _clear_dataset_cache(dataset)
        if parent_index % 8 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if tuple(str(row["parent_sample_id"]) for row in rows) != expected:
        raise ValueError(f"V26h {split} output parent coverage differs")
    return rows


def _reference_fit(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("V26h cannot fit an empty train reference")
    states = np.stack([np.asarray(row["state"], dtype=np.float64) for row in rows], axis=0)
    class_ids = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    fitted = fit_class_conditional_reference(states, class_ids)
    metadata = {
        "fit_population": "per_fold_train_known_only_parent_final",
        "parent_count": int(len(rows)),
        "parent_ids_sha256": parent_id_sequence_sha256(
            [str(row["parent_sample_id"]) for row in rows]
        ),
        "state_matrix_sha256": state_matrix_sha256(states),
        "state_dimension": int(states.shape[1]),
        "class_counts": dict(fitted["class_counts"]),
        "pooled_degrees_of_freedom": int(fitted["pooled_degrees_of_freedom"]),
        "variance_floor": float(fitted["variance_floor"]),
        "unknown_reference_fit_exposure": 0,
    }
    return fitted, metadata


def _scored_rows(
    rows: Sequence[Mapping[str, Any]], fitted: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("V26h cannot score an empty mixed population")
    states = np.stack([np.asarray(row["state"], dtype=np.float64) for row in rows], axis=0)
    scores, closest = mahalanobis_scores(states, fitted)
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = {key: value for key, value in raw.items() if key != "state"}
        row.update(
            {
                "closest_reference_class": L1_CLASSES[int(closest[index])],
                "mahalanobis_novelty_score": float(scores[index]),
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
    return result


def _per_label_summary(
    rows: Sequence[Mapping[str, Any]], *, score_key: str
) -> list[dict[str, Any]]:
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


def _capture_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["capture_id"])].append(row)
    result: list[dict[str, Any]] = []
    for capture_id, members in sorted(grouped.items()):
        unknown = [
            float(row["mahalanobis_novelty_score"])
            for row in members
            if bool(row["is_unknown_truth"])
        ]
        known = [
            float(row["mahalanobis_novelty_score"])
            for row in members
            if not bool(row["is_unknown_truth"])
        ]
        result.append(
            {
                "capture_id": capture_id,
                "parent_support": int(len(members)),
                "known_parent_support": int(len(known)),
                "unknown_parent_support": int(len(unknown)),
                "unknown_score_median": float(np.median(unknown)) if unknown else None,
                "known_score_median": float(np.median(known)) if known else None,
                "unknown_vs_known_auroc_within_capture": (
                    binary_ranking_metrics(known, unknown)["auroc_unknown_vs_known"]
                    if known and unknown
                    else None
                ),
            }
        )
    return result


def _bootstrap_ranking(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    known = [row for row in rows if not bool(row["is_unknown_truth"])]
    unknown = [row for row in rows if bool(row["is_unknown_truth"])]
    if not known or not unknown:
        raise ValueError("V26h bootstrap needs both known and Unknown parents")
    rng = np.random.default_rng(seed)
    known_indices = np.arange(len(known), dtype=np.int64)
    unknown_indices = np.arange(len(unknown), dtype=np.int64)
    values: dict[str, list[float]] = {
        "auroc_unknown_vs_known": [],
        "average_precision_unknown": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_known = rng.choice(known_indices, size=len(known_indices), replace=True)
        sampled_unknown = rng.choice(unknown_indices, size=len(unknown_indices), replace=True)
        metric = binary_ranking_metrics(
            [float(known[index]["mahalanobis_novelty_score"]) for index in sampled_known],
            [float(unknown[index]["mahalanobis_novelty_score"]) for index in sampled_unknown],
        )
        for key in values:
            values[key].append(float(metric[key]))
    return {
        "method": "parent bootstrap; known and Unknown parent rows resampled independently",
        "replicates": int(BOOTSTRAP_REPLICATES),
        "seed": int(seed),
        "confidence_level": 0.95,
        "known_parent_support": int(len(known)),
        "unknown_parent_support": int(len(unknown)),
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
    known_mahalanobis = [
        float(row["mahalanobis_novelty_score"])
        for row in rows
        if not bool(row["is_unknown_truth"])
    ]
    unknown_mahalanobis = [
        float(row["mahalanobis_novelty_score"])
        for row in rows
        if bool(row["is_unknown_truth"])
    ]
    known_msp = [
        float(row["max_softmax_unknownness"])
        for row in rows
        if not bool(row["is_unknown_truth"])
    ]
    unknown_msp = [
        float(row["max_softmax_unknownness"])
        for row in rows
        if bool(row["is_unknown_truth"])
    ]
    return {
        "mahalanobis": binary_ranking_metrics(known_mahalanobis, unknown_mahalanobis),
        "max_softmax_unknownness_recomputed_comparator": binary_ranking_metrics(
            known_msp, unknown_msp
        ),
        "per_label_mahalanobis": _per_label_summary(
            rows, score_key="mahalanobis_novelty_score"
        ),
        "per_label_msp_unknownness": _per_label_summary(
            rows, score_key="max_softmax_unknownness"
        ),
        "bootstrap": _bootstrap_ranking(rows, seed=BOOTSTRAP_SEED + int(rows[0]["fold_id"])),
        "capture_strata": _capture_summary(rows),
    }


def _aggregate_fold_metrics(
    folds: Sequence[Mapping[str, Any]], *, split: str
) -> dict[str, Any]:
    measures = (
        ("mahalanobis", "auroc_unknown_vs_known"),
        ("mahalanobis", "average_precision_unknown"),
        ("max_softmax_unknownness_recomputed_comparator", "auroc_unknown_vs_known"),
        ("max_softmax_unknownness_recomputed_comparator", "average_precision_unknown"),
    )
    summary: dict[str, Any] = {"split": split, "fold_count": len(folds), "fold_values": {}}
    for method, metric in measures:
        values = np.asarray(
            [float(fold[split]["metrics"][method][metric]) for fold in folds],
            dtype=np.float64,
        )
        summary["fold_values"][f"{method}.{metric}"] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


def _parent_detail_csv(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
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
        description="parent detail CSV",
    )


def _assert_mixed_population(rows: Sequence[Mapping[str, Any]]) -> None:
    for split, expected_labels in EXPECTED_PARENT_COUNTS.items():
        for fold_id in range(FOLD_COUNT):
            fold_rows = [
                row
                for row in rows
                if str(row["split"]) == split and int(row["fold_id"]) == fold_id
            ]
            observed: dict[str, int] = defaultdict(int)
            parent_ids: list[str] = []
            for row in fold_rows:
                observed[str(row["label_name"])] += 1
                parent_ids.append(str(row["parent_sample_id"]))
            if dict(observed) != dict(expected_labels):
                raise ValueError(
                    f"V26h {split} fold {fold_id} mixed parent support differs from V26f"
                )
            if len(parent_ids) != len(set(parent_ids)):
                raise ValueError(f"V26h {split} fold {fold_id} repeats a mixed parent")
            if len(fold_rows) != int(sum(expected_labels.values())):
                raise ValueError(f"V26h {split} fold {fold_id} parent coverage differs")


def _isolated_fold_command(root: Path, *, fold_id: int) -> list[str]:
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "report.known_gate_v26h",
        "--project-root",
        str(root),
        "--isolated-fold",
        "--fold-id",
        str(int(fold_id)),
    ]


def _run_isolated_fold(root: Path, *, fold_id: int) -> dict[str, Any]:
    """Run one bounded-memory fold worker and parse its compact JSON result."""

    environment = os.environ.copy()
    # The worker is a diagnostic replay, not a training run.  CPU avoids CUDA
    # allocator retention and makes the fold subprocess' memory bound stable;
    # checkpoint/model/data semantics remain unchanged and are recorded.
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
            "V26h isolated fold worker failed; "
            f"fold={fold_id}; returncode={completed.returncode}; "
            f"stderr={completed.stderr[-4000:]}"
        )
    output = completed.stdout.strip()
    if not output or len(output.encode("utf-8")) > MAX_ISOLATED_FOLD_OUTPUT_BYTES:
        raise MemoryError("V26h isolated fold output is missing or exceeds its bound")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("V26h isolated fold worker emitted non-JSON output") from exc
    if not isinstance(payload, dict):
        raise ValueError("V26h isolated fold worker output must be an object")
    if (
        payload.get("status") != "passed"
        or int(payload.get("fold_id", -1)) != int(fold_id)
        or set(payload.get("validation", {})) != {"parent_count", "parent_ids_sha256", "metrics", "rows"}
        or set(payload.get("test", {})) != {"parent_count", "parent_ids_sha256", "metrics", "rows"}
    ):
        raise ValueError("V26h isolated fold worker binding differs")
    return payload


def _run_isolated_fold_worker(root: Path, *, fold_id: int) -> dict[str, Any]:
    """Worker entry point; all source validation happens before inference."""

    # Keep the short-lived CPU worker bounded on Windows and avoid creating a
    # large BLAS thread pool for each fold.  This is an execution safeguard;
    # it does not alter model weights, inputs, ordering or score semantics.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    prepared_contract, _readiness = validate_prepared_contract(root)
    model, config, device, binding = _fold_l1_runtime(
        root, fold_id=fold_id, prepared_contract=prepared_contract, force_cpu=True
    )
    if device.type != "cpu":
        raise AssertionError("V26h isolated fold worker escaped its CPU boundary")
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
    fitted, reference_metadata = _reference_fit(train_rows)
    _clear_dataset_cache(train_dataset)
    del train_dataset, train_rows
    gc.collect()

    split_results: dict[str, dict[str, Any]] = {}
    for split, cache_path in (("val", L1_VALIDATION_CACHE), ("test", L1_TEST_CACHE)):
        mixed_dataset = V26OpenSetAuditDataset(
            root,
            config,
            split_name=split,
            cache_path=root / cache_path,
        )
        expected_ids = tuple(str(value) for value in mixed_dataset.parent_ids)
        expected_counts = dict(EXPECTED_PARENT_COUNTS[split])
        if mixed_dataset.label_counts() != expected_counts:
            raise ValueError(f"V26h {split} mixed labels differ from frozen V26f counts")
        mixed_rows = _forward_dataset(
            mixed_dataset,
            model=model,
            device=device,
            fold_id=fold_id,
            split=split,
            expected_parent_ids=expected_ids,
        )
        scored = _scored_rows(mixed_rows, fitted)
        split_results[split] = {
            "parent_count": int(len(scored)),
            "parent_ids_sha256": parent_id_sequence_sha256(
                [str(row["parent_sample_id"]) for row in scored]
            ),
            "metrics": _fold_metrics(scored),
            "rows": scored,
        }
        _clear_dataset_cache(mixed_dataset)
        del mixed_dataset, mixed_rows, scored
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
    del model, config, fitted, split_results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prepared_contract, readiness = validate_prepared_contract(root)
    del prepared_contract, readiness
    folds: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = _run_isolated_fold(root, fold_id=fold_id)
        folds.append(fold)
        details.extend(list(fold["validation"]["rows"]))
        details.extend(list(fold["test"]["rows"]))
    _assert_mixed_population(details)
    detail_status = _parent_detail_csv(root, details)
    aggregate = {
        "validation": _aggregate_fold_metrics(folds, split="validation"),
        "test": _aggregate_fold_metrics(folds, split="test"),
        "interpretation": (
            "fold-level threshold-free diagnostics; no state, centre, variance, "
            "threshold or semantic action is pooled for deployment"
        ),
    }
    report = {
        "status": "completed_v26h_train_only_mahalanobis_ood_ranking_diagnostic",
        "schema_version": REPORT_SCHEMA,
        "experiment": "V26h frozen L1 train-only Mahalanobis OOD ranking diagnostic",
        "scope": (
            "CQ08 V26e train-only reference fitting plus V26f frozen mixed "
            "validation/test ranking; no action replay or calibration"
        ),
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "readiness_path": relative_path(root, root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "ood_score_policy": OOD_SCORE_POLICY,
        "representation": {
            "name": REPRESENTATION_NAME,
            "verified_classifier_path": (
                "KnownGateClassifier.classifier[0] LayerNorm before classifier[1] Linear"
            ),
            "raw_liquid_final_state_used_directly": False,
            "gradient_tracking": False,
        },
        "implementation_sha256": implementation_sha256(root),
        "source_protocol": {
            "v26e_l1": (
                "five fixed epoch-32 Known-only OOF checkpoints; each fold "
                "reference fit uses only its own train parents"
            ),
            "v26f_mixed_data": "frozen mixed val/test caches after model/checkpoint freeze",
            "unknown_training_exposure": 0,
            "unknown_normalization_exposure": 0,
            "unknown_reference_fit_exposure": 0,
            "unknown_checkpoint_selection_exposure": 0,
            "unknown_threshold_fit_exposure": 0,
            "unknown_truth_use": "final threshold-free ranking report only",
        },
        "folds": folds,
        "aggregate_fold_diagnostics": aggregate,
        "parent_detail_csv": relative_path(root, root / DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / DETAIL_CSV),
        "parent_detail_csv_write": detail_status,
        "comparison_boundary": {
            "msp": (
                "recomputed descriptive max-softmax unknownness comparator; "
                "V26a threshold and mixed-validation selection are not reused"
            ),
            "v12_v13": (
                "not a direct replication: those used old DBB specialist subclass "
                "state; V26h uses current L1 KnownTarget/Clutter readout state"
            ),
            "threshold_selection": "not_executed",
            "semantic_action_change": "not_executed",
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
    write_immutable_json(
        root / REPORT,
        report,
        max_bytes=MAX_REPORT_BYTES,
        description="report",
    )
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
    parser = argparse.ArgumentParser(description="Run V26h train-only Mahalanobis OOD diagnostic.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--isolated-fold", action="store_true")
    parser.add_argument("--fold-id", type=int, choices=range(FOLD_COUNT))
    args = parser.parse_args()
    if args.isolated_fold:
        if args.fold_id is None:
            parser.error("--isolated-fold requires --fold-id")
        print(
            json.dumps(
                _run_isolated_fold_worker(
                    args.project_root.resolve(), fold_id=int(args.fold_id)
                ),
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