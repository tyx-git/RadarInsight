"""Evaluate V26 known-only metrics and the post-checkpoint open-set audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v26 import (
    ARTIFACT,
    AUDIT_VALIDATION_CACHE,
    CONFIG,
    EXPECTED_PARENT_COUNTS,
    EXPERIMENT_CONTRACT,
    FEATURE_MANIFEST,
    READINESS,
    materialize_test_audit_cache,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    KNOWN_GATE_V26_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    known_gate_device,
    load_known_gate_config,
    sha256_file,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v26_audit import (
    V26OpenSetAuditDataset,
    build_v26_open_set_audit_dataloader,
)
from train.known_gate_v21 import (
    _load_checkpoint,
    _model_inputs,
    _verify_contract,
    _verify_readiness,
)


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26-known-gate-known-only-open-set-development.json"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 26043


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


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


def _metric_vector(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics = summarize_known_gate_records(rows, scope="v26_bootstrap")
    by_class = {str(row["class"]): row for row in metrics["per_class"]}
    target = by_class["KNOWN_TARGET_EVIDENCE"]
    clutter = by_class["KNOWN_CLUTTER_EVIDENCE"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "known_target_recall": float(target["recall"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_f1": float(clutter["f1"]),
    }


def _bootstrap(parent_rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(parent_rows) if row["label_name"] == label],
            dtype=np.int64,
        )
        for label in KNOWN_LABELS
    }
    support = {label: int(indices.size) for label, indices in strata.items()}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in strata.values()]
        )
        vector = _metric_vector([parent_rows[int(index)] for index in sampled])
        for name, value in vector.items():
            values.setdefault(name, []).append(value)
    return {
        "method": "stratified parent bootstrap by known base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "confidence_level": 0.95,
        "stratum_support": support,
        "metrics": {
            name: {
                "median": float(np.median(data)),
                "lower_95": float(np.quantile(data, 0.025)),
                "upper_95": float(np.quantile(data, 0.975)),
            }
            for name, data in values.items()
        },
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    manifest_path = root / FEATURE_MANIFEST
    artifact = root / ARTIFACT
    summary_path = artifact / "training-summary.json"
    checkpoint_path = artifact / "best_model.pt"
    detail_path = artifact / "sample_detail_information.json"
    required = (
        config_path,
        readiness_path,
        contract_path,
        manifest_path,
        summary_path,
        checkpoint_path,
        detail_path,
        root / AUDIT_VALIDATION_CACHE,
    )
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"V26 report inputs are missing: {missing}")
    config = load_known_gate_config(config_path)
    contract_evidence = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract_evidence)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != config.variant
        or len(list(summary.get("history") or ())) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V26 known-only checkpoint is not a completed frozen 20-epoch artifact")
    checkpoint_frozen_sha256 = sha256_file(checkpoint_path)
    device = known_gate_device(config)
    model, reference, pairing = build_known_gate_model_for_config(root, config)
    del reference, pairing
    # ``build_known_gate_model_for_config`` constructs the model on CPU.  The
    # audit batches are moved to the selected inference device below, so move
    # the model before loading and evaluating the frozen checkpoint as well.
    model.to(device)
    _load_checkpoint(
        checkpoint_path,
        model,
        None,
        config=config,
        config_path=config_path,
        readiness_sha256=readiness["sha256"],
        map_location=device,
    )
    # The test audit is intentionally called only after the immutable best
    # checkpoint has been loaded and its file hash recorded above.
    validation = _run_split(
        model,
        config,
        root,
        split="val",
        cache_path=root / AUDIT_VALIDATION_CACHE,
        device=device,
        minimum_evidence_events=config.train.prefix_auxiliary_min_events,
    )
    test_cache_path, test_cache_statistics = materialize_test_audit_cache(root)
    test = _run_split(
        model,
        config,
        root,
        split="test",
        cache_path=test_cache_path,
        device=device,
        minimum_evidence_events=config.train.prefix_auxiliary_min_events,
    )
    val_known_parent = list(validation["records"]["parent"])
    val_known_parent = [row for row in val_known_parent if row["true_id"] >= 0]
    test_known_parent = [row for row in test["records"]["parent"] if row["true_id"] >= 0]
    report = {
        "status": "completed_development_open_set_audit",
        "schema_version": "RADAR-LN3-V26-known-gate-known-only-open-set-development-v1",
        "variant": config.variant,
        "task_name": "known_gate",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_frozen_sha256,
        "checkpoint_epoch": int(summary.get("best_epoch", -1)),
        "checkpoint_frozen_before_test_audit": True,
        "test_audit_cache_materialized_after_checkpoint_frozen": True,
        "test_audit_cache_statistics": test_cache_statistics,
        "known_only_training": {
            "train_parent_count": int(summary.get("train_parent_count", -1)),
            "validation_parent_count": int(summary.get("validation_parent_count", -1)),
            "unknown_is_model_class": summary.get("unknown_is_model_class"),
            "test_training_access": summary.get("test_evaluation"),
            "history_epochs": len(list(summary.get("history") or ())),
        },
        "validation_mixed_open_set_audit": validation,
        "test_mixed_open_set_audit": test,
        "parent_bootstrap": {
            "validation_known_only": _bootstrap(val_known_parent, seed=BOOTSTRAP_SEED),
            "test_known_only": _bootstrap(test_known_parent, seed=BOOTSTRAP_SEED + 1),
        },
        "support_contract": {
            "validation": EXPECTED_PARENT_COUNTS["val"],
            "test": EXPECTED_PARENT_COUNTS["test"],
            "clutter_validation_parent_support": 9,
            "clutter_test_parent_support": 8,
        },
        "unknown_semantics": {
            "unknown_is_strict_open_set_truth": True,
            "unknown_is_not_a_binary_model_class": True,
            "binary_gate_false_accept_rate_is_diagnostic_only": True,
            "unknown_rejection": "not_available_without_knownness_or_rejection_head",
            "pending_abstain": "not_available_and_not_counted_as_unknown_rejection",
        },
        "contract_hashes": {
            "data_contract_sha256": sha256_file(root / config.data.contract_path),
            "normalization_sha256": sha256_file(root / config.data.normalization_stats_path),
            "experiment_contract_sha256": sha256_file(contract_path),
            "feature_manifest_sha256": sha256_file(manifest_path),
        },
        "OOF": "not_executed",
        "full_cascade": "not_executed",
        "formal_rejection": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, report)
    return {
        "status": report["status"],
        "report": str(root / OUTPUT),
        "report_sha256": sha256_file(root / OUTPUT),
        "validation_known_parent_support": len(val_known_parent),
        "validation_unknown_parent_support": validation["unknown_parent_count"],
        "test_known_parent_support": len(test_known_parent),
        "test_unknown_parent_support": test["unknown_parent_count"],
        "test": "completed_post_checkpoint_open_set_audit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the V26 known-only checkpoint and mixed open-set audit.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
