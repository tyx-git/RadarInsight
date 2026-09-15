"""Paired r21 evaluation for adapter-only radial-sidecar training."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from v27.v27_baseline.metrics import summarize_records

from .data import R21Dataset, build_r21_loader, r21_model_input_kwargs
from .protocol import (
    R21_ARMS,
    R21_EXPERIMENT_ID,
    R21_L2_CLASSES,
    R21_REPORT_SCHEMA,
    R21Config,
    artifact_path,
    load_manifest,
    seal,
    sha256_file,
    write_json_if_identical,
)
from .training import load_checkpoint


def _device(config: R21Config, requested: str | None) -> torch.device:
    value = str(requested or config.training.get("device", "cpu")).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("r21 evaluation requested CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r21 evaluation device: {value}")


def _records(
    model: torch.nn.Module,
    dataset: R21Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r21_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r21_model_input_kwargs(batch, device))
            logits = output["window_logits"].detach().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("r21 evaluation logits are non-finite")
            for index, parent_id in enumerate(batch["parent_sample_ids"]):
                final = logits[index]
                rows.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "source_suffix": str(parent_id).split("|")[2],
                        "label_name": str(batch["label_names"][index]),
                        "true_id": int(batch["l2_labels"][index]),
                        "pred_id": int(final.argmax().item()),
                        "probabilities": [float(value) for value in torch.softmax(final, dim=-1).tolist()],
                        "event_count": int(batch["lengths"][index]),
                    }
                )
    if [row["parent_sample_id"] for row in rows] != dataset.parent_ids:
        raise ValueError("r21 evaluation changed parent order")
    return rows


def _source_metrics(records: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_suffix"])].append(row)
    return {
        source: {
            "support": len(rows),
            "metrics": summarize_records(rows, class_names=R21_L2_CLASSES, scope=f"{scope}:source={source}"),
        }
        for source, rows in sorted(grouped.items())
    }


def _paired(control: list[dict[str, Any]], treatment: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["parent_sample_id"]: row for row in control}
    right = {row["parent_sample_id"]: row for row in treatment}
    if set(left) != set(right):
        raise ValueError("r21 arms have different parent populations")
    transitions: dict[str, int] = defaultdict(int)
    source_transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    changed: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        before, after = left[parent_id], right[parent_id]
        if before["true_id"] != after["true_id"]:
            raise ValueError("r21 arms have different labels")
        before_correct = before["true_id"] == before["pred_id"]
        after_correct = after["true_id"] == after["pred_id"]
        key = f"{'correct' if before_correct else 'wrong'}_to_{'correct' if after_correct else 'wrong'}"
        source = str(before["source_suffix"])
        transitions[key] += 1
        source_transitions[source][key] += 1
        if before["pred_id"] != after["pred_id"]:
            changed.append(
                {
                    "parent_sample_id": parent_id,
                    "source_suffix": source,
                    "label_name": before["label_name"],
                    "control_prediction": before["pred_id"],
                    "treatment_prediction": after["pred_id"],
                    "control_correct": before_correct,
                    "treatment_correct": after_correct,
                }
            )
    return {
        "parent_count": len(left),
        "correctness_transitions": dict(sorted(transitions.items())),
        "source_correctness_transitions": {
            source: dict(sorted(values.items()))
            for source, values in sorted(source_transitions.items())
        },
        "changed_predictions": changed,
    }


def evaluate_comparison(
    project_root: str | Path,
    config: R21Config,
    *,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    _, manifest_path, manifest_sha256 = load_manifest(root, config, allow_code_drift=True)
    runtime_device = _device(config, device)
    reports: dict[str, Any] = {}
    for arm in R21_ARMS:
        model, checkpoint = load_checkpoint(root, config, arm=arm, seed=int(seed))
        split_reports: dict[str, Any] = {}
        for split in ("train", "validation"):
            dataset = R21Dataset(root, config, split_name=split, arm=arm)
            rows = _records(
                model.to(runtime_device),
                dataset,
                device=runtime_device,
                batch_size=int(config.training["batch_size"]),
            )
            split_reports[split] = {
                "metrics": summarize_records(rows, class_names=R21_L2_CLASSES, scope=f"{split}:{arm}:parent_final"),
                "source_metrics": _source_metrics(rows, f"{split}:{arm}"),
                "records": rows,
            }
        checkpoint_path = artifact_path(root, config.artifacts["checkpoint_template"], arm=arm, seed=int(seed))
        reports[arm] = {
            "checkpoint": {
                "path": checkpoint_path.relative_to(root).as_posix(),
                "sha256": sha256_file(checkpoint_path),
                "epoch": int(checkpoint["epoch"]),
                "model_state_sha256": checkpoint["model_state_sha256"],
                "frozen_base_state_sha256": checkpoint["frozen_base_state_sha256"],
                "manifest_sha256": checkpoint["manifest_sha256"],
            },
            "splits": split_reports,
        }
    payload = {
        "schema_version": R21_REPORT_SCHEMA,
        "experiment_id": R21_EXPERIMENT_ID,
        "status": "completed_paired_radial_adapter_only",
        "device": str(runtime_device),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "data_scope": {
            "source": "r20 exact data and sidecar population",
            "unknown_payload": "not_created",
            "test": "not_created",
            "rd_payload": "not_available",
        },
        "variable": {
            "main": "trainable_scope",
            "control": "zero_control adapter-only; input sidecar zero",
            "treatment": "radial_decomposed adapter-only; r20 four-field sidecar",
            "fixed": "r20 data, structure, split, seed, optimizer, lr, weight_decay, loss, epoch budget",
        },
        "reports": reports,
        "paired": {
            split: _paired(
                reports["zero_control"]["splits"][split]["records"],
                reports["radial_decomposed"]["splits"][split]["records"],
            )
            for split in ("train", "validation")
        },
        "training": "executed_for_both_arms",
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "rd_payload_access_count": 0,
        "l3": "not_executed",
        "not_proven": [
            "Unknown rejection",
            "open-set calibration",
            "cross-source unseen-domain generalization",
            "deployment performance",
        ],
    }
    path = artifact_path(root, config.artifacts["comparison_report_path"])
    write_json_if_identical(path, seal(payload))
    return {"report_path": str(path), **payload}


__all__ = ["evaluate_comparison"]
