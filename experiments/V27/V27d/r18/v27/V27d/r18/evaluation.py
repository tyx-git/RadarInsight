"""Formal paired evaluation for the r18 trained arms."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from v27.v27_baseline.metrics import summarize_records

from .data import R18Dataset, build_r18_loader, r17_l2_inputs
from .protocol import (
    R18_COMPARISON_REPORT_SCHEMA,
    R18_L2_CLASSES,
    R18Config,
    artifact_path,
    load_manifest,
    seal,
    sha256_file,
    write_json_if_identical,
)
from .training import load_checkpoint


def _device(config: R18Config, requested: str | None) -> torch.device:
    value = str(requested or config.training.get("device", "cpu")).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("r18 evaluation requested CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r18 evaluation device: {value}")


def _records(
    model: torch.nn.Module,
    dataset: R18Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r18_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r17_l2_inputs(batch, device))
            logits = output["window_logits"].detach().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("r18 evaluation logits are non-finite")
            for index, parent_id in enumerate(batch["parent_sample_ids"]):
                final = logits[index]
                rows.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "source_suffix": str(parent_id).split("|")[2],
                        "label_name": str(batch["label_names"][index]),
                        "true_id": int(batch["l2_labels"][index]),
                        "pred_id": int(final.argmax().item()),
                        "probabilities": [
                            float(value)
                            for value in torch.softmax(final, dim=-1).tolist()
                        ],
                        "event_count": int(batch["lengths"][index]),
                        "range_clipped_event_count": int(
                            batch["l2_range_clipped_count"][index]
                        ),
                        "speed_clipped_event_count": int(
                            batch["l2_speed_clipped_count"][index]
                        ),
                    }
                )
    if [row["parent_sample_id"] for row in rows] != dataset.parent_ids:
        raise ValueError("r18 evaluation changed deterministic parent order")
    return rows


def _source_metrics(records: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_suffix"])].append(row)
    return {
        source: {
            "support": len(rows),
            "metrics": summarize_records(
                rows,
                class_names=R18_L2_CLASSES,
                scope=f"{scope}:source={source}",
            ),
        }
        for source, rows in sorted(grouped.items())
    }


def _paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    before = {row["parent_sample_id"]: row for row in left}
    after = {row["parent_sample_id"]: row for row in right}
    if set(before) != set(after):
        raise ValueError("r18 trained arms have different parent populations")
    transitions: dict[str, int] = defaultdict(int)
    source_transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    changed: list[dict[str, Any]] = []
    for parent_id in sorted(before):
        old = before[parent_id]
        new = after[parent_id]
        if old["true_id"] != new["true_id"]:
            raise ValueError("r18 trained arms have different labels")
        old_correct = old["true_id"] == old["pred_id"]
        new_correct = new["true_id"] == new["pred_id"]
        key = f"{'correct' if old_correct else 'wrong'}_to_{'correct' if new_correct else 'wrong'}"
        source = str(old["source_suffix"])
        transitions[key] += 1
        source_transitions[source][key] += 1
        if old["pred_id"] != new["pred_id"]:
            changed.append(
                {
                    "parent_sample_id": parent_id,
                    "source_suffix": source,
                    "label_name": old["label_name"],
                    "full_prediction": old["pred_id"],
                    "ablation_prediction": new["pred_id"],
                    "full_correct": old_correct,
                    "ablation_correct": new_correct,
                }
            )
    return {
        "parent_count": len(before),
        "correctness_transitions": dict(sorted(transitions.items())),
        "source_correctness_transitions": {
            source: dict(sorted(values.items()))
            for source, values in sorted(source_transitions.items())
        },
        "changed_predictions": changed,
    }


def evaluate_comparison(
    project_root: str | Path,
    config: R18Config,
    *,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate both trained arms and write one immutable comparison report."""

    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    runtime_device = _device(config, device)
    frozen = None
    reports: dict[str, Any] = {}
    loaded: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for arm in ("full", "radial_speed_ablation"):
        model, checkpoint = load_checkpoint(
            root,
            config,
            arm=arm,
            seed=int(seed),
            frozen=frozen,
        )
        if frozen is None:
            # load_checkpoint internally loads the frozen source.  Keeping the
            # models independent is intentional; evaluation never mutates it.
            pass
        dataset_reports: dict[str, Any] = {}
        for split in ("train", "validation"):
            dataset = R18Dataset(
                root,
                config,
                split_name=split,
                radial_speed_ablation=arm == "radial_speed_ablation",
            )
            rows = _records(
                model.to(runtime_device),
                dataset,
                device=runtime_device,
                batch_size=int(config.training["batch_size"]),
            )
            dataset_reports[split] = {
                "metrics": summarize_records(
                    rows,
                    class_names=R18_L2_CLASSES,
                    scope=f"{split}:{arm}:parent_final",
                ),
                "source_metrics": _source_metrics(rows, f"{split}:{arm}"),
                "records": rows,
            }
        loaded[arm] = (model.cpu(), checkpoint)
        reports[arm] = {
            "checkpoint": {
                "path": str(
                    artifact_path(
                        root,
                        config.artifacts["checkpoint_template"],
                        arm=arm,
                        seed=int(seed),
                    )
                ),
                "sha256": sha256_file(
                    artifact_path(
                        root,
                        config.artifacts["checkpoint_template"],
                        arm=arm,
                        seed=int(seed),
                    )
                ),
                "epoch": int(checkpoint["epoch"]),
                "model_state_sha256": checkpoint["model_state_sha256"],
                "manifest_sha256": checkpoint["manifest_sha256"],
            },
            "splits": dataset_reports,
        }
    paired: dict[str, Any] = {}
    for split in ("train", "validation"):
        paired[split] = _paired(
            reports["full"]["splits"][split]["records"],
            reports["radial_speed_ablation"]["splits"][split]["records"],
        )
    payload = {
        "schema_version": R18_COMPARISON_REPORT_SCHEMA,
        "experiment_id": "V27d-r18",
        "status": "completed_paired_source_stratified_training_comparison",
        "device": str(runtime_device),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "split_protocol": manifest["split"],
        "data_scope": {
            "source": "OriData/excel",
            "unknown_payload": "not_created",
            "test": "not_created",
            "rd_payload": "not_available",
            "snr_used": False,
        },
        "arms": {
            "full": "all frozen r17 TR fields",
            "radial_speed_ablation": "radial_speed_mps and its radial-derived causal increments removed",
        },
        "reports": reports,
        "paired": paired,
        "training": "executed_for_both_arms",
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "l3": "not_executed",
        "not_proven": [
            "unseen-source generalization",
            "strict open-set Unknown rejection",
            "L3 calibrated actions",
            "deployment performance",
        ],
    }
    path = artifact_path(root, config.artifacts["comparison_report_path"])
    write_json_if_identical(path, seal(payload))
    return {"report_path": str(path), **payload}


__all__ = ["evaluate_comparison"]