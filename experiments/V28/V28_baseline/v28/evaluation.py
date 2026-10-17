"""Known-only V28 endpoint evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import V28TaskDataset, build_dataloader, model_input_kwargs
from .checkpoint import load_checkpoint
from .metrics import confusion_metrics
from .model import V28TaskModel
from .protocol import V28Context, V28_REPORT_SCHEMA, artifact_path
from radar_ln3.runtime import atomic_write_text, json_text, relative_project_path, sha256_file


def evaluate_model(model: V28TaskModel, dataset: V28TaskDataset, *, context: V28Context, device: torch.device, epoch: int | None = None) -> dict[str, Any]:
    if model.task != dataset.task:
        raise ValueError("V28 evaluation model/dataset task differs")
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_dataloader(dataset, batch_size=int(context.config.training["batch_size"]), shuffle=False, seed=int(context.config.training["dataloader_seed"])):
            output = model(**model_input_kwargs(batch, device))
            probabilities = torch.softmax(output["sequence_logits"], dim=-1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("V28 evaluation probabilities are non-finite")
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][row])
                final_probability = probabilities[row, length - 1]
                records.append({
                    "parent_sample_id": str(parent_id),
                    "track_id": str(batch["track_ids"][row]),
                    "source_name": str(batch["source_names"][row]),
                    "true_id": int(batch["labels"][row]),
                    "pred_id": int(final_probability.argmax()),
                    "probabilities": [float(value) for value in final_probability.cpu().tolist()],
                    "label_name": str(batch["task_label_names"][row]),
                    "event_count": length,
                    "final_raw_state_norm": float(torch.linalg.vector_norm(output["final_raw_state"][row]).item()),
                })
    metrics = confusion_metrics(records, class_names=dataset.class_names, scope="parent_final")
    return {
        "schema_version": V28_REPORT_SCHEMA,
        "experiment_id": "V28",
        "task_name": model.task,
        "split": dataset.split,
        "checkpoint_epoch": epoch,
        "class_names": list(dataset.class_names),
        "parent_count": len(records),
        "parent_metrics": metrics,
        "records": records,
        "evaluation_scope": {"unknown": "excluded_known_only", "test": "sealed", "classification_endpoint": "known_only"},
    }


def evaluate_checkpoint(
    context: V28Context,
    *,
    task: str,
    checkpoint: str | Path,
    seed: int,
    split: str = "validation",
    device: torch.device,
) -> dict[str, Any]:
    """Load one provenance-bound checkpoint and write a reproducible report."""

    if split not in {"train", "validation"}:
        raise ValueError("V28 evaluation split is invalid")
    model = V28TaskModel(context, task=task).to(device)
    checkpoint_path = Path(checkpoint)
    payload = load_checkpoint(context, model, checkpoint_path, task=task, seed=seed, map_location=device)
    dataset = V28TaskDataset(context, task=task, split=split)
    report = evaluate_model(model, dataset, context=context, device=device, epoch=int(payload["epoch"]))
    resolved = checkpoint_path if checkpoint_path.is_absolute() else context.project_root / checkpoint_path
    report.update(
        {
            "checkpoint_path": relative_project_path(context.project_root, resolved),
            "checkpoint_sha256": sha256_file(resolved),
            "checkpoint_kind": str(payload["kind"]),
            "checkpoint_manifest_compatibility": payload.get("manifest_compatibility"),
            "seed": int(seed),
            "manifest_sha256": context.manifest_sha256,
            "config_sha256": context.config.sha256,
        }
    )
    output = artifact_path(context.project_root, str(context.config.artifacts["evaluation_template"]), task=task, seed=seed)
    atomic_write_text(output, json_text(report))
    return {"report_path": relative_project_path(context.project_root, output), "report_sha256": sha256_file(output), "report": report}


__all__ = ["evaluate_checkpoint", "evaluate_model"]
