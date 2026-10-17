"""Known-class evaluation for either independent V27 task model."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.runtime import canonical_sha256

from .data import V27TaskDataset, build_v27_dataloader, model_input_kwargs
from .metrics import parent_balanced_prefix_records, parent_bootstrap, summarize_records
from .model import V27LiquidV2TaskModel
from .protocol import V27Config, V27_REPORT_SCHEMA, task_class_names


def evaluate_v27(model: V27LiquidV2TaskModel, dataset: V27TaskDataset, *, config: V27Config, device: torch.device, batch_size: int | None = None, bootstrap_seed: int = 2700, bootstrap_replicates: int = 0, checkpoint_epoch: int | None = None) -> dict[str, Any]:
    if model.task != dataset.task:
        raise ValueError("V27 evaluation model/dataset task differs")
    classes = task_class_names(model.task)
    model.eval()
    parent_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_v27_dataloader(dataset, batch_size=int(batch_size or config.training.batch_size), shuffle=False):
            output = model(**model_input_kwargs(batch, device), return_trace=True)
            logits = output["sequence_logits"]
            probabilities = torch.softmax(logits, dim=-1)
            if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
                raise FloatingPointError("V27 evaluation produced non-finite logits")
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][row])
                if not 0 < length <= config.data.max_sequence_length:
                    raise ValueError("V27 evaluation parent length differs")
                label = int(batch["labels"][row])
                final = probabilities[row, length - 1]
                parent_records.append({"parent_sample_id": str(parent_id), "track_id": str(batch["track_ids"][row]), "source_name": str(batch["source_names"][row]), "label_name": str(batch["task_label_names"][row]), "base_label_name": str(batch["base_label_names"][row]), "true_id": label, "pred_id": int(final.argmax()), "probabilities": [float(value) for value in final.cpu().tolist()], "event_count": length, "final_raw_state_norm": float(torch.linalg.vector_norm(output["liquid_sequence"][row, length - 1]).item()), "final_knownness_logit": float(output["knownness_logit"][row].item()), "final_quality_logit_unfitted": float(output["quality_logit"][row].item())})
                for event in range(length):
                    probability = probabilities[row, event]
                    event_records.append({"parent_sample_id": str(parent_id), "track_id": str(batch["track_ids"][row]), "source_name": str(batch["source_names"][row]), "true_id": label, "pred_id": int(probability.argmax()), "probabilities": [float(value) for value in probability.cpu().tolist()], "event_index": event, "window_length": event + 1, "parent_length": length, "label_name": str(batch["task_label_names"][row])})
    eligible = [row for row in event_records if row["window_length"] >= config.training.eligible_prefix_min_events and row["window_length"] < row["parent_length"]]
    balanced = parent_balanced_prefix_records(eligible, minimum_events=config.training.eligible_prefix_min_events, class_names=classes)
    bootstrap = parent_bootstrap(parent_records, class_names=classes, seed=bootstrap_seed, replicates=bootstrap_replicates) if bootstrap_replicates else {"method": "not_executed", "replicates": 0}
    return {"schema_version": V27_REPORT_SCHEMA, "experiment_id": "V27", "task_name": model.task, "split": dataset.split_name, "checkpoint_epoch": checkpoint_epoch, "class_names": list(classes), "parent_count": len(parent_records), "window_count": len(event_records), "parent_metrics": summarize_records(parent_records, class_names=classes, scope="parent_final"), "parent_balanced_eligible_prefix_metrics": summarize_records(balanced, class_names=classes, scope="parent_balanced_eligible_prefix_diagnostic"), "parent_bootstrap": bootstrap, "knownness_quality_status": {"knownness": config.model.knownness_score_status, "quality": config.model.quality_head_status, "calibration": config.l3.calibration_status}, "evaluation_scope": {"unknown": "sealed", "test": "sealed", "open_set_actions": "not_executed_unfitted", "deployment": "not_executed"}, "records": {"parent": parent_records, "eligible_prefix": eligible, "window": event_records}, "record_sha256": canonical_sha256({"parent": parent_records, "eligible_prefix": eligible, "window": event_records})}


def assess_v27_checkpoint(evaluation: Mapping[str, Any], *, epoch: int) -> dict[str, Any]:
    task = str(evaluation.get("task_name"))
    classes = task_class_names(task)
    metrics = dict(evaluation.get("parent_metrics") or {})
    rows = {str(row["class"]): dict(row) for row in metrics.get("per_class") or ()}
    if set(rows) != set(classes) or int(epoch) <= 0:
        raise ValueError("V27 checkpoint assessment class/task differs")
    values = [float(row[field]) for row in rows.values() for field in ("precision", "recall", "f1")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("V27 checkpoint assessment is non-finite")
    minimum = min(values)
    macro_f1 = float(metrics["macro_f1"])
    if task == "dbb_specialist":
        matrix = np.asarray(metrics["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
        db_errors = int(matrix[0, 1] + matrix[1, 0])
        key = [minimum, -float(db_errors), macro_f1, -float(epoch)]
    else:
        db_errors = None
        key = [minimum, macro_f1, -float(epoch)]
    return {"schema_version": "RADAR-LN3-V27-task-checkpoint-selector-v2", "task": task, "epoch": int(epoch), "feasible": True, "minimum_per_class_precision_recall_f1": minimum, "bidirectional_drone_bird_errors": db_errors, "macro_f1": macro_f1, "ranking_key": key}


def is_better_v27_checkpoint(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if candidate.get("feasible") is not True:
        return False
    return incumbent is None or tuple(candidate["ranking_key"]) > tuple(incumbent["ranking_key"])


__all__ = ["assess_v27_checkpoint", "evaluate_v27", "is_better_v27_checkpoint"]
