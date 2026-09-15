"""Paired known-class evaluation for V27d-r14 anchored L2 models."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.runtime import relative_project_path
from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records

from .data import R14L2Dataset, build_r14_dataloader, r14_model_input_kwargs, r14_sidecar_kwargs
from .protocol import R14_ARMS, R14_CLASS_NAMES, R14_REPORT_SCHEMA, R14Context, artifact_path, read_json, read_sealed_json, write_sealed_json_if_identical
from .training import load_checkpoint


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _records(batch: Mapping[str, Any], logits: np.ndarray, *, arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    padding = batch["tr_padding_mask"].cpu().numpy()
    lengths = batch["lengths"].cpu().numpy().astype(np.int64)
    labels = batch["labels"].cpu().numpy().astype(np.int64)
    probabilities = _softmax(logits)
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row, parent_id in enumerate(batch["parent_sample_ids"]):
        length = int(lengths[row])
        if length <= 0 or not np.array_equal(padding[row], np.arange(logits.shape[1]) >= length):
            raise ValueError("V27d-r14 evaluation padding differs from lengths")
        final = length - 1
        true_id = int(labels[row])
        parents.append({"parent_sample_id": str(parent_id), "track_id": str(batch["track_ids"][row]), "source_name": str(batch["source_names"][row]), "true_id": true_id, "pred_id": int(np.argmax(probabilities[row, final])), "probabilities": [float(value) for value in probabilities[row, final]], "logits": [float(value) for value in logits[row, final]], "event_count": length, "arm": arm})
        for event in range(length):
            events.append({"parent_sample_id": str(parent_id), "track_id": str(batch["track_ids"][row]), "source_name": str(batch["source_names"][row]), "true_id": true_id, "pred_id": int(np.argmax(probabilities[row, event])), "probabilities": [float(value) for value in probabilities[row, event]], "logits": [float(value) for value in logits[row, event]], "event_index": event, "window_length": event + 1, "parent_length": length, "arm": arm})
    return parents, events


def evaluate_model(context: R14Context, *, model: torch.nn.Module, dataset: R14L2Dataset, device: torch.device, checkpoint_epoch: int, reference_logits: Mapping[str, torch.Tensor] | None = None) -> dict[str, Any]:
    model.eval()
    parent_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r14_dataloader(dataset, batch_size=int(context.config.training["batch_size"]), shuffle=False):
            kwargs = r14_model_input_kwargs(batch, device)
            reference = None
            if reference_logits is not None:
                reference = torch.stack([reference_logits[str(parent_id)] for parent_id in batch["parent_sample_ids"]], dim=0)[:, :kwargs["tr_padding_mask"].shape[1]].to(device=device, non_blocking=True)
            output = model(base_inputs=kwargs, reference_sequence_logits=reference, **r14_sidecar_kwargs(batch, device))
            logits = output["sequence_logits"].detach().cpu().numpy()
            if not np.isfinite(logits).all():
                raise FloatingPointError("V27d-r14 evaluation produced non-finite logits")
            parents, events = _records(batch, logits, arm=model.arm)
            parent_rows.extend(parents)
            event_rows.extend(events)
    eligible = [row for row in event_rows if int(row["window_length"]) >= 3 and int(row["window_length"]) < int(row["parent_length"])]
    balanced = parent_balanced_prefix_records(eligible, minimum_events=3, class_names=R14_CLASS_NAMES)
    return {"schema_version": R14_REPORT_SCHEMA, "experiment_id": "V27d-r14", "arm": model.arm, "checkpoint_epoch": int(checkpoint_epoch), "class_names": list(R14_CLASS_NAMES), "parent_metrics": summarize_records(parent_rows, class_names=R14_CLASS_NAMES, scope="parent_final"), "parent_balanced_eligible_prefix_metrics": summarize_records(balanced, class_names=R14_CLASS_NAMES, scope="parent_balanced_eligible_prefix_diagnostic"), "records": {"parent": parent_rows, "eligible_prefix": eligible, "window": event_rows}}


def _reference_metrics(context: R14Context) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = context.config.source["paths"]
    first = read_json(context.project_root / paths["comparator_report_seed42"])
    second = read_json(context.project_root / paths["comparator_report_seed43"])
    first_metrics = dict(first["arms"]["frozen_v27a_main_reference"]["parent_metrics"])
    second_metrics = dict(second["arms"]["frozen_v27a_main_reference"]["parent_metrics"])
    if first_metrics != second_metrics:
        raise ValueError("V27d-r14 V27a Main comparator metrics differ between reference reports")
    return first_metrics, {"experiment_id": "V27a", "variant": "V27a-r3-main-epoch-020", "source_report_seed42": paths["comparator_report_seed42"], "source_report_seed43": paths["comparator_report_seed43"], "support": int(first_metrics["support"])}


def _class_rows(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["class"]): dict(row) for row in metrics.get("per_class", ())}
    if set(rows) != set(R14_CLASS_NAMES):
        raise ValueError("V27d-r14 metric classes differ")
    return rows


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"):
        value = float(left[key]) - float(right[key])
        result[key] = value
        result[f"{key}_pp"] = 100.0 * value
    left_rows, right_rows = _class_rows(left), _class_rows(right)
    result["per_class"] = {name: {metric: {"delta": float(left_rows[name][metric]) - float(right_rows[name][metric]), "delta_pp": 100.0 * (float(left_rows[name][metric]) - float(right_rows[name][metric]))} for metric in ("precision", "recall", "f1")} for name in R14_CLASS_NAMES}
    return result


def _paired_transitions(control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right):
        raise ValueError("V27d-r14 paired parent sets differ")
    transitions: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if int(a["true_id"]) != int(b["true_id"]):
            raise ValueError("V27d-r14 paired labels differ")
        a_correct = int(a["pred_id"]) == int(a["true_id"])
        b_correct = int(b["pred_id"]) == int(b["true_id"])
        transitions[f"{'correct' if a_correct else 'wrong'}_to_{'correct' if b_correct else 'wrong'}"] += 1
        if int(a["pred_id"]) != int(b["pred_id"]):
            changed.append({"parent_sample_id": parent_id, "true_id": int(a["true_id"]), "control_pred_id": int(a["pred_id"]), "treatment_pred_id": int(b["pred_id"]), "control_correct": a_correct, "treatment_correct": b_correct})
    return {"parent_count": len(left), "control_correct_count": sum(int(int(row["pred_id"]) == int(row["true_id"])) for row in left.values()), "treatment_correct_count": sum(int(int(row["pred_id"]) == int(row["true_id"])) for row in right.values()), "correctness_transitions": dict(sorted(transitions.items())), "changed_predictions": changed}


def _confusion_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[list[int]]:
    a = np.asarray(left["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
    b = np.asarray(right["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("V27d-r14 confusion matrix shape differs")
    return (a - b).tolist()


def evaluate_arm(context: R14Context, *, arm: str, seed: int) -> dict[str, Any]:
    if arm not in R14_ARMS or int(seed) not in (42, 44):
        raise ValueError("V27d-r14 evaluation arm or seed is invalid")
    device = torch.device("cuda:0") if context.config.training["device"] == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    epoch = int(context.config.training["primary_epoch"])
    model, checkpoint = load_checkpoint(context, arm=arm, seed=seed, epoch=epoch, device=device)
    validation_dataset = R14L2Dataset(context, split="val")
    with torch.inference_mode():
        reference_logits: dict[str, torch.Tensor] = {}
        model.frozen_reference_base_model.eval()
        for batch in build_r14_dataloader(validation_dataset, batch_size=int(context.config.training["batch_size"]), shuffle=False):
            values_raw = model.frozen_reference_base_model(**r14_model_input_kwargs(batch, device), return_trace=False)["sequence_logits"].detach().cpu()
            values = torch.zeros((values_raw.shape[0], 29, values_raw.shape[-1]), dtype=values_raw.dtype)
            values[:, :values_raw.shape[1]] = values_raw
            for index, parent_id in enumerate(batch["parent_sample_ids"]):
                reference_logits[str(parent_id)] = values[index]
    evaluation = evaluate_model(context, model=model, dataset=validation_dataset, device=device, checkpoint_epoch=epoch, reference_logits=reference_logits)
    checkpoint_path = artifact_path(context.project_root, context.config.artifacts["checkpoint_template"], arm=arm, seed=seed, epoch=epoch)
    payload = {"schema_version": R14_REPORT_SCHEMA, "status": "completed_known_class_l2_evaluation_only", "experiment_id": "V27d-r14", "arm": arm, "seed": int(seed), "mode": context.config.modes[arm], "manifest_sha256": context.manifest_sha256, "checkpoint_path": relative_project_path(context.project_root, checkpoint_path), "checkpoint_model_state_sha256": checkpoint["model_state_sha256"], "parent_metrics": evaluation["parent_metrics"], "parent_balanced_eligible_prefix_metrics": evaluation["parent_balanced_eligible_prefix_metrics"], "records": evaluation["records"], "scope": {"unknown_training_exposure": 0, "unknown_evaluation_exposure": 0, "test_payload_access_count": 0, "knownness": "not_in_scope", "quality": "not_in_scope", "l3": "not_in_scope"}, "deployment_eligible": False}
    path = artifact_path(context.project_root, context.config.artifacts["evaluation_template"], arm=arm, seed=seed, epoch=epoch)
    write_sealed_json_if_identical(path, payload)
    return {**payload, "report_path": str(path)}


def build_paired_report(context: R14Context, *, seed: int) -> dict[str, Any]:
    epoch = int(context.config.training["primary_epoch"])
    paths = {arm: artifact_path(context.project_root, context.config.artifacts["evaluation_template"], arm=arm, seed=seed, epoch=epoch, must_exist=True) for arm in R14_ARMS}
    control = read_sealed_json(paths["control"], expected_schema=R14_REPORT_SCHEMA)
    treatment = read_sealed_json(paths["treatment"], expected_schema=R14_REPORT_SCHEMA)
    reference, reference_source = _reference_metrics(context)
    control_metrics, treatment_metrics = control["parent_metrics"], treatment["parent_metrics"]
    payload = {"schema_version": R14_REPORT_SCHEMA, "status": "paired_known_class_l2_validation", "experiment_id": "V27d-r14", "seed": int(seed), "manifest_sha256": context.manifest_sha256, "arms": {"control": {"mode": control["mode"], "parent_metrics": control_metrics}, "treatment": {"mode": treatment["mode"], "parent_metrics": treatment_metrics}, "best_l2_candidate": {"identity": reference_source, "parent_metrics": reference}}, "paired_control_vs_treatment": _paired_transitions(control["records"]["parent"], treatment["records"]["parent"]), "confusion_delta_treatment_minus_control": _confusion_delta(treatment_metrics, control_metrics), "deltas": {"treatment_minus_control": _metric_delta(treatment_metrics, control_metrics), "control_minus_best_l2_candidate": _metric_delta(control_metrics, reference), "treatment_minus_best_l2_candidate": _metric_delta(treatment_metrics, reference)}, "scope": {"comparison": "known_class_parent_final_and_eligible_prefix_only", "unknown_training_exposure": 0, "unknown_evaluation_exposure": 0, "test_payload_access_count": 0, "knownness": "not_in_scope", "quality": "not_in_scope", "l3": "not_in_scope", "oof": "not_executed", "cross_source": "not_executed"}, "deployment_eligible": False}
    path = artifact_path(context.project_root, context.config.artifacts["paired_template"], seed=seed)
    write_sealed_json_if_identical(path, payload)
    return {**payload, "report_path": str(path)}


def summarize_seeds(context: R14Context) -> dict[str, Any]:
    reports = {str(seed): read_sealed_json(artifact_path(context.project_root, context.config.artifacts["paired_template"], seed=seed, must_exist=True), expected_schema=R14_REPORT_SCHEMA) for seed in (42, 44)}
    def aggregate(arm: str, metric: str) -> dict[str, Any]:
        values = [float(reports[str(seed)]["arms"][arm]["parent_metrics"][metric]) for seed in (42, 44)]
        return {"values": values, "mean": float(np.mean(values)), "median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
    result = {"schema_version": "RADAR-LN3-V27d-r14-density-weighted-summary-v1", "status": "two_seed_known_class_summary", "experiment_id": "V27d-r14", "manifest_sha256": context.manifest_sha256, "seeds": [42, 44], "per_seed": {}, "aggregate": {"control_accuracy": aggregate("control", "accuracy"), "treatment_accuracy": aggregate("treatment", "accuracy"), "treatment_minus_control_accuracy_pp": {"values": [float(reports[str(seed)]["deltas"]["treatment_minus_control"]["accuracy_pp"]) for seed in (42, 44)]}, "treatment_minus_control_macro_f1_pp": {"values": [float(reports[str(seed)]["deltas"]["treatment_minus_control"]["macro_f1_pp"]) for seed in (42, 44)]}}, "scope": {"unknown_training_exposure": 0, "test_payload_access_count": 0, "knownness": "not_in_scope", "quality": "not_in_scope", "l3": "not_in_scope", "oof": "not_executed", "cross_source": "not_executed"}, "deployment_eligible": False}
    for seed in (42, 44):
        report = reports[str(seed)]
        result["per_seed"][str(seed)] = {"control": report["arms"]["control"]["parent_metrics"], "treatment": report["arms"]["treatment"]["parent_metrics"], "treatment_minus_control": report["deltas"]["treatment_minus_control"], "treatment_minus_best_l2_candidate": report["deltas"]["treatment_minus_best_l2_candidate"], "correctness_transitions": report["paired_control_vs_treatment"]["correctness_transitions"]}
    path = artifact_path(context.project_root, context.config.artifacts["summary_path"])
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path)}


__all__ = ["build_paired_report", "evaluate_arm", "evaluate_model", "summarize_seeds"]
