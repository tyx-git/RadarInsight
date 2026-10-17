"""Independent frozen-endpoint L1 to L2 to L3 test replay."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from .data import V28ReplayDataset, build_dataloader, model_input_kwargs
from .heads import V28TrajectoryHeads, deterministic_quality_target, route_knownness_score, trajectory_consistency_score
from .l3 import resolve_action_verified
from .model import V28TaskModel
from .protocol import V28Context, V28_L1_TASK, V28_L2_TASK, V28_L2_CLASSES, artifact_path, relative_project_path, resolve_project_path, write_sealed_json
from .checkpoint import load_checkpoint
from .registry import load_registry
from radar_ln3.runtime import sha256_file


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits / float(temperature)
    shifted -= shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _candidate_set(probability: np.ndarray, class_names: list[str], floors: Mapping[str, Any]) -> list[str]:
    return [name for index, name in enumerate(class_names) if float(probability[index]) + 1.0e-12 >= float(floors.get(name, 0.0))]


def _wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"estimate": None, "lower_95": None, "upper_95": None, "support": 0}
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return {"estimate": float(estimate), "lower_95": float(max(0.0, center - radius)), "upper_95": float(min(1.0, center + radius)), "support": int(total)}


def _action_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    actions = ("DroneTarget", "BirdTarget", "BalloonTarget", "Clutter", "UNKNOWN_REJECTED", "PENDING_ABSTAIN")
    true_names = {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}
    rows: list[dict[str, Any]] = []
    matrix = np.zeros((len(true_names), len(actions)), dtype=np.int64)
    true_order = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget")
    for record in records:
        true_name = str(record["label_name"])
        action = str(record["action"])
        if true_name not in true_names or action not in actions:
            raise ValueError("V28 replay action label escaped contract")
        matrix[true_order.index(true_name), actions.index(action)] += 1
    action_true_index = {
        "DroneTarget": 0,
        "BirdTarget": 1,
        "BalloonTarget": 2,
        "Clutter": 3,
        "UNKNOWN_REJECTED": 4,
    }
    for index, action in enumerate(actions):
        predicted = int(matrix[:, index].sum())
        true_index = action_true_index.get(action)
        true_support = int(matrix[true_index].sum()) if true_index is not None else None
        tp = int(matrix[true_index, index]) if true_index is not None else None
        precision = (tp / predicted) if tp is not None and predicted else None
        recall = (tp / true_support) if tp is not None and true_support else None
        f1 = (2.0 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
        rows.append({"action": action, "precision": None if precision is None else float(precision), "recall": None if recall is None else float(recall), "f1": None if f1 is None else float(f1), "predicted_support": predicted, "true_support": true_support, "true_positive": tp, "metric_defined": true_index is not None})
    total = int(matrix.sum())
    accepted = [record for record in records if record["action"] != "PENDING_ABSTAIN"]
    accepted_correct = sum((record["action"] == record["label_name"] or (record["label_name"] == "ClutterTarget" and record["action"] == "Clutter") or (record["label_name"] == "UnknownTarget" and record["action"] == "UNKNOWN_REJECTED")) for record in accepted)
    unknown = [record for record in records if record["label_name"] == "UnknownTarget"]
    clutter = [record for record in records if record["label_name"] == "ClutterTarget"]
    known = [record for record in records if record["label_name"] in V28_L2_CLASSES]
    known_by_class = {
        label: {
            "accepted": sum(record["action"] == label for record in known if record["label_name"] == label),
            "pending": sum(record["action"] == "PENDING_ABSTAIN" for record in known if record["label_name"] == label),
            "wrong_action": sum(record["action"] not in {label, "PENDING_ABSTAIN"} for record in known if record["label_name"] == label),
            "support": sum(record["label_name"] == label for record in known),
        }
        for label in V28_L2_CLASSES
    }
    for values in known_by_class.values():
        values["acceptance_rate"] = _wilson(values["accepted"], values["support"])
        values["false_rejection_rate"] = _wilson(values["support"] - values["accepted"], values["support"])
    return {"actions": rows, "true_labels": list(true_order), "action_labels": list(actions), "confusion_matrix_rows_true_columns_action": matrix.tolist(), "support": total, "pending_rate": _wilson(sum(record["action"] == "PENDING_ABSTAIN" for record in records), total), "coverage": _wilson(len(accepted), total), "accepted_accuracy": _wilson(accepted_correct, len(accepted)), "selective_risk": _wilson(len(accepted) - accepted_correct, len(accepted)), "unknown_rejection": {"rejected": sum(record["action"] == "UNKNOWN_REJECTED" for record in unknown), "error_acceptance": sum(record["action"] in V28_L2_CLASSES or record["action"] == "Clutter" for record in unknown), "pending": sum(record["action"] == "PENDING_ABSTAIN" for record in unknown), "rate": _wilson(sum(record["action"] == "UNKNOWN_REJECTED" for record in unknown), len(unknown))}, "clutter_acceptance": {"accepted": sum(record["action"] == "Clutter" for record in clutter), "misrouted": sum(record["action"] != "Clutter" for record in clutter), "rate": _wilson(sum(record["action"] == "Clutter" for record in clutter), len(clutter))}, "known_target_acceptance": {"accepted": sum(record["action"] == record["label_name"] for record in known), "pending": sum(record["action"] == "PENDING_ABSTAIN" for record in known), "wrong_action": sum(record["action"] not in {*V28_L2_CLASSES, "PENDING_ABSTAIN"} for record in known), "rate": _wilson(sum(record["action"] == record["label_name"] for record in known), len(known)), "by_class": known_by_class}}


def run_replay(context: V28Context, *, device: torch.device, audit_mode: bool = False) -> dict[str, Any]:
    registry = load_registry(context)
    calibration = registry["calibration"]
    l1_binding = registry["endpoint_binding"][V28_L1_TASK]
    l2_binding = registry["endpoint_binding"][V28_L2_TASK]
    l1_head_binding = registry["head_bindings"][V28_L1_TASK]
    l2_head_binding = registry["head_bindings"][V28_L2_TASK]
    l1_model = V28TaskModel(context, task=V28_L1_TASK).to(device)
    l2_model = V28TaskModel(context, task=V28_L2_TASK).to(device)
    l1_payload = load_checkpoint(context, l1_model, l1_binding["path"], task=V28_L1_TASK, seed=int(l1_binding["seed"]), map_location=device)
    l2_payload = load_checkpoint(context, l2_model, l2_binding["path"], task=V28_L2_TASK, seed=int(l2_binding["seed"]), map_location=device)
    l1_head = V28TrajectoryHeads(context, task=V28_L1_TASK).to(device)
    l2_head = V28TrajectoryHeads(context, task=V28_L2_TASK).to(device)
    l1_head_payload = torch.load(resolve_project_path(context.project_root, l1_head_binding["path"], must_exist=True), map_location=device, weights_only=False)
    l2_head_payload = torch.load(resolve_project_path(context.project_root, l2_head_binding["path"], must_exist=True), map_location=device, weights_only=False)
    l1_head.load_state_dict(l1_head_payload["state_dict"], strict=True)
    l2_head.load_state_dict(l2_head_payload["state_dict"], strict=True)
    l1_dataset = V28ReplayDataset(context, task=V28_L1_TASK)
    l2_dataset = V28ReplayDataset(context, task=V28_L2_TASK)
    l1_loader = build_dataloader(l1_dataset, batch_size=int(context.config.training["batch_size"]), shuffle=False, seed=int(context.config.training["dataloader_seed"]))
    l2_loader = build_dataloader(l2_dataset, batch_size=int(context.config.training["batch_size"]), shuffle=False, seed=int(context.config.training["dataloader_seed"]))
    l1_task = calibration["tasks"][V28_L1_TASK]
    l2_task = calibration["tasks"][V28_L2_TASK]
    l1_names = l1_task["class_names"]
    records: list[dict[str, Any]] = []
    l1_model.eval(); l2_model.eval(); l1_head.eval(); l2_head.eval()
    with torch.inference_mode():
        for l1_batch, l2_batch in zip(l1_loader, l2_loader, strict=True):
            if list(l1_batch["parent_sample_ids"]) != list(l2_batch["parent_sample_ids"]):
                raise ValueError("V28 replay L1/L2 parent order differs")
            l1_output = l1_model(**model_input_kwargs(l1_batch, device))
            l2_output = l2_model(**model_input_kwargs(l2_batch, device))
            l1_aux = l1_head(l1_output)
            l2_aux = l2_head(l2_output)
            l1_lengths = l1_output["lengths"].long()
            l2_lengths = l2_output["lengths"].long()
            rows = torch.arange(l1_lengths.shape[0], device=device)
            l1_quality = deterministic_quality_target(context, task=V28_L1_TASK, task_facts=l1_output["task_facts_sequence"], event_facts=l1_output["event_facts_sequence"], padding_mask=l1_output["padding_mask"])[rows, l1_lengths - 1]
            l2_quality = deterministic_quality_target(context, task=V28_L2_TASK, task_facts=l2_output["task_facts_sequence"], event_facts=l2_output["event_facts_sequence"], padding_mask=l2_output["padding_mask"])[rows, l2_lengths - 1]
            l1_probs = _probabilities(l1_output["final_logits"].cpu().numpy(), float(l1_task["temperature"].get("temperature", 1.0)))
            l2_probs = _probabilities(l2_output["final_logits"].cpu().numpy(), float(l2_task["temperature"].get("temperature", 1.0)))
            l1_target_scores = torch.sigmoid(l1_aux["target_knownness_logits"][rows, l1_lengths - 1]).cpu().numpy()
            l1_clutter_scores = torch.sigmoid(l1_aux["clutter_knownness_logits"][rows, l1_lengths - 1]).cpu().numpy()
            l2_scores = torch.sigmoid(l2_aux["target_knownness_logits"][rows, l2_lengths - 1]).cpu().numpy()
            l2_consistency = trajectory_consistency_score(l2_output)[rows, l2_lengths - 1].cpu().numpy()
            for index, parent_id in enumerate(l1_batch["parent_sample_ids"]):
                l1_floors = dict(l1_task["conformal"].get("candidate_probability_floor_by_class", {}))
                l1_floors["KNOWN_TARGET_EVIDENCE"] = float(context.config.selection["l1_target_candidate_probability_floor"])
                l1_floors["KNOWN_CLUTTER_EVIDENCE"] = float(context.config.selection["l1_clutter_candidate_probability_floor"])
                l1_candidates = _candidate_set(l1_probs[index], l1_names, l1_floors)
                l1_prediction = l1_names[int(l1_probs[index].argmax())] if l1_probs.shape[1] else None
                l1_target = l1_prediction == "KNOWN_TARGET_EVIDENCE"
                l1_clutter = l1_prediction == "KNOWN_CLUTTER_EVIDENCE"
                l2_candidates = _candidate_set(l2_probs[index], list(V28_L2_CLASSES), l2_task["conformal"].get("candidate_probability_floor_by_class", {}))
                l2_prediction = V28_L2_CLASSES[int(l2_probs[index].argmax())] if l2_probs.shape[1] else None
                l1_knownness = bool(l1_target_scores[index] >= float(l1_task["knownness"]["target"]["selected"]["threshold"])) if l1_target else bool(l1_clutter_scores[index] >= float(l1_task["knownness"]["clutter"]["selected"]["threshold"]))
                route_threshold = float(l2_task["knownness"].get(l2_prediction, {}).get("selected", {}).get("threshold", 1.0))
                if l2_prediction == "BirdTarget":
                    l2_route_score = float(route_knownness_score(
                        l2_scores[index],
                        l2_consistency[index],
                        route="BirdTarget",
                        consistency_weight=float(context.config.heads["bird_consistency_weight"]),
                    ))
                    l2_knownness = bool(l2_route_score >= route_threshold)
                else:
                    l2_route_score = float(l2_scores[index])
                    l2_knownness = bool(l2_scores[index] >= route_threshold)
                if not l1_candidates and l1_target and l1_probs[index].max() >= float(context.config.selection["l1_target_candidate_probability_floor"]):
                    l1_candidates = [l1_prediction]
                if not l2_candidates and context.config.selection["l2_candidate_empty_fallback"] and l2_knownness:
                    l2_candidates = [l2_prediction]
                decision = resolve_action_verified(registry=registry, l1_candidate_set=l1_candidates, l1_prediction=l1_prediction, l1_knownness=l1_knownness, l1_quality_eligible=bool(l1_quality[index]), l1_target=l1_target, l1_clutter=l1_clutter, l2_candidate_set=l2_candidates, l2_prediction=l2_prediction, l2_knownness=l2_knownness, l2_quality_eligible=bool(l2_quality[index]), streaming=False, audit_mode=audit_mode)
                records.append({"parent_sample_id": str(parent_id), "label_name": str(l1_batch["base_label_names"][index]), "action": decision.action, "reason_code": decision.reason_code, "l1_prediction": l1_prediction, "l2_prediction": l2_prediction, "l1_candidates": l1_candidates, "l2_candidates": l2_candidates, "l1_knownness": float(l1_target_scores[index] if l1_target else l1_clutter_scores[index]), "l2_knownness": float(l2_scores[index]), "l2_route_knownness": l2_route_score, "l2_trajectory_consistency": float(l2_consistency[index]), "l1_quality_eligible": bool(l1_quality[index]), "l2_quality_eligible": bool(l2_quality[index]), "event_count": int(l1_lengths[index].item())})
    audit_pool = calibration.get("post_selection_audit", {})
    report = {"schema_version": "RADAR-LN3-V28-l3-replay-v2", "experiment_id": "V28", "mode": "audit" if audit_mode else "replay", "status": "completed", "registry_status": registry["status"], "deployment_eligible": bool(registry.get("deployment_eligible")), "system_manifest_sha256": context.manifest_sha256, "config_sha256": context.config.sha256, "registry_sha256": registry["registry_sha256"], "registry_file_sha256": sha256_file(artifact_path(context.project_root, str(context.config.artifacts["registry_path"]))), "checkpoint_epochs": {V28_L1_TASK: int(l1_payload["epoch"]), V28_L2_TASK: int(l2_payload["epoch"])}, "parent_count": len(records), "evaluation_scope": {"split": context.config.data["test_split_name"], "test_payload_access_count": 0 if audit_mode else len(records), "audit_payload_access_count": len(records) if audit_mode else 0, "post_selection_audit": "completed" if audit_mode else "not_executed", "audit_pool_parent_count": int(audit_pool.get("parent_count", 0)), "audit_pool_parent_ids_sha256": audit_pool.get("parent_ids_sha256"), "thresholds_frozen_before_audit": bool(audit_mode), "classification_training_unknown_exposure": 0}, "action_metrics": _action_metrics(records), "records": records}
    path = artifact_path(context.project_root, str(context.config.artifacts["audit_report_path"] if audit_mode else context.config.artifacts["replay_report_path"]))
    write_sealed_json(path, report)
    return {"path": relative_project_path(context.project_root, path), "sha256": sha256_file(path), "report": report}


__all__ = ["run_replay"]
