"""Aggregate frozen V21e KnownGate treatment-only or paired OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from radar_ln3.known_gate import summarize_known_gate_records
from radar_ln3.known_gate_oof import FOLD_COUNT, PREDICTION_SCHEMA, fold_record, load_fold_contract, prediction_relative, sequence_sha256, sha256_file




def _load_arm(root: Path, arm: str) -> dict[str, Any]:
    contract, assignment = load_fold_contract(root)
    predictions: list[dict[str, Any]] = []
    records_by_scope = {scope: [] for scope in ("parent", "window", "eligible_prefix")}
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        path = root / prediction_relative(arm, fold_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluation = dict(payload.get("evaluation") or {})
        records = dict(payload.get("records") or {})
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != PREDICTION_SCHEMA
            or payload.get("arm") != arm
            or int(payload.get("fold_id", -1)) != fold_id
            or payload.get("grouping") != "capture_content_provenance_component"
            or payload.get("prediction_protocol")
            != "out_of_fold_fixed_final_epoch"
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
            or int(payload.get("held_out_prediction_pass_count", -1)) != 1
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or set(records) != set(records_by_scope)
        ):
            raise ValueError(f"V21e OOF {arm} fold {fold_id} prediction is invalid")
        parent_ids = sorted(str(row["parent_sample_id"]) for row in records["parent"])
        if parent_ids != fold["held_out_parent_ids"]:
            raise ValueError(f"V21e OOF {arm} fold {fold_id} held-out set changed")
        history = list(payload.get("fold_training_history") or [])
        if len(history) != 20 or [int(row["epoch"]) for row in history] != list(range(1, 21)):
            raise ValueError(f"V21e OOF {arm} fold {fold_id} lacks 20 epochs")
        for scope in records_by_scope:
            records_by_scope[scope].extend(dict(row) for row in records[scope])
        predictions.append(payload)
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
                "train_parent_count": int(payload["train_scope"]["parent_count"]),
                "held_out_parent_count": len(parent_ids),
                "held_out_label_parent_counts": fold["held_out_label_parent_counts"],
                "parent_metrics": evaluation["parent_metrics"],
                "eligible_prefix_metrics": evaluation["eligible_prefix_metrics"],
                "initial_model_state_sha256": payload["initial_model_state_sha256"],
                "paired_baseline_reference_state_sha256": payload.get(
                    "paired_baseline_reference_state_sha256"
                ),
                "epoch_parent_order_hashes": [
                    str(row["parent_order_sha256"]) for row in history
                ],
            }
        )
    parent_ids = sorted(str(row["parent_sample_id"]) for row in records_by_scope["parent"])
    assignment_ids = sorted(
        str(row["parent_sample_id"]) for row in assignment["assignments"]
    )
    if (
        len(parent_ids) != 986
        or len(set(parent_ids)) != 986
        or parent_ids != assignment_ids
        or sequence_sha256(parent_ids) != assignment["known_parent_ids_sha256"]
    ):
        raise ValueError(f"V21e OOF {arm} does not cover every known train parent once")
    metrics_by_scope = {
        scope: summarize_known_gate_records(rows, scope=f"oof_{arm}_{scope}")
        for scope, rows in records_by_scope.items()
    }
    return {
        "arm": arm,
        "predictions": predictions,
        "fold_summaries": fold_summaries,
        "records_by_scope": records_by_scope,
        "metrics_by_scope": metrics_by_scope,
    }

























