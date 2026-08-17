"""Audit whether V21g modal consistency is worth a V21e nested OOF run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from radar_ln3.known_gate_features import (
    FEATURE_INDICES_BY_VARIANT,
    MODAL_FEATURE_NAMES,
    V21B_VARIANT,
)
from radar_ln3.known_gate_v21e_oof import write_new_or_identical, sha256_file


INDEX_RELATIVE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
CORE_CACHE_RELATIVE = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-train-feature-cache.npz"
)
V21B_REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21b-known-gate-modal-consistency-clutter-drone-"
    "paired-validation.json"
)
V21E_ARTIFACT_ROOT_RELATIVE = Path(
    "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof"
)
OUTPUT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21g-modal-consistency-cancellation-audit.json"
)
KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
SCHEMA = "RADAR-LN3-V21g-modal-consistency-cancellation-audit-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("V21g AUC groups must be non-empty")
    return float(
        (
            (pos[:, None] > neg[None, :]).sum()
            + 0.5 * (pos[:, None] == neg[None, :]).sum()
        )
        / (len(pos) * len(neg))
    )


def _direction_free_auc(
    positive: Sequence[float], negative: Sequence[float]
) -> float:
    value = _auc(positive, negative)
    return float(max(value, 1.0 - value))


def _summary(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "support": int(len(data)),
        "mean": float(np.mean(data)) if len(data) else None,
        "median": float(np.median(data)) if len(data) else None,
        "q25": float(np.quantile(data, 0.25)) if len(data) else None,
        "q75": float(np.quantile(data, 0.75)) if len(data) else None,
    }


def _load_v21e_parent_predictions(root: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for fold_id in range(5):
        path = (
            root
            / V21E_ARTIFACT_ROOT_RELATIVE
            / f"fold-{fold_id:02d}"
            / "treatment"
            / "held-out-predictions.json"
        )
        payload = _load_json(path)
        if (
            payload.get("status") != "passed"
            or payload.get("prediction_protocol")
            != "out_of_fold_fixed_final_epoch"
            or payload.get("validation_access_count") != 0
            or payload.get("test_payload_access_count") != 0
            or payload.get("unknown_training_exposure") != 0
        ):
            raise ValueError(f"V21e OOF prediction is outside the sealed scope: {path}")
        for row in payload.get("records", {}).get("parent", []):
            parent_id = str(row["parent_sample_id"])
            if parent_id in predictions:
                raise ValueError(f"V21e OOF parent repeats: {parent_id}")
            predictions[parent_id] = dict(row)
    if len(predictions) != 986:
        raise ValueError(f"V21e OOF parent support changed: {len(predictions)}")
    return predictions


def _load_parent_feature_rows(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / INDEX_RELATIVE, low_memory=False)
    selected = frame[
        frame["split_name"].astype(str).eq("train")
        & frame["label_name"].astype(str).isin(KNOWN_LABELS)
    ].sort_values(["parent_sample_id", "causal_event_index"], kind="stable")
    with np.load(root / CORE_CACHE_RELATIVE, allow_pickle=False) as payload:
        sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if [str(value) for value in selected["sample_id"]] != sample_ids:
        raise ValueError("V21g core cache no longer aligns with the train index")
    indices = FEATURE_INDICES_BY_VARIANT[V21B_VARIANT]
    if len(indices) != len(MODAL_FEATURE_NAMES):
        raise ValueError("V21g modal feature width changed")
    if values.dtype != np.float32 or valid.dtype != np.bool_:
        raise ValueError("V21g core cache dtype changed")
    rows: list[dict[str, Any]] = []
    for parent_id, group in selected.groupby("parent_sample_id", sort=True):
        labels = group["label_name"].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"V21g parent label is not unique: {parent_id}")
        row_indices = group.index.to_numpy(dtype=np.int64)
        # selected was reset by positional order only; use the explicit row order
        # mapping instead of assuming the pandas index is contiguous.
        positions = selected.index.get_indexer(group.index)
        row: dict[str, Any] = {
            "parent_sample_id": str(parent_id),
            "label_name": labels[0],
        }
        for feature_index, feature_name in enumerate(MODAL_FEATURE_NAMES):
            column = indices[feature_index]
            active = valid[positions, column]
            if not active.any():
                raise ValueError(f"V21g parent has no valid modal event: {parent_id}")
            data = values[positions[active], column].astype(np.float64)
            row[f"{feature_name}__mean"] = float(np.mean(data))
            row[f"{feature_name}__final"] = float(data[-1])
            row[f"{feature_name}__max"] = float(np.max(data))
        rows.append(row)
    return pd.DataFrame(rows)


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prior_report = _load_json(root / V21B_REPORT_RELATIVE)
    if (
        prior_report.get("status") != "completed"
        or prior_report.get("frozen_decision", {}).get("positive_signal") is not False
        or prior_report.get("base_label_paired_outcomes", {})
        .get("DroneTarget", {})
        .get("changed")
        != 0
    ):
        raise ValueError("V21b formal result is not the frozen negative reference")
    predictions = _load_v21e_parent_predictions(root)
    parents = _load_parent_feature_rows(root)
    parents["pred_id"] = parents["parent_sample_id"].map(
        lambda value: int(predictions[str(value)]["pred_id"])
    )
    parents["true_id"] = parents["parent_sample_id"].map(
        lambda value: int(predictions[str(value)]["true_id"])
    )
    if len(parents) != 986:
        raise ValueError("V21g train parent support changed")

    feature_results: dict[str, Any] = {}
    for feature_name in MODAL_FEATURE_NAMES:
        feature_results[feature_name] = {}
        for reducer in ("mean", "final", "max"):
            column = f"{feature_name}__{reducer}"
            selected = lambda mask: parents.loc[mask, column].to_numpy(dtype=float)
            feature_results[feature_name][reducer] = {
                "ClutterTarget_vs_DroneTarget": _direction_free_auc(
                    selected(parents["label_name"].eq("ClutterTarget")),
                    selected(parents["label_name"].eq("DroneTarget")),
                ),
                "ClutterTarget_vs_BirdTarget": _direction_free_auc(
                    selected(parents["label_name"].eq("ClutterTarget")),
                    selected(parents["label_name"].eq("BirdTarget")),
                ),
                "ClutterTarget_vs_BalloonTarget": _direction_free_auc(
                    selected(parents["label_name"].eq("ClutterTarget")),
                    selected(parents["label_name"].eq("BalloonTarget")),
                ),
                "V21e_hard_Bird_vs_correct_Bird": _direction_free_auc(
                    selected(
                        parents["label_name"].eq("BirdTarget")
                        & parents["pred_id"].ne(parents["true_id"])
                    ),
                    selected(
                        parents["label_name"].eq("BirdTarget")
                        & parents["pred_id"].eq(parents["true_id"])
                    ),
                ),
                "V21e_Clutter_error_vs_correct_Clutter": _direction_free_auc(
                    selected(
                        parents["label_name"].eq("ClutterTarget")
                        & parents["pred_id"].eq(0)
                    ),
                    selected(
                        parents["label_name"].eq("ClutterTarget")
                        & parents["pred_id"].eq(1)
                    ),
                ),
            }

    clutter_error_mask = parents["label_name"].eq("ClutterTarget") & parents[
        "pred_id"
    ].eq(0)
    target_error_mask = parents["label_name"].ne("ClutterTarget") & parents[
        "pred_id"
    ].eq(1)
    hard_bird_mask = parents["label_name"].eq("BirdTarget") & parents[
        "pred_id"
    ].ne(parents["true_id"])
    balloon_error_count = int(
        (
            parents["label_name"].eq("BalloonTarget")
            & parents["pred_id"].ne(parents["true_id"])
        ).sum()
    )
    max_drone = max(
        feature_results[name][reducer]["ClutterTarget_vs_DroneTarget"]
        for name in MODAL_FEATURE_NAMES
        for reducer in ("mean", "final", "max")
    )
    max_hard_bird = max(
        feature_results[name][reducer]["V21e_hard_Bird_vs_correct_Bird"]
        for name in MODAL_FEATURE_NAMES
        for reducer in ("mean", "final", "max")
    )
    conditional_error_values = {
        name: {
            reducer: {
                "Clutter_error_support": int(clutter_error_mask.sum()),
                "target_to_clutter_error_support": int(target_error_mask.sum()),
                "hard_Bird_support": int(hard_bird_mask.sum()),
                "Clutter_error_mean": float(
                    parents.loc[clutter_error_mask, f"{name}__{reducer}"].mean()
                ),
                "correct_Clutter_mean": float(
                    parents.loc[
                        parents["label_name"].eq("ClutterTarget")
                        & parents["pred_id"].eq(1),
                        f"{name}__{reducer}",
                    ].mean()
                ),
            }
            for reducer in ("mean", "final", "max")
        }
        for name in MODAL_FEATURE_NAMES
    }
    reasons = [
        "Clutter/Drone direction-free train-parent AUC remains below 0.60 for every modal reducer",
        "V21e hard-Bird versus correct-Bird separation remains below 0.60 for every modal reducer",
        "V21b standalone paired training produced zero parent prediction changes and no positive signal",
        "V21e has zero Balloon parent errors, so the strong Clutter/Balloon univariate separation cannot be tested against the active error boundary",
        "conditional error supports are only 6 Clutter errors and 14 hard Bird errors; they cannot justify a new fitted model",
    ]
    report = {
        "status": "cancelled_before_training",
        "schema_version": SCHEMA,
        "experiment": "V21g modal consistency on V21e KnownGate",
        "decision": {
            "training_required": False,
            "cancelled": True,
            "next_retained_experiment": "V21h target-row iso",
            "primary_clutter_drone_auc_max": float(max_drone),
            "hard_bird_auc_max": float(max_hard_bird),
            "balloon_error_support": balloon_error_count,
            "cancellation_reasons": reasons,
        },
        "scope": {
            "source": "V21e train-parent-grouped OOF parent-final records",
            "parent_support": int(len(parents)),
            "error_support": int((parents["pred_id"] != parents["true_id"]).sum()),
            "clutter_error_support": int(clutter_error_mask.sum()),
            "target_to_clutter_error_support": int(target_error_mask.sum()),
            "hard_bird_support": int(hard_bird_mask.sum()),
            "unknown_training_exposure": 0,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
        },
        "feature_names": list(MODAL_FEATURE_NAMES),
        "direction_free_parent_auc": feature_results,
        "conditional_error_diagnostic": conditional_error_values,
        "prior_v21b_report_sha256": sha256_file(root / V21B_REPORT_RELATIVE),
        "test": "未执行",
        "OOF": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / OUTPUT_RELATIVE
    write_new_or_identical(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and cancel V21g if its signal is weak.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
