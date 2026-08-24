"""Contract smoke test for V24d's OOF-only L3 score-band calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build.known_gate_v24d import FOLD_COUNT, READINESS, SOURCE_PARENT_COUNT
from report.known_gate_v21e_oof import _load_arm
from report.known_gate_v24d import (
    CLUTTER_ACTION,
    PENDING_ACTION,
    TARGET_ACTION,
    _annotate_actions,
    _cross_fit_thresholds,
    _require_preparation,
    _score_band_actions,
    _select_thresholds,
    _validate_source_records,
)


def _synthetic_parent_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        for index, (true_id, score) in enumerate(
            ((0, 0.01), (0, 0.15), (1, 0.85), (1, 0.99))
        ):
            rows.append(
                {
                    "parent_sample_id": f"synthetic-{fold_id}-{index}",
                    "track_id": f"synthetic-{fold_id}-{index}",
                    "fold_id": fold_id,
                    "true_id": true_id,
                    "pred_id": 1 if score > 0.5 else 0,
                    "known_clutter_probability": score,
                    "known_target_probability": 1.0 - score,
                    "product_action_eligible": True,
                }
            )
    return rows


def _assert_policy_semantics() -> dict[str, bool]:
    scores = np.asarray([0.05, 0.20, 0.50, 0.80, 0.95], dtype=np.float64)
    action = _score_band_actions(
        scores,
        target_threshold=0.20,
        clutter_threshold=0.80,
        eligible=np.asarray([True, True, True, True, False]),
    )
    if action.tolist() != [TARGET_ACTION, TARGET_ACTION, PENDING_ACTION, CLUTTER_ACTION, PENDING_ACTION]:
        raise AssertionError("V24d score-band action semantics differ from the protocol")
    degenerate = _score_band_actions(
        np.asarray([0.49, 0.50, 0.51], dtype=np.float64),
        target_threshold=0.5,
        clutter_threshold=0.5,
    )
    if degenerate.tolist() != [TARGET_ACTION, TARGET_ACTION, CLUTTER_ACTION]:
        raise AssertionError("V24d degenerate score band is not deterministic")
    try:
        _score_band_actions(
            np.asarray([0.5], dtype=np.float64),
            target_threshold=0.81,
            clutter_threshold=0.80,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("V24d accepted an invalid lower threshold")
    try:
        _score_band_actions(
            np.asarray([0.4], dtype=np.float64),
            target_threshold=0.4,
            clutter_threshold=0.4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("V24d accepted a non-raw zero-width score band")
    return {
        "score_band_maps_only_middle_scores_to_pending": True,
        "product_ineligible_event_is_forced_pending": True,
        "raw_argmax_degenerate_band_is_deterministic": True,
        "only_the_0_5_raw_argmax_zero_width_band_is_legal": True,
        "invalid_thresholds_are_rejected": True,
    }


def _assert_cross_fold_isolation() -> dict[str, bool]:
    rows = _synthetic_parent_rows()
    thresholds = _cross_fit_thresholds(rows)
    if set(thresholds) != set(range(FOLD_COUNT)):
        raise AssertionError("V24d synthetic cross-fit omitted an outer fold")
    for fold_id, selection in thresholds.items():
        fit_folds = set(int(value) for value in selection["fit_fold_ids"])
        if fold_id in fit_folds or fit_folds != set(range(FOLD_COUNT)) - {fold_id}:
            raise AssertionError("V24d held-out fold leaked into synthetic threshold fit")
        if selection["fit_parent_support"] != 16 or selection["held_out_parent_support"] != 4:
            raise AssertionError("V24d synthetic cross-fit support differs from fold isolation")
    window_row = dict(rows[0])
    window_row["product_action_eligible"] = False
    annotated = _annotate_actions(
        [window_row], scope="window", thresholds=thresholds
    )[0]
    if (
        int(annotated["raw_action_id"]) != PENDING_ACTION
        or int(annotated["calibrated_action_id"]) != PENDING_ACTION
    ):
        raise AssertionError("V24d allowed an ineligible window to emit a class action")
    return {
        "threshold_fit_excludes_held_out_outer_fold": True,
        "threshold_fit_uses_parent_records_only": True,
        "ineligible_windows_remain_pending_before_and_after_score_band": True,
    }


def _assert_held_out_label_invariance() -> dict[str, bool]:
    """Changing a held-out fold's labels cannot influence its fitted thresholds."""

    rows = _synthetic_parent_rows()
    before = _cross_fit_thresholds(rows)[0]
    relabelled = [dict(row) for row in rows]
    for row in relabelled:
        if int(row["fold_id"]) == 0:
            row["true_id"] = 1 - int(row["true_id"])
    after = _cross_fit_thresholds(relabelled)[0]
    if json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True):
        raise AssertionError("V24d held-out labels changed their own fitted threshold")
    return {
        "held_out_fold_label_mutation_does_not_change_its_threshold_selection": True,
    }


def _assert_exact_breakpoint_search() -> dict[str, bool]:
    rows = _synthetic_parent_rows()
    selection = _select_thresholds(rows, fit_fold_ids=list(range(FOLD_COUNT)))
    unique_scores = {
        0.0,
        1.0,
        *(float(row["known_clutter_probability"]) for row in rows),
    }
    expected_candidate_count = 1 + len(unique_scores) * (len(unique_scores) - 1) // 2
    if int(selection["candidate_count"]) != expected_candidate_count:
        raise AssertionError("V24d exact search omitted or duplicated breakpoint candidates")
    scores = np.asarray(
        [float(row["known_clutter_probability"]) for row in rows], dtype=np.float64
    )
    observed = _score_band_actions(
        scores,
        target_threshold=float(selection["target_threshold"]),
        clutter_threshold=float(selection["clutter_threshold"]),
    )
    expected_matrix = np.zeros((2, 3), dtype=np.float64)
    np.add.at(
        expected_matrix,
        (
            np.asarray([int(row["true_id"]) for row in rows], dtype=np.int64),
            observed,
        ),
        1.0,
    )
    selected_matrix = np.asarray(
        selection["metrics"]["action_confusion_rows_true_columns_action"], dtype=np.float64
    )
    if not np.array_equal(expected_matrix, selected_matrix):
        raise AssertionError("V24d exact-search candidate metrics differ from executable actions")
    return {
        "exact_breakpoint_pair_count_is_complete": True,
        "selected_exact_candidate_metrics_match_executable_score_band": True,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    preparation = _require_preparation(root)
    source = _load_arm(root, "treatment")
    records = {
        scope: [dict(row) for row in source["records_by_scope"][scope]]
        for scope in ("parent", "eligible_prefix", "window")
    }
    _validate_source_records(records)
    readiness = preparation["readiness"]
    if (
        int(readiness["known_train_parent_count"]) != SOURCE_PARENT_COUNT
        or readiness["threshold_fit_held_out_labels_accessed"] is not False
        or int(readiness["validation_payload_access_count"]) != 0
        or int(readiness["test_payload_access_count"]) != 0
        or int(readiness["unknown_training_exposure"]) != 0
        or readiness["deployment_eligible"] is not False
        or not (root / READINESS).is_file()
    ):
        raise AssertionError("V24d readiness does not preserve sealed OOF-only semantics")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V24d-known-gate-l3-oof-score-band-smoke-v1",
        "source_oof_parent_count": SOURCE_PARENT_COUNT,
        "source_scope_counts": {scope: len(rows) for scope, rows in records.items()},
        **_assert_policy_semantics(),
        **_assert_cross_fold_isolation(),
        **_assert_held_out_label_invariance(),
        **_assert_exact_breakpoint_search(),
        "l1_model_training": "not_executed_not_applicable",
        "validation": "not_executed",
        "test": "not_executed",
        "unknown": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V24d's L3 OOF score-band contract.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
