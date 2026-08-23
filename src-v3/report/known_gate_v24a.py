"""Paired V24-versus-V24a development report for RD peak-drift evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from build.known_gate_v24 import (
    ARTIFACT as CONTROL_ARTIFACT,
    CONFIG as CONTROL_CONFIG,
)
from build.known_gate_v24a import (
    ARTIFACT,
    CONFIG,
    EXPERIMENT_CONTRACT,
    READINESS,
    VALIDATION_CLUTTER_PARENT_SUPPORT,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V24A_READINESS_SCHEMA,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_features import (
    V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
)
from train.known_gate_v21 import _verify_contract, _verify_readiness
from radar_ln3.known_gate import load_known_gate_config, sha256_file


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-paired-development.json"
)
SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 24043
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
EXPECTED_SUPPORT = {
    "DroneTarget": 113,
    "BirdTarget": 56,
    "BalloonTarget": 54,
    "ClutterTarget": 9,
}
COURSE_HIGH_THRESHOLD = 0.05


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _records(detail: Mapping[str, Any], *, identity: str) -> dict[str, list[dict[str, Any]]]:
    rows = {
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in ("parent", "eligible_prefix", "window")
    }
    expected_counts = {"parent": 232, "eligible_prefix": 3756, "window": 4452}
    if {scope: len(value) for scope, value in rows.items()} != expected_counts:
        raise ValueError(f"{identity} V24a record counts differ from frozen V24 scope")
    parent_ids = [str(row.get("parent_sample_id", "")) for row in rows["parent"]]
    if len(parent_ids) != len(set(parent_ids)) or not all(parent_ids):
        raise ValueError(f"{identity} V24a parent records are incomplete or duplicated")
    support = {
        label: sum(str(row.get("label_name")) == label for row in rows["parent"])
        for label in LABELS
    }
    if support != EXPECTED_SUPPORT:
        raise ValueError(f"{identity} V24a parent label support changed: {support}")
    return rows


def _metrics(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "parent_final": summarize_known_gate_records(rows["parent"], scope="parent_final"),
        "eligible_prefix": summarize_known_gate_records(
            rows["eligible_prefix"], scope="eligible_causal_prefix"
        ),
        "parent_balanced_eligible_prefix": (
            summarize_parent_balanced_eligible_prefix_records(
                rows["eligible_prefix"], scope="parent_balanced_eligible_causal_prefix"
            )
        ),
        "window": summarize_known_gate_records(rows["window"], scope="causal_windows"),
    }


def _wilson(
    successes: int, support: int, *, z: float = 1.959963984540054
) -> dict[str, float | int | None | str]:
    if support < 0 or successes < 0 or successes > support:
        raise ValueError("Wilson interval count is invalid")
    if support == 0:
        return {
            "support": 0,
            "successes": 0,
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "status": "undefined_zero_denominator",
        }
    p = float(successes / support)
    denominator = 1.0 + z * z / support
    center = (p + z * z / (2.0 * support)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / support + z * z / (4.0 * support * support)) / denominator
    return {
        "support": int(support),
        "successes": int(successes),
        "estimate": p,
        "lower_95": float(max(0.0, center - half)),
        "upper_95": float(min(1.0, center + half)),
        "status": "defined",
    }


def _clutter_wilson(parent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in parent_rows:
        matrix[int(row["true_id"]), int(row["pred_id"])] += 1
    tp = int(matrix[1, 1])
    return {
        "method": "Wilson score interval, two-sided 95 percent",
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
        "clutter_precision": _wilson(tp, int(matrix[:, 1].sum())),
        "clutter_recall": _wilson(tp, int(matrix[1].sum())),
        "warning": "Only nine validation Clutter parents; development-only uncertainty remains wide.",
    }


def _metric_vector(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics = summarize_known_gate_records(records, scope="v24a_parent_bootstrap")
    per_class = {str(row["class"]): row for row in metrics["per_class"]}
    target = per_class["KNOWN_TARGET_EVIDENCE"]
    clutter = per_class["KNOWN_CLUTTER_EVIDENCE"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "known_target_recall": float(target["recall"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_f1": float(clutter["f1"]),
    }


def _ci(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _ordered_pair(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        result: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["parent_sample_id"]), int(row["event_index"]))
            if key in result:
                raise ValueError(f"V24a paired {scope} records repeat key {key}")
            result[key] = dict(row)
        return result

    before = keyed(control)
    after = keyed(treatment)
    if set(before) != set(after):
        raise ValueError(f"V24a paired {scope} record keys differ")
    ordered = sorted(before)
    for key in ordered:
        if (
            str(before[key]["label_name"]) != str(after[key]["label_name"])
            or int(before[key]["true_id"]) != int(after[key]["true_id"])
            or int(before[key]["parent_length"]) != int(after[key]["parent_length"])
        ):
            raise ValueError(f"V24a paired {scope} record identity differs")
    return [before[key] for key in ordered], [after[key] for key in ordered]


def _paired_bootstrap(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    before, after = _ordered_pair(control, treatment, scope="parent")
    strata = {
        label: np.asarray(
            [index for index, row in enumerate(before) if str(row["label_name"]) == label],
            dtype=np.int64,
        )
        for label in LABELS
    }
    if {label: len(indices) for label, indices in strata.items()} != EXPECTED_SUPPORT:
        raise ValueError("V24a bootstrap strata differ from frozen support")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    before_values: dict[str, list[float]] = {}
    after_values: dict[str, list[float]] = {}
    delta_values: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        before_vector = _metric_vector([before[int(index)] for index in sampled])
        after_vector = _metric_vector([after[int(index)] for index in sampled])
        for key in before_vector:
            before_values.setdefault(key, []).append(before_vector[key])
            after_values.setdefault(key, []).append(after_vector[key])
            delta_values.setdefault(key, []).append(after_vector[key] - before_vector[key])
    return {
        "method": "paired parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": EXPECTED_SUPPORT,
        "control": {key: _ci(value) for key, value in before_values.items()},
        "treatment": {key: _ci(value) for key, value in after_values.items()},
        "treatment_minus_control": {key: _ci(value) for key, value in delta_values.items()},
    }


def _load_feature_maps(root: Path) -> tuple[dict[str, float], dict[str, bool]]:
    config = load_known_gate_config(root / CONFIG)
    path = root / str(config.data.validation_feature_cache_path)
    with np.load(path, allow_pickle=False) as cache:
        sample_ids = [str(value) for value in np.asarray(cache["sample_ids"]).tolist()]
        values = np.asarray(cache["values"])
        valid = np.asarray(cache["valid"])
    if (
        values.shape != (4452, 2)
        or valid.shape != values.shape
        or values.dtype != np.float32
        or valid.dtype != np.bool_
        or np.any(values[~valid] != 0.0)
    ):
        raise ValueError("V24a validation cache does not match its frozen shape/invalid policy")
    return (
        {sample_id: float(values[index, 1]) for index, sample_id in enumerate(sample_ids)},
        {sample_id: bool(valid[index, 1]) for index, sample_id in enumerate(sample_ids)},
    )


def _index_event_metadata(root: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    import pandas as pd

    frame = pd.read_csv(root / SOURCE_INDEX, low_memory=False)
    selected = frame[
        frame["split_name"].astype(str).eq("val")
        & frame["label_name"].astype(str).isin(LABELS)
    ].copy()
    if len(selected) != 4452:
        raise ValueError("V24a validation index event support differs from frozen contract")
    events: dict[tuple[str, int], dict[str, Any]] = {}
    capture_by_parent: dict[str, str] = {}
    for row in selected.itertuples(index=False):
        parent_id = str(row.parent_sample_id)
        key = (parent_id, int(row.causal_event_index))
        if key in events:
            raise ValueError("V24a validation index repeats parent/event key")
        capture_id = str(row.capture_id)
        previous_capture = capture_by_parent.setdefault(parent_id, capture_id)
        if previous_capture != capture_id:
            raise ValueError("V24a parent crosses capture strata")
        events[key] = {
            "sample_id": str(row.sample_id),
            "capture_id": capture_id,
            "rd_frame_count": int(row.rd_frame_count),
        }
    return events, capture_by_parent


def _course_stratum(row: Mapping[str, Any], course_value_by_event: Mapping[tuple[str, int], float | None]) -> str:
    value = course_value_by_event[(str(row["parent_sample_id"]), int(row["event_index"]))]
    if value is None:
        return "invalid"
    return "valid_high" if value >= COURSE_HIGH_THRESHOLD else "valid_low"


def _extract_course_values(root: Path) -> dict[tuple[str, int], float | None]:
    config = load_known_gate_config(root / CONTROL_CONFIG)
    path = root / str(config.data.validation_feature_cache_path)
    with np.load(path, allow_pickle=False) as cache:
        sample_ids = [str(value) for value in np.asarray(cache["sample_ids"]).tolist()]
        values = np.asarray(cache["values"])
        valid = np.asarray(cache["valid"])
    if values.shape != (4452, 1) or valid.shape != values.shape:
        raise ValueError("V24 course cache shape differs from frozen control")
    events, _ = _index_event_metadata(root)
    key_by_sample = {value["sample_id"]: key for key, value in events.items()}
    if set(sample_ids) != set(key_by_sample):
        raise ValueError("V24 course cache sample set differs from validation index")
    return {
        key_by_sample[sample_id]: (float(values[index, 0]) if bool(valid[index, 0]) else None)
        for index, sample_id in enumerate(sample_ids)
    }


def _outcome_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    known_to_clutter = sum(int(row["true_id"]) == 0 and int(row["pred_id"]) == 1 for row in rows)
    clutter_to_known = sum(int(row["true_id"]) == 1 and int(row["pred_id"]) == 0 for row in rows)
    bird_to_clutter = sum(
        str(row["label_name"]) == "BirdTarget" and int(row["pred_id"]) == 1
        for row in rows
    )
    return {
        "support": len(rows),
        "KnownTarget_to_Clutter": int(known_to_clutter),
        "Clutter_to_KnownTarget": int(clutter_to_known),
        "BirdTarget_to_Clutter": int(bird_to_clutter),
    }


def _stratified_outcomes(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    stratum_by_key: Mapping[tuple[str, int], str],
    scope: str,
) -> dict[str, Any]:
    before, after = _ordered_pair(control, treatment, scope=scope)
    grouped_before: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_after: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for left, right in zip(before, after, strict=True):
        key = (str(left["parent_sample_id"]), int(left["event_index"]))
        stratum = stratum_by_key[key]
        grouped_before[stratum].append(left)
        grouped_after[stratum].append(right)
    result: dict[str, Any] = {}
    for stratum in sorted(set(grouped_before) | set(grouped_after)):
        left = grouped_before[stratum]
        right = grouped_after[stratum]
        if len(left) != len(right):
            raise ValueError("V24a stratum pair length differs")
        changes = sum(
            int(a["pred_id"]) != int(b["pred_id"])
            for a, b in zip(left, right, strict=True)
        )
        result[stratum] = {
            "control": _outcome_counts(left),
            "treatment": _outcome_counts(right),
            "prediction_changed_count": int(changes),
        }
    return result


def _strata_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_strata: Mapping[tuple[str, int], str],
    scope: str,
) -> dict[tuple[str, int], str]:
    """Bind a cached event stratum to one evaluated record per causal event."""

    result: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (str(row["parent_sample_id"]), int(row["event_index"]))
        if key in result:
            raise ValueError(f"V24a {scope} rows repeat stratum key {key}")
        try:
            result[key] = event_strata[key]
        except KeyError as exc:
            raise ValueError(f"V24a {scope} row lacks cached event stratum {key}") from exc
    return result


def _high_course_bird_audit(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    course_strata: Mapping[tuple[str, int], str],
    scope: str,
) -> dict[str, Any]:
    """Report the preregistered Bird-to-Clutter direction at one scope."""

    before, after = _ordered_pair(control, treatment, scope=scope)
    selected_before = [
        row
        for row in before
        if str(row["label_name"]) == "BirdTarget"
        and course_strata[(str(row["parent_sample_id"]), int(row["event_index"]))]
        == "valid_high"
    ]
    selected_after = [
        row
        for row in after
        if str(row["label_name"]) == "BirdTarget"
        and course_strata[(str(row["parent_sample_id"]), int(row["event_index"]))]
        == "valid_high"
    ]
    control_counts = _outcome_counts(selected_before)
    treatment_counts = _outcome_counts(selected_after)
    return {
        "control": control_counts,
        "treatment": treatment_counts,
        "BirdTarget_to_Clutter_delta": int(
            treatment_counts["BirdTarget_to_Clutter"]
            - control_counts["BirdTarget_to_Clutter"]
        ),
    }


def _parent_changes(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    capture_by_parent: Mapping[str, str],
    course_by_event: Mapping[tuple[str, int], float | None],
    drift_by_sample: Mapping[str, float],
    drift_valid_by_sample: Mapping[str, bool],
    metadata_by_event: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(control, treatment, scope="parent")
    changes: list[dict[str, Any]] = []
    for left, right in zip(before, after, strict=True):
        parent_id = str(left["parent_sample_id"])
        key = (parent_id, int(left["event_index"]))
        meta = metadata_by_event[key]
        if str(left["label_name"]) != str(right["label_name"]):
            raise ValueError("V24a parent label differs in paired change audit")
        if int(left["pred_id"]) == int(right["pred_id"]):
            continue
        control_correct = int(left["pred_id"]) == int(left["true_id"])
        treatment_correct = int(right["pred_id"]) == int(right["true_id"])
        changes.append(
            {
                "parent_sample_id": parent_id,
                "track_id": str(left["track_id"]),
                "capture_id": capture_by_parent[parent_id],
                "label_name": str(left["label_name"]),
                "event_index": int(left["event_index"]),
                "parent_length": int(left["parent_length"]),
                "course_circular_incoherence": course_by_event[key],
                "course_stratum": _course_stratum(left, course_by_event),
                "rd_frame_count": int(meta["rd_frame_count"]),
                "rd_peak_drift_valid": bool(drift_valid_by_sample[meta["sample_id"]]),
                "rd_peak_drift_mean_fraction": (
                    float(drift_by_sample[meta["sample_id"]])
                    if drift_valid_by_sample[meta["sample_id"]]
                    else None
                ),
                "control_prediction": int(left["pred_id"]),
                "treatment_prediction": int(right["pred_id"]),
                "control_known_clutter_probability": float(left["known_clutter_probability"]),
                "treatment_known_clutter_probability": float(right["known_clutter_probability"]),
                "control_correct": bool(control_correct),
                "treatment_correct": bool(treatment_correct),
                "change_type": (
                    "rescue" if (not control_correct and treatment_correct)
                    else "harm" if (control_correct and not treatment_correct)
                    else "incorrect_direction_swap"
                ),
            }
        )
    changes.sort(key=lambda row: (row["label_name"], row["parent_sample_id"]))
    return {
        "changed_parent_count": len(changes),
        "rescued_parent_count": sum(row["change_type"] == "rescue" for row in changes),
        "harmed_parent_count": sum(row["change_type"] == "harm" for row in changes),
        "changes": changes,
    }


def _capture_strata(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    capture_by_parent: Mapping[str, str],
) -> list[dict[str, Any]]:
    before, after = _ordered_pair(control, treatment, scope="parent")
    grouped_before: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_after: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for left, right in zip(before, after, strict=True):
        capture = capture_by_parent[str(left["parent_sample_id"])]
        grouped_before[capture].append(left)
        grouped_after[capture].append(right)
    return [
        {
            "capture_id": capture,
            "control": _outcome_counts(grouped_before[capture]),
            "treatment": _outcome_counts(grouped_after[capture]),
            "parent_prediction_changed_count": sum(
                int(left["pred_id"]) != int(right["pred_id"])
                for left, right in zip(
                    grouped_before[capture], grouped_after[capture], strict=True
                )
            ),
        }
        for capture in sorted(grouped_before)
    ]


def _load_run(
    root: Path,
    *,
    artifact: Path,
    config_path: Path,
    variant: str,
    readiness_sha256: str | None,
) -> dict[str, Any]:
    summary_path = root / artifact / "training-summary.json"
    detail_path = root / artifact / "sample_detail_information.json"
    checkpoint_path = root / artifact / "best_model.pt"
    if not all(path.is_file() for path in (summary_path, detail_path, checkpoint_path)):
        raise FileNotFoundError(f"V24a {variant} training output is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != variant
        or int(summary.get("seed", -1)) != 43
        or len(list(summary.get("history") or ())) != 20
        or Path(str(summary.get("config_path", ""))).resolve() != config_path.resolve()
        or Path(str(summary.get("best_checkpoint", ""))).resolve() != checkpoint_path.resolve()
        or Path(str(summary.get("sample_detail_information", ""))).resolve()
        != detail_path.resolve()
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or (
            readiness_sha256 is not None
            and dict(summary.get("training_readiness") or {}).get("sha256")
            != readiness_sha256
        )
    ):
        raise ValueError(f"V24a {variant} summary violates the sealed development contract")
    rows = _records(detail, identity=variant)
    parent_metrics = _metrics(rows)["parent_final"]
    recorded = dict(dict(summary.get("best_validation_metrics") or {}).get("parent_metrics") or {})
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(parent_metrics[key]) - float(recorded.get(key, np.nan))) > 1.0e-12:
            raise ValueError(f"V24a {variant} summary/recomputed metric differs at {key}")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "detail_path": str(detail_path),
        "detail_sha256": sha256_file(detail_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "rows": rows,
        "metrics": _metrics(rows),
    }


def _scope_safety(
    control_metrics: Mapping[str, Any], treatment_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    scopes = ("parent_final", "eligible_prefix", "parent_balanced_eligible_prefix", "window")
    result: dict[str, Any] = {}
    for scope in scopes:
        before = {row["class"]: row for row in control_metrics[scope]["per_class"]}
        after = {row["class"]: row for row in treatment_metrics[scope]["per_class"]}
        result[scope] = {
            "known_target_recall_delta": float(
                after["KNOWN_TARGET_EVIDENCE"]["recall"]
                - before["KNOWN_TARGET_EVIDENCE"]["recall"]
            ),
            "clutter_precision_delta": float(
                after["KNOWN_CLUTTER_EVIDENCE"]["precision"]
                - before["KNOWN_CLUTTER_EVIDENCE"]["precision"]
            ),
            "clutter_recall_delta": float(
                after["KNOWN_CLUTTER_EVIDENCE"]["recall"]
                - before["KNOWN_CLUTTER_EVIDENCE"]["recall"]
            ),
            "macro_f1_delta": float(
                treatment_metrics[scope]["macro_f1"] - control_metrics[scope]["macro_f1"]
            ),
        }
    eligible = result["eligible_prefix"]
    parent_balanced = result["parent_balanced_eligible_prefix"]
    return {
        "by_scope": result,
        "eligible_prefix_known_target_recall_not_lower": eligible["known_target_recall_delta"] >= 0.0,
        "eligible_prefix_clutter_recall_not_lower": eligible["clutter_recall_delta"] >= 0.0,
        "parent_balanced_eligible_prefix_known_target_recall_not_lower": (
            parent_balanced["known_target_recall_delta"] >= 0.0
        ),
        "parent_balanced_eligible_prefix_clutter_recall_not_lower": (
            parent_balanced["clutter_recall_delta"] >= 0.0
        ),
    }


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    if not all(path.is_file() for path in (config_path, readiness_path, contract_path)):
        raise FileNotFoundError("V24a config/readiness/contract is incomplete")
    config = load_known_gate_config(config_path)
    contract_evidence = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract_evidence)
    if (
        config.variant != V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT
        or config.feature_names
        != (
            "causal_prefix_course_circular_incoherence",
            "current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
        )
        or config.model.evidence_dim != 2
        or readiness.get("schema_version") != KNOWN_GATE_V24A_READINESS_SCHEMA
    ):
        raise ValueError("V24a config/readiness identity differs")
    control = _load_run(
        root,
        artifact=CONTROL_ARTIFACT,
        config_path=root / CONTROL_CONFIG,
        variant=V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
        readiness_sha256=None,
    )
    treatment = _load_run(
        root,
        artifact=ARTIFACT,
        config_path=config_path,
        variant=V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
        readiness_sha256=readiness["sha256"],
    )
    pair = dict(treatment["summary"].get("paired_initialization") or {})
    if (
        pair.get("paired_baseline") is not True
        or pair.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pair.get("new_fusion_columns_exactly_zero") is not True
        or int(pair.get("new_fusion_columns", -1)) != 1
        or pair.get("initial_logits_exactly_equal") is not True
        or pair.get("sealed_v24_control_initial_model_state_reproduced") is not True
        or pair.get("sealed_v24_control_initial_model_state_sha256")
        != control["summary"].get("initial_model_state_sha256")
    ):
        raise ValueError("V24a treatment does not prove frozen V24 paired initialization")
    control_orders = list(
        dict(control["summary"].get("dataloader_contract") or {}).get("epoch_parent_order_hashes") or []
    )
    treatment_orders = list(
        dict(treatment["summary"].get("dataloader_contract") or {}).get("epoch_parent_order_hashes") or []
    )
    if control_orders != treatment_orders or len(control_orders) != 20:
        raise ValueError("V24a control/treatment parent orders differ")
    metadata_by_event, capture_by_parent = _index_event_metadata(root)
    course_by_event = _extract_course_values(root)
    drift_by_sample, drift_valid_by_sample = _load_feature_maps(root)
    event_drift_strata = {
        key: (
            "valid" if drift_valid_by_sample[str(meta["sample_id"])] else "invalid"
        )
        for key, meta in metadata_by_event.items()
    }
    event_course_strata = {
        key: (
            "invalid"
            if course_by_event[key] is None
            else "valid_high"
            if float(course_by_event[key]) >= COURSE_HIGH_THRESHOLD
            else "valid_low"
        )
        for key in metadata_by_event
    }
    control_parent, treatment_parent = _ordered_pair(
        control["rows"]["parent"], treatment["rows"]["parent"], scope="parent"
    )
    parent_course_strata = _strata_for_rows(
        control_parent, event_strata=event_course_strata, scope="parent_course"
    )
    parent_drift_strata = _strata_for_rows(
        control_parent, event_strata=event_drift_strata, scope="parent_rd_drift_validity"
    )
    changes = _parent_changes(
        control["rows"]["parent"],
        treatment["rows"]["parent"],
        capture_by_parent=capture_by_parent,
        course_by_event=course_by_event,
        drift_by_sample=drift_by_sample,
        drift_valid_by_sample=drift_valid_by_sample,
        metadata_by_event=metadata_by_event,
    )
    metrics_delta = {
        scope: {
            key: float(treatment["metrics"][scope][key] - control["metrics"][scope][key])
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        }
        for scope in ("parent_final", "eligible_prefix", "parent_balanced_eligible_prefix", "window")
    }
    safety = _scope_safety(control["metrics"], treatment["metrics"])
    course_parent = _stratified_outcomes(
        control_parent,
        treatment_parent,
        stratum_by_key=parent_course_strata,
        scope="parent_course",
    )
    drift_parent = _stratified_outcomes(
        control_parent,
        treatment_parent,
        stratum_by_key=parent_drift_strata,
        scope="parent_rd_drift_validity",
    )
    scope_rows = {
        "parent_final": (control_parent, treatment_parent),
        "eligible_prefix": _ordered_pair(
            control["rows"]["eligible_prefix"],
            treatment["rows"]["eligible_prefix"],
            scope="eligible_prefix",
        ),
        "window": _ordered_pair(
            control["rows"]["window"],
            treatment["rows"]["window"],
            scope="window",
        ),
    }
    scope_course_strata = {
        scope: _strata_for_rows(
            rows[0], event_strata=event_course_strata, scope=f"{scope}_course"
        )
        for scope, rows in scope_rows.items()
    }
    scope_drift_strata = {
        scope: _strata_for_rows(
            rows[0], event_strata=event_drift_strata, scope=f"{scope}_rd_drift_validity"
        )
        for scope, rows in scope_rows.items()
    }
    course_strata_by_scope = {
        scope: _stratified_outcomes(
            rows[0],
            rows[1],
            stratum_by_key=scope_course_strata[scope],
            scope=f"{scope}_course",
        )
        for scope, rows in scope_rows.items()
    }
    drift_strata_by_scope = {
        scope: _stratified_outcomes(
            rows[0],
            rows[1],
            stratum_by_key=scope_drift_strata[scope],
            scope=f"{scope}_rd_drift_validity",
        )
        for scope, rows in scope_rows.items()
    }
    high_course_bird_by_scope = {
        scope: _high_course_bird_audit(
            rows[0],
            rows[1],
            course_strata=scope_course_strata[scope],
            scope=scope,
        )
        for scope, rows in scope_rows.items()
    }
    high_bird_delta = int(
        high_course_bird_by_scope["parent_final"]["BirdTarget_to_Clutter_delta"]
    )
    parent_known_to_clutter_delta = (
        _outcome_counts(treatment_parent)["KnownTarget_to_Clutter"]
        - _outcome_counts(control_parent)["KnownTarget_to_Clutter"]
    )
    parent_clutter_to_known_delta = (
        _outcome_counts(treatment_parent)["Clutter_to_KnownTarget"]
        - _outcome_counts(control_parent)["Clutter_to_KnownTarget"]
    )
    report = {
        "status": "completed_development_paired_only",
        "schema_version": "RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-paired-development-v1",
        "scope": "V24a development KnownGate L1 paired experiment only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "feature_semantics": {
            "feature": "current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
            "current_event_only": True,
            "preceding_event_frame_bridge_used": False,
            "fewer_than_two_current_rd_frames": "invalid_value_zero_not_a_valid_zero",
            "rd_drift_validity": (
                "retained for cache/audit stratification only; gating-only validity is not a "
                "fusion input, so invalid and valid physical zero both contribute numeric zero"
            ),
            "course_strata": {
                "invalid": "causal course feature invalid",
                "valid_low": f"valid course incoherence < {COURSE_HIGH_THRESHOLD}",
                "valid_high": f"valid course incoherence >= {COURSE_HIGH_THRESHOLD}",
            },
        },
        "control": {
            "metrics": control["metrics"],
            "summary_path": control["summary_path"],
            "summary_sha256": control["summary_sha256"],
            "checkpoint_path": control["checkpoint_path"],
            "checkpoint_sha256": control["checkpoint_sha256"],
        },
        "treatment": {
            "metrics": treatment["metrics"],
            "summary_path": treatment["summary_path"],
            "summary_sha256": treatment["summary_sha256"],
            "checkpoint_path": treatment["checkpoint_path"],
            "checkpoint_sha256": treatment["checkpoint_sha256"],
        },
        "treatment_minus_control": metrics_delta,
        "paired_protocol": {
            "seed": 43,
            "initial_shared_state_exact": True,
            "new_drift_fusion_column_initially_zero": True,
            "initial_logits_exactly_equal": True,
            "all_20_epoch_parent_order_hashes_equal": True,
            "full_parameter_training": True,
            "control_checkpoint_is_not_a_finetune_source": True,
        },
        "paired_parent_bootstrap": _paired_bootstrap(control_parent, treatment_parent),
        "clutter_parent_wilson_intervals": {
            "control": _clutter_wilson(control_parent),
            "treatment": _clutter_wilson(treatment_parent),
            "validation_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        },
        "error_direction_audit": {
            "parent_final_control": _outcome_counts(control_parent),
            "parent_final_treatment": _outcome_counts(treatment_parent),
            "high_course_bird_by_scope": high_course_bird_by_scope,
            "KnownTarget_to_Clutter_delta": int(parent_known_to_clutter_delta),
            "Clutter_to_KnownTarget_delta": int(parent_clutter_to_known_delta),
        },
        "eligible_prefix_safety": safety,
        "course_incoherence_strata_by_scope": course_strata_by_scope,
        "rd_drift_validity_strata_by_scope": drift_strata_by_scope,
        # Preserve the original parent aliases for downstream readers while
        # making the prefix/window strata explicit and independently auditable.
        "course_incoherence_parent_strata": course_parent,
        "rd_drift_validity_parent_strata": drift_parent,
        "capture_parent_strata": _capture_strata(
            control_parent, treatment_parent, capture_by_parent
        ),
        "per_parent_changes": changes,
        "frozen_decision": {
            "high_course_bird_to_clutter_not_increased": high_bird_delta <= 0,
            "high_course_bird_eligible_prefix_to_clutter_not_increased": (
                high_course_bird_by_scope["eligible_prefix"]["BirdTarget_to_Clutter_delta"]
                <= 0
            ),
            "high_course_bird_window_to_clutter_not_increased": (
                high_course_bird_by_scope["window"]["BirdTarget_to_Clutter_delta"]
                <= 0
            ),
            "overall_known_to_clutter_not_increased": parent_known_to_clutter_delta <= 0,
            "clutter_to_known_not_increased": parent_clutter_to_known_delta <= 0,
            "parent_final_known_target_recall_not_lower": (
                safety["by_scope"]["parent_final"]["known_target_recall_delta"] >= 0.0
            ),
            "eligible_prefix_safety_passed": (
                safety["eligible_prefix_known_target_recall_not_lower"]
                and safety["eligible_prefix_clutter_recall_not_lower"]
                and safety["parent_balanced_eligible_prefix_known_target_recall_not_lower"]
                and safety["parent_balanced_eligible_prefix_clutter_recall_not_lower"]
            ),
            "development_only_no_release_claim": True,
        },
        "experiment_contract_path": str(contract_path),
        "experiment_contract_sha256": sha256_file(contract_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": readiness["sha256"],
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, report)
    return {**report, "report_path": str(root / OUTPUT)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V24a current-event RD peak drift against V24 control."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
