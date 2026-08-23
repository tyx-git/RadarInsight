"""Paired V24-versus-V24c development report for strict same-second skipping."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from build.known_gate_v24 import ARTIFACT as CONTROL_ARTIFACT
from build.known_gate_v24 import CONFIG as CONTROL_CONFIG
from build.known_gate_v24c import (
    ARTIFACT,
    CONFIG,
    EXPERIMENT_CONTRACT,
    READINESS,
    VALIDATION_CLUTTER_PARENT_SUPPORT,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V24C_READINESS_SCHEMA,
    load_known_gate_config,
    sha256_file,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_features import (
    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
)
from train.known_gate_v21 import _verify_contract, _verify_readiness


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-paired-development.json"
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
ZERO_DELTA_REASON = "zero_delta_at_second_precision"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _records(
    detail: Mapping[str, Any], *, identity: str
) -> dict[str, list[dict[str, Any]]]:
    rows = {
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in ("parent", "eligible_prefix", "window")
    }
    expected_counts = {"parent": 232, "eligible_prefix": 3756, "window": 4452}
    if {scope: len(value) for scope, value in rows.items()} != expected_counts:
        raise ValueError(f"{identity} V24c record counts differ from frozen V24 scope")
    parent_ids = [str(row.get("parent_sample_id", "")) for row in rows["parent"]]
    if len(parent_ids) != len(set(parent_ids)) or not all(parent_ids):
        raise ValueError(f"{identity} V24c parent records are incomplete or duplicated")
    support = {
        label: sum(str(row.get("label_name")) == label for row in rows["parent"])
        for label in LABELS
    }
    if support != EXPECTED_SUPPORT:
        raise ValueError(f"{identity} V24c parent label support changed: {support}")
    return rows


def _metrics(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "parent_final": summarize_known_gate_records(rows["parent"], scope="parent_final"),
        "eligible_prefix": summarize_known_gate_records(
            rows["eligible_prefix"], scope="eligible_causal_prefix"
        ),
        "parent_balanced_eligible_prefix": (
            summarize_parent_balanced_eligible_prefix_records(
                rows["eligible_prefix"],
                scope="parent_balanced_eligible_causal_prefix",
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
    half = z * np.sqrt(
        p * (1.0 - p) / support + z * z / (4.0 * support * support)
    ) / denominator
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
    metrics = summarize_known_gate_records(records, scope="v24c_parent_bootstrap")
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
                raise ValueError(f"V24c paired {scope} records repeat key {key}")
            result[key] = dict(row)
        return result

    before = keyed(control)
    after = keyed(treatment)
    if set(before) != set(after):
        raise ValueError(f"V24c paired {scope} record keys differ")
    ordered = sorted(before)
    for key in ordered:
        if (
            str(before[key]["label_name"]) != str(after[key]["label_name"])
            or int(before[key]["true_id"]) != int(after[key]["true_id"])
            or int(before[key]["parent_length"]) != int(after[key]["parent_length"])
        ):
            raise ValueError(f"V24c paired {scope} record identity differs")
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
        raise ValueError("V24c bootstrap strata differ from frozen support")
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
            delta_values.setdefault(key, []).append(
                after_vector[key] - before_vector[key]
            )
    return {
        "method": "paired parent bootstrap stratified by original base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": EXPECTED_SUPPORT,
        "control": {key: _ci(value) for key, value in before_values.items()},
        "treatment": {key: _ci(value) for key, value in after_values.items()},
        "treatment_minus_control": {
            key: _ci(value) for key, value in delta_values.items()
        },
    }


def _index_metadata(
    root: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    frame = pd.read_csv(root / SOURCE_INDEX, low_memory=False)
    selected = frame[
        frame["split_name"].astype(str).eq("val")
        & frame["label_name"].astype(str).isin(LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if len(selected) != 4452:
        raise ValueError("V24c validation index event support differs from frozen contract")
    events: dict[tuple[str, int], dict[str, Any]] = {}
    parents: dict[str, dict[str, Any]] = {}
    for parent_id, parent in selected.groupby("parent_sample_id", sort=True):
        parent = parent.sort_values("causal_event_index", kind="stable")
        labels = parent["label_name"].astype(str).unique().tolist()
        captures = parent["capture_id"].astype(str).unique().tolist()
        reasons = parent["tr_dt_reason"].astype(str).tolist()
        if len(labels) != 1 or len(captures) != 1:
            raise ValueError(f"V24c validation parent metadata differs: {parent_id}")
        ordinal = 0
        maximum_ordinal = 0
        zero_count = 0
        for row in parent.itertuples(index=False):
            reason = str(row.tr_dt_reason)
            if reason in {
                "first_event_no_predecessor",
                "positive_delta_at_second_precision",
            }:
                ordinal = 1
            elif reason == ZERO_DELTA_REASON:
                ordinal += 1
                zero_count += 1
                maximum_ordinal = max(maximum_ordinal, ordinal)
            else:
                raise ValueError(f"V24c validation index has unsupported reason: {reason}")
            key = (str(parent_id), int(row.causal_event_index))
            if key in events:
                raise ValueError("V24c validation index repeats parent/event key")
            events[key] = {
                "sample_id": str(row.sample_id),
                "capture_id": captures[0],
                "tr_dt_reason": reason,
                "same_second_ordinal": int(ordinal) if reason == ZERO_DELTA_REASON else 0,
                "rd_present": bool(str(row.rd_present).strip().lower() in {"1", "true", "yes"}),
            }
        parents[str(parent_id)] = {
            "capture_id": captures[0],
            "label_name": labels[0],
            "parent_length": int(len(parent)),
            "same_second_followup_count": int(zero_count),
            "has_same_second_followup": bool(zero_count > 0),
            "maximum_same_second_ordinal": int(maximum_ordinal),
        }
    if len(parents) != 232 or len(events) != 4452:
        raise ValueError("V24c validation metadata is incomplete")
    return events, parents


def _outcome_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    known_to_clutter = sum(
        int(row["true_id"]) == 0 and int(row["pred_id"]) == 1 for row in rows
    )
    clutter_to_known = sum(
        int(row["true_id"]) == 1 and int(row["pred_id"]) == 0 for row in rows
    )
    return {
        "support": len(rows),
        "KnownTarget_to_Clutter": int(known_to_clutter),
        "Clutter_to_KnownTarget": int(clutter_to_known),
        "BirdTarget_to_Clutter": int(
            sum(
                str(row["label_name"]) == "BirdTarget" and int(row["pred_id"]) == 1
                for row in rows
            )
        ),
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
    return {
        stratum: {
            "control": _outcome_counts(grouped_before[stratum]),
            "treatment": _outcome_counts(grouped_after[stratum]),
            "prediction_changed_count": int(
                sum(
                    int(left["pred_id"]) != int(right["pred_id"])
                    for left, right in zip(
                        grouped_before[stratum], grouped_after[stratum], strict=True
                    )
                )
            ),
        }
        for stratum in sorted(grouped_before)
    }


def _parent_changes(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    events: Mapping[tuple[str, int], Mapping[str, Any]],
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(control, treatment, scope="parent")
    changes: list[dict[str, Any]] = []
    for left, right in zip(before, after, strict=True):
        if int(left["pred_id"]) == int(right["pred_id"]):
            continue
        parent_id = str(left["parent_sample_id"])
        key = (parent_id, int(left["event_index"]))
        event = events[key]
        parent = parents[parent_id]
        control_correct = int(left["pred_id"]) == int(left["true_id"])
        treatment_correct = int(right["pred_id"]) == int(right["true_id"])
        changes.append(
            {
                "parent_sample_id": parent_id,
                "track_id": str(left["track_id"]),
                "capture_id": parent["capture_id"],
                "label_name": str(left["label_name"]),
                "event_index": int(left["event_index"]),
                "parent_length": int(left["parent_length"]),
                "parent_has_same_second_followup": bool(
                    parent["has_same_second_followup"]
                ),
                "parent_same_second_followup_count": int(
                    parent["same_second_followup_count"]
                ),
                "parent_maximum_same_second_ordinal": int(
                    parent["maximum_same_second_ordinal"]
                ),
                "event_dt_reason": event["tr_dt_reason"],
                "event_same_second_ordinal": int(event["same_second_ordinal"]),
                "control_prediction": int(left["pred_id"]),
                "treatment_prediction": int(right["pred_id"]),
                "control_known_clutter_probability": float(
                    left["known_clutter_probability"]
                ),
                "treatment_known_clutter_probability": float(
                    right["known_clutter_probability"]
                ),
                "control_correct": bool(control_correct),
                "treatment_correct": bool(treatment_correct),
                "change_type": (
                    "rescue"
                    if (not control_correct and treatment_correct)
                    else "harm"
                    if (control_correct and not treatment_correct)
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
    parents: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    before, after = _ordered_pair(control, treatment, scope="parent_capture")
    grouped_before: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_after: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for left, right in zip(before, after, strict=True):
        capture = str(parents[str(left["parent_sample_id"])]["capture_id"])
        grouped_before[capture].append(left)
        grouped_after[capture].append(right)
    return [
        {
            "capture_id": capture,
            "control": _outcome_counts(grouped_before[capture]),
            "treatment": _outcome_counts(grouped_after[capture]),
            "parent_prediction_changed_count": int(
                sum(
                    int(left["pred_id"]) != int(right["pred_id"])
                    for left, right in zip(
                        grouped_before[capture], grouped_after[capture], strict=True
                    )
                )
            ),
        }
        for capture in sorted(grouped_before)
    ]


def _scope_safety(
    control_metrics: Mapping[str, Any], treatment_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    scopes = (
        "parent_final",
        "eligible_prefix",
        "parent_balanced_eligible_prefix",
        "window",
    )
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
                treatment_metrics[scope]["macro_f1"]
                - control_metrics[scope]["macro_f1"]
            ),
        }
    eligible = result["eligible_prefix"]
    parent_balanced = result["parent_balanced_eligible_prefix"]
    return {
        "by_scope": result,
        "eligible_prefix_known_target_recall_not_lower": (
            eligible["known_target_recall_delta"] >= 0.0
        ),
        "eligible_prefix_clutter_recall_not_lower": (
            eligible["clutter_recall_delta"] >= 0.0
        ),
        "parent_balanced_eligible_prefix_known_target_recall_not_lower": (
            parent_balanced["known_target_recall_delta"] >= 0.0
        ),
        "parent_balanced_eligible_prefix_clutter_recall_not_lower": (
            parent_balanced["clutter_recall_delta"] >= 0.0
        ),
    }


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
        raise FileNotFoundError(f"V24c {variant} training output is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != variant
        or int(summary.get("seed", -1)) != 43
        or len(list(summary.get("history") or ())) != 20
        or Path(str(summary.get("config_path", ""))).resolve() != config_path.resolve()
        or Path(str(summary.get("best_checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
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
        raise ValueError(f"V24c {variant} summary violates the sealed development contract")
    rows = _records(detail, identity=variant)
    parent_metrics = _metrics(rows)["parent_final"]
    recorded = dict(
        dict(summary.get("best_validation_metrics") or {}).get("parent_metrics") or {}
    )
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(float(parent_metrics[key]) - float(recorded.get(key, np.nan))) > 1.0e-12:
            raise ValueError(f"V24c {variant} summary/recomputed metric differs at {key}")
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


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    if not all(path.is_file() for path in (config_path, readiness_path, contract_path)):
        raise FileNotFoundError("V24c config/readiness/contract is incomplete")
    config = load_known_gate_config(config_path)
    contract_evidence = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract_evidence)
    if (
        config.variant != V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT
        or config.feature_names != ("causal_prefix_course_circular_incoherence",)
        or config.model.evidence_dim != 1
        or config.model.liquid_invalid_dt_update_mode != "strict_skip"
        or readiness.get("schema_version") != KNOWN_GATE_V24C_READINESS_SCHEMA
    ):
        raise ValueError("V24c config/readiness identity differs")
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
        variant=V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
        readiness_sha256=readiness["sha256"],
    )
    pair = dict(treatment["summary"].get("paired_initialization") or {})
    if (
        pair.get("paired_baseline") is not True
        or pair.get("dynamics_only_pair") is not True
        or pair.get("shared_parameters_and_buffers_exactly_equal") is not True
        or int(pair.get("new_fusion_columns", -1)) != 0
        or pair.get("sealed_v24_control_initial_model_state_reproduced") is not True
        or pair.get("sealed_v24_control_initial_model_state_sha256")
        != control["summary"].get("initial_model_state_sha256")
        or pair.get("initial_logits_exactly_equal_without_zero_delta") is not True
        or pair.get("strict_skip_state_unchanged_at_every_probe_same_second_event")
        is not True
    ):
        raise ValueError("V24c treatment does not prove frozen V24 dynamics-only pairing")
    control_orders = list(
        dict(control["summary"].get("dataloader_contract") or {}).get(
            "epoch_parent_order_hashes"
        )
        or []
    )
    treatment_orders = list(
        dict(treatment["summary"].get("dataloader_contract") or {}).get(
            "epoch_parent_order_hashes"
        )
        or []
    )
    if control_orders != treatment_orders or len(control_orders) != 20:
        raise ValueError("V24c control/treatment parent orders differ")
    events, parents = _index_metadata(root)
    control_parent, treatment_parent = _ordered_pair(
        control["rows"]["parent"], treatment["rows"]["parent"], scope="parent"
    )
    scope_rows = {
        "parent_final": (control_parent, treatment_parent),
        "eligible_prefix": _ordered_pair(
            control["rows"]["eligible_prefix"],
            treatment["rows"]["eligible_prefix"],
            scope="eligible_prefix",
        ),
        "window": _ordered_pair(
            control["rows"]["window"], treatment["rows"]["window"], scope="window"
        ),
    }
    event_reason_strata = {
        key: str(metadata["tr_dt_reason"]) for key, metadata in events.items()
    }
    event_ordinal_strata = {
        key: (
            "not_same_second"
            if int(metadata["same_second_ordinal"]) == 0
            else "same_second_ordinal_2"
            if int(metadata["same_second_ordinal"]) == 2
            else "same_second_ordinal_3"
            if int(metadata["same_second_ordinal"]) == 3
            else "same_second_ordinal_4_or_more"
        )
        for key, metadata in events.items()
    }
    parent_followup_strata = {
        (str(row["parent_sample_id"]), int(row["event_index"])): (
            "has_same_second_followup"
            if parents[str(row["parent_sample_id"])]["has_same_second_followup"]
            else "no_same_second_followup"
        )
        for row in control_parent
    }
    parent_multiplicity_strata = {
        (str(row["parent_sample_id"]), int(row["event_index"])): (
            "no_same_second_followup"
            if parents[str(row["parent_sample_id"])]["same_second_followup_count"] == 0
            else "one_same_second_followup"
            if parents[str(row["parent_sample_id"])]["same_second_followup_count"] == 1
            else "two_or_more_same_second_followups"
        )
        for row in control_parent
    }
    track_length_strata = {
        (str(row["parent_sample_id"]), int(row["event_index"])): (
            "short_le_5"
            if int(row["parent_length"]) <= 5
            else "medium_6_15"
            if int(row["parent_length"]) <= 15
            else "long_ge_16"
        )
        for row in control_parent
    }
    metrics_delta = {
        scope: {
            key: float(treatment["metrics"][scope][key] - control["metrics"][scope][key])
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        }
        for scope in (
            "parent_final",
            "eligible_prefix",
            "parent_balanced_eligible_prefix",
            "window",
        )
    }
    safety = _scope_safety(control["metrics"], treatment["metrics"])
    same_second_reason_by_scope = {
        scope: _stratified_outcomes(
            rows[0],
            rows[1],
            stratum_by_key=event_reason_strata,
            scope=f"{scope}_dt_reason",
        )
        for scope, rows in scope_rows.items()
    }
    same_second_ordinal_by_scope = {
        scope: _stratified_outcomes(
            rows[0],
            rows[1],
            stratum_by_key=event_ordinal_strata,
            scope=f"{scope}_same_second_ordinal",
        )
        for scope, rows in scope_rows.items()
    }
    parent_followup = _stratified_outcomes(
        control_parent,
        treatment_parent,
        stratum_by_key=parent_followup_strata,
        scope="parent_same_second_followup",
    )
    parent_multiplicity = _stratified_outcomes(
        control_parent,
        treatment_parent,
        stratum_by_key=parent_multiplicity_strata,
        scope="parent_same_second_multiplicity",
    )
    track_length = _stratified_outcomes(
        control_parent,
        treatment_parent,
        stratum_by_key=track_length_strata,
        scope="parent_track_length",
    )
    changes = _parent_changes(
        control_parent, treatment_parent, events=events, parents=parents
    )
    known_to_clutter_delta = (
        _outcome_counts(treatment_parent)["KnownTarget_to_Clutter"]
        - _outcome_counts(control_parent)["KnownTarget_to_Clutter"]
    )
    clutter_to_known_delta = (
        _outcome_counts(treatment_parent)["Clutter_to_KnownTarget"]
        - _outcome_counts(control_parent)["Clutter_to_KnownTarget"]
    )
    report = {
        "status": "completed_development_paired_only",
        "schema_version": "RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-paired-development-v1",
        "scope": "V24c development KnownGate L1 paired dynamics diagnostic only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "dynamics_variable": {
            "control": "bounded_observation alpha=0.25 at zero_delta",
            "treatment": "strict_skip restores pre-step state after recurrent LayerNorm at zero_delta",
            "first_and_positive_dt_updates_unchanged": True,
            "padding_unchanged": True,
            "legacy_v20_skip_not_reused": True,
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
            "no_new_fusion_columns": True,
            "initial_logits_equal_without_same_second_events": True,
            "strict_state_unchanged_at_same_second_probe_events": True,
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
            "KnownTarget_to_Clutter_delta": int(known_to_clutter_delta),
            "Clutter_to_KnownTarget_delta": int(clutter_to_known_delta),
        },
        "eligible_prefix_safety": safety,
        "same_second_reason_strata_by_scope": same_second_reason_by_scope,
        "same_second_ordinal_strata_by_scope": same_second_ordinal_by_scope,
        "same_second_parent_followup_strata": parent_followup,
        "same_second_parent_multiplicity_strata": parent_multiplicity,
        "track_length_parent_strata": track_length,
        "capture_parent_strata": _capture_strata(
            control_parent, treatment_parent, parents
        ),
        "per_parent_changes": changes,
        "frozen_decision": {
            "same_second_is_a_descriptive_potential_confound_not_an_adoption_rule": True,
            "overall_known_to_clutter_not_increased": known_to_clutter_delta <= 0,
            "clutter_to_known_not_increased": clutter_to_known_delta <= 0,
            "parent_final_known_target_recall_not_lower": (
                safety["by_scope"]["parent_final"]["known_target_recall_delta"] >= 0.0
            ),
            "eligible_prefix_safety_passed": (
                safety["eligible_prefix_known_target_recall_not_lower"]
                and safety["eligible_prefix_clutter_recall_not_lower"]
                and safety[
                    "parent_balanced_eligible_prefix_known_target_recall_not_lower"
                ]
                and safety[
                    "parent_balanced_eligible_prefix_clutter_recall_not_lower"
                ]
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
        description="Evaluate V24c strict same-second skip against the V24 control."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
