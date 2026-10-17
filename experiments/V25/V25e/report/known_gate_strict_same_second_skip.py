"""Paired V24-versus-V24c development report for strict same-second skipping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate import sha256_file, summarize_known_gate_records, summarize_parent_balanced_eligible_prefix_records


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 24043
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
EXPECTED_SUPPORT = {
    "DroneTarget": 113,
    "BirdTarget": 56,
    "BalloonTarget": 54,
    "ClutterTarget": 9,
}


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





