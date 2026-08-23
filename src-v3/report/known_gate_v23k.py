"""Development-only report for V23k causal TR/RD observation intensity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radar_ln3.known_gate import (
    build_known_gate_dataset,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23e_selection import (
    V23E_SELECTOR_SCHEMA,
    clutter_prefix_action_trajectories,
    frozen_v23a_legacy_reference,
)
from radar_ln3.known_gate_v23k import (
    V23K_REPORT_SCHEMA,
    V23K_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_config,
    readiness_relative,
    report_relative,
)
from report.known_gate_v21e import _bootstrap, _metric_vector, _ordered_pair
from report.known_gate_v23e_selection import _selection_bootstrap


SCOPES = ("parent", "eligible_prefix", "window")
MIN_STABLE_STRATUM_SUPPORT = 8


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _records(detail: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in SCOPES
    }
    if len(rows["parent"]) != 232 or len(rows["eligible_prefix"]) != 3756 or len(rows["window"]) != 4452:
        raise ValueError("V23k detail record counts differ from frozen V21 scope")
    return rows


def _metrics(rows: Mapping[str, Sequence[Mapping[str, Any]]], *, prefix: str) -> dict[str, Any]:
    return {
        "parent_final": summarize_known_gate_records(rows["parent"], scope=f"{prefix}_parent_final"),
        "eligible_prefix": summarize_known_gate_records(rows["eligible_prefix"], scope=f"{prefix}_eligible_prefix"),
        "parent_balanced_eligible_prefix": summarize_parent_balanced_eligible_prefix_records(rows["eligible_prefix"], scope=f"{prefix}_parent_balanced_eligible_prefix"),
        "window": summarize_known_gate_records(rows["window"], scope=f"{prefix}_window"),
    }


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left, right = _metric_vector(before), _metric_vector(after)
    return {key: float(right[key] - left[key]) for key in left}


def _parent_strata(root: Path, source_config: Any) -> dict[str, Any]:
    """Make the pre-registered length × multiplicity partition from val only."""

    from radar_ln3.data import DT_REASON_ZERO_DELTA

    dataset = build_known_gate_dataset(root, source_config, split_name="val")
    length_groups = {
        "short_12_to_16_events": set(),
        "medium_17_to_21_events": set(),
        "long_22_to_29_events": set(),
    }
    multiplicity_groups = {
        "no_same_second_followup": set(),
        "same_second_maximum_ordinal_2": set(),
        "same_second_maximum_ordinal_3_or_more": set(),
    }
    cells: dict[str, set[str]] = {
        f"{length_key}__{multiplicity_key}": set()
        for length_key in length_groups
        for multiplicity_key in multiplicity_groups
    }
    details: dict[str, dict[str, Any]] = {}
    for item in dataset:
        parent_id = str(item.base.parent_sample_id)
        length = int(item.base.event_count)
        if 12 <= length <= 16:
            length_group = "short_12_to_16_events"
        elif 17 <= length <= 21:
            length_group = "medium_17_to_21_events"
        elif 22 <= length <= 29:
            length_group = "long_22_to_29_events"
        else:
            raise ValueError(f"V23k validation parent has unexpected length: {length}")
        ordinal = 0
        maximum = 1
        zero_count = 0
        for reason in item.base.tr_dt_reason.tolist():
            if reason in (1, 3):
                ordinal = 1
            elif reason == DT_REASON_ZERO_DELTA:
                ordinal += 1
                zero_count += 1
                maximum = max(maximum, ordinal)
            else:
                raise ValueError(f"V23k unexpected active dt reason: {reason}")
        multiplicity_group = (
            "no_same_second_followup"
            if zero_count == 0
            else "same_second_maximum_ordinal_2"
            if maximum == 2
            else "same_second_maximum_ordinal_3_or_more"
        )
        length_groups[length_group].add(parent_id)
        multiplicity_groups[multiplicity_group].add(parent_id)
        cells[f"{length_group}__{multiplicity_group}"].add(parent_id)
        details[parent_id] = {
            "label_name": str(item.base.label_name),
            "track_length_events": length,
            "track_length_group": length_group,
            "zero_delta_event_count": zero_count,
            "maximum_causal_same_second_ordinal": maximum,
            "same_second_multiplicity_group": multiplicity_group,
        }
    if sum(len(values) for values in cells.values()) != len(dataset):
        raise ValueError("V23k length/multiplicity cells do not partition validation parents")
    if set().union(*cells.values()) != set(details):
        raise ValueError("V23k length/multiplicity cells omit or duplicate validation parents")
    return {
        "length_groups": length_groups,
        "multiplicity_groups": multiplicity_groups,
        "cells": cells,
        "details": details,
        "validation_parent_count": len(dataset),
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
    }


def _rows_for_ids(rows: Sequence[Mapping[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows if str(row["parent_sample_id"]) in ids]
    if len(result) != len(ids):
        raise ValueError("V23k stratum parent records are incomplete")
    return result


def _stratum_metrics(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    groups: Mapping[str, set[str]],
    *,
    prefix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
    for name, ids in groups.items():
        if not ids:
            # The frozen validation partition has two empty cross-cells.  They
            # are part of the requested Cartesian table, but do not constitute
            # an estimate: avoid inventing zero-valued performance or trying to
            # bootstrap an absent population.
            result[name] = {
                "support": 0,
                "minimum_stable_support": MIN_STABLE_STRATUM_SUPPORT,
                "stable_estimate": False,
                "comparison_status": "not_applicable_empty_stratum",
                "base_label_support": {label: 0 for label in labels},
                "v23a_fixed_retention_control": None,
                "v23k_observation_intensity_treatment": None,
                "v23k_minus_v23a": None,
                "warning": "empty descriptive stratum; no estimate or comparison is available",
            }
            continue
        before, after = _ordered_pair(_rows_for_ids(control, ids), _rows_for_ids(treatment, ids), scope="parent")
        before_metrics = summarize_known_gate_records(before, scope=f"v23a_{prefix}_{name}")
        after_metrics = summarize_known_gate_records(after, scope=f"v23k_{prefix}_{name}")
        support_by_label = {label: sum(str(row["label_name"]) == label for row in before) for label in labels}
        result[name] = {
            "support": len(before),
            "minimum_stable_support": MIN_STABLE_STRATUM_SUPPORT,
            "stable_estimate": len(before) >= MIN_STABLE_STRATUM_SUPPORT,
            "comparison_status": "descriptive",
            "base_label_support": support_by_label,
            "v23a_fixed_retention_control": before_metrics,
            "v23k_observation_intensity_treatment": after_metrics,
            "v23k_minus_v23a": _delta(before_metrics, after_metrics),
            "warning": None if len(before) >= MIN_STABLE_STRATUM_SUPPORT else "low-support descriptive stratum; do not infer a stable effect",
        }
    return result


def _load_v23a_epoch20_rows(
    root: Path,
    *,
    source_v23a: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load V23a's fixed epoch-20 diagnostic, never its selected checkpoint."""

    import torch

    legacy_best = Path(
        str(source_v23a["legacy_best_checkpoint_path"]).replace("\\", "/")
    )
    if legacy_best.name != "best_model.pt":
        raise ValueError("V23k V23a control does not declare a canonical best checkpoint path")
    legacy_best = legacy_best if legacy_best.is_absolute() else root / legacy_best
    path = (legacy_best.parent / "epochs" / "epoch-020.pt").resolve()
    if not path.is_file():
        raise FileNotFoundError("V23k requires frozen V23a epoch-020 checkpoint for same-epoch strata")
    from radar_ln3.known_gate_official_cfc import build_official_cfc_model_from_frozen_source
    from radar_ln3.known_gate import evaluate_known_gate, known_gate_device
    from radar_ln3.known_gate_v23_adapters import load_adapter_config

    control_config, source_config, _source_path = load_adapter_config(root, "official_cfc_positive_dt_fixed_retention")
    model, _audit = build_official_cfc_model_from_frozen_source(
        root, source_config, state_adapter_spec=control_config.adapter
    )
    device = known_gate_device(source_config)
    model.to(device)
    payload = dict(torch.load(path, map_location=device, weights_only=False))
    if (
        payload.get("checkpoint_schema")
        != "RADAR-LN3-V23a-known-gate-official-cfc-adapter-seed43-checkpoint-v1"
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != "official_cfc_positive_dt_fixed_retention"
        or payload.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or int(payload.get("epoch", -1)) != 20
        or payload.get("config_name") != control_config.config_name
        or payload.get("config_identity_sha256") != source_v23a["config_identity_sha256"]
        or payload.get("config_file_sha256") != source_v23a["config_sha256"]
        or payload.get("readiness_sha256") != source_v23a["readiness_sha256"]
        or payload.get("external_state_adapter") != model.external_state_adapter_specification()
        or payload.get("official_cfc") != model.backbone_specification()
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V23a epoch-020 checkpoint differs from frozen fixed-retention control")
    model.load_state_dict(payload["model_state"], strict=True)
    dataset = build_known_gate_dataset(root, source_config, split_name="val")
    evaluation = evaluate_known_gate(model, dataset, config=source_config, device=device)
    return (
        {
            scope: [dict(row) for row in evaluation["records"][scope]]
            for scope in SCOPES
        },
        {
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "epoch": 20,
            "binding": "derived from the V23a control artifact declared in the V23k frozen config; checkpoint payload is verified against that config and readiness binding",
        },
    )


def _fixed_epoch20_diagnostic(
    control_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    treatment_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Describe the pre-registered same-epoch comparison without selection."""

    control_metrics = _metrics(control_rows, prefix="v23a_epoch20")
    treatment_metrics = _metrics(treatment_rows, prefix="v23k_epoch20")
    parent_before, parent_after = _ordered_pair(
        control_rows["parent"], treatment_rows["parent"], scope="parent"
    )
    return {
        "comparison": "fixed same epoch 20 V23k minus frozen V23a; descriptive only and never a checkpoint-selection override",
        "v23a_fixed_retention_control": control_metrics,
        "v23k_observation_intensity_treatment": treatment_metrics,
        "v23k_minus_v23a": {
            scope: _delta(control_metrics[scope], treatment_metrics[scope])
            for scope in (
                "parent_final",
                "eligible_prefix",
                "parent_balanced_eligible_prefix",
                "window",
            )
        },
        "paired_parent_bootstrap": _bootstrap(parent_before, parent_after),
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, source_config, _source_path = load_config(root)
    config_path = root / config_relative()
    readiness_path = root / readiness_relative()
    artifact = root / artifact_relative()
    summary_path = artifact / "training-summary.json"
    if not all(path.is_file() for path in (config_path, readiness_path, summary_path)):
        raise FileNotFoundError("V23k training output is incomplete")
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    readiness = dict(json.loads(readiness_path.read_text(encoding="utf-8")))
    if (
        summary.get("variant") != V23K_VARIANT
        or summary.get("architecture") != "official_ncps_cfc_external_state_adapter_causal_observation_intensity"
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != config_identity_sha256(config)
        or len(list(summary.get("history") or ())) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or dict(summary.get("observation_intensity") or {}) != config.observation_intensity.as_dict()
        or dict(summary.get("selector") or {}) != config.selector
        or dict(summary.get("training_readiness") or {}).get("sha256") != sha256_file(readiness_path)
        or readiness.get("status") != "passed"
        or readiness.get("variant") != V23K_VARIANT
    ):
        raise ValueError("V23k training summary violates the prepared contract")
    base = {
        "status": "completed_development_observation_intensity_only",
        "schema_version": V23K_REPORT_SCHEMA,
        "experiment": "V23k V23a official-CfC causal log1p decayed TR/RD observation-intensity inputs only",
        "scope": "V21 development KnownGate only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "source_v23a": config.source_v23a,
        "observation_intensity": config.observation_intensity.as_dict(),
        "selector": config.selector,
        "training_summary_path": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    epoch20_detail_value = str(summary.get("epoch20_sample_detail_information") or "")
    epoch20_detail_path = Path(epoch20_detail_value.replace("\\", "/"))
    if not epoch20_detail_path.is_absolute():
        epoch20_detail_path = root / epoch20_detail_path
    if not epoch20_detail_path.is_file():
        raise FileNotFoundError("V23k fixed epoch-20 diagnostic detail is missing")
    epoch20_detail = dict(json.loads(epoch20_detail_path.read_text(encoding="utf-8")))
    if (
        epoch20_detail.get("variant") != V23K_VARIANT
        or int(epoch20_detail.get("epoch", -1)) != 20
        or dict(epoch20_detail.get("observation_intensity") or {})
        != config.observation_intensity.as_dict()
    ):
        raise ValueError("V23k fixed epoch-20 diagnostic detail binding is stale")
    epoch20_rows = _records(epoch20_detail)
    v23a_epoch20_rows, v23a_epoch20_control = _load_v23a_epoch20_rows(
        root,
        source_v23a=config.source_v23a,
    )
    epoch20_before, epoch20_after = _ordered_pair(
        v23a_epoch20_rows["parent"], epoch20_rows["parent"], scope="parent"
    )
    strata = _parent_strata(root, source_config)
    fixed_epoch20 = _fixed_epoch20_diagnostic(v23a_epoch20_rows, epoch20_rows)
    mandatory_strata = {
        "comparison": "fixed same epoch 20 V23k minus frozen V23a; independent of selection and never a best-checkpoint substitution",
        "definition": {
            "track_length": {
                "short_12_to_16_events": "12 <= causal event count <= 16",
                "medium_17_to_21_events": "17 <= causal event count <= 21",
                "long_22_to_29_events": "22 <= causal event count <= 29",
            },
            "same_second_multiplicity": {
                "no_same_second_followup": "zero_delta event count = 0",
                "same_second_maximum_ordinal_2": "at least one zero_delta and maximum causal same-second ordinal = 2",
                "same_second_maximum_ordinal_3_or_more": "maximum causal same-second ordinal >= 3",
            },
            "low_support_policy": f"support < {MIN_STABLE_STRATUM_SUPPORT} is descriptive only and flagged unstable",
        },
        "track_length": _stratum_metrics(epoch20_before, epoch20_after, strata["length_groups"], prefix="track_length"),
        "same_second_multiplicity": _stratum_metrics(epoch20_before, epoch20_after, strata["multiplicity_groups"], prefix="same_second_multiplicity"),
        "track_length_by_same_second_multiplicity": _stratum_metrics(epoch20_before, epoch20_after, strata["cells"], prefix="track_length_by_same_second_multiplicity"),
        "parent_details": strata["details"],
        "v23a_epoch20_control_checkpoint": v23a_epoch20_control,
        "validation_parent_count": strata["validation_parent_count"],
        "unknown_dataset_access_count": strata["unknown_dataset_access_count"],
        "test_dataset_access_count": strata["test_dataset_access_count"],
    }
    if summary.get("status") == "completed_no_feasible_checkpoint":
        report = {
            **base,
            "selection_status": "no_feasible_checkpoint",
            "best_epoch": None,
            "metrics": None,
            "fixed_epoch20_diagnostic": fixed_epoch20,
            "length_by_same_second_multiplicity_parent_final": mandatory_strata,
            "conclusion": "V23k completed all 20 epochs but no checkpoint met the frozen V23e feasibility constraints; causal observation intensity did not create an admissible development checkpoint.",
            "not_final_product_performance": True,
        }
    elif summary.get("status") == "completed":
        checkpoint_path = artifact / "best_model.pt"
        detail_path = artifact / "sample_detail_information.json"
        if not all(path.is_file() for path in (checkpoint_path, detail_path)):
            raise FileNotFoundError("V23k selected checkpoint/detail is missing")
        detail = dict(json.loads(detail_path.read_text(encoding="utf-8")))
        if (
            int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -2))
            or Path(str(detail.get("checkpoint", ""))).resolve() != checkpoint_path.resolve()
            or detail.get("variant") != V23K_VARIANT
            or dict(detail.get("observation_intensity") or {}) != config.observation_intensity.as_dict()
        ):
            raise ValueError("V23k selected detail binding is stale")
        rows = _records(detail)
        metrics = _metrics(rows, prefix="v23k")
        selector = dict(summary.get("best_checkpoint_selection") or {})
        if selector.get("schema_version") != V23E_SELECTOR_SCHEMA or selector.get("feasible") is not True:
            raise ValueError("V23k best checkpoint is not feasible under V23e selector")
        v23a_artifact = root / "artifacts/RADAR-LN3-V23a-cq08-known-gate-official-cfc-positive-dt-fixed-retention-seed43-formal"
        v23a_detail_path = v23a_artifact / "sample_detail_information.json"
        if not v23a_detail_path.is_file():
            raise FileNotFoundError("V23k direct V23a sample detail is missing")
        v23a_rows = _records(dict(json.loads(v23a_detail_path.read_text(encoding="utf-8"))))
        v23a_metrics = _metrics(v23a_rows, prefix="v23a")
        deltas = {
            scope: _delta(v23a_metrics[scope], metrics[scope])
            for scope in ("parent_final", "eligible_prefix", "parent_balanced_eligible_prefix", "window")
        }
        parent_before, parent_after = _ordered_pair(v23a_rows["parent"], rows["parent"], scope="parent")
        report = {
            **base,
            "selection_status": "feasible_checkpoint_selected",
            "best_epoch": int(summary["best_epoch"]),
            "best_checkpoint_path": str(checkpoint_path),
            "best_checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics": {"v23a_control": v23a_metrics, "v23k_treatment": metrics, "v23k_minus_v23a": deltas},
            "fixed_epoch20_diagnostic": fixed_epoch20,
            "selector_assessment": selector,
            "paired_parent_bootstrap": _bootstrap(parent_before, parent_after),
            "clutter_parent_bootstrap": _selection_bootstrap(rows),
            "clutter_prefix_action_trajectories": clutter_prefix_action_trajectories(rows),
            "length_by_same_second_multiplicity_parent_final": mandatory_strata,
            "conclusion": "V23k found a V23e-feasible checkpoint. This is development-only local KnownGate evidence; the mandatory length × same-second-multiplicity table is descriptive and does not establish a product or cross-source claim.",
            "not_final_product_performance": True,
        }
    else:
        raise ValueError(f"V23k training did not complete: {summary.get('status')}")
    output = root / report_relative()
    _write_json(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report completed V23k causal observation-intensity training.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
