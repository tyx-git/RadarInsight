"""Development-only report for V23l's two-backbone temporal comparison.

The report keeps two comparisons deliberately separate:

* fixed epoch 20: official V23l versus self-developed V23g, a direct
  same-epoch descriptive architecture comparison;
* selected checkpoints: V23l only if it meets the frozen V23e selector.

V23g's historical best checkpoint is retained as context, but it did not meet
the V23e feasibility rule and is therefore never presented as an equally
admissible selected counterpart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radar_ln3.known_gate import (
    build_known_gate_dataset,
    evaluate_known_gate,
    known_gate_device,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23e_selection import (
    V23E_SELECTOR_SCHEMA,
    clutter_prefix_action_trajectories,
)
from radar_ln3.known_gate_v23l import (
    V23L_REPORT_SCHEMA,
    V23L_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_config,
    readiness_relative,
    report_relative,
)
from radar_ln3.known_gate_v23g import (
    V23G_VARIANT,
    artifact_relative as v23g_artifact_relative,
)
from report.known_gate_common import _bootstrap, _metric_vector, _ordered_pair
from report.known_gate_v23e_selection import _selection_bootstrap


SCOPES = ("parent", "eligible_prefix", "window")
MIN_STABLE_STRATUM_SUPPORT = 8


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _records(detail: Mapping[str, Any], *, name: str) -> dict[str, list[dict[str, Any]]]:
    rows = {
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in SCOPES
    }
    if len(rows["parent"]) != 232 or len(rows["eligible_prefix"]) != 3756 or len(rows["window"]) != 4452:
        raise ValueError(f"{name} record counts differ from frozen V21e development")
    return rows


def _metrics(rows: Mapping[str, Sequence[Mapping[str, Any]]], *, prefix: str) -> dict[str, Any]:
    return {
        "parent_final": summarize_known_gate_records(
            rows["parent"], scope=f"{prefix}_parent_final"
        ),
        "eligible_prefix": summarize_known_gate_records(
            rows["eligible_prefix"], scope=f"{prefix}_eligible_prefix"
        ),
        "parent_balanced_eligible_prefix": summarize_parent_balanced_eligible_prefix_records(
            rows["eligible_prefix"], scope=f"{prefix}_parent_balanced_eligible_prefix"
        ),
        "window": summarize_known_gate_records(rows["window"], scope=f"{prefix}_window"),
    }


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left, right = _metric_vector(before), _metric_vector(after)
    return {name: float(right[name] - left[name]) for name in left}


def _parent_strata(root: Path, source_config: Any) -> dict[str, Any]:
    """Create the frozen V23k-style length × multiplicity partition from val."""

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
    cells = {
        f"{length_key}__{multiplicity_key}": set()
        for length_key in length_groups
        for multiplicity_key in multiplicity_groups
    }
    details: dict[str, dict[str, Any]] = {}
    for item in dataset:
        parent_id = str(item.base.parent_sample_id)
        length = int(item.base.event_count)
        if 12 <= length <= 16:
            length_key = "short_12_to_16_events"
        elif 17 <= length <= 21:
            length_key = "medium_17_to_21_events"
        elif 22 <= length <= 29:
            length_key = "long_22_to_29_events"
        else:
            raise ValueError(f"V23l validation parent has unexpected length: {length}")
        ordinal, maximum, zero_count = 0, 1, 0
        for reason in item.base.tr_dt_reason.tolist():
            if reason in (1, 3):
                ordinal = 1
            elif reason == DT_REASON_ZERO_DELTA:
                ordinal += 1
                zero_count += 1
                maximum = max(maximum, ordinal)
            else:
                raise ValueError("V23l has unsupported active dt reason")
        multiplicity_key = (
            "no_same_second_followup"
            if zero_count == 0
            else "same_second_maximum_ordinal_2"
            if maximum == 2
            else "same_second_maximum_ordinal_3_or_more"
        )
        length_groups[length_key].add(parent_id)
        multiplicity_groups[multiplicity_key].add(parent_id)
        cells[f"{length_key}__{multiplicity_key}"].add(parent_id)
        details[parent_id] = {
            "label_name": str(item.base.label_name),
            "track_length_events": length,
            "track_length_group": length_key,
            "zero_delta_event_count": zero_count,
            "maximum_causal_same_second_ordinal": maximum,
            "same_second_multiplicity_group": multiplicity_key,
        }
    if sum(len(values) for values in cells.values()) != len(dataset) or set().union(*cells.values()) != set(details):
        raise ValueError("V23l length/multiplicity strata do not partition validation parents")
    clutter = [row for row in details.values() if row["label_name"] == "ClutterTarget"]
    if len(clutter) != 9 or any(row["zero_delta_event_count"] != 0 for row in clutter):
        raise ValueError("V23l development Clutter same-second exposure boundary changed")
    return {
        "length_groups": length_groups,
        "multiplicity_groups": multiplicity_groups,
        "cells": cells,
        "details": details,
        "validation_parent_count": len(dataset),
    }


def _stratum_metrics(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    groups: Mapping[str, set[str]],
    *,
    prefix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parent_ids in groups.items():
        before = [dict(row) for row in control if str(row["parent_sample_id"]) in parent_ids]
        after = [dict(row) for row in treatment if str(row["parent_sample_id"]) in parent_ids]
        if not parent_ids:
            result[name] = {
                "support": 0,
                "minimum_stable_support": MIN_STABLE_STRATUM_SUPPORT,
                "interpretation": "empty stratum; no estimate",
            }
            continue
        before, after = _ordered_pair(before, after, scope="parent")
        before_metrics = summarize_known_gate_records(before, scope=f"{prefix}_{name}_control")
        after_metrics = summarize_known_gate_records(after, scope=f"{prefix}_{name}_treatment")
        result[name] = {
            "support": len(parent_ids),
            "minimum_stable_support": MIN_STABLE_STRATUM_SUPPORT,
            "stable_support": len(parent_ids) >= MIN_STABLE_STRATUM_SUPPORT,
            "base_label_support": {
                label: sum(str(row["label_name"]) == label for row in before)
                for label in ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
            },
            "self_developed_v23g": before_metrics,
            "official_v23l": after_metrics,
            "official_minus_self_developed": _delta(before_metrics, after_metrics),
        }
    return result


def _load_epoch20_rows(
    root: Path, *, source_config: Any, config: Any
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read V23l's saved epoch-20 detail and reproduce V23g epoch 20 exactly."""

    artifact = root / artifact_relative()
    summary = _read_json(artifact / "training-summary.json")
    v23l_detail_path = Path(str(summary["epoch20_sample_detail_information"]).replace("\\", "/"))
    if not v23l_detail_path.is_absolute():
        v23l_detail_path = root / v23l_detail_path
    v23l_detail = _read_json(v23l_detail_path)
    if v23l_detail.get("variant") != V23L_VARIANT or int(v23l_detail.get("epoch", -1)) != 20:
        raise ValueError("V23l epoch-20 detail binding is stale")
    v23l_rows = _records(v23l_detail, name="V23l epoch 20")

    v23g_artifact = root / v23g_artifact_relative()
    v23g_epoch_path = v23g_artifact / "epochs" / "epoch-020.pt"
    frozen_epoch_path = Path(
        str(config.source_v23g["epoch20_checkpoint_path"]).replace("\\", "/")
    )
    if not frozen_epoch_path.is_absolute():
        frozen_epoch_path = root / frozen_epoch_path
    if v23g_epoch_path.resolve() != frozen_epoch_path.resolve() or not v23g_epoch_path.is_file():
        raise FileNotFoundError("V23l requires V23g epoch-020 checkpoint")
    if sha256_file(v23g_epoch_path) != str(config.source_v23g["epoch20_checkpoint_sha256"]):
        raise ValueError("V23l V23g epoch-020 comparator checkpoint binding is stale")
    from radar_ln3.known_gate_v23g import load_config as load_v23g_config
    from train.known_gate_v23g import _build_density_model, _density_config

    wrapper, v23g_source, _source_path = load_v23g_config(root)
    if v23g_source != source_config:
        raise ValueError("V23l and V23g source configurations differ")
    density_config = _density_config(source_config, wrapper)
    model, _pairing = _build_density_model(root, source_config, density_config)
    device = known_gate_device(source_config)
    model.to(device)
    payload = dict(torch.load(v23g_epoch_path, map_location=device, weights_only=False))
    if (
        payload.get("variant") != V23G_VARIANT
        or int(payload.get("epoch", -1)) != 20
        or payload.get("source_control")
        != config.source_v23g["source_v21e_control"]
    ):
        raise ValueError("V23g epoch-020 checkpoint does not match V23l frozen comparator")
    model.load_state_dict(payload["model_state"], strict=True)
    evaluation = evaluate_known_gate(
        model,
        build_known_gate_dataset(root, source_config, split_name="val"),
        config=density_config,
        device=device,
    )
    v23g_rows = {scope: [dict(row) for row in evaluation["records"][scope]] for scope in SCOPES}
    _records({"records": v23g_rows}, name="V23g epoch 20")
    return v23g_rows, v23l_rows, {
        "checkpoint_path": str(v23g_epoch_path),
        "checkpoint_sha256": sha256_file(v23g_epoch_path),
        "epoch": 20,
    }


def _fixed_epoch20_diagnostic(
    v23g_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    v23l_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    control_metrics = _metrics(v23g_rows, prefix="v23g_epoch20")
    treatment_metrics = _metrics(v23l_rows, prefix="v23l_epoch20")
    before, after = _ordered_pair(v23g_rows["parent"], v23l_rows["parent"], scope="parent")
    return {
        "comparison": "fixed same epoch 20 V23l official minus V23g self-developed; descriptive only and never a checkpoint-selection override",
        "self_developed_v23g": control_metrics,
        "official_v23l": treatment_metrics,
        "official_minus_self_developed": {
            scope: _delta(control_metrics[scope], treatment_metrics[scope])
            for scope in ("parent_final", "eligible_prefix", "parent_balanced_eligible_prefix", "window")
        },
        "paired_parent_bootstrap": _bootstrap(before, after),
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, source_config, _source_path = load_config(root)
    artifact = root / artifact_relative()
    summary_path = artifact / "training-summary.json"
    readiness_path = root / readiness_relative()
    if not all(path.is_file() for path in (summary_path, readiness_path)):
        raise FileNotFoundError("V23l training output is incomplete")
    summary = _read_json(summary_path)
    readiness = _read_json(readiness_path)
    if (
        summary.get("variant") != V23L_VARIANT
        or summary.get("architecture")
        != "official_ncps_cfc_external_fixed_retention_density_adapter_course_prefix"
        or summary.get("config_sha256") != sha256_file(root / config_relative())
        or summary.get("config_identity_sha256") != config_identity_sha256(config)
        or len(list(summary.get("history") or ())) != 20
        or summary.get("source_v23g_density") != config.source_v23g
        or summary.get("selector") != config.selector
        or summary.get("model_state_changed_from_initialization") is not True
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or readiness.get("status") != "passed"
        or readiness.get("variant") != V23L_VARIANT
    ):
        raise ValueError("V23l training summary violates its frozen contract")
    v23g_rows, epoch20_rows, v23g_epoch20 = _load_epoch20_rows(
        root, source_config=source_config, config=config
    )
    strata = _parent_strata(root, source_config)
    fixed_epoch20 = _fixed_epoch20_diagnostic(v23g_rows, epoch20_rows)
    mandatory_strata = {
        "comparison": "fixed same epoch 20 V23l official minus V23g self-developed; descriptive only",
        "definition": {
            "track_length": {
                "short_12_to_16_events": "12 <= causal event count <= 16",
                "medium_17_to_21_events": "17 <= causal event count <= 21",
                "long_22_to_29_events": "22 <= causal event count <= 29",
            },
            "same_second_multiplicity": {
                "no_same_second_followup": "zero_delta event count = 0",
                "same_second_maximum_ordinal_2": "at least one zero_delta and maximum causal ordinal = 2",
                "same_second_maximum_ordinal_3_or_more": "maximum causal ordinal >= 3",
            },
            "low_support_policy": f"support < {MIN_STABLE_STRATUM_SUPPORT} is descriptive only",
        },
        "track_length": _stratum_metrics(v23g_rows["parent"], epoch20_rows["parent"], strata["length_groups"], prefix="track_length"),
        "same_second_multiplicity": _stratum_metrics(v23g_rows["parent"], epoch20_rows["parent"], strata["multiplicity_groups"], prefix="same_second_multiplicity"),
        "track_length_by_same_second_multiplicity": _stratum_metrics(v23g_rows["parent"], epoch20_rows["parent"], strata["cells"], prefix="track_length_by_same_second_multiplicity"),
        "parent_details": strata["details"],
        "v23g_epoch20_control_checkpoint": v23g_epoch20,
        "clutter_density_effect_not_estimable": True,
        "reason": "all nine development Clutter parents have no same-second follow-up",
    }
    base = {
        "status": "completed_development_two_backbone_comparison_only",
        "schema_version": V23L_REPORT_SCHEMA,
        "experiment": "V23l official CfC fixed positive-dt retention plus same-second density versus V23g self-developed density",
        "scope": "V21 development KnownGate only; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "source_v23g_density": config.source_v23g,
        "official_adapter": config.adapter.as_dict(),
        "selector": config.selector,
        "training_summary_path": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config_path": str(root / config_relative()),
        "config_sha256": sha256_file(root / config_relative()),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "fixed_epoch20_diagnostic": fixed_epoch20,
        "length_by_same_second_multiplicity_parent_final": mandatory_strata,
        "statistical_boundary": {
            "development_parent_count": 232,
            "clutter_parent_support": 9,
            "parent_bootstrap_is_descriptive_not_new_holdout": True,
            "clutter_same_second_density_effect_is_not_estimable": True,
            "v23g_legacy_best_checkpoint_is_not_v23e_feasible": True,
        },
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    if summary.get("status") == "completed_no_feasible_checkpoint":
        report = {
            **base,
            "selection_status": "no_feasible_checkpoint",
            "best_epoch": None,
            "selected_metrics": None,
            "conclusion": "V23l completed all 20 epochs but no checkpoint satisfied the frozen V23e constraints; only the fixed epoch-20 two-backbone diagnostic is available.",
            "not_final_product_performance": True,
        }
    elif summary.get("status") == "completed":
        detail_path = artifact / "sample_detail_information.json"
        checkpoint_path = artifact / "best_model.pt"
        if not all(path.is_file() for path in (detail_path, checkpoint_path)):
            raise FileNotFoundError("V23l feasible selected checkpoint/detail is missing")
        detail = _read_json(detail_path)
        selector = dict(summary.get("best_checkpoint_selection") or {})
        if (
            detail.get("variant") != V23L_VARIANT
            or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -2))
            or selector.get("schema_version") != V23E_SELECTOR_SCHEMA
            or selector.get("feasible") is not True
        ):
            raise ValueError("V23l selected checkpoint is not V23e-feasible")
        selected_rows = _records(detail, name="V23l selected")
        selected_metrics = _metrics(selected_rows, prefix="v23l_selected")
        report = {
            **base,
            "selection_status": "v23e_feasible_checkpoint_selected",
            "best_epoch": int(summary["best_epoch"]),
            "best_checkpoint_path": str(checkpoint_path),
            "best_checkpoint_sha256": sha256_file(checkpoint_path),
            "selected_metrics": selected_metrics,
            "selector_assessment": selector,
            "clutter_parent_bootstrap": _selection_bootstrap(selected_rows),
            "clutter_prefix_action_trajectories": clutter_prefix_action_trajectories(selected_rows),
            "conclusion": "V23l found a V23e-feasible official-CfC checkpoint. This remains development-only local KnownGate evidence, not a product or cross-source conclusion.",
            "not_final_product_performance": True,
        }
    else:
        raise ValueError(f"V23l training did not complete: {summary.get('status')}")
    output = root / report_relative()
    _write_json(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report V23l official/full-temporal KnownGate comparison.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
