"""Train-only readiness and zero-delta shortcut audit for V20/V20r."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar_ln3.task_models import (
    LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
    V20_DENSITY_PAIRED_PROTOCOL,
    V20_SKIP_PAIRED_PROTOCOL,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)


VARIANTS = {
    "skip": {
        "label": "V20-skip",
        "protocol": V20_SKIP_PAIRED_PROTOCOL,
        "slug": "zero-delta-skip",
    },
    "density": {
        "label": "V20r-density",
        "protocol": V20_DENSITY_PAIRED_PROTOCOL,
        "slug": "density-weighted",
    },
}
DBB_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
SOURCE_INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
FIRST_REASON = "first_event_no_predecessor"
ZERO_REASON = "zero_delta_at_second_precision"
POSITIVE_REASON = "positive_delta_at_second_precision"


def _cramers_v(labels: np.ndarray, categories: np.ndarray) -> float:
    label_values, label_ids = np.unique(np.asarray(labels, dtype=np.str_), return_inverse=True)
    category_values, category_ids = np.unique(
        np.asarray(categories, dtype=np.str_), return_inverse=True
    )
    if len(label_values) < 2 or len(category_values) < 2:
        return 0.0
    table = np.zeros((len(label_values), len(category_values)), dtype=np.float64)
    np.add.at(table, (label_ids, category_ids), 1.0)
    total = float(table.sum())
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total
    chi_square = float(np.sum(np.square(table - expected) / np.maximum(expected, 1.0e-12)))
    denominator = total * float(min(table.shape[0] - 1, table.shape[1] - 1))
    return float(np.sqrt(max(0.0, chi_square) / denominator))


def _config_paths(variant: str) -> dict[str, Path]:
    prefix = (
        "RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip"
        if variant == "skip"
        else "RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted"
    )
    return {
        arm: Path(f"config/RADAR-LN3/{prefix}-{arm}-seed42-config.json")
        for arm in ("control", "treatment")
    }


def _training_core(config: Any) -> dict[str, Any]:
    excluded = {
        "checkpoint_path",
        "checkpoint_schema",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "paired_protocol",
        "paired_arm",
        "paired_counterpart_config_path",
    }
    return {
        key: value
        for key, value in config.train.__dict__.items()
        if key not in excluded
    }


def _density_shared_control_binding(root: Path, density_control: Any) -> dict[str, Any]:
    config_path = root / _config_paths("skip")["control"]
    config = load_task_config(config_path)
    if (
        config.task != density_control.task
        or config.data != density_control.data
        or config.model != density_control.model
        or _training_core(config) != _training_core(density_control)
        or config.train.paired_protocol != V20_SKIP_PAIRED_PROTOCOL
        or config.train.paired_arm != "control"
    ):
        raise ValueError(
            "V20r density Control is not exactly equivalent to the completed V20 skip Control"
        )
    artifact = root / config.train.formal_artifact_dir
    summary_path = artifact / "training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "V20r requires the completed shared V20 skip Control summary: "
            f"{summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readiness = dict(summary.get("v20_training_readiness") or {})
    history_epochs = {int(row.get("epoch", -1)) for row in summary.get("history") or []}
    readiness_path = Path(str(readiness.get("path", "")))
    epoch_dir = root / str(config.train.epoch_checkpoint_dir)
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "dbb_specialist"
        or summary.get("stop_reason") != "epoch_limit"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != "RADAR-LN3-V20-zero-delta-skip-training-readiness-v1"
        or readiness.get("paired_protocol") != V20_SKIP_PAIRED_PROTOCOL
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != readiness.get("sha256")
        or history_epochs != set(range(1, 21))
        or not (root / config.train.checkpoint_path).is_file()
        or not (root / config.train.last_checkpoint_path).is_file()
        or any(
            not (epoch_dir / f"epoch-{epoch:03d}.pt").is_file()
            for epoch in range(1, 21)
        )
    ):
        raise ValueError(
            "V20r shared V20 skip Control is incomplete or violates sealed-test scope"
        )
    return {
        "source_experiment": "V20-skip",
        "reuse_role": "V20r-density shared Control; not retrained",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint": str((root / config.train.checkpoint_path).resolve()),
        "best_checkpoint_sha256": sha256_file(root / config.train.checkpoint_path),
        "best_epoch": int(summary.get("best_epoch", -1)),
        "paired_protocol": V20_SKIP_PAIRED_PROTOCOL,
        "validation_metrics_used_for_readiness_gate": False,
        "shared_control_retrained": False,
    }


def _with_density_ordinal(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    ordered = value.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    )
    ordinals = pd.Series(0, index=value.index, dtype=np.int64)
    for parent_id, group in ordered.groupby("parent_sample_id", sort=False):
        event_indices = group["causal_event_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(event_indices, np.arange(len(group), dtype=np.int64)):
            raise ValueError(
                f"V20r parent has a non-causal event index sequence: {parent_id}"
            )
        current = 0
        for row_index, reason in zip(
            group.index, group["tr_dt_reason"].astype(str), strict=True
        ):
            if reason == FIRST_REASON:
                if current != 0:
                    raise ValueError(f"V20r parent repeats first-event reason: {parent_id}")
                current = 1
            elif reason == POSITIVE_REASON:
                if current < 1:
                    raise ValueError(f"V20r parent starts without first event: {parent_id}")
                current = 1
            elif reason == ZERO_REASON:
                if current < 1:
                    raise ValueError(f"V20r zero-delta precedes first event: {parent_id}")
                current += 1
            else:
                raise ValueError(
                    f"V20r active event has an unsupported dt reason: {reason}"
                )
            ordinals.loc[row_index] = current
    if bool((ordinals <= 0).any()):
        raise ValueError("V20r causal density ordinal must be positive for every event")
    value["density_ordinal"] = ordinals
    value["density_bucket"] = np.select(
        [
            value["density_ordinal"].eq(1),
            value["density_ordinal"].eq(2),
            value["density_ordinal"].ge(3),
        ],
        ["k1_control_unchanged", "k2_alpha_0_50", "k3plus_alpha_0_75"],
        default="invalid",
    )
    if bool(value["density_bucket"].astype(str).eq("invalid").any()):
        raise ValueError("V20r density ordinal bucket derivation failed")
    return value


def _parent_summary(frame: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    frame = frame[frame["label_name"].astype(str).isin(DBB_LABELS)].copy()
    frame["zero_delta"] = frame["tr_dt_reason"].astype(str).eq(ZERO_REASON)
    frame["raw_zero_dt"] = frame["tr_dt_is_zero"].astype(bool)
    for label in DBB_LABELS:
        selected = frame[frame["label_name"].astype(str).eq(label)]
        parent = selected.groupby("parent_sample_id", sort=False)
        zero_fractions = parent["zero_delta"].mean().to_numpy(dtype=np.float64)
        raw_zero_fractions = parent["raw_zero_dt"].mean().to_numpy(dtype=np.float64)
        density_exposure = selected["density_ordinal"].ge(2)
        density_exposure_fractions = parent["density_ordinal"].apply(
            lambda values: float(values.ge(2).mean())
        ).to_numpy(dtype=np.float64)
        k2_count = int(selected["density_ordinal"].eq(2).sum())
        k3plus_count = int(selected["density_ordinal"].ge(3).sum())
        exposed_count = k2_count + k3plus_count
        rows[label] = {
            "parent_count": int(selected["parent_sample_id"].nunique()),
            "event_count": int(len(selected)),
            "zero_delta_event_count": int(selected["zero_delta"].sum()),
            "zero_delta_event_fraction": float(selected["zero_delta"].mean())
            if len(selected)
            else 0.0,
            "parent_balanced_zero_delta_fraction_mean": float(zero_fractions.mean())
            if len(zero_fractions)
            else 0.0,
            "parent_balanced_zero_delta_fraction_median": float(np.median(zero_fractions))
            if len(zero_fractions)
            else 0.0,
            "raw_zero_dt_event_fraction_including_first_event": float(selected["raw_zero_dt"].mean())
            if len(selected)
            else 0.0,
            "parent_balanced_raw_zero_dt_fraction_mean": float(raw_zero_fractions.mean())
            if len(raw_zero_fractions)
            else 0.0,
            "density_k1_control_unchanged_event_count": int(
                selected["density_ordinal"].eq(1).sum()
            ),
            "density_k2_alpha_0_50_event_count": k2_count,
            "density_k3plus_alpha_0_75_event_count": k3plus_count,
            "density_treatment_exposed_event_count": exposed_count,
            "density_treatment_exposed_event_fraction": float(density_exposure.mean())
            if len(selected)
            else 0.0,
            "parent_balanced_density_exposure_fraction_mean": float(
                density_exposure_fractions.mean()
            )
            if len(density_exposure_fractions)
            else 0.0,
            "mean_density_gain_on_exposed_events": (
                float((0.50 * k2_count + 0.75 * k3plus_count) / exposed_count)
                if exposed_count
                else None
            ),
        }
    return rows


def _split_report(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = frame[
        frame["split_name"].astype(str).eq(split)
        & frame["label_name"].astype(str).isin(DBB_LABELS)
    ].copy()
    selected = _with_density_ordinal(selected)
    selected["zero_delta"] = selected["tr_dt_reason"].astype(str).eq(ZERO_REASON)
    selected["raw_zero_dt"] = selected["tr_dt_is_zero"].astype(bool)
    labels = selected["label_name"].astype(str).to_numpy()
    zero = selected["zero_delta"].astype(str).to_numpy()
    raw_zero = selected["raw_zero_dt"].astype(str).to_numpy()
    density_bucket = selected["density_bucket"].astype(str).to_numpy()
    summary = _parent_summary(selected)
    return {
        "split": split,
        "parent_count": int(selected["parent_sample_id"].nunique()),
        "event_count": int(len(selected)),
        "class_parent_counts": {
            label: int(summary[label]["parent_count"]) for label in DBB_LABELS
        },
        "class_event_counts": {
            label: int(summary[label]["event_count"]) for label in DBB_LABELS
        },
        "by_label": summary,
        "zero_delta_definition": "tr_dt_reason == zero_delta_at_second_precision; first_event_no_predecessor excluded",
        "raw_zero_dt_definition": "tr_dt_is_zero; includes first_event_no_predecessor",
        "density_ordinal_definition": (
            "first/positive reset k=1; zero_delta increments k; Treatment differs only "
            "at k=2 (alpha=0.50) and k>=3 (alpha=0.75); padding excluded"
        ),
        "density_ordinal_event_counts": {
            bucket: int(selected["density_bucket"].astype(str).eq(bucket).sum())
            for bucket in (
                "k1_control_unchanged",
                "k2_alpha_0_50",
                "k3plus_alpha_0_75",
            )
        },
        "zero_delta_label_cramers_v_event_level": _cramers_v(labels, zero),
        "raw_zero_dt_label_cramers_v_event_level": _cramers_v(labels, raw_zero),
        "density_ordinal_label_cramers_v_event_level": _cramers_v(
            labels, density_bucket
        ),
    }


def run(project_root: Path, variant: str = "skip") -> dict[str, Any]:
    root = project_root.resolve()
    if variant not in VARIANTS:
        raise ValueError(f"unsupported V20 readiness variant: {variant}")
    configs = _config_paths(variant)
    loaded = {arm: load_task_config(root / path) for arm, path in configs.items()}
    contracts = {arm: verify_task_data_contract(root, config) for arm, config in loaded.items()}
    protocol = VARIANTS[variant]["protocol"]
    if any(
        config.schema_version != LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION
        or config.task.name != "dbb_specialist"
        or config.train.paired_protocol != protocol
        or config.train.seed != 42
        for config in loaded.values()
    ):
        raise ValueError("V20 readiness config pair violates the frozen protocol")
    expected_modes = {
        "control": "bounded_observation",
        "treatment": "skip" if variant == "skip" else "density_weighted",
    }
    if any(
        loaded[arm].model.liquid_invalid_dt_update_mode != expected_modes[arm]
        or loaded[arm].model.liquid_invalid_dt_observation_gain != 0.25
        for arm in expected_modes
    ):
        raise ValueError("V20 readiness config pair has invalid dynamics modes or gain")
    if contracts["control"]["hashes"] != contracts["treatment"]["hashes"]:
        raise ValueError("V20 Control/Treatment data contract hashes differ")
    index_path = root / SOURCE_INDEX
    frame = pd.read_csv(
        index_path,
        usecols=[
            "parent_sample_id",
            "split_name",
            "label_name",
            "causal_event_index",
            "tr_dt_is_zero",
            "tr_dt_reason",
        ],
        low_memory=False,
    )
    reports = {split: _split_report(frame, split) for split in ("train", "val")}
    shared_control = (
        _density_shared_control_binding(root, loaded["control"])
        if variant == "density"
        else None
    )
    checks = {
        "paired_configs_share_one_frozen_v16_data_contract": True,
        "dbb_train_and_validation_support_are_positive": all(
            reports[split]["parent_count"] > 0
            and all(reports[split]["class_parent_counts"][label] > 0 for label in DBB_LABELS)
            for split in reports
        ),
        "zero_delta_reason_statistics_are_finite": all(
            np.isfinite(reports[split]["zero_delta_label_cramers_v_event_level"])
            and np.isfinite(reports[split]["raw_zero_dt_label_cramers_v_event_level"])
            and np.isfinite(
                reports[split]["density_ordinal_label_cramers_v_event_level"]
            )
            for split in reports
        ),
        "density_ordinal_is_causal_parent_local_and_padding_free": True,
        "density_exposure_statistics_are_report_only_not_model_inputs": True,
        "density_shared_v20_control_is_complete_and_test_sealed": (
            shared_control is not None if variant == "density" else True
        ),
        "zero_delta_association_is_report_only_not_a_training_gate": True,
        "test_payload_access_zero": True,
    }
    output = root / (
        "src-v3/processed/report/"
        f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-training-readiness.json"
    )
    result = {
        "status": "passed" if all(checks.values()) else "blocked",
        "schema_version": f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-training-readiness-v1",
        "experiment": VARIANTS[variant]["label"],
        "paired_protocol": protocol,
        "paired_configs": {
            arm: {
                "path": str((root / path).resolve()),
                "sha256": sha256_file(root / path),
                "mode": loaded[arm].model.liquid_invalid_dt_update_mode,
            }
            for arm, path in configs.items()
        },
        "contract_hashes": contracts["control"]["hashes"],
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "split_reports": reports,
        "shared_control_reuse": shared_control,
        "checks": checks,
        "interpretation": (
            "zero-delta, density ordinal, and raw-zero associations are diagnostic only; "
            "they do not alter sampling, labels, thresholds, or model inputs"
        ),
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training_at_report_generation": "未执行",
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    formal_artifact_nonempty = any(
        (root / config.train.formal_artifact_dir).is_dir()
        and any((root / config.train.formal_artifact_dir).iterdir())
        for config in loaded.values()
    )
    if formal_artifact_nonempty:
        if not output.exists():
            raise FileNotFoundError(
                f"V20 readiness is missing after formal artifacts exist: {output}"
            )
        if output.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to change V20 readiness after formal artifacts exist: {output}"
            )
    else:
        output.write_text(text, encoding="utf-8")
    result["report_path"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V20 train-only zero-delta readiness report.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="skip")
    args = parser.parse_args()
    run(args.project_root, args.variant)


if __name__ == "__main__":
    main()
