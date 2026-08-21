"""Minimal runtime-compatibility audit for V23's unmodified official CfC.

This probe intentionally uses only deterministic random tensors.  It does not
open Radar-LN3 data, checkpoints, validation, test, or OOF payloads.  Its job
is to persist the upstream API constraints that force V23 to use a thin
per-parent sequence adapter rather than silently changing time semantics.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.known_gate_v23 import (
    RUNTIME_COMPATIBILITY_REPORT_RELATIVE,
    RUNTIME_COMPATIBILITY_REQUIRED_CHECKS,
    RUNTIME_COMPATIBILITY_REPORT_SCHEMA,
)
from radar_ln3.official_cfc import (
    OFFICIAL_CFC_ACTIVATION,
    OFFICIAL_CFC_CLASS_MODULE,
    OFFICIAL_CFC_DIRECT_URL,
    OFFICIAL_CFC_DISTRIBUTION_VERSION,
    OFFICIAL_CFC_GIT_COMMIT,
    OFFICIAL_CFC_MODE,
    official_cfc_runtime_binding,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dense_cfc() -> torch.nn.Module:
    from ncps.torch import CfC

    return CfC(
        48,
        48,
        proj_size=None,
        return_sequences=True,
        batch_first=True,
        mixed_memory=False,
        mode=OFFICIAL_CFC_MODE,
        activation=OFFICIAL_CFC_ACTIVATION,
        backbone_units=48,
        backbone_layers=1,
        backbone_dropout=0.0,
    ).eval()


def _captures(callable_: Any) -> dict[str, Any]:
    try:
        callable_()
    except Exception as error:  # expected upstream compatibility boundary
        return {"accepted": False, "error_type": type(error).__name__, "error": str(error)}
    return {"accepted": True, "error_type": None, "error": None}


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    from ncps.torch import CfC, LTC

    torch.manual_seed(20260821)
    runtime = official_cfc_runtime_binding()
    x = torch.randn(2, 3, 48)
    per_parent_timespans = torch.tensor([[0.0, 0.5, 2.0], [0.0, 1.0, 3.0]])
    cfc = _dense_cfc()
    heterogeneous = _captures(lambda: cfc(x, timespans=per_parent_timespans))

    x_one = torch.randn(1, 1, 48)
    state_before = torch.randn(1, 48)
    zero_output, zero_state = cfc(
        x_one, hx=state_before.clone(), timespans=torch.zeros(1, 1)
    )
    zero_dt = {
        "accepted": True,
        "output_finite": bool(torch.isfinite(zero_output).all()),
        "state_finite": bool(torch.isfinite(zero_state).all()),
        "state_equal_to_input_state": bool(torch.equal(zero_state, state_before)),
        "max_abs_state_change": float((zero_state - state_before).abs().max()),
    }
    padding = _captures(
        lambda: cfc(
            x_one,
            timespans=torch.zeros(1, 1),
            padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
    )
    dense = {
        "units_argument_is_integer": isinstance(cfc.wiring_or_units, int),
        "wired_false": bool(getattr(cfc, "wired_false", False)),
        "has_wiring_attribute": hasattr(cfc, "wiring"),
        "class_module": cfc.__class__.__module__,
    }
    ltc = LTC(48, 48, return_sequences=True, batch_first=True, mixed_memory=False).eval()
    ltc_output, ltc_state = ltc(
        x_one, hx=state_before.clone(), timespans=torch.zeros(1, 1)
    )
    ltc_zero_dt = {
        "accepted": True,
        "output_finite": bool(torch.isfinite(ltc_output).all()),
        "state_finite": bool(torch.isfinite(ltc_state).all()),
        "nan_count": int(torch.isnan(ltc_output).sum() + torch.isnan(ltc_state).sum()),
    }
    checks = {
        "exact_ncps_distribution_version": runtime["distribution_version"]
        == OFFICIAL_CFC_DISTRIBUTION_VERSION,
        "exact_official_runtime_binding": (
            runtime["distribution_version"] == OFFICIAL_CFC_DISTRIBUTION_VERSION
            and runtime["git_commit"] == OFFICIAL_CFC_GIT_COMMIT
            and runtime["direct_url"] == OFFICIAL_CFC_DIRECT_URL
        ),
        "official_cfc_forward_has_no_padding_mask": not padding["accepted"]
        and padding["error_type"] == "TypeError",
        "official_cfc_batch_two_heterogeneous_timespans_are_not_supported": not heterogeneous[
            "accepted"
        ] and heterogeneous["error_type"] == "RuntimeError",
        "official_cfc_zero_dt_is_finite_but_not_state_identity": (
            zero_dt["output_finite"]
            and zero_dt["state_finite"]
            and not zero_dt["state_equal_to_input_state"]
        ),
        "official_ltc_zero_dt_is_not_safe_for_v23_direct_swap": not ltc_zero_dt[
            "output_finite"
        ] and ltc_zero_dt["nan_count"] > 0,
        "official_cfc_baseline_is_dense_not_wired": (
            dense["units_argument_is_integer"]
            and dense["wired_false"]
            and not dense["has_wiring_attribute"]
            and dense["class_module"] == OFFICIAL_CFC_CLASS_MODULE
        ),
    }
    if set(checks) != set(RUNTIME_COMPATIBILITY_REQUIRED_CHECKS) or not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"V23 official runtime compatibility changed: {failed}")
    report = {
        "status": "passed",
        "schema_version": RUNTIME_COMPATIBILITY_REPORT_SCHEMA,
        "scope": "deterministic random-tensor upstream API audit; no Radar-LN3 dataset/checkpoint/validation/test/OOF access",
        "runtime": runtime,
        "api": {
            "cfc_forward_signature": str(inspect.signature(CfC.forward)),
            "ltc_forward_signature": str(inspect.signature(LTC.forward)),
        },
        "core_constraints": {
            "cfc_zero_dt": zero_dt,
            "cfc_batch_two_heterogeneous_timespans": heterogeneous,
            "cfc_padding_mask_keyword": padding,
            "cfc_dense_instance": dense,
            "ltc_zero_dt": ltc_zero_dt,
        },
        "adapter_decision": {
            "recurrent_operator": "unmodified ncps.torch.CfC.forward",
            "valid_event_time_input": "raw non-negative dt; zero is passed unchanged",
            "zero_dt_means_state_update_not_skip": True,
            "padding_strategy": "adapter removes right padding before each official per-parent forward call",
            "per_parent_reason": "upstream CfC batch=2 heterogeneous timespans RuntimeError",
            "wiring": "forbidden in V23 dense baseline",
            "ltc": "not a V23 direct replacement because dt=0 is numerically unsafe",
            "no_project_time_semantic_reintroduction": [
                "zero_dt_floor",
                "skip_update",
                "bounded_observation_gain",
                "positive_dt_cap",
            ],
        },
        "performance_boundary": {
            "per_parent_adapter_required": True,
            "expected_overhead_relative_to_hypothetical_vectorized_batch": "higher",
            "semantically_equivalent_upstream_batch_two_reference": "unavailable because batch-two heterogeneous timespans raise RuntimeError",
            "formal_latency_measurement": "not_executed; training-summary elapsed_seconds is descriptive only",
        },
        "checks": checks,
        "dataset_access_count": 0,
        "checkpoint_access_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }
    output = root / RUNTIME_COMPATIBILITY_REPORT_RELATIVE
    _write_json(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V23 official-CfC runtime compatibility.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
