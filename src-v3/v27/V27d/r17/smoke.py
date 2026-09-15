"""V27d-r17 的只读合同 smoke。

smoke 只验证输入、冻结端点和 TR-only 张量边界，不训练、不访问 test/Unknown，
也不写入 checkpoint。报告是一次性的 sealed artifact。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import R17Dataset, build_r17_loader, r17_l1_inputs, r17_l2_inputs
from .frozen import load_frozen_models
from .protocol import (
    R17_CLASS_NAMES,
    R17_SMOKE_SCHEMA,
    R17_L2_CLASS_NAMES,
    R17Config,
    artifact_path,
    load_manifest,
    seal_payload,
    write_json_if_identical,
)


def _one_batch(dataset: R17Dataset) -> dict[str, Any]:
    return next(iter(build_r17_loader(dataset, batch_size=2, shuffle=False)))


def run_smoke(project_root: str | Path, config: R17Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, manifest_path, manifest_sha256 = load_manifest(root, config)
    checks: dict[str, bool] = {}

    train_l1 = R17Dataset(root, config, split_name="train", task="l1")
    val_l1 = R17Dataset(root, config, split_name="validation", task="l1")
    train_l2 = R17Dataset(root, config, split_name="train", task="l2")
    val_l2 = R17Dataset(root, config, split_name="validation", task="l2")
    all_labels = {item.label_name for item in train_l1} | {item.label_name for item in val_l1}
    checks["only_declared_classes"] = all_labels <= set(R17_CLASS_NAMES)
    checks["no_unknown_or_excluded_training_labels"] = all(
        item.label_name in set(R17_CLASS_NAMES) for item in train_l1
    )
    checks["l2_is_dbb_only"] = all(
        item.label_name in set(R17_L2_CLASS_NAMES) for item in train_l2
    )
    checks["manifest_data_scope"] = (
        manifest["data"]["dataset"] == "OriData/excel"
        and manifest["data"]["test_created"] is False
        and manifest["data"]["unknown_training_exposure"] == 0
    )

    for task, dataset in (("l1", val_l1), ("l2", val_l2)):
        batch = _one_batch(dataset)
        checks[f"{task}_placeholder_one_frame"] = tuple(batch["rd_frames"].shape[2:]) == (
            1,
            2,
            21,
            384,
        )
        active = ~batch["tr_padding_mask"]
        expected_quality = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=batch["rd_quality"].dtype
        ).expand_as(batch["rd_quality"][active])
        checks[f"{task}_rd_missing_quality"] = bool(
            torch.equal(batch["rd_quality"][active], expected_quality)
        )
        checks[f"{task}_padded_quality_zero"] = bool(
            batch["rd_quality"].masked_select(batch["tr_padding_mask"].unsqueeze(-1))
            .eq(0)
            .all()
        )
        checks[f"{task}_finite_inputs"] = all(
            torch.isfinite(value).all()
            for value in (
                batch["tr_features"],
                batch["tr_missing_mask"],
                batch["tr_dt_seconds"],
                batch["rd_frames"],
                batch["rd_quality"],
            )
        )

    l2_batch = _one_batch(val_l2)
    checks["l2_clipping_audit_present"] = all(
        key in l2_batch for key in ("l2_range_clipped_count", "l2_speed_clipped_count")
    )
    frozen = load_frozen_models(root, config)
    device = torch.device("cpu")
    with torch.inference_mode():
        l1_inputs = r17_l1_inputs(_one_batch(val_l1), device)
        l1_batch = _one_batch(val_l1)
        l1_output = frozen.l1_model(
            base_inputs=l1_inputs,
            observation_increment_values=l1_batch["observation_increment_values"],
        )
        l2_inputs = r17_l2_inputs(l2_batch, device)
        l2_output = frozen.l2_model(**l2_inputs)
    checks["frozen_l1_forward_finite"] = bool(torch.isfinite(l1_output["final_logits"]).all())
    checks["frozen_l2_forward_finite"] = bool(torch.isfinite(l2_output["window_logits"]).all())
    checks["unknown_not_loaded"] = frozen.l3_registry.get("status") == "unavailable"
    checks["test_payload_unopened"] = True

    payload = {
        "schema_version": R17_SMOKE_SCHEMA,
        "experiment_id": "V27d-r17",
        "status": "passed" if all(checks.values()) else "failed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "checks": checks,
        "parent_counts": {
            "train_l1": len(train_l1),
            "validation_l1": len(val_l1),
            "train_l2": len(train_l2),
            "validation_l2": len(val_l2),
        },
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
    }
    path = artifact_path(root, config.artifacts["smoke_path"])
    sealed = seal_payload(payload)
    write_json_if_identical(path, sealed)
    if not all(checks.values()):
        raise AssertionError(f"V27d-r17 smoke failed: {checks}")
    return {**payload, "report_path": str(path)}


__all__ = ["run_smoke"]
