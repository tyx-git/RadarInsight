"""V27d-r02 合同 smoke：验证隔离、形状和 parent 级边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from v27.V27d.r20.data import R20Dataset, build_r20_loader, r20_model_input_kwargs
from v27.V27d.r20.training import load_checkpoint as load_r20_checkpoint

from .model import R02BirdDroneModel
from .protocol import (
    R02_CLASSES,
    R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS,
    R02_TRAINING_HARD_NEGATIVE_IDS,
    R02Config,
    prepare_manifest,
    project_path,
    write_json,
)


def run_smoke(project_root: str | Path, config: R02Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path, manifest = prepare_manifest(root, config)
    r20_config = config.r20
    base_model, _ = load_r20_checkpoint(
        root,
        r20_config,
        arm="zero_control",
        seed=42,
        checkpoint_path=project_path(root, config.source["r20_checkpoint_path"], must_exist=True),
    )
    model = R02BirdDroneModel(
        base_model,
        reliability_hidden_dim=int(config.model["reliability_hidden_dim"]),
        maturity_hidden_dim=int(config.model["maturity_hidden_dim"]),
    )
    train = R20Dataset(root, r20_config, split_name="train", arm="zero_control")
    validation = R20Dataset(root, r20_config, split_name="validation", arm="zero_control")
    if set(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS) & set(train.parent_ids):
        raise AssertionError("validation audit hard negatives leaked into train")
    if not set(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS).issubset(set(validation.parent_ids)):
        raise AssertionError("validation audit hard negatives are absent")
    if not set(R02_TRAINING_HARD_NEGATIVE_IDS).issubset(set(train.parent_ids)):
        raise AssertionError("training hard negatives are absent")
    batch = next(iter(build_r20_loader(train, batch_size=2, shuffle=False)))
    output = model(**r20_model_input_kwargs(batch, torch.device("cpu")))
    expected_shape = (2, int(batch["tr_features"].shape[1]))
    for key in ("bird_reliability", "maturity"):
        if tuple(output[key].shape) != expected_shape:
            raise AssertionError(f"r02 {key} shape differs")
    if tuple(output["window_logits"].shape) != (2, len(R02_CLASSES)):
        raise AssertionError("r02 window logits shape differs")
    if not torch.isfinite(output["window_logits"]).all():
        raise AssertionError("r02 logits are not finite")
    payload = {
        "schema_version": "RADAR-LN3-V27d-r02-contract-smoke-v1",
        "experiment_id": "V27d-r02",
        "status": "passed",
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "train_parent_count": len(train),
        "validation_parent_count": len(validation),
        "training_hard_negative_count": len(R02_TRAINING_HARD_NEGATIVE_IDS),
        "validation_audit_hard_negative_count": len(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS),
        "complete_parent_sequence": True,
        "event_random_split": False,
        "base_r20_checkpoint_untouched": True,
        "model_contract": model.model_contract(),
        "batch_shape": {
            "tr_features": list(batch["tr_features"].shape),
            "window_logits": list(output["window_logits"].shape),
        },
    }
    output_path = project_path(root, config.artifacts["smoke_path"])
    write_json(output_path, payload)
    return {"report_path": output_path.relative_to(root).as_posix(), **payload}


__all__ = ["run_smoke"]
