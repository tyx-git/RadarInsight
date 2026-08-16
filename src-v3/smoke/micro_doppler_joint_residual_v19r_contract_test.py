from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from build.micro_doppler_joint_residual_v19r import _build_initial_model
from build.micro_doppler_residual_v19r import (
    _validate_v18r_source,
    _validate_v19_feature_source,
)
from radar_ln3.micro_doppler_joint_residual_v19r import (
    V19R_JOINT_CHECKPOINT_SCHEMA_VERSION,
    V19R_JOINT_CONFIG_RELATIVE_PATH,
    V19R_JOINT_SCHEMA_VERSION,
    load_v19r_joint_checkpoint,
    save_v19r_joint_checkpoint,
    tensor_state_sha256,
)
from radar_ln3.task_models import (
    build_task_dataloader,
    build_task_dataset,
    evaluate_task,
    task_training_objective,
)
from train.micro_doppler_joint_residual_v19r import (
    _forward_model,
    _load_runtime,
    _provenance,
)


def _actual_joint_probe(root: Path) -> dict[str, object]:
    v18r = _validate_v18r_source(root)
    v19 = _validate_v19_feature_source(root, v18r)
    model, initialization, _ = _build_initial_model(root, v18r, v19)
    config = v18r["config"]
    dataset = build_task_dataset(
        root,
        config,
        split_name=config.data.train_split_name,
        parent_limit=3,
    )
    batch = next(iter(build_task_dataloader(dataset, batch_size=3, shuffle=False)))
    tensors = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model.eval()
    with torch.inference_mode():
        initial = _forward_model(model, tensors)
    if (
        not torch.equal(
            initial["sequence_logits"], initial["backbone_sequence_logits"]
        )
        or bool((initial["residual_logits"] != 0.0).any())
    ):
        raise ValueError("V19r joint initial residual does not preserve V18r logits")
    initial_backbone_hash = tensor_state_sha256(model.backbone.state_dict())
    if initial_backbone_hash != initialization[
        "source_v18_initial_treatment_state_sha256"
    ]:
        raise ValueError("V19r joint initial backbone differs from V18r")
    if (
        model.backbone_trainable_parameter_count <= 24
        or model.residual_trainable_parameter_count != 24
        or model.trainable_parameter_count
        != model.backbone_trainable_parameter_count + 24
        or any(not parameter.requires_grad for parameter in model.backbone.parameters())
    ):
        raise ValueError("V19r joint does not expose complete trainable parameters")

    model.train()
    output = _forward_model(model, tensors)
    objective = task_training_objective(
        output,
        tensors,
        config,
        class_weights=torch.tensor(config.train.class_weights, dtype=torch.float32),
    )
    objective["loss"].backward()
    residual_gradient = model.residual.linear.weight.grad
    backbone_gradients = [
        parameter.grad
        for parameter in model.backbone.parameters()
        if parameter.grad is not None
    ]
    if (
        residual_gradient is None
        or not torch.isfinite(residual_gradient).all()
        or float(residual_gradient.abs().max().item()) <= 0.0
        or not backbone_gradients
        or any(not torch.isfinite(value).all() for value in backbone_gradients)
        or max(float(value.abs().max().item()) for value in backbone_gradients) <= 0.0
    ):
        raise ValueError("V19r joint gradients do not reach backbone and residual")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    residual_before = model.residual.linear.weight.detach().clone()
    optimizer.step()
    trained_backbone_hash = tensor_state_sha256(model.backbone.state_dict())
    if (
        trained_backbone_hash == initial_backbone_hash
        or torch.equal(residual_before, model.residual.linear.weight.detach())
    ):
        raise ValueError("V19r joint optimizer did not update both model branches")

    with tempfile.TemporaryDirectory(prefix="radar-ln3-v19r-joint-smoke-") as directory:
        checkpoint_path = Path(directory) / "joint.pt"
        provenance = {"probe": "v19r-joint"}
        save_v19r_joint_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config_name="smoke",
            config_sha256="0" * 64,
            epoch=1,
            global_step=1,
            metrics={"status": "smoke"},
            provenance=provenance,
            training_state={"status": "smoke"},
        )
        restored, _, _ = _build_initial_model(root, v18r, v19)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-4)
        payload = load_v19r_joint_checkpoint(
            checkpoint_path,
            restored,
            optimizer=restored_optimizer,
            expected_config_name="smoke",
            expected_config_sha256="0" * 64,
            expected_provenance=provenance,
        )
        if (
            payload.get("checkpoint_schema")
            != V19R_JOINT_CHECKPOINT_SCHEMA_VERSION
            or tensor_state_sha256(restored.state_dict())
            != tensor_state_sha256(model.state_dict())
        ):
            raise ValueError("V19r joint checkpoint round-trip differs")
    evaluation = evaluate_task(
        restored,
        dataset,
        config=config,
        device=torch.device("cpu"),
        batch_size=3,
    )
    return {
        "actual_parent_ids": list(batch["parent_sample_ids"]),
        "actual_labels": list(batch["label_names"]),
        "source_initial_backbone_state_sha256": initial_backbone_hash,
        "initial_combined_logits_exactly_equal_backbone": True,
        "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
        "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
        "total_trainable_parameter_count": model.trainable_parameter_count,
        "backbone_gradient_absolute_max": max(
            float(value.abs().max().item()) for value in backbone_gradients
        ),
        "residual_gradient_absolute_max": float(
            residual_gradient.abs().max().item()
        ),
        "optimizer_updates_backbone": True,
        "optimizer_updates_residual": True,
        "checkpoint_roundtrip": True,
        "bounded_evaluation_parent_count": evaluation["parent_count"],
    }


def run(project_root: Path, *, prepared: bool) -> dict[str, object]:
    root = project_root.resolve()
    result: dict[str, object] = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V19r-joint-late-residual-smoke-v1",
        "actual_joint_probe": _actual_joint_probe(root),
        "prepared_contract": "not_requested",
        "formal_training": "未执行",
        "test": "未执行",
    }
    if prepared:
        config_path = root / V19R_JOINT_CONFIG_RELATIVE_PATH
        runtime = _load_runtime(root, config_path)
        model, _, _ = _build_initial_model(
            root, runtime["v18r"], runtime["v19"]
        )
        result["prepared_contract"] = {
            "status": "passed",
            "config": str(config_path),
            "config_sha256": runtime["config_sha256"],
            "provenance": _provenance(runtime),
            "backbone_trainable_parameter_count": model.backbone_trainable_parameter_count,
            "residual_trainable_parameter_count": model.residual_trainable_parameter_count,
            "total_trainable_parameter_count": model.trainable_parameter_count,
            "full_backbone_training": True,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused Smoke for corrected V19r full joint training."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    run(args.project_root.resolve(), prepared=args.prepared)


if __name__ == "__main__":
    main()
