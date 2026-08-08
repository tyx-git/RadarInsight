from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.rd_physical import extract_current_event_rd_physical
from radar_ln3.task_models import (
    RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION,
    build_task_dataloader,
    build_task_dataset,
    build_v11_paired_models,
    capture_training_state,
    load_task_checkpoint,
    load_task_config,
    restore_training_state,
    save_task_checkpoint,
    seed_everything,
    task_prefix_loss,
    verify_task_data_contract,
)
from train.tasks import parent_order_sha256


CONTROL_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-control-seed42-config.json"
)
TREATMENT_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-treatment-seed42-config.json"
)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _shared_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    return {
        "tr_features": batch["tr_features"],
        "tr_missing_mask": batch["tr_missing_mask"],
        "tr_dt_seconds": batch["tr_dt_seconds"],
        "tr_dt_valid": batch["tr_dt_valid"],
        "tr_motion_features": batch.get("tr_motion_features"),
        "tr_motion_valid_mask": batch.get("tr_motion_valid_mask"),
        "rd_frames": batch["rd_frames"],
        "rd_frame_mask": batch["rd_frame_mask"],
        "rd_quality": batch["rd_quality"],
        "tr_padding_mask": batch["tr_padding_mask"],
    }


def _physical_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "rd_physical_features": batch["rd_physical_features"],
        "rd_physical_valid": batch["rd_physical_valid"],
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _real_event_online_cache_check(
    project_root: Path,
    treatment_dataset: Any,
    treatment_model: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    parent_id = treatment_dataset.parent_ids[0]
    group = treatment_dataset.base._groups[parent_id].iloc[: treatment_dataset.base.max_sequence_length]
    item = treatment_dataset[0]
    checked = 0
    valid_checked = 0
    invalid_checked = 0
    maximum_error = 0.0
    for event_index, (_, row) in enumerate(group.iterrows()):
        if str(row["sample_id"]) != f"{parent_id}#point{event_index + 1:04d}":
            raise AssertionError("V11 online/cache smoke found an unexpected sample ordering contract")
        present = str(row["rd_present"]).strip().lower() in {"1", "true", "yes", "y"}
        if present:
            indices = [int(value) for value in json.loads(str(row["rd_tensor_frame_indices_json"]))]
            indices = indices[: treatment_dataset.base.max_rd_frames_per_point]
            path = Path(str(row["rd_tensor_path"]).replace("\\", "/"))
            path = path if path.is_absolute() else project_root / path
            with np.load(path, allow_pickle=False) as payload:
                amplitude = np.asarray(payload["amplitude"])[np.asarray(indices, dtype=np.int64)]
                velocity = np.asarray(payload["velocity_mps"])
            extracted = extract_current_event_rd_physical(amplitude, velocity)
            prepared = treatment_model.prepare_online_rd_physical(amplitude, velocity)
            valid_checked += 1
        else:
            path = project_root / "data/processed/cq08_rd_physical_canonical_v3_1/cq08__track__1_rd_tensor.npz"
            with np.load(path, allow_pickle=False) as payload:
                amplitude = np.asarray(payload["amplitude"][:0])
                velocity = np.asarray(payload["velocity_mps"])
            extracted = extract_current_event_rd_physical(amplitude, velocity)
            prepared = treatment_model.prepare_online_rd_physical(amplitude, velocity)
            invalid_checked += 1
        cached = item.rd_physical_features[event_index].numpy()
        error = float(np.max(np.abs(cached - extracted.values)))
        maximum_error = max(maximum_error, error)
        if not np.array_equal(cached, extracted.values):
            raise AssertionError("V11 online extractor differs from the versioned cache")
        if bool(item.rd_physical_valid[event_index].item()) != extracted.valid:
            raise AssertionError("V11 online extractor valid differs from the versioned cache")
        if not torch.equal(
            prepared["rd_physical_features"].detach().cpu()[0],
            item.rd_physical_features[event_index],
        ):
            raise AssertionError("V11 model online preparation differs from the Dataset cache")
        if bool(prepared["rd_physical_valid"].item()) != extracted.valid:
            raise AssertionError("V11 model online preparation valid differs from the Dataset cache")
        checked += 1
    reference_path = project_root / "data/processed/cq08_rd_physical_canonical_v3_1/cq08__track__1_rd_tensor.npz"
    with np.load(reference_path, allow_pickle=False) as payload:
        empty_amplitude = np.asarray(payload["amplitude"][:0])
        reference_velocity = np.asarray(payload["velocity_mps"])
    missing = treatment_model.prepare_online_rd_physical(empty_amplitude, reference_velocity)
    if (
        bool(missing["rd_physical_valid"].item())
        or bool((missing["rd_physical_features"] != 0.0).any())
    ):
        raise AssertionError("V11 online missing-RD preparation must be zero and invalid")
    frame_cap_rejected = False
    with np.load(reference_path, allow_pickle=False) as payload:
        excessive_amplitude = np.asarray(
            payload["amplitude"][: treatment_dataset.base.max_rd_frames_per_point + 1]
        )
    try:
        treatment_model.prepare_online_rd_physical(excessive_amplitude, reference_velocity)
    except ValueError as exc:
        frame_cap_rejected = "frame cap" in str(exc)
    if not frame_cap_rejected:
        raise AssertionError("V11 online preparation accepted more frames than the frozen cache/CNN cap")
    return {
        "parent_sample_id": parent_id,
        "event_count": checked,
        "valid_event_count": valid_checked,
        "invalid_event_count": invalid_checked,
        "cache_and_shared_extractor_exact": True,
        "cache_and_model_online_preparation_exact": True,
        "maximum_absolute_error": maximum_error,
        "causal_scope": "one current RD event at a time",
        "explicit_missing_online_zero_invalid": True,
        "excess_frames_rejected": frame_cap_rejected,
    }


def _online_sequence_check(
    model: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    parent_index = 0
    length = int(batch["lengths"][parent_index].item())
    shared = _shared_inputs(batch)
    with torch.inference_mode():
        offline = model(
            **shared,
            **_physical_inputs(batch),
        )["sequence_logits"][parent_index, :length]
        state: torch.Tensor | None = None
        online_logits: list[torch.Tensor] = []
        for event_index in range(length):
            output = model.forward_online_point(
                tr_features=batch["tr_features"][parent_index : parent_index + 1, event_index],
                tr_missing_mask=batch["tr_missing_mask"][parent_index : parent_index + 1, event_index],
                tr_dt_seconds=batch["tr_dt_seconds"][parent_index : parent_index + 1, event_index],
                tr_dt_valid=batch["tr_dt_valid"][parent_index : parent_index + 1, event_index],
                tr_motion_features=batch["tr_motion_features"][parent_index : parent_index + 1, event_index],
                tr_motion_valid_mask=batch["tr_motion_valid_mask"][parent_index : parent_index + 1, event_index],
                rd_frames=batch["rd_frames"][parent_index : parent_index + 1, event_index],
                rd_frame_mask=batch["rd_frame_mask"][parent_index : parent_index + 1, event_index],
                rd_quality=batch["rd_quality"][parent_index : parent_index + 1, event_index],
                rd_physical_features=batch["rd_physical_features"][parent_index : parent_index + 1, event_index],
                rd_physical_valid=batch["rd_physical_valid"][parent_index : parent_index + 1, event_index],
                liquid_state=state,
            )
            state = output["liquid_state"]
            online_logits.append(output["window_logits"][0])
        online = torch.stack(online_logits, dim=0)
    difference = (offline - online).abs()
    tolerance = 2.0e-5 if offline.is_cuda else 2.0e-6
    if not torch.allclose(offline, online, atol=tolerance, rtol=tolerance):
        raise AssertionError(
            "V11 offline and online causal logits differ beyond tolerance: "
            f"maximum={float(difference.max().item())}, tolerance={tolerance}"
        )
    return {
        "parent_sample_id": batch["parent_sample_ids"][parent_index],
        "event_count": length,
        "maximum_absolute_logit_error": float(difference.max().item()),
        "tolerance": tolerance,
        "passed": True,
        "state_reset_boundary": "new model state per parent",
    }


def _loader_order_check(dataset: Any, seed: int, batch_size: int) -> dict[str, Any]:
    orders: list[list[str]] = []
    hashes: list[str] = []
    for _ in range(2):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        loader = build_task_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        order: list[str] = []
        for batch in loader:
            order.extend(str(value) for value in batch["parent_sample_ids"])
        orders.append(order)
        hashes.append(parent_order_sha256(order))
    if orders[0] != orders[1] or len(orders[0]) != len(dataset) or len(set(orders[0])) != len(dataset):
        raise AssertionError("V11 same-seed independent DataLoader generators did not reproduce one parent order")
    return {
        "seed": seed,
        "parent_count": len(orders[0]),
        "unique_parent_count": len(set(orders[0])),
        "independent_generator_hashes": hashes,
        "hashes_equal": hashes[0] == hashes[1],
    }


def _checkpoint_recovery_check(
    model: Any,
    config: Any,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    loader_generator: torch.Generator,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    history = [
        {
            "epoch": 1,
            "global_step": 1,
            "parent_order_sha256": parent_order_sha256(list(batch["parent_sample_ids"])),
        }
    ]
    training_state = capture_training_state(
        loader_generator,
        run_state={
            "next_epoch": 2,
            "global_step": 1,
            "best_epoch": 1,
            "best_selection": {
                "metric": "parent_min_precision_recall",
                "primary_value": 0.0,
                "ranking_key": [0.0, 0.0],
            },
            "history": history,
        },
    )
    save_task_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config=config,
        epoch=1,
        global_step=1,
        metrics={"smoke": True},
        training_state=training_state,
    )
    model.eval()
    with torch.inference_mode():
        expected_logits = model(
            **_shared_inputs(batch),
            **_physical_inputs(batch),
        )["window_logits"]
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch_cpu = torch.rand(4)
    expected_loader = torch.randperm(17, generator=loader_generator)
    expected_cuda = torch.rand(4, device=device) if device.type == "cuda" else None

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)
    loader_generator.manual_seed(999)
    restored = type(model)(config).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=config.train.learning_rate)
    payload = load_task_checkpoint(
        checkpoint_path,
        restored,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
        optimizer=restored_optimizer,
    )
    run_state = restore_training_state(payload, loader_generator)
    restored.eval()
    with torch.inference_mode():
        actual_logits = restored(
            **_shared_inputs(batch),
            **_physical_inputs(batch),
        )["window_logits"]
    actual_python = random.random()
    actual_numpy = float(np.random.random())
    actual_torch_cpu = torch.rand(4)
    actual_loader = torch.randperm(17, generator=loader_generator)
    actual_cuda = torch.rand(4, device=device) if device.type == "cuda" else None
    checks = {
        "model_logits_exact": torch.equal(expected_logits, actual_logits),
        "optimizer_state_present": "optimizer_state" in payload,
        "python_rng_exact": expected_python == actual_python,
        "numpy_rng_exact": expected_numpy == actual_numpy,
        "torch_cpu_rng_exact": torch.equal(expected_torch_cpu, actual_torch_cpu),
        "dataloader_generator_exact": torch.equal(expected_loader, actual_loader),
        "cuda_rng_exact": True if expected_cuda is None else torch.equal(expected_cuda, actual_cuda),
        "run_state_exact": run_state["next_epoch"] == 2 and run_state["global_step"] == 1,
    }
    if not all(checks.values()):
        raise AssertionError(f"V11 checkpoint recovery smoke failed: {checks}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_schema": payload["checkpoint_schema"],
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "epoch_boundary_recovery": checks,
    }


def run(project_root: Path, *, device_name: str, parent_limit: int = 6) -> dict[str, Any]:
    root = project_root.resolve()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA V11 smoke was requested but CUDA is unavailable")
    device = torch.device(device_name)
    control_config = load_task_config(root / CONTROL_CONFIG)
    treatment_config = load_task_config(root / TREATMENT_CONFIG)
    if control_config.schema_version != RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        raise ValueError("V11 paired smoke loaded an unexpected Control schema")
    formal_artifact_paths: list[str] = []
    for seed in (42, 43, 44):
        for arm in ("control", "treatment"):
            path = root / (
                f"config/RADAR-LN3/RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-{arm}-seed{seed}-config.json"
            )
            configured = load_task_config(path)
            artifact = root / configured.train.formal_artifact_dir
            if artifact.exists() and any(artifact.iterdir()):
                raise AssertionError(f"V11 formal artifact exists before user training: {artifact}")
            formal_artifact_paths.append(str(artifact))
    control_contract = verify_task_data_contract(root, control_config)
    treatment_contract = verify_task_data_contract(root, treatment_config)
    control_dataset = build_task_dataset(
        root,
        control_config,
        split_name=control_config.data.train_split_name,
        parent_limit=parent_limit,
    )
    treatment_dataset = build_task_dataset(
        root,
        treatment_config,
        split_name=treatment_config.data.train_split_name,
        parent_limit=parent_limit,
    )
    if control_dataset.parent_ids != treatment_dataset.parent_ids:
        raise AssertionError("V11 Control/Treatment smoke parent sets differ")
    control_batch_cpu = next(
        iter(build_task_dataloader(control_dataset, batch_size=2, shuffle=False))
    )
    treatment_batch_cpu = next(
        iter(build_task_dataloader(treatment_dataset, batch_size=2, shuffle=False))
    )
    immutable_keys = (
        "tr_features",
        "tr_missing_mask",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_motion_features",
        "tr_motion_valid_mask",
        "rd_frames",
        "rd_frame_mask",
        "rd_quality",
        "tr_padding_mask",
        "labels",
        "lengths",
    )
    if any(not torch.equal(control_batch_cpu[key], treatment_batch_cpu[key]) for key in immutable_keys):
        raise AssertionError("V11 Control/Treatment immutable real-batch tensors differ")
    if "rd_physical_features" in control_batch_cpu or "rd_physical_valid" in control_batch_cpu:
        raise AssertionError("V11 Control batch must not expose RD physical tensors")
    if "rd_physical_features" not in treatment_batch_cpu or "rd_physical_valid" not in treatment_batch_cpu:
        raise AssertionError("V11 Treatment batch lacks RD physical tensors")

    seed_everything(control_config.train.seed)
    control_model, treatment_model, initialization = build_v11_paired_models(
        control_config,
        treatment_config,
    )
    control_model.eval()
    treatment_model.eval()
    cpu_shared = _shared_inputs(control_batch_cpu)
    cpu_active = ~control_batch_cpu["tr_padding_mask"]
    cpu_nonzero_probe = torch.zeros(
        (*cpu_active.shape, 4),
        dtype=control_batch_cpu["tr_features"].dtype,
    )
    cpu_nonzero_probe[cpu_active] = 0.5
    with torch.inference_mode():
        cpu_control_initial = control_model(**cpu_shared)["sequence_logits"]
        cpu_treatment_initial = treatment_model(
            **cpu_shared,
            rd_physical_features=cpu_nonzero_probe,
            rd_physical_valid=cpu_active,
        )["sequence_logits"]
    cpu_initial_difference = float((cpu_control_initial - cpu_treatment_initial).abs().max().item())
    if not torch.equal(cpu_control_initial, cpu_treatment_initial):
        raise AssertionError(
            "V11 CPU reference initial logits differ despite zero new columns: "
            f"maximum_absolute_difference={cpu_initial_difference}"
        )
    initialization.update(
        {
            "initial_cpu_reference_logits_exactly_equal_with_nonzero_probe": True,
            "initial_cpu_reference_maximum_absolute_difference": cpu_initial_difference,
            "initial_cpu_reference_logits_sha256": hashlib.sha256(
                cpu_control_initial.detach().contiguous().numpy().tobytes(order="C")
            ).hexdigest(),
        }
    )
    control_model = control_model.to(device)
    treatment_model = treatment_model.to(device)
    control_batch = _to_device(control_batch_cpu, device)
    treatment_batch = _to_device(treatment_batch_cpu, device)
    shared = _shared_inputs(control_batch)
    active = ~control_batch["tr_padding_mask"]
    nonzero_probe = torch.zeros(
        (*active.shape, 4),
        dtype=control_batch["tr_features"].dtype,
        device=device,
    )
    nonzero_probe[active] = 0.5
    control_model.eval()
    treatment_model.eval()
    with torch.inference_mode():
        control_initial = control_model(**shared)["sequence_logits"]
        treatment_initial = treatment_model(
            **shared,
            rd_physical_features=nonzero_probe,
            rd_physical_valid=active,
        )["sequence_logits"]
    initial_difference = float((control_initial - treatment_initial).abs().max().item())
    device_tolerance = 5.0e-6 if device.type == "cuda" else 0.0
    device_close = (
        torch.equal(control_initial, treatment_initial)
        if device_tolerance == 0.0
        else torch.allclose(control_initial, treatment_initial, atol=device_tolerance, rtol=device_tolerance)
    )
    if not device_close:
        raise AssertionError(
            "V11 device initial logits differ beyond the frozen numerical contract: "
            f"maximum_absolute_difference={initial_difference}, tolerance={device_tolerance}"
        )
    initialization.update(
        {
            "initial_device_logits_exactly_equal_with_nonzero_probe": torch.equal(
                control_initial, treatment_initial
            ),
            "initial_device_logits_numerically_equal_with_nonzero_probe": True,
            "initial_device_logits_maximum_absolute_difference": initial_difference,
            "initial_device_logits_tolerance": device_tolerance,
            "initial_device_logits_sha256_control": hashlib.sha256(
                control_initial.detach().cpu().contiguous().numpy().tobytes(order="C")
            ).hexdigest(),
            "cuda_bitwise_determinism_claimed": False,
        }
    )

    online_cache = _real_event_online_cache_check(
        root,
        treatment_dataset,
        treatment_model,
        device=device,
    )
    online_sequence = _online_sequence_check(treatment_model, treatment_batch)
    loader_order = _loader_order_check(
        treatment_dataset,
        treatment_config.train.seed,
        treatment_config.train.batch_size,
    )

    treatment_model.train()
    optimizer = torch.optim.AdamW(
        treatment_model.parameters(),
        lr=treatment_config.train.learning_rate,
    )
    output = treatment_model(
        **_shared_inputs(treatment_batch),
        **_physical_inputs(treatment_batch),
    )
    weights = torch.tensor(
        treatment_config.train.class_weights,
        dtype=torch.float32,
        device=device,
    )
    loss = task_prefix_loss(
        output["sequence_logits"],
        treatment_batch["labels"],
        treatment_batch["tr_padding_mask"],
        class_weights=weights,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    fusion_gradient = treatment_model.fusion[0].weight.grad
    if fusion_gradient is None:
        raise AssertionError("V11 Treatment Fusion has no gradient")
    new_column_gradient = fusion_gradient[:, control_model.fusion_input_dim :]
    new_column_gradient_norm = float(new_column_gradient.pow(2).sum().sqrt().item())
    if not np.isfinite(new_column_gradient_norm) or new_column_gradient_norm <= 0.0:
        raise AssertionError("V11 four new Fusion columns did not receive a finite nonzero gradient")
    total_gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            treatment_model.parameters(),
            treatment_config.train.gradient_clip_norm,
        ).item()
    )
    before_step = treatment_model.fusion[0].weight[:, control_model.fusion_input_dim :].detach().clone()
    optimizer.step()
    after_step = treatment_model.fusion[0].weight[:, control_model.fusion_input_dim :].detach()
    new_columns_updated = not torch.equal(before_step, after_step)
    if not new_columns_updated:
        raise AssertionError("V11 optimizer did not update the four RD physical Fusion columns")
    if not torch.isfinite(loss).item() or not np.isfinite(total_gradient_norm):
        raise AssertionError("V11 Treatment smoke loss or total gradient is non-finite")

    output_dir = root / "src-v3" / "smoke" / f"RADAR-LN3-V11-rd-physical-paired-{device_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(treatment_config.train.dataloader_seed or treatment_config.train.seed)
    # Advance the generator so recovery verifies a non-initial state.
    torch.randperm(31, generator=generator)
    checkpoint = _checkpoint_recovery_check(
        treatment_model,
        treatment_config,
        optimizer,
        treatment_batch,
        generator,
        output_dir / "epoch-boundary-checkpoint.pt",
        device=device,
    )

    result = {
        "status": "passed",
        "schema_version": RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION,
        "device": str(device),
        "scope": "bounded paired contract smoke; not formal training or performance evaluation",
        "configs": {
            "control": str(root / CONTROL_CONFIG),
            "treatment": str(root / TREATMENT_CONFIG),
        },
        "data_contract_hashes": {
            "control": control_contract["hashes"],
            "treatment": treatment_contract["hashes"],
            "rd_physical": treatment_contract["rd_physical"]["hashes"],
        },
        "parent_limit": parent_limit,
        "batch_parent_sample_ids": treatment_batch_cpu["parent_sample_ids"],
        "batch_shapes": {
            key: list(value.shape)
            for key, value in treatment_batch_cpu.items()
            if isinstance(value, torch.Tensor)
        },
        "immutable_control_treatment_tensors_exact": True,
        "control_has_no_rd_physical_tensor": True,
        "formal_artifact_directories_clean": True,
        "formal_artifact_paths": formal_artifact_paths,
        "paired_initialization": initialization,
        "online_cache_contract": online_cache,
        "offline_online_contract": online_sequence,
        "loader_order_contract": loader_order,
        "optimization": {
            "loss": float(loss.item()),
            "total_gradient_norm": total_gradient_norm,
            "new_column_gradient_norm": new_column_gradient_norm,
            "new_columns_updated_after_one_step": new_columns_updated,
            "formal_training_step": False,
        },
        "checkpoint": checkpoint,
        "model_state_sha256_after_step": _state_sha256(treatment_model.state_dict()),
        "formal_training": "未执行",
        "formal_validation": "未执行",
        "test_evaluation": "未执行",
        "OOF": "未执行",
        "P95": "未执行",
    }
    result_path = output_dir / "result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated V11 RD physical paired contract smoke.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--parent-limit", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.project_root, device_name=args.device, parent_limit=args.parent_limit),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
