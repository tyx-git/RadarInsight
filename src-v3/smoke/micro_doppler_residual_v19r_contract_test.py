from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from build.micro_doppler_residual_v19r import (
    _forward_v18r,
    _validate_v18r_source,
    _validate_v19_feature_source,
)
from radar_ln3.micro_doppler_residual_v19r import (
    MicroDopplerLateResidual,
    V19R_CACHE_FIELDS,
    V19R_CACHE_SCHEMA_VERSION,
    V19R_CHECKPOINT_SCHEMA_VERSION,
    V19R_CONFIG_NAME,
    V19R_CONFIG_RELATIVE_PATH,
    V19R_SCHEMA_VERSION,
    V19_CLASS_NAMES,
    V19_FEATURE_DIM,
    V19_RESIDUAL_BOUND,
    V19_TRAINABLE_PARAMETER_COUNT,
    V19rResidualCacheDataset,
    build_residual_dataloader,
    evaluate_residual,
    load_v19r_residual_checkpoint,
    normalize_stable8,
    residual_objective,
    save_v19r_residual_checkpoint,
    sha256_file,
    tensor_state_sha256,
)
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
)
from report.micro_doppler_residual_v19r import run as run_v19r_report
from train.micro_doppler_residual_v19r import (
    _load_runtime,
    _provenance,
    train as train_v19r,
)
from train.tasks import parent_order_sha256


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _synthetic_cache(path: Path, *, split: str, prefix: str) -> None:
    parent_count = 6
    sequence = 4
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    padding = np.zeros((parent_count, sequence), dtype=np.bool_)
    valid = np.ones_like(padding)
    counts = np.tile(np.arange(1, sequence + 1, dtype=np.int16), (parent_count, 1))
    features = np.zeros((parent_count, sequence, V19_FEATURE_DIM), dtype=np.float32)
    class_vectors = np.asarray(
        [
            [2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 1.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    for parent_index, label in enumerate(labels):
        for event_index in range(sequence):
            features[parent_index, event_index] = class_vectors[label] * (
                0.75 + 0.25 * (event_index + 1)
            )
    arrays = {
        "schema_version": np.asarray(V19R_CACHE_SCHEMA_VERSION),
        "split_name": np.asarray(split),
        "parent_sample_ids": np.asarray(
            [f"{prefix}-parent-{index}" for index in range(parent_count)]
        ),
        "track_ids": np.asarray([f"{prefix}-{index}" for index in range(parent_count)]),
        "source_names": np.asarray(["synthetic"] * parent_count),
        "label_names": np.asarray([V19_CLASS_NAMES[value] for value in labels]),
        "labels": labels,
        "lengths": np.full(parent_count, sequence, dtype=np.int64),
        "padding_mask": padding,
        "micro_doppler_valid": valid,
        "valid_counts": counts,
        "baseline_logits": np.zeros((parent_count, sequence, 3), dtype=np.float32),
        "causal_features": features,
    }
    if tuple(arrays) != V19R_CACHE_FIELDS:
        raise ValueError("V19r synthetic cache field order differs")
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _mechanism_probe() -> dict[str, object]:
    model = MicroDopplerLateResidual()
    if model.linear.bias is not None or model.trainable_parameter_count != 24:
        raise ValueError("V19r residual is not the frozen 24-weight bias-free layer")
    baseline = torch.randn((2, 4, 3), generator=torch.Generator().manual_seed(7))
    features = torch.randn((2, 4, 8), generator=torch.Generator().manual_seed(8))
    padding = torch.tensor(
        [[False, False, False, False], [False, False, False, True]]
    )
    initial = model(baseline, features, padding)
    if not torch.equal(initial["sequence_logits"], baseline):
        raise ValueError("V19r zero initialization does not preserve baseline logits")
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.arange(1, 25, dtype=torch.float32).reshape(3, 8) * 100.0
        )
    residual = model(baseline, features, padding)["residual_logits"]
    active = ~padding
    absolute_max = float(residual[active].abs().max().item())
    zero_sum_max = float(residual[active].sum(dim=-1).abs().max().item())
    pairwise_max = float(
        (residual[active].amax(dim=-1) - residual[active].amin(dim=-1)).max().item()
    )
    if (
        absolute_max > V19_RESIDUAL_BOUND + 2.0e-6
        or zero_sum_max > 2.0e-6
        or pairwise_max > 2.0 * V19_RESIDUAL_BOUND + 2.0e-6
        or bool((residual[padding] != 0.0).any())
    ):
        raise ValueError("V19r bounded zero-sum residual contract failed")
    return {
        "zero_initialization_exact": True,
        "trainable_parameter_count": model.trainable_parameter_count,
        "absolute_max": absolute_max,
        "zero_sum_absolute_max": zero_sum_max,
        "pairwise_max": pairwise_max,
    }


def _checkpoint_and_cache_roundtrip() -> dict[str, object]:
    model = MicroDopplerLateResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    baseline = torch.zeros((2, 4, 3))
    features = torch.randn((2, 4, 8), generator=torch.Generator().manual_seed(9))
    padding = torch.zeros((2, 4), dtype=torch.bool)
    objective = residual_objective(
        model(baseline, features, padding)["sequence_logits"],
        {"label": torch.tensor([0, 1]), "padding_mask": padding},
        class_weights=torch.ones(3),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    optimizer.step()
    provenance = {"probe": "v19r"}
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v19r-roundtrip-") as directory:
        root = Path(directory)
        checkpoint_path = root / "checkpoint.pt"
        cache_path = root / "cache.npz"
        save_v19r_residual_checkpoint(
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
        restored = MicroDopplerLateResidual()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-4)
        payload = load_v19r_residual_checkpoint(
            checkpoint_path,
            restored,
            optimizer=restored_optimizer,
            expected_config_name="smoke",
            expected_config_sha256="0" * 64,
            expected_provenance=provenance,
        )
        if (
            payload.get("checkpoint_schema") != V19R_CHECKPOINT_SCHEMA_VERSION
            or not torch.equal(model.linear.weight, restored.linear.weight)
        ):
            raise ValueError("V19r checkpoint round-trip differs")
        _synthetic_cache(cache_path, split="train", prefix="roundtrip")
        dataset = V19rResidualCacheDataset(cache_path)
        if len(dataset) != 6 or dataset[0]["length"].item() != 4:
            raise ValueError("V19r cache round-trip differs")
    return {
        "checkpoint_roundtrip": True,
        "cache_roundtrip": True,
        "checkpoint_epoch": int(payload["epoch"]),
    }


def _synthetic_end_to_end() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v19r-e2e-") as directory:
        root = Path(directory)
        config_path = root / "config.json"
        readiness_path = root / "readiness.json"
        contract_path = root / "contract.json"
        normalization_path = root / "normalization.json"
        train_cache_path = root / "train.npz"
        validation_cache_path = root / "val.npz"
        baseline_checkpoint_path = root / "baseline.pt"
        baseline_summary_path = root / "baseline-summary.json"
        paired_path = root / "paired.json"
        source_v19_config_path = root / "source-v19-config.json"
        source_v19_readiness_path = root / "source-v19-readiness.json"
        artifact = root / "artifact"
        report_path = root / "report.json"
        _synthetic_cache(train_cache_path, split="train", prefix="train")
        _synthetic_cache(validation_cache_path, split="val", prefix="val")
        _write_json(normalization_path, {"schema_version": "synthetic"})
        baseline_checkpoint_path.write_bytes(b"synthetic-v18r-best")
        train_dataset = V19rResidualCacheDataset(train_cache_path)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        loader = build_residual_dataloader(
            train_dataset, batch_size=2, shuffle=True, generator=generator
        )
        order_hashes: dict[str, str] = {}
        history: list[dict[str, object]] = []
        for epoch in range(1, 21):
            order: list[str] = []
            for batch in loader:
                order.extend(str(value) for value in batch["parent_sample_id"])
            digest = parent_order_sha256(order)
            order_hashes[str(epoch)] = digest
            history.append({"epoch": epoch, "parent_order_sha256": digest})
        _write_json(
            baseline_summary_path,
            {"status": "completed", "best_epoch": 15, "history": history},
        )
        _write_json(source_v19_config_path, {"schema_version": "synthetic-v19"})
        _write_json(
            source_v19_readiness_path,
            {
                "status": "passed",
                "config_sha256": sha256_file(source_v19_config_path),
                "test": "未执行",
                "deployment_eligible": False,
            },
        )
        baseline_hash = sha256_file(baseline_checkpoint_path)
        _write_json(
            paired_path,
            {
                "status": "completed_but_gate_failed",
                "test": "未执行",
                "runs": {
                    "treatment": {
                        "best_epoch": 15,
                        "best_checkpoint_sha256": baseline_hash,
                    }
                },
            },
        )
        validation_dataset = V19rResidualCacheDataset(validation_cache_path)
        baseline_evaluation = evaluate_residual(
            None,
            validation_dataset,
            device=torch.device("cpu"),
            batch_size=2,
            prefix_auxiliary_min_events=3,
        )
        _write_json(
            contract_path,
            {
                "schema_version": V19R_SCHEMA_VERSION,
                "source_v18r": {"checkpoint_epoch": 15},
                "baseline_validation_parent_metrics": baseline_evaluation[
                    "parent_metrics"
                ],
            },
        )
        config: dict[str, object] = {
            "schema_version": V19R_SCHEMA_VERSION,
            "config_name": V19R_CONFIG_NAME,
            "class_names": list(V19_CLASS_NAMES),
            "contract_path": str(contract_path),
            "expected_contract_sha256": sha256_file(contract_path),
            "readiness_path": str(readiness_path),
            "normalization_path": str(normalization_path),
            "expected_normalization_sha256": sha256_file(normalization_path),
            "train_cache_path": str(train_cache_path),
            "expected_train_cache_sha256": sha256_file(train_cache_path),
            "validation_cache_path": str(validation_cache_path),
            "expected_validation_cache_sha256": sha256_file(validation_cache_path),
            "baseline_checkpoint_path": str(baseline_checkpoint_path),
            "expected_baseline_checkpoint_sha256": baseline_hash,
            "expected_baseline_backbone_state_sha256": "1" * 64,
            "baseline_training_summary_path": str(baseline_summary_path),
            "expected_baseline_training_summary_sha256": sha256_file(
                baseline_summary_path
            ),
            "v18r_paired_report_path": str(paired_path),
            "expected_v18r_paired_report_sha256": sha256_file(paired_path),
            "source_v19_config_path": str(source_v19_config_path),
            "expected_source_v19_config_sha256": sha256_file(source_v19_config_path),
            "source_v19_readiness_path": str(source_v19_readiness_path),
            "expected_source_v19_readiness_sha256": sha256_file(
                source_v19_readiness_path
            ),
            "model": {
                "baseline": "frozen V18r density+stable8 best logits",
                "linear_shape": [3, 8],
                "bias": False,
                "zero_initialized": True,
                "zero_sum_across_classes": True,
                "absolute_residual_bound": V19_RESIDUAL_BOUND,
                "maximum_pairwise_logit_correction": 2.0 * V19_RESIDUAL_BOUND,
                "trainable_parameter_count": V19_TRAINABLE_PARAMETER_COUNT,
            },
            "train": {
                "seed": 42,
                "dataloader_seed": 42,
                "batch_size": 2,
                "learning_rate": 0.01,
                "optimizer": "AdamW_default_weight_decay",
                "epochs": 20,
                "early_stopping_patience": 0,
                "gradient_clip_norm": 100.0,
                "class_weights": [1.0, 1.0, 1.0],
                "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
                "prefix_auxiliary_weight": 0.1,
                "prefix_auxiliary_min_events": 3,
                "checkpoint_selection_metric": "parent_min_precision_recall_f1_then_min_drone_bird_errors_then_macro_f1",
                "save_every_epoch": True,
                "record_parent_order_hash": True,
                "reference_parent_order_sha256_by_epoch": order_hashes,
                "device": "cpu",
                "checkpoint_schema": V19R_CHECKPOINT_SCHEMA_VERSION,
                "artifact_dir": str(artifact),
                "checkpoint_path": str(artifact / "best_model.pt"),
                "last_checkpoint_path": str(artifact / "last_model.pt"),
                "epoch_checkpoint_dir": str(artifact / "epochs"),
            },
            "adoption": {
                "minimum_per_class_precision_recall_f1": 0.95,
                "strict_macro_f1_improvement_over_frozen_v18r": 0.02,
                "strict_drone_bird_weak_floor_improvement_over_frozen_v18r": 0.02,
                "drone_to_bird_must_not_increase": True,
                "bird_to_drone_must_not_increase": True,
                "minimum_bidirectional_error_reduction": 1,
                "balloon_floor_must_not_decrease": True,
            },
        }
        _write_json(config_path, config)
        _write_json(
            readiness_path,
            {
                "status": "passed",
                "config_sha256": sha256_file(config_path),
                "contract_sha256": sha256_file(contract_path),
                "baseline_checkpoint_epoch": 15,
                "baseline_checkpoint_sha256": baseline_hash,
                "baseline_backbone_state_sha256": "1" * 64,
                "checks": {"synthetic": True},
                "validation_used_for_readiness_gate": False,
                "test": "未执行",
                "deployment_eligible": False,
            },
        )
        summary = train_v19r(root, config_path)
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_v19r_report(root, config_path, report_path)
        epoch_files = sorted((artifact / "epochs").glob("epoch-*.pt"))
        if (
            summary["status"] != "completed"
            or len(summary["history"]) != 20
            or len(epoch_files) != 20
            or report["test"] != "未执行"
            or report["deployment_eligible"] is not False
        ):
            raise ValueError("V19r synthetic end-to-end pipeline contract failed")
        return {
            "training_status": summary["status"],
            "performance_status": summary["performance_status"],
            "best_epoch": int(summary["best_epoch"]),
            "epoch_checkpoint_count": len(epoch_files),
            "report_status": report["status"],
        }


def _actual_frozen_probe(root: Path) -> dict[str, object]:
    v18r = _validate_v18r_source(root)
    v19 = _validate_v19_feature_source(root, v18r)
    dataset = build_task_dataset(
        root,
        v18r["config"],
        split_name=v18r["config"].data.train_split_name,
        parent_limit=3,
    )
    loader = build_task_dataloader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    cache_positions = {
        parent_id: index
        for index, parent_id in enumerate(v19["train_cache"].parent_sample_ids.tolist())
    }
    selected_positions = [
        cache_positions[str(parent_id)] for parent_id in batch["parent_sample_ids"]
    ]
    device = torch.device("cpu")
    backbone = LiquidTaskClassifier(v18r["config"]).to(device)
    checkpoint = load_task_checkpoint(
        v18r["checkpoint_path"],
        backbone,
        expected_schema=v18r["config"].train.checkpoint_schema,
        map_location=device,
    )
    if int(checkpoint.get("epoch", -1)) != 15:
        raise ValueError("V19r actual probe did not load V18r epoch 15")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    backbone_before = tensor_state_sha256(backbone.state_dict())
    tensors = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.inference_mode():
        baseline = _forward_v18r(backbone, tensors)["sequence_logits"]
        causal, counts = normalize_stable8(
            tensors["micro_doppler_features"],
            tensors["micro_doppler_valid"],
            tensors["tr_padding_mask"],
            median=torch.tensor(v19["normalization"]["median"], dtype=torch.float32),
            scale=torch.tensor(v19["normalization"]["scale"], dtype=torch.float32),
        )
    baseline = baseline.clone()
    causal = causal.clone()
    counts = counts.clone()
    expected = torch.from_numpy(
        v19["train_cache"].causal_features[selected_positions, : causal.shape[1]]
    )
    if not torch.allclose(causal, expected, rtol=0.0, atol=2.0e-6):
        raise ValueError("V19r actual causal stable8 does not reproduce V19 inputs")
    residual = MicroDopplerLateResidual().to(device)
    initial = residual(baseline, causal, tensors["tr_padding_mask"])
    if not torch.equal(initial["sequence_logits"], baseline):
        raise ValueError("V19r actual zero-init logits differ from frozen V18r")
    objective = residual_objective(
        initial["sequence_logits"],
        {"label": tensors["labels"], "padding_mask": tensors["tr_padding_mask"]},
        class_weights=torch.tensor(v18r["config"].train.class_weights),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    if residual.linear.weight.grad is None or not torch.isfinite(
        residual.linear.weight.grad
    ).all():
        raise ValueError("V19r actual residual gradient is missing or non-finite")
    gradient_max = float(residual.linear.weight.grad.abs().max().item())
    if gradient_max <= 0.0:
        raise ValueError("V19r actual residual gradient is zero")
    backbone_after = tensor_state_sha256(backbone.state_dict())
    if backbone_before != backbone_after or any(
        parameter.grad is not None for parameter in backbone.parameters()
    ):
        raise ValueError("V19r residual probe changed the frozen V18r backbone")
    return {
        "baseline_epoch": 15,
        "actual_parent_ids": list(batch["parent_sample_ids"]),
        "actual_labels": list(batch["label_names"]),
        "initial_logits_exact": True,
        "causal_stable8_reproduced": True,
        "residual_gradient_absolute_max": gradient_max,
        "backbone_state_sha256": backbone_before,
        "backbone_unchanged": True,
        "final_valid_counts": [
            int(counts[index, int(batch["lengths"][index].item()) - 1].item())
            for index in range(len(batch["parent_sample_ids"]))
        ],
    }


def run(project_root: Path, *, prepared: bool) -> dict[str, object]:
    root = project_root.resolve()
    result: dict[str, object] = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V19r-micro-doppler-late-residual-smoke-v1",
        "mechanism": _mechanism_probe(),
        "roundtrip": _checkpoint_and_cache_roundtrip(),
        "synthetic_end_to_end": _synthetic_end_to_end(),
        "actual_frozen_probe": _actual_frozen_probe(root),
        "prepared_contract": "not_requested",
        "formal_training": "未执行",
        "test": "未执行",
    }
    if prepared:
        config_path = root / V19R_CONFIG_RELATIVE_PATH
        runtime = _load_runtime(root, config_path)
        train_cache = V19rResidualCacheDataset(runtime["resolved"]["train_cache_path"])
        validation_cache = V19rResidualCacheDataset(
            runtime["resolved"]["validation_cache_path"]
        )
        result["prepared_contract"] = {
            "status": "passed",
            "config": str(config_path),
            "config_sha256": runtime["config_sha256"],
            "provenance": _provenance(runtime),
            "train_parent_count": len(train_cache),
            "validation_parent_count": len(validation_cache),
            "parent_overlap_count": len(
                set(train_cache.parent_sample_ids)
                & set(validation_cache.parent_sample_ids)
            ),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused contract Smoke for the V19r late-residual experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    run(args.project_root.resolve(), prepared=args.prepared)


if __name__ == "__main__":
    main()
