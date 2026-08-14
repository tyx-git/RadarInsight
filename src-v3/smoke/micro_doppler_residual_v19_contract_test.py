from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from build.micro_doppler_residual_v19 import (
    _fit_parent_balanced_normalization,
    _forward_control,
    _validate_source_pair,
)
from radar_ln3.micro_doppler_residual import (
    MicroDopplerLateResidual,
    ResidualCacheDataset,
    V19_CACHE_FIELDS,
    V19_CACHE_SCHEMA_VERSION,
    V19_CLASS_NAMES,
    V19_CONFIG_RELATIVE_PATH,
    V19_RESIDUAL_BOUND,
    V19_SCHEMA_VERSION,
    V19_FEATURE_DIM,
    V19_TRAINABLE_PARAMETER_COUNT,
    build_residual_dataloader,
    causal_valid_mean,
    evaluate_residual,
    load_residual_checkpoint,
    normalize_stable8,
    residual_objective,
    require_exact_parent_order,
    save_residual_checkpoint,
    sha256_file,
    tensor_state_sha256,
)
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
)
from report.micro_doppler_residual_v19 import run as run_v19_report
from train.micro_doppler_residual_v19 import (
    _load_runtime,
    _provenance,
    train as train_v19,
)
from train.tasks import parent_order_sha256


class _ParentIdDataset(Dataset[str]):
    def __init__(self, parent_ids: list[str]) -> None:
        self.parent_ids = list(parent_ids)

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> str:
        return self.parent_ids[index]


def _synthetic_causality() -> dict[str, object]:
    values = torch.arange(1, 1 + 4 * V19_FEATURE_DIM, dtype=torch.float32).reshape(
        1, 4, V19_FEATURE_DIM
    )
    valid = torch.tensor([[True, False, True, True]])
    padding = torch.tensor([[False, False, False, False]])
    causal, counts = causal_valid_mean(values, valid, padding)
    changed = values.clone()
    changed[:, 3] += 10000.0
    changed_causal, _ = causal_valid_mean(changed, valid, padding)
    if not torch.equal(causal[:, :3], changed_causal[:, :3]):
        raise ValueError("V19 causal stable8 summary reads a future event")
    if not torch.equal(causal[:, 1], causal[:, 0]):
        raise ValueError("V19 invalid current event does not hold prior evidence")
    if counts.tolist() != [[1, 1, 2, 3]]:
        raise ValueError(f"V19 causal counts differ: {counts.tolist()}")
    empty_valid = torch.zeros((1, 2), dtype=torch.bool)
    empty_padding = torch.zeros((1, 2), dtype=torch.bool)
    empty, empty_counts = causal_valid_mean(
        torch.ones((1, 2, V19_FEATURE_DIM)), empty_valid, empty_padding
    )
    if bool((empty != 0.0).any()) or bool((empty_counts != 0).any()):
        raise ValueError("V19 pre-evidence causal state must be exactly zero")
    return {
        "future_perturbation_preserves_prior_prefixes": True,
        "invalid_event_holds_prior_summary": True,
        "pre_evidence_summary_is_zero": True,
        "counts": counts.tolist(),
    }


def _synthetic_residual() -> dict[str, object]:
    model = MicroDopplerLateResidual()
    if model.linear.bias is not None:
        raise ValueError("V19 residual linear layer must not have a bias")
    if model.trainable_parameter_count != V19_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("V19 residual does not expose exactly 24 trainable weights")
    baseline = torch.randn((2, 5, 3), generator=torch.Generator().manual_seed(7))
    features = torch.randn((2, 5, 8), generator=torch.Generator().manual_seed(8))
    padding = torch.tensor(
        [[False, False, False, False, False], [False, False, False, True, True]]
    )
    initial = model(baseline, features, padding)
    if not torch.equal(initial["sequence_logits"], baseline):
        raise ValueError("V19 zero initialization does not preserve baseline logits exactly")
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.arange(1, 25, dtype=torch.float32).reshape(3, 8) * 100.0
        )
    output = model(baseline, features, padding)
    residual = output["residual_logits"]
    active = ~padding
    absolute_max = float(residual[active].abs().max().item())
    zero_sum_max = float(residual[active].sum(dim=-1).abs().max().item())
    pairwise_max = float(
        (residual[active].amax(dim=-1) - residual[active].amin(dim=-1)).max().item()
    )
    if absolute_max > V19_RESIDUAL_BOUND + 2.0e-6:
        raise ValueError("V19 residual exceeds its absolute bound")
    if zero_sum_max > 2.0e-6:
        raise ValueError("V19 residual violates zero-sum")
    if pairwise_max > 2.0 * V19_RESIDUAL_BOUND + 2.0e-6:
        raise ValueError("V19 residual exceeds its pairwise bound")
    if bool((residual[padding] != 0.0).any()):
        raise ValueError("V19 padded residual must be zero")
    return {
        "zero_initialization_exact": True,
        "trainable_parameter_count": model.trainable_parameter_count,
        "absolute_max": absolute_max,
        "zero_sum_absolute_max": zero_sum_max,
        "pairwise_max": pairwise_max,
    }


def _checkpoint_roundtrip() -> dict[str, object]:
    model = MicroDopplerLateResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    baseline = torch.zeros((2, 4, 3))
    features = torch.randn((2, 4, 8), generator=torch.Generator().manual_seed(9))
    padding = torch.zeros((2, 4), dtype=torch.bool)
    labels = torch.tensor([0, 1])
    output = model(baseline, features, padding)
    objective = residual_objective(
        output["sequence_logits"],
        {"label": labels, "padding_mask": padding},
        class_weights=torch.ones(3),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    optimizer.step()
    provenance = {"probe": "v19"}
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v19-smoke-") as directory:
        path = Path(directory) / "checkpoint.pt"
        save_residual_checkpoint(
            path,
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
        payload = load_residual_checkpoint(
            path,
            restored,
            optimizer=restored_optimizer,
            expected_config_name="smoke",
            expected_config_sha256="0" * 64,
            expected_provenance=provenance,
        )
        if not torch.equal(model.linear.weight, restored.linear.weight):
            raise ValueError("V19 checkpoint round-trip changed residual weights")
    return {
        "checkpoint_roundtrip": True,
        "epoch": int(payload["epoch"]),
        "gradient_absolute_max": float(model.linear.weight.grad.abs().max().item()),
    }


def _cache_roundtrip() -> dict[str, object]:
    parent_count = 3
    sequence = 4
    labels = np.arange(parent_count, dtype=np.int64)
    padding = np.asarray(
        [
            [False, False, False, False],
            [False, False, False, True],
            [False, False, True, True],
        ],
        dtype=np.bool_,
    )
    valid = ~padding
    counts = np.cumsum(valid, axis=1, dtype=np.int64)
    counts[padding] = 0
    arrays = {
        "schema_version": np.asarray(V19_CACHE_SCHEMA_VERSION),
        "split_name": np.asarray("train"),
        "parent_sample_ids": np.asarray(["a", "b", "c"]),
        "track_ids": np.asarray(["1", "2", "3"]),
        "source_names": np.asarray(["cq08", "cq08", "cq08"]),
        "label_names": np.asarray(V19_CLASS_NAMES),
        "labels": labels,
        "lengths": (~padding).sum(axis=1).astype(np.int64),
        "padding_mask": padding,
        "micro_doppler_valid": valid,
        "valid_counts": counts.astype(np.int16),
        "baseline_logits": np.zeros((parent_count, sequence, 3), dtype=np.float32),
        "causal_features": np.where(
            (counts > 0)[..., None],
            np.ones((parent_count, sequence, 8), dtype=np.float32),
            0.0,
        ).astype(np.float32),
    }
    if tuple(arrays) != V19_CACHE_FIELDS:
        raise ValueError("V19 synthetic cache field order differs")
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v19-cache-smoke-") as directory:
        path = Path(directory) / "cache.npz"
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        dataset = ResidualCacheDataset(path)
        if len(dataset) != parent_count or dataset[1]["length"].item() != 3:
            raise ValueError("V19 residual cache round-trip differs")
    return {"cache_roundtrip": True, "parent_count": parent_count}


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
    features = np.zeros((parent_count, sequence, 8), dtype=np.float32)
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
        "schema_version": np.asarray(V19_CACHE_SCHEMA_VERSION),
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
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _synthetic_end_to_end_pipeline() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v19-e2e-smoke-") as directory:
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
        artifact = root / "artifact"
        report_path = root / "report.json"
        _synthetic_cache(train_cache_path, split="train", prefix="train")
        _synthetic_cache(validation_cache_path, split="val", prefix="val")
        normalization = {"schema_version": "synthetic", "fit_population": "train"}
        _write_json(normalization_path, normalization)
        baseline_checkpoint_path.write_bytes(b"synthetic-frozen-control")
        train_dataset = ResidualCacheDataset(train_cache_path)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        loader = build_residual_dataloader(
            train_dataset, batch_size=2, shuffle=True, generator=generator
        )
        order_hashes: dict[str, str] = {}
        baseline_history: list[dict[str, object]] = []
        for epoch in range(1, 21):
            order: list[str] = []
            for batch in loader:
                order.extend(str(value) for value in batch["parent_sample_id"])
            digest = parent_order_sha256(order)
            order_hashes[str(epoch)] = digest
            baseline_history.append(
                {"epoch": epoch, "parent_order_sha256": digest}
            )
        baseline_summary = {
            "status": "completed",
            "history": baseline_history,
            "best_epoch": 1,
        }
        _write_json(baseline_summary_path, baseline_summary)
        _write_json(
            paired_path,
            {
                "status": "completed_but_gate_failed",
                "test": "未执行",
                "OOF": "未执行",
                "full_cascade": "未执行",
            },
        )
        validation_dataset = ResidualCacheDataset(validation_cache_path)
        baseline_evaluation = evaluate_residual(
            None,
            validation_dataset,
            device=torch.device("cpu"),
            batch_size=2,
            prefix_auxiliary_min_events=3,
        )
        contract = {
            "schema_version": V19_SCHEMA_VERSION,
            "source": {"control_checkpoint_epoch": 1},
            "baseline_validation_parent_metrics": baseline_evaluation[
                "parent_metrics"
            ],
        }
        _write_json(contract_path, contract)
        config: dict[str, object] = {
            "schema_version": V19_SCHEMA_VERSION,
            "config_name": "RADAR-LN3-V19-cq08-dbb-specialist-micro-doppler-late-residual-seed42",
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
            "expected_baseline_checkpoint_sha256": sha256_file(
                baseline_checkpoint_path
            ),
            "expected_baseline_backbone_state_sha256": "1" * 64,
            "baseline_training_summary_path": str(baseline_summary_path),
            "expected_baseline_training_summary_sha256": sha256_file(
                baseline_summary_path
            ),
            "v18_paired_report_path": str(paired_path),
            "expected_v18_paired_report_sha256": sha256_file(paired_path),
            "model": {
                "baseline": "frozen V18 Control best logits",
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
                "checkpoint_schema": "RADAR-LN3-V19-dbb-micro-doppler-late-residual-checkpoint-v1",
                "artifact_dir": str(artifact),
                "checkpoint_path": str(artifact / "best_model.pt"),
                "last_checkpoint_path": str(artifact / "last_model.pt"),
                "epoch_checkpoint_dir": str(artifact / "epochs"),
            },
            "adoption": {
                "minimum_per_class_precision_recall_f1": 0.95,
                "strict_macro_f1_improvement_over_frozen_control": 0.02,
                "strict_drone_bird_weak_floor_improvement_over_frozen_control": 0.02,
                "drone_to_bird_must_not_increase": True,
                "bird_to_drone_must_not_increase": True,
                "minimum_bidirectional_error_reduction": 1,
            },
            "scope": {
                "formal_training": "manual_only",
                "validation_checkpoint_selection": True,
                "validation_threshold_tuning": False,
                "test": "未执行",
                "deployment_eligible": False,
            },
        }
        _write_json(config_path, config)
        readiness = {
            "status": "passed",
            "config_sha256": sha256_file(config_path),
            "contract_sha256": sha256_file(contract_path),
            "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint_path),
            "baseline_backbone_state_sha256": "1" * 64,
            "checks": {"synthetic": True},
            "validation_used_for_readiness_gate": False,
            "test": "未执行",
            "deployment_eligible": False,
        }
        _write_json(readiness_path, readiness)
        summary = train_v19(root, config_path)
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_v19_report(root, config_path, report_path)
        epoch_files = sorted((artifact / "epochs").glob("epoch-*.pt"))
        if (
            summary["status"] != "completed"
            or len(summary["history"]) != 20
            or len(epoch_files) != 20
            or report["test"] != "未执行"
            or report["deployment_eligible"] is not False
        ):
            raise ValueError("V19 synthetic end-to-end pipeline contract failed")
        return {
            "training_status": summary["status"],
            "performance_status": summary["performance_status"],
            "best_epoch": int(summary["best_epoch"]),
            "epoch_checkpoint_count": len(epoch_files),
            "report_status": report["status"],
            "report_test": report["test"],
            "deployment_eligible": report["deployment_eligible"],
        }


def _actual_frozen_probe(root: Path) -> dict[str, object]:
    source = _validate_source_pair(root)
    normalization = _fit_parent_balanced_normalization(root, source["feature"])
    full_dataset = build_task_dataset(
        root,
        source["feature"],
        split_name=source["feature"].data.train_split_name,
    )
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(42)
    order_loader = DataLoader(
        _ParentIdDataset(list(full_dataset.parent_ids)),
        batch_size=source["feature"].train.batch_size,
        shuffle=True,
        num_workers=0,
        generator=order_generator,
    )
    order_hashes: list[str] = []
    for epoch in range(1, 21):
        order: list[str] = []
        for values in order_loader:
            order.extend(str(value) for value in values)
        order_hashes.append(
            require_exact_parent_order(
                order, source["order_hashes"][str(epoch)], epoch=epoch
            )
        )
    dataset = build_task_dataset(
        root,
        source["feature"],
        split_name=source["feature"].data.train_split_name,
        parent_limit=3,
    )
    loader = build_task_dataloader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    device = torch.device("cpu")
    control = LiquidTaskClassifier(source["control"]).to(device)
    load_task_checkpoint(
        source["checkpoint_path"],
        control,
        expected_schema=source["control"].train.checkpoint_schema,
        map_location=device,
    )
    for parameter in control.parameters():
        parameter.requires_grad_(False)
    control.eval()
    backbone_before = tensor_state_sha256(control.state_dict())
    tensors = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.inference_mode():
        baseline = _forward_control(control, tensors)["sequence_logits"]
        causal, counts = normalize_stable8(
            tensors["micro_doppler_features"],
            tensors["micro_doppler_valid"],
            tensors["tr_padding_mask"],
            median=torch.tensor(normalization["median"], dtype=torch.float32),
            scale=torch.tensor(normalization["scale"], dtype=torch.float32),
        )
    baseline = baseline.clone()
    causal = causal.clone()
    counts = counts.clone()
    residual = MicroDopplerLateResidual().to(device)
    initial = residual(baseline, causal, tensors["tr_padding_mask"])
    if not torch.equal(initial["sequence_logits"], baseline):
        raise ValueError("V19 actual zero-init logits differ from frozen V18 Control")
    optimizer = torch.optim.AdamW(residual.parameters(), lr=1.0e-4)
    weights = torch.tensor(source["control"].train.class_weights, dtype=torch.float32)
    objective = residual_objective(
        initial["sequence_logits"],
        {"label": tensors["labels"], "padding_mask": tensors["tr_padding_mask"]},
        class_weights=weights,
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    if residual.linear.weight.grad is None or not torch.isfinite(
        residual.linear.weight.grad
    ).all():
        raise ValueError("V19 actual residual gradient is missing or non-finite")
    gradient_max = float(residual.linear.weight.grad.abs().max().item())
    if gradient_max <= 0.0:
        raise ValueError("V19 actual residual gradient is zero")
    optimizer.step()
    backbone_after = tensor_state_sha256(control.state_dict())
    if backbone_before != backbone_after:
        raise ValueError("V19 residual optimizer changed the frozen Control backbone")
    if any(parameter.grad is not None for parameter in control.parameters()):
        raise ValueError("V19 frozen Control received a gradient")
    return {
        "actual_parent_ids": list(batch["parent_sample_ids"]),
        "actual_labels": list(batch["label_names"]),
        "train_normalization_parent_count": int(normalization["parent_count"]),
        "all_20_parent_orders_match_frozen_v18_control": True,
        "parent_order_sha256": order_hashes,
        "initial_logits_exact": True,
        "residual_gradient_absolute_max": gradient_max,
        "backbone_state_sha256_before": backbone_before,
        "backbone_state_sha256_after": backbone_after,
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
        "schema_version": "RADAR-LN3-V19-micro-doppler-late-residual-smoke-v1",
        "synthetic_causality": _synthetic_causality(),
        "synthetic_residual": _synthetic_residual(),
        "checkpoint": _checkpoint_roundtrip(),
        "cache": _cache_roundtrip(),
        "synthetic_end_to_end_pipeline": _synthetic_end_to_end_pipeline(),
        "actual_frozen_probe": _actual_frozen_probe(root),
        "prepared_contract": "not_requested",
        "formal_training": "未执行",
        "test": "未执行",
    }
    if prepared:
        config_path = root / V19_CONFIG_RELATIVE_PATH
        runtime = _load_runtime(root, config_path)
        train_cache = ResidualCacheDataset(runtime["resolved"]["train_cache_path"])
        validation_cache = ResidualCacheDataset(
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
        description="Focused contract Smoke for the V19 late-residual experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    run(args.project_root.resolve(), prepared=args.prepared)


if __name__ == "__main__":
    main()
