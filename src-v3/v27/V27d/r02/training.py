"""V27d-r02 Bird→Drone 混淆感知训练。"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v27.V27d.r20.data import R20Dataset, build_r20_loader, r20_model_input_kwargs
from v27.V27d.r20.protocol import load_config as load_r20_config
from v27.V27d.r20.training import load_checkpoint as load_r20_checkpoint
from v27.v27_baseline.metrics import summarize_records

from .calibration import calibration_metrics, fit_class_conditional_calibration
from .losses import combined_loss
from .model import R02BirdDroneModel
from .protocol import (
    R02_CHECKPOINT_SCHEMA,
    R02_CLASSES,
    R02_REPORT_SCHEMA,
    R02_TRAINING_HARD_NEGATIVE_IDS,
    R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS,
    R02Config,
    load_json,
    prepare_manifest,
    project_path,
    sha256_file,
    write_json,
)


def _seed(seed: int) -> None:
    value = int(seed)
    if value < 0:
        raise ValueError("r02 seed must be non-negative")
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device(requested: str | None) -> torch.device:
    value = str(requested or "cuda").strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("r02 requested CUDA but CUDA is unavailable")
        return torch.device(value if value.startswith("cuda:") else "cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r02 device: {value}")


def _class_weights(dataset: R20Dataset, device: torch.device) -> torch.Tensor:
    counts = {name: 0 for name in R02_CLASSES}
    for item in dataset:
        label = item.sequence.label_name
        if label not in counts:
            raise ValueError(f"r02 received non-DBB label: {label}")
        counts[label] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"r02 training support is incomplete: {counts}")
    total = float(sum(counts.values()))
    return torch.tensor(
        [total / (len(counts) * counts[name]) for name in R02_CLASSES],
        dtype=torch.float32,
        device=device,
    )


def _parent_weights(dataset: R20Dataset, *, hard_negative_weight: float, hard_negative_ids: set[str], device: torch.device) -> torch.Tensor:
    values: list[float] = []
    for item in dataset:
        parent_id = str(item.sequence.parent_sample_id)
        if parent_id in hard_negative_ids and item.sequence.label_name != "BirdTarget":
            raise ValueError(f"r02 hard-negative is not BirdTarget: {parent_id}")
        values.append(float(hard_negative_weight if parent_id in hard_negative_ids else 1.0))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _freeze_trajectory_heads(model: R02BirdDroneModel) -> None:
    heads = getattr(model.base_model.base_model, "trajectory_heads", None)
    if heads is None:
        raise ValueError("r02 base model has no trajectory heads")
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)


def _metrics(true_ids: list[int], pred_ids: list[int]) -> dict[str, Any]:
    records = [
        {"true_id": int(true), "pred_id": int(pred), "parent_sample_id": str(index)}
        for index, (true, pred) in enumerate(zip(true_ids, pred_ids, strict=True))
    ]
    return summarize_records(records, class_names=R02_CLASSES, scope="parent_final")


def _records_from_logits(
    dataset: R20Dataset,
    logits: torch.Tensor,
    *,
    calibrated_probabilities: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    values = logits.detach().cpu()
    if calibrated_probabilities is None:
        probabilities = torch.softmax(values, dim=-1)
    else:
        probabilities = calibrated_probabilities.detach().cpu()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        final = values[index]
        probability = probabilities[index]
        rows.append(
            {
                "parent_sample_id": str(item.sequence.parent_sample_id),
                "source_suffix": str(item.sequence.parent_sample_id).split("|")[2],
                "label_name": str(item.sequence.label_name),
                "true_id": int(item.sequence.l2_label),
                "pred_id": int(probability.argmax().item()),
                "probabilities": [float(value) for value in probability.tolist()],
                "logits": [float(value) for value in final.tolist()],
                "event_count": int(item.sequence.event_count),
            }
        )
    if [row["parent_sample_id"] for row in rows] != dataset.parent_ids:
        raise ValueError("r02 record construction changed parent order")
    return rows


def _run_model(
    model: R02BirdDroneModel,
    dataset: R20Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    final_logits: list[torch.Tensor] = []
    reliabilities: list[torch.Tensor] = []
    maturities: list[torch.Tensor] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in build_r20_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r20_model_input_kwargs(batch, device))
            final_logits.append(output["window_logits"].detach().cpu())
            if "bird_reliability" in output:
                reliabilities.append(output["bird_reliability"].detach().cpu())
            if "maturity" in output:
                maturities.append(output["maturity"].detach().cpu())
            labels.extend(int(value) for value in batch["l2_labels"].tolist())
    logits = torch.cat(final_logits, dim=0) if final_logits else torch.zeros((0, len(R02_CLASSES)))
    def _pad_prefix(values: list[torch.Tensor]) -> torch.Tensor:
        if not values:
            return torch.zeros((0, 0), dtype=torch.float32)
        width = max(int(value.shape[1]) for value in values)
        output = torch.zeros((sum(int(value.shape[0]) for value in values), width), dtype=values[0].dtype)
        offset = 0
        for value in values:
            length = int(value.shape[1])
            output[offset : offset + value.shape[0], :length] = value
            offset += int(value.shape[0])
        return output

    return {
        "logits": logits,
        "labels": torch.tensor(labels, dtype=torch.long),
        "reliability": _pad_prefix(reliabilities),
        "maturity": _pad_prefix(maturities),
        "records": _records_from_logits(dataset, logits),
    }


def _run_base_model(
    model: torch.nn.Module,
    dataset: R20Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    final_logits: list[torch.Tensor] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in build_r20_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r20_model_input_kwargs(batch, device))
            final_logits.append(output["window_logits"].detach().cpu())
            labels.extend(int(value) for value in batch["l2_labels"].tolist())
    logits = torch.cat(final_logits, dim=0) if final_logits else torch.zeros((0, len(R02_CLASSES)))
    return {"logits": logits, "labels": torch.tensor(labels, dtype=torch.long), "records": _records_from_logits(dataset, logits)}


def _source_metrics(records: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row["source_suffix"]), []).append(row)
    return {
        source: {
            "support": len(rows),
            "metrics": summarize_records(rows, class_names=R02_CLASSES, scope=f"{scope}:source={source}"),
        }
        for source, rows in sorted(groups.items())
    }


def _hard_negative_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = set(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS)
    return [
        {
            "parent_sample_id": row["parent_sample_id"],
            "source_suffix": row["source_suffix"],
            "label_name": row["label_name"],
            "baseline_or_after_prediction": row["pred_id"],
            "is_bird_to_drone": row["label_name"] == "BirdTarget" and row["pred_id"] == 0,
            "event_count": row["event_count"],
        }
        for row in records
        if row["parent_sample_id"] in selected
    ]


def _collect_train_logits(model: R02BirdDroneModel, dataset: R20Dataset, *, device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    result = _run_model(model, dataset, device=device, batch_size=batch_size)
    return result["logits"], result["labels"]


def train(
    project_root: str | Path,
    config: R02Config,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """从 r20 zero_control checkpoint 开始，训练完整 parent 序列模型。"""
    root = Path(project_root).resolve()
    _seed(int(config.training["seed"]))
    runtime_device = _device(device)
    manifest_path, manifest = prepare_manifest(root, config)
    r20_config = load_r20_config(root, config.source["r20_config_path"])
    base_model, base_payload = load_r20_checkpoint(
        root,
        r20_config,
        arm="zero_control",
        seed=42,
        checkpoint_path=project_path(root, config.source["r20_checkpoint_path"], must_exist=True),
    )
    model = R02BirdDroneModel(
        base_model.to(runtime_device),
        reliability_hidden_dim=int(config.model["reliability_hidden_dim"]),
        maturity_hidden_dim=int(config.model["maturity_hidden_dim"]),
    ).to(runtime_device)
    _freeze_trajectory_heads(model)
    train_set = R20Dataset(root, r20_config, split_name="train", arm="zero_control")
    validation_set = R20Dataset(root, r20_config, split_name="validation", arm="zero_control")
    assignments = manifest["parent_assignments"]
    if train_set.parent_ids != [key for key, value in assignments.items() if value == "train"]:
        raise ValueError("r02 train order differs from r20 sealed parent split")
    if validation_set.parent_ids != [key for key, value in assignments.items() if value == "validation"]:
        raise ValueError("r02 validation order differs from r20 sealed parent split")
    if set(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS) & set(train_set.parent_ids):
        raise ValueError("r02 validation audit IDs leaked into train")
    class_weights = _class_weights(train_set, runtime_device)
    parent_weights = _parent_weights(
        train_set,
        hard_negative_weight=float(config.training["hard_negative_weight"]),
        hard_negative_ids=set(R02_TRAINING_HARD_NEGATIVE_IDS),
        device=runtime_device,
    )
    loader = build_r20_loader(
        train_set,
        batch_size=int(config.training["batch_size"]),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(config.training["seed"])),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("r02 has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    # r20 validation 仅用于最终审计，禁止逐 epoch 参与 checkpoint 选择。
    best_key: tuple[float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    started = time.perf_counter()
    batch_index_by_id = {parent_id: index for index, parent_id in enumerate(train_set.parent_ids)}
    for epoch in range(1, int(config.training["epochs"]) + 1):
        model.train()
        model.base_model.base_model.trajectory_heads.eval()
        losses: list[float] = []
        components: dict[str, list[float]] = {"parent_ce": [], "bird_drone_margin": [], "bird_reliability": [], "maturity": []}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**r20_model_input_kwargs(batch, runtime_device))
            ids = [str(value) for value in batch["parent_sample_ids"]]
            row_weights = torch.stack([parent_weights[batch_index_by_id[parent_id]] for parent_id in ids])
            loss_values = combined_loss(
                output["sequence_logits"],
                batch["l2_labels"].to(runtime_device),
                batch["tr_padding_mask"].to(runtime_device),
                output["bird_reliability_logits"],
                output["maturity_logits"],
                class_weights=class_weights,
                parent_weights=row_weights,
                margin=float(config.training["margin"]),
                margin_weight=float(config.training["margin_weight"]),
                reliability_weight=float(config.training["reliability_weight"]),
                maturity_weight=float(config.training["maturity_weight"]),
            )
            loss = loss_values["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("r02 loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable,
                float(config.training["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            for name in components:
                components[name].append(float(loss_values[name].detach().cpu().item()))
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        train_epoch = _run_model(model, train_set, device=runtime_device, batch_size=int(config.training["batch_size"]))
        train_metrics = summarize_records(train_epoch["records"], class_names=R02_CLASSES, scope="train:r02:parent_final")
        train_bird_drone_errors = sum(
            row["label_name"] == "BirdTarget" and row["pred_id"] == 0 for row in train_epoch["records"]
        )
        key = (float(train_metrics["macro_f1"]),)
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch = key, state, epoch
        validation = _run_model(model, validation_set, device=runtime_device, batch_size=int(config.training["batch_size"]))
        validation_metrics = summarize_records(validation["records"], class_names=R02_CLASSES, scope="validation:r02:parent_final_audit")
        bird_drone_errors = sum(
            row["label_name"] == "BirdTarget" and row["pred_id"] == 0 for row in validation["records"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "loss_components": {name: float(np.mean(values)) for name, values in components.items()},
                "train_metrics": train_metrics,
                "train_bird_to_drone_errors": int(train_bird_drone_errors),
                "validation_audit_metrics": validation_metrics,
                "validation_audit_bird_to_drone_errors": int(bird_drone_errors),
                "selection_key_train_only": list(key),
            }
        )
        print(
            f"[V27d-r02] epoch={epoch}/{config.training['epochs']} "
            f"loss={np.mean(losses):.6f} train_macro_f1={train_metrics['macro_f1']:.6f} "
            f"validation_audit_macro_f1={validation_metrics['macro_f1']:.6f} "
            f"validation_audit_bird_to_drone={bird_drone_errors}",
            flush=True,
        )
    if best_state is None or best_key is None:
        raise RuntimeError("r02 produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    train_logits, train_labels = _collect_train_logits(
        model, train_set, device=runtime_device, batch_size=int(config.training["batch_size"])
    )
    calibration = fit_class_conditional_calibration(
        train_logits,
        train_labels,
        regularization=float(config.training["calibration_regularization"]),
        log_temperature_bound=float(config.training["calibration_log_temperature_bound"]),
        bias_bound=float(config.training["calibration_bias_bound"]),
    )
    calibration_path = project_path(root, config.artifacts["calibration_path"])
    write_json(calibration_path, calibration.to_json())
    validation_raw = _run_model(model, validation_set, device=runtime_device, batch_size=int(config.training["batch_size"]))
    calibrated_probabilities = calibration.probabilities(validation_raw["logits"])
    baseline_model, _ = load_r20_checkpoint(
        root,
        r20_config,
        arm="zero_control",
        seed=42,
        checkpoint_path=project_path(root, config.source["r20_checkpoint_path"], must_exist=True),
    )
    baseline = _run_base_model(baseline_model.to(runtime_device), validation_set, device=runtime_device, batch_size=int(config.training["batch_size"]))
    baseline_records = baseline["records"]
    after_records = _records_from_logits(validation_set, validation_raw["logits"], calibrated_probabilities=calibrated_probabilities)
    training_summary = {
        "schema_version": "RADAR-LN3-V27d-r02-training-summary-v1",
        "experiment_id": "V27d-r02",
        "status": "completed_train_only_selection",
        "best_epoch": int(best_epoch),
        "epochs": len(history),
        "training_parent_count": len(train_set),
        "validation_parent_count": len(validation_set),
        "checkpoint_selection": "train_parent_macro_f1_only; validation audit not used for selection",
        "validation_used_for_checkpoint_selection": False,
        "history": history,
    }
    training_summary_path = project_path(root, config.artifacts["training_report_path"])
    write_json(training_summary_path, training_summary)
    report = {
        "schema_version": R02_REPORT_SCHEMA,
        "experiment_id": "V27d-r02",
        "status": "completed_bird_drone_training",
        "seed": int(config.training["seed"]),
        "device": str(runtime_device),
        "best_epoch": int(best_epoch),
        "epochs": len(history),
        "training_summary_path": training_summary_path.relative_to(root).as_posix(),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "source_r20_checkpoint": {
            "path": config.source["r20_checkpoint_path"],
            "sha256": sha256_file(project_path(root, config.source["r20_checkpoint_path"], must_exist=True)),
            "model_state_sha256": base_payload.get("model_state_sha256"),
        },
        "training_contract": {
            "complete_parent_sequence": True,
            "event_random_split": False,
            "loader_shuffles_parents_only": True,
            "validation_used_for_training": False,
            "validation_used_for_calibration": False,
            "checkpoint_selection_uses_validation": False,
            "calibration_fit_is_post_selection_and_train_only": True,
            "training_hard_negative_count": len(R02_TRAINING_HARD_NEGATIVE_IDS),
            "validation_audit_hard_negative_count": len(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS),
        },
        "hard_negative_audit": {
            "training_ids": list(R02_TRAINING_HARD_NEGATIVE_IDS),
            "validation_ids": list(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS),
            "validation_baseline": _hard_negative_audit(baseline_records),
            "validation_after_calibration": _hard_negative_audit(after_records),
        },
        "baseline": {
            "metrics": summarize_records(baseline_records, class_names=R02_CLASSES, scope="validation:r20:baseline"),
            "source_metrics": _source_metrics(baseline_records, "validation:r20:baseline"),
        },
        "after_r02_calibration": {
            "metrics": summarize_records(after_records, class_names=R02_CLASSES, scope="validation:r02:calibrated"),
            "source_metrics": _source_metrics(after_records, "validation:r02:calibrated"),
        },
        "calibration": {
            **calibration.to_json(),
            "train_metrics": calibration_metrics(train_logits, train_labels, calibration),
            "validation_metrics": calibration_metrics(validation_raw["logits"], validation_raw["labels"], calibration),
        },
        "history": history,
        "limitations": [
            "四个指定 validation hard-negative 只做审计，没有用于训练或校准。",
            "class-conditional calibration 使用 train logits 拟合，不是严格 OOF calibration；结果需视为 train-only exploratory。",
            "validation 只有 84 个 parent，Bird→Drone 计数变化必须结合 source 分层和 bootstrap 解释。",
            "本实验使用 Excel TR-only 数据，尚未接入 X5A 原始点集。",
        ],
    }
    checkpoint_path = project_path(root, config.artifacts["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "checkpoint_schema": R02_CHECKPOINT_SCHEMA,
        "experiment_id": "V27d-r02",
        "seed": int(config.training["seed"]),
        "best_epoch": int(best_epoch),
        "config_path": config.path.relative_to(root).as_posix(),
        "config_sha256": config.sha256,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "source_r20_checkpoint_path": config.source["r20_checkpoint_path"],
        "source_r20_checkpoint_sha256": sha256_file(project_path(root, config.source["r20_checkpoint_path"], must_exist=True)),
        "model_contract": model.model_contract(),
        "model_state": best_state,
        "training_hard_negative_ids": list(R02_TRAINING_HARD_NEGATIVE_IDS),
        "validation_audit_hard_negative_ids": list(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    report["checkpoint_path"] = checkpoint_path.relative_to(root).as_posix()
    report["calibration_path"] = calibration_path.relative_to(root).as_posix()
    report["elapsed_seconds"] = time.perf_counter() - started
    report_path = project_path(root, config.artifacts["report_path"])
    write_json(report_path, report)
    return report


def load_r02_checkpoint(project_root: str | Path, config: R02Config, *, device: str | None = None) -> tuple[R02BirdDroneModel, dict[str, Any]]:
    root = Path(project_root).resolve()
    runtime_device = _device(device)
    path = project_path(root, config.artifacts["checkpoint_path"], must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != R02_CHECKPOINT_SCHEMA:
        raise ValueError("r02 checkpoint schema differs")
    r20_config = load_r20_config(root, config.source["r20_config_path"])
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
    if payload.get("model_contract") != model.model_contract():
        raise ValueError("r02 checkpoint model contract differs")
    model.load_state_dict(dict(payload["model_state"]), strict=True)
    _freeze_trajectory_heads(model)
    return model.to(runtime_device).eval(), dict(payload)


__all__ = ["load_r02_checkpoint", "train"]
