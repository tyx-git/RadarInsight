"""r01 Excel before/after 逐点 soft-gate 评估。

该模块只读加载现有 r20 checkpoint 与 r20 validation parent，所有报告写入 r01。
X5A 不进入这条分类评估链；X5A 仅有独立的格式/质量诊断。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from v27.V27d.r17.data import R17Dataset
from v27.V27d.r17.protocol import load_config as load_r17_config
from v27.V27d.r20.data import R20Dataset, build_r20_loader, r20_model_input_kwargs
from v27.V27d.r20.protocol import R20_ARMS, load_config as load_r20_config
from v27.V27d.r20.training import load_checkpoint

from .denoising import DenoisingConfig, fit_denoiser, score_sequence

CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")


def _metrics(true_ids: list[int], pred_ids: list[int]) -> dict[str, Any]:
    true = np.asarray(true_ids, dtype=np.int64)
    pred = np.asarray(pred_ids, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(CLASS_NAMES):
        actual = true == index
        selected = pred == index
        tp = int(np.sum(actual & selected))
        fp = int(np.sum(~actual & selected))
        fn = int(np.sum(actual & ~selected))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        rows.append({
            "class": name,
            "support": int(np.sum(actual)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, selected in zip(true, pred, strict=True):
        matrix[int(actual), int(selected)] += 1
    return {
        "support": int(true.size),
        "accuracy": float(np.mean(true == pred)) if true.size else 0.0,
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "per_class": rows,
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
    }


def _apply_event_weights(inputs: dict[str, torch.Tensor], weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """逐点 soft gate；不改变 dt、padding 和事件顺序。"""

    result = dict(inputs)
    event_weights = weights.to(
        device=inputs["tr_features"].device,
        dtype=inputs["tr_features"].dtype,
    )
    for key in (
        "tr_features",
        "rd_frames",
        "rd_quality",
        "rd_physical_features",
        "micro_doppler_features",
        "radial_sidecar_values",
    ):
        value = result.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            continue
        if value.shape[:2] != event_weights.shape:
            continue
        gate = event_weights
        while gate.ndim < value.ndim:
            gate = gate.unsqueeze(-1)
        result[key] = value * gate
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0, "removed_fraction": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p50": float(np.quantile(array, 0.50)),
        "p99": float(np.quantile(array, 0.99)),
        "removed_fraction": float(np.mean(array <= 0.0)),
    }


def evaluate_excel(
    project_root: str | Path,
    *,
    arm: str = "zero_control",
    seed: int = 42,
    device: str = "cpu",
    target_retention_floor: float = 0.985,
) -> dict[str, Any]:
    """执行真实 r20 checkpoint 的逐点 gate 前后 logits/argmax 对照。"""

    root = Path(project_root).resolve()
    if arm not in R20_ARMS:
        raise ValueError(f"r01 Excel evaluation arm is invalid: {arm}")
    r17_config = load_r17_config(root)
    r17_train = R17Dataset(root, r17_config, task="l1", split_name="train")
    r17_all = R17Dataset(root, r17_config, task="l1", split_name=None)
    calibration = fit_denoiser(
        r17_train,
        config=DenoisingConfig(target_retention_floor=float(target_retention_floor)),
    )
    scored_by_parent = {
        sequence.parent_sample_id: score_sequence(calibration, sequence)
        for sequence in r17_all
        if sequence.label_name in CLASS_NAMES
    }
    r20_config = load_r20_config(root)
    model, checkpoint = load_checkpoint(root, r20_config, arm=arm, seed=int(seed))
    model.to(torch.device(device))
    model.eval()
    dataset = R20Dataset(root, r20_config, split_name="validation", arm=arm)
    loader = build_r20_loader(
        dataset,
        batch_size=int(r20_config.training["batch_size"]),
        shuffle=False,
    )
    before_true: list[int] = []
    after_true: list[int] = []
    before_pred: list[int] = []
    after_pred: list[int] = []
    before_logits: list[np.ndarray] = []
    after_logits: list[np.ndarray] = []
    weights_by_label: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    removed_by_label: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    events_by_label: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    parent_alignment = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = r20_model_input_kwargs(batch, torch.device(device))
            pointwise: list[np.ndarray] = []
            for parent_id, label in zip(batch["parent_sample_ids"], batch["label_names"], strict=True):
                result = scored_by_parent.get(str(parent_id))
                if result is None:
                    raise ValueError(f"r01 denoising result missing parent: {parent_id}")
                label = str(label)
                if label not in CLASS_NAMES:
                    raise ValueError(f"r20 validation contains non-DBB label: {label}")
                parent_alignment += 1
                pointwise.append(result.keep_weight)
                weights_by_label[label].extend(result.keep_weight.astype(float).tolist())
                removed_by_label[label] += int(result.removed_count)
                events_by_label[label] += int(len(result.keep_weight))
            width = int(batch["tr_features"].shape[1])
            weight_tensor = torch.zeros((len(pointwise), width), dtype=torch.float32)
            for row, values in enumerate(pointwise):
                length = min(width, len(values))
                weight_tensor[row, :length] = torch.from_numpy(values[:length].astype(np.float32))
            base = model(**inputs)["window_logits"].detach().cpu()
            gated = model(**_apply_event_weights(inputs, weight_tensor))["window_logits"].detach().cpu()
            before_logits.append(base.numpy())
            after_logits.append(gated.numpy())
            true = [int(value) for value in batch["l2_labels"].tolist()]
            before_true.extend(true)
            after_true.extend(true)
            before_pred.extend(base.argmax(dim=-1).tolist())
            after_pred.extend(gated.argmax(dim=-1).tolist())
    before_logits_array = np.concatenate(before_logits, axis=0) if before_logits else np.zeros((0, 3))
    after_logits_array = np.concatenate(after_logits, axis=0) if after_logits else np.zeros((0, 3))
    delta = after_logits_array - before_logits_array
    return {
        "schema_version": "RADAR-LN3-V27d-r01-excel-pointwise-evaluation-v1",
        "experiment_id": "V27d-r01",
        "arm": arm,
        "seed": int(seed),
        "device": str(device),
        "scope": {
            "classification_dataset": "OriData/excel/TR-only-processed-r17",
            "evaluation_split": "r20 validation parent_final",
            "training_parent_source": "r17 train",
            "denoiser_fit_uses_validation": False,
            "parent_alignment_count": parent_alignment,
            "checkpoint_epoch": int(checkpoint["epoch"]),
        },
        "denoiser": {
            "threshold": float(calibration.threshold),
            "threshold_selection": dict(calibration.threshold_selection),
            "keep_weight_by_label": {name: _summary(values) for name, values in sorted(weights_by_label.items())},
            "removed_by_label": removed_by_label,
            "events_by_label": events_by_label,
        },
        "before_denoising": _metrics(before_true, before_pred),
        "after_denoising": _metrics(after_true, after_pred),
        "logit_audit": {
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
            "prediction_changed_count": int(np.sum(np.asarray(before_pred) != np.asarray(after_pred))),
            "prediction_changed_fraction": float(np.mean(np.asarray(before_pred) != np.asarray(after_pred))) if before_pred else 0.0,
            "identity_gate": bool(np.allclose(before_logits_array, after_logits_array)),
        },
        "x5a_to_excel_mapping": False,
        "classification_effect_measured": True,
        "limitations": [
            "X5A 原始点集没有进入这条 Excel TR 分类链。",
            "X5A 没有逐点类别标签，因此不能用 X5A 直接计算 Drone/Bird P/R。",
            "keep_weight 的事件级保留率不是 parent-final 分类 Recall。",
        ],
    }


__all__ = ["evaluate_excel"]
