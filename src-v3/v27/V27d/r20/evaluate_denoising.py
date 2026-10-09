"""评估 V27d-r20 模型接入密度-空间联合降噪前后的指标。

用法（项目根目录执行）：

    PYTHONPATH=src-v3 .venv/Scripts/python.exe \
      -m v27.V27d.r20.evaluate_denoising --arm zero_control

评估严格使用 r20 validation parent；降噪器只从 r17 train parent 拟合。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v27.V27d.r17.data import R17Dataset
from v27.V27d.r17.protocol import load_config as load_r17_config
from v27.V27d.r20.data import R20Dataset, build_r20_loader, r20_model_input_kwargs
from v27.V27d.r20.denoising import DenoisingConfig, fit_denoiser, score_sequence
from v27.V27d.r20.protocol import R20_ARMS, load_config as load_r20_config
from v27.V27d.r20.training import load_checkpoint


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
        rows.append(
            {
                "class": name,
                "support": int(np.sum(actual)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "support": int(true.size),
        "accuracy": float(np.mean(true == pred)) if true.size else 0.0,
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "per_class": rows,
    }


def _apply_event_weights(inputs: dict[str, torch.Tensor], weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """软门控事件证据，不改变时间顺序和 padding 语义。"""

    result = dict(inputs)
    event_weights = weights.to(device=inputs["tr_features"].device, dtype=inputs["tr_features"].dtype)
    for key in (
        "tr_features",
        "rd_frames",
        "rd_quality",
        "rd_physical_features",
        "micro_doppler_features",
        "radial_sidecar_values",
    ):
        if key not in result:
            continue
        value = result[key]
        if value.ndim < 2 or value.shape[1] != event_weights.shape[1]:
            continue
        gate = event_weights
        while gate.ndim < value.ndim:
            gate = gate.unsqueeze(-1)
        result[key] = value * gate
    return result


def evaluate_denoising(
    project_root: str | Path,
    *,
    arm: str = "zero_control",
    seed: int = 42,
    target_retention_floor: float = 0.985,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if arm not in R20_ARMS:
        raise ValueError(f"降噪评估 arm 无效：{arm}")
    r17_config = load_r17_config(root)
    r17_train = R17Dataset(root, r17_config, task="l1", split_name="train")
    # r20 继承 r18 的 source-stratified parent 划分；用全量 r17 只做
    # 事件分数回连，不参与任何阈值拟合。
    r17_all = R17Dataset(root, r17_config, task="l1", split_name=None)
    denoiser = fit_denoiser(
        r17_train,
        config=DenoisingConfig(target_retention_floor=float(target_retention_floor)),
    )
    denoising_by_parent = {
        sequence.parent_sample_id: score_sequence(denoiser, sequence)
        for sequence in r17_all
        if sequence.label_name in CLASS_NAMES
    }
    r20_config = load_r20_config(root)
    model, checkpoint = load_checkpoint(root, r20_config, arm=arm, seed=int(seed))
    model.eval()
    dataset = R20Dataset(root, r20_config, split_name="validation", arm=arm)
    loader = build_r20_loader(
        dataset,
        batch_size=int(r20_config.training["batch_size"]),
        shuffle=False,
    )
    before_true: list[int] = []
    before_pred: list[int] = []
    after_true: list[int] = []
    after_pred: list[int] = []
    weight_values: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    parent_alignment = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = r20_model_input_kwargs(batch, torch.device("cpu"))
            weights: list[np.ndarray] = []
            for parent_id, label in zip(batch["parent_sample_ids"], batch["label_names"], strict=True):
                result = denoising_by_parent.get(str(parent_id))
                if result is None:
                    raise ValueError(f"降噪验证结果缺少 parent：{parent_id}")
                if str(label) not in CLASS_NAMES:
                    raise ValueError(f"r20 validation 出现非 DBB 类别：{label}")
                parent_alignment += 1
                values = result.keep_weight
                weights.append(values)
                weight_values[str(label)].extend(values.tolist())
            width = int(batch["tr_features"].shape[1])
            weight_tensor = torch.zeros((len(weights), width), dtype=torch.float32)
            for row, values in enumerate(weights):
                length = min(width, len(values))
                weight_tensor[row, :length] = torch.from_numpy(values[:length].astype(np.float32))
            base_output = model(**inputs)["window_logits"]
            denoised_output = model(**_apply_event_weights(inputs, weight_tensor))["window_logits"]
            true = batch["l2_labels"].tolist()
            before_true.extend(int(value) for value in true)
            after_true.extend(int(value) for value in true)
            before_pred.extend(base_output.argmax(dim=-1).tolist())
            after_pred.extend(denoised_output.argmax(dim=-1).tolist())
    report = {
        "schema_version": "RADAR-LN3-V27d-r20-denoising-evaluation-v1",
        "experiment_id": "V27d-r20",
        "arm": arm,
        "seed": int(seed),
        "scope": {
            "training_parent_source": "r17 train, task=l1, known classes plus ClutterTarget",
            "denoiser_score_lookup_source": "r17 all parents, no refit",
            "evaluation_parent_source": "r20 validation, task=l2",
            "denoiser_fit_uses_validation": False,
            "model_checkpoint_epoch": int(checkpoint["epoch"]),
            "soft_event_gate": True,
            "parent_alignment_count": parent_alignment,
        },
        "denoiser": {
            "threshold": float(denoiser.threshold),
            "threshold_selection": dict(denoiser.threshold_selection),
            "train_event_counts": dict(denoiser.train_event_counts),
            "validation_keep_weight": {
                label: {
                    "count": len(values),
                    "mean": float(np.mean(values)) if values else 0.0,
                    "p01": float(np.quantile(values, 0.01)) if values else 0.0,
                    "p50": float(np.quantile(values, 0.50)) if values else 0.0,
                    "removed_fraction": float(np.mean(np.asarray(values) <= 0.0)) if values else 0.0,
                }
                for label, values in weight_values.items()
            },
        },
        "before_denoising": _metrics(before_true, before_pred),
        "after_denoising": _metrics(after_true, after_pred),
    }
    for before, after in zip(
        report["before_denoising"]["per_class"],
        report["after_denoising"]["per_class"],
        strict=True,
    ):
        after["delta_precision"] = after["precision"] - before["precision"]
        after["delta_recall"] = after["recall"] - before["recall"]
        after["delta_f1"] = after["f1"] - before["f1"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 V27d-r20 降噪前后 Drone/Bird 指标")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--arm", choices=R20_ARMS, default="zero_control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-retention-floor", type=float, default=0.985)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_denoising(
        args.project_root,
        arm=args.arm,
        seed=args.seed,
        target_retention_floor=args.target_retention_floor,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
