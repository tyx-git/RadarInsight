"""r02 仅使用训练集拟合的 class-conditional calibration。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .protocol import R02_CLASSES


@dataclass(frozen=True)
class ClassConditionalCalibration:
    """向量缩放校准器：每个类别一个温度和偏置。"""

    log_temperature: tuple[float, ...]
    bias: tuple[float, ...]
    class_names: tuple[str, ...] = R02_CLASSES
    fit_scope: str = "train_only"
    optimizer: str = "LBFGS"
    steps: int = 0

    def __post_init__(self) -> None:
        if len(self.log_temperature) != len(self.class_names) or len(self.bias) != len(self.class_names):
            raise ValueError("r02 calibration parameter dimensions differ from class count")
        if not np.isfinite(np.asarray(self.log_temperature, dtype=np.float64)).all():
            raise ValueError("r02 calibration log temperatures must be finite")
        if not np.isfinite(np.asarray(self.bias, dtype=np.float64)).all():
            raise ValueError("r02 calibration biases must be finite")

    @property
    def temperatures(self) -> tuple[float, ...]:
        return tuple(float(np.exp(value)) for value in self.log_temperature)

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 1 or logits.shape[-1] != len(self.class_names):
            raise ValueError("r02 calibration logits have an invalid class dimension")
        if not torch.isfinite(logits).all():
            raise ValueError("r02 calibration logits must be finite")
        log_temperature = torch.as_tensor(
            self.log_temperature, dtype=logits.dtype, device=logits.device
        )
        bias = torch.as_tensor(self.bias, dtype=logits.dtype, device=logits.device)
        return logits / torch.exp(log_temperature) + bias

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.transform_logits(logits), dim=-1)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-r02-class-conditional-calibration-v1",
            "class_names": list(self.class_names),
            "log_temperature": list(self.log_temperature),
            "temperature": list(self.temperatures),
            "bias": list(self.bias),
            "fit_scope": self.fit_scope,
            "optimizer": self.optimizer,
            "steps": int(self.steps),
            "validation_used_for_fit": False,
        }


def _validate_fit_inputs(logits: np.ndarray, labels: np.ndarray, class_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != int(class_count) or values.shape[0] == 0:
        raise ValueError("r02 calibration logits must have shape [N,C] with N>0")
    if targets.shape != (values.shape[0],) or np.any((targets < 0) | (targets >= int(class_count))):
        raise ValueError("r02 calibration labels have an invalid shape or class id")
    if not np.isfinite(values).all():
        raise ValueError("r02 calibration logits contain non-finite values")
    return torch.from_numpy(values), torch.from_numpy(targets)


def fit_class_conditional_calibration(
    logits: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    class_names: Sequence[str] = R02_CLASSES,
    max_steps: int = 100,
    regularization: float = 0.10,
    log_temperature_bound: float = 0.70,
    bias_bound: float = 1.00,
) -> ClassConditionalCalibration:
    """在 train-only parent-final logits 上拟合 vector scaling。

    参数仅用于校准，绝不读取 validation/test。温度和偏置均有限幅，且偏置
    去均值以消除 softmax 不可识别的整体平移。
    """
    names = tuple(str(value) for value in class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("r02 calibration requires unique class names")
    raw_logits = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    raw_labels = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    values, targets = _validate_fit_inputs(raw_logits, raw_labels, len(names))
    parameter = nn.Parameter(torch.zeros(len(names) * 2, dtype=torch.float32))
    temperature_bound = max(0.05, float(log_temperature_bound))
    bias_bound_value = max(0.05, float(bias_bound))
    regularization_value = max(0.0, float(regularization))
    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=0.10,
        max_iter=max(1, int(max_steps)),
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        log_temperature = parameter[: len(names)].clamp(-temperature_bound, temperature_bound)
        bias = parameter[len(names) :].clamp(-bias_bound_value, bias_bound_value)
        bias = bias - bias.mean()
        calibrated = values / torch.exp(log_temperature) + bias
        loss = nn.functional.cross_entropy(calibrated, targets)
        loss = loss + regularization_value * (log_temperature.square().mean() + bias.square().mean())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        log_temperature = parameter[: len(names)].clamp(-temperature_bound, temperature_bound)
        bias = parameter[len(names) :].clamp(-bias_bound_value, bias_bound_value)
        bias = bias - bias.mean()
    return ClassConditionalCalibration(
        log_temperature=tuple(float(value) for value in log_temperature.tolist()),
        bias=tuple(float(value) for value in bias.tolist()),
        class_names=names,
        fit_scope="train_only",
        optimizer="LBFGS",
        steps=int(max_steps),
    )


def calibration_metrics(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    calibration: ClassConditionalCalibration,
) -> dict[str, float]:
    """返回校准前后的 NLL 与 ECE，输入可来自 train 或 validation。"""
    raw_logits = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, dtype=torch.float32)
    raw_labels = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.long)
    if raw_logits.ndim != 2 or raw_labels.shape != (raw_logits.shape[0],):
        raise ValueError("r02 calibration metric input shape differs")
    with torch.no_grad():
        before_probability = torch.softmax(raw_logits, dim=-1)
        after_logits = calibration.transform_logits(raw_logits)
        after_probability = torch.softmax(after_logits, dim=-1)
        before_nll = nn.functional.cross_entropy(raw_logits, raw_labels).item()
        after_nll = nn.functional.cross_entropy(after_logits, raw_labels).item()

    def ece(probability: torch.Tensor, bins: int = 10) -> float:
        confidence, prediction = probability.max(dim=-1)
        correct = prediction.eq(raw_labels)
        result = 0.0
        for index in range(bins):
            lower = float(index) / bins
            upper = float(index + 1) / bins
            mask = (confidence >= lower) & ((confidence < upper) if index + 1 < bins else (confidence <= upper))
            if bool(mask.any()):
                result += float(mask.float().mean()) * abs(
                    float(confidence[mask].mean()) - float(correct[mask].float().mean())
                )
        return result

    return {
        "before_nll": float(before_nll),
        "after_nll": float(after_nll),
        "before_ece": ece(before_probability),
        "after_ece": ece(after_probability),
    }


__all__ = [
    "ClassConditionalCalibration",
    "calibration_metrics",
    "fit_class_conditional_calibration",
]
