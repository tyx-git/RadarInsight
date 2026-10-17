"""r21 adapter-only wrapper over the r20 L2 model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from v27.V27d.r20.model import R20L2Model

from .protocol import R21_ARMS, R21Config


class R21L2Model(nn.Module):
    """Use the exact r20 structure while exposing only its radial adapter."""

    def __init__(self, config: R21Config, *, base_model: R20L2Model, arm: str) -> None:
        super().__init__()
        if arm not in R21_ARMS:
            raise ValueError("r21 model arm is invalid")
        if not isinstance(base_model, R20L2Model):
            raise TypeError("r21 base_model must be R20L2Model")
        self.config = config
        self.arm = arm
        self.base_model = base_model

    @property
    def radial_adapter(self) -> nn.Module:
        return self.base_model.radial_adapter

    def model_contract(self) -> dict[str, Any]:
        contract = dict(self.base_model.model_contract())
        return {
            "schema_version": "RADAR-LN3-V27d-r21-l2-model-contract-v1",
            "experiment_id": self.config["experiment_id"],
            "arm": self.arm,
            "structure_source": "V27d-r20",
            "base_model_contract": contract,
            "trainable_scope": "radial_adapter_only",
            "frozen_scope": "base_model_all_parameters",
            "unknown_classifier_class": False,
        }

    def base_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.base_model.base_model.state_dict().items()
        }

    def forward(self, **inputs: torch.Tensor) -> dict[str, Any]:
        return self.base_model(**inputs)


def freeze_base_train_adapter(model: R21L2Model) -> None:
    for parameter in model.base_model.base_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.base_model.radial_adapter.parameters():
        parameter.requires_grad_(True)
    model.base_model.base_model.eval()
    model.base_model.radial_adapter.train()
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any(not name.startswith("base_model.radial_adapter.") for name in names):
        raise AssertionError(f"r21 trainable scope differs: {names}")


def assert_base_state_unchanged(model: R21L2Model, reference: Mapping[str, torch.Tensor]) -> None:
    current = model.base_state()
    if set(current) != set(reference):
        raise AssertionError("r21 frozen base state keys changed")
    for name in current:
        if not torch.equal(current[name], reference[name]):
            raise AssertionError(f"r21 frozen base parameter changed: {name}")


__all__ = ["R21L2Model", "assert_base_state_unchanged", "freeze_base_train_adapter"]
