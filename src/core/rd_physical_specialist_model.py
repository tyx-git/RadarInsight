from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data.rd_physical_specialist_dataset import specialist_label_names
from .rd_physical_five_model import PhysicalRDFiveClassifier


SPECIALIST_MODEL_SCHEMA_VERSION = "physical_rd_specialist_model_v2_local_heads"


@dataclass(frozen=True)
class PhysicalRDSpecialistOutput:
    logits: torch.Tensor
    frame_logits: torch.Tensor
    micro_logits: torch.Tensor
    parent_logits: torch.Tensor


class PhysicalRDSpecialistClassifier(nn.Module):
    """Independent specialist parameter set using the audited RD encoder.

    The five-class encoder is instantiated afresh for every specialist; no
    five-class checkpoint is loaded.  Only the requested local logits are
    exposed and optimized.
    """

    supports_rejection = False

    def __init__(self, *, kind: str, base: PhysicalRDFiveClassifier) -> None:
        super().__init__()
        self.kind = str(kind)
        self.label_names = specialist_label_names(self.kind)
        self.base = base

    def forward(self, *args: torch.Tensor) -> PhysicalRDSpecialistOutput:
        output = self.base(*args)
        if self.kind == "parent_gate":
            logits = output.parent_logits
            # ``parent_gate`` uses a native two-class auxiliary head as well;
            # no global five-class auxiliary logits are constructed or sliced.
            frame = output.frame_logits
            micro = output.micro_logits
            parent = output.parent_logits
        else:
            # The base model is instantiated with a local head for DBB/CU.
            # Do not slice global five-class logits here: doing so would make
            # the local head contract depend on an unused five-class head.
            logits = output.flat_logits
            frame = output.frame_logits
            micro = output.micro_logits
            parent = output.parent_logits
        return PhysicalRDSpecialistOutput(logits=logits, frame_logits=frame, micro_logits=micro, parent_logits=parent)


PhysicalRDParentGateClassifier = PhysicalRDSpecialistClassifier
PhysicalRDDBBSpecialist = PhysicalRDSpecialistClassifier
PhysicalRDCUSpecialist = PhysicalRDSpecialistClassifier


__all__ = [
    "PhysicalRDSpecialistClassifier",
    "PhysicalRDSpecialistOutput",
    "PhysicalRDParentGateClassifier",
    "PhysicalRDDBBSpecialist",
    "PhysicalRDCUSpecialist",
    "SPECIALIST_MODEL_SCHEMA_VERSION",
]
