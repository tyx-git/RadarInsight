from __future__ import annotations

from dataclasses import dataclass

import torch

from ..trajectory import TrackRepresentationBuilder


@dataclass(frozen=True)
class SharedExpertFeatures:
    sequence_features: torch.Tensor
    deep_features: torch.Tensor
    raw_physical_features: torch.Tensor
    physical_features: torch.Tensor
    fused_features: torch.Tensor


def build_shared_expert_features(
    representation_builder: TrackRepresentationBuilder,
    sequence_features: torch.Tensor,
    physical_features: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> SharedExpertFeatures:
    rep = representation_builder(
        sequence_features=sequence_features,
        physical_features=physical_features,
        padding_mask=padding_mask,
    )
    return SharedExpertFeatures(
        sequence_features=rep.sequence_features,
        deep_features=rep.deep_features,
        raw_physical_features=rep.raw_physical_features,
        physical_features=rep.physical_features,
        fused_features=rep.fused_features,
    )
