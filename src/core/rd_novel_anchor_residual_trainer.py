from __future__ import annotations

from .rd_novel_anchor_residual_config import RDNovelAnchorResidualConfig


def train_rd_novel_anchor_residual(config: RDNovelAnchorResidualConfig) -> dict[str, object]:
    raise RuntimeError(
        "anchor-residual training is retired: the legacy baseline logit cache is incompatible with the active Physical RD contract"
    )


__all__ = ["train_rd_novel_anchor_residual"]
