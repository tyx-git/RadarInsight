from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA, DT_REASON_ZERO_DELTA
from .model import RadarLiquidClassifier
from .stable import StableDecision, TargetStableStateMachine


@dataclass
class OnlineStepResult:
    window_logits: torch.Tensor
    window_label_id: int
    window_label_name: str
    stable_label_id: int | None
    stable_label_name: str | None
    stable_confidence: float | None
    stable_confirmed: bool
    dt_clamped: bool
    dt_zero: bool
    dt_invalid: bool


class OnlineTargetClassifier:
    """Per-target causal wrapper for window and stable results.

    One instance owns one target's liquid state and stable state. A service can
    keep one instance per active target ID; the target ID itself never enters
    the tensor contract.
    """

    def __init__(self, model: RadarLiquidClassifier, *, confirm_count: int = 2, ema_decay: float = 0.65) -> None:
        self.model = model
        self.stable = TargetStableStateMachine(confirm_count=confirm_count, ema_decay=ema_decay)
        self.liquid_state: torch.Tensor | None = None
        self.zero_ordinal = 0

    def reset(self) -> None:
        self.liquid_state = None
        self.zero_ordinal = 0
        self.stable = TargetStableStateMachine(confirm_count=self.stable.confirm_count, ema_decay=self.stable.ema_decay)

    @torch.inference_mode()
    def step(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor | None = None,
        tr_dt_reason: torch.Tensor | None = None,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
    ) -> OnlineStepResult:
        if tr_features.shape[0] != 1:
            raise ValueError("OnlineTargetClassifier currently requires batch size 1 per target")
        zero_ordinal = None
        if tr_dt_reason is not None:
            if tr_dt_reason.shape != tr_dt_seconds.shape:
                raise ValueError("OnlineTargetClassifier tr_dt_reason shape mismatch")
            reason_id = int(tr_dt_reason.reshape(-1)[0].item())
            if reason_id in {DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA}:
                self.zero_ordinal = 1
            elif reason_id == DT_REASON_ZERO_DELTA:
                self.zero_ordinal += 1
            else:
                raise ValueError(f"unsupported online dt reason: {reason_id}")
            zero_ordinal = torch.tensor(
                [self.zero_ordinal], dtype=torch.long, device=tr_features.device
            )
        output = self.model.forward_online_point(
            tr_features=tr_features,
            tr_missing_mask=tr_missing_mask,
            tr_dt_seconds=tr_dt_seconds,
            tr_dt_valid=tr_dt_valid,
            tr_dt_reason=tr_dt_reason,
            zero_ordinal=zero_ordinal,
            rd_frames=rd_frames,
            rd_frame_mask=rd_frame_mask,
            rd_quality=rd_quality,
            liquid_state=self.liquid_state,
        )
        self.liquid_state = output["liquid_state"]
        decision: StableDecision = self.stable.update(output["window_logits"][0].detach().cpu())
        return OnlineStepResult(
            window_logits=output["window_logits"],
            window_label_id=decision.window_label_id,
            window_label_name=decision.window_label_name,
            stable_label_id=decision.stable_label_id,
            stable_label_name=decision.stable_label_name,
            stable_confidence=decision.stable_confidence,
            stable_confirmed=decision.stable_confirmed,
            dt_clamped=bool(output["dt_clamped_mask"][0].item()),
            dt_zero=bool(output["dt_zero_mask"][0].item()),
            dt_invalid=bool(output["dt_invalid_mask"][0].item()),
        )


__all__ = ["OnlineStepResult", "OnlineTargetClassifier"]
