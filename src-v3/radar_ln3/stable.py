from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StableDecision:
    window_label_id: int
    window_label_name: str
    window_confidence: float
    stable_label_id: int | None
    stable_label_name: str | None
    stable_confidence: float | None
    stable_confirmed: bool
    candidate_label_id: int
    candidate_count: int
    update_count: int


LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)


class TargetStableStateMachine:
    """Causal target-level smoothing over window logits.

    This state is external to the liquid encoder and can be kept per target
    identity by the online service. It never reads future windows.
    """

    def __init__(self, *, ema_decay: float = 0.65, confirm_count: int = 2) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.ema_decay = float(ema_decay)
        self.confirm_count = max(1, int(confirm_count))
        self.ema_logits: torch.Tensor | None = None
        self.ema_probabilities: torch.Tensor | None = None
        self.candidate_label: int | None = None
        self.candidate_count = 0
        self.stable_label: int | None = None
        self.update_count = 0
        self._input_mode: str | None = None

    def _lock_input_mode(self, mode: str) -> None:
        if self._input_mode is None:
            self._input_mode = mode
            return
        if self._input_mode != mode:
            raise RuntimeError(
                "one stable state machine cannot mix logits and probability updates; "
                f"existing_mode={self._input_mode}, requested_mode={mode}"
            )

    def update(self, window_logits: torch.Tensor) -> StableDecision:
        logits = window_logits.detach().float().reshape(-1)
        if logits.shape != (5,) or not torch.isfinite(logits).all():
            raise ValueError("stable state expects finite logits with shape [5]")
        self._lock_input_mode("logits")
        self.ema_logits = logits if self.ema_logits is None else self.ema_decay * self.ema_logits + (1.0 - self.ema_decay) * logits
        window_probabilities = torch.softmax(logits, dim=0)
        stable_probabilities = torch.softmax(self.ema_logits, dim=0)
        return self._update_decision(window_probabilities, stable_probabilities)

    def update_probabilities(self, window_probabilities: torch.Tensor) -> StableDecision:
        """Update from normalized class probabilities without a second softmax.

        The probabilities themselves are averaged, so no second softmax or
        accidental confidence compression is introduced.
        """

        probabilities = window_probabilities.detach().float().reshape(-1)
        if probabilities.shape != (5,) or not torch.isfinite(probabilities).all():
            raise ValueError("stable state expects finite probabilities with shape [5]")
        if bool((probabilities < 0.0).any()):
            raise ValueError("stable probabilities must be non-negative")
        total = probabilities.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            raise ValueError("stable probabilities must have a positive finite sum")
        self._lock_input_mode("probabilities")
        probabilities = probabilities / total
        self.ema_probabilities = probabilities if self.ema_probabilities is None else self.ema_decay * self.ema_probabilities + (1.0 - self.ema_decay) * probabilities
        self.ema_probabilities = self.ema_probabilities / self.ema_probabilities.sum().clamp_min(torch.finfo(probabilities.dtype).tiny)
        return self._update_decision(probabilities, self.ema_probabilities)

    def _update_decision(self, window_probabilities: torch.Tensor, stable_probabilities: torch.Tensor) -> StableDecision:
        window_label = int(window_probabilities.argmax().item())
        stable_candidate = int(stable_probabilities.argmax().item())
        if stable_candidate == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = stable_candidate
            self.candidate_count = 1
        if self.candidate_count >= self.confirm_count:
            self.stable_label = stable_candidate
        self.update_count += 1
        stable_confirmed = self.stable_label is not None
        return StableDecision(
            window_label_id=window_label,
            window_label_name=LABELS[window_label],
            window_confidence=float(window_probabilities[window_label].item()),
            stable_label_id=self.stable_label,
            stable_label_name=None if self.stable_label is None else LABELS[self.stable_label],
            stable_confidence=None if self.stable_label is None else float(stable_probabilities[self.stable_label].item()),
            stable_confirmed=stable_confirmed,
            candidate_label_id=stable_candidate,
            candidate_count=self.candidate_count,
            update_count=self.update_count,
        )


__all__ = ["StableDecision", "TargetStableStateMachine"]
