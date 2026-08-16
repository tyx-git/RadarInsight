"""Legacy stable-state implementations retained for historical artifacts.

The V21 semantic decision layer and its Pending/Abstain rules are not yet
calibrated or executed here; no V21 local KnownGate result may be reported as
stable performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rejection import HierarchicalActionDecision


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


@dataclass(frozen=True)
class HierarchicalStableDecision:
    event_action: str
    state_name: str
    stable_confirmed: bool
    stable_label_id: int | None
    stable_label_name: str | None
    stable_rejected_to_unknown: bool
    candidate_action: str | None
    candidate_label_id: int | None
    candidate_count: int
    update_count: int
    transition_reset: bool


class HierarchicalStableStateMachine:
    """Consecutive-confirmation state machine over calibrated product actions.

    A conflicting event immediately withdraws the previous stable output and
    starts a new pending transition.  Consequently a rejected or pending event
    can never coexist with a confirmed concrete target label.
    """

    _FINAL_ACTIONS = frozenset({"DBB_ACCEPTED", "CLUTTER_ACCEPTED", "UNKNOWN_REJECTED"})

    def __init__(self, *, confirm_count: int = 2) -> None:
        self.confirm_count = max(1, int(confirm_count))
        self.reset()

    def reset(self) -> None:
        self.candidate_key: tuple[str, int | None] | None = None
        self.candidate_count = 0
        self.stable_key: tuple[str, int | None] | None = None
        self.update_count = 0

    @staticmethod
    def _key(decision: HierarchicalActionDecision) -> tuple[str, int | None] | None:
        if decision.final_action == "DBB_ACCEPTED":
            if decision.accepted_label_id not in {0, 1, 2}:
                raise ValueError("DBB_ACCEPTED must carry a DBB subclass label")
            return decision.final_action, int(decision.accepted_label_id)
        if decision.final_action == "CLUTTER_ACCEPTED":
            if decision.accepted_label_id != 3:
                raise ValueError("CLUTTER_ACCEPTED must carry ClutterTarget")
            return decision.final_action, 3
        if decision.final_action == "UNKNOWN_REJECTED":
            if decision.accepted_label_id is not None or decision.rejected_to_unknown is not True:
                raise ValueError("UNKNOWN_REJECTED must not carry an accepted class")
            return decision.final_action, None
        if decision.final_action == "NOT_CALIBRATED":
            return None
        raise ValueError(f"unsupported hierarchical event action: {decision.final_action!r}")

    @staticmethod
    def _pending_state(action: str) -> str:
        return "DBB_PENDING" if action == "DBB_ACCEPTED" else "CU_PENDING"

    def update(self, decision: HierarchicalActionDecision) -> HierarchicalStableDecision:
        if decision.final_action in {"DBB_PENDING", "CU_PENDING"}:
            transition_reset = self.candidate_key is not None or self.stable_key is not None
            self.update_count += 1
            self.candidate_key = None
            self.candidate_count = 0
            self.stable_key = None
            return HierarchicalStableDecision(
                event_action=decision.final_action,
                state_name=decision.final_action,
                stable_confirmed=False,
                stable_label_id=None,
                stable_label_name=None,
                stable_rejected_to_unknown=False,
                candidate_action=None,
                candidate_label_id=None,
                candidate_count=0,
                update_count=self.update_count,
                transition_reset=transition_reset,
            )
        key = self._key(decision)
        self.update_count += 1
        if key is None:
            transition_reset = self.candidate_key is not None or self.stable_key is not None
            self.candidate_key = None
            self.candidate_count = 0
            self.stable_key = None
            return HierarchicalStableDecision(
                event_action=decision.final_action,
                state_name="NOT_CALIBRATED",
                stable_confirmed=False,
                stable_label_id=None,
                stable_label_name=None,
                stable_rejected_to_unknown=False,
                candidate_action=None,
                candidate_label_id=None,
                candidate_count=0,
                update_count=self.update_count,
                transition_reset=transition_reset,
            )
        transition_reset = key != self.candidate_key
        if transition_reset:
            self.candidate_key = key
            self.candidate_count = 1
            self.stable_key = None
        else:
            self.candidate_count += 1
        if self.candidate_count >= self.confirm_count:
            self.stable_key = key
        if self.stable_key is None:
            return HierarchicalStableDecision(
                event_action=decision.final_action,
                state_name=self._pending_state(key[0]),
                stable_confirmed=False,
                stable_label_id=None,
                stable_label_name=None,
                stable_rejected_to_unknown=False,
                candidate_action=key[0],
                candidate_label_id=key[1],
                candidate_count=self.candidate_count,
                update_count=self.update_count,
                transition_reset=transition_reset,
            )
        action, label_id = self.stable_key
        rejected = action == "UNKNOWN_REJECTED"
        if rejected and label_id is not None:
            raise RuntimeError("rejected stable action unexpectedly carries a class label")
        return HierarchicalStableDecision(
            event_action=decision.final_action,
            state_name=action,
            stable_confirmed=True,
            stable_label_id=label_id,
            stable_label_name=None if label_id is None else LABELS[label_id],
            stable_rejected_to_unknown=rejected,
            candidate_action=key[0],
            candidate_label_id=key[1],
            candidate_count=self.candidate_count,
            update_count=self.update_count,
            transition_reset=transition_reset,
        )


__all__ = [
    "HierarchicalStableDecision",
    "HierarchicalStableStateMachine",
    "StableDecision",
    "TargetStableStateMachine",
]
