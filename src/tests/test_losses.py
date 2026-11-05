from __future__ import annotations

import torch

from radar_ln3.losses import parent_final_fixed_parent_count_ce


def test_l2_prefix_loss_uses_last_valid_events_per_parent() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]],
            [[0.0, 4.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    labels = torch.tensor([1, 1])
    padding = torch.tensor(
        [[False, False, False, False], [False, False, True, True]]
    )
    result = parent_final_fixed_parent_count_ce(
        logits,
        labels,
        padding,
        class_weights=torch.ones(2),
        prefix_weight=0.15,
        prefix_events=4,
    )

    lengths = torch.tensor([4, 2])
    final = torch.nn.functional.cross_entropy(
        logits[torch.arange(2), lengths - 1], labels, reduction="none"
    ).mean()
    # Parent 0 contributes its last three non-final events; parent 1 contributes
    # its single non-final event. The parent means are averaged equally.
    parent0 = torch.nn.functional.cross_entropy(
        logits[0, :3], torch.ones(3, dtype=torch.long), reduction="none"
    ).mean()
    parent1 = torch.nn.functional.cross_entropy(
        logits[1, :1], torch.ones(1, dtype=torch.long), reduction="none"
    ).mean()
    prefix = (parent0 + parent1) / 2

    torch.testing.assert_close(result["final_state_loss"], final)
    torch.testing.assert_close(result["prefix_auxiliary_loss"], prefix)
    torch.testing.assert_close(result["loss"], final + 0.15 * prefix)
    assert result["prefix_event_count"] == 3
