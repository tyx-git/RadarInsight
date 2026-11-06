"""L3 knownness head 标签映射回归测试。"""

from __future__ import annotations

import torch

from radar_ln3.training.l3_heads import _epoch_permutation, _route_targets


def test_profile_epochs_use_distinct_deterministic_shuffle_orders() -> None:
    first = _epoch_permutation(32, seed=42, epoch=1, phase_offset=20)
    second = _epoch_permutation(32, seed=42, epoch=2, phase_offset=20)
    repeat = _epoch_permutation(32, seed=42, epoch=1, phase_offset=20)

    assert not torch.equal(first, second)
    assert torch.equal(first, repeat)


def test_l2_non_classwise_counts_all_known_target_labels() -> None:
    labels = (
        "DroneTarget",
        "BirdTarget",
        "BalloonTarget",
        "UnknownTarget",
        "ClutterTarget",
    )

    targets = _route_targets("l2", labels, 1, torch.device("cpu"), l2_classwise=False)
    assert targets.sum(dim=(0, 1)).tolist() == [3.0]


def test_l2_classwise_keeps_per_class_support() -> None:
    labels = (
        "DroneTarget",
        "BirdTarget",
        "BirdTarget",
        "UnknownTarget",
    )

    targets = _route_targets("l2", labels, 1, torch.device("cpu"), l2_classwise=True)
    assert targets.sum(dim=(0, 1)).tolist() == [1.0, 2.0, 0.0]


def test_l1_route_support_keeps_target_and_clutter_semantics() -> None:
    labels = (
        "DroneTarget",
        "BalloonTarget",
        "ClutterTarget",
        "UnknownTarget",
    )

    targets = _route_targets("l1", labels, 1, torch.device("cpu"))
    assert targets.sum(dim=(0, 1)).tolist() == [2.0, 1.0]
