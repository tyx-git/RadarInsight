"""L3 knownness head 标签映射回归测试。"""

from __future__ import annotations

import torch

from radar_ln3.training.l3_heads import _route_targets


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
