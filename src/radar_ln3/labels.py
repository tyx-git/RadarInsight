"""L1/L2 共享的标签词表。

标签字符串只在这里定义一次：``TASK_LABEL_TO_ID`` 是唯一真源，
``TASK_LABELS`` 由其键顺序派生，避免训练侧与特征侧各写一份。
"""

from __future__ import annotations

TASK_LABEL_TO_ID: dict[str, dict[str, int]] = {
    # L1 把三个细类折叠到 0，Clutter 为 1。
    "l1": {"DroneTarget": 0, "BirdTarget": 0, "BalloonTarget": 0, "ClutterTarget": 1},
    "l2": {"DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2},
}
TASK_LABELS: dict[str, tuple[str, ...]] = {
    task: tuple(mapping) for task, mapping in TASK_LABEL_TO_ID.items()
}
L2_LABELS: tuple[str, ...] = TASK_LABELS["l2"]
KNOWN_TRAINING_LABELS: tuple[str, ...] = tuple(
    sorted({label for labels in TASK_LABELS.values() for label in labels})
)

__all__ = [
    "TASK_LABEL_TO_ID",
    "TASK_LABELS",
    "L2_LABELS",
    "KNOWN_TRAINING_LABELS",
]
