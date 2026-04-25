from __future__ import annotations

from dataclasses import dataclass


STANDARD_LABELS = [
    "AircraftTarget",
    "BalloonTarget",
    "BirdTarget",
    "CarTarget",
    "ClutterTarget",
    "DroneTarget",
    "PeopleTarget",
    "ShipTarget",
]


@dataclass(frozen=True)
class LabelVocabulary:
    labels: list[str]

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    def encode(self, label_name: str) -> int:
        return self.label_to_index[label_name]

    def decode(self, label_index: int) -> str:
        return self.index_to_label[label_index]


DEFAULT_VOCAB = LabelVocabulary(labels=STANDARD_LABELS)
