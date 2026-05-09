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
    "UnknownTarget",
]

FINE_LABELS = [
    "AircraftTarget",
    "BalloonTarget",
    "BirdTarget",
    "CarTarget",
    "ClutterTarget",
    "DroneTarget",
    "PeopleTarget",
    "RotorDroneTarget",
    "ShipTarget",
    "UnknownTarget",
]

FINE_TO_COARSE_LABEL = {
    "RotorDroneTarget": "DroneTarget",
}

ROTOR_DRONE_RAW_LABELS = {
    "Rotary drone",
    "小型旋翼无人机",
    "轻型旋翼无人机",
}


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
FINE_VOCAB = LabelVocabulary(labels=FINE_LABELS)


def normalize_coarse_label(label_name: str) -> str:
    label = str(label_name)
    return FINE_TO_COARSE_LABEL.get(label, label)


def infer_fine_label(
    label_name: str,
    *,
    source_name: str = "",
    label_id: object = None,
    batch_id: object = None,
    label_name_raw: str = "",
) -> str:
    coarse_label = normalize_coarse_label(label_name)
    source = str(source_name)
    raw = str(label_name_raw)
    id_value = "" if label_id is None else str(label_id).strip()
    batch_value = "" if batch_id is None else str(batch_id).strip()
    if coarse_label == "DroneTarget":
        if source == "cq08_track" and (id_value in {"1", "2"} or batch_value in {"1", "2"}):
            return "RotorDroneTarget"
        if raw in ROTOR_DRONE_RAW_LABELS:
            return "RotorDroneTarget"
    return coarse_label


def two_layer_label_fields(
    label_name: str,
    *,
    source_name: str = "",
    label_id: object = None,
    batch_id: object = None,
    label_name_raw: str = "",
) -> dict[str, str]:
    coarse_label = normalize_coarse_label(label_name)
    fine_label = infer_fine_label(
        coarse_label,
        source_name=source_name,
        label_id=label_id,
        batch_id=batch_id,
        label_name_raw=label_name_raw,
    )
    return {
        "coarse_label": coarse_label,
        "fine_label": fine_label,
    }


__all__ = [
    "STANDARD_LABELS",
    "FINE_LABELS",
    "FINE_TO_COARSE_LABEL",
    "LabelVocabulary",
    "DEFAULT_VOCAB",
    "FINE_VOCAB",
    "normalize_coarse_label",
    "infer_fine_label",
    "two_layer_label_fields",
]
