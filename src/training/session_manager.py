import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import Subset

from src.training.dataset import RadarFusionDataset


@dataclass(slots=True)
class SessionStatus:
    session_id: str
    ready_for_training: bool
    has_track_csv: bool
    has_track_clips: bool
    has_matched_labels: bool
    has_rd_clips: bool
    num_track_clips: int
    num_matched_tracks: int
    labels: list[str]


@dataclass(slots=True)
class SessionSplitManifest:
    train_sessions: list[str]
    val_sessions: list[str]
    test_sessions: list[str]
    session_statuses: list[SessionStatus]

    def to_dict(self) -> dict[str, object]:
        return {
            "train_sessions": self.train_sessions,
            "val_sessions": self.val_sessions,
            "test_sessions": self.test_sessions,
            "session_statuses": [asdict(status) for status in self.session_statuses],
        }


def _load_jsonl(file_path: Path) -> list[dict[str, object]]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def scan_session_directories(data_root: str, include_rd: bool = True) -> list[SessionStatus]:
    root = Path(data_root)
    if not root.exists():
        return []

    statuses: list[SessionStatus] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        track_csv = session_dir / "radar/track/track_data.csv"
        track_clip_file = session_dir / "labels/labels_trackclip.jsonl"
        matched_file = session_dir / "labels/labels_matched.jsonl"
        rd_clip_file = session_dir / "radar/rd/rd_clips.h5"

        track_clips = _load_jsonl(track_clip_file) if track_clip_file.exists() else []
        matched_items = _load_jsonl(matched_file) if matched_file.exists() else []
        labels = sorted(
            {
                str(item.get("label_main"))
                for item in matched_items
                if item.get("label_main") is not None
            }
        )
        ready = track_csv.exists() and track_clip_file.exists() and matched_file.exists()
        if include_rd:
            ready = ready and rd_clip_file.exists()
        ready = ready and bool(track_clips) and bool(matched_items)

        statuses.append(
            SessionStatus(
                session_id=session_dir.name,
                ready_for_training=ready,
                has_track_csv=track_csv.exists(),
                has_track_clips=track_clip_file.exists(),
                has_matched_labels=matched_file.exists(),
                has_rd_clips=rd_clip_file.exists(),
                num_track_clips=len(track_clips),
                num_matched_tracks=len(matched_items),
                labels=labels,
            )
        )
    return statuses


def _greedy_session_split(
    session_to_indices: dict[str, list[int]],
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    sessions = list(session_to_indices.items())
    if not sessions:
        return [], [], []

    rng = random.Random(random_seed)
    rng.shuffle(sessions)
    sessions.sort(key=lambda item: len(item[1]), reverse=True)
    session_ids = [session_id for session_id, _ in sessions]

    if len(session_ids) == 1:
        return session_ids, [], []
    if len(session_ids) == 2:
        return [session_ids[0]], [session_ids[1]], []

    total_samples = sum(len(indices) for _, indices in sessions)
    val_target = max(1, int(round(total_samples * val_ratio))) if val_ratio > 0 else 0
    test_target = max(1, int(round(total_samples * test_ratio))) if test_ratio > 0 else 0

    train_sessions = [session_ids[0]]
    val_sessions = [session_ids[1]] if val_target > 0 else []
    test_sessions = [session_ids[2]] if test_target > 0 else []

    assigned = set(train_sessions + val_sessions + test_sessions)
    sample_counts = {session_id: len(indices) for session_id, indices in sessions}
    train_count = sum(sample_counts[session_id] for session_id in train_sessions)
    val_count = sum(sample_counts[session_id] for session_id in val_sessions)
    test_count = sum(sample_counts[session_id] for session_id in test_sessions)

    target_train = max(total_samples - val_target - test_target, 1)

    for session_id in session_ids:
        if session_id in assigned:
            continue

        candidates: list[tuple[float, str]] = []
        candidates.append((train_count / max(target_train, 1), "train"))
        if val_target > 0:
            candidates.append((val_count / max(val_target, 1), "val"))
        if test_target > 0:
            candidates.append((test_count / max(test_target, 1), "test"))

        _, split_name = min(candidates, key=lambda item: item[0])
        if split_name == "train":
            train_sessions.append(session_id)
            train_count += sample_counts[session_id]
        elif split_name == "val":
            val_sessions.append(session_id)
            val_count += sample_counts[session_id]
        else:
            test_sessions.append(session_id)
            test_count += sample_counts[session_id]

    return train_sessions, val_sessions, test_sessions


def _split_label_group(
    session_ids: list[str],
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if not session_ids:
        return [], [], []

    rng = random.Random(random_seed)
    ordered = list(session_ids)
    rng.shuffle(ordered)
    total = len(ordered)

    if total == 1:
        return ordered, [], []
    if total == 2:
        return [ordered[0]], [ordered[1]], []

    val_count = max(1, int(round(total * val_ratio))) if val_ratio > 0 else 0
    test_count = max(1, int(round(total * test_ratio))) if test_ratio > 0 else 0
    while val_count + test_count >= total:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    test_sessions = ordered[:test_count]
    val_sessions = ordered[test_count : test_count + val_count]
    train_sessions = ordered[test_count + val_count :]
    if not train_sessions:
        train_sessions = [val_sessions.pop()] if val_sessions else [test_sessions.pop()]
    return train_sessions, val_sessions, test_sessions


def build_session_subsets(
    dataset: RadarFusionDataset,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    session_ids: list[str] | None = None,
) -> tuple[Subset, Subset | None, Subset | None, dict[str, list[str]]]:
    session_to_indices = dataset.build_session_index()
    if session_ids is not None:
        allowed = set(session_ids)
        session_to_indices = {key: value for key, value in session_to_indices.items() if key in allowed}

    session_to_label: dict[str, str] = {}
    for record in dataset.samples:
        if record.session_id not in session_to_indices:
            continue
        session_to_label.setdefault(record.session_id, record.label_name)

    label_to_sessions: dict[str, list[str]] = {}
    for session_id in session_to_indices:
        label = session_to_label.get(session_id, "UNKNOWN")
        label_to_sessions.setdefault(label, []).append(session_id)

    train_sessions: list[str] = []
    val_sessions: list[str] = []
    test_sessions: list[str] = []
    for label_index, label in enumerate(sorted(label_to_sessions)):
        label_train, label_val, label_test = _split_label_group(
            session_ids=label_to_sessions[label],
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed + label_index,
        )
        train_sessions.extend(label_train)
        val_sessions.extend(label_val)
        test_sessions.extend(label_test)

    train_indices = [index for session_id in train_sessions for index in session_to_indices[session_id]]
    val_indices = [index for session_id in val_sessions for index in session_to_indices[session_id]]
    test_indices = [index for session_id in test_sessions for index in session_to_indices[session_id]]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None
    test_subset = Subset(dataset, test_indices) if test_indices else None
    split_map = {
        "train": train_sessions,
        "val": val_sessions,
        "test": test_sessions,
    }
    return train_subset, val_subset, test_subset, split_map


def build_session_manifest(
    data_root: str,
    include_rd: bool,
    split_map: dict[str, list[str]],
) -> SessionSplitManifest:
    statuses = scan_session_directories(data_root=data_root, include_rd=include_rd)
    return SessionSplitManifest(
        train_sessions=split_map.get("train", []),
        val_sessions=split_map.get("val", []),
        test_sessions=split_map.get("test", []),
        session_statuses=statuses,
    )


def save_session_manifest(manifest: SessionSplitManifest, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_session_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, indent=2, ensure_ascii=False)
    return manifest_path


def count_session_labels(statuses: list[SessionStatus]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for status in statuses:
        for label in status.labels:
            counter[label] += 1
    return dict(counter)
