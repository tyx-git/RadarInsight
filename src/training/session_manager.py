from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import Subset

from src.training.dataset import RadarFusionDataset
from src.training.tsv_dataset import TrackOnlyTsvDataset

scan_statuses: list[SessionStatus] = []


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
            'train_sessions': self.train_sessions,
            'val_sessions': self.val_sessions,
            'test_sessions': self.test_sessions,
            'session_statuses': [asdict(status) for status in self.session_statuses],
        }



def _load_jsonl(file_path: Path) -> list[dict[str, object]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]



def _scan_tsv_directories(data_root: str) -> list[SessionStatus]:
    root = Path(data_root)
    if not root.exists():
        return []
    statuses: list[SessionStatus] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        points_file = session_dir / 'track_points.tsv'
        labels_file = session_dir / 'track_labels.tsv'
        notes_file = session_dir / 'track_notes.tsv'
        meta_file = session_dir / 'session_meta.json'
        labels: list[str] = []
        num_tracks = 0
        if labels_file.exists():
            with labels_file.open('r', encoding='utf-8') as handle:
                lines = [line.rstrip('\r\n') for line in handle]
            if len(lines) > 1:
                header = lines[0].split('	')
                try:
                    label_index = header.index('label_name')
                except ValueError:
                    label_index = -1
                if label_index >= 0:
                    labels = sorted({line.split('	')[label_index] for line in lines[1:] if line.strip()})
                    num_tracks = len(lines) - 1
        ready = points_file.exists() and labels_file.exists() and notes_file.exists() and meta_file.exists() and num_tracks > 0
        statuses.append(
            SessionStatus(
                session_id=session_dir.name,
                ready_for_training=ready,
                has_track_csv=points_file.exists(),
                has_track_clips=points_file.exists(),
                has_matched_labels=labels_file.exists(),
                has_rd_clips=False,
                num_track_clips=num_tracks,
                num_matched_tracks=num_tracks,
                labels=labels,
            )
        )
    return statuses



def scan_session_directories(data_root: str, include_rd: bool = True) -> list[SessionStatus]:
    root = Path(data_root)
    if (root / 'dataset_manifest.json').exists() or any((root / child).is_dir() and (root / child / 'track_points.tsv').exists() for child in root.iterdir()) if root.exists() else False:
        return _scan_tsv_directories(data_root)

    if not root.exists():
        return []

    statuses: list[SessionStatus] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        track_csv = session_dir / 'radar/track/track_data.csv'
        track_clip_file = session_dir / 'labels/labels_trackclip.jsonl'
        matched_file = session_dir / 'labels/labels_matched.jsonl'
        rd_clip_file = session_dir / 'radar/rd/rd_clips.h5'

        track_clips = _load_jsonl(track_clip_file) if track_clip_file.exists() else []
        matched_items = _load_jsonl(matched_file) if matched_file.exists() else []
        labels = sorted({str(item.get('label_main')) for item in matched_items if item.get('label_main') is not None})
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



def _greedy_session_split(session_to_indices, val_ratio, test_ratio, random_seed):
    sessions = list(session_to_indices.items())
    if not sessions:
        return [], [], []

    sample_counts = {session_id: len(indices) for session_id, indices in sessions}
    session_ids = [session_id for session_id, _ in sessions]
    if len(session_ids) <= 3:
        if len(session_ids) == 1:
            return session_ids, [], []
        if len(session_ids) == 2:
            return [session_ids[0]], [session_ids[1]], []
        return [session_ids[0]], [session_ids[1]], [session_ids[2]]

    rng = random.Random(random_seed)
    status_map = {status.session_id: status for status in scan_statuses}

    def label_of(session_id: str) -> str:
        labels = list(status_map.get(session_id, SessionStatus(session_id, False, False, False, False, False, 0, 0, [])).labels)
        return labels[0] if labels else 'Unknown'

    label_to_sessions: dict[str, list[str]] = {}
    for session_id in session_ids:
        label_to_sessions.setdefault(label_of(session_id), []).append(session_id)

    for label_sessions in label_to_sessions.values():
        label_sessions.sort(key=lambda sid: sample_counts[sid], reverse=True)

    total_sessions = len(session_ids)
    desired_val_sessions = max(1, int(round(total_sessions * val_ratio))) if val_ratio > 0 else 0
    desired_test_sessions = max(1, int(round(total_sessions * test_ratio))) if test_ratio > 0 else 0

    classes = sorted(label_to_sessions.keys())
    base_test = {label: 1 if len(label_to_sessions[label]) >= 3 and desired_test_sessions > 0 else 0 for label in classes}
    base_val = {label: 1 if len(label_to_sessions[label]) >= 3 and desired_val_sessions > 0 else 0 for label in classes}

    test_sessions: list[str] = []
    val_sessions: list[str] = []
    used: set[str] = set()

    def pop_session(label: str, prefer_large: bool = True) -> str | None:
        candidates = [sid for sid in label_to_sessions.get(label, []) if sid not in used]
        if not candidates:
            return None
        candidates.sort(key=lambda sid: sample_counts[sid], reverse=prefer_large)
        return candidates[0]

    for label, count in base_test.items():
        for _ in range(count):
            sid = pop_session(label, prefer_large=True)
            if sid is not None:
                test_sessions.append(sid)
                used.add(sid)

    for label, count in base_val.items():
        for _ in range(count):
            sid = pop_session(label, prefer_large=False)
            if sid is not None:
                val_sessions.append(sid)
                used.add(sid)

    remaining = [sid for sid in session_ids if sid not in used]

    def fill_bucket(bucket: list[str], target_count: int, prefer_large: bool):
        nonlocal remaining
        while len(bucket) < target_count and remaining:
            remaining.sort(key=lambda sid: sample_counts[sid], reverse=prefer_large)
            sid = remaining.pop(0)
            bucket.append(sid)

    fill_bucket(test_sessions, desired_test_sessions, True)
    fill_bucket(val_sessions, desired_val_sessions, False)

    train_sessions = [sid for sid in session_ids if sid not in set(test_sessions) | set(val_sessions)]

    def covers_all_major_labels(session_list: list[str]) -> bool:
        labels = {label_of(sid) for sid in session_list}
        required = {label for label, sessions_for_label in label_to_sessions.items() if len(sessions_for_label) >= 2}
        return required.issubset(labels)

    if not covers_all_major_labels(val_sessions):
        for label, sessions_for_label in label_to_sessions.items():
            if len(sessions_for_label) < 2:
                continue
            if label in {label_of(sid) for sid in val_sessions}:
                continue
            candidate = next((sid for sid in train_sessions if label_of(sid) == label), None)
            if candidate is None:
                continue
            swap_out = min(val_sessions, key=lambda sid: sample_counts[sid]) if val_sessions else None
            if swap_out is not None:
                val_sessions.remove(swap_out)
                train_sessions.append(swap_out)
            train_sessions.remove(candidate)
            val_sessions.append(candidate)

    if not covers_all_major_labels(test_sessions):
        for label, sessions_for_label in label_to_sessions.items():
            if len(sessions_for_label) < 2:
                continue
            if label in {label_of(sid) for sid in test_sessions}:
                continue
            candidate = next((sid for sid in train_sessions if label_of(sid) == label), None)
            if candidate is None:
                continue
            swap_out = min(test_sessions, key=lambda sid: sample_counts[sid]) if test_sessions else None
            if swap_out is not None:
                test_sessions.remove(swap_out)
                train_sessions.append(swap_out)
            train_sessions.remove(candidate)
            test_sessions.append(candidate)

    train_sessions.sort()
    val_sessions.sort()
    test_sessions.sort()
    return train_sessions, val_sessions, test_sessions


def build_dataset(data_root: str, config, session_ids=None, label_order=None):
    root = Path(data_root)
    is_tsv = (root / 'dataset_manifest.json').exists() or any((root / child).is_dir() and (root / child / 'track_points.tsv').exists() for child in root.iterdir()) if root.exists() else False
    if is_tsv:
        return TrackOnlyTsvDataset(data_root=data_root, config=config, session_ids=session_ids, label_order=label_order)
    return RadarFusionDataset(
        data_root=data_root,
        session_ids=session_ids,
        label_order=label_order,
        sequence_length=config.sequence_length,
        rd_frames=config.rd_frames,
        rd_height=config.rd_height,
        rd_width=config.rd_width,
        rd_input_channels=config.rd_input_channels,
        feature_columns=config.track_feature_columns,
        global_feature_columns=config.track_global_feature_columns,
        include_rd=config.include_rd,
        conditional_ship_drone_session_ids=getattr(config, 'conditional_ship_drone_session_ids', ()),
    )



def build_session_split_manifest(dataset, statuses, config) -> SessionSplitManifest:
    global scan_statuses
    scan_statuses = list(statuses)
    ready_ids = {status.session_id for status in statuses if status.ready_for_training}
    session_to_indices: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        session_to_indices.setdefault(sample.session_id, []).append(index)
    session_to_indices = {session_id: indices for session_id, indices in session_to_indices.items() if session_id in ready_ids}
    train_sessions, val_sessions, test_sessions = _greedy_session_split(
        session_to_indices=session_to_indices,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
    )
    return SessionSplitManifest(train_sessions=train_sessions, val_sessions=val_sessions, test_sessions=test_sessions, session_statuses=statuses)



def build_split_subsets(dataset, manifest: SessionSplitManifest):
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    train_set = set(manifest.train_sessions)
    val_set = set(manifest.val_sessions)
    test_set = set(manifest.test_sessions)
    for index, sample in enumerate(dataset.samples):
        if sample.session_id in train_set:
            train_indices.append(index)
        elif sample.session_id in val_set:
            val_indices.append(index)
        elif sample.session_id in test_set:
            test_indices.append(index)
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def count_session_labels(statuses: list[SessionStatus]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for status in statuses:
        for label in status.labels:
            counter[label] += 1
    return counter


def build_session_manifest(data_root: str, include_rd: bool, split_map: dict[str, list[str]]) -> SessionSplitManifest:
    statuses = scan_session_directories(data_root=data_root, include_rd=include_rd)
    return SessionSplitManifest(
        train_sessions=list(split_map.get('train', [])),
        val_sessions=list(split_map.get('val', [])),
        test_sessions=list(split_map.get('test', [])),
        session_statuses=statuses,
    )


def save_session_manifest(manifest: SessionSplitManifest, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / 'training_session_manifest.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest.to_dict(), handle, indent=2, ensure_ascii=False)
    return output_path


def build_session_subsets(dataset, val_ratio: float, test_ratio: float, random_seed: int, session_ids=None):
    statuses = scan_session_directories(getattr(dataset, 'data_root', ''), include_rd=False)
    class _Config:
        pass
    cfg = _Config()
    cfg.val_ratio = val_ratio
    cfg.test_ratio = test_ratio
    cfg.random_seed = random_seed
    manifest = build_session_split_manifest(dataset, statuses, cfg)
    train_subset, val_subset, test_subset = build_split_subsets(dataset, manifest)
    split_map = {
        'train': manifest.train_sessions,
        'val': manifest.val_sessions,
        'test': manifest.test_sessions,
    }
    return train_subset, val_subset, test_subset, split_map
