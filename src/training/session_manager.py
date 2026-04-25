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
SCAN_CACHE_FILENAME = 'session_scan_cache.json'
SPLIT_CACHE_FILENAME = 'session_split_cache.json'


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
    source_name: str = 'Unknown'
    group_id: str | None = None


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
                source_name='TSVSession',
                group_id=session_dir.name,
            )
        )
    return statuses



def scan_session_directories(data_root: str, include_rd: bool = True) -> list[SessionStatus]:
    root = Path(data_root)
    if (root / 'dataset_manifest.json').exists() or any((root / child).is_dir() and (root / child / 'track_points.tsv').exists() for child in root.iterdir()) if root.exists() else False:
        return _scan_tsv_directories(data_root)

    if not root.exists():
        return []

    cache_path = root / SCAN_CACHE_FILENAME
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding='utf-8'))
            cached_include_rd = bool(payload.get('include_rd', True))
            if cached_include_rd == bool(include_rd):
                cached_statuses = payload.get('statuses', [])
                if isinstance(cached_statuses, list) and cached_statuses:
                    return [SessionStatus(**item) for item in cached_statuses]
        except Exception:
            pass

    statuses: list[SessionStatus] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        track_csv = session_dir / 'radar/track/track_data.csv'
        track_clip_file = session_dir / 'labels/labels_trackclip.jsonl'
        matched_file = session_dir / 'labels/labels_matched.jsonl'
        rd_clip_file = session_dir / 'radar/rd/rd_clips.h5'
        meta_file = session_dir / 'meta_session.json'

        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding='utf-8'))
            except Exception:
                meta = {}
        source_name = str(meta.get('source_name') or meta.get('source') or 'Unknown')
        parent_instance_id = meta.get('parent_instance_id') or meta.get('instance_id')
        sequence_id = meta.get('sequence_id')
        if source_name == 'Carrada' and parent_instance_id is not None and sequence_id is not None:
            group_id = f"Carrada::{sequence_id}::{parent_instance_id}"
        else:
            group_id = session_dir.name

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
                source_name=source_name,
                group_id=group_id,
            )
        )

    try:
        cache_payload = {
            'include_rd': bool(include_rd),
            'statuses': [asdict(item) for item in statuses],
        }
        cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass
    return statuses



def _split_group_pool(group_ids, group_to_sessions, group_sample_counts, status_map, val_ratio, test_ratio, random_seed):
    group_ids = list(group_ids)
    if not group_ids:
        return [], [], []

    random.Random(random_seed).shuffle(group_ids)

    def label_of_group(group_id: str) -> str:
        labels = []
        for sid in group_to_sessions[group_id]:
            status = status_map.get(sid)
            if status is not None:
                labels.extend(list(status.labels))
        return labels[0] if labels else 'Unknown'

    label_to_groups: dict[str, list[str]] = {}
    for group_id in group_ids:
        label_to_groups.setdefault(label_of_group(group_id), []).append(group_id)

    for label_groups in label_to_groups.values():
        label_groups.sort(key=lambda gid: group_sample_counts[gid], reverse=True)

    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []

    def split_label_groups(groups_for_label: list[str]) -> tuple[list[str], list[str], list[str]]:
        count = len(groups_for_label)
        if count == 1:
            return groups_for_label[:], [], []
        if count == 2:
            if val_ratio >= test_ratio:
                return [groups_for_label[0]], [groups_for_label[1]], []
            return [groups_for_label[0]], [], [groups_for_label[1]]

        desired_val = int(round(count * val_ratio)) if val_ratio > 0 else 0
        desired_test = int(round(count * test_ratio)) if test_ratio > 0 else 0
        if val_ratio > 0:
            desired_val = max(1, desired_val)
        if test_ratio > 0:
            desired_test = max(1, desired_test)

        max_allocatable = count - 1
        while desired_val + desired_test > max_allocatable:
            if desired_test >= desired_val and desired_test > 0:
                desired_test -= 1
            elif desired_val > 0:
                desired_val -= 1
            else:
                break

        train_count = max(1, count - desired_val - desired_test)
        train = groups_for_label[:train_count]
        val = groups_for_label[train_count:train_count + desired_val]
        test = groups_for_label[train_count + desired_val:train_count + desired_val + desired_test]
        return train, val, test

    for label in sorted(label_to_groups.keys()):
        groups_for_label = label_to_groups[label]
        train_part, val_part, test_part = split_label_groups(groups_for_label)
        train_groups.extend(train_part)
        val_groups.extend(val_part)
        test_groups.extend(test_part)

    return train_groups, val_groups, test_groups


def _greedy_session_split(session_to_indices, val_ratio, test_ratio, random_seed):
    sessions = list(session_to_indices.items())
    if not sessions:
        return [], [], []

    status_map = {status.session_id: status for status in scan_statuses}
    group_to_sessions: dict[str, list[str]] = {}
    sample_counts = {session_id: len(indices) for session_id, indices in sessions}

    for session_id, _ in sessions:
        status = status_map.get(session_id)
        group_id = status.group_id if status is not None and status.group_id is not None else session_id
        group_to_sessions.setdefault(group_id, []).append(session_id)

    group_sample_counts = {gid: sum(sample_counts[sid] for sid in sids) for gid, sids in group_to_sessions.items()}

    source_to_groups: dict[str, list[str]] = {}
    for group_id, session_ids in group_to_sessions.items():
        first_session = session_ids[0]
        status = status_map.get(first_session)
        source_name = status.source_name if status is not None else 'Unknown'
        source_to_groups.setdefault(source_name, []).append(group_id)

    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []
    for source_name, source_groups in sorted(source_to_groups.items()):
        source_train, source_val, source_test = _split_group_pool(
            group_ids=source_groups,
            group_to_sessions=group_to_sessions,
            group_sample_counts=group_sample_counts,
            status_map=status_map,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
        train_groups.extend(source_train)
        val_groups.extend(source_val)
        test_groups.extend(source_test)

    train_sessions = sorted([sid for gid in train_groups for sid in group_to_sessions[gid]])
    val_sessions = sorted([sid for gid in val_groups for sid in group_to_sessions[gid]])
    test_sessions = sorted([sid for gid in test_groups for sid in group_to_sessions[gid]])
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
        label_granularity=getattr(config, 'label_granularity', 'coarse'),
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
    data_root = Path(getattr(dataset, 'data_root', ''))
    cache_path = data_root / SPLIT_CACHE_FILENAME if str(data_root) else None
    selected_session_ids = sorted({sample.session_id for sample in getattr(dataset, 'samples', [])})
    cache_key = {
        'val_ratio': float(config.val_ratio),
        'test_ratio': float(config.test_ratio),
        'random_seed': int(config.random_seed),
        'selected_session_ids': selected_session_ids,
        'ready_sessions': sorted(status.session_id for status in statuses if status.ready_for_training),
        'group_ids': sorted(f'{status.session_id}:{status.group_id}:{status.source_name}:{"|".join(status.labels)}' for status in statuses),
    }
    if cache_path is not None and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding='utf-8'))
            if payload.get('cache_key') == cache_key:
                manifest_payload = payload.get('manifest', {})
                return SessionSplitManifest(
                    train_sessions=list(manifest_payload.get('train_sessions', [])),
                    val_sessions=list(manifest_payload.get('val_sessions', [])),
                    test_sessions=list(manifest_payload.get('test_sessions', [])),
                    session_statuses=statuses,
                )
        except Exception:
            pass

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
    manifest = SessionSplitManifest(train_sessions=train_sessions, val_sessions=val_sessions, test_sessions=test_sessions, session_statuses=statuses)
    if cache_path is not None:
        try:
            cache_payload = {
                'cache_key': cache_key,
                'manifest': {
                    'train_sessions': manifest.train_sessions,
                    'val_sessions': manifest.val_sessions,
                    'test_sessions': manifest.test_sessions,
                },
            }
            cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass
    return manifest



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
