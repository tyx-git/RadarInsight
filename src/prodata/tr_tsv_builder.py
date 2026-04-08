from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, copytree, rmtree

import h5py

from src.prodata.excel_tsv_builder import write_tsv

SOURCE_FORMAT = 'mat_tr_only_v1'


@dataclass(slots=True)
class PreparedTrSession:
    session_id: str
    output_dir: Path
    label_name: str
    num_track_points: int


def _map_raw_label_to_token(raw_label: str) -> str | None:
    label = raw_label.strip().lower()
    if 'bird' in label:
        return 'bird'
    if 'rotary drone' in label:
        return 'drone'
    if 'fixed-wing' in label:
        return 'large_aircraft'
    if 'helicopter' in label:
        return 'large_aircraft'
    if 'speedboat' in label:
        return 'vehicle'
    if 'passenger ship' in label:
        return 'vessel'
    return None


def _token_to_display(token: str) -> str:
    mapping = {
        'bird': '鸟',
        'drone': '无人机',
        'large_aircraft': '大型机',
        'vehicle': '车',
        'vessel': '船',
    }
    return mapping.get(token, token)


def _read_vector(handle: h5py.File, key: str) -> list[float]:
    if key not in handle:
        return []
    data = handle[key][()]
    flat = data.reshape(-1).tolist()
    return [float(value) for value in flat]


def _parse_tr_file(path: Path) -> tuple[str, str, list[dict[str, object]]] | None:
    match = re.match(r'^(\d{14})_DAUR_TR_(.+)_([0-9]{2})_([0-9]+)\.mat$', path.name)
    if not match:
        return None
    session_id = match.group(1)
    raw_label = match.group(2)
    track_id = f'{match.group(3)}_{match.group(4)}'
    label_token = _map_raw_label_to_token(raw_label)
    if label_token is None:
        return None

    with h5py.File(path, 'r') as handle:
        range_values = _read_vector(handle, 'R_m') or _read_vector(handle, 'R')
        azimuth_values = _read_vector(handle, 'A_m') or _read_vector(handle, 'A')
        elevation_values = _read_vector(handle, 'E_m') or _read_vector(handle, 'E')
        doppler_values = _read_vector(handle, 'V_m') or _read_vector(handle, 'V')
        snr_values = _read_vector(handle, 'SNR')
        data_time_values = _read_vector(handle, 'DATA_time')
        frame_values = _read_vector(handle, 'Iframecnt')

    length = min(len(range_values), len(azimuth_values), len(elevation_values), len(doppler_values))
    if length == 0:
        return None

    if len(data_time_values) < length:
        if frame_values:
            data_time_values = frame_values[:length]
        else:
            data_time_values = list(range(length))

    rows: list[dict[str, object]] = []
    object_key = f'{session_id}_{track_id}'
    for index in range(length):
        amplitude = float(snr_values[index]) if index < len(snr_values) else 0.0
        noise = 0.0
        rows.append(
            {
                'session_id': session_id,
                'track_id': track_id,
                'object_key': object_key,
                'timestamp_ms': int(float(data_time_values[index]) * 1000.0),
                'range_m': float(range_values[index]),
                'azimuth_deg': float(azimuth_values[index]),
                'doppler_speed_mps': float(doppler_values[index]),
                'speed_mps': abs(float(doppler_values[index])),
                'height_m': 0.0,
                'elevation_deg': float(elevation_values[index]),
                'course_deg': 0.0,
                'amplitude_db': amplitude,
                'noise_db': noise,
                'snr_db': amplitude - noise,
                'source_file': path.name,
                'source_format': SOURCE_FORMAT,
            }
        )
    return session_id, label_token, rows


def prepare_tr_tsv_sessions(raw_root: str | Path = 'pro/raw/TR', output_root: str | Path = 'pro/tsv_tr') -> list[PreparedTrSession]:
    raw_root_path = Path(raw_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    session_buffers: dict[str, list[dict[str, object]]] = {}
    session_labels: dict[str, set[str]] = {}

    for path in sorted(raw_root_path.glob('*.mat')):
        if '(1)' in path.name:
            continue
        parsed = _parse_tr_file(path)
        if parsed is None:
            continue
        session_id, label_token, rows = parsed
        session_buffers.setdefault(session_id, []).extend(rows)
        session_labels.setdefault(session_id, set()).add(label_token)

    prepared: list[PreparedTrSession] = []
    manifest_rows: list[dict[str, object]] = []
    for session_id, rows in sorted(session_buffers.items()):
        labels = sorted(session_labels.get(session_id, []))
        if len(labels) != 1:
            continue
        label_token = labels[0]
        session_dir = output_root_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        track_points = rows
        object_keys = sorted({str(row['object_key']) for row in rows})
        track_ids = sorted({str(row['track_id']) for row in rows})
        track_labels = [
            {
                'session_id': session_id,
                'track_id': track_id,
                'object_key': f'{session_id}_{track_id}',
                'label_name': label_token,
                'label_display_name': _token_to_display(label_token),
                'label_id': label_token,
                'label_source': 'filename_tr',
                'is_valid': 1,
            }
            for track_id in track_ids
        ]
        track_notes = [
            {
                'session_id': session_id,
                'track_id': track_id,
                'object_key': f'{session_id}_{track_id}',
                'notes_name': '',
                'notes_display_name': '',
                'notes_id': '',
                'notes_source': 'filename_tr',
                'is_multilabel': 0,
            }
            for track_id in track_ids
        ]

        write_tsv(
            session_dir / 'track_points.tsv',
            track_points,
            [
                'session_id', 'track_id', 'object_key', 'timestamp_ms', 'range_m', 'azimuth_deg',
                'doppler_speed_mps', 'speed_mps', 'height_m', 'elevation_deg', 'course_deg',
                'amplitude_db', 'noise_db', 'snr_db', 'source_file', 'source_format',
            ],
        )
        write_tsv(
            session_dir / 'track_labels.tsv',
            track_labels,
            ['session_id', 'track_id', 'object_key', 'label_name', 'label_display_name', 'label_id', 'label_source', 'is_valid'],
        )
        write_tsv(
            session_dir / 'track_notes.tsv',
            track_notes,
            ['session_id', 'track_id', 'object_key', 'notes_name', 'notes_display_name', 'notes_id', 'notes_source', 'is_multilabel'],
        )
        meta = {
            'session_id': session_id,
            'source_format': SOURCE_FORMAT,
            'raw_tr_files': sorted({str(row['source_file']) for row in rows}),
            'num_tracks': len(track_ids),
            'num_track_points': len(track_points),
            'label_set': [label_token],
            'notes_set': [],
            'has_rd': False,
            'dedup_status': 'kept',
            'processor_version': 'tr_tsv_v1',
        }
        (session_dir / 'session_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        prepared.append(PreparedTrSession(session_id=session_id, output_dir=session_dir, label_name=label_token, num_track_points=len(track_points)))
        manifest_rows.append(meta)

    dataset_manifest = {
        'dataset_version': 'tr_tsv_v1',
        'source_format': SOURCE_FORMAT,
        'num_sessions': len(prepared),
        'sessions': [item.session_id for item in prepared],
        'labels': {item.session_id: item.label_name for item in prepared},
    }
    (output_root_path / 'dataset_manifest.json').write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return prepared


def sync_tr_tsv_sessions(processed_root: str | Path = 'pro/tsv_tr', dataset_root: str | Path = 'data/tsv', session_ids: list[str] | None = None) -> list[str]:
    processed_root_path = Path(processed_root)
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    session_filter = set(session_ids) if session_ids else None
    for session_dir in sorted(path for path in processed_root_path.iterdir() if path.is_dir()):
        if session_filter is not None and session_dir.name not in session_filter:
            continue
        target_dir = dataset_root_path / session_dir.name
        if target_dir.exists():
            rmtree(target_dir)
        copytree(session_dir, target_dir)
        copied.append(session_dir.name)
    manifest_file = processed_root_path / 'dataset_manifest.json'
    if manifest_file.exists():
        copy2(manifest_file, dataset_root_path / 'dataset_manifest_tr_supplement.json')
    return copied
