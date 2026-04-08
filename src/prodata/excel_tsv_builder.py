from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import re
from collections import Counter, defaultdict
from shutil import copy2, copytree, rmtree
from typing import Iterator

NS_MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
NS_REL = '{http://schemas.openxmlformats.org/package/2006/relationships}'
NS_DOCREL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
SOURCE_FORMAT = 'excel_bz_tck_v1'
KEEP_DUPLICATE_SESSION_IDS = {'20250327_191432'}
DROP_DUPLICATE_SESSION_IDS = {'20241128_191432'}


LABEL_TOKEN_MAP = {
    "人": "human",
    "鸟": "bird",
    "无人机": "drone",
    "微型机": "micro_aircraft",
    "小型机": "small_aircraft",
    "大型机": "large_aircraft",
    "车": "vehicle",
    "船": "vessel",
    "疑似气球": "suspected_balloon",
    "待定": "unknown",
    "杂波": "clutter",
}

NOTE_TOKEN_MAP = {
    "非机动": "non_maneuvering",
    "机动": "maneuvering",
    "往返": "back_and_forth",
    "圆周": "circular",
    "8字": "figure_eight",
    "干扰": "interference",
    "栅格": "grid",
    "矩形": "rectangle",
    "后期固杂": "post_static_clutter",
    "固杂": "static_clutter",
}


@dataclass(slots=True)
class PreparedTsvSession:
    session_id: str
    output_dir: Path
    num_tracks: int
    num_track_points: int
    labels: list[str]
    notes: list[str]
    duplicate_status: str



def col_to_index(column_name: str) -> int:
    index = 0
    for character in column_name:
        if 'A' <= character <= 'Z':
            index = index * 26 + (ord(character) - ord('A') + 1)
    return index - 1



def parse_shared_strings(zip_file: ZipFile) -> list[str]:
    path = 'xl/sharedStrings.xml'
    if path not in zip_file.namelist():
        return []
    root = ET.fromstring(zip_file.read(path))
    strings: list[str] = []
    for shared_item in root.findall(f'{NS_MAIN}si'):
        fragments: list[str] = []
        for node in shared_item.iter():
            if node.tag == f'{NS_MAIN}t':
                fragments.append(node.text or '')
        strings.append(''.join(fragments))
    return strings



def read_relationships(zip_file: ZipFile, rel_path: str) -> dict[str, str]:
    root = ET.fromstring(zip_file.read(rel_path))
    relationships: dict[str, str] = {}
    for rel in root.findall(f'{NS_REL}Relationship'):
        relationships[rel.attrib['Id']] = rel.attrib['Target']
    return relationships



def workbook_sheets(zip_file: ZipFile) -> list[tuple[str, str]]:
    workbook_root = ET.fromstring(zip_file.read('xl/workbook.xml'))
    relationships = read_relationships(zip_file, 'xl/_rels/workbook.xml.rels')
    sheet_entries: list[tuple[str, str]] = []
    for sheet in workbook_root.find(f'{NS_MAIN}sheets').findall(f'{NS_MAIN}sheet'):
        relationship_id = sheet.attrib[f'{NS_DOCREL}id']
        target = relationships[relationship_id]
        if not target.startswith('worksheets/'):
            target = 'worksheets/' + Path(target).name
        sheet_entries.append((sheet.attrib['name'], 'xl/' + target))
    return sheet_entries



def iter_sheet_rows(zip_file: ZipFile, sheet_path: str, shared_strings: list[str]) -> Iterator[list[str]]:
    sheet_root = ET.fromstring(zip_file.read(sheet_path))
    sheet_data = sheet_root.find(f'{NS_MAIN}sheetData')
    if sheet_data is None:
        return
    for row in sheet_data.findall(f'{NS_MAIN}row'):
        values: dict[int, str] = {}
        max_col = -1
        for cell in row.findall(f'{NS_MAIN}c'):
            reference = cell.attrib.get('r', '')
            match = re.match(r'([A-Z]+)(\d+)', reference)
            if not match:
                continue
            column_index = col_to_index(match.group(1))
            max_col = max(max_col, column_index)
            cell_type = cell.attrib.get('t')
            value_node = cell.find(f'{NS_MAIN}v')
            text = ''
            if cell_type == 's' and value_node is not None and value_node.text is not None:
                text = shared_strings[int(value_node.text)]
            elif cell_type == 'inlineStr':
                inline_text = cell.find(f'{NS_MAIN}is/{NS_MAIN}t')
                text = inline_text.text if inline_text is not None and inline_text.text is not None else ''
            elif value_node is not None and value_node.text is not None:
                text = value_node.text
            values[column_index] = text
        if max_col >= 0:
            resolved_row = [''] * (max_col + 1)
            for index, value in values.items():
                resolved_row[index] = value
            yield resolved_row



def read_first_sheet_table(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as zip_file:
        shared_strings = parse_shared_strings(zip_file)
        sheets = workbook_sheets(zip_file)
        if not sheets:
            return []
        _, sheet_path = sheets[0]
        rows = list(iter_sheet_rows(zip_file, sheet_path, shared_strings))
    if not rows:
        return []
    header = rows[0]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [''] * (len(header) - len(row))
        records.append({header[i]: row[i] if i < len(row) else '' for i in range(len(header))})
    return records



def to_float(value: str) -> float:
    if value in {'', '(null)'}:
        return 0.0
    return float(value)



def normalize_notes_id(value: str) -> str:
    return ';'.join(part.strip() for part in value.split(';') if part.strip()) if value else ''


def normalize_label_token(value: str) -> str:
    normalized = value.strip()
    return LABEL_TOKEN_MAP.get(normalized, normalized.lower().replace(' ', '_')) if normalized else ''


def normalize_notes_token(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return ''
    parts = [part.strip() for part in normalized.split(';') if part.strip()]
    return ';'.join(NOTE_TOKEN_MAP.get(part, part.lower().replace(' ', '_')) for part in parts)



def collect_excel_pairs(raw_root: str | Path) -> dict[str, dict[str, Path]]:
    root = Path(raw_root)
    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(root.glob('*.xlsx')):
        prefix, rest = path.stem.split('_', 1)
        pairs.setdefault(rest, {})[prefix] = path
    return {session_id: mapping for session_id, mapping in pairs.items() if {'bz', 'tck'} <= set(mapping)}



def deduplicate_pairs(pairs: dict[str, dict[str, Path]]) -> tuple[dict[str, dict[str, Path]], list[dict[str, str]]]:
    filtered: dict[str, dict[str, Path]] = {}
    duplicate_rows: list[dict[str, str]] = []
    for session_id, mapping in sorted(pairs.items()):
        if session_id in DROP_DUPLICATE_SESSION_IDS:
            duplicate_rows.append({
                'session_id': session_id,
                'action': 'drop',
                'reason': 'confirmed_duplicate_of_20250327_191432',
                'kept_session_id': '20250327_191432',
                'bz_file': mapping['bz'].name,
                'tck_file': mapping['tck'].name,
            })
            continue
        status = 'kept'
        if session_id in KEEP_DUPLICATE_SESSION_IDS:
            status = 'keep_preferred'
        duplicate_rows.append({
            'session_id': session_id,
            'action': status,
            'reason': 'manually_reviewed' if session_id in KEEP_DUPLICATE_SESSION_IDS else '',
            'kept_session_id': session_id,
            'bz_file': mapping['bz'].name,
            'tck_file': mapping['tck'].name,
        })
        filtered[session_id] = mapping
    return filtered, duplicate_rows



def build_session_records(session_id: str, bz_path: Path, tck_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object], list[dict[str, str]]]:
    bz_rows = read_first_sheet_table(bz_path)
    tck_rows = read_first_sheet_table(tck_path)

    bz_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in bz_rows:
        key = (row.get('StartTimer', ''), row.get('Batch', ''))
        bz_map[key] = row

    grouped_points: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tck_rows:
        key = (row.get('StartTimer', ''), row.get('Batch', ''))
        grouped_points[key].append(row)

    track_points: list[dict[str, object]] = []
    track_labels: list[dict[str, object]] = []
    track_notes: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []

    label_values: set[str] = set()
    notes_values: set[str] = set()

    for key, label_row in sorted(bz_map.items()):
        start_timer, batch = key
        points = grouped_points.get(key, [])
        if not points:
            rejected.append({
                'session_id': session_id,
                'object_key': start_timer,
                'track_id': batch,
                'reason': 'missing_tck_points',
                'source_file': tck_path.name,
            })
            continue

        object_key = start_timer
        label_display_name = label_row.get('MarkType', '').strip()
        label_name = normalize_label_token(label_display_name)
        label_id = label_row.get('MarkTypeID', '').strip()
        notes_display_name = label_row.get('Notes', '').strip()
        notes_name = normalize_notes_token(notes_display_name)
        notes_id = normalize_notes_id(label_row.get('NotesID', '').strip())

        label_values.add(label_name)
        if notes_name:
            notes_values.add(notes_name)

        track_labels.append({
            'session_id': session_id,
            'track_id': batch,
            'object_key': object_key,
            'label_name': label_name,
            'label_display_name': label_display_name,
            'label_id': label_id,
            'label_source': 'manual_bz',
            'is_valid': 1,
        })

        track_notes.append({
            'session_id': session_id,
            'track_id': batch,
            'object_key': object_key,
            'notes_name': notes_name,
            'notes_display_name': notes_display_name,
            'notes_id': notes_id,
            'notes_source': 'manual_bz',
            'is_multilabel': int(';' in notes_id) if notes_id else 0,
        })

        sorted_points = sorted(points, key=lambda item: int(float(item.get('Time', '0') or '0')))
        for point in sorted_points:
            amplitude_db = to_float(point.get('Amplitude', '0'))
            noise_db = to_float(point.get('Noise', '0'))
            track_points.append({
                'session_id': session_id,
                'track_id': batch,
                'object_key': object_key,
                'timestamp_ms': int(float(point.get('Time', '0') or '0')),
                'range_m': to_float(point.get('Distance', '0')),
                'azimuth_deg': to_float(point.get('Azimuth', '0')),
                'doppler_speed_mps': to_float(point.get('Speed_R', '0')),
                'speed_mps': to_float(point.get('Speed', '0')),
                'height_m': to_float(point.get('Height', '0')),
                'elevation_deg': to_float(point.get('Elevation', '0')),
                'course_deg': to_float(point.get('Course', '0')),
                'amplitude_db': amplitude_db,
                'noise_db': noise_db,
                'snr_db': amplitude_db - noise_db,
                'source_file': tck_path.name,
                'source_format': SOURCE_FORMAT,
            })

    for key, points in sorted(grouped_points.items()):
        if key not in bz_map:
            start_timer, batch = key
            rejected.append({
                'session_id': session_id,
                'object_key': start_timer,
                'track_id': batch,
                'reason': 'missing_bz_label',
                'source_file': tck_path.name,
            })

    meta = {
        'session_id': session_id,
        'raw_bz_file': bz_path.name,
        'raw_tck_file': tck_path.name,
        'source_format': SOURCE_FORMAT,
        'num_tracks': len(track_labels),
        'num_track_points': len(track_points),
        'label_set': sorted(label_values),
        'notes_set': sorted(notes_values),
        'has_rd': False,
        'dedup_status': 'keep_preferred' if session_id in KEEP_DUPLICATE_SESSION_IDS else 'unique',
        'processor_version': 'tsv_v1.0',
    }

    return track_points, track_labels, track_notes, meta, rejected



def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='	')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def prepare_excel_tsv_sessions(raw_root: str | Path = 'pro/excel', output_root: str | Path = 'pro/tsv', session_ids: list[str] | None = None) -> list[PreparedTsvSession]:
    pairs = collect_excel_pairs(raw_root)
    pairs, duplicate_rows = deduplicate_pairs(pairs)
    if session_ids:
        session_filter = set(session_ids)
        pairs = {session_id: mapping for session_id, mapping in pairs.items() if session_id in session_filter}

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    all_rejected: list[dict[str, str]] = []
    prepared: list[PreparedTsvSession] = []
    manifest_rows: list[dict[str, object]] = []
    label_counter: Counter[str] = Counter()
    notes_counter: Counter[str] = Counter()

    for session_id, mapping in sorted(pairs.items()):
        session_dir = output_root_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        track_points, track_labels, track_notes, meta, rejected = build_session_records(session_id, mapping['bz'], mapping['tck'])
        all_rejected.extend(rejected)

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
        (session_dir / 'session_meta.json').write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding='utf-8')

        for row in track_labels:
            label_counter[str(row['label_name'])] += 1
        for row in track_notes:
            if row['notes_name']:
                notes_counter[str(row['notes_name'])] += 1

        prepared.append(
            PreparedTsvSession(
                session_id=session_id,
                output_dir=session_dir,
                num_tracks=meta['num_tracks'],
                num_track_points=meta['num_track_points'],
                labels=meta['label_set'],
                notes=meta['notes_set'],
                duplicate_status=meta['dedup_status'],
            )
        )
        manifest_rows.append(meta)

    write_tsv(
        output_root_path / 'duplicates_report.tsv',
        duplicate_rows,
        ['session_id', 'action', 'reason', 'kept_session_id', 'bz_file', 'tck_file'],
    )
    write_tsv(
        output_root_path / 'rejected_records.tsv',
        all_rejected,
        ['session_id', 'object_key', 'track_id', 'reason', 'source_file'],
    )
    dataset_manifest = {
        'dataset_version': 'tsv_v1.0',
        'source_format': SOURCE_FORMAT,
        'num_sessions': len(prepared),
        'sessions': [item.session_id for item in prepared],
        'label_counts': dict(sorted(label_counter.items())),
        'notes_counts': dict(sorted(notes_counter.items())),
        'duplicates_report': 'duplicates_report.tsv',
        'rejected_records': 'rejected_records.tsv',
    }
    (output_root_path / 'dataset_manifest.json').write_text(json.dumps(dataset_manifest, ensure_ascii=True, indent=2), encoding='utf-8')
    return prepared



def sync_tsv_sessions(processed_root: str | Path = 'pro/tsv', dataset_root: str | Path = 'data/tsv', session_ids: list[str] | None = None) -> list[str]:
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

    for filename in ['dataset_manifest.json', 'duplicates_report.tsv', 'rejected_records.tsv']:
        source_file = processed_root_path / filename
        if source_file.exists():
            copy2(source_file, dataset_root_path / filename)
    return copied
