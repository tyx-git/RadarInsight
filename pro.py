#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pro.processing.carrada_session_builder import prepare_carrada_sessions, sync_carrada_sessions
from pro.processing.cq08_mat_session_builder import prepare_cq08_mat_sessions, sync_cq08_mat_sessions
from pro.processing.cq08_bird_bucket_builder import build_cq08_bird_buckets
from pro.processing.label_interface_builder import build_label_interface
from pro.processing.session_unifier import unify_session_roots, rebuild_sessions_root
from src.utils.logger import logger


def _guard_main_sessions_root_cleanup(dataset_root: str, clean_outputs: bool, command_name: str) -> None:
    normalized = Path(dataset_root)
    if clean_outputs and normalized.as_posix().rstrip('/') == 'data/sessions':
        raise ValueError(
            f"{command_name} refuses to delete the canonical data/sessions root. "
            "Please use a staging dataset_root such as data/sessions_cq08_mat or data/sessions_carrada, "
            "then merge into data/sessions via pro.py unify-sessions."
        )


def cmd_prepare_carrada(args):
    raw_root = args.raw_root
    output_root = args.output_root
    dataset_root = args.dataset_root
    frame_margin = int(args.frame_margin)
    sub_session_window_specs = args.sub_session_window_specs or ['128:64', '64:32', '192:96']

    _guard_main_sessions_root_cleanup(dataset_root, bool(args.clean_outputs), 'prepare-carrada')

    _guard_main_sessions_root_cleanup(dataset_root, bool(args.clean_outputs), 'prepare-cq08-mat')

    if args.clean_outputs:
        output_path = Path(output_root)
        dataset_path = Path(dataset_root)
        if output_path.exists():
            shutil.rmtree(output_path)
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        logger.info(f"[*] Cleared previous Carrada outputs: {output_root}")
        logger.info(f"[*] Cleared previous Carrada dataset sync: {dataset_root}")

    prepared = prepare_carrada_sessions(
        carrada_root=raw_root,
        output_root=output_root,
        sequence_ids=args.sequence_ids or None,
        frame_margin=frame_margin,
        sub_session_window_specs=sub_session_window_specs,
    )
    copied = sync_carrada_sessions(
        processed_root=output_root,
        dataset_root=dataset_root,
        sequence_ids=args.sequence_ids or None,
    )

    logger.info("=" * 30)
    logger.info("Carrada Session Summary")
    logger.info("=" * 30)
    logger.info(f"Prepared sessions: {len(prepared)}")
    logger.info(f"Copied sessions: {len(copied)}")
    logger.info(f"Processed root: {output_root}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info("=" * 30)


def cmd_prepare_cq08_mat(args):
    raw_root = args.raw_root
    output_root = args.output_root
    dataset_root = args.dataset_root

    _guard_main_sessions_root_cleanup(dataset_root, bool(args.clean_outputs), 'prepare-cq08-mat')

    if args.clean_outputs:
        output_path = Path(output_root)
        dataset_path = Path(dataset_root)
        if output_path.exists():
            shutil.rmtree(output_path)
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        logger.info(f"[*] Cleared previous CQ-08 MAT outputs: {output_root}")
        logger.info(f"[*] Cleared previous CQ-08 MAT dataset sync: {dataset_root}")

    prepared = prepare_cq08_mat_sessions(
        source_root=raw_root,
        output_root=output_root,
        session_ids=args.session_ids or None,
        batch_nos=args.batch_nos or None,
    )
    copied = sync_cq08_mat_sessions(
        processed_root=output_root,
        dataset_root=dataset_root,
        session_ids=args.session_ids or None,
    )

    logger.info("=" * 30)
    logger.info("CQ-08 MAT Session Summary")
    logger.info("=" * 30)
    logger.info(f"Prepared sessions: {len(prepared)}")
    logger.info(f"Copied sessions: {len(copied)}")
    logger.info(f"Processed root: {output_root}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info("=" * 30)


def cmd_unify_sessions(args):
    manifest = rebuild_sessions_root(
        source_roots=args.source_roots,
        final_sessions_root=args.output_root,
        clean_old_roots=not bool(args.keep_old_roots),
        strict=bool(args.strict),
    )
    logger.info("=" * 30)
    logger.info("Unified Session Summary")
    logger.info("=" * 30)
    logger.info(f"Final sessions root: {manifest['final_sessions_root']}")
    logger.info(f"Num sessions: {manifest['num_sessions']}")
    logger.info(f"Source counts: {manifest['source_counts']}")
    logger.info(f"Label counts: {manifest['label_counts']}")
    logger.info(f"Skipped sessions: {len(manifest['skipped_sessions'])}")
    logger.info(f"Cleanup performed: {manifest['cleanup_performed']}")
    logger.info("=" * 30)


def cmd_clean_data_roots(args):
    data_root = Path(args.data_root)
    preserved_name = args.preserve_name
    removed = []
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == preserved_name:
            continue
        if child.name.startswith('sessions'):
            shutil.rmtree(child)
            removed.append(child.name)
    logger.info("=" * 30)
    logger.info("Data Root Cleanup Summary")
    logger.info("=" * 30)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Preserved: {preserved_name}")
    logger.info(f"Removed: {removed}")
    logger.info("=" * 30)


def cmd_clean_zero_clip_sessions(args):
    data_root = Path(args.data_root)
    removed = []
    for session_dir in data_root.iterdir():
        if not session_dir.is_dir():
            continue
        clip_file = session_dir / 'labels' / 'labels_trackclip.jsonl'
        if not clip_file.exists():
            continue
        try:
            num_clips = sum(1 for _ in clip_file.open('r', encoding='utf-8'))
        except Exception:
            continue
        if num_clips == 0:
            removed.append(session_dir.name)
            shutil.rmtree(session_dir)

    cache_file = data_root / 'session_scan_cache.json'
    if cache_file.exists():
        cache_file.unlink()

    logger.info("=" * 30)
    logger.info("Zero-Clip Session Cleanup Summary")
    logger.info("=" * 30)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Removed sessions: {len(removed)}")
    logger.info(f"Removed ids: {removed}")
    logger.info("=" * 30)


def cmd_bucket_cq08_bird(args):
    summary = build_cq08_bird_buckets(
        data_root=args.data_root,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    logger.info("=" * 30)
    logger.info("CQ-08 Bird Bucket Summary")
    logger.info("=" * 30)
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Num sessions: {summary['num_sessions']}")
    logger.info(f"Bucket counts: {summary['bucket_counts']}")
    logger.info(f"CSV: {summary['output_csv']}")
    logger.info(f"JSON: {args.output_json}")
    logger.info("=" * 30)


def cmd_build_label_interface(args):
    summary = build_label_interface(
        data_root=args.data_root,
        output_json=args.output_json,
        rewrite_meta=not bool(args.no_rewrite_meta),
    )
    logger.info("=" * 30)
    logger.info("Label Interface Summary")
    logger.info("=" * 30)
    logger.info(f"Data root: {summary['data_root']}")
    logger.info(f"Updated sessions: {summary['updated_sessions']}")
    logger.info(f"Coarse counts: {summary['coarse_counts']}")
    logger.info(f"Detail counts: {summary['detail_counts']}")
    logger.info(f"JSON: {args.output_json}")
    logger.info("=" * 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone preprocessing entrypoint for RadarInsight")
    subparsers = parser.add_subparsers(dest="command", required=True)

    carrada = subparsers.add_parser("prepare-carrada", help="Build standard Carrada sessions with pseudo-track + RD")
    carrada.add_argument("--raw-root", default="pro/Carrada", help="Carrada root directory")
    carrada.add_argument("--output-root", default="pro/processed/carrada_sessions", help="Processed Carrada session root")
    carrada.add_argument("--dataset-root", default="data/sessions_carrada", help="Destination dataset root for synced sessions")
    carrada.add_argument("--sequence-ids", nargs="*", default=None, help="Optional Carrada sequence ids to process")
    carrada.add_argument("--frame-margin", type=int, default=5, help="Frame margin added on both sides of each instance active span")
    carrada.add_argument("--sub-session-window-specs", nargs='*', default=None, help="Carrada mixed window specs, e.g. 128:64 64:32 192:96")
    carrada.add_argument("--clean-outputs", action="store_true", help="Delete previous processed and synced Carrada outputs before rebuild")

    cq08mat = subparsers.add_parser("prepare-cq08-mat", help="Build standard sessions from CQ-08 processed MAT files")
    cq08mat.add_argument("--raw-root", default="pro/23/CQ-08", help="CQ-08 root containing MAT, ??, ??")
    cq08mat.add_argument("--output-root", default="pro/processed/cq08_mat_sessions", help="Processed CQ-08 MAT session root")
    cq08mat.add_argument("--dataset-root", default="data/sessions_cq08_mat", help="Destination dataset root for synced sessions")
    cq08mat.add_argument("--session-ids", nargs="*", default=None, help="Optional CQ-08 MAT session ids to process")
    cq08mat.add_argument("--batch-nos", nargs="*", type=int, default=None, help="Optional CQ-08 MAT batch numbers to process")
    cq08mat.add_argument("--clean-outputs", action="store_true", help="Delete previous processed and synced CQ-08 MAT outputs before rebuild")

    unify = subparsers.add_parser("unify-sessions", help="Merge multiple standardized session roots into one unified root")
    unify.add_argument("--source-roots", nargs="+", required=True, help="Input session roots to merge")
    unify.add_argument("--output-root", default="data/sessions", help="Final unified sessions root (default: data/sessions)")
    unify.add_argument("--keep-old-roots", action="store_true", help="Keep old source roots after successful rebuild (default behavior is to delete them)")
    unify.add_argument("--strict", action="store_true", help="Enable strict validation behavior")

    clean = subparsers.add_parser("clean-data-roots", help="Remove all session-like roots under data/ except the preserved one")
    clean.add_argument("--data-root", default="data", help="Top-level data directory")
    clean.add_argument("--preserve-name", default="sessions", help="Directory name to preserve")

    clean_zero = subparsers.add_parser("clean-zero-clips", help="Remove sessions whose labels_trackclip.jsonl is empty")
    clean_zero.add_argument("--data-root", default="data/sessions", help="Canonical sessions root")

    bucket_bird = subparsers.add_parser("bucket-cq08-bird", help="Build bucket analysis files for CQ-08 BirdTarget sessions")
    bucket_bird.add_argument("--data-root", default="data/sessions", help="Canonical sessions root")
    bucket_bird.add_argument("--output-csv", default="artifacts/cq08_bird_buckets.csv", help="CSV output path")
    bucket_bird.add_argument("--output-json", default="artifacts/cq08_bird_bucket_summary.json", help="JSON summary output path")

    label_interface = subparsers.add_parser("build-label-interface", help="Add coarse+detail label interface to all sessions")
    label_interface.add_argument("--data-root", default="data/sessions", help="Canonical sessions root")
    label_interface.add_argument("--output-json", default="artifacts/label_interface_summary.json", help="JSON summary output path")
    label_interface.add_argument("--no-rewrite-meta", action="store_true", help="Do not rewrite meta_session.json, only compute summary")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare-carrada":
        cmd_prepare_carrada(args)
    elif args.command == "prepare-cq08-mat":
        cmd_prepare_cq08_mat(args)
    elif args.command == "unify-sessions":
        cmd_unify_sessions(args)
    elif args.command == "clean-data-roots":
        cmd_clean_data_roots(args)
    elif args.command == "clean-zero-clips":
        cmd_clean_zero_clip_sessions(args)
    elif args.command == "bucket-cq08-bird":
        cmd_bucket_cq08_bird(args)
    elif args.command == "build-label-interface":
        cmd_build_label_interface(args)


if __name__ == "__main__":
    main()
