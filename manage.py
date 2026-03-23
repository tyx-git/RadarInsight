#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
import sys
from pathlib import Path

import yaml

sys.path.append(os.path.abspath("."))

from src.labeling.truth_matcher import TruthMatcher
from src.preprocessing.slicer import DataSlicer
from src.prodata.mat_session_builder import prepare_real_sessions, sync_prepared_sessions
from src.qc.integrity_check import DataIntegrityChecker
from src.utils.logger import logger


def load_config(config_path: str = "config/system_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def cmd_prepare_pro(args, config):
    radar_config = dict(config.get("radar", {}))
    processing_config = dict(config.get("pro_processing", {}))
    preprocessing_config = dict(config.get("preprocessing", {}))
    raw_root = args.raw_root or processing_config.get("raw_root", "pro/raw")
    output_root = args.output_root or processing_config.get("output_root", "pro/sessions")
    dataset_root = args.dataset_root or processing_config.get("dataset_root", "data/sessions")
    range_scale_to_meter = float(processing_config.get("range_scale_to_meter", 1000.0))
    track_window_size_s = float(preprocessing_config.get("window_size_s", 8.0))
    track_overlap = float(preprocessing_config.get("overlap", 0.5))
    rd_crop_range_win = int(preprocessing_config.get("rd_crop_range_win", 32))
    rd_crop_doppler_win = int(preprocessing_config.get("rd_crop_doppler_win", 32))
    rd_align_smoothing_window = int(preprocessing_config.get("rd_align_smoothing_window", 5))
    rd_align_refine_range_radius = int(preprocessing_config.get("rd_align_refine_range_radius", 6))
    rd_align_refine_doppler_radius = int(preprocessing_config.get("rd_align_refine_doppler_radius", 6))
    doppler_window = int(processing_config.get("doppler_window", 32))
    doppler_windows_to_test = list(processing_config.get("doppler_windows_to_test", processing_config.get("doppler_windows", [16, 32, 64])))
    use_multires_rd = bool(args.multires_rd or processing_config.get("use_multires_rd", False))
    doppler_fft_size = int(processing_config.get("doppler_fft_size", 64))
    range_window_kind = str(processing_config.get("range_window_kind", "hann"))
    doppler_window_kind = str(processing_config.get("doppler_window_kind", "hann"))
    clutter_suppression_mode = str(processing_config.get("clutter_suppression_mode", "median_subtract"))
    micro_motion_alpha = float(processing_config.get("micro_motion_alpha", 0.35))
    axis_mapping_mode = str(processing_config.get("axis_mapping_mode", "calibrated_preferred"))
    range_resolution_m = float(radar_config.get("range_resolution_m", 8.0))
    doppler_resolution_mps = float(radar_config.get("doppler_resolution_mps", 0.5))
    zero_doppler_idx = int(radar_config.get("zero_doppler_idx", doppler_fft_size // 2))
    rd_align_clip_range_margin = int(preprocessing_config.get("rd_align_clip_range_margin", 10))
    rd_align_clip_doppler_margin = int(preprocessing_config.get("rd_align_clip_doppler_margin", 6))
    rd_align_peak_guard_ratio = float(preprocessing_config.get("rd_align_peak_guard_ratio", 1.2))
    rd_align_max_range_shift = int(preprocessing_config.get("rd_align_max_range_shift", 4))
    rd_align_max_doppler_shift = int(preprocessing_config.get("rd_align_max_doppler_shift", 3))

    summaries = prepare_real_sessions(
        raw_root=raw_root,
        output_root=output_root,
        range_scale_to_meter=range_scale_to_meter,
        track_window_size_s=track_window_size_s,
        track_overlap=track_overlap,
        rd_crop_range_win=rd_crop_range_win,
        rd_crop_doppler_win=rd_crop_doppler_win,
        rd_align_smoothing_window=rd_align_smoothing_window,
        rd_align_refine_range_radius=rd_align_refine_range_radius,
        rd_align_refine_doppler_radius=rd_align_refine_doppler_radius,
        doppler_window=doppler_window,
        doppler_windows_to_test=doppler_windows_to_test,
        use_multires_rd=use_multires_rd,
        doppler_fft_size=doppler_fft_size,
        range_window_kind=range_window_kind,
        doppler_window_kind=doppler_window_kind,
        clutter_suppression_mode=clutter_suppression_mode,
        micro_motion_alpha=micro_motion_alpha,
        axis_mapping_mode=axis_mapping_mode,
        range_resolution_m=range_resolution_m,
        doppler_resolution_mps=doppler_resolution_mps,
        zero_doppler_idx=zero_doppler_idx,
        rd_align_clip_range_margin=rd_align_clip_range_margin,
        rd_align_clip_doppler_margin=rd_align_clip_doppler_margin,
        rd_align_peak_guard_ratio=rd_align_peak_guard_ratio,
        rd_align_max_range_shift=rd_align_max_range_shift,
        rd_align_max_doppler_shift=rd_align_max_doppler_shift,
        session_ids=args.session_ids or None,
    )
    copied_sessions = sync_prepared_sessions(
        processed_root=output_root,
        dataset_root=dataset_root,
        session_ids=args.session_ids or None,
    )

    logger.info("=" * 30)
    logger.info("Pro Preparation Summary")
    logger.info("=" * 30)
    logger.info(f"Prepared sessions: {len(summaries)}")
    logger.info(f"RD pipeline mode: {'toTest_multiresolution' if use_multires_rd else 'default_single_scale'}")
    logger.info(f"Copied sessions to dataset root: {copied_sessions}")
    for summary in summaries:
        logger.info(
            f"- {summary['session_id']} | label={summary['label']} | track_rows={summary['num_track_rows']} | "
            f"track_clips={summary['num_track_clips']} | rd_shape={summary['rd_shape']} | qc={summary['qc_status']}"
        )
    logger.info("=" * 30)


def cmd_process(args, config):
    session_dir = Path(f"data/sessions/{args.session_id}")
    if not session_dir.exists():
        logger.error(f"[!] Session directory does not exist: {session_dir}")
        return

    summary = {
        "session": args.session_id,
        "qc": "Failed",
        "num_track_clips": 0,
        "num_rd_clips": 0,
        "num_matches": 0,
    }

    try:
        logger.info(f"[*] Running QC for session: {args.session_id}")
        checker = DataIntegrityChecker(session_dir)
        qc_result = checker.check_integrity()
        summary["qc"] = qc_result.get("status", "Unknown")

        logger.info("[*] Running track slicing.")
        slicer = DataSlicer(session_dir)
        preprocessing_config = config["preprocessing"]
        track_clips = slicer.slice_tracks(
            window_size_s=preprocessing_config["window_size_s"],
            overlap=preprocessing_config["overlap"],
        )
        summary["num_track_clips"] = len(track_clips)

        if args.rd:
            logger.info("[*] Running RD slicing.")
            slicer.slice_rd_clips(
                track_clips,
                range_win=preprocessing_config["rd_crop_range_win"],
                doppler_win=preprocessing_config["rd_crop_doppler_win"],
                smoothing_window=preprocessing_config.get("rd_align_smoothing_window", 5),
                refine_range_radius=preprocessing_config.get("rd_align_refine_range_radius", 6),
                refine_doppler_radius=preprocessing_config.get("rd_align_refine_doppler_radius", 6),
                clip_range_margin=preprocessing_config.get("rd_align_clip_range_margin", 10),
                clip_doppler_margin=preprocessing_config.get("rd_align_clip_doppler_margin", 6),
                peak_guard_ratio=preprocessing_config.get("rd_align_peak_guard_ratio", 1.2),
                max_range_shift=preprocessing_config.get("rd_align_max_range_shift", 4),
                max_doppler_shift=preprocessing_config.get("rd_align_max_doppler_shift", 3),
            )
            summary["num_rd_clips"] = len(track_clips)

        logger.info("[*] Running truth matching.")
        labeling_config = config["labeling"]
        matcher = TruthMatcher(
            session_dir,
            radar_yaw_deg=labeling_config["radar_yaw_deg"],
        )
        matches = matcher.match_by_time_and_space(
            t_thresh=labeling_config["time_threshold_s"],
            d_thresh=labeling_config["dist_threshold_m"],
            v_thresh=labeling_config["speed_threshold_mps"],
            az_thresh=labeling_config.get("azimuth_threshold_deg", 5.0),
        )
        summary["num_matches"] = len(matches)

        logger.info("=" * 30)
        logger.info("Pipeline Summary")
        logger.info("=" * 30)
        logger.info(f"Session: {summary['session']}")
        logger.info(f"QC: {summary['qc']}")
        logger.info(f"Track clips: {summary['num_track_clips']}")
        if args.rd:
            logger.info(f"RD clips: {summary['num_rd_clips']}")
        logger.info(f"Matched labels: {summary['num_matches']}")
        logger.info("=" * 30)
    except Exception:
        logger.exception("[!] Processing pipeline failed.")


def cmd_train(args, config):
    try:
        from src.training.trainer import run_training
    except ImportError as exc:
        logger.error("[!] Training dependencies are not available.")
        logger.error("[!] Install packages from requirements.txt before running train.")
        logger.error(f"[!] Import error: {exc}")
        return

    training_config = dict(config.get("training", {}))

    if args.data_root:
        training_config["data_root"] = args.data_root
    if args.output_dir:
        training_config["output_dir"] = args.output_dir
    else:
        resolved_epochs = args.epochs if args.epochs is not None else int(training_config.get("epochs", 30))
        resolved_batch_size = args.batch_size if args.batch_size is not None else int(training_config.get("batch_size", 4))
        resolved_device = (args.device or "cpu").split(":", 1)[0].lower()
        output_name = f"{datetime.now().strftime('%Y%m%d%H%M')}-{resolved_epochs}-{resolved_batch_size}-{resolved_device}"
        training_config["output_dir"] = f"artifacts/{output_name}"
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        training_config["num_workers"] = args.num_workers
    if args.include_rd is not None:
        training_config["include_rd"] = args.include_rd

    result = run_training(
        config_values=training_config,
        session_ids=args.session_ids or None,
        device=args.device,
    )

    logger.info("=" * 30)
    logger.info("Training Summary")
    logger.info("=" * 30)
    logger.info(f"Samples: {result['num_samples']}")
    logger.info(f"Classes: {result['labels']}")
    logger.info(f"Session splits: {result['session_splits']}")
    logger.info(f"Best checkpoint: {result['best_checkpoint']}")
    logger.info(f"History file: {result['history_path']}")
    logger.info(f"Manifest file: {result['manifest_path']}")
    if result.get("test_metrics") is not None:
        logger.info(f"Test metrics: {result['test_metrics']}")
    logger.info("=" * 30)


def cmd_evaluate(args, config):
    try:
        from src.training.evaluator import run_evaluation
    except ImportError as exc:
        logger.error("[!] Evaluation dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    training_config = dict(config.get("training", {}))
    resolved_output_dir = args.output_dir
    if not resolved_output_dir:
        resolved_output_dir = str(Path(args.checkpoint).parent)
    result = run_evaluation(
        checkpoint_path=args.checkpoint,
        config_values=training_config,
        session_ids=args.session_ids or None,
        data_root=args.data_root,
        output_dir=resolved_output_dir,
        device=args.device,
    )

    logger.info("=" * 30)
    logger.info("Evaluation Summary")
    logger.info("=" * 30)
    logger.info(f"Sessions: {result['evaluated_sessions']}")
    logger.info(f"Samples: {result['num_samples']}")
    logger.info(f"Loss: {result['loss']:.4f}")
    logger.info(f"Accuracy: {result['accuracy']:.4f}")
    logger.info(f"Macro F1: {result['macro_f1']:.4f}")
    logger.info(f"Report file: {result['report_path']}")
    logger.info("=" * 30)


def cmd_sessions(args, config):
    try:
        from src.training.session_manager import count_session_labels, scan_session_directories
    except ImportError as exc:
        logger.error("[!] Session manager dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    data_root = args.data_root or config.get("training", {}).get("data_root", "data/sessions")
    include_rd = args.include_rd if args.include_rd is not None else config.get("training", {}).get("include_rd", True)
    statuses = scan_session_directories(data_root=data_root, include_rd=include_rd)
    label_distribution = count_session_labels(statuses)

    logger.info("=" * 30)
    logger.info("Session Summary")
    logger.info("=" * 30)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Ready sessions: {[status.session_id for status in statuses if status.ready_for_training]}")
    logger.info(f"Label coverage: {label_distribution}")
    for status in statuses:
        logger.info(
            f"- {status.session_id} | ready={status.ready_for_training} | "
            f"track_clips={status.num_track_clips} | matched_tracks={status.num_matched_tracks} | labels={status.labels}"
        )
    logger.info("=" * 30)


def cmd_report(args, config):
    try:
        from src.reporting.report_builder import build_real_report
    except ImportError as exc:
        logger.error("[!] Reporting dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    artifact_path = Path(args.artifact_dir)
    resolved_real_root = args.real_root or "reports/real"
    real_result = build_real_report(str(artifact_path), report_root=resolved_real_root)

    logger.info("=" * 30)
    logger.info("Report Summary")
    logger.info("=" * 30)
    logger.info(f"Real report: {real_result['report_dir']}")
    logger.info("=" * 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RadarInsight command line interface")
    parser.add_argument("--config", default="config/system_config.yaml", help="Path to YAML config file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    prepare_parser = subparsers.add_parser("prepare-pro", help="Prepare real TR/RD mat sessions under pro/sessions")
    prepare_parser.add_argument("session_ids", nargs="*", help="Optional session IDs; defaults to all paired raw sessions")
    prepare_parser.add_argument("--raw-root", dest="raw_root", help="Root folder containing pro raw TR and RD mats")
    prepare_parser.add_argument("--output-root", dest="output_root", help="Output folder for prepared sessions")
    prepare_parser.add_argument("--dataset-root", dest="dataset_root", help="Final standardized dataset root copied for training")
    prepare_parser.add_argument("--multires-rd", dest="multires_rd", action="store_true", help="Enable the toTest multiresolution DPL-to-RD branch")

    process_parser = subparsers.add_parser("process", help="Run QC, slicing, and matching")
    process_parser.add_argument("session_id", help="Session identifier")
    process_parser.add_argument("--rd", action="store_true", help="Enable RD clip extraction")

    train_parser = subparsers.add_parser("train", help="Train the X3D-M plus TCN/Transformer classifier")
    train_parser.add_argument("session_ids", nargs="*", help="Optional session IDs; defaults to all session folders")
    train_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    train_parser.add_argument("--output-dir", dest="output_dir", help="Directory for checkpoints and history")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", dest="learning_rate", type=float, help="Optimizer learning rate")
    train_parser.add_argument("--num-workers", dest="num_workers", type=int, help="DataLoader worker count")
    train_parser.add_argument("--device", help="Training device, for example cpu or cuda")
    train_parser.add_argument("--include-rd", dest="include_rd", action="store_true", help="Use the RD CNN branch")
    train_parser.add_argument("--no-rd", dest="include_rd", action="store_false", help="Disable the RD CNN branch")
    train_parser.set_defaults(include_rd=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on held-out or specified sessions")
    eval_parser.add_argument("checkpoint", help="Path to a saved checkpoint file")
    eval_parser.add_argument("session_ids", nargs="*", help="Optional explicit session IDs to evaluate")
    eval_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    eval_parser.add_argument("--output-dir", dest="output_dir", help="Directory for evaluation outputs")
    eval_parser.add_argument("--device", help="Evaluation device, for example cpu or cuda")

    sessions_parser = subparsers.add_parser("sessions", help="Inspect available sessions for training")
    sessions_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    sessions_parser.add_argument("--include-rd", dest="include_rd", action="store_true", help="Require RD clip files")
    sessions_parser.add_argument("--no-rd", dest="include_rd", action="store_false", help="Ignore RD clip requirements")
    sessions_parser.set_defaults(include_rd=None)

    report_parser = subparsers.add_parser("report", help="Generate real visualization reports")
    report_parser.add_argument("artifact_dir", help="Artifact directory containing training and evaluation outputs")
    report_parser.add_argument("--real-root", dest="real_root", default="reports/real", help="Output root for real reports; final folder name follows the artifact directory name")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        config = load_config(args.config)
    except Exception as exc:
        logger.error(f"[!] Failed to load config file: {exc}")
        return

    if args.command == "prepare-pro":
        cmd_prepare_pro(args, config)
    elif args.command == "process":
        cmd_process(args, config)
    elif args.command == "train":
        cmd_train(args, config)
    elif args.command == "evaluate":
        cmd_evaluate(args, config)
    elif args.command == "sessions":
        cmd_sessions(args, config)
    elif args.command == "report":
        cmd_report(args, config)


if __name__ == "__main__":
    main()
