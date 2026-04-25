#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.append(os.path.abspath("."))

from src.labeling.truth_matcher import TruthMatcher
from pro.preprocessing.slicer import DataSlicer
from pro.processing.excel_tsv_builder import prepare_excel_tsv_sessions, sync_tsv_sessions
from pro.processing.tr_tsv_builder import prepare_tr_tsv_sessions, sync_tr_tsv_sessions
from pro.processing.mat_session_builder import prepare_real_sessions, sync_prepared_sessions
from pro.processing.cq08_session_builder import prepare_cq08_sessions, sync_cq08_sessions
from pro.processing.carrada_pseudotrack_builder import build_carrada_pseudo_tracks, sync_carrada_pseudo_track_sessions
from pro.processing.carrada_session_builder import prepare_carrada_sessions, sync_carrada_sessions
from src.qc.integrity_check import DataIntegrityChecker
from src.utils.logger import logger
from src.utils.artifact_cleaner import clean_artifacts


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def load_config(config_path: str = "config/system_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_train_control(control_path: str = "train.json") -> dict:
    path = Path(control_path)
    if not path.exists():
        return {}
    import json
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def merge_train_control(training_config: dict, control: dict) -> dict:
    merged = dict(training_config)
    for key, value in control.items():
        if key in {"track", "rd"}:
            continue
        merged[key] = value
    merged["include_track"] = bool(control.get("track", merged.get("include_track", True)))
    merged["include_rd"] = bool(control.get("rd", merged.get("include_rd", True)))
    merged["track_backbone_type"] = str(control.get("track_backbone_type", merged.get("track_backbone_type", "temporal")))
    merged["label_granularity"] = str(control.get("label_granularity", merged.get("label_granularity", "coarse")))
    return merged


def apply_eval_runtime_overrides(training_config: dict, control: dict) -> tuple[dict, dict]:
    resolved = dict(training_config)
    applied: dict[str, object] = {}
    if not bool(control.get("eval_fusion_override_enabled", False)):
        return resolved, applied

    override_map = {
        "eval_rd_dominant_fusion_bias": "rd_dominant_fusion_bias",
        "eval_rd_unimodal_logit_residual_weight": "rd_unimodal_logit_residual_weight",
        "eval_min_rd_fusion_weight": "min_rd_fusion_weight",
        "eval_ambiguous_rd_fusion_boost": "ambiguous_rd_fusion_boost",
        "eval_surface_rd_fusion_boost": "surface_rd_fusion_boost",
        "eval_use_weighted_fusion_feature": "use_weighted_fusion_feature",
        "eval_rd_fusion_curriculum_enabled": "rd_fusion_curriculum_enabled",
        "eval_rd_fusion_curriculum_start_epoch": "rd_fusion_curriculum_start_epoch",
        "eval_rd_fusion_curriculum_full_epoch": "rd_fusion_curriculum_full_epoch",
        "eval_min_rd_fusion_weight_start": "min_rd_fusion_weight_start",
        "eval_ambiguous_rd_fusion_boost_start": "ambiguous_rd_fusion_boost_start",
        "eval_surface_rd_fusion_boost_start": "surface_rd_fusion_boost_start",
    }
    for source_key, target_key in override_map.items():
        if source_key in control:
            resolved[target_key] = control[source_key]
            applied[target_key] = control[source_key]
    return resolved, applied


def cmd_prepare_tsv(args, config):
    processing_config = dict(config.get("tsv_processing", {}))
    raw_root = args.raw_root or processing_config.get("raw_root", "pro/excel")
    output_root = args.output_root or processing_config.get("output_root", "pro/tsv")
    dataset_root = args.dataset_root or processing_config.get("dataset_root", "data/tsv")

    prepared = prepare_excel_tsv_sessions(
        raw_root=raw_root,
        output_root=output_root,
        session_ids=args.session_ids or None,
    )
    copied_sessions = sync_tsv_sessions(
        processed_root=output_root,
        dataset_root=dataset_root,
        session_ids=args.session_ids or None,
    )

    logger.info("=" * 80)
    logger.info("[*] Excel-to-TSV preparation complete")
    logger.info(f"[*] Prepared TSV sessions: {len(prepared)}")
    logger.info(f"[*] Synced TSV sessions to training root: {len(copied_sessions)}")
    for item in prepared:
        logger.info(
            f"[*] {item.session_id} | tracks={item.num_tracks} | points={item.num_track_points} | "
            f"labels={','.join(item.labels)} | notes={','.join(item.notes)} | duplicate={item.duplicate_status}"
        )




def cmd_prepare_tr_tsv(args, config):
    control = load_train_control(getattr(args, "train_control", "train.json"))
    raw_root = args.raw_root or 'pro/raw/TR'
    output_root = args.output_root or 'pro/tsv_tr'
    dataset_root = args.dataset_root or control.get('data_root') or config.get('tsv_processing', {}).get('dataset_root', 'data/tsv')
    prepared = prepare_tr_tsv_sessions(raw_root=raw_root, output_root=output_root)
    copied = sync_tr_tsv_sessions(processed_root=output_root, dataset_root=dataset_root)
    logger.info('=' * 80)
    logger.info('[*] TR-only supplement preparation complete')
    logger.info(f'[*] Prepared sessions: {len(prepared)}')
    logger.info(f'[*] Synced sessions: {len(copied)}')
    for item in prepared:
        logger.info(f"[*] {item.session_id} | label={item.label_name} | points={item.num_track_points}")

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
    control = load_train_control(getattr(args, "train_control", "train.json"))
    try:
        from src.training.trainer import run_training
    except ImportError as exc:
        logger.error("[!] Training dependencies are not available.")
        logger.error("[!] Install packages from requirements.txt before running train.")
        logger.error(f"[!] Import error: {exc}")
        return

    training_config = merge_train_control(dict(config.get("training", {})), control)

    if args.data_root:
        training_config["data_root"] = args.data_root
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        training_config["num_workers"] = args.num_workers
    if args.output_dir:
        training_config["output_dir"] = args.output_dir
    else:
        resolved_epochs = int(training_config.get("epochs", 30))
        resolved_batch_size = int(
            training_config.get(
                "session_bag_batch_size" if bool(training_config.get("use_session_bag_training", False)) else "batch_size",
                1 if bool(training_config.get("use_session_bag_training", False)) else 4,
            )
        )
        resolved_device = (args.device or str(control.get("device", "cpu"))).split(":", 1)[0].lower()
        output_name = f"{datetime.now().strftime('%Y%m%d%H%M')}-{resolved_epochs}-{resolved_batch_size}-{resolved_device}"
        training_config["output_dir"] = f"artifacts/{output_name}"
    if bool(control.get("clean_bad_artifacts", False)):
        removed = clean_artifacts("artifacts", control.get("artifact_keep", []))
        if removed:
            logger.info(f"[*] Removed low-value artifacts: {removed}")

    resolved_session_ids = list(args.session_ids or [])
    session_id_file = getattr(args, 'session_id_file', None)
    if session_id_file:
        session_file_path = Path(session_id_file)
        if not session_file_path.exists():
            logger.error(f"[!] Session id file not found: {session_id_file}")
            return
        try:
            payload = json.loads(session_file_path.read_text(encoding='utf-8'))
            if not isinstance(payload, list):
                raise ValueError('session id file must contain a JSON array')
            resolved_session_ids.extend(str(item) for item in payload)
        except Exception as exc:
            logger.error(f"[!] Failed to read session id file: {exc}")
            return

    result = run_training(
        config_values=training_config,
        session_ids=resolved_session_ids or None,
        device=args.device or control.get("device"),
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
    control = load_train_control(getattr(args, "train_control", "train.json"))
    try:
        from src.training.evaluator import run_evaluation
    except ImportError as exc:
        logger.error("[!] Evaluation dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    training_config = merge_train_control(dict(config.get("training", {})), control)
    training_config, applied_overrides = apply_eval_runtime_overrides(training_config, control)
    resolved_output_dir = args.output_dir
    if not resolved_output_dir:
        resolved_output_dir = str(Path(args.checkpoint).parent)
    resolved_session_ids = list(args.session_ids or [])
    session_id_file = getattr(args, 'session_id_file', None)
    if session_id_file:
        session_file_path = Path(session_id_file)
        if not session_file_path.exists():
            logger.error(f"[!] Session id file not found: {session_id_file}")
            return
        try:
            payload = json.loads(session_file_path.read_text(encoding='utf-8'))
            if not isinstance(payload, list):
                raise ValueError('session id file must contain a JSON array')
            resolved_session_ids.extend(str(item) for item in payload)
        except Exception as exc:
            logger.error(f"[!] Failed to read session id file: {exc}")
            return

    result = run_evaluation(
        checkpoint_path=args.checkpoint,
        config_values=training_config,
        session_ids=resolved_session_ids or None,
        data_root=args.data_root or control.get("data_root"),
        output_dir=resolved_output_dir,
        device=args.device or control.get("device"),
    )

    logger.info("=" * 30)
    logger.info("Evaluation Summary")
    logger.info("=" * 30)
    logger.info(f"Sessions: {result['evaluated_sessions']}")
    logger.info(f"Samples: {result['num_samples']}")
    logger.info(f"Loss: {result['loss']:.4f}")
    logger.info(f"Accuracy: {result['accuracy']:.4f}")
    logger.info(f"Macro F1: {result['macro_f1']:.4f}")
    if applied_overrides:
        logger.info(f"Eval overrides: {applied_overrides}")
    logger.info(f"Report file: {result['report_path']}")
    logger.info("=" * 30)


def cmd_prepare_cq08(args, config):
    processing_config = dict(config.get("pro_processing", {}))
    preprocessing_config = dict(config.get("preprocessing", {}))
    raw_root = args.raw_root or "pro/23/CQ-08"
    output_root = args.output_root or "pro/sessions_cq08"
    dataset_root = args.dataset_root or processing_config.get("dataset_root", "data/sessions")

    summaries = prepare_cq08_sessions(
        source_root=raw_root,
        output_root=output_root,
        session_ids=args.session_ids or None,
        track_window_size_s=float(preprocessing_config.get("window_size_s", 8.0)),
        track_overlap=float(preprocessing_config.get("overlap", 0.5)),
        rd_crop_range_win=int(preprocessing_config.get("rd_crop_range_win", 16)),
        rd_crop_doppler_win=int(preprocessing_config.get("rd_crop_doppler_win", 16)),
        rd_align_smoothing_window=int(preprocessing_config.get("rd_align_smoothing_window", 5)),
        rd_align_refine_range_radius=int(preprocessing_config.get("rd_align_refine_range_radius", 4)),
        rd_align_refine_doppler_radius=int(preprocessing_config.get("rd_align_refine_doppler_radius", 4)),
        rd_align_clip_range_margin=int(preprocessing_config.get("rd_align_clip_range_margin", 8)),
        rd_align_clip_doppler_margin=int(preprocessing_config.get("rd_align_clip_doppler_margin", 5)),
        rd_align_peak_guard_ratio=float(preprocessing_config.get("rd_align_peak_guard_ratio", 1.2)),
        rd_align_max_range_shift=int(preprocessing_config.get("rd_align_max_range_shift", 4)),
        rd_align_max_doppler_shift=int(preprocessing_config.get("rd_align_max_doppler_shift", 3)),
    )
    copied_sessions = sync_cq08_sessions(
        processed_root=output_root,
        dataset_root=dataset_root,
        session_ids=args.session_ids or None,
    )

    logger.info("=" * 30)
    logger.info("CQ-08 Preparation Summary")
    logger.info("=" * 30)
    logger.info(f"Prepared sessions: {len(summaries)}")
    logger.info(f"Copied sessions to dataset root: {copied_sessions}")
    for summary in summaries[:20]:
        logger.info(
            f"- {summary['session_id']} | label={summary['label']} | track_rows={summary['num_track_rows']} | "
            f"point_rows={summary['num_point_rows']} | track_clips={summary['num_track_clips']} | "
            f"rd_shape={summary['rd_shape']} | qc={summary['qc_status']}"
        )
    logger.info("=" * 30)


def cmd_session_infer(args, config):
    control = load_train_control(getattr(args, "train_control", "train.json"))
    try:
        from src.training.evaluator import run_session_inference
    except ImportError as exc:
        logger.error("[!] Session inference dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    training_config = merge_train_control(dict(config.get("training", {})), control)
    training_config, applied_overrides = apply_eval_runtime_overrides(training_config, control)
    resolved_output_dir = args.output_dir
    if not resolved_output_dir:
        resolved_output_dir = str(Path(args.checkpoint).parent)
    result = run_session_inference(
        checkpoint_path=args.checkpoint,
        config_values=training_config,
        session_ids=args.session_ids or None,
        data_root=args.data_root or control.get("data_root"),
        output_dir=resolved_output_dir,
        device=args.device or control.get("device"),
    )

    session_level = result["session_level"]
    logger.info("=" * 30)
    logger.info("Session Inference Summary")
    logger.info("=" * 30)
    logger.info(f"Sessions: {result['evaluated_sessions']}")
    logger.info(f"Clip accuracy: {result['clip_level']['accuracy']:.4f}")
    logger.info(f"Clip macro F1: {result['clip_level']['macro_f1']:.4f}")
    logger.info(f"Session accuracy: {session_level['accuracy']:.4f}")
    logger.info(f"Session macro F1: {session_level['macro_f1']:.4f}")
    if applied_overrides:
        logger.info(f"Eval overrides: {applied_overrides}")
    logger.info(f"Report file: {result['report_path']}")
    logger.info("=" * 30)


def cmd_session_diagnose(args, config):
    control = load_train_control(getattr(args, "train_control", "train.json"))
    try:
        from src.training.evaluator import run_session_diagnosis
    except ImportError as exc:
        logger.error("[!] Session diagnosis dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    training_config = merge_train_control(dict(config.get("training", {})), control)
    training_config, applied_overrides = apply_eval_runtime_overrides(training_config, control)
    resolved_output_dir = args.output_dir
    if not resolved_output_dir:
        resolved_output_dir = str(Path(args.checkpoint).parent)
    result = run_session_diagnosis(
        checkpoint_path=args.checkpoint,
        config_values=training_config,
        session_ids=args.session_ids or None,
        data_root=args.data_root or control.get("data_root"),
        output_dir=resolved_output_dir,
        device=args.device or control.get("device"),
    )

    logger.info("=" * 30)
    logger.info("Session Diagnosis Summary")
    logger.info("=" * 30)
    logger.info(f"Sessions: {result['evaluated_sessions']}")
    logger.info(f"Clip rows: {result['num_clips']}")
    logger.info(f"Session rows: {result['num_sessions']}")
    logger.info(f"Session CSV: {result['session_csv_path']}")
    logger.info(f"Clip CSV: {result['clip_csv_path']}")
    if applied_overrides:
        logger.info(f"Eval overrides: {applied_overrides}")
    logger.info(f"Report file: {result['report_path']}")
    logger.info("=" * 30)


def cmd_mine_hard_sessions(args, config):
    control = load_train_control(getattr(args, "train_control", "train.json"))
    try:
        from src.training.evaluator import run_hard_session_mining
    except ImportError as exc:
        logger.error("[!] Hard-session mining dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    training_config = merge_train_control(dict(config.get("training", {})), control)
    training_config, applied_overrides = apply_eval_runtime_overrides(training_config, control)
    resolved_output_dir = args.output_dir
    if not resolved_output_dir:
        resolved_output_dir = str(Path(args.checkpoint).parent / "hard_session_mining")
    result = run_hard_session_mining(
        checkpoint_path=args.checkpoint,
        config_values=training_config,
        session_ids=args.session_ids or None,
        data_root=args.data_root or control.get("data_root"),
        output_dir=resolved_output_dir,
        device=args.device or control.get("device"),
        top_k=args.top_k,
        min_error_clips=args.min_error_clips,
    )

    logger.info("=" * 30)
    logger.info("Hard Session Mining Summary")
    logger.info("=" * 30)
    logger.info(f"Top K: {args.top_k}")
    logger.info(f"Selected sessions: {[row['session_id'] for row in result['selected_sessions']]}")
    if applied_overrides:
        logger.info(f"Eval overrides: {applied_overrides}")
    logger.info(f"Report file: {result['report_path']}")
    logger.info("=" * 30)


def cmd_prepare_carrada_pseudotrack(args, config):
    processing_config = dict(config.get("carrada_processing", {}))
    raw_root = args.raw_root or processing_config.get("raw_root", "pro/Carrada")
    output_root = args.output_root or processing_config.get("output_root", "pro/processed/carrada_pseudotrack_sessions")
    dataset_root = args.dataset_root or processing_config.get("dataset_root", "data/sessions")

    prepared = build_carrada_pseudo_tracks(
        carrada_root=raw_root,
        output_root=output_root,
        sequence_ids=args.sequence_ids or None,
    )
    copied_sessions = sync_carrada_pseudo_track_sessions(
        processed_root=output_root,
        dataset_root=dataset_root,
        sequence_ids=args.sequence_ids or None,
    )

    logger.info("=" * 30)
    logger.info("Carrada Pseudo-Track Summary")
    logger.info("=" * 30)
    logger.info(f"Prepared sessions: {len(prepared)}")
    logger.info(f"Copied sessions: {len(copied_sessions)}")
    logger.info(f"Processed root: {output_root}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info("=" * 30)


def cmd_prepare_carrada(args, config):
    processing_config = dict(config.get("carrada_processing", {}))
    raw_root = args.raw_root or processing_config.get("raw_root", "pro/Carrada")
    output_root = args.output_root or processing_config.get("output_root", "pro/processed/carrada_sessions")
    dataset_root = args.dataset_root or processing_config.get("dataset_root", "data/sessions_carrada")
    frame_margin = int(args.frame_margin if args.frame_margin is not None else processing_config.get("frame_margin", 5))
    sub_session_window_specs = args.sub_session_window_specs or processing_config.get("sub_session_window_specs", ['128:64', '64:32', '192:96'])
    clean_outputs = bool(args.clean_outputs or processing_config.get("clean_outputs", False))

    if clean_outputs:
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
    copied_sessions = sync_carrada_sessions(
        processed_root=output_root,
        dataset_root=dataset_root,
        session_ids=args.session_ids or None,
        sequence_ids=args.sequence_ids or None,
    )

    logger.info("=" * 30)
    logger.info("Carrada Session Summary")
    logger.info("=" * 30)
    logger.info(f"Prepared sessions: {len(prepared)}")
    logger.info(f"Copied sessions: {len(copied_sessions)}")
    logger.info(f"Processed root: {output_root}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info("=" * 30)


def cmd_build_curated_manifest(args, config):
    try:
        from src.training.curated_manifest import build_curated_manifest_from_spans_csv
    except ImportError as exc:
        logger.error("[!] Curated manifest builder dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    output_path = args.output or str(Path(args.source_csv).with_suffix(".manifest.json"))
    result_path = build_curated_manifest_from_spans_csv(
        source_csv=args.source_csv,
        output_path=output_path,
        true_label=args.true_label,
        predicted_label=args.predicted_label,
        min_clips=args.min_clips,
    )
    logger.info("=" * 30)
    logger.info("Curated Manifest Summary")
    logger.info("=" * 30)
    logger.info(f"Source CSV: {args.source_csv}")
    logger.info(f"Manifest file: {result_path}")
    logger.info("=" * 30)


def cmd_sessions(args, config):
    control = load_train_control(getattr(args, "train_control", "train.json"))
    try:
        from src.training.session_manager import count_session_labels, scan_session_directories
    except ImportError as exc:
        logger.error("[!] Session manager dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    data_root = args.data_root or control.get("data_root") or config.get("training", {}).get("data_root", "data/tsv")
    include_rd = bool(control.get("rd", config.get("training", {}).get("include_rd", True)))
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

def cmd_calibrate(args, config):
    control = load_train_control(getattr(args, "train_control", "train.json"))
    try:
        from src.training.calibration import run_group_calibration
    except ImportError as exc:
        logger.error("[!] Calibration dependencies are not available.")
        logger.error(f"[!] Import error: {exc}")
        return

    resolved_output_dir = args.output_dir
    if not resolved_output_dir:
        checkpoint_path = Path(args.checkpoint)
        source_name = checkpoint_path.parent.name
        calibrated_name = source_name if source_name.endswith('-calibrated') else f"{source_name}-calibrated"
        resolved_output_dir = str(Path('artifacts') / 'calibrated' / calibrated_name)

    result = run_group_calibration(
        checkpoint_path=args.checkpoint,
        output_dir=resolved_output_dir,
        data_root=args.data_root or control.get("data_root"),
        device=args.device or control.get("device"),
    )

    logger.info("=" * 30)
    logger.info("Calibration Summary")
    logger.info("=" * 30)
    logger.info(f"Output dir: {result['output_dir']}")
    logger.info(f"Bias: {result['bias']}")
    logger.info(f"Val before: {result['base_val_metrics']}")
    logger.info(f"Val after: {result['biased_val_metrics']}")
    logger.info("=" * 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RadarInsight command line interface")
    parser.add_argument("--config", default="config/system_config.yaml", help="Path to YAML config file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    prepare_tsv_parser = subparsers.add_parser("prepare-tsv", help="Prepare Excel track annotations into standardized TSV sessions")
    prepare_tsv_parser.add_argument("session_ids", nargs="*", help="Optional session IDs; defaults to all paired Excel sessions")
    prepare_tsv_parser.add_argument("--raw-root", dest="raw_root", help="Root folder containing paired bz/tck Excel files")
    prepare_tsv_parser.add_argument("--output-root", dest="output_root", help="Output folder for processed TSV sessions")
    prepare_tsv_parser.add_argument("--dataset-root", dest="dataset_root", help="Final TSV dataset root copied for training")

    prepare_tr_parser = subparsers.add_parser("prepare-tr-tsv", help="Prepare TR-only mat sessions into standardized TSV supplements")
    prepare_tr_parser.add_argument("--raw-root", dest="raw_root", help="Root folder containing TR mat files")
    prepare_tr_parser.add_argument("--output-root", dest="output_root", help="Output folder for processed TR TSV sessions")
    prepare_tr_parser.add_argument("--dataset-root", dest="dataset_root", help="Final TSV dataset root copied for training")
    prepare_tr_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")

    prepare_parser = subparsers.add_parser("prepare-pro", help="Prepare real TR/RD mat sessions under pro/sessions")
    prepare_parser.add_argument("session_ids", nargs="*", help="Optional session IDs; defaults to all paired raw sessions")
    prepare_parser.add_argument("--raw-root", dest="raw_root", help="Root folder containing pro raw TR and RD mats")
    prepare_parser.add_argument("--output-root", dest="output_root", help="Output folder for prepared sessions")
    prepare_parser.add_argument("--dataset-root", dest="dataset_root", help="Final standardized dataset root copied for training")
    prepare_parser.add_argument("--multires-rd", dest="multires_rd", action="store_true", help="Enable the toTest multiresolution DPL-to-RD branch")

    prepare_cq08_parser = subparsers.add_parser("prepare-cq08", help="Prepare CQ-08 raw echo, point-track, and track files into standardized sessions")
    prepare_cq08_parser.add_argument("session_ids", nargs="*", help="Optional session IDs; defaults to all discovered CQ-08 sessions")
    prepare_cq08_parser.add_argument("--raw-root", dest="raw_root", help="Root folder containing CQ-08 source directories")
    prepare_cq08_parser.add_argument("--output-root", dest="output_root", help="Output folder for prepared CQ-08 sessions")
    prepare_cq08_parser.add_argument("--dataset-root", dest="dataset_root", help="Final standardized dataset root copied for training")

    process_parser = subparsers.add_parser("process", help="Run QC, slicing, and matching")
    process_parser.add_argument("session_id", help="Session identifier")
    process_parser.add_argument("--rd", action="store_true", help="Enable RD clip extraction")

    train_parser = subparsers.add_parser("train", help="Train the track/RD classifier; modality switches are controlled by TRACK and RD environment variables")
    train_parser.add_argument("session_ids", nargs="*", help="Optional session IDs; defaults to all session folders")
    train_parser.add_argument("--session-id-file", dest="session_id_file", help="JSON file containing an array of session IDs")
    train_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    train_parser.add_argument("--output-dir", dest="output_dir", help="Directory for checkpoints and history")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", dest="learning_rate", type=float, help="Optimizer learning rate")
    train_parser.add_argument("--num-workers", dest="num_workers", type=int, help="DataLoader worker count")
    train_parser.add_argument("--device", help="Training device, for example cpu or cuda")
    train_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on held-out or specified sessions")
    eval_parser.add_argument("checkpoint", help="Path to a saved checkpoint file")
    eval_parser.add_argument("session_ids", nargs="*", help="Optional explicit session IDs to evaluate")
    eval_parser.add_argument("--session-id-file", dest="session_id_file", help="JSON file containing an array of session IDs")
    eval_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    eval_parser.add_argument("--output-dir", dest="output_dir", help="Directory for evaluation outputs")
    eval_parser.add_argument("--device", help="Evaluation device, for example cpu or cuda")
    eval_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")

    session_infer_parser = subparsers.add_parser("session-infer", help="Run clip-level evaluation and session-level aggregated inference")
    session_infer_parser.add_argument("checkpoint", help="Path to a saved checkpoint file")
    session_infer_parser.add_argument("session_ids", nargs="*", help="Optional explicit session IDs to evaluate")
    session_infer_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    session_infer_parser.add_argument("--output-dir", dest="output_dir", help="Directory for inference outputs")
    session_infer_parser.add_argument("--device", help="Inference device, for example cpu or cuda")
    session_infer_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")

    session_diagnose_parser = subparsers.add_parser("session-diagnose", help="Export clip-level and session-level diagnosis files for selected sessions")
    session_diagnose_parser.add_argument("checkpoint", help="Path to a saved checkpoint file")
    session_diagnose_parser.add_argument("session_ids", nargs="*", help="Optional explicit session IDs to diagnose")
    session_diagnose_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    session_diagnose_parser.add_argument("--output-dir", dest="output_dir", help="Directory for diagnosis outputs")
    session_diagnose_parser.add_argument("--device", help="Diagnosis device, for example cpu or cuda")
    session_diagnose_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")

    mine_hard_sessions_parser = subparsers.add_parser("mine-hard-sessions", help="Automatically mine the hardest sessions and export focused diagnosis files")
    mine_hard_sessions_parser.add_argument("checkpoint", help="Path to a saved checkpoint file")
    mine_hard_sessions_parser.add_argument("session_ids", nargs="*", help="Optional explicit session IDs to rank instead of using the default evaluation split")
    mine_hard_sessions_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    mine_hard_sessions_parser.add_argument("--output-dir", dest="output_dir", help="Directory for mining outputs")
    mine_hard_sessions_parser.add_argument("--device", help="Mining device, for example cpu or cuda")
    mine_hard_sessions_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")
    mine_hard_sessions_parser.add_argument("--top-k", dest="top_k", type=int, default=3, help="Number of hardest sessions to export")
    mine_hard_sessions_parser.add_argument("--min-error-clips", dest="min_error_clips", type=int, default=1, help="Minimum error clips required for a session to be selected")

    carrada_parser = subparsers.add_parser("prepare-carrada-pseudotrack", help="Build pseudo-track sessions from Carrada RD-only sequences")
    carrada_parser.add_argument("--raw-root", default=None, help="Carrada root directory")
    carrada_parser.add_argument("--output-root", default=None, help="Processed Carrada pseudo-track session root")
    carrada_parser.add_argument("--dataset-root", default=None, help="Destination dataset root for synced sessions")
    carrada_parser.add_argument("--sequence-ids", nargs="*", default=None, help="Optional Carrada sequence ids to process")

    carrada_full_parser = subparsers.add_parser("prepare-carrada", help="Build standard Carrada sessions with pseudo-track + RD")
    carrada_full_parser.add_argument("--raw-root", default=None, help="Carrada root directory")
    carrada_full_parser.add_argument("--output-root", default=None, help="Processed Carrada session root")
    carrada_full_parser.add_argument("--dataset-root", default=None, help="Destination dataset root for synced sessions")
    carrada_full_parser.add_argument("--sequence-ids", nargs="*", default=None, help="Optional Carrada sequence ids to process")
    carrada_full_parser.add_argument("--session-ids", nargs="*", default=None, help="Optional prepared session ids to sync")
    carrada_full_parser.add_argument("--frame-margin", type=int, default=None, help="Frame margin added on both sides of each instance active span")
    carrada_full_parser.add_argument("--sub-session-window-specs", nargs='*', default=None, help="Carrada mixed window specs, e.g. 128:64 64:32 192:96")
    carrada_full_parser.add_argument("--clean-outputs", action="store_true", help="Delete previous processed and synced Carrada outputs before rebuild")

    curated_manifest_parser = subparsers.add_parser("build-curated-manifest", help="Build a curated optimization manifest from suspicious span CSV files")
    curated_manifest_parser.add_argument("source_csv", help="Path to a suspicious spans CSV file")
    curated_manifest_parser.add_argument("--output", dest="output", help="Output manifest JSON path")
    curated_manifest_parser.add_argument("--true-label", dest="true_label", help="Optional true label filter")
    curated_manifest_parser.add_argument("--predicted-label", dest="predicted_label", help="Optional predicted label filter")
    curated_manifest_parser.add_argument("--min-clips", dest="min_clips", type=int, default=1, help="Minimum number of clips per span")

    sessions_parser = subparsers.add_parser("sessions", help="Inspect available sessions for training")
    sessions_parser.add_argument("--data-root", dest="data_root", help="Root folder containing session directories")
    sessions_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")

    report_parser = subparsers.add_parser("report", help="Generate real visualization reports")
    report_parser.add_argument("artifact_dir", help="Artifact directory containing training and evaluation outputs")
    report_parser.add_argument("--real-root", dest="real_root", default="reports/real", help="Output root for real reports; final folder name follows the artifact directory name")

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate a checkpoint with validation-driven group logit bias")
    calibrate_parser.add_argument("checkpoint", help="Path to a saved checkpoint file")
    calibrate_parser.add_argument("--output-dir", dest="output_dir", help="Directory for calibrated checkpoint and copied artifacts")
    calibrate_parser.add_argument("--data-root", dest="data_root", help="Optional dataset root override")
    calibrate_parser.add_argument("--device", help="Calibration device, for example cpu or cuda")
    calibrate_parser.add_argument("--train-control", dest="train_control", default="train.json", help="Unified training control JSON file")

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

    if args.command == "prepare-tsv":
        cmd_prepare_tsv(args, config)
    elif args.command == "prepare-tr-tsv":
        cmd_prepare_tr_tsv(args, config)
    elif args.command == "prepare-pro":
        cmd_prepare_pro(args, config)
    elif args.command == "prepare-cq08":
        cmd_prepare_cq08(args, config)
    elif args.command == "process":
        cmd_process(args, config)
    elif args.command == "train":
        cmd_train(args, config)
    elif args.command == "evaluate":
        cmd_evaluate(args, config)
    elif args.command == "session-infer":
        cmd_session_infer(args, config)
    elif args.command == "session-diagnose":
        cmd_session_diagnose(args, config)
    elif args.command == "mine-hard-sessions":
        cmd_mine_hard_sessions(args, config)
    elif args.command == "prepare-carrada-pseudotrack":
        cmd_prepare_carrada_pseudotrack(args, config)
    elif args.command == "prepare-carrada":
        cmd_prepare_carrada(args, config)
    elif args.command == "build-curated-manifest":
        cmd_build_curated_manifest(args, config)
    elif args.command == "sessions":
        cmd_sessions(args, config)
    elif args.command == "report":
        cmd_report(args, config)
    elif args.command == "calibrate":
        cmd_calibrate(args, config)


if __name__ == "__main__":
    main()
