from __future__ import annotations

import argparse
import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from .expert1_config import Expert1TrainConfig
from .expert1_data import build_expert1_dataloaders
from .expert1_trainer import _diagnostic_checkpoint_test_recommendation
from .expert1_trainer import _build_class_weights_from_index
from .expert1_trainer import _build_optimizer
from .expert1_trainer import _build_rd_class_weights_from_index
from .expert1_trainer import _checkpoint_eval_runtime
from .expert1_trainer import _checkpoint_joint_logit_delta_scale
from .expert1_trainer import _count_trainable_parameters
from .expert1_trainer import _diagnostic_checkpoint_test_record
from .expert1_trainer import _fit_source_feature_stats_from_index
from .expert1_trainer import _joint_logit_delta_probe
from .expert1_trainer import _load_checkpoint_into_modules
from .expert1_trainer import _load_window_metadata
from .expert1_trainer import _parent_weak_class_metrics
from .expert1_trainer import _parent_window_aggregation_metrics
from .expert1_trainer import _precision_candidate_delta_vs_selected
from .expert1_trainer import _precision_candidate_from_diagnostic_checkpoint_tests
from .expert1_trainer import _precision_candidate_readiness_summary
from .expert1_trainer import _precision_calibration_probe
from .expert1_trainer import _precision_logit_bias_dict
from .expert1_trainer import _rd_joint_bottleneck_summary
from .expert1_trainer import _normalize_precision_logit_bias
from .expert1_trainer import _rel
from .expert1_trainer import _run_epoch
from .expert1_trainer import _serializable_joint_logit_delta_probe
from .expert1_trainer import resolve_device
from .experts import LowAltitudeSuspectExpert
from .physical import PHYSICAL_FEATURE_COLUMNS
from .reporting import write_real_experiment_report
from .trajectory import TrackRepresentationBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Posthoc reselect and export Expert1 precision candidate from an existing "
            "run without rerunning full training."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Existing Expert1 artifact run directory.")
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=None,
        help="Existing evaluation_report.json. Defaults to <run-dir>/evaluation_report.json.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Existing training_summary.json. Defaults to <run-dir>/training_summary.json when present.",
    )
    parser.add_argument(
        "--output-checkpoint-name",
        type=str,
        default="precision_candidate_model_reselected.pt",
        help="Checkpoint filename written inside run-dir.",
    )
    parser.add_argument(
        "--timestamp-label",
        type=str,
        default="",
        help="reports/real/<timestamp-label> output folder. Defaults to current timestamp with posthoc suffix.",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Reload the exported precision checkpoint and evaluate it on the test loader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Override checkpoint config device when --full-validation is used.",
    )
    parser.add_argument(
        "--validation-max-batches",
        type=int,
        default=0,
        help="Max test batches for --full-validation. 0 means the full test loader.",
    )
    parser.add_argument(
        "--validation-num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for --full-validation. Defaults to 0 to avoid Windows multiprocessing issues.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_path(value: object) -> Path:
    path = Path(str(value))
    if path.exists() or path.is_absolute():
        return path
    return Path.cwd() / path


def _write_precision_checkpoint(
    precision_candidate: dict[str, object],
    destination_path: Path,
) -> dict[str, object]:
    if not precision_candidate:
        return {}
    source_path = _checkpoint_path(precision_candidate.get("path", ""))
    if not source_path.exists():
        return {
            "status": "missing_source_checkpoint",
            "source_path": _rel(source_path),
        }
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    recommended_scale = float(
        precision_candidate.get(
            "recommended_joint_logit_delta_scale",
            payload.get("selected_joint_logit_delta_scale", 1.0),
        )
    )
    payload["selected_joint_logit_delta_scale"] = recommended_scale
    payload["selection_reason"] = "precision_candidate_posthoc_reselection"
    payload["precision_candidate_source_checkpoint"] = _rel(source_path)
    payload["precision_candidate_source_checkpoint_name"] = str(precision_candidate.get("checkpoint_name", ""))
    payload["precision_candidate_metrics"] = dict(precision_candidate)
    payload["precision_candidate_recommended_joint_logit_delta_scale"] = recommended_scale
    precision_logit_bias = _normalize_precision_logit_bias(precision_candidate.get("precision_logit_bias", None))
    payload["precision_logit_bias"] = precision_logit_bias
    payload["precision_logit_bias_by_label"] = _precision_logit_bias_dict(precision_logit_bias)
    if "precision_calibration_probe" in precision_candidate:
        payload["precision_calibration_probe"] = precision_candidate.get("precision_calibration_probe", {})
    payload_config = payload.get("config", {})
    if isinstance(payload_config, dict):
        payload_config = dict(payload_config)
        payload_config["joint_logit_delta_scale"] = recommended_scale
        payload_config["selected_joint_logit_delta_scale"] = recommended_scale
        payload_config["precision_logit_bias"] = precision_logit_bias
        payload_config["precision_logit_bias_by_label"] = _precision_logit_bias_dict(precision_logit_bias)
        payload["config"] = payload_config
    checkpoint_probe = payload.get("joint_logit_delta_probe", {})
    if isinstance(checkpoint_probe, dict):
        checkpoint_probe = dict(checkpoint_probe)
        checkpoint_probe["precision_selected_scale"] = recommended_scale
        payload["joint_logit_delta_probe"] = checkpoint_probe
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination_path)
    return {
        "status": "exported",
        "path": _rel(destination_path),
        "source_path": _rel(source_path),
        "source_checkpoint_name": str(precision_candidate.get("checkpoint_name", "")),
        "recommended_joint_logit_delta_scale": recommended_scale,
        "precision_logit_bias": precision_logit_bias,
        "precision_logit_bias_by_label": _precision_logit_bias_dict(precision_logit_bias),
    }


def _metadata_validation(
    precision_candidate: dict[str, object],
    precision_checkpoint_export: dict[str, object],
) -> dict[str, object]:
    if not precision_candidate:
        return {"status": "no_precision_candidate"}
    if precision_checkpoint_export.get("status") != "exported":
        return {
            "status": "not_exported",
            "note": str(precision_checkpoint_export.get("status", "")),
        }
    checkpoint_path = _checkpoint_path(precision_checkpoint_export.get("path", ""))
    if not checkpoint_path.exists():
        return {
            "status": "missing",
            "path": _rel(checkpoint_path),
        }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_scale = float(precision_candidate.get("recommended_joint_logit_delta_scale", 0.0))
    selected_scale = float(checkpoint.get("selected_joint_logit_delta_scale", 0.0))
    return {
        "status": "metadata_only_not_evaluated",
        "path": _rel(checkpoint_path),
        "source_checkpoint_name": str(precision_checkpoint_export.get("source_checkpoint_name", "")),
        "selected_joint_logit_delta_scale": selected_scale,
        "expected_joint_logit_delta_scale": expected_scale,
        "selected_joint_logit_delta_scale_delta": selected_scale - expected_scale,
        "scale_matches": abs(selected_scale - expected_scale) <= 1.0e-6,
        "expected_macro_f1": float(precision_candidate.get("macro_f1", 0.0)),
        "matches_precision_candidate": False,
        "note": "Posthoc export metadata check only; full test-side reload validation was not run.",
    }


def _config_from_checkpoint(checkpoint_path: Path, device_override: str = "") -> Expert1TrainConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload_config = checkpoint.get("config", {})
    if not isinstance(payload_config, dict):
        payload_config = {}
    allowed = {item.name for item in fields(Expert1TrainConfig)}
    values = {key: value for key, value in payload_config.items() if key in allowed}
    if device_override:
        values["device"] = device_override
    return Expert1TrainConfig(**values)


def _build_modules(config: Expert1TrainConfig, device: torch.device) -> tuple[TrackRepresentationBuilder, nn.Module, LowAltitudeSuspectExpert]:
    representation_builder = TrackRepresentationBuilder(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=128,
        dropout=config.dropout,
    ).to(device)
    rd_dim = 16
    rd_proj = nn.Sequential(
        nn.Linear(6, rd_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(rd_dim, rd_dim),
        nn.LayerNorm(rd_dim),
    ).to(device)
    model = LowAltitudeSuspectExpert(
        input_dim=128 + 32,
        rd_input_dim=4,
        rd_patch_input_dim=3,
        rd_context_dim=rd_dim,
        traj_sequence_dim=128,
        use_rd_patch_cnn=config.use_rd_patch_cnn,
        dropout=config.dropout,
        cosface_scale=config.cosface_scale,
        cosface_margin=config.cosface_margin,
        reject_threshold=config.reject_threshold,
        rd_bird_logit_scale=config.rd_bird_logit_scale,
        rd_unknown_logit_scale=config.rd_unknown_logit_scale,
        rd_clutter_logit_scale=config.rd_clutter_logit_scale,
        rd_residual_logit_scale=config.rd_residual_logit_scale,
        rd_unknown_clutter_logit_scale=config.rd_unknown_clutter_logit_scale,
        rd_clutter_morphology_logit_scale=config.rd_clutter_morphology_logit_scale,
        joint_logit_delta_scale=config.joint_logit_delta_scale,
        rd_specialist_residual_scale=config.rd_specialist_residual_scale,
        joint_agreement_prior_scale=config.low_altitude_joint_agreement_prior_scale,
        rd_joint_segment_count=config.rd_joint_segment_count,
        enable_rd_segment_joint_tokens=config.enable_rd_segment_joint_tokens,
    ).to(device)
    return representation_builder, rd_proj, model


def _full_validation(
    precision_candidate: dict[str, object],
    precision_checkpoint_export: dict[str, object],
    *,
    device_override: str = "",
    max_batches: int = 0,
    num_workers: int = 0,
) -> dict[str, object]:
    if not precision_candidate:
        return {"status": "no_precision_candidate"}
    if precision_checkpoint_export.get("status") != "exported":
        return {
            "status": "not_exported",
            "note": str(precision_checkpoint_export.get("status", "")),
        }
    checkpoint_path = _checkpoint_path(precision_checkpoint_export.get("path", ""))
    if not checkpoint_path.exists():
        return {
            "status": "missing",
            "path": _rel(checkpoint_path),
        }
    config = _config_from_checkpoint(checkpoint_path, device_override=device_override)
    config = Expert1TrainConfig(**{
        **config.__dict__,
        "num_workers": int(max(0, num_workers)),
        "eval_num_workers": int(max(0, num_workers)),
        "persistent_workers": bool(config.persistent_workers and int(max(0, num_workers)) > 0),
    })
    device, _device_info = resolve_device(config.device)
    loaders = build_expert1_dataloaders(
        data_root=config.data_root,
        index_name=config.index_name,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        eval_num_workers=config.eval_num_workers,
        rd_sample_boost=config.rd_sample_boost,
        raw_source_boost=config.raw_source_boost,
        excel_source_boost=config.excel_source_boost,
        cq08_source_boost=config.cq08_source_boost,
        source_balance_power=config.source_balance_power,
        source_label_balance_power=config.source_label_balance_power,
        parent_window_balance_power=config.parent_window_balance_power,
        stage2_rd_aware_sampling=config.stage2_rd_aware_sampling,
        stage2_rd_batch_fraction=config.stage2_rd_batch_fraction,
        stage2_rd_shard_local_sampling=config.stage2_rd_shard_local_sampling,
        max_sampler_weight=config.max_sampler_weight,
        unknown_sample_boost=config.unknown_sample_boost,
        clutter_sample_boost=config.clutter_sample_boost,
        bird_sample_boost=config.bird_sample_boost,
        dataset_cache_size=config.dataset_cache_size,
        rd_encoded_cache_size=config.rd_encoded_cache_size,
        enable_window_rd_time_crop=config.enable_window_rd_time_crop,
        rd_window_crop_min_frames=config.rd_window_crop_min_frames,
        use_rd_segment_store=config.use_rd_segment_store,
        rd_segment_manifest_name=config.rd_segment_manifest_name,
        window_segment_map_name=config.window_segment_map_name,
        rd_segment_store_cache_size=config.rd_segment_store_cache_size,
        allow_rd_segment_store_fallback=config.allow_rd_segment_store_fallback,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    representation_builder, rd_proj, model = _build_modules(config, device)
    class_weights = _build_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
    rd_class_weights = _build_rd_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
    source_feature_stats = (
        _fit_source_feature_stats_from_index(loaders.train_loader.dataset.index_df, config.data_root, config.index_name)
        if config.source_normalization
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _build_optimizer(representation_builder, rd_proj, model, config)
    print(
        "[*] Validating precision checkpoint: "
        f"device={device} max_batches={int(max_batches)} params="
        f"{_count_trainable_parameters(representation_builder) + _count_trainable_parameters(rd_proj) + _count_trainable_parameters(model):,}"
    )
    checkpoint_state = _load_checkpoint_into_modules(
        checkpoint_path,
        representation_builder,
        rd_proj,
        model,
        device,
    )
    selected_scale = _checkpoint_joint_logit_delta_scale(checkpoint_state, config)
    model.joint_logit_delta_scale = selected_scale
    checkpoint_epoch = int(checkpoint_state.get("epoch", precision_candidate.get("epoch", -1)))
    stage2_enabled, stage2_scale, stage2_epoch = _checkpoint_eval_runtime(checkpoint_epoch, config)
    checkpoint_metrics = _run_epoch(
        representation_builder,
        rd_proj,
        model,
        loaders.test_loader,
        criterion,
        optimizer,
        device,
        False,
        config,
        stage2_enabled,
        stage2_scale,
        stage2_epoch,
        rd_class_weights,
        source_feature_stats,
        max_batches=int(max_batches),
        compute_aux_loss=config.compute_eval_aux_loss,
    )
    checkpoint_probe = _joint_logit_delta_probe(
        checkpoint_metrics,
        config,
        stage2_enabled=stage2_enabled,
        selected_scale_override=selected_scale,
    )
    checkpoint_selected_probe_metrics = checkpoint_probe.get("selected_metrics")
    if isinstance(checkpoint_selected_probe_metrics, dict):
        checkpoint_metrics.update(checkpoint_selected_probe_metrics)
    checkpoint_metrics["selected_joint_logit_delta_scale"] = float(selected_scale)
    checkpoint_metrics["joint_logit_delta_probe"] = _serializable_joint_logit_delta_probe(checkpoint_probe)
    window_metadata = _load_window_metadata(config.data_root, config.index_name)
    checkpoint_parent_aggregation = _parent_window_aggregation_metrics(checkpoint_metrics, window_metadata)
    checkpoint_parent_mean_logit = checkpoint_parent_aggregation.get("mean_logit", {})
    if isinstance(checkpoint_parent_mean_logit, dict):
        checkpoint_metrics["parent_mean_logit_macro_f1"] = float(checkpoint_parent_mean_logit.get("macro_f1", 0.0))
    checkpoint_metrics.update(_parent_weak_class_metrics(checkpoint_parent_aggregation))
    record = _diagnostic_checkpoint_test_record(
        "precision_checkpoint_export",
        checkpoint_path,
        checkpoint_state,
        checkpoint_metrics,
        is_selected_checkpoint=False,
    )
    expected_scale = float(
        precision_checkpoint_export.get(
            "recommended_joint_logit_delta_scale",
            precision_candidate.get("recommended_joint_logit_delta_scale", 0.0),
        )
    )
    tolerance = 5.0e-5 if int(max_batches) <= 0 else 1.0e-4
    record["source_checkpoint_name"] = str(precision_checkpoint_export.get("source_checkpoint_name", ""))
    record["expected_joint_logit_delta_scale"] = expected_scale
    record["selected_joint_logit_delta_scale_delta"] = float(record.get("selected_joint_logit_delta_scale", 0.0)) - expected_scale
    record["scale_matches"] = abs(float(record["selected_joint_logit_delta_scale_delta"])) <= 1.0e-6
    record["metric_tolerance"] = tolerance
    record["validation_max_batches"] = int(max_batches)
    for key in [
        "macro_f1",
        "final_unknown_recall",
        "final_clutter_recall",
        "final_net_gain_rate_on_rd",
        "final_harm_rate_on_rd",
        "parent_mean_logit_macro_f1",
    ]:
        expected_value = float(precision_candidate.get(key, 0.0))
        actual_value = float(record.get(key, 0.0))
        record[f"expected_{key}"] = expected_value
        record[f"{key}_delta"] = actual_value - expected_value
    record["macro_f1_matches"] = abs(float(record.get("macro_f1_delta", 0.0))) <= tolerance
    record["matches_precision_candidate"] = bool(
        int(max_batches) <= 0
        and record["scale_matches"]
        and record["macro_f1_matches"]
    )
    record["status"] = "ok" if record["matches_precision_candidate"] else ("partial" if int(max_batches) > 0 else "mismatch")
    record["rd_joint_bottleneck_summary"] = _rd_joint_bottleneck_summary(checkpoint_metrics, config)
    return record


def _replace_precision_fields(
    evaluation_report: dict[str, object],
    summary: dict[str, object],
    precision_candidate: dict[str, object],
    precision_readiness: dict[str, object],
    precision_delta: dict[str, object],
    precision_checkpoint_export: dict[str, object],
    precision_checkpoint_validation: dict[str, object],
) -> dict[str, object]:
    checkpoint_selection = evaluation_report.setdefault("checkpoint_selection", {})
    if not isinstance(checkpoint_selection, dict):
        raise TypeError("evaluation_report.checkpoint_selection must be a dict")
    precision_checkpoint = str(precision_checkpoint_export.get("path", precision_candidate.get("path", "")))
    precision_checkpoint_name = str(precision_candidate.get("checkpoint_name", ""))
    checkpoint_selection["precision_checkpoint"] = precision_checkpoint
    checkpoint_selection["precision_checkpoint_name"] = precision_checkpoint_name
    checkpoint_selection["precision_checkpoint_metrics"] = precision_candidate
    checkpoint_selection["precision_checkpoint_export"] = precision_checkpoint_export
    checkpoint_selection["precision_checkpoint_validation"] = precision_checkpoint_validation
    checkpoint_selection["precision_candidate_checkpoint"] = precision_candidate.get("path", "")
    checkpoint_selection["precision_candidate_checkpoint_name"] = precision_checkpoint_name
    checkpoint_selection["precision_candidate_metrics"] = precision_candidate
    checkpoint_selection["precision_candidate_delta_vs_selected"] = precision_delta
    rd_joint_summary = precision_checkpoint_validation.get("rd_joint_bottleneck_summary", {})
    if not isinstance(rd_joint_summary, dict) or not rd_joint_summary:
        rd_joint_summary = precision_candidate.get("rd_joint_bottleneck_summary", {})
    if (
        isinstance(rd_joint_summary, dict)
        and bool(rd_joint_summary.get("full_training_rerun_recommended_now", False))
    ):
        precision_readiness = {
            **precision_readiness,
            "full_training_rerun_recommended_now": True,
            "next_action": str(rd_joint_summary.get("next_action", precision_readiness.get("next_action", ""))),
            "posthoc_validation_overrides_readiness": True,
        }
    checkpoint_selection["precision_candidate_readiness"] = precision_readiness
    if isinstance(rd_joint_summary, dict) and rd_joint_summary:
        checkpoint_selection["rd_joint_bottleneck_summary"] = rd_joint_summary

    diagnostic_tests = evaluation_report.setdefault("diagnostic_checkpoint_tests", {})
    if isinstance(diagnostic_tests, dict):
        diagnostic_tests["precision_candidate"] = precision_candidate
        diagnostic_tests["precision_candidate_readiness"] = precision_readiness
        diagnostic_tests["precision_candidate_delta_vs_selected"] = precision_delta
        diagnostic_tests["precision_checkpoint_validation"] = precision_checkpoint_validation
        if isinstance(rd_joint_summary, dict) and rd_joint_summary:
            diagnostic_tests["rd_joint_bottleneck_summary"] = rd_joint_summary
        results = diagnostic_tests.get("results", [])
        if isinstance(results, list):
            diagnostic_tests["diagnostic_test_recommendation"] = _diagnostic_checkpoint_test_recommendation(results)

    evaluation_report["precision_checkpoint"] = precision_checkpoint
    evaluation_report["precision_checkpoint_name"] = precision_checkpoint_name
    evaluation_report["precision_checkpoint_export"] = precision_checkpoint_export
    evaluation_report["precision_checkpoint_validation"] = precision_checkpoint_validation
    evaluation_report["precision_candidate"] = precision_candidate
    evaluation_report["precision_candidate_readiness"] = precision_readiness
    evaluation_report["precision_candidate_delta_vs_selected"] = precision_delta
    if isinstance(rd_joint_summary, dict) and rd_joint_summary:
        evaluation_report["rd_joint_bottleneck_summary"] = rd_joint_summary
    evaluation_report["posthoc_precision_reselection"] = {
        "status": "applied",
        "source": "diagnostic_checkpoint_tests.results",
        "precision_checkpoint": precision_checkpoint,
        "precision_checkpoint_name": precision_checkpoint_name,
    }

    summary["precision_checkpoint"] = precision_checkpoint
    summary["precision_checkpoint_name"] = precision_checkpoint_name
    summary["precision_checkpoint_metrics"] = precision_candidate
    summary["precision_checkpoint_export"] = precision_checkpoint_export
    summary["precision_checkpoint_validation"] = precision_checkpoint_validation
    summary["precision_candidate"] = precision_candidate
    summary["precision_candidate_checkpoint"] = precision_candidate.get("path", "")
    summary["precision_candidate_checkpoint_name"] = precision_checkpoint_name
    summary["precision_candidate_metrics"] = precision_candidate
    summary["precision_candidate_readiness"] = precision_readiness
    summary["precision_candidate_delta_vs_selected"] = precision_delta
    if isinstance(rd_joint_summary, dict) and rd_joint_summary:
        summary["rd_joint_bottleneck_summary"] = rd_joint_summary
    return precision_readiness


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    evaluation_report_path = args.evaluation_report or run_dir / "evaluation_report.json"
    summary_path = args.summary or run_dir / "training_summary.json"
    if not evaluation_report_path.exists():
        raise FileNotFoundError(f"Missing evaluation report: {evaluation_report_path}")

    evaluation_report = _load_json(evaluation_report_path)
    summary = _load_json(summary_path) if summary_path.exists() else {}
    diagnostic_tests = evaluation_report.get("diagnostic_checkpoint_tests", {})
    if not isinstance(diagnostic_tests, dict):
        raise ValueError("evaluation_report.diagnostic_checkpoint_tests is missing or invalid")
    records = diagnostic_tests.get("results", [])
    if not isinstance(records, list) or not records:
        raise ValueError("diagnostic checkpoint test results are missing")

    precision_candidate = _precision_candidate_from_diagnostic_checkpoint_tests(records)
    if args.full_validation:
        source_checkpoint_path = _checkpoint_path(precision_candidate.get("path", ""))
        config = _config_from_checkpoint(source_checkpoint_path, device_override=str(args.device or ""))
        config = Expert1TrainConfig(**{
            **config.__dict__,
            "num_workers": int(max(0, args.validation_num_workers)),
            "eval_num_workers": int(max(0, args.validation_num_workers)),
            "persistent_workers": bool(config.persistent_workers and int(max(0, args.validation_num_workers)) > 0),
        })
        device, _device_info = resolve_device(config.device)
        loaders = build_expert1_dataloaders(
            data_root=config.data_root,
            index_name=config.index_name,
            batch_size=config.batch_size,
            seed=config.seed,
            num_workers=config.num_workers,
            eval_num_workers=config.eval_num_workers,
            rd_sample_boost=config.rd_sample_boost,
            raw_source_boost=config.raw_source_boost,
            excel_source_boost=config.excel_source_boost,
            cq08_source_boost=config.cq08_source_boost,
            source_balance_power=config.source_balance_power,
            source_label_balance_power=config.source_label_balance_power,
            parent_window_balance_power=config.parent_window_balance_power,
            stage2_rd_aware_sampling=config.stage2_rd_aware_sampling,
            stage2_rd_batch_fraction=config.stage2_rd_batch_fraction,
            stage2_rd_shard_local_sampling=config.stage2_rd_shard_local_sampling,
            max_sampler_weight=config.max_sampler_weight,
            unknown_sample_boost=config.unknown_sample_boost,
            clutter_sample_boost=config.clutter_sample_boost,
            bird_sample_boost=config.bird_sample_boost,
            dataset_cache_size=config.dataset_cache_size,
            rd_encoded_cache_size=config.rd_encoded_cache_size,
            enable_window_rd_time_crop=config.enable_window_rd_time_crop,
            rd_window_crop_min_frames=config.rd_window_crop_min_frames,
            use_rd_segment_store=config.use_rd_segment_store,
            rd_segment_manifest_name=config.rd_segment_manifest_name,
            window_segment_map_name=config.window_segment_map_name,
            rd_segment_store_cache_size=config.rd_segment_store_cache_size,
            allow_rd_segment_store_fallback=config.allow_rd_segment_store_fallback,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
        )
        representation_builder, rd_proj, model = _build_modules(config, device)
        class_weights = _build_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
        rd_class_weights = _build_rd_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
        source_feature_stats = (
            _fit_source_feature_stats_from_index(loaders.train_loader.dataset.index_df, config.data_root, config.index_name)
            if config.source_normalization
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = _build_optimizer(representation_builder, rd_proj, model, config)
        checkpoint_state = _load_checkpoint_into_modules(
            source_checkpoint_path,
            representation_builder,
            rd_proj,
            model,
            device,
        )
        selected_scale = float(precision_candidate.get("recommended_joint_logit_delta_scale", _checkpoint_joint_logit_delta_scale(checkpoint_state, config)))
        model.joint_logit_delta_scale = selected_scale
        checkpoint_epoch = int(checkpoint_state.get("epoch", precision_candidate.get("epoch", -1)))
        stage2_enabled, stage2_scale, stage2_epoch = _checkpoint_eval_runtime(checkpoint_epoch, config)
        checkpoint_metrics = _run_epoch(
            representation_builder,
            rd_proj,
            model,
            loaders.test_loader,
            criterion,
            optimizer,
            device,
            False,
            config,
            stage2_enabled,
            stage2_scale,
            stage2_epoch,
            rd_class_weights,
            source_feature_stats,
            max_batches=int(args.validation_max_batches),
            compute_aux_loss=config.compute_eval_aux_loss,
        )
        checkpoint_probe = _joint_logit_delta_probe(
            checkpoint_metrics,
            config,
            stage2_enabled=stage2_enabled,
            selected_scale_override=selected_scale,
        )
        checkpoint_selected_probe_metrics = checkpoint_probe.get("selected_metrics")
        if isinstance(checkpoint_selected_probe_metrics, dict):
            checkpoint_metrics.update(checkpoint_selected_probe_metrics)
        checkpoint_metrics["selected_joint_logit_delta_scale"] = selected_scale
        checkpoint_metrics["joint_logit_delta_probe"] = _serializable_joint_logit_delta_probe(checkpoint_probe)
        window_metadata = _load_window_metadata(config.data_root, config.index_name)
        checkpoint_parent_aggregation = _parent_window_aggregation_metrics(checkpoint_metrics, window_metadata)
        checkpoint_parent_mean_logit = checkpoint_parent_aggregation.get("mean_logit", {})
        if isinstance(checkpoint_parent_mean_logit, dict):
            checkpoint_metrics["parent_mean_logit_macro_f1"] = float(checkpoint_parent_mean_logit.get("macro_f1", 0.0))
        checkpoint_metrics.update(_parent_weak_class_metrics(checkpoint_parent_aggregation))
        checkpoint_metrics["rd_joint_bottleneck_summary"] = _rd_joint_bottleneck_summary(checkpoint_metrics, config)
        precision_calibration_probe = _precision_calibration_probe(
            checkpoint_metrics,
            precision_candidate,
            config,
            stage2_enabled=stage2_enabled,
            window_metadata=window_metadata,
        )
        precision_candidate["precision_calibration_probe"] = precision_calibration_probe
        precision_candidate["precision_logit_bias"] = _normalize_precision_logit_bias(None)
        precision_candidate["precision_logit_bias_by_label"] = {}
        recommended_calibration = precision_calibration_probe.get("recommended", {})
        if isinstance(recommended_calibration, dict) and precision_calibration_probe.get("ready", False):
            precision_candidate.update(
                {
                    "score": float(recommended_calibration.get("score", precision_candidate.get("score", 0.0))),
                    "macro_f1": float(recommended_calibration.get("macro_f1", precision_candidate.get("macro_f1", 0.0))),
                    "traj_macro_f1": float(recommended_calibration.get("traj_macro_f1", precision_candidate.get("traj_macro_f1", 0.0))),
                    "joint_minus_traj_macro_f1_on_rd": float(
                        recommended_calibration.get(
                            "joint_minus_traj_macro_f1_on_rd",
                            precision_candidate.get("joint_minus_traj_macro_f1_on_rd", 0.0),
                        )
                    ),
                    "rd_macro_f1_on_rd": float(recommended_calibration.get("rd_macro_f1_on_rd", precision_candidate.get("rd_macro_f1_on_rd", 0.0))),
                    "final_net_gain_rate_on_rd": float(
                        recommended_calibration.get("final_net_gain_rate_on_rd", precision_candidate.get("final_net_gain_rate_on_rd", 0.0))
                    ),
                    "final_harm_rate_on_rd": float(
                        recommended_calibration.get("final_harm_rate_on_rd", precision_candidate.get("final_harm_rate_on_rd", 0.0))
                    ),
                    "final_unknown_recall": float(recommended_calibration.get("final_unknown_recall", precision_candidate.get("final_unknown_recall", 0.0))),
                    "final_clutter_recall": float(recommended_calibration.get("final_clutter_recall", precision_candidate.get("final_clutter_recall", 0.0))),
                    "final_unknown_clutter_recall_mean": float(
                        recommended_calibration.get(
                            "final_unknown_clutter_recall_mean",
                            precision_candidate.get("final_unknown_clutter_recall_mean", 0.0),
                        )
                    ),
                    "source_balanced_supported_macro_f1": float(
                        recommended_calibration.get(
                            "source_balanced_supported_macro_f1",
                            precision_candidate.get("source_balanced_supported_macro_f1", 0.0),
                        )
                    ),
                    "parent_mean_logit_macro_f1": float(
                        recommended_calibration.get("parent_mean_logit_macro_f1", precision_candidate.get("parent_mean_logit_macro_f1", 0.0))
                    ),
                    "precision_threshold_failures": list(recommended_calibration.get("precision_threshold_failures", [])),
                    "precision_threshold_gaps": dict(recommended_calibration.get("precision_threshold_gaps", {})),
                    "precision_logit_bias": _normalize_precision_logit_bias(recommended_calibration.get("precision_logit_bias", None)),
                    "precision_logit_bias_by_label": _precision_logit_bias_dict(recommended_calibration.get("precision_logit_bias", None)),
                    "recommendation_basis": "diagnostic_test_precision_candidate_calibrated",
                }
            )
            if isinstance(recommended_calibration.get("rd_joint_bottleneck_summary"), dict):
                precision_candidate["rd_joint_bottleneck_summary"] = recommended_calibration.get("rd_joint_bottleneck_summary", {})
            precision_candidate.update(dict(precision_candidate.get("precision_threshold_gaps", {})))
    precision_readiness = _precision_candidate_readiness_summary(records, precision_candidate)
    precision_delta = _precision_candidate_delta_vs_selected(precision_candidate, records)
    destination_path = run_dir / args.output_checkpoint_name
    precision_checkpoint_export = _write_precision_checkpoint(precision_candidate, destination_path)
    if args.full_validation:
        precision_checkpoint_validation = _full_validation(
            precision_candidate,
            precision_checkpoint_export,
            device_override=str(args.device or ""),
            max_batches=int(args.validation_max_batches),
            num_workers=int(args.validation_num_workers),
        )
    else:
        precision_checkpoint_validation = _metadata_validation(precision_candidate, precision_checkpoint_export)
    precision_readiness = _replace_precision_fields(
        evaluation_report,
        summary,
        precision_candidate,
        precision_readiness,
        precision_delta,
        precision_checkpoint_export,
        precision_checkpoint_validation,
    )

    timestamp_label = args.timestamp_label or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-expert1-posthoc-precision"
    report_paths = write_real_experiment_report(
        output_dir=run_dir,
        evaluation_report=evaluation_report,
        summary=summary,
        timestamp_label=timestamp_label,
    )
    print(
        json.dumps(
            {
                "precision_checkpoint": precision_checkpoint_export.get("path", ""),
                "precision_checkpoint_name": precision_candidate.get("checkpoint_name", ""),
                "precision_checkpoint_validation_status": precision_checkpoint_validation.get("status", ""),
                "precision_candidate_macro_f1": precision_candidate.get("macro_f1", 0.0),
                "precision_candidate_unknown_recall": precision_candidate.get("final_unknown_recall", 0.0),
                "precision_candidate_clutter_recall": precision_candidate.get("final_clutter_recall", 0.0),
                "precision_candidate_threshold_failures": precision_candidate.get("precision_threshold_failures", []),
                "precision_existing_scale_ready": precision_readiness.get("existing_scale_ready", False),
                "precision_calibration_ready": precision_readiness.get("calibration_ready", False),
                "precision_logit_bias_by_label": precision_candidate.get("precision_logit_bias_by_label", {}),
                "precision_next_action": precision_readiness.get("next_action", ""),
                "rd_joint_primary_bottleneck": (
                    precision_checkpoint_validation.get("rd_joint_bottleneck_summary", {})
                    if isinstance(precision_checkpoint_validation.get("rd_joint_bottleneck_summary", {}), dict)
                    else (
                        precision_candidate.get("rd_joint_bottleneck_summary", {})
                        if isinstance(precision_candidate.get("rd_joint_bottleneck_summary", {}), dict)
                        else {}
                    )
                ).get("primary_bottleneck", ""),
                "real_report_markdown": _rel(report_paths.markdown_path),
                "real_report_json": _rel(report_paths.json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
