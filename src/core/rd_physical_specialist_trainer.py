from __future__ import annotations

import json
import random
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES
from .data.rd_physical_window_dataset import PhysicalRDWindowBatch
from .data.rd_physical_specialist_dataset import PhysicalRDSpecialistWindowDataset, collate_physical_rd_specialist_windows, remap_specialist_label, specialist_label_names
from .device import resolve_cuda_device
from .rd_physical_five_trainer import _masked_ce
from .rd_physical_specialist_artifacts import build_physical_rd_specialist_model, load_physical_rd_specialist_artifacts, specialist_checkpoint_payload
from .rd_physical_five_artifacts import sha256_file
from .rd_physical_specialist_config import PhysicalRDSpecialistConfig
from .rd_physical_specialist_model import PhysicalRDSpecialistClassifier
from .rd_physical_parent_aggregation import mean_parent_probabilities
from .run_paths import prepare_run_output_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _verify_reference_provenance(config: PhysicalRDSpecialistConfig, project_root: Path) -> dict[str, object]:
    """Verify that configured validation guards come from one immutable val report."""
    if not config.reference_metrics:
        return {"verified": True, "source": "no_reference_metrics"}
    if not config.reference_validation_report:
        return {
            "verified": False,
            "source": "legacy_unverified_config",
            "reason": "reference metrics have no report/checkpoint provenance",
        }
    report_path = project_root / config.reference_validation_report
    if not report_path.is_file():
        raise FileNotFoundError(f"reference validation report does not exist: {report_path}")
    report_sha256 = sha256_file(report_path)
    if report_sha256.lower() != config.reference_validation_report_sha256.lower():
        raise ValueError("reference validation report SHA-256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    completion = report.get("evaluation_completion", {})
    if (
        report.get("kind") != config.kind
        or report.get("test_evaluation_executed") is not False
        or completion.get("full_split_evaluated") is not True
        or int(completion.get("excluded_by_batch_limit_count", -1)) != 0
    ):
        raise ValueError("reference metrics must come from a full validation-only specialist report")
    report_metrics = report.get("metrics", {}).get("per_class", {})
    for label, expected_metrics in config.reference_metrics.items():
        actual_metrics = report_metrics.get(label)
        if not isinstance(actual_metrics, dict):
            raise ValueError(f"reference validation report is missing class metrics: {label}")
        for metric, expected in expected_metrics.items():
            actual_value = float(actual_metrics.get(metric, float("nan")))
            if not np.isfinite(actual_value) or abs(actual_value - float(expected)) > 1.0e-9:
                raise ValueError(f"reference metric mismatch for {label}.{metric}")
    checkpoint_value = report.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("reference validation report is missing checkpoint path")
    checkpoint_path = project_root / checkpoint_value
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"reference checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256.lower() != config.reference_checkpoint_sha256.lower():
        raise ValueError("reference checkpoint SHA-256 mismatch")
    route_recall_provenance: dict[str, object] | None = None
    if config.reference_route_recalls is not None:
        predictions_path = project_root / config.reference_predictions_csv
        if not predictions_path.is_file():
            raise FileNotFoundError(f"reference predictions CSV does not exist: {predictions_path}")
        predictions_sha256 = sha256_file(predictions_path)
        if predictions_sha256.lower() != config.reference_predictions_csv_sha256.lower():
            raise ValueError("reference predictions CSV SHA-256 mismatch")
        supports = {label: 0 for label in RD_FIVE_LABELS}
        correct = {label: 0 for label in RD_FIVE_LABELS}
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                label = str(row.get("label_name_true", ""))
                prediction = str(row.get("label_name_pred", ""))
                if label not in supports:
                    raise ValueError(f"reference predictions contain an unknown original label: {label!r}")
                if prediction not in specialist_label_names("parent_gate"):
                    raise ValueError(f"reference predictions contain an unknown parent label: {prediction!r}")
                supports[label] += 1
                expected = specialist_label_names("parent_gate")[remap_specialist_label("parent_gate", label)]
                correct[label] += int(prediction == expected)
        if sum(supports.values()) != int(completion.get("evaluated_window_count", -1)):
            raise ValueError("reference predictions row count does not match validation report")
        if any(support == 0 for support in supports.values()):
            raise ValueError("reference predictions must support every original five-class label")
        actual_route_recalls = {
            label: correct[label] / supports[label] if supports[label] else 0.0
            for label in RD_FIVE_LABELS
        }
        for label, expected in config.reference_route_recalls.items():
            if abs(actual_route_recalls[label] - float(expected)) > 1.0e-9:
                raise ValueError(f"reference route recall mismatch for {label}")
        route_recall_provenance = {
            "predictions_csv": config.reference_predictions_csv,
            "predictions_csv_sha256": predictions_sha256,
            "support": supports,
            "correct": correct,
            "route_recalls": actual_route_recalls,
        }
    return {
        "verified": True,
        "source": "full_validation_report",
        "report": config.reference_validation_report,
        "report_sha256": report_sha256,
        "checkpoint": checkpoint_value,
        "checkpoint_sha256": checkpoint_sha256,
        "route_recall_provenance": route_recall_provenance,
        "test_evaluation_executed": False,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _make_loader(data_root: Path, config: PhysicalRDSpecialistConfig, split: str, shuffle: bool) -> DataLoader:
    dataset = PhysicalRDSpecialistWindowDataset(
        data_root,
        kind=config.kind,
        index_name=config.index_name,
        split_name=split,
        normalization_stats_name=config.normalization_stats_name,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=True, persistent_workers=False, collate_fn=collate_physical_rd_specialist_windows, generator=torch.Generator().manual_seed(config.seed + (1 if shuffle else 2)))


def _evidence_weights(batch: PhysicalRDWindowBatch, config: PhysicalRDSpecialistConfig) -> torch.Tensor:
    """Build loss weights from per-window RD evidence only.

    Parent/sample/content identifiers are intentionally not consulted.  Those
    fields remain metadata for split integrity and parent-level reporting.
    """
    evidence = torch.where(
        batch.is_prefix_windows,
        config.prefix_max_loss_weight * (batch.window_frame_counts.float() / float(config.evidence_target_frames)).clamp(max=1.0),
        torch.ones_like(batch.window_frame_counts, dtype=torch.float32),
    )
    present = batch.quality_features[:, RD_FIVE_QUALITY_NAMES.index("rd_present")].bool()
    return evidence * present.to(evidence.dtype)


def _metrics(logits: torch.Tensor, labels: torch.Tensor, labels_names: tuple[str, ...]) -> dict[str, object]:
    if logits.ndim != 2 or logits.shape[1] != len(labels_names) or labels.shape != (logits.shape[0],):
        raise ValueError("specialist metrics shape contract failed")
    matrix = [[0 for _ in labels_names] for _ in labels_names]
    for truth, pred in zip(labels.tolist(), logits.argmax(-1).tolist()):
        matrix[int(truth)][int(pred)] += 1
    per_class = {}; ps = []; rs = []; fs = []
    for i, name in enumerate(labels_names):
        tp = matrix[i][i]; fp = sum(matrix[row][i] for row in range(len(labels_names)) if row != i); fn = sum(matrix[i][col] for col in range(len(labels_names)) if col != i)
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); f = 2*p*r / max(1e-12, p+r)
        per_class[name] = {"precision": p, "recall": r, "f1": f, "support": sum(matrix[i]), "support_stability": "low" if 0 < sum(matrix[i]) < 30 else ("no_support" if not sum(matrix[i]) else "adequate")}
        ps.append(p); rs.append(r); fs.append(f)
    directional = {
        f"{labels_names[truth]}_to_{labels_names[prediction]}": matrix[truth][prediction]
        for truth in range(len(labels_names))
        for prediction in range(len(labels_names))
        if truth != prediction
    }
    return {"accuracy": float((logits.argmax(-1) == labels).float().mean()) if labels.numel() else 0.0, "macro_precision": float(np.mean(ps)) if ps else 0.0, "macro_recall": float(np.mean(rs)) if rs else 0.0, "macro_f1": float(np.mean(fs)) if fs else 0.0, "balanced_accuracy": float(np.mean(rs)) if rs else 0.0, "sample_count": int(labels.numel()), "per_class": per_class, "confusion_matrix": {"labels": list(labels_names), "matrix": matrix}, "directional_errors": directional}


def _parent_metrics(logits: torch.Tensor, labels: torch.Tensor, parent_ids: list[str], labels_names: tuple[str, ...]) -> dict[str, object]:
    if not parent_ids:
        return {"parent_count": 0, "aggregation": "arithmetic mean of per-window probabilities, then argmax", "metrics": _metrics(logits.new_empty((0, len(labels_names))), labels.new_empty((0,), dtype=torch.long), labels_names)}
    parent_probabilities, parent_labels, parent_count = mean_parent_probabilities(logits, labels, parent_ids)
    return {"parent_count": parent_count, "aggregation": "arithmetic mean of per-window probabilities, then argmax", "metrics": _metrics(parent_probabilities, parent_labels, labels_names)}


def _selection_record(metrics: dict[str, object], config: PhysicalRDSpecialistConfig) -> dict[str, object]:
    """Build a validation-only checkpoint eligibility record.

    References are fixed in the independent config and must come from a
    previously completed validation report.  No test score or threshold is
    consulted.  Every configured local class is checked independently, which
    prevents a macro-F1 gain from hiding a Drone/Bird or Clutter/Unknown
    regression.
    """
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict) or int(metrics.get("sample_count", 0)) == 0:
        return {
            "checks": {},
            "route_recall_checks": {},
            "non_degradation": False,
            "reference_source": "none",
            "macro_f1": 0.0,
            "min_class_f1": 0.0,
            "min_class_recall": 0.0,
            "min_class_precision": 0.0,
            "min_reference_ratio": 0.0,
            "mean_reference_ratio": 0.0,
            "passed_reference_metric_count": 0,
            "reference_metric_count": 0,
            "parent_macro_f1": 0.0,
            "parent_non_degradation": False,
            "selection_policy": config.selection_policy,
            "eligible": False,
        }
    references = config.reference_metrics or {}
    checks: dict[str, bool] = {}
    reference_ratios: dict[str, dict[str, float]] = {}
    flat_ratios: list[float] = []
    passed_reference_metric_count = 0
    for label, reference in references.items():
        if label not in per_class:
            raise ValueError(f"specialist reference label is absent from validation metrics: {label}")
        current = per_class[label]
        reference_ratios[label] = {}
        label_checks: list[bool] = []
        for metric, expected in reference.items():
            current_value = float(current.get(metric, 0.0))
            expected_value = float(expected)
            passed = current_value + float(config.non_degradation_tolerance) >= expected_value
            label_checks.append(passed)
            passed_reference_metric_count += int(passed)
            ratio = 1.0 if expected_value <= 0.0 else current_value / expected_value
            reference_ratios[label][metric] = ratio
            flat_ratios.append(ratio)
        checks[label] = all(label_checks)
    route_recall_checks: dict[str, bool] = {}
    route_recalls = metrics.get("route_recall_by_original_class", {})
    for label, expected in (config.reference_route_recalls or {}).items():
        if label not in route_recalls:
            raise ValueError(f"validation metrics are missing original-class route recall: {label}")
        current_value = float(route_recalls[label]["recall"])
        expected_value = float(expected)
        passed = current_value + float(config.non_degradation_tolerance) >= expected_value
        route_recall_checks[label] = passed
        passed_reference_metric_count += int(passed)
        ratio = 1.0 if expected_value <= 0.0 else current_value / expected_value
        reference_ratios[f"route:{label}"] = {"recall": ratio}
        flat_ratios.append(ratio)
    parent = metrics.get("parent_metrics")
    parent_metrics = parent.get("metrics") if isinstance(parent, dict) else None
    parent_macro = float(parent_metrics.get("macro_f1", 0.0)) if isinstance(parent_metrics, dict) else 0.0
    all_checks = [*checks.values(), *route_recall_checks.values()]
    local_ok = bool(all(all_checks)) if config.enforce_non_degradation else True
    parent_ok = True
    return {
        "checks": checks,
        "route_recall_checks": route_recall_checks,
        "non_degradation": local_ok,
        "reference_source": config.reference_validation_report if references and config.reference_validation_report else ("config.reference_metrics_unverified" if references else "none"),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "min_class_f1": min(float(item.get("f1", 0.0)) for item in per_class.values()),
        "min_class_recall": min(float(item.get("recall", 0.0)) for item in per_class.values()),
        "min_class_precision": min(float(item.get("precision", 0.0)) for item in per_class.values()),
        "reference_metric_ratios": reference_ratios,
        "min_reference_ratio": min(flat_ratios) if flat_ratios else 1.0,
        "mean_reference_ratio": float(np.mean(flat_ratios)) if flat_ratios else 1.0,
        "passed_reference_metric_count": passed_reference_metric_count,
        "reference_metric_count": len(flat_ratios),
        "parent_macro_f1": parent_macro,
        "parent_non_degradation": parent_ok,
        "selection_policy": config.selection_policy,
        "eligible": bool(local_ok and parent_ok),
    }


def _selection_key(selection: dict[str, object]) -> tuple[float, ...]:
    policy = str(selection.get("selection_policy", "eligible_macro_f1"))
    if policy == "eligible_reference_guard":
        # If no epoch clears every guard, keep the diagnostic candidate that
        # is closest to all configured precision/recall/F1 references.  This
        # avoids improving CU recall by silently sacrificing DBB precision or
        # recall.  An ineligible candidate is still never deployment-ready.
        return tuple(
            float(selection[name])
            for name in (
                "eligible",
                "min_reference_ratio",
                "passed_reference_metric_count",
                "mean_reference_ratio",
                "macro_f1",
                "parent_macro_f1",
            )
        )
    if policy == "eligible_weak_class_guard":
        # Eligibility remains the primary guard. If every epoch is
        # ineligible, compare the weakest local class before macro-F1 so a
        # majority-group solution cannot hide CU/rare-class failures.
        return tuple(
            float(selection[name])
            for name in (
                "eligible",
                "min_class_recall",
                "min_class_f1",
                "min_class_precision",
                "macro_f1",
                "parent_macro_f1",
            )
        )
    return tuple(
        float(selection[name])
        for name in (
            "eligible",
            "macro_f1",
            "min_class_f1",
            "min_class_recall",
            "min_class_precision",
            "parent_macro_f1",
        )
    )


def _route_recall_by_original_class(
    logits: torch.Tensor,
    original_label_names: list[str],
    *,
    kind: str,
) -> dict[str, dict[str, float | int]]:
    if kind != "parent_gate":
        return {}
    if logits.shape[0] != len(original_label_names):
        raise ValueError("original labels must align with parent-gate validation logits")
    predictions = logits.argmax(-1).tolist()
    result: dict[str, dict[str, float | int]] = {}
    for label in RD_FIVE_LABELS:
        indices = [index for index, value in enumerate(original_label_names) if value == label]
        expected = remap_specialist_label("parent_gate", label)
        correct = sum(int(predictions[index] == expected) for index in indices)
        support = len(indices)
        result[label] = {
            "support": support,
            "correct_route_count": correct,
            "recall": correct / support if support else 0.0,
        }
    return result


def _run_epoch(model: PhysicalRDSpecialistClassifier, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, config: PhysicalRDSpecialistConfig, phase: str) -> tuple[dict[str, object], float]:
    training = optimizer is not None; model.train(training)
    limit = config.max_train_batches_per_epoch if training else config.max_val_batches
    limit = int(limit) if limit else len(loader)
    names = specialist_label_names(config.kind)
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
    logits_rows: list[torch.Tensor] = []; labels_rows: list[torch.Tensor] = []; parent_ids: list[str] = []; original_label_names: list[str] = []; losses: list[float] = []; norms: list[float] = []
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > limit: break
        args = (batch.features.to(device), batch.time_seconds.to(device), batch.padding_mask.to(device), batch.quality_features.to(device), batch.range_offset_m.to(device), batch.velocity_mps.to(device), batch.raw_amplitude_db.to(device))
        labels = batch.labels.to(device)
        valid = batch.quality_features[:, RD_FIVE_QUALITY_NAMES.index("rd_present")].bool().to(device)
        weights = _evidence_weights(batch, config).to(device)
        with torch.set_grad_enabled(training):
            output = model(*args)
            final = (batch.lengths.to(device) - 1).clamp_min(0)
            rows = torch.arange(output.frame_logits.shape[0], device=device)
            loss = _masked_ce(output.logits, labels, valid, weights, class_weights, config.focal_gamma)
            loss = loss + config.prefix_aux_weight * _masked_ce(output.frame_logits[rows, final], labels, valid, weights, class_weights, config.focal_gamma)
            loss = loss + config.micro_aux_weight * _masked_ce(output.micro_logits, labels, valid, weights, class_weights, config.focal_gamma)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite specialist {phase} loss at batch={batch_index}")
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm): raise RuntimeError(f"non-finite specialist {phase} gradient at batch={batch_index}")
                norms.append(float(norm.detach().cpu())); optimizer.step()
        valid_cpu = valid.detach().cpu(); logits_rows.append(output.logits.detach().cpu()[valid_cpu]); labels_rows.append(batch.labels.detach().cpu()[valid_cpu]); parent_ids.extend([p for p, ok in zip(batch.parent_sample_ids, valid_cpu.tolist()) if ok]); original_label_names.extend([name for name, ok in zip(batch.label_names, valid_cpu.tolist()) if ok]); losses.append(float(loss.detach().cpu()))
        if batch_index == 1 or batch_index % config.log_interval_batches == 0 or batch_index == limit: print(f"[physical-rd-specialist:{config.kind}][{phase}] batch={batch_index}/{limit} loss={losses[-1]:.4f}", flush=True)
    logits = torch.cat(logits_rows) if logits_rows else torch.empty((0, len(names))); labels = torch.cat(labels_rows) if labels_rows else torch.empty((0,), dtype=torch.long)
    metrics = _metrics(logits, labels, names); metrics["parent_metrics"] = _parent_metrics(logits, labels, parent_ids, names); metrics["route_recall_by_original_class"] = _route_recall_by_original_class(logits, original_label_names, kind=config.kind); metrics["gradient_norm"] = {"sample_count": len(norms), "min": min(norms) if norms else None, "mean": float(np.mean(norms)) if norms else None, "max": max(norms) if norms else None, "all_finite": True}
    return metrics, float(np.mean(losses)) if losses else 0.0


def train_physical_rd_specialist(config: PhysicalRDSpecialistConfig) -> dict[str, object]:
    if config.run_test_evaluation: raise ValueError("specialist training forbids test evaluation")
    _set_seed(config.seed); device, device_info = resolve_cuda_device(config.device); project_root = Path.cwd(); data_root = project_root / "data"
    reference_provenance = _verify_reference_provenance(config, project_root)
    artifacts = load_physical_rd_specialist_artifacts(project_root, config); train_loader = _make_loader(data_root, config, "train", True); val_loader = _make_loader(data_root, config, "val", False); model = build_physical_rd_specialist_model(artifacts, config, device); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay); paths = prepare_run_output_dir(config.output_dir)
    data = {"kind": config.kind, "labels": list(specialist_label_names(config.kind)), "train_window_count": len(train_loader.dataset), "validation_window_count": len(val_loader.dataset), "train_label_counts": train_loader.dataset.frame["label_name"].astype(str).value_counts().sort_index().to_dict(), "validation_label_counts": val_loader.dataset.frame["label_name"].astype(str).value_counts().sort_index().to_dict(), "train_parent_count": int(train_loader.dataset.frame["parent_sample_id"].astype(str).nunique()), "validation_parent_count": int(val_loader.dataset.frame["parent_sample_id"].astype(str).nunique()), "test_split_loaded": False, "tr_inputs_loaded": False}
    history = []; candidate = None; candidate_selection = None; candidate_epoch = -1; candidate_checkpoint = None; qualified_checkpoint = None
    _write_json(paths.output_dir / "training_progress.json", {"status": "started", "config": asdict(config), "data": data, "reference_provenance": reference_provenance, "test_evaluation_executed": False})
    for epoch in range(1, config.epochs + 1):
        train_metrics, train_loss = _run_epoch(model, train_loader, optimizer, device, config, "train")
        with torch.no_grad(): val_metrics, val_loss = _run_epoch(model, val_loader, None, device, config, "val")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics, "val": val_metrics})
        selection = _selection_record(val_metrics, config)
        if candidate_selection is None or _selection_key(selection) > _selection_key(candidate_selection):
            candidate = val_metrics; candidate_selection = selection; candidate_epoch = epoch
            checkpoint_stem = "candidate" if config.require_qualified_checkpoint else "best"
            candidate_checkpoint = paths.output_dir / f"{checkpoint_stem}_rd_physical_{config.kind}_specialist.pt"
            payload = specialist_checkpoint_payload(epoch=epoch, config=config, artifacts=artifacts, model=model, validation=val_metrics, selection=selection)
            torch.save(payload, candidate_checkpoint)
            if config.require_qualified_checkpoint and bool(selection["eligible"]):
                qualified_checkpoint = paths.output_dir / f"best_rd_physical_{config.kind}_specialist.pt"
                torch.save(payload, qualified_checkpoint)
            elif not config.require_qualified_checkpoint:
                qualified_checkpoint = candidate_checkpoint
        qualified = bool(candidate_selection and candidate_selection["eligible"])
        deliverable = qualified if config.require_qualified_checkpoint else candidate_selection is not None
        _write_json(paths.output_dir / "training_history.json", history); _write_json(paths.output_dir / "training_progress.json", {"status": "running", "epoch": epoch, "candidate_epoch": candidate_epoch, "candidate_checkpoint": str(candidate_checkpoint) if candidate_checkpoint else "", "candidate_validation": candidate, "candidate_selection": candidate_selection, "qualified": qualified, "best_epoch": candidate_epoch if deliverable else -1, "best_checkpoint": str(qualified_checkpoint) if deliverable and qualified_checkpoint else "", "best_validation": candidate if deliverable else None, "best_selection": candidate_selection if deliverable else None, "reference_provenance": reference_provenance, "test_evaluation_executed": False})
    qualified = bool(candidate_selection and candidate_selection["eligible"])
    deliverable = qualified if config.require_qualified_checkpoint else candidate_selection is not None
    summary = {"scope": f"physical RD-only {config.kind} specialist; no TR input, fusion, reject tuning, or test evaluation", "config": asdict(config), "device": device_info, "data": data, "training_outcome": ("qualified" if qualified else "no_qualified_checkpoint") if config.require_qualified_checkpoint else "legacy_best_candidate", "candidate_checkpoint": str(candidate_checkpoint) if candidate_checkpoint else "", "candidate_epoch": candidate_epoch, "candidate_validation": candidate, "candidate_selection": candidate_selection, "qualified": qualified, "best_checkpoint": str(qualified_checkpoint) if deliverable and qualified_checkpoint else "", "best_epoch": candidate_epoch if deliverable else -1, "best_validation": candidate if deliverable else None, "best_selection": candidate_selection if deliverable else None, "reference_provenance": reference_provenance, "supports_rejection": False, "test_evaluation_executed": False}
    _write_json(paths.output_dir / "training_summary.json", summary); _write_json(paths.output_dir / "training_progress.json", {"status": "completed", "epoch": config.epochs, "training_outcome": summary["training_outcome"], "candidate_epoch": candidate_epoch, "candidate_checkpoint": str(candidate_checkpoint) if candidate_checkpoint else "", "candidate_validation": candidate, "candidate_selection": candidate_selection, "qualified": qualified, "best_epoch": summary["best_epoch"], "best_checkpoint": summary["best_checkpoint"], "best_validation": summary["best_validation"], "best_selection": summary["best_selection"], "reference_provenance": reference_provenance, "test_evaluation_executed": False})
    return summary


__all__ = ["train_physical_rd_specialist"]
