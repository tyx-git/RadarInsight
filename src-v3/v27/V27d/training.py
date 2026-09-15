"""Independent V27d knownness and quality head training."""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from radar_ln3.runtime import canonical_sha256, relative_project_path, write_torch_if_absent_or_identical

from .features import load_feature_cache
from .heads import V27dHeads, build_heads
from .logging_utils import close_logger, configure_logger
from .metrics import binary_auroc
from .protocol import V27DContext, V27D_KNOWN_LABELS, artifact_path, write_sealed_json


HEAD_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-independent-head-checkpoint-v1"


def _tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d head state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def _component_keys(model: V27dHeads, component: str) -> tuple[str, ...]:
    if component == "knownness":
        prefixes = ("trajectory_encoder.", "knownness_projection.", "knownness_head.")
    elif component == "quality":
        prefixes = ("quality_head.",)
    else:
        raise ValueError(f"V27d unknown head component: {component}")
    keys = tuple(name for name in model.state_dict() if name.startswith(prefixes))
    if not keys:
        raise ValueError(f"V27d component has no parameters: {component}")
    return keys


def _component_state(model: V27dHeads, component: str) -> dict[str, torch.Tensor]:
    keys = set(_component_keys(model, component))
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in keys}


def _load_component_state(model: V27dHeads, component: str, state: Mapping[str, Any]) -> None:
    expected = set(_component_keys(model, component))
    if set(state) != expected:
        raise ValueError(f"V27d {component} checkpoint keys differ")
    current = model.state_dict()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(current[name].shape) or value.dtype != current[name].dtype:
            raise ValueError(f"V27d {component} checkpoint tensor differs: {name}")
    merged = model.state_dict()
    for name, value in state.items():
        merged[name] = value
    model.load_state_dict(merged, strict=True)


def _device(context: V27DContext) -> torch.device:
    if context.config.training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d configuration requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _batch_slices(count: int, batch_size: int, *, generator: torch.Generator) -> Iterable[torch.Tensor]:
    order = torch.randperm(count, generator=generator)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def _head_inputs(tensors: Mapping[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "event_facts_sequence": tensors["event_facts_sequence"][indices].to(device),
        "raw_scale_features": tensors["raw_scale_features"][indices].to(device),
        "stable8_causal_features": tensors["stable8_causal_features"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }


def _positive_weight(target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = target.masked_select(mask)
    positive = int(selected.sum().item())
    negative = int(selected.numel() - positive)
    if positive <= 0 or negative <= 0:
        raise ValueError(f"V27d binary target lacks support: positive={positive}, negative={negative}")
    return max(1.0e-3, negative / positive)


def _parent_balanced_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, positive_weight: float) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(positive_weight), reduction="none")
    count = mask.sum(dim=1)
    eligible = count > 0
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    per_parent = (values * mask.to(dtype=values.dtype)).sum(dim=1) / count.clamp_min(1)
    return per_parent[eligible].mean()


def _candidate_thresholds(values: torch.Tensor) -> list[float]:
    finite = sorted({float(value) for value in values.detach().cpu().tolist() if math.isfinite(float(value))})
    candidates = [float("-inf"), *finite]
    if finite:
        candidates.append(math.nextafter(finite[-1], math.inf))
    return sorted(set(candidates))


def _knownness_selection(
    scores: torch.Tensor,
    payload: Mapping[str, Any],
    *,
    task: str,
    context: V27DContext,
) -> dict[str, Any]:
    """Select a development checkpoint at a per-class known-recall-safe point.

    This is only a checkpoint-selection diagnostic.  Product thresholds are
    still fitted later on the independent calibration pool.
    """

    tensors = payload["tensors"]
    labels = [str(value) for value in payload["label_names"]]
    if scores.ndim != 1 or scores.shape[0] != len(labels) or not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("V27d development knownness scores do not align")
    eligible = tensors["knownness_population"].bool() & tensors["quality_target_final"].bool()
    target = tensors["knownness_target"].bool()
    positive = eligible & target
    negative = eligible & ~target
    class_names = (
        ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
        if task == "known_gate"
        else V27D_KNOWN_LABELS
    )
    class_masks = {
        name: positive & torch.tensor([label == name for label in labels], dtype=torch.bool)
        for name in class_names
    }
    if not bool(positive.any()):
        raise ValueError(f"V27d {task} development selection lacks positive knownness support")
    if not bool(negative.any()):
        raise ValueError(f"V27d {task} development selection lacks negative knownness support")
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in _candidate_thresholds(scores[positive]):
        accepted = scores.ge(threshold)
        recalls = {
            name: float(accepted[mask].float().mean().item()) if bool(mask.any()) else 0.0
            for name, mask in class_masks.items()
        }
        safe = all(value >= context.config.calibration.known_target_recall_floor for name, value in recalls.items() if name != "ClutterTarget")
        if task == "known_gate":
            safe = safe and recalls["ClutterTarget"] >= context.config.calibration.clutter_recall_floor
        unknown_rejection = float((~accepted[negative]).float().mean().item()) if bool(negative.any()) else 0.0
        if safe:
            ranking = (unknown_rejection, min(recalls.values()), float(threshold))
            selection = {
                "criterion": "development_unknown_rejection_at_per_class_known_recall_floor",
                "feasible": True,
                "threshold": threshold,
                "known_recall_by_class": recalls,
                "unknown_rejection": unknown_rejection,
                "known_support": int(positive.sum().item()),
                "unknown_support": int(negative.sum().item()),
                "selection_rank": list(ranking),
            }
            if best is None or ranking > best[0]:
                best = (ranking, threshold, selection)
    if best is None:
        return {
            "selection_score": float("-inf"),
            "selection": {
                "criterion": "development_unknown_rejection_at_per_class_known_recall_floor",
                "feasible": False,
                "known_support": int(positive.sum().item()),
                "unknown_support": int(negative.sum().item()),
            },
        }
    return {"selection_score": best[0][0], "selection_rank": list(best[0]), "selection": best[2]}


def _component_selection(
    model: V27dHeads,
    payload: Mapping[str, Any],
    *,
    component: str,
    task: str,
    context: V27DContext,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    tensors = payload["tensors"]
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        output = model(**_head_inputs(tensors, indices, device))
    lengths = tensors["lengths"]
    rows = torch.arange(lengths.shape[0], device=device)
    if component == "knownness":
        scores = torch.sigmoid(output["knownness_logit_sequence"][rows, lengths.to(device) - 1]).cpu()
        return _knownness_selection(scores, payload, task=task, context=context)
    scores = torch.sigmoid(output["quality_logit_sequence"][rows, lengths.to(device) - 1]).cpu()
    target = tensors["quality_target_final"].cpu()
    auc = binary_auroc(scores, target)
    return {
        "selection_score": float("-inf") if auc is None else auc,
        "selection_rank": [] if auc is None else [auc],
        "selection": {
            "criterion": "development_quality_auroc",
            "feasible": auc is not None,
            "auroc": auc,
            "positive_support": int(target.sum().item()),
            "negative_support": int((~target).sum().item()),
        },
    }


def _run_component(
    model: V27dHeads,
    payload: Mapping[str, Any],
    *,
    component: str,
    development_payload: Mapping[str, Any],
    context: V27DContext,
    device: torch.device,
    logger: Any,
) -> dict[str, Any]:
    tensors = payload["tensors"]
    count = len(payload["parent_ids"])
    target_parent = tensors["knownness_target"] if component == "knownness" else tensors["quality_target_final"].to(dtype=torch.float32)
    if component == "knownness":
        prefix_number = torch.arange(1, tensors["padding_mask"].shape[1] + 1).unsqueeze(0)
        # A prefix is a training example only after its own deterministic
        # quality gate; a later high-quality event must not label an earlier
        # low-quality prefix as known.
        mask = (
            (~tensors["padding_mask"])
            & tensors["knownness_population"].unsqueeze(1)
            & tensors["quality_target_sequence"]
            & prefix_number.ge(context.config.calibration.minimum_quality_events)
        )
        target = target_parent.unsqueeze(1).expand_as(mask).to(dtype=torch.float32)
    else:
        mask = ~tensors["padding_mask"]
        target = tensors["quality_target_sequence"].to(dtype=torch.float32)
    positive_weight = _positive_weight(target, mask)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name in _component_keys(model, component):
            parameter.requires_grad = True
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=context.config.training.learning_rate, weight_decay=context.config.training.weight_decay)
    generator = torch.Generator().manual_seed(int(context.config.training.dataloader_seed))
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_rank: tuple[float, ...] | None = None
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(1, context.config.training.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_index, indices in enumerate(_batch_slices(count, context.config.training.batch_size, generator=generator), start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**_head_inputs(tensors, indices, device))
            batch_target = target[indices].to(device)
            batch_mask = mask[indices].to(device)
            loss = _parent_balanced_loss(output["knownness_logit_sequence"] if component == "knownness" else output["quality_logit_sequence"], batch_target, batch_mask, positive_weight=positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"V27d {component} loss is non-finite")
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(parameters, context.config.training.gradient_clip_norm, error_if_nonfinite=True).item())
            optimizer.step()
            losses.append(float(loss.detach().item()))
            if batch_index == 1 or batch_index % 10 == 0:
                logger.info("component=%s epoch=%d batch=%d loss=%.8f gradient_norm=%.6f", component, epoch, batch_index, losses[-1], gradient_norm)
        record = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "gradient_norm": gradient_norm, "elapsed_seconds": time.perf_counter() - started}
        selection = _component_selection(
            model,
            development_payload,
            component=component,
            task=str(payload["task"]),
            context=context,
            device=device,
        )
        record.update(selection)
        history.append(record)
        logger.info("component=%s epoch_complete=%d loss=%.8f", component, epoch, record["loss"])
        selection_rank = tuple(float(value) for value in selection.get("selection_rank", (selection["selection_score"],)))
        if best_rank is None or selection_rank > best_rank or best_state is None:
            best_score = float(selection["selection_score"])
            best_rank = selection_rank
            best_epoch = epoch
            best_state = _component_state(model, component)
    if best_state is None or best_epoch is None:
        raise RuntimeError(f"V27d {component} produced no selectable checkpoint")
    if component == "knownness" and not history[best_epoch - 1]["selection"].get("feasible", False):
        raise RuntimeError(
            "V27d knownness training produced no development checkpoint that "
            "satisfies the configured per-class known-recall safety floor"
        )
    _load_component_state(model, component, best_state)
    fit_parent_mask = (
        tensors["knownness_population"].bool() & tensors["quality_target_final"].bool()
        if component == "knownness"
        else torch.ones(count, dtype=torch.bool)
    )
    return {
        "component": component,
        "positive_weight": positive_weight,
        "history": history,
        "state": _component_state(model, component),
        "component_state_sha256": _tensor_state_hash(_component_state(model, component)),
        "selected_epoch": best_epoch,
        "selection_score": best_score,
        "selection": history[best_epoch - 1]["selection"],
        "fit_parent_count": int(fit_parent_mask.sum().item()),
        "fit_prefix_count": int(mask.sum().item()),
    }


def _evaluate(model: V27dHeads, payload: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    model.eval()
    tensors = payload["tensors"]
    with torch.inference_mode():
        output = model(**_head_inputs(tensors, torch.arange(len(payload["parent_ids"])), device))
    lengths = tensors["lengths"]
    rows = torch.arange(lengths.shape[0], device=device)
    known = output["knownness_logit_sequence"][rows, lengths.to(device) - 1].cpu()
    quality = output["quality_logit_sequence"][rows, lengths.to(device) - 1].cpu()
    labels = [str(value) for value in payload["label_names"]]
    quality_mask = tensors["quality_target_final"].cpu()
    known_mask = tensors["knownness_population"].cpu() & quality_mask
    target = tensors["knownness_target"].cpu()
    knownness_auroc = binary_auroc(known[known_mask], target[known_mask]) if bool(known_mask.any()) else None
    if bool(known_mask.any()):
        known_pred = known.ge(0.0)
        known_accuracy = float(known_pred[known_mask].eq(target[known_mask].bool()).float().mean().item())
    else:
        known_accuracy = None
    quality_pred = quality.ge(0.0)
    quality_accuracy = float(quality_pred.eq(quality_mask).float().mean().item()) if len(labels) else None
    return {
        "parent_count": len(labels),
        "knownness_fit_eligible_count": int(known_mask.sum().item()),
        "knownness_parent_accuracy_at_zero": known_accuracy,
        "knownness_parent_accuracy_at_logit_zero": known_accuracy,
        "knownness_auroc": knownness_auroc,
        "quality_parent_accuracy_at_zero": quality_accuracy,
        "quality_positive_count": int(quality_mask.sum().item()),
        "knownness_logit_mean": float(known.mean().item()),
        "quality_logit_mean": float(quality.mean().item()),
    }


def train_heads(context: V27DContext, *, variant: str, task: str, seed: int) -> dict[str, Any]:
    if int(seed) not in context.config.training.seeds:
        raise ValueError("V27d seed is not registered")
    feature_path = artifact_path(context.project_root, context.config.artifacts.feature_cache_template, variant=variant, role="representation_fit", task=task)
    dev_path = artifact_path(context.project_root, context.config.artifacts.feature_cache_template, variant=variant, role="representation_development", task=task)
    fit_payload = load_feature_cache(context, feature_path)
    dev_payload = load_feature_cache(context, dev_path)
    device = _device(context)
    log_path = artifact_path(context.project_root, context.config.artifacts.training_log_template, variant=variant, task=task, seed=seed)
    logger = configure_logger(log_path, name=f"v27d.{variant}.{task}.{seed}")
    try:
        known_model = build_heads(context, task=task, variant=variant, seed=seed).to(device)
        known_result = _run_component(known_model, fit_payload, component="knownness", development_payload=dev_payload, context=context, device=device, logger=logger)
        quality_model = build_heads(context, task=task, variant=variant, seed=seed).to(device)
        quality_result = _run_component(quality_model, fit_payload, component="quality", development_payload=dev_payload, context=context, device=device, logger=logger)
        final_model = build_heads(context, task=task, variant=variant, seed=seed).to(device)
        _load_component_state(final_model, "knownness", known_result["state"])
        _load_component_state(final_model, "quality", quality_result["state"])
        final_model.eval()
        summary = {
            "schema_version": "RADAR-LN3-V27d-head-training-summary-v1",
            "experiment_id": "V27d",
            "variant": variant,
            "task": task,
            "seed": int(seed),
            "feature_cache_path": relative_project_path(context.project_root, feature_path),
            "feature_cache_sha256": fit_payload["cache_sha256"],
            "knownness": {key: value for key, value in known_result.items() if key != "state"},
            "quality": {key: value for key, value in quality_result.items() if key != "state"},
            "development_metrics": _evaluate(final_model, dev_payload, device=device),
            "model_contract": final_model.model_contract(),
            "model_contract_sha256": canonical_sha256(final_model.model_contract()),
            "classification_logits_changed": False,
            "unknown_is_classifier_class": False,
            "test_payload_access_count": 0,
        }
        summary_path = artifact_path(context.project_root, context.config.artifacts.head_summary_template, variant=variant, task=task, seed=seed)
        write_sealed_json(summary_path, summary)
        checkpoint = {
            "checkpoint_schema": HEAD_CHECKPOINT_SCHEMA,
            "experiment_id": "V27d",
            "variant": variant,
            "task": task,
            "seed": int(seed),
            "feature_cache_sha256": fit_payload["cache_sha256"],
            "v27d_manifest_sha256": context.manifest_sha256,
            "model_contract": final_model.model_contract(),
            "model_contract_sha256": canonical_sha256(final_model.model_contract()),
            "knownness_state": known_result["state"],
            "quality_state": quality_result["state"],
            "head_state": {name: value.detach().cpu().clone() for name, value in final_model.state_dict().items()},
            "knownness_state_sha256": known_result["component_state_sha256"],
            "quality_state_sha256": quality_result["component_state_sha256"],
            "metrics": summary["development_metrics"],
            "deployment_eligible": False,
            "test_payload_access_count": 0,
        }
        checkpoint["head_state_sha256"] = _tensor_state_hash(checkpoint["head_state"])
        checkpoint_path = artifact_path(context.project_root, context.config.artifacts.head_checkpoint_template, variant=variant, task=task, seed=seed)
        write_torch_if_absent_or_identical(checkpoint_path, checkpoint)
        return {
            "status": "completed",
            "checkpoint_path": relative_project_path(context.project_root, checkpoint_path),
            "summary_path": relative_project_path(context.project_root, summary_path),
            "log_path": relative_project_path(context.project_root, log_path),
            "knownness": {key: value for key, value in known_result.items() if key != "state"},
            "quality": {key: value for key, value in quality_result.items() if key != "state"},
            "development_metrics": summary["development_metrics"],
        }
    finally:
        close_logger(logger)


def load_head_checkpoint(context: V27DContext, *, variant: str, task: str, seed: int) -> tuple[V27dHeads, dict[str, Any]]:
    path = artifact_path(context.project_root, context.config.artifacts.head_checkpoint_template, variant=variant, task=task, seed=seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != HEAD_CHECKPOINT_SCHEMA:
        raise ValueError("V27d head checkpoint schema differs")
    if payload.get("v27d_manifest_sha256") != context.manifest_sha256 or payload.get("variant") != variant or payload.get("task") != task or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("V27d head checkpoint provenance differs")
    feature_path = artifact_path(
        context.project_root,
        context.config.artifacts.feature_cache_template,
        variant=variant,
        role="representation_fit",
        task=task,
    )
    feature_cache = load_feature_cache(context, feature_path)
    if payload.get("feature_cache_sha256") != feature_cache.get("cache_sha256"):
        raise ValueError("V27d head checkpoint feature-cache binding differs")
    model = build_heads(context, task=task, variant=variant, seed=seed)
    if payload.get("model_contract") != model.model_contract() or payload.get("model_contract_sha256") != canonical_sha256(model.model_contract()):
        raise ValueError("V27d head checkpoint model contract differs")
    state = payload.get("head_state")
    if not isinstance(state, Mapping) or payload.get("head_state_sha256") != _tensor_state_hash(state):
        raise ValueError("V27d head checkpoint state hash differs")
    for component in ("knownness", "quality"):
        component_state = payload.get(f"{component}_state")
        if not isinstance(component_state, Mapping) or payload.get(f"{component}_state_sha256") != _tensor_state_hash(component_state):
            raise ValueError(f"V27d {component} checkpoint state hash differs")
        expected_state = {name: state[name] for name in _component_keys(model, component)}
        if set(component_state) != set(expected_state) or any(
            not isinstance(component_state[name], torch.Tensor)
            or not torch.equal(component_state[name], expected_state[name])
            for name in expected_state
        ):
            raise ValueError(f"V27d {component} checkpoint state is inconsistent with head state")
    model.load_state_dict(state, strict=True)
    return model, dict(payload)


def _device(context: V27DContext) -> torch.device:
    if context.config.training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d configuration requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


__all__ = ["HEAD_CHECKPOINT_SCHEMA", "load_head_checkpoint", "train_heads"]
