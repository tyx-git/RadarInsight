"""Training and development threshold selection for V27d-r8."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical
from v27.V27d.logging_utils import close_logger, configure_logger, log_event
from v27.V27d.r6.training import load_checkpoint as load_r6_checkpoint

from .features import load_population_cache
from .model import BirdVerifierR8, R8_ARM_FLAGS, build_verifier, initialize_from_r6
from .protocol import R8_ARMS, R8_CHECKPOINT_SCHEMA, R8_EXPERIMENT_ID, R8_SEEDS, artifact_path, load_context, source_path, write_sealed_json


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cuda")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r8 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r8 device must be cpu or cuda")


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True)


def _inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device, *, use_course: bool, use_stable8: bool) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    dispersion = tensors["dispersion_features"] if use_stable8 else torch.zeros_like(tensors["dispersion_features"])
    course = tensors["course_features"] if use_course else torch.zeros_like(tensors["course_features"])
    return {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "event_facts_sequence": tensors["event_facts_sequence"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
        "dispersion_features": dispersion[indices].to(device),
        "course_features": course[indices].to(device),
    }


def _prefix_mask(payload: Mapping[str, Any]) -> torch.Tensor:
    return payload["tensors"]["prefix_route_mask"].bool() & payload["tensors"]["candidate_population"].unsqueeze(1)


def _parent_balanced_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, positive_rows: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    values = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    active = mask.sum(dim=1)
    eligible = active.gt(0)
    if not bool(eligible.any()):
        return logits.sum() * 0.0, {"bird": 0.0, "false_positive": 0.0}
    parent_loss = (values * mask.to(values.dtype)).sum(dim=1) / active.clamp_min(1)
    bird = eligible & positive_rows
    false_positive = eligible & ~positive_rows
    components = {
        "bird": float(parent_loss[bird].mean().detach().item()) if bool(bird.any()) else 0.0,
        "false_positive": float(parent_loss[false_positive].mean().detach().item()) if bool(false_positive.any()) else 0.0,
    }
    parts = []
    if bool(bird.any()):
        parts.append(parent_loss[bird].mean())
    if bool(false_positive.any()):
        parts.append(parent_loss[false_positive].mean())
    return (0.5 * (parts[0] + parts[1]) if len(parts) == 2 else parent_loss[eligible].mean()), components


def _load_r6(root: Path, config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    r6_config = source_path(root, config["source"]["r6_config_path"], must_exist=True)
    model, checkpoint, _ = load_r6_checkpoint(root, arm="liquid_treatment", seed=seed, config_path=r6_config)
    if checkpoint.get("checkpoint_schema") != "RADAR-LN3-V27d-r6-checkpoint-v1":
        raise ValueError("V27d-r6 checkpoint schema differs")
    return {**checkpoint, "_path": source_path(root, config["source"]["r6_liquid_checkpoint_template"].format(seed=seed), must_exist=True), "_model": model}


def _masks(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    candidate = tensors["candidate_population"].bool()
    positive = tensors["bird_positive"].bool() & candidate
    negative = tensors["route_false_positive_negative"].bool() & candidate
    return candidate, positive, negative


def _output(model: BirdVerifierR8, payload: Mapping[str, Any], device: torch.device, *, use_course: bool, use_stable8: bool) -> dict[str, Any]:
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        return model(**_inputs(payload, indices, device, use_course=use_course, use_stable8=use_stable8))


def select_threshold(model: BirdVerifierR8, payload: Mapping[str, Any], *, device: torch.device, use_course: bool, use_stable8: bool, minimum_accepted_bird: int) -> dict[str, Any]:
    scores = _output(model, payload, device, use_course=use_course, use_stable8=use_stable8)["verifier_score_final"].detach().cpu()
    _, positive, negative = _masks(payload)
    candidates = sorted(set(float(value) for value in scores[positive].tolist()))
    if not candidates:
        raise ValueError("V27d-r8 threshold pool has no Bird positive")
    candidates.append(float(torch.nextafter(torch.tensor(candidates[-1]), torch.tensor(float("inf"))).item()))
    best: tuple[tuple[float, int, float], dict[str, Any]] | None = None
    for threshold in candidates:
        accepted = scores.ge(threshold)
        accepted_bird = int((accepted & positive).sum())
        if accepted_bird < minimum_accepted_bird:
            continue
        rejected_negative = int((negative & ~accepted).sum())
        accepted_candidate = accepted & (positive | negative)
        record = {
            "feasible": True,
            "threshold": threshold,
            "accepted_bird": accepted_bird,
            "bird_support": int(positive.sum()),
            "rejected_bird": int((positive & ~accepted).sum()),
            "accepted_false_positive": int((accepted & negative).sum()),
            "rejected_false_positive": rejected_negative,
            "false_positive_rejection": float(rejected_negative / max(1, int(negative.sum()))),
            "bird_precision": float(accepted_bird / max(1, int(accepted_candidate.sum()))),
        }
        ranking = (record["false_positive_rejection"], accepted_bird, -threshold)
        if best is None or ranking > best[0]:
            best = (ranking, record)
    if best is None:
        raise RuntimeError("V27d-r8 found no recall-safe threshold")
    return best[1]


def _initialize(model: BirdVerifierR8, r6_checkpoint: Mapping[str, Any]) -> None:
    initialize_from_r6(model, r6_checkpoint["head_state"])


def _freeze_core(model: BirdVerifierR8) -> list[nn.Parameter]:
    for parameter in model.core.parameters():
        parameter.requires_grad = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("V27d-r8 no trainable readout parameters")
    return parameters


def train_arm(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if arm not in R8_ARMS or int(seed) not in R8_SEEDS:
        raise ValueError("V27d-r8 arm or seed is not registered")
    fit = load_population_cache(_cache_path(root, config, "representation_fit"))
    development = load_population_cache(_cache_path(root, config, "representation_development"))
    use_course, use_stable8 = R8_ARM_FLAGS[arm]
    device = _device(config)
    model = build_verifier(config, arm=arm, seed=seed)
    r6_checkpoint = _load_r6(root, config, seed)
    _initialize(model, r6_checkpoint)
    model = model.to(device)
    trainable = _freeze_core(model)
    selected = torch.where(fit["tensors"]["candidate_population"].bool())[0]
    target = fit["tensors"]["knownness_target"]
    mask = _prefix_mask(fit)
    positive_rows = fit["tensors"]["bird_positive"][selected].bool()
    optimizer = torch.optim.AdamW(trainable, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(int(seed))
    baseline_accepted_bird = int(r6_checkpoint["selection"]["accepted_bird"])
    minimum_accepted_bird = baseline_accepted_bird - int(config["selection"]["max_additional_bird_rejections_per_pool"])
    log_path = artifact_path(root, config["artifacts"]["log_template"].format(arm=arm, seed=seed))
    logger = configure_logger(log_path, name=f"v27d-r8.{arm}.{seed}")
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    started = time.perf_counter()
    try:
        log_event(logger, "training_start", experiment_id=R8_EXPERIMENT_ID, arm=arm, seed=seed, device=str(device), candidate_population=int(selected.numel()), use_course=use_course, use_stable8=use_stable8, frozen_core=True)
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            order = selected[torch.randperm(selected.numel(), generator=generator)]
            losses: list[float] = []
            components: list[dict[str, float]] = []
            for batch_no, start in enumerate(range(0, len(order), int(config["training"]["batch_size"])), start=1):
                indices = order[start : start + int(config["training"]["batch_size"])]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_inputs(fit, indices, device, use_course=use_course, use_stable8=use_stable8))
                batch_target = target[indices].to(device).unsqueeze(1).expand_as(output["verifier_logit_sequence"])
                batch_mask = mask[indices].to(device)
                loss, component = _parent_balanced_loss(output["verifier_logit_sequence"], batch_target, batch_mask, fit["tensors"]["bird_positive"][indices].bool().to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r8 loss is non-finite")
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(trainable, float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True).item())
                optimizer.step()
                losses.append(float(loss.detach().item()))
                components.append(component)
                if batch_no == 1 or batch_no == (len(order) + int(config["training"]["batch_size"]) - 1) // int(config["training"]["batch_size"]):
                    log_event(logger, "batch", epoch=epoch, batch=batch_no, loss=losses[-1], gradient_norm=gradient_norm)
            selection = select_threshold(model, development, device=device, use_course=use_course, use_stable8=use_stable8, minimum_accepted_bird=minimum_accepted_bird)
            record = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "loss_components": {key: sum(item[key] for item in components) / len(components) for key in components[0]}, "selection": selection, "elapsed_seconds": time.perf_counter() - started}
            history.append(record)
            log_event(logger, "development_selection", epoch=epoch, bird_precision=selection["bird_precision"], false_positive_rejection=selection["false_positive_rejection"], accepted_bird=selection["accepted_bird"])
            if best_selection is None or (selection["false_positive_rejection"], selection["accepted_bird"]) > (best_selection["false_positive_rejection"], best_selection["accepted_bird"]):
                best_selection = dict(selection)
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if best_state is None or best_selection is None or best_epoch is None:
            raise RuntimeError("V27d-r8 produced no checkpoint")
        model.load_state_dict(best_state, strict=True)
        contract = model.model_contract()
        checkpoint = {
            "checkpoint_schema": R8_CHECKPOINT_SCHEMA,
            "experiment_id": R8_EXPERIMENT_ID,
            "arm": arm,
            "seed": int(seed),
            "manifest_sha256": manifest["manifest_sha256"],
            "fit_cache_sha256": fit["cache_sha256"],
            "r6_initial_checkpoint_sha256": sha256_file(r6_checkpoint["_path"]),
            "r6_initial_checkpoint_manifest_sha256": r6_checkpoint["manifest_sha256"],
            "model_contract": contract,
            "model_contract_sha256": canonical_sha256(contract),
            "selected_epoch": best_epoch,
            "selection": best_selection,
            "history": history,
            "use_course": use_course,
            "use_stable8": use_stable8,
            "classification_logits_changed": False,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
            "formal_l3_enabled": False,
            "test_payload_access_count": 0,
            "frozen_liquid_core": True,
            "head_state": best_state,
        }
        checkpoint["head_state_sha256"] = canonical_sha256({name: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.numpy().tobytes().hex()} for name, value in sorted(best_state.items())})
        checkpoint_path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed))
        write_torch_if_absent_or_identical(checkpoint_path, checkpoint)
        summary = {"schema_version": "RADAR-LN3-V27d-r8-training-summary-v1", "experiment_id": R8_EXPERIMENT_ID, "arm": arm, "seed": int(seed), "selected_epoch": best_epoch, "selection": best_selection, "checkpoint_path": relative_project_path(root, checkpoint_path), "log_path": relative_project_path(root, log_path), "r6_initial_checkpoint_path": relative_project_path(root, r6_checkpoint["_path"]), "use_course": use_course, "use_stable8": use_stable8, "frozen_liquid_core": True, "formal_l3_enabled": False}
        summary_path = artifact_path(root, config["artifacts"]["summary_template"].format(arm=arm, seed=seed))
        write_sealed_json(summary_path, summary)
        log_event(logger, "training_complete", selected_epoch=best_epoch, bird_precision=best_selection["bird_precision"], false_positive_rejection=best_selection["false_positive_rejection"])
        return {"status": "completed", "checkpoint_path": relative_project_path(root, checkpoint_path), "summary_path": relative_project_path(root, summary_path), "log_path": relative_project_path(root, log_path), "selection": best_selection}
    finally:
        close_logger(logger)


def load_checkpoint(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> tuple[BirdVerifierR8, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != R8_CHECKPOINT_SCHEMA or payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("V27d-r8 checkpoint provenance/schema differs")
    model = build_verifier(config, arm=arm, seed=seed)
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V27d-r8 checkpoint contract differs")
    model.load_state_dict(payload["head_state"], strict=True)
    return model, dict(payload), config


__all__ = ["_inputs", "load_checkpoint", "train_arm"]