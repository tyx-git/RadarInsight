"""Training and threshold selection for V27d-r7."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical
from v27.V27d.logging_utils import close_logger, configure_logger, log_event
from v27.V27d.r6.model import BirdVerifier as R6BirdVerifier

from .features import load_population_cache
from .model import BirdVerifierR7, build_verifier, initialize_from_r6
from .protocol import R7_CHECKPOINT_SCHEMA, R7_EXPERIMENT_ID, R7_ROLES, artifact_path, load_context, source_path, write_sealed_json


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cuda")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r7 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r7 device must be cpu or cuda")


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True)


def _inputs(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device, *, use_branch: bool) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    dispersion = tensors["dispersion_features"] if use_branch else torch.zeros_like(tensors["dispersion_features"])
    return {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "event_facts_sequence": tensors["event_facts_sequence"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
        "dispersion_features": dispersion[indices].to(device),
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
    losses = []
    if bool(bird.any()):
        losses.append(parent_loss[bird].mean())
    if bool(false_positive.any()):
        losses.append(parent_loss[false_positive].mean())
    return (0.5 * (losses[0] + losses[1]) if len(losses) == 2 else parent_loss[eligible].mean()), components


def _load_r6_checkpoint(root: Path, config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    path = source_path(root, config["source"]["r6_liquid_checkpoint_template"].format(seed=seed), must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("arm") != "liquid_treatment" or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("V27d-r6 Liquid checkpoint identity differs")
    if payload.get("checkpoint_schema") != "RADAR-LN3-V27d-r6-checkpoint-v1":
        raise ValueError("V27d-r6 Liquid checkpoint schema differs")
    r6_manifest_path = source_path(root, config["source"]["r6_manifest_path"], must_exist=True)
    import json

    manifest_payload = json.loads(r6_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("V27d-r6 manifest is not an object")
    declared_manifest_hash = canonical_sha256({key: value for key, value in manifest_payload.items() if key != "payload_sha256"})
    if manifest_payload.get("payload_sha256") != declared_manifest_hash or payload.get("manifest_sha256") != sha256_file(r6_manifest_path):
        raise ValueError("V27d-r6 checkpoint manifest provenance differs")
    fit_path = source_path(root, config["source"]["r6_cache_template"].format(role="representation_fit"), must_exist=True)
    fit_payload = torch.load(fit_path, map_location="cpu", weights_only=False)
    if not isinstance(fit_payload, Mapping) or payload.get("fit_cache_sha256") != fit_payload.get("cache_sha256"):
        raise ValueError("V27d-r6 checkpoint fit-cache provenance differs")
    return {**dict(payload), "_path": path}


def _initialize(model: BirdVerifierR7, r6_checkpoint: Mapping[str, Any]) -> None:
    initialize_from_r6(model, r6_checkpoint["head_state"])


def _candidate_masks(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    candidate = tensors["candidate_population"].bool()
    positive = tensors["bird_positive"].bool() & candidate
    negative = tensors["route_false_positive_negative"].bool() & candidate
    return candidate, positive, negative


def _output(model: BirdVerifierR7, payload: Mapping[str, Any], device: torch.device, *, use_branch: bool) -> dict[str, Any]:
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        return model(**_inputs(payload, indices, device, use_branch=use_branch))


def select_threshold(model: BirdVerifierR7, payload: Mapping[str, Any], *, device: torch.device, use_branch: bool, minimum_accepted_bird: int) -> dict[str, Any]:
    scores = _output(model, payload, device, use_branch=use_branch)["verifier_score_final"].detach().cpu()
    _, positive, negative = _candidate_masks(payload)
    candidates = sorted(set(float(value) for value in scores[positive].tolist()))
    if not candidates:
        raise ValueError("V27d-r7 threshold pool has no Bird positive")
    candidates.append(float(torch.nextafter(torch.tensor(candidates[-1]), torch.tensor(float("inf"))).item()))
    best: tuple[tuple[float, int, float], dict[str, Any]] | None = None
    for threshold in candidates:
        accepted = scores.ge(threshold)
        accepted_bird = int((accepted & positive).sum())
        if accepted_bird < minimum_accepted_bird:
            continue
        rejected_negative = int((negative & ~accepted).sum())
        false_positive_rejection = float(rejected_negative / max(1, int(negative.sum())))
        accepted_candidate = accepted & (positive | negative)
        precision = float(accepted_bird / max(1, int(accepted_candidate.sum())))
        record = {
            "feasible": True,
            "threshold": threshold,
            "accepted_bird": accepted_bird,
            "bird_support": int(positive.sum()),
            "rejected_bird": int((positive & ~accepted).sum()),
            "accepted_false_positive": int((negative & accepted).sum()),
            "rejected_false_positive": rejected_negative,
            "false_positive_rejection": false_positive_rejection,
            "bird_precision": precision,
        }
        ranking = (false_positive_rejection, accepted_bird, -threshold)
        if best is None or ranking > best[0]:
            best = (ranking, record)
    if best is None:
        raise RuntimeError("V27d-r7 found no recall-safe threshold")
    return best[1]


def _freeze_core(model: BirdVerifierR7) -> list[nn.Parameter]:
    for parameter in model.core.parameters():
        parameter.requires_grad = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("V27d-r7 no trainable readout parameters")
    return parameters


def train_arm(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if arm not in {"zero_control", "stable8_treatment"} or int(seed) not in (42, 43, 44):
        raise ValueError("V27d-r7 arm or seed is not registered")
    fit = load_population_cache(_cache_path(root, config, "representation_fit"))
    development = load_population_cache(_cache_path(root, config, "representation_development"))
    device = _device(config)
    use_branch = arm == "stable8_treatment"
    model = build_verifier(config, arm=arm, seed=seed)
    r6_checkpoint = _load_r6_checkpoint(root, config, seed)
    _initialize(model, r6_checkpoint)
    model = model.to(device)
    model.train()
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
    logger = configure_logger(log_path, name=f"v27d-r7.{arm}.{seed}")
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    started = time.perf_counter()
    try:
        log_event(logger, "training_start", experiment_id=R7_EXPERIMENT_ID, arm=arm, seed=seed, device=str(device), candidate_population=int(selected.numel()), use_branch=use_branch, frozen_core=True)
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            order = selected[torch.randperm(selected.numel(), generator=generator)]
            losses: list[float] = []
            components: list[dict[str, float]] = []
            for batch_no, start in enumerate(range(0, len(order), int(config["training"]["batch_size"])), start=1):
                indices = order[start : start + int(config["training"]["batch_size"])]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_inputs(fit, indices, device, use_branch=use_branch))
                batch_target = target[indices].to(device).unsqueeze(1).expand_as(output["verifier_logit_sequence"])
                batch_mask = mask[indices].to(device)
                loss, component = _parent_balanced_loss(output["verifier_logit_sequence"], batch_target, batch_mask, fit["tensors"]["bird_positive"][indices].bool().to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r7 loss is non-finite")
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(trainable, float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True).item())
                optimizer.step()
                losses.append(float(loss.detach().item()))
                components.append(component)
                if batch_no == 1 or batch_no == (len(order) + int(config["training"]["batch_size"]) - 1) // int(config["training"]["batch_size"]):
                    log_event(logger, "batch", epoch=epoch, batch=batch_no, loss=losses[-1], gradient_norm=gradient_norm)
            selection = select_threshold(model, development, device=device, use_branch=use_branch, minimum_accepted_bird=minimum_accepted_bird)
            record = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "loss_components": {key: sum(item[key] for item in components) / len(components) for key in components[0]}, "selection": selection, "elapsed_seconds": time.perf_counter() - started}
            history.append(record)
            log_event(logger, "development_selection", epoch=epoch, bird_precision=selection["bird_precision"], false_positive_rejection=selection["false_positive_rejection"], accepted_bird=selection["accepted_bird"])
            if best_selection is None or (selection["false_positive_rejection"], selection["accepted_bird"]) > (best_selection["false_positive_rejection"], best_selection["accepted_bird"]):
                best_selection = dict(selection)
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if best_state is None or best_selection is None or best_epoch is None:
            raise RuntimeError("V27d-r7 produced no checkpoint")
        model.load_state_dict(best_state, strict=True)
        contract = model.model_contract()
        checkpoint = {
            "checkpoint_schema": R7_CHECKPOINT_SCHEMA,
            "experiment_id": R7_EXPERIMENT_ID,
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
        summary = {"schema_version": "RADAR-LN3-V27d-r7-training-summary-v1", "experiment_id": R7_EXPERIMENT_ID, "arm": arm, "seed": int(seed), "selected_epoch": best_epoch, "selection": best_selection, "checkpoint_path": relative_project_path(root, checkpoint_path), "log_path": relative_project_path(root, log_path), "r6_initial_checkpoint_path": relative_project_path(root, r6_checkpoint["_path"]), "frozen_liquid_core": True, "use_stable8_branch": use_branch, "formal_l3_enabled": False}
        summary_path = artifact_path(root, config["artifacts"]["summary_template"].format(arm=arm, seed=seed))
        write_sealed_json(summary_path, summary)
        log_event(logger, "training_complete", selected_epoch=best_epoch, bird_precision=best_selection["bird_precision"], false_positive_rejection=best_selection["false_positive_rejection"])
        return {"status": "completed", "checkpoint_path": relative_project_path(root, checkpoint_path), "summary_path": relative_project_path(root, summary_path), "log_path": relative_project_path(root, log_path), "selection": best_selection}
    finally:
        close_logger(logger)


def load_checkpoint(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> tuple[BirdVerifierR7, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != R7_CHECKPOINT_SCHEMA or payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("V27d-r7 checkpoint provenance/schema differs")
    model = build_verifier(config, arm=arm, seed=seed)
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V27d-r7 checkpoint contract differs")
    model.load_state_dict(payload["head_state"], strict=True)
    return model, dict(payload), config


__all__ = ["_inputs", "load_checkpoint", "train_arm"]
