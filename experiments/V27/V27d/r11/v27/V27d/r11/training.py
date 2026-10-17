"""Train the isolated V27d-r11 causal stable8 shape projection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.runtime import (
    relative_project_path,
    verify_file_hash,
)
from v27.V27d.logging_utils import (
    close_logger,
    configure_logger,
    log_event,
    log_metric_set,
    progress_interval,
)
from v27.V27d.metrics import binary_auroc
from v27.V27d.r6.training import load_checkpoint as load_r6_checkpoint

from .checkpoint import save_checkpoint
from .features import load_population_cache, validate_cache_binding
from .model import BirdVerifierR11, build_model, initialize_from_r6
from .protocol import (
    R11_EXPERIMENT_ID,
    R11_ROLES,
    R11_SEEDS,
    artifact_path,
    load_context,
    source_path,
    write_sealed_json_if_identical,
)


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cuda")).strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r11 requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r11 training.device must be cpu or cuda")


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    if role not in R11_ROLES:
        raise ValueError("V27d-r11 cache role is invalid")
    return artifact_path(
        root,
        config["artifacts"]["cache_template"].format(role=role),
        must_exist=True,
    )


def _inputs(
    payload: Mapping[str, Any],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    names = (
        "summary_features",
        "event_trajectory",
        "task_facts_sequence",
        "event_facts_sequence",
        "padding_mask",
        "event_frame_values",
        "event_frame_mask",
    )
    return {name: tensors[name][indices].to(device) for name in names}


def _r6_inputs(
    payload: Mapping[str, Any],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    names = (
        "summary_features",
        "event_trajectory",
        "task_facts_sequence",
        "event_facts_sequence",
        "padding_mask",
    )
    return {name: tensors[name][indices].to(device) for name in names}


def _prefix_mask(payload: Mapping[str, Any]) -> torch.Tensor:
    tensors = payload["tensors"]
    return tensors["prefix_route_mask"].bool() & tensors["candidate_population"].unsqueeze(1).bool()


def _candidate_masks(
    payload: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    candidate = tensors["candidate_population"].bool()
    positive = candidate & tensors["bird_positive"].bool()
    negative = candidate & tensors["route_false_positive_negative"].bool()
    if not torch.equal(candidate, positive | negative) or bool((positive & negative).any()):
        raise ValueError("V27d-r11 population masks are invalid")
    return candidate, positive, negative


def _parent_balanced_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equalize positive/negative parent groups after per-parent prefix averaging."""

    values = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    active = mask.sum(dim=1)
    eligible = active.gt(0)
    if not bool(eligible.any()):
        return logits.sum() * 0.0, {"bird": 0.0, "false_positive": 0.0}
    parent_loss = (values * mask.to(values.dtype)).sum(dim=1) / active.clamp_min(1)
    bird = eligible & positive_rows.bool()
    false_positive = eligible & ~positive_rows.bool()
    components = {
        "bird": float(parent_loss[bird].mean().detach().item()) if bool(bird.any()) else 0.0,
        "false_positive": float(parent_loss[false_positive].mean().detach().item()) if bool(false_positive.any()) else 0.0,
    }
    groups = [parent_loss[bird].mean(), parent_loss[false_positive].mean()]
    groups = [value for value, present in zip(groups, (bool(bird.any()), bool(false_positive.any())), strict=True) if present]
    return sum(groups) / len(groups), components


def _freeze_except_shape_projection(model: BirdVerifierR11) -> tuple[str, ...]:
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("shape_projection.")
        if parameter.requires_grad:
            trainable.append(name)
    expected = ("shape_projection.weight",)
    if tuple(trainable) != expected:
        raise ValueError(f"V27d-r11 trainable parameters differ: {trainable}")
    return tuple(trainable)


def _load_r6_reference(
    root: Path,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    if int(seed) not in R11_SEEDS:
        raise ValueError("V27d-r11 seed is not registered")
    source = config["source"]
    checkpoint_path = source_path(
        root,
        source["r6_checkpoint_template"].format(seed=seed),
        must_exist=True,
    )
    verify_file_hash(
        checkpoint_path,
        source["expected_checkpoint_file_sha256s"][str(seed)],
        field=f"V27d-r11 r6 checkpoint seed {seed}",
    )
    r6_config_path = source_path(root, source["r6_config_path"], must_exist=True)
    model, payload, _ = load_r6_checkpoint(
        root,
        arm="liquid_treatment",
        seed=int(seed),
        config_path=r6_config_path,
    )
    if payload.get("head_state_sha256") != source["expected_checkpoint_head_state_sha256s"][str(seed)]:
        raise ValueError(f"V27d-r11 r6 head state hash differs for seed {seed}")
    return model, dict(payload), checkpoint_path


def _scores(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    r6: bool,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(len(payload["parent_ids"]), dtype=torch.long)
    inputs = _r6_inputs(payload, indices, device) if r6 else _inputs(payload, indices, device)
    model.eval()
    with torch.inference_mode():
        output = model(**inputs)
    return output["verifier_score_final"].detach().cpu()


def _selection_from_scores(
    payload: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    baseline_accepted_positive: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _, positive, negative = _candidate_masks(payload)
    candidates = sorted(float(value) for value in scores[positive].tolist())
    if not candidates:
        raise ValueError("V27d-r11 development pool has no Bird positive")
    last = torch.tensor(candidates[-1], dtype=scores.dtype)
    candidates.append(float(torch.nextafter(last, torch.tensor(float("inf"), dtype=scores.dtype)).item()))
    maximum_loss = int(config["selection"]["max_additional_bird_rejections_per_pool"])
    minimum_positive = max(0, int(baseline_accepted_positive) - maximum_loss)
    best: tuple[tuple[float, int, float], dict[str, Any]] | None = None
    for threshold in candidates:
        accepted = scores.ge(threshold)
        accepted_positive = int((accepted & positive).sum())
        if accepted_positive < minimum_positive:
            continue
        accepted_negative = int((accepted & negative).sum())
        rejected_negative = int((negative & ~accepted).sum())
        accepted_candidate = accepted & (positive | negative)
        precision = accepted_positive / max(1, int(accepted_candidate.sum()))
        record = {
            "feasible": True,
            "threshold": threshold,
            "higher_is_known": True,
            "accepted_bird": accepted_positive,
            "bird_support": int(positive.sum()),
            "rejected_bird": int(positive.sum()) - accepted_positive,
            "accepted_false_positive": accepted_negative,
            "false_positive_support": int(negative.sum()),
            "rejected_false_positive": rejected_negative,
            "false_positive_rejection": rejected_negative / max(1, int(negative.sum())),
            "bird_precision": precision,
        }
        ranking = (record["false_positive_rejection"], accepted_positive, -threshold)
        if best is None or ranking > best[0]:
            best = (ranking, record)
    if best is None:
        raise RuntimeError("V27d-r11 found no recall-safe development threshold")
    return best[1]


def _ranking(payload: Mapping[str, Any], scores: torch.Tensor) -> dict[str, Any]:
    _, positive, negative = _candidate_masks(payload)
    mask = positive | negative
    if not bool(mask.any()):
        raise ValueError("V27d-r11 ranking pool is empty")
    target = positive[mask]
    return {
        "support": int(mask.sum()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "auroc_higher_is_known": binary_auroc(scores[mask], target),
        "score_mean_positive": float(scores[positive].mean()) if bool(positive.any()) else None,
        "score_mean_negative": float(scores[negative].mean()) if bool(negative.any()) else None,
    }


def _action_metrics(
    payload: Mapping[str, Any],
    scores: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    candidate, positive, negative = _candidate_masks(payload)
    labels = [str(value) for value in payload["label_names"]]
    accepted = candidate & scores.ge(float(threshold))
    unknown = candidate & torch.tensor(
        [label == "UnknownTarget" for label in labels], dtype=torch.bool
    )
    by_label: dict[str, dict[str, int]] = {}
    for label in sorted(set(labels)):
        label_mask = candidate & torch.tensor(
            [item == label for item in labels], dtype=torch.bool
        )
        by_label[label] = {
            "support": int(label_mask.sum()),
            "accepted": int((accepted & label_mask).sum()),
            "rejected": int((label_mask & ~accepted).sum()),
        }
    return {
        "threshold": float(threshold),
        "candidate_support": int(candidate.sum()),
        "bird_support": int(positive.sum()),
        "false_positive_support": int(negative.sum()),
        "accepted_bird": int((accepted & positive).sum()),
        "rejected_bird": int((positive & ~accepted).sum()),
        "bird_action_precision": float((accepted & positive).sum().item() / max(1, accepted.sum().item())),
        "bird_action_recall": float((accepted & positive).sum().item() / max(1, positive.sum().item())),
        "accepted_false_positive": int((accepted & negative).sum()),
        "rejected_false_positive": int((negative & ~accepted).sum()),
        "false_positive_rejection": float((negative & ~accepted).sum().item() / max(1, negative.sum().item())),
        "unknown_support": int(unknown.sum()),
        "unknown_rejection": float((unknown & ~accepted).sum().item() / max(1, unknown.sum().item())),
        "pending_low_quality_excluded": int((~payload["tensors"]["quality_target_final"].bool()).sum()),
        "by_label": by_label,
    }


def train_seed(
    project_root: str | Path,
    *,
    seed: int,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) not in R11_SEEDS:
        raise ValueError("V27d-r11 seed is not registered")
    caches = {
        role: load_population_cache(_cache_path(root, config, role))
        for role in R11_ROLES
    }
    for role, payload in caches.items():
        validate_cache_binding(payload, root=root, config=config, role=role)
        if payload.get("r11_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"V27d-r11 cache manifest differs: {role}")
    fit = caches["representation_fit"]
    development = caches["representation_development"]
    device = _device(config)
    reference, r6_payload, r6_checkpoint_path = _load_r6_reference(root, config, seed=seed)
    reference.to(device).eval()
    model = build_model(config, arm="shape_treatment", seed=seed).to(device)
    initialize_from_r6(model, r6_payload["head_state"])
    trainable_names = _freeze_except_shape_projection(model)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    selected = torch.where(fit["tensors"]["candidate_population"].bool())[0]
    if not len(selected):
        raise ValueError("V27d-r11 fit population has no route candidates")
    prefix_mask = _prefix_mask(fit)
    if not bool(prefix_mask[selected].any()):
        raise ValueError("V27d-r11 fit population has no eligible prefixes")
    target = fit["tensors"]["knownness_target"].float()
    positive_rows = fit["tensors"]["bird_positive"].bool()
    baseline_scores = _scores(reference, development, r6=True, device=device)
    baseline_threshold = float(r6_payload["selection"]["threshold"])
    baseline_actions = _action_metrics(development, baseline_scores, baseline_threshold)
    baseline_accepted_positive = int(baseline_actions["accepted_bird"])
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["training"]["dataloader_seed"]) + int(seed)
    )
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    steps_per_epoch = (len(selected) + batch_size - 1) // batch_size
    log_path = artifact_path(
        root,
        config["artifacts"]["log_template"].format(seed=seed),
    )
    logger = configure_logger(log_path, name=f"v27d.r11.seed.{seed}")
    started = time.perf_counter()
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    try:
        log_event(
            logger,
            "training_start",
            experiment_id=R11_EXPERIMENT_ID,
            seed=seed,
            device=str(device),
            fit_candidate_parents=int(selected.numel()),
            trainable_parameters=",".join(trainable_names),
            frozen_reference_checkpoint=relative_project_path(root, r6_checkpoint_path),
            formal_l3_enabled=False,
            unknown_payload_access_count=0,
            test_payload_access_count=0,
        )
        for epoch in range(1, epochs + 1):
            model.train()
            shuffled = selected[torch.randperm(len(selected), generator=generator)]
            losses: list[float] = []
            components: list[dict[str, float]] = []
            gradient_max = 0.0
            for batch_number, start in enumerate(
                range(0, len(shuffled), batch_size), start=1
            ):
                batch_indices = shuffled[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = model(**_inputs(fit, batch_indices, device))
                batch_target = target[batch_indices].to(device).unsqueeze(1).expand_as(
                    output["verifier_logit_sequence"]
                )
                batch_mask = prefix_mask[batch_indices].to(device)
                loss, component = _parent_balanced_loss(
                    output["verifier_logit_sequence"],
                    batch_target,
                    batch_mask,
                    positive_rows[batch_indices].to(device),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27d-r11 loss is non-finite")
                loss.backward()
                gradient_norm = float(
                    nn.utils.clip_grad_norm_(
                        trainable,
                        float(config["training"]["gradient_clip_norm"]),
                        error_if_nonfinite=True,
                    ).item()
                )
                optimizer.step()
                losses.append(float(loss.detach().item()))
                components.append(component)
                gradient_max = max(gradient_max, gradient_norm)
                if (
                    batch_number == 1
                    or batch_number == steps_per_epoch
                    or batch_number % progress_interval(steps_per_epoch) == 0
                ):
                    log_event(
                        logger,
                        "batch",
                        epoch=epoch,
                        batch=f"{batch_number}/{steps_per_epoch}",
                        loss=losses[-1],
                        gradient_norm=gradient_norm,
                    )
            dev_scores = _scores(model, development, r6=False, device=device)
            selection = _selection_from_scores(
                development,
                dev_scores,
                baseline_accepted_positive=baseline_accepted_positive,
                config=config,
            )
            ranking = _ranking(development, dev_scores)
            row = {
                "epoch": epoch,
                "train_loss": sum(losses) / max(1, len(losses)),
                "loss_components": {
                    key: sum(item[key] for item in components) / max(1, len(components))
                    for key in ("bird", "false_positive")
                },
                "gradient_norm_max": gradient_max,
                "development_selection": selection,
                "development_ranking": ranking,
                "unknown_payload_access_count": 0,
                "test_payload_access_count": 0,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            log_metric_set(
                logger,
                "development_knownness",
                {
                    "support": ranking["support"],
                    "auroc": ranking["auroc_higher_is_known"],
                },
            )
            log_event(
                logger,
                "epoch_complete",
                epoch=epoch,
                train_loss=row["train_loss"],
                bird_precision=selection["bird_precision"],
                bird_recall=selection["accepted_bird"] / max(1, selection["bird_support"]),
                false_positive_rejection=selection["false_positive_rejection"],
                elapsed_seconds=row["elapsed_seconds"],
            )
            if best_selection is None or (
                selection["false_positive_rejection"],
                selection["accepted_bird"],
                -selection["threshold"],
            ) > (
                best_selection["false_positive_rejection"],
                best_selection["accepted_bird"],
                -best_selection["threshold"],
            ):
                best_selection = dict(selection)
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        if best_state is None or best_selection is None or best_epoch is None:
            raise RuntimeError("V27d-r11 did not select a checkpoint")
        model.load_state_dict(best_state, strict=True)
        cache_hashes = {
            role: caches[role]["cache_sha256"] for role in R11_ROLES
        }
        checkpoint_path = artifact_path(
            root,
            config["artifacts"]["checkpoint_template"].format(seed=seed),
        )
        save_checkpoint(
            checkpoint_path,
            root=root,
            config=config,
            manifest=manifest,
            model=model,
            seed=seed,
            epoch=best_epoch,
            global_step=best_epoch * steps_per_epoch,
            metrics=best_selection,
            cache_hashes=cache_hashes,
        )
        summary = {
            "schema_version": "RADAR-LN3-V27d-r11-training-summary-v1",
            "experiment_id": R11_EXPERIMENT_ID,
            "status": "completed",
            "seed": int(seed),
            "selected_epoch": best_epoch,
            "checkpoint_path": relative_project_path(root, checkpoint_path),
            "log_path": relative_project_path(root, log_path),
            "direct_baseline": "V27d-r6-liquid_treatment",
            "direct_baseline_checkpoint": relative_project_path(root, r6_checkpoint_path),
            "direct_baseline_development_actions": baseline_actions,
            "selected_development_selection": best_selection,
            "history": history,
            "trainable_parameters": list(trainable_names),
            "frozen_liquid_v2_core": True,
            "frozen_base_readout": True,
            "frozen_verifier_head": True,
            "formal_l3_enabled": False,
            "unknown_payload_access_count": 0,
            "test_payload_access_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary_path = artifact_path(
            root,
            config["artifacts"]["summary_template"].format(seed=seed),
        )
        write_sealed_json_if_identical(summary_path, summary)
        log_event(
            logger,
            "training_complete",
            selected_epoch=best_epoch,
            checkpoint_path=relative_project_path(root, checkpoint_path),
            summary_path=relative_project_path(root, summary_path),
            elapsed_seconds=summary["elapsed_seconds"],
        )
        return {
            "status": "completed",
            "checkpoint_path": relative_project_path(root, checkpoint_path),
            "summary_path": relative_project_path(root, summary_path),
            "log_path": relative_project_path(root, log_path),
            "selected_epoch": best_epoch,
            "development_selection": best_selection,
        }
    finally:
        close_logger(logger)


__all__ = ["_action_metrics", "_inputs", "_load_r6_reference", "_r6_inputs", "_scores", "train_seed"]