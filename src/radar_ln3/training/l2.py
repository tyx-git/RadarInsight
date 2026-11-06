"""L2 正式训练流程。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from radar_ln3.checkpoint import Checkpoint, canonical_hash, sha256_file
from radar_ln3.cache import load_event_logit_anchor
from radar_ln3.data import load_manifest
from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.data.derived import register_product_artifacts
from radar_ln3.data.manifest import data_manifest_hash
from radar_ln3.features import FeatureCache
from radar_ln3.data.contracts import RawEchoProvider
from radar_ln3.model import build_l2_model, load_model_config
from radar_ln3.model.l2 import trainable_parameter_groups
from radar_ln3.logger import log_classification_metrics

from .common import (
    ModelBatchCollator,
    _l2_inputs,
    _l2_reference_inputs,
    build_training_dataset,
    build_training_loader,
    checkpoint_payload,
    evaluate_l2,
    evaluate_l2_direct,
    train_l2_step,
    train_l2_direct_step,
)


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value) % (2**32))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前环境不可用")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"不支持的训练 device：{requested}")


def _emit(
    progress_callback: ProgressCallback | None,
    event: Mapping[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback(event)
    if "validation" in event:
        log_classification_metrics(
            event["validation"], task="l2", epoch=event["epoch"]
        )


def _cached_stable8_valid_event_count(
    feature_caches: Mapping[str, FeatureCache] | None,
) -> int | None:
    """返回 train cache 中有效 stable8 事件数，并检查 summary/frame 对齐。"""

    if feature_caches is None:
        return None
    summary = feature_caches.get("micro_doppler_stable8")
    frames = feature_caches.get("micro_doppler_stable8_frame")
    if summary is None and frames is None:
        return None
    if summary is None or frames is None:
        raise ValueError("L2 stable8 summary 和 frame cache 必须成对存在")
    if summary.sample_ids != frames.sample_ids:
        raise ValueError("L2 stable8 summary 和 frame cache 的 sample_id 未对齐")
    summary_valid = np.asarray(summary.valid, dtype=np.bool_)
    frame_valid = np.asarray(frames.valid, dtype=np.bool_)
    event_count = len(summary.sample_ids)
    if summary_valid.shape != (event_count,) or frame_valid.ndim != 2:
        raise ValueError("L2 stable8 cache validity shape 无效")
    if frame_valid.shape[0] != event_count:
        raise ValueError("L2 stable8 summary 和 frame cache 数量不一致")
    if not np.array_equal(summary_valid, frame_valid.any(axis=-1)):
        raise ValueError("L2 stable8 summary validity 与 frame validity 不一致")
    return int(summary_valid.sum())


def _l2_checkpoint_rank(
    metrics: Mapping[str, Any],
    *,
    epoch: int,
    class_order: tuple[str, ...],
    bird_precision_floor: float | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    expected_classes = ("DroneTarget", "BirdTarget", "BalloonTarget")
    if class_order != expected_classes:
        raise ValueError("L2 checkpoint selection 的类别顺序不完整")
    rows_value = metrics.get("per_class")
    if not isinstance(rows_value, list):
        raise ValueError("L2 checkpoint selection 缺少 per_class 指标")
    rows = {str(row["class"]): row for row in rows_value}
    if set(rows) != set(expected_classes):
        raise ValueError("L2 checkpoint selection 的类别集合不完整")
    confusion = np.asarray(metrics.get("confusion_matrix"), dtype=np.int64)
    if confusion.shape != (3, 3):
        raise ValueError("L2 checkpoint selection 的 confusion matrix shape 无效")
    class_index = {name: index for index, name in enumerate(class_order)}
    drone_bird_confusions = int(
        confusion[class_index["DroneTarget"], class_index["BirdTarget"]]
        + confusion[class_index["BirdTarget"], class_index["DroneTarget"]]
    )
    minimum_class_f1 = min(float(rows[name]["f1"]) for name in expected_classes)
    macro_f1 = float(metrics["macro_f1"])
    bird_precision = float(rows["BirdTarget"]["precision"])
    bird_recall = float(rows["BirdTarget"]["recall"])
    if bird_precision_floor is None:
        rank = (
            minimum_class_f1,
            -float(drone_bird_confusions),
            macro_f1,
            -float(epoch),
        )
    else:
        if not 0.0 <= float(bird_precision_floor) <= 1.0:
            raise ValueError("L2 Bird precision floor 必须位于 [0,1]")
        rank = (
            float(bird_precision >= float(bird_precision_floor)),
            bird_precision,
            minimum_class_f1,
            -float(drone_bird_confusions),
            macro_f1,
            bird_recall,
            -float(epoch),
        )
    selection = {
        "minimum_class_f1": minimum_class_f1,
        "drone_bird_confusions": drone_bird_confusions,
        "macro_f1": macro_f1,
        "epoch": int(epoch),
    }
    if bird_precision_floor is not None:
        selection.update({
            "bird_precision": bird_precision,
            "bird_recall": bird_recall,
            "bird_precision_floor": float(bird_precision_floor),
            "bird_precision_floor_met": bool(
                bird_precision >= float(bird_precision_floor)
            ),
        })
    return rank, selection


def load_l2_base_model(
    project_root: str | Path,
    *,
    device: torch.device | None = None,
    arm: str = "robust_control",
    mode: str = "residual",
    config_path: str | Path = "src/config/model.json",
    artifact_root: str | Path | None = None,
) -> tuple[Any, Any]:
    """加载 l2base1.pt 与 l2base2.pt，构造正式 L2 模型。"""

    root = Path(project_root).resolve()
    config = load_model_config(root, config_path, artifact_root=artifact_root)
    base = Checkpoint.load(root, config.artifacts.l2_base_weight_path)
    stable = Checkpoint.load(root, config.artifacts.l2_stable_weight_path)
    model = build_l2_model(
        config,
        arm=arm,
        mode=mode,
        base_state=base.state,
        stable_state=stable.state,
        seed=config.training.seed,
    )
    if device is not None:
        model = model.to(device)
    return config, model


def train_l2(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    device: str = "cpu",
    arm: str = "robust_control",
    manifest: DatasetManifest | None = None,
    progress_callback: ProgressCallback | None = None,
    raw_echo_provider: RawEchoProvider | None = None,
    train_feature_caches: Mapping[str, FeatureCache] | None = None,
    validation_feature_caches: Mapping[str, FeatureCache] | None = None,
    validate_caches: bool = True,
    mode: str = "residual",
    progress_interval: int = 25,
    resume: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    bird_precision_floor: float | None = None,
    config_path: str | Path = "src/config/model.json",
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """执行正式 L2 训练并保存 epoch/final checkpoint。"""

    root = Path(project_root).resolve()
    if isinstance(progress_interval, bool) or int(progress_interval) <= 0:
        raise ValueError("progress_interval 必须是正整数")
    config = load_model_config(root, config_path, artifact_root=artifact_root)
    run_seed = int(config.training.seed)
    _seed(run_seed)
    cache_names = set(train_feature_caches or {})
    validation_cache_names = set(validation_feature_caches or {})
    required_cache_names = {
        "micro_doppler_stable8",
        "micro_doppler_stable8_frame",
    }
    if raw_echo_provider is None and (
        not required_cache_names.issubset(cache_names)
        or not required_cache_names.issubset(validation_cache_names)
    ):
        raise ValueError("正式 L2 训练必须注入 raw_echo_provider 以生成 stable8")
    if mode not in {"residual", "direct", "direct_grouped"}:
        raise ValueError("L2 training mode 无效")
    if bird_precision_floor is not None and (
        not np.isfinite(float(bird_precision_floor))
        or not 0.0 <= float(bird_precision_floor) <= 1.0
    ):
        raise ValueError("L2 Bird precision floor 必须是 [0,1] 内的有限值")
    manifest = manifest or load_manifest(
        manifest_path, root, verify_files=True, allow_unmaterialized=False
    )
    source_manifest_hash = data_manifest_hash(manifest)
    baseline_anchor = (
        None
        if mode in {"direct", "direct_grouped"}
        else load_event_logit_anchor(root, config.artifacts.l2_baseline_anchor_path)
    )
    train_set = build_training_dataset(
        root,
        task="l2",
        split_name="train",
        manifest_path=manifest_path,
        manifest=manifest,
        validate_event_cache=validate_caches,
    )
    cached_stable8_valid_events = _cached_stable8_valid_event_count(
        train_feature_caches
    )
    if (
        mode == "residual"
        and cached_stable8_valid_events is not None
        and cached_stable8_valid_events == 0
    ):
        raise ValueError(
            "L2 train feature cache 全部没有有效 stable8 证据；"
            "请检查 raw-IQ 输入或重建 stable8 cache"
        )
    validation_set = build_training_dataset(
        root,
        task="l2",
        split_name="validation",
        manifest_path=manifest_path,
        manifest=manifest,
        validate_event_cache=validate_caches,
    )
    if baseline_anchor is not None:
        expected_anchor_ids = set(train_set.view.sample_ids) | set(
            validation_set.view.sample_ids
        )
        actual_anchor_ids = set(baseline_anchor.sample_ids)
        if actual_anchor_ids != expected_anchor_ids:
            raise ValueError(
                "L2 baseline anchor 与 train/validation view 覆盖范围不一致："
                f"missing={sorted(expected_anchor_ids - actual_anchor_ids)[:3]}, "
                f"extra={sorted(actual_anchor_ids - expected_anchor_ids)[:3]}"
            )
        if any(split_name == "test" for split_name in baseline_anchor.split_names):
            raise ValueError("L2 baseline anchor 不得包含 test 事件")
        if baseline_anchor.payload.get("source_manifest_hash") != source_manifest_hash:
            raise ValueError("L2 baseline anchor source manifest hash 不匹配")
    effective_batch_size = int(batch_size or config.training.batch_size)
    if effective_batch_size <= 0:
        raise ValueError("L2 batch_size 必须为正整数")
    target_device = _device(device)
    effective_workers = (
        (2 if target_device.type == "cuda" else 0)
        if num_workers is None
        else int(num_workers)
    )
    if effective_workers < 0:
        raise ValueError("num_workers 必须是非负整数")
    effective_pin_memory = (
        target_device.type == "cuda" if pin_memory is None else bool(pin_memory)
    )
    clipping = {
        "bounds": {
            name: {"lower": bounds[0], "upper": bounds[1]}
            for name, bounds in config.data.l2.tr_clipping_bounds.items()
        }
    }
    collator_args = {
        "task": "l2",
        "max_rd_frames": config.data.max_rd_frames_per_point,
        "max_sequence_length": config.data.max_sequence_length,
        "stable_indices": config.data.l2.stable8_indices,
        "clipping": clipping,
        "raw_echo_provider": raw_echo_provider,
        "stable_median": config.data.l2.stable8_median,
        "stable_scale": config.data.l2.stable8_scale,
        "stable_input_mode": (
            "raw" if mode in {"direct", "direct_grouped"} else "normalized"
        ),
    }
    train_loader = build_training_loader(
        train_set,
        collate_fn=ModelBatchCollator(
            **collator_args,
            feature_caches=train_feature_caches,
            expected_view_hash=train_set.view.view_hash,
            baseline_anchor=baseline_anchor,
        ),
        batch_size=effective_batch_size,
        shuffle=True,
        seed=config.training.dataloader_seed + run_seed,
        num_workers=effective_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=effective_pin_memory,
    )
    validation_loader = build_training_loader(
        validation_set,
        collate_fn=ModelBatchCollator(
            **collator_args,
            feature_caches=validation_feature_caches,
            expected_view_hash=validation_set.view.view_hash,
            baseline_anchor=baseline_anchor,
        ),
        batch_size=effective_batch_size,
        shuffle=False,
        seed=config.training.dataloader_seed + run_seed,
        num_workers=effective_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=effective_pin_memory,
    )
    _, model = load_l2_base_model(
        root, device=target_device, arm=arm, mode=mode, config_path=config_path,
        artifact_root=artifact_root,
    )
    groups = trainable_parameter_groups(model)
    if mode == "direct":
        optimizer = torch.optim.AdamW(
            [{"params": groups["base"], "lr": config.training.learning_rate}],
            weight_decay=config.training.weight_decay,
        )
    elif mode == "direct_grouped":
        pretrained_encoder = []
        newly_initialized = []
        for name, parameter in model.trainable_base_model.named_parameters():
            if not parameter.requires_grad or name.startswith("trajectory_heads."):
                continue
            if name.startswith(("tr_encoder.", "rd_encoder.")):
                pretrained_encoder.append(parameter)
            else:
                newly_initialized.append(parameter)
        if not pretrained_encoder or not newly_initialized:
            raise ValueError("direct_grouped 参数分组不完整")
        optimizer = torch.optim.AdamW(
            [
                {"params": pretrained_encoder, "lr": config.training.base_learning_rate},
                {"params": newly_initialized, "lr": config.training.learning_rate},
            ],
            weight_decay=config.training.l2_weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": groups["base"], "lr": config.training.base_learning_rate},
                {"params": groups["stable"], "lr": config.training.stable_learning_rate},
                {"params": groups["motion"], "lr": config.training.motion_learning_rate},
            ],
            weight_decay=config.training.l2_weight_decay,
        )
    class_weights = torch.tensor(config.training.class_weights["l2"], dtype=torch.float32, device=target_device)
    output_path_for_mode = (
        config.artifacts.l2_direct_grouped_output_path
        if mode == "direct_grouped"
        else config.artifacts.l2_direct_output_path
        if mode == "direct"
        else config.artifacts.l2_output_path
    )
    # epoch checkpoint 必须和 final checkpoint 使用同一个配置路径的 parent。
    # 临时实验配置可以因此完整隔离到 tmp/L2，正式配置仍保持原目录。
    output_root = Path(root / output_path_for_mode).parent
    epoch_root = output_root / "epochs"
    history: list[dict[str, Any]] = []
    best_rank: tuple[float, ...] | None = None
    best_payload: dict[str, Any] | None = None
    start_epoch = 1
    resumed_from_epoch: int | None = None
    if resume:
        checkpoint_paths = sorted(
            epoch_root.glob("epoch-*.pt"),
            key=lambda path: int(path.stem.removeprefix("epoch-")),
        )
        if not checkpoint_paths:
            raise ValueError(f"L2 resume 找不到 epoch checkpoint：{epoch_root}")
        expected_epochs = list(range(1, len(checkpoint_paths) + 1))
        actual_epochs = [
            int(path.stem.removeprefix("epoch-")) for path in checkpoint_paths
        ]
        if actual_epochs != expected_epochs:
            raise ValueError(
                "L2 resume epoch checkpoint 不连续："
                f"expected={expected_epochs[:3]}..{expected_epochs[-1:]}, "
                f"actual={actual_epochs[:3]}..{actual_epochs[-1:]}"
            )
        current_config_hash = sha256_file(config.path)
        expected_contract_hash = canonical_hash(model.model_contract())
        latest_checkpoint: Checkpoint | None = None
        class_order = tuple(str(name) for name in config.class_names["l2"])
        for checkpoint_path in checkpoint_paths:
            checkpoint = Checkpoint.load(
                root,
                checkpoint_path,
                expected_schema="RADAR-LN3-CHECKPOINT-V1",
                expected_task="l2",
            )
            payload = checkpoint.payload
            epoch = int(checkpoint_path.stem.removeprefix("epoch-"))
            if (
                int(payload.get("epoch", -1)) != epoch
                or payload.get("config_sha256") != current_config_hash
                or payload.get("manifest_sha256") != source_manifest_hash
                or payload.get("model_contract_sha256") != expected_contract_hash
            ):
                raise ValueError(
                    f"L2 resume checkpoint 绑定与当前训练不一致：{checkpoint_path}"
                )
            contract = payload.get("model_contract")
            if not isinstance(contract, Mapping) or contract.get("mode") != mode:
                raise ValueError(
                    f"L2 resume checkpoint mode 与当前训练不一致：{checkpoint_path}"
                )
            record = payload.get("metrics")
            if not isinstance(record, Mapping) or int(record.get("epoch", -1)) != epoch:
                raise ValueError(
                    f"L2 resume checkpoint 缺少 epoch 指标：{checkpoint_path}"
                )
            validation = record.get("validation")
            if not isinstance(validation, Mapping):
                raise ValueError(
                    f"L2 resume checkpoint 缺少 validation 指标：{checkpoint_path}"
                )
            rank, selection = _l2_checkpoint_rank(
                validation,
                epoch=epoch,
                class_order=class_order,
                bird_precision_floor=bird_precision_floor,
            )
            if record.get("checkpoint_selection_metrics") != selection:
                raise ValueError(
                    f"L2 resume checkpoint selection 指标不一致：{checkpoint_path}"
                )
            history.append(dict(record))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_payload = dict(payload)
            latest_checkpoint = checkpoint
        assert latest_checkpoint is not None
        model.load_state_dict(latest_checkpoint.state, strict=True)
        resumed_from_epoch = actual_epochs[-1]
        start_epoch = resumed_from_epoch + 1
    started = time.perf_counter()
    start_event = {
        "stage": "l2_start",
        "train_parents": len(train_set),
        "validation_parents": len(validation_set),
        "num_workers": effective_workers,
        "prefetch_factor": int(prefetch_factor) if effective_workers else 0,
        "pin_memory": effective_pin_memory,
    }
    if cached_stable8_valid_events is not None:
        start_event["stable8_valid_train_events"] = cached_stable8_valid_events
    start_event["start_epoch"] = start_epoch
    if resumed_from_epoch is not None:
        start_event["resumed_from_epoch"] = resumed_from_epoch
    _emit(progress_callback, start_event)
    baseline_reference_verified = False
    for epoch in range(start_epoch, config.training.epochs + 1):
        model.train()
        model.frozen_reference_base_model.eval()
        model.trainable_base_model.trajectory_heads.eval()
        losses: list[torch.Tensor] = []
        loss_component_totals: dict[str, torch.Tensor] = {}
        prefix_event_count = 0
        epoch_stable8_valid_events = 0
        loader_iterator = iter(train_loader)
        for step in range(1, len(train_loader) + 1):
            wait_started = time.perf_counter()
            batch = next(loader_iterator)
            batch_wait_seconds = time.perf_counter() - wait_started
            compute_started = time.perf_counter()
            if mode == "residual" and not baseline_reference_verified:
                inputs = _l2_inputs(batch, target_device)
                reference_inputs = _l2_reference_inputs(
                    batch, target_device, base_inputs=inputs
                )
                anchor = batch["baseline_logits"].to(
                    device=target_device,
                    non_blocking=target_device.type == "cuda",
                )
                with torch.inference_mode():
                    reference_logits = model.frozen_reference_logits(
                        base_inputs=reference_inputs
                    )
                    active = ~inputs["tr_padding_mask"]
                    max_error = float(
                        (reference_logits - anchor)
                        .abs()
                        .masked_select(active.unsqueeze(-1))
                        .max()
                        .item()
                    )
                if max_error > 1.0e-5:
                    raise ValueError(
                        "L2 baseline anchor 与当前 frozen reference 不一致："
                        f"max_abs_error={max_error:.8g}，请重建 baseline anchor"
                    )
                baseline_reference_verified = True
                _emit(
                    progress_callback,
                    {
                        "stage": "l2_anchor_verified",
                        "max_abs_error": max_error,
                        "tolerance": 1.0e-5,
                    },
                )
            if mode == "residual":
                epoch_stable8_valid_events += int(
                    batch["stable_summary_valid"].sum().item()
                )
            if mode in {"direct", "direct_grouped"}:
                loss = train_l2_direct_step(
                    model, batch, device=target_device,
                    class_weights=class_weights, optimizer=optimizer,
                    gradient_clip_norm=config.training.gradient_clip_norm,
                    return_loss_tensor=True,
                )
                assert isinstance(loss, torch.Tensor)
                loss_components: dict[str, torch.Tensor | int] = {
                    "final_state_loss": loss.detach(),
                    "prefix_auxiliary_loss": loss.detach().new_zeros(()),
                    "total_loss": loss.detach(),
                    "prefix_event_count": 0,
                }
            else:
                step_loss = train_l2_step(
                    model, batch, device=target_device,
                    class_weights=class_weights, optimizer=optimizer,
                    gradient_clip_norm=config.training.gradient_clip_norm,
                    prefix_weight=config.training.l2_prefix_loss_weight,
                    prefix_events=config.training.l2_prefix_loss_events,
                    return_loss_components=True,
                    baseline_reference_verified=baseline_reference_verified,
                )
                assert isinstance(step_loss, tuple)
                loss, loss_components = step_loss
            prefix_event_count = int(loss_components["prefix_event_count"])
            for name in ("final_state_loss", "prefix_auxiliary_loss", "total_loss"):
                value = loss_components[name]
                assert isinstance(value, torch.Tensor)
                loss_component_totals[name] = (
                    loss_component_totals.get(name, torch.zeros_like(value)) + value
                )
            losses.append(loss)
            if step % int(progress_interval) == 0 or step == len(train_loader):
                _emit(
                    progress_callback,
                    {
                        "stage": "l2_step",
                        "epoch": epoch,
                        "step": step,
                        "total": len(train_loader),
                        "loss": float(loss.item()),
                        "batch_wait_seconds": batch_wait_seconds,
                        "step_compute_seconds": time.perf_counter() - compute_started,
                        "loss_components": {
                            name: float(value.item())
                            for name, value in loss_components.items()
                            if isinstance(value, torch.Tensor)
                        },
                        "prefix_event_count": prefix_event_count,
                    },
                )
        if mode == "residual" and epoch_stable8_valid_events == 0:
            raise ValueError(
                "L2 当前 epoch 的 train batch 全部没有有效 stable8 证据；"
                "请检查 raw-IQ 输入或 feature cache 对齐"
            )
        if mode in {"direct", "direct_grouped"}:
            metrics = evaluate_l2_direct(
                model, validation_loader, device=target_device,
            )
        else:
            metrics = evaluate_l2(
                model, validation_loader, device=target_device,
                baseline_reference_verified=baseline_reference_verified,
            )
        record = {
            "epoch": epoch,
            "train_loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
            "train_loss_components": {
                name: float((value / max(1, len(losses))).item())
                for name, value in loss_component_totals.items()
            },
            "prefix_loss": {
                "weight": float(config.training.l2_prefix_loss_weight),
                "last_valid_events_excluding_final": int(
                    config.training.l2_prefix_loss_events
                ),
            },
            "validation": metrics,
        }
        if (
            config.training.l2_checkpoint_selection
            != "minimum_class_f1_drone_bird_confusion_macro_f1"
        ):
            raise ValueError(
                "不支持的 L2 checkpoint selection："
                f"{config.training.l2_checkpoint_selection}"
            )
        class_order = tuple(str(name) for name in config.class_names["l2"])
        rank, selection_metrics = _l2_checkpoint_rank(
            metrics,
            epoch=epoch,
            class_order=class_order,
            bird_precision_floor=bird_precision_floor,
        )
        record["checkpoint_selection_metrics"] = selection_metrics
        history.append(record)
        payload = checkpoint_payload(
            model,
            config=config,
            manifest_sha256=source_manifest_hash,
            task="l2",
            epoch=epoch,
            metrics=record,
        )
        Checkpoint.save(root, epoch_root / f"epoch-{epoch:03d}.pt", payload)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = payload
        _emit(progress_callback, {"stage": "l2_epoch", **record, "elapsed_seconds": time.perf_counter() - started})
    if best_payload is None:
        raise RuntimeError("L2 训练没有产生 checkpoint")
    if bird_precision_floor is not None:
        selected_metrics = best_payload.get("metrics", {}).get("checkpoint_selection_metrics", {})
        if not bool(selected_metrics.get("bird_precision_floor_met")):
            best_precision = float(selected_metrics.get("bird_precision", 0.0))
            raise ValueError(
                "L2 validation Bird precision 未达到最低要求："
                f"best={best_precision:.6f}, required={float(bird_precision_floor):.6f}；"
                "候选仅保留在 epoch checkpoints，未生成可用 final checkpoint。"
            )
    best_payload["selection"] = config.training.l2_checkpoint_selection
    if bird_precision_floor is not None:
        best_payload["selection_constraints"] = {
            "bird_precision_floor": float(bird_precision_floor),
            "validation_bird_precision": float(
                best_payload["metrics"]["checkpoint_selection_metrics"]["bird_precision"]
            ),
        }
    output_path = (
        config.artifacts.l2_direct_grouped_output_path
        if mode == "direct_grouped"
        else config.artifacts.l2_direct_output_path
        if mode == "direct"
        else config.artifacts.l2_output_path
    )
    final_checkpoint, _ = Checkpoint.save(root, output_path, best_payload)
    registered_manifest = None
    if mode == "residual" and (
        artifact_root is None
        or Path(artifact_root).as_posix().rstrip("/") == "artifacts/radar_ln3"
    ):
        registered_manifest = register_product_artifacts(
            root,
            manifest_path=manifest_path,
            manifest=manifest,
            overwrite=True,
            artifact_root="artifacts/radar_ln3",
        )
    summary = {
        "schema_version": "RADAR-LN3-TRAINING-SUMMARY-V1",
        "task": "l2",
        "arm": arm,
        "mode": mode,
        "selected_epoch": int(best_payload["epoch"]),
        "train_parent_count": len(train_set),
        "validation_parent_count": len(validation_set),
        "batch_size": effective_batch_size,
        "data_loader": {
            "num_workers": effective_workers,
            "prefetch_factor": int(prefetch_factor) if effective_workers else 0,
            "pin_memory": effective_pin_memory,
        },
        "history": history,
        "checkpoint": final_checkpoint.provenance(kind="final", task="l2"),
        "manifest_sha256": source_manifest_hash,
        "dataloader_seed": config.training.dataloader_seed + run_seed,
        "optimizer_groups": {
            "base_learning_rate": config.training.base_learning_rate if mode == "direct_grouped" else config.training.learning_rate if mode == "direct" else config.training.base_learning_rate,
            "stable_learning_rate": None if mode in {"direct", "direct_grouped"} else config.training.stable_learning_rate,
            "motion_learning_rate": None if mode in {"direct", "direct_grouped"} else config.training.motion_learning_rate,
            "prefix_loss_weight": config.training.l2_prefix_loss_weight if mode == "residual" else 0.0,
            "prefix_loss_events": config.training.l2_prefix_loss_events if mode == "residual" else 0,
            "weight_decay": config.training.l2_weight_decay if mode == "direct_grouped" else config.training.weight_decay if mode == "direct" else config.training.l2_weight_decay,
        },
        "baseline_anchor_path": config.artifacts.l2_baseline_anchor_path,
        "checkpoint_selection": config.training.l2_checkpoint_selection,
        "selection_constraints": (
            None
            if bird_precision_floor is None
            else {
                "bird_precision_floor": float(bird_precision_floor),
                "validation_bird_precision": float(
                    best_payload["metrics"]["checkpoint_selection_metrics"]["bird_precision"]
                ),
            }
        ),
        "resume_mode": (
            "warm_model_state_optimizer_and_sampler_reset"
            if resume
            else "fresh"
        ),
        "resumed_from_epoch": resumed_from_epoch,
        "manifest_path": str(registered_manifest or (root / manifest_path).resolve()),
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _emit(progress_callback, {"stage": "l2_complete", **summary["checkpoint"], "selected_epoch": summary["selected_epoch"]})
    return summary


__all__ = ["ProgressCallback", "load_l2_base_model", "train_l2"]
