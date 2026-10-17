"""Audit-only gradient telemetry for the V25a KnownGate replication.

The recorder is deliberately external to the model, loss, optimiser and data
pipeline.  It observes gradients after ``loss.backward()`` and before the
existing global-norm clipping call, then writes one immutable JSON payload per
completed epoch.  Therefore it cannot become a model input or alter an update.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


V25A_GRADIENT_TELEMETRY_SCHEMA = (
    "RADAR-LN3-V25a-known-gate-preclip-gradient-batch-telemetry-v1"
)
V25A_GRADIENT_TELEMETRY_DIRECTORY = "gradient-telemetry"
V25A_GRADIENT_MODULE_NAMES = (
    "tr_encoder",
    "rd_encoder",
    "fusion",
    "liquid",
    "classifier",
)
V25A_BASE_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
)
V25A_TASK_LABELS = (
    "KNOWN_TARGET_EVIDENCE",
    "KNOWN_CLUTTER_EVIDENCE",
)
V25A_RECORD_FIELDS = frozenset(
    {
        "epoch",
        "step",
        "global_step",
        "parent_count",
        "parent_batch_sha256",
        "task_label_counts",
        "base_label_counts",
        "task_label_signature",
        "base_label_signature",
        "contains_clutter_parent",
        "batch_observation_counts",
        "loss",
        "pre_clip_module_l2",
        "pre_clip_module_squared_fraction",
        "pre_clip_module_nonzero_gradient",
        "pre_clip_module_total_l2",
        "pre_clip_global_l2",
        "gradient_clip_threshold",
        "was_globally_clipped",
        "global_clip_scale",
        "post_clip_global_l2_upper_bound",
        "module_vs_global_l2_absolute_difference",
        "globally_affected_nonzero_modules",
    }
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> None:
    content = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite a different V25a telemetry artifact: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _count_names(values: Sequence[str], *, allowed: Sequence[str]) -> dict[str, int]:
    observed = [str(value) for value in values]
    invalid = sorted(set(observed) - set(allowed))
    if invalid:
        raise ValueError(f"V25a telemetry received unsupported label names: {invalid}")
    return {name: observed.count(name) for name in allowed}


def _signature(values: Sequence[str]) -> str:
    return "+".join(sorted(str(value) for value in values))


def _as_finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V25a telemetry {name} is non-finite")
    return result


def partition_gradient_modules(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Partition every trainable parameter into one product-model module.

    The explicit partition is a contract check: a newly introduced trainable
    parameter must be assigned deliberately instead of being silently omitted
    from the diagnostic report.
    """

    groups = {name: [] for name in V25A_GRADIENT_MODULE_NAMES}
    unknown: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"V25a requires full parameter training, found frozen {name}")
        matched = [
            module
            for module in V25A_GRADIENT_MODULE_NAMES
            if name.startswith(module + ".")
        ]
        if len(matched) != 1:
            unknown.append(name)
            continue
        groups[matched[0]].append((name, parameter))
    if unknown or any(not values for values in groups.values()):
        raise ValueError(
            "V25a gradient module partition is incomplete: "
            f"unknown={unknown}, empty={[name for name, values in groups.items() if not values]}"
        )
    all_names = [name for values in groups.values() for name, _ in values]
    expected = [name for name, _ in model.named_parameters()]
    if sorted(all_names) != sorted(expected) or len(all_names) != len(set(all_names)):
        raise AssertionError("V25a gradient parameter partition does not cover each parameter once")
    return groups


def gradient_module_metadata(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> dict[str, Any]:
    if tuple(groups) != V25A_GRADIENT_MODULE_NAMES:
        raise ValueError("V25a gradient module order differs from the frozen protocol")
    return {
        name: {
            "parameter_tensor_count": len(values),
            "parameter_element_count": int(sum(parameter.numel() for _, parameter in values)),
            "parameter_names": [parameter_name for parameter_name, _ in values],
        }
        for name, values in groups.items()
    }


def capture_preclip_gradient_snapshot(
    *,
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
    batch: Mapping[str, Any],
    tensors: Mapping[str, Any],
    objective: Mapping[str, torch.Tensor],
    epoch: int,
    step: int,
    global_step: int,
) -> dict[str, Any]:
    """Capture a V25a record after backpropagation and before clipping.

    No gradient is modified here.  The one host transfer of the scalar vector
    is intentionally deferred until all per-module reductions have completed.
    """

    if tuple(groups) != V25A_GRADIENT_MODULE_NAMES:
        raise ValueError("V25a gradient groups differ from the frozen protocol")
    parent_ids = [str(value) for value in list(batch.get("parent_sample_ids") or ())]
    task_names = [str(value) for value in list(batch.get("task_label_names") or ())]
    base_names = [str(value) for value in list(batch.get("base_label_names") or ())]
    if (
        not parent_ids
        or len(parent_ids) != len(set(parent_ids))
        or len(parent_ids) != len(task_names)
        or len(parent_ids) != len(base_names)
    ):
        raise ValueError("V25a telemetry batch parent/label metadata is invalid")
    labels = tensors.get("labels")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.shape[0] != len(parent_ids):
        raise ValueError("V25a telemetry batch task labels are invalid")
    expected_ids = [0 if name == V25A_TASK_LABELS[0] else 1 for name in task_names]
    if labels.detach().cpu().tolist() != expected_ids:
        raise AssertionError("V25a telemetry task-label names differ from tensor labels")

    active = ~tensors["tr_padding_mask"]
    reasons = tensors["tr_dt_reason"]
    if active.shape != reasons.shape:
        raise ValueError("V25a telemetry dt/padding shapes differ")
    rd_present = tensors["rd_frame_mask"].any(dim=2) & active
    course_valid = tensors["known_gate_feature_valid"][..., 0] & active

    squares: list[torch.Tensor] = []
    finite_flags: list[torch.Tensor] = []
    nonzero_counts: list[torch.Tensor] = []
    for module in V25A_GRADIENT_MODULE_NAMES:
        values = groups[module]
        first_parameter = values[0][1]
        square = torch.zeros((), dtype=torch.float32, device=first_parameter.device)
        finite = torch.ones((), dtype=torch.bool, device=first_parameter.device)
        nonzero_count = torch.zeros((), dtype=torch.int64, device=first_parameter.device)
        for _, parameter in values:
            gradient = parameter.grad
            if gradient is None:
                continue
            detached = gradient.detach().to(dtype=torch.float32)
            finite = finite & torch.isfinite(detached).all()
            square = square + torch.sum(detached.square())
            nonzero_count = nonzero_count + torch.count_nonzero(detached)
        squares.append(square)
        finite_flags.append(finite)
        nonzero_counts.append(nonzero_count)
    square_vector = torch.stack(squares)
    finite_vector = torch.stack(finite_flags)
    nonzero_vector = torch.stack(nonzero_counts)
    scalar_diagnostics = torch.cat(
        [
            square_vector.detach().cpu(),
            finite_vector.to(dtype=torch.int8).detach().cpu(),
            nonzero_vector.detach().cpu(),
        ]
    )
    square_values = scalar_diagnostics[: len(V25A_GRADIENT_MODULE_NAMES)].tolist()
    finite_values = scalar_diagnostics[
        len(V25A_GRADIENT_MODULE_NAMES) : 2 * len(V25A_GRADIENT_MODULE_NAMES)
    ].tolist()
    nonzero_values = scalar_diagnostics[2 * len(V25A_GRADIENT_MODULE_NAMES) :].tolist()
    if (
        not all(bool(value) for value in finite_values)
        or not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in square_values)
    ):
        raise ValueError("V25a pre-clip module gradient squares are invalid")
    norm_vector = torch.sqrt(square_vector)
    total_square = torch.sum(square_vector)
    total_norm = torch.sqrt(total_square)
    fractions = (
        square_vector / total_square
        if float(total_square.detach().item()) > 0.0
        else torch.zeros_like(square_vector)
    )
    module_norms = {
        module: _as_finite_float(value, name=f"{module} pre-clip norm")
        for module, value in zip(
            V25A_GRADIENT_MODULE_NAMES, norm_vector.detach().cpu().tolist()
        )
    }
    module_fractions = {
        module: _as_finite_float(value, name=f"{module} squared fraction")
        for module, value in zip(V25A_GRADIENT_MODULE_NAMES, fractions.detach().cpu().tolist())
    }
    if abs(sum(module_fractions.values()) - (1.0 if total_norm.item() > 0.0 else 0.0)) > 1.0e-5:
        raise AssertionError("V25a module squared-gradient fractions do not sum correctly")

    task_counts = _count_names(task_names, allowed=V25A_TASK_LABELS)
    base_counts = _count_names(base_names, allowed=V25A_BASE_LABELS)
    losses = {
        "total": _as_finite_float(objective["loss"].detach().item(), name="total loss"),
        "final_state": _as_finite_float(
            objective["final_state_loss"].detach().item(), name="final-state loss"
        ),
        "prefix_auxiliary": _as_finite_float(
            objective["prefix_auxiliary_loss"].detach().item(), name="prefix loss"
        ),
    }
    return {
        "epoch": int(epoch),
        "step": int(step),
        "global_step": int(global_step),
        "parent_count": len(parent_ids),
        "parent_batch_sha256": _sha256_text(
            json.dumps(parent_ids, ensure_ascii=False, separators=(",", ":"))
        ),
        "task_label_counts": task_counts,
        "base_label_counts": base_counts,
        "task_label_signature": _signature(task_names),
        "base_label_signature": _signature(base_names),
        "contains_clutter_parent": bool(base_counts["ClutterTarget"] > 0),
        "batch_observation_counts": {
            "active_event_count": int(active.sum().item()),
            "zero_delta_event_count": int((active & (reasons == 2)).sum().item()),
            "positive_delta_event_count": int((active & (reasons == 3)).sum().item()),
            "rd_present_event_count": int(rd_present.sum().item()),
            "course_feature_valid_event_count": int(course_valid.sum().item()),
        },
        "loss": losses,
        "pre_clip_module_l2": module_norms,
        "pre_clip_module_squared_fraction": module_fractions,
        "pre_clip_module_nonzero_gradient": {
            module: bool(value)
            for module, value in zip(V25A_GRADIENT_MODULE_NAMES, nonzero_values)
        },
        "pre_clip_module_total_l2": _as_finite_float(
            total_norm.detach().cpu().item(), name="module total gradient norm"
        ),
    }


class V25AGradientTelemetryRecorder:
    """Write and validate immutable per-epoch V25a telemetry payloads."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        config_path: Path,
        config_file_sha256: str,
        config_identity_sha256: str,
        source_summary_path: Path,
        source_summary_sha256: str,
        expected_parent_order_hashes: Mapping[int, str],
    ) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.config_path = config_path.resolve()
        self.config_file_sha256 = str(config_file_sha256)
        self.config_identity_sha256 = str(config_identity_sha256)
        self.source_summary_path = source_summary_path.resolve()
        self.source_summary_sha256 = str(source_summary_sha256)
        self.expected_parent_order_hashes = {
            int(epoch): str(value) for epoch, value in expected_parent_order_hashes.items()
        }
        self.directory = self.artifact_dir / V25A_GRADIENT_TELEMETRY_DIRECTORY
        self.metadata_path = self.directory / "metadata.json"
        self._groups: dict[str, list[tuple[str, nn.Parameter]]] | None = None
        self._records: list[dict[str, Any]] = []
        self._steps_per_epoch: int | None = None
        self._initial_model_state_sha256: str | None = None
        self._completed_epochs: set[int] = set()

    @staticmethod
    def epoch_path(directory: Path, epoch: int) -> Path:
        return directory / f"epoch-{int(epoch):03d}.json"

    def _metadata(self) -> dict[str, Any]:
        if self._groups is None or self._steps_per_epoch is None or self._initial_model_state_sha256 is None:
            raise RuntimeError("V25a telemetry recorder is not initialized")
        return {
            "schema_version": V25A_GRADIENT_TELEMETRY_SCHEMA,
            "artifact_directory": str(self.artifact_dir),
            "config_path": str(self.config_path),
            "config_file_sha256": self.config_file_sha256,
            "config_identity_sha256": self.config_identity_sha256,
            "source_v24_summary_path": str(self.source_summary_path),
            "source_v24_summary_sha256": self.source_summary_sha256,
            "initial_model_state_sha256": self._initial_model_state_sha256,
            "steps_per_epoch": int(self._steps_per_epoch),
            "expected_parent_order_hashes": {
                str(epoch): value
                for epoch, value in sorted(self.expected_parent_order_hashes.items())
            },
            "gradient_module_metadata": gradient_module_metadata(self._groups),
            "capture_timing": "after_loss_backward_before_existing_global_norm_clip",
            "gradient_semantics": {
                "pre_clip_global_norm_owner": "existing torch.nn.utils.clip_grad_norm_ return value",
                "module_norms": "read-only L2 reductions over parameter.grad before clipping",
                "telemetry_changes_loss": False,
                "telemetry_changes_gradients": False,
                "telemetry_changes_optimizer": False,
                "telemetry_changes_data_order": False,
                "raw_parent_sample_ids_stored": False,
            },
        }

    def start(
        self,
        *,
        model: nn.Module,
        steps_per_epoch: int,
        initial_model_state_sha256: str,
        completed_history: Sequence[Mapping[str, Any]],
        resumed: bool,
    ) -> None:
        if int(steps_per_epoch) <= 0:
            raise ValueError("V25a telemetry steps_per_epoch must be positive")
        if len(self.expected_parent_order_hashes) != 20 or set(self.expected_parent_order_hashes) != set(range(1, 21)):
            raise ValueError("V25a telemetry requires all twenty V24 parent-order hashes")
        if not self.source_summary_path.is_file() or _sha256_file(self.source_summary_path) != self.source_summary_sha256:
            raise ValueError("V25a source V24 summary is missing or stale")
        self._groups = partition_gradient_modules(model)
        self._steps_per_epoch = int(steps_per_epoch)
        self._initial_model_state_sha256 = str(initial_model_state_sha256)
        self.directory.mkdir(parents=True, exist_ok=True)
        _write_new_or_identical(self.metadata_path, self._metadata())

        completed_epochs = [int(row.get("epoch", -1)) for row in completed_history]
        if completed_epochs != list(range(1, len(completed_epochs) + 1)):
            raise ValueError("V25a resume history must contain contiguous completed epochs")
        expected_files = {self.epoch_path(self.directory, epoch) for epoch in completed_epochs}
        actual_files = set(self.directory.glob("epoch-*.json"))
        pending_files = set(self.directory.glob("epoch-*.json.pending"))
        if actual_files == expected_files and pending_files:
            # The pending payload belongs to updates made after the most recent
            # durable checkpoint.  It must never become training evidence on a
            # resumed run, so retain it only as an explicitly orphaned audit
            # artifact and regenerate that epoch from the checkpoint state.
            next_epoch = len(completed_epochs) + 1
            expected_pending = self.epoch_path(self.directory, next_epoch).with_suffix(
                ".json.pending"
            )
            if pending_files != {expected_pending}:
                raise ValueError(
                    "V25a pending telemetry does not belong to the uncheckpointed next epoch"
                )
            archived = expected_pending.with_suffix(
                expected_pending.suffix + f".orphaned-before-checkpoint-{time.time_ns()}"
            )
            os.replace(expected_pending, archived)
            pending_files = set()
        if actual_files != expected_files:
            # A checkpoint is written before its staged telemetry file is
            # promoted.  The only recoverable interruption is therefore one
            # final pending file for the last checkpointed epoch.
            if not completed_epochs:
                raise ValueError(
                    "V25a telemetry files exist without a completed checkpoint history"
                )
            latest_epoch = completed_epochs[-1]
            latest_final = self.epoch_path(self.directory, latest_epoch)
            latest_pending = latest_final.with_suffix(latest_final.suffix + ".pending")
            expected_without_latest = expected_files - {latest_final}
            if actual_files != expected_without_latest or pending_files != {latest_pending}:
                raise ValueError(
                    "V25a telemetry files do not match resumable completed epochs: "
                    f"expected_final={sorted(path.name for path in expected_files)}, "
                    f"actual_final={sorted(path.name for path in actual_files)}, "
                    f"actual_pending={sorted(path.name for path in pending_files)}"
                )
            latest_history = dict(completed_history[-1])
            staged = dict(latest_history.get("gradient_telemetry") or {})
            if (
                str(staged.get("staged_path", "")) != str(latest_pending)
                or str(staged.get("staged_sha256", "")) != _sha256_file(latest_pending)
            ):
                raise ValueError(
                    "V25a latest checkpoint is not bound to its staged telemetry payload"
                )
            os.replace(latest_pending, latest_final)
            actual_files = set(self.directory.glob("epoch-*.json"))
            pending_files = set(self.directory.glob("epoch-*.json.pending"))
            # The checkpoint still serializes the staged descriptor.  It is
            # repaired in the trainer's resumed in-memory history before any
            # subsequent checkpoint is written, but validate the promoted
            # immutable payload here as well.
            self._validate_epoch_file(
                latest_final,
                expected_epoch=latest_epoch,
                expected_global_step_end=int(completed_history[-1]["global_step"]),
                expected_parent_order_sha256=str(
                    completed_history[-1]["parent_order_sha256"]
                ),
            )
        if actual_files != expected_files or pending_files:
            raise ValueError("V25a telemetry recovery did not produce an exact committed epoch set")
        if not resumed and completed_epochs:
            raise ValueError("fresh V25a telemetry run unexpectedly has completed history")
        for history_row in completed_history:
            epoch = int(history_row["epoch"])
            self._validate_epoch_file(
                self.epoch_path(self.directory, epoch),
                expected_epoch=epoch,
                expected_global_step_end=int(history_row["global_step"]),
                expected_parent_order_sha256=str(history_row["parent_order_sha256"]),
            )
        self._completed_epochs = set(completed_epochs)

    def committed_epoch_descriptor(
        self,
        *,
        epoch: int,
        expected_global_step_end: int,
        expected_parent_order_sha256: str,
    ) -> dict[str, Any]:
        """Return the immutable final descriptor for one committed epoch.

        This is deliberately usable after ``start`` repairs the one recoverable
        checkpoint/pending-file interruption.  The trainer then rewrites its
        in-memory resumed history to final paths before it emits any later
        checkpoint or summary.
        """

        if self._steps_per_epoch is None:
            raise RuntimeError("V25a telemetry descriptor requested before recorder start")
        epoch_value = int(epoch)
        if epoch_value not in self._completed_epochs:
            raise ValueError("V25a telemetry descriptor requested for an uncommitted epoch")
        path = self.epoch_path(self.directory, epoch_value)
        self._validate_epoch_file(
            path,
            expected_epoch=epoch_value,
            expected_global_step_end=int(expected_global_step_end),
            expected_parent_order_sha256=str(expected_parent_order_sha256),
        )
        return {
            "path": str(path),
            "sha256": _sha256_file(path),
            "epoch": epoch_value,
            "record_count": int(self._steps_per_epoch),
        }

    def capture_after_backward(
        self,
        *,
        batch: Mapping[str, Any],
        tensors: Mapping[str, Any],
        objective: Mapping[str, torch.Tensor],
        epoch: int,
        step: int,
        global_step: int,
    ) -> dict[str, Any]:
        if self._groups is None or self._steps_per_epoch is None:
            raise RuntimeError("V25a telemetry capture occurred before recorder start")
        if epoch in self._completed_epochs or step < 1 or step > self._steps_per_epoch:
            raise ValueError("V25a telemetry received an invalid active epoch/step")
        return capture_preclip_gradient_snapshot(
            groups=self._groups,
            batch=batch,
            tensors=tensors,
            objective=objective,
            epoch=epoch,
            step=step,
            global_step=global_step,
        )

    def record_after_clip_before_optimizer_step(
        self,
        record: Mapping[str, Any],
        *,
        pre_clip_global_norm: float,
        clip_threshold: float,
    ) -> None:
        if self._steps_per_epoch is None:
            raise RuntimeError("V25a telemetry record occurred before recorder start")
        value = _as_finite_float(pre_clip_global_norm, name="global pre-clip gradient norm")
        threshold = _as_finite_float(clip_threshold, name="gradient clip threshold")
        if value < 0.0 or threshold <= 0.0:
            raise ValueError("V25a telemetry clip norm/threshold is invalid")
        item = dict(record)
        module_total = _as_finite_float(
            item.get("pre_clip_module_total_l2"), name="module total gradient norm"
        )
        tolerance = max(1.0e-4, 1.0e-4 * max(1.0, value))
        difference = abs(module_total - value)
        if difference > tolerance:
            raise AssertionError(
                "V25a module pre-clip norm differs from clip_grad_norm_ return: "
                f"module={module_total}, global={value}, tolerance={tolerance}"
            )
        clipped = bool(value > threshold)
        item.update(
            {
                "pre_clip_global_l2": value,
                "gradient_clip_threshold": threshold,
                "was_globally_clipped": clipped,
                "global_clip_scale": float(threshold / value) if clipped and value > 0.0 else 1.0,
                "post_clip_global_l2_upper_bound": float(min(value, threshold)),
                "module_vs_global_l2_absolute_difference": difference,
                "globally_affected_nonzero_modules": [
                    name
                    for name, active in dict(
                        item["pre_clip_module_nonzero_gradient"]
                    ).items()
                    if bool(active)
                ]
                if clipped
                else [],
            }
        )
        if len(self._records) >= self._steps_per_epoch:
            raise ValueError("V25a telemetry received too many records in one epoch")
        self._records.append(item)

    def stage_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        parent_order_sha256: str,
    ) -> dict[str, Any]:
        if self._steps_per_epoch is None:
            raise RuntimeError("V25a telemetry flush occurred before recorder start")
        if epoch in self._completed_epochs:
            raise ValueError("V25a telemetry attempts to rewrite a completed epoch")
        if len(self._records) != self._steps_per_epoch:
            raise ValueError(
                "V25a telemetry epoch record count differs from dataloader steps: "
                f"records={len(self._records)}, steps={self._steps_per_epoch}"
            )
        expected_hash = self.expected_parent_order_hashes.get(int(epoch))
        if expected_hash is None or str(parent_order_sha256) != expected_hash:
            raise AssertionError(
                "V25a parent order differs from the frozen V24 control at "
                f"epoch {epoch}: observed={parent_order_sha256}, expected={expected_hash}"
            )
        expected_start = (int(epoch) - 1) * self._steps_per_epoch + 1
        expected_globals = list(range(expected_start, int(global_step) + 1))
        actual_globals = [int(record["global_step"]) for record in self._records]
        if actual_globals != expected_globals or len(expected_globals) != self._steps_per_epoch:
            raise AssertionError("V25a telemetry global-step sequence is not contiguous")
        if any(
            int(record["epoch"]) != int(epoch)
            or int(record["step"]) != index
            for index, record in enumerate(self._records, start=1)
        ):
            raise AssertionError("V25a telemetry epoch/step labels are not contiguous")
        payload = {
            "schema_version": V25A_GRADIENT_TELEMETRY_SCHEMA,
            "epoch": int(epoch),
            "step_count": int(self._steps_per_epoch),
            "global_step_start": expected_start,
            "global_step_end": int(global_step),
            "parent_order_sha256": str(parent_order_sha256),
            "records": self._records,
        }
        final_path = self.epoch_path(self.directory, epoch)
        pending_path = final_path.with_suffix(final_path.suffix + ".pending")
        if final_path.exists():
            raise FileExistsError("V25a final telemetry epoch already exists before staging")
        _write_new_or_identical(pending_path, payload)
        self._validate_epoch_file(
            pending_path,
            expected_epoch=epoch,
            expected_global_step_end=global_step,
            expected_parent_order_sha256=parent_order_sha256,
        )
        result = {
            "staged_path": str(pending_path),
            "staged_sha256": _sha256_file(pending_path),
            "final_path": str(final_path),
            "epoch": int(epoch),
            "record_count": int(self._steps_per_epoch),
        }
        self._records = []
        return result

    def commit_epoch(
        self,
        *,
        epoch: int,
        staged: Mapping[str, Any],
        expected_global_step_end: int,
        expected_parent_order_sha256: str,
    ) -> dict[str, Any]:
        """Promote a checkpoint-bound staged epoch to its final JSON file.

        The caller saves the epoch checkpoint between ``stage_epoch`` and this
        method.  A restart can consequently recover an interrupted promotion
        without ever treating telemetry from an uncheckpointed epoch as valid.
        """

        if self._steps_per_epoch is None:
            raise RuntimeError("V25a telemetry commit occurred before recorder start")
        if epoch in self._completed_epochs:
            raise ValueError("V25a telemetry attempts to recommit a completed epoch")
        final_path = self.epoch_path(self.directory, epoch)
        pending_path = final_path.with_suffix(final_path.suffix + ".pending")
        if (
            str(staged.get("staged_path", "")) != str(pending_path)
            or str(staged.get("final_path", "")) != str(final_path)
            or str(staged.get("staged_sha256", "")) != _sha256_file(pending_path)
        ):
            raise ValueError("V25a staged telemetry binding is invalid before commit")
        if final_path.exists():
            raise FileExistsError("V25a final telemetry epoch already exists before commit")
        self._validate_epoch_file(
            pending_path,
            expected_epoch=epoch,
            expected_global_step_end=expected_global_step_end,
            expected_parent_order_sha256=expected_parent_order_sha256,
        )
        os.replace(pending_path, final_path)
        self._completed_epochs.add(int(epoch))
        return self.committed_epoch_descriptor(
            epoch=epoch,
            expected_global_step_end=expected_global_step_end,
            expected_parent_order_sha256=expected_parent_order_sha256,
        )

    def _validate_epoch_file(
        self,
        path: Path,
        *,
        expected_epoch: int,
        expected_global_step_end: int,
        expected_parent_order_sha256: str,
    ) -> None:
        if self._steps_per_epoch is None:
            raise RuntimeError("V25a telemetry validation occurred before recorder start")
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = list(payload.get("records") or ())
        expected_start = (int(expected_epoch) - 1) * self._steps_per_epoch + 1
        if (
            payload.get("schema_version") != V25A_GRADIENT_TELEMETRY_SCHEMA
            or int(payload.get("epoch", -1)) != int(expected_epoch)
            or int(payload.get("step_count", -1)) != self._steps_per_epoch
            or int(payload.get("global_step_start", -1)) != expected_start
            or int(payload.get("global_step_end", -1)) != int(expected_global_step_end)
            or str(payload.get("parent_order_sha256", ""))
            != str(expected_parent_order_sha256)
            or len(records) != self._steps_per_epoch
        ):
            raise ValueError(f"V25a telemetry epoch payload is stale or malformed: {path}")
        for step, record in enumerate(records, start=1):
            if set(record) != V25A_RECORD_FIELDS:
                raise ValueError(
                    "V25a telemetry record fields differ from the sealed no-identifier schema: "
                    f"{path}/{step}"
                )
            task_counts = dict(record.get("task_label_counts") or {})
            base_counts = dict(record.get("base_label_counts") or {})
            observation_counts = dict(record.get("batch_observation_counts") or {})
            module_norms = dict(record.get("pre_clip_module_l2") or {})
            module_fractions = dict(record.get("pre_clip_module_squared_fraction") or {})
            module_nonzero = dict(record.get("pre_clip_module_nonzero_gradient") or {})
            loss = dict(record.get("loss") or {})
            expected_parent_count = int(record.get("parent_count", -1))
            expected_task_total = sum(int(task_counts.get(name, -1)) for name in V25A_TASK_LABELS)
            expected_base_total = sum(int(base_counts.get(name, -1)) for name in V25A_BASE_LABELS)
            if (
                int(record.get("epoch", -1)) != int(expected_epoch)
                or int(record.get("step", -1)) != step
                or int(record.get("global_step", -1)) != expected_start + step - 1
                or expected_parent_count != 2
                or expected_task_total != expected_parent_count
                or expected_base_total != expected_parent_count
                or set(task_counts) != set(V25A_TASK_LABELS)
                or set(base_counts) != set(V25A_BASE_LABELS)
                or set(module_norms) != set(V25A_GRADIENT_MODULE_NAMES)
                or set(module_fractions) != set(V25A_GRADIENT_MODULE_NAMES)
                or set(module_nonzero) != set(V25A_GRADIENT_MODULE_NAMES)
                or set(loss) != {"total", "final_state", "prefix_auxiliary"}
                or set(observation_counts)
                != {
                    "active_event_count",
                    "zero_delta_event_count",
                    "positive_delta_event_count",
                    "rd_present_event_count",
                    "course_feature_valid_event_count",
                }
                or not isinstance(record.get("parent_batch_sha256"), str)
                or len(str(record.get("parent_batch_sha256"))) != 64
                or not isinstance(record.get("was_globally_clipped"), bool)
                or not math.isfinite(float(record.get("pre_clip_global_l2", float("nan"))))
                or not math.isfinite(float(record.get("global_clip_scale", float("nan"))))
                or not math.isfinite(float(record.get("gradient_clip_threshold", float("nan"))))
                or not math.isfinite(float(record.get("pre_clip_module_total_l2", float("nan"))))
                or any(
                    not math.isfinite(float(module_norms.get(module, float("nan"))))
                    or float(module_norms[module]) < 0.0
                    or not math.isfinite(float(module_fractions.get(module, float("nan"))))
                    or float(module_fractions[module]) < 0.0
                    for module in V25A_GRADIENT_MODULE_NAMES
                )
                or abs(
                    sum(float(module_fractions[module]) for module in V25A_GRADIENT_MODULE_NAMES)
                    - (1.0 if float(record["pre_clip_global_l2"]) > 0.0 else 0.0)
                )
                > 1.0e-5
                or any(
                    not math.isfinite(float(loss[name]))
                    for name in ("total", "final_state", "prefix_auxiliary")
                )
                or int(observation_counts["active_event_count"]) <= 0
                or int(observation_counts["zero_delta_event_count"]) < 0
                or int(observation_counts["positive_delta_event_count"]) < 0
                or int(observation_counts["rd_present_event_count"]) < 0
                or int(observation_counts["course_feature_valid_event_count"]) < 0
                or int(observation_counts["zero_delta_event_count"])
                + int(observation_counts["positive_delta_event_count"])
                > int(observation_counts["active_event_count"])
                or int(observation_counts["rd_present_event_count"])
                > int(observation_counts["active_event_count"])
                or int(observation_counts["course_feature_valid_event_count"])
                > int(observation_counts["active_event_count"])
            ):
                raise ValueError(f"V25a telemetry step record is malformed: {path}/{step}")

    def finalize(self, *, completed_history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if self._steps_per_epoch is None:
            raise RuntimeError("V25a telemetry finalization occurred before recorder start")
        if self._records:
            raise RuntimeError("V25a telemetry has unflushed active-epoch records")
        completed_epochs = [int(row.get("epoch", -1)) for row in completed_history]
        if completed_epochs != list(range(1, 21)) or self._completed_epochs != set(completed_epochs):
            raise ValueError("V25a telemetry requires exactly twenty completed epochs")
        for history_row in completed_history:
            self._validate_epoch_file(
                self.epoch_path(self.directory, int(history_row["epoch"])),
                expected_epoch=int(history_row["epoch"]),
                expected_global_step_end=int(history_row["global_step"]),
                expected_parent_order_sha256=str(history_row["parent_order_sha256"]),
            )
        expected_steps = 20 * self._steps_per_epoch
        return {
            "schema_version": V25A_GRADIENT_TELEMETRY_SCHEMA,
            "directory": str(self.directory),
            "metadata_path": str(self.metadata_path),
            "metadata_sha256": _sha256_file(self.metadata_path),
            "epoch_file_count": 20,
            "record_count": expected_steps,
            "steps_per_epoch": self._steps_per_epoch,
            "capture_timing": "after_loss_backward_before_existing_global_norm_clip",
            "gradient_telemetry_changes_training_semantics": False,
            "source_v24_parent_order_hashes_exactly_reproduced": True,
        }


__all__ = ["V25A_BASE_LABELS", "V25A_GRADIENT_MODULE_NAMES", "V25A_GRADIENT_TELEMETRY_DIRECTORY", "V25A_GRADIENT_TELEMETRY_SCHEMA", "V25A_RECORD_FIELDS", "V25A_TASK_LABELS", "V25AGradientTelemetryRecorder", "capture_preclip_gradient_snapshot", "gradient_module_metadata", "partition_gradient_modules"]