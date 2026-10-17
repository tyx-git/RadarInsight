"""Run the V26h frozen train-only L1 representation OOD diagnostic.

V26h is deliberately a diagnostic, not a product change.  It reloads each
frozen V26e L1 epoch-32 checkpoint, fits a class-conditional reference from
that fold's Known train parents only, and evaluates a threshold-free novelty
ranking on the already frozen V26f mixed validation/test populations.

The expensive RD forward is run in one short-lived CPU subprocess per fold.
That boundary is intentional on Windows: the project RD tensors are large and
PyTorch/CUDA may retain allocator pages across many variable-length parents.
The worker returns only final 48-D states (for train fitting) and compact
scored mixed rows; it never writes or mutates a model, cache, threshold, or
semantic action.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.known_gate import KnownGateConfig, build_known_gate_model_for_config, collate_known_gate_sequences, known_gate_config_identity_sha256, known_gate_device
from radar_ln3.known_gate_v26_audit import collate_v26_open_set_audit
from radar_ln3.known_gate_v26e_oof import CONTRACT as V26E_CONTRACT, READINESS as V26E_READINESS, fold_record
from radar_ln3.known_gate_v26h import KNOWN_BASE_LABELS, L1_CLASSES, L1_EPOCH, UNKNOWN_BASE_LABEL, read_json, relative_path, sha256_file
from train.known_gate_common import _model_inputs as l1_model_inputs
from train.known_gate_common import _state_dict_sha256
from train.known_gate_v26c_oof import _load_checkpoint
from train.known_gate_v26e_oof import _runtime_l1_config, _source_protocol


V26E_CONTRACT_PATH = Path(str(V26E_CONTRACT))
V26E_READINESS_PATH = Path(str(V26E_READINESS))



def _clear_dataset_cache(dataset: Any) -> None:
    """Release decoded RD arrays after a parent/fold forward boundary."""

    backing = getattr(dataset, "_dataset", dataset)
    base = getattr(backing, "base", None)
    cache = getattr(base, "_tensor_cache", None)
    items = getattr(cache, "_items", None)
    if items is not None and hasattr(items, "clear"):
        items.clear()


def _fold_l1_runtime(
    root: Path,
    *,
    fold_id: int,
    prepared_contract: Mapping[str, Any],
    force_cpu: bool = False,
) -> tuple[Any, KnownGateConfig, torch.device, dict[str, Any]]:
    """Construct the exact V26e fold model and load its bound snapshot."""

    v26e_contract = read_json(root / V26E_CONTRACT_PATH, description="V26e contract")
    v26e_fold = fold_record(v26e_contract, fold_id)
    l1_config, l1_config_path, _declaration = _runtime_l1_config(
        root, fold_id, dict(v26e_fold["l1"])
    )
    source, source_sha = _source_protocol(root)
    model, reference, pairing = build_known_gate_model_for_config(root, source)
    del reference, pairing
    model.config = l1_config
    initial_state_sha = _state_dict_sha256(model.state_dict())
    # CUDA_VISIBLE_DEVICES can leave ``torch.cuda.is_available()`` true in a
    # Windows subprocess while exposing zero logical devices.  In that state
    # ``known_gate_device`` would incorrectly return cuda:0 and torch.load
    # would fail before the diagnostic starts.  The isolated worker is
    # intentionally CPU-only, so make that execution boundary explicit rather
    # than relying on process-global CUDA discovery.
    device = torch.device("cpu") if force_cpu else known_gate_device(l1_config)
    if force_cpu and device.type != "cpu":
        raise AssertionError("V26h isolated fold runtime must use CPU")
    model.to(device)

    source_fold = next(
        value
        for value in list(dict(prepared_contract["source_v26e"]).get("folds") or ())
        if int(value["fold_id"]) == int(fold_id)
    )
    checkpoint_binding = dict(source_fold["checkpoint"])
    checkpoint = root / str(checkpoint_binding["path"])
    if (
        relative_path(root, checkpoint) != checkpoint_binding["path"]
        or sha256_file(checkpoint) != checkpoint_binding["sha256"]
    ):
        raise ValueError(f"V26h fold {fold_id} checkpoint binding differs")
    _load_checkpoint(
        checkpoint,
        task="known_gate",
        schema=l1_config.train.checkpoint_schema,
        fold_id=fold_id,
        model=model,
        optimizer=None,
        config_identity=known_gate_config_identity_sha256(l1_config),
        config_path=l1_config_path,
        contract_sha256=sha256_file(root / V26E_CONTRACT_PATH),
        readiness_sha256=sha256_file(root / V26E_READINESS_PATH),
        source_config_sha256=source_sha,
        initial_state_sha256=initial_state_sha,
        device=device,
    )
    model.eval()
    return model, l1_config, device, {
        "config_path": relative_path(root, l1_config_path),
        "config_sha256": sha256_file(l1_config_path),
        "checkpoint_path": relative_path(root, checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": _state_dict_sha256(model.state_dict()),
        "fixed_epoch": int(L1_EPOCH),
        "device": str(device),
    }


def _classifier_readout_state(
    model: Any,
    output: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return the exact state consumed by the final L1 Linear layer."""

    state = output.get("liquid_final_state")
    classifier = getattr(model, "classifier", None)
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise ValueError("V26h model output lacks a [batch,hidden] final state")
    if not isinstance(classifier, torch.nn.Sequential) or len(classifier) != 2:
        raise ValueError("V26h expects the frozen LayerNorm -> Linear classifier")
    normalizer, linear = classifier[0], classifier[1]
    if not isinstance(normalizer, torch.nn.LayerNorm) or not isinstance(linear, torch.nn.Linear):
        raise ValueError("V26h classifier structure differs from the frozen L1 contract")
    value = normalizer(state)
    if value.shape[-1] != linear.in_features or not torch.isfinite(value).all():
        raise ValueError("V26h classifier readout state is invalid")
    reconstructed = linear(value)
    declared = output.get("window_logits")
    if (
        not isinstance(declared, torch.Tensor)
        or declared.shape != reconstructed.shape
        or not torch.allclose(reconstructed, declared, rtol=1.0e-5, atol=1.0e-6)
    ):
        raise ValueError("V26h readout state does not reconstruct frozen final logits")
    return value


def _forward_dataset(
    dataset: Any,
    *,
    model: Any,
    device: torch.device,
    fold_id: int,
    split: str,
    expected_parent_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Forward parent-final states serially and retain only compact rows."""

    if split not in {"train_reference", "val", "test"}:
        raise ValueError("V26h received an unsupported split")
    expected = tuple(str(value) for value in expected_parent_ids)
    observed = tuple(str(value) for value in dataset.parent_ids)
    if observed != expected:
        raise ValueError(f"V26h {split} parent order differs from its frozen contract")

    rows: list[dict[str, Any]] = []
    for parent_index, requested_parent_id in enumerate(expected, start=1):
        item = dataset[parent_index - 1]
        batch = (
            collate_known_gate_sequences([item])
            if split == "train_reference"
            else collate_v26_open_set_audit([item])
        )
        tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            output = model(**l1_model_inputs(tensors))
            state = _classifier_readout_state(model, output)
            logits = output["window_logits"]
            probabilities = torch.softmax(logits, dim=-1)
        parent_id = str(batch["parent_sample_ids"][0])
        if parent_id != requested_parent_id:
            raise ValueError(f"V26h {split} parent identity changed during forward")
        label = str(batch["base_label_names"][0])
        allowed = KNOWN_BASE_LABELS if split == "train_reference" else (*KNOWN_BASE_LABELS, UNKNOWN_BASE_LABEL)
        if label not in allowed:
            raise ValueError(f"V26h {split} contains an unsupported truth label: {label}")
        length = int(batch["lengths"][0].item())
        if length < 1 or state.shape != (1, 48) or probabilities.shape != (1, 2):
            raise ValueError(f"V26h {split} produced an invalid parent output: {parent_id}")
        state_row = state[0].detach().to("cpu", dtype=torch.float64).numpy().copy()
        probability_row = probabilities[0].detach().to("cpu", dtype=torch.float64).numpy().copy()
        if not np.isfinite(state_row).all() or not np.isfinite(probability_row).all():
            raise ValueError(f"V26h {split} emitted a non-finite parent output: {parent_id}")
        first = dataset.base._groups[parent_id].iloc[0]
        rows.append(
            {
                "parent_sample_id": parent_id,
                "label_name": label,
                "class_id": int(batch["labels"][0].item()),
                "state": state_row,
                "known_target_probability": float(probability_row[0]),
                "known_clutter_probability": float(probability_row[1]),
                "l1_predicted_class": L1_CLASSES[int(probability_row.argmax())],
                "parent_length": length,
                "track_id": str(first.get("track_id", "")),
                "capture_id": str(first.get("capture_id", "")),
                "provenance_component_id": str(first.get("provenance_component_id", "")),
                "rd_present_at_final": bool(batch["rd_frame_mask"][0, length - 1].any().item()),
                "course_valid_at_final": bool(
                    batch["known_gate_feature_valid"][0, length - 1, 0].item()
                ),
                "fold_id": int(fold_id),
                "split": split,
            }
        )
        del tensors, output, state, logits, probabilities, batch, item
        _clear_dataset_cache(dataset)
        if parent_index % 8 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if tuple(str(row["parent_sample_id"]) for row in rows) != expected:
        raise ValueError(f"V26h {split} output parent coverage differs")
    return rows





























