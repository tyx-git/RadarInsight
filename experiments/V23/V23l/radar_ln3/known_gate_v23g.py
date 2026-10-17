"""Contracts for V23g: self-developed CfC density-weighted KnownGate replay.

This experiment intentionally begins from the completed V21e seed43 course
KnownGate, rather than historical DBB V20r.  It isolates the recurrent
same-second update schedule only:

* Control: ``bounded_observation`` with fixed 0.25 gain;
* Treatment: ``density_weighted`` with ``min(0.25*k, 0.75)``.

No feature, loss, seed, parent order, data split, normalization or checkpoint
rule is changed.  Development is not an open-set/product evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .known_gate import KnownGateConfig, known_gate_config_identity_sha256, load_known_gate_config


V23G_VARIANT = "v23g_course_prefix_density_weighted"
V23G_CONFIG_SCHEMA = "RADAR-LN3-V23g-known-gate-density-weighted-config-v1"
V23G_READINESS_SCHEMA = "RADAR-LN3-V23g-known-gate-density-weighted-readiness-v1"
V23G_CHECKPOINT_SCHEMA = "RADAR-LN3-V23g-known-gate-density-weighted-seed43-checkpoint-v1"
V23G_CONTROL_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-seed43-config.json"
)
V23G_CONTROL_ARTIFACT_RELATIVE = Path(
    "artifacts/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-seed43-formal"
)
V23G_CONTROL_REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-paired-validation.json"
)
V23G_TRAINING_SEED = 43
V23G_EPOCHS = 20


def config_relative() -> Path:
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V23g-cq08-liquid-known-gate-causal-course-consistency-"
        "density-weighted-seed43-config.json"
    )


def artifact_relative() -> Path:
    return Path(
        "artifacts/"
        "RADAR-LN3-V23g-cq08-liquid-known-gate-causal-course-consistency-"
        "density-weighted-seed43-formal"
    )


def contract_relative() -> Path:
    return Path(
        "data/processed/"
        "RADAR-LN3-V23g-cq08-liquid-known-gate-causal-course-consistency-"
        "density-weighted-contract.json"
    )


def readiness_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23g-known-gate-density-weighted-training-readiness.json"
    )




def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class V23GConfig:
    schema_version: str
    config_name: str
    variant: str
    source_control: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]


def config_identity_sha256(config: V23GConfig) -> str:
    return canonical_sha256(asdict(config))


def source_control(root: Path) -> dict[str, Any]:
    """Bind V23g to the completed self-developed V21e seed43 Control."""

    root = root.resolve()
    config_path = (root / V23G_CONTROL_CONFIG_RELATIVE).resolve()
    artifact = (root / V23G_CONTROL_ARTIFACT_RELATIVE).resolve()
    summary_path = artifact / "training-summary.json"
    checkpoint_path = artifact / "best_model.pt"
    detail_path = artifact / "sample_detail_information.json"
    report_path = (root / V23G_CONTROL_REPORT_RELATIVE).resolve()
    required = (config_path, summary_path, checkpoint_path, detail_path, report_path)
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("V23g requires completed V21e seed43 control: " + ", ".join(missing))
    config = load_known_gate_config(config_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    history = list(summary.get("history") or ())
    if (
        config.variant != "v21e_course_prefix"
        or config.train.seed != V23G_TRAINING_SEED
        or config.train.dataloader_seed != V23G_TRAINING_SEED
        or config.train.epochs != V23G_EPOCHS
        or config.model.liquid_invalid_dt_update_mode != "bounded_observation"
        or abs(config.model.liquid_invalid_dt_observation_gain - 0.25) > 1.0e-12
        or summary.get("status") != "completed"
        or summary.get("variant") != "v21e_course_prefix"
        or len(history) != V23G_EPOCHS
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or detail.get("variant") != "v21e_course_prefix"
        or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -2))
        or report.get("status") != "completed"
        or report.get("schema_version")
        != "RADAR-LN3-V21e-known-gate-causal-course-consistency-paired-validation-v1"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V23g V21e seed43 control is incomplete or outside frozen scope")
    parent_orders = []
    for epoch, row in enumerate(history, start=1):
        value = dict(row)
        order = str(value.get("parent_order_sha256", ""))
        if (
            int(value.get("epoch", -1)) != epoch
            or len(order) != 64
            or int(value.get("parent_order_count", -1)) != 986
            or int(value.get("parent_order_unique_count", -1)) != 986
        ):
            raise ValueError("V23g Control parent-order contract is invalid")
        parent_orders.append(order)
    return {
        "variant": config.variant,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": known_gate_config_identity_sha256(config),
        "summary_path": _relative(root, summary_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": _relative(root, checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "detail_path": _relative(root, detail_path),
        "detail_sha256": sha256_file(detail_path),
        "report_path": _relative(root, report_path),
        "report_sha256": sha256_file(report_path),
        "initial_model_state_sha256": str(summary["initial_model_state_sha256"]),
        "parent_order_sha256_by_epoch": parent_orders,
        "fixed_zero_delta_gain": 0.25,
        "control_mode": "bounded_observation",
    }


def load_config(root: Path) -> tuple[V23GConfig, KnownGateConfig, Path]:
    root = root.resolve()
    path = (root / config_relative()).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    if set(payload) != {"schema_version", "config_name", "variant", "source_control", "model", "train"}:
        raise ValueError("V23g config top-level fields differ")
    if (
        payload.get("schema_version") != V23G_CONFIG_SCHEMA
        or payload.get("config_name") != path.stem.removesuffix("-config")
        or payload.get("variant") != V23G_VARIANT
    ):
        raise ValueError("V23g config identity differs")
    control = dict(payload["source_control"])
    if control != source_control(root):
        raise ValueError("V23g V21e seed43 direct-control binding is stale")
    control_path = _resolve(root, str(control["config_path"]))
    source_config = load_known_gate_config(control_path)
    model = dict(payload["model"])
    expected_model = dict(json.loads(control_path.read_text(encoding="utf-8"))["model"])
    expected_model["liquid_invalid_dt_update_mode"] = "density_weighted"
    if model != expected_model:
        raise ValueError("V23g model differs beyond density_weighted mode")
    train = dict(payload["train"])
    expected_train = dict(json.loads(control_path.read_text(encoding="utf-8"))["train"])
    artifact = str(artifact_relative()).replace("\\", "/")
    expected_train.update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": V23G_CHECKPOINT_SCHEMA,
            "paired_baseline_config_path": str(V23G_CONTROL_CONFIG_RELATIVE).replace("\\", "/"),
        }
    )
    if train != expected_train:
        raise ValueError("V23g training protocol differs beyond its isolated dynamics mode")
    return (
        V23GConfig(
            schema_version=V23G_CONFIG_SCHEMA,
            config_name=str(payload["config_name"]),
            variant=V23G_VARIANT,
            source_control=control,
            model=model,
            train=train,
        ),
        source_config,
        control_path,
    )


__all__ = ["V23G_CHECKPOINT_SCHEMA", "V23G_CONFIG_SCHEMA", "V23G_CONTROL_ARTIFACT_RELATIVE", "V23G_CONTROL_CONFIG_RELATIVE", "V23G_CONTROL_REPORT_RELATIVE", "V23G_EPOCHS", "V23G_READINESS_SCHEMA", "V23G_TRAINING_SEED", "V23G_VARIANT", "V23GConfig", "artifact_relative", "canonical_sha256", "config_identity_sha256", "config_relative", "contract_relative", "load_config", "readiness_relative", "sha256_file", "source_control"]