"""V27d-r02 Bird→Drone 混淆治理实验协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from v27.V27d.r20.protocol import load_config as load_r20_config
from v27.V27d.r20.protocol import load_manifest as load_r20_manifest
from v27.V27d.r17.protocol import canonical_sha256

R02_EXPERIMENT_ID = "V27d-r02"
R02_CONFIG_SCHEMA = "RADAR-LN3-V27d-r02-config-v1"
R02_MANIFEST_SCHEMA = "RADAR-LN3-V27d-r02-manifest-v1"
R02_REPORT_SCHEMA = "RADAR-LN3-V27d-r02-report-v1"
R02_CHECKPOINT_SCHEMA = "RADAR-LN3-V27d-r02-checkpoint-v1"
R02_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
R02_TRAINING_HARD_NEGATIVE_IDS = (
    "r17|excel|20241128_191432|start20250327191948524|batch524",
    "r17|excel|20241128_191432|start20250327192855704|batch704",
    "r17|excel|20241128_191432|start20250327200420350|batch350",
    "r17|excel|20241130_213057|start20250321093258966|batch966",
    "r17|excel|20250401_091500|start20250401102404567|batch567",
    "r17|excel|20250401_091500|start2025040110335666|batch66",
    "r17|excel|20250401_091500|start20250401104311440|batch440",
    "r17|excel|20250401_091500|start20250401105504886|batch886",
    "r17|excel|20250403_190653|start20250403192905724|batch724",
)
R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS = (
    "r17|excel|20241128_191432|start20250327192615657|batch657",
    "r17|excel|20241128_191432|start20250327192730675|batch675",
    "r17|excel|20241128_191432|start2025032720011493|batch93",
    "r17|excel|20241128_191432|start20250327200139134|batch134",
)


def project_path(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    path = (root / Path(value)).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"r02 path escaped project root: {path}") from exc
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"r02 JSON must be an object: {path}")
    return dict(payload)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


class R02Config:
    """校验后的 r02 配置，同时保留只读的 r20 父配置。"""

    def __init__(self, payload: dict[str, Any], path: Path, r20: Any) -> None:
        self.payload = payload
        self.path = path
        self.r20 = r20
        self.sha256 = sha256_file(path)
        self.identity_sha256 = canonical_sha256(payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    @property
    def source(self) -> Mapping[str, Any]:
        return self.payload["source"]

    @property
    def model(self) -> Mapping[str, Any]:
        return self.payload["model"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> R02Config:
    root = Path(project_root).resolve()
    path = project_path(
        root,
        config_path or Path("src-v3/v27/V27d/r02/config/V27d-r02-bird-drone.json"),
        must_exist=True,
    )
    payload = load_json(path)
    if payload.get("schema_version") != R02_CONFIG_SCHEMA or payload.get("experiment_id") != R02_EXPERIMENT_ID:
        raise ValueError("r02 configuration identity differs")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("r02 source must be an object")
    r20_config_path = project_path(root, str(source.get("r20_config_path")), must_exist=True)
    r20 = load_r20_config(root, r20_config_path)
    if sha256_file(r20_config_path) != source.get("r20_config_sha256"):
        raise ValueError("r02 r20 config hash differs")
    if source.get("r20_experiment_id") != "V27d-r20":
        raise ValueError("r02 parent experiment must be V27d-r20")
    if source.get("dataset") != "OriData/excel/TR-only-processed-r17":
        raise ValueError("r02 dataset differs from registered r20 dataset")
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("r02 training must be an object")
    for key in ("seed", "epochs", "batch_size"):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"r02 training.{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay", "gradient_clip_norm", "margin", "margin_weight", "reliability_weight", "maturity_weight", "hard_negative_weight", "calibration_regularization", "calibration_log_temperature_bound", "calibration_bias_bound"):
        value = float(training.get(key))
        if value < 0.0 or not (value == value) or value in (float("inf"), float("-inf")):
            raise ValueError(f"r02 training.{key} must be finite and non-negative")
    if training.get("sequence_policy") != "complete_parent_sequence_no_event_random_split":
        raise ValueError("r02 must keep complete parent sequences")
    model = payload.get("model")
    if not isinstance(model, Mapping) or int(model.get("hidden_dim", 0)) <= 0:
        raise ValueError("r02 model hidden_dim is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("r02 artifacts must be an object")
    for key in ("manifest_path", "hard_negative_path", "checkpoint_path", "training_report_path", "report_path", "calibration_path"):
        if not isinstance(artifacts.get(key), str) or not artifacts[key].strip():
            raise ValueError(f"r02 artifacts.{key} is required")
    return R02Config(payload, path, r20)


def build_manifest(project_root: str | Path, config: R02Config) -> dict[str, Any]:
    root = Path(project_root).resolve()
    r20_manifest, r20_manifest_path, r20_manifest_sha256 = load_r20_manifest(
        root, config.r20, allow_code_drift=True
    )
    assignments = dict(r20_manifest.get("parent_assignments", {}))
    train_ids = [parent_id for parent_id, split in assignments.items() if split == "train"]
    validation_ids = [parent_id for parent_id, split in assignments.items() if split == "validation"]
    if set(R02_TRAINING_HARD_NEGATIVE_IDS) - set(train_ids):
        raise ValueError("r02 training hard-negative IDs are not all in r20 train split")
    if set(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS) - set(validation_ids):
        raise ValueError("r02 validation audit IDs are not all in r20 validation split")
    if set(R02_TRAINING_HARD_NEGATIVE_IDS) & set(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS):
        raise ValueError("r02 hard-negative train and validation sets overlap")
    return {
        "schema_version": R02_MANIFEST_SCHEMA,
        "experiment_id": R02_EXPERIMENT_ID,
        "parent_experiment": "V27d-r20",
        "config": {
            "path": config.path.relative_to(root).as_posix(),
            "sha256": config.sha256,
            "identity_sha256": config.identity_sha256,
        },
        "source": {
            "dataset": config.source["dataset"],
            "r20_config_path": config.source["r20_config_path"],
            "r20_config_sha256": config.source["r20_config_sha256"],
            "r20_manifest_path": r20_manifest_path.relative_to(root).as_posix(),
            "r20_manifest_sha256": r20_manifest_sha256,
        },
        "split": {
            "protocol": "r20 sealed source_suffix+label parent split",
            "train_parent_count": len(train_ids),
            "validation_parent_count": len(validation_ids),
            "event_random_split": False,
            "sequence_policy": "complete_parent_sequence_no_event_random_split",
        },
        "hard_negative_policy": {
            "training_ids": list(R02_TRAINING_HARD_NEGATIVE_IDS),
            "training_count": len(R02_TRAINING_HARD_NEGATIVE_IDS),
            "validation_audit_ids": list(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS),
            "validation_audit_count": len(R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS),
            "validation_ids_are_never_used_for_training": True,
            "loss_weight": float(config.training["hard_negative_weight"]),
        },
        "model_policy": {
            "base_checkpoint_arm": "zero_control",
            "base_checkpoint_seed": 42,
            "bird_specific_reliability_head": True,
            "bird_maturity_head": True,
            "class_conditional_calibration": "train-only logits fit; validation apply-only",
            "bird_drone_margin_loss": True,
        },
        "parent_assignments_sha256": canonical_sha256(assignments),
        "parent_assignments": assignments,
    }


def prepare_manifest(project_root: str | Path, config: R02Config) -> tuple[Path, dict[str, Any]]:
    root = Path(project_root).resolve()
    manifest = build_manifest(root, config)
    path = project_path(root, config.artifacts["manifest_path"])
    write_json(path, manifest)
    return path, manifest


__all__ = ["R02_CHECKPOINT_SCHEMA", "R02_CLASSES", "R02_CONFIG_SCHEMA", "R02_EXPERIMENT_ID", "R02_MANIFEST_SCHEMA", "R02_REPORT_SCHEMA", "R02_TRAINING_HARD_NEGATIVE_IDS", "R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS", "R02Config", "build_manifest", "load_config", "load_json", "prepare_manifest", "project_path", "sha256_file", "write_json"]