"""V28-v2 configuration, role boundaries, and immutable provenance."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)

V28_SCHEMA_VERSION = "RADAR-LN3-V28-unified-product-v2"
V28_MANIFEST_SCHEMA = "RADAR-LN3-V28-manifest-v2"
V28_POOL_SCHEMA = "RADAR-LN3-V28-pool-registry-v2"
V28_REPRESENTATION_SCHEMA = "RADAR-LN3-V28-representation-cache-v2"
V28_HEAD_SCHEMA = "RADAR-LN3-V28-head-checkpoint-v2"
V28_REPORT_SCHEMA = "RADAR-LN3-V28-task-evaluation-v2"
V28_L3_SCHEMA = "RADAR-LN3-V28-l3-registry-v2"
V28_ROOT = Path("src-v3") / "v28"
V28_L1_TASK = "l1_known_gate"
V28_L2_TASK = "l2_dbb_specialist"
V28_TASKS = (V28_L1_TASK, V28_L2_TASK)
V28_ROLES = ("representation_fit", "representation_development", "calibration")
V28_AUDIT_ROLE = "post_selection_audit"
V28_L1_CLASSES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
V28_L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
V28_DEFAULT_CONFIG = V28_ROOT / "config" / "V28-default.json"
V28_CHECKPOINT_CODE_SCOPE = (
    "src-v3/v28/model.py",
    "src-v3/v28/data.py",
)

#: Tooling that measures or inspects the frozen product but does not define
#: model semantics, feature construction, training, or the decision policy.
#: These files are excluded from the code manifest so that instrumentation can
#: be added or edited without invalidating every checkpoint and the L3
#: registry that binds ``system_manifest_sha256``.
V28_NON_CONTRACT_SOURCES = frozenset(
    {
        "benchmark.py",
    }
)


@dataclass(frozen=True)
class V28Config:
    payload: dict[str, Any]
    path: Path
    sha256: str
    identity_sha256: str

    @property
    def data(self) -> Mapping[str, Any]:
        return self.payload["data"]

    @property
    def model(self) -> Mapping[str, Any]:
        return self.payload["model"]

    @property
    def heads(self) -> Mapping[str, Any]:
        return self.payload["heads"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def selection(self) -> Mapping[str, Any]:
        return self.payload["selection"]

    @property
    def calibration(self) -> Mapping[str, Any]:
        return self.payload["calibration"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.payload["artifacts"]

    @property
    def protocol(self) -> Mapping[str, Any]:
        return self.payload["protocol"]


@dataclass(frozen=True)
class V28Context:
    project_root: Path
    config: V28Config
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V28 {field} must be an object")
    return dict(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V28 {field} must be non-empty text")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V28 {field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V28 {field} must be a non-negative integer")
    return int(value)


def _finite_float(value: Any, field: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V28 {field} must be numeric") from exc
    if not result == result or result in (float("inf"), float("-inf")):
        raise ValueError(f"V28 {field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"V28 {field} must be positive")
    return result


def _sha(value: Any, field: str) -> str:
    result = _text(value, field).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"V28 {field} must be a SHA-256 digest")
    return result


def artifact_path(project_root: str | Path, template: str, *, task: str | None = None, seed: int | None = None, epoch: int | None = None, role: str | None = None, component: str | None = None) -> Path:
    """Render one artifact and enforce the ``src-v3/v28`` boundary."""

    root = Path(project_root).resolve()
    template = _text(template, "artifact template")
    if "{task}" in template and task not in V28_TASKS:
        raise ValueError("V28 artifact requires a registered task")
    if "{role}" in template and role not in V28_ROLES:
        raise ValueError("V28 artifact requires a registered role")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("V28 artifact requires a non-negative seed")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("V28 artifact requires a positive epoch")
    try:
        rendered = template.format(task=task, seed=seed, epoch=epoch, role=role, component=component)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid V28 artifact template: {template}") from exc
    target = resolve_project_path(root, rendered)
    scope = resolve_project_path(root, V28_ROOT)
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V28 artifact escaped src-v3/v28") from exc
    return target


def _validate_config(root: Path, payload: Mapping[str, Any], *, allow_missing_derived: bool = False) -> None:
    if payload.get("schema_version") != V28_SCHEMA_VERSION or payload.get("experiment_id") != "V28":
        raise ValueError("V28 configuration identity differs")
    data = _mapping(payload.get("data"), "data")
    for field in ("index_path", "index_schema_version", "source_name", "train_split_name", "validation_split_name", "test_split_name"):
        _text(data.get(field), f"data.{field}")
    if len({str(data["train_split_name"]), str(data["validation_split_name"]), str(data["test_split_name"])}) != 3:
        raise ValueError("V28 data split names must be distinct")
    _positive_int(data.get("max_sequence_length"), "data.max_sequence_length")
    _positive_int(data.get("max_rd_frames_per_point"), "data.max_rd_frames_per_point")
    resolve_project_path(root, data["index_path"], must_exist=True)

    l1 = _mapping(data.get("l1"), "data.l1")
    for split in ("train", "validation"):
        source = _mapping(l1.get(split), f"data.l1.{split}")
        resolve_project_path(root, source.get("course_cache"), must_exist=True)
    _text(l1.get("course_feature_name"), "data.l1.course_feature_name")
    _text(l1.get("course_feature_variant"), "data.l1.course_feature_variant")
    _positive_int(l1.get("course_minimum_valid_nonfirst_points"), "data.l1.course_minimum_valid_nonfirst_points")
    for field in ("rd_amplitude_center", "rd_amplitude_scale"):
        _finite_float(l1.get(field), f"data.l1.{field}", positive=field.endswith("scale"))

    def require_path(value: Any, field: str) -> None:
        text = _text(value, field)
        # V28 derived caches are created by an explicit preparation stage.
        # Their absence must not prevent the manifest or classifier config
        # from loading; the consuming dataset validates them when opened.
        if allow_missing_derived and text.replace("\\", "/").startswith("src-v3/v28/") and not (root / text).is_file():
            return
        resolve_project_path(root, text, must_exist=True)

    l2 = _mapping(data.get("l2"), "data.l2")
    for split in ("train", "validation", "test"):
        source = _mapping(l2.get(split), f"data.l2.{split}")
        require_path(source.get("rd_physical_cache"), f"data.l2.{split}.rd_physical_cache")
        require_path(source.get("micro_doppler_cache"), f"data.l2.{split}.micro_doppler_cache")
    open_set = _mapping(l2.get("open_set_caches"), "data.l2.open_set_caches")
    for role in V28_ROLES:
        source = _mapping(open_set.get(role), f"data.l2.open_set_caches.{role}")
        require_path(source.get("rd_physical_cache"), f"data.l2.open_set_caches.{role}.rd_physical_cache")
        require_path(source.get("micro_doppler_cache"), f"data.l2.open_set_caches.{role}.micro_doppler_cache")
    clipping = _mapping(l2.get("tr_clipping"), "data.l2.tr_clipping")
    if clipping.get("method") != "known_train_only_one_sided_upper_quantile":
        raise ValueError("V28 L2 clipping method differs")
    for field in ("quantile", "range_lower_m", "range_upper_m", "speed_lower_mps", "speed_upper_mps"):
        _finite_float(clipping.get(field), f"data.l2.tr_clipping.{field}")
    if clipping["range_upper_m"] < clipping["range_lower_m"] or clipping["speed_upper_mps"] < clipping["speed_lower_mps"]:
        raise ValueError("V28 L2 clipping bounds are invalid")
    for field in ("rd_amplitude_center", "rd_amplitude_scale"):
        _finite_float(l2.get(field), f"data.l2.{field}", positive=field.endswith("scale"))

    model = _mapping(payload.get("model"), "model")
    for field in ("tr_value_dim", "tr_missing_dim", "rd_quality_dim", "rd_height", "rd_width", "rd_embedding_dim", "hidden_dim", "liquid_input_dim", "liquid_hidden_dim", "observation_residual_hidden_dim"):
        _positive_int(model.get(field), f"model.{field}")
    for field in ("tau_min_seconds", "tau_max_seconds", "zero_dt_event_floor_seconds", "dt_max_seconds", "invalid_dt_observation_gain"):
        _finite_float(model.get(field), f"model.{field}", positive=True)
    if model["tau_max_seconds"] < model["tau_min_seconds"]:
        raise ValueError("V28 tau bounds are invalid")
    if model.get("invalid_dt_update_mode") not in {"bounded_observation", "floor", "skip", "strict_skip", "density_weighted"}:
        raise ValueError("V28 Liquid V2 update mode is unsupported")
    if model.get("classifier_readout_normalization") != "independent_layer_norm_on_raw_state":
        raise ValueError("V28 classifier readout normalization differs")

    heads = _mapping(payload.get("heads"), "heads")
    for field in ("projection_dim", "trajectory_hidden_dim", "knownness_hidden_dim", "quality_hidden_dim", "epochs"):
        _positive_int(heads.get(field), f"heads.{field}")
    for field in ("learning_rate", "weight_decay", "gradient_clip_norm"):
        _finite_float(heads.get(field), f"heads.{field}", positive=field == "gradient_clip_norm")
    forbidden = heads.get("forbidden_inputs")
    if not isinstance(forbidden, list) or set(map(str, forbidden)) != {"l1_logits", "l2_logits", "softmax", "label_identity"}:
        raise ValueError("V28 knownness forbidden-input contract differs")

    training = _mapping(payload.get("training"), "training")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("V28 training.seeds are invalid")
    for field in ("batch_size", "epochs"):
        _positive_int(training.get(field), f"training.{field}")
    for field in ("learning_rate", "gradient_clip_norm"):
        _finite_float(training.get(field), f"training.{field}", positive=True)
    _finite_float(training.get("weight_decay"), "training.weight_decay")
    for key, expected in (("l1_known_gate_class_weights", 2), ("l2_dbb_specialist_class_weights", 3)):
        weights = training.get(key)
        if not isinstance(weights, list) or len(weights) != expected or any(_finite_float(v, f"training.{key}") <= 0 for v in weights):
            raise ValueError(f"V28 {key} are invalid")

    selection = _mapping(payload.get("selection"), "selection")
    for field in ("knownness_recall_floor", "clutter_recall_floor", "quality_recall_floor"):
        value = _finite_float(selection.get(field), f"selection.{field}")
        if not 0.0 < value <= 1.0:
            raise ValueError(f"V28 selection.{field} must be in (0,1]")
    _nonnegative_int(selection.get("max_correct_known_new_rejections"), "selection.max_correct_known_new_rejections")
    _positive_int(selection.get("minimum_negative_support"), "selection.minimum_negative_support")
    for field in ("knownness_min_events", "quality_min_events"):
        _positive_int(heads.get(field), f"heads.{field}")
    for field in ("quality_min_rd_fraction", "quality_min_dt_fraction", "quality_min_aux_fraction"):
        value = _finite_float(heads.get(field), f"heads.{field}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"V28 heads.{field} must be within [0,1]")
    for field in (
        "l1_target_knownness_recall_floor",
        "l1_clutter_knownness_recall_floor",
        "l2_drone_knownness_recall_floor",
        "l2_bird_knownness_recall_floor",
        "l2_balloon_knownness_recall_floor",
    ):
        value = _finite_float(heads.get(field), f"heads.{field}")
        if not 0.0 < value <= 1.0:
            raise ValueError(f"V28 heads.{field} must be in (0,1]")
    for field in ("l2_bird_max_correct_known_new_rejections", "l2_route_max_correct_known_new_rejections"):
        _nonnegative_int(heads.get(field), f"heads.{field}")
    _finite_float(heads.get("bird_consistency_weight"), "heads.bird_consistency_weight")
    for field in ("l1_target_candidate_probability_floor", "l1_clutter_candidate_probability_floor"):
        value = _finite_float(selection.get(field), f"selection.{field}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"V28 selection.{field} must be within [0,1]")
    route_thresholds = _mapping(selection.get("l2_route_thresholds"), "selection.l2_route_thresholds")
    if set(route_thresholds) != set(V28_L2_CLASSES):
        raise ValueError("V28 L2 route thresholds must cover exactly the DBB classes")
    for name, value in route_thresholds.items():
        _finite_float(value, f"selection.l2_route_thresholds.{name}")
    calibration = _mapping(payload.get("calibration"), "calibration")
    for field in ("known_target_recall_floor", "clutter_recall_floor", "conformal_error_rate"):
        value = _finite_float(calibration.get(field), f"calibration.{field}")
        if not 0.0 < value <= 1.0:
            raise ValueError(f"V28 calibration.{field} must be in (0,1]")
    _positive_int(calibration.get("minimum_unknown_support_for_formal"), "calibration.minimum_unknown_support_for_formal")
    for field in ("temperature_min", "temperature_max"):
        _finite_float(calibration.get(field), f"calibration.{field}", positive=True)
    _positive_int(calibration.get("temperature_grid_size"), "calibration.temperature_grid_size")

    migration = _mapping(payload.get("migration_sources"), "migration_sources")
    for name, source in migration.items():
        resolve_project_path(root, source, must_exist=True)
        _text(name, "migration source name")
    protocol = _mapping(payload.get("protocol"), "protocol")
    required = {
        "unknown_is_classifier_class": False,
        "unknown_classifier_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "l2_clutter_training": False,
        "l2_clutter_knownness": False,
        "all_valid_events_update_both_tasks": True,
        "quality_output_feedback": False,
        "fixed_softmax_threshold_allowed": False,
        "single_global_distance_gate_allowed": False,
        "l3_fail_closed_before_calibration": True,
        "offline_chunked_online_must_match": True,
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V28 protocol.{key} differs")


def load_config(project_root: str | Path, config_path: str | Path | None = None, *, allow_missing_derived: bool = False) -> V28Config:
    root = Path(project_root).resolve()
    path = (root / V28_DEFAULT_CONFIG if config_path is None else Path(config_path)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("V28 configuration escaped project root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V28 configuration root must be an object")
    value = dict(payload)
    _validate_config(root, value, allow_missing_derived=allow_missing_derived)
    return V28Config(value, path, sha256_file(path), canonical_sha256(value))


def _source_hashes(root: Path, config: V28Config) -> dict[str, str]:
    paths: dict[str, str] = {"index": str(config.data["index_path"])}
    for task in ("l1", "l2"):
        data = config.data[task]
        split_names = ("train", "validation") if task == "l1" else ("train", "validation", "test")
        for split in split_names:
            for name, source in dict(data[split]).items():
                normalized = str(source).replace("\\", "/")
                # V28-owned feature caches are derived artifacts. Their
                # hashes live in feature-manifest-v2; including them here
                # would make the system manifest change after preparation.
                if not normalized.startswith("src-v3/v28/"):
                    paths[f"{task}:{split}:{name}"] = str(source)
    for name, source in _mapping(config.payload.get("migration_sources"), "migration_sources").items():
        paths[f"migration:{name}"] = str(source)
    expected = _mapping(config.payload.get("expected_source_hashes", {}), "expected_source_hashes")
    observed: dict[str, str] = {}
    for name, source in paths.items():
        digest = sha256_file(resolve_project_path(root, source, must_exist=True))
        if name in expected and digest != _sha(expected[name], f"expected_source_hashes.{name}"):
            raise ValueError(f"V28 source hash differs: {name}")
        observed[name] = digest
    return observed


def _public_dependency_hashes(root: Path) -> dict[str, str]:
    files = (
        Path("src-v3/radar_ln3/liquid_v2.py"),
        Path("src-v3/radar_ln3/data.py"),
        Path("src-v3/radar_ln3/features.py"),
        Path("src-v3/radar_ln3/known_gate_features.py"),
        Path("src-v3/radar_ln3/rd_physical.py"),
        Path("src-v3/radar_ln3/micro_doppler_broadband.py"),
        Path("src-v3/radar_ln3/runtime.py"),
        Path("src-v3/radar_ln3/losses.py"),
    )
    return {path.as_posix(): sha256_file(root / path) for path in files}


def build_manifest(root: Path, config: V28Config, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    code_files = sorted(
        path
        for path in (root / V28_ROOT).rglob("*.py")
        if "__pycache__" not in path.parts and path.name not in V28_NON_CONTRACT_SOURCES
    )
    dependencies: dict[str, str] = {}
    for package in ("torch", "numpy", "pandas", "scipy"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "schema_version": V28_MANIFEST_SCHEMA,
        "experiment_id": "V28",
        "variant": "unified-liquid-v2-strict-open-set-v2",
        "config": {"path": relative_project_path(root, config.path), "sha256": config.sha256, "identity_sha256": config.identity_sha256},
        "source_hashes": dict(source_hashes),
        "code_hashes": {relative_project_path(root, path): sha256_file(path) for path in code_files},
        "public_dependency_hashes": _public_dependency_hashes(root),
        "dependencies": dependencies,
        "tasks": {
            V28_L1_TASK: {"class_names": list(V28_L1_CLASSES), "classifier_labels": ["DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"], "unknown_is_classifier_class": False, "unknown_training_exposure": 0, "clutter_training": True},
            V28_L2_TASK: {"class_names": list(V28_L2_CLASSES), "classifier_labels": list(V28_L2_CLASSES), "unknown_is_classifier_class": False, "unknown_training_exposure": 0, "clutter_training": False, "clutter_knownness": False},
        },
        "liquid_v2": {"equation_version": str(config.model["equation_version"]), "memory_state": "raw_r_t", "post_update_normalization": False, "classifier_readout": "independent_layer_norm_on_raw_state", "all_valid_events_update_both_tasks": True},
        "selection": dict(config.selection),
        "calibration": dict(config.calibration),
        "protocol": dict(config.protocol),
        "evaluation_scope": {"test_payload_access_count": 0, "test_split": str(config.data["test_split_name"]), "post_selection_audit": "not_exposed_by_source"},
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "payload_sha256" in value:
        raise ValueError("V28 payload is already sealed")
    value["payload_sha256"] = canonical_sha256(payload)
    return value


def write_sealed_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json_text(_seal(payload)))
    return path


def read_sealed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V28 artifact is not an object: {path}")
    declared = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError(f"V28 artifact payload hash differs: {path}")
    return dict(value)


def checkpoint_manifest_compatibility(
    project_root: str | Path,
    current_manifest: Mapping[str, Any],
    trained_manifest_sha256: str,
    *,
    task: str,
) -> dict[str, Any]:
    """Validate a checkpoint against the current model-affecting contract.

    A classifier checkpoint may outlive downstream representation/cache code.
    The formal Unknown-support gate is an L3-only policy and may change
    without retraining the classifier; model and input contracts remain strict.
    """

    current_digest = sha256_file(
        resolve_project_path(
            Path(project_root),
            current_manifest.get("_manifest_path", ""),
            must_exist=True,
        )
    ) if current_manifest.get("_manifest_path") else None
    if current_digest == trained_manifest_sha256:
        return {"status": "exact", "trained_manifest_sha256": trained_manifest_sha256}
    root = Path(project_root).resolve()
    archive = root / V28_ROOT / "manifests" / "archive"
    candidates = sorted(archive.glob(f"V28-manifest-v2.json.stale-before-{trained_manifest_sha256[:16]}"))
    old_path = next((path for path in candidates if sha256_file(path) == trained_manifest_sha256), None)
    if old_path is None:
        raise ValueError("V28 checkpoint manifest differs and its archived manifest is unavailable")
    old = read_sealed_json(old_path)
    old_selection = dict(old.get("selection", {}))
    current_selection = dict(current_manifest.get("selection", {}))
    for key in (
        "l1_target_candidate_probability_floor",
        "l1_clutter_candidate_probability_floor",
        "l2_candidate_empty_fallback",
        "l2_route_thresholds",
    ):
        old_selection.pop(key, None)
        current_selection.pop(key, None)
    for key in ("source_hashes", "public_dependency_hashes", "liquid_v2", "protocol"):
        if old.get(key) != current_manifest.get(key):
            raise ValueError(f"V28 checkpoint compatibility differs: {key}")
    if old_selection != current_selection:
        raise ValueError("V28 checkpoint compatibility differs: selection")
    old_config = old.get("config", {})
    current_config = current_manifest.get("config", {})
    if old_config.get("path") != current_config.get("path"):
        raise ValueError("V28 checkpoint compatibility differs: config/path")
    config_path = resolve_project_path(root, str(current_config.get("path")), must_exist=True)
    current_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(current_payload, Mapping):
        raise ValueError("V28 checkpoint compatibility differs: config payload")
    if canonical_sha256(current_payload) != old_config.get("identity_sha256"):
        normalized_payload = dict(current_payload)
        normalized_calibration = dict(normalized_payload.get("calibration", {}))
        normalized_calibration["minimum_unknown_support_for_formal"] = old.get("calibration", {}).get("minimum_unknown_support_for_formal")
        normalized_payload["calibration"] = normalized_calibration
        normalized_heads = dict(normalized_payload.get("heads", {}))
        for key in (
            "l1_target_knownness_recall_floor",
            "l1_clutter_knownness_recall_floor",
            "l2_drone_knownness_recall_floor",
            "l2_bird_knownness_recall_floor",
            "l2_balloon_knownness_recall_floor",
            "l2_bird_max_correct_known_new_rejections",
            "l2_route_max_correct_known_new_rejections",
            "bird_consistency_weight",
        ):
            normalized_heads.pop(key, None)
        normalized_payload["heads"] = normalized_heads
        normalized_selection = dict(normalized_payload.get("selection", {}))
        for key in (
            "l1_target_candidate_probability_floor",
            "l1_clutter_candidate_probability_floor",
            "l2_candidate_empty_fallback",
            "l2_route_thresholds",
        ):
            normalized_selection.pop(key, None)
        normalized_payload["selection"] = normalized_selection
        normalized_artifacts = dict(normalized_payload.get("artifacts", {}))
        normalized_artifacts.pop("audit_report_path", None)
        normalized_payload["artifacts"] = normalized_artifacts
    else:
        normalized_payload = None
    if normalized_payload is not None and canonical_sha256(normalized_payload) != old_config.get("identity_sha256"):
        raise ValueError("V28 checkpoint compatibility differs: config outside L3 support policy")
    old_calibration = old.get("calibration", {})
    current_calibration = current_manifest.get("calibration", {})
    for key in set(old_calibration) | set(current_calibration):
        if key == "minimum_unknown_support_for_formal":
            continue
        if old_calibration.get(key) != current_calibration.get(key):
            raise ValueError(f"V28 checkpoint compatibility differs: calibration/{key}")
    for key in V28_CHECKPOINT_CODE_SCOPE:
        if old.get("code_hashes", {}).get(key) != current_manifest.get("code_hashes", {}).get(key):
            raise ValueError(f"V28 checkpoint compatibility differs: {key}")
    if old.get("tasks", {}).get(task) != current_manifest.get("tasks", {}).get(task):
        raise ValueError(f"V28 checkpoint compatibility differs: tasks/{task}")
    changed_code = sorted(
        key
        for key in set(old.get("code_hashes", {})) | set(current_manifest.get("code_hashes", {}))
        if old.get("code_hashes", {}).get(key) != current_manifest.get("code_hashes", {}).get(key)
        and key not in V28_CHECKPOINT_CODE_SCOPE
    )
    return {
        "status": "compatible_archived_manifest",
        "trained_manifest_sha256": trained_manifest_sha256,
        "current_manifest_sha256": current_digest,
        "archived_manifest_path": relative_project_path(root, old_path),
        "unchanged_code_scope": list(V28_CHECKPOINT_CODE_SCOPE),
        "downstream_only_changes": changed_code,
    }


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path, allow_missing_derived=True)
    expected = build_manifest(root, config, _source_hashes(root, config))
    path = artifact_path(root, str(config.artifacts["manifest_path"]))
    if path.exists():
        archive = root / V28_ROOT / "manifests" / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / f"{path.name}.stale-before-{sha256_file(path)[:16]}"
        if destination.exists():
            if sha256_file(destination) != sha256_file(path):
                raise FileExistsError(f"V28 manifest archive collision: {destination}")
            path.unlink()
        else:
            path.replace(destination)
    write_sealed_json(path, expected)
    return {"path": relative_project_path(root, path), "sha256": sha256_file(path), "payload": expected}


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> V28Context:
    root = Path(project_root).resolve()
    # V28-owned classifier/role caches are staged artifacts. Context loading
    # verifies external sources and the active code manifest, while each
    # consuming Dataset verifies that its own derived cache exists.
    config = load_config(root, config_path, allow_missing_derived=True)
    expected = build_manifest(root, config, _source_hashes(root, config))
    path = artifact_path(root, str(config.artifacts["manifest_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"V28 manifest is missing; run prepare-manifest: {path}")
    manifest = read_sealed_json(path)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V28 manifest differs from current code, config, or sources")
    return V28Context(root, config, manifest, path, sha256_file(path))


def task_classes(task: str) -> tuple[str, ...]:
    if task == V28_L1_TASK:
        return V28_L1_CLASSES
    if task == V28_L2_TASK:
        return V28_L2_CLASSES
    raise ValueError(f"unknown V28 task: {task}")


__all__ = ["V28Config", "V28Context", "V28_AUDIT_ROLE", "V28_HEAD_SCHEMA", "V28_L1_CLASSES", "V28_L1_TASK", "V28_L2_CLASSES", "V28_L2_TASK", "V28_L3_SCHEMA", "V28_MANIFEST_SCHEMA", "V28_POOL_SCHEMA", "V28_REPORT_SCHEMA", "V28_REPRESENTATION_SCHEMA", "V28_ROLES", "V28_ROOT", "V28_SCHEMA_VERSION", "V28_TASKS", "artifact_path", "build_manifest", "checkpoint_manifest_compatibility", "load_config", "load_context", "prepare_manifest", "read_sealed_json", "task_classes", "write_sealed_json"]