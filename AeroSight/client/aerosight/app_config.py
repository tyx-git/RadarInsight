# -*- coding: utf-8 -*-
"""Configuration loading, normalization and legacy migration."""
from __future__ import annotations

import copy
import os
import yaml

USER_SETTINGS_PATH = os.path.join(
    os.path.dirname(__file__), "config", "user_settings.yaml")

# 内置默认配置，不依赖随包发布的 yaml 文件。
BUILTIN_DEFAULT_CONFIG = yaml.safe_load("""
instances:
  - id: asd-directory-replay
    name: "ASD 目录回放"
    enabled: true
    source:
      kind: rrtsp
      rrtsp:
        url: "rtsp://127.0.0.1:8554/radar"
        username: ""
        password: ""
        remote_inference: true
      simulation:
        rate_hz: 5
    display:
      plane_range_m: 3000
    map:
      image_path: ""
      radar_x: 680
      radar_y: 680
      pixels_per_100m: 20
      heading_deg: 0
    model:
      onnx_path: ""
pipeline:
  message_queue_size: 64
  inference_queue_size: 16
  result_queue_size: 32
ui:
  render_fps: 30
  window_title: "AeroSight 雷达终端"
  selected_instance_id: asd-directory-replay
logging:
  level: INFO
  dir: logs
""")

DEFAULT_MAP = {
    "image_path": "",
    "radar_x": 680.0,
    "radar_y": 680.0,
    "pixels_per_100m": 20.0,
    "heading_deg": 0.0,
}


def _deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None = None) -> dict:
    # Explicit path (argument or AEROSIGHT_CONFIG) loads that file; otherwise
    # use built-in defaults and merge the local, never-published user settings.
    path = path or os.environ.get("AEROSIGHT_CONFIG")
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = copy.deepcopy(BUILTIN_DEFAULT_CONFIG)
        if os.path.exists(USER_SETTINGS_PATH):
            with open(USER_SETTINGS_PATH, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, user_cfg)
    return normalize_config(cfg)


def normalize_map(value=None, fallback=None) -> dict:
    result = dict(DEFAULT_MAP)
    result.update(fallback or {})
    if isinstance(value, dict):
        result.update(value)
    return {
        "image_path": str(result.get("image_path", "")),
        "radar_x": float(result.get("radar_x", 680.0)),
        "radar_y": float(result.get("radar_y", 680.0)),
        "pixels_per_100m": max(float(result.get("pixels_per_100m", 20.0)), 1.0),
        "heading_deg": float(result.get("heading_deg", 0.0)),
    }


def default_instance(instance_id: str, name: str, kind: str = "simulation") -> dict:
    return {
        "id": instance_id,
        "name": name,
        "enabled": True,
        "source": {
            "kind": kind,
            "rrtsp": {
                "url": "",
                "username": "",
                "password": "",
                "remote_inference": False,
            },
            "simulation": {
                "rate_hz": 5.0,
            },
        },
        "display": {
            "plane_range_m": 3000.0,
        },
        "map": normalize_map(),
        "model": {
            "onnx_path": "",
        },
    }


def normalize_instance(raw: dict, index: int, fallback_map=None) -> dict:
    instance_id = str(raw.get("id") or f"radar-instance-{index + 1}")
    default_name = "\u5185\u7f6e\u4eff\u771f" if raw.get("source", {}).get("kind") == "simulation" else f"\u96f7\u8fbe {index + 1}"
    source_raw = raw.get("source", {})
    kind = source_raw.get("kind", "rrtsp")
    if kind not in ("rrtsp", "simulation"):
        kind = "rrtsp"
    rrtsp_raw = source_raw.get("rrtsp", {})
    simulation_raw = source_raw.get("simulation", {})

    item = default_instance(instance_id, str(raw.get("name") or default_name), kind)
    item["enabled"] = bool(raw.get("enabled", True))
    item["source"]["rrtsp"] = {
        "url": str(rrtsp_raw.get("url", "")),
        "username": str(rrtsp_raw.get("username", "")),
        "password": str(rrtsp_raw.get("password", "")),
        "remote_inference": bool(rrtsp_raw.get("remote_inference", False)),
    }
    item["source"]["simulation"] = {
        "rate_hz": max(float(simulation_raw.get("rate_hz", 5.0)), 0.1),
    }
    item["display"] = {
        "plane_range_m": max(float(raw.get("display", {}).get("plane_range_m", 3000.0)), 1.0),
    }
    item["map"] = normalize_map(raw.get("map", {}), fallback_map)
    item["model"] = {
        "onnx_path": str(raw.get("model", {}).get("onnx_path", "")),
    }
    return item


def normalize_config(cfg: dict) -> dict:
    legacy_streams = cfg.get("streams")
    if legacy_streams is not None:
        cfg = _migrate_legacy(cfg)

    raw_instances = cfg.get("instances", [])
    if not isinstance(raw_instances, list) or not raw_instances:
        raw_instances = [
            {
                **default_instance("asd-directory-replay", "ASD \u76ee\u5f55\u56de\u653e", "rrtsp"),
                "source": {
                    "kind": "rrtsp",
                    "rrtsp": {
                        "url": "rtsp://127.0.0.1:8554/radar",
                        "username": "",
                        "password": "",
                        "remote_inference": True,
                    },
                    "simulation": {"rate_hz": 5.0},
                },
            }
        ]
    instances = [
        normalize_instance(raw, i)
        for i, raw in enumerate(raw_instances)
    ]
    cfg["instances"] = instances[:8]

    pipeline = cfg.setdefault("pipeline", {})
    pipeline.setdefault("message_queue_size", 64)
    pipeline.setdefault("inference_queue_size", 16)
    pipeline.setdefault("result_queue_size", 32)

    ui = cfg.setdefault("ui", {})
    ui.setdefault("render_fps", 30)
    ui.setdefault("selected_instance_id", "")
    ui.setdefault("window_title", "AeroSight \u96f7\u8fbe\u7ec8\u7aef")
    ui.pop("selected_stream_id", None)
    selected = ui.get("selected_instance_id", "")
    ids = {item["id"] for item in instances}
    if selected not in ids:
        ui["selected_instance_id"] = next(iter(ids))

    logging_cfg = cfg.setdefault("logging", {})
    logging_cfg.setdefault("level", "INFO")
    logging_cfg.setdefault("dir", "logs")
    return cfg


def _migrate_legacy(cfg: dict) -> dict:
    streams = cfg.get("streams", {})
    global_map = normalize_map(cfg.get("map", {}))
    display_range = float(cfg.get("display", {}).get("plane_range_m", 3000.0))
    onnx_path = str(cfg.get("model", {}).get("onnx_path", ""))
    instances = []

    if streams.get("simulation_enabled", True):
        item = default_instance("legacy-simulation", "\u5185\u7f6e\u4eff\u771f", "simulation")
        item["source"]["simulation"]["rate_hz"] = float(streams.get("simulation_rate_hz", 5.0))
        item["display"]["plane_range_m"] = display_range
        item["map"] = normalize_map({}, global_map)
        item["model"]["onnx_path"] = onnx_path
        instances.append(item)

    for i, raw in enumerate(streams.get("items", [])):
        if not raw.get("enabled", True):
            continue
        item = default_instance(
            str(raw.get("id") or f"legacy-rrtsp-{i + 1}"),
            str(raw.get("name") or f"\u96f7\u8fbe {i + 1}"),
            "rrtsp",
        )
        item["source"]["rrtsp"] = {
            "url": str(raw.get("url", "")),
            "username": str(raw.get("username", "")),
            "password": str(raw.get("password", "")),
        }
        item["display"]["plane_range_m"] = display_range
        item["map"] = normalize_map(raw.get("map", {}), global_map)
        item["model"]["onnx_path"] = onnx_path
        instances.append(item)

    selected = cfg.get("ui", {}).get("selected_stream_id", "")
    cfg.pop("streams", None)
    cfg.pop("display", None)
    cfg.pop("model", None)
    cfg.pop("map", None)
    cfg["instances"] = instances
    cfg.setdefault("ui", {})["selected_instance_id"] = selected
    return cfg
