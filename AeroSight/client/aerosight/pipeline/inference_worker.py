# -*- coding: utf-8 -*-
"""ONNX inference process with one independent backend per radar instance."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _q

from aerosight.core.radar import RadarInference
from aerosight.inference.onnx_backend import InferenceBackend

log = logging.getLogger(__name__)


def inference_main(cfg: dict, frame_q: mp.Queue, result_q: mp.Queue,
                   cmd_q: mp.Queue, stop_event: mp.Event):
    model_paths = _model_path_map(cfg)
    backends: dict[str, InferenceBackend] = {}
    log.info("Inference process started: %s independent instance model slots",
             len(model_paths))

    def apply_config(new_cfg: dict):
        nonlocal model_paths
        wanted = _model_path_map(new_cfg)
        for instance_id, path in wanted.items():
            backend = backends.get(instance_id)
            if backend is None or backend.onnx_path != path:
                if backend is not None:
                    backend.close()
                backends[instance_id] = InferenceBackend(path)
        for removed in set(backends) - set(wanted):
            backends[removed].close()
            backends.pop(removed, None)
        model_paths = wanted

    while not stop_event.is_set():
        drained_config = False
        try:
            while True:
                cmd = cmd_q.get_nowait()
                drained_config = True
                if cmd.get("cmd") == "apply":
                    apply_config(cmd.get("cfg", cfg))
                elif cmd.get("cmd") == "model":
                    # Legacy one-model command is applied only when a stream id exists.
                    instance_id = cmd.get("stream_id")
                    if instance_id:
                        path = cmd.get("path", "")
                        backend = backends.get(instance_id)
                        if backend is None or backend.onnx_path != path:
                            backends[instance_id] = InferenceBackend(path)
                        model_paths[instance_id] = path
        except _q.Empty:
            pass

        try:
            frame = frame_q.get(timeout=0.2 if not drained_config else 0.02)
        except _q.Empty:
            continue
        except (FileNotFoundError, EOFError):
            break

        instance_id = frame.stream_id
        if instance_id not in backends:
            backends[instance_id] = InferenceBackend(
                model_paths.get(instance_id, ""))
        backend = backends[instance_id]
        labels, status = backend.run(frame)
        result = RadarInference(
            stream_id=instance_id,
            frame_id=frame.frame_id,
            labels=labels,
            status=status)
        try:
            while True:
                result_q.get_nowait()
        except _q.Empty:
            pass
        try:
            result_q.put_nowait(result)
        except Exception:
            pass

    for backend in backends.values():
        backend.close()
    log.info("Inference process exited")


def _model_path_map(cfg: dict) -> dict[str, str]:
    return {
        item["id"]: item.get("model", {}).get("onnx_path", "")
        for item in cfg.get("instances", [])
    }
