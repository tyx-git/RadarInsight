# -*- coding: utf-8 -*-
"""AeroSight v3 ONNX Runtime inference backend.

Required data:
* latest 29 track points: float32 [B,29,14], left-padded with zeros
* current complete RD map: normally float32 [B,2,H,W]

Accepted outputs are class probabilities, class ids, or class names.
"""
from __future__ import annotations

import json
import logging
from collections import deque

import numpy as np

from aerosight.inference.features import (
    TRACK_FEATURE_SIZE,
    TRACK_HISTORY_LENGTH,
    clean_rd_map,
    track_features,
)

log = logging.getLogger(__name__)



def build_session(onnx_path: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        try:
            session = ort.InferenceSession(
                onnx_path, providers=["DmlExecutionProvider"]
            )
            log.info("ONNX Runtime has enabled DirectML")
            return session
        except Exception as exc:
            log.warning("DirectML initialization failed; using CPU: %s", exc)
    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    log.info("ONNX Runtime has enabled CPU")
    return session


class InferenceBackend:
    def __init__(self, onnx_path: str = ""):
        self.onnx_path = ""
        self.sess = None
        self.track_input_name = ""
        self.rd_inputs: list[dict] = []
        self.output_name = ""
        self.output_mode = ""
        self.class_names: list[str] = []
        self.class_ids: list[int] = []
        self.class_map: dict[int, str] = {}
        self.class_count = 0
        self.sequence_length = TRACK_HISTORY_LENGTH
        self.status = "No ONNX selected; target classification disabled"
        self.usable = False
        self.histories: dict[tuple[str, int], dict] = {}
        if onnx_path:
            self.set_model(onnx_path)

    def set_model(self, onnx_path: str) -> str:
        self.close()
        if not onnx_path:
            self.onnx_path = ""
            self.sess = None
            self.status = "No ONNX selected; target classification disabled"
            self.usable = False
            return self.status

        try:
            self.sess = build_session(onnx_path)
            self.onnx_path = onnx_path
            self.class_ids, self.class_names, self.class_map = self._read_class_mapping()
            self.usable, self.status = self._check_adapter()
        except Exception as exc:
            self.sess = None
            self.usable = False
            self.status = f"Model load failed: {type(exc).__name__}: {exc}"
            log.warning(self.status)
        if not self.usable:
            self.sess = None
        return self.status

    def close(self):
        self.sess = None
        self.histories = {}
        self.rd_inputs = []
        self.track_input_name = ""
        self.output_name = ""
        self.output_mode = ""
        self.class_names = []
        self.class_ids = []
        self.class_map = {}
        self.class_count = 0

    def run(self, frame) -> tuple[dict[int, tuple[str, float]], str]:
        remote_labels = getattr(frame, "remote_labels", None)
        if remote_labels:
            return dict(remote_labels), "RadarInsight Full Cascade ONNX (remote)"
        if not self.onnx_path:
            return {}, "No ONNX selected; target classification disabled"
        if self.sess is None or not self.usable:
            return {}, self.status

        try:
            current_keys = set()
            for obs in frame.observations:
                key = (frame.stream_id, obs.track_id)
                current_keys.add(key)
                history = self.histories.get(key)
                if history is None:
                    history = {
                        "items": deque(maxlen=TRACK_HISTORY_LENGTH),
                        "last_seen": frame.frame_id,
                    }
                    self.histories[key] = history
                previous_time = (
                    history["items"][-1][0] if history["items"] else frame.timestamp
                )
                dt = max(0.0, frame.timestamp - previous_time)
                vector = track_features(obs, dt)
                history["items"].append((frame.timestamp, vector))
                history["last_seen"] = frame.frame_id

            self._prune_old_histories(frame.frame_id, current_keys)
            current_observations = [
                obs for obs in frame.observations
                if (frame.stream_id, obs.track_id) in self.histories
            ]
            if not current_observations:
                return {}, self.status

            track_batch = []
            for obs in current_observations:
                items = self.histories[(frame.stream_id, obs.track_id)]["items"]
                steps = np.zeros(
                    (TRACK_HISTORY_LENGTH, TRACK_FEATURE_SIZE), dtype=np.float32
                )
                start = TRACK_HISTORY_LENGTH - len(items)
                for offset, (_timestamp, vector) in enumerate(items):
                    steps[start + offset] = vector
                track_batch.append(steps)
            track_array = np.stack(track_batch).astype(np.float32)

            batch_size = len(current_observations)
            rd_feed = self._rd_feed(frame, batch_size)
            feed = {self.track_input_name: track_array}
            feed.update(rd_feed)
            outputs = self.sess.run(None, feed)
            labels = self._parse_outputs(
                outputs, [obs.track_id for obs in current_observations]
            )
            return labels, self.status
        except Exception as exc:
            message = f"Inference failed: {type(exc).__name__}: {exc}"
            log.warning(message)
            return {}, message

    def _rd_feed(self, frame, batch_size: int) -> dict[str, np.ndarray]:
        amp, ph, current_shape = self._current_rd_map(frame)
        for spec in self.rd_inputs:
            expected = spec.get("shape_hw")
            if expected is not None and tuple(expected) != current_shape:
                raise ValueError(
                    f"Complete RD map shape {current_shape} does not match "
                    f"model input {spec['name']} expected {tuple(expected)}; "
                    "the map is not resized"
                )

        feed: dict[str, np.ndarray] = {}
        for spec in self.rd_inputs:
            mode = spec["mode"]
            if mode == "fused":
                channels = int(spec.get("channels") or 2)
                if channels == 1:
                    image = amp[None, :, :]
                else:
                    image = np.stack((amp, ph), axis=0)
            elif mode == "amplitude":
                image = amp[None, :, :]
            elif mode == "phase":
                image = ph[None, :, :]
            elif mode == "amplitude_3d":
                image = amp
            elif mode == "phase_3d":
                image = ph
            else:
                raise ValueError(f"Unsupported RD input mode {mode}")
            feed[spec["name"]] = self._repeat_image(image, batch_size)
        return feed

    def _current_rd_map(self, frame) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        if frame.rd_amplitude:
            amp, ph = clean_rd_map(frame.rd_amplitude, frame.rd_phase)
            return amp, ph, amp.shape

        fallbacks = [
            tuple(spec["shape_hw"])
            for spec in self.rd_inputs
            if spec.get("shape_hw") is not None
        ]
        if fallbacks:
            shape = fallbacks[0]
            amp = np.zeros(shape, dtype=np.float32)
            return amp, np.zeros_like(amp), shape
        raise ValueError(
            "Current complete RD map is missing and model RD dimensions are dynamic"
        )

    @staticmethod
    def _repeat_image(image: np.ndarray, batch_size: int) -> np.ndarray:
        return np.broadcast_to(
            image, (batch_size,) + image.shape
        ).copy()

    def _check_adapter(self) -> tuple[bool, str]:
        inputs = self.sess.get_inputs()
        track_info = None
        rank4 = []
        rank3_rd = []

        for info in inputs:
            shape = tuple(getattr(info, "shape", ()))
            name = info.name.lower()
            if len(shape) == 3 and str(shape[-1]) == str(TRACK_FEATURE_SIZE):
                if track_info is not None:
                    return False, "More than one [B,T,14] track input was found"
                if not self._dynamic_dim(shape[0]):
                    return False, f"Track input {info.name} batch B must be dynamic"
                if isinstance(shape[1], int) and int(shape[1]) != TRACK_HISTORY_LENGTH:
                    return False, (
                        f"Track input {info.name} time dimension must be "
                        f"{TRACK_HISTORY_LENGTH}, got {shape[1]}"
                    )
                if "float" not in info.type:
                    return False, f"Track input {info.name} must be tensor(float32)"
                track_info = info
            elif len(shape) == 4:
                if not self._dynamic_dim(shape[0]):
                    return False, f"RD input {info.name} batch B must be dynamic"
                if "float" not in info.type:
                    return False, f"RD input {info.name} must be tensor(float32)"
                rank4.append(info)
            elif len(shape) == 3 and str(shape[-1]) != str(TRACK_FEATURE_SIZE):
                if not self._dynamic_dim(shape[0]):
                    return False, f"RD input {info.name} batch B must be dynamic"
                if "float" not in info.type:
                    return False, f"RD input {info.name} must be tensor(float32)"
                rank3_rd.append(info)

        if track_info is None:
            return False, (
                f"Model must contain a float32 track input [B,{TRACK_HISTORY_LENGTH},"
                f"{TRACK_FEATURE_SIZE}]"
            )
        self.track_input_name = track_info.name

        if rank4:
            ok, message = self._configure_rank4_rd(rank4)
        else:
            ok, message = self._configure_rank3_rd(rank3_rd)
        if not ok:
            return False, message

        metadata = self.sess.get_modelmeta().custom_metadata_map or {}

        for key in ("track_history", "sequence_length"):
            if key in metadata:
                try:
                    value = int(metadata[key])
                except (TypeError, ValueError):
                    return False, f"{key} must be an integer"
                if value != TRACK_HISTORY_LENGTH:
                    return False, f"{key} must be {TRACK_HISTORY_LENGTH}, got {value}"

        output_info = self._select_supported_output()
        if output_info is None:
            names = ", ".join(item.name for item in self.sess.get_outputs())
            return False, (
                "No supported output found. Use probabilities [B,C]/[B,T,C], "
                f"class_ids [B], or class_names [B]. Current outputs: {names}"
            )
        self.output_name = output_info.name
        self.output_mode = self._output_mode_for(output_info)

        if self.output_mode == "probabilities":
            class_dim = output_info.shape[-1]
            try:
                self.class_count = int(class_dim)
            except (TypeError, ValueError):
                return False, "Probability class dimension C must be a fixed integer"
            if self.class_count < 2:
                return False, "Probability class count must be at least 2"
            if not self.class_map:
                return False, (
                    "The model must declare classes metadata mapping IDs to names, "
                    'for example classes={"101":"drone","202":"bird"}'
                )
            column_ids = self._probability_column_ids(metadata)
            if column_ids is None:
                if len(self.class_ids) != self.class_count:
                    return False, (
                        "Probability columns do not match declared classes; add "
                        "prob_class_ids metadata listing the class ID of each "
                        "probability column, for example prob_class_ids=[101,202]"
                    )
                column_ids = list(self.class_ids)
            if len(column_ids) != self.class_count:
                return False, (
                    f"prob_class_ids length {len(column_ids)} does not match "
                    f"probability columns {self.class_count}"
                )
            undeclared = [cid for cid in column_ids if cid not in self.class_map]
            if undeclared:
                return False, f"prob_class_ids references undeclared class ids {undeclared}"
            self.class_ids = column_ids
            self.class_names = [self.class_map[cid] for cid in column_ids]
        elif self.output_mode == "class_id" and not self.class_map:
            return False, (
                "The model must declare classes metadata mapping IDs to names, "
                'for example classes={"303":"drone"}'
            )

        rd_modes = ",".join(spec["mode"] for spec in self.rd_inputs)
        return True, (
            f"Track+full-RD ONNX loaded: track={TRACK_HISTORY_LENGTH}, "
            f"rd={rd_modes}, output={self.output_mode}"
        )

    def _configure_rank4_rd(self, infos) -> tuple[bool, str]:
        if len(infos) == 1:
            info = infos[0]
            name = info.name.lower()
            channels = info.shape[1]
            fixed_channels = isinstance(channels, int)
            if fixed_channels and int(channels) not in (1, 2):
                return False, f"RD image input {info.name} must have 1 or 2 channels"
            if fixed_channels and int(channels) == 1 and "phase" in name and "amplitude" not in name:
                mode = "phase"
            elif fixed_channels and int(channels) == 1:
                mode = "amplitude"
            else:
                mode = "fused"
            self.rd_inputs = [{
                "name": info.name,
                "mode": mode,
                "channels": int(channels) if fixed_channels else 0,
                "shape_hw": self._fixed_hw(info, 2),
            }]
            return True, ""

        if len(infos) == 2:
            amp_info = next((item for item in infos if "amp" in item.name.lower()), None)
            phase_info = next((item for item in infos if "phase" in item.name.lower()), None)
            if amp_info is None or phase_info is None:
                amp_info, phase_info = infos
            for info, mode in ((amp_info, "amplitude"), (phase_info, "phase")):
                channels = info.shape[1]
                if isinstance(channels, int) and int(channels) != 1:
                    return False, f"Separate RD input {info.name} must have one channel"
                self.rd_inputs.append({
                    "name": info.name,
                    "mode": mode,
                    "channels": 1 if isinstance(channels, int) else 0,
                    "shape_hw": self._fixed_hw(info, 2),
                })
            return True, ""

        return False, "Model can contain one fused RD image or separate amplitude/phase inputs"

    def _configure_rank3_rd(self, infos) -> tuple[bool, str]:
        if len(infos) == 1:
            info = infos[0]
            mode = "phase_3d" if "phase" in info.name.lower() and "amp" not in info.name.lower() else "amplitude_3d"
            self.rd_inputs = [{
                "name": info.name,
                "mode": mode,
                "channels": 1,
                "shape_hw": self._fixed_hw(info, 1),
            }]
            return True, ""
        if len(infos) == 2:
            amp_info = next((item for item in infos if "amp" in item.name.lower()), None)
            phase_info = next((item for item in infos if "phase" in item.name.lower()), None)
            if amp_info is None or phase_info is None:
                amp_info, phase_info = infos
            for info, mode in ((amp_info, "amplitude_3d"), (phase_info, "phase_3d")):
                self.rd_inputs.append({
                    "name": info.name,
                    "mode": mode,
                    "channels": 1,
                    "shape_hw": self._fixed_hw(info, 1),
                })
            return True, ""
        return False, "Unsupported number of rank-3 RD inputs"

    @staticmethod
    def _dynamic_dim(value) -> bool:
        return not isinstance(value, int) or int(value) <= 0

    @staticmethod
    def _fixed_hw(info, offset: int) -> tuple[int, int] | None:
        shape = tuple(getattr(info, "shape", ()))
        height = shape[offset]
        width = shape[offset + 1]
        if isinstance(height, int) and isinstance(width, int):
            return int(height), int(width)
        return None

    def _select_supported_output(self):
        outputs = self.sess.get_outputs()
        preferred = ("probabilities", "probs", "scores", "class_ids",
                     "labels", "class_names", "classes")
        for name in preferred:
            for output in outputs:
                if output.name.lower() == name:
                    mode = self._output_mode_for(output)
                    if mode:
                        return output
        for output in outputs:
            if self._output_mode_for(output):
                return output
        return None

    @staticmethod
    def _output_mode_for(output) -> str:
        shape = tuple(getattr(output, "shape", ()))
        output_type = getattr(output, "type", "").lower()
        name = output.name.lower()
        if "string" in output_type and len(shape) in (1, 2):
            return "class_name"
        if "float" in output_type and len(shape) in (2, 3):
            return "probabilities"
        if len(shape) == 1 and any(kind in output_type for kind in ("int", "float", "tensor")):
            if "float" in output_type or name in ("class_ids", "labels", "classes", "class"):
                return "class_id"
        if len(shape) == 1 and name in ("class_ids", "labels", "classes"):
            return "class_id"
        return ""

    def _parse_outputs(self, outputs, track_ids) -> dict[int, tuple[str, float]]:
        for info, value in zip(self.sess.get_outputs(), outputs):
            if info.name != self.output_name:
                continue
            array = np.asarray(value)
            if self.output_mode == "probabilities":
                return self._labels_from_probabilities(array, track_ids)
            if self.output_mode == "class_id":
                return self._labels_from_ids(array, track_ids)
            if self.output_mode == "class_name":
                return self._labels_from_names(array, track_ids)
        raise ValueError(f"Output {self.output_name} was not returned")

    def _labels_from_probabilities(self, array, track_ids) -> dict[int, tuple]:
        if array.ndim == 3:
            array = array[:, -1, :]
        if array.ndim != 2:
            raise ValueError(f"Probability output must be [B,C] or [B,T,C], got {array.shape}")
        if array.shape[0] != len(track_ids):
            raise ValueError(f"Output batch {array.shape[0]} does not match {len(track_ids)} tracks")
        if not np.isfinite(array).all():
            raise ValueError("Probability output contains NaN or Inf")
        if np.any(array < 0.0) or np.any(array > 1.0):
            shifted = array - np.max(array, axis=-1, keepdims=True)
            exp = np.exp(shifted)
            array = exp / np.sum(exp, axis=-1, keepdims=True)

        labels = {}
        for track_id, scores in zip(track_ids, array):
            index = int(np.argmax(scores))
            class_id = self.class_ids[index]
            labels[int(track_id)] = self.class_map[class_id], float(scores[index])
        return labels

    def _labels_from_ids(self, array, track_ids) -> dict[int, tuple]:
        array = np.asarray(array).reshape(-1)
        if array.size != len(track_ids):
            raise ValueError(f"Class id output size {array.size} does not match {len(track_ids)} tracks")
        labels = {}
        for track_id, value in zip(track_ids, array):
            raw_id = int(round(float(value)))
            name = self.class_map.get(raw_id)
            if name is None:
                raise ValueError(f"Model returned undeclared class id {raw_id}")
            labels[int(track_id)] = name, 1.0
        return labels

    def _labels_from_names(self, array, track_ids) -> dict[int, tuple]:
        flat = np.asarray(array).reshape(-1)
        if flat.size != len(track_ids):
            raise ValueError(f"Class name output size {flat.size} does not match {len(track_ids)} tracks")
        return {
            int(track_id): (str(name), 1.0)
            for track_id, name in zip(track_ids, flat)
        }

    def _probability_column_ids(self, metadata):
        """Optional explicit mapping from probability column index to class ID."""
        for key in ("prob_class_ids", "prob_ids", "column_ids"):
            if key not in metadata:
                continue
            raw = self._json_value(metadata[key])
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"{key} must be a non-empty list of integer ids")
            ids = []
            for item in raw:
                try:
                    ids.append(int(str(item).strip()))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} entries must be integers") from exc
            if len(set(ids)) != len(ids):
                raise ValueError(f"{key} contains duplicate ids")
            return ids
        return None

    def _read_class_mapping(self) -> tuple[list[int], list[str], dict[int, str]]:
        """Read model-declared class IDs and display names.

        Preferred metadata is a JSON object ``classes`` / ``label_map`` /
        ``class_map`` mapping integer IDs to names. A list of IDs plus aligned
        names is also accepted. No client-defined zero/one-based default is used.
        """
        try:
            metadata = self.sess.get_modelmeta().custom_metadata_map or {}
        except Exception:
            metadata = {}

        for key in ("classes", "label_map", "class_map", "id_to_name"):
            if metadata.get(key):
                raw_object = self._json_value(metadata[key])
                if isinstance(raw_object, dict):
                    return self._mapping_from_object(raw_object)

        class_ids_raw = None
        for key in ("class_ids", "label_ids", "ids"):
            if key in metadata:
                class_ids_raw = self._json_value(metadata[key])
                break
        names_raw = None
        for key in ("class_names", "labels", "names"):
            if key in metadata:
                names_raw = self._json_value(metadata[key])
                break

        if class_ids_raw is not None and names_raw is not None:
            return self._mapping_from_parallel_lists(class_ids_raw, names_raw)
        return [], [], {}

    @staticmethod
    def _json_value(raw):
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return raw

    @staticmethod
    def _mapping_from_object(raw: dict) -> tuple[list[int], list[str], dict[int, str]]:
        mapping: dict[int, str] = {}
        for key, value in raw.items():
            try:
                class_id = int(str(key).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Class id must be an integer, got {key!r}") from exc
            name = str(value).strip()
            if not name:
                raise ValueError(f"Class id {class_id} has an empty name")
            if class_id in mapping:
                raise ValueError(f"Duplicate class id {class_id}")
            mapping[class_id] = name
        return list(mapping.keys()), list(mapping.values()), mapping

    @staticmethod
    def _mapping_from_parallel_lists(ids_raw, names_raw) -> tuple[list[int], list[str], dict[int, str]]:
        if not isinstance(ids_raw, list) or not isinstance(names_raw, list):
            raise ValueError("class_ids and class_names must both be lists")
        if len(ids_raw) != len(names_raw) or not ids_raw:
            raise ValueError("class_ids and class_names must be non-empty and aligned")
        raw = {str(ids_raw[index]): names_raw[index] for index in range(len(ids_raw))}
        return InferenceBackend._mapping_from_object(raw)

    def _prune_old_histories(self, frame_id: int, current_keys: set):
        if frame_id < 200:
            return
        dead = [
            key for key, history in self.histories.items()
            if key not in current_keys and frame_id - history["last_seen"] > 100
        ]
        for key in dead:
            self.histories.pop(key, None)
