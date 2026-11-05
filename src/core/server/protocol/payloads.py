"""R-RTSP v1 RTP payload encoders."""

from __future__ import annotations

import math
import struct
from typing import Mapping

import numpy as np


RD_HEADER = struct.Struct(">BBHIHHBBHf")
TRACK_FRAME_HEADER = struct.Struct(">BHBB")
TLV_HEADER = struct.Struct(">BH")
TRACK_HEADER = struct.Struct(">IBBI")
INFERENCE_RESULT = struct.Struct(">IBf")

RD_FRAME_KEY = 3
TRACK_STATE_NAMES = {
    "suspect": 0,
    "confirmed": 1,
    "coasted": 2,
    "dropped": 3,
}
CLASS_TO_ID = {
    "drone": 1,
    "dronetarget": 1,
    "bird": 2,
    "birdtarget": 2,
    "balloon": 3,
    "balloontarget": 3,
    "clutter": 4,
    "cluttertarget": 4,
    "unknown": 5,
    "unknowntarget": 5,
    "unknown-rejected": 5,
    "other": 6,
}

# This order is part of the PT=97 FSPEC contract.
TRACK_FIELDS = (
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_speed_mps",
    "speed_mps",
    "course_deg",
    "snr_db",
    "amplitude_db",
)


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def build_rd_header(
    *,
    frame_sequence: int,
    frame_id: int,
    range_bins: int,
    doppler_bins: int,
    with_phase: bool,
    bits_per_sample: int = 32,
    quant: int = 0,
    compression: int = 0,
    scale: float = 1.0,
) -> bytes:
    if not 1 <= int(range_bins) <= 0xFFFF or not 1 <= int(doppler_bins) <= 0xFFFF:
        raise ValueError("RD dimensions must fit uint16")
    if not math.isfinite(float(scale)):
        raise ValueError("RD scale must be finite")
    flags = 0x04  # little-endian float32 samples
    if with_phase:
        flags |= 0x02
    return RD_HEADER.pack(
        RD_FRAME_KEY,
        flags,
        int(frame_sequence) & 0xFFFF,
        int(frame_id) & 0xFFFFFFFF,
        int(range_bins),
        int(doppler_bins),
        int(bits_per_sample),
        int(quant),
        int(compression),
        float(scale),
    )


def fragment_rd_frame(
    *,
    amplitude: np.ndarray,
    phase: np.ndarray | None,
    frame_sequence: int,
    frame_id: int,
    max_packet_size: int = 1200,
) -> list[bytes]:
    """Serialize one RD frame into RTP payload fragments."""

    amp = np.asarray(amplitude, dtype=np.float32)
    if amp.ndim != 2 or min(amp.shape) <= 0:
        raise ValueError(f"RD amplitude must be a non-empty 2D array: {amp.shape}")
    if not np.isfinite(amp).all():
        raise ValueError("RD amplitude contains non-finite values")
    phase_array: np.ndarray | None = None
    if phase is not None:
        phase_array = np.asarray(phase, dtype=np.float32)
        if phase_array.shape != amp.shape:
            raise ValueError("RD amplitude and phase shapes differ")
        if not np.isfinite(phase_array).all():
            raise ValueError("RD phase contains non-finite values")
    if int(max_packet_size) <= RD_HEADER.size + 1:
        raise ValueError("max_packet_size is too small for an RD header")
    sample_bytes = np.ascontiguousarray(amp, dtype="<f4").tobytes()
    if phase_array is not None:
        sample_bytes += np.ascontiguousarray(phase_array, dtype="<f4").tobytes()
    header = build_rd_header(
        frame_sequence=frame_sequence,
        frame_id=frame_id,
        range_bins=amp.shape[0],
        doppler_bins=amp.shape[1],
        with_phase=phase_array is not None,
    )
    chunk_size = int(max_packet_size) - RD_HEADER.size
    return [header + sample_bytes[offset:offset + chunk_size]
            for offset in range(0, len(sample_bytes), chunk_size)]


def encode_track(
    track_id: int,
    state: int,
    class_id: int,
    values: Mapping[str, object],
) -> bytes:
    """Encode one PT=97 track TLV, omitting absent/non-finite fields."""

    fspec = 0
    body = bytearray()
    for bit, name in enumerate(TRACK_FIELDS):
        value = _finite(values.get(name))
        if value is not None:
            fspec |= 1 << bit
            body.extend(struct.pack(">f", value))
    value = TRACK_HEADER.pack(
        int(track_id) & 0xFFFFFFFF,
        int(state) & 0xFF,
        int(class_id) & 0xFF,
        fspec,
    ) + bytes(body)
    return TLV_HEADER.pack(1, len(value)) + value


def build_track_frame(
    tracks: list[bytes],
    *,
    update_sequence: int,
    frame_type: int = 1,
) -> bytes:
    if len(tracks) > 0xFFFF:
        raise ValueError("too many tracks in one update")
    return TRACK_FRAME_HEADER.pack(
        int(frame_type) & 0xFF,
        len(tracks),
        int(update_sequence) & 0xFF,
        0,
    ) + b"".join(tracks)


def encode_inference_result(
    track_id: int,
    class_id: int,
    confidence: float,
) -> bytes:
    """Encode one classification result as a type-2 TLV."""

    score = float(confidence)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("inference confidence must be finite and in [0, 1]")
    if not 1 <= int(class_id) <= 6:
        raise ValueError("inference class_id must be in [1, 6]")
    value = INFERENCE_RESULT.pack(
        int(track_id) & 0xFFFFFFFF,
        int(class_id),
        score,
    )
    return TLV_HEADER.pack(2, len(value)) + value


def build_inference_frame(
    results: list[Mapping[str, object]],
    *,
    update_sequence: int,
) -> bytes:
    tlvs = [
        encode_inference_result(
            int(result["track_id"]),
            int(result["class_id"]),
            float(result["confidence"]),
        )
        for result in results
    ]
    return build_track_frame(tlvs, update_sequence=update_sequence, frame_type=2)


__all__ = [
    "CLASS_TO_ID",
    "RD_HEADER",
    "INFERENCE_RESULT",
    "TRACK_FIELDS",
    "TRACK_STATE_NAMES",
    "build_rd_header",
    "build_track_frame",
    "build_inference_frame",
    "encode_track",
    "encode_inference_result",
    "fragment_rd_frame",
]
