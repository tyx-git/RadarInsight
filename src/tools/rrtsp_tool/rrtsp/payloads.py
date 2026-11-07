# -*- coding: utf-8 -*-
"""R-RTSP v1 自定义 RTP 载荷：RD 图谱与航迹 TLV。"""
from __future__ import annotations

import math
import struct

RD_HEADER = struct.Struct(">BBHIHHBBHf")
TRACK_FRAME_HEADER = struct.Struct(">BHBB")
TLV_HEADER = struct.Struct(">BH")
TRACK_HEAD = struct.Struct(">IBBI")

RD_FRAME_FULL = 1
RD_FRAME_KEY = 3

TRACK_STATE_NAMES = {
    "suspect": 0,
    "confirmed": 1,
    "coasted": 2,
    "dropped": 3,
}

CLASS_TO_ID = {
    "drone": 1,
    "bird": 2,
    "balloon": 3,
    "clutter": 4,
    "unknown-l2": 5,
    "other": 6,
}

# FSPEC 位定义。SDP fields 必须保持同一顺序。
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


def _finite_float(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def build_rd_header(*, frame_sequence: int, frame_id: int,
                    range_bins: int, doppler_bins: int,
                    with_phase: bool, bits_per_sample: int = 32,
                    quant: int = 0, compression: int = 0,
                    scale: float = 1.0) -> bytes:
    flags = 0x04  # 样本为小端 float32
    if with_phase:
        flags |= 0x02
    return RD_HEADER.pack(
        RD_FRAME_KEY, flags, frame_sequence, frame_id,
        range_bins, doppler_bins, bits_per_sample, quant,
        compression, scale)


def parse_rd_header(data: bytes) -> dict:
    values = RD_HEADER.unpack(data[:RD_HEADER.size])
    keys = ("frame_type", "flags", "frame_sequence", "frame_id",
            "range_bins", "doppler_bins", "bits_per_sample",
            "quant", "compression", "scale")
    return dict(zip(keys, values))


def fragment_rd_frame(*, amplitude, phase=None, frame_sequence: int,
                      frame_id: int, max_packet_size: int = 1200) -> list[bytes]:
    """把一帧 RD 图切成多个 RTP payload。每个分片都携带完整头部。"""
    amp = amplitude.astype("<f4", copy=False)
    expected_shape = amp.shape
    sample_bytes = amp.tobytes()
    with_phase = phase is not None
    if with_phase:
        ph = phase.astype("<f4", copy=False)
        if ph.shape != expected_shape:
            raise ValueError("幅度和相位矩阵形状不一致")
        sample_bytes += ph.tobytes()

    if max_packet_size <= RD_HEADER.size + 1:
        raise ValueError("maxPacketSize 太小")
    chunk_size = max_packet_size - RD_HEADER.size
    header = build_rd_header(
        frame_sequence=frame_sequence,
        frame_id=frame_id,
        range_bins=expected_shape[0],
        doppler_bins=expected_shape[1],
        with_phase=with_phase)

    chunks = []
    for offset in range(0, len(sample_bytes), chunk_size):
        chunks.append(header + sample_bytes[offset:offset + chunk_size])
    return chunks


def encode_track(track_id: int, state: int, class_id: int,
                 values: dict[str, float]) -> bytes:
    """按 FSPEC 编码单个 Track TLV。"""
    fspec = 0
    body = bytearray()
    for bit, name in enumerate(TRACK_FIELDS):
        if name in values and _finite_float(values[name]):
            fspec |= 1 << bit
            body.extend(struct.pack(">f", float(values[name])))

    value = TRACK_HEAD.pack(track_id, state, class_id, fspec) + bytes(body)
    # v1 只定义 type=1
    return TLV_HEADER.pack(1, len(value)) + value


def build_track_frame(tracks: list[bytes], update_sequence: int,
                      frame_type: int = 1) -> bytes:
    return TRACK_FRAME_HEADER.pack(
        frame_type, len(tracks), update_sequence & 0xFF, 0) + b"".join(tracks)


def parse_track_frame(payload: bytes) -> dict:
    frame_type, n_tracks, seq, _reserved = TRACK_FRAME_HEADER.unpack_from(payload)
    offset = TRACK_FRAME_HEADER.size
    tracks = []
    for _ in range(n_tracks):
        tlv_type, length = TLV_HEADER.unpack_from(payload, offset)
        offset += TLV_HEADER.size
        value = payload[offset:offset + length]
        offset += length
        track_id, state, class_id, fspec = TRACK_HEAD.unpack_from(value)
        values = {}
        p = TRACK_HEAD.size
        for bit, name in enumerate(TRACK_FIELDS):
            if fspec & (1 << bit):
                values[name] = struct.unpack(">f", value[p:p + 4])[0]
                p += 4
        tracks.append({
            "type": tlv_type,
            "track_id": track_id,
            "state": state,
            "class_id": class_id,
            "fspec": fspec,
            "values": values,
        })
    return {"frame_type": frame_type, "sequence": seq, "tracks": tracks}
