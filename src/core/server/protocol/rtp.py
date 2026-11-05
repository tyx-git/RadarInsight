"""RTP and the small RTCP subset required by R-RTSP v1."""

from __future__ import annotations

import struct
import time
from typing import Any


RTP_VERSION = 2
RTP_HEADER = struct.Struct(">BBHII")
RTCP_HEADER = struct.Struct(">BBH")
PT_RTCP_SR = 200
PT_RTCP_RR = 201


def build_rtp_packet(
    *,
    payload_type: int,
    ssrc: int,
    sequence: int,
    timestamp: int,
    payload: bytes,
    marker: bool = False,
) -> bytes:
    """Build an RTP packet without CSRCs or header extensions."""

    if not 0 <= int(payload_type) <= 127:
        raise ValueError("RTP payload_type must be in [0, 127]")
    if not 0 <= int(sequence) <= 0xFFFF:
        raise ValueError("RTP sequence must be a uint16")
    if not 0 <= int(timestamp) <= 0xFFFFFFFF:
        raise ValueError("RTP timestamp must be a uint32")
    first = RTP_VERSION << 6
    second = (0x80 if marker else 0) | int(payload_type)
    return RTP_HEADER.pack(
        first,
        second,
        int(sequence) & 0xFFFF,
        int(timestamp) & 0xFFFFFFFF,
        int(ssrc) & 0xFFFFFFFF,
    ) + bytes(payload)


def parse_rtp_packet(data: bytes) -> dict[str, Any]:
    """Parse an RTP packet and return its payload and header fields."""

    if len(data) < RTP_HEADER.size:
        raise ValueError("RTP packet is shorter than the fixed header")
    first, second, sequence, timestamp, ssrc = RTP_HEADER.unpack_from(data)
    if first >> 6 != RTP_VERSION:
        raise ValueError("unsupported RTP version")
    csrc_count = first & 0x0F
    offset = RTP_HEADER.size + 4 * csrc_count
    if len(data) < offset:
        raise ValueError("RTP CSRC list is truncated")
    has_extension = bool(first & 0x10)
    extension_length = 0
    if has_extension:
        if len(data) < offset + 4:
            raise ValueError("RTP header extension is truncated")
        extension_words = struct.unpack_from(">H", data, offset + 2)[0]
        extension_length = 4 + 4 * extension_words
        if len(data) < offset + extension_length:
            raise ValueError("RTP header extension payload is truncated")
        offset += extension_length
    payload = data[offset:]
    if first & 0x20:
        if not payload:
            raise ValueError("RTP padding flag set on an empty payload")
        padding = payload[-1]
        if padding == 0 or padding > len(payload):
            raise ValueError("invalid RTP padding")
        payload = payload[:-padding]
    return {
        "version": first >> 6,
        "marker": bool(second & 0x80),
        "payload_type": second & 0x7F,
        "sequence": sequence,
        "timestamp": timestamp,
        "ssrc": ssrc,
        "payload": payload,
        "csrc_count": csrc_count,
        "header_extension_bytes": extension_length,
    }


def _ntp_timestamp(now: float | None = None) -> tuple[int, int]:
    current = time.time() if now is None else float(now)
    seconds = int(current) + 2208988800
    fraction = int((current - int(current)) * (1 << 32)) & 0xFFFFFFFF
    return seconds & 0xFFFFFFFF, fraction


def build_sender_report(
    *,
    ssrc: int,
    rtp_timestamp: int,
    packet_count: int,
    octet_count: int,
    now: float | None = None,
) -> bytes:
    """Build an RTCP Sender Report with no reception report blocks."""

    ntp_high, ntp_low = _ntp_timestamp(now)
    body = struct.pack(">IIIII", int(ssrc), ntp_high, ntp_low,
                       int(rtp_timestamp) & 0xFFFFFFFF,
                       int(packet_count) & 0xFFFFFFFF)
    body += struct.pack(">I", int(octet_count) & 0xFFFFFFFF)
    # length is measured in 32-bit words after the common header.
    return RTCP_HEADER.pack(0x80, PT_RTCP_SR, (len(body) + 4) // 4 - 1) + body


def build_receiver_report(
    *,
    reporter_ssrc: int,
    source_ssrc: int,
    highest_sequence: int,
    jitter: int = 0,
) -> bytes:
    """Build a minimal RTCP Receiver Report with one source block."""

    body = struct.pack(">II", int(reporter_ssrc), int(source_ssrc))
    body += struct.pack(">I", 0)  # fraction lost + cumulative lost
    body += struct.pack(">I", int(highest_sequence) & 0xFFFFFFFF)
    body += struct.pack(">III", int(jitter) & 0xFFFFFFFF, 0, 0)
    return RTCP_HEADER.pack(0x81, PT_RTCP_RR, (len(body) + 4) // 4 - 1) + body


def parse_rtcp(data: bytes) -> list[dict[str, int]]:
    """Parse enough RTCP to identify Sender and Receiver Reports."""

    packets: list[dict[str, int]] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < RTCP_HEADER.size:
            raise ValueError("truncated RTCP header")
        first, packet_type, length_words = RTCP_HEADER.unpack_from(data, offset)
        if first >> 6 != RTP_VERSION:
            raise ValueError("unsupported RTCP version")
        size = (int(length_words) + 1) * 4
        if size < 4 or offset + size > len(data):
            raise ValueError("invalid RTCP packet length")
        packets.append({
            "version": first >> 6,
            "count": first & 0x1F,
            "type": int(packet_type),
            "length": size,
        })
        offset += size
    return packets


__all__ = [
    "PT_RTCP_RR",
    "PT_RTCP_SR",
    "RTP_HEADER",
    "build_receiver_report",
    "build_rtp_packet",
    "build_sender_report",
    "parse_rtcp",
    "parse_rtp_packet",
]
