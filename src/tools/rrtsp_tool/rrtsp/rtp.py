# -*- coding: utf-8 -*-
"""RTP/RTCP 二进制头部工具。

仅实现 R-RTSP v1 需要的最小、可互操作子集：
- RTP fixed header（RFC 3550）
- RTCP Sender Report / Receiver Report / APP
"""
from __future__ import annotations

import struct
import time

RTP_VERSION = 2
PT_RTCP_SR = 200
PT_RTCP_RR = 201
PT_RTCP_APP = 204


def build_rtp_packet(*, payload_type: int, ssrc: int, sequence: int,
                     timestamp: int, payload: bytes,
                     marker: bool = False) -> bytes:
    """构造一个 RTP 包。CSRC=0，无扩展。"""
    first = (RTP_VERSION << 6)
    second = (0x80 if marker else 0) | (payload_type & 0x7F)
    return struct.pack(
        ">BBHII", first, second, sequence & 0xFFFF,
        timestamp & 0xFFFFFFFF, ssrc & 0xFFFFFFFF) + payload


def parse_rtp_packet(data: bytes) -> dict:
    if len(data) < 12:
        raise ValueError("RTP 包长度不足")
    first, second, seq, ts, ssrc = struct.unpack(">BBHII", data[:12])
    cc = first & 0x0F
    header_len = 12 + 4 * cc
    return {
        "version": first >> 6,
        "padding": bool(first & 0x20),
        "extension": bool(first & 0x10),
        "csrc_count": cc,
        "marker": bool(second & 0x80),
        "payload_type": second & 0x7F,
        "sequence": seq,
        "timestamp": ts,
        "ssrc": ssrc,
        "payload": data[header_len:],
    }


def ntp_now() -> tuple[int, int]:
    """当前 UTC 时间 -> NTP 64 位时间戳的高/低 32 位。"""
    unix_seconds = time.time()
    ntp_seconds = int(unix_seconds) + 2208988800
    fraction = int((unix_seconds - int(unix_seconds)) * (1 << 32))
    return ntp_seconds, fraction & 0xFFFFFFFF


def build_sender_report(*, ssrc: int, rtp_timestamp: int,
                        packet_count: int, octet_count: int) -> bytes:
    ntp_hi, ntp_lo = ntp_now()
    return struct.pack(
        ">BBHIIIIII",
        (RTP_VERSION << 6), PT_RTCP_SR, 6, ssrc,
        ntp_hi, ntp_lo, rtp_timestamp, packet_count, octet_count)


def build_receiver_report(*, reporter_ssrc: int, source_ssrc: int,
                          highest_sequence: int, jitter: int,
                          fraction_lost: int = 0, packets_lost: int = 0,
                          last_sr: int = 0, delay_since_last_sr: int = 0,
                          report_blocks: int = 1) -> bytes:
    lost_word = ((fraction_lost & 0xFF) << 24) | (packets_lost & 0x00FFFFFF)
    return struct.pack(
        ">BBHIIIIIII",
        (RTP_VERSION << 6) | (report_blocks & 0x1F), PT_RTCP_RR, 7,
        reporter_ssrc, source_ssrc, lost_word, highest_sequence,
        jitter, last_sr, delay_since_last_sr)


def build_app_packet(*, ssrc: int, name: str, data: bytes,
                     subtype: int = 0) -> bytes:
    if len(name) != 4:
        raise ValueError("RTCP APP name 必须为 4 字节")
    padding = (-len(data)) % 4
    body = data + (b"\0" * padding)
    length = (len(body) // 4) + 1
    return struct.pack(">BBHI", (RTP_VERSION << 6) | (subtype & 0x1F),
                       PT_RTCP_APP, length, ssrc) + name.encode("ascii") + body


def parse_rtcp(data: bytes) -> list[dict]:
    """解析 SR/RR/APP；对未知类型返回头部信息。"""
    out = []
    offset = 0
    while offset + 4 <= len(data):
        first, pt, words = struct.unpack(">BBH", data[offset:offset + 4])
        length = (words + 1) * 4
        if offset + length > len(data):
            raise ValueError("RTCP 长度字段越界")
        item = {"type": pt, "ssrc": None, "raw": data[offset:offset + length]}
        if length >= 8 and pt in (PT_RTCP_SR, PT_RTCP_RR, PT_RTCP_APP):
            item["ssrc"] = struct.unpack(">I", data[offset + 4:offset + 8])[0]
        if pt == PT_RTCP_APP and length >= 12:
            item["name"] = data[offset + 8:offset + 12].decode("ascii", "replace")
            item["data"] = data[offset + 12:offset + length]
        out.append(item)
        offset += length
    return out
