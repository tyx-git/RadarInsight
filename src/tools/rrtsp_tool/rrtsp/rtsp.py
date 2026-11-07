# -*- coding: utf-8 -*-
"""RTSP 文本信令与 SDP 描述。"""
from __future__ import annotations

from dataclasses import dataclass

PROTOCOL = "RTSP/1.0"
DEFAULT_PORT = 8554
RESOURCE = "radar"

STATUS_TEXT = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Valid",
    454: "Session Not Found",
    455: "Method Not Valid In This State",
    461: "Unsupported Transport",
    500: "Internal Server Error",
    501: "Not Implemented",
}


@dataclass
class Request:
    method: str
    uri: str
    headers: dict[str, str]


def _read_line(conn) -> str:
    data = bytearray()
    while True:
        ch = conn.recv(1)
        if not ch:
            break
        if ch == b"\n":
            break
        data.extend(ch)
    return data.decode("utf-8", "replace").rstrip("\r")


def read_request(conn) -> Request | None:
    line = _read_line(conn)
    if not line:
        return None
    parts = line.split()
    if len(parts) < 3:
        raise ValueError("无效 RTSP 请求行")
    method, uri, version = parts[0], parts[1], parts[2]
    if not version.startswith("RTSP/"):
        raise ValueError("不是 RTSP 请求")

    headers: dict[str, str] = {}
    while True:
        line = _read_line(conn)
        if not line:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return Request(method=method.upper(), uri=uri, headers=headers)


def send_response(conn, status: int, *, cseq: str = "",
                  headers: dict[str, str] | None = None,
                  body: bytes = b"", content_type: str | None = None) -> None:
    reason = STATUS_TEXT.get(status, "Error")
    lines = [
        f"{PROTOCOL} {status} {reason}",
        f"CSeq: {cseq}",
    ]
    if body:
        lines.append(f"Content-Length: {len(body)}")
        if content_type:
            lines.append(f"Content-Type: {content_type}")
    for key, value in (headers or {}).items():
        lines.append(f"{key}: {value}")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
    conn.sendall(head + body)


def parse_transport(value: str) -> dict:
    """解析 RTP/AVP;unicast;client_port=5004-5005。"""
    out: dict[str, str] = {}
    for token in value.split(";"):
        token = token.strip()
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip().lower()] = v.strip()
        elif token:
            out[token.lower()] = ""
    return out


def media_track_id(uri: str) -> int | None:
    lowered = uri.lower()
    marker = "trackid="
    if marker in lowered:
        tail = lowered.split(marker, 1)[1]
        number = tail.split("&", 1)[0]
        if number.isdigit():
            return int(number)
    return None


def base_url(request_uri: str, host_header: str, server_host: str,
             server_port: int) -> str:
    if host_header:
        host = host_header
    else:
        host = f"{'127.0.0.1' if server_host in ('0.0.0.0', '') else server_host}:{server_port}"
    return f"rtsp://{host}/{RESOURCE}"


def build_sdp(*, base: str, server_ip: str, ssrc_rd: int, ssrc_track: int,
              has_rd: bool, has_track: bool, frame_rate: float,
              max_packet_size: int = 1200,
              rd_shape: tuple[int, int] | None = None) -> str:
    lines = [
        "v=0",
        f"o=radar {ssrc_track} 1 IN IP4 {server_ip or '127.0.0.1'}",
        "s=R-RTSP Radar Stream",
        "t=0 0",
        "a=control:*",
        "a=range:npt=0-",
    ]

    if has_rd:
        range_bins, doppler_bins = rd_shape or (0, 0)
        rd_fmt = (
            f"numRangeBins={range_bins};numDopplerBins={doppler_bins};"
            "bitsPerSample=32;"
            "quant=linear;compression=none;"
            f"frameRate={frame_rate:g};maxPacketSize={max_packet_size};"
            "samples=amplitude,phase")
        lines.extend([
            "m=application 5004 RTP/AVP 96",
            "a=control:trackID=0",
            "a=rtpmap:96 radar-rdmap/90000",
            f"a=fmtp:96 {rd_fmt}",
            f"a=ssrc:{ssrc_rd:08x} cname:radar@aerosight",
        ])

    if has_track:
        fields = ",".join([
            "range", "azimuth", "elevation", "radial_speed",
            "speed", "course", "snr", "amplitude", "class"])
        track_fmt = (
            "encoding=tlv;updateRate="
            f"{frame_rate:g};coord=polar;fields={fields}")
        lines.extend([
            "m=application 5006 RTP/AVP 97",
            "a=control:trackID=1",
            "a=rtpmap:97 radar-track/90000",
            f"a=fmtp:97 {track_fmt}",
            f"a=ssrc:{ssrc_track:08x} cname:radar@aerosight",
        ])

    return "\r\n".join(lines) + "\r\n"


def header_get(headers: dict[str, str], name: str, default: str = "") -> str:
    return headers.get(name.lower(), default)
