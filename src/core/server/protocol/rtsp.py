"""RTSP/1.0 signaling and SDP generation for R-RTSP v1."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


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
    503: "Service Unavailable",
}


@dataclass(frozen=True)
class Request:
    method: str
    uri: str
    headers: dict[str, str]
    body: bytes = b""


def _read_line(conn, *, max_bytes: int = 64 * 1024) -> str:
    data = bytearray()
    while len(data) < max_bytes:
        char = conn.recv(1)
        if not char:
            break
        if char == b"\n":
            break
        data.extend(char)
    if len(data) >= max_bytes:
        raise ValueError("RTSP line is too long")
    return data.decode("utf-8", "replace").rstrip("\r")


def read_request(conn) -> Request | None:
    line = _read_line(conn)
    if not line:
        return None
    parts = line.split()
    if len(parts) != 3 or not parts[2].startswith("RTSP/"):
        raise ValueError("invalid RTSP request line")
    headers: dict[str, str] = {}
    while True:
        current = _read_line(conn)
        if not current:
            break
        if ":" not in current:
            raise ValueError("invalid RTSP header")
        key, value = current.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    try:
        length = int(headers.get("content-length", "0"))
    except ValueError as exc:
        raise ValueError("invalid RTSP Content-Length") from exc
    if length < 0 or length > 4 * 1024 * 1024:
        raise ValueError("RTSP request body is too large")
    body = b""
    while len(body) < length:
        chunk = conn.recv(length - len(body))
        if not chunk:
            raise ConnectionError("RTSP connection closed while reading body")
        body += chunk
    return Request(parts[0].upper(), parts[1], headers, body)


def send_response(
    conn,
    status: int,
    *,
    cseq: str = "",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
    content_type: str | None = None,
) -> None:
    lines = [f"{PROTOCOL} {int(status)} {STATUS_TEXT.get(status, 'Error')}"]
    if cseq:
        lines.append(f"CSeq: {cseq}")
    if body:
        lines.append(f"Content-Length: {len(body)}")
        if content_type:
            lines.append(f"Content-Type: {content_type}")
    for key, value in (headers or {}).items():
        lines.append(f"{key}: {value}")
    conn.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8") + body)


def parse_transport(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in str(value).split(";"):
        token = token.strip()
        if "=" in token:
            key, item = token.split("=", 1)
            result[key.strip().lower()] = item.strip()
        elif token:
            result[token.lower()] = ""
    return result


def media_track_id(uri: str) -> int | None:
    path = urlparse(uri).path.lower()
    marker = "trackid="
    if marker not in path:
        return None
    tail = path.split(marker, 1)[1].split("/", 1)[0]
    return int(tail) if tail.isdigit() else None


def request_path(uri: str) -> str:
    return urlparse(uri).path.rstrip("/") or "/"


def base_url(request_uri: str, host_header: str, server_host: str, server_port: int) -> str:
    if host_header:
        host = host_header
    else:
        host = f"{'127.0.0.1' if server_host in ('0.0.0.0', '') else server_host}:{server_port}"
    path = request_path(request_uri)
    if path == "/":
        path = f"/{RESOURCE}"
    return f"rtsp://{host}{path}"


def build_sdp(
    *,
    base: str,
    server_ip: str,
    ssrc_rd: int,
    ssrc_track: int,
    has_rd: bool,
    has_track: bool,
    update_rate_hz: float,
    max_packet_size: int = 1200,
    rd_shape: tuple[int, int] | None = None,
    has_inference: bool = False,
    ssrc_inference: int = 0,
) -> str:
    lines = [
        "v=0",
        f"o=radar {int(ssrc_track)} 1 IN IP4 {server_ip or '127.0.0.1'}",
        "s=R-RTSP Radar Stream",
        "t=0 0",
        "a=control:*",
        "a=range:npt=0-",
    ]
    if has_rd:
        rows, columns = rd_shape or (0, 0)
        fmtp = (
            f"numRangeBins={int(rows)};numDopplerBins={int(columns)};"
            "bitsPerSample=32;quant=linear;compression=none;"
            f"frameRate={float(update_rate_hz):g};maxPacketSize={int(max_packet_size)};"
            "samples=amplitude,phase"
        )
        lines.extend([
            "m=application 5004 RTP/AVP 96",
            "a=control:trackID=0",
            "a=rtpmap:96 radar-rdmap/90000",
            f"a=fmtp:96 {fmtp}",
            f"a=ssrc:{int(ssrc_rd):08x} cname:radar@radarinsight",
        ])
    if has_track:
        fields = ",".join([
            "range", "azimuth", "elevation", "radial_speed", "speed",
            "course", "snr", "amplitude", "class",
        ])
        fmtp = f"encoding=tlv;updateRate={float(update_rate_hz):g};coord=polar;fields={fields}"
        lines.extend([
            "m=application 5006 RTP/AVP 97",
            "a=control:trackID=1",
            "a=rtpmap:97 radar-track/90000",
            f"a=fmtp:97 {fmtp}",
            f"a=ssrc:{int(ssrc_track):08x} cname:radar@radarinsight",
        ])
    if has_inference:
        lines.extend([
            "m=application 5008 RTP/AVP 98",
            "a=control:trackID=2",
            "a=rtpmap:98 radar-classification/90000",
            "a=fmtp:98 encoding=tlv;classes=drone,bird,balloon,clutter,unknown,pending",
            f"a=ssrc:{int(ssrc_inference):08x} cname:radarinsight-classification",
        ])
    return "\r\n".join(lines) + "\r\n"


def header_get(headers: dict[str, str], name: str, default: str = "") -> str:
    return headers.get(name.lower(), default)


__all__ = [
    "DEFAULT_PORT",
    "PROTOCOL",
    "RESOURCE",
    "Request",
    "base_url",
    "build_sdp",
    "header_get",
    "media_track_id",
    "parse_transport",
    "read_request",
    "request_path",
    "send_response",
]
