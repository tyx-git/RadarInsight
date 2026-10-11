# -*- coding: utf-8 -*-
"""R-RTSP v1 track + RD client connection.

Flow: DESCRIBE -> SETUP trackID=1 -> optional SETUP trackID=0 -> PLAY.
After PLAY, track and RTP data are read on separate UDP sockets and paired by
RTP timestamp.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import socket
import struct
import threading
import time
import queue as _queue
import re
from urllib.parse import unquote, urlparse


log = logging.getLogger(__name__)

TRACK_FRAME_HEAD = struct.Struct(">BHBB")
TLV_HEAD = struct.Struct(">BH")
TRACK_VALUE_HEAD = struct.Struct(">IBBI")
RTP_HEAD = struct.Struct(">BBHII")
RD_HEADER = struct.Struct(">BBHIHHBBHf")
INFERENCE_FRAME_HEAD = struct.Struct(">BHBB")
TLV_HEAD = struct.Struct(">BH")
INFERENCE_RESULT_HEAD = struct.Struct(">IBf")

TRACK_FIELD_NAMES = (
    "range_m", "azimuth_deg", "elevation_deg", "radial_speed_mps",
    "speed_mps", "course_deg", "snr_db", "amplitude_db",
)
CLASS_NAMES = {
    1: "drone", 2: "bird", 3: "balloon", 4: "clutter",
    5: "unknown", 6: "other",
}
STATE_NAMES = {0: "suspect", 1: "confirmed", 2: "coasted", 3: "dropped"}
INFERENCE_CLASS_NAMES = {
    1: "drone",
    2: "bird",
    3: "balloon",
    4: "clutter",
    5: "unknown",
    6: "pending",
}
DEFAULT_SCAN_INTERVAL_SECONDS = 3.0
RTP_CLOCK_HZ = 90000.0


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_response(username, password, realm, method, uri, nonce,
                    nc, cnonce, qop):
    ha1 = sha256(f"{username}:{realm}:{password}")
    ha2 = sha256(f"{method}:{uri}")
    return sha256(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")


def parse_sdp_update_rate(sdp: str) -> float | None:
    """Read the track media ``updateRate`` from an R-RTSP SDP body."""

    in_track_media = False
    for raw_line in str(sdp or "").splitlines():
        line = raw_line.strip()
        if line.startswith("m="):
            in_track_media = bool(re.match(r"m=\S+\s+\d+\s+\S+\s+97(?:\s|$)", line))
            continue
        if not in_track_media or not line.lower().startswith("a=fmtp:"):
            continue
        match = re.search(r"(?:^|;)\s*updateRate\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)", line)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                return None
            return value if value > 0.0 else None
    return None


def parse_track_payload(payload: bytes) -> list[dict]:
    frame_type, n_tracks, _seq, _reserved = TRACK_FRAME_HEAD.unpack_from(payload)
    offset = TRACK_FRAME_HEAD.size
    tracks = []

    for _ in range(n_tracks):
        tlv_type, length = TLV_HEAD.unpack_from(payload, offset)
        offset += TLV_HEAD.size
        value = payload[offset:offset + length]
        offset += length
        if tlv_type != 1 or len(value) < TRACK_VALUE_HEAD.size:
            continue

        track_id, state, class_id, fspec = TRACK_VALUE_HEAD.unpack_from(value)
        p = TRACK_VALUE_HEAD.size
        fields = {}
        for bit, name in enumerate(TRACK_FIELD_NAMES):
            if fspec & (1 << bit):
                fields[name] = struct.unpack(">f", value[p:p + 4])[0]
                p += 4

        tracks.append({
            "id": track_id,
            "state": STATE_NAMES.get(state, str(state)),
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, "unknown"),
            "fields": fields,
        })
    return tracks


def parse_inference_payload(payload: bytes) -> dict[int, tuple[str, float]]:
    """Decode the optional PT=98 classification result frame."""

    if len(payload) < INFERENCE_FRAME_HEAD.size:
        raise ValueError("Inference payload is shorter than its frame header")
    frame_type, count, _sequence, _reserved = INFERENCE_FRAME_HEAD.unpack_from(payload)
    if frame_type != 2:
        raise ValueError(f"Unsupported inference frame type: {frame_type}")
    offset = INFERENCE_FRAME_HEAD.size
    labels: dict[int, tuple[str, float]] = {}
    for _ in range(count):
        if offset + TLV_HEAD.size > len(payload):
            raise ValueError("Inference TLV header is truncated")
        tlv_type, length = TLV_HEAD.unpack_from(payload, offset)
        offset += TLV_HEAD.size
        end = offset + length
        if end > len(payload):
            raise ValueError("Inference TLV value is truncated")
        value = payload[offset:end]
        offset = end
        if tlv_type != 2 or len(value) != INFERENCE_RESULT_HEAD.size:
            continue
        track_id, class_id, confidence = INFERENCE_RESULT_HEAD.unpack(value)
        if class_id not in INFERENCE_CLASS_NAMES:
            raise ValueError(f"Unknown inference class id: {class_id}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Inference confidence is outside [0, 1]")
        labels[int(track_id)] = (INFERENCE_CLASS_NAMES[class_id], float(confidence))
    if offset != len(payload):
        raise ValueError("Inference payload contains trailing bytes")
    return labels


class RRTSPConnection:
    def __init__(self, url: str, username: str = "", password: str = ""):
        parsed = urlparse(url)
        if parsed.scheme != "rtsp" or not parsed.hostname:
            raise ValueError(f"Invalid R-RTSP URL: {url}")
        self.host = parsed.hostname
        self.port = parsed.port or 8554
        self.path = parsed.path or "/radar"
        self.username = username or unquote(parsed.username or "")
        self.password = password or unquote(parsed.password or "")
        self.url = url

        self.tcp: socket.socket | None = None
        self.rtp: socket.socket | None = None
        self.rtcp: socket.socket | None = None
        self.rd_rtp: socket.socket | None = None
        self.rd_rtcp: socket.socket | None = None
        self.inference_rtp: socket.socket | None = None
        self.inference_rtcp: socket.socket | None = None

        self.cseq = 0
        self.session_id = ""
        self.playing = False
        self.has_rd = False
        self.has_inference = False
        self.sdp = ""
        self.update_rate_hz: float | None = None
        self.scan_interval_seconds = DEFAULT_SCAN_INTERVAL_SECONDS
        self.last_response_body = b""
        self.challenge: dict[str, str] | None = None
        self.nc = 0
        self.cnonce = secrets.token_hex(8)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.track_q: _queue.Queue = _queue.Queue(maxsize=200)
        self.rd_q: _queue.Queue = _queue.Queue(maxsize=200)
        self.inference_q: _queue.Queue = _queue.Queue(maxsize=200)
        self._rd_pending: dict[int, tuple] = {}
        self._inference_pending: dict[int, dict[int, tuple[str, float]]] = {}
        self.last_remote_labels: dict[int, tuple[str, float]] = {}

    def open(self):
        self.tcp = socket.create_connection((self.host, self.port), timeout=5)
        self.command("DESCRIBE")
        self.sdp = self.last_response_body.decode("utf-8", "replace")
        self.update_rate_hz = parse_sdp_update_rate(self.sdp)
        if self.update_rate_hz is not None and self.update_rate_hz > 0.0:
            self.scan_interval_seconds = 1.0 / self.update_rate_hz
        else:
            self.scan_interval_seconds = DEFAULT_SCAN_INTERVAL_SECONDS

        self.rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rd_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rd_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inference_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inference_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for sock in (
            self.rtp, self.rtcp, self.rd_rtp, self.rd_rtcp,
            self.inference_rtp, self.inference_rtcp,
        ):
            sock.bind(("", 0))

        rtp_port = self.rtp.getsockname()[1]
        rtcp_port = self.rtcp.getsockname()[1]
        self.command(
            "SETUP",
            path=f"{self.url}/trackID=1",
            headers={
                "Transport": "RTP/AVP;unicast;"
                f"client_port={rtp_port}-{rtcp_port}"
            },
        )

        rd_rtp_port = self.rd_rtp.getsockname()[1]
        rd_rtcp_port = self.rd_rtcp.getsockname()[1]
        status, _headers = self.command(
            "SETUP",
            path=f"{self.url}/trackID=0",
            headers={
                "Transport": "RTP/AVP;unicast;"
                f"client_port={rd_rtp_port}-{rd_rtcp_port}"
            },
            accepted_statuses=(200, 461),
        )
        self.has_rd = status == 200

        assert self.inference_rtp is not None and self.inference_rtcp is not None
        inference_rtp_port = self.inference_rtp.getsockname()[1]
        inference_rtcp_port = self.inference_rtcp.getsockname()[1]
        status, _headers = self.command(
            "SETUP",
            path=f"{self.url}/trackID=2",
            headers={
                "Transport": "RTP/AVP;unicast;"
                f"client_port={inference_rtp_port}-{inference_rtcp_port}"
            },
            accepted_statuses=(200, 461),
        )
        self.has_inference = status == 200

        self.command("PLAY")
        self.playing = True
        self._start_receivers()

    def close(self):
        self._stop.set()
        self._rd_pending.clear()
        self._inference_pending.clear()
        self.last_remote_labels = {}
        try:
            if self.tcp and self.playing:
                self.command("TEARDOWN", accepted_statuses=(200, 404, 454))
                self.playing = False
        except Exception:
            pass

        for thread in self._threads:
            thread.join(timeout=0.5)
        for pending_queue in (self.track_q, self.rd_q, self.inference_q):
            while True:
                try:
                    pending_queue.get_nowait()
                except _queue.Empty:
                    break
        for sock in (
            self.tcp, self.rtp, self.rtcp,
            self.rd_rtp, self.rd_rtcp,
            self.inference_rtp, self.inference_rtcp,
        ):
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        self.tcp = None
        self.rtp = self.rtcp = None
        self.rd_rtp = self.rd_rtcp = None
        self.inference_rtp = self.inference_rtcp = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def command(self, method: str, path: str | None = None,
                headers: dict | None = None, accepted_statuses=(200,)):
        uri = path or self.url
        body = dict(headers or {})
        if self.challenge:
            body["Authorization"] = self._authorization(method, uri)
        lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {self.next_cseq()}"]
        if self.session_id and method != "DESCRIBE":
            lines.append(f"Session: {self.session_id}")
        for key, value in body.items():
            lines.append(f"{key}: {value}")
        data = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
        assert self.tcp is not None
        self.tcp.sendall(data)
        status, resp_headers, response_body = self._read_response()
        self.last_response_body = response_body

        if status == 401:
            challenge_text = resp_headers.get("www-authenticate", "")
            # Only retry once: the first 401 carries the challenge and no
            # Authorization has been sent yet. A second 401 after credentials
            # were supplied means the username or password is wrong.
            if self.username and challenge_text and not self.challenge:
                self.challenge = self._parse_digest(challenge_text)
                self.cseq -= 1
                return self.command(
                    method, path, headers, accepted_statuses
                )
            if self.challenge:
                raise PermissionError("R-RTSP 账号或密码错误")
            raise PermissionError("R-RTSP requires username and password")
        if status not in accepted_statuses:
            raise RuntimeError(
                f"{method} failed: status={status}, accepted={accepted_statuses}"
            )
        session = resp_headers.get("session")
        if session:
            self.session_id = session.split(";")[0]
        return status, resp_headers

    def next_cseq(self) -> int:
        self.cseq += 1
        return self.cseq

    @staticmethod
    def _parse_digest(value: str) -> dict[str, str]:
        out = {}
        for part in value.split(","):
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            out[key.strip().lower()] = val.strip().strip('"')
        return out

    def _read_response(self):
        status_line = self._read_line()
        headers = {}
        while True:
            line = self._read_line()
            if not line:
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        body = b""
        length = int(headers.get("content-length", "0"))
        while len(body) < length:
            chunk = self.tcp.recv(length - len(body))  # type: ignore[union-attr]
            if not chunk:
                break
            body += chunk
        if not status_line:
            raise RuntimeError("RTSP connection closed")
        status = int(status_line.split()[1])
        return status, headers, body

    def _read_line(self) -> str:
        assert self.tcp is not None
        data = bytearray()
        while True:
            ch = self.tcp.recv(1)
            if not ch or ch == b"\n":
                break
            data.extend(ch)
        return data.decode("utf-8", "replace").rstrip("\r")

    def _authorization(self, method: str, uri: str) -> str:
        self.nc += 1
        assert self.challenge is not None
        realm = self.challenge.get("realm", "AeroSight")
        nonce = self.challenge.get("nonce", "")
        qop = self.challenge.get("qop", "auth")
        nc = f"{self.nc:08d}"
        response = digest_response(
            self.username, self.password, realm, method, uri,
            nonce, nc, self.cnonce, qop)
        return (
            "Digest "
            f'username="{self.username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{uri}", response="{response}", qop={qop}, nc={nc}, '
            f'cnonce="{self.cnonce}", algorithm=SHA-256')

    def _start_receivers(self):
        self._stop.clear()
        watchdog_thread = threading.Thread(
            target=self._connection_watchdog,
            name="aerosight-rrtsp-watchdog",
            daemon=True,
        )
        self._threads.append(watchdog_thread)
        watchdog_thread.start()
        track_thread = threading.Thread(
            target=self._track_loop, name="aerosight-rrtsp-track", daemon=True
        )
        self._threads.append(track_thread)
        track_thread.start()
        if self.has_rd:
            rd_thread = threading.Thread(
                target=self._rd_loop, name="aerosight-rrtsp-rd", daemon=True
            )
            self._threads.append(rd_thread)
            rd_thread.start()
        if self.has_inference:
            inference_thread = threading.Thread(
                target=self._inference_loop,
                name="aerosight-rrtsp-inference",
                daemon=True,
            )
            self._threads.append(inference_thread)
            inference_thread.start()

    def _connection_watchdog(self):
        """Notice a closed RTSP control socket while UDP media is idle."""

        tcp = self.tcp
        if tcp is None:
            return
        tcp.settimeout(0.2)
        while not self._stop.is_set():
            try:
                data = tcp.recv(1, socket.MSG_PEEK)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            # A response may be waiting for a control command. Do not consume
            # it here; just avoid a busy loop while the command reader drains it.
            self._stop.wait(0.2)
        self._stop.set()

    def _track_loop(self):
        assert self.rtp is not None
        self.rtp.settimeout(0.2)
        while not self._stop.is_set():
            try:
                data, _addr = self.rtp.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed = self._parse_rtp(data)
            if parsed is None or parsed["payload_type"] != 97:
                continue
            tracks = parse_track_payload(parsed["payload"])
            try:
                self.track_q.put(
                    (parsed["timestamp"], tracks), timeout=0.2
                )
            except _queue.Full:
                try:
                    self.track_q.get_nowait()
                    self.track_q.put_nowait((parsed["timestamp"], tracks))
                except Exception:
                    pass

    def _rd_loop(self):
        assert self.rd_rtp is not None
        self.rd_rtp.settimeout(0.2)
        buffers: dict[int, dict] = {}
        while not self._stop.is_set():
            try:
                data, _addr = self.rd_rtp.recvfrom(65535)
            except socket.timeout:
                buffers.clear()
                continue
            except OSError:
                break
            parsed = self._parse_rtp(data)
            if parsed is None or parsed["payload_type"] != 96:
                continue
            payload = parsed["payload"]
            if len(payload) < RD_HEADER.size:
                continue
            rd_header = self._parse_rd_header(payload)
            frame_sequence = rd_header["frame_sequence"]
            buffer = buffers.setdefault(
                frame_sequence,
                {"packets": [], "timestamp": parsed["timestamp"]},
            )
            buffer["packets"].append((parsed["sequence"], payload[RD_HEADER.size:]))

            if parsed["marker"]:
                buffer["packets"].sort(key=lambda item: item[0])
                body = b"".join(chunk for _seq, chunk in buffer["packets"])
                buffers.pop(frame_sequence, None)
                try:
                    amp, phase = self._decode_rd(rd_header, body)
                    self.rd_q.put(
                        (parsed["timestamp"], amp, phase), timeout=0.2
                    )
                except Exception as exc:
                    log.debug("RD decode failed: %s", exc)
            if len(buffers) > 20:
                oldest = next(iter(buffers))
                buffers.pop(oldest, None)

    def _inference_loop(self):
        assert self.inference_rtp is not None
        self.inference_rtp.settimeout(0.2)
        while not self._stop.is_set():
            try:
                data, _addr = self.inference_rtp.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed = self._parse_rtp(data)
            if parsed is None or parsed["payload_type"] != 98:
                continue
            try:
                labels = parse_inference_payload(parsed["payload"])
                self.inference_q.put(
                    (parsed["timestamp"], labels), timeout=0.2
                )
            except _queue.Full:
                try:
                    self.inference_q.get_nowait()
                    self.inference_q.put_nowait((parsed["timestamp"], labels))
                except Exception:
                    pass
            except (ValueError, struct.error) as exc:
                log.debug("RRTSP inference decode failed: %s", exc)

    @staticmethod
    def _parse_rtp(data: bytes) -> dict | None:
        if len(data) < RTP_HEAD.size:
            return None
        first, second, sequence, timestamp, _ssrc = RTP_HEAD.unpack_from(data)
        cc = first & 0x0F
        payload_start = RTP_HEAD.size + 4 * cc
        return {
            "sequence": sequence,
            "timestamp": timestamp,
            "payload_type": second & 0x7F,
            "marker": bool(second & 0x80),
            "payload": data[payload_start:],
        }

    @staticmethod
    def _parse_rd_header(payload: bytes) -> dict:
        values = RD_HEADER.unpack_from(payload)
        keys = (
            "frame_type", "flags", "frame_sequence", "frame_id",
            "range_bins", "doppler_bins", "bits_per_sample",
            "quant", "compression", "scale",
        )
        return dict(zip(keys, values))

    @staticmethod
    def _decode_rd(header: dict, body: bytes):
        import numpy as np

        n = int(header["range_bins"]) * int(header["doppler_bins"]) * 4
        if header["bits_per_sample"] != 32:
            raise ValueError("Only float32 RD samples are supported")
        if len(body) < n:
            raise ValueError("RD amplitude payload is incomplete")
        amplitude = np.frombuffer(body[:n], dtype="<f4").copy()
        amplitude = amplitude.reshape(
            int(header["range_bins"]), int(header["doppler_bins"])
        )
        if header["flags"] & 0x02 and len(body) >= 2 * n:
            phase = np.frombuffer(body[n:2 * n], dtype="<f4").copy()
            phase = phase.reshape(amplitude.shape)
        else:
            phase = np.zeros_like(amplitude)
        return amplitude, phase

    def receive_paired(self, timeout: float = 0.5):
        """Return one complete timestamp-matched radar batch.

        When the server advertises PT=98, the current track batch remains
        blocked until its classification packet arrives. This preserves batch
        order and prevents later track packets from reaching the UI first.
        """
        try:
            timestamp, tracks = self.track_q.get(timeout=timeout)
        except _queue.Empty:
            if self._stop.is_set():
                raise ConnectionError("R-RTSP control connection is closed")
            raise
        rd = None
        if self.has_rd:
            rd = self._wait_for_rd(timestamp)
        self.last_remote_labels = (
            self._wait_for_inference(timestamp) if self.has_inference else {}
        )
        return timestamp, tracks, rd

    def _wait_for_inference(
        self,
        timestamp: int,
        timeout: float = 0.5,
    ) -> dict[int, tuple[str, float]]:
        if timestamp in self._inference_pending:
            return self._inference_pending.pop(timestamp)
        # ``timeout`` is retained for source compatibility. Inference is a
        # required part of a remote-inference batch and must not time out.
        while not self._stop.is_set():
            try:
                result_timestamp, labels = self.inference_q.get(
                    timeout=0.1
                )
            except _queue.Empty:
                continue
            if result_timestamp == timestamp:
                return labels
            self._inference_pending[result_timestamp] = labels
        raise TimeoutError("R-RTSP connection closed while waiting for inference")

    def _wait_for_rd(self, timestamp: int, timeout: float = 0.8):
        if timestamp in self._rd_pending:
            return self._rd_pending.pop(timestamp)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                rd_timestamp, amplitude, phase = self.rd_q.get(
                    timeout=min(0.1, remaining)
                )
            except _queue.Empty:
                continue
            if rd_timestamp == timestamp:
                return amplitude, phase
            self._rd_pending[rd_timestamp] = (amplitude, phase)
        return None

    # Backward-compatible alias.
    def receive_track(self, timeout: float = 0.5):
        timestamp, tracks, _rd = self.receive_paired(timeout)
        return timestamp, tracks


_RRTSPConnection = RRTSPConnection
