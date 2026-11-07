# -*- coding: utf-8 -*-
"""R-RTSP v1 最小观看客户端。"""
from __future__ import annotations

import socket
import threading
import time

from . import auth as digest_auth
from .netutil import bind_udp_pair
from . import rtp as rtp_text
from .rtsp import PROTOCOL, RESOURCE


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


class RRTSPViewerClient:
    def __init__(self, host: str, port: int = 8554,
                 resource: str = RESOURCE, username: str = "",
                 password: str = ""):
        self.host = host
        self.port = port
        self.url = f"rtsp://{host}:{port}/{resource}"
        self.conn: socket.socket | None = None
        self.session_id = ""
        self.cseq = 0
        self.media: dict[int, dict] = {}
        self.username = username
        self.password = password
        self.challenge: dict[str, str] | None = None
        self.nc = 0
        self.cnonce = digest_auth.secrets.token_hex(8)

    def connect(self):
        self.conn = socket.create_connection((self.host, self.port), timeout=5)

    def close(self):
        for item in self.media.values():
            for key in ("rtp_socket", "rtcp_socket"):
                try:
                    item[key].close()
                except Exception:
                    pass
        try:
            self.conn and self.conn.close()
        except Exception:
            pass

    def _request(self, method: str, url: str | None = None,
                 headers: dict[str, str] | None = None,
                 allow_auth_retry: bool = True):
        self.cseq += 1
        target = url or self.url
        request_headers = dict(headers or {})
        if self.challenge and self.username:
            request_headers["Authorization"] = self._authorization(method, target)

        lines = [f"{method} {target} {PROTOCOL}", f"CSeq: {self.cseq}"]
        if self.session_id:
            lines.append(f"Session: {self.session_id}")
        for key, value in request_headers.items():
            lines.append(f"{key}: {value}")
        self.conn.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())

        status_line = _read_line(self.conn)
        response_headers: dict[str, str] = {}
        while True:
            line = _read_line(self.conn)
            if not line:
                break
            key, value = line.split(":", 1)
            response_headers[key.strip().lower()] = value.strip()

        body = b""
        length = int(response_headers.get("content-length", "0"))
        while len(body) < length:
            chunk = self.conn.recv(length - len(body))
            if not chunk:
                break
            body += chunk

        if status_line and status_line.startswith('RTSP/') and ' 401 ' in status_line:
            challenge_text = response_headers.get("www-authenticate", "")
            if allow_auth_retry and self.username and challenge_text:
                self.challenge = digest_auth.parse_digest_header(challenge_text)
                self.cseq -= 1
                return self._request(
                    method, url, headers, allow_auth_retry=False)

        if not status_line:
            raise RuntimeError("RTSP 连接已关闭")
        status = int(status_line.split()[1])
        return status, response_headers, body

    def _authorization(self, method: str, uri: str) -> str:
        self.nc += 1
        realm = self.challenge.get("realm", "AeroSight")
        return digest_auth.build_authorization(
            username=self.username,
            password=self.password,
            realm=realm,
            method=method,
            uri=uri,
            nonce=self.challenge["nonce"],
            cnonce=self.cnonce,
            nc=f"{self.nc:08x}")

    def options(self):
        return self._request("OPTIONS")

    def describe(self) -> tuple[int, dict, bytes]:
        return self._request("DESCRIBE")

    def setup(self, track_id: int) -> dict:
        rtp_sock, rtcp_sock, rtp_port, rtcp_port = bind_udp_pair()
        headers = {
            "Transport": (
                "RTP/AVP;unicast;"
                f"client_port={rtp_port}-{rtcp_port}")
        }
        url = f"{self.url}/trackID={track_id}"
        status, resp, _body = self._request("SETUP", url, headers)
        if status != 200:
            rtp_sock.close(); rtcp_sock.close()
            raise RuntimeError(f"SETUP 失败：{status}")
        self.session_id = resp.get("session", self.session_id).split(";")[0]
        transport = self._parse_transport(resp.get("transport", ""))
        item = {
            "rtp_socket": rtp_sock,
            "rtcp_socket": rtcp_sock,
            "client_rtp_port": rtp_port,
            "client_rtcp_port": rtcp_port,
            "server_rtp_port": int(transport.get("server_port", "0-0").split("-")[0]),
            "server_rtcp_port": int(transport.get("server_port", "0-0").split("-")[1]),
            "ssrc": int(transport.get("ssrc", "0"), 16),
            "packets": [],
        }
        self.media[track_id] = item
        return item

    def play(self):
        status, _h, _b = self._request("PLAY")
        if status != 200:
            raise RuntimeError(f"PLAY 失败：{status}")

    def pause(self):
        return self._request("PAUSE")

    def teardown(self):
        try:
            return self._request("TEARDOWN")
        finally:
            self.close()

    @staticmethod
    def _parse_transport(value: str) -> dict[str, str]:
        out = {}
        for token in value.split(";"):
            if "=" in token:
                k, v = token.split("=", 1)
                out[k.strip().lower()] = v.strip()
        return out

    def receive_for(self, duration: float, send_rr: bool = True) -> dict:
        stop = threading.Event()
        threads = []
        for track_id, item in self.media.items():
            thread = threading.Thread(
                target=self._receive_one,
                args=(track_id, item, duration, stop, send_rr),
                daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(duration + 2)
        return {
            track_id: item["packets"]
            for track_id, item in self.media.items()
        }

    def _receive_one(self, track_id: int, item: dict, duration: float,
                     stop: threading.Event, send_rr: bool):
        sock = item["rtp_socket"]
        sock.settimeout(0.2)
        deadline = time.monotonic() + duration
        next_rr = time.monotonic() + 0.5
        reporter_ssrc = (id(self) ^ track_id) & 0xFFFFFFFF

        while time.monotonic() < deadline:
            try:
                data, _addr = sock.recvfrom(4096)
                packet = rtp_text.parse_rtp_packet(data)
                item["packets"].append(packet)
            except socket.timeout:
                pass
            if send_rr and time.monotonic() >= next_rr and item["packets"]:
                highest = item["packets"][-1]["sequence"]
                rr = rtp_text.build_receiver_report(
                    reporter_ssrc=reporter_ssrc,
                    source_ssrc=item["ssrc"],
                    highest_sequence=highest,
                    jitter=0)
                try:
                    item["rtcp_socket"].sendto(
                        rr, (self.host, item["server_rtcp_port"]))
                except OSError:
                    pass
                next_rr = time.monotonic() + 1.0
        stop.set()

