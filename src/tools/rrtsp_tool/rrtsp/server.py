# -*- coding: utf-8 -*-
"""R-RTSP v1 服务端。

运行方式类似网络摄像头：工具监听 TCP 信令端口；观看端用
OPTIONS/DESCRIBE/SETUP/PLAY 发起会话，随后服务端通过 UDP 发送 RTP。
"""
from __future__ import annotations

import random
import socket
import threading
import time

from . import auth as digest_auth
from . import payloads, rtp
from .asd_format import AsdData
from .netutil import bind_udp_pair
from . import rtsp as rtsp_text

MAX_PACKET_SIZE = 1200


def _new_ssrc() -> int:
    return random.randint(1, 0xFFFFFFFF)


def _new_sequence() -> int:
    return random.randint(0, 0xFFFF)


class MediaTransport:
    def __init__(self, track_id: int, name: str):
        self.track_id = track_id
        self.name = name
        self.ssrc = _new_ssrc()
        self.sequence = _new_sequence()
        self.rtp_socket: socket.socket | None = None
        self.rtcp_socket: socket.socket | None = None
        self.rtp_port = 0
        self.rtcp_port = 0
        self.client_rtp_port = 0
        self.client_rtcp_port = 0
        self.client_addr = ""
        self.setup = False
        self.packets_sent = 0
        self.octets_sent = 0
        self.last_timestamp = 0
        self.rtcp_thread: threading.Thread | None = None

    def setup_transport(self, client_addr: str, client_rtp: int,
                        client_rtcp: int):
        self.rtp_socket, self.rtcp_socket, self.rtp_port, self.rtcp_port = (
            bind_udp_pair())
        self.client_addr = client_addr
        self.client_rtp_port = client_rtp
        self.client_rtcp_port = client_rtcp
        self.setup = True

    def send_rtp(self, data: bytes, timestamp: int):
        if not self.rtp_socket:
            return
        self.rtp_socket.sendto(
            data, (self.client_addr, self.client_rtp_port))
        self.sequence = (self.sequence + 1) & 0xFFFF
        self.packets_sent += 1
        self.octets_sent += len(data)
        self.last_timestamp = timestamp

    def close(self):
        for sock in (self.rtp_socket, self.rtcp_socket):
            try:
                sock and sock.close()
            except Exception:
                pass
        self.setup = False


class ViewerSession:
    def __init__(self, session_id: str, data: AsdData, rate: float,
                 ssrc_rd: int, ssrc_track: int, on_log=None):
        self.id = session_id
        self.data = data
        self.rate = rate
        self.on_log = on_log or (lambda text: None)

        self.media: dict[int, MediaTransport] = {
            0: MediaTransport(0, "RD map"),
            1: MediaTransport(1, "track"),
        }
        self.media[0].ssrc = ssrc_rd
        self.media[1].ssrc = ssrc_track
        self.playing = threading.Event()
        self.alive = True
        self._stop = threading.Event()
        self._active_elapsed = 0.0
        self._resume_time: float | None = None
        self.event_index = 0
        self.rd_frame_index = 0
        self._media_origin = None

        self.sender = threading.Thread(target=self._send_loop, daemon=True)
        self.sender.start()

    def log(self, text: str):
        self.on_log(f"观看端 {self.id}: {text}")

    @property
    def has_rd(self) -> bool:
        return any(self.data.rd_count(i) for i in range(self.data.n_events))

    @property
    def source_track_id(self) -> int:
        source = self.data.manifest.get("source", {})
        try:
            return int(source.get("track_id", 1))
        except (TypeError, ValueError):
            return 1

    @property
    def class_id(self) -> int:
        name = self.data.label_name.strip().lower().replace(" ", "")
        aliases = {
            "dronetarget": "drone",
            "birdtarget": "bird",
            "balloontarget": "balloon",
            "cluttertarget": "clutter",
            "unknowntarget": "unknown-l2",
        }
        return payloads.CLASS_TO_ID.get(aliases.get(name, name), 6)

    def setup_media(self, track_id: int, client_addr: str,
                    transport: dict[str, str]) -> MediaTransport:
        if track_id == 0 and not self.has_rd:
            raise ValueError("当前数据集不包含 RD 图谱")
        if track_id not in self.media:
            raise ValueError("未知 trackID")
        ports = transport.get("client_port", "")
        try:
            rtp_text, rtcp_text = ports.split("-", 1)
            client_rtp, client_rtcp = int(rtp_text), int(rtcp_text)
        except ValueError:
            raise ValueError("Transport 中缺少有效的 client_port")
        media = self.media[track_id]
        if not media.setup:
            media.setup_transport(client_addr, client_rtp, client_rtcp)
            media.rtcp_thread = threading.Thread(
                target=self._rtcp_loop, args=(media,), daemon=True)
            media.rtcp_thread.start()
        return media

    def play(self):
        if self._resume_time is None:
            self._resume_time = time.monotonic()
        self.playing.set()

    def pause(self):
        if self._resume_time is not None:
            self._active_elapsed += time.monotonic() - self._resume_time
            self._resume_time = None
        self.playing.clear()

    def close(self):
        self.alive = False
        self.pause()
        self._stop.set()
        self.sender.join(timeout=0.3)
        for media in self.media.values():
            media.close()

    def _elapsed(self) -> float:
        if self._resume_time is None:
            return self._active_elapsed
        return self._active_elapsed + time.monotonic() - self._resume_time

    def _send_loop(self):
        while not self._stop.is_set():
            if not self.playing.wait(timeout=0.2):
                continue
            source_index = self.event_index % self.data.n_events
            event = self.data.read_event(source_index)
            elapsed = self._elapsed()
            if self._media_origin is None:
                self._media_origin = elapsed
            elapsed -= self._media_origin
            timestamp = int(elapsed * 90000)

            self._send_track(event, timestamp)
            self._send_rd(event, elapsed)

            self.event_index += 1
            self._interruptible_sleep(1.0 / self.rate)

    def _send_track(self, event: dict, timestamp: int):
        media = self.media[1]
        if not media.setup:
            return
        values = {
            key: event["tr"].get(key)
            for key in payloads.TRACK_FIELDS
            if key in event["tr"]}
        track = payloads.encode_track(
            self.source_track_id,
            payloads.TRACK_STATE_NAMES["confirmed"],
            self.class_id, values)
        frame = payloads.build_track_frame(
            [track], update_sequence=max(0, self.event_index) & 0xFF)
        packet = rtp.build_rtp_packet(
            payload_type=97,
            ssrc=media.ssrc,
            sequence=media.sequence,
            timestamp=timestamp,
            payload=frame)
        media.send_rtp(packet, timestamp)

    def _send_rd(self, event: dict, elapsed: float):
        media = self.media[0]
        if not media.setup or not event["rd"]:
            return
        event_origin = float(event["rd"][0][0])
        for j, (frame_time, amp, phase) in enumerate(event["rd"]):
            frame_seq = self.rd_frame_index & 0xFFFF
            frame_id = self.event_index * 100 + j
            chunks = payloads.fragment_rd_frame(
                amplitude=amp,
                phase=phase,
                frame_sequence=frame_seq,
                frame_id=frame_id,
                max_packet_size=MAX_PACKET_SIZE)
            frame_offset = max(0.0, float(frame_time) - event_origin)
            frame_ts = int((elapsed + frame_offset) * 90000)
            for k, chunk in enumerate(chunks):
                packet = rtp.build_rtp_packet(
                    payload_type=96,
                    ssrc=media.ssrc,
                    sequence=media.sequence,
                    timestamp=frame_ts,
                    payload=chunk,
                    marker=(k == len(chunks) - 1))
                media.send_rtp(packet, frame_ts)
            self.rd_frame_index += 1

    def _interruptible_sleep(self, seconds: float):
        end = time.monotonic() + seconds
        while not self._stop.is_set() and self.playing.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.02, remaining))

    def _rtcp_loop(self, media: MediaTransport):
        last_sr_time = 0.0
        last_sr_ntp = 0
        media.rtcp_socket.settimeout(0.2)
        while self.alive and media.setup:
            now = time.time()
            if now - last_sr_time >= 1.0:
                sr = rtp.build_sender_report(
                    ssrc=media.ssrc,
                    rtp_timestamp=media.last_timestamp,
                    packet_count=media.packets_sent,
                    octet_count=media.octets_sent)
                try:
                    media.rtcp_socket.sendto(
                        sr, (media.client_addr, media.client_rtcp_port))
                    last_sr_ntp = rtp.ntp_now()[0]
                    last_sr_time = now
                except OSError:
                    pass
            try:
                data, _addr = media.rtcp_socket.recvfrom(2048)
                packets = rtp.parse_rtcp(data)
                for packet in packets:
                    if packet["type"] == rtp.PT_RTCP_RR:
                        self.log(f"{media.name} 收到 RTCP RR")
            except socket.timeout:
                pass
            except (OSError, ValueError):
                time.sleep(0.1)


class RRTSPServer:
    def __init__(self, host: str, port: int, asd_path: str, rate: float,
                 auth_mode: str = "open", username: str = "viewer",
                 password: str = "", realm: str = "AeroSight",
                 on_log=None):
        self.host = host
        self.port = port
        self.asd_path = asd_path
        self.rate = max(float(rate), 0.1)
        self.auth_mode = auth_mode
        self.username = username
        self.password = password
        self.realm = realm
        self.nonce = digest_auth.new_nonce()
        self.nonce_created = time.time()
        self.on_log = on_log or (lambda text: None)

        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._sessions: dict[str, ViewerSession] = {}
        self._lock = threading.Lock()
        self._data: AsdData | None = None
        self.ssrc_rd = _new_ssrc()
        self.ssrc_track = _new_ssrc()

    def start(self):
        if self.is_running():
            return
        self._data = AsdData(self.asd_path)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(8)
        self._socket.settimeout(0.5)
        self._stop.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self.on_log(
            f"R-RTSP 服务已启动：rtsp://<本机IP>:{self.port}/radar")

    def stop(self):
        self._stop.set()
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
        try:
            self._socket and self._socket.close()
        except Exception:
            pass
        if self._data:
            self._data.close()
            self._data = None
        self.on_log("R-RTSP 服务已停止")

    def is_running(self) -> bool:
        return bool(self._accept_thread and self._accept_thread.is_alive())

    def _first_rd_shape(self):
        if self._data is None:
            return None
        for i in range(self._data.n_events):
            if self._data.rd_count(i):
                return self._data.read_frame_meta(i, 0)["shape"]
        return None

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._handle_connection, args=(conn, addr),
                daemon=True)
            thread.start()

    def _handle_connection(self, conn: socket.socket, addr):
        with conn:
            while not self._stop.is_set():
                try:
                    req = rtsp_text.read_request(conn)
                except (ValueError, OSError):
                    break
                if req is None:
                    break
                try:
                    self._route(conn, req, addr)
                except Exception as exc:
                    cseq = rtsp_text.header_get(req.headers, "CSeq", "0")
                    rtsp_text.send_response(conn, 500, cseq=cseq)
                    self.on_log(f"请求处理失败：{exc}")

    def _route(self, conn, req, addr):
        method = req.method
        cseq = rtsp_text.header_get(req.headers, "CSeq", "0")

        if method == "OPTIONS":
            rtsp_text.send_response(conn, 200, cseq=cseq, headers={
                "Public": (
                    "OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, "
                    "TEARDOWN, GET_PARAMETER")
            })
            return

        if self.auth_mode == "digest":
            authorization = rtsp_text.header_get(req.headers, "Authorization")
            valid = digest_auth.validate_authorization(
                header=authorization,
                username=self.username,
                password=self.password,
                realm=self.realm,
                method=method,
                uri=req.uri,
                nonce=self.nonce,
                issued_at=self.nonce_created)
            if not valid:
                rtsp_text.send_response(conn, 401, cseq=cseq, headers={
                    "WWW-Authenticate": self._challenge_header()
                })
                self.on_log(f"{addr[0]} 需要 Digest 鉴权")
                return

        if method == "DESCRIBE":
            has_rd = any(self._data.rd_count(i)
                         for i in range(self._data.n_events))
            rd_shape = self._first_rd_shape()
            local_ip = conn.getsockname()[0]
            host = rtsp_text.header_get(req.headers, "Host")
            base = rtsp_text.base_url(req.uri, host, self.host, self.port)
            # 为保持 SDP 稳定，RD/track SSRC 使用首个会话或临时值。
            sdp = rtsp_text.build_sdp(
                base=base,
                server_ip=local_ip,
                ssrc_rd=self.ssrc_rd,
                ssrc_track=self.ssrc_track,
                has_rd=has_rd,
                has_track=True,
                frame_rate=self.rate,
                max_packet_size=MAX_PACKET_SIZE,
                rd_shape=rd_shape)
            rtsp_text.send_response(
                conn, 200, cseq=cseq,
                headers={"Content-Base": base + "/"},
                body=sdp.encode("utf-8"),
                content_type="application/sdp")
            self.on_log(f"{addr[0]} 请求 SDP")
            return

        if method == "SETUP":
            self._setup(conn, req, addr, cseq)
            return

        session_id = rtsp_text.header_get(req.headers, "Session").split(";")[0].strip()
        with self._lock:
            session = self._sessions.get(session_id)

        if not session:
            rtsp_text.send_response(conn, 454, cseq=cseq)
            return

        if method == "PLAY":
            session.play()
            info = []
            for track_id in (0, 1):
                media = session.media[track_id]
                if media.setup:
                    info.append(
                        f"url=trackID={track_id};seq={media.sequence};"
                        f"rtptime={media.last_timestamp}")
            rtsp_text.send_response(conn, 200, cseq=cseq, headers={
                "Session": f"{session.id};timeout=60",
                "RTP-Info": ",".join(info),
            })
            self.on_log(f"{addr[0]} 开始播放 {session.id}")
            return

        if method == "PAUSE":
            session.pause()
            rtsp_text.send_response(conn, 200, cseq=cseq, headers={
                "Session": f"{session.id};timeout=60"})
            return

        if method == "TEARDOWN":
            with self._lock:
                self._sessions.pop(session.id, None)
            session.close()
            rtsp_text.send_response(conn, 200, cseq=cseq)
            self.on_log(f"{addr[0]} 拆除会话 {session.id}")
            return

        if method == "GET_PARAMETER":
            body = f"events: {self._data.n_events}\r\nrate: {self.rate}\r\n".encode()
            rtsp_text.send_response(
                conn, 200, cseq=cseq, body=body,
                content_type="text/parameters")
            return

        rtsp_text.send_response(conn, 501, cseq=cseq)

    def _challenge_header(self) -> str:
        return (
            'Digest realm="' + self.realm + '", '
            f'nonce="{self.nonce}", algorithm=SHA-256, qop="auth"')

    def _setup(self, conn, req, addr, cseq):
        session_id = rtsp_text.header_get(
            req.headers, "Session").split(";")[0].strip()
        track_id = rtsp_text.media_track_id(req.uri)
        if track_id is None:
            track_id = 0
        if track_id == 0 and not any(
                self._data.rd_count(i)
                for i in range(self._data.n_events)):
            rtsp_text.send_response(conn, 461, cseq=cseq)
            self.on_log("SETUP 失败：当前数据集没有 RD 图谱")
            return

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ViewerSession(
                    f"{random.randint(0, 0xFFFFFFFF):08x}",
                    self._data, self.rate, self.ssrc_rd,
                    self.ssrc_track, self.on_log)
                self._sessions[session.id] = session

        transport = rtsp_text.parse_transport(
            rtsp_text.header_get(req.headers, "Transport"))
        try:
            media = session.setup_media(track_id, addr[0], transport)
        except ValueError as exc:
            rtsp_text.send_response(conn, 461, cseq=cseq)
            self.on_log(f"SETUP 失败：{exc}")
            return

        transport_out = (
            "RTP/AVP;unicast;"
            f"client_port={media.client_rtp_port}-{media.client_rtcp_port};"
            f"server_port={media.rtp_port}-{media.rtcp_port};"
            f"ssrc={media.ssrc:08x}")
        rtsp_text.send_response(conn, 200, cseq=cseq, headers={
            "Session": f"{session.id};timeout=60",
            "Transport": transport_out,
        })
        self.on_log(
            f"{addr[0]} SETUP trackID={track_id}，服务端 "
            f"{media.rtp_port}/{media.rtcp_port}")
