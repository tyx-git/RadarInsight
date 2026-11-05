"""R-RTSP v1 server for AeroSight-compatible clients."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import socket
import threading
from typing import Callable

from ..protocol import auth, rtsp
from ..source.base import StreamSource
from .session import MAX_PACKET_SIZE, ViewerSession


@dataclass(frozen=True, init=False)
class RRTSPServerConfig:
    """R-RTSP listener and one authoritative packet send interval."""

    host: str = "0.0.0.0"
    port: int = 8554
    resource: str = "/radar"
    send_interval_seconds: float = 1.0
    loop: bool = True
    max_clients: int = 8
    auth_mode: str = "open"
    username: str = "viewer"
    password: str = ""
    # AeroSight v1 clients use this realm as their compatibility default.
    realm: str = "AeroSight"
    socket_timeout_seconds: float = 60.0
    inference_mode: str = "causal"

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8554,
        resource: str = "/radar",
        send_interval_seconds: float = 1.0,
        loop: bool = True,
        max_clients: int = 8,
        auth_mode: str = "open",
        username: str = "viewer",
        password: str = "",
        realm: str = "AeroSight",
        socket_timeout_seconds: float = 60.0,
        inference_mode: str = "causal",
        *,
        playback_rate: float | None = None,
    ) -> None:
        # Accept the old keyword once for callers outside this repository, but
        # keep one stored interval and one internal name everywhere else.
        if playback_rate is not None:
            if not math.isfinite(float(playback_rate)) or float(playback_rate) <= 0.0:
                raise ValueError("playback_rate must be positive and finite")
            send_interval_seconds = 1.0 / float(playback_rate)
        object.__setattr__(self, "host", str(host))
        object.__setattr__(self, "port", int(port))
        object.__setattr__(self, "resource", str(resource))
        object.__setattr__(self, "send_interval_seconds", float(send_interval_seconds))
        object.__setattr__(self, "loop", bool(loop))
        object.__setattr__(self, "max_clients", int(max_clients))
        object.__setattr__(self, "auth_mode", str(auth_mode))
        object.__setattr__(self, "username", str(username))
        object.__setattr__(self, "password", str(password))
        object.__setattr__(self, "realm", str(realm))
        object.__setattr__(self, "socket_timeout_seconds", float(socket_timeout_seconds))
        object.__setattr__(self, "inference_mode", str(inference_mode))
        if not 0 <= int(self.port) <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if not math.isfinite(self.send_interval_seconds) or self.send_interval_seconds <= 0.0:
            raise ValueError("send_interval_seconds must be positive and finite")
        if int(self.max_clients) <= 0:
            raise ValueError("max_clients must be positive")
        if self.inference_mode not in {"causal", "queue", "package"}:
            raise ValueError("inference_mode must be causal, queue, or package")
        resource = str(self.resource)
        if not resource.startswith("/"):
            raise ValueError("resource must start with '/'")


class RRTSPServer:
    """Serve a finite or looping ``StreamSource`` over R-RTSP.

    The server publishes PT=97 track observations and optional PT=96 RD
    frames. An inference factory under ``core.server.model`` can supply the
    matching PT=98 classification results without coupling the transport to a
    particular ONNX model.
    """

    def __init__(
        self,
        source: StreamSource,
        *,
        config: RRTSPServerConfig | None = None,
        inference_factory=None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.source = source
        self.inference_factory = inference_factory
        self.config = config or RRTSPServerConfig()
        self.on_log = on_log or (lambda message: None)
        self.authenticator = auth.DigestAuthenticator(
            mode=self.config.auth_mode,
            username=self.config.username,
            password=self.config.password,
            realm=self.config.realm,
        )
        self._socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._sessions: dict[str, ViewerSession] = {}
        self._connection_sessions: dict[int, set[str]] = {}
        self._session_owners: dict[str, int] = {}
        self._connections: set[socket.socket] = set()
        self._ssrc_rd = random.randint(1, 0xFFFFFFFF)
        self._ssrc_track = random.randint(1, 0xFFFFFFFF)
        self._ssrc_inference = random.randint(1, 0xFFFFFFFF)
        self._bound_port = int(self.config.port)

    @classmethod
    def from_asd(
        cls,
        path,
        *,
        config: RRTSPServerConfig | None = None,
        timestamp_unit: str | None = None,
        amplitude_mode: str | None = None,
        max_events: int | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> "RRTSPServer":
        """Build a replay server from one standard ASD package."""

        from ..source.asd import (
            ASDStreamSource,
            MODEL_CONTEXT_EVENTS,
            QUEUE_MAX_EVENTS,
        )

        effective_config = config or RRTSPServerConfig()
        if max_events is None:
            max_events = (
                QUEUE_MAX_EVENTS
                if effective_config.inference_mode == "queue"
                else MODEL_CONTEXT_EVENTS
            )

        return cls(
            ASDStreamSource(
                path,
                timestamp_unit=timestamp_unit,
                amplitude_mode=amplitude_mode,
                max_events=max_events,
            ),
            config=effective_config,
            on_log=on_log,
        )

    @property
    def host(self) -> str:
        return self.config.host

    @property
    def port(self) -> int:
        return self._bound_port

    @property
    def url(self) -> str:
        host = self.config.host if self.config.host not in {"", "0.0.0.0"} else "127.0.0.1"
        return f"rtsp://{host}:{self.port}{self.config.resource}"

    def is_running(self) -> bool:
        return bool(self._accept_thread and self._accept_thread.is_alive())

    def start(self) -> None:
        if self.is_running():
            return
        if not self.source.events:
            raise ValueError("RRTSP source has no events")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.config.host, self.config.port))
        listener.listen(int(self.config.max_clients))
        listener.settimeout(0.5)
        self._socket = listener
        self._bound_port = int(listener.getsockname()[1])
        self._stop.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="rrtsp-accept",
            daemon=True,
        )
        self._accept_thread.start()
        self.on_log(f"R-RTSP server started: {self.url}")

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._socket is not None:
                self._socket.close()
        except OSError:
            pass
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._connection_sessions.clear()
            self._session_owners.clear()
            connections = list(self._connections)
            self._connections.clear()
        for connection in connections:
            try:
                connection.close()
            except OSError:
                pass
        for session in sessions:
            session.close()
        thread = self._accept_thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._accept_thread = None
        self._socket = None
        self.on_log("R-RTSP server stopped")

    def __enter__(self) -> "RRTSPServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _accept_loop(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                conn, address = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(float(self.config.socket_timeout_seconds))
            with self._lock:
                self._connections.add(conn)
            threading.Thread(
                target=self._connection_loop,
                args=(conn, address),
                name=f"rrtsp-client-{address[0]}:{address[1]}",
                daemon=True,
            ).start()

    def _connection_loop(self, conn: socket.socket, address) -> None:
        connection_key = id(conn)
        self._connection_sessions.setdefault(connection_key, set())
        try:
            with conn:
                while not self._stop.is_set():
                    try:
                        request = rtsp.read_request(conn)
                    except socket.timeout:
                        if self._connection_is_playing(connection_key):
                            continue
                        break
                    except (ConnectionError, OSError, ValueError) as exc:
                        self.on_log(f"R-RTSP request read failed from {address[0]}: {exc}")
                        break
                    if request is None:
                        break
                    try:
                        self._route(conn, request, address, connection_key)
                    except (OSError, ValueError, RuntimeError) as exc:
                        self.on_log(f"R-RTSP request failed from {address[0]}: {exc}")
                        try:
                            rtsp.send_response(
                                conn,
                                500,
                                cseq=rtsp.header_get(request.headers, "CSeq", "0"),
                            )
                        except OSError:
                            break
        finally:
            with self._lock:
                self._connections.discard(conn)
                session_ids = self._connection_sessions.pop(connection_key, set())
                sessions = []
                for value in session_ids:
                    self._session_owners.pop(value, None)
                    sessions.append(self._sessions.pop(value, None))
            for session in sessions:
                if session is not None:
                    session.close()

    def _connection_is_playing(self, connection_key: int) -> bool:
        with self._lock:
            session_ids = self._connection_sessions.get(connection_key, set())
            return any(
                (session := self._sessions.get(session_id)) is not None
                and session.playing.is_set()
                for session_id in session_ids
            )

    def _route(self, conn: socket.socket, request: rtsp.Request, address, connection_key: int) -> None:
        cseq = rtsp.header_get(request.headers, "CSeq", "0")
        if not self._valid_resource(request.uri):
            rtsp.send_response(conn, 404, cseq=cseq)
            return
        if not self.authenticator.valid(request.headers, request.method, request.uri):
            rtsp.send_response(
                conn,
                401,
                cseq=cseq,
                headers={"WWW-Authenticate": self.authenticator.challenge()},
            )
            return
        method = request.method
        if method == "OPTIONS":
            rtsp.send_response(
                conn,
                200,
                cseq=cseq,
                headers={
                    "Public": "OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, GET_PARAMETER, SET_PARAMETER",
                },
            )
            return
        if method == "DESCRIBE":
            self._describe(conn, request, cseq)
            return
        if method == "SETUP":
            self._setup(conn, request, address, cseq, connection_key)
            return
        session_id = rtsp.header_get(request.headers, "Session").split(";", 1)[0].strip()
        with self._lock:
            session = (
                self._sessions.get(session_id)
                if self._session_owners.get(session_id) == connection_key
                else None
            )
        if session is None:
            rtsp.send_response(conn, 454, cseq=cseq)
            return
        if method == "PLAY":
            session.play()
            rtsp.send_response(
                conn,
                200,
                cseq=cseq,
                headers={"Session": self._session_header(session), "RTP-Info": self._rtp_info(session)},
            )
            return
        if method == "PAUSE":
            session.pause()
            rtsp.send_response(conn, 200, cseq=cseq, headers={"Session": self._session_header(session)})
            return
        if method == "TEARDOWN":
            with self._lock:
                self._sessions.pop(session.id, None)
                self._session_owners.pop(session.id, None)
                self._connection_sessions.get(connection_key, set()).discard(session.id)
            session.close()
            rtsp.send_response(conn, 200, cseq=cseq)
            return
        if method == "GET_PARAMETER":
            body = (
                f"events: {len(self.source.events)}\r\n"
                f"send_interval_seconds: {session.send_interval_seconds:g}\r\n"
                f"loop: {str(self.config.loop).lower()}\r\n"
                f"rd: {str(self.source.has_rd).lower()}\r\n"
                f"inference: {str(self.inference_factory is not None).lower()}\r\n"
            ).encode("utf-8")
            rtsp.send_response(conn, 200, cseq=cseq, body=body, content_type="text/parameters")
            return
        if method == "SET_PARAMETER":
            requested = self._parameter_float(request.body, "send_interval_seconds")
            if requested is None:
                rtsp.send_response(conn, 400, cseq=cseq)
                return
            session.set_send_interval_seconds(requested)
            body = f"send_interval_seconds: {session.send_interval_seconds:g}\r\n".encode(
                "utf-8"
            )
            rtsp.send_response(
                conn,
                200,
                cseq=cseq,
                headers={"Session": self._session_header(session)},
                body=body,
                content_type="text/parameters",
            )
            return
        rtsp.send_response(conn, 501, cseq=cseq)

    @staticmethod
    def _parameter_float(body: bytes, name: str) -> float | None:
        prefix = f"{name}:".lower()
        for raw_line in body.decode("utf-8", "replace").splitlines():
            line = raw_line.strip()
            if not line.lower().startswith(prefix):
                continue
            try:
                value = float(line.split(":", 1)[1].strip())
            except (IndexError, TypeError, ValueError):
                return None
            return value if math.isfinite(value) and value > 0.0 else None
        return None

    def _valid_resource(self, uri: str) -> bool:
        path = rtsp.request_path(uri)
        resource = self.config.resource.rstrip("/") or "/"
        return path == resource or path.startswith(resource + "/")

    def _describe(self, conn: socket.socket, request: rtsp.Request, cseq: str) -> None:
        host = rtsp.header_get(request.headers, "Host")
        base = rtsp.base_url(request.uri, host, self.config.host, self.port)
        local_ip = conn.getsockname()[0]
        sdp = rtsp.build_sdp(
            base=base,
            server_ip=local_ip,
            ssrc_rd=self._ssrc_rd,
            ssrc_track=self._ssrc_track,
            has_rd=self.source.has_rd,
            has_track=True,
            update_rate_hz=1.0 / self.config.send_interval_seconds,
            max_packet_size=MAX_PACKET_SIZE,
            rd_shape=self.source.first_rd_shape,
            has_inference=self.inference_factory is not None,
            ssrc_inference=self._ssrc_inference,
        )
        rtsp.send_response(
            conn,
            200,
            cseq=cseq,
            headers={"Content-Base": base + "/"},
            body=sdp.encode("utf-8"),
            content_type="application/sdp",
        )

    def _setup(self, conn: socket.socket, request: rtsp.Request, address, cseq: str, connection_key: int) -> None:
        track_id = rtsp.media_track_id(request.uri)
        if track_id is None:
            rtsp.send_response(conn, 400, cseq=cseq)
            return
        if track_id == 0 and not self.source.has_rd:
            rtsp.send_response(conn, 461, cseq=cseq)
            return
        if track_id == 2 and self.inference_factory is None:
            rtsp.send_response(conn, 461, cseq=cseq)
            return
        if track_id not in (0, 1, 2):
            rtsp.send_response(conn, 461, cseq=cseq)
            return
        session_id = rtsp.header_get(request.headers, "Session").split(";", 1)[0].strip()
        created_session = False
        with self._lock:
            if session_id and self._session_owners.get(session_id) not in {None, connection_key}:
                rtsp.send_response(conn, 454, cseq=cseq)
                return
            session = self._sessions.get(session_id) if session_id else None
            if session is None:
                if len(self._sessions) >= int(self.config.max_clients):
                    rtsp.send_response(conn, 503, cseq=cseq)
                    return
                session_id = f"{random.randint(1, 0xFFFFFFFF):08x}"
                while session_id in self._sessions:
                    session_id = f"{random.randint(1, 0xFFFFFFFF):08x}"
                session = ViewerSession(
                    session_id,
                    self.source,
                    send_interval_seconds=self.config.send_interval_seconds,
                    loop=self.config.loop,
                    inference_mode=self.config.inference_mode,
                    ssrc_rd=self._ssrc_rd,
                    ssrc_track=self._ssrc_track,
                    ssrc_inference=self._ssrc_inference,
                    inference_factory=self.inference_factory,
                    on_log=self.on_log,
                )
                self._sessions[session.id] = session
                self._connection_sessions.setdefault(connection_key, set()).add(session.id)
                self._session_owners[session.id] = connection_key
                created_session = True
        try:
            transport = rtsp.parse_transport(rtsp.header_get(request.headers, "Transport"))
            media = session.setup_media(track_id, address[0], transport)
        except ValueError as exc:
            if created_session:
                with self._lock:
                    self._sessions.pop(session.id, None)
                    self._session_owners.pop(session.id, None)
                    self._connection_sessions.get(connection_key, set()).discard(session.id)
                session.close()
            rtsp.send_response(conn, 461, cseq=cseq)
            self.on_log(f"SETUP rejected: {exc}")
            return
        transport_out = (
            "RTP/AVP;unicast;"
            f"client_port={media.client_rtp_port}-{media.client_rtcp_port};"
            f"server_port={media.rtp_port}-{media.rtcp_port};"
            f"ssrc={media.ssrc:08x}"
        )
        rtsp.send_response(
            conn,
            200,
            cseq=cseq,
            headers={"Session": self._session_header(session), "Transport": transport_out},
        )

    @staticmethod
    def _session_header(session: ViewerSession) -> str:
        return f"{session.id};timeout=60"

    @staticmethod
    def _rtp_info(session: ViewerSession) -> str:
        values = []
        for track_id, media in session.media.items():
            if media.setup:
                values.append(
                    f"url=trackID={track_id};seq={media.sequence};rtptime={media.last_timestamp}"
                )
        return ",".join(values)


__all__ = ["RRTSPServer", "RRTSPServerConfig"]
