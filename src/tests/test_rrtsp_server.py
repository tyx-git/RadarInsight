"""Contract tests for the RadarInsight R-RTSP backend."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pytest

from core.server import (
    FullCascadeInferenceFactory,
    FullCascadePackageInferenceFactory,
    RRTSPEvent,
    RRTSPRDFrame,
    RRTSPServer,
    RRTSPServerConfig,
)
from core.server.protocol.payloads import INFERENCE_RESULT, TLV_HEADER
from core.server.protocol.rtp import parse_rtp_packet
from radar_ln3.data.contracts import EventData


@dataclass
class _Source:
    _events: tuple[RRTSPEvent, ...]

    @property
    def events(self):
        return self._events

    @property
    def has_rd(self):
        return True

    @property
    def first_rd_shape(self):
        return (2, 3)

    @property
    def source_name(self):
        return "fixture"

    @property
    def metadata(self):
        return {}


def _request(sock: socket.socket, method: str, uri: str, cseq: int, *, session: str = "", transport: str = ""):
    lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {cseq}"]
    if session:
        lines.append(f"Session: {session}")
    if transport:
        lines.append(f"Transport: {transport}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
    raw = bytearray()
    while b"\r\n\r\n" not in raw:
        raw.extend(sock.recv(1))
    header, body = bytes(raw).split(b"\r\n\r\n", 1)
    lines = header.decode("utf-8").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        key, value = line.split(":", 1)
        headers[key.lower()] = value.strip()
    content_length = int(headers.get("content-length", "0"))
    while len(body) < content_length:
        body += sock.recv(content_length - len(body))
    return status, headers, body[:content_length]


def test_rrtsp_server_handshake_and_media() -> None:
    amplitude = np.arange(6, dtype=np.float32).reshape(2, 3)
    phase = np.zeros_like(amplitude)
    source = _Source(
        (
            RRTSPEvent(
                event_index=0,
                timestamp_seconds=0.0,
                track_id=7,
                state=1,
                class_id=1,
                tr_values={"range_m": 100.0, "azimuth_deg": 2.0},
                rd_frames=(RRTSPRDFrame(0.0, amplitude, phase),),
            ),
        )
    )
    server = RRTSPServer(
        source,
        config=RRTSPServerConfig(
            host="127.0.0.1",
            port=0,
            playback_rate=100.0,
            loop=False,
        ),
    )
    server.start()
    tcp = socket.create_connection(("127.0.0.1", server.port), timeout=3.0)
    tcp.settimeout(3.0)
    track = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    track.bind(("127.0.0.1", 0))
    track.settimeout(3.0)
    track_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    track_rtcp.bind(("127.0.0.1", 0))
    rd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rd.bind(("127.0.0.1", 0))
    rd.settimeout(3.0)
    rd_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rd_rtcp.bind(("127.0.0.1", 0))
    try:
        assert _request(tcp, "OPTIONS", server.url, 1)[0] == 200
        status, _, sdp = _request(tcp, "DESCRIBE", server.url, 2)
        assert status == 200
        assert b"radar-rdmap/90000" in sdp
        status, headers, _ = _request(
            tcp,
            "SETUP",
            server.url + "/trackID=1",
            3,
            transport=(
                "RTP/AVP;unicast;"
                f"client_port={track.getsockname()[1]}-{track_rtcp.getsockname()[1]}"
            ),
        )
        assert status == 200
        session = headers["session"].split(";", 1)[0]
        status, _, _ = _request(
            tcp,
            "SETUP",
            server.url + "/trackID=0",
            4,
            session=session,
            transport=(
                "RTP/AVP;unicast;"
                f"client_port={rd.getsockname()[1]}-{rd_rtcp.getsockname()[1]}"
            ),
        )
        assert status == 200
        assert _request(tcp, "PLAY", server.url, 5, session=session)[0] == 200
        track_packet, _ = track.recvfrom(4096)
        rd_packet, _ = rd.recvfrom(4096)
        assert parse_rtp_packet(track_packet)["payload_type"] == 97
        assert parse_rtp_packet(rd_packet)["payload_type"] == 96
        assert _request(tcp, "TEARDOWN", server.url, 6, session=session)[0] == 200
    finally:
        tcp.close()
        for sock in (track, track_rtcp, rd, rd_rtcp):
            sock.close()
        server.stop()


class _FakeInferenceSession:
    def predict(self, event):
        return {
            "track_id": int(event.track_id),
            "class_id": 2,
            "confidence": 0.875,
        }

    def predict_batch(self, events):
        return [self.predict(event) for event in events]


@pytest.mark.parametrize("mode", ["causal", "queue"])
def test_client_controls_session_interval_and_receives_complete_batches(mode, monkeypatch):
    # Exercise the real client/server control and UDP media path together.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "AeroSight" / "client"))
    from aerosight.pipeline.rrtsp_source import RRTSPConnection

    class SlowStartInference(_FakeInferenceSession):
        primed = False

        def predict_batch(self, events):
            if not self.primed:
                time.sleep(0.25)
                self.primed = True
            return super().predict_batch(events)

    amplitude = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = _Source(tuple(
        RRTSPEvent(
            event_index=step,
            timestamp_seconds=step * 3.0,
            track_id=track_id,
            state=0,
            class_id=5,
            tr_values={"range_m": 100.0 + step, "azimuth_deg": 2.0, "elevation_deg": 0.0},
            rd_frames=(RRTSPRDFrame(step * 3.0, amplitude, np.zeros_like(amplitude)),),
            parent_sample_id=f"target-{track_id}",
        )
        for step in range(3)
        for track_id in (11, 22)
    ))
    server = RRTSPServer(
        source,
        config=RRTSPServerConfig(
            host="127.0.0.1", port=0, send_interval_seconds=3.0,
            loop=False, inference_mode=mode,
        ),
        inference_factory=SlowStartInference,
    )
    server.start()
    client = RRTSPConnection(server.url, send_interval_seconds=0.08)
    other_client = RRTSPConnection(server.url, send_interval_seconds=0.4)
    try:
        client.open()
        other_client.open()
        assert client.send_interval_seconds == pytest.approx(0.08)
        assert client.update_rate_hz == pytest.approx(12.5)
        assert other_client.send_interval_seconds == pytest.approx(0.4)
        assert server.config.send_interval_seconds == 3.0
        _, headers = client.command("OPTIONS")
        assert "SET_PARAMETER" in headers["public"]

        for value in ("0", "-1", "nan", "inf", "text"):
            client.command(
                "SET_PARAMETER", body=f"send_interval_seconds: {value}\r\n",
                accepted_statuses=(400,),
            )
        client.command("GET_PARAMETER")
        assert b"send_interval_seconds: 0.08\r\n" in client.last_response_body
        other_client.command("GET_PARAMETER")
        assert b"send_interval_seconds: 0.4\r\n" in other_client.last_response_body

        timestamps = []
        for step in range(3):
            timestamp, tracks, rd = client.receive_paired(timeout=2.0)
            timestamps.append(timestamp)
            assert [track["id"] for track in tracks] == [11, 22]
            assert [track["fields"]["range_m"] for track in tracks] == [100.0 + step] * 2
            assert set(client.last_remote_labels) == {11, 22}
            assert client.last_remote_labels[11] == ("bird", 0.875)
            assert rd is not None
            np.testing.assert_array_equal(rd[0], amplitude)
        # The selected interval can shorten the 3 s source spacing. Even slow
        # queue priming must not cause two completed batches to burst together.
        gaps = np.diff(timestamps) / 90_000.0
        assert np.all(gaps >= 0.05), gaps
        assert np.all(gaps < 0.5), gaps
    finally:
        client.close()
        other_client.close()
        server.stop()


def test_rrtsp_optional_pt98_classification_channel() -> None:
    source = _Source(
        (
            RRTSPEvent(
                event_index=0,
                timestamp_seconds=0.0,
                track_id=19,
                state=0,
                class_id=5,
                tr_values={"range_m": 250.0},
                parent_sample_id="p1",
            ),
        )
    )
    server = RRTSPServer(
        source,
        config=RRTSPServerConfig(
            host="127.0.0.1",
            port=0,
            playback_rate=50.0,
            loop=False,
        ),
        inference_factory=_FakeInferenceSession,
    )
    server.start()
    tcp = socket.create_connection(("127.0.0.1", server.port), timeout=3.0)
    tcp.settimeout(3.0)
    track_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    track_rtp.bind(("127.0.0.1", 0))
    track_rtp.settimeout(3.0)
    track_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    track_rtcp.bind(("127.0.0.1", 0))
    inference_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inference_rtp.bind(("127.0.0.1", 0))
    inference_rtp.settimeout(3.0)
    inference_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inference_rtcp.bind(("127.0.0.1", 0))
    try:
        status, _, sdp = _request(tcp, "DESCRIBE", server.url, 1)
        assert status == 200
        assert b"radar-classification/90000" in sdp
        status, headers, _ = _request(
            tcp,
            "SETUP",
            server.url + "/trackID=1",
            2,
            transport=(
                "RTP/AVP;unicast;"
                f"client_port={track_rtp.getsockname()[1]}-{track_rtcp.getsockname()[1]}"
            ),
        )
        assert status == 200
        session = headers["session"].split(";", 1)[0]
        status, _, _ = _request(
            tcp,
            "SETUP",
            server.url + "/trackID=2",
            3,
            session=session,
            transport=(
                "RTP/AVP;unicast;"
                f"client_port={inference_rtp.getsockname()[1]}-"
                f"{inference_rtcp.getsockname()[1]}"
            ),
        )
        assert status == 200
        assert _request(tcp, "PLAY", server.url, 4, session=session)[0] == 200
        packet, _ = inference_rtp.recvfrom(4096)
        parsed = parse_rtp_packet(packet)
        assert parsed["payload_type"] == 98
        frame_type, count, _, _ = struct.unpack_from(">BHBB", parsed["payload"])
        tlv_type, length = TLV_HEADER.unpack_from(parsed["payload"], 5)
        track_id, class_id, confidence = INFERENCE_RESULT.unpack_from(
            parsed["payload"], 5 + TLV_HEADER.size
        )
        assert (frame_type, count, tlv_type, length) == (2, 1, 2, INFERENCE_RESULT.size)
        assert track_id == 19
        assert class_id == 2
        assert confidence == pytest.approx(0.875)
        assert parse_rtp_packet(track_rtp.recvfrom(4096)[0])["payload_type"] == 97
        assert _request(tcp, "TEARDOWN", server.url, 5, session=session)[0] == 200
    finally:
        tcp.close()
        for sock in (track_rtp, track_rtcp, inference_rtp, inference_rtcp):
            sock.close()
        server.stop()


def test_active_playback_survives_idle_rtsp_control_timeout() -> None:
    source = _Source(
        tuple(
            RRTSPEvent(
                event_index=index,
                timestamp_seconds=float(index),
                track_id=19,
                state=0,
                class_id=5,
                tr_values={"range_m": 250.0 + index},
            )
            for index in range(2)
        )
    )
    server = RRTSPServer(
        source,
        config=RRTSPServerConfig(
            host="127.0.0.1",
            port=0,
            playback_rate=50.0,
            loop=True,
            socket_timeout_seconds=0.1,
        ),
    )
    server.start()
    tcp = socket.create_connection(("127.0.0.1", server.port), timeout=3.0)
    tcp.settimeout(3.0)
    track_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    track_rtp.bind(("127.0.0.1", 0))
    track_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    track_rtcp.bind(("127.0.0.1", 0))
    try:
        status, headers, _ = _request(
            tcp,
            "SETUP",
            server.url + "/trackID=1",
            1,
            transport=(
                "RTP/AVP;unicast;"
                f"client_port={track_rtp.getsockname()[1]}-"
                f"{track_rtcp.getsockname()[1]}"
            ),
        )
        assert status == 200
        session = headers["session"].split(";", 1)[0]
        assert _request(tcp, "PLAY", server.url, 2, session=session)[0] == 200
        time.sleep(0.35)
        status, _, body = _request(
            tcp,
            "GET_PARAMETER",
            server.url,
            3,
            session=session,
        )
        assert status == 200
        assert b"events: 2" in body
    finally:
        tcp.close()
        track_rtp.close()
        track_rtcp.close()
        server.stop()


def test_package_inference_is_cached_for_all_replayed_events() -> None:
    class _Pipeline:
        calls = 0

        def predict_sequence(self, sequence):
            self.calls += 1
            return {
                "action": "BirdTarget",
                "reason_code": "KNOWN_DBB_CLASS_ACCEPTED",
                "cascade_probabilities": [0.1, 0.7, 0.1, 0.05, 0.03, 0.02],
            }

    pipeline = _Pipeline()
    session = FullCascadePackageInferenceFactory(pipeline)()
    first = RRTSPEvent(
        event_index=0,
        timestamp_seconds=0.0,
        track_id=81,
        state=0,
        class_id=5,
        tr_values={},
        parent_sample_id="parent-1",
        model_sequence=object(),
    )
    second = RRTSPEvent(
        event_index=1,
        timestamp_seconds=0.1,
        track_id=81,
        state=0,
        class_id=5,
        tr_values={},
        parent_sample_id="parent-1",
        model_sequence=object(),
    )
    first_result = session.predict(first)
    second_result = session.predict(second)
    assert pipeline.calls == 1
    assert first_result["class_id"] == second_result["class_id"] == 2
    assert first_result["confidence"] == second_result["confidence"] == pytest.approx(0.7)
    assert first_result["is_final"] and second_result["is_final"]


def test_causal_inference_keeps_independent_histories_for_interleaved_targets() -> None:
    class _Pipeline:
        lengths = []

        def predict_sequence(self, sequence):
            self.lengths.append((sequence.parent_sample_id, len(sequence.events)))
            return {
                "action": "BirdTarget",
                "reason_code": "KNOWN_DBB_CLASS_ACCEPTED",
                "cascade_probabilities": [0.1, 0.7, 0.1, 0.05, 0.03, 0.02],
            }

    def event(parent: str, track_id: int, index: int) -> RRTSPEvent:
        model_event = EventData(
            sample_id=f"{parent}:event-{index}",
            parent_sample_id=parent,
            track_id=str(track_id),
            causal_event_index=index,
            label_id=4,
            label_name="UnknownTarget",
            split_name="inference",
            tr_values={"time_seconds": float(index)},
            rd_values=None,
            tr_available=True,
            rd_available=False,
        )
        return RRTSPEvent(
            event_index=index,
            timestamp_seconds=float(index),
            track_id=track_id,
            state=0,
            class_id=5,
            tr_values={"range_m": 100.0},
            parent_sample_id=parent,
            model_event=model_event,
        )

    pipeline = _Pipeline()
    session = FullCascadeInferenceFactory(pipeline)()
    session.predict(event("parent-a", 11, 0))
    session.predict(event("parent-b", 22, 0))
    session.predict(event("parent-a", 11, 1))
    session.predict(event("parent-b", 22, 1))
    session.predict(event("parent-a", 11, 0))

    assert pipeline.lengths == [
        ("parent-a", 1),
        ("parent-b", 1),
        ("parent-a", 2),
        ("parent-b", 2),
        ("parent-a", 1),
    ]
