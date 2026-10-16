"""Regression coverage for large radar batches and damaged UDP packets."""
from __future__ import annotations

import socket
import struct
import threading
from copy import deepcopy
import queue

import pytest

from aerosight.core.radar import RadarObservation
from aerosight.app_config import default_instance
from aerosight.pipeline.receiver import InstanceSupervisor
from aerosight.pipeline.rrtsp_source import (
    INFERENCE_RESULT_HEAD,
    RRTSPConnection,
    RTP_HEAD,
    TLV_HEAD,
    TRACK_FRAME_HEAD,
    TRACK_VALUE_HEAD,
    parse_track_payload,
)


def track_payload(count: int) -> bytes:
    fields = struct.pack(">8f", 1000.0, -20.0, 5.0, -14.0, 18.0, 10.0, 30.0, 40.0)
    tracks = []
    for track_id in range(count):
        value = TRACK_VALUE_HEAD.pack(track_id, 0, 5, 0xFF) + fields
        tracks.append(TLV_HEAD.pack(1, len(value)) + value)
    return TRACK_FRAME_HEAD.pack(1, count, 0, 0) + b"".join(tracks)


@pytest.mark.parametrize("payload", [
    b"",
    TRACK_FRAME_HEAD.pack(1, 1, 0, 0),
    TRACK_FRAME_HEAD.pack(1, 1, 0, 0) + TLV_HEAD.pack(1, 1) + b"\x00",
    track_payload(1)[:-1],
    track_payload(1) + b"\x00",
])
def test_incomplete_track_packets_are_rejected(payload):
    with pytest.raises(ValueError):
        parse_track_payload(payload)


@pytest.mark.parametrize("fields", [
    {"range_m": 1000.0, "elevation_deg": 5.0},
    {"range_m": 1000.0, "azimuth_deg": 5.0},
    {"range_m": float("nan"), "azimuth_deg": 5.0, "elevation_deg": 5.0},
    {"range_m": -1.0, "azimuth_deg": 5.0, "elevation_deg": 5.0},
    {"range_m": 1000.0, "azimuth_deg": float("inf"), "elevation_deg": 5.0},
    {"range_m": 1000.0, "azimuth_deg": 181.0, "elevation_deg": 5.0},
    {"range_m": 1000.0, "azimuth_deg": 5.0, "elevation_deg": 91.0},
])
def test_invalid_positions_cannot_become_zero_angle_observations(fields):
    with pytest.raises(ValueError):
        RadarObservation.from_track("radar", {"id": 1, "fields": fields})


@pytest.mark.parametrize("channel,count", [(97, 100), (98, 400)])
def test_udp_receiver_survives_bad_packet_and_delivers_large_batch(channel, count):
    if channel == 97:
        payload = track_payload(count)
    else:
        results = []
        for track_id in range(count):
            value = INFERENCE_RESULT_HEAD.pack(track_id, 1, 0.9)
            results.append(TLV_HEAD.pack(2, len(value)) + value)
        payload = TRACK_FRAME_HEAD.pack(2, count, 0, 0) + b"".join(results)
    packet = RTP_HEAD.pack(0x80, channel, 2, 540_000, 7) + payload
    assert len(packet) > 4096

    connection = RRTSPConnection("rtsp://127.0.0.1:8554/radar")
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    if channel == 97:
        connection.rtp = receiver
        receive_loop = connection._track_loop
        result_q = connection.track_q
    else:
        connection.inference_rtp = receiver
        receive_loop = connection._inference_loop
        result_q = connection.inference_q
    worker = threading.Thread(target=receive_loop, daemon=True)
    worker.start()
    try:
        damaged = RTP_HEAD.pack(0x80, channel, 1, 270_000, 7) + payload[:-1]
        sender.sendto(damaged, receiver.getsockname())
        sender.sendto(packet, receiver.getsockname())
        timestamp, results = result_q.get(timeout=2.0)
        assert timestamp == 540_000
        assert len(results) == count
        assert worker.is_alive()
        if channel == 97:
            observations = [RadarObservation.from_track("radar", track)
                            for track in results]
            assert [observation.track_id for observation in observations] == list(range(count))
            assert observations[0].azimuth_deg == -20.0
            assert observations[0].altitude_m > 0.0
        else:
            assert results[count - 1][0] == "drone"
    finally:
        connection._stop.set()
        worker.join(timeout=1.0)
        receiver.close()
        sender.close()
    assert not worker.is_alive()


def test_settings_do_not_restart_receiver_when_only_scan_speed_changes(monkeypatch):
    cfg = {"instances": [default_instance("radar", "Radar", "rrtsp")]}
    supervisor = InstanceSupervisor(cfg, queue.Queue())
    monkeypatch.setattr(supervisor, "_worker_entry", lambda conf, stop: stop.wait())
    try:
        supervisor.start()
        original = supervisor.workers["radar"]
        cfg = deepcopy(cfg)
        cfg["instances"][0]["display"]["frontend_scan_period_seconds"] = 7.5
        supervisor.apply_config(cfg)
        assert supervisor.workers["radar"] is original
        cfg["instances"][0]["source"]["rrtsp"]["send_interval_seconds"] = 0.75
        supervisor.apply_config(cfg)
        assert not original.is_alive()
        assert supervisor.workers["radar"] is not original
        assert supervisor.workers["radar"].is_alive()
    finally:
        supervisor.stop()


def test_apply_cancels_a_receiver_waiting_for_missing_model_labels():
    conn = RRTSPConnection("rtsp://127.0.0.1:8554/radar")
    conn.has_inference = True
    conn.track_q.put((90_000, []))
    stop = threading.Event()
    done = threading.Event()
    errors = []

    def receive():
        try:
            conn.receive_paired(stop_event=stop)
        except TimeoutError as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=receive, daemon=True)
    worker.start()
    try:
        assert not done.wait(timeout=0.15)
        stop.set()
        assert done.wait(timeout=1.0)
        assert len(errors) == 1
    finally:
        stop.set()
        worker.join(timeout=1.0)
