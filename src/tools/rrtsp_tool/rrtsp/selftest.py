# -*- coding: utf-8 -*-
"""R-RTSP v1 自检：实际启动服务端和观看端进行端到端验证。"""
from __future__ import annotations

import socket
import time

from . import payloads, rtp
from .client import RRTSPViewerClient
from .server import RRTSPServer


def _library() -> str:
    import os
    root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "library")


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _reassemble_rd(packets):
    groups = {}
    for packet in packets:
        header = payloads.parse_rd_header(packet["payload"])
        groups.setdefault(header["frame_sequence"], []).append(
            (packet["sequence"], packet))
    complete = []
    for seq, items in groups.items():
        items.sort(key=lambda x: x[0])
        if items[-1][1]["marker"]:
            body = b"".join(p["payload"][20:] for _seq, p in items)
            complete.append((payloads.parse_rd_header(items[0][1]["payload"]), body))
    return complete


def run_selftest() -> int:
    lib = _library()
    rd_asd = f"{lib}/sessions/session_20250616120945_seq3178_unknownl2.asd"
    track_asd = f"{lib}/tracks/track_0001_drone.asd"

    port = _free_port()
    server = RRTSPServer("127.0.0.1", port, rd_asd, 3)
    server.start(); time.sleep(0.2)
    client = RRTSPViewerClient("127.0.0.1", port); client.connect()
    status, _h, sdp = client.describe()
    assert status == 200 and b"radar-rdmap" in sdp and b"radar-track" in sdp
    client.setup(1); client.setup(0); client.play()
    packets = client.receive_for(1.2)
    rtcp_reports = []
    for track_id in (0, 1):
        sock = client.media[track_id]["rtcp_socket"]
        sock.settimeout(0.5)
        try:
            data, _addr = sock.recvfrom(2048)
            rtcp_reports.extend(rtp.parse_rtcp(data))
        except socket.timeout:
            pass
    assert any(x["type"] == rtp.PT_RTCP_SR for x in rtcp_reports)
    client.teardown(); server.stop()
    frames = _reassemble_rd(packets[0])
    # 新封包保留原始 MAT 形状；该场次首帧为 31x226。
    assert frames and frames[0][0]["range_bins"] == 31
    assert frames[0][0]["doppler_bins"] == 226
    assert len(packets[1]) >= 2

    port = _free_port()
    server = RRTSPServer("127.0.0.1", port, track_asd, 3)
    server.start(); time.sleep(0.2)
    client = RRTSPViewerClient("127.0.0.1", port); client.connect()
    status, _h, sdp = client.describe()
    assert status == 200 and b"radar-track" in sdp and b"radar-rdmap" not in sdp
    client.setup(1); client.play()
    packets = client.receive_for(0.8)
    client.teardown(); server.stop()
    assert packets[1]

    port = _free_port()
    server = RRTSPServer(
        "127.0.0.1", port, rd_asd, 3, auth_mode="digest", password="secret")
    server.start(); time.sleep(0.2)
    good = RRTSPViewerClient("127.0.0.1", port, "radar", "viewer", "secret")
    good.connect(); status, _h, _body = good.describe(); assert status==200; good.close()
    bad = RRTSPViewerClient("127.0.0.1", port, "radar", "viewer", "wrong")
    bad.connect(); status, _h, _body = bad.describe(); assert status==401; bad.close()
    server.stop()
    print("R-RTSP SELFTEST OK"); return 0


if __name__ == "__main__":
    raise SystemExit(run_selftest())

