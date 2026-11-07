# -*- coding: utf-8 -*-
"""UDP 端口分配工具。"""
from __future__ import annotations

import socket


def bind_udp_pair(host: str = ""):
    """绑定连续的 RTP/RTCP UDP 端口，返回 sockets 和两个端口。"""
    for _ in range(40):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind((host, 0))
            rtp_port = probe.getsockname()[1]
        finally:
            probe.close()

        rtcp_port = rtp_port + 1
        if rtcp_port > 65535:
            continue
        try:
            rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rtp.bind((host, rtp_port))
            rtcp.bind((host, rtcp_port))
            return rtp, rtcp, rtp_port, rtcp_port
        except OSError:
            try:
                rtp.close()
            except Exception:
                pass
            try:
                rtcp.close()
            except Exception:
                pass
    raise OSError("无法分配连续 UDP 端口")


def detect_lan_ip() -> str:
    """获取本机用于访问 IPv4 网络的首选局域网 IP。

    UDP connect 不会真正发出数据包，只让操作系统根据路由表选择本机
    出口网卡；失败时再尝试主机名解析。
    """
    targets = (("8.8.8.8", 80), ("114.114.114.114", 80))
    for target in targets:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(target)
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("169.254."):
                return ip
        except OSError:
            pass
        finally:
            sock.close()

    try:
        infos = socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET,
            socket.SOCK_DGRAM)
        for info in infos:
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")):
                return ip
    except OSError:
        pass
    return "127.0.0.1"
