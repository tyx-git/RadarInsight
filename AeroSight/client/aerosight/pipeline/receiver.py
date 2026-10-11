# -*- coding: utf-8 -*-
"""Receiver process: independently supervise radar instances."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _q
import socket
import threading
import time

from aerosight.core.radar import RadarFrame, RadarObservation, StreamStatus
from aerosight.pipeline.rrtsp_source import RRTSPConnection, RTP_CLOCK_HZ
from aerosight.pipeline.simulation import build_simulation_observations

log = logging.getLogger(__name__)


def put_latest(q: mp.Queue, item) -> bool:
    try:
        q.put_nowait(item)
        return True
    except Exception:
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(item)
            return True
        except Exception:
            return False


def receiver_main(cfg: dict, out_q: mp.Queue, cmd_q: mp.Queue,
                  stop_event: mp.Event):
    supervisor = InstanceSupervisor(cfg, out_q)
    supervisor.start()

    while not stop_event.is_set():
        try:
            cmd = cmd_q.get(timeout=0.2)
        except _q.Empty:
            continue
        except (FileNotFoundError, EOFError):
            break
        if cmd.get("cmd") == "apply":
            supervisor.apply_config(cmd.get("cfg", cfg))
        elif cmd.get("cmd") == "stop":
            break

    supervisor.stop()
    log.info("Receiver process exited")


class InstanceSupervisor:
    def __init__(self, cfg: dict, out_q: mp.Queue):
        self.cfg = cfg
        self.out_q = out_q
        self.workers: dict[str, threading.Thread] = {}
        self.stops: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self):
        self.apply_config(self.cfg)

    def stop(self):
        with self._lock:
            events = list(self.stops.values())
        for event in events:
            event.set()
        with self._lock:
            workers = list(self.workers.values())
        for worker in workers:
            worker.join(timeout=1.2)

    def apply_config(self, cfg: dict):
        self.cfg = cfg
        wanted: dict[str, dict] = {}
        for item in cfg.get("instances", []):
            if item.get("enabled", True):
                wanted[item["id"]] = item

        with self._lock:
            for instance_id in set(self.workers) - set(wanted):
                self.stops[instance_id].set()

            for instance_id, conf in wanted.items():
                worker = self.workers.get(instance_id)
                if worker is None or not worker.is_alive():
                    event = threading.Event()
                    worker = threading.Thread(
                        target=self._worker_entry,
                        args=(conf, event),
                        name=f"aerosight-instance-{instance_id}",
                        daemon=True)
                    self.stops[instance_id] = event
                    self.workers[instance_id] = worker
                    worker.start()
                else:
                    self.stops[instance_id].set()
                    worker.join(timeout=0.8)
                    event = threading.Event()
                    worker = threading.Thread(
                        target=self._worker_entry,
                        args=(conf, event),
                        name=f"aerosight-instance-{instance_id}",
                        daemon=True)
                    self.stops[instance_id] = event
                    self.workers[instance_id] = worker
                    worker.start()

        for missing in set(self.stops) - set(wanted):
            worker = self.workers.get(missing)
            if worker:
                worker.join(timeout=0.8)

    def _worker_entry(self, conf: dict, stop: threading.Event):
        source = conf.get("source", {})
        if source.get("kind") == "simulation":
            self._simulation_worker(conf, stop)
        else:
            self._rrtsp_worker(conf, stop)

    def _emit_status(self, conf, state, detail):
        put_latest(
            self.out_q,
            ("status", StreamStatus(
                id=conf["id"], name=conf.get("name", conf["id"]),
                state=state, detail=detail)))

    def _simulation_worker(self, conf: dict, stop: threading.Event):
        self._emit_status(conf, "connected", "\u5185\u7f6e\u4eff\u771f\u822a\u8ff9")
        rate = max(float(conf["source"]["simulation"].get("rate_hz", 5)), 0.5)
        period = 1.0 / rate
        start = time.monotonic()
        frame_id = 0
        next_at = start
        while not stop.is_set():
            now = time.monotonic()
            if now < next_at:
                if stop.wait(min(0.02, next_at - now)):
                    break
                continue
            elapsed = now - start
            observations = build_simulation_observations(elapsed)
            frame = RadarFrame(
                stream_id=conf["id"],
                stream_name=conf.get("name", "\u5185\u7f6e\u4eff\u771f"),
                frame_id=frame_id, timestamp=elapsed,
                observations=observations)
            put_latest(self.out_q, ("frame", frame))
            frame_id += 1
            next_at += period
            if next_at < now:
                next_at = now

    def _rrtsp_worker(self, conf: dict, stop: threading.Event):
        failure_count = 0
        rrtsp = conf["source"].get("rrtsp", {})
        while not stop.is_set():
            self._emit_status(conf, "connecting", "\u6b63\u5728\u5efa\u7acb\u8fde\u63a5")
            conn = None
            try:
                conn = RRTSPConnection(
                    rrtsp.get("url", ""),
                    rrtsp.get("username", ""),
                    rrtsp.get("password", ""))
                conn.open()
                failure_count = 0
                self._emit_status(conf, "connected", "\u6b63\u5728\u63a5\u6536\u822a\u8ff9")
                frame_id = 0
                while not stop.is_set():
                    try:
                        timestamp, tracks, rd = conn.receive_paired(
                            timeout=0.5
                        )
                    except (_q.Empty, socket.timeout, TimeoutError):
                        if conn.stopped:
                            break
                        # A 0.5 s receive timeout is shorter than the normal
                        # replay scan interval.  An empty track queue means
                        # that the next radar batch has not arrived yet; it is
                        # not a connection failure and must not trigger a
                        # reconnect (which would discard trajectory continuity).
                        continue
                    rd_amplitude = []
                    rd_phase = []
                    if rd is not None:
                        amplitude, phase = rd
                        rd_amplitude = amplitude.tolist()
                        rd_phase = phase.tolist()

                    observations = []
                    for track in tracks:
                        if track.get("fields", {}).get("range_m") is None:
                            continue
                        observations.append(
                            RadarObservation.from_track(conf["id"], track)
                        )
                    frame = RadarFrame(
                        stream_id=conf["id"],
                        stream_name=conf.get("name", conf["id"]),
                        frame_id=frame_id,
                        timestamp=float(timestamp) / RTP_CLOCK_HZ,
                        rtp_timestamp=int(timestamp),
                        scan_interval_seconds=conn.scan_interval_seconds,
                        observations=observations,
                        rd_amplitude=rd_amplitude,
                        rd_phase=rd_phase,
                        remote_labels=dict(conn.last_remote_labels))
                    put_latest(self.out_q, ("frame", frame))
                    frame_id += 1
            except Exception as exc:
                failure_count += 1
                detail = self._friendly_error(exc)
                self._emit_status(conf, "error", detail)
                log.warning("Instance %s error: %s", conf.get("name"), detail)
                retry_delay = min(10.0, 2.0 * failure_count)
                if stop.wait(retry_delay):
                    break
            finally:
                if conn is not None:
                    conn.close()

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        if isinstance(exc, ConnectionRefusedError):
            return "\u63a8\u6d41\u672a\u542f\u52a8\u3001\u7aef\u53e3\u672a\u76d1\u542c\uff0c\u6216\u5730\u5740/\u7aef\u53e3\u586b\u5199\u9519\u8bef"
        if isinstance(exc, (socket.timeout, TimeoutError)):
            return "\u8fde\u63a5\u6216\u7b49\u5f85\u6570\u636e\u8d85\u65f6\uff1a\u8bf7\u68c0\u67e5 IP\u3001\u9632\u706b\u5899\u548c\u63a8\u6d41\u72b6\u6001"
        if isinstance(exc, PermissionError):
            return f"\u9274\u6743\u5931\u8d25\uff1a{exc}"
        return f"{type(exc).__name__}: {exc}"
