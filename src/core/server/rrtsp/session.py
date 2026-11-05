"""Per-viewer R-RTSP playback session."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import math
import random
import socket
import threading
import time

from ..protocol import payloads, rtp
from ..source.base import RRTSPEvent, StreamSource


MAX_PACKET_SIZE = 1200
RTP_CLOCK_HZ = 90_000


def _new_ssrc() -> int:
    return random.randint(1, 0xFFFFFFFF)


def _new_sequence() -> int:
    return random.randint(0, 0xFFFF)


def _bind_udp_pair() -> tuple[socket.socket, socket.socket, int, int]:
    """Bind adjacent RTP/RTCP ports without leaving a probing race."""

    for _ in range(40):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(("", 0))
            rtp_port = int(probe.getsockname()[1])
        finally:
            probe.close()
        rtcp_port = rtp_port + 1
        if rtcp_port > 65535:
            continue
        rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rtcp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            rtp_socket.bind(("", rtp_port))
            rtcp_socket.bind(("", rtcp_port))
            return rtp_socket, rtcp_socket, rtp_port, rtcp_port
        except OSError:
            rtp_socket.close()
            rtcp_socket.close()
    raise OSError("unable to allocate adjacent RTP/RTCP ports")


class MediaTransport:
    def __init__(self, track_id: int, name: str, ssrc: int | None = None) -> None:
        self.track_id = int(track_id)
        self.name = str(name)
        self.ssrc = int(ssrc or _new_ssrc())
        self.sequence = _new_sequence()
        self.rtp_socket: socket.socket | None = None
        self.rtcp_socket: socket.socket | None = None
        self.rtp_port = 0
        self.rtcp_port = 0
        self.client_addr = ""
        self.client_rtp_port = 0
        self.client_rtcp_port = 0
        self.packets_sent = 0
        self.octets_sent = 0
        self.last_timestamp = 0
        self.setup = False

    def setup_transport(self, client_addr: str, client_rtp: int, client_rtcp: int) -> None:
        if self.setup:
            if (self.client_addr, self.client_rtp_port, self.client_rtcp_port) != (
                client_addr, int(client_rtp), int(client_rtcp)
            ):
                raise ValueError("media transport already belongs to another client")
            return
        rtp_socket, rtcp_socket, rtp_port, rtcp_port = _bind_udp_pair()
        self.rtp_socket = rtp_socket
        self.rtcp_socket = rtcp_socket
        self.rtp_port = rtp_port
        self.rtcp_port = rtcp_port
        self.client_addr = str(client_addr)
        self.client_rtp_port = int(client_rtp)
        self.client_rtcp_port = int(client_rtcp)
        self.setup = True

    def send_rtp(self, payload: bytes, *, payload_type: int, timestamp: int, marker: bool = False) -> None:
        if not self.setup or self.rtp_socket is None:
            return
        packet = rtp.build_rtp_packet(
            payload_type=payload_type,
            ssrc=self.ssrc,
            sequence=self.sequence,
            timestamp=timestamp,
            payload=payload,
            marker=marker,
        )
        self.rtp_socket.sendto(packet, (self.client_addr, self.client_rtp_port))
        self.sequence = (self.sequence + 1) & 0xFFFF
        self.packets_sent += 1
        self.octets_sent += len(payload)
        self.last_timestamp = int(timestamp) & 0xFFFFFFFF

    def close(self) -> None:
        self.setup = False
        for sock in (self.rtp_socket, self.rtcp_socket):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        self.rtp_socket = None
        self.rtcp_socket = None


class QueuePlaybackBuffer:
    """Ten-target producer/consumer buffer for queue inference mode."""

    MAX_TARGETS = 10
    MAX_QUEUE_SIZE = 60
    LOW_WATERMARK = 29

    def __init__(
        self,
        events: Sequence[RRTSPEvent],
        predictor,
        *,
        loop: bool,
        on_error,
    ) -> None:
        self.predictor = predictor
        self.loop = bool(loop)
        self.on_error = on_error
        grouped: dict[str, list[RRTSPEvent]] = {}
        order: list[str] = []
        for event in events:
            parent = str(event.parent_sample_id or event.track_id)
            if parent not in grouped:
                grouped[parent] = []
                order.append(parent)
            grouped[parent].append(event)
        self._groups = [
            tuple(order[offset:offset + self.MAX_TARGETS])
            for offset in range(0, len(order), self.MAX_TARGETS)
        ]
        self._events_by_parent = {
            parent: tuple(values) for parent, values in grouped.items()
        }
        self._group_index = 0
        self._parents: tuple[str, ...] = ()
        self._queues: dict[str, deque] = {}
        self._cursors: dict[str, int] = {}
        self._primed = False
        self._exhausted = False
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._producer_loop,
            name="rrtsp-queue-producer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.wake()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _select_group_locked(self) -> None:
        if self._parents:
            return
        if self._group_index >= len(self._groups):
            if not self.loop:
                self._exhausted = True
                return
            self._group_index = 0
        self._parents = self._groups[self._group_index]
        self._group_index += 1
        self._queues = {parent: deque() for parent in self._parents}
        self._cursors = {parent: 0 for parent in self._parents}
        self._primed = False

    def _group_finished_locked(self) -> bool:
        return bool(self._parents) and all(
            self._cursors[parent] >= len(self._events_by_parent[parent])
            and not self._queues[parent]
            for parent in self._parents
        )

    def _events_for_fill_locked(self) -> list[RRTSPEvent]:
        self._select_group_locked()
        if self._exhausted:
            return []
        if self._primed and not any(
            len(self._queues[parent]) < self.LOW_WATERMARK
            and self._cursors[parent] < len(self._events_by_parent[parent])
            for parent in self._parents
        ):
            return []
        # Prime to the full 60-point buffer. Once primed, refill the whole
        # active group only when at least one queue falls below 29 points;
        # this keeps the ONNX call batched instead of running one inference
        # for every packet consumed by the sender.
        target_size = self.MAX_QUEUE_SIZE
        selected: list[RRTSPEvent] = []
        for parent in self._parents:
            queue = self._queues[parent]
            events = self._events_by_parent[parent]
            cursor = self._cursors[parent]
            if len(queue) >= target_size or cursor >= len(events):
                continue
            selected.append(events[cursor])
        return selected

    def _append_predictions(
        self,
        events: list[RRTSPEvent],
        predictions: Sequence[Mapping[str, object]],
    ) -> None:
        if len(events) != len(predictions):
            raise ValueError(
                f"queue prediction count mismatch: expected={len(events)}, "
                f"actual={len(predictions)}"
            )
        for event, prediction in zip(events, predictions, strict=True):
            parent = str(event.parent_sample_id or event.track_id)
            queue = self._queues[parent]
            if len(queue) >= self.MAX_QUEUE_SIZE:
                continue
            queue.append((event, dict(prediction)))
            self._cursors[parent] += 1
        self._primed = True

    def _producer_loop(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                self._select_group_locked()
                if self._exhausted:
                    self._condition.notify_all()
                    return
                if self._group_finished_locked():
                    self._parents = ()
                    self._queues = {}
                    self._cursors = {}
                    self._select_group_locked()
                    self._condition.notify_all()
                    continue
                events = self._events_for_fill_locked()
                if not events:
                    self._condition.wait(timeout=0.05)
                    continue
            try:
                predictions = self.predictor.predict_batch(events)
                with self._condition:
                    self._append_predictions(events, predictions)
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                self.on_error(exc)
                return

    def pop_batch(self, *, playing: threading.Event) -> list[tuple[RRTSPEvent, Mapping[str, object]]] | None:
        with self._condition:
            while not self._stop.is_set():
                if not playing.is_set():
                    return None
                if self._error is not None:
                    raise self._error
                self._select_group_locked()
                active_parents = [
                    parent
                    for parent in self._parents
                    if self._queues[parent]
                    or self._cursors[parent] < len(self._events_by_parent[parent])
                ]
                if active_parents and any(
                    not self._queues[parent] for parent in active_parents
                ):
                    self._condition.wait(timeout=0.1)
                    continue
                values = [
                    self._queues[parent].popleft()
                    for parent in active_parents
                    if self._queues[parent]
                ]
                if values:
                    self._condition.notify_all()
                    return values
                if self._exhausted:
                    return None
                self._condition.wait(timeout=0.1)
        return None


class ViewerSession:
    """Replay one source sequence for one RTSP viewer."""

    def __init__(
        self,
        session_id: str,
        source: StreamSource,
        *,
        send_interval_seconds: float,
        loop: bool,
        inference_mode: str = "causal",
        ssrc_rd: int,
        ssrc_track: int,
        ssrc_inference: int,
        inference_factory,
        on_log,
    ) -> None:
        self.id = str(session_id)
        self.source = source
        self.send_interval_seconds = max(float(send_interval_seconds), 1.0e-6)
        self.loop = bool(loop)
        self.inference_mode = str(inference_mode)
        self.on_log = on_log
        self.inference_factory = inference_factory
        self.inference_session = inference_factory() if inference_factory is not None else None
        self.media: dict[int, MediaTransport] = {
            0: MediaTransport(0, "RD map", ssrc_rd),
            1: MediaTransport(1, "track", ssrc_track),
        }
        if self.inference_session is not None:
            self.media[2] = MediaTransport(2, "classification", ssrc_inference)
        self.playing = threading.Event()
        self.alive = True
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._position = 0.0
        self._resume_at: float | None = None
        self._event_cursor = 0
        self._frame_sequence = 0
        self._update_sequence = 0
        self._cycle_origin = 0.0
        self._rtcp_started: set[int] = set()
        self._queue_buffer = None
        if self.inference_mode == "queue" and self.inference_session is not None:
            self._queue_buffer = QueuePlaybackBuffer(
                source.events,
                self.inference_session,
                loop=self.loop,
                on_error=lambda exc: self.on_log(
                    f"session={self.id} queue inference failed: {type(exc).__name__}: {exc}"
                ),
            )
        self._timeline = self._build_timeline(
            source.events,
            send_interval_seconds=self.send_interval_seconds,
        )
        self._sender = threading.Thread(
            target=self._send_loop,
            name=f"rrtsp-send-{self.id}",
            daemon=True,
        )
        self._sender.start()
        self._rtcp_threads: list[threading.Thread] = []

    @staticmethod
    def _build_timeline(
        events: tuple[RRTSPEvent, ...] | list[RRTSPEvent] | object,
        *,
        send_interval_seconds: float,
    ) -> tuple[float, ...]:
        values = list(events)  # type: ignore[arg-type]
        if not values:
            return ()
        offsets = [0.0]
        for previous, current in zip(values, values[1:]):
            delta = float(current.timestamp_seconds) - float(previous.timestamp_seconds)
            if delta <= 0.0:
                # Equal source timestamps represent one radar scan containing
                # multiple target updates. Keep them in the same RTP frame.
                offsets.append(offsets[-1])
                continue
            # Source timestamps group observations; the configured interval
            # controls replay cadence. Model inputs retain their source time.
            offsets.append(offsets[-1] + float(send_interval_seconds))
        return tuple(offsets)

    @property
    def event_count(self) -> int:
        return len(self.source.events)

    @property
    def has_rd(self) -> bool:
        return self.source.has_rd

    @property
    def has_inference(self) -> bool:
        return self.inference_session is not None

    def setup_media(self, track_id: int, client_addr: str, transport: Mapping[str, str]) -> MediaTransport:
        if int(track_id) not in self.media:
            raise ValueError(f"unknown trackID={track_id}")
        media = self.media[int(track_id)]
        if int(track_id) == 0 and not self.has_rd:
            raise ValueError("source does not contain RD frames")
        port_pair = str(transport.get("client_port", ""))
        try:
            client_rtp, client_rtcp = (int(item) for item in port_pair.split("-", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Transport must contain client_port=RTP-RTCP") from exc
        if not 1 <= client_rtp <= 65535 or not 1 <= client_rtcp <= 65535:
            raise ValueError("client UDP ports are out of range")
        media.setup_transport(client_addr, client_rtp, client_rtcp)
        if int(track_id) not in self._rtcp_started:
            self._start_rtcp_thread(media)
        return media

    def set_send_interval_seconds(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("send_interval_seconds must be positive and finite")
        with self._lock:
            previous_timeline = self._timeline
            self.send_interval_seconds = value
            self._timeline = self._build_timeline(
                self.source.events,
                send_interval_seconds=self.send_interval_seconds,
            )
            if self._event_cursor:
                # Rebase on the last transmitted batch without rewinding the
                # stream or turning an interval change into a catch-up burst.
                previous_offset = previous_timeline[self._event_cursor - 1]
                self._cycle_origin += (
                    previous_offset - self._timeline[self._event_cursor - 1]
                )

    def play(self) -> None:
        with self._lock:
            if self._resume_at is None:
                self._resume_at = time.monotonic()
            self.playing.set()
            if self._queue_buffer is not None:
                self._queue_buffer.start()

    def pause(self) -> None:
        with self._lock:
            self._capture_position()
            self._resume_at = None
            self.playing.clear()
            if self._queue_buffer is not None:
                self._queue_buffer.wake()

    def _capture_position(self) -> None:
        if self._resume_at is not None:
            self._position += time.monotonic() - self._resume_at
            self._resume_at = time.monotonic()

    def _current_position(self) -> float:
        with self._lock:
            value = self._position
            if self._resume_at is not None:
                value += time.monotonic() - self._resume_at
            return value

    def close(self) -> None:
        with self._lock:
            self.alive = False
            self.playing.clear()
            self._stop.set()
        self._sender.join(timeout=1.0)
        if self._queue_buffer is not None:
            self._queue_buffer.stop()
        for thread in self._rtcp_threads:
            thread.join(timeout=0.5)
        for media in self.media.values():
            media.close()

    def _reset_cycle(self) -> None:
        cycle_duration = (
            self._timeline[-1] + self.send_interval_seconds
            if self._timeline
            else 0.0
        )
        self._cycle_origin += cycle_duration
        self._event_cursor = 0

    def _send_loop(self) -> None:
        if self._queue_buffer is not None:
            self._send_queue_loop()
            return
        while not self._stop.is_set():
            if not self.playing.wait(timeout=0.2):
                continue
            if self._event_cursor >= self.event_count:
                if not self.loop:
                    self.playing.clear()
                    continue
                self._reset_cycle()
            if not self._timeline:
                self.playing.clear()
                continue
            scan_start = self._event_cursor
            target_offset = self._timeline[scan_start]
            scan_end = scan_start + 1
            while (
                scan_end < self.event_count
                and self._timeline[scan_end] == target_offset
            ):
                scan_end += 1
            scan_events = self.source.events[scan_start:scan_end]
            while not self._stop.is_set() and self.playing.is_set():
                with self._lock:
                    target = self._cycle_origin + self._timeline[scan_start]
                    wait = target - self._current_position()
                if wait <= 0.0:
                    break
                time.sleep(min(wait, 0.05))
            if self._stop.is_set() or not self.playing.is_set():
                continue
            timestamp = int(self._current_position() * RTP_CLOCK_HZ) & 0xFFFFFFFF
            try:
                predictions: list[Mapping[str, object]] = []
                if self.inference_session is not None and self.media[2].setup:
                    inference_started = time.monotonic()
                    for event in scan_events:
                        try:
                            prediction = self.inference_session.predict(event)
                        except Exception as exc:
                            self.on_log(
                                f"session={self.id} inference failed at "
                                f"event={event.event_index}: {type(exc).__name__}: {exc}"
                            )
                        else:
                            predictions.append(prediction)
                    inference_delay = time.monotonic() - inference_started
                    # Keep the media clock aligned with transmitted scans;
                    # expensive package inference must not make later scans
                    # burst through the replay.
                    self._cycle_origin += inference_delay
                    timestamp = (
                        timestamp + int(inference_delay * RTP_CLOCK_HZ)
                    ) & 0xFFFFFFFF
                if predictions:
                    self._send_inference(predictions, timestamp)
                self._send_track(scan_events, timestamp)
                self._send_rd(scan_events, timestamp)
            except (OSError, ValueError) as exc:
                self.on_log(
                    f"session={self.id} media send failed at scan "
                    f"{scan_start}: {exc}"
                )
                self.playing.clear()
                continue
            self._event_cursor = scan_end
            self._update_sequence = (self._update_sequence + 1) & 0xFF

    def _send_queue_loop(self) -> None:
        """Send one point per non-empty queue at the single send interval."""

        last_send: float | None = None
        while not self._stop.is_set():
            if not self.playing.wait(timeout=0.2):
                continue
            wait = (
                last_send + self.send_interval_seconds - self._current_position()
                if last_send is not None else 0.0
            )
            if wait > 0.0:
                if self._stop.wait(min(wait, 0.05)):
                    return
                continue
            try:
                batch = self._queue_buffer.pop_batch(playing=self.playing)
            except Exception as exc:
                self.on_log(
                    f"session={self.id} queue media failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.playing.clear()
                return
            if not batch:
                if not self.loop:
                    self.playing.clear()
                    return
                continue

            timestamp = int(self._current_position() * RTP_CLOCK_HZ) & 0xFFFFFFFF
            events = [event for event, _prediction in batch]
            predictions = [prediction for _event, prediction in batch]
            try:
                if predictions:
                    self._send_inference(predictions, timestamp)
                self._send_track(events, timestamp)
                self._send_rd(events, timestamp)
            except (OSError, ValueError) as exc:
                self.on_log(
                    f"session={self.id} queue media send failed: {exc}"
                )
                self.playing.clear()
                return
            self._update_sequence = (self._update_sequence + 1) & 0xFF
            # Inference may have blocked for longer than one interval. Start
            # the next interval here, never flush accumulated deadlines.
            last_send = self._current_position()

    def _send_track(
        self,
        events: Sequence[RRTSPEvent],
        timestamp: int,
    ) -> None:
        media = self.media[1]
        payloads_for_events = [
            payloads.encode_track(
                event.track_id,
                event.state,
                event.class_id,
                event.tr_values,
            )
            for event in events
        ]
        frame = payloads.build_track_frame(
            payloads_for_events, update_sequence=self._update_sequence
        )
        media.send_rtp(frame, payload_type=97, timestamp=timestamp)

    def _send_inference(
        self,
        predictions: Sequence[Mapping[str, object]],
        timestamp: int,
    ) -> None:
        media = self.media[2]
        if not media.setup:
            return
        frame = payloads.build_inference_frame(
            list(predictions),
            update_sequence=self._update_sequence,
        )
        media.send_rtp(frame, payload_type=98, timestamp=timestamp)

    def _send_rd(
        self,
        events: Sequence[RRTSPEvent],
        timestamp: int,
    ) -> None:
        media = self.media[0]
        if not media.setup:
            return
        # PT=96 carries one complete RD map per radar scan. Directory replay
        # packages have target-specific maps, so use the first available map
        # as the scan-level map shared by the target batch.
        event = next((item for item in events if item.rd_frames), None)
        if event is None:
            return
        for frame_index, frame in enumerate(event.rd_frames):
            chunks = payloads.fragment_rd_frame(
                amplitude=frame.amplitude,
                phase=frame.phase,
                frame_sequence=self._frame_sequence,
                frame_id=event.event_index * 100 + frame_index,
                max_packet_size=MAX_PACKET_SIZE,
            )
            # RD packets belong to the same logical scan batch as PT=97 and
            # PT=98. Keep one RTP timestamp for every RD frame in that batch;
            # the client uses the timestamp to perform strict cross-track
            # pairing and does not use the source RD sub-frame time as a
            # second synchronization clock.
            frame_timestamp = int(timestamp) & 0xFFFFFFFF
            for packet_index, chunk in enumerate(chunks):
                media.send_rtp(
                    chunk,
                    payload_type=96,
                    timestamp=frame_timestamp,
                    marker=packet_index == len(chunks) - 1,
                )
            self._frame_sequence = (self._frame_sequence + 1) & 0xFFFF

    def _start_rtcp_thread(self, media: MediaTransport) -> None:
        if not media.setup or media.rtcp_socket is None:
            return
        thread = threading.Thread(
            target=self._rtcp_loop,
            args=(media,),
            name=f"rrtsp-rtcp-{self.id}-{media.track_id}",
            daemon=True,
        )
        self._rtcp_started.add(media.track_id)
        self._rtcp_threads.append(thread)
        thread.start()

    def _rtcp_loop(self, media: MediaTransport) -> None:
        assert media.rtcp_socket is not None
        media.rtcp_socket.settimeout(0.2)
        last_report = 0.0
        while self.alive and media.setup:
            now = time.monotonic()
            if now - last_report >= 1.0:
                report = rtp.build_sender_report(
                    ssrc=media.ssrc,
                    rtp_timestamp=media.last_timestamp,
                    packet_count=media.packets_sent,
                    octet_count=media.octets_sent,
                )
                try:
                    media.rtcp_socket.sendto(
                        report,
                        (media.client_addr, media.client_rtcp_port),
                    )
                except OSError:
                    return
                last_report = now
            try:
                packet, _ = media.rtcp_socket.recvfrom(4096)
                if any(item["type"] == rtp.PT_RTCP_RR for item in rtp.parse_rtcp(packet)):
                    self.on_log(f"session={self.id} received RTCP RR for {media.name}")
            except socket.timeout:
                pass
            except (OSError, ValueError):
                return


__all__ = [
    "MAX_PACKET_SIZE",
    "MediaTransport",
    "QueuePlaybackBuffer",
    "ViewerSession",
]
