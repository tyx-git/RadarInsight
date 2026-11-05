"""Start an AeroSight-compatible R-RTSP replay server.

Example:

    python -m core.server --asd-dir data/asd --host 0.0.0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading

from .model import ONNXProductBackend
from .rrtsp import RRTSPServer, RRTSPServerConfig
from .source import (
    ASDDirectoryStreamSource,
    ASDStreamSource,
    MODEL_CONTEXT_EVENTS,
    QUEUE_MAX_EVENTS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _event_limit(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > QUEUE_MAX_EVENTS:
        raise argparse.ArgumentTypeError(
            f"the current source adapter supports at most {QUEUE_MAX_EVENTS} events per package"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RadarInsight R-RTSP backend server")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--asd", type=Path, default=None, help="replay one ASD package")
    source.add_argument(
        "--asd-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "asd",
        help="replay top-level ASD packages in filename order (default: data/asd)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="RTSP bind address")
    parser.add_argument("--port", type=int, default=8554, help="RTSP TCP port; 0 chooses a free port")
    parser.add_argument(
        "--send-interval",
        type=_positive_float,
        default=1.0,
        help="seconds between RTP batches (default: 1)",
    )
    parser.add_argument("--no-loop", action="store_true", help="stop after one ASD sequence")
    parser.add_argument(
        "--max-events",
        type=_event_limit,
        default=None,
        help="source events per ASD package; defaults to 29 for causal/package and 60 for queue",
    )
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"), default=None)
    parser.add_argument("--amplitude-mode", default=None)
    parser.add_argument("--auth", choices=("open", "digest"), default="open")
    parser.add_argument("--username", default="viewer")
    parser.add_argument("--password", default="")
    parser.add_argument("--realm", default="AeroSight")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/experimental/v28/LV28.onnx"),
        help="server-side Full Cascade ONNX path",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cpu", action="store_true", help="do not try DirectML for the optional model")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="disable server-side inference and PT=98 classification results",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("causal", "queue", "package"),
        default="causal",
        help=(
            "causal: one inference per target event; queue: batched ten-target "
            "buffered inference; package: classify each complete ASD parent once"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    send_interval_seconds = float(args.send_interval)
    max_events = (
        int(args.max_events)
        if args.max_events is not None
        else (QUEUE_MAX_EVENTS if args.inference_mode == "queue" else MODEL_CONTEXT_EVENTS)
    )
    source = (
        ASDStreamSource(
            args.asd,
            timestamp_unit=args.timestamp_unit,
            amplitude_mode=args.amplitude_mode,
            max_events=max_events,
        )
        if args.asd is not None
        else ASDDirectoryStreamSource(
            args.asd_dir,
            timestamp_unit=args.timestamp_unit,
            amplitude_mode=args.amplitude_mode,
            max_events_per_package=max_events,
            send_interval_seconds=send_interval_seconds,
        )
    )
    print(f"source: {source.metadata}")
    inference_factory = None
    if not args.no_model:
        from .model.streaming import (
            FullCascadeInferenceFactory,
            FullCascadeQueueInferenceFactory,
            FullCascadePackageInferenceFactory,
        )

        model_path = args.onnx if args.onnx.is_absolute() else args.project_root / args.onnx
        backend = ONNXProductBackend(
            args.project_root,
            model_path,
            prefer_directml=not args.cpu,
        )
        readiness = backend.readiness()
        print(f"model readiness: {readiness}")
        if readiness["research_only"] or not readiness["deployment_eligible"]:
            print("WARNING: bound cascade is research-only; rendered predictions are not deployment-certified")
        factory_type = {
            "causal": FullCascadeInferenceFactory,
            "queue": FullCascadeQueueInferenceFactory,
            "package": FullCascadePackageInferenceFactory,
        }[args.inference_mode]
        inference_factory = factory_type(backend.pipeline)
    server = RRTSPServer(
        source,
        config=RRTSPServerConfig(
            host=args.host,
            port=args.port,
            send_interval_seconds=send_interval_seconds,
            inference_mode=args.inference_mode,
            loop=not args.no_loop,
            auth_mode=args.auth,
            username=args.username,
            password=args.password,
            realm=args.realm,
        ),
        inference_factory=inference_factory,
        on_log=print,
    )
    stop = threading.Event()

    def request_stop(*_signal_args: object) -> None:
        stop.set()

    for signum in (signal.SIGTERM,):
        try:
            signal.signal(signum, request_stop)
        except (OSError, ValueError):
            pass
    try:
        server.start()
        print(f"connect AeroSight to {server.url}")
        while not stop.wait(0.25):
            pass
    except KeyboardInterrupt:
        print("shutdown requested")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
