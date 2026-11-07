# RadarInsight backend server

`core.server` is the backend side of the current AeroSight flow. It serves an
`.asd` event sequences over the custom R-RTSP v1 protocol:

```text
data/asd/*.asd -> ASDDirectoryStreamSource -> RRTSPServer
                                            -> PT=96/97 observations -> AeroSight
                                            -> Full Cascade ONNX -> PT=98 results
```

The RRTSP stream separates observations from optional model results:

- PT=97 carries the eight finite TR fields in the protocol's TLV format;
- PT=96 carries amplitude and phase RD matrices, fragmented at 1200 bytes;
- optional PT=98 carries Full Cascade class and confidence keyed by track ID;
- RTSP `DESCRIBE/SETUP/PLAY/PAUSE/TEARDOWN` controls a per-client session;
- `SET_PARAMETER` sets that session's `send_interval_seconds` and
  `GET_PARAMETER` returns its current value;
- RTCP Sender Reports are emitted and Receiver Reports are accepted;
- optional Digest SHA-256 authentication is supported.

For ASD replay, the package label is evaluation ground truth and is never sent
as the radar class. PT=97 therefore uses `class_id=5` (unknown) and a suspect
state; the Full Cascade result is sent separately on PT=98.

Directory replay defaults to one aggregated multi-target batch every 1 second.
Each scan takes the next causal point from every available ASD package and
sends the batch in one PT=97 frame; the matching PT=98 labels use the same RTP
timestamp. Original TR timestamps remain unchanged in model input. Use
`--send-interval 1` to set the server's default interval in seconds. Source
timestamp gaps do not impose a second replay interval. Causal inference runs
before sending, so inference latency can lengthen the actual batch interval.

In AeroSight settings, each radar instance has two independent periods:

- Data source: backend send period (`s`), default `1.0`;
- Display: frontend scan period (`s`), default `3.0`.

Applying a send-period change reconnects the affected instance. Before `PLAY`,
the client sends `SET_PARAMETER` with `Content-Type: text/parameters` and body
`send_interval_seconds: 1.0`. The server acknowledges the effective interval;
the change affects only this viewing session, without restarting the backend
or changing other viewers. A scan-period-only change keeps the connection and
current scan angle. The frontend's animation period is never sent to the server.

`ASDStreamSource` reuses `radar_ln3.data.asd.load_asd`, so timestamp units,
finite numeric checks, physical-axis canonicalization and the 12-frame RD cap
are applied before transmission. Consequently, the ASD replay source emits
the current canonical `21x384` RD shape; a future `StreamSource` may publish
another shape if the AeroSight model contract allows it. The fixed product
window in causal/package mode keeps only event indices `0..28`. Queue mode
can read up to 60 source events per target, with a 29-event model context.

Run a replay server from the source root:

```text
python -m core.server --asd-dir data/asd --host 0.0.0.0 --port 8554
```

`--asd` can be used to replay exactly one package. The directory defaults to
the backend's `src/data/asd` folder.

Then configure AeroSight with:

```text
rtsp://<server-ip>:8554/radar
```

With the default directory source, the server scans `src/data/asd/*.asd` in
filename order, reads at most 29 causal events from each package, and
interleaves equal event indices as successive radar scans. A shorter package
simply contributes no point after its final event. Default `causal` inference
re-evaluates the Full Cascade for each event prefix, so early observations can
remain low-confidence or `PENDING` until more points arrive. Use
`--inference-mode package` explicitly to run Full Cascade once per package and
reuse the result for every point in that target's replay. The built-in AeroSight
configuration connects to `rtsp://127.0.0.1:8554/radar` and consumes PT=98
results without loading a local compact ONNX model.

Use `--inference-mode queue` to prepare predictions in batches of up to ten
targets and buffer up to 60 points per target. Queues refill below 29 points.
The sender takes one prepared point from each available target at the session's
send interval. An empty queue waits for inference; completed batches are not
sent in a catch-up burst.

`core.server.model.ONNXRuntimeModel` is a generic tensor-level ONNX Runtime
adapter. `ONNXProductBackend` binds the current Full Cascade ONNX artifact to
the product preprocessing pipeline. Start without inference using `--no-model`.
Future models can provide another `StreamingInferenceFactory` without changing
the RRTSP session or AeroSight rendering contract.
