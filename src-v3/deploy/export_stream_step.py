"""RK3588/RK3568 streaming deployment prep for the V28 Liquid V2 cell.

Two things, in order:

  PART 1  NUMERICAL PROOF
          The upstream LiquidV2Cell._time_parameters() computes
              floor = (safe_dt <= 1e-9) | (~dt_valid)
              safe_dt = where(floor, floor_value, safe_dt)
              alpha   = where(~dt_valid, gain, alpha)
          Both `~dt_valid` use sites emit an ONNX `Not`, which RK NPU does not
          support.  This script re-derives the same two lines using
          `where(dt_valid, a, b)` (no negation, no `Not`) and asserts the two
          formulations agree bit-for-bit across the full dt_reason space.

  PART 2  ONNX EXPORT
          Export the streaming step (batch=1, one event, one state) to ONNX and
          print the operator histogram, cross-checked against the RK ops that
          the NPU does not implement.

The streaming path is LiquidV2Cell.step(), reached from
V28TaskModel.online_encoded_step() (src-v3/v28/model.py:205).  Under
invalid_dt_update_mode="bounded_observation" -- which V28-default.json uses --
step() returns _compute_step() immediately and never touches the
torch.nonzero strict_skip branch at liquid_v2.py:442.  So the NonZero /
ScatterElements that dominate the T=29 padded batch graph are simply absent.

Usage:
    ./.venv/Scripts/python.exe src-v3/deploy/export_stream_step.py
    ./.venv/Scripts/python.exe src-v3/deploy/export_stream_step.py --out my_step.onnx

NOTE ON LOCATION
    This file lives in src-v3/deploy/, NOT src-v3/v28/.  The V28 system
    manifest rglobs src-v3/v28/**/*.py and hashes every hit, so dropping a new
    .py inside that tree changes code_hashes and breaks the frozen manifest
    that every checkpoint and the L3 registry bind to.  Keeping tooling
    outside the tree means it can be edited freely.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

# ---------------------------------------------------------------- constants --

DT_REASON_FIRST_EVENT = 1
DT_REASON_ZERO_DELTA = 2
DT_REASON_POSITIVE_DELTA = 3

# RK NPU ops that are NOT implemented by the compiler.  Sourced from the
# RKNN Compiler Support Operator List.  Anything here will be pushed to CPU
# fallback (RKNN v2.x does fall back, but it breaks the NPU pipeline).
RK_UNSUPPORTED = frozenset(
    {"NonZero", "ScatterElements", "GatherElements", "Expand", "Not"}
)


def _defaults() -> dict[str, float | int]:
    """Mirror src-v3/v28/config/V28-default.json."""
    return {
        "hidden_dim": 48,
        "input_dim": 48,
        "tau_min_seconds": 0.05,
        "tau_max_seconds": 30.0,
        "zero_dt_event_floor_seconds": 0.001,
        "dt_max_seconds": 120.0,
        "invalid_dt_observation_gain": 0.25,
    }


# ------------------------------------------------------------ reference impl --


class OriginalCell(nn.Module):
    """Faithful copy of LiquidV2Cell._compute_step + _time_parameters.

    Uses `~dt_valid` in two places, which is exactly what emits the ONNX `Not`.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = int(cfg["hidden_dim"])
        i = int(cfg["input_dim"])
        self.tau_min = float(cfg["tau_min_seconds"])
        self.tau_max = float(cfg["tau_max_seconds"])
        self.zero_floor = float(cfg["zero_dt_event_floor_seconds"])
        self.dt_max = float(cfg["dt_max_seconds"])
        self.gain = float(cfg["invalid_dt_observation_gain"])

        self.compute_norm = nn.LayerNorm(d)
        self.input_projection = nn.Linear(i, d)
        self.state_projection = nn.Linear(d, d, bias=False)
        self.candidate_projection = nn.Linear(i + d, d)
        self.gate_projection = nn.Linear(i + d, d)
        self.tau_unconstrained = nn.Parameter(torch.zeros(d))

    def _tau(self, ref: torch.Tensor) -> torch.Tensor:
        span = self.tau_max - self.tau_min
        return (self.tau_min + span * torch.sigmoid(self.tau_unconstrained)).to(
            dtype=ref.dtype, device=ref.device
        )

    def time_parameters(
        self,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (alpha, invalid_dt_mask) -- the two Not-bearing outputs."""
        raw_dt = dt_seconds
        safe_dt = raw_dt.clamp(min=0.0, max=self.dt_max)
        floor = (safe_dt <= 1.0e-9) | (~dt_valid)                    # NOT #1
        safe_dt = torch.where(
            floor, torch.full_like(safe_dt, self.zero_floor), safe_dt
        )
        tau = self._tau(raw_dt)
        alpha = 1.0 - torch.exp(-safe_dt / tau)

        observation_gain = torch.full_like(alpha, self.gain)
        alpha = torch.where(~dt_valid, observation_gain, alpha)      # NOT #2

        first = dt_reason == DT_REASON_FIRST_EVENT
        zero = dt_reason == DT_REASON_ZERO_DELTA
        zero_gain = torch.full_like(alpha, self.gain)
        alpha = torch.where(first, torch.ones_like(alpha), alpha)
        alpha = torch.where(zero, zero_gain, alpha).clamp(0.0, 1.0)
        # float, so the equivalence check compares like with like
        invalid_dt = (1.0 - dt_valid.to(dtype=raw_dt.dtype))          # NOT #3
        return alpha, invalid_dt

    def forward(
        self,
        event: torch.Tensor,
        state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.compute_norm(state)
        combined = torch.cat((event, normalized), dim=-1)
        candidate = torch.tanh(
            self.candidate_projection(combined)
            + self.input_projection(event)
            + self.state_projection(normalized)
        )
        gate = torch.sigmoid(self.gate_projection(combined))
        mixed = gate * candidate + (1.0 - gate) * state
        alpha, invalid_dt = self.time_parameters(dt_seconds, dt_valid, dt_reason)
        next_state = (1.0 - alpha) * state + alpha * mixed
        return next_state, invalid_dt


class NotFreeCell(nn.Module):
    """Same cell with every `~dt_valid` re-expressed as a positive Where.
        floor    : (safe_dt <= eps) | (~dt_valid)
                 -> where(dt_valid, (safe_dt <= eps), TRUE)
        alpha    : where(~dt_valid, gain, alpha)
                 -> where(dt_valid, alpha, gain)
        invalid  : ~dt_valid
                 -> where(dt_valid, FALSE, TRUE)

    All three produce no `Not` in the exported graph.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = int(cfg["hidden_dim"])
        i = int(cfg["input_dim"])
        self.tau_min = float(cfg["tau_min_seconds"])
        self.tau_max = float(cfg["tau_max_seconds"])
        self.zero_floor = float(cfg["zero_dt_event_floor_seconds"])
        self.dt_max = float(cfg["dt_max_seconds"])
        self.gain = float(cfg["invalid_dt_observation_gain"])

        self.compute_norm = nn.LayerNorm(d)
        self.input_projection = nn.Linear(i, d)
        self.state_projection = nn.Linear(d, d, bias=False)
        self.candidate_projection = nn.Linear(i + d, d)
        self.gate_projection = nn.Linear(i + d, d)
        self.tau_unconstrained = nn.Parameter(torch.zeros(d))

    def _tau(self, ref: torch.Tensor) -> torch.Tensor:
        span = self.tau_max - self.tau_min
        return (self.tau_min + span * torch.sigmoid(self.tau_unconstrained)).to(
            dtype=ref.dtype, device=ref.device
        )

    def time_parameters(
        self,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not-free alpha, with the [B] -> [B,H] broadcast kept implicit.

        Four things are needed to land on a graph RK NPU can execute whole:

        1. every `~dt_valid` becomes `where(dt_valid, a, b)`
        2. the per-event scalars arrive shaped [B, 1], so the alpha arithmetic
           broadcasts directly.  Shaped [B], the tracer inserts Unsqueeze +
           Expand to line them up with [B, H].
        3. dt_reason arrives as FLOAT.  The upstream contract types it int64,
           and comparing int64 against a python int makes the tracer emit a
           Cast plus an Expand of the int64 scalar.  The caller casts the
           reason id once, outside the graph.
        4. no `Where` with BOOL branches.  `Where(cond, bool, bool)` has no CPU
           kernel in ONNX Runtime and is a poor fit for the NPU; the floor
           selection is done in float arithmetic instead.
        """
        raw_dt = dt_seconds
        safe_dt = raw_dt.clamp(min=0.0, max=self.dt_max)

        # floor = (safe_dt <= eps) OR (not valid)
        # Written as float arithmetic: keep_own = valid AND safe_dt > eps.
        valid_f = dt_valid.to(dtype=safe_dt.dtype)
        above_eps = (safe_dt > 1.0e-9).to(dtype=safe_dt.dtype)
        keep_own = valid_f * above_eps
        floor_value = torch.full_like(safe_dt, self.zero_floor)
        safe_dt = keep_own * safe_dt + (1.0 - keep_own) * floor_value

        # NOTE: no .expand() here.  tau is [H] and safe_dt is [B,1], so the
        # division already broadcasts to [B,H].  Calling .expand() first makes
        # the tracer emit an Expand node, which RK NPU does not implement.
        tau = self._tau(raw_dt)
        alpha = 1.0 - torch.exp(-safe_dt / tau)

        observation_gain = torch.full_like(alpha, self.gain)
        alpha = torch.where(dt_valid, alpha, observation_gain)

        reason = dt_reason if dt_reason.is_floating_point() else dt_reason.float()
        first = reason == float(DT_REASON_FIRST_EVENT)
        zero = reason == float(DT_REASON_ZERO_DELTA)
        zero_gain = torch.full_like(alpha, self.gain)
        alpha = torch.where(first, torch.ones_like(alpha), alpha)
        alpha = torch.where(zero, zero_gain, alpha).clamp(0.0, 1.0)

        # invalid = not valid, in float land (no bool Where anywhere).
        invalid_dt = 1.0 - valid_f
        return alpha, invalid_dt

    def forward(
        self,
        event: torch.Tensor,
        state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.compute_norm(state)
        combined = torch.cat((event, normalized), dim=-1)
        candidate = torch.tanh(
            self.candidate_projection(combined)
            + self.input_projection(event)
            + self.state_projection(normalized)
        )
        gate = torch.sigmoid(self.gate_projection(combined))
        mixed = gate * candidate + (1.0 - gate) * state
        alpha, invalid_dt = self.time_parameters(dt_seconds, dt_valid, dt_reason)
        next_state = (1.0 - alpha) * state + alpha * mixed
        return next_state, invalid_dt


# ------------------------------------------------------------------- helpers --


def verify_with_onnxruntime(path: Path, cell: "NotFreeCell") -> None:
    """Cross-check the exported graph against the exact module that was exported.

    `cell` MUST be the same instance that was passed to torch.onnx.export --
    building a fresh one here gives different random weights and the comparison
    becomes meaningless.

    ONNX Runtime's CPU EP has no kernel for `Where` when the branches are BOOL.
    That shape does not appear in this graph, but if a future edit reintroduces
    it this function reports the gap instead of failing opaquely.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed -- skipping runtime cross-check")
        return

    try:
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"  ORT-CPU could not load the graph: {first}")
        print("    -> verify on the board with rknn_toolkit_lite2 instead.")
        return

    import numpy as np

    print("  ORT-CPU loaded the graph; running the cross-check")
    cases = [
        ("first event (r=1,v=F)", 0.0, False, 1),
        ("zero delta (r=2,v=F)", 0.0, False, 2),
        ("positive dt=0.5 (r=3)", 0.5, True, 3),
        ("positive dt>dt_max", 137.0, True, 3),
        ("negative dt (r=3,v=T)", -2.0, True, 3),
    ]
    d_model = cell.compute_norm.normalized_shape[0]
    d_input = cell.input_projection.in_features

    print(f"  {'case':<24}{'max_abs_err':>14}{'verdict':>12}")
    print("  " + "-" * 48)
    worst = 0.0
    failed = 0
    for name, dt, valid, reason in cases:
        event = torch.randn(1, d_input)
        state = torch.randn(1, d_model)
        dt_t = torch.tensor([[dt]], dtype=torch.float32)
        v_t = torch.tensor([[valid]], dtype=torch.bool)
        r_t = torch.tensor([[float(reason)]], dtype=torch.float32)
        with torch.no_grad():
            ref, _ = cell(event, state, dt_t, v_t, r_t)
        got = session.run(
            None,
            {
                "event": event.numpy(),
                "state": state.numpy(),
                "dt_seconds": dt_t.numpy(),
                "dt_valid": v_t.numpy(),
                "dt_reason": r_t.numpy(),
            },
        )[0]
        err = float(np.max(np.abs(ref.numpy() - got)))
        worst = max(worst, err)
        ok = err < 1e-5
        failed += 0 if ok else 1
        print(
            f"  {name:<24}{err:>14.2e}"
            f"{'MATCH' if ok else '*** WRONG ***':>12}"
        )
    print(f"  worst error: {worst:.2e}")
    if failed:
        print(f"  *** {failed} case(s) disagree -- the export is not faithful.")


def op_histogram(path: Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(path))
    hist: dict[str, int] = {}
    for node in model.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    return hist


def print_histogram(hist: dict[str, int], *, total: int) -> list[str]:
    print(f"  nodes: {total}")
    for name, count in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])):
        flag = "  <-- NOT SUPPORTED" if name in RK_UNSUPPORTED else ""
        print(f"    {name:<24} {count:>4}{flag}")
    return sorted(RK_UNSUPPORTED & set(hist))


# ---------------------------------------------------------------------- main --


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="stream_step_notfree.onnx",
        help="output ONNX path (default: stream_step_notfree.onnx)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="random cases for the equivalence check (default: 4096)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = _defaults()

    original = OriginalCell(cfg).eval()
    notfree = NotFreeCell(cfg).eval()
    # Same weights so the comparison is apples-to-apples.
    notfree.load_state_dict(original.state_dict())

    # ---------------------------------------------------------- PART 1 ------
    print("=" * 78)
    print("PART 1  equivalence of the Not-free rewrite")
    print("=" * 78)

    n = args.samples
    dt_seconds = torch.cat(
        [
            torch.zeros(n // 4),                                   # first / zero
            torch.rand(n // 4) * cfg["dt_max_seconds"],            # in range
            torch.full((n // 4,), float(cfg["dt_max_seconds"]) * 2),
            -torch.rand(n // 4),                                   # negative
        ]
    )
    dt_valid = torch.rand(n) < 0.6
    dt_reason = torch.randint(
        DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA + 1, (n,)
    ).long()
    dt_valid = dt_reason == DT_REASON_POSITIVE_DELTA
    dt_seconds = torch.where(
        dt_reason == DT_REASON_POSITIVE_DELTA,
        dt_seconds.abs().clamp(min=1e-6),
        torch.zeros_like(dt_seconds),
    )
    # The streaming shape is [B,1] for the per-event scalars.  This is also
    # what keeps Unsqueeze/Expand out of the exported graph.
    dt_seconds = dt_seconds.unsqueeze(-1)
    dt_valid = dt_valid.unsqueeze(-1)
    dt_reason = dt_reason.unsqueeze(-1)

    with torch.no_grad():
        alpha_o, invalid_o = original.time_parameters(dt_seconds, dt_valid, dt_reason)
        alpha_n, invalid_n = notfree.time_parameters(dt_seconds, dt_valid, dt_reason)

    alpha_err = float((alpha_o - alpha_n).abs().max())
    invalid_same = bool(torch.equal(invalid_o, invalid_n))
    print(f"  cases                 : {n}")
    print(f"  max |alpha_o - alpha_n|: {alpha_err:.3e}")
    print(f"  invalid_dt identical  : {invalid_same}")
    ok = alpha_err == 0.0 and invalid_same
    print(f"  verdict               : {'IDENTICAL' if ok else '*** DIFFERS ***'}")
    if not ok:
        print("  Do NOT proceed -- the rewrite is not equivalent.")
        return 1

    # ---------------------------------------------------------- PART 2 ------
    print()
    print("=" * 78)
    print("PART 2  streaming step ONNX export (batch=1, T=1)")
    print("=" * 78)

    event = torch.randn(1, cfg["input_dim"])
    state = torch.randn(1, cfg["hidden_dim"])
    #                       ^^^ [B,1] on purpose: see NotFreeCell.time_parameters.
    #                       Passing [B] makes the tracer insert Unsqueeze+Expand.
    dt = torch.tensor([[0.5]])
    valid = torch.tensor([[True]])
    # dt_reason as FLOAT: see NotFreeCell.time_parameters.  An int64 reason
    # against a python int makes the tracer emit Cast + Expand.
    reason = torch.tensor([[float(DT_REASON_POSITIVE_DELTA)]])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    # Export the INFERENCE outputs only.  `invalid_dt` is a diagnostic that the
    # board never consumes, and carrying it forces a broadcast (Expand) into
    # the graph.  The deployment module returns next_state alone.
    class _InferenceOnly(nn.Module):
        def __init__(self, cell: NotFreeCell) -> None:
            super().__init__()
            self.cell = cell

        def forward(
            self,
            event: torch.Tensor,
            state: torch.Tensor,
            dt_seconds: torch.Tensor,
            dt_valid: torch.Tensor,
            dt_reason: torch.Tensor,
        ) -> torch.Tensor:
            next_state, _ = self.cell(
                event, state, dt_seconds, dt_valid, dt_reason
            )
            return next_state

    module = _InferenceOnly(notfree).eval()

    torch.onnx.export(
        module,
        (event, state, dt, valid, reason),
        str(out_path),
        input_names=["event", "state", "dt_seconds", "dt_valid", "dt_reason"],
        output_names=["next_state"],
        opset_version=17,
        dynamo=False,
    )
    print(f"  exported -> {out_path}")

    hist = op_histogram(out_path)
    hits = print_histogram(hist, total=sum(hist.values()))

    print()
    print("  runtime cross-check")
    verify_with_onnxruntime(out_path, notfree)

    print()
    if hits:
        print(f"  *** RK-unsupported ops still present: {hits}")
        print("  These will be pushed to CPU fallback on RKNPU.")
        return 1

    print("  RK-unsupported ops: NONE")
    print()
    print("=" * 78)
    print("next steps")
    print("=" * 78)
    print("  1. On an x86_64 Linux box (WSL2 is fine):")
    print("       from rknn.api import RKNN")
    print("       rk = RKNN(verbose=True)          # verbose shows the Target column")
    print(f"       rk.config(target_platform='rk3588')")
    print(f"       rk.load_onnx(model='{out_path.name}')")
    print("       rk.build(do_quantization=False)  # start FP16, not INT8")
    print("       rk.export_rknn('stream_step.rknn')")
    print("  2. In the build log, confirm every row reads Target=NPU.")
    print("  3. Measure a single rknn_run() on the board -- that is your latency")
    print("     floor.  Streaming has no batch to amortise it over.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
